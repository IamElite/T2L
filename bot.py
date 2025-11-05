"""
Telegram File-to-Link Bot - FULLY WORKING VERSION
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus

import httpx
from motor.motor_asyncio import AsyncIOMotorClient
from pyrogram import Client, filters, idle
from pyrogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# Clean logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable verbose logs
logging.getLogger("pyrogram").setLevel(logging.WARNING)
logging.getLogger("pyrogram.session").setLevel(logging.ERROR)
logging.getLogger("pyrogram.connection").setLevel(logging.ERROR)

# Config
API_ID = int(os.getenv("API_ID", 0))
API_HASH = os.getenv("API_HASH", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
MONGO_URL = os.getenv("MONGO_URL", "")
OWNER_ID = int(os.getenv("OWNER_ID", 0))
LOG_CHANNEL_ID = int(os.getenv("LOG_CHANNEL_ID", 0))
BASE_URL = os.getenv("BASE_URL", "http://localhost:8080").rstrip("/")
SHORTLINK_ENABLED = os.getenv("SHORTLINK_ENABLED", "false").lower() == "true"
SHORTLINK_API_KEY = os.getenv("SHORTLINK_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
STORE_EXTRA = os.getenv("STORE_EXTRA", "false").lower() == "true"
STORE_USER = os.getenv("STORE_USER", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8080))

logger.info(f"✅ Config loaded - PORT: {PORT}")

# MongoDB
try:
    mongo = AsyncIOMotorClient(MONGO_URL)
    db = mongo.filetolinks
    files_collection = db.files
    users_collection = db.users if STORE_USER else None
    logger.info("✅ MongoDB connected")
except Exception as e:
    logger.error(f"❌ MongoDB error: {e}")
    db = None

# FastAPI & Pyrogram
app = FastAPI(title="File Bot", version="1.0")
bot = Client("file_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN, workers=4)

# ==================== UTILITIES ====================

def get_name(log_msg: Message) -> str:
    if log_msg.video:
        name = log_msg.video.file_name or f"file_{log_msg.id}"
    elif log_msg.document:
        name = log_msg.document.file_name or f"file_{log_msg.id}"
    else:
        name = f"file_{log_msg.id}"
    return name.replace('/', '-').replace('\\', '-')

def get_hash(log_msg: Message) -> str:
    key = f"{log_msg.chat.id}:{log_msg.id}:{SECRET_KEY}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]

async def get_shortlink(url: str) -> Optional[str]:
    if not SHORTLINK_ENABLED:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://shorten.example.com/create",
                json={"url": url, "key": SHORTLINK_API_KEY}
            )
            return resp.json().get("short_url")
    except:
        return None

def format_size(bytes_size: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = bytes_size
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.1f} {units[unit_idx]}"

def is_owner(user_id: int) -> bool:
    return user_id == OWNER_ID

async def save_file_record(log_msg_id, uploader_id, file_name, file_size, tg_file_id, hash_val, short_stream=None, short_download=None):
    if not db:
        return
    try:
        doc = {
            "log_msg_id": log_msg_id,
            "uploader_id": uploader_id,
            "file_name": file_name,
            "file_size": file_size,
            "tg_file_id": tg_file_id,
            "hash": hash_val,
            "date": datetime.utcnow(),
            "short_stream": short_stream,
            "short_download": short_download
        }
        await files_collection.insert_one(doc)
        
        if STORE_USER and users_collection:
            await users_collection.update_one(
                {"user_id": uploader_id},
                {"$inc": {"file_count": 1}, "$setOnInsert": {"first_seen": datetime.utcnow()}},
                upsert=True
            )
        logger.info(f"✅ Saved: {file_name}")
    except Exception as e:
        logger.error(f"DB save error: {e}")

async def delete_file_record(log_msg_id, requester_id):
    if not db:
        return False
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id})
        if not record:
            return False
        if not (is_owner(requester_id) or record["uploader_id"] == requester_id):
            return False
        await files_collection.delete_one({"log_msg_id": log_msg_id})
        return True
    except:
        return False

# ==================== FASTAPI ROUTES ====================

@app.head("/")
@app.get("/")
async def root():
    return {"status": "running", "bot": "File-to-Link Bot"}

@app.head("/health")
@app.get("/health")
async def health():
    return {"status": "OK"}

@app.get("/meta/{log_msg_id}")
async def meta(log_msg_id: int):
    if not db:
        raise HTTPException(503, "DB unavailable")
    
    record = await files_collection.find_one({"log_msg_id": log_msg_id})
    if not record:
        raise HTTPException(404, "Not found")
    
    return {
        "file_name": record["file_name"],
        "file_size": record["file_size"],
        "uploader_id": record["uploader_id"],
        "mime_type": "video/mp4" if record["file_name"].endswith(".mp4") else "application/octet-stream",
    }

@app.get("/watch/{log_msg_id}/{name}")
async def watch(log_msg_id: int, name: str, hash: str):
    if not db:
        raise HTTPException(503, "DB unavailable")
    
    record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
    if not record:
        raise HTTPException(403, "Invalid hash")
    
    file_name = record["file_name"]
    file_size = format_size(record["file_size"])
    download_url = f"{BASE_URL}/{log_msg_id}/{name}?hash={hash}"
    stream_url = f"{BASE_URL}/stream/{log_msg_id}/{name}?hash={hash}"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{file_name}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial; max-width: 900px; margin: 50px auto; padding: 20px; background: #f0f0f0; }}
            h1 {{ color: #333; }}
            video {{ width: 100%; max-width: 100%; border: 2px solid #333; margin: 20px 0; background: #000; }}
            .info {{ background: #fff; padding: 15px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .buttons {{ margin: 20px 0; }}
            a {{ color: #fff; text-decoration: none; margin-right: 15px; padding: 12px 20px; background: #007bff; border-radius: 5px; display: inline-block; }}
            a:hover {{ background: #0056b3; }}
            .download {{ background: #28a745; }}
            .download:hover {{ background: #1e7e34; }}
        </style>
    </head>
    <body>
        <h1>▶️ {file_name}</h1>
        <div class="info">
            <p><strong>Size:</strong> {file_size}</p>
        </div>
        <video controls preload="metadata" controlsList="nodownload">
            <source src="{stream_url}" type="video/mp4">
            Your browser does not support video playback.
        </video>
        <div class="buttons">
            <a href="{download_url}" class="download">⬇️ Download File</a>
            <a href="javascript:window.close()">❌ Close</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/stream/{log_msg_id}/{name}")
async def stream(log_msg_id: int, name: str, hash: str, request: Request):
    if not db:
        raise HTTPException(503, "DB unavailable")
    
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
        if not record:
            raise HTTPException(403, "Invalid hash")
        
        log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
        if not log_msg or not (log_msg.video or log_msg.document):
            raise HTTPException(404, "File not found")
        
        # Get file size
        if log_msg.video:
            total_size = log_msg.video.file_size
        else:
            total_size = log_msg.document.file_size
        
        range_header = request.headers.get("range")
        
        if range_header:
            # Parse range header
            byte_range = range_header.replace("bytes=", "").split("-")
            start = int(byte_range[0]) if byte_range[0] else 0
            end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else total_size - 1
            
            # Calculate chunk size
            chunk_size = end - start + 1
            
            # Stream with range using proper Pyrogram stream_media
            async def range_streamer():
                offset = start
                remaining = chunk_size
                async for chunk in bot.stream_media(log_msg, offset=offset, limit=remaining):
                    yield chunk
            
            return StreamingResponse(
                range_streamer(),
                status_code=206,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{total_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(chunk_size),
                    "Content-Type": "video/mp4"
                }
            )
        else:
            # Full file streaming
            async def full_streamer():
                async for chunk in bot.stream_media(log_msg):
                    yield chunk
            
            return StreamingResponse(
                full_streamer(),
                headers={
                    "Content-Type": "video/mp4",
                    "Accept-Ranges": "bytes"
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(500, f"Server error: {str(e)}")

@app.get("/{log_msg_id}/{name}")
async def download(log_msg_id: int, name: str, hash: str):
    if not db:
        raise HTTPException(503, "DB unavailable")
    
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
        if not record:
            raise HTTPException(403, "Invalid hash")
        
        file_name = record["file_name"]
        log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
        
        if not log_msg or not (log_msg.video or log_msg.document):
            raise HTTPException(404, "File not found")
        
        # Stream file for download
        async def download_streamer():
            async for chunk in bot.stream_media(log_msg):
                yield chunk
        
        return StreamingResponse(
            download_streamer(),
            headers={
                "Content-Disposition": f'attachment; filename="{file_name}"',
                "Content-Type": "application/octet-stream"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(500, f"Server error: {str(e)}")

# ==================== BOT HANDLERS ====================

@bot.on_message(filters.command("start") & filters.private)
async def cmd_start(client, message):
    await message.reply(
        "🎬 **Telegram File-to-Link Bot**\n\n"
        "Send me any video or document and I'll convert it to streamable links!\n\n"
        "📌 **Commands:**\n"
        "/myfiles - View your uploaded files\n"
        "/help - Get help\n\n"
        "Just send a file to get started! 🚀"
    )
    logger.info(f"Start: {message.from_user.id}")

@bot.on_message(filters.command("help") & filters.private)
async def cmd_help(client, message):
    await message.reply(
        "📖 **How to use:**\n\n"
        "1️⃣ Send me a video or document\n"
        "2️⃣ Get instant stream & download links\n"
        "3️⃣ Share links with anyone\n"
        "4️⃣ Use /myfiles to manage your files\n\n"
        "**Features:**\n"
        "✅ Browser streaming\n"
        "✅ Direct downloads\n"
        "✅ Secure hash-based URLs\n"
        "✅ Up to 4GB file support"
    )

@bot.on_message(filters.video & filters.private)
async def handle_video(client, message):
    if not db:
        await message.reply("❌ Database error! Contact admin.")
        return
    
    file_size = message.video.file_size
    if file_size > 4 * 1024 * 1024 * 1024:
        await message.reply("❌ File too large! Maximum 4GB allowed.")
        return
    
    status = await message.reply("⏳ Processing your video...")
    
    try:
        # Copy to log channel
        log_msg = await message.copy(LOG_CHANNEL_ID)
        logger.info(f"Copied video to log: {log_msg.id}")
        
        # Generate hash and URLs
        hash_val = get_hash(log_msg)
        file_name = get_name(log_msg)
        
        stream_url = f"{BASE_URL}/watch/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
        download_url = f"{BASE_URL}/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
        
        # Get shortlinks if enabled
        short_stream = await get_shortlink(stream_url)
        short_download = await get_shortlink(download_url)
        
        # Save to database
        await save_file_record(
            log_msg.id, message.from_user.id, file_name, file_size,
            log_msg.video.file_id, hash_val, short_stream, short_download
        )
        
        # Create response keyboard
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("▶️ Stream Online", url=short_stream or stream_url),
                InlineKeyboardButton("⬇️ Download", url=short_download or download_url)
            ],
            [
                InlineKeyboardButton("🗑️ Delete", callback_data=f"delete:{log_msg.id}"),
                InlineKeyboardButton("ℹ️ Info", callback_data=f"info:{log_msg.id}")
            ]
        ])
        
        await status.edit_text(
            f"✅ **Video Processed Successfully!**\n\n"
            f"📁 **Name:** `{file_name}`\n"
            f"📊 **Size:** `{format_size(file_size)}`\n\n"
            f"Click below to stream or download:",
            reply_markup=keyboard
        )
        logger.info(f"✅ Video processed: {file_name}")
        
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        await status.edit_text(f"❌ Error processing video: {str(e)}")

@bot.on_message(filters.document & filters.private)
async def handle_document(client, message):
    if not db:
        await message.reply("❌ Database error! Contact admin.")
        return
    
    file_size = message.document.file_size
    if file_size > 4 * 1024 * 1024 * 1024:
        await message.reply("❌ File too large! Maximum 4GB allowed.")
        return
    
    status = await message.reply("⏳ Processing your document...")
    
    try:
        # Copy to log channel
        log_msg = await message.copy(LOG_CHANNEL_ID)
        logger.info(f"Copied document to log: {log_msg.id}")
        
        # Generate hash and URLs
        hash_val = get_hash(log_msg)
        file_name = get_name(log_msg)
        
        download_url = f"{BASE_URL}/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
        short_download = await get_shortlink(download_url)
        
        # Save to database
        await save_file_record(
            log_msg.id, message.from_user.id, file_name, file_size,
            log_msg.document.file_id, hash_val, None, short_download
        )
        
        # Create response keyboard
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("⬇️ Download File", url=short_download or download_url)],
            [
                InlineKeyboardButton("🗑️ Delete", callback_data=f"delete:{log_msg.id}"),
                InlineKeyboardButton("ℹ️ Info", callback_data=f"info:{log_msg.id}")
            ]
        ])
        
        await status.edit_text(
            f"✅ **Document Processed Successfully!**\n\n"
            f"📁 **Name:** `{file_name}`\n"
            f"📊 **Size:** `{format_size(file_size)}`\n\n"
            f"Click below to download:",
            reply_markup=keyboard
        )
        logger.info(f"✅ Document processed: {file_name}")
        
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        await status.edit_text(f"❌ Error processing document: {str(e)}")

@bot.on_callback_query(filters.regex(r'(delete|info):\d+'))
async def handle_callback(client, query):
    try:
        action, msg_id_str = query.data.split(":")
        msg_id = int(msg_id_str)
        
        if action == "info":
            record = await files_collection.find_one({"log_msg_id": msg_id})
            if not record:
                await query.answer("❌ File not found in database", show_alert=True)
                return
            
            info_text = (
                f"📁 **File Information**\n\n"
                f"**Name:** `{record['file_name']}`\n"
                f"**Size:** `{format_size(record['file_size'])}`\n"
                f"**Uploaded:** `{record['date'].strftime('%Y-%m-%d %H:%M')}`\n"
                f"**Hash:** `{record['hash']}`"
            )
            await query.message.reply(info_text)
            await query.answer("✅ Info sent!")
        
        elif action == "delete":
            success = await delete_file_record(msg_id, query.from_user.id)
            if not success:
                await query.answer("❌ Unauthorized or file not found", show_alert=True)
                return
            
            try:
                await bot.delete_messages(LOG_CHANNEL_ID, msg_id)
            except:
                logger.warning(f"Could not delete message {msg_id} from log channel")
            
            await query.message.reply("✅ File deleted successfully!")
            await query.answer("✅ Deleted!")
    
    except Exception as e:
        logger.error(f"Callback error: {e}")
        await query.answer(f"❌ Error: {str(e)}", show_alert=True)

@bot.on_message(filters.command("myfiles") & filters.private)
async def cmd_myfiles(client, message):
    if not db:
        await message.reply("❌ Database error!")
        return
    
    try:
        cursor = files_collection.find({"uploader_id": message.from_user.id}).sort("date", -1).limit(10)
        records = await cursor.to_list(length=None)
        
        if not records:
            await message.reply("📭 **No files found!**\n\nSend me a video or document to get started.")
            return
        
        keyboard = []
        text = "📂 **Your Uploaded Files:**\n\n"
        
        for i, r in enumerate(records, 1):
            text += f"{i}. `{r['file_name'][:30]}...` - {format_size(r['file_size'])}\n"
            
            # Check if it's a video (has stream option)
            if r.get("short_stream") or "watch" in str(r.get("short_stream", "")):
                btns = [
                    InlineKeyboardButton("▶️", url=r.get("short_stream") or f"{BASE_URL}/watch/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}"),
                    InlineKeyboardButton("⬇️", url=r.get("short_download") or f"{BASE_URL}/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}")
                ]
            else:
                btns = [
                    InlineKeyboardButton("⬇️ Download", url=r.get("short_download") or f"{BASE_URL}/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}")
                ]
            keyboard.append(btns)
        
        text += f"\n*Showing last 10 files*"
        await message.reply(text, reply_markup=InlineKeyboardMarkup(keyboard))
        
    except Exception as e:
        logger.error(f"Myfiles error: {e}")
        await message.reply(f"❌ Error: {str(e)}")

@bot.on_message(filters.command("stats") & filters.user([OWNER_ID]) & filters.private)
async def cmd_stats(client, message):
    if not db:
        await message.reply("❌ Database error!")
        return
    
    try:
        total_files = await files_collection.count_documents({})
        total_size = 0
        
        if STORE_EXTRA:
            result = await files_collection.aggregate([
                {"$group": {"_id": None, "total_size": {"$sum": "$file_size"}}}
            ]).to_list(1)
            total_size = result[0]["total_size"] if result else 0
        
        total_users = 0
        if STORE_USER and users_collection:
            total_users = await users_collection.count_documents({})
        
        await message.reply(
            f"📊 **Bot Statistics**\n\n"
            f"📁 **Total Files:** `{total_files}`\n"
            f"💾 **Total Storage:** `{format_size(total_size)}`\n"
            f"👥 **Total Users:** `{total_users}`\n\n"
            f"⚡ Status: Running smoothly!"
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        await message.reply(f"❌ Error: {str(e)}")

# ==================== MAIN ====================

async def main():
    logger.info("🚀 Starting bot...")
    
    await bot.start()
    logger.info("✅ Bot is online!")
    
    # FastAPI server config
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        access_log=False
    )
    server = uvicorn.Server(config)
    
    logger.info(f"✅ Server starting on port {PORT}")
    
    # Run both services
    try:
        await asyncio.gather(
            idle(),
            server.serve()
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("⛔ Shutting down...")
    finally:
        await bot.stop()
        logger.info("✅ Stopped cleanly")

if __name__ == "__main__":
    asyncio.run(main())
