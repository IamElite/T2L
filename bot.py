"""
Telegram File-to-Link Bot with FastAPI - PYTHON 3.10 COMPATIBLE
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

# Clean logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable verbose Pyrogram logs
logging.getLogger("pyrogram").setLevel(logging.WARNING)
logging.getLogger("pyrogram.session").setLevel(logging.ERROR)
logging.getLogger("pyrogram.connection").setLevel(logging.ERROR)

# Load config
try:
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
except Exception as e:
    logger.error(f"❌ Config error: {e}")

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
        <style>
            body {{ font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            video {{ width: 100%; border: 1px solid #ddd; margin: 20px 0; }}
            .info {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            a {{ color: #0066cc; text-decoration: none; margin-right: 20px; padding: 10px 15px; background: #e8e8e8; border-radius: 5px; display: inline-block; }}
            a:hover {{ background: #d0d0d0; }}
        </style>
    </head>
    <body>
        <h1>▶️ {file_name}</h1>
        <div class="info">
            <p><strong>Size:</strong> {file_size}</p>
        </div>
        <video controls preload="metadata">
            <source src="{stream_url}" type="video/mp4">
            Your browser does not support video.
        </video>
        <div>
            <a href="{download_url}">⬇️ Download</a>
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
    
    record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
    if not record:
        raise HTTPException(403, "Invalid")
    
    log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
    if not log_msg:
        raise HTTPException(404, "Not found")
    
    total_size = log_msg.video.file_size if log_msg.video else log_msg.document.file_size
    range_header = request.headers.get("range")
    
    if range_header:
        parts = range_header.split('=')[1].split('-')
        start = int(parts[0] or 0)
        end = int(parts[1]) if len(parts) > 1 and parts[1] else total_size - 1
        new_len = end - start + 1
        
        async def range_iterator():
            chunk_size = 256 * 1024
            offset = start
            remaining = new_len
            while remaining > 0:
                chunk = min(chunk_size, remaining)
                async for data in bot.stream_media(log_msg, offset=offset, limit=chunk):
                    yield data
                    offset += len(data)
                    remaining -= len(data)
        
        return StreamingResponse(
            range_iterator(),
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{total_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(new_len),
                "Content-Type": "video/mp4"
            }
        )
    else:
        async def full_iterator():
            async for chunk in bot.stream_media(log_msg):
                yield chunk
        
        return StreamingResponse(full_iterator(), headers={"Content-Type": "video/mp4"})

@app.get("/{log_msg_id}/{name}")
async def download(log_msg_id: int, name: str, hash: str):
    if not db:
        raise HTTPException(503, "DB unavailable")
    
    record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
    if not record:
        raise HTTPException(403, "Invalid")
    
    file_name = record["file_name"]
    log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
    if not log_msg:
        raise HTTPException(404, "Not found")
    
    async def download_iterator():
        async for chunk in bot.stream_media(log_msg):
            yield chunk
    
    return StreamingResponse(
        download_iterator(),
        headers={
            "Content-Disposition": f'attachment; filename="{file_name}"',
            "Content-Type": "application/octet-stream"
        }
    )

# ==================== BOT HANDLERS ====================

@bot.on_message(filters.command("start") & filters.private)
async def cmd_start(client, message):
    await message.reply(
        "🎬 **File-to-Link Bot**\n\n"
        "Send me videos/documents!\n\n"
        "/myfiles - Your files\n"
        "/help - More info"
    )
    logger.info(f"Start: {message.from_user.id}")

@bot.on_message(filters.command("help") & filters.private)
async def cmd_help(client, message):
    await message.reply(
        "📖 **Help**\n\n"
        "1. Send file\n"
        "2. Get stream/download links\n"
        "3. /myfiles to see your files"
    )

@bot.on_message(filters.video & filters.private)
async def handle_video(client, message):
    if not db:
        await message.reply("❌ DB error")
        return
    
    file_size = message.video.file_size
    if file_size > 4 * 1024 * 1024 * 1024:
        await message.reply("❌ File > 4GB")
        return
    
    status = await message.reply("⏳ Processing...")
    
    log_msg = await message.copy(LOG_CHANNEL_ID)
    hash_val = get_hash(log_msg)
    file_name = get_name(log_msg)
    
    stream_url = f"{BASE_URL}/watch/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
    download_url = f"{BASE_URL}/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
    
    short_stream = await get_shortlink(stream_url)
    short_download = await get_shortlink(download_url)
    
    await save_file_record(
        log_msg.id, message.from_user.id, file_name, file_size,
        log_msg.video.file_id, hash_val, short_stream, short_download
    )
    
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶️ Stream", url=short_stream or stream_url),
            InlineKeyboardButton("⬇️ Download", url=short_download or download_url)
        ],
        [
            InlineKeyboardButton("🗑️ Delete", callback_data=f"delete:{log_msg.id}"),
            InlineKeyboardButton("ℹ️ Info", callback_data=f"info:{log_msg.id}")
        ]
    ])
    
    await status.edit_text(
        f"✅ **Done!**\n\n"
        f"📁 {file_name}\n"
        f"📊 {format_size(file_size)}",
        reply_markup=keyboard
    )
    logger.info(f"Processed: {file_name}")

@bot.on_message(filters.document & filters.private)
async def handle_document(client, message):
    if not db:
        await message.reply("❌ DB error")
        return
    
    file_size = message.document.file_size
    if file_size > 4 * 1024 * 1024 * 1024:
        await message.reply("❌ File > 4GB")
        return
    
    status = await message.reply("⏳ Processing...")
    
    log_msg = await message.copy(LOG_CHANNEL_ID)
    hash_val = get_hash(log_msg)
    file_name = get_name(log_msg)
    
    download_url = f"{BASE_URL}/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
    short_download = await get_shortlink(download_url)
    
    await save_file_record(
        log_msg.id, message.from_user.id, file_name, file_size,
        log_msg.document.file_id, hash_val, None, short_download
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("⬇️ Download", url=short_download or download_url)],
        [
            InlineKeyboardButton("🗑️ Delete", callback_data=f"delete:{log_msg.id}"),
            InlineKeyboardButton("ℹ️ Info", callback_data=f"info:{log_msg.id}")
        ]
    ])
    
    await status.edit_text(
        f"✅ **Done!**\n\n"
        f"📁 {file_name}\n"
        f"📊 {format_size(file_size)}",
        reply_markup=keyboard
    )

@bot.on_callback_query(filters.regex(r'(delete|info):\d+'))
async def handle_callback(client, query):
    action, msg_id_str = query.data.split(":")
    msg_id = int(msg_id_str)
    
    if action == "info":
        record = await files_collection.find_one({"log_msg_id": msg_id})
        if not record:
            await query.answer("❌ Not found")
            return
        
        info_text = (
            f"📁 **Info**\n\n"
            f"Name: `{record['file_name']}`\n"
            f"Size: `{format_size(record['file_size'])}`\n"
            f"Date: `{record['date']}`"
        )
        await query.message.reply(info_text)
        await query.answer("✅")
    
    elif action == "delete":
        success = await delete_file_record(msg_id, query.from_user.id)
        if not success:
            await query.answer("❌ Unauthorized")
            return
        
        try:
            await bot.delete_messages(LOG_CHANNEL_ID, msg_id)
        except:
            pass
        
        await query.message.reply("✅ Deleted!")
        await query.answer("✅")

@bot.on_message(filters.command("myfiles") & filters.private)
async def cmd_myfiles(client, message):
    if not db:
        await message.reply("❌ DB error")
        return
    
    cursor = files_collection.find({"uploader_id": message.from_user.id}).sort("date", -1).limit(10)
    records = await cursor.to_list(length=None)
    
    if not records:
        await message.reply("📭 No files")
        return
    
    keyboard = []
    text = "📂 **Your Files:**\n\n"
    
    for i, r in enumerate(records, 1):
        text += f"{i}. `{r['file_name']}` - {format_size(r['file_size'])}\n"
        btns = [
            InlineKeyboardButton("▶️", url=r.get("short_stream") or f"{BASE_URL}/watch/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}"),
            InlineKeyboardButton("⬇️", url=r.get("short_download") or f"{BASE_URL}/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}")
        ]
        keyboard.append(btns)
    
    await message.reply(text, reply_markup=InlineKeyboardMarkup(keyboard))

@bot.on_message(filters.command("stats") & filters.user([OWNER_ID]) & filters.private)
async def cmd_stats(client, message):
    if not db:
        await message.reply("❌ DB error")
        return
    
    total_files = await files_collection.count_documents({})
    total_size = 0
    
    if STORE_EXTRA:
        result = await files_collection.aggregate([{"$group": {"_id": None, "total_size": {"$sum": "$file_size"}}}]).to_list(1)
        total_size = result[0]["total_size"] if result else 0
    
    total_users = 0
    if STORE_USER and users_collection:
        total_users = await users_collection.count_documents({})
    
    await message.reply(
        f"📊 **Stats**\n\n"
        f"Files: `{total_files}`\n"
        f"Size: `{format_size(total_size)}`\n"
        f"Users: `{total_users}`"
    )

# ==================== MAIN ====================

async def main():
    logger.info("🚀 Starting bot...")
    
    await bot.start()
    logger.info("✅ Bot running!")
    
    # Configure server
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    
    # Run both together with asyncio.gather (Python 3.10 compatible)
    try:
        await asyncio.gather(
            idle(),
            server.serve()
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("⛔ Shutting down...")
    finally:
        await bot.stop()
        logger.info("✅ Stopped")

if __name__ == "__main__":
    asyncio.run(main())
