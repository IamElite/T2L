"""
Telegram File-to-Link Bot - HEALTH-CHECK FRIENDLY VERSION
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
from pyrogram.errors import FloodWait
from pyrogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("pyrogram").setLevel(logging.ERROR)

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

logger.info(f"✅ Config loaded")

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

# FastAPI
app = FastAPI(title="File Bot")

# Bot status flag
bot_ready = False

# Pyrogram with session in /tmp (writable on Koyeb)
SESSION_FILE = "/tmp/bot_session"
bot = Client(SESSION_FILE, api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN, workers=4)

# ==================== UTILITIES ====================

def get_name(log_msg: Message) -> str:
    if log_msg.video:
        return (log_msg.video.file_name or f"file_{log_msg.id}").replace('/', '-').replace('\\', '-')
    elif log_msg.document:
        return (log_msg.document.file_name or f"file_{log_msg.id}").replace('/', '-').replace('\\', '-')
    return f"file_{log_msg.id}"

def get_hash(log_msg: Message) -> str:
    return hashlib.sha256(f"{log_msg.chat.id}:{log_msg.id}:{SECRET_KEY}".encode()).hexdigest()[:16]

async def get_shortlink(url: str) -> Optional[str]:
    if not SHORTLINK_ENABLED:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post("https://shorten.example.com/create", json={"url": url, "key": SHORTLINK_API_KEY})
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

async def save_file_record(log_msg_id, uploader_id, file_name, file_size, tg_file_id, hash_val, short_stream=None, short_download=None):
    if not db:
        return
    try:
        await files_collection.insert_one({
            "log_msg_id": log_msg_id, "uploader_id": uploader_id, "file_name": file_name,
            "file_size": file_size, "tg_file_id": tg_file_id, "hash": hash_val,
            "date": datetime.utcnow(), "short_stream": short_stream, "short_download": short_download
        })
        if STORE_USER and users_collection:
            await users_collection.update_one(
                {"user_id": uploader_id},
                {"$inc": {"file_count": 1}, "$setOnInsert": {"first_seen": datetime.utcnow()}},
                upsert=True
            )
    except Exception as e:
        logger.error(f"DB error: {e}")

async def delete_file_record(log_msg_id, requester_id):
    if not db:
        return False
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id})
        if not record or not (OWNER_ID == requester_id or record["uploader_id"] == requester_id):
            return False
        await files_collection.delete_one({"log_msg_id": log_msg_id})
        return True
    except:
        return False

# ==================== FASTAPI ROUTES ====================

@app.head("/")
@app.get("/")
async def root():
    return {"status": "running" if bot_ready else "starting", "bot": "File-to-Link Bot"}

@app.head("/health")
@app.get("/health")
async def health():
    return {"status": "OK", "bot_ready": bot_ready}

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
        raise HTTPException(403, "Invalid")
    
    file_name = record["file_name"]
    file_size = format_size(record["file_size"])
    download_url = f"{BASE_URL}/{log_msg_id}/{name}?hash={hash}"
    stream_url = f"{BASE_URL}/stream/{log_msg_id}/{name}?hash={hash}"
    
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>{file_name}</title><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>body{{font-family:Arial;max-width:900px;margin:50px auto;padding:20px;background:#f0f0f0}}h1{{color:#333}}
video{{width:100%;border:2px solid #333;margin:20px 0;background:#000}}.info{{background:#fff;padding:15px;border-radius:8px;margin:20px 0}}
a{{color:#fff;text-decoration:none;margin-right:15px;padding:12px 20px;background:#007bff;border-radius:5px;display:inline-block}}
a:hover{{background:#0056b3}}.download{{background:#28a745}}.download:hover{{background:#1e7e34}}</style></head>
<body><h1>▶️ {file_name}</h1><div class="info"><p><strong>Size:</strong> {file_size}</p></div>
<video controls preload="metadata"><source src="{stream_url}" type="video/mp4">Your browser does not support video.</video>
<div><a href="{download_url}" class="download">⬇️ Download</a><a href="javascript:window.close()">❌ Close</a></div></body></html>""")

@app.get("/stream/{log_msg_id}/{name}")
async def stream(log_msg_id: int, name: str, hash: str, request: Request):
    if not db or not bot_ready:
        raise HTTPException(503, "Service unavailable")
    
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
        if not record:
            raise HTTPException(403, "Invalid")
        
        log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
        if not log_msg or not (log_msg.video or log_msg.document):
            raise HTTPException(404, "Not found")
        
        total_size = log_msg.video.file_size if log_msg.video else log_msg.document.file_size
        range_header = request.headers.get("range")
        
        if range_header:
            byte_range = range_header.replace("bytes=", "").split("-")
            start = int(byte_range[0]) if byte_range[0] else 0
            end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else total_size - 1
            chunk_size = end - start + 1
            
            async def range_streamer():
                async for chunk in bot.stream_media(log_msg, offset=start, limit=chunk_size):
                    yield chunk
            
            return StreamingResponse(range_streamer(), status_code=206, headers={
                "Content-Range": f"bytes {start}-{end}/{total_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": "video/mp4"
            })
        else:
            async def full_streamer():
                async for chunk in bot.stream_media(log_msg):
                    yield chunk
            return StreamingResponse(full_streamer(), headers={"Content-Type": "video/mp4", "Accept-Ranges": "bytes"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(500, str(e))

@app.get("/{log_msg_id}/{name}")
async def download(log_msg_id: int, name: str, hash: str):
    if not db or not bot_ready:
        raise HTTPException(503, "Service unavailable")
    
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
        if not record:
            raise HTTPException(403, "Invalid")
        
        log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
        if not log_msg or not (log_msg.video or log_msg.document):
            raise HTTPException(404, "Not found")
        
        async def download_streamer():
            async for chunk in bot.stream_media(log_msg):
                yield chunk
        
        return StreamingResponse(download_streamer(), headers={
            "Content-Disposition": f'attachment; filename="{record["file_name"]}"',
            "Content-Type": "application/octet-stream"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(500, str(e))

# ==================== BOT HANDLERS ====================

@bot.on_message(filters.command("start") & filters.private)
async def cmd_start(client, message):
    await message.reply("🎬 **File-to-Link Bot**\n\nSend videos/documents for stream & download links!\n\n/myfiles - Your files\n/help - More info")

@bot.on_message(filters.command("help") & filters.private)
async def cmd_help(client, message):
    await message.reply("📖 **Usage:**\n1️⃣ Send file\n2️⃣ Get links\n3️⃣ /myfiles to view\n\n✅ Browser streaming\n✅ Direct downloads\n✅ Up to 4GB")

@bot.on_message(filters.video & filters.private)
async def handle_video(client, message):
    if not db:
        await message.reply("❌ DB error")
        return
    
    if message.video.file_size > 4 * 1024 * 1024 * 1024:
        await message.reply("❌ Max 4GB")
        return
    
    status = await message.reply("⏳ Processing...")
    
    try:
        log_msg = await message.copy(LOG_CHANNEL_ID)
        hash_val = get_hash(log_msg)
        file_name = get_name(log_msg)
        
        stream_url = f"{BASE_URL}/watch/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
        download_url = f"{BASE_URL}/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
        
        short_stream = await get_shortlink(stream_url)
        short_download = await get_shortlink(download_url)
        
        await save_file_record(log_msg.id, message.from_user.id, file_name, message.video.file_size,
                                log_msg.video.file_id, hash_val, short_stream, short_download)
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("▶️ Stream", url=short_stream or stream_url),
             InlineKeyboardButton("⬇️ Download", url=short_download or download_url)],
            [InlineKeyboardButton("🗑️ Delete", callback_data=f"delete:{log_msg.id}"),
             InlineKeyboardButton("ℹ️ Info", callback_data=f"info:{log_msg.id}")]
        ])
        
        await status.edit_text(
            f"✅ **Done!**\n\n📁 `{file_name}`\n📊 `{format_size(message.video.file_size)}`",
            reply_markup=keyboard
        )
    except Exception as e:
        logger.error(f"Video error: {e}")
        await status.edit_text(f"❌ Error: {str(e)}")

@bot.on_message(filters.document & filters.private)
async def handle_document(client, message):
    if not db:
        await message.reply("❌ DB error")
        return
    
    if message.document.file_size > 4 * 1024 * 1024 * 1024:
        await message.reply("❌ Max 4GB")
        return
    
    status = await message.reply("⏳ Processing...")
    
    try:
        log_msg = await message.copy(LOG_CHANNEL_ID)
        hash_val = get_hash(log_msg)
        file_name = get_name(log_msg)
        
        download_url = f"{BASE_URL}/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
        short_download = await get_shortlink(download_url)
        
        await save_file_record(log_msg.id, message.from_user.id, file_name, message.document.file_size,
                                log_msg.document.file_id, hash_val, None, short_download)
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("⬇️ Download", url=short_download or download_url)],
            [InlineKeyboardButton("🗑️ Delete", callback_data=f"delete:{log_msg.id}"),
             InlineKeyboardButton("ℹ️ Info", callback_data=f"info:{log_msg.id}")]
        ])
        
        await status.edit_text(
            f"✅ **Done!**\n\n📁 `{file_name}`\n📊 `{format_size(message.document.file_size)}`",
            reply_markup=keyboard
        )
    except Exception as e:
        logger.error(f"Document error: {e}")
        await status.edit_text(f"❌ Error: {str(e)}")

@bot.on_callback_query(filters.regex(r'(delete|info):\d+'))
async def handle_callback(client, query):
    action, msg_id = query.data.split(":")
    msg_id = int(msg_id)
    
    if action == "info":
        record = await files_collection.find_one({"log_msg_id": msg_id})
        if not record:
            await query.answer("❌ Not found", show_alert=True)
            return
        await query.message.reply(f"📁 **Info**\n\nName: `{record['file_name']}`\nSize: `{format_size(record['file_size'])}`\nDate: `{record['date']}`")
        await query.answer("✅")
    elif action == "delete":
        if await delete_file_record(msg_id, query.from_user.id):
            try:
                await bot.delete_messages(LOG_CHANNEL_ID, msg_id)
            except:
                pass
            await query.message.reply("✅ Deleted!")
            await query.answer("✅")
        else:
            await query.answer("❌ Unauthorized", show_alert=True)

@bot.on_message(filters.command("myfiles") & filters.private)
async def cmd_myfiles(client, message):
    if not db:
        await message.reply("❌ DB error")
        return
    
    records = await files_collection.find({"uploader_id": message.from_user.id}).sort("date", -1).limit(10).to_list(None)
    
    if not records:
        await message.reply("📭 No files")
        return
    
    keyboard = []
    text = "📂 **Your Files:**\n\n"
    
    for i, r in enumerate(records, 1):
        text += f"{i}. `{r['file_name'][:30]}` - {format_size(r['file_size'])}\n"
        if r.get("short_stream"):
            keyboard.append([
                InlineKeyboardButton("▶️", url=r.get("short_stream") or f"{BASE_URL}/watch/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}"),
                InlineKeyboardButton("⬇️", url=r.get("short_download") or f"{BASE_URL}/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}")
            ])
        else:
            keyboard.append([InlineKeyboardButton("⬇️", url=r.get("short_download") or f"{BASE_URL}/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}")])
    
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
    
    total_users = await users_collection.count_documents({}) if STORE_USER and users_collection else 0
    
    await message.reply(f"📊 **Stats**\n\n📁 Files: `{total_files}`\n💾 Storage: `{format_size(total_size)}`\n👥 Users: `{total_users}`")

# ==================== MAIN WITH BACKGROUND BOT START ====================

async def start_bot_background():
    """Start bot in background with FloodWait handling"""
    global bot_ready
    
    retry_delay = 10
    max_delay = 3600  # Max 1 hour wait
    
    while True:
        try:
            logger.info("🤖 Attempting to start bot...")
            await bot.start()
            bot_ready = True
            logger.info("✅ Bot started successfully!")
            await idle()
            break
        except FloodWait as e:
            logger.warning(f"⏰ FloodWait: {e.value}s ({e.value//60} min)")
            await asyncio.sleep(e.value + 10)
        except Exception as e:
            logger.error(f"❌ Bot start error: {e}")
            await asyncio.sleep(min(retry_delay, max_delay))
            retry_delay *= 2

async def main():
    logger.info("🚀 Starting services...")
    
    # Start FastAPI server first (for health checks)
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    
    # Run server and bot concurrently
    await asyncio.gather(
        server.serve(),
        start_bot_background()
    )

if __name__ == "__main__":
    asyncio.run(main())
