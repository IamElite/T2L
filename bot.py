"""
Telegram File-to-Link Bot with FastAPI Web Server

README:
Install dependencies: pip install -r requirements.txt

Environment Variables (set in Koyeb env):
- API_ID: Telegram API ID
- API_HASH: Telegram API Hash
- BOT_TOKEN: Bot token from BotFather
- MONGO_URL: MongoDB connection string
- OWNER_ID: Telegram user ID of bot owner (int)
- LOG_CHANNEL_ID: Channel ID to store files (e.g., -1001234567890)
- BASE_URL: Public base URL (e.g., https://your-app.koyeb.app/)
- SHORTLINK_ENABLED: "true" or "false" (default false if not set)
- SHORTLINK_API_KEY: API key for shortlink service (if enabled)
- SECRET_KEY: Random secret string for hash generation
- STORE_EXTRA: "true" or "false" (default false)
- STORE_USER: "true" or "false" (default false)
- PORT: Port for server (default 8080 if not set for local dev)

Koyeb Deployment Tips:
- Use Dockerfile or direct Python runner.
- Start command: python3 bot.py
- Logs go to stdout.
- For production, ensure MongoDB is connected via MONGO_URL.

Sample Flow:
User sends movie.mp4 -> bot copies to LOG_CHANNEL_ID -> generates URLs with hash
- Stream: BASE_URL/watch/12345/movie.mp4?hash=abcd...
- Download: BASE_URL/12345/movie.mp4?hash=abcd...
If SHORTLINK_ENABLED, short links saved.
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
from urllib.parse import quote_plus

import httpx
from motor.motor_asyncio import AsyncIOMotorClient
from pyrogram import Client, filters
from pyrogram.errors import MessageIdInvalid, ChannelInvalid
from pyrogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load env vars
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")
MONGO_URL = os.getenv("MONGO_URL")
OWNER_ID = int(os.getenv("OWNER_ID"))
LOG_CHANNEL_ID = int(os.getenv("LOG_CHANNEL_ID"))
BASE_URL = os.getenv("BASE_URL").rstrip("/")
SHORTLINK_ENABLED = os.getenv("SHORTLINK_ENABLED", "false").lower() == "true"
SHORTLINK_API_KEY = os.getenv("SHORTLINK_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
STORE_EXTRA = os.getenv("STORE_EXTRA", "false").lower() == "true"
STORE_USER = os.getenv("STORE_USER", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8080))

# MongoDB setup
mongo = AsyncIOMotorClient(MONGO_URL)
db = mongo.filetolinks
files_collection = db.files
users_collection = db.users if STORE_USER else None

# FastAPI app
app = FastAPI(root_path="/" if BASE_URL.endswith("/") else "")

# Pyrogram client
bot = Client("bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

def get_name(log_msg: Message) -> str:
    """Get cleaned filename from log message."""
    if log_msg.video:
        name = log_msg.video.file_name or log_msg.document.file_name or f"file_{log_msg.id}"
    elif log_msg.document:
        name = log_msg.document.file_name or f"file_{log_msg.id}"
    else:
        name = f"file_{log_msg.id}"
    return name.replace('/', '-').replace('\\', '-')

def get_hash(log_msg: Message) -> str:
    """Generate deterministic secret-backed hash."""
    key = f"{log_msg.chat.id}:{log_msg.id}:{SECRET_KEY}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]

async def get_shortlink(url: str) -> Optional[str]:
    """Async wrapper for shortlink with retries."""
    import time
    import random

    def backoff(tries: int) -> float:
        return random.uniform(1, min(2 ** tries, 60))

    # Placeholder - implement actual API call
    for _ in range(3):
        try:
            # Example: use SHORTLINK_API_KEY to shorten url
            async with httpx.AsyncClient() as client:
                resp = await client.post("https://shorten.example.com/create", json={
                    "url": url,
                    "key": SHORTLINK_API_KEY
                }, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return data.get("short_url")
        except Exception as e:
            logging.error(f"Shortlink failed: {e}")
            await asyncio.sleep(backoff(_))
    return None

def format_size(bytes: int) -> str:
    """Human readable file size."""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = bytes
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.1f} {units[unit_idx]}"

def is_owner(user_id: int) -> bool:
    """Check if user is owner."""
    return user_id == OWNER_ID

async def save_file_record(log_msg_id: int, uploader_id: int, file_name: str, file_size: int, tg_file_id: str, hash: str, short_stream: Optional[str] = None, short_download: Optional[str] = None):
    """Save minimal file record to Mongo."""
    doc = {
        "log_msg_id": log_msg_id,
        "uploader_id": uploader_id,
        "file_name": file_name,
        "file_size": file_size,
        "tg_file_id": tg_file_id,
        "hash": hash,
        "date": datetime.utcnow(),
    }
    if STORE_EXTRA or short_stream:
        doc["short_stream"] = short_stream
    if STORE_EXTRA or short_download:
        doc["short_download"] = short_download
    await files_collection.insert_one(doc)

    if STORE_USER:
        await users_collection.update_one(
            {"user_id": uploader_id},
            {"$inc": {"file_count": 1}, "$setOnInsert": {"first_seen": datetime.utcnow()}},
            upsert=True
        )

async def delete_file_record(log_msg_id: int, requester_id: int) -> bool:
    """Delete file record if allowed."""
    record = await files_collection.find_one({"log_msg_id": log_msg_id})
    if not record:
        return False
    if not (is_owner(requester_id) or record["uploader_id"] == requester_id):
        return False
    await files_collection.delete_one({"log_msg_id": log_msg_id})
    return True

async def stream_telegram_file(log_msg_id: int, range_header: Optional[str] = None):
    """Generator to stream file from Telegram with range support."""
    log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
    if not log_msg:
        raise HTTPException(status_code=404, detail="File not found")

    total_size = log_msg.media.file_size

    start = 0
    end = total_size - 1
    if range_header:
        # Parse range: bytes=start-end
        parts = range_header.split('=')[1].split('-')
        start = int(parts[0] or 0)
        end = int(parts[1]) if len(parts) > 1 and parts[1] else total_size - 1

    chunk_size = 64 * 1024  # 64KB

    async for chunk in log_msg.media.download_iter(start=start, size=end - start + 1 if end < total_size else total_size - start, chunk_size=chunk_size):
        yield chunk

# FastAPI routes
@app.get("/health")
async def health():
    return {"status": "OK"}

@app.get("/meta/{log_msg_id}")
async def meta(log_msg_id: int):
    record = await files_collection.find_one({"log_msg_id": log_msg_id})
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "file_name": record["file_name"],
        "file_size": record["file_size"],
        "uploader_id": record["uploader_id"],
        "mime_type": "video/mp4" if record["file_name"].endswith(".mp4") else "application/octet-stream",
        "hls_manifest": None  # Optional, set to URL if HLS available
    }

@app.get("/meta/{log_msg_id}/thumb")
async def thumb(log_msg_id: int):
    # Poster not implemented, return 404
    raise HTTPException(status_code=404, detail="No thumbnail available")

@app.get("/watch/{log_msg_id}/{name}")
async def watch(log_msg_id: int, name: str, hash: str):
    record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
    if not record:
        raise HTTPException(status_code=403, detail="Invalid hash")

    file_name = record["file_name"]
    file_size = format_size(record["file_size"])
    uploader_id = record["uploader_id"]
    download_url = f"{BASE_URL}/{log_msg_id}/{name}?hash={hash}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>{file_name}</title></head>
    <body>
        <h1>{file_name}</h1>
        <p>Size: {file_size}</p>
        <p>Uploader: {uploader_id}</p>
        <video controls preload="metadata" style="max-width: 100%;">
            <source src="{BASE_URL}/stream/{log_msg_id}/{name}?hash={hash}" type="video/mp4">
        </video>
        <p><a href="{download_url}">Download</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/stream/{log_msg_id}/{name}")
async def stream(log_msg_id: int, name: str, hash: str, request: Request):
    record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
    if not record:
        raise HTTPException(status_code=403, detail="Invalid hash")

    log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
    total_size = log_msg.media.file_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse range
        parts = range_header.split('=')[1].split('-')
        start = int(parts[0] or 0)
        end = int(parts[1]) if len(parts) > 1 and parts[1] else total_size - 1
        new_len = end - start + 1

        def iterator():
            async for chunk in log_msg.media.download_iter(start=start, size=new_len, chunk_size=64*1024):
                yield chunk

        return StreamingResponse(iterator(), status_code=206, headers={
            "Content-Range": f"bytes {start}-{end}/{total_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(new_len),
            "Content-Type": "video/mp4"
        })
    else:
        def iterator():
            async for chunk in log_msg.media.download_iter(chunk_size=64*1024):
                yield chunk

        return StreamingResponse(iterator(), headers={"Content-Type": "video/mp4"})

@app.get("/{log_msg_id}/{name}")
async def download(log_msg_id: int, name: str, hash: str):
    record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
    if not record:
        raise HTTPException(status_code=403, detail="Invalid hash")

    file_name = record["file_name"]

    def iterator():
        async for chunk in stream_telegram_file(log_msg_id):
            yield chunk

    return StreamingResponse(iterator(), headers={
        "Content-Disposition": f"attachment; filename=\"{file_name}\"",
        "Content-Type": "application/octet-stream"
    })

# Pyrogram handlers
@bot.on_message(filters.video & filters.private)
async def handle_video(client: Client, message: Message):
    if message.video.file_size > 4 * 1024 * 1024 * 1024:  # 4GB
        await message.reply("File too large (>4GB)")
        return

    logging.info("File received")
    log_msg = await message.copy(LOG_CHANNEL_ID)
    logging.info("Copied to log channel")

    hash_val = get_hash(log_msg)
    name = get_name(log_msg)

    stream_url = f"{BASE_URL}/watch/{log_msg.id}/{quote_plus(name)}?hash={hash_val}"
    download_url = f"{BASE_URL}/{log_msg.id}/{quote_plus(name)}?hash={hash_val}"

    short_stream = None
    short_download = None
    if SHORTLINK_ENABLED:
        logging.info("Creating shortlinks")
        short_stream = await get_shortlink(stream_url)
        short_download = await get_shortlink(download_url)

    await save_file_record(
        log_msg_id=log_msg.id,
        uploader_id=message.from_user.id,
        file_name=name,
        file_size=log_msg.video.file_size,
        tg_file_id=log_msg.media.file_id,
        hash=hash_val,
        short_stream=short_stream,
        short_download=short_download
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

    await message.reply("✅ File processed!\n▶️ Stream   ⬇️ Download   🗑️ Delete   ℹ️ Info", reply_markup=keyboard)

@bot.on_callback_query(filters.regex(r'(delete|info):\d+'))
async def handle_callback(client: Client, query: CallbackQuery):
    action, msg_id_str = query.data.split(":")
    msg_id = int(msg_id_str)

    if action == "info":
        record = await files_collection.find_one({"log_msg_id": msg_id})
        if not record:
            await query.answer("File not found")
            return
        link = f"https://t.me/c/{str(LOG_CHANNEL_ID).lstrip('-')}/{msg_id}"
        info = f"Name: {record['file_name']}\nSize: {format_size(record['file_size'])}\nUploader: {record['uploader_id']}\nLink: {link}\nDate: {record['date']}"
        await query.message.reply(info)
        await query.answer()

    elif action == "delete":
        success = await delete_file_record(msg_id, query.from_user.id)
        if not success:
            await query.answer("Unauthorized")
            return
        try:
            await bot.delete_messages(LOG_CHANNEL_ID, msg_id)
            await query.message.reply("Deleted")
            logging.info("File deleted")
        except:
            logging.error("Failed to delete message")
        await query.answer("Deleted")

# Admin commands
@bot.on_message(filters.command("stats") & filters.user([OWNER_ID]))
async def cmd_stats(client: Client, message: Message):
    total_files = await files_collection.count_documents({})
    total_size = 0
    if STORE_EXTRA:
        pipeline = [{"$group": {"_id": None, "total_size": {"$sum": "$file_size"}}}]
        result = await files_collection.aggregate(pipeline).to_list(length=1)
        total_size = result[0]["total_size"] if result else 0
    total_users = 0 if not STORE_USER else await users_collection.count_documents({})
    await message.reply(f"Files: {total_files}\nSize: {format_size(total_size)}\nUsers: {total_users}")

@bot.on_message(filters.command("users") & filters.user([OWNER_ID]))
async def cmd_users(client: Client, message: Message):
    if not STORE_USER:
        await message.reply("User tracking disabled")
        return
    cursor = users_collection.find().sort("file_count", -1).limit(50)
    users_list = await cursor.to_list(length=None)
    text = "\n".join([f"{u['user_id']}: {u['file_count']}" for u in users_list])
    await message.reply(text or "No users")

@bot.on_message(filters.command("broadcast") & filters.user([OWNER_ID]))
async def cmd_broadcast(client: Client, message: Message):
    if not STORE_USER:
        return
    text = message.text.split(" ", 1)[1] if " " in message.text else None
    if not text:
        return
    cursor = users_collection.find({}, {"user_id": 1})
    users_list = await cursor.to_list(length=None)
    users = [u["user_id"] for u in users_list]
    sent = 0
    for uid in users:
        try:
            await client.send_message(uid, text)
            sent += 1
            await asyncio.sleep(0.05)
        except:
            pass
    await message.reply(f"Broadcasted to {sent}/{len(users)} users")

@bot.on_message(filters.command("restart") & filters.user([OWNER_ID]))
async def cmd_restart(client: Client, message: Message):
    await message.reply("Restarting...")
    import sys
    sys.exit(0)

@bot.on_message(filters.command("help_admin") & filters.user([OWNER_ID]))
async def cmd_help_admin(client: Client, message: Message):
    help_text = "/stats - file stats\n/users - top users\n/broadcast <msg> - send to users\n/restart - shutdown\n"
    await message.reply(help_text)

# User commands
@bot.on_message(filters.command("start"))
async def cmd_start(client: Client, message: Message):
    await message.reply("Send me videos and I'll give you stream/download links!")

@bot.on_message(filters.command("myfiles"))
async def cmd_myfiles(client: Client, message: Message):
    uploader_id = message.from_user.id
    cursor = files_collection.find({"uploader_id": uploader_id}).sort("date", -1).limit(10)  # Simple pagination, limit 10
    records_list = await cursor.to_list(length=None)
    if not records_list:
        await message.reply("No files")
        return
    text = ""
    keyboard = []
    for r in records_list:
        text += f"{r['file_name']} - {format_size(r['file_size'])}\n"
        btns = [
            InlineKeyboardButton("Stream", url=r.get("short_stream") or f"{BASE_URL}/watch/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}"),
            InlineKeyboardButton("Download", url=r.get("short_download") or f"{BASE_URL}/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}")
        ]
        keyboard.append(btns)
    keyboard.append([InlineKeyboardButton("Close", callback_data="close")])  # Placeholder
    await message.reply(text, reply_markup=InlineKeyboardMarkup(keyboard))

# Main
async def start_both():
    await mongo.connect()
    await bot.start()
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(start_both())

# Pytest skeletons
import pytest

def test_get_name():
    # Mock Message
    pass

def test_get_hash():
    # Assert hash length and deterministic
    pass

@pytest.mark.asyncio
async def test_save_file_record():
    pass

@pytest.mark.asyncio
async def test_delete_file_record():
    pass
