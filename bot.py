"""
Telegram File-to-Link Bot with FastAPI Web Server - FIXED VERSION

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
from pyrogram import Client, filters
from pyrogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
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
    
    logger.info(f"✅ Config loaded - API_ID: {API_ID}, PORT: {PORT}")
except Exception as e:
    logger.error(f"❌ Error loading config: {e}")

# MongoDB setup
try:
    mongo = AsyncIOMotorClient(MONGO_URL)
    db = mongo.filetolinks
    files_collection = db.files
    users_collection = db.users if STORE_USER else None
    logger.info("✅ MongoDB connected")
except Exception as e:
    logger.error(f"❌ MongoDB connection error: {e}")
    mongo = None
    db = None

# FastAPI app
app = FastAPI(title="Telegram File Bot", version="1.0")

# Pyrogram bot client
bot = Client(
    "file_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    workers=4
)

# ==================== UTILITY FUNCTIONS ====================

def get_name(log_msg: Message) -> str:
    """Get cleaned filename from log message."""
    name = None
    if log_msg.video:
        name = log_msg.video.file_name or f"file_{log_msg.id}"
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
    if not SHORTLINK_ENABLED:
        return None
    
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://shorten.example.com/create",
                    json={"url": url, "key": SHORTLINK_API_KEY}
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("short_url")
        except Exception as e:
            logger.error(f"Shortlink attempt {attempt+1} failed: {e}")
            await asyncio.sleep(2 ** attempt)
    return None

def format_size(bytes_size: int) -> str:
    """Human readable file size."""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = bytes_size
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.1f} {units[unit_idx]}"

def is_owner(user_id: int) -> bool:
    """Check if user is owner."""
    return user_id == OWNER_ID

async def save_file_record(
    log_msg_id: int,
    uploader_id: int,
    file_name: str,
    file_size: int,
    tg_file_id: str,
    hash_val: str,
    short_stream: Optional[str] = None,
    short_download: Optional[str] = None
):
    """Save file record to MongoDB."""
    if not db:
        logger.error("Database not connected")
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
        logger.info(f"✅ File record saved: {file_name}")
        
        if STORE_USER and users_collection:
            await users_collection.update_one(
                {"user_id": uploader_id},
                {
                    "$inc": {"file_count": 1},
                    "$setOnInsert": {"first_seen": datetime.utcnow()}
                },
                upsert=True
            )
    except Exception as e:
        logger.error(f"Error saving file record: {e}")

async def delete_file_record(log_msg_id: int, requester_id: int) -> bool:
    """Delete file record if allowed."""
    if not db:
        return False
    
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id})
        if not record:
            return False
        if not (is_owner(requester_id) or record["uploader_id"] == requester_id):
            return False
        await files_collection.delete_one({"log_msg_id": log_msg_id})
        logger.info(f"✅ File deleted: {log_msg_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return False

# ==================== FASTAPI ROUTES ====================

@app.head("/")
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "✅ Bot is running",
        "bot": "Telegram File-to-Link Bot",
        "version": "1.0"
    }

@app.head("/health")
@app.get("/health")
async def health():
    """Health check."""
    return {"status": "OK"}

@app.get("/meta/{log_msg_id}")
async def meta(log_msg_id: int):
    """Get file metadata."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id})
        if not record:
            raise HTTPException(status_code=404, detail="File not found")
        return {
            "file_name": record["file_name"],
            "file_size": record["file_size"],
            "uploader_id": record["uploader_id"],
            "mime_type": "video/mp4" if record["file_name"].endswith(".mp4") else "application/octet-stream",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /meta: {e}")
        raise HTTPException(status_code=500, detail="Server error")

@app.get("/watch/{log_msg_id}/{name}")
async def watch(log_msg_id: int, name: str, hash: str):
    """Watch video in browser."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
        if not record:
            raise HTTPException(status_code=403, detail="Invalid hash")
        
        file_name = record["file_name"]
        file_size = format_size(record["file_size"])
        uploader_id = record["uploader_id"]
        download_url = f"{BASE_URL}/{log_msg_id}/{name}?hash={hash}"
        stream_url = f"{BASE_URL}/stream/{log_msg_id}/{name}?hash={hash}"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{file_name}</title>
            <style>
                body {{ font-family: Arial; max-width: 800px; margin: 50px auto; }}
                h1 {{ color: #333; }}
                video {{ max-width: 100%; border: 1px solid #ddd; margin: 20px 0; }}
                .info {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
                a {{ color: #0066cc; text-decoration: none; margin-right: 15px; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>▶️ {file_name}</h1>
            <div class="info">
                <p><strong>Size:</strong> {file_size}</p>
                <p><strong>Uploader ID:</strong> {uploader_id}</p>
            </div>
            <video controls preload="metadata" style="width: 100%;">
                <source src="{stream_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div>
                <a href="{download_url}">⬇️ Download</a>
                <a href="javascript:window.close()">❌ Close</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /watch: {e}")
        raise HTTPException(status_code=500, detail="Server error")

@app.get("/stream/{log_msg_id}/{name}")
async def stream(log_msg_id: int, name: str, hash: str, request: Request):
    """Stream video file."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
        if not record:
            raise HTTPException(status_code=403, detail="Invalid hash")
        
        log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
        if not log_msg:
            raise HTTPException(status_code=404, detail="Message not found")
        
        total_size = log_msg.media.file_size
        range_header = request.headers.get("range")
        
        if range_header:
            parts = range_header.split('=')[1].split('-')
            start = int(parts[0] or 0)
            end = int(parts[1]) if len(parts) > 1 and parts[1] else total_size - 1
            new_len = end - start + 1
            
            async def range_iterator():
                async for chunk in bot.download_media(log_msg, in_memory=True, offset=start, limit=new_len):
                    yield chunk
            
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
                async for chunk in bot.download_media(log_msg, in_memory=True):
                    yield chunk
            
            return StreamingResponse(full_iterator(), headers={"Content-Type": "video/mp4"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /stream: {e}")
        raise HTTPException(status_code=500, detail="Server error")

@app.get("/{log_msg_id}/{name}")
async def download(log_msg_id: int, name: str, hash: str):
    """Download file."""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        record = await files_collection.find_one({"log_msg_id": log_msg_id, "hash": hash})
        if not record:
            raise HTTPException(status_code=403, detail="Invalid hash")
        
        file_name = record["file_name"]
        log_msg = await bot.get_messages(LOG_CHANNEL_ID, log_msg_id)
        if not log_msg:
            raise HTTPException(status_code=404, detail="Message not found")
        
        async def download_iterator():
            async for chunk in bot.download_media(log_msg, in_memory=True):
                yield chunk
        
        return StreamingResponse(
            download_iterator(),
            headers={
                "Content-Disposition": f'attachment; filename="{file_name}"',
                "Content-Type": "application/octet-stream"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download: {e}")
        raise HTTPException(status_code=500, detail="Server error")

# ==================== PYROGRAM BOT HANDLERS ====================

@bot.on_message(filters.command("start") & filters.private)
async def cmd_start(client: Client, message: Message):
    """Start command."""
    try:
        await message.reply(
            "🎬 **Telegram File-to-Link Bot**\n\n"
            "Send me videos or documents aur main tumhe stream/download links de dunga!\n\n"
            "Commands:\n"
            "/myfiles - Apni uploaded files dekho\n"
            "/help - Aur info"
        )
        logger.info(f"✅ Start command from {message.from_user.id}")
    except Exception as e:
        logger.error(f"Error in start command: {e}")

@bot.on_message(filters.command("help") & filters.private)
async def cmd_help(client: Client, message: Message):
    """Help command."""
    try:
        await message.reply(
            "📖 **Help**\n\n"
            "1️⃣ Video/file bhejo\n"
            "2️⃣ Links milenge stream/download ke liye\n"
            "3️⃣ /myfiles se apni files dekh sakta hai\n\n"
            "❓ Koi issue? Owner ko contact kar!"
        )
    except Exception as e:
        logger.error(f"Error in help command: {e}")

@bot.on_message(filters.video & filters.private)
async def handle_video(client: Client, message: Message):
    """Handle video upload."""
    try:
        if not db:
            await message.reply("❌ Database connection error!")
            return
        
        file_size = message.video.file_size
        if file_size > 4 * 1024 * 1024 * 1024:  # 4GB limit
            await message.reply("❌ File too large (>4GB)")
            return
        
        status_msg = await message.reply("⏳ Processing your file...")
        
        # Copy to log channel
        log_msg = await message.copy(LOG_CHANNEL_ID)
        logger.info(f"✅ File copied to log channel: {log_msg.id}")
        
        # Generate hash and name
        hash_val = get_hash(log_msg)
        file_name = get_name(log_msg)
        
        # Generate URLs
        stream_url = f"{BASE_URL}/watch/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
        download_url = f"{BASE_URL}/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
        
        # Try shortlinks if enabled
        short_stream = await get_shortlink(stream_url)
        short_download = await get_shortlink(download_url)
        
        # Save to database
        await save_file_record(
            log_msg_id=log_msg.id,
            uploader_id=message.from_user.id,
            file_name=file_name,
            file_size=file_size,
            tg_file_id=log_msg.video.file_id,
            hash_val=hash_val,
            short_stream=short_stream,
            short_download=short_download
        )
        
        # Create keyboard
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
        
        # Send response
        await status_msg.edit_text(
            f"✅ **File Processed!**\n\n"
            f"📁 Name: `{file_name}`\n"
            f"📊 Size: `{format_size(file_size)}`\n\n"
            f"Choose action:",
            reply_markup=keyboard
        )
        logger.info(f"✅ Video processed successfully: {file_name}")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        await message.reply(f"❌ Error: {str(e)}")

@bot.on_message(filters.document & filters.private)
async def handle_document(client: Client, message: Message):
    """Handle document upload."""
    try:
        if not db:
            await message.reply("❌ Database connection error!")
            return
        
        file_size = message.document.file_size
        if file_size > 4 * 1024 * 1024 * 1024:  # 4GB limit
            await message.reply("❌ File too large (>4GB)")
            return
        
        status_msg = await message.reply("⏳ Processing your file...")
        
        # Copy to log channel
        log_msg = await message.copy(LOG_CHANNEL_ID)
        logger.info(f"✅ Document copied: {log_msg.id}")
        
        # Generate hash and name
        hash_val = get_hash(log_msg)
        file_name = get_name(log_msg)
        
        # Generate URLs
        download_url = f"{BASE_URL}/{log_msg.id}/{quote_plus(file_name)}?hash={hash_val}"
        short_download = await get_shortlink(download_url)
        
        # Save to database
        await save_file_record(
            log_msg_id=log_msg.id,
            uploader_id=message.from_user.id,
            file_name=file_name,
            file_size=file_size,
            tg_file_id=log_msg.document.file_id,
            hash_val=hash_val,
            short_download=short_download
        )
        
        # Create keyboard
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("⬇️ Download", url=short_download or download_url)
            ],
            [
                InlineKeyboardButton("🗑️ Delete", callback_data=f"delete:{log_msg.id}"),
                InlineKeyboardButton("ℹ️ Info", callback_data=f"info:{log_msg.id}")
            ]
        ])
        
        await status_msg.edit_text(
            f"✅ **Document Processed!**\n\n"
            f"📁 Name: `{file_name}`\n"
            f"📊 Size: `{format_size(file_size)}`",
            reply_markup=keyboard
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        await message.reply(f"❌ Error: {str(e)}")

@bot.on_callback_query(filters.regex(r'(delete|info):\d+'))
async def handle_callback(client: Client, query: CallbackQuery):
    """Handle callback queries."""
    try:
        action, msg_id_str = query.data.split(":")
        msg_id = int(msg_id_str)
        
        if action == "info":
            if not db:
                await query.answer("Database error!")
                return
            
            record = await files_collection.find_one({"log_msg_id": msg_id})
            if not record:
                await query.answer("❌ File not found")
                return
            
            info_text = (
                f"📁 **File Info**\n\n"
                f"Name: `{record['file_name']}`\n"
                f"Size: `{format_size(record['file_size'])}`\n"
                f"Uploader ID: `{record['uploader_id']}`\n"
                f"Date: `{record['date']}`\n"
                f"Hash: `{record['hash']}`"
            )
            await query.message.reply(info_text)
            await query.answer("✅ Info sent")
        
        elif action == "delete":
            success = await delete_file_record(msg_id, query.from_user.id)
            if not success:
                await query.answer("❌ Unauthorized or file not found")
                return
            
            try:
                await bot.delete_messages(LOG_CHANNEL_ID, msg_id)
            except:
                logger.warning("Could not delete from log channel")
            
            await query.message.reply("✅ File deleted successfully!")
            await query.answer("✅ Deleted")
    
    except Exception as e:
        logger.error(f"Error in callback: {e}")
        await query.answer(f"❌ Error: {str(e)}")

@bot.on_message(filters.command("myfiles") & filters.private)
async def cmd_myfiles(client: Client, message: Message):
    """Show user's files."""
    if not db:
        await message.reply("❌ Database error!")
        return
    
    try:
        uploader_id = message.from_user.id
        cursor = files_collection.find({"uploader_id": uploader_id}).sort("date", -1).limit(10)
        records = await cursor.to_list(length=None)
        
        if not records:
            await message.reply("📭 No files found")
            return
        
        keyboard = []
        text = "📂 **Your Files:**\n\n"
        
        for i, r in enumerate(records, 1):
            text += f"{i}. `{r['file_name']}` - {format_size(r['file_size'])}\n"
            btns = [
                InlineKeyboardButton("▶️ Stream", url=r.get("short_stream") or f"{BASE_URL}/watch/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}"),
                InlineKeyboardButton("⬇️ DL", url=r.get("short_download") or f"{BASE_URL}/{r['log_msg_id']}/{quote_plus(r['file_name'])}?hash={r['hash']}")
            ]
            keyboard.append(btns)
        
        await message.reply(text, reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"Error in myfiles: {e}")
        await message.reply(f"❌ Error: {str(e)}")

# ==================== ADMIN COMMANDS ====================

@bot.on_message(filters.command("stats") & filters.user([OWNER_ID]) & filters.private)
async def cmd_stats(client: Client, message: Message):
    """Show statistics (owner only)."""
    if not db:
        await message.reply("❌ Database error!")
        return
    
    try:
        total_files = await files_collection.count_documents({})
        total_size = 0
        
        if STORE_EXTRA:
            pipeline = [{"$group": {"_id": None, "total_size": {"$sum": "$file_size"}}}]
            result = await files_collection.aggregate(pipeline).to_list(length=1)
            total_size = result[0]["total_size"] if result else 0
        
        total_users = 0
        if STORE_USER and users_collection:
            total_users = await users_collection.count_documents({})
        
        stats_text = (
            f"📊 **Bot Statistics**\n\n"
            f"📁 Total Files: `{total_files}`\n"
            f"📊 Total Size: `{format_size(total_size)}`\n"
            f"👥 Total Users: `{total_users}`"
        )
        await message.reply(stats_text)
    except Exception as e:
        logger.error(f"Error in stats: {e}")
        await message.reply(f"❌ Error: {str(e)}")

@bot.on_message(filters.command("help_admin") & filters.user([OWNER_ID]) & filters.private)
async def cmd_help_admin(client: Client, message: Message):
    """Admin help."""
    help_text = (
        "👑 **Admin Commands**\n\n"
        "/stats - Bot statistics\n"
        "/help_admin - This menu"
    )
    await message.reply(help_text)

# ==================== MAIN FUNCTION ====================

async def main():
    """Main function to run bot and server concurrently."""
    try:
        logger.info("🚀 Starting Telegram File Bot...")
        
        # Start bot first
        await bot.start()
        logger.info("✅ Bot started successfully!")
        
        # Configure FastAPI server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=PORT,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Create tasks for both services
        bot_task = asyncio.create_task(bot.idle())
        server_task = asyncio.create_task(server.serve())
        
        logger.info("✅ Both services are running!")
        
        # Run until interrupted
        await asyncio.gather(bot_task, server_task)
    
    except KeyboardInterrupt:
        logger.info("⛔ Shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
    finally:
        await bot.stop()
        logger.info("✅ Bot stopped cleanly")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⛔ Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
