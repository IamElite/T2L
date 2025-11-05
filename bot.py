import os
import logging
from pyrogram import Client, filters
from pyrogram.types import Message
from pymongo import MongoClient
from flask import Flask, render_template_string
from threading import Thread
import time

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
API_ID = int(os.getenv("API_ID", "24168862"))
API_HASH = os.getenv("API_HASH", "916a9424dd1e58ab7955001ccc0172b3")
BOT_TOKEN = os.getenv("BOT_TOKEN", "8017518988:AAGV3PhzhDtJjxcjC-CaJtMoByNM4x39kR0")
BASE_URL = os.getenv("BASE_URL", "era-steamer.koyeb.app").rstrip("/")
LOG_CHANNEL_ID = int(os.getenv("LOG_CHANNEL_ID", "-1003119165774"))
MONGO_URL = os.getenv("MONGO_URL")
OWNER_ID = int(os.getenv("OWNER_ID", "1679112664"))
PORT = int(os.getenv("PORT", 8080))

# Pyrogram Client
app_bot = Client("StreamBot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# Flask Server
flask_app = Flask(__name__)

# MongoDB
try:
    mongo_client = MongoClient(MONGO_URL)
    db = mongo_client["stream_bot"]
    files_collection = db["files"]
    logger.info("✅ MongoDB Connected!")
except Exception as e:
    logger.error(f"❌ MongoDB Error: {e}")
    files_collection = None

# HTML PLAYER INTERFACE
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stream Bot Player</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            max-width: 800px;
            width: 90%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .player-container {
            background: #000;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        video {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .info-box h3 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .info-box p {
            color: #666;
            word-break: break-all;
            font-size: 0.95em;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s;
            font-weight: bold;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .feature-card h4 {
            margin-bottom: 5px;
        }
        
        .feature-card p {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .status {
            text-align: center;
            padding: 15px;
            margin-top: 20px;
            border-radius: 8px;
            display: none;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            display: block;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Stream Bot Player</h1>
            <p style="color: #999;">Fast & Easy Streaming</p>
        </div>
        
        <div class="player-container">
            <video id="player" controls>
                <source src="" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <div class="input-group">
            <input 
                type="text" 
                id="fileId" 
                placeholder="Telegram File ID paste karo..."
                autocomplete="off"
            >
            <button onclick="loadStream()">▶️ Play</button>
        </div>
        
        <div id="status" class="status"></div>
        
        <div class="info-box">
            <h3>📋 Kaise use kare?</h3>
            <p>1. Bot ko file bhejo Telegram mein<br>
               2. File ID copy karo<br>
               3. Yahan paste kar aur Play karo<br>
               4. Stream shuru ho jayega!</p>
        </div>
        
        <div class="features">
            <div class="feature-card">
                <h4>⚡ Fast</h4>
                <p>Direct streaming, no wait</p>
            </div>
            <div class="feature-card">
                <h4>🎯 Easy</h4>
                <p>Simple interface</p>
            </div>
            <div class="feature-card">
                <h4>🔒 Secure</h4>
                <p>Safe & reliable</p>
            </div>
        </div>
    </div>
    
    <script>
        function loadStream() {
            const fileId = document.getElementById('fileId').value;
            const statusDiv = document.getElementById('status');
            
            if (!fileId) {
                statusDiv.className = 'status error';
                statusDiv.innerHTML = '❌ File ID khali hai!';
                return;
            }
            
            const streamUrl = `/stream/${fileId}`;
            document.getElementById('player').src = streamUrl;
            
            statusDiv.className = 'status success';
            statusDiv.innerHTML = '✅ Stream loading... Play karo!';
        }
        
        // Enter key support
        document.getElementById('fileId').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') loadStream();
        });
    </script>
</body>
</html>
"""

# Save file info
def save_file_info(file_id, file_name, file_size, user_id):
    if not files_collection:
        return False
    try:
        files_collection.insert_one({
            "file_id": file_id,
            "file_name": file_name,
            "file_size": file_size,
            "user_id": user_id,
            "timestamp": time.time()
        })
        return True
    except Exception as e:
        logger.error(f"Save Error: {e}")
        return False

# ============ FLASK ROUTES ============

@flask_app.route("/")
def home():
    """Player HTML"""
    return render_template_string(HTML_TEMPLATE)

@flask_app.route("/stream/<file_id>")
async def stream(file_id):
    """Stream file directly"""
    try:
        # Telegram se file download karte hue stream karo
        async def generate():
            async with app_bot:
                async for chunk in app_bot.get_file(file_id):
                    yield chunk
        
        return app_bot.get_file(file_id), 206, {
            'Content-Type': 'video/mp4',
            'Accept-Ranges': 'bytes',
        }
    except Exception as e:
        logger.error(f"Stream Error: {e}")
        return f"❌ Error: {str(e)}", 404

@flask_app.route("/download/<file_id>")
async def download(file_id):
    """Download file"""
    try:
        async with app_bot:
            file = await app_bot.get_file(file_id)
            return file, 200, {
                'Content-Disposition': 'attachment',
                'Content-Type': 'application/octet-stream',
            }
    except Exception as e:
        logger.error(f"Download Error: {e}")
        return f"❌ Error: {str(e)}", 404

@flask_app.route("/player")
def player():
    """Player page"""
    return render_template_string(HTML_TEMPLATE)

# ============ TELEGRAM BOT COMMANDS ============

@app_bot.on_message(filters.command("start"))
async def start_handler(client, message: Message):
    welcome_text = """
🎬 **Welcome to Stream Bot!**

📤 Mujhe koi bhi file bhejo aur main tujhe uska stream link de dunga!

✨ **Features:**
• Direct stream links
• Fast download support
• Beautiful player interface
• Simple aur fast!

🔗 Just send any file and get instant link!

🎯 /help - Help dekho
📺 /player - Player open karo
    """
    await message.reply(welcome_text, quote=True)
    
    try:
        await client.send_message(LOG_CHANNEL_ID, 
            f"👤 New user: {message.from_user.mention}\nID: `{message.from_user.id}`")
    except:
        pass

@app_bot.on_message(filters.command("help"))
async def help_handler(client, message: Message):
    help_text = f"""
📚 **Help Guide:**

1️⃣ Bot ko koi bhi file bhejo
2️⃣ Bot stream link aur player URL generate karega
3️⃣ Link se direct play/download kar sakte ho

🔗 **Player URL:** `{BASE_URL}/player`

📝 File bhejne ke baad:
• Stream Link: Direct streaming ke liye
• File ID: Manual stream karne ke liye

Ye link kisi ko bhi share kar sakte ho!
    """
    await message.reply(help_text, quote=True)

@app_bot.on_message(filters.document | filters.video | filters.audio)
async def file_handler(client, message: Message):
    try:
        file = message.document or message.video or message.audio
        file_id = file.file_id
        file_name = file.file_name or "Unknown"
        file_size = file.file_size
        
        save_file_info(file_id, file_name, file_size, message.from_user.id)
        
        stream_link = f"{BASE_URL}/stream/{file_id}"
        player_link = f"{BASE_URL}/player"
        
        response_text = f"""
✅ **File Received!**

📁 **Name:** `{file_name}`
📊 **Size:** `{file_size / 1024 / 1024:.2f} MB`
🔗 **File ID:** `{file_id}`

🎬 **Stream Link:**
`{stream_link}`

📺 **Player:**
`{player_link}`

Enter File ID in player to stream!
        """
        
        await message.reply(response_text, quote=True)
        logger.info(f"File received: {file_name} from {message.from_user.id}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await message.reply(f"❌ Error: {str(e)}", quote=True)

@app_bot.on_message(filters.command("ping"))
async def ping_handler(client, message: Message):
    await message.reply("🟢 **Pong!** Bot active hai!", quote=True)

# ============ RUN BOT & SERVER ============

def run_flask():
    """Flask server ko separate thread mein run karo"""
    logger.info(f"🌐 Flask Server starting on port {PORT}...")
    flask_app.run(host="0.0.0.0", port=PORT, debug=False)

def run_bot():
    """Telegram bot run karo"""
    logger.info("🚀 Telegram Bot Starting...")
    app_bot.run()

if __name__ == "__main__":
    # Bot aur Server dono ko parallel run karo
    bot_thread = Thread(target=run_bot, daemon=True)
    server_thread = Thread(target=run_flask, daemon=True)
    
    bot_thread.start()
    server_thread.start()
    
    # Forever run karo
    bot_thread.join()
    server_thread.join()
