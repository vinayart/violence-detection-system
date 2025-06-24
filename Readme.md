
# 🔍 Violence Detection System (Real-time + Video Upload)

This project is a real-time **violence detection system** built with:
- **YOLO** for person detection
- **VideoMAE** for classifying violence from video chunks
- **Flask + SocketIO** for the live web interface
- **Telegram Bot** for instant alerts to authorities
- **Alarm System** using Pygame for immediate feedback

---

## 📦 Features

- 🎥 Live webcam violence detection
- 🎞️ Video file upload support
- 🧠 YOLO for detecting people
- 🧠 VideoMAE for violence classification
- 🔊 Plays alarm sound on violence detection
- 📍 Gets your IP-based location
- 🛂 Sends alerts to the nearest police authority via Telegram
- 🖼️ Saves violence-detected frames

---

## 🗂️ Folder Structure

```
violence-detection-system/
├── app.py                  # Main Flask backend
├── bot.py                  # Telegram bot code (called when violence detected)
├── yolov5/ or yolov8/      # YOLO model directory
├── static/
│   └── alarm.mp3           # Alarm sound file
├── templates/
│   └── index.html          # Web UI
├── detected_frames/        # Stores frames detected as violent
├── uploaded_videos/        # Video uploads from user
├── violence_words.docx     # File with violence-related keywords
├── yolo11n.pt              # YOLO model weights (download separately)
├── requirements.txt        # All Python dependencies
└── README.md               # You're reading this!
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/violence-detection-system.git
cd violence-detection-system
```

### 2. Create Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Add Model Files

- `yolo11n.pt` → download and place in the project root.
- `violence_words.docx` → add your violence-related terms.
- `alarm.mp3` → save in `static/` folder.

---

## ▶️ Running the App

```bash
python app.py
```

Now open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🔐 Telegram Alert Setup

Edit your `app.py` and replace these lines with your bot credentials:

```python
TELEGRAM_BOT_TOKEN = 'your-telegram-bot-token'
TELEGRAM_CHAT_ID = 'your-chat-id'
```

> Use this URL to get your chat ID after sending a message to your bot:
> ```
> https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
> ```

---

## 🧠 Detection Flow

1. YOLO detects people in frames.
2. Person crops grouped into 16-frame chunks.
3. Chunks sent to **VideoMAE** for classification.
4. If violence detected:
   - Alarm plays
   - Frame saved
   - Alert sent to Telegram with location

---

## 📍 Location Detection

- Uses IP from `https://ipinfo.io` to get:
  - City, region, country
  - Latitude, longitude
- Haversine formula finds **nearest authority** from predefined list.

---

## 🚫 GitHub File Management

❌ Do NOT upload:
- `.pt`, `.h5`, `.mp4`, full datasets

✅ Instead, use:
- `.gitignore` to ignore large files
- Git LFS or external storage (Google Drive, HuggingFace Hub)

---

## 📜 License

MIT License — use for academic and personal projects.
