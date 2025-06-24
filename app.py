import eventlet
eventlet.monkey_patch()

import os
import cv2
import torch
import pygame
import base64
import uuid
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from ultralytics import YOLO
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from io import BytesIO
from PIL import Image
from math import radians, sin, cos, sqrt, atan2
from docx import Document
import subprocess

# === Telegram Bot Setup ===
TELEGRAM_BOT_TOKEN = 'your-token'
TELEGRAM_CHAT_ID = '23748377483'

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    # You can uncomment to send alert:
    # requests.post(url, data=payload)

# === Flask & SocketIO ===
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

UPLOAD_FOLDER = "uploaded_videos"
FRAMES_FOLDER = "detected_frames"
MODEL_PATH = "yolo11n.pt"
ALARM_PATH = r"C:\mega project\static\alarm.mp3"
DOCX_PATH = "violence_words.docx"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

live_video_capture = None
stop_processing = False

# Load YOLO model
yolo_model = YOLO(MODEL_PATH)

# Load VideoMAE model
try:
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
    video_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    video_model.to(device)
except Exception as e:
    print(f"Error loading video model: {e}")

pygame.mixer.init()

def play_alarm():
    try:
        pygame.mixer.music.load(ALARM_PATH)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"[ERROR] Alarm couldn't be played: {e}")

def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

def save_frame_to_file(encoded_frame, output_folder):
    image_data = base64.b64decode(encoded_frame)
    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(output_folder, filename)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return file_path

def load_violence_words(docx_path):
    doc = Document(docx_path)
    violence_words = []
    for para in doc.paragraphs:
        violence_words.extend(para.text.split())
    return violence_words

def enlarge_bbox(x, y, w, h, factor, img_width, img_height):
    new_w = w * factor
    new_h = h * factor
    new_x = max(0, x - (new_w - w) / 2)
    new_y = max(0, y - (new_h - h) / 2)
    new_w = min(new_w, img_width - new_x)
    new_h = min(new_h, img_height - new_y)
    return int(new_x), int(new_y), int(new_w), int(new_h)

def get_current_location():
    try:
        data = requests.get("https://ipinfo.io/json").json()
        lat, lon = map(float, data.get('loc').split(','))
        return {
            'city': data.get('city'),
            'region': data.get('region'),
            'country': data.get('country'),
            'latitude': lat,
            'longitude': lon
        }
    except Exception as e:
        # Fallback location
        return {
            'city': 'Ichalkaranji',
            'region': 'Kolhapur',
            'country': 'India',
            'latitude': 16.7,
            'longitude': 74.4
        }

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

authorities = [
    {'id': 'AUTH001', 'name': 'Bangalore Police', 'lat': 12.9716, 'lon': 77.5946},
    {'id': 'AUTH002', 'name': 'Kolhapur Police', 'lat': 16.7050, 'lon': 74.2433},
    {'id': 'AUTH003', 'name': 'Delhi Police', 'lat': 28.6139, 'lon': 77.2090}
]

def find_nearest_authority(lat, lon):
    return min(authorities, key=lambda auth: haversine(lat, lon, auth['lat'], auth['lon']))

def extract_people_frames_live(video_stream, output_folder, enlargement_factor=1.0):
    global stop_processing
    chunk_buffer, chunk_counter, frame_count = [], 0, 0
    os.makedirs(output_folder, exist_ok=True)
    violence_words = load_violence_words(DOCX_PATH)

    # Get current location once
    location_info = get_current_location()
    nearest_authority = find_nearest_authority(location_info['latitude'], location_info['longitude'])

    while video_stream.isOpened() and not stop_processing:
        ret, frame = video_stream.read()
        if not ret:
            break

        frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        results = yolo_model.predict(frame, conf=0.5)

        for result in results:
            boxes = result.boxes.xywh.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, class_id in zip(boxes, class_ids):
                if int(class_id) == 0:
                    x, y, w, h = box
                    x, y, w, h = enlarge_bbox(x - w/2, y - h/2, w, h, enlargement_factor, frame_width, frame_height)
                    cropped = frame[y:y+h, x:x+w]
                    image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    chunk_buffer.append(image)

                    if len(chunk_buffer) == 16:
                        inputs = image_processor(chunk_buffer, return_tensors="pt")
                        inputs = {k: v.to(video_model.device) for k, v in inputs.items()}
                        with torch.no_grad():
                            logits = video_model(**inputs).logits
                        pred = video_model.config.id2label[logits.argmax(-1).item()]
                        is_violent = any(word.lower() in pred.lower() for word in violence_words)

                        chunk_counter += 1
                        encoded_frame = encode_frame_to_base64(frame)

                        socketio.emit('live_feed', {
                            'frame': encoded_frame,
                            'is_violent': is_violent,
                            'location': f"{location_info['city']}, {location_info['region']}, {location_info['country']}",
                            'authority': nearest_authority['name']
                        })

                        socketio.emit('chunk_status', {
                            'chunk': chunk_counter,
                            'status': 'Violence' if is_violent else 'Non-Violence'
                        })

                        if is_violent:
                            
                            play_alarm()
                            saved_frame_path = save_frame_to_file(encoded_frame, output_folder)

                            # Prepare location text
                            location_text = f"{location_info['city']}, {location_info['region']}, {location_info['country']}"

                            # Call bot.py with args: image path and location string
                            try:
                               subprocess.Popen(['python', 'bot.py', saved_frame_path, location_text])
                            except Exception as e:
                               print(f"[ERROR] Failed to call bot.py: {e}")


                        chunk_buffer = []

        eventlet.sleep(0.01)

    video_stream.release()
    print("[INFO] Live video processing ended.")

def extract_people_frames_from_file(video_path, output_folder, enlargement_factor=1.0):
    cap = cv2.VideoCapture(video_path)
    extract_people_frames_live(cap, output_folder, enlargement_factor)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_live_detection', methods=['POST'])
def start_live_detection():
    global live_video_capture, stop_processing
    stop_processing = False
    live_video_capture = cv2.VideoCapture(0)
    if not live_video_capture.isOpened():
        return jsonify({'error': 'Webcam could not be opened'}), 500

    socketio.start_background_task(extract_people_frames_live, live_video_capture, FRAMES_FOLDER)
    return jsonify({'status': 'Live detection started'})

@app.route('/start_video_detection', methods=['POST'])
def start_video_detection():
    global stop_processing
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    video_file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)
    stop_processing = False
    socketio.start_background_task(extract_people_frames_from_file, video_path, FRAMES_FOLDER)
    return jsonify({'status': 'Video detection started'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global stop_processing, live_video_capture
    stop_processing = True
    if live_video_capture:
        live_video_capture.release()
    return jsonify({'status': 'Detection stopped'})

@socketio.on('connect')
def on_connect():
    print('Client connected')

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
