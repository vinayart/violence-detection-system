import cv2
import numpy as np
from ultralytics import YOLO
import time

# For alarm sound:
try:
    import winsound
    def play_alarm():
        duration = 1000  # milliseconds
        freq = 1000  # Hz
        winsound.Beep(freq, duration)
except ImportError:
    # For non-Windows or if winsound is unavailable
    from playsound import playsound
    import threading
    def play_alarm():
        # Play alarm sound asynchronously (add your alarm.mp3 path here if you want)
        threading.Thread(target=playsound, args=("alarm.mp3",), daemon=True).start()


# Load YOLO model
model = YOLO("yolov8n.pt")

class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
               "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
               "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
               "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
               "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
               "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
               "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
               "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
               "toothbrush"]

harmful_objects = {
    "knife",
    "baseball bat",
    "scissors",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "truck",
    "train",
    "boat"
}

def detect_blood_mask(frame, red_threshold=0.01):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    red_area_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return red_area_ratio > red_threshold

def detect_harmful_objects(frame):
    results = model(frame)
    detected_objects = set()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id < len(class_names):
                label = class_names[class_id].lower()
                if label in harmful_objects:
                    detected_objects.add(label)
    return len(detected_objects) > 0, detected_objects

def detect_violence(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    last_alarm_time = 0
    alarm_cooldown = 5  # seconds between alarms so it doesn't spam

    print("[INFO] Violence detection started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        blood_present = detect_blood_mask(frame)
        weapon_present, detected_objs = detect_harmful_objects(frame)

        if blood_present and weapon_present:
            current_time = time.time()
            if current_time - last_alarm_time > alarm_cooldown:
                print(f"[ALERT] Violence detected! Objects: {detected_objs}")
                play_alarm()
                last_alarm_time = current_time

    cap.release()

if __name__ == "__main__":
    video_path = r".mp4"
    detect_violence(video_path)
