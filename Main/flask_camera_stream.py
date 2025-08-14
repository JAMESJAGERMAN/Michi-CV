

from flask import Flask, Response
import cv2
import time
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json

# URL kamera ESP32
ESP32_STREAM_URL = "http://172.20.10.11:81/stream"

# Model deteksi
CUSTOM_MODEL_PATH = "C:/Users/Rakaputu Banardi A/Documents/A capstone/github/train/weights/best.pt"
HUMAN_MODEL_PATH = "yolov8l.pt"
CUSTOM_CONFIDENCE_THRESHOLD = 0.75
HUMAN_CONFIDENCE_THRESHOLD = 0.3
PERSON_LABEL = "person"

custom_model = YOLO(CUSTOM_MODEL_PATH)
human_model = YOLO(HUMAN_MODEL_PATH)

# MQTT setup
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "testtopic/mwtt"
mqtt_client = mqtt.Client(protocol=mqtt.MQTTv5)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# Cooldown untuk kirim per objek
OBJECT_SEND_COOLDOWN = 3  # detik
last_object_sent_time = {}

app = Flask(__name__)


def send_mqtt_object(label):
    message = {"command": label}
    mqtt_client.publish(MQTT_TOPIC, json.dumps(message))
    print("MQTT Object Sent:", message)

def send_mqtt_greeting():
    message = {"response": "greetings"}
    mqtt_client.publish(MQTT_TOPIC, json.dumps(message))
    print("MQTT Greeting Sent:", message)

def gen_frames():
    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    prev_time = 0
    global last_object_sent_time
    human_detected_start = None
    greeted = False
    last_greeted_time = 0
    GREETING_COOLDOWN = 3  # seconds
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            cap.release()
            cap = cv2.VideoCapture(ESP32_STREAM_URL)
            continue
        # Deteksi objek dan manusia
        custom_results = custom_model(frame, verbose=False)[0]
        human_results = human_model(frame, verbose=False)[0]
        annotated_frame = frame.copy()
        # Get person class id from human model
        person_cls_id = None
        for k, v in human_model.names.items():
            if v.lower() == PERSON_LABEL:
                person_cls_id = k
                break
        # Draw custom model detections (objects) + MQTT send
        for box in custom_results.boxes:
            conf = float(box.conf[0])
            if conf < CUSTOM_CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = custom_model.names[cls_id]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 100, 100), 2)
            cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            # MQTT object detection send with cooldown per label
            current_time = time.time()
            if (label not in last_object_sent_time) or (current_time - last_object_sent_time[label] > OBJECT_SEND_COOLDOWN):
                send_mqtt_object(label)
                last_object_sent_time[label] = current_time
        # Draw human model detections (person)
        human_detected = False
        for box in human_results.boxes:
            conf = float(box.conf[0])
            if conf < HUMAN_CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = human_model.names[cls_id]
            if person_cls_id is not None and cls_id == person_cls_id:
                human_detected = True
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Human {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Human greeting logic
        current_time = time.time()
        if human_detected:
            if human_detected_start is None:
                human_detected_start = time.time()
            elif time.time() - human_detected_start >= 2 and not greeted:
                if current_time - last_greeted_time > GREETING_COOLDOWN:
                    send_mqtt_greeting()
                    greeted = True
                    last_greeted_time = current_time
        else:
            human_detected_start = None
            greeted = False
        # FPS calculation
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
