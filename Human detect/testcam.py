from ultralytics import YOLO
import cv2
import time
import pyttsx3
import paho.mqtt.client as mqtt
import json

# ====== MQTT CONFIGURATION ======
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "testtopic/mwtt"

mqtt_client = mqtt.Client(protocol=mqtt.MQTTv5)  # Use MQTTv5 to avoid deprecation
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

def send_mqtt_greeting():
    message = {"response": "greetings"}
    mqtt_client.publish(MQTT_TOPIC, json.dumps(message))
    print("MQTT Message Sent:", message)

# ====== YOLO SETUP ======  
model = YOLO("yolov8n.pt")
PERSON_CLASS_ID = 0

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ESP32-CAM stream
cap = cv2.VideoCapture("http://172.20.10.11:81/stream")
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

human_detected_start = None
greeted = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame from ESP32-CAM stream")
        break

    results = model(frame)
    detections = results[0].boxes
    annotated_frame = frame.copy()

    human_detected = False

    if detections is not None:
        for box in detections:
            cls_id = int(box.cls[0].item())
            if cls_id == PERSON_CLASS_ID:
                human_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Human", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Human detection duration check
    if human_detected:
        if human_detected_start is None:
            human_detected_start = time.time()
        elif time.time() - human_detected_start >= 2 and not greeted:
            print("Hello, human!")
            engine.say("Hello, human!")
            engine.runAndWait()
            send_mqtt_greeting()
            greeted = True
    else:
        human_detected_start = None
        greeted = False

    # Show greeting on frame
    if greeted:
        cv2.putText(annotated_frame, "Hello, Human!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Human Detection - YOLOv8 + MQTT", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()

