from ultralytics import YOLO
import cv2
import time
import pyttsx3
import paho.mqtt.client as mqtt
import json

# ====== CONFIGURATION ======
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "testtopic/mwtt"
ESP32_STREAM_URL = "http://172.20.10.11:81/stream"
CONFIDENCE_THRESHOLD = 0.5
GREETING_COOLDOWN = 3  # seconds

# ====== INIT MQTT ======
mqtt_client = mqtt.Client(protocol=mqtt.MQTTv5)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# ====== TEXT-TO-SPEECH SETUP ======
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ====== LOAD YOLO MODEL ======
model = YOLO("yolov8l.pt")  # Replace with your trained model if available
print("Model loaded with classes:", model.names)

# ====== OPEN ESP32-CAM STREAM ======
print("Connecting to ESP32-CAM...")
cap = cv2.VideoCapture(ESP32_STREAM_URL)
retry_count = 0
while not cap.isOpened() and retry_count < 5:
    print("Retrying ESP32-CAM stream...")
    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    retry_count += 1
    time.sleep(1)

if not cap.isOpened():
    print("Error: Could not connect to ESP32-CAM after multiple retries.")
    exit()

# ====== MAIN LOOP ======
last_sent_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame from ESP32-CAM stream")
            continue

        results = model(frame)
        detections = results[0].boxes
        annotated_frame = frame.copy()
        detected_classes = set()

        if detections is not None:
            for box in detections:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if conf >= CONFIDENCE_THRESHOLD:
                    label = model.names[cls_id]
                    detected_classes.add(label)

                    # Draw box and label
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Send MQTT + TTS if object detected and cooldown passed
        current_time = time.time()
        if detected_classes and current_time - last_sent_time > GREETING_COOLDOWN:
            for class_name in detected_classes:
                payload = {"product": class_name}
                mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
                print("MQTT Message Sent:", payload)

                # Speak detected object
                engine.say(f"{class_name} detected")
                engine.runAndWait()

            last_sent_time = current_time

        # Show live feed
        cv2.imshow("Object Detection - YOLOv8 + MQTT", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram stopped with Ctrl+C")

finally:
    cap.release()
    cv2.destroyAllWindows()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("Resources released. Goodbye!")
