from ultralytics import YOLO
import cv2
import time
import json
import paho.mqtt.client as mqtt

# ============================
# CONFIGURATION
# ============================
model_path = "C:/Users/Rakaputu Banardi A/Documents/A capstone/dataset/my_modell/train4/weights/best.pt"
confidence_threshold = 0.65
OBJECT_COOLDOWN = 3  # seconds

# MQTT Configuration    
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "testtopic/mwtt"

mqtt_client = mqtt.Client(protocol=mqtt.MQTTv5)
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print("Connected to MQTT broker.")
except Exception as e:
    print("Failed to connect to MQTT broker:", e)
    exit()

# ============================
# LOAD MODEL & CAMERA
# ============================
model = YOLO(model_path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ============================
# 3-SECOND DELAY BEFORE STARTING DETECTION
# ============================
print("Starting detection in 3 seconds...")
time.sleep(3)

# Dictionary to store last sent time for each label
last_sent_time = {}

# ============================
# DETECTION LOOP
# ============================
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    results = model(frame, verbose=False)[0]
    annotated_frame = frame.copy()
    count = 0

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Draw detection
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{label} {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        # MQTT Cooldown Check
        now = time.time()
        if label not in last_sent_time or (now - last_sent_time[label]) >= OBJECT_COOLDOWN:
            mqtt_payload = {
                "command": label
            }
            mqtt_client.publish(MQTT_TOPIC, json.dumps(mqtt_payload))
            print("MQTT Published:", mqtt_payload)
            last_sent_time[label] = now

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # Overlay info
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Detections: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

    cv2.imshow("YOLOv8 - MQTT Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================
# CLEANUP
# ============================
cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()
    