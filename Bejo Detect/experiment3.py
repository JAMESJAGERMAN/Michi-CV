from ultralytics import YOLO
import cv2
import time
import json
import paho.mqtt.client as mqtt

# ============================
# CONFIGURATION
# ============================
MODEL_PATH = "C:/Users/Rakaputu Banardi A/Documents/A capstone/dataset/my_modell/train4/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.65
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "testtopic/mwtt"
GREETING_COOLDOWN = 3  # seconds between MQTT messages

# ============================
# MQTT SETUP
# ============================
mqtt_client = mqtt.Client(protocol=mqtt.MQTTv5)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

def send_mqtt_detection(label):
    message = {"detected": label}
    mqtt_client.publish(MQTT_TOPIC, json.dumps(message))
    print("MQTT Sent:", message)

# ============================
# LOAD MODEL
# ============================
model = YOLO(MODEL_PATH)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ============================
# MAIN LOOP
# ============================
prev_time = 0
last_mqtt_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Inference
        results = model(frame, verbose=False)[0]
        annotated_frame = frame.copy()

        count = 0
        label_sent = None

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # MQTT sending condition
            current_time = time.time()
            if label_sent is None and current_time - last_mqtt_time > GREETING_COOLDOWN:
                label_sent = label
                send_mqtt_detection(label)
                last_mqtt_time = current_time

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        # Overlay text
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Detections: {count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

        # Show window
        cv2.imshow("YOLOv8 + MQTT (Custom Model)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped manually")

finally:
    cap.release()
    cv2.destroyAllWindows()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("Cleaned up successfully.")


