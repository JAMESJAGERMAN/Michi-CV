from ultralytics import YOLO
import cv2
import time
import pyttsx3
import paho.mqtt.client as mqtt
import json

# ============================
# CONFIGURATION
# ============================
model_path = "C:/Users/Rakaputu Banardi A/Documents/A capstone/dataset/my_modell/train4/weights/best.pt"
confidence_threshold = 0.8  # Only show detections above 80%
GREETING_COOLDOWN = 5  # seconds (increased from 3 to 5)

# MQTT Config
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "testtopic/mwtt"

# ============================
# INITIALIZATION
# ============================
# Load YOLO model
model = YOLO(model_path)
print("Model loaded with classes:", model.names)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# MQTT setup
mqtt_client = mqtt.Client(protocol=mqtt.MQTTv5)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ============================
# MAIN LOOP
# ============================
prev_time = 0
last_sent_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Run detection
        results = model(frame, verbose=False)[0]
        annotated_frame = frame.copy()
        detected_classes = set()
        count = 0

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            detected_classes.add(label)

            # Draw box and label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Send MQTT and TTS with cooldown
        current_time = time.time()
        if detected_classes and current_time - last_sent_time > GREETING_COOLDOWN:
            for class_name in detected_classes:
                payload = {"product": class_name}
                mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
                print("MQTT Message Sent:", payload)

                engine.say(f"{class_name} detected")
                engine.runAndWait()

            last_sent_time = current_time

        # FPS display
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        # Overlay info
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Detections: {count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

        # Show frame
        cv2.imshow("YOLOv8 - Custom Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped with Ctrl+C")

finally:
    cap.release()
    cv2.destroyAllWindows()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("Resources released. Goodbye!")
