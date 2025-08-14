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
CONFIDENCE_THRESHOLD = 0.5
GREETING_COOLDOWN = 3  # seconds

# ====== INIT MQTT ======
mqtt_client = mqtt.Client(protocol=mqtt.MQTTv5)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

def send_mqtt_greeting():
    message = {"response": "greetings"}
    mqtt_client.publish(MQTT_TOPIC, json.dumps(message))
    print("MQTT Message Sent:", message)

# ====== LOAD YOLO MODEL ======
model = YOLO("yolov8l.pt")  # Pretrained model
PERSON_CLASS_ID = 0

# ====== TEXT-TO-SPEECH SETUP ======
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ====== OPEN LOCAL WEBCAM ======
print("Opening local webcam...")
cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    print("Error: Could not open local webcam.")
    exit()

# ====== MAIN LOOP ======
human_detected_start = None
greeted = False
last_sent_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            continue

        results = model(frame)
        detections = results[0].boxes
        annotated_frame = frame.copy()

        human_detected = False

        if detections is not None:
            for box in detections:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id == PERSON_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
                    human_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Human {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Detection confirmation and cooldown logic
        if human_detected:
            if human_detected_start is None:
                human_detected_start = time.time()
            elif time.time() - human_detected_start >= 2 and not greeted:
                current_time = time.time()
                if current_time - last_sent_time > GREETING_COOLDOWN:
                    print("Hello, human!")
                    engine.say("Hello, human!")
                    engine.runAndWait()
                    send_mqtt_greeting()
                    greeted = True
                    last_sent_time = current_time
        else:
            human_detected_start = None
            greeted = False

        # Display greeting text
        if greeted:
            cv2.putText(annotated_frame, "Hello, Human!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Show live feed
        cv2.imshow("Human Detection - YOLOv8 + MQTT", annotated_frame)

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
