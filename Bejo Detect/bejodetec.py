from ultralytics import YOLO
import cv2
import time

# ============================
# CONFIGURATION
# ============================
model_path = "C:/Users/Rakaputu Banardi A/Documents/A capstone/github/train/weights/best.pt"  # Path to your custom model
confidence_threshold = 0.65  # Only show detections above this confidence

# Load model
model = YOLO(model_path)

# Start webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ============================
# START DETECTION LOOP
# ============================
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # === Inference ===
    results = model(frame, verbose=False)[0]  # Get first (only) result

    # Create copy to draw
    annotated_frame = frame.copy()

    # Count and draw detections
    count = 0
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        count += 1

        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]  # Get class name

        # Draw rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(
            annotated_frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # === FPS calculation ===
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # === Overlay text info ===
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Detections: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

    # === Display frame ===
    cv2.imshow("YOLOv8 - Custom Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================
# CLEANUP
# ============================
cap.release()
cv2.destroyAllWindows()
