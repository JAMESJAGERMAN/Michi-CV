# Michi Computer Vision

Welcome! This repository contains the code for a real-time Michi detection system built using YOLOv8, MQTT communication. This repo contains the MICHI computer vision part of the robot. It provides:


---

## 📁 Project Structure

```
Michi-CV/
├── Main/                       # Main fungtion for Computer vision Robot 
├── Code_Test                   # Seperate of each fungtion for test on Espcam and Webcam
├── Train/                      # Dataset of object Detection
├── Chart/                      # Result of Data training
├── README_beverage.md          # (this file)
```

---

## 🤖 Main System ( `Main/` )

This main folder contain core system for manage computer vision of the robot

### Features:

- **flask_camera_stream.py**: Detects person infornt camera and products in the camera feed.
- **Confidence Filtering**: Ignores detections below parameter confidence.
- **Yolo8l.pt**: File Large model Yolov8 for detect person part.
- **MQTT Communication**: Publishes detected product labels to an MQTT broker.

---

### Key Features

- Real-time detection via OpenCV Esp32 stream.
- Confidence threshold filtering for accuracy.
- MQTT output for integration with other devices (robots, dashboards, etc.).

---

### Dependencies

Install required Python packages:

```pwsh
pip install flask
pip install ultralytics 
pip install opencv-python
pip install paho-mqtt pipwin
pipwin install pyaudio
```

---

## 🚀 Code Tester (`Code_test/`)

This main folder contain testing fungtion of main system but seperated, each program have diffrent fungtion not like the main core one that have all in 

### Features:


- **Human_greating_cam.py**: Test Feature for Detects person infornt camera feed .
- **Human_greating_ESP32.py**: Test Feature for Detects person infornt camera ESP32 feed.
- **object_detect_cam.py**: Test Feature for Detects custom data infornt camera ESP32 feed.
- **object_detect_ESP32.py**: Test Feature for Detects custom data infornt camera ESP32 feed.
- **MQTT Communication**: Publishes detected product labels to an MQTT broker.

---

### Key Features

- Real-time detection via OpenCV Esp32 stream.
- Confidence threshold filtering for accuracy.
- MQTT output for integration with other devices (robots, dashboards, etc.).

Awesome—here are 2 tight, friendly sections you can drop straight into your README.

---

## 🚀 Getting Started

### 1) Prerequisites

* **Python** 3.10–3.11
* **Git**

### 2) Clone the repo

```bash
git clone https://github.com/<your-username>/Michi-CV.git
cd Michi-CV
```

### 3) Create & activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4) Install dependencies

Prefer using a `requirements.txt` if you have one:

```bash
pip install -r requirements.txt
```

…or install individually:

```bash
pip install flask 
pip ultralytics 
pip opencv-python
pip paho-mqtt
pip install pipwin
pipwin install pyaudio
```

> **GPU (optional):** install PyTorch that matches your CUDA. Check [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) then verify with:

```python
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 5) Put your models in place

* **Human detection:** `yolov8l.pt` (Ultralytics will auto-download on first run)
* **Custom object model:** copy your `best.pt` to
  `Train/weights/best.pt` *(or keep it anywhere and update the path in the script)*

### 6) Configure your settings

Open `Main/flask_camera_stream.py` and set the constants at the top:

* `ESP32_STREAM_URL` → your ESP32-CAM stream URL (e.g. `http://<esp32-ip>:81/stream`)
* `CUSTOM_MODEL_PATH` → path to your custom `best.pt`
* `HUMAN_MODEL_PATH` → usually `yolov8l.pt`
* `MQTT_BROKER`, `MQTT_PORT`, `MQTT_TOPIC` → MQTT settings (default uses `broker.emqx.io`)


### 7) Run the main app

```bash
cd Main
python flask_camera_stream.py
```

Open the stream in your browser:

```
http://localhost:5000/video_feed
```

### 8) (Optional) Watch MQTT messages

Use any MQTT client :

---

## 🛠️ Troubleshooting

**No video / black page at `/video_feed`**

* Make sure the Flask app is running with no errors.
* Ensure the `<img>` in your HTML points to the correct URL (e.g. `http://<server-ip>:5000/video_feed`).
* Windows firewall may block port **5000**—allow Python through or change the port.

**ESP32-CAM stream won’t open**

* Confirm `ESP32_STREAM_URL` is correct and reachable from your PC (same Wi-Fi).
* Only one viewer can access the ESP32-CAM stream at a time—close other tabs/apps.
* Try lowering ESP32 resolution/quality in its firmware to reduce bandwidth.

**Low FPS / lag**

* Prefer a GPU (CUDA). Verify with:

  ```python
  python -c "import torch; print('CUDA:', torch.cuda.is_available())"
  ```
* Reduce model size: switch `yolov8l.pt` → `yolov8s.pt`/`yolov8n.pt` for testing.
* Lower input size: in code, call `model(frame, imgsz=480)` (or 416/320).
* Limit camera resolution:

  ```python
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  ```
* Skip frames if needed (process every 2nd/3rd frame).
* Close other apps using the GPU/CPU.

**“Model not found” or wrong labels**

* Check `CUSTOM_MODEL_PATH` actually points to your `best.pt`.
* Print class map to verify (Ultralytics: `print(model.names)`).

**MQTT messages not received**

* Verify broker, port, and topic match on both sender & subscriber.
* Test network connectivity (same LAN or public broker).
* Some corporate networks block MQTT—try a different network or port.

**Mic / Wake word not working (Code\_Test only)**

* Windows: install via pipwin: `pipwin install pyaudio`.
* macOS: `brew install portaudio && pip install pyaudio`.
* Linux: `sudo apt install portaudio19-dev && pip install pyaudio`.
* Check OS microphone permissions.

**Flask crashes on start**

* Another app might be using port 5000. Run with a different port:

  ```bash
  python flask_camera_stream.py --port 8000
  ```

  (or change the `app.run(..., port=8000)` line)

**“Failed to read frame” loop**

* ESP32 power/Wi-Fi might be unstable; try a better power supply and stronger signal.
* Add simple reconnect logic (recreate `VideoCapture` if `read()` fails).

---

## 📄 License

This project is part of the Michi Robot ecosystem. Please refer to the main project repository for licensing information.

## 📞 Support

For technical support and questions:
- Check the component tests for hardware validation
- Review serial output for debugging information
- Consult the main project documentation for system integration

---