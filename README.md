# 🤖 Dual-Arm Robot Controller with Vision-Based Object Detection

## 📌 Overview

This repository contains the evolution of a **dual-arm robotic system** designed for autonomous object detection and manipulation using stereo vision and AI models.

The project demonstrates a progression from a **feature-rich GUI-based control system** to a **lightweight, deployment-ready solution optimized for embedded platforms like the Raspberry Pi 5**.

The system detects a **ping-pong ball** using computer vision, estimates its 3D position via stereo cameras, and commands robotic arms to reach and grab the object.

---

## 🚀 Key Features

* 🦾 Dual robotic arm control (left & right arms)
* 🎥 Stereo vision for depth estimation (3D localization)
* 🧠 AI-based object detection (YOLO → TensorFlow Lite)
* ⚙️ Inverse kinematics for precise arm movement
* 🖥️ GUI interface (early versions)
* 🔋 Battery monitoring (voltage, current, power consumption)
* 🔊 Text-to-Speech feedback for unreachable targets
* ⚡ Optimized for Raspberry Pi 5 deployment

---

## 📂 Project Evolution

### 🔹 Version 4 – Full GUI System

**File:** `FullGUIV4.py`

* Multi-tab interface:

  * Manual control
  * Auto tracking
  * Recording & playback
  * Settings
* Color-based object detection (HSV filtering)
* Movement recording and replay system
* Camera search algorithm for object acquisition
* Threaded architecture for performance

👉 This version focuses on **flexibility, control, and experimentation**

---

### 🔹 Version 5 – YOLO-Based Detection

**File:** `FullGUIV5.py`

* Transition from color detection → **YOLO deep learning model**
* Improved object detection robustness
* Simplified GUI for faster operation
* Maintains stereo vision and IK pipeline

👉 This version improves **accuracy and robustness in real-world conditions**

---

### 🔹 Version 7 – TFLite Optimized Deployment

**File:** `FullGUIV7_tflite.py`

* Conversion of YOLO model → **TensorFlow Lite (INT8 optimized)**
* Designed for **low-latency inference on Raspberry Pi 5**
* Further simplified GUI for fast deployment
* Added:

  * Battery monitoring system
  * Text-to-Speech feedback
  * Workspace validation (reachable vs unreachable objects)

👉 This version focuses on **efficiency, real-time performance, and embedded deployment**

---

## 🧠 System Architecture

### 1. Vision Pipeline

* Stereo camera input
* Image rectification
* Object detection (YOLO / TFLite)
* Disparity calculation
* 3D coordinate estimation

### 2. Motion Planning

* Coordinate transformation (camera → robot frame)
* Inverse kinematics (2-link planar arm)
* Joint angle computation

### 3. Actuation

* Serial communication with servo controller
* Smooth trajectory interpolation
* Dual-arm coordination

---

## ⚙️ Hardware Requirements

* Raspberry Pi 5
* Stereo camera setup
* Dual robotic arms (4 DOF each)
* Servo controller (e.g., SSC-32)
* Power monitoring sensors (optional)

---

## 🧪 Software Dependencies

Install required packages:

```bash
pip install opencv-python numpy pillow pyserial tkinter pyttsx3 tensorflow
```

For YOLO version:

```bash
pip install ultralytics
```

---

## ▶️ Usage

Run the latest optimized version:

```bash
python3 FullGUIV7_tflite.py
```

For earlier versions:

```bash
python3 FullGUIV5.py
python3 FullGUIV4.py
```

---

## 📊 Optimization Strategy

Key improvements implemented across versions:

* ✅ Reduced GUI complexity for faster responsiveness
* ✅ Switched from HSV detection → Deep Learning (YOLO)
* ✅ Converted model to **TensorFlow Lite (INT8 quantization)**
* ✅ Reduced computation overhead for embedded deployment
* ✅ Improved real-time performance on Raspberry Pi 5

---

## ⚠️ Known Limitations

* Requires proper stereo camera calibration
* Performance depends on lighting conditions
* Limited workspace for robotic arms
* Serial communication latency may affect responsiveness

---

## 🔮 Future Improvements

* ROS2 integration
* Multi-object detection and prioritization
* Improved grasping strategy
* Edge TPU acceleration
* Autonomous navigation integration

---

## 👨‍💻 Author

Developed as part of a robotics project focused on:

* Embedded AI systems
* Autonomous robotics
* Real-time computer vision

---

## 📜 License

This project is open-source and available for educational and research purposes.

---

## ⭐ Notes

This repository reflects a **progressive engineering approach**, moving from:

> Complex → Practical
> General-purpose → Embedded optimized
> Prototype → Deployable system

---

Feel free to fork, experiment, and improve 🚀
