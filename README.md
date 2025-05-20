# 🚦 Traffic Flow Optimization – AI-Powered Smart Traffic System

## 🎯 Objective

The objective of this project is to reduce urban traffic congestion by dynamically adjusting traffic signal timings using AI and real-time IoT data. The system is designed to optimize traffic flow at intersections by detecting vehicles and calculating green light durations based on vehicle density and type.

---

## 🌟 Features

- 🚘 **Vehicle Detection using YOLOv5**  
  Detects and classifies different vehicle types (cars, buses, bikes) in real-time.

- ⏱ **Dynamic Signal Timing**  
  Calculates optimal green light durations using a weighted formula based on vehicle type and count.

- 🔄 **Real-time Traffic Light Simulation**  
  Simulates red/yellow/green signal changes with countdown based on traffic analysis.

- 📡 **IoT Sensor Integration**  
  Accepts real-time or simulated data from traffic sensors (cameras, GPS, loop detectors).

- 📊 **Performance Monitoring**  
  Tracks metrics like system latency, detection accuracy, and responsiveness.

- 🔐 **Secure & Scalable Architecture**  
  Backend optimizations and encrypted communication for smart city readiness.

---

## 🛠 Technology Used

- **Languages:** Python  
- **Libraries/Frameworks:**  
  - PyTorch (YOLOv5 for object detection)  
  - OpenCV (image/video processing)  
  - FastAPI (backend REST API)  
- **Tools:**  
  - IoT sensor simulators  
  - Uvicorn (API server)  
  - JSON & REST APIs for data communication  

---

## ⚙️ How It Works

1. **Input**: Live traffic feeds or uploaded images + IoT sensor data.
2. **Detection**: Vehicles are detected using YOLOv5 and counted.
3. **Calculation**: Green light duration is calculated using weighted logic (e.g., truck > car > bike).
4. **Control**: The simulated traffic light display changes based on calculated timing.
5. **Feedback Loop**: Real-time adjustments are made as new data flows in.

---

## 📊 Data Collection

- **Source**: Simulated traffic images and video data for initial development.
- **IoT Input**: Simulated sensor data (e.g., GPS, loop detectors) for integration testing.
- **Model Training Dataset**: Pre-trained YOLOv5 model trained on COCO dataset for vehicle detection. Fine-tuned with traffic scene images where required.

---

## 🎮 Controls

- **Start/Stop Simulation** – Initiates or halts the signal control system.
- **Upload Image/Video** – Allows user to test vehicle detection accuracy.
- **Sensor Feed Toggle** – Switch between real-time and simulated IoT data.

---

## 🧠 ML Techniques Used

- **YOLOv5 (You Only Look Once)** object detection for real-time identification and classification of vehicles.
- **Weighted Heuristics** for determining optimal green light durations based on detected vehicle types.

---

## 📈 Model Training

- **Model Used**: YOLOv5s (small variant for speed and lightweight deployment)
- **Training Data**: COCO dataset and traffic scene images
- **Training Tools**: PyTorch, Google Colab (optional for experimentation)
- **Improvements**: Model tuned for low-light conditions and optimized for real-time inference speed.

---

## 📤 Output Explanation

- A live or simulated traffic intersection with real-time signal changes.
- Visual feedback on:
  - Number and types of vehicles detected
  - Calculated green light durations
  - System response and adaptation to new input data

---


## 🎥 Demo Video (Optional)

Link: *[To be added by team]*

---

## 📦 Project Package Includes

- ✅ Source code and model files
- ✅ Setup instructions
- ✅ Test images/videos
- ✅ API documentation
- ✅ Project report and presentation

---
