# ------------------------------------
# ✅ Install Dependencies
# ------------------------------------
!pip install easyocr opencv-python-headless --quiet

# ------------------------------------
# 📦 Imports and Setup
# ------------------------------------
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from IPython.display import display, HTML, clear_output
from google.colab import files
import easyocr
import os

# Load YOLOv5 models
vehicle_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
plate_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Placeholder

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Vehicle classes
vehicle_classes = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']


# ------------------------------------
# 🔍 Detection Functions
# ------------------------------------
def detect_vehicles_and_plates(image_path):
    img = Image.open(image_path)
    img_cv = cv2.imread(image_path)
    
    # Vehicle detection
    vehicle_results = vehicle_model(img)
    vehicle_preds = vehicle_results.pandas().xyxy[0]
    vehicles = vehicle_preds[vehicle_preds['name'].isin(vehicle_classes)]
    
    # Simplified plate detection (placeholder)
    plate_results = plate_model(img)
    plate_preds = plate_results.pandas().xyxy[0]
    plates = plate_preds[plate_preds['confidence'] > 0.5]
    
    plate_info = []
    for _, plate in plates.iterrows():
        x1, y1, x2, y2 = map(int, [plate['xmin'], plate['ymin'], plate['xmax'], plate['ymax']])
        plate_img = img_cv[y1:y2, x1:x2]
        results = reader.readtext(plate_img)
        if results:
            text = results[0][1]
            plate_info.append({
                'coordinates': (x1, y1, x2, y2),
                'text': text,
                'confidence': plate['confidence']
            })
    
    vehicle_counts = {
        cls: len(vehicles[vehicles['name'] == cls]) for cls in vehicle_classes
    }
    vehicle_counts['total'] = sum(vehicle_counts.values())

    annotated_img = vehicle_results.render()[0]

    for plate in plate_info:
        x1, y1, x2, y2 = plate['coordinates']
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, plate['text'], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return vehicle_counts, annotated_img, plate_info


# ------------------------------------
# ⏱️ Green Light Timing
# ------------------------------------
def calculate_green_light_time(vehicle_counts, base_time=30, vehicle_weight=2):
    weights = {'car': 1.0, 'motorcycle': 0.8, 'bicycle': 0.5, 'bus': 2.5, 'truck': 2.0}
    weighted_sum = sum(
        count * weights.get(vehicle, 1) for vehicle, count in vehicle_counts.items() if vehicle in weights
    )
    green_time = base_time + (weighted_sum * vehicle_weight)
    return min(int(green_time), 90)


# ------------------------------------
# 🚦 Fixed Traffic Light Simulation
# ------------------------------------
def traffic_light_simulation(green_time):
    phases = [('red', 5), ('yellow', 2), ('green', green_time)]

    for color, duration in phases:
        for sec in range(duration, 0, -1):
            red_color = 'red' if color == 'red' else '#330000'
            yellow_color = 'yellow' if color == 'yellow' else '#332200'
            green_color = '#00FF00' if color == 'green' else '#003300'

            clear_output(wait=True)
            display(HTML(f"""
                <div style="width: 120px; height: 300px; background-color: black;
                            border-radius: 10px; padding: 10px; display: flex; flex-direction: column;
                            justify-content: space-between; align-items: center;">
                    <div style="width: 90px; height: 90px; border-radius: 50%;
                                background-color: {red_color}; border: 2px solid #fff;"></div>
                    <div style="width: 90px; height: 90px; border-radius: 50%;
                                background-color: {yellow_color}; border: 2px solid #fff;"></div>
                    <div style="width: 90px; height: 90px; border-radius: 50%;
                                background-color: {green_color}; border: 2px solid #fff;"></div>
                </div>
                <p style="font-size: 20px; font-family: monospace;">🚦 Current signal: <strong>{color.upper()}</strong></p>
                <p style="font-size: 16px;">⏳ Time remaining: <strong>{sec}s</strong></p>
            """))
            time.sleep(1)


# ------------------------------------
# 🖼️ Process and Visualize
# ------------------------------------
def process_image_and_calculate_timing(image_path):
    print("🔍 Detecting vehicles and number plates...")
    vehicle_counts, annotated_img, plate_info = detect_vehicles_and_plates(image_path)

    # Show original image
    print("\n🖼️ Original Image:")
    img = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.show()

    # Show annotated image
    print("\n✅ Detection Results:")
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Vehicle counts
    print("\n🚗 Vehicle Counts:")
    for vehicle, count in vehicle_counts.items():
        if vehicle != 'total':
            print(f"{vehicle.capitalize()}: {count}")
    print(f"Total Vehicles: {vehicle_counts['total']}")

    # Plate info
    if plate_info:
        print("\n🔢 Detected Number Plates:")
        for i, plate in enumerate(plate_info, 1):
            print(f"Plate {i}: {plate['text']} (Confidence: {plate['confidence']:.2f})")
    else:
        print("\n❌ No number plates detected.")

    # Green light timing
    green_time = calculate_green_light_time(vehicle_counts)
    print(f"\n⏱️ Calculated Green Light Time: {green_time} seconds")

    # Save result
    output_path = "annotated_output.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"\n📝 Annotated image saved to: {output_path}")

    # Simulation
    simulate = input("\nDo you want to see the traffic light simulation? (y/n): ")
    if simulate.strip().lower() == 'y':
        traffic_light_simulation(green_time)


# ------------------------------------
# 🚀 Upload and Run
# ------------------------------------
print("📤 Please upload an image with traffic:")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"\n📸 Processing image: {filename}")
    process_image_and_calculate_timing(filename)
