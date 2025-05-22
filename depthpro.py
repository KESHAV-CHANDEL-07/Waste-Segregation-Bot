import cv2
from pyparsing import C
import torch
import numpy as np
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

# Load YOLOv10 Model (Change path to your trained model)
yolo_model = YOLO(r"C:\Users\kesha\test venv\runs\detect\train6\weights\best.pt")  # Ensure your trained model is available

# Load Depth Estimation Model
depth_model_name = "LiheYoung/depth-anything-small-hf"  # Pre-trained model
image_processor = AutoImageProcessor.from_pretrained(depth_model_name)
depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Camera Parameters (Adjust according to your camera)
FOCAL_LENGTH_PIXELS = 500  # Example: 500px (check camera specs)
BASELINE = 0.1  # Assumed 10cm distance between stereo cameras (if using stereo)
CAMERA_WIDTH = 640  # Change according to your camera resolution

# Read the Image
image_path = r"C:\Users\kesha\Downloads\WhatsApp Image 2025-03-25 at 21.55.30_f0ad5128.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image.")
    exit()

# Convert Image to RGB (for depth model)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(image_rgb)

# Run Depth Estimation
inputs = image_processor(images=pil_image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    depth_output = depth_model(**inputs).predicted_depth

# Convert Depth Map to Numpy
depth_map = depth_output.squeeze().cpu().numpy()
depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

# Run YOLO Object Detection
results = yolo_model(image)

# Process YOLO results and estimate distance
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        cls_id = int(box.cls[0].item())  # Class ID
        conf = box.conf[0].item()  # Confidence score

        # Get depth value at the center of the bounding box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        depth_value = depth_map[center_y, center_x]  # Depth at object center

        # Estimate real-world distance
        distance = depth_value * FOCAL_LENGTH_PIXELS / max(abs(center_x - CAMERA_WIDTH / 2), 1)

        print(f"Detected: Class {cls_id} | Confidence: {conf:.2f} | Distance: {distance:.2f} meters")

        # Draw Bounding Box & Distance
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{distance:.2f}m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow("Object Detection with Distance Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
