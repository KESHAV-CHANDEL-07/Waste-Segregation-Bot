import cv2
import math
from ultralytics import YOLO
import time

# ==== CONFIGURATION ====
MODEL_PATH = r"C:\Users\kesha\test venv\runs\detect\train5\weights\best.pt"
CAMERA_HEIGHT_M = 0.45             # Height of camera from ground (meters)
CAMERA_ANGLE_DEG = 30              # Tilt angle from horizontal
VERTICAL_FOV_DEG = 49.5            # Camera vertical FOV
FRAME_WIDTH = 480
FRAME_HEIGHT = 480

# ==== FUNCTION TO ESTIMATE DISTANCE ====
def estimate_distance(y_pixel):
    delta_angle_deg = ((y_pixel / FRAME_HEIGHT) - 0.5) * VERTICAL_FOV_DEG
    total_angle_deg = CAMERA_ANGLE_DEG + delta_angle_deg
    total_angle_rad = math.radians(total_angle_deg)

    if math.tan(total_angle_rad) <= 0:
        return None

    return CAMERA_HEIGHT_M / math.tan(total_angle_rad)

# ==== LOAD CUSTOM YOLO MODEL ====
model = YOLO(MODEL_PATH)

# ==== INITIALIZE CAMERA ====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

cooldown_time = 2  # seconds between detections
last_print_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame,conf=0.3, verbose=False)[0]

    nearest_distance = float('inf')
    target_info = None

    # Find the nearest relevant object
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label not in ["biodegradable", "non-biodegradable"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        object_y = y2
        distance = estimate_distance(object_y)
        if distance and distance < nearest_distance:
            nearest_distance = distance
            target_info = {
                'label': label,
                'distance': distance,
                'box': (x1, y1, x2, y2)
            }

    # Display only nearest object (if any)
    if target_info:
        x1, y1, x2, y2 = target_info['box']
        label = target_info['label']
        distance = target_info['distance']
        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {distance:.2f} m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Print LEFT or RIGHT only once every few seconds
        current_time = time.time()
        if current_time - last_print_time > cooldown_time:
            if label == "biodegradable":
                print("LEFT")
            elif label == "non-biodegradable":
                print("RIGHT")
            last_print_time = current_time

    # Show frame
    cv2.imshow("Waste Detection + Distance Estimation", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
