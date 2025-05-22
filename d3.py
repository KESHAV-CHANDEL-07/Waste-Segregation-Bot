import cv2
import math
from ultralytics import YOLO

#Config
MODEL_PATH = r"C:\Users\kesha\test venv\runs\detect\train5\weights\best.pt"# Path to your custom YOLO model
CAMERA_HEIGHT_M = 0.95             # Camera height from the ground (in meters)
CAMERA_ANGLE_DEG = 30             # Downward tilt angle from horizontal
VERTICAL_FOV_DEG = 49.5            # Vertical Field of View of your camera (in degrees)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

#FUNCTION TO ESTIMATE DISTANCE
def estimate_distance(y_pixel):
    delta_angle_deg = ((y_pixel / FRAME_HEIGHT) - 0.5) * VERTICAL_FOV_DEG
    total_angle_deg = CAMERA_ANGLE_DEG + delta_angle_deg
    total_angle_rad = math.radians(total_angle_deg)

    if math.tan(total_angle_rad) <= 0:
        return None

    distance = CAMERA_HEIGHT_M / math.tan(total_angle_rad)
    return distance

#load your model
model = YOLO(MODEL_PATH)

# camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]  

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        object_y = y2 

        distance = estimate_distance(object_y)
        if distance:
            # Draw bounding box and distance text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)
            cv2.putText(frame, f"{label}: {distance:.2f} m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display result
    cv2.imshow("Waste Detection + Distance Estimation", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
