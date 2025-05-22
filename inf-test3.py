import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
import RPi.GPIO as GPIO
import math

# === SERVO SETUP ===
servo_pins = [17, 18, 27, 22]  # [base, shoulder, elbow, gripper]
GPIO.setmode(GPIO.BCM)
for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)

servos = [GPIO.PWM(pin, 50) for pin in servo_pins]
for servo in servos:
    servo.start(0)
    time.sleep(0.2)

def set_angle(servo, angle):
    duty = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.4)
    servo.ChangeDutyCycle(0)

# === CAMERA PARAMETERS ===
CAMERA_FOV = 62  # horizontal FOV in degrees
IMG_WIDTH = 640  # frame width

# === ARM DIMENSIONS ===
L1 = 10  # shoulder to elbow length (cm)
L2 = 10  # elbow to gripper length (cm)
FOCAL_LENGTH = 615  # in pixels
REAL_WIDTH = 7.5  # actual object width (cm) — adjust per your object

# === MATH HELPERS ===
def pixel_to_angle(x_pixel):
    center = IMG_WIDTH / 2
    offset = x_pixel - center
    angle_per_pixel = CAMERA_FOV / IMG_WIDTH
    return offset * angle_per_pixel  # degrees

def get_object_position(x_angle_deg, distance_cm):
    x_rad = math.radians(x_angle_deg)
    X = distance_cm * math.sin(x_rad)
    Y = distance_cm * math.cos(x_rad)
    return X, Y, 0

def calculate_angles(X, Y):
    D = math.sqrt(X**2 + Y**2)
    D = min(D, L1 + L2)

    try:
        elbow_angle = math.degrees(math.acos((L1**2 + L2**2 - D**2) / (2 * L1 * L2)))
        alpha = math.atan2(Y, X)
        beta = math.acos((L1**2 + D**2 - L2**2) / (2 * L1 * D))
        shoulder_angle = math.degrees(alpha + beta)
    except:
        shoulder_angle, elbow_angle = 90, 90  # fallback values if math domain error

    return shoulder_angle, elbow_angle

def move_arm(base_angle, shoulder_angle, elbow_angle):
    set_angle(servos[0], 90 + base_angle)
    set_angle(servos[1], shoulder_angle)
    set_angle(servos[2], 180 - elbow_angle)
    set_angle(servos[3], 80)  # close gripper

# === YOLO DETECTION + ARM CONTROL ===
def infer_webcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load custom YOLO model
    yolo_model = YOLO(r'C:\Users\kesha\test venv\runs\detect\train5\weights\best.pt')
    yolo_model.to(device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            results = yolo_model(frame, device=device, half=True)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    cls_name = result.names[cls_id] if result.names else f"Class {cls_id}"
                    label = f"{cls_name}: {conf:.2f}"

                    center_x = (x1 + x2) // 2
                    box_width = x2 - x1
                    distance_cm = (FOCAL_LENGTH * REAL_WIDTH) / box_width

                    # Angle + IK
                    base_angle = pixel_to_angle(center_x)
                    X, Y, _ = get_object_position(base_angle, distance_cm)
                    shoulder, elbow = calculate_angles(X, Y)

                    # Move arm
                    move_arm(base_angle, shoulder, elbow)

                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("YOLO Webcam Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        for servo in servos:
            servo.ChangeDutyCycle(0)
            servo.stop()
        GPIO.cleanup()

infer_webcam()
