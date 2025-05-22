import cv2
import torch
import math
import numpy as np
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time

# ==== CONFIGURATION ====
model = YOLO('best.pt')  # Load YOLO model

# Camera parameters
cam_height = 35  # in cm
image_width = 1280
image_height = 720
hfov = 60  # horizontal FOV in degrees
vfov = 45  # vertical FOV in degrees
CAMERA_TILT_DEG = 30  # downward tilt from horizontal

# Robotic arm dimensions in cm
L1, L2, L3 = 20, 17, 7

# Servo GPIO pins
servo_pins = [15, 18, 23, 24]  # Update to your pin numbers for servos 1 to 4 (claw)

# ==== GPIO SETUP ====
GPIO.setmode(GPIO.BCM)
pwms = []
for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, 50)  # 50Hz
    pwm.start(0)
    pwms.append(pwm)

def set_servo_angle(pwm, angle):
    duty = 2.5 + (angle / 180.0) * 10
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)

# ==== MATH FUNCTIONS ====
def pixel_to_world(px, py):
    x_angle = (px - image_width / 2) * (hfov / image_width)
    y_angle = (py - image_height / 2) * (vfov / image_height)
    y_angle_total = y_angle + CAMERA_TILT_DEG

    x_rad = math.radians(x_angle)
    y_rad = math.radians(y_angle_total)

    y = cam_height / math.tan(y_rad)
    x = y * math.tan(x_rad)

    return x, y, 0  # z=0 for ground

def inverse_kinematics(x, y, z):
    theta1 = math.atan2(y, x)
    r = math.sqrt(x**2 + y**2)
    reach = r - L3
    height = z
    D = (reach**2 + height**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = max(min(D, 1), -1)
    theta3 = math.atan2(math.sqrt(1 - D**2), D)
    theta2 = math.atan2(height, reach) - math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta4 = -(theta2 + theta3)
    return [math.degrees(a) for a in [theta1, theta2, theta3, theta4]]

# ==== CAMERA SETUP ====
cap = cv2.VideoCapture(0)
cap.set(3, image_width)
cap.set(4, image_height)


try:
    # Take only the first frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera read failed")

    results = model(frame)[0]

    found = False
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        x, y, z = pixel_to_world(cx, cy)

        try:
            angles = inverse_kinematics(x, y, z)
            print(f"Angles: {angles}")

            # Move 3 arm servos
            for i in range(3):
                set_servo_angle(pwms[i], angles[i])

            # Claw close
            print("Closing claw...")
            set_servo_angle(pwms[3], 60)  # Adjust angle to close claw
            time.sleep(3)

            found = True
            break
        except:
            print("Target out of reach or error in IK.")

    if not found:
        print("No valid object detected.")

finally:
    for pwm in pwms:
        pwm.stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()

