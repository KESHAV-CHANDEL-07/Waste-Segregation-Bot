import RPi.GPIO as GPIO
import subprocess
import time

BUTTON_PIN = 17  # GPIO pin number

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def button_callback(channel):
    print("Button pressed! Executing script...")
    subprocess.run(["python3", "/home/pi/d4.py"])  # Update with your script's path

GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=button_callback, bouncetime=300)

print("Waiting for button press...")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    GPIO.cleanup()
