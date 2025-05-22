from ultralytics import YOLO
import cv2
import time
import torch
import threading
import queue

# Force CPU
torch.cuda.is_available = lambda : False

# Load and export model (run once)
model = YOLO(r"runs\detect\train5\weights\best.pt", task="detect")
model.export(format="ncnn", half=True)  # Uncomment first time
ncnn_model = YOLO("./yolo11n_ncnn_model", task="detect")

# Threading queues
frame_queue = queue.Queue(maxsize=4)
result_queue = queue.Queue(maxsize=4)

def capture_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 640))  # Smaller input
        frame_queue.put(frame)
    cap.release()

def inference():
    while True:
        frame = frame_queue.get()
        results = ncnn_model(frame)
        result_queue.put((frame, results))

# Start threads
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=inference, daemon=True).start()

# Main loop
prev_time = time.time()
while True:
    frame, results = result_queue.get()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf)
            if confidence > 0.5:  # Filter low-confidence
                class_name = ncnn_model.names[int(box.cls)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("YOLO11 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

