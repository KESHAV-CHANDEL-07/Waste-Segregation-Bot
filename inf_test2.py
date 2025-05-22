from ultralytics import YOLO
import cv2
import time
import torch
import threading
import queue
import os

# Pi-specific optimizations
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Use all cores for OpenBLAS
cv2.ocl.setUseOpenCL(True)  # Enable OpenCL if available
torch.cuda.is_available = lambda : False  # Force CPU

# Load and export model (run once)
#model = YOLO(r"C:\Users\kesha\test venv\runs\detect\train5\weights\best.pt", task="detect")  # Use smallest model possible
#model.export(format="ncnn", half=True)  # Uncomment first time
ncnn_model = YOLO(r"C:\Users\kesha\test venv\runs\detect\train5\weights\best_ncnn_model\model.ncnn.bin", task="detect")
ncnn_model.to('cpu')


# Threading queues with small buffers to reduce memory usage
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def capture_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every other frame
        frame_counter += 1
        if frame_counter % 2 != 0:
            continue
            
        # Lower resolution for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Don't block if queue is full - drop frames instead
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

def inference():
    while True:
        if frame_queue.empty():
            time.sleep(0.01)  # Prevent CPU spinning
            continue
            
        frame = frame_queue.get()
        try:
            results = ncnn_model(frame, verbose=False)
            result_queue.put((frame, results))
        except Exception as e:
            print(f"Inference error: {e}")
            continue

def check_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
        return temp
    except:
        return 0

# Start threads
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=inference, daemon=True).start()

# Main loop - with adaptive frame rate
prev_time = time.time()
temp_check_time = time.time()
fps_values = []
running = True

while running:
    # Check temperature every 30 seconds
    current_time = time.time()
    if current_time - temp_check_time > 30:
        temp = check_cpu_temp()
        if temp > 80:  # Temperature threshold
            print(f"Warning: CPU temperature high ({temp}°C), throttling may occur")
        temp_check_time = current_time
    
    try:
        # Non-blocking get with timeout
        try:
            frame, results = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        # Simplified visualization - draw boxes for all detections
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                class_id = int(box.cls)
                class_name = ncnn_model.names[class_id]
                
                # Draw boxes with confidence and class names
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Calculate FPS with smoothing
        current_time = time.time()
        instantaneous_fps = 1 / (current_time - prev_time)
        fps_values.append(instantaneous_fps)
        if len(fps_values) > 10:
            fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values)
        prev_time = current_time
        
        # Only update FPS display every 10 frames
        if len(fps_values) % 5 == 0:
            frame = cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("YOLO Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
    
    except Exception as e:
        print(f"Display error: {e}")
        continue

cv2.destroyAllWindows()
