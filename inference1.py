import torch
from ultralytics import YOLO
import cv2
import time

def infer_webcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load YOLO model
    yolo_model = YOLO(r'runs\detect\train23\weights\best.pt')
    yolo_model.to(device)

    # Uncomment the following line if you want to use FP16 precision (half)
    # yolo_model.half()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Inference
        results = yolo_model(frame)  # Automatically uses the correct device

        # Draw bounding boxes and labels
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = result.names[cls_id] if result.names else f"Class {cls_id}"
                label = f"{cls_name}: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Calculate and display FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLO Webcam Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the webcam inference
infer_webcam()
