import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

def infer_webcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load YOLO
    yolo_model = YOLO(r'runs/detect/train5/weights/best.pt')
    yolo_model.export(format="ncnn")
    model = YOLO(r"C:\Users\kesha\test venv\runs\detect\train5\weights\best_ncnn_model")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        results = yolo_model(frame, device=device, conf=0.5)

        #Draw bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                #Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                #labels & confidence
                if hasattr(box, 'cls') and hasattr(box, 'conf'):
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    cls_name = result.names[cls_id] if result.names else f"Class {cls_id}"
                    label = f"{cls_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLO Webcam Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
infer_webcam()
