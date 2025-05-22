import cv2
import torch
import numpy as np
import json
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time
import matplotlib.pyplot as plt

# ========== Load Scale Factor ==========
with open("multi_distance_scale.json", "r") as f:
    scale_data = json.load(f)
scale_factor = scale_data["average_scale_factor"]
print(f"✅ Loaded scale factor: {scale_factor:.5f} meters/unit")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)

video_path = 0  # Use 0 for webcam or replace with video path (e.g., "test_video.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video or webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_idx = 0
start_time = time.time()
processing_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame_start_time = time.time()

    # Convert BGR to RGB PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess and run through model
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted_depth = model(**inputs).predicted_depth

    # Interpolate to match original frame size
    depth_raw = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(frame_height, frame_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Real-world distance at center
    cx, cy = frame_width // 2, frame_height // 2
    raw_depth = depth_raw[cy, cx]
    real_distance_m = raw_depth * scale_factor
    print(f"📏 Depth at center: {real_distance_m:.2f} meters")

    # Visualize depth
    depth_normalized = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min())
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

    # Annotate on frame
    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    cv2.putText(frame, f"{real_distance_m:.2f} m", (cx - 50, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Combine views
    combined = np.hstack((frame, depth_colored))
    cv2.imshow("RGB (Left) | Depth Map (Right)", combined)

    # FPS tracking
    frame_time = time.time() - frame_start_time
    processing_times.append(frame_time)

    if frame_idx % 10 == 0:
        fps_now = 1.0 / (sum(processing_times[-10:]) / len(processing_times[-10:]))
        print(f"FPS: {fps_now:.2f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ========== Cleanup ==========
cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()

# ========== Summary ==========
total_time = time.time() - start_time
avg_fps = frame_idx / total_time
print(f"\n✅ Processed {frame_idx} frames in {total_time:.2f}s (Avg FPS: {avg_fps:.2f})")

# Optional FPS graph
if len(processing_times) > 1:
    plt.plot([1.0 / t for t in processing_times])
    plt.title("Frame Processing FPS")
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.grid(True)
    plt.show()
