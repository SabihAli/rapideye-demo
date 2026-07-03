import os
import cv2
import torch
from ultralytics import YOLO

# 1-indexed selector for the videos:
# 1 -> 001_gun.mp4
# 2 -> 002_gun.mp4
VIDEO_INDEX = 2

def main():
    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_path = os.path.join(project_root, "data", "models", "weapons_yolov8.pt")

    # Explicit list of video paths requested by the user
    video_paths = [
        "c:/Programming/Projects/03_PROTOTYPE/rapideye_prototype/rapideye-demo/data/videos/weapons/001_gun.mp4",
        "c:/Programming/Projects/03_PROTOTYPE/rapideye_prototype/rapideye-demo/data/videos/weapons/002_gun.mp4",
    ]

    # 1. Check if model exists
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    # 2. Validate selected index
    if VIDEO_INDEX < 1 or VIDEO_INDEX > len(video_paths):
        print(f"[ERROR] Invalid VIDEO_INDEX: {VIDEO_INDEX}. Must be between 1 and {len(video_paths)}.")
        return

    selected_video_path = video_paths[VIDEO_INDEX - 1]
    video_filename = os.path.basename(selected_video_path)
    print(f"Selected Video Index: {VIDEO_INDEX}")
    print(f"Playing video: {selected_video_path}")

    if not os.path.exists(selected_video_path):
        print(f"[ERROR] Video file does not exist at: {selected_video_path}")
        return

    # 3. Setup GPU / CUDA device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 4. Load YOLOv8 Weapons Model
    print(f"Loading YOLOv8 model from {model_path}...")
    try:
        model = YOLO(model_path)
        # Move model to the selected device (CUDA/CPU)
        model.to(device)
        print("Model loaded successfully!")
        print("Model Class Names:", model.names)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # 5. Open Video Stream
    cap = cv2.VideoCapture(selected_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {selected_video_path}")
        return

    window_name = f"RapidEye Weapons Detection - {video_filename}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\nStarting video playback on {device}. Press 'q' or 'ESC' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Run inference using YOLOv8 natively on the selected device
        results = model(frame, conf=0.40, device=device, verbose=False)

        # Plot bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow(window_name, annotated_frame)

        # Break loop on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test completed.")

if __name__ == "__main__":
    main()
