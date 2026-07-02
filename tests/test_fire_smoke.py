import os
import sys
import cv2
import torch

# 1-indexed selector for the videos:
# 1 -> 001_fire.mp4
# 2 -> 002_fire.mp4
# 3 -> 003_fire.mp4
VIDEO_INDEX = 3

def main():
    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_path = os.path.join(project_root, "data", "models", "fire_smoke_yolov5.pt")
    
    # Explicit list of video paths requested by the user
    video_paths = [
        "c:/Programming/Projects/03_PROTOTYPE/rapideye_prototype/rapideye-demo/data/videos/fire/001_fire.mp4",
        "c:/Programming/Projects/03_PROTOTYPE/rapideye_prototype/rapideye-demo/data/videos/fire/002_fire.mp4",
        "c:/Programming/Projects/03_PROTOTYPE/rapideye_prototype/rapideye-demo/data/videos/fire/003_fire.mp4",
    ]

    # Validate model exists
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    # Validate selected index
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

    # 3. Load YOLOv5 Model via PyTorch Hub (with trust_repo=True to auto-accept the prompt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading YOLOv5 model from {model_path} via ultralytics/yolov5 PyTorch Hub...")
    try:
        # Load the custom model using official yolov5 github hub config and move to device
        # Note: trust_repo=True allows it to download and load without prompt blocking
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True).to(device)
        model.conf = 0.25  # NMS confidence threshold
        print("Model loaded successfully!")
        print("Model Class Names:", model.names)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # 4. Open Video Stream
    cap = cv2.VideoCapture(selected_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {selected_video_path}")
        return

    window_name = f"RapidEye Fire/Smoke Detection - {video_filename}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nStarting video playback. Press 'q' or 'ESC' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Run inference (YOLOv5 hub model expects RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        # Draw detections on the frame (results.render() returns list of images with detections in RGB)
        annotated_frames = results.render()
        annotated_frame = cv2.cvtColor(annotated_frames[0], cv2.COLOR_RGB2BGR)

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
