"""Offline video annotation: run fire/smoke or weapon detection on a video file.

Reads every frame, runs the same models as the live pipeline (server.inference),
draws overlays with the shared Annotator, and writes an MP4 that preserves the
source frame rate, resolution, and audio track.

Usage:
    python scripts/annotate_video.py --input assets/camera_1.mp4 \
        --output annotated_assets/camera_1.mp4 --mode fire --camera-id 1
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.inference.yolo_runner import yolo_runner
from server.inference.annotator import Annotator

BATCH_SIZE = 8


def probe_stream(path: str) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height,nb_frames",
        "-of", "json", path,
    ])
    return json.loads(out)["streams"][0]


def has_audio(path: str) -> bool:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_name", "-of", "csv=p=0", path,
    ])
    return bool(out.strip())


def open_writer(output: str, source: str, width: int, height: int, fps: str) -> subprocess.Popen:
    """ffmpeg process consuming raw BGR frames; muxes original audio back in."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-framerate", fps, "-i", "pipe:0",
    ]
    audio = has_audio(source)
    if audio:
        cmd += ["-i", source]
    cmd += ["-map", "0:v"]
    if audio:
        cmd += ["-map", "1:a", "-c:a", "copy"]
    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-shortest", output,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=["fire", "weapon"])
    parser.add_argument("--camera-id", type=int, required=True)
    args = parser.parse_args()

    info = probe_stream(args.input)
    fps_rational = info["r_frame_rate"]
    fps = eval(fps_rational)  # e.g. "30000/1001"

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(args.output, args.input, width, height, fps_rational)

    run_batch = (
        yolo_runner.run_fire_batch if args.mode == "fire" else yolo_runner.run_weapon_batch
    )

    total = 0
    start = time.time()
    while True:
        frames = []
        for _ in range(BATCH_SIZE):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            break

        t0 = time.time()
        detections = run_batch(frames)
        latency_ms = (time.time() - t0) * 1000 / len(frames)

        for frame, dets in zip(frames, detections):
            annotated = Annotator.draw_overlays(
                frame, dets,
                zone_config=None, is_alert=False,
                current_fps=fps, latency_ms=latency_ms,
                camera_id=args.camera_id,
            )
            writer.stdin.write(annotated.tobytes())
        total += len(frames)
        if total % 96 == 0:
            elapsed = time.time() - start
            print(f"  {total} frames ({total / elapsed:.1f} fps processing)", flush=True)

    cap.release()
    writer.stdin.close()
    if writer.wait() != 0:
        sys.exit("ffmpeg encoding failed")
    print(f"Done: {total} frames -> {args.output} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
