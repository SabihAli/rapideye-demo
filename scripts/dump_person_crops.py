"""
Offline verification: run the person detector on a video and save the padded
crops that would actually be sent to the weapon detector in production
(server.inference.yolo_runner.YoloRunner._padded_bbox), so crop generosity
can be checked visually against weapons that extend outside the raw person
bounding box (e.g. an extended rifle).

Usage:
    .venv/bin/python scripts/dump_person_crops.py [video] [--every N] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.config import settings  # noqa: E402
from server.inference.ort_models import OrtYoloDetector  # noqa: E402
from server.inference.yolo_runner import YoloRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default="assets/test_sample_weapon.mp4")
    parser.add_argument("--every", type=int, default=5, help="Sample every Nth frame")
    parser.add_argument("--out", default="data/person_crops")
    args = parser.parse_args()

    out_dir = settings.project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the fp32 model explicitly: production's own fallback logic
    # (YoloRunner._needs_fp32_fallback) currently always lands person here too
    # since the int8 model won't build on TensorRT, so this matches what's
    # actually live rather than what settings.detector_onnx() would prefer.
    detector = OrtYoloDetector(
        settings.onnx_dir / "person.onnx",
        name="person_crop_check",
        conf=settings.person_conf,
        imgsz=settings.person_imgsz,
        keep_classes=[0],
    )

    video_path = settings.project_root / args.video
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[dump_person_crops] {video_path.name}: {total} frames, sampling every {args.every}")

    saved = 0
    for idx in range(0, total, args.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = detector.infer([frame])[0]
        for i, row in enumerate(dets):
            x1, y1, x2, y2, conf, _ = row.tolist()
            raw_bbox = (int(x1), int(y1), int(x2), int(y2))
            padded = YoloRunner._padded_bbox(frame, raw_bbox)
            if padded is None:
                continue
            px1, py1, px2, py2 = padded
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            raw_w, raw_h = x2 - x1, y2 - y1
            pad_w, pad_h = px2 - px1, py2 - py1
            out_path = out_dir / (
                f"frame{idx:04d}_p{i}_conf{conf:.2f}"
                f"_raw{int(raw_w)}x{int(raw_h)}_pad{pad_w}x{pad_h}.jpg"
            )
            cv2.imwrite(str(out_path), crop)
            saved += 1
    cap.release()
    print(f"[dump_person_crops] saved {saved} padded person crops -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
