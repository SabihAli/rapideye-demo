#!/usr/bin/env python3
"""Facial recognition test pipeline CLI — see FACIAL_REC_FEASIBILITY.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.facial_rec.benchmark import (  # noqa: E402
    load_frame_sources,
    process_video,
    run_multistream_bench,
    run_single_stream_bench,
)
from server.facial_rec.config import FacialRecConfig  # noqa: E402
from server.facial_rec.pipeline import FacialRecTestPipeline  # noqa: E402


def _synthetic_face_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Fallback frame when no sample images exist (tracker/pipeline smoke only)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 120), (440, 400), (180, 160, 140), -1)
    cv2.circle(frame, (280, 220), 20, (30, 30, 30), -1)
    cv2.circle(frame, (360, 220), 20, (30, 30, 30), -1)
    return frame


def _resolve_source(source: Path | None) -> list:
    if source is None:
        sample_dir = ROOT / "data" / "facial_rec" / "samples"
        if sample_dir.exists() and any(sample_dir.iterdir()):
            return load_frame_sources(sample_dir, streams=1)
        print("No sample images found — using synthetic frame (detection may be empty).")
        return [_synthetic_face_frame()]
    return load_frame_sources(source, streams=1)


def cmd_bench(args: argparse.Namespace) -> int:
    config = FacialRecConfig.from_env()
    pipeline = FacialRecTestPipeline(config)
    frames = _resolve_source(Path(args.source) if args.source else None)
    report = run_single_stream_bench(pipeline, frames[0], args.iterations)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_multistream(args: argparse.Namespace) -> int:
    config = FacialRecConfig.from_env()
    pipeline = FacialRecTestPipeline(config)
    source = Path(args.source) if args.source else ROOT / "data" / "facial_rec" / "samples"
    if not source.exists():
        frames = [_synthetic_face_frame() for _ in range(args.streams)]
    else:
        frames = load_frame_sources(source, streams=args.streams)
    report = run_multistream_bench(pipeline, frames, args.frames)
    print(json.dumps(report.to_dict(), indent=2))
    per_stream_fps = report.effective_fps / max(report.streams, 1)
    print(f"\nApprox FPS per stream: {per_stream_fps:.2f}")
    target = 15.0
    status = "PASS" if per_stream_fps >= target else "BELOW_TARGET"
    print(f"Target {target} FPS/stream: {status}")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    config = FacialRecConfig.from_env()
    pipeline = FacialRecTestPipeline(config)
    video_path = Path(args.input)
    count = 0
    for result in process_video(pipeline, video_path, max_frames=args.max_frames):
        count += 1
        if count % 30 == 0:
            print(
                f"frame={result.frame_index} faces={len(result.detections)} "
                f"det_ms={result.detect_ms:.1f} rec_ms={result.recognize_ms:.1f}"
            )
    print(f"Processed {count} frames from {video_path}")
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    config = FacialRecConfig.from_env()
    pipeline = FacialRecTestPipeline(config)
    image = cv2.imread(str(Path(args.image)))
    if image is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 1
    sim = pipeline.enroll_from_frame(image, args.identity_id, args.name)
    print(f"Enrolled {args.name} ({args.identity_id}), self-match similarity={sim:.3f}")
    print(f"Gallery: {config.gallery_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Facial recognition test pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="Single-stream detect+track+recognize benchmark")
    bench.add_argument("--source", default=None, help="Image file or directory")
    bench.add_argument("--iterations", type=int, default=100)
    bench.set_defaults(func=cmd_bench)

    multi = sub.add_parser("multistream", help="Simulate N streams (FACIAL_REC.md load test)")
    multi.add_argument("--source", default=None, help="Image file or directory")
    multi.add_argument("--streams", type=int, default=4)
    multi.add_argument("--frames", type=int, default=200, help="Iterations per stream cycle")
    multi.set_defaults(func=cmd_multistream)

    video = sub.add_parser("video", help="Process a video file")
    video.add_argument("--input", required=True)
    video.add_argument("--max-frames", type=int, default=300)
    video.set_defaults(func=cmd_video)

    enroll = sub.add_parser("enroll", help="Enroll identity into gallery JSON")
    enroll.add_argument("--image", required=True)
    enroll.add_argument("--identity-id", required=True)
    enroll.add_argument("--name", required=True)
    enroll.set_defaults(func=cmd_enroll)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
