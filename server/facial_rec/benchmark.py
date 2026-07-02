from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from server.facial_rec.config import FacialRecConfig
from server.facial_rec.pipeline import FacialRecTestPipeline
from server.facial_rec.schemas import FrameResult


@dataclass
class BenchmarkReport:
    frames: int
    streams: int
    total_ms: float
    mean_detect_ms: float
    mean_recognize_ms: float
    mean_faces_per_frame: float
    effective_fps: float
    vram_mb: float | None

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "streams": self.streams,
            "total_ms": round(self.total_ms, 2),
            "mean_detect_ms": round(self.mean_detect_ms, 2),
            "mean_recognize_ms": round(self.mean_recognize_ms, 2),
            "mean_faces_per_frame": round(self.mean_faces_per_frame, 3),
            "effective_fps": round(self.effective_fps, 2),
            "vram_mb": round(self.vram_mb, 1) if self.vram_mb is not None else None,
        }


def _read_vram_mb() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return None


def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_frame_sources(source: Path, streams: int) -> list[np.ndarray]:
    if source.is_file():
        img = load_image_bgr(source)
        return [img.copy() for _ in range(streams)]
    if source.is_dir():
        images = sorted(
            [p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if not images:
            raise FileNotFoundError(f"No images in {source}")
        frames = [load_image_bgr(p) for p in images]
        while len(frames) < streams:
            frames.append(frames[-1].copy())
        return frames[:streams]
    raise FileNotFoundError(f"Source not found: {source}")


def run_single_stream_bench(
    pipeline: FacialRecTestPipeline,
    frame: np.ndarray,
    iterations: int,
) -> BenchmarkReport:
    detect_ms: list[float] = []
    recognize_ms: list[float] = []
    faces: list[int] = []
    t0 = time.perf_counter()
    for i in range(iterations):
        result = pipeline.process_frame(frame, stream_id=0)
        detect_ms.append(result.detect_ms)
        recognize_ms.append(result.recognize_ms)
        faces.append(len(result.detections))
    total_ms = (time.perf_counter() - t0) * 1000.0
    return BenchmarkReport(
        frames=iterations,
        streams=1,
        total_ms=total_ms,
        mean_detect_ms=float(np.mean(detect_ms)),
        mean_recognize_ms=float(np.mean(recognize_ms)),
        mean_faces_per_frame=float(np.mean(faces)),
        effective_fps=iterations / (total_ms / 1000.0) if total_ms > 0 else 0.0,
        vram_mb=_read_vram_mb(),
    )


def run_multistream_bench(
    pipeline: FacialRecTestPipeline,
    frames: list[np.ndarray],
    iterations: int,
) -> BenchmarkReport:
    streams = len(frames)
    detect_ms: list[float] = []
    recognize_ms: list[float] = []
    faces: list[int] = []
    total_frames = iterations * streams
    t0 = time.perf_counter()
    for _ in range(iterations):
        for stream_id, frame in enumerate(frames):
            result = pipeline.process_frame(frame, stream_id=stream_id)
            detect_ms.append(result.detect_ms)
            recognize_ms.append(result.recognize_ms)
            faces.append(len(result.detections))
    total_ms = (time.perf_counter() - t0) * 1000.0
    return BenchmarkReport(
        frames=total_frames,
        streams=streams,
        total_ms=total_ms,
        mean_detect_ms=float(np.mean(detect_ms)),
        mean_recognize_ms=float(np.mean(recognize_ms)),
        mean_faces_per_frame=float(np.mean(faces)),
        effective_fps=total_frames / (total_ms / 1000.0) if total_ms > 0 else 0.0,
        vram_mb=_read_vram_mb(),
    )


def process_video(
    pipeline: FacialRecTestPipeline,
    video_path: Path,
    max_frames: int = 300,
    stream_id: int = 0,
) -> Iterable[FrameResult]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            yield pipeline.process_frame(frame, stream_id=stream_id)
    finally:
        cap.release()
