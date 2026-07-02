#!/usr/bin/env python3
"""
Standalone facial recognition video test — FACIAL_REC.md Option B.

Runs SCRFD + buffalo_l (InsightFace), tracks faces, matches against a gallery,
and writes an annotated output video. No imports from demo-app server packages.

Usage:
  pip install opencv-python-headless insightface onnxruntime numpy
  python facial_rec_video_test.py --input data/facial_rec/samples/P1E_S2_C1 --output out/annotated.mp4
  python facial_rec_video_test.py --input clip.mp4 --gallery data/facial_rec/gallery_built/gallery.json

Gallery layout (optional):
  gallery/
    Alice/photo1.jpg
    Bob/photo2.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Gallery — cosine match against enrolled ArcFace embeddings
# ---------------------------------------------------------------------------


@dataclass
class GalleryEntry:
    identity_id: str
    display_name: str
    embedding: np.ndarray


class FaceGallery:
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        self.entries: list[GalleryEntry] = []

    def enroll(self, identity_id: str, display_name: str, embedding: np.ndarray) -> None:
        emb = embedding.astype(np.float32)
        emb /= max(np.linalg.norm(emb), 1e-12)
        self.entries = [e for e in self.entries if e.identity_id != identity_id]
        self.entries.append(GalleryEntry(identity_id, display_name, emb))

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        if not self.entries:
            return "Unknown", 0.0
        query = embedding.astype(np.float32)
        query /= max(np.linalg.norm(query), 1e-12)
        best_name = "Unknown"
        best_sim = -1.0
        for entry in self.entries:
            sim = float(np.dot(query, entry.embedding))
            if sim > best_sim:
                best_sim = sim
                best_name = entry.display_name if sim >= self.threshold else "Unknown"
        return best_name, best_sim


def load_gallery_from_json(gallery_path: Path, threshold: float | None = None) -> FaceGallery:
    data = json.loads(gallery_path.read_text(encoding="utf-8"))
    gallery = FaceGallery(threshold=threshold if threshold is not None else float(data.get("threshold", 0.4)))
    for entry in data.get("identities", []):
        emb = np.array(entry["embedding"], dtype=np.float32)
        identity_id = entry["identity_id"]
        display_name = entry.get("display_name", identity_id)
        gallery.enroll(identity_id, display_name, emb)
        print(f"Enrolled {display_name} from {gallery_path.name}")
    return gallery


def load_gallery(gallery_arg: Path, app: "FaceAnalysisApp", threshold: float) -> FaceGallery:
    if gallery_arg.is_file() and gallery_arg.suffix.lower() == ".json":
        return load_gallery_from_json(gallery_arg, threshold)
    return load_gallery_from_dir(gallery_arg, app, threshold)


def load_gallery_from_dir(gallery_dir: Path, app: "FaceAnalysisApp", threshold: float) -> FaceGallery:
    gallery = FaceGallery(threshold=threshold)
    if not gallery_dir.is_dir():
        return gallery
    for person_dir in sorted(gallery_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: could not read {img_path}", file=sys.stderr)
                continue
            faces = app.analyze(img)
            if not faces:
                print(f"Warning: no face in gallery image {img_path}", file=sys.stderr)
                continue
            face = max(faces, key=lambda f: f.det_score)
            gallery.enroll(person_dir.name, person_dir.name, face.embedding)
            print(f"Enrolled {person_dir.name} from {img_path.name}")
            break
    return gallery


# ---------------------------------------------------------------------------
# Tracker — lightweight IoU (gates how often we relabel)
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    score: float
    label: str = "Unknown"
    similarity: float = 0.0
    missed: int = 0
    frames_since_label: int = 0
    detect_only: bool = False


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 20):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: list[tuple[tuple[int, int, int, int], float]]) -> list[Track]:
        matched: set[int] = set()
        for bbox, score in detections:
            best_idx = -1
            best_iou = 0.0
            for i, track in enumerate(self.tracks):
                if i in matched:
                    continue
                iou = _iou(track.bbox, bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                t = self.tracks[best_idx]
                t.bbox = bbox
                t.score = score
                t.missed = 0
                t.frames_since_label += 1
                matched.add(best_idx)
            else:
                self.tracks.append(
                    Track(track_id=self._next_id, bbox=bbox, score=score, frames_since_label=0)
                )
                self._next_id += 1
                matched.add(len(self.tracks) - 1)

        for i, track in enumerate(self.tracks):
            if i not in matched:
                track.missed += 1
                track.frames_since_label += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks


# ---------------------------------------------------------------------------
# InsightFace wrapper
# ---------------------------------------------------------------------------


@dataclass
class AnalyzedFace:
    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: np.ndarray


class FaceAnalysisApp:
    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
    ):
        from insightface.app import FaceAnalysis

        if ctx_id < 0:
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._app = FaceAnalysis(name=model_pack, providers=providers)
        self._app.prepare(ctx_id=max(ctx_id, 0), det_size=det_size)

    def analyze(self, image_bgr: np.ndarray) -> list[AnalyzedFace]:
        faces = self._app.get(image_bgr)
        out: list[AnalyzedFace] = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            emb = np.asarray(face.embedding, dtype=np.float32)
            out.append(
                AnalyzedFace(
                    bbox=(x1, y1, x2, y2),
                    det_score=float(face.det_score),
                    embedding=emb,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

COLOR_KNOWN = (40, 200, 40)
COLOR_UNKNOWN = (60, 60, 255)
COLOR_TRACK = (255, 200, 0)


def draw_annotations(
    frame: np.ndarray,
    tracks: list[Track],
    fps: float | None = None,
    frame_idx: int | None = None,
) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        color = COLOR_KNOWN if track.label != "Unknown" else COLOR_UNKNOWN
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"#{track.track_id}"
        if not track.detect_only:
            tag += f" {track.label}"
            if track.similarity > 0:
                tag += f" {track.similarity:.2f}"
        cv2.putText(out, tag, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if fps is not None:
        cv2.putText(
            out,
            f"FPS {fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COLOR_TRACK,
            2,
        )
    if frame_idx is not None:
        cv2.putText(
            out,
            f"frame {frame_idx}",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TRACK,
            2,
        )
    return out


# ---------------------------------------------------------------------------
# Input — video file, webcam, or frame directory (ChokePoint-style dataset)
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_frames_from_dir(frames_dir: Path) -> list[Path]:
    """Load ordered frame paths from all_file.txt or sorted image glob."""
    frames_dir = frames_dir.resolve()
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frames_dir}")

    manifest = frames_dir / "all_file.txt"
    if manifest.is_file():
        names = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
        paths = [frames_dir / name for name in names]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} frames listed in {manifest.name} are missing under {frames_dir} "
                f"(e.g. {missing[0].name}). Copy the JPG sequence into this folder."
            )
        return paths

    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No images in {frames_dir}")
    return paths


def resolve_input(input_arg: str) -> tuple[str, list[Path] | None]:
    """Return ('video', None) or ('frames', [paths])."""
    if input_arg.isdigit():
        return "video", None
    path = Path(input_arg)
    if path.is_dir():
        return "frames", list_frames_from_dir(path)
    if path.is_file():
        return "video", None
    raise FileNotFoundError(
        f"Input not found: {input_arg}\n"
        "Use a frame folder (e.g. data/facial_rec/samples/P1E_S2_C1) or a video file."
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _process_one_frame(
    frame: np.ndarray,
    frame_idx: int,
    *,
    gallery: FaceGallery,
    app: FaceAnalysisApp,
    tracker: FaceTracker,
    rec_interval: int,
    detect_only: bool,
    first_face_frame: int | None,
    faces_total: int,
    t_start: float,
) -> tuple[np.ndarray, int | None, int]:
    faces = app.analyze(frame)
    faces_total += len(faces)
    if faces and first_face_frame is None:
        first_face_frame = frame_idx
    detections = [(f.bbox, f.det_score) for f in faces]
    tracks = tracker.update(detections)
    for t in tracks:
        t.detect_only = detect_only

    if not detect_only:
        for track in tracks:
            if track.frames_since_label > 0 and track.frames_since_label % rec_interval != 0:
                continue
            best_face: AnalyzedFace | None = None
            best_dist = float("inf")
            tcx = (track.bbox[0] + track.bbox[2]) / 2
            tcy = (track.bbox[1] + track.bbox[3]) / 2
            for face in faces:
                fcx = (face.bbox[0] + face.bbox[2]) / 2
                fcy = (face.bbox[1] + face.bbox[3]) / 2
                dist = (tcx - fcx) ** 2 + (tcy - fcy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_face = face
            if best_face is None:
                continue
            name, sim = gallery.match(best_face.embedding)
            track.label = name
            track.similarity = sim
            track.frames_since_label = 0

    elapsed = time.perf_counter() - t_start
    proc_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
    annotated = draw_annotations(frame, tracks, fps=proc_fps, frame_idx=frame_idx)
    return annotated, first_face_frame, faces_total


def process_frame_sequence(
    frame_paths: list[Path],
    output_path: Path,
    gallery: FaceGallery,
    app: FaceAnalysisApp,
    fps: float = 25.0,
    rec_interval: int = 5,
    max_frames: int | None = None,
    detect_only: bool = False,
) -> dict:
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {output_path}")

    tracker = FaceTracker()
    frame_idx = 0
    first_face_frame: int | None = None
    t_start = time.perf_counter()
    faces_total = 0
    limit = min(len(frame_paths), max_frames) if max_frames else len(frame_paths)

    print(f"Frame sequence: {limit} frames from {frame_paths[0].parent.name}")

    try:
        for frame_path in frame_paths[:limit]:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"Warning: skip unreadable {frame_path.name}", file=sys.stderr)
                continue
            annotated, first_face_frame, faces_total = _process_one_frame(
                frame,
                frame_idx,
                gallery=gallery,
                app=app,
                tracker=tracker,
                rec_interval=rec_interval,
                detect_only=detect_only,
                first_face_frame=first_face_frame,
                faces_total=faces_total,
                t_start=t_start,
            )
            writer.write(annotated)
            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.perf_counter() - t_start
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                print(f"  frame {frame_idx}/{limit} | proc_fps={proc_fps:.1f}")
    finally:
        writer.release()

    elapsed = time.perf_counter() - t_start
    if first_face_frame is None:
        print(
            "Warning: no faces detected in entire sequence. "
            "For P1E_S2_C1, faces typically start around frame 360.",
            file=sys.stderr,
        )
    elif first_face_frame > 0:
        print(f"First face detected at frame {first_face_frame}")
    return {
        "frames": frame_idx,
        "seconds": round(elapsed, 2),
        "proc_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0,
        "mean_faces_per_frame": round(faces_total / max(frame_idx, 1), 2),
        "output": str(output_path),
    }


def process_video(
    input_path: str | int,
    output_path: Path,
    gallery: FaceGallery,
    app: FaceAnalysisApp,
    rec_interval: int = 5,
    max_frames: int | None = None,
    detect_only: bool = False,
) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        in_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {output_path}")

    tracker = FaceTracker()
    frame_idx = 0
    first_face_frame: int | None = None
    t_start = time.perf_counter()
    faces_total = 0

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            annotated, first_face_frame, faces_total = _process_one_frame(
                frame,
                frame_idx,
                gallery=gallery,
                app=app,
                tracker=tracker,
                rec_interval=rec_interval,
                detect_only=detect_only,
                first_face_frame=first_face_frame,
                faces_total=faces_total,
                t_start=t_start,
            )
            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 30 == 0:
                elapsed = time.perf_counter() - t_start
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                print(f"  frame {frame_idx} | proc_fps={proc_fps:.1f}")
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t_start
    if first_face_frame is None:
        print(
            "Warning: no faces detected in entire video. "
            "Check clip content, or for P1E_S2_C1 faces start ~frame 360.",
            file=sys.stderr,
        )
    elif first_face_frame > 0:
        print(f"First face detected at frame {first_face_frame}")
    return {
        "frames": frame_idx,
        "seconds": round(elapsed, 2),
        "proc_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0,
        "mean_faces_per_frame": round(faces_total / max(frame_idx, 1), 2),
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Facial recognition video test — writes annotated MP4 (FACIAL_REC.md Option B)"
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Frame folder (e.g. data/facial_rec/samples/P1E_S2_C1), video file, or 0 for webcam",
    )
    p.add_argument(
        "--output",
        "-o",
        default="data/facial_rec/output/annotated.mp4",
        help="Output annotated video path",
    )
    p.add_argument(
        "--gallery",
        "-g",
        default="data/facial_rec/gallery",
        help="Gallery folder, or gallery.json from build_gallery_from_dataset.py",
    )
    p.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    p.add_argument("--det-size", type=int, default=640, help="SCRFD input size")
    p.add_argument("--threshold", type=float, default=0.4, help="Cosine match threshold")
    p.add_argument("--rec-interval", type=int, default=5, help="Re-recognize every N frames per track")
    p.add_argument("--fps", type=float, default=25.0, help="Output FPS when --input is a frame folder")
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    p.add_argument("--gpu", type=int, default=0, help="GPU device id (-1 for CPU)")
    p.add_argument(
        "--detect-only",
        action="store_true",
        help="Detection + tracking only (no gallery matching). Use when clip has no GT.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    gallery_path = Path(args.gallery)

    print("Loading InsightFace", args.model, f"det={args.det_size}...")
    app = FaceAnalysisApp(model_pack=args.model, det_size=(args.det_size, args.det_size), ctx_id=args.gpu)

    if args.detect_only:
        gallery = FaceGallery(threshold=args.threshold)
        print("Mode: detect-only (track IDs, no recognition labels)")
    else:
        gallery = load_gallery(gallery_path, app, args.threshold)
    print(f"Gallery: {len(gallery.entries)} identities")

    mode, frame_paths = resolve_input(args.input)
    print(f"Processing {args.input} -> {output_path}")

    if mode == "frames":
        stats = process_frame_sequence(
            frame_paths,
            output_path,
            gallery,
            app,
            fps=args.fps,
            rec_interval=args.rec_interval,
            max_frames=args.max_frames,
            detect_only=args.detect_only,
        )
    else:
        input_src: str | int = int(args.input) if args.input.isdigit() else args.input
        stats = process_video(
            input_src,
            output_path,
            gallery,
            app,
            rec_interval=args.rec_interval,
            max_frames=args.max_frames,
            detect_only=args.detect_only,
        )
    print("Done.")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
