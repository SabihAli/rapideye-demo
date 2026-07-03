#!/usr/bin/env python3
"""
Standalone facial recognition video test — FACIAL_REC.md Option B.

Runs SCRFD + buffalo_l (InsightFace), tracks faces, matches against a gallery,
and writes an annotated output video. No imports from demo-app server packages.

Two raw videos, no labels required:
  # 1. Build a gallery by clustering faces in an unlabelled enrollment video:
  python build_gallery_from_dataset.py --video enroll.mp4 --output data/facial_rec/gallery_built
  # 2. Infer on another raw video with the same people -> annotated output.mp4:
  python facial_rec_video_test.py -i probe.mp4 \
      --gallery data/facial_rec/gallery_built/gallery.json -o output.mp4

Multi-video (batched YOLO person detection + ByteTrack + face recognition):
  # Detect full bodies with a light YOLO model (batched across all videos in one
  # GPU call), track with ByteTrack, attach a gallery identity via face crops.
  # Inputs are downsampled to --target-fps; outputs are written at the same rate.
  python facial_rec_video_test.py --inputs cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
      --gallery data/facial_rec/gallery_built/gallery.json \
      --output-dir out/ --target-fps 20 --yolo-model yolo11n.pt

Performance (see PIPELINE_OPTIMIZATIONS.md) — implemented here:
  - Batched detection: all cameras in one YOLO call; fp16 (--no-half to disable),
    --imgsz to trade recall for speed, optional TensorRT engine (--trt).
  - Threaded decode + encode overlap CPU I/O with GPU compute; --gpu-decode asks
    ffmpeg for NVDEC hardware decode.
  - Recognition is throttled per-track (--rec-interval), gated by person-box height
    (--min-person-box), runs only on tracked person crops, and shares the GPU with
    detection (--face-cpu to force CPU). Gallery matching is one vectorized matmul.
  - --adaptive raises the rec interval under load; --log-gpu prints VRAM; the run
    prints a per-stage timing breakdown (timings_pct) so you can see the bottleneck.
  Not implemented (need real cameras / a bigger build): RTSP/DeepStream ingestion,
  int8 calibration on camera footage, a separate OSNet ReID stage, CUDA-stream overlap.

Usage:
  pip install opencv-python-headless insightface onnxruntime numpy
  python facial_rec_video_test.py --input data/facial_rec/samples/P1E_S2_C1 --output out/annotated.mp4
  python facial_rec_video_test.py --input clip.mp4 --gallery data/facial_rec/gallery_built/gallery.json

Gallery sources:
  - gallery.json from build_gallery_from_dataset.py (XML dataset or --video clustering)
  - a folder of  gallery/<Name>/photo.jpg  images
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
        self._matrix: np.ndarray | None = None  # (N, D) L2-normalized prototypes
        self._names: list[str] = []

    def enroll(self, identity_id: str, display_name: str, embedding: np.ndarray) -> None:
        emb = embedding.astype(np.float32)
        emb /= max(np.linalg.norm(emb), 1e-12)
        self.entries = [e for e in self.entries if e.identity_id != identity_id]
        self.entries.append(GalleryEntry(identity_id, display_name, emb))
        self._rebuild()

    def _rebuild(self) -> None:
        if self.entries:
            self._matrix = np.stack([e.embedding for e in self.entries]).astype(np.float32)
            self._names = [e.display_name for e in self.entries]
        else:
            self._matrix = None
            self._names = []

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        return self.match_batch(embedding[None, :])[0]

    def match_batch(self, embeddings: np.ndarray) -> list[tuple[str, float]]:
        """Cosine match a stack of queries against the gallery via one matmul (opt #4)."""
        q = np.atleast_2d(embeddings).astype(np.float32)
        if self._matrix is None or len(q) == 0:
            return [("Unknown", 0.0)] * len(q)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        sims = q @ self._matrix.T  # (Q, N) — all queries vs all identities at once
        best = sims.argmax(axis=1)
        out: list[tuple[str, float]] = []
        for i, j in enumerate(best):
            s = float(sims[i, j])
            out.append((self._names[j] if s >= self.threshold else "Unknown", s))
        return out


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
    # Accumulated matching confidence per identity name (temporal voting).
    votes: dict[str, float] = field(default_factory=dict)


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
        # Keep briefly-lost tracks internally (bridges short detection dropouts so an
        # ID survives a flicker), but only RETURN tracks detected in this frame.
        # Otherwise a person who leaves keeps a frozen box at their last position
        # (the frame edge) for up to max_missed frames, and a newcomer entering near
        # that spot inherits the stale ID/label.
        return [t for t in self.tracks if t.missed == 0]


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
        # We only use bbox + det_score + ArcFace embedding, so skip the 2D/3D landmark
        # and gender-age models buffalo_l also ships — a large per-face speedup.
        self._app = FaceAnalysis(
            name=model_pack,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
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
# Live display (optional --show) — falls back cleanly on headless OpenCV
# ---------------------------------------------------------------------------

_SHOW_OK: bool | None = None


def show_frame(winname: str, frame: np.ndarray) -> bool:
    """Display a frame; return False if the user pressed 'q' (or display is dead)."""
    global _SHOW_OK
    if _SHOW_OK is False:
        return True
    try:
        cv2.imshow(winname, frame)
        _SHOW_OK = True
        return (cv2.waitKey(1) & 0xFF) != ord("q")
    except cv2.error as e:
        if _SHOW_OK is None:
            print(
                f"Live display unavailable (headless OpenCV / no DISPLAY): {e}\n"
                "Writing output file(s) only. For a window, run on a machine with a display "
                "and `pip install opencv-python` (not -headless).",
                file=sys.stderr,
            )
        _SHOW_OK = False
        return True


def mosaic(frames: list[np.ndarray | None], names: list[str], tile=(480, 360)) -> np.ndarray:
    """Tile up to 4 annotated frames into a single 2x2 view for --show."""
    tw, th = tile
    tiles: list[np.ndarray] = []
    for i in range(4):
        f = frames[i] if i < len(frames) else None
        t = np.zeros((th, tw, 3), np.uint8) if f is None else cv2.resize(f, (tw, th))
        label = names[i] if i < len(names) else ""
        if label:
            cv2.putText(t, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(t)
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])


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
            # Temporal voting: a single frame can mislabel a face, so accumulate
            # matching confidence per identity across the track and show the winner.
            if name != "Unknown":
                track.votes[name] = track.votes.get(name, 0.0) + sim
                track.similarity = sim
            track.label = max(track.votes, key=track.votes.get) if track.votes else "Unknown"
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
    show: bool = False,
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
            if show and not show_frame("facial_rec - press q to quit", annotated):
                print("Live view: quit requested.")
                break
            if frame_idx % 30 == 0:
                elapsed = time.perf_counter() - t_start
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                print(f"  frame {frame_idx}/{limit} | proc_fps={proc_fps:.1f}")
    finally:
        writer.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

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
    show: bool = False,
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

            if show and not show_frame("facial_rec - press q to quit", annotated):
                print("Live view: quit requested.")
                break
            if frame_idx % 30 == 0:
                elapsed = time.perf_counter() - t_start
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                print(f"  frame {frame_idx} | proc_fps={proc_fps:.1f}")
    finally:
        cap.release()
        writer.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

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


# ---------------------------------------------------------------------------
# YOLO person detection + ByteTrack — batched, multi-video pipeline
#
# Detect full bodies with a light YOLO model (batched across all videos in one
# GPU call), track them with ByteTrack (one tracker state per video), then run
# InsightFace recognition on each person crop to attach a gallery identity.
# ---------------------------------------------------------------------------


class PersonDetector:
    """Light YOLO person detector: batched inference, fp16, optional TensorRT (opts #2,#7)."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        gpu: int = 0,
        conf: float = 0.25,
        imgsz: int = 640,
        half: bool = True,
        use_trt: bool = False,
    ):
        from ultralytics import YOLO

        self.device = gpu if gpu >= 0 else "cpu"
        self.conf = conf
        self.imgsz = imgsz
        self.half = half and gpu >= 0  # fp16 only meaningful on GPU
        path = model_path
        if use_trt and gpu >= 0:
            path = self._ensure_engine(model_path, imgsz, self.half)
        self.is_engine = str(path).endswith(".engine")
        self.model = YOLO(path)

    def _ensure_engine(self, model_path: str, imgsz: int, half: bool) -> str:
        """Build once, cache, and reuse a TensorRT engine next to the .pt (opt #2 cache)."""
        from ultralytics import YOLO

        engine = Path(model_path).with_suffix(".engine")
        if engine.exists():
            print(f"Using cached TensorRT engine {engine.name}")
            return str(engine)
        try:
            print(f"Exporting {Path(model_path).name} -> TensorRT fp16 (one-time, minutes)...")
            YOLO(model_path).export(
                format="engine", half=half, imgsz=imgsz, device=self.device, batch=4
            )
            return str(engine) if engine.exists() else model_path
        except Exception as e:  # tensorrt missing / build failure -> fp16 PyTorch
            print(f"TensorRT export failed ({e}); using PyTorch fp16 instead.", file=sys.stderr)
            return model_path

    def detect_batch(self, frames: list[np.ndarray]) -> list:
        """One batched GPU call for all frames; return per-frame ultralytics Boxes (CPU).

        The Boxes object is what BYTETracker.update expects (it reads .conf/.xywh/.cls
        and supports boolean indexing), so we hand it through directly.
        """
        results = self.model.predict(
            frames,
            classes=[0],
            conf=self.conf,
            imgsz=self.imgsz,
            half=self.half,
            device=self.device,
            verbose=False,
        )
        return [r.boxes.cpu() for r in results]


def make_bytetrack(track_buffer: int = 30):
    from types import SimpleNamespace

    from ultralytics.trackers.byte_tracker import BYTETracker

    # Keys mirror ultralytics' default bytetrack.yaml.
    args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=track_buffer,
        match_thresh=0.8,
        fuse_score=True,
    )
    return BYTETracker(args)


def recognize_person(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    app: FaceAnalysisApp,
    gallery: FaceGallery,
) -> tuple[str, float] | None:
    """Detect the largest face inside a person crop and match it to the gallery."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    faces = app.analyze(frame[y1:y2, x1:x2])
    if not faces:
        return None
    face = max(faces, key=lambda f: f.det_score)
    return gallery.match(face.embedding)


def _open_capture(path: str, hw_accel: bool) -> cv2.VideoCapture:
    """Open a video, optionally requesting GPU/NVDEC hardware decode (opt #1)."""
    if hw_accel:
        try:
            cap = cv2.VideoCapture(
                path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
            )
            if cap.isOpened():
                return cap
        except (cv2.error, AttributeError):
            pass  # older OpenCV without the enum, or no HW decoder -> CPU decode
    return cv2.VideoCapture(path)


class _EmitStream:
    """Video reader that downsamples to target_fps via a frame-time accumulator."""

    def __init__(self, path: str, target_fps: float, hw_accel: bool = False):
        self.path = path
        self.cap = _open_capture(path, hw_accel)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open input: {path}")
        self.in_fps = self.cap.get(cv2.CAP_PROP_FPS) or target_fps
        self.out_fps = min(self.in_fps, target_fps)
        self.step = self.out_fps / self.in_fps  # frames kept per source frame (<=1)
        self.accum = 0.0
        self.ended = False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def next_emitted(self) -> np.ndarray | None:
        """Return the next frame to keep after downsampling, or None at EOF."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                self.ended = True
                return None
            self.accum += self.step
            if self.accum >= 1.0:
                self.accum -= 1.0
                return frame

    def release(self) -> None:
        self.cap.release()


class ThreadedReader:
    """Decode+downsample a video on a background thread so it overlaps GPU compute (opt #6).

    Frames land in a bounded queue; the main loop just pops the next one. This is the
    offline stand-in for the doc's producer/consumer ingestion pipeline.
    """

    def __init__(self, path: str, target_fps: float, hw_accel: bool = False, queue_size: int = 8):
        import queue
        import threading

        self.stream = _EmitStream(path, target_fps, hw_accel=hw_accel)
        self.width, self.height, self.out_fps = self.stream.width, self.stream.height, self.stream.out_fps
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop:
            frame = self.stream.next_emitted()
            self._q.put(frame)  # backpressure keeps all frames; None is the EOF sentinel
            if frame is None:
                break

    def read(self) -> np.ndarray | None:
        return self._q.get()

    def release(self) -> None:
        self._stop = True
        try:
            self._q.get_nowait()
        except Exception:
            pass
        self.stream.release()


class ThreadedWriter:
    """Encode annotated frames on a background thread so mp4 muxing overlaps compute (opt #6)."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], queue_size: int = 16):
        import queue
        import threading

        self.path = path
        self._w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._w.isOpened():
            raise RuntimeError(f"Cannot open output writer: {path}")
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                break
            self._w.write(frame)

    def write(self, frame: np.ndarray) -> None:
        self._q.put(frame)

    def close(self) -> None:
        self._q.put(None)
        self._t.join()
        self._w.release()


def _gpu_mem_mb() -> float | None:
    """Reserved GPU memory in MB, or None if torch/CUDA unavailable (opt #8 monitoring)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1e6
    except Exception:
        return None
    return None


def process_videos_batched(
    inputs: list[str],
    out_dir: Path,
    gallery: FaceGallery,
    app: FaceAnalysisApp,
    detector: PersonDetector,
    target_fps: float = 20.0,
    rec_interval: int = 5,
    max_frames: int | None = None,
    detect_only: bool = False,
    track_buffer: int = 30,
    show: bool = False,
    min_person_box: int = 0,
    hw_accel: bool = False,
    adaptive: bool = False,
    log_gpu: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(inputs)
    names = [Path(p).stem for p in inputs]
    last_annot: list[np.ndarray | None] = [None] * n

    # Threaded decode + encode so CPU I/O overlaps GPU inference (opts #1, #6).
    readers = [ThreadedReader(p, target_fps, hw_accel=hw_accel) for p in inputs]
    writers: list[tuple[Path, ThreadedWriter]] = []
    for p, r in zip(inputs, readers):
        out_path = out_dir / f"annotated_{Path(p).stem}.mp4"
        writers.append((out_path, ThreadedWriter(out_path, r.out_fps, (r.width, r.height))))
        print(f"  {Path(p).name}: {r.width}x{r.height} @ {r.out_fps:.0f}fps -> {out_path.name}")

    trackers = [make_bytetrack(track_buffer) for _ in inputs]
    id_states: list[dict[int, dict]] = [dict() for _ in inputs]
    frame_counts = [0] * n
    faces_total = [0] * n
    timings = {"decode": 0.0, "detect": 0.0, "track": 0.0, "recognize": 0.0, "draw_write": 0.0}
    t_start = time.perf_counter()
    total_emitted = 0
    cur_rec = max(1, rec_interval)  # adaptive throttling adjusts this (opt #8)
    peak_gpu_mb = 0.0
    active = set(range(n))

    while active:
        batch_frames: list[np.ndarray] = []
        batch_vi: list[int] = []
        _t = time.perf_counter()
        for vi in sorted(active):
            if max_frames is not None and frame_counts[vi] >= max_frames:
                active.discard(vi)
                continue
            frame = readers[vi].read()  # prefetched on a thread -> usually instant
            if frame is None:
                active.discard(vi)
                continue
            batch_frames.append(frame)
            batch_vi.append(vi)
        timings["decode"] += time.perf_counter() - _t
        if not batch_frames:
            break

        _t = time.perf_counter()
        dets_batch = detector.detect_batch(batch_frames)  # single batched GPU call (opt #3)
        timings["detect"] += time.perf_counter() - _t
        for frame, vi, boxes in zip(batch_frames, batch_vi, dets_batch):
            _t = time.perf_counter()
            tracks_arr = np.asarray(trackers[vi].update(boxes, frame))
            timings["track"] += time.perf_counter() - _t
            state = id_states[vi]
            draw_tracks: list[Track] = []
            for row in tracks_arr:
                x1, y1, x2, y2 = row[:4]
                tid = int(row[4])
                bbox = (int(x1), int(y1), int(x2), int(y2))
                st = state.setdefault(
                    tid, {"votes": {}, "label": "Unknown", "sim": 0.0, "seen": 0}
                )
                # Throttle recognition (opt #5) + skip person boxes too small for a
                # reliable face match (opt #5 min-size gate).
                big_enough = (bbox[3] - bbox[1]) >= min_person_box
                if not detect_only and big_enough and st["seen"] % cur_rec == 0:
                    _t = time.perf_counter()
                    res = recognize_person(frame, bbox, app, gallery)
                    timings["recognize"] += time.perf_counter() - _t
                    if res is not None:
                        name, sim = res
                        faces_total[vi] += 1
                        if name != "Unknown":
                            st["votes"][name] = st["votes"].get(name, 0.0) + sim
                            st["sim"] = sim
                        st["label"] = (
                            max(st["votes"], key=st["votes"].get) if st["votes"] else "Unknown"
                        )
                st["seen"] += 1
                draw_tracks.append(
                    Track(
                        track_id=tid,
                        bbox=bbox,
                        score=float(row[5]),
                        label=st["label"],
                        similarity=st["sim"],
                        detect_only=detect_only,
                    )
                )
            _t = time.perf_counter()
            elapsed = time.perf_counter() - t_start
            agg_fps = total_emitted / elapsed if elapsed > 0 else 0.0
            annotated = draw_annotations(frame, draw_tracks, fps=agg_fps, frame_idx=frame_counts[vi])
            writers[vi][1].write(annotated)
            timings["draw_write"] += time.perf_counter() - _t
            last_annot[vi] = annotated
            frame_counts[vi] += 1

        total_emitted += len(batch_frames)

        # Adaptive throttle (opt #8): if we can't keep up with real-time across all
        # streams, recognize less often before we'd ever have to drop frames.
        if adaptive:
            elapsed = time.perf_counter() - t_start
            proc_fps = total_emitted / max(elapsed, 1e-9)
            target_agg = target_fps * len(active) if active else target_fps
            if proc_fps < 0.9 * target_agg:
                cur_rec = min(cur_rec + 1, 60)
            elif proc_fps > 1.2 * target_agg and cur_rec > rec_interval:
                cur_rec -= 1

        if show and not show_frame("facial_rec (2x2) - press q to quit", mosaic(last_annot, names)):
            print("Live view: quit requested.")
            break
        if total_emitted % (n * 30) < n:
            elapsed = time.perf_counter() - t_start
            mem = _gpu_mem_mb()
            if mem:
                peak_gpu_mb = max(peak_gpu_mb, mem)
            extra = f" | gpu={mem:.0f}MB" if (log_gpu and mem) else ""
            extra += f" | rec_every={cur_rec}" if adaptive else ""
            print(
                f"  emitted {total_emitted} / {len(active)} active | "
                f"agg_fps={total_emitted / max(elapsed, 1e-9):.1f}{extra}"
            )

    for _, w in writers:
        w.close()
    for r in readers:
        r.release()
    if show:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    elapsed = time.perf_counter() - t_start
    return {
        "videos": n,
        "frames_per_video": frame_counts,
        "total_frames": total_emitted,
        "seconds": round(elapsed, 2),
        "aggregate_fps": round(total_emitted / elapsed, 2) if elapsed > 0 else 0.0,
        "per_video_fps": [round(c / elapsed, 2) if elapsed > 0 else 0.0 for c in frame_counts],
        "faces_recognized": faces_total,
        "rec_interval_final": cur_rec,
        "peak_gpu_mb": round(peak_gpu_mb, 1),
        "timings_sec": {k: round(v, 2) for k, v in timings.items()},
        "timings_pct": {k: round(100 * v / max(elapsed, 1e-9)) for k, v in timings.items()},
        "outputs": [str(p) for p, _ in writers],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Facial recognition video test — writes annotated MP4 (FACIAL_REC.md Option B)"
    )
    p.add_argument(
        "--input",
        "-i",
        default=None,
        help="Single frame folder (e.g. data/facial_rec/samples/P1E_S2_C1), video file, or 0 for webcam",
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Multiple videos -> batched YOLO+ByteTrack full-body recognition (e.g. --inputs a.mp4 b.mp4 c.mp4 d.mp4)",
    )
    p.add_argument(
        "--output",
        "-o",
        default="data/facial_rec/output/annotated.mp4",
        help="Output path (single --input mode)",
    )
    p.add_argument(
        "--output-dir",
        default="data/facial_rec/output",
        help="Output directory (multi --inputs mode); writes annotated_<name>.mp4",
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
    p.add_argument("--target-fps", type=float, default=20.0, help="Downsample input(s) to this FPS (multi --inputs mode)")
    p.add_argument("--yolo-model", default="yolo11n.pt", help="Light YOLO model for person detection (multi mode)")
    p.add_argument("--person-conf", type=float, default=0.25, help="YOLO person confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size (512/640); lower is faster")
    p.add_argument("--no-half", dest="half", action="store_false", help="Disable fp16; default is fp16 on GPU (opt #7)")
    p.add_argument("--trt", action="store_true", help="Build/reuse a TensorRT fp16 engine for YOLO (opt #2)")
    p.add_argument("--track-buffer", type=int, default=30, help="ByteTrack frames to keep a lost track")
    p.add_argument("--min-person-box", type=int, default=0, help="Skip face-rec for person boxes shorter than N px (opt #5)")
    p.add_argument("--gpu-decode", action="store_true", help="Request GPU/NVDEC hardware video decode (opt #1)")
    p.add_argument("--face-cpu", action="store_true", help="Force InsightFace onto CPU (default: same GPU as YOLO)")
    p.add_argument("--adaptive", action="store_true", help="Auto-raise rec interval if falling behind real-time (opt #8)")
    p.add_argument("--log-gpu", action="store_true", help="Print reserved GPU memory during the run (opt #8)")
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames processed (per video)")
    p.add_argument("--gpu", type=int, default=0, help="GPU device id (-1 for CPU)")
    p.add_argument(
        "--detect-only",
        action="store_true",
        help="Detection + tracking only (no gallery matching). Use when clip has no GT.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show live annotated output in a window (2x2 mosaic for --inputs). Press q to quit.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.input) == bool(args.inputs):
        print("Provide exactly one of --input (single) or --inputs (multi-video YOLO).", file=sys.stderr)
        return 2
    gallery_path = Path(args.gallery)

    # Detection (YOLO/torch) and recognition (InsightFace/onnxruntime) share one CUDA
    # context/process (opt #6). Use --face-cpu to force recognition onto CPU (e.g. if a
    # torch/onnxruntime cuDNN mismatch prevents both running on the GPU together).
    face_ctx = -1 if args.face_cpu else args.gpu
    print("Loading InsightFace", args.model, f"det={args.det_size} on {'CPU' if face_ctx < 0 else f'GPU{face_ctx}'}...")
    app = FaceAnalysisApp(model_pack=args.model, det_size=(args.det_size, args.det_size), ctx_id=face_ctx)

    if args.detect_only:
        gallery = FaceGallery(threshold=args.threshold)
        print("Mode: detect-only (track IDs, no recognition labels)")
    else:
        gallery = load_gallery(gallery_path, app, args.threshold)
    print(f"Gallery: {len(gallery.entries)} identities")

    # Multi-video: batched YOLO person detection + ByteTrack + face recognition.
    if args.inputs:
        print(f"Loading YOLO {args.yolo_model} (imgsz={args.imgsz}, fp16={args.half and args.gpu >= 0}, trt={args.trt})...")
        detector = PersonDetector(
            model_path=args.yolo_model,
            gpu=args.gpu,
            conf=args.person_conf,
            imgsz=args.imgsz,
            half=args.half,
            use_trt=args.trt,
        )
        print(f"Batched inference over {len(args.inputs)} videos @ {args.target_fps:.0f}fps -> {args.output_dir}")
        stats = process_videos_batched(
            args.inputs,
            Path(args.output_dir),
            gallery,
            app,
            detector,
            target_fps=args.target_fps,
            rec_interval=args.rec_interval,
            max_frames=args.max_frames,
            detect_only=args.detect_only,
            track_buffer=args.track_buffer,
            show=args.show,
            min_person_box=args.min_person_box,
            hw_accel=args.gpu_decode,
            adaptive=args.adaptive,
            log_gpu=args.log_gpu,
        )
        print("Done.")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    output_path = Path(args.output)
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
            show=args.show,
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
            show=args.show,
        )
    print("Done.")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
