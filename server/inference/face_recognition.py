"""
4-camera facial recognition pipeline — YOLO person detection + ByteTrack + InsightFace.

Library module for batched multi-stream inference and gallery management.
CLI entry points: ``run`` (inference) and ``build-gallery`` (enrollment).

See docs/PIPELINE_OPTIMIZATIONS.md for performance notes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from server.config import settings

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = Path("data/facial_rec")
DEFAULT_SAMPLES_DIR = DEFAULT_DATA_ROOT / "samples"
# Sourced from settings rather than a second hardcoded "gallery_P1E_S2_C1.json" literal
# — this is exactly the filename that previously drifted out of sync with
# server/config.py's own copy (see settings.facial_gallery_filename).
DEFAULT_GALLERY_JSON = settings.facial_gallery_path
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "output"
DEFAULT_YOLO_MODEL = Path("data/models/yolo11n.pt")
MAX_CAMERA_INPUTS = 4

COLOR_KNOWN = (40, 200, 40)
COLOR_UNKNOWN = (60, 60, 255)
COLOR_TRACK = (255, 200, 0)

_SHOW_OK: bool | None = None


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
        self._matrix: np.ndarray | None = None
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
        q = np.atleast_2d(embeddings).astype(np.float32)
        if self._matrix is None or len(q) == 0:
            return [("Unknown", 0.0)] * len(q)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        sims = q @ self._matrix.T
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
    return gallery


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
                continue
            faces = app.analyze(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: f.det_score)
            gallery.enroll(person_dir.name, person_dir.name, face.embedding)
            break
    return gallery


def load_gallery(gallery_arg: Path, app: "FaceAnalysisApp", threshold: float) -> FaceGallery:
    if gallery_arg.is_file() and gallery_arg.suffix.lower() == ".json":
        return load_gallery_from_json(gallery_arg, threshold)
    return load_gallery_from_dir(gallery_arg, app, threshold)


def discover_sample_videos(
    samples_dir: Path | None = None,
    max_inputs: int = MAX_CAMERA_INPUTS,
) -> list[Path]:
    """Discover up to ``max_inputs`` ``*.mp4`` files under ``data/facial_rec/samples/``."""
    root = samples_dir or DEFAULT_SAMPLES_DIR
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.mp4"))[:max_inputs]


# ---------------------------------------------------------------------------
# Tracks & InsightFace
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    score: float
    label: str = "Unknown"
    similarity: float = 0.0
    detect_only: bool = False


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
        from server.inference.ort_models import face_providers

        providers = ["CPUExecutionProvider"] if ctx_id < 0 else face_providers()
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
# Annotation & display
# ---------------------------------------------------------------------------


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
        cv2.putText(out, f"FPS {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TRACK, 2)
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


def show_frame(winname: str, frame: np.ndarray) -> bool:
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
                "Writing output file(s) only.",
                file=sys.stderr,
            )
        _SHOW_OK = False
        return True


def mosaic(frames: list[np.ndarray | None], names: list[str], tile=(480, 360)) -> np.ndarray:
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
# YOLO + ByteTrack multi-camera pipeline
# ---------------------------------------------------------------------------


class PersonDetector:
    """Light YOLO person detector: batched inference, fp16, optional TensorRT."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_YOLO_MODEL,
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
        self.half = half and gpu >= 0
        path = str(model_path)
        if use_trt and gpu >= 0:
            path = self._ensure_engine(path, imgsz, self.half)
        self.is_engine = path.endswith(".engine")
        self.model = YOLO(path)

    def _ensure_engine(self, model_path: str, imgsz: int, half: bool) -> str:
        from ultralytics import YOLO

        engine = Path(model_path).with_suffix(".engine")
        if engine.exists():
            print(f"Using cached TensorRT engine {engine.name}")
            return str(engine)
        try:
            print(f"Exporting {Path(model_path).name} -> TensorRT fp16 (one-time)...")
            YOLO(model_path).export(
                format="engine", half=half, imgsz=imgsz, device=self.device, batch=MAX_CAMERA_INPUTS
            )
            return str(engine) if engine.exists() else model_path
        except Exception as e:
            print(f"TensorRT export failed ({e}); using PyTorch fp16 instead.", file=sys.stderr)
            return model_path

    def detect_batch(self, frames: list[np.ndarray]) -> list:
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

    args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=track_buffer,
        match_thresh=0.8,
        fuse_score=True,
    )
    return BYTETracker(args)


def _face_embedding_from_person_crop(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    app: FaceAnalysisApp,
    min_det_score: float = 0.0,
    min_face_px: int = 0,
) -> tuple[np.ndarray | None, bool]:
    """Quality gate: a recognition attempt that runs on a blurry, tiny, or
    low-confidence face detection produces an unreliable embedding, and one
    bad embedding is enough to poison a track's identity — so a candidate
    face has to clear both a detector-confidence floor and a minimum size
    (of the face itself, not the surrounding person box) before its
    embedding is even extracted. Callers that don't pass thresholds get the
    old permissive behavior (any detected face).

    Returns ``(embedding_or_None, face_seen)``: ``face_seen`` is True
    whenever InsightFace found a face in the crop at all, even if it was
    then rejected by the quality gate — letting callers (see
    face_pipeline.TrackState.consecutive_no_face) tell "no face visible at
    all" apart from "a face was visible but this attempt's capture quality
    was poor," which matters for identity-staleness decay: the former means
    the person likely isn't facing the camera anymore, the latter is normal
    noise that shouldn't cost an established identity anything.
    """
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None, False
    faces = app.analyze(frame[y1:y2, x1:x2])
    if not faces:
        return None, False
    face = max(faces, key=lambda f: f.det_score)
    if face.det_score < min_det_score:
        return None, True
    if min_face_px > 0:
        fx1, fy1, fx2, fy2 = face.bbox
        if (fx2 - fx1) < min_face_px or (fy2 - fy1) < min_face_px:
            return None, True
    return face.embedding, True


def recognize_person(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    app: FaceAnalysisApp,
    gallery: FaceGallery,
    min_det_score: float = 0.0,
    min_face_px: int = 0,
) -> tuple[str, float] | None:
    emb, _face_seen = _face_embedding_from_person_crop(frame, bbox, app, min_det_score, min_face_px)
    if emb is None:
        return None
    return gallery.match(emb)


def recognize_persons_batch(
    items: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    app: FaceAnalysisApp,
    gallery: FaceGallery,
    min_det_score: float = 0.0,
    min_face_px: int = 0,
) -> list[tuple[tuple[str, float] | None, bool]]:
    """Extract embeddings from person crops and match against gallery in one
    batch. Returns one ``(match_or_None, face_seen)`` pair per item, in the
    same order — see _face_embedding_from_person_crop for what face_seen
    distinguishes and why."""
    if not items:
        return []

    results: list[tuple[str, float] | None] = [None] * len(items)
    face_seen: list[bool] = [False] * len(items)
    embeddings: list[np.ndarray] = []
    index_map: list[int] = []

    for i, (frame, bbox) in enumerate(items):
        emb, seen = _face_embedding_from_person_crop(frame, bbox, app, min_det_score, min_face_px)
        face_seen[i] = seen
        if emb is not None:
            embeddings.append(emb)
            index_map.append(i)

    if embeddings:
        matches = gallery.match_batch(np.stack(embeddings, axis=0))
        for j, orig_i in enumerate(index_map):
            results[orig_i] = matches[j]

    return list(zip(results, face_seen))


def _open_capture(path: str, hw_accel: bool) -> cv2.VideoCapture:
    if hw_accel:
        try:
            cap = cv2.VideoCapture(
                path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
            )
            if cap.isOpened():
                return cap
        except (cv2.error, AttributeError):
            pass
    return cv2.VideoCapture(path)


class _EmitStream:
    def __init__(self, path: str, target_fps: float, hw_accel: bool = False):
        self.path = path
        self.cap = _open_capture(path, hw_accel)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open input: {path}")
        self.in_fps = self.cap.get(cv2.CAP_PROP_FPS) or target_fps
        self.out_fps = min(self.in_fps, target_fps)
        self.step = self.out_fps / self.in_fps
        self.accum = 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def next_emitted(self) -> np.ndarray | None:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.accum += self.step
            if self.accum >= 1.0:
                self.accum -= 1.0
                return frame

    def release(self) -> None:
        self.cap.release()


class ThreadedReader:
    def __init__(self, path: str, target_fps: float, hw_accel: bool = False, queue_size: int = 8):
        import queue
        import threading

        self.stream = _EmitStream(path, target_fps, hw_accel=hw_accel)
        self.width, self.height, self.out_fps = self.stream.width, self.stream.height, self.stream.out_fps
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop:
            frame = self.stream.next_emitted()
            self._q.put(frame)
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
    def __init__(self, path: Path, fps: float, size: tuple[int, int], queue_size: int = 16):
        import queue
        import threading

        self.path = path
        self._w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._w.isOpened():
            raise RuntimeError(f"Cannot open output writer: {path}")
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
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
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1e6
    except Exception:
        return None
    return None


def process_videos_batched(
    inputs: list[str | Path],
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
    input_strs = [str(p) for p in inputs]
    n = len(input_strs)
    names = [Path(p).stem for p in input_strs]
    last_annot: list[np.ndarray | None] = [None] * n

    readers = [ThreadedReader(p, target_fps, hw_accel=hw_accel) for p in input_strs]
    writers: list[tuple[Path, ThreadedWriter]] = []
    for p, r in zip(input_strs, readers):
        out_path = out_dir / f"annotated_{Path(p).stem}.mp4"
        writers.append((out_path, ThreadedWriter(out_path, r.out_fps, (r.width, r.height))))
        print(f"  {Path(p).name}: {r.width}x{r.height} @ {r.out_fps:.0f}fps -> {out_path.name}")

    trackers = [make_bytetrack(track_buffer) for _ in input_strs]
    id_states: list[dict[int, dict]] = [dict() for _ in input_strs]
    frame_counts = [0] * n
    faces_total = [0] * n
    timings = {"decode": 0.0, "detect": 0.0, "track": 0.0, "recognize": 0.0, "draw_write": 0.0}
    t_start = time.perf_counter()
    total_emitted = 0
    cur_rec = max(1, rec_interval)
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
            frame = readers[vi].read()
            if frame is None:
                active.discard(vi)
                continue
            batch_frames.append(frame)
            batch_vi.append(vi)
        timings["decode"] += time.perf_counter() - _t
        if not batch_frames:
            break

        _t = time.perf_counter()
        dets_batch = detector.detect_batch(batch_frames)
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
            # One gallery identity per frame: the track with the strongest
            # accumulated vote keeps the name, the rest fall back to Unknown.
            by_label: dict[str, list[Track]] = {}
            for trk in draw_tracks:
                if trk.label != "Unknown":
                    by_label.setdefault(trk.label, []).append(trk)
            for label, claimants in by_label.items():
                if len(claimants) > 1:
                    claimants.sort(
                        key=lambda t: state[t.track_id]["votes"].get(label, 0.0),
                        reverse=True,
                    )
                    for trk in claimants[1:]:
                        trk.label = "Unknown"
                        trk.similarity = 0.0
            _t = time.perf_counter()
            elapsed = time.perf_counter() - t_start
            agg_fps = total_emitted / elapsed if elapsed > 0 else 0.0
            annotated = draw_annotations(frame, draw_tracks, fps=agg_fps, frame_idx=frame_counts[vi])
            writers[vi][1].write(annotated)
            timings["draw_write"] += time.perf_counter() - _t
            last_annot[vi] = annotated
            frame_counts[vi] += 1

        total_emitted += len(batch_frames)

        if adaptive:
            elapsed = time.perf_counter() - t_start
            proc_fps = total_emitted / max(elapsed, 1e-9)
            target_agg = target_fps * len(active) if active else target_fps
            if proc_fps < 0.9 * target_agg:
                cur_rec = min(cur_rec + 1, 60)
            elif proc_fps > 1.2 * target_agg and cur_rec > rec_interval:
                cur_rec -= 1

        if show and not show_frame("face_recognition (2x2) - press q to quit", mosaic(last_annot, names)):
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


# ---------------------------------------------------------------------------
# Gallery building (ChokePoint XML + unlabelled video clustering)
# ---------------------------------------------------------------------------


@dataclass
class EyeAnnotation:
    frame_id: str
    person_id: str
    left_eye: tuple[int, int]
    right_eye: tuple[int, int]


@dataclass
class FaceRecord:
    frame_idx: int
    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: np.ndarray
    crop: np.ndarray


def parse_chokepoint_xml(xml_path: Path) -> list[EyeAnnotation]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations: list[EyeAnnotation] = []
    for frame_el in root.findall("frame"):
        frame_id = frame_el.attrib.get("number", "")
        for person_el in frame_el.findall("person"):
            person_id = person_el.attrib.get("id", "")
            left = person_el.find("leftEye")
            right = person_el.find("rightEye")
            if left is None or right is None:
                continue
            annotations.append(
                EyeAnnotation(
                    frame_id=frame_id,
                    person_id=person_id,
                    left_eye=(int(left.attrib["x"]), int(left.attrib["y"])),
                    right_eye=(int(right.attrib["x"]), int(right.attrib["y"])),
                )
            )
    return annotations


def load_bg_frames(sequence_dir: Path) -> set[str]:
    bg_file = sequence_dir / "bg_img.txt"
    if not bg_file.exists():
        return set()
    lines = bg_file.read_text(encoding="utf-8").strip().splitlines()
    return {line.strip() for line in lines if line.strip()}


def frame_filename(frame_id: str) -> str:
    return f"{frame_id}.jpg" if not frame_id.endswith(".jpg") else frame_id


def eye_bbox(
    left: tuple[int, int],
    right: tuple[int, int],
    img_w: int,
    img_h: int,
    scale: float = 2.8,
) -> tuple[int, int, int, int]:
    lx, ly = left
    rx, ry = right
    cx = (lx + rx) / 2
    cy = (ly + ry) / 2
    eye_dist = max(np.hypot(rx - lx, ry - ly), 1.0)
    half = eye_dist * scale
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half * 1.1))
    x2 = int(min(img_w, cx + half))
    y2 = int(min(img_h, cy + half * 1.2))
    return x1, y1, x2, y2


def sharpness_score(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_score(ann: EyeAnnotation, crop_bgr: np.ndarray) -> float:
    lx, ly = ann.left_eye
    rx, ry = ann.right_eye
    eye_dist = np.hypot(rx - lx, ry - ly)
    frontal = 1.0 / (1.0 + abs(ly - ry))
    sharp = sharpness_score(crop_bgr)
    return float(eye_dist * 0.5 + sharp * 0.003 + frontal * 20.0)


def embed_crop(app: FaceAnalysisApp, crop_bgr: np.ndarray) -> np.ndarray | None:
    faces = app.analyze(crop_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.det_score)
    emb = face.embedding.astype(np.float32)
    return emb / max(np.linalg.norm(emb), 1e-12)


def discover_sequences(data_root: Path) -> list[str]:
    samples = data_root / "samples"
    if not samples.is_dir():
        return []
    sequences = []
    for seq_dir in sorted(samples.iterdir()):
        if not seq_dir.is_dir():
            continue
        xml = data_root / "gallery" / f"{seq_dir.name}.xml"
        if xml.is_file():
            sequences.append(seq_dir.name)
    return sequences


def build_gallery_for_sequence(
    sequence: str,
    data_root: Path,
    out_dir: Path,
    app: FaceAnalysisApp,
    top_k: int,
    min_quality: float,
) -> dict:
    seq_dir = data_root / "samples" / sequence
    xml_path = data_root / "gallery" / f"{sequence}.xml"
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
    if not xml_path.is_file():
        raise FileNotFoundError(f"Annotation XML not found: {xml_path}")

    bg_frames = load_bg_frames(seq_dir)
    annotations = parse_chokepoint_xml(xml_path)
    by_person: dict[str, list[tuple[float, EyeAnnotation, np.ndarray]]] = {}

    for ann in annotations:
        fname = frame_filename(ann.frame_id)
        if fname in bg_frames:
            continue
        img_path = seq_dir / fname
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = eye_bbox(ann.left_eye, ann.right_eye, w, h)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        crop = img[y1:y2, x1:x2]
        score = quality_score(ann, crop)
        if score < min_quality:
            continue
        by_person.setdefault(ann.person_id, []).append((score, ann, crop))

    identities: list[dict] = []
    seq_out = out_dir / sequence
    seq_out.mkdir(parents=True, exist_ok=True)

    for person_id, candidates in sorted(by_person.items()):
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:top_k]
        embeddings: list[np.ndarray] = []
        person_dir = seq_out / person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        for rank, (score, ann, crop) in enumerate(selected):
            out_img = person_dir / f"enroll_{rank:02d}_f{ann.frame_id}.jpg"
            cv2.imwrite(str(out_img), crop)
            emb = embed_crop(app, crop)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            continue
        prototype = np.mean(np.stack(embeddings, axis=0), axis=0)
        prototype /= max(np.linalg.norm(prototype), 1e-12)
        identities.append(
            {
                "identity_id": person_id,
                "display_name": f"Person_{person_id}",
                "sequence": sequence,
                "num_enrollment_crops": len(embeddings),
                "embedding": prototype.tolist(),
            }
        )

    return {
        "sequence": sequence,
        "identities": identities,
        "threshold": 0.4,
        "source": "chokepoint_xml_auto",
    }


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / max(np.linalg.norm(vec), 1e-12)


def collect_faces_from_video(
    app: FaceAnalysisApp,
    video_path: Path,
    sample_every: int,
    max_frames: int | None,
    min_det_score: float,
    min_face_px: int,
) -> list[FaceRecord]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    records: list[FaceRecord] = []
    frame_idx = 0
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue
            h, w = frame.shape[:2]
            for face in app.analyze(frame):
                if face.det_score < min_det_score:
                    continue
                x1, y1, x2, y2 = face.bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < min_face_px or y2 - y1 < min_face_px:
                    continue
                emb = face.embedding.astype(np.float32)
                emb = _normalize(emb)
                records.append(
                    FaceRecord(
                        frame_idx=frame_idx,
                        bbox=(x1, y1, x2, y2),
                        det_score=face.det_score,
                        embedding=emb,
                        crop=frame[y1:y2, x1:x2].copy(),
                    )
                )
            frame_idx += 1
    finally:
        cap.release()

    return records


def cluster_faces(
    records: list[FaceRecord],
    join_threshold: float,
    merge_threshold: float,
) -> list[dict]:
    clusters: list[dict] = []
    for idx, rec in enumerate(records):
        best_j, best_sim = -1, -1.0
        for j, c in enumerate(clusters):
            sim = float(np.dot(rec.embedding, c["centroid"]))
            if sim > best_sim:
                best_sim, best_j = sim, j
        if best_j >= 0 and best_sim >= join_threshold:
            c = clusters[best_j]
            c["members"].append(idx)
            c["sum"] += rec.embedding
            c["centroid"] = _normalize(c["sum"])
        else:
            clusters.append(
                {"members": [idx], "sum": rec.embedding.copy(), "centroid": rec.embedding.copy()}
            )

    merged = True
    while merged:
        merged = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if float(np.dot(clusters[a]["centroid"], clusters[b]["centroid"])) >= merge_threshold:
                    clusters[a]["members"].extend(clusters[b]["members"])
                    clusters[a]["sum"] += clusters[b]["sum"]
                    clusters[a]["centroid"] = _normalize(clusters[a]["sum"])
                    del clusters[b]
                    merged = True
                    break
            if merged:
                break
    return clusters


def build_gallery_from_video(
    app: FaceAnalysisApp,
    video_path: Path,
    out_dir: Path,
    *,
    sample_every: int = 5,
    max_frames: int | None = None,
    min_det_score: float = 0.5,
    min_face_px: int = 40,
    join_threshold: float = 0.5,
    merge_threshold: float = 0.6,
    min_samples: int = 5,
    top_k: int = 5,
    name_prefix: str = "Person",
    match_threshold: float = 0.4,
) -> dict:
    records = collect_faces_from_video(
        app, video_path, sample_every, max_frames, min_det_score, min_face_px
    )
    if not records:
        raise RuntimeError("No faces detected in enrollment video.")

    clusters = cluster_faces(records, join_threshold, merge_threshold)
    kept = [c for c in clusters if len(c["members"]) >= min_samples]
    kept.sort(key=lambda c: len(c["members"]), reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    identities: list[dict] = []
    for rank, c in enumerate(kept, start=1):
        identity_id = f"id{rank:03d}"
        display_name = f"{name_prefix} {rank}"
        prototype = _normalize(c["sum"])

        members = sorted((records[i] for i in c["members"]), key=lambda r: r.det_score, reverse=True)
        person_dir = out_dir / identity_id
        person_dir.mkdir(parents=True, exist_ok=True)
        for k, rec in enumerate(members[:top_k]):
            cv2.imwrite(str(person_dir / f"enroll_{k:02d}_f{rec.frame_idx}.jpg"), rec.crop)

        identities.append(
            {
                "identity_id": identity_id,
                "display_name": display_name,
                "num_enrollment_crops": len(members),
                "embedding": prototype.tolist(),
            }
        )

    return {
        "threshold": match_threshold,
        "source": "unlabelled_video_cluster",
        "enrollment_video": str(video_path),
        "identities": identities,
    }


def write_gallery_json(result: dict, out_dir: Path, filename: str = settings.facial_gallery_filename) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery_json = out_dir / filename
    gallery_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return gallery_json


def run_build_gallery(args: argparse.Namespace) -> int:
    face_ctx = -1 if getattr(args, "face_cpu", False) else args.gpu
    print(f"Loading InsightFace {args.model}...")
    app = FaceAnalysisApp(model_pack=args.model, det_size=(args.det_size, args.det_size), ctx_id=face_ctx)

    if args.video is not None:
        if not args.video.is_file():
            print(f"Video not found: {args.video}", file=sys.stderr)
            return 1
        result = build_gallery_from_video(
            app,
            args.video,
            args.output,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            min_det_score=args.min_det_score,
            min_face_px=args.min_face,
            join_threshold=args.cluster_threshold,
            merge_threshold=args.merge_threshold,
            min_samples=args.min_samples,
            top_k=args.top_k,
            name_prefix=args.name_prefix,
            match_threshold=args.match_threshold,
        )
        gallery_json = write_gallery_json(result, args.output)
        print(f"Gallery ready: {gallery_json} ({len(result['identities'])} identities)")
        return 0

    sequences = [args.sequence] if args.sequence else discover_sequences(args.data_root)
    if args.enroll_sequence:
        sequences = [args.enroll_sequence]
    if not sequences:
        print("No sequences found. Expected samples/<SEQ>/ + gallery/<SEQ>.xml", file=sys.stderr)
        return 1

    all_identities: list[dict] = []
    args.output.mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        print(f"Building gallery from {seq}...")
        result = build_gallery_for_sequence(
            seq, args.data_root, args.output, app, args.top_k, args.min_quality
        )
        write_gallery_json(result, args.output, f"gallery_{seq}.json")
        all_identities.extend(result["identities"])
        print(f"  {len(result['identities'])} identities")

    combined = {
        "threshold": 0.4,
        "source": "chokepoint_xml_auto",
        "identities": all_identities,
    }
    gallery_json = write_gallery_json(combined, args.output)
    print(f"Gallery ready: {gallery_json} ({len(all_identities)} identities)")
    return 0


def resolve_run_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        paths = [Path(p) for p in args.inputs]
        if len(paths) > MAX_CAMERA_INPUTS:
            raise ValueError(f"At most {MAX_CAMERA_INPUTS} inputs supported, got {len(paths)}")
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Input video(s) not found: {missing[0]}")
        return paths

    discovered = discover_sample_videos()
    if not discovered:
        raise FileNotFoundError(
            f"No *.mp4 files found under {DEFAULT_SAMPLES_DIR}. "
            "Place up to 4 sample videos there or pass --inputs explicitly."
        )
    print(f"Auto-discovered {len(discovered)} video(s) under {DEFAULT_SAMPLES_DIR}")
    return discovered


def run_inference(args: argparse.Namespace) -> int:
    inputs = resolve_run_inputs(args)
    gallery_path = Path(args.gallery)

    face_ctx = -1 if args.face_cpu else args.gpu
    print(
        f"Loading InsightFace {args.model} "
        f"det={args.det_size} on {'CPU' if face_ctx < 0 else f'GPU{face_ctx}'}..."
    )
    app = FaceAnalysisApp(model_pack=args.model, det_size=(args.det_size, args.det_size), ctx_id=face_ctx)

    if args.detect_only:
        gallery = FaceGallery(threshold=args.threshold)
        print("Mode: detect-only (track IDs, no recognition labels)")
    else:
        gallery = load_gallery(gallery_path, app, args.threshold)
    print(f"Gallery: {len(gallery.entries)} identities")

    yolo_path = Path(args.yolo_model)
    print(
        f"Loading YOLO {yolo_path} "
        f"(imgsz={args.imgsz}, fp16={args.half and args.gpu >= 0}, trt={args.trt})..."
    )
    detector = PersonDetector(
        model_path=yolo_path,
        gpu=args.gpu,
        conf=args.person_conf,
        imgsz=args.imgsz,
        half=args.half,
        use_trt=args.trt,
    )
    print(f"Batched inference over {len(inputs)} video(s) @ {args.target_fps:.0f}fps -> {args.output_dir}")
    stats = process_videos_batched(
        inputs,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_run_parser(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help=f"Up to {MAX_CAMERA_INPUTS} input videos (default: auto-discover *.mp4 under data/facial_rec/samples/)",
    )
    sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    sub.add_argument("--gallery", "-g", type=Path, default=DEFAULT_GALLERY_JSON)
    sub.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    sub.add_argument("--det-size", type=int, default=640)
    sub.add_argument("--threshold", type=float, default=0.4)
    sub.add_argument("--rec-interval", type=int, default=5)
    sub.add_argument("--target-fps", type=float, default=20.0)
    sub.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    sub.add_argument("--person-conf", type=float, default=0.25)
    sub.add_argument("--imgsz", type=int, default=640)
    sub.add_argument("--no-half", dest="half", action="store_false")
    sub.add_argument("--trt", action="store_true")
    sub.add_argument("--track-buffer", type=int, default=30)
    sub.add_argument("--min-person-box", type=int, default=0)
    sub.add_argument("--gpu-decode", action="store_true")
    sub.add_argument("--face-cpu", action="store_true")
    sub.add_argument("--adaptive", action="store_true")
    sub.add_argument("--log-gpu", action="store_true")
    sub.add_argument("--max-frames", type=int, default=None)
    sub.add_argument("--gpu", type=int, default=0)
    sub.add_argument("--detect-only", action="store_true")
    sub.add_argument("--show", action="store_true")


def _add_build_gallery_parser(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    sub.add_argument("--output", type=Path, default=DEFAULT_DATA_ROOT / "gallery_built")
    sub.add_argument("--sequence", default=None)
    sub.add_argument("--top-k", type=int, default=5)
    sub.add_argument("--min-quality", type=float, default=30.0)
    sub.add_argument("--model", default="buffalo_l")
    sub.add_argument("--det-size", type=int, default=640)
    sub.add_argument("--gpu", type=int, default=0)
    sub.add_argument("--face-cpu", action="store_true")
    sub.add_argument("--enroll-sequence", default=None)
    sub.add_argument("--video", type=Path, default=None, help="Raw video to cluster faces from")
    sub.add_argument("--sample-every", type=int, default=5)
    sub.add_argument("--max-frames", type=int, default=None)
    sub.add_argument("--min-det-score", type=float, default=0.5)
    sub.add_argument("--min-face", type=int, default=40)
    sub.add_argument("--cluster-threshold", type=float, default=0.5)
    sub.add_argument("--merge-threshold", type=float, default=0.6)
    sub.add_argument("--min-samples", type=int, default=5)
    sub.add_argument("--name-prefix", default="Person")
    sub.add_argument("--match-threshold", type=float, default=0.4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="4-camera facial recognition — YOLO + ByteTrack + InsightFace",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    run_p = subs.add_parser("run", help="Run batched multi-camera inference")
    _add_run_parser(run_p)

    build_p = subs.add_parser("build-gallery", help="Build face gallery from video or ChokePoint XML")
    _add_build_gallery_parser(build_p)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_inference(args)
    if args.command == "build-gallery":
        return run_build_gallery(args)
    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
