"""
Live facial recognition integration for the demo-app inference pipeline.

Runs YOLO11n person detection + ByteTrack + InsightFace gallery matching on
cameras where ``face_enabled`` is true. Identity labels bind to person tracks.

Optimizations (see docs/PIPELINE_OPTIMIZATIONS.md):
- Batched YOLO person detection across face-enabled cameras
- Parallel per-camera ByteTrack (CPU-bound, independent streams)
- Throttled + batched face recognition with gallery match_batch
- Adaptive recognition interval when pipeline latency grows
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from server.config import settings
from server.inference.yolo_runner import RawDetection


@dataclass
class _TrackWork:
    camera_id: int
    frame: np.ndarray
    boxes: object
    rec_interval: int


@dataclass
class _RecCandidate:
    camera_id: int
    det_index: int
    frame: np.ndarray
    bbox: tuple[int, int, int, int]
    state: dict


class FacePipelineService:
    """Per-camera person tracking and identity labeling for live streams."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._detector = None
        self._face_app = None
        self._gallery = None
        self._trackers: Dict[int, object] = {}
        self._track_states: Dict[int, Dict[int, dict]] = {
            i: {} for i in range(1, 5)
        }
        self.loaded_model_names: List[str] = []
        self._rec_interval_effective = settings.facial_rec_interval
        self._recognize_persons_batch = None

    def ensure_loaded(self) -> bool:
        """Lazy-load GPU models and gallery. Returns False if unavailable."""
        with self._lock:
            if self._loaded:
                return self._detector is not None

            try:
                from server.inference.face_recognition import (
                    FaceAnalysisApp,
                    PersonDetector,
                    load_gallery_from_json,
                    recognize_persons_batch,
                )

                self._recognize_persons_batch = recognize_persons_batch

                gpu = settings.cuda_device if settings.use_cuda else -1
                model_path = settings.models_dir / settings.model_person
                self._detector = PersonDetector(
                    model_path=model_path,
                    gpu=gpu,
                    conf=settings.person_conf,
                    imgsz=settings.person_imgsz,
                    half=settings.use_cuda,
                    use_trt=settings.person_use_trt,
                )
                self._face_app = FaceAnalysisApp(
                    model_pack=settings.insightface_model,
                    det_size=(settings.insightface_det_size, settings.insightface_det_size),
                    ctx_id=gpu if settings.use_cuda else -1,
                )

                gallery_path = settings.facial_gallery_path
                if gallery_path.is_file():
                    self._gallery = load_gallery_from_json(
                        gallery_path,
                        threshold=settings.facial_match_threshold,
                    )
                else:
                    from server.inference.face_recognition import FaceGallery

                    print(
                        f"[FacePipeline] Gallery not found at {gallery_path}; "
                        "labels will show Unknown until gallery is built."
                    )
                    self._gallery = FaceGallery(threshold=settings.facial_match_threshold)

                self.loaded_model_names = [
                    "Person (YOLO11n)",
                    f"InsightFace ({settings.insightface_model})",
                ]
                if self._detector.is_engine:
                    self.loaded_model_names[0] += " TensorRT"

                self._loaded = True
                print(
                    f"[FacePipeline] Ready — gallery identities: "
                    f"{len(self._gallery.entries)}, rec_every={self._rec_interval_effective}"
                )
                return True
            except Exception as exc:
                print(f"[FacePipeline] Failed to load models: {exc}")
                self._loaded = True
                self._detector = None
                return False

    def report_pipeline_pressure(self, latency_ms: float, batch_size: int) -> None:
        """Adapt face-recognition throttle when inference falls behind."""
        if not settings.adaptive_facial_throttle:
            self._rec_interval_effective = settings.facial_rec_interval
            return

        base = settings.facial_rec_interval
        ceiling = settings.facial_rec_interval_max
        threshold = settings.pipeline_lag_threshold_ms

        if latency_ms > threshold or batch_size >= 3:
            self._rec_interval_effective = min(ceiling, self._rec_interval_effective + 1)
        elif latency_ms < threshold * 0.5 and self._rec_interval_effective > base:
            self._rec_interval_effective = max(base, self._rec_interval_effective - 1)

    @property
    def rec_interval_effective(self) -> int:
        return max(1, self._rec_interval_effective)

    def reset_camera(self, camera_id: int) -> None:
        """Drop tracker state when facial recognition is disabled for a camera."""
        with self._lock:
            self._trackers.pop(camera_id, None)
            self._track_states[camera_id] = {}

    def _get_tracker(self, camera_id: int, target_fps: float | None = None):
        if camera_id not in self._trackers:
            from server.inference.face_recognition import make_bytetrack

            buffer = settings.track_buffer
            if target_fps and target_fps > 0:
                buffer = max(15, int(settings.track_buffer * target_fps / max(settings.base_fps, 1)))
            self._trackers[camera_id] = make_bytetrack(buffer)
        return self._trackers[camera_id]

    @staticmethod
    def _should_recognize(seen: int, rec_interval: int) -> bool:
        """Run on track init and every N frames thereafter."""
        return seen == 0 or seen % rec_interval == 0

    @staticmethod
    def _apply_match(state: dict, match: tuple[str, float] | None) -> None:
        if match is None:
            return
        name, sim = match
        if name != "Unknown":
            state["votes"][name] = state["votes"].get(name, 0.0) + sim
            state["sim"] = sim
        state["label"] = (
            max(state["votes"], key=state["votes"].get) if state["votes"] else "Unknown"
        )

    def _track_camera(
        self,
        work: _TrackWork,
        target_fps: float | None = None,
    ) -> tuple[int, List[RawDetection], List[_RecCandidate]]:
        with self._lock:
            tracker = self._get_tracker(work.camera_id, target_fps=target_fps)

        state = self._track_states[work.camera_id]
        tracks_arr = np.asarray(tracker.update(work.boxes, work.frame))
        dets: List[RawDetection] = []
        candidates: List[_RecCandidate] = []

        for row in tracks_arr:
            x1, y1, x2, y2 = row[:4]
            track_id = int(row[4])
            score = float(row[5])
            bbox = (int(x1), int(y1), int(x2), int(y2))

            st = state.setdefault(
                track_id,
                {"votes": {}, "label": "Unknown", "sim": 0.0, "seen": 0},
            )
            big_enough = (bbox[3] - bbox[1]) >= settings.min_person_box
            if big_enough and self._should_recognize(st["seen"], work.rec_interval):
                candidates.append(
                    _RecCandidate(
                        camera_id=work.camera_id,
                        det_index=len(dets),
                        frame=work.frame,
                        bbox=bbox,
                        state=st,
                    )
                )

            st["seen"] += 1
            dets.append(
                RawDetection(
                    bbox=bbox,
                    class_name="person",
                    confidence=score,
                    track_id=track_id,
                    identity=st["label"],
                    similarity=st["sim"],
                )
            )

        return work.camera_id, dets, candidates

    def process_batch(
        self,
        items: List[Tuple[int, np.ndarray]],
        *,
        target_fps_by_camera: Dict[int, float] | None = None,
    ) -> Dict[int, List[RawDetection]]:
        """
        Run batched person detection + tracking + identity labeling.

        ``items`` is a list of ``(camera_id, bgr_frame)`` for face-enabled cameras.
        """
        if not items:
            return {}

        if not self.ensure_loaded() or self._detector is None or self._face_app is None:
            return {cam_id: [] for cam_id, _ in items}

        camera_ids = [cam_id for cam_id, _ in items]
        frames = [frame for _, frame in items]
        rec_interval = self.rec_interval_effective
        fps_map = target_fps_by_camera or {}

        with self._lock:
            boxes_batch = self._detector.detect_batch(frames)

        work_items = [
            _TrackWork(cam_id, frame, boxes, rec_interval)
            for cam_id, frame, boxes in zip(camera_ids, frames, boxes_batch)
        ]

        out: Dict[int, List[RawDetection]] = {cam_id: [] for cam_id in camera_ids}
        all_candidates: List[_RecCandidate] = []

        if len(work_items) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(work_items))) as pool:
                futures = [
                    pool.submit(self._track_camera, work, fps_map.get(work.camera_id))
                    for work in work_items
                ]
                track_results = [f.result() for f in futures]
        else:
            work = work_items[0]
            track_results = [self._track_camera(work, fps_map.get(work.camera_id))]

        for cam_id, dets, candidates in track_results:
            out[cam_id] = dets
            all_candidates.extend(candidates)

        if all_candidates and self._recognize_persons_batch is not None:
            batch_items = [(c.frame, c.bbox) for c in all_candidates]
            with self._lock:
                matches = self._recognize_persons_batch(
                    batch_items, self._face_app, self._gallery
                )
            for candidate, match in zip(all_candidates, matches):
                self._apply_match(candidate.state, match)
                det = out[candidate.camera_id][candidate.det_index]
                det.identity = candidate.state["label"]
                det.similarity = candidate.state["sim"]

        return out


face_pipeline_service = FacePipelineService()
