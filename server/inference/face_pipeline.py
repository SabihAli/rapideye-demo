"""
Live facial recognition integration for the demo-app inference pipeline.

Runs YOLO11n person detection + ByteTrack + InsightFace gallery matching on
cameras where ``face_enabled`` is true. Identity labels bind to person tracks.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

import numpy as np

from server.config import settings
from server.inference.yolo_runner import RawDetection


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
                    recognize_person,
                )

                self._recognize_person = recognize_person

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
                    f"{len(self._gallery.entries)}"
                )
                return True
            except Exception as exc:
                print(f"[FacePipeline] Failed to load models: {exc}")
                self._loaded = True
                self._detector = None
                return False

    def reset_camera(self, camera_id: int) -> None:
        """Drop tracker state when facial recognition is disabled for a camera."""
        with self._lock:
            self._trackers.pop(camera_id, None)
            self._track_states[camera_id] = {}

    def _get_tracker(self, camera_id: int):
        if camera_id not in self._trackers:
            from server.inference.face_recognition import make_bytetrack

            self._trackers[camera_id] = make_bytetrack(settings.track_buffer)
        return self._trackers[camera_id]

    def process_batch(
        self,
        items: List[Tuple[int, np.ndarray]],
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

        with self._lock:
            boxes_batch = self._detector.detect_batch(frames)
            out: Dict[int, List[RawDetection]] = {cam_id: [] for cam_id in camera_ids}

            for cam_id, frame, boxes in zip(camera_ids, frames, boxes_batch):
                tracker = self._get_tracker(cam_id)
                state = self._track_states[cam_id]
                tracks_arr = np.asarray(tracker.update(boxes, frame))

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
                    if big_enough and st["seen"] % max(1, settings.facial_rec_interval) == 0:
                        res = self._recognize_person(
                            frame, bbox, self._face_app, self._gallery
                        )
                        if res is not None:
                            name, sim = res
                            if name != "Unknown":
                                st["votes"][name] = st["votes"].get(name, 0.0) + sim
                                st["sim"] = sim
                            st["label"] = (
                                max(st["votes"], key=st["votes"].get)
                                if st["votes"]
                                else "Unknown"
                            )
                    st["seen"] += 1

                    out[cam_id].append(
                        RawDetection(
                            bbox=bbox,
                            class_name="person",
                            confidence=score,
                            track_id=track_id,
                            identity=st["label"],
                            similarity=st["sim"],
                        )
                    )

            return out


face_pipeline_service = FacePipelineService()
