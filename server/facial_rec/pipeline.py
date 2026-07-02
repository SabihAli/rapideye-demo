from __future__ import annotations

import time
from typing import Any

import numpy as np

from server.facial_rec.backend import FaceDetectorBackend, FaceRecognizerBackend, create_backend
from server.facial_rec.config import FacialRecConfig
from server.facial_rec.gallery import FaceGallery
from server.facial_rec.schemas import FaceDetection, FaceTrack, FrameResult, MatchResult
from server.facial_rec.tracker import FaceTracker


class FacialRecTestPipeline:
    """
    Option B test pipeline: detect (SCRFD) -> track -> gated recognize (ArcFace) -> gallery match.

    Recognition runs when a track is new or every `recognition_interval` frames.
    """

    def __init__(
        self,
        config: FacialRecConfig | None = None,
        backend: Any | None = None,
        detector: FaceDetectorBackend | None = None,
        recognizer: FaceRecognizerBackend | None = None,
    ):
        self.config = config or FacialRecConfig()
        self.gallery = FaceGallery(self.config.gallery_path, self.config.match_threshold)
        self.tracker = FaceTracker()
        self._frame_counts: dict[int, int] = {}

        if backend is not None:
            self._backend = backend
            self._detector = backend
            self._recognizer = backend
        elif detector is not None and recognizer is not None:
            self._backend = None
            self._detector = detector
            self._recognizer = recognizer
        else:
            self._backend = create_backend(
                self.config.model_pack,
                self.config.det_size,
                self.config.ctx_id,
                self.config.providers,
            )
            self._detector = self._backend
            self._recognizer = self._backend

    def process_frame(self, image_bgr: np.ndarray, stream_id: int = 0) -> FrameResult:
        frame_index = self._frame_counts.get(stream_id, 0)
        self._frame_counts[stream_id] = frame_index + 1

        t0 = time.perf_counter()
        detections = self._detector.detect(image_bgr)
        detect_ms = (time.perf_counter() - t0) * 1000.0

        tracks = self.tracker.update(stream_id, detections)
        matches: list[MatchResult] = []
        recognize_ms = 0.0

        det_by_bbox = {det.bbox: det for det in detections}
        for track in tracks:
            needs_rec = (
                track.frames_since_recognition == 0
                or track.frames_since_recognition >= self.config.recognition_interval
            )
            if not needs_rec:
                if track.embedding is not None:
                    identity_id, display_name, sim = self.gallery.match(np.array(track.embedding))
                    matches.append(
                        MatchResult(
                            identity_id=identity_id,
                            display_name=display_name,
                            similarity=sim,
                            known=identity_id is not None,
                        )
                    )
                continue

            det = self._nearest_detection(track, det_by_bbox)
            if det is None:
                continue

            t1 = time.perf_counter()
            embedding = self._recognizer.embed(image_bgr, det)
            recognize_ms += (time.perf_counter() - t1) * 1000.0
            self.tracker.set_embedding(stream_id, track.track_id, embedding.tolist())

            identity_id, display_name, sim = self.gallery.match(embedding)
            matches.append(
                MatchResult(
                    identity_id=identity_id,
                    display_name=display_name,
                    similarity=sim,
                    known=identity_id is not None,
                )
            )

        return FrameResult(
            stream_id=stream_id,
            frame_index=frame_index,
            detections=detections,
            tracks=tracks,
            matches=matches,
            detect_ms=detect_ms,
            recognize_ms=recognize_ms,
        )

    def enroll_from_frame(
        self, image_bgr: np.ndarray, identity_id: str, display_name: str
    ) -> float:
        if self._backend is not None and hasattr(self._backend, "detect_and_embed"):
            faces = self._backend.detect_and_embed(image_bgr)
            if not faces:
                raise ValueError("No face detected for enrollment")
            emb = faces[0]["embedding"]
        else:
            dets = self._detector.detect(image_bgr)
            if not dets:
                raise ValueError("No face detected for enrollment")
            emb = self._recognizer.embed(image_bgr, dets[0])
        self.gallery.enroll(identity_id, display_name, emb)
        _, _, sim = self.gallery.match(emb)
        return sim

    @staticmethod
    def _nearest_detection(
        track: FaceTrack, det_by_bbox: dict[tuple[float, float, float, float], FaceDetection]
    ) -> FaceDetection | None:
        if not det_by_bbox:
            return None
        best_det: FaceDetection | None = None
        best_dist = float("inf")
        tcx = (track.bbox[0] + track.bbox[2]) / 2
        tcy = (track.bbox[1] + track.bbox[3]) / 2
        for det in det_by_bbox.values():
            dcx = (det.bbox[0] + det.bbox[2]) / 2
            dcy = (det.bbox[1] + det.bbox[3]) / 2
            dist = (tcx - dcx) ** 2 + (tcy - dcy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_det = det
        return best_det
