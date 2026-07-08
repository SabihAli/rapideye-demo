from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.byte_tracker import STrack

from server.inference.yolo_runner import RawDetection


def clip_bbox(bbox: Tuple[float, float, float, float], frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(int(round(x1)), w - 1)),
        max(0, min(int(round(y1)), h - 1)),
        max(0, min(int(round(x2)), w)),
        max(0, min(int(round(y2)), h)),
    )


@dataclass
class TrackState:
    """Per-track bookkeeping exposed to callers (geometry mirrors the backing STrack;
    identity fields are owned by this wrapper since ByteTrack has no concept of them)."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"
    rec_attempts: int = 0
    label: str = "Unknown"
    similarity: float = 0.0
    votes: Dict[str, float] = field(default_factory=dict)


class _DetResults:
    """Minimal Results-like object satisfying BYTETracker.update()'s interface
    (`.conf` / `.cls` / `.xywh`, boolean-mask indexing, `len()`)."""

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_DetResults":
        return _DetResults(self.xywh[mask], self.conf[mask], self.cls[mask])


class ByteTrackAdapter:
    """Wraps ultralytics' BYTETracker so it can be dropped into the existing
    per-camera tracker slots (person + fire) with the same interface the
    previous hand-rolled tracker exposed: active_tracks(), mean_confidence(),
    predict_only() and update_with_detections().

    Two behaviors that were the source of the reported false-positive /
    stuck-box issues are handled deliberately differently from a naive
    BYTETracker.update()-every-tick usage:

    - New tracks require a second consecutive detection-cycle match before
      they are surfaced (ByteTrack's own confirmation gate), which filters
      out one-frame noise blips that used to become visible tracks instantly.
    - Only currently-Tracked (matched-this-cycle) tracks are surfaced at all.
      A track that a detection cycle fails to re-match disappears from the
      output on the very next cycle rather than continuing to be drawn while
      it coasts through the full track_buffer window. track_buffer is used
      purely as internal re-identification memory now, not display lifetime.
    - Coasting between detection cycles (person/fire run every n-th frame)
      is done via the tracker's own Kalman filter (STrack.multi_predict),
      which predicts in (x, y, aspect, height) space instead of extrapolating
      raw per-corner pixel velocity: aspect ratio is part of the state, so
      boxes don't stretch into "elongated" shapes the way independent-corner
      linear extrapolation did.
    """

    def __init__(
        self,
        track_buffer: int = 30,
        track_high_thresh: float = 0.4,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
    ) -> None:
        self._args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
        )
        self._tracker = BYTETracker(self._args)
        self._class_names: List[str] = []
        self._meta: Dict[int, TrackState] = {}
        self._last_frame_shape: Tuple[int, int, int] = (1, 1, 3)

    def reset(self) -> None:
        # Deliberately not BYTETracker.reset(): that also calls STrack.reset_id(),
        # which resets a class-level (process-wide) counter shared by every
        # camera's tracker instance, and would risk track_id collisions with
        # other still-active cameras' trackers.
        self._tracker.tracked_stracks = []
        self._tracker.lost_stracks = []
        self._tracker.removed_stracks = []
        self._tracker.frame_id = 0
        self._meta.clear()

    def _class_id(self, name: str) -> float:
        if name not in self._class_names:
            self._class_names.append(name)
        return float(self._class_names.index(name))

    def _class_name(self, idx: float) -> str:
        i = int(idx)
        return self._class_names[i] if 0 <= i < len(self._class_names) else "object"

    @staticmethod
    def _to_xywh(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(0.0, x2 - x1), max(0.0, y2 - y1))

    def _to_state(self, strack: STrack) -> TrackState:
        meta = self._meta.get(strack.track_id)
        if meta is None:
            meta = TrackState(track_id=strack.track_id, bbox=(0.0, 0.0, 0.0, 0.0), confidence=0.0)
            self._meta[strack.track_id] = meta
        meta.bbox = clip_bbox(tuple(strack.xyxy.tolist()), self._last_frame_shape)
        meta.confidence = float(strack.score)
        meta.class_name = self._class_name(strack.cls)
        return meta

    def _prune_meta(self) -> None:
        alive = {t.track_id for t in self._tracker.tracked_stracks} | {t.track_id for t in self._tracker.lost_stracks}
        for track_id in list(self._meta.keys()):
            if track_id not in alive:
                self._meta.pop(track_id, None)

    def active_tracks(self) -> List[TrackState]:
        return [self._to_state(t) for t in self._tracker.tracked_stracks if t.is_activated]

    def mean_confidence(self) -> float:
        active = self.active_tracks()
        if not active:
            return 0.0
        return float(sum(t.confidence for t in active) / len(active))

    def predict_only(self, frame_shape: Tuple[int, int, int]) -> List[TrackState]:
        self._last_frame_shape = frame_shape
        STrack.multi_predict(self._tracker.tracked_stracks)
        return self.active_tracks()

    def update_with_detections(
        self,
        detections: List[RawDetection],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[List[TrackState], List[TrackState]]:
        self._last_frame_shape = frame_shape
        before_ids = {t.track_id for t in self._tracker.tracked_stracks if t.is_activated}

        if detections:
            xywh = np.asarray([self._to_xywh(det.bbox) for det in detections], dtype=np.float32)
            conf = np.asarray([det.confidence for det in detections], dtype=np.float32)
            cls = np.asarray([self._class_id(det.class_name) for det in detections], dtype=np.float32)
        else:
            xywh = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)

        self._tracker.update(_DetResults(xywh, conf, cls))
        self._prune_meta()

        active = self.active_tracks()
        new_candidates = [state for state in active if state.track_id not in before_ids]
        return active, new_candidates
