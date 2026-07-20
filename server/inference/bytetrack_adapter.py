from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.byte_tracker import STrack

from server.inference.yolo_runner import RawDetection


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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
    vote_counts: Dict[str, int] = field(default_factory=dict)
    # Camera-local frame_index a recognition attempt is next due — cadence
    # must be driven by elapsed frames, not by rec_attempts (which only
    # advances when an attempt actually happens, so gating attempts on
    # rec_attempts's own value is self-referential and never reaches
    # anything past the first one for interval > 1).
    next_rec_frame: int = 0
    # Set by _to_state when this cycle's bbox has implausibly low IoU
    # against the same track_id's last known bbox — BYTETracker is pure
    # IOU/motion association (no appearance model), so when two people's
    # boxes overlap, the Hungarian match can swap which detection continues
    # which track_id. The swapped-to detection's position is generally far
    # from where this track_id's Kalman filter predicted it, which is what
    # this flags. Consumers (see FacePipelineService) use it to reset
    # identity rather than let a stale label ride along onto a different
    # physical person.
    jumped: bool = False
    # Consecutive recognition attempts in a row where no face was detected
    # in the crop at all (distinct from a face being detected but rejected
    # by the quality gate — see face_recognition._face_embedding_from_person_crop's
    # face_seen return). FacePipelineService uses this to decay an
    # established identity once the face has been durably absent (e.g. the
    # person turned their back), rather than letting it ride indefinitely —
    # a match of None on its own doesn't distinguish "no face visible" from
    # "face visible but blurry/bad angle this attempt," which still
    # shouldn't decay.
    consecutive_no_face: int = 0


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
        track_buffer: int = 60,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        jump_iou_threshold: float = 0.3,
        use_reid: bool = False,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
    ) -> None:
        base_args = dict(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
        )
        if use_reid:
            # model="auto" makes BOTSORT's internal encoder a pure pass-through
            # for externally-supplied embeddings (see update_with_detections's
            # feats= param) instead of loading its own AutoBackend/torch-based
            # ReID model — embeddings are computed by OrtReidEncoder (this
            # project's onnxruntime-gpu + TensorRT stack) and threaded in.
            # gmc_method="none": these are static CCTV cameras, so camera-
            # motion compensation has no clear benefit here, and enabling it
            # would need img passed to .update() (see update_with_detections),
            # which also needs _DetResults.xyxy — not implemented, since GMC
            # isn't used. Revisit both together if a moving/PTZ camera is
            # added.
            self._args = SimpleNamespace(
                **base_args,
                gmc_method="none",
                proximity_thresh=proximity_thresh,
                appearance_thresh=appearance_thresh,
                with_reid=True,
                model="auto",
            )
            self._tracker = BOTSORT(self._args)
        else:
            self._args = SimpleNamespace(**base_args)
            self._tracker = BYTETracker(self._args)
        self._use_reid = use_reid
        self._class_names: List[str] = []
        self._meta: Dict[int, TrackState] = {}
        self._last_frame_shape: Tuple[int, int, int] = (1, 1, 3)
        self._jump_iou_threshold = jump_iou_threshold

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
        is_new = strack.track_id not in self._meta
        meta = self._meta.get(strack.track_id)
        if meta is None:
            meta = TrackState(track_id=strack.track_id, bbox=(0.0, 0.0, 0.0, 0.0), confidence=0.0)
            self._meta[strack.track_id] = meta
        new_bbox = clip_bbox(tuple(strack.xyxy.tolist()), self._last_frame_shape)
        meta.jumped = not is_new and _iou(meta.bbox, new_bbox) < self._jump_iou_threshold
        meta.bbox = new_bbox
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
        # Dispatch through the tracker instance's own multi_predict, not the
        # base STrack static method: BOTSORT's BOTrack overrides this to use
        # KalmanFilterXYWH (its own shared_kalman), a different state
        # parameterization than base STrack's KalmanFilterXYAH. Calling
        # STrack.multi_predict directly on BOTrack instances would silently
        # run the wrong Kalman model on their mean/covariance during every
        # coasting frame — no crash, just quietly wrong box interpolation.
        self._tracker.multi_predict(self._tracker.tracked_stracks)
        return self.active_tracks()

    def update_with_detections(
        self,
        detections: List[RawDetection],
        frame_shape: Tuple[int, int, int],
        feats: Optional[np.ndarray] = None,
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

        if self._use_reid and detections and feats is None:
            # BOTSORT's model="auto" encoder is a pure pass-through for
            # externally-supplied feats (see __init__) — it has no fallback
            # for feats=None, it just crashes trying to iterate it (ultralytics
            # utils/reid.py's _auto_encoder). That's a real, reachable case
            # here: the caller has no embeddings to supply on a given tick
            # whenever the ReID encoder failed to load, or every crop this
            # tick was too degenerate to encode (see face_pipeline's
            # _feats_for_camera). A zero-vector placeholder keeps BOTSORT
            # running: embedding_distance's zero-norm check (see
            # ultralytics utils/matching.py) already treats a missing/zero
            # feature as "ignore appearance, fall back to motion/IoU" rather
            # than crashing or forcing a false match — exactly the degraded
            # behavior we want when embeddings aren't available this tick.
            feats = np.zeros((len(detections), 1), dtype=np.float32)

        self._tracker.update(_DetResults(xywh, conf, cls), feats=feats)
        self._prune_meta()

        active = self.active_tracks()
        new_candidates = [state for state in active if state.track_id not in before_ids]
        return active, new_candidates
