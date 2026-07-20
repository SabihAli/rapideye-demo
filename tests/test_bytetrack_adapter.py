"""Unit tests for the ByteTrack ID-switch detection heuristic (TrackState.jumped).

_to_state's jump computation is a deterministic function of (previous bbox,
new bbox) for a given track_id — tested directly against it with a minimal
STrack-like double, rather than trying to coerce the real BYTETracker's
Hungarian assignment into a specific ID-switch scenario (an emergent,
non-deterministic property of the full tracker that's a poor fit for a
unit test).
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

from server.inference.bytetrack_adapter import ByteTrackAdapter, _iou
from server.inference.yolo_runner import RawDetection


class _FakeSTrack:
    def __init__(self, track_id: int, bbox: tuple[float, float, float, float], score: float = 0.9, cls: float = 0.0):
        self.track_id = track_id
        self.xyxy = np.array(bbox, dtype=np.float32)
        self.score = score
        self.cls = cls


class TestIou:
    def test_identical_boxes(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert _iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0

    def test_partial_overlap(self):
        # Two 10x10 boxes overlapping in a 5x10 region: intersection=50,
        # union = 100+100-50=150.
        assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def _adapter(jump_iou_threshold: float = 0.3) -> ByteTrackAdapter:
    adapter = ByteTrackAdapter(jump_iou_threshold=jump_iou_threshold)
    # _to_state clips through clip_bbox against _last_frame_shape, which
    # only update_with_detections()/predict_only() normally set — driving
    # _to_state directly (to isolate the jump computation from BYTETracker's
    # own association) needs a real frame size or every bbox clips to (1,1).
    adapter._last_frame_shape = (480, 640, 3)
    return adapter


class TestJumpDetection:
    def test_first_appearance_is_never_a_jump(self):
        adapter = _adapter()
        state = adapter._to_state(_FakeSTrack(1, (100, 100, 160, 300)))
        assert state.jumped is False

    def test_small_movement_is_not_a_jump(self):
        adapter = _adapter()
        adapter._to_state(_FakeSTrack(1, (100, 100, 160, 300)))
        state = adapter._to_state(_FakeSTrack(1, (106, 100, 166, 300)))
        assert state.jumped is False

    def test_large_position_jump_on_same_track_id_is_flagged(self):
        """The actual failure mode this guards against: BYTETracker's
        Hungarian match reassigns an existing track_id to a detection far
        from where it last was — the signature of an ID switch during an
        overlap/crossing between two people."""
        adapter = _adapter()
        adapter._to_state(_FakeSTrack(1, (100, 100, 160, 300)))
        state = adapter._to_state(_FakeSTrack(1, (400, 120, 460, 320)))
        assert state.jumped is True

    def test_jump_flag_is_per_cycle_not_sticky(self):
        adapter = _adapter()
        adapter._to_state(_FakeSTrack(1, (100, 100, 160, 300)))
        jumped_state = adapter._to_state(_FakeSTrack(1, (400, 120, 460, 320)))
        assert jumped_state.jumped is True
        # Next cycle continues smoothly from the new position — not a jump.
        settled_state = adapter._to_state(_FakeSTrack(1, (404, 120, 464, 320)))
        assert settled_state.jumped is False

    def test_threshold_is_configurable(self):
        # A jump that clears a lenient threshold shouldn't clear a strict one.
        lenient = _adapter(jump_iou_threshold=0.05)
        lenient._to_state(_FakeSTrack(1, (100, 100, 160, 300)))
        # Boxes still overlap a bit (shifted by 50px out of 60px width).
        state = lenient._to_state(_FakeSTrack(1, (150, 100, 210, 300)))
        assert state.jumped is False

        strict = _adapter(jump_iou_threshold=0.6)
        strict._to_state(_FakeSTrack(1, (100, 100, 160, 300)))
        state = strict._to_state(_FakeSTrack(1, (150, 100, 210, 300)))
        assert state.jumped is True


class TestReidTracker:
    """use_reid switches the underlying tracker class between plain
    BYTETracker (pure IOU/motion — fire tracking, and person tracking with
    the flag off) and BOTSORT (IOU + appearance-embedding fusion). Not
    tested here: whether appearance fusion actually reduces ID swaps during
    a real overlap/crossing — like the jump heuristic above, that's an
    emergent property of the full tracker's Hungarian assignment, a poor
    fit for a unit test (see manual end-to-end verification in the plan
    instead). These tests only cover the plumbing: the right class gets
    built, feats actually reach the tracker, and predict_only dispatches
    correctly for both.
    """

    def test_use_reid_false_builds_bytetracker(self):
        adapter = ByteTrackAdapter(use_reid=False)
        assert type(adapter._tracker) is BYTETracker

    def test_use_reid_true_builds_botsort(self):
        adapter = ByteTrackAdapter(use_reid=True)
        assert isinstance(adapter._tracker, BOTSORT)

    def test_feats_reach_the_tracked_strack(self):
        adapter = ByteTrackAdapter(use_reid=True)
        detections = [RawDetection(bbox=(100.0, 100.0, 160.0, 300.0), class_name="person", confidence=0.9)]
        feats = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        # BYTETracker/BOTSORT both require a second consecutive matching
        # cycle before a new track is activated/surfaced (see class
        # docstring) — call twice with the same detection+embedding.
        adapter.update_with_detections(detections, (480, 640, 3), feats=feats)
        active, _ = adapter.update_with_detections(detections, (480, 640, 3), feats=feats)

        assert len(active) == 1
        strack = adapter._tracker.tracked_stracks[0]
        assert strack.curr_feat is not None
        assert np.allclose(strack.curr_feat, [1.0, 0.0, 0.0])

    def test_feats_none_with_use_reid_falls_back_instead_of_crashing(self):
        """Regression test: BOTSORT's model="auto" encoder (utils/reid.py's
        _auto_encoder) has no None-handling at all — it just crashes trying
        to iterate feats=None. That's a reachable case, not hypothetical:
        the caller has nothing to supply whenever the ReID encoder failed to
        load, or every crop this tick was too degenerate to encode (see
        face_pipeline._feats_for_camera, which returns None for exactly that
        case). update_with_detections must substitute a placeholder rather
        than let feats=None reach the tracker when detections exist."""
        adapter = ByteTrackAdapter(use_reid=True)
        detections = [RawDetection(bbox=(100.0, 100.0, 160.0, 300.0), class_name="person", confidence=0.9)]
        active, _ = adapter.update_with_detections(detections, (480, 640, 3), feats=None)
        assert isinstance(active, list)

    @pytest.mark.parametrize("use_reid", [False, True])
    def test_predict_only_dispatches_through_tracker_instance(self, use_reid):
        """Regression test: predict_only used to hardcode STrack.multi_predict,
        which silently runs the wrong Kalman filter (KalmanFilterXYAH) against
        BOTrack state (KalmanFilterXYWH) when use_reid=True. It must dispatch
        through self._tracker.multi_predict, which each tracker class
        overrides correctly — verified here by replacing the instance's
        bound method and asserting predict_only calls it."""
        adapter = ByteTrackAdapter(use_reid=use_reid)
        adapter._tracker.multi_predict = MagicMock()
        adapter.predict_only((480, 640, 3))
        adapter._tracker.multi_predict.assert_called_once_with(adapter._tracker.tracked_stracks)
