"""Unit tests for the ByteTrack ID-switch detection heuristic (TrackState.jumped).

_to_state's jump computation is a deterministic function of (previous bbox,
new bbox) for a given track_id — tested directly against it with a minimal
STrack-like double, rather than trying to coerce the real BYTETracker's
Hungarian assignment into a specific ID-switch scenario (an emergent,
non-deterministic property of the full tracker that's a poor fit for a
unit test).
"""

from dataclasses import dataclass

import numpy as np
import pytest

from server.inference.bytetrack_adapter import ByteTrackAdapter, _iou


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
