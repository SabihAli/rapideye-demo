"""Unit tests for live face pipeline integration (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from server.inference.face_pipeline import FacePipelineService
from server.inference.yolo_runner import RawDetection, YoloRunner


class TestRawDetectionPayload:
    def test_person_track_includes_identity_fields(self):
        det = RawDetection(
            bbox=(10.0, 20.0, 110.0, 220.0),
            class_name="person",
            confidence=0.91,
            track_id=7,
            identity="Alice",
            similarity=0.88,
        )
        payload = det.to_normalized(640, 480)
        assert payload["class_name"] == "person"
        assert payload["track_id"] == 7
        assert payload["identity"] == "Alice"
        assert payload["is_unknown"] is False
        assert payload["similarity"] == pytest.approx(0.88)

    def test_unknown_identity_sets_flag(self):
        det = RawDetection(
            bbox=(0.0, 0.0, 10.0, 10.0),
            class_name="person",
            confidence=0.5,
            track_id=1,
            identity="Unknown",
            similarity=0.0,
        )
        payload = det.to_normalized(100, 100)
        assert payload["is_unknown"] is True


class TestYoloRunnerMerge:
    def test_merge_detection_lists(self):
        a = [RawDetection((0, 0, 1, 1), "fire", 0.9)]
        b = [RawDetection((1, 1, 2, 2), "person", 0.8, track_id=2, identity="Bob")]
        merged = YoloRunner.merge_detection_lists(a, b)
        assert len(merged) == 2
        assert merged[1].identity == "Bob"


class TestFacePipelineService:
    def test_reset_camera_clears_tracker_state(self):
        service = FacePipelineService()
        state = service._camera_states[1]
        state.tracker.update_with_detections(
            [RawDetection((0, 0, 10, 10), "person", 0.8)],
            (480, 640, 3),
        )
        state.frame_index = 9
        assert state.tracker.active_tracks() != []
        service.reset_camera(1)
        assert service._camera_states[1].frame_index == 0
        assert service._camera_states[1].tracker.active_tracks() == []

    def test_process_batch_returns_empty_when_not_loaded(self):
        service = FacePipelineService()
        service._loaded = True
        service._detector = None
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        out = service.process_batch([(1, frame)])
        assert out == {1: []}

    @patch("server.inference.face_pipeline.FacePipelineService.ensure_loaded", return_value=True)
    def test_process_batch_emits_person_detections(self, _mock_loaded):
        service = FacePipelineService()
        service._loaded = True
        service._face_app = MagicMock()
        service._gallery = MagicMock()
        service._recognize_persons_batch = MagicMock(return_value=[("Alice", 0.91)])

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        det = RawDetection((10, 20, 60, 120), "person", 0.95)
        with patch("server.inference.face_pipeline.yolo_runner.run_person_batch", return_value=[[det]]):
            out = service.process_batch(
                [(2, frame)],
                detect_allowed={2: True},
                recognition_allowed={2: True},
            )

        assert 2 in out
        assert len(out[2]) == 1
        result = out[2][0]
        assert result.class_name == "person"
        assert result.track_id == 1
        assert result.identity == "Alice"
        assert result.similarity == pytest.approx(0.91)

    def test_adaptive_throttle_increases_interval_under_load(self):
        service = FacePipelineService()
        service._rec_interval_effective = 5
        service._person_interval_ceiling_effective = 10
        service._fire_interval_effective = 3
        with patch("server.inference.face_pipeline.settings") as mock_settings:
            mock_settings.adaptive_facial_throttle = True
            mock_settings.facial_rec_interval = 5
            mock_settings.facial_rec_interval_max = 30
            mock_settings.pipeline_lag_threshold_ms = 150.0
            mock_settings.person_interval_max = 10
            mock_settings.person_interval_max_ceiling = 20
            mock_settings.fire_det_interval = 3
            mock_settings.fire_det_interval_max = 10
            service.report_pipeline_pressure(200.0, 2)
        assert service.rec_interval_effective == 6
        assert service._person_interval_ceiling_effective == 11
        assert service.fire_interval_effective == 4

    def test_should_recognize_on_init_and_interval(self):
        from server.inference.face_pipeline import TrackState

        track = TrackState(track_id=1, bbox=(0, 0, 60, 80), confidence=0.9)
        assert FacePipelineService._should_attempt_recognition(track, 5, 40) is True
        track.rec_attempts = 5
        assert FacePipelineService._should_attempt_recognition(track, 5, 40) is True
        track.rec_attempts = 3
        assert FacePipelineService._should_attempt_recognition(track, 5, 40) is False
