"""Unit tests for live face pipeline integration (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from server.config import settings
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
        # A single match isn't enough to commit an identity anymore
        # (identity_min_matches default is 2) — recognition ran (hence
        # identity isn't None, unlike a track it's never been attempted on)
        # but the track reports "Unknown" until corroborated.
        assert result.identity == "Unknown"
        assert result.similarity == 0.0

        track = service._camera_states[2].tracker.active_tracks()[0]
        assert track.rec_attempts == 1
        assert track.votes.get("Alice") == pytest.approx(0.91)

    @patch("server.inference.face_pipeline.FacePipelineService.ensure_loaded", return_value=True)
    def test_process_batch_reattempts_recognition_after_the_cadence_elapses(self, _mock_loaded):
        """Regression test for the rec_attempts-modulo deadlock: driving
        process_batch across enough real frames (using the real, pinned
        facial_rec_interval=30) must eventually produce a second recognition
        attempt on the same still-alive track — not get permanently stuck
        after the first one."""
        service = FacePipelineService()
        service._loaded = True
        service._face_app = MagicMock()
        service._gallery = MagicMock()
        service._recognize_persons_batch = MagicMock(return_value=[("Alice", 0.6)])

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        det = RawDetection((10, 20, 60, 120), "person", 0.95)
        with patch("server.inference.face_pipeline.yolo_runner.run_person_batch", return_value=[[det]]):
            for _ in range(80):
                service.process_batch(
                    [(2, frame)],
                    detect_allowed={2: True},
                    recognition_allowed={2: True},
                )
                track = service._camera_states[2].tracker.active_tracks()[0]
                if track.rec_attempts >= 2:
                    break

        assert track.rec_attempts >= 2
        # Two consistent "Alice" matches should have committed the identity.
        assert track.label == "Alice"

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
        # Fresh track: next_rec_frame defaults to 0, so it's due immediately.
        assert FacePipelineService._should_attempt_recognition(track, 0, 40) is True
        track.next_rec_frame = 30
        assert FacePipelineService._should_attempt_recognition(track, 29, 40) is False
        assert FacePipelineService._should_attempt_recognition(track, 30, 40) is True

    def test_should_recognize_cadence_is_frame_based_not_attempt_count_based(self):
        """Regression test: cadence must be driven by elapsed frames
        (next_rec_frame), not by rec_attempts's own value. Gating on
        `rec_attempts % interval == 0` is self-referential — rec_attempts
        only advances when an attempt happens, so once it reaches 1 it can
        never land on a multiple of any interval > 1 again, permanently
        stopping recognition after the very first attempt on every track."""
        from server.inference.face_pipeline import TrackState

        track = TrackState(track_id=1, bbox=(0, 0, 60, 80), confidence=0.9)
        track.rec_attempts = 1
        track.next_rec_frame = 30
        assert FacePipelineService._should_attempt_recognition(track, 15, 40) is False
        assert FacePipelineService._should_attempt_recognition(track, 30, 40) is True

    def test_should_recognize_keeps_reverifying_already_labeled_tracks(self):
        """A track that already has a name must stay eligible for
        re-recognition on its cadence, not lock forever — otherwise a wrong
        identity (e.g. from a ByteTrack ID swap during an overlap between
        two people) never gets a chance to self-correct once they separate."""
        from server.inference.face_pipeline import TrackState

        track = TrackState(track_id=1, bbox=(0, 0, 60, 80), confidence=0.9)
        track.label = "Ammer"
        track.next_rec_frame = 30
        assert FacePipelineService._should_attempt_recognition(track, 30, 40) is True
        assert FacePipelineService._should_attempt_recognition(track, 10, 40) is False


class TestDedupeIdentities:
    @staticmethod
    def _track(track_id, label="Unknown", votes=None):
        from server.inference.face_pipeline import TrackState

        t = TrackState(track_id=track_id, bbox=(0, 0, 60, 80), confidence=0.9)
        t.label = label
        t.votes = votes or {}
        return t

    @staticmethod
    def _det(track_id, identity):
        return RawDetection(
            bbox=(0, 0, 60, 80),
            class_name="person",
            confidence=0.9,
            track_id=track_id,
            identity=identity,
            similarity=0.7,
        )

    def test_keeps_highest_vote_and_demotes_the_rest(self):
        tracks = {
            1: self._track(1, "Ammer", {"Ammer": 1.2}),
            2: self._track(2, "Ammer", {"Ammer": 3.5}),
            3: self._track(3, "Mohsin", {"Mohsin": 2.0}),
        }
        dets = [self._det(1, "Ammer"), self._det(2, "Ammer"), self._det(3, "Mohsin")]

        FacePipelineService._dedupe_identities(dets, tracks)

        assert dets[0].identity == "Unknown"
        assert dets[0].similarity == 0.0
        assert dets[1].identity == "Ammer"
        assert dets[2].identity == "Mohsin"

    def test_no_conflict_leaves_everyone_alone(self):
        tracks = {1: self._track(1, "Ammer", {"Ammer": 1.0}), 2: self._track(2, "Mohsin", {"Mohsin": 1.0})}
        dets = [self._det(1, "Ammer"), self._det(2, "Mohsin")]

        FacePipelineService._dedupe_identities(dets, tracks)

        assert dets[0].identity == "Ammer"
        assert dets[1].identity == "Mohsin"

    def test_ignores_unknown_and_none_identities(self):
        tracks = {1: self._track(1), 2: self._track(2)}
        dets = [self._det(1, "Unknown"), self._det(2, None)]

        FacePipelineService._dedupe_identities(dets, tracks)

        assert dets[0].identity == "Unknown"
        assert dets[1].identity is None

    @patch("server.inference.face_pipeline.FacePipelineService.ensure_loaded", return_value=True)
    def test_process_batch_deduplicates_overlapping_tracks(self, _mock_loaded):
        """End-to-end through process_batch: two tracks that both match the
        same gallery identity in one recognition batch (e.g. two people
        passing close to each other) must not both display that name.
        identity_min_matches is patched to 1 here — this test is about the
        same-frame conflict resolution, not the accumulation gate (covered
        separately in TestApplyMatch)."""
        service = FacePipelineService()
        service._loaded = True
        service._face_app = MagicMock()
        service._gallery = MagicMock()
        # Track 1 matches "Ammer" more confidently than track 2 does.
        service._recognize_persons_batch = MagicMock(return_value=[("Ammer", 0.9), ("Ammer", 0.4)])

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        det_a = RawDetection((10, 20, 60, 220), "person", 0.95)
        det_b = RawDetection((70, 20, 120, 220), "person", 0.95)
        with patch.object(settings, "identity_min_matches", 1), patch(
            "server.inference.face_pipeline.yolo_runner.run_person_batch",
            return_value=[[det_a, det_b]],
        ):
            out = service.process_batch(
                [(3, frame)],
                detect_allowed={3: True},
                recognition_allowed={3: True},
            )

        identities = [d.identity for d in out[3]]
        assert identities.count("Ammer") == 1
        assert "Unknown" in identities


class TestApplyMatch:
    """Direct coverage of the identity-commitment state machine, independent
    of process_batch's cadence gating (identity_min_matches/
    identity_override_margin use the real settings values throughout)."""

    @staticmethod
    def _track():
        from server.inference.face_pipeline import TrackState

        return TrackState(track_id=1, bbox=(0, 0, 60, 80), confidence=0.9)

    def test_single_match_does_not_commit_an_identity(self):
        track = self._track()
        FacePipelineService._apply_match(track, ("Ammer", 0.8))
        assert track.label == "Unknown"
        assert track.rec_attempts == 1
        assert track.votes["Ammer"] == pytest.approx(0.8)

    def test_second_consistent_match_commits_the_identity(self):
        track = self._track()
        FacePipelineService._apply_match(track, ("Ammer", 0.8))
        FacePipelineService._apply_match(track, ("Ammer", 0.7))
        assert track.label == "Ammer"
        assert track.rec_attempts == 2
        # similarity is the average of the accumulated matches for the name.
        assert track.similarity == pytest.approx((0.8 + 0.7) / 2)

    def test_split_votes_do_not_commit_until_one_name_reaches_the_threshold(self):
        track = self._track()
        FacePipelineService._apply_match(track, ("Ammer", 0.6))
        FacePipelineService._apply_match(track, ("Mohsin", 0.6))
        # Neither name has 2 matches yet — still Unknown despite 2 attempts.
        assert track.label == "Unknown"
        FacePipelineService._apply_match(track, ("Ammer", 0.6))
        assert track.label == "Ammer"

    def test_weak_competing_match_is_ignored_once_identified(self):
        track = self._track()
        FacePipelineService._apply_match(track, ("Ammer", 0.8))
        FacePipelineService._apply_match(track, ("Ammer", 0.8))
        assert track.label == "Ammer"

        # A different identity that isn't substantially stronger must not
        # take over — this is the "stays attached to the track" guarantee.
        FacePipelineService._apply_match(track, ("Mohsin", 0.81))
        assert track.label == "Ammer"

    def test_single_strong_match_does_not_override_even_if_it_clears_the_margin(self):
        """Regression test for the actual bug behind identities flipping
        once and then sticking: a single (even very confident) competing
        match must not be enough to displace an established identity — it
        has to accumulate identity_min_matches of its own first, exactly
        like an initial commit does. Comparing one fresh `sim` straight
        against the bar and swapping immediately was the bug."""
        track = self._track()
        FacePipelineService._apply_match(track, ("Ammer", 0.5))
        FacePipelineService._apply_match(track, ("Ammer", 0.5))
        assert track.label == "Ammer"

        # Clears similarity (0.5) + override margin (0.15) on magnitude
        # alone, but it's only ONE match for "Mohsin" — must not take over.
        FacePipelineService._apply_match(track, ("Mohsin", 0.99))
        assert track.label == "Ammer"
        assert track.votes["Mohsin"] == pytest.approx(0.99)
        assert track.vote_counts["Mohsin"] == 1

    def test_substantially_stronger_match_overrides_and_resets_evidence(self):
        track = self._track()
        FacePipelineService._apply_match(track, ("Ammer", 0.5))
        FacePipelineService._apply_match(track, ("Ammer", 0.5))
        assert track.label == "Ammer"

        # Two corroborating matches, averaging well past similarity (0.5) +
        # override margin (0.15 default) = 0.65+.
        FacePipelineService._apply_match(track, ("Mohsin", 0.9))
        FacePipelineService._apply_match(track, ("Mohsin", 0.9))
        assert track.label == "Mohsin"
        assert track.similarity == pytest.approx(0.9)
        # Evidence reset: the old "Ammer" votes don't linger to make a
        # future swap back easier than it should be.
        assert track.votes == {"Mohsin": pytest.approx(1.8)}
        assert track.vote_counts == {"Mohsin": 2}

    def test_repeated_confirmation_raises_the_override_bar(self):
        track = self._track()
        FacePipelineService._apply_match(track, ("Ammer", 0.5))
        FacePipelineService._apply_match(track, ("Ammer", 0.5))
        FacePipelineService._apply_match(track, ("Ammer", 0.9))
        assert track.label == "Ammer"
        # similarity tracks the running average for the held identity.
        assert track.similarity == pytest.approx((0.5 + 0.5 + 0.9) / 3)

        # Two matches that would have cleared the old 0.5-average-based bar
        # (0.65) but not the new, reinforced one (~0.63 + 0.15 = ~0.78).
        FacePipelineService._apply_match(track, ("Mohsin", 0.7))
        FacePipelineService._apply_match(track, ("Mohsin", 0.7))
        assert track.label == "Ammer"

    def test_unknown_match_only_counts_the_attempt(self):
        track = self._track()
        FacePipelineService._apply_match(track, ("Unknown", 0.1))
        assert track.label == "Unknown"
        assert track.rec_attempts == 1
        assert track.votes == {}

    def test_no_match_only_counts_the_attempt(self):
        track = self._track()
        FacePipelineService._apply_match(track, None)
        assert track.label == "Unknown"
        assert track.rec_attempts == 1


class TestResetJumpedTracks:
    """Covers the ByteTrack-ID-switch mitigation: a track flagged `jumped`
    (its box landed somewhere its own motion doesn't explain — see
    TrackState.jumped / bytetrack_adapter tests) must lose whatever identity
    it was carrying, since it may now be following a different physical
    person than whoever earned that label."""

    @staticmethod
    def _track(label="Unknown", jumped=False, rec_attempts=0):
        from server.inference.face_pipeline import TrackState

        t = TrackState(track_id=1, bbox=(0, 0, 60, 80), confidence=0.9)
        t.label = label
        t.jumped = jumped
        t.rec_attempts = rec_attempts
        if label != "Unknown":
            t.votes = {label: 1.6}
            t.vote_counts = {label: 2}
            t.similarity = 0.8
        return t

    def test_jumped_identified_track_is_reset(self):
        track = self._track(label="Ammer", jumped=True, rec_attempts=2)
        FacePipelineService._reset_jumped_tracks([track])
        assert track.label == "Unknown"
        assert track.similarity == 0.0
        assert track.votes == {}
        assert track.vote_counts == {}

    def test_jumped_track_that_never_had_a_label_is_left_alone(self):
        # No identity to lose — rec_attempts=0 too, so nothing to reset.
        track = self._track(label="Unknown", jumped=True, rec_attempts=0)
        FacePipelineService._reset_jumped_tracks([track])
        assert track.rec_attempts == 0

    def test_non_jumped_identified_track_is_untouched(self):
        track = self._track(label="Ammer", jumped=False, rec_attempts=2)
        FacePipelineService._reset_jumped_tracks([track])
        assert track.label == "Ammer"

    @patch("server.inference.face_pipeline.FacePipelineService.ensure_loaded", return_value=True)
    def test_process_batch_drops_identity_when_tracker_flags_a_jump(self, _mock_loaded):
        """End-to-end through process_batch: a track carrying a committed
        identity loses it the moment the tracker reports a jump, without
        needing a new (necessarily face-less, per the reported bug)
        recognition attempt to trigger it. Reported identity goes to None
        (not "Unknown") because no recognition has been attempted on the
        track since the reset — same as a track that's never run
        recognition at all; see test_reset_jumped_tracks for the
        label/votes-level assertions."""
        service = FacePipelineService()
        service._loaded = True
        service._face_app = MagicMock()
        service._gallery = MagicMock()

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        det = RawDetection((10, 20, 60, 120), "person", 0.95)
        with patch("server.inference.face_pipeline.yolo_runner.run_person_batch", return_value=[[det]]):
            service.process_batch(
                [(2, frame)], detect_allowed={2: True}, recognition_allowed={2: True}
            )
            tracker = service._camera_states[2].tracker
            track = tracker.active_tracks()[0]
            # Simulate an already-committed identity from prior cycles.
            track.label = "Ammer"
            track.votes = {"Ammer": 1.6}
            track.vote_counts = {"Ammer": 2}
            track.similarity = 0.8
            track.rec_attempts = 2

            # Simulate the tracker detecting an ID switch on the next cycle
            # (this is what _to_state would set from real geometry — bypass
            # the real tracker here so the test isn't at the mercy of
            # BYTETracker's own association deciding whether a given
            # synthetic jump still counts as the same track_id, which is
            # covered separately in test_bytetrack_adapter.py).
            track.jumped = True
            tracker.update_with_detections = MagicMock(return_value=([track], []))
            tracker.predict_only = MagicMock(return_value=[track])

            out = service.process_batch(
                [(2, frame)], detect_allowed={2: True}, recognition_allowed={2: True}
            )

        result = next(d for d in out[2] if d.track_id == track.track_id)
        assert result.identity is None
        assert track.label == "Unknown"
        assert track.votes == {}
