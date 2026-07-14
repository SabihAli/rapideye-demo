from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from server.config import settings
from server.inference.bytetrack_adapter import ByteTrackAdapter, TrackState, clip_bbox
from server.inference.yolo_runner import RawDetection, yolo_runner


def _new_tracker() -> ByteTrackAdapter:
    return ByteTrackAdapter(
        track_buffer=settings.track_buffer,
        track_high_thresh=settings.track_high_thresh,
        track_low_thresh=settings.track_low_thresh,
        new_track_thresh=settings.track_new_thresh,
        match_thresh=settings.track_match_thresh,
        jump_iou_threshold=settings.track_jump_iou_threshold,
    )


@dataclass
class CameraState:
    tracker: ByteTrackAdapter = field(default_factory=_new_tracker)
    frame_index: int = 0
    next_detect_frame: int = 0


class FacePipelineService:
    """Per-camera person tracking, cadence control, and identity labeling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._face_app = None
        self._gallery = None
        self._recognize_persons_batch = None
        self._camera_states: Dict[int, CameraState] = {i: CameraState() for i in range(1, 5)}
        self.loaded_model_names: List[str] = []
        self._rec_interval_effective = settings.facial_rec_interval
        self._person_interval_ceiling_effective = settings.person_interval_max
        self._fire_interval_effective = settings.fire_det_interval

    def ensure_loaded(self) -> bool:
        with self._lock:
            if self._loaded:
                return self._face_app is not None
            try:
                from server.inference.face_recognition import (
                    FaceAnalysisApp,
                    FaceGallery,
                    load_gallery_from_json,
                    recognize_persons_batch,
                )

                self._recognize_persons_batch = recognize_persons_batch
                gpu = settings.cuda_device if settings.use_cuda else -1
                self._face_app = FaceAnalysisApp(
                    model_pack=settings.insightface_model,
                    det_size=(settings.insightface_det_size, settings.insightface_det_size),
                    ctx_id=gpu,
                )
                gallery_path = settings.facial_gallery_path
                if gallery_path.is_file():
                    self._gallery = load_gallery_from_json(
                        gallery_path,
                        threshold=settings.facial_match_threshold,
                    )
                else:
                    self._gallery = FaceGallery(threshold=settings.facial_match_threshold)
                self.loaded_model_names = [f"InsightFace ({settings.insightface_model})"]
                self._loaded = True
                return True
            except Exception as exc:
                print(f"[FacePipeline] Failed to load models: {exc}")
                self._loaded = True
                self._face_app = None
                return False

    def reset_camera(self, camera_id: int) -> None:
        with self._lock:
            self._camera_states[camera_id] = CameraState()

    def report_pipeline_pressure(self, latency_ms: float, batch_size: int) -> None:
        """Shared backpressure signal for the whole tick: as measured latency
        climbs, both the facial-rec retry cadence AND the person/fire
        detection cadence ceilings widen (more frames get interpolated/
        coasted between real detector calls instead of running inference
        every tick); as latency recovers they narrow back down. Previously
        only facial-rec responded to load, so person/fire kept requesting
        detector time every `person_interval_max`/`fire_det_interval` frames
        regardless of how backed up the pipeline already was.
        """
        if not settings.adaptive_facial_throttle:
            self._rec_interval_effective = settings.facial_rec_interval
            self._person_interval_ceiling_effective = settings.person_interval_max
            self._fire_interval_effective = settings.fire_det_interval
            return

        threshold = settings.pipeline_lag_threshold_ms
        under_load = latency_ms > threshold or batch_size >= 3
        relieved = latency_ms < threshold * 0.5

        if under_load:
            self._rec_interval_effective = min(settings.facial_rec_interval_max, self._rec_interval_effective + 1)
            self._person_interval_ceiling_effective = min(
                settings.person_interval_max_ceiling, self._person_interval_ceiling_effective + 1
            )
            self._fire_interval_effective = min(settings.fire_det_interval_max, self._fire_interval_effective + 1)
        elif relieved:
            if self._rec_interval_effective > settings.facial_rec_interval:
                self._rec_interval_effective -= 1
            if self._person_interval_ceiling_effective > settings.person_interval_max:
                self._person_interval_ceiling_effective -= 1
            if self._fire_interval_effective > settings.fire_det_interval:
                self._fire_interval_effective -= 1

    @property
    def rec_interval_effective(self) -> int:
        return max(1, self._rec_interval_effective)

    @property
    def fire_interval_effective(self) -> int:
        return max(1, self._fire_interval_effective)

    @staticmethod
    def _should_attempt_recognition(track: TrackState, frame_index: int, min_box: int) -> bool:
        # Deliberately keeps periodically re-checking tracks that already
        # have a label, not just "Unknown" ones (mirrors the reference CLI
        # pipeline's `run` command). ByteTrack is pure IOU/motion — no
        # appearance model — so a close pass or overlap between two people
        # can swap which physical person a track_id follows. These re-checks
        # don't cheaply overwrite an established identity though (see
        # _apply_match) — they're just how a substantially-stronger
        # competing match ever gets the chance to be observed at all.
        #
        # Cadence is driven by next_rec_frame (elapsed camera frames), not
        # rec_attempts — rec_attempts only advances when an attempt actually
        # runs, so gating on rec_attempts's own value is self-referential:
        # once it's 1, `1 % interval` never lands on 0 again for interval > 1
        # and no track would ever get a second attempt.
        x1, y1, x2, y2 = map(int, track.bbox)
        if (y2 - y1) < min_box or (x2 - x1) < min_box:
            return False
        return frame_index >= track.next_rec_frame

    def _dynamic_interval(self, mean_conf: float) -> int:
        low = settings.track_conf_low
        high = settings.track_conf_high
        ceiling = max(settings.person_interval_min, self._person_interval_ceiling_effective)
        if mean_conf <= low:
            return settings.person_interval_min
        if mean_conf >= high:
            return ceiling
        ratio = (mean_conf - low) / max(high - low, 1e-6)
        span = ceiling - settings.person_interval_min
        return settings.person_interval_min + int(round(span * ratio))

    @staticmethod
    def _apply_match(track: TrackState, match: Tuple[str, float] | None) -> None:
        track.rec_attempts += 1
        if match is None:
            return
        name, sim = match
        if name == "Unknown":
            return

        # Every match accumulates evidence for its name, regardless of
        # what's currently assigned (if anything). A name only ever
        # displaces the current label by repeatedly, consistently winning
        # recognition attempts — never by a single observation, no matter
        # how strong that one match's similarity is. A single embedding can
        # be spuriously confident (blur, angle, a moment of cross-talk
        # between two overlapping person crops), and treating one such match
        # as decisive was the actual bug behind identities flipping once and
        # then sticking: the old code compared a single new `sim` against
        # the bar and swapped immediately.
        track.votes[name] = track.votes.get(name, 0.0) + sim
        track.vote_counts[name] = track.vote_counts.get(name, 0) + 1

        if track.label == "Unknown":
            best_name = max(track.votes, key=track.votes.get)
            if track.vote_counts[best_name] >= settings.identity_min_matches:
                track.label = best_name
                track.similarity = track.votes[best_name] / track.vote_counts[best_name]
            return

        if name == track.label:
            # Reinforces the current identity: raise the bar a competing
            # name has to clear to take over.
            track.similarity = track.votes[name] / track.vote_counts[name]
            return

        # A different, already-attached identity: only take over once the
        # challenger has its own accumulated evidence (identity_min_matches
        # independent attempts, same bar as an initial commit gets) AND that
        # accumulated evidence is substantially stronger, on average, than
        # what's currently held — not just a lucky single match.
        if track.vote_counts[name] < settings.identity_min_matches:
            return
        challenger_avg = track.votes[name] / track.vote_counts[name]
        if challenger_avg >= track.similarity + settings.identity_override_margin:
            track.label = name
            track.similarity = challenger_avg
            # Drop the displaced identity's evidence so it doesn't retain a
            # head start if it starts winning matches again later — it has
            # to re-earn the identity_min_matches bar from scratch, same as
            # any other challenger.
            track.votes = {name: track.votes[name]}
            track.vote_counts = {name: track.vote_counts[name]}
        # else: the challenger hasn't cleared the bar yet — ignored for now.
        # The identity stays attached to the track (see class docstring:
        # only track loss or a substantially stronger, corroborated match
        # revisits it), but the challenger's votes are kept so it can keep
        # accumulating toward a future override instead of starting over
        # every attempt.

    def process_batch(
        self,
        items: List[Tuple[int, np.ndarray]],
        *,
        detect_allowed: Optional[Dict[int, bool]] = None,
        recognition_allowed: Optional[Dict[int, bool]] = None,
    ) -> Dict[int, List[RawDetection]]:
        if not items:
            return {}
        if yolo_runner.person_detector is None:
            return {camera_id: [] for camera_id, _ in items}

        face_ready = self.ensure_loaded()
        camera_ids = [camera_id for camera_id, _ in items]
        frames = [frame for _, frame in items]

        detect_mask: List[bool] = []
        for camera_id in camera_ids:
            state = self._camera_states[camera_id]
            state.frame_index += 1
            can_detect = True if detect_allowed is None else detect_allowed.get(camera_id, True)
            detect_mask.append(can_detect and state.frame_index >= state.next_detect_frame)

        detections_batch = yolo_runner.run_person_batch(frames, person_mask=detect_mask)
        out: Dict[int, List[RawDetection]] = {camera_id: [] for camera_id in camera_ids}
        rec_candidates: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        rec_tracks: List[TrackState] = []

        for idx, (camera_id, frame) in enumerate(items):
            state = self._camera_states[camera_id]
            tracker = state.tracker
            rec_allowed = True if recognition_allowed is None else recognition_allowed.get(camera_id, False)

            if detect_mask[idx]:
                active_tracks, _ = tracker.update_with_detections(detections_batch[idx], frame.shape)
                state.next_detect_frame = state.frame_index + self._dynamic_interval(tracker.mean_confidence())
            else:
                active_tracks = tracker.predict_only(frame.shape)

            self._reset_jumped_tracks(active_tracks)

            if not rec_allowed:
                # Face rec just got turned off (or was never on) for this
                # camera: drop any identity a track picked up earlier so it
                # reverts to a plain "Person" box instead of keeping a stale
                # name/Unknown label until the track is lost.
                for track in active_tracks:
                    if track.rec_attempts or track.label != "Unknown":
                        self._reset_identity(track)

            for track in active_tracks:
                bbox = clip_bbox(track.bbox, frame.shape)
                # identity/similarity stay None until recognition has actually
                # been attempted at least once on this track — the annotator
                # uses that to tell "plain person detection" (blue, no rec
                # run) apart from "recognition ran, no match" (red, Unknown).
                identified = track.rec_attempts > 0
                out[camera_id].append(
                    RawDetection(
                        bbox=bbox,
                        class_name="person",
                        confidence=track.confidence,
                        track_id=track.track_id,
                        identity=track.label if identified else None,
                        similarity=track.similarity if identified else None,
                    )
                )
                if (
                    detect_mask[idx]
                    and face_ready
                    and rec_allowed
                    and self._should_attempt_recognition(track, state.frame_index, settings.min_person_box)
                ):
                    track.next_rec_frame = state.frame_index + self.rec_interval_effective
                    rec_candidates.append((frame, bbox))
                    rec_tracks.append(track)

        if rec_candidates and self._recognize_persons_batch is not None:
            matches = self._recognize_persons_batch(
                rec_candidates,
                self._face_app,
                self._gallery,
                min_det_score=settings.face_min_det_score,
                min_face_px=settings.face_min_size_px,
            )
            for track, match in zip(rec_tracks, matches):
                self._apply_match(track, match)

            for camera_id, detections in out.items():
                for det in detections:
                    track = next(
                        (
                            track
                            for track in self._camera_states[camera_id].tracker.active_tracks()
                            if track.track_id == det.track_id
                        ),
                        None,
                    )
                    # Only tracks recognition has actually run on get an
                    # identity — otherwise this would stamp "Unknown" back
                    # onto every other track in the camera regardless of the
                    # rec_attempts gating above.
                    if track is not None and track.rec_attempts > 0:
                        det.identity = track.label
                        det.similarity = track.similarity

        # Never show the same identity on two tracks in the same camera at
        # once. A ByteTrack ID swap during an overlap (or two tracks
        # independently, spuriously matching the same person) can otherwise
        # have both tracks claiming the same name simultaneously. Runs every
        # call, not just ticks where new recognition happened, since a stale
        # duplicate from an earlier mismatch can otherwise sit in `out`
        # unresolved on coasting frames.
        for camera_id, detections in out.items():
            tracks_by_id = {
                t.track_id: t for t in self._camera_states[camera_id].tracker.active_tracks()
            }
            self._dedupe_identities(detections, tracks_by_id)

        return out

    @staticmethod
    def _dedupe_identities(detections: List[RawDetection], tracks_by_id: Dict[int, TrackState]) -> None:
        """Mutates ``detections`` in place: if more than one shares a
        non-Unknown identity, the one with the strongest accumulated vote for
        that label keeps it and the rest revert to Unknown — not
        permanently, since _should_attempt_recognition keeps re-verifying
        every track, so a demoted track can win the name back (or pick up
        its correct one) once its votes catch up."""
        by_label: Dict[str, List[RawDetection]] = {}
        for det in detections:
            if det.identity and det.identity != "Unknown":
                by_label.setdefault(det.identity, []).append(det)
        for label, claimants in by_label.items():
            if len(claimants) <= 1:
                continue
            claimants.sort(
                key=lambda d: tracks_by_id[d.track_id].votes.get(label, 0.0)
                if d.track_id in tracks_by_id
                else 0.0,
                reverse=True,
            )
            for loser in claimants[1:]:
                loser.identity = "Unknown"
                loser.similarity = 0.0

    @classmethod
    def _reset_jumped_tracks(cls, tracks: List[TrackState]) -> None:
        """Drops identity on any track flagged jumped this cycle (see
        TrackState.jumped): its box just landed somewhere its own motion
        doesn't explain — the signature of a ByteTrack ID switch, most
        commonly two people's boxes overlapping and separating with the
        Hungarian match picking the wrong pairing. Better an honest
        "Unknown" than a stale label riding along onto a different physical
        person."""
        for track in tracks:
            if track.jumped and (track.rec_attempts or track.label != "Unknown"):
                cls._reset_identity(track)

    @staticmethod
    def _reset_identity(track: TrackState) -> None:
        track.label = "Unknown"
        track.similarity = 0.0
        track.rec_attempts = 0
        track.votes.clear()
        track.vote_counts.clear()
        track.next_rec_frame = 0


face_pipeline_service = FacePipelineService()
