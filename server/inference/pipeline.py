from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2

from server.config import settings
from server.ingest.stream_manager import stream_manager
from server.inference.annotator import Annotator
from server.inference.bytetrack_adapter import ByteTrackAdapter
from server.inference.face_pipeline import face_pipeline_service
from server.inference.model_switches import model_switch_manager
from server.inference.motion_detector import motion_gate
from server.inference.yolo_runner import PersonCropRequest, RawDetection, YoloRunner, yolo_runner
from server.inference.zone_engine import zone_engine
from server.recording.camera_recorder import camera_recorder
from server.schemas.alerts import AlertEvent, Detection


@dataclass
class _PendingFrame:
    camera_id: int
    frame: Any
    decoder: Any
    switches: Any


def _new_fire_tracker() -> ByteTrackAdapter:
    return ByteTrackAdapter(
        track_buffer=settings.track_buffer,
        track_high_thresh=settings.track_high_thresh,
        track_low_thresh=settings.track_low_thresh,
        new_track_thresh=settings.track_new_thresh,
        match_thresh=settings.track_match_thresh,
        jump_iou_threshold=settings.track_jump_iou_threshold,
    )


@dataclass
class _FireState:
    tracker: ByteTrackAdapter = field(default_factory=_new_fire_tracker)
    frame_index: int = 0
    next_detect_frame: int = 0


class InferencePipeline:
    def __init__(self) -> None:
        self.alerts_history: List[AlertEvent] = []
        self._alerts_lock = threading.Lock()

        self.active_alerts: Dict[int, Optional[str]] = {i: None for i in range(1, 5)}
        # Consecutive-frame counters backing the alert debounce: a new alert
        # only arms after alert_trigger_frames consecutive in-zone frames,
        # and an active alert only disarms after alert_clear_frames
        # consecutive no-detection frames (see _pipeline_loop). Only ever
        # touched from the single pipeline loop thread, same as active_alerts.
        self._alert_hit_streak: Dict[int, int] = {i: 0 for i in range(1, 5)}
        self._alert_miss_streak: Dict[int, int] = {i: 0 for i in range(1, 5)}
        self.websockets: Dict[int, List[Any]] = {i: [] for i in range(1, 5)}
        self._ws_lock = threading.Lock()

        self._last_processed_ts: Dict[int, float] = {i: 0.0 for i in range(1, 5)}
        self._last_clip_trigger_ts: Dict[int, float] = {i: 0.0 for i in range(1, 5)}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.clip_writer = None
        self._fire_states: Dict[int, _FireState] = {i: _FireState() for i in range(1, 5)}
        # Runs the fire stage concurrently with the person/face-rec stage
        # (see _run_batched_inference), plus per-camera JPEG encode/broadcast
        # off the main loop thread (see _broadcast_frame).
        self._stage_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cv-stage")
        self._broadcast_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cv-broadcast")
        self._cleanup_last_run = 0.0
        self._stats_last_log = 0.0
        self._stats_accum: Dict[str, float] = {
            "motion_ms": 0.0,
            "person_ms": 0.0,
            "fire_ms": 0.0,
            "weapon_ms": 0.0,
            "post_ms": 0.0,
            "batches": 0.0,
            "motion_open": 0.0,
        }

    def start(self, clip_writer_ref=None):
        self.clip_writer = clip_writer_ref
        if self.clip_writer is not None:
            self.clip_writer.on_complete = self._mark_clip_ready
        if self._running:
            return
        self._running = True
        face_pipeline_service.ensure_loaded()
        self._thread = threading.Thread(target=self._pipeline_loop, name="InferencePipeline", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._stage_pool.shutdown(wait=False)
        self._broadcast_pool.shutdown(wait=False)

    def clear_alerts(self):
        print("[Pipeline] Clearing all alert logs and active states...")
        with self._alerts_lock:
            self.alerts_history.clear()
            self.active_alerts = {i: None for i in range(1, 5)}
        self._save_alerts_history()

    def register_websocket(self, camera_id: int, websocket: Any):
        with self._ws_lock:
            if websocket not in self.websockets[camera_id]:
                self.websockets[camera_id].append(websocket)

    def unregister_websocket(self, camera_id: int, websocket: Any):
        with self._ws_lock:
            if websocket in self.websockets[camera_id]:
                self.websockets[camera_id].remove(websocket)

    def get_alerts(self, limit: int = 50, offset: int = 0) -> List[AlertEvent]:
        with self._alerts_lock:
            sorted_alerts = sorted(self.alerts_history, key=lambda x: x.timestamp, reverse=True)
            return sorted_alerts[offset : offset + limit]

    def add_alert(self, alert: AlertEvent):
        with self._alerts_lock:
            self.alerts_history.append(alert)

    def _collect_pending_frames(self) -> List[_PendingFrame]:
        pending: List[_PendingFrame] = []
        for cam_id in range(1, 5):
            decoder = stream_manager.get_decoder(cam_id)
            if decoder is None or not decoder.is_active:
                continue

            latest_ts = decoder._last_push_time
            if latest_ts == 0.0 or latest_ts <= self._last_processed_ts[cam_id]:
                continue

            self._last_processed_ts[cam_id] = latest_ts
            frame = decoder.get_latest_frame()
            if frame is None:
                continue

            switches = model_switch_manager.get_switches(cam_id)
            pending.append(_PendingFrame(cam_id, frame, decoder, switches))
        return pending

    def _run_fire_stage(
        self,
        pending: List[_PendingFrame],
        motion_open: Dict[int, bool],
    ) -> Tuple[Dict[int, List[RawDetection]], float]:
        t_start = time.perf_counter()
        frames = [item.frame for item in pending]
        fire_mask: List[bool] = []
        for item in pending:
            fire_state = self._fire_states[item.camera_id]
            fire_state.frame_index += 1
            fire_mask.append(
                item.switches.fire_enabled
                and motion_open.get(item.camera_id, True)
                and fire_state.frame_index >= fire_state.next_detect_frame
            )

        detected = yolo_runner.run_fire_batch(frames, fire_mask=fire_mask)
        output: Dict[int, List[RawDetection]] = {}
        for idx, item in enumerate(pending):
            fire_state = self._fire_states[item.camera_id]

            if not item.switches.fire_enabled:
                # Fire disabled for this camera: drop any coasting state
                # outright instead of continuing to call predict_only().
                # predict_only() has no detection cycle to ever mark a track
                # lost/expire it (that's what lets it coast smoothly between
                # real detection frames while enabled) — with fire off there
                # is no re-detection to ever reanchor it, so Kalman
                # predictions would keep extrapolating forever, drifting into
                # an ever-expanding box instead of disappearing.
                if fire_state.tracker.active_tracks():
                    fire_state.tracker.reset()
                output[item.camera_id] = []
                continue

            if fire_mask[idx]:
                active_tracks, _ = fire_state.tracker.update_with_detections(detected[idx], item.frame.shape)
                fire_state.next_detect_frame = fire_state.frame_index + face_pipeline_service.fire_interval_effective
            else:
                active_tracks = fire_state.tracker.predict_only(item.frame.shape)

            output[item.camera_id] = [
                RawDetection(
                    bbox=tuple(map(float, track.bbox)),
                    class_name=track.class_name,
                    confidence=track.confidence,
                    track_id=track.track_id,
                )
                for track in active_tracks
            ]
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return output, elapsed_ms

    def _run_batched_inference(self, pending: List[_PendingFrame]) -> Dict[int, List[RawDetection]]:
        t0 = time.perf_counter()
        motion_open = {item.camera_id: motion_gate.update(item.camera_id, item.frame) for item in pending}
        t_motion = time.perf_counter()

        # Person (+ face rec) and fire only both read `pending`/`motion_open`
        # and write to their own per-camera tracker state, so they have no
        # data dependency on each other. Run fire on a worker thread while
        # person runs on this one: ORT releases the GIL for the actual
        # session.run() call, so the two stages' GPU work (and cv2
        # pre/post-processing) overlaps instead of paying person_ms + fire_ms
        # back-to-back every tick. Weapon still runs after, since it depends
        # on person's output crops.
        fire_future = self._stage_pool.submit(self._run_fire_stage, pending, motion_open)

        face_items = [(item.camera_id, item.frame) for item in pending]
        recognition_allowed = {item.camera_id: item.switches.face_enabled for item in pending}
        person_batch = face_pipeline_service.process_batch(
            face_items,
            detect_allowed=motion_open,
            recognition_allowed=recognition_allowed,
        )
        t_person = time.perf_counter()

        fire_batch, fire_elapsed_ms = fire_future.result()

        weapon_requests: List[PersonCropRequest] = []
        for item in pending:
            if not item.switches.weapon_enabled:
                continue
            for det in person_batch.get(item.camera_id, []):
                weapon_requests.append(
                    PersonCropRequest(
                        camera_id=item.camera_id,
                        frame=item.frame,
                        bbox=tuple(map(int, det.bbox)),
                        track_id=det.track_id,
                    )
                )
        weapon_batch = yolo_runner.run_weapon_batch_on_crops(weapon_requests)
        t_weapon = time.perf_counter()

        latency_ms = (t_weapon - t0) * 1000.0
        per_camera_latency = latency_ms / max(len(pending), 1)
        face_pipeline_service.report_pipeline_pressure(latency_ms, len(pending))

        self._stats_accum["motion_ms"] += (t_motion - t0) * 1000.0
        self._stats_accum["person_ms"] += (t_person - t_motion) * 1000.0
        self._stats_accum["fire_ms"] += fire_elapsed_ms
        self._stats_accum["weapon_ms"] += (t_weapon - t_person) * 1000.0
        self._stats_accum["batches"] += 1.0
        self._stats_accum["motion_open"] += sum(1 for is_open in motion_open.values() if is_open)

        detections_by_camera: Dict[int, List[RawDetection]] = {}
        for item in pending:
            detections_by_camera[item.camera_id] = YoloRunner.merge_detection_lists(
                person_batch.get(item.camera_id, []),
                weapon_batch.get(item.camera_id, []),
                fire_batch.get(item.camera_id, []),
            )
            item.decoder._last_inference_latency_ms = per_camera_latency
        return detections_by_camera

    def _maybe_log_pipeline_stats(self) -> None:
        now = time.time()
        interval = settings.pipeline_stats_interval_sec
        if interval <= 0 or now - self._stats_last_log < interval:
            return

        batches = max(self._stats_accum["batches"], 1.0)
        avg_motion_open = self._stats_accum["motion_open"] / batches
        print(
            "[Pipeline] avg ms/batch "
            f"motion={self._stats_accum['motion_ms'] / batches:.1f} "
            f"person={self._stats_accum['person_ms'] / batches:.1f} "
            f"fire={self._stats_accum['fire_ms'] / batches:.1f} "
            f"weapon={self._stats_accum['weapon_ms'] / batches:.1f} "
            f"post={self._stats_accum['post_ms'] / batches:.1f} "
            f"| rec_every={face_pipeline_service.rec_interval_effective} "
            f"| motion_open_cams={avg_motion_open:.1f}"
        )
        self._stats_last_log = now
        for key in self._stats_accum:
            self._stats_accum[key] = 0.0

    def _pipeline_loop(self):
        print("[Pipeline] Main processing loop started.")
        self._load_alerts_history()

        while self._running:
            pending = self._collect_pending_frames()
            if not pending:
                time.sleep(0.005)
                continue

            detections_by_camera = self._run_batched_inference(pending)

            t_post = time.perf_counter()
            for item in pending:
                cam_id = item.camera_id
                frame = item.frame
                decoder = item.decoder
                detections = detections_by_camera.get(cam_id, [])
                h, w, _ = frame.shape
                latency_ms = getattr(decoder, "_last_inference_latency_ms", 0.0)

                if camera_recorder.is_recording(cam_id):
                    camera_recorder.write_frame(cam_id, frame, fps=decoder.target_fps)

                is_alert_now, triggering = zone_engine.check_detections(cam_id, detections, w, h)

                # Debounce: zone_engine's check is a stateless per-frame test,
                # so a new alert only arms after alert_trigger_frames (M)
                # consecutive in-zone frames, and an already-active alert only
                # disarms after alert_clear_frames (N) consecutive
                # no-detection frames — a gap shorter than that is absorbed
                # into the same alert rather than fragmenting into a new one.
                if is_alert_now:
                    self._alert_hit_streak[cam_id] += 1
                    self._alert_miss_streak[cam_id] = 0
                else:
                    self._alert_miss_streak[cam_id] += 1
                    self._alert_hit_streak[cam_id] = 0

                alert_id = self.active_alerts[cam_id]
                if alert_id is None:
                    if self._alert_hit_streak[cam_id] >= settings.alert_trigger_frames:
                        alert_id = str(uuid.uuid4())
                        self.active_alerts[cam_id] = alert_id
                        alert_detections = [Detection(**det.to_normalized(w, h)) for det in triggering]
                        now_ts = time.time()
                        # Clip compilation is throttled independently of the
                        # debounce above: each clip job costs a fixed ~10s in
                        # ClipWriter plus encode time, so a camera re-arming
                        # faster than that would pile up unbounded raw-frame
                        # snapshots in ClipWriter's queue (see
                        # alert_clip_cooldown_sec). The alert itself still
                        # records/shows live either way; it just skips the
                        # clip when on cooldown.
                        on_cooldown = (
                            now_ts - self._last_clip_trigger_ts[cam_id]
                        ) < settings.alert_clip_cooldown_sec
                        event = AlertEvent(
                            id=alert_id,
                            camera_id=cam_id,
                            alert_type=self._determine_alert_type(triggering),
                            timestamp=now_ts,
                            detections=alert_detections,
                            clip_path=None if on_cooldown else f"/api/recordings/{alert_id}",
                        )
                        self.add_alert(event)
                        self._save_alerts_history()
                        if self.clip_writer and not on_cooldown:
                            self._last_clip_trigger_ts[cam_id] = now_ts
                            self.clip_writer.trigger_clip(cam_id, alert_id, decoder.ring_buffer)
                elif self._alert_miss_streak[cam_id] >= settings.alert_clear_frames:
                    self.active_alerts[cam_id] = None

                # Overlay/broadcast reflect the debounced alert lifecycle
                # (stays "active" through brief gaps) rather than the raw,
                # noisy single-frame signal.
                is_alert = self.active_alerts[cam_id] is not None

                zone_config = zone_engine.get_zone(cam_id)
                annotated = Annotator.draw_overlays(
                    frame=frame,
                    detections=detections,
                    zone_config=zone_config,
                    is_alert=is_alert,
                    current_fps=decoder.target_fps,
                    latency_ms=latency_ms,
                    camera_id=cam_id,
                )

                # JPEG-encode + base64 + WS-queue push is pure CPU work with no
                # dependency on the next tick's inference; run it on a worker
                # thread so the main loop can start pulling/inferring the next
                # batch of frames immediately instead of blocking on encode.
                self._broadcast_pool.submit(
                    self._broadcast_frame_safe,
                    cam_id,
                    annotated,
                    detections,
                    is_alert,
                    decoder.target_fps,
                    latency_ms,
                )

            self._stats_accum["post_ms"] += (time.perf_counter() - t_post) * 1000.0
            self._maybe_log_pipeline_stats()
            self._maybe_cleanup_old_recordings()

        print("[Pipeline] Process loop stopped.")

    def _determine_alert_type(self, triggering: List[RawDetection]) -> str:
        class_names = [d.class_name.lower() for d in triggering]
        if any(w in class_names for w in ["fire", "smoke"]):
            return "fire.detected"
        if any(w in class_names for w in ["gun", "weapon", "knife", "pistol", "handgun", "rifle"]):
            return "weapon.detected"
        return "zone.intrusion"

    def _broadcast_frame_safe(self, *args, **kwargs) -> None:
        """Wraps _broadcast_frame for submission to the broadcast thread pool:
        exceptions raised inside a submitted task are stored on its Future and
        silently dropped unless something calls .result() on it, which nothing
        here does (fire-and-forget by design)."""
        try:
            self._broadcast_frame(*args, **kwargs)
        except Exception as exc:
            print(f"[Pipeline] Error broadcasting frame for cam {args[0] if args else '?'}: {exc}")

    def _broadcast_frame(
        self,
        camera_id: int,
        frame: Any,
        detections: List[RawDetection],
        is_alert: bool,
        target_fps: float,
        latency_ms: float,
    ):
        with self._ws_lock:
            subs = self.websockets[camera_id]
            if not subs:
                return

        ret, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return

        base64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
        h, w, _ = frame.shape
        payload = {
            "camera_id": camera_id,
            "frame": f"data:image/jpeg;base64,{base64_str}",
            "detections": [d.to_normalized(w, h) for d in detections],
            "is_alert": is_alert,
            "target_fps": round(target_fps, 1),
            "latency_ms": round(latency_ms, 1),
        }

        for ws in list(subs):
            try:
                if hasattr(ws, "push_queue"):
                    ws.push_queue.put_nowait(payload)
            except Exception as exc:
                print(f"[Pipeline] Error queuing WS frame for cam {camera_id}: {exc}")

    def _load_alerts_history(self):
        history_file = settings.recordings_dir / "alerts_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                with self._alerts_lock:
                    self.alerts_history = [AlertEvent(**item) for item in data]
            except Exception as exc:
                print(f"[Pipeline] Error reading alerts history file: {exc}")
                return
            self._reconcile_clip_state()

    def _reconcile_clip_state(self) -> None:
        """One-time pass right after loading persisted history: entries
        written before clip_ready existed all default to False, which would
        otherwise show a permanent "Compiling..." state even for clips that
        already succeeded. Reconcile against what's actually on disk instead
        — no pending compile job survives a restart, so if the file isn't
        there now, it never will be."""
        changed = False
        with self._alerts_lock:
            for alert in self.alerts_history:
                if alert.clip_ready or alert.clip_path is None:
                    continue
                if (settings.recordings_dir / f"{alert.id}.mp4").exists():
                    alert.clip_ready = True
                else:
                    alert.clip_path = None
                changed = True
        if changed:
            self._save_alerts_history()

    def _save_alerts_history(self):
        # Held for the whole read+write, not just the serialize step: this is
        # now called from both the pipeline loop thread (new alerts) and
        # ClipWriter worker threads (_mark_clip_ready), so two concurrent
        # writers racing to open/write the same file could otherwise corrupt
        # or truncate it.
        history_file = settings.recordings_dir / "alerts_history.json"
        with self._alerts_lock:
            try:
                serialized = [item.model_dump() for item in self.alerts_history]
                with open(history_file, "w") as f:
                    json.dump(serialized, f, indent=4)
            except Exception as exc:
                print(f"[Pipeline] Error writing alerts history file: {exc}")

    def _mark_clip_ready(self, alert_id: str, success: bool) -> None:
        """ClipWriter's on_complete callback: flips clip_ready once the file
        actually exists, instead of the alert publishing a clip_path that
        404s until the (10s+) compile finishes. On failure, clip_path is
        cleared too — there will never be a file to serve, so the frontend
        should just show no clip rather than a permanently-pending one."""
        with self._alerts_lock:
            alert = next((a for a in self.alerts_history if a.id == alert_id), None)
            if alert is None:
                return
            alert.clip_ready = success
            if not success:
                alert.clip_path = None
        self._save_alerts_history()

    def _maybe_cleanup_old_recordings(self) -> None:
        now = time.time()
        if now - self._cleanup_last_run < settings.recordings_cleanup_interval_sec:
            return
        self._cleanup_last_run = now

        cutoff = now - settings.recordings_retention_days * 86400.0
        with self._alerts_lock:
            keep = [a for a in self.alerts_history if a.timestamp >= cutoff]
            removed_ids = {a.id for a in self.alerts_history if a.timestamp < cutoff}
            self.alerts_history = keep

        removed_clips = 0
        for alert_id in removed_ids:
            clip_path = settings.recordings_dir / f"{alert_id}.mp4"
            try:
                clip_path.unlink()
                removed_clips += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[Pipeline] Retention: failed to delete {clip_path}: {exc}")

        # Sweep clip files that no longer have a matching alert (e.g. left
        # over from before this cleanup existed) once they're also stale.
        known_ids = {a.id for a in keep}
        orphans = 0
        for clip_path in settings.recordings_dir.glob("*.mp4"):
            if clip_path.stem in known_ids:
                continue
            try:
                if clip_path.stat().st_mtime < cutoff:
                    clip_path.unlink()
                    orphans += 1
            except (FileNotFoundError, OSError) as exc:
                print(f"[Pipeline] Retention: failed to delete orphan {clip_path}: {exc}")

        if removed_ids or orphans:
            print(
                f"[Pipeline] Retention cleanup: removed {len(removed_ids)} alerts "
                f"({removed_clips} clips) + {orphans} orphaned clip files older than "
                f"{settings.recordings_retention_days:.0f}d"
            )
            self._save_alerts_history()


inference_pipeline = InferencePipeline()
