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
        self.websockets: Dict[int, List[Any]] = {i: [] for i in range(1, 5)}
        self._ws_lock = threading.Lock()

        self._last_processed_ts: Dict[int, float] = {i: 0.0 for i in range(1, 5)}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.clip_writer = None
        self._fire_states: Dict[int, _FireState] = {i: _FireState() for i in range(1, 5)}
        # Runs the fire stage concurrently with the person/face-rec stage
        # (see _run_batched_inference), plus per-camera JPEG encode/broadcast
        # off the main loop thread (see _broadcast_frame).
        self._stage_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cv-stage")
        self._broadcast_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cv-broadcast")
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
                    camera_recorder.write_frame(cam_id, frame)

                is_alert, triggering = zone_engine.check_detections(cam_id, detections, w, h)

                alert_id = self.active_alerts[cam_id]
                if is_alert:
                    if alert_id is None:
                        alert_id = str(uuid.uuid4())
                        self.active_alerts[cam_id] = alert_id
                        alert_detections = [Detection(**det.to_normalized(w, h)) for det in triggering]
                        event = AlertEvent(
                            id=alert_id,
                            camera_id=cam_id,
                            alert_type=self._determine_alert_type(triggering),
                            timestamp=time.time(),
                            detections=alert_detections,
                            clip_path=f"/api/recordings/{alert_id}",
                        )
                        self.add_alert(event)
                        self._save_alerts_history()
                        if self.clip_writer:
                            self.clip_writer.trigger_clip(cam_id, alert_id, decoder.ring_buffer)
                elif alert_id is not None:
                    self.active_alerts[cam_id] = None

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

    def _save_alerts_history(self):
        history_file = settings.recordings_dir / "alerts_history.json"
        try:
            with self._alerts_lock:
                serialized = [item.model_dump() for item in self.alerts_history]
            with open(history_file, "w") as f:
                json.dump(serialized, f, indent=4)
        except Exception as exc:
            print(f"[Pipeline] Error writing alerts history file: {exc}")


inference_pipeline = InferencePipeline()
