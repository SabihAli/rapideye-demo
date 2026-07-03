import time
import cv2
import json
import base64
import uuid
import threading
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from server.config import settings
from server.ingest.stream_manager import stream_manager
from server.inference.yolo_runner import yolo_runner, RawDetection, YoloRunner
from server.inference.model_switches import model_switch_manager
from server.inference.zone_engine import zone_engine
from server.inference.scheduler import fps_scheduler
from server.inference.annotator import Annotator
from server.inference.face_pipeline import face_pipeline_service
from server.recording.camera_recorder import camera_recorder
from server.schemas.alerts import AlertEvent, Detection


@dataclass
class _PendingFrame:
    camera_id: int
    frame: Any
    decoder: Any
    switches: Any


class InferencePipeline:
    """
    Main processing pipeline that orchestrates GPU inference, zone validation,
    annotation rendering, clip trigger detection, and WebSocket broadcasting.

    Frames from multiple cameras are batched when fire/weapon/person models run
    on more than one feed in the same tick.
    """

    def __init__(self):
        self.alerts_history: List[AlertEvent] = []
        self._alerts_lock = threading.Lock()

        self.active_alerts: Dict[int, Optional[str]] = {i: None for i in range(1, 5)}
        self.websockets: Dict[int, List[Any]] = {i: [] for i in range(1, 5)}
        self._ws_lock = threading.Lock()

        self._last_processed_ts: Dict[int, float] = {i: 0.0 for i in range(1, 5)}
        self._face_enabled_cache: Dict[int, bool] = {i: False for i in range(1, 5)}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.clip_writer = None
        self._stats_last_log = 0.0
        self._stats_accum: Dict[str, float] = {
            "fire_ms": 0.0,
            "weapon_ms": 0.0,
            "face_ms": 0.0,
            "post_ms": 0.0,
            "batches": 0.0,
        }

    def start(self, clip_writer_ref=None):
        """Starts the pipeline thread."""
        self.clip_writer = clip_writer_ref
        if self._running:
            return
        self._running = True
        self._prime_face_pipeline()
        self._thread = threading.Thread(target=self._pipeline_loop, name="InferencePipeline", daemon=True)
        self._thread.start()

    def _prime_face_pipeline(self):
        if any(
            model_switch_manager.get_switches(cam_id).face_enabled
            for cam_id in range(1, 5)
        ):
            face_pipeline_service.ensure_loaded()

    def stop(self):
        """Stops the pipeline thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def clear_alerts(self):
        """Clears all historical and active alerts both in memory and on disk."""
        print("[Pipeline] Clearing all alert logs and active states...")
        with self._alerts_lock:
            self.alerts_history.clear()
            self.active_alerts = {i: None for i in range(1, 5)}
        self._save_alerts_history()

    def register_websocket(self, camera_id: int, websocket: Any):
        with self._ws_lock:
            if websocket not in self.websockets[camera_id]:
                self.websockets[camera_id].append(websocket)
                print(f"[Pipeline] Registered WS client for camera {camera_id}")

    def unregister_websocket(self, camera_id: int, websocket: Any):
        with self._ws_lock:
            if websocket in self.websockets[camera_id]:
                self.websockets[camera_id].remove(websocket)
                print(f"[Pipeline] Unregistered WS client for camera {camera_id}")

    def get_alerts(self, limit: int = 50, offset: int = 0) -> List[AlertEvent]:
        with self._alerts_lock:
            sorted_alerts = sorted(self.alerts_history, key=lambda x: x.timestamp, reverse=True)
            return sorted_alerts[offset : offset + limit]

    def add_alert(self, alert: AlertEvent):
        with self._alerts_lock:
            self.alerts_history.append(alert)

    def _sync_face_toggle(self, camera_id: int, face_enabled: bool) -> None:
        prev = self._face_enabled_cache.get(camera_id, False)
        if prev and not face_enabled:
            face_pipeline_service.reset_camera(camera_id)
        elif not prev and face_enabled:
            face_pipeline_service.ensure_loaded()
        self._face_enabled_cache[camera_id] = face_enabled

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
            self._sync_face_toggle(cam_id, switches.face_enabled)
            pending.append(_PendingFrame(cam_id, frame, decoder, switches))
        return pending

    def _run_batched_inference(
        self, pending: List[_PendingFrame]
    ) -> Dict[int, List[RawDetection]]:
        frames = [item.frame for item in pending]
        fire_mask = [item.switches.fire_enabled for item in pending]
        weapon_mask = [item.switches.weapon_enabled for item in pending]

        t0 = time.perf_counter()
        fire_batch = yolo_runner.run_fire_batch(frames, fire_mask=fire_mask)
        t_fire = time.perf_counter()
        weapon_batch = yolo_runner.run_weapon_batch(frames, weapon_mask=weapon_mask)
        t_weapon = time.perf_counter()

        face_items = [
            (item.camera_id, item.frame)
            for item in pending
            if item.switches.face_enabled
        ]
        target_fps_by_camera = {
            item.camera_id: item.decoder.target_fps for item in pending if item.switches.face_enabled
        }
        face_batch = face_pipeline_service.process_batch(
            face_items,
            target_fps_by_camera=target_fps_by_camera,
        )
        t_face = time.perf_counter()

        latency_ms = (t_face - t0) * 1000.0
        per_camera_latency = latency_ms / max(len(pending), 1)
        face_pipeline_service.report_pipeline_pressure(latency_ms, len(pending))

        self._stats_accum["fire_ms"] += (t_fire - t0) * 1000.0
        self._stats_accum["weapon_ms"] += (t_weapon - t_fire) * 1000.0
        self._stats_accum["face_ms"] += (t_face - t_weapon) * 1000.0
        self._stats_accum["batches"] += 1.0

        detections_by_camera: Dict[int, List[RawDetection]] = {}
        for idx, item in enumerate(pending):
            detections_by_camera[item.camera_id] = YoloRunner.merge_detection_lists(
                fire_batch[idx],
                weapon_batch[idx],
                face_batch.get(item.camera_id, []),
            )
            item.decoder._last_inference_latency_ms = per_camera_latency

        return detections_by_camera

    def _maybe_log_pipeline_stats(self) -> None:
        now = time.time()
        interval = settings.pipeline_stats_interval_sec
        if interval <= 0 or now - self._stats_last_log < interval:
            return

        batches = max(self._stats_accum["batches"], 1.0)
        rec_every = face_pipeline_service.rec_interval_effective
        print(
            "[Pipeline] stage avg ms/batch — "
            f"fire={self._stats_accum['fire_ms'] / batches:.1f} "
            f"weapon={self._stats_accum['weapon_ms'] / batches:.1f} "
            f"face={self._stats_accum['face_ms'] / batches:.1f} "
            f"| rec_every={rec_every}"
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
                fps_scheduler.report_detection(cam_id, is_alert)

                alert_id = self.active_alerts[cam_id]
                if is_alert:
                    if alert_id is None:
                        alert_id = str(uuid.uuid4())
                        self.active_alerts[cam_id] = alert_id
                        print(f"[Pipeline] CAM {cam_id} Triggered ALERT: {alert_id}")

                        alert_detections = [
                            Detection(**det.to_normalized(w, h)) for det in triggering
                        ]
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
                else:
                    if alert_id is not None:
                        print(f"[Pipeline] CAM {cam_id} ALERT Cleared: {alert_id}")
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

                self._broadcast_frame(
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
            except Exception as e:
                print(f"[Pipeline] Error queuing WS frame for cam {camera_id}: {e}")

    def _load_alerts_history(self):
        history_file = settings.recordings_dir / "alerts_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    with self._alerts_lock:
                        self.alerts_history = [AlertEvent(**item) for item in data]
                print(f"[Pipeline] Loaded {len(self.alerts_history)} alert records from disk.")
            except Exception as e:
                print(f"[Pipeline] Error reading alerts history file: {e}")

    def _save_alerts_history(self):
        history_file = settings.recordings_dir / "alerts_history.json"
        try:
            with self._alerts_lock:
                serialized = [item.model_dump() for item in self.alerts_history]
            with open(history_file, "w") as f:
                json.dump(serialized, f, indent=4)
        except Exception as e:
            print(f"[Pipeline] Error writing alerts history file: {e}")


inference_pipeline = InferencePipeline()
