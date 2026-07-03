import time
import cv2
import json
import base64
import uuid
import threading
from typing import Dict, List, Any, Optional
from server.config import settings
from server.ingest.stream_manager import stream_manager
from server.inference.yolo_runner import yolo_runner, RawDetection
from server.inference.zone_engine import zone_engine
from server.inference.scheduler import fps_scheduler
from server.inference.annotator import Annotator
from server.recording.camera_recorder import camera_recorder
from server.schemas.alerts import AlertEvent, Detection

class InferencePipeline:
    """
    Main processing pipeline that orchestrates GPU inference, zone validation,
    annotation rendering, clip trigger detection, and WebSocket broadcasting.
    """
    def __init__(self):
        self.alerts_history: List[AlertEvent] = []
        self._alerts_lock = threading.Lock()
        
        # Track active alert state per camera to avoid spamming alerts
        # Stores {camera_id: active_alert_id} or None
        self.active_alerts: Dict[int, Optional[str]] = {i: None for i in range(1, 5)}
        
        # Track websocket connections: {camera_id: [websocket_instances]}
        self.websockets: Dict[int, List[Any]] = {i: [] for i in range(1, 5)}
        self._ws_lock = threading.Lock()
        
        self._last_processed_ts: Dict[int, float] = {i: 0.0 for i in range(1, 5)}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.clip_writer = None  # Injected later to break circular dependency

    def start(self, clip_writer_ref=None):
        """Starts the pipeline thread."""
        self.clip_writer = clip_writer_ref
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._pipeline_loop, name="InferencePipeline", daemon=True)
        self._thread.start()

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
        """Registers a client WebSocket for real-time frame streaming."""
        with self._ws_lock:
            if websocket not in self.websockets[camera_id]:
                self.websockets[camera_id].append(websocket)
                print(f"[Pipeline] Registered WS client for camera {camera_id}")

    def unregister_websocket(self, camera_id: int, websocket: Any):
        """Unregisters a client WebSocket."""
        with self._ws_lock:
            if websocket in self.websockets[camera_id]:
                self.websockets[camera_id].remove(websocket)
                print(f"[Pipeline] Unregistered WS client for camera {camera_id}")

    def get_alerts(self, limit: int = 50, offset: int = 0) -> List[AlertEvent]:
        """Thread-safe retrieval of alert history."""
        with self._alerts_lock:
            # Sorted newest first
            sorted_alerts = sorted(self.alerts_history, key=lambda x: x.timestamp, reverse=True)
            return sorted_alerts[offset:offset + limit]

    def add_alert(self, alert: AlertEvent):
        """Thread-safe registration of a new alert."""
        with self._alerts_lock:
            self.alerts_history.append(alert)

    def _pipeline_loop(self):
        print("[Pipeline] Main processing loop started.")
        
        # Load existing alerts from files if any (optional extension)
        self._load_alerts_history()

        while self._running:
            processed_any = False
            
            for cam_id in range(1, 5):
                decoder = stream_manager.get_decoder(cam_id)
                if decoder is None or not decoder.is_active:
                    continue

                # Only process if decoder has a new frame
                latest_ts = decoder._last_push_time
                if latest_ts == 0.0 or latest_ts <= self._last_processed_ts[cam_id]:
                    continue

                self._last_processed_ts[cam_id] = latest_ts
                frame = decoder.get_latest_frame()
                if frame is None:
                    continue

                processed_any = True
                h, w, _ = frame.shape

                if camera_recorder.is_recording(cam_id):
                    camera_recorder.write_frame(cam_id, frame)
                
                # 1. Run YOLO Models
                start_time = time.time()
                detections = yolo_runner.run_inference(frame)
                latency_ms = (time.time() - start_time) * 1000.0

                # 2. Evaluate Zone Intrusions
                is_alert, triggering = zone_engine.check_detections(cam_id, detections, w, h)
                
                # Update scheduler on detection states
                fps_scheduler.report_detection(cam_id, is_alert)

                # 3. Handle Alert Lifecycle (Transitions & Video Clips)
                alert_id = self.active_alerts[cam_id]
                
                if is_alert:
                    if alert_id is None:
                        # New alert boundary transition!
                        alert_id = str(uuid.uuid4())
                        self.active_alerts[cam_id] = alert_id
                        print(f"[Pipeline] CAM {cam_id} Triggered ALERT: {alert_id}")
                        
                        # Gather triggering detections
                        alert_detections = [
                            Detection(**det.to_normalized(w, h)) 
                            for det in triggering
                        ]
                        
                        # Create Alert Event Schema
                        event = AlertEvent(
                            id=alert_id,
                            camera_id=cam_id,
                            alert_type=self._determine_alert_type(triggering),
                            timestamp=time.time(),
                            detections=alert_detections,
                            clip_path=f"/api/recordings/{alert_id}"
                        )
                        self.add_alert(event)
                        self._save_alerts_history()
                        
                        # Trigger Clip Writer to capture pre-alert + post-alert MP4
                        if self.clip_writer:
                            self.clip_writer.trigger_clip(cam_id, alert_id, decoder.ring_buffer)
                else:
                    if alert_id is not None:
                        # Transition back to secure!
                        print(f"[Pipeline] CAM {cam_id} ALERT Cleared: {alert_id}")
                        self.active_alerts[cam_id] = None

                # 4. Annotate Frame
                zone_config = zone_engine.get_zone(cam_id)
                annotated = Annotator.draw_overlays(
                    frame=frame,
                    detections=detections,
                    zone_config=zone_config,
                    is_alert=is_alert,
                    current_fps=decoder.target_fps,
                    latency_ms=latency_ms,
                    camera_id=cam_id
                )

                # 5. Broadcast to WebSockets
                self._broadcast_frame(cam_id, annotated, detections, is_alert, decoder.target_fps, latency_ms)

            # Avoid tight spin lock if no streams had new frames
            if not processed_any:
                time.sleep(0.005)

        print("[Pipeline] Process loop stopped.")

    def _determine_alert_type(self, triggering: List[RawDetection]) -> str:
        """Determines type classification based on triggering classes."""
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
        latency_ms: float
    ):
        """Encodes frame and sends JSON update containing metadata to WS subscribers."""
        with self._ws_lock:
            subs = self.websockets[camera_id]
            if not subs:
                return

        # Encode to JPEG
        ret, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return
        
        # Base64 Encode
        base64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
        h, w, _ = frame.shape
        
        # Compile metadata
        payload = {
            "camera_id": camera_id,
            "frame": f"data:image/jpeg;base64,{base64_str}",
            "detections": [d.to_normalized(w, h) for d in detections],
            "is_alert": is_alert,
            "target_fps": round(target_fps, 1),
            "latency_ms": round(latency_ms, 1)
        }

        # Send asynchronously to all clients on this camera
        import asyncio
        for ws in list(subs):
            try:
                # Use call_soon_threadsafe if loops are running or handle in FastAPI endpoint directly
                # To make it robust and easy, the websocket router endpoint will pull from a queue or
                # we call the WebSocket's send_json directly if compatible (FastAPI WebSockets are not thread safe,
                # so we can push to a thread-safe Queue per WS connection, which the WS handler reads!).
                # This queue-based pattern is extremely thread-safe and robust!
                if hasattr(ws, "push_queue"):
                    ws.push_queue.put_nowait(payload)
            except Exception as e:
                print(f"[Pipeline] Error queuing WS frame for cam {camera_id}: {e}")

    def _load_alerts_history(self):
        """Loads alert history from a JSON file in recordings directory."""
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
        """Saves alert history to a JSON file in recordings directory."""
        history_file = settings.recordings_dir / "alerts_history.json"
        try:
            with self._alerts_lock:
                serialized = [item.model_dump() for item in self.alerts_history]
            with open(history_file, "w") as f:
                json.dump(serialized, f, indent=4)
        except Exception as e:
            print(f"[Pipeline] Error writing alerts history file: {e}")

# Global pipeline instance
inference_pipeline = InferencePipeline()
