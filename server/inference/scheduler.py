import time
import threading
from typing import Dict
from server.config import settings

class AdaptiveFPSScheduler:
    """
    Dynamically adjusts and balances target FPS allocations across the 4 cameras
    based on detection activity to prevent GPU saturation.
    
    Rules from PLAN.md:
    - Quiet state (no detections in last N frames) -> Increase priority (up to ~20-25 FPS).
    - Active state (entity/fire/weapon or zone alert) -> Decrease target FPS (down to ~8-10 FPS).
    - Global budget constraint: sum(FPS) <= 60 FPS.
    """
    def __init__(self, check_interval_sec: float = 0.5):
        self.check_interval_sec = check_interval_sec
        self.total_fps_cap = 60.0
        
        # State tracking per camera (1-4)
        self.last_detection_time: Dict[int, float] = {i: 0.0 for i in range(1, 5)}
        self.active_alert_state: Dict[int, bool] = {i: False for i in range(1, 5)}
        self.current_targets: Dict[int, float] = {i: float(settings.base_fps) for i in range(1, 5)}
        
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self, stream_manager_ref):
        """Starts the periodic rebalancing background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._scheduler_loop, 
            args=(stream_manager_ref,), 
            name="FPSScheduler", 
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stops the scheduler thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def report_detection(self, camera_id: int, has_detections: bool):
        """Called by the pipeline to report if detections were found in the latest frame."""
        with self._lock:
            if has_detections:
                self.last_detection_time[camera_id] = time.time()
            self.active_alert_state[camera_id] = has_detections

    def _scheduler_loop(self, stream_manager):
        print("[FPSScheduler] Adaptive scheduler thread started.")
        while self._running:
            time.sleep(self.check_interval_sec)
            
            with self._lock:
                now = time.time()
                requested_fps = {}
                
                for cam_id in range(1, 5):
                    decoder = stream_manager.get_decoder(cam_id)
                    if decoder is None or not decoder.is_active:
                        requested_fps[cam_id] = 0.0
                        continue
                    
                    # Check if stream has been quiet
                    time_since_detection = now - self.last_detection_time[cam_id]
                    is_quiet = time_since_detection > 5.0 and not self.active_alert_state[cam_id]
                    
                    if is_quiet:
                        # Quiet: request high priority to sweep area quickly (22 FPS target)
                        requested_fps[cam_id] = 22.0
                    else:
                        # Active: lower FPS to maintain slow/steady tracking (9 FPS target)
                        requested_fps[cam_id] = 9.0

                # Normalize to fit total budget (60 FPS)
                total_requested = sum(requested_fps.values())
                
                if total_requested > 0:
                    scale = min(1.0, self.total_fps_cap / total_requested)
                    
                    for cam_id, requested in requested_fps.items():
                        decoder = stream_manager.get_decoder(cam_id)
                        if decoder:
                            # Apply normalized target, keeping at least 5 FPS for basic visibility
                            target = max(5.0, requested * scale)
                            decoder.target_fps = target
                            self.current_targets[cam_id] = target
                else:
                    # Fallback if no streams active
                    for cam_id in range(1, 5):
                        self.current_targets[cam_id] = float(settings.base_fps)

# Global scheduler instance
fps_scheduler = AdaptiveFPSScheduler()
