import time
import cv2
import threading
import subprocess
import shutil
from typing import Optional, Any
from pathlib import Path
from server.ingest.ring_buffer import RingBuffer

class StreamDecoder:
    """
    Decodes a hardcoded local video file or RTSP stream on an independent background thread.
    Maintains a 10s pre-alert RingBuffer and throttles decoding/pushing according to target_fps.
    """
    def __init__(self, camera_id: int, url: str, base_fps: int = 15):
        self.camera_id = camera_id
        self.url = url
        self.target_fps = float(base_fps)
        self.ring_buffer = RingBuffer(max_len=int(self.target_fps * 10))
        
        self.is_active = False
        self.error_count = 0
        self.fps_measure = 0.0
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[Any] = None
        self._last_push_time = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._reset_triggered = False
        
        # Lock for accessing latest frame
        self._frame_lock = threading.Lock()

    def start(self):
        """Starts the background decoder thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name=f"Decoder-Cam{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the background decoder thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.is_active = False

    def reset(self):
        """Signals the decoder to reset video position to frame 0 and clears stats/buffers."""
        print(f"[Decoder Cam {self.camera_id}] Reset triggered.")
        self._reset_triggered = True
        self.ring_buffer.clear()
        self.error_count = 0

    def get_latest_frame(self) -> Optional[Any]:
        """Thread-safe retrieval of the latest frame."""
        with self._frame_lock:
            return self._latest_frame

    def _run_loop(self):
        print(f"[Decoder Cam {self.camera_id}] Thread started for {self.url}")
        
        # Resolve path relative to project root if not absolute
        resolved_url = self.url
        path_obj = Path(resolved_url)
        if not path_obj.is_absolute():
            from server.config import settings
            resolved_url = str(settings.project_root / resolved_url)

        is_file = Path(resolved_url).exists() or resolved_url.endswith((".mp4", ".avi", ".mkv"))
        
        cap = None
        reconnect_delay = 5.0
        
        while self._running:
            # Handle reset trigger in thread context
            if self._reset_triggered:
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._reset_triggered = False
                self._last_push_time = time.time()
                self._frame_count = 0
                self._fps_start_time = time.time()

            if cap is None or not cap.isOpened():
                self.is_active = False
                print(f"[Decoder Cam {self.camera_id}] Opening file/stream: {resolved_url}")
                cap = cv2.VideoCapture(resolved_url)
                if not cap.isOpened():
                    self.error_count += 1
                    print(f"[Decoder Cam {self.camera_id}] Open failed. Retrying in {reconnect_delay}s...")
                    time.sleep(reconnect_delay)
                    continue
                self.is_active = True
                self._last_push_time = time.time()
                self._fps_start_time = time.time()
                self._frame_count = 0

            ret, frame = cap.read()
            if not ret:
                self.error_count += 1
                if is_file:
                    # Loop video back to start
                    print(f"[Decoder Cam {self.camera_id}] End of video file. Looping back to start.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1)
                    continue
                else:
                    print(f"[Decoder Cam {self.camera_id}] Read error. Reconnecting stream...")
                    cap.release()
                    cap = None
                    time.sleep(1.0)
                    continue

            now = time.time()
            self._frame_count += 1
            elapsed = now - self._fps_start_time
            if elapsed >= 2.0:
                self.fps_measure = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = now

            target_interval = 1.0 / max(self.target_fps, 1.0)
            time_since_push = now - self._last_push_time
            
            if is_file:
                sleep_needed = target_interval - time_since_push
                if sleep_needed > 0:
                    time.sleep(sleep_needed)
                    now = time.time()
                
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                self.ring_buffer.append(now, self._latest_frame)
                self._last_push_time = now
            else:
                if time_since_push >= target_interval:
                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                    self.ring_buffer.append(now, self._latest_frame)
                    self._last_push_time = now
                time.sleep(0.001)

        if cap is not None:
            cap.release()
        self.is_active = False
        print(f"[Decoder Cam {self.camera_id}] Stopped.")
