import os
import time
import cv2
import threading
import queue
from typing import List, Tuple, Any
from pathlib import Path
from server.config import settings
from server.ingest.ring_buffer import RingBuffer

class ClipWriter:
    """
    Asynchronously compiles 20-second video clips (10s pre-alert + 10s post-alert)
    upon receiving alert triggers. Saves output as MP4 in data/recordings/.
    """
    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread = None

    def start(self):
        """Starts the background worker thread for writing clips."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name="ClipWriterWorker", daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the background worker thread."""
        self._running = False
        self.queue.put(None)  # Wake up queue
        if self._thread:
            self._thread.join(timeout=2.0)

    def trigger_clip(self, camera_id: int, alert_id: str, ring_buffer: RingBuffer):
        """
        Signals the worker thread to begin compiling a clip for the alert.
        Takes a snapshot of the pre-alert frames immediately.
        """
        pre_alert_snapshot = ring_buffer.get_all()
        # Enqueue the task
        self.queue.put((camera_id, alert_id, ring_buffer, pre_alert_snapshot))
        print(f"[ClipWriter] Enqueued clip request for alert {alert_id}")

    def _worker_loop(self):
        print("[ClipWriter] Worker loop started.")
        while self._running:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break

            camera_id, alert_id, ring_buffer, pre_alert_snapshot = task
            try:
                self._compile_clip(camera_id, alert_id, ring_buffer, pre_alert_snapshot)
            except Exception as e:
                print(f"[ClipWriter] Error compiling clip for alert {alert_id}: {e}")
            finally:
                self.queue.task_done()

        print("[ClipWriter] Worker loop stopped.")

    def _compile_clip(
        self, 
        camera_id: int, 
        alert_id: str, 
        ring_buffer: RingBuffer, 
        pre_frames: List[Tuple[float, Any, Any]]
    ):
        """
        Sleeps to capture post-alert frames, merges with pre-alert, and writes MP4 file.
        """
        # Sleep for 10.0 seconds to allow the post-alert frames to accumulate in the ring buffer
        time.sleep(10.0)
        
        # Get the post-alert frames
        post_frames = ring_buffer.get_all()
        
        # Combine and remove duplicates by timestamp to ensure clean transitions
        seen_ts = set()
        merged: List[Tuple[float, Any]] = []
        
        for ts, frame, _ in pre_frames + post_frames:
            if ts not in seen_ts:
                seen_ts.add(ts)
                merged.append((ts, frame))
                
        # Sort chronologically
        merged.sort(key=lambda x: x[0])
        
        if not merged:
            print(f"[ClipWriter] Warning: No frames found to write for alert {alert_id}")
            return

        # Determine video writer properties
        first_frame = merged[0][1]
        h, w, _ = first_frame.shape
        
        output_path = settings.recordings_dir / f"{alert_id}.mp4"
        
        # Determine average frame rate of captured sequence
        fps = settings.base_fps
        if len(merged) > 1:
            total_duration = merged[-1][0] - merged[0][0]
            if total_duration > 0:
                fps = len(merged) / total_duration
                # Bound FPS to sensible limits
                fps = max(5.0, min(fps, 30.0))

        print(f"[ClipWriter] Writing {len(merged)} frames to {output_path} at {fps:.1f} FPS")
        
        # Open OpenCV VideoWriter
        # 'mp4v' represents standard MPEG-4 video inside MP4 container
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        if not out.isOpened():
            print(f"[ClipWriter] ERROR: Could not open VideoWriter for {output_path}")
            return
            
        try:
            for _, frame in merged:
                out.write(frame)
        finally:
            out.release()
            
        print(f"[ClipWriter] Successfully wrote video clip for alert {alert_id}")

# Global clip writer instance
clip_writer = ClipWriter()
