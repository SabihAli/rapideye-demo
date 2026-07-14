import os
import time
import cv2
import threading
import queue
from typing import Callable, List, Optional, Tuple, Any
from pathlib import Path
from server.config import settings
from server.ingest.ring_buffer import RingBuffer
from server.recording.transcode import transcode_to_h264_inplace

class ClipWriter:
    """
    Asynchronously compiles 20-second video clips (10s pre-alert + 10s post-alert)
    upon receiving alert triggers. Saves output as MP4 in data/recordings/.
    """
    def __init__(self, num_workers: int = 3, max_queue: int = 12):
        # Bounded: each queued task carries a full raw-frame ring-buffer
        # snapshot (hundreds of MB). Workers are inherently slow (a fixed
        # ~10s sleep per clip in _compile_clip before encoding even starts),
        # so with no cap a camera that re-arms faster than the pool can drain
        # would pile up snapshots in memory without bound. alert_clip_cooldown_sec
        # is the primary throttle; this is the backstop — trigger_clip drops
        # and logs instead of blocking/growing when the backstop is also hit.
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._running = False
        self._threads: List[threading.Thread] = []
        self._num_workers = max(1, num_workers)
        # Invoked as on_complete(alert_id, success) from whichever worker
        # thread finishes a clip — lets callers (InferencePipeline) flip the
        # alert's clip_ready flag once the file actually exists, instead of
        # publishing clip_path before the clip is even compiled.
        self.on_complete: Optional[Callable[[str, bool], None]] = None

    def start(self):
        """Starts the background worker thread pool for writing clips."""
        if self._running:
            return
        self._running = True
        for i in range(self._num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"ClipWriterWorker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        """Stops the background worker threads."""
        self._running = False
        for _ in self._threads:
            self.queue.put(None)  # One wake-up per worker
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def trigger_clip(self, camera_id: int, alert_id: str, ring_buffer: RingBuffer):
        """
        Signals a worker thread to begin compiling a clip for the alert.
        Takes a snapshot of the pre-alert frames immediately.
        """
        pre_alert_snapshot = ring_buffer.get_all()
        try:
            self.queue.put_nowait((camera_id, alert_id, ring_buffer, pre_alert_snapshot))
            print(f"[ClipWriter] Enqueued clip request for alert {alert_id}")
        except queue.Full:
            print(
                f"[ClipWriter] Queue full ({self.queue.qsize()} pending) — "
                f"dropping clip for alert {alert_id}"
            )
            if self.on_complete:
                try:
                    self.on_complete(alert_id, False)
                except Exception as exc:
                    print(f"[ClipWriter] on_complete callback failed for alert {alert_id}: {exc}")

    def _worker_loop(self):
        print(f"[ClipWriter] Worker loop started ({threading.current_thread().name}).")
        while self._running:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break

            camera_id, alert_id, ring_buffer, pre_alert_snapshot = task
            success = False
            try:
                success = self._compile_clip(camera_id, alert_id, ring_buffer, pre_alert_snapshot)
            except Exception as e:
                print(f"[ClipWriter] Error compiling clip for alert {alert_id}: {e}")
            finally:
                self.queue.task_done()
                if self.on_complete:
                    try:
                        self.on_complete(alert_id, success)
                    except Exception as exc:
                        print(f"[ClipWriter] on_complete callback failed for alert {alert_id}: {exc}")

        print(f"[ClipWriter] Worker loop stopped ({threading.current_thread().name}).")

    def _compile_clip(
        self,
        camera_id: int,
        alert_id: str,
        ring_buffer: RingBuffer,
        pre_frames: List[Tuple[float, Any, Any]]
    ) -> bool:
        """
        Sleeps to capture post-alert frames, merges with pre-alert, and writes MP4 file.
        Returns True iff the clip was written successfully.
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
            return False

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
            return False

        try:
            for _, frame in merged:
                out.write(frame)
        finally:
            out.release()

        # OpenCV wrote MPEG-4 Part 2 (see transcode.py) — no browser can
        # play that. Re-encode to H.264 in place; if ffmpeg fails for some
        # reason, leave the raw file so the clip isn't lost outright, just
        # not browser-playable.
        if not transcode_to_h264_inplace(output_path):
            print(f"[ClipWriter] Warning: H.264 transcode failed for alert {alert_id}; clip may not play in-browser")

        print(f"[ClipWriter] Successfully wrote video clip for alert {alert_id}")
        return True

# Global clip writer instance
clip_writer = ClipWriter(num_workers=settings.clip_writer_workers)
