import time
import cv2
import threading
from typing import Optional, Any
from pathlib import Path
from server.ingest.ring_buffer import RingBuffer
from server.ingest.nvdec_reader import NvdecFrameReader, NvdecUnavailable

class StreamDecoder:
    """
    Decodes a local video file or RTSP stream on an independent background thread.

    Decode path: NVDEC (GPU, ffmpeg cuvid) when NVDEC_ENABLE is set, with
    automatic fallback to CPU OpenCV capture. Streams run at their native
    frame rate — files are paced in real time (no slowdown/speedup) and loop
    forever. There is no FPS throttling; the pipeline consumes the latest
    frame only, so a slow consumer never delays or distorts playback.

    Maintains a thread-safe latest-frame slot, plus owns the lifecycle
    (creation, native-FPS resize) of a ~10 s pre-alert RingBuffer — but does
    NOT populate that buffer itself. It's filled with *annotated* frames
    (boxes/labels/zone overlay baked in) by InferencePipeline's main loop,
    once per processed tick, so alert clips and manual recordings show the
    same overlays a live viewer would see rather than a clean raw feed.
    """
    def __init__(self, camera_id: int, url: str, base_fps: int = 30):
        self.camera_id = camera_id
        self.url = url
        # Native stream rate once known; base_fps until probed.
        self.target_fps = float(base_fps)
        self.ring_buffer = RingBuffer(max_len=int(self.target_fps * 10))
        self.decode_backend = "none"

        self.is_active = False
        self.error_count = 0
        self.fps_measure = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[Any] = None
        self._last_push_time = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()

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
            self._thread.join(timeout=4.0)
        self.is_active = False

    def get_latest_frame(self) -> Optional[Any]:
        """Thread-safe retrieval of the latest frame."""
        with self._frame_lock:
            return self._latest_frame

    def _push_frame(self, frame) -> None:
        now = time.time()
        self._frame_count += 1
        elapsed = now - self._fps_start_time
        if elapsed >= 2.0:
            self.fps_measure = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = now
        with self._frame_lock:
            self._latest_frame = frame
        self._last_push_time = now

    def _run_loop(self):
        print(f"[Decoder Cam {self.camera_id}] Thread started for {self.url}")

        # Resolve path relative to project root if not absolute
        from server.config import settings
        resolved_url = self.url
        path_obj = Path(resolved_url)
        if not path_obj.is_absolute() and not resolved_url.lower().startswith(
            ("rtsp://", "http://", "https://")
        ):
            resolved_url = str(settings.project_root / resolved_url)

        is_file = Path(resolved_url).exists()

        while self._running:
            if settings.nvdec_enable and self._run_nvdec(resolved_url, is_file):
                continue  # NVDEC loop exited (stop or stream error) — retry/exit
            if not self._running:
                break
            self._run_opencv(resolved_url, is_file)

        self.is_active = False
        print(f"[Decoder Cam {self.camera_id}] Stopped.")

    def _apply_native_fps(self, fps: float) -> None:
        # Called on every (re)connect, not just the first probe. Replacing
        # self.ring_buffer unconditionally would abandon the old object mid-
        # clip: ClipWriter.trigger_clip() captures a specific RingBuffer
        # reference and reads it again ~10s later for post-alert frames — if
        # a reconnect happens in between, new frames would go to a new object
        # while the in-flight clip keeps reading the now-frozen old one,
        # silently truncating it. Only replace when the size actually needs
        # to change; otherwise keep appending to the same buffer.
        if fps and fps > 0:
            self.target_fps = float(fps)
            new_max_len = int(self.target_fps * 10)
            if new_max_len != self.ring_buffer.max_len:
                self.ring_buffer = RingBuffer(max_len=new_max_len)

    def _run_nvdec(self, url: str, is_file: bool) -> bool:
        """Decode via ffmpeg NVDEC until stop/stream end. Returns False if
        NVDEC is unavailable (caller falls back to OpenCV)."""
        try:
            reader = NvdecFrameReader(url, is_file=is_file)
        except NvdecUnavailable as e:
            print(f"[Decoder Cam {self.camera_id}] NVDEC unavailable ({e}); using CPU decode.")
            return False
        except Exception as e:
            print(f"[Decoder Cam {self.camera_id}] NVDEC init error ({e}); using CPU decode.")
            return False

        self._apply_native_fps(reader.fps)
        self.decode_backend = "nvdec"
        self.is_active = True
        print(
            f"[Decoder Cam {self.camera_id}] NVDEC decode {reader.width}x{reader.height} "
            f"@ {reader.fps:.1f}fps (native pacing)"
        )
        try:
            while self._running:
                frame = reader.read()
                if frame is None:
                    self.error_count += 1
                    print(f"[Decoder Cam {self.camera_id}] NVDEC stream ended/broke. Reopening...")
                    break
                self._push_frame(frame)
        finally:
            reader.close()
            self.is_active = False
        if self._running:
            time.sleep(1.0)
        return True

    def _run_opencv(self, url: str, is_file: bool) -> None:
        """CPU OpenCV fallback. Files are paced at native FPS; loops forever."""
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            self.error_count += 1
            print(f"[Decoder Cam {self.camera_id}] Open failed. Retrying in 5s...")
            time.sleep(5.0)
            return

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._apply_native_fps(native_fps)
        self.decode_backend = "opencv"
        self.is_active = True
        frame_interval = 1.0 / max(native_fps, 1.0)
        next_due = time.time()
        print(f"[Decoder Cam {self.camera_id}] CPU decode @ {native_fps:.1f}fps (native pacing)")

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.error_count += 1
                    print(f"[Decoder Cam {self.camera_id}] Read error. Reconnecting stream...")
                    break
                if is_file:
                    # Real-time pacing so file playback matches wall clock.
                    now = time.time()
                    if now < next_due:
                        time.sleep(next_due - now)
                    next_due = max(next_due + frame_interval, time.time() - frame_interval)
                self._push_frame(frame)
        finally:
            cap.release()
            self.is_active = False
        if self._running:
            time.sleep(1.0)
