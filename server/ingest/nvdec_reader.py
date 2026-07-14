"""
NVDEC (GPU) video decoding via an ffmpeg subprocess.

Decodes with NVIDIA CUVID/NVDEC and pipes raw BGR24 frames over stdout.
File inputs are paced at native speed with ``-re`` (no slowdown/speedup) and
looped forever with ``-stream_loop -1`` to match the demo's looping behavior.
Network streams (rtsp/http/hls) pace themselves.

Falls back is handled by the caller (StreamDecoder) — if ffmpeg/NVDEC is not
usable this module raises and the decoder drops to CPU OpenCV capture.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional

import numpy as np

_CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
    "mjpeg": "mjpeg_cuvid",
}


class NvdecUnavailable(RuntimeError):
    pass


def probe(url: str) -> dict:
    """Return {width, height, fps, codec} for the first video stream."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise NvdecUnavailable("ffprobe not found")
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,avg_frame_rate,r_frame_rate",
        "-of", "json", url,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        raise NvdecUnavailable(f"ffprobe failed: {out.stderr.strip()[:200]}")
    streams = json.loads(out.stdout).get("streams") or []
    if not streams:
        raise NvdecUnavailable("no video stream found")
    s = streams[0]

    def _rate(expr: str) -> float:
        try:
            num, _, den = expr.partition("/")
            return float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            return 0.0

    fps = _rate(s.get("avg_frame_rate", "0/1")) or _rate(s.get("r_frame_rate", "0/1")) or 30.0
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": fps,
        "codec": s.get("codec_name", ""),
    }


class NvdecFrameReader:
    """Blocking frame reader over an ffmpeg NVDEC subprocess."""

    def __init__(self, url: str, is_file: bool, loop: bool = True):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise NvdecUnavailable("ffmpeg not found")

        info = probe(url)
        self.width: int = info["width"]
        self.height: int = info["height"]
        self.fps: float = info["fps"]
        self._frame_bytes = self.width * self.height * 3

        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda"]
        cuvid = _CUVID_DECODERS.get(info["codec"])
        if cuvid:
            cmd += ["-c:v", cuvid]
        if is_file:
            # -re paces reads at native rate (real-time playback, no drift);
            # -stream_loop -1 loops the file forever.
            if loop:
                cmd += ["-stream_loop", "-1"]
            cmd += ["-re"]
        elif url.startswith("rtsp"):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-i", url, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_bytes * 4,
        )
        # Fail fast if NVDEC init dies immediately (e.g. unsupported codec).
        first = self._read_exact(self._frame_bytes)
        if first is None:
            err = b""
            if self._proc.stderr is not None:
                try:
                    err = self._proc.stderr.read() or b""
                except Exception:
                    pass
            self.close()
            raise NvdecUnavailable(
                f"ffmpeg NVDEC produced no frames: {err.decode(errors='replace')[:300]}"
            )
        self._pending: Optional[np.ndarray] = self._to_frame(first)

    def read(self) -> Optional[np.ndarray]:
        """Next decoded frame, or None on EOF/process exit."""
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        buf = self._read_exact(self._frame_bytes)
        return None if buf is None else self._to_frame(buf)

    def _to_frame(self, buf: bytes) -> np.ndarray:
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3).copy()

    def _read_exact(self, n: int) -> Optional[bytes]:
        stdout = self._proc.stdout
        if stdout is None:
            return None
        chunks = bytearray()
        while len(chunks) < n:
            chunk = stdout.read(n - len(chunks))
            if not chunk:
                return None
            chunks.extend(chunk)
        return bytes(chunks)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        for pipe in (self._proc.stdout, self._proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass
