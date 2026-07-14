"""Post-write H.264 transcode for recorded MP4s.

OpenCV's VideoWriter in this environment can't produce real H.264: its
FFMPEG backend only exposes a hardware V4L2 encoder here (no matching
device present, so it fails), and falls back silently to old MPEG-4 Part 2
("mp4v" fourcc) instead. That's a perfectly valid MP4 file — ffprobe reads
it fine, VLC plays it fine — but no browser's <video> element can decode
MPEG-4 Part 2, so every recorded clip and manual recording appeared broken
in the frontend player. The system `ffmpeg` binary (already a hard
dependency for NVDEC ingest) does have libx264, so recordings are
re-encoded through it after OpenCV finishes writing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def transcode_to_h264_inplace(path: Path, timeout: float = 120.0) -> bool:
    """Re-encodes ``path`` to H.264/yuv420p with a faststart MP4 moov atom,
    replacing it in place. Returns True on success; leaves the original
    file untouched (still whatever OpenCV wrote) if ffmpeg fails, so a
    transcode failure degrades to today's behavior rather than losing the
    recording."""
    tmp_path = path.parent / f"{path.stem}.h264tmp.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[Transcode] ffmpeg failed for {path.name}: {exc}")
        tmp_path.unlink(missing_ok=True)
        return False

    if result.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        stderr = result.stderr.decode(errors="replace")[:500]
        print(f"[Transcode] ffmpeg failed for {path.name} (exit {result.returncode}): {stderr}")
        tmp_path.unlink(missing_ok=True)
        return False

    tmp_path.replace(path)
    return True
