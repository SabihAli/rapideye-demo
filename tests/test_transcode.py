"""Real (not mocked) coverage for the post-write H.264 transcode: writes an
actual OpenCV mp4v file and runs it through ffmpeg, same as production."""

import subprocess

import cv2
import numpy as np
import pytest

from server.recording.transcode import transcode_to_h264_inplace


def _probe_codec(path) -> str:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _write_mp4v_clip(path, frames=10, size=(64, 48)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def test_transcode_converts_mp4v_to_h264(tmp_path):
    clip_path = tmp_path / "clip.mp4"
    _write_mp4v_clip(clip_path)
    assert _probe_codec(clip_path) == "mpeg4"

    ok = transcode_to_h264_inplace(clip_path)

    assert ok is True
    assert clip_path.is_file()
    assert _probe_codec(clip_path) == "h264"


def test_transcode_leaves_original_untouched_on_failure(tmp_path):
    not_a_video = tmp_path / "clip.mp4"
    not_a_video.write_bytes(b"not actually a video file")

    ok = transcode_to_h264_inplace(not_a_video)

    assert ok is False
    # Original file is left in place rather than deleted on failure.
    assert not_a_video.read_bytes() == b"not actually a video file"
    assert not (tmp_path / "clip.h264tmp.mp4").exists()
