import time
import uuid
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from server.config import settings
from server.db import database as db
from server.recording.transcode import transcode_to_h264_inplace
from server.schemas.camera_recordings import CameraRecording


def _playback_url(recording_id: str) -> str:
    return f"/api/camera-recordings/{recording_id}"


def _row_to_schema(row: Dict[str, Any]) -> CameraRecording:
    return CameraRecording(
        id=row["id"],
        camera_id=row["camera_id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        duration_seconds=row["duration_seconds"],
        playback_url=_playback_url(row["id"]),
        file_size_bytes=row["file_size_bytes"],
        status=row["status"],
    )


@dataclass
class _ActiveRecording:
    recording_id: str
    camera_id: int
    started_at: float
    output_path: Path
    writer: Optional[cv2.VideoWriter]
    frame_size: tuple[int, int]
    frame_count: int = 0


class CameraRecorderManager:
    """
    Manages per-camera manual recordings.
    Raw decoded frames are written to MP4 on disk; metadata is stored in SQLite.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active: Dict[int, _ActiveRecording] = {}

    def is_recording(self, camera_id: int) -> bool:
        with self._lock:
            return camera_id in self._active

    def get_status(self, camera_id: int) -> Optional[CameraRecording]:
        with self._lock:
            active = self._active.get(camera_id)
            if active is None:
                return None
            row = db.get_recording(active.recording_id)
            return _row_to_schema(row) if row else None

    def start_recording(self, camera_id: int) -> CameraRecording:
        if camera_id not in range(1, 5):
            raise ValueError(f"Invalid camera_id: {camera_id}")

        with self._lock:
            if camera_id in self._active:
                raise RuntimeError(f"Camera {camera_id} is already recording")

            recording_id = str(uuid.uuid4())
            started_at = time.time()
            camera_dir = settings.camera_recordings_dir / f"camera_{camera_id}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{recording_id}.mp4"
            output_path = camera_dir / file_name

            db.insert_recording(
                {
                    "id": recording_id,
                    "camera_id": camera_id,
                    "started_at": started_at,
                    "ended_at": None,
                    "duration_seconds": None,
                    "file_name": file_name,
                    "file_size_bytes": None,
                    "status": "recording",
                    "created_at": started_at,
                }
            )

            self._active[camera_id] = _ActiveRecording(
                recording_id=recording_id,
                camera_id=camera_id,
                started_at=started_at,
                output_path=output_path,
                writer=None,
                frame_size=(0, 0),
            )

        row = db.get_recording(recording_id)
        assert row is not None
        print(f"[CameraRecorder] Started recording {recording_id} for camera {camera_id}")
        return _row_to_schema(row)

    def write_frame(self, camera_id: int, frame: Any, fps: Optional[float] = None) -> None:
        with self._lock:
            active = self._active.get(camera_id)
            if active is None:
                return

            h, w, _ = frame.shape
            if active.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(active.output_path),
                    fourcc,
                    float(fps) if fps else float(settings.base_fps),
                    (w, h),
                )
                if not writer.isOpened():
                    self._fail_recording_locked(active, "Could not open VideoWriter")
                    return
                active.writer = writer
                active.frame_size = (w, h)

            active.writer.write(frame)
            active.frame_count += 1

    def stop_recording(self, camera_id: int) -> CameraRecording:
        with self._lock:
            active = self._active.pop(camera_id, None)
            if active is None:
                raise RuntimeError(f"Camera {camera_id} is not recording")
            return self._finalize_active_locked(active)

    def stop_all(self) -> None:
        with self._lock:
            active_ids = list(self._active.keys())
            for camera_id in active_ids:
                active = self._active.pop(camera_id)
                self._finalize_active_locked(active)

    def list_recordings(
        self,
        camera_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CameraRecording]:
        rows = db.list_recordings(camera_id=camera_id, limit=limit, offset=offset)
        return [_row_to_schema(row) for row in rows]

    def get_recording(self, recording_id: str) -> Optional[CameraRecording]:
        row = db.get_recording(recording_id)
        return _row_to_schema(row) if row else None

    def get_recording_file_path(self, recording_id: str) -> Optional[Path]:
        row = db.get_recording(recording_id)
        if row is None:
            return None
        path = settings.camera_recordings_dir / f"camera_{row['camera_id']}" / row["file_name"]
        return path if path.exists() else None

    def _finalize_active_locked(self, active: _ActiveRecording) -> CameraRecording:
        ended_at = time.time()
        duration = ended_at - active.started_at

        if active.writer is not None:
            active.writer.release()

        # OpenCV wrote MPEG-4 Part 2 (see transcode.py) — no browser can
        # play that. Re-encode to H.264 in place; if ffmpeg fails, leave the
        # raw file so the recording isn't lost outright, just not
        # browser-playable.
        if active.frame_count > 0 and active.output_path.exists():
            if not transcode_to_h264_inplace(active.output_path):
                print(
                    f"[CameraRecorder] Warning: H.264 transcode failed for "
                    f"{active.recording_id}; recording may not play in-browser"
                )

        file_size = active.output_path.stat().st_size if active.output_path.exists() else 0
        status = "completed" if active.frame_count > 0 and file_size > 0 else "failed"

        db.update_recording(
            active.recording_id,
            {
                "ended_at": ended_at,
                "duration_seconds": round(duration, 2),
                "file_size_bytes": file_size,
                "status": status,
            },
        )

        row = db.get_recording(active.recording_id)
        assert row is not None
        print(
            f"[CameraRecorder] Stopped recording {active.recording_id} "
            f"({active.frame_count} frames, {status})"
        )
        return _row_to_schema(row)

    def _fail_recording_locked(self, active: _ActiveRecording, reason: str) -> None:
        print(f"[CameraRecorder] Recording {active.recording_id} failed: {reason}")
        if active.writer is not None:
            active.writer.release()
        db.update_recording(
            active.recording_id,
            {
                "ended_at": time.time(),
                "duration_seconds": 0,
                "file_size_bytes": 0,
                "status": "failed",
            },
        )
        self._active.pop(active.camera_id, None)


camera_recorder = CameraRecorderManager()
