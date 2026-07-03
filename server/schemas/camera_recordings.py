from typing import Literal, Optional

from pydantic import BaseModel, Field


RecordingStatus = Literal["recording", "completed", "failed"]


class CameraRecording(BaseModel):
    id: str = Field(..., description="Unique UUID for this camera recording.")
    camera_id: int = Field(..., description="Camera ID index (1-4).")
    started_at: float = Field(..., description="Epoch timestamp when recording started.")
    ended_at: Optional[float] = Field(None, description="Epoch timestamp when recording ended.")
    duration_seconds: Optional[float] = Field(None, description="Recorded duration in seconds.")
    playback_url: str = Field(..., description="API URL to stream/download the MP4.")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes after completion.")
    status: RecordingStatus = Field(..., description="recording, completed, or failed.")


class CameraRecordingStatus(BaseModel):
    camera_id: int
    is_recording: bool
    recording: Optional[CameraRecording] = None
