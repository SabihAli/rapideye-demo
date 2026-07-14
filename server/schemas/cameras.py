from typing import Literal, Optional

from pydantic import BaseModel, Field


CameraSourceType = Literal["url", "file"]


class CameraOut(BaseModel):
    camera_id: int = Field(..., description="Camera slot index (1-4).")
    name: str = Field(..., description="User-provided display name.")
    source_type: CameraSourceType = Field(..., description="'url' (RTSP/HTTP stream) or 'file' (uploaded video).")
    source_value: str = Field(..., description="Stream URL, or server-side path to the uploaded file.")
    original_filename: Optional[str] = Field(None, description="Original uploaded filename, when source_type is 'file'.")
    created_at: float = Field(..., description="Epoch timestamp when this camera was added.")
    is_active: bool = Field(..., description="Whether the decode thread is currently running and reading frames.")
    current_fps: float = Field(..., description="Current frame decode rate.")
    error_count: int = Field(..., description="Total cumulative decode errors encountered by this stream.")
