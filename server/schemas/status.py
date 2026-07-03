from pydantic import BaseModel, Field
from typing import List

class StreamStatus(BaseModel):
    camera_id: int = Field(..., description="Camera ID index (1-4).")
    is_active: bool = Field(..., description="Flag indicating if the camera decode thread is running and reading frames.")
    current_fps: float = Field(..., description="Current frame decode rate.")
    error_count: int = Field(..., description="Total cumulative decode errors encountered by this stream.")

class SystemHealth(BaseModel):
    gpu_available: bool = Field(..., description="Boolean indicating if CUDA-capable GPU is available and detected by PyTorch.")
    gpu_device: str | None = Field(None, description="CUDA device name when GPU inference is active.")
    models_loaded: List[str] = Field(..., description="Names of YOLO models loaded on the GPU.")
    streams: List[StreamStatus] = Field(..., description="List of stream status records for all configured cameras.")
