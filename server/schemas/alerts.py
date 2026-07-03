from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Detection(BaseModel):
    bbox: List[float] = Field(
        ..., 
        description="Bounding box coordinates in [xmin, ymin, xmax, ymax] normalized format."
    )
    class_name: str = Field(..., description="Detected class label (e.g., person, fire, smoke, handgun).")
    confidence: float = Field(..., description="Model confidence value between 0.0 and 1.0.")

class AlertEvent(BaseModel):
    id: str = Field(..., description="Unique UUID for this alert event.")
    camera_id: int = Field(..., description="Camera ID index (1-4) that triggered the alert.")
    alert_type: str = Field(..., description="Type of alert, e.g., 'zone.intrusion', 'fire.detected', 'weapon.detected'.")
    timestamp: float = Field(..., description="Epoch timestamp when the alert event occurred.")
    detections: List[Detection] = Field(default_factory=list, description="List of detections associated with this alert.")
    clip_path: Optional[str] = Field(None, description="Path/URL to the recorded video clip of the event.")
