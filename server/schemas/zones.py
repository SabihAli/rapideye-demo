from pydantic import BaseModel, Field
from typing import List

class ZoneConfig(BaseModel):
    camera_id: int = Field(..., description="Camera ID (1 to 4)", ge=1, le=4)
    polygon: List[List[float]] = Field(
        default_factory=list, 
        description="Normalized coordinates of the polygon vertices, each point represented as [x, y] with values in range [0, 1]."
    )
    alert_classes: List[str] = Field(
        default_factory=lambda: ["person", "vehicle", "fire", "smoke", "weapon"],
        description="List of classes that will trigger zone alerts when detected inside the polygon."
    )
