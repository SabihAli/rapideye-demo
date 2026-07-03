from pydantic import BaseModel, Field


class CameraModelSwitches(BaseModel):
    camera_id: int = Field(..., description="Camera ID (1 to 4)", ge=1, le=4)
    fire_enabled: bool = Field(
        True,
        description="When false, the fire/smoke model is not run for this camera.",
    )
    weapon_enabled: bool = Field(
        True,
        description="When false, the weapon model is not run for this camera.",
    )
    face_enabled: bool = Field(
        False,
        description="When true, run YOLO11n person detection + facial recognition on this camera.",
    )
