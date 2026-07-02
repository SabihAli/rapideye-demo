import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    camera_1_url: str = "assets/camera_1.mp4"
    camera_2_url: str = "assets/camera_2.mp4"
    camera_3_url: str = "assets/camera_3.mp4"
    camera_4_url: str = "assets/camera_4.mp4"
    
    model_entity: str = "yolo11s.pt"
    model_fire: str = "data/models/fire_smoke_yolov5.pt"
    model_weapon: str = "data/models/weapons_yolov8.pt"
    
    conf_entity: float = 0.5
    conf_fire: float = 0.4
    conf_weapon: float = 0.4
    
    base_fps: int = 15
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Paths resolved dynamically
    project_root: Path = Path(__file__).resolve().parent.parent
    assets_dir: Path = project_root / "assets"
    zones_dir: Path = project_root / "data" / "zones"
    recordings_dir: Path = project_root / "data" / "recordings"
    models_dir: Path = project_root / "data" / "models"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.zones_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_camera_url(self, camera_id: int) -> str:
        """Helper to get URL by camera_id index (1-4)."""
        mapping = {
            1: self.camera_1_url,
            2: self.camera_2_url,
            3: self.camera_3_url,
            4: self.camera_4_url
        }
        return mapping.get(camera_id, "")

settings = Settings()
