import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    camera_1_url: str = "assets/camera_1.mp4"
    camera_2_url: str = "assets/camera_2.mp4"
    camera_3_url: str = "assets/camera_3.mp4"
    camera_4_url: str = "assets/camera_4.mp4"
    
    model_entity: str = "yolo11s.pt"
    model_person: str = "yolo11n.pt"
    model_fire: str = "data/models/fire_smoke_yolov5.pt"
    model_weapon: str = "data/models/weapons_yolov8.pt"
    
    conf_entity: float = 0.5
    conf_fire: float = 0.25
    conf_weapon: float = 0.4
    person_conf: float = 0.25
    person_imgsz: int = 640
    person_use_trt: bool = True
    weapon_imgsz: int = 640
    weapon_half: bool = True

    facial_rec_interval: int = 5
    facial_rec_interval_max: int = 30
    adaptive_facial_throttle: bool = True
    pipeline_lag_threshold_ms: float = 150.0
    facial_match_threshold: float = 0.4
    insightface_model: str = "buffalo_l"
    insightface_det_size: int = 640
    track_buffer: int = 30
    min_person_box: int = 40
    pipeline_stats_interval_sec: float = 30.0

    # Inference toggles
    enable_entity_detection: bool = False
    use_cuda: bool = True
    cuda_device: int = 0
    
    base_fps: int = 15
    api_host: str = "0.0.0.0"
    api_port: int = 8001

    # Paths resolved dynamically
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    assets_dir: Path = project_root / "assets"
    zones_dir: Path = project_root / "data" / "zones"
    recordings_dir: Path = project_root / "data" / "recordings"
    camera_recordings_dir: Path = project_root / "data" / "recordings" / "cameras"
    models_dir: Path = project_root / "data" / "models"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.zones_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.camera_recordings_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def facial_gallery_path(self) -> Path:
        return self.data_dir / "facial_rec" / "gallery_built" / "gallery.json"

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
