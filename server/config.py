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
    weapon_imgsz: int = 640
    fire_imgsz: int = 640

    # --- ONNX Runtime / TensorRT ---------------------------------------------
    # All models run through onnxruntime-gpu. Provider chain per model:
    # TensorRT EP (fp16, int8 for detectors) -> CUDA EP -> CPU.
    trt_enable: bool = True
    trt_fp16: bool = True
    int8_detectors: bool = False         # use int8-QDQ detector models when present
    ort_max_batch: int = 8               # TRT profile max batch / chunk size

    # --- Person stage (always-on, all cameras) --------------------------------
    # Person detection runs every n-th frame per camera; between detection
    # frames tracks coast on ByteTrack Kalman predictions. n adapts to mean
    # active-track confidence within [person_interval_min, person_interval_max].
    # Pinned to a fixed every-2nd-frame cadence: wider intervals (the old
    # default went up to 10, and the adaptive ceiling below could push it to
    # 20 under load) meant too many consecutive frames were pure Kalman
    # coasting with no real detection to correct drift, which is what was
    # producing sparse/poor-quality boxes.
    person_interval_min: int = 2
    person_interval_max: int = 2
    # Ceiling person_interval_max is allowed to widen to under sustained
    # pipeline lag (see FacePipelineService.report_pipeline_pressure) — the
    # same backpressure signal that already throttles facial-rec cadence.
    # Capped equal to person_interval_max so load can no longer silently
    # widen cadence past every-2nd-frame.
    person_interval_max_ceiling: int = 2
    track_conf_low: float = 0.5          # mean track score <= low  -> detect every frame
    track_conf_high: float = 0.7         # mean track score >= high -> longest interval

    # --- Fire stage ------------------------------------------------------------
    # Constant every-n cadence (fire bbox geometry is too volatile for adaptive n);
    # interpolated via a dedicated ByteTrack between detection frames. Pinned
    # to every-other-frame, same as person_interval_min/max above: fire boxes
    # were spending too many frames on pure Kalman interpolation with no real
    # detection to correct them, and fire_det_interval_max let sustained
    # pipeline lag silently widen the effective interval further still.
    # Capping the ceiling equal to the base interval closes that off.
    fire_det_interval: int = 2
    fire_det_interval_max: int = 2

    # --- Weapon stage (person-crop gated) --------------------------------------
    # Weapon detection runs only on padded person-track crops, batched across
    # cameras. Padding is a fraction of the person box added on each side.
    weapon_crop_padding: float = 0.25
    weapon_min_crop_px: int = 48         # skip crops smaller than this on either side

    # --- Motion gating ----------------------------------------------------------
    # Person + fire inference only run while a camera's motion gate is open
    # (EMA background diff on a downscaled grayscale frame).
    motion_enabled: bool = True
    motion_downscale_width: int = 320
    motion_pixel_delta: int = 25         # grayscale delta for a pixel to count as changed
    motion_area_ratio: float = 0.003     # changed-pixel ratio that counts as motion
    motion_cooldown_sec: float = 2.0     # keep gate open this long after last motion

    # --- Facial recognition ------------------------------------------------------
    facial_rec_interval: int = 5
    facial_rec_interval_max: int = 30
    adaptive_facial_throttle: bool = True
    pipeline_lag_threshold_ms: float = 150.0
    facial_match_threshold: float = 0.4
    insightface_model: str = "buffalo_l"   # buffalo_l | buffalo_m | buffalo_s
    insightface_det_size: int = 640

    # --- Tracking (ByteTrack) ---------------------------------------------------
    # Two-stage IOU association: a track must clear track_new_thresh to be born,
    # and is only matched against track_high_thresh/track_low_thresh detections
    # thereafter. A track not re-matched on a detection cycle drops out of the
    # displayed set immediately (track_buffer only keeps it alive internally, for
    # re-identification if the same object reappears) instead of continuing to
    # coast/render for the full buffer window.
    track_buffer: int = 30
    track_high_thresh: float = 0.4
    track_low_thresh: float = 0.1
    track_new_thresh: float = 0.6
    track_match_thresh: float = 0.8
    min_person_box: int = 40
    pipeline_stats_interval_sec: float = 30.0

    # --- Alerting (zone intrusion / fire / weapon debounce) ---------------------
    # zone_engine.check_detections() is a stateless per-frame check; without
    # debounce, a single flickering frame in/out of the zone polygon mints a
    # brand-new alert + clip every time. alert_trigger_frames requires M
    # consecutive in-zone frames before arming a new alert; alert_clear_frames
    # requires N consecutive no-detection frames before disarming — a brief
    # gap shorter than that is absorbed into the same still-active alert
    # instead of starting a new one.
    alert_trigger_frames: int = 5
    alert_clear_frames: int = 20

    # --- Clip / recording lifecycle ----------------------------------------------
    clip_writer_workers: int = 3          # concurrent clip-compile workers
    recordings_retention_days: float = 30.0
    recordings_cleanup_interval_sec: float = 3600.0

    # Inference toggles
    enable_entity_detection: bool = False
    use_cuda: bool = True
    cuda_device: int = 0

    # Ingestion: NVDEC (GPU) decode via ffmpeg, CPU OpenCV fallback.
    nvdec_enable: bool = True

    base_fps: int = 30
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
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.trt_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def onnx_dir(self) -> Path:
        return self.models_dir / "onnx"

    @property
    def trt_cache_dir(self) -> Path:
        return self.data_dir / "trt_cache"

    @property
    def facial_gallery_path(self) -> Path:
        return self.data_dir / "facial_rec" / "gallery_built" / "gallery.json"

    def detector_onnx(self, key: str) -> Path:
        """Resolve the ONNX file for a detector ('person'|'fire'|'weapon'),
        preferring the int8-QDQ variant when int8_detectors is enabled."""
        if self.int8_detectors:
            int8_path = self.onnx_dir / f"{key}.int8.onnx"
            if int8_path.is_file():
                return int8_path
        return self.onnx_dir / f"{key}.onnx"

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
