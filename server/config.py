import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_entity: str = "yolo11s.pt"
    model_person: str = "yolo11n.pt"
    model_fire: str = "data/models/fire_smoke_yolov5.pt"
    model_weapon: str = "data/models/weapon.pt"

    conf_entity: float = 0.5
    conf_fire: float = 0.25
    conf_weapon: float = 0.6
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
    # Pinned to a fixed every-30th-frame cadence (min == max, so
    # adaptive_facial_throttle has no room to shrink it under low load): more
    # frequent attempts meant more chances for a single low-quality embedding
    # (blur, bad angle, partial occlusion during a close pass) to corrupt a
    # track's identity. facial_rec_interval_max is still the ceiling the
    # backpressure logic can widen to under sustained pipeline lag.
    facial_rec_interval: int = 10
    facial_rec_interval_max: int = 10
    # Tracks with no confirmed identity yet (brand-new, or just reset after a
    # ByteTrack ID jump/lost-track revival) use this much shorter cadence
    # instead of facial_rec_interval: there's no established identity yet for
    # a bad embedding to corrupt, so the corruption-risk reasoning above
    # doesn't apply, and reacquiring a label quickly matters more here than
    # in the steady-state case. Once identity_min_matches lands and the
    # track has a label, it falls back to the conservative facial_rec_interval
    # cadence. Floored at person_interval_min/max (2): recognition attempts
    # only ever run on a detect cycle (process_batch's detect_mask gate), so
    # anything below that just wastes the retry rather than firing sooner.
    # With identity_min_matches now at 1, every retry is a full attempt at
    # commit (not partial corroboration) — so this interval alone is what
    # determines "how many real seconds until a newly-entered, unidentified
    # person gets a name," and it's now pinned to the fastest cadence
    # recognition can actually run at.
    facial_rec_interval_probation: int = 2
    adaptive_facial_throttle: bool = True
    pipeline_lag_threshold_ms: float = 150.0
    facial_match_threshold: float = 0.5
    # Single source of truth for gallery filenames — previously each of
    # server/config.py, server/inference/face_recognition.py's CLI defaults,
    # and scripts/build_office_gallery.py hardcoded their own "gallery_P1E_S2_C1.json"/
    # "gallery_office.json" literal, which is exactly how they drifted out of
    # sync with each other before. facial_gallery_filename is the "live"
    # gallery FacePipelineService actually loads (see facial_gallery_path
    # below); facial_gallery_office_filename is the human-labelled snapshot
    # scripts/build_office_gallery.py also writes alongside it, for
    # inspection — nothing in the live pipeline reads it.
    facial_gallery_filename: str = "gallery_P1E_S2_C1.json"
    facial_gallery_office_filename: str = "gallery_office.json"
    insightface_model: str = "buffalo_m"   # buffalo_l | buffalo_m | buffalo_s
    insightface_det_size: int = 640
    # Quality gate on the extracted face itself (not the surrounding person
    # box) before an embedding is even matched against the gallery — see
    # _face_embedding_from_person_crop.
    face_min_det_score: float = 0.5
    face_min_size_px: int = 40
    # Identity commitment: a track needs this many recognition attempts
    # voting for the same name (accumulated evidence, not a single
    # prediction) before it's assigned at all. Once assigned, a competing
    # name can only take over if its single-match similarity beats the
    # current identity's best-seen similarity by this margin — otherwise
    # the identity stays attached to the track through noisy re-checks.
    identity_min_matches: int = 1
    identity_override_margin: float = 0.15
    # Identity staleness: an established identity stays attached through a
    # brief bad-angle/blur attempt (that's what identity_override_margin
    # above already protects against on the "wrong new identity" side), but
    # nothing previously cleared it when the face is durably gone entirely —
    # e.g. the person turned their back and never faces the camera again.
    # After this many consecutive recognition attempts with no face detected
    # in the crop at all (not just below the quality gate — see
    # TrackState.consecutive_no_face), the identity decays to Unknown.
    identity_stale_after_no_face: int = 5

    # --- Tracking (ByteTrack) ---------------------------------------------------
    # Two-stage IOU association: a track must clear track_new_thresh to be born,
    # and is only matched against track_high_thresh/track_low_thresh detections
    # thereafter. A track not re-matched on a detection cycle drops out of the
    # displayed set immediately (track_buffer only keeps it alive internally, for
    # re-identification if the same object reappears) instead of continuing to
    # coast/render for the full buffer window.
    track_buffer: int = 60
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    track_new_thresh: float = 0.6
    track_match_thresh: float = 0.8
    # BYTETracker has no appearance model — it's pure IOU/motion, so when two
    # people's boxes overlap, its Hungarian match can swap which detection
    # continues which track_id. A track_id's box landing at IOU below this
    # threshold vs its own last known box (one detection cycle earlier, so
    # ~2 frames of real motion) is the signature of that: too big a jump for
    # normal movement, consistent with the detection now belonging to a
    # different physical person. FacePipelineService resets identity on it
    # rather than let a stale label ride along onto the wrong person.
    track_jump_iou_threshold: float = 0.3
    min_person_box: int = 40
    pipeline_stats_interval_sec: float = 30.0

    # --- Appearance-embedding (ReID) tracking, person tracker only -------------
    # BYTETracker is pure IOU/motion (see track_jump_iou_threshold above) — it
    # has no signal to tell "same person, ID continues" apart from "different
    # person, ID happens to land on the same track_id" during an overlap or a
    # lost-track revival. Enabling this switches the person tracker (not the
    # fire tracker, which has no identity concept) from BYTETracker to
    # BoT-SORT, which fuses IOU with appearance-embedding distance in the
    # association step itself, so bad associations happen less often instead
    # of only being detected/patched after the fact. Master rollout flag:
    # kept False until validated on real footage (see docs/PIPELINE_OPTIMIZATIONS.md
    # section 4, "ReID — The Biggest Lever", never previously implemented).
    track_reid_enabled: bool = True
    model_reid: str = "yolo26n-reid.onnx"
    # Min IoU to consider a track/detection pair for appearance comparison at
    # all, and min cosine similarity for the appearance match itself to count
    # — both are BoT-SORT upstream defaults (ultralytics cfg/trackers/botsort.yaml).
    track_proximity_thresh: float = 0.5
    track_appearance_thresh: float = 0.25

    # --- Camera management (add/remove via /api/cameras, max 4 slots) ----------
    camera_upload_max_bytes: int = 500 * 1024 * 1024
    camera_upload_allowed_ext: set[str] = {".mp4", ".mov", ".avi", ".mkv"}

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
    # Separate throttle on clip *compilation* (not the debounce above): each
    # clip job costs a fixed ~10s (ClipWriter._compile_clip sleeps to capture
    # post-alert frames) plus encode time, so a camera that keeps re-arming
    # faster than that starves the fixed-size worker pool and piles up
    # unbounded raw-frame snapshots in ClipWriter's queue. A new alert still
    # records/shows live even on cooldown; it just won't get its own clip.
    alert_clip_cooldown_sec: float = 15.0

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
    videos_uploads_dir: Path = project_root / "data" / "videos" / "uploads"

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
        self.videos_uploads_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.trt_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def onnx_dir(self) -> Path:
        return self.models_dir / "onnx"

    @property
    def trt_cache_dir(self) -> Path:
        return self.data_dir / "trt_cache"

    @property
    def facial_gallery_dir(self) -> Path:
        return self.data_dir / "facial_rec" / "gallery_built"

    @property
    def facial_gallery_path(self) -> Path:
        return self.facial_gallery_dir / self.facial_gallery_filename

    @property
    def facial_gallery_office_path(self) -> Path:
        return self.facial_gallery_dir / self.facial_gallery_office_filename

    def detector_onnx(self, key: str) -> Path:
        """Resolve the ONNX file for a detector ('person'|'fire'|'weapon'),
        preferring the int8-QDQ variant when int8_detectors is enabled."""
        if self.int8_detectors:
            int8_path = self.onnx_dir / f"{key}.int8.onnx"
            if int8_path.is_file():
                return int8_path
        return self.onnx_dir / f"{key}.onnx"

settings = Settings()
