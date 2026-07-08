# Pipeline Rework Progress — 30fps × 4-camera ORT/TensorRT stack

Tracking checklist for the perf/cv-pipeline rework (see conversation plan).
Goal: ≥30 fps on all 4 feeds at native playback speed, no FPS caps.

## Done

- [x] **Environment** — repaired `.venv`: removed conflicting CPU `onnxruntime`,
      reinstalled `onnxruntime-gpu 1.24.4`, replaced TensorRT 11 wheels with
      TensorRT **10.13** (ORT 1.24 links `libnvinfer.so.10`). TRT + CUDA execution
      providers verified working on the RTX 3090.
- [x] **`server/config.py`** — new knobs: TRT/int8 toggles, dynamic person
      interval bounds, fire cadence, weapon crop padding, motion gating, NVDEC
      toggle, `base_fps=30`, `detector_onnx()` resolver (prefers int8 variant),
      `onnx_dir` / `trt_cache_dir`.
- [x] **`.env`** — fixed stale `MODEL_FIRE`/`MODEL_WEAPON` paths (were pointing at
      nonexistent files → fire/weapon models silently didn't load), added all new
      settings incl. `INSIGHTFACE_MODEL` (buffalo_l/buffalo_m), `BASE_FPS=30`.
- [x] **`scripts/export_models.py`** — exports person (yolo11n), weapon (yolov8),
      fire (yolov5) → ONNX (dynamic batch, opset 17) + `.meta.json` sidecars
      (class names, output layout) + **int8 QDQ quantization calibrated on
      `assets/camera_*.mp4` real footage**. All 6 model files generated in
      `data/models/onnx/` (fp32 + int8 for each detector).
- [x] **`server/inference/ort_models.py`** — ONNX Runtime layer used by all
      models: provider chain TensorRT EP (fp16 + int8, disk engine cache,
      explicit batch profiles 1..8) → CUDA EP → CPU; batched letterbox
      preprocess; v8 (yolo11/v8) and v5 output decoding; class-aware NMS;
      warmup; `face_providers()` for InsightFace sessions (TRT fp16, never int8).
- [x] **`server/inference/motion_detector.py`** — per-camera motion gate
      (downscaled gray EMA-background diff, ~<1 ms/frame CPU, cooldown after
      last motion). Person+fire inference will run only while gate is open.
- [x] **`server/ingest/nvdec_reader.py`** — NVDEC GPU decode via ffmpeg cuvid
      subprocess piping raw BGR; `-re` native-speed pacing for files (fixes
      slowdown/speedup), `-stream_loop -1` looping; ffprobe stream probing.
      Verified: h264_cuvid decodes camera_1.mp4 (1920×1080) on GPU.
- [x] **`server/ingest/decoder.py`** — reworked around NVDEC with automatic
      CPU OpenCV fallback; native-fps pacing; no FPS throttling; ring buffer
      resized to native rate; latest-frame slot unchanged.

## In progress

- [ ] **Confirm all 4 assets decode via NVDEC** (camera_1 verified; probe of
      2–4 pending).
- [ ] **End-to-end verification** — run backend; confirm per-camera ≥30 fps,
      stage timings, NVDEC active, TRT engines cached, int8 detectors loaded;
      exercise motion/no-motion, weapon-with/without-person, face
      identify-then-stop scenarios.

## Left

- [x] **`server/inference/yolo_runner.py` rewrite** — drop torch.hub/Ultralytics
      runtime; wrap `OrtYoloDetector` for person/fire/weapon;
      `run_person_batch(frames)`, `run_fire_batch(frames)` (full frames, batched
      across cameras), `run_weapon_batch_on_crops(...)` (padded person crops
      batched across cameras, boxes mapped back to frame coords, per-camera NMS
      dedupe of overlapping-crop hits).
- [x] **`server/inference/face_pipeline.py` refactor** — person stage owns ALL
      cameras (person always-on; `face_enabled` = recognition only,
      `weapon_enabled` = crop stage): dynamic every-n person detection
      (n ∈ [1,10] from mean track confidence), Kalman-style coasting on
      in-between frames (interpolated boxes), face rec retry-until-identified
      then stop per track, all candidate crops batched across cameras, batched
      ArcFace embedding path,
      InsightFace on TRT-fp16/CUDA providers.
- [x] **Fire interpolation** — constant every-n fire cadence with per-camera
      tracker coasting between detection frames.
- [x] **`server/inference/pipeline.py` rewire** — tick flow: motion gate →
      person batch → parallel track/interpolate → batched face rec → weapon on
      crops → fire batch/coast → zones/alerts/annotate/broadcast; person boxes
      broadcast on all cameras; extended per-stage stats (motion/person/weapon
      + achieved fps per camera).
- [x] **Retire `scheduler.py`** — remove `fps_scheduler` from `main.py` +
      `pipeline.py` (no FPS caps anywhere); model warmup at startup.
- [x] **`.env.example` + `requirements.txt`** — document new settings; pin
      `tensorrt-cu12==10.13.3.9`; note buffalo_m option.

## Requirement → status map (from task list)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Face rec on person crops only | Done |
| 2 | Identity binds to track; face rec only for unidentified tracks | Done |
| 3 | Weapon detection on padded person crops | Done |
| 4 | Weapon runs only when persons detected | Done |
| 5 | Fire + person full-frame | Done |
| 6 | Person detection every dynamic-n + tracker interpolation | Done |
| 7 | Fire every constant-n + interpolation | Done |
| 8 | onnxruntime-gpu for all models | Done |
| 9 | Batch frames across cameras for person/fire | Done |
| 10 | Multiple person crops batched into face rec | Done |
| 11 | buffalo_m option in .env | Done (`INSIGHTFACE_MODEL`) |
| 12 | NVDEC GPU decoding | Done (verified on camera_1) |
| 13 | TensorRT fp16 on all models | Done |
| 14 | int8 (detectors; face stays fp16 per agreed split) | Done (calibrated QDQ models) |
| + | Motion gating for person/fire (added during Q&A) | Done |
| + | No FPS caps, min 30 fps all feeds | Code path done; runtime verification pending |
