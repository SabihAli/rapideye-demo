# Pipeline Optimizations — 4-Camera Real-Time CV Stack
YOLO Detection + ByteTrack + ReID + InsightFace Face Recognition

Implemented in `server/inference/face_recognition.py` (sections 1–3, 5 partial, 6 partial, 8 partial). ReID (section 4) is not implemented — face recognition on person crops replaces it for identity.

---

## 1. Ingestion & Decode

- Use NVDEC (GPU hardware decode) for all 4 RTSP streams. Do not decode on CPU and copy to GPU.
- Use `nvcr.io`-style GStreamer/DeepStream or PyAV with `hwaccel=cuda` for decode pipelines.
- Decode directly into GPU memory buffers (avoid CPU roundtrip: decode → host → device copy).
- Set camera streams to constant bitrate, capped resolution (e.g. 1280x720) at the source if the camera supports it — don't decode 4K and downscale on GPU every frame.
- Use a frame-skipping/latest-frame-only queue per camera (drop stale frames) rather than a FIFO buffer, to prevent latency drift under load.
- Batch decode across cameras into a single CUDA stream where the decode SDK supports multi-stream batching.

---

## 2. Detection (YOLO)

- Use YOLOv8n/v11n or YOLOv8s only. Do not use m/l/x variants for 4-camera real-time on a single consumer GPU.
- Export to TensorRT (fp16 minimum, int8 with calibration if accuracy budget allows). Do not run raw PyTorch eager mode in production.
- Batch all 4 camera frames into a single inference call (batch=4) instead of 4 sequential calls.
- Fix input resolution at 640x640 or 512x512. Lower resolution trades detection recall on small/distant persons for latency — tune per camera mount height.
- Use NMS on GPU (TensorRT EfficientNMS plugin or torchvision GPU NMS), not CPU NMS.
- Restrict detection classes to `person` only (and `face` if using a joint detector) — skip full COCO class output.
- Cache and reuse the TensorRT engine; do not rebuild per session.
- Consider INT8 quantization with a calibration set from your actual camera footage (not generic COCO calibration) — accuracy holds better this way.

---

## 3. Tracking (ByteTrack)

- ByteTrack is CPU-only — do not attempt to GPU-accelerate the Kalman filter/Hungarian matching, it's not the bottleneck.
- Run ByteTrack per-camera in parallel threads/processes (not sequential) since it's CPU-bound and cameras are independent until cross-camera fusion.
- Tune `track_thresh`, `match_thresh`, and `track_buffer` per camera based on actual frame rate achieved (not assumed) — mismatched buffer length vs. real FPS causes premature track death.
- Keep tracking running every frame even when ReID/face recognition are throttled (see below) — motion continuity is what lets you skip expensive re-identification most frames.

---

## 4. ReID — The Biggest Lever

- **Do not run ReID embedding extraction on every detection in every frame.** This is the single highest-impact optimization available.
  - Run ReID on: track initialization, then every N frames (5–10) per track, plus on track re-acquisition after occlusion.
  - Between checks, rely on ByteTrack's motion association for identity continuity.
- Use OSNet (x0.25 or x0.5) over TransReID/CLIP-ReID/SOLIDER for the real-time tier. Transformer-backbone ReID models are too slow for per-camera real-time budgets on consumer GPUs.
- Batch all pending ReID crops across all 4 cameras into one inference call rather than looping per-detection or per-camera.
- Export ReID model to TensorRT fp16.
- Resize crops to the model's native input (typically 128x256 or 256x128) using GPU-side resize, not CPU/PIL resize.
- Cache embeddings per track ID in GPU memory (or pinned host memory) rather than recomputing on every comparison.
- Use cosine similarity via batched matrix multiply (GPU) for gallery matching, not a Python loop over gallery entries.

---

## 5. Face Recognition (InsightFace)

- Use MobileFaceNet or EdgeFace instead of ArcFace r100 for the real-time tier. ArcFace r100 is accuracy-optimal but too heavy for every-frame use across 4 streams.
- If accuracy requirements force ArcFace, use r50 and throttle frequency (see below), not r100 at full frequency.
- Use RetinaFace mobile variant for face detection, not the ResNet50 backbone version, unless face detection accuracy is failing at your camera distances.
- **Throttle face recognition the same way as ReID**: run on track init and periodically (e.g. every 10–15 frames), not every frame. Face recognition is typically your most expensive per-identity operation — it should run least often.
- Only run face detection/recognition on person crops that ByteTrack has already localized — do not run full-frame face detection independently of the person pipeline.
- Export RetinaFace and the recognition backbone to TensorRT fp16.
- Batch face crops across all 4 cameras into one recognition inference call.
- Skip face recognition entirely for tracks below a minimum bounding-box size (faces too small/far to be reliably matched anyway) — saves compute on low-value inference.

---

## 6. Cross-Component / Architecture

- Run detection, ReID, and face recognition in a single process sharing one CUDA context. Separate processes each carry their own CUDA context overhead (~300–500MB and added latency per call) — this compounds across 3 models.
- Use a single shared TensorRT execution context per model where possible, reused across all 4 camera batches, rather than one context per camera.
- Pipeline stages asynchronously: decode, detect, track, ReID/face should run as a producer-consumer pipeline with queues between stages, not a single blocking sequential loop per frame. This lets slow stages (ReID/face) run on their throttled schedule without blocking detection+tracking, which must run every frame.
- Use CUDA streams to overlap detection inference on camera batch N+1 with ReID/face postprocessing on batch N.
- Pin host memory for any CPU↔GPU transfers (crop extraction, embedding retrieval) to reduce copy latency.

---

## 7. Precision & Quantization

- fp16 across all three models is close to mandatory for the real-time tier; establish this as baseline, not an optional tuning step.
- int8 is worth pursuing for detection first (largest FPS gain, most mature tooling), then ReID, then face recognition last (accuracy is most sensitive here since it drives identity matching correctness).
- Always calibrate int8 on real camera footage, not stock datasets — domain shift from generic calibration sets measurably hurts accuracy on surveillance camera angles/lighting.

---

## 8. Monitoring & Adaptive Behavior

- Track actual achieved FPS per camera and per pipeline stage in production, not assumed FPS — ByteTrack parameters and throttle intervals (N frames) should be tuned against measured values.
- Implement adaptive throttling: if system falls behind (queue depth growing), automatically increase the ReID/face-recognition throttle interval before dropping frames entirely.
- Log GPU memory headroom continuously — gallery embedding cache growth (as more identities are enrolled) is unbounded unless you cap or evict, and can silently erode your inference memory budget over time.
- Set a hard frame-drop policy (drop oldest, keep newest) per camera queue so one slow camera doesn't stall the batch for the other three.

---

## 9. Priority Order (Highest Impact First)

1. Throttle ReID and face recognition to every-N-frames instead of every-frame.
2. Convert all three models to TensorRT fp16.
3. Batch across 4 cameras at every stage (detection, ReID, face).
4. Switch to lightweight model variants (YOLOn/s, OSNet x0.25/x0.5, MobileFaceNet).
5. GPU decode (NVDEC) to remove CPU bottleneck at ingestion.
6. Single shared CUDA context / single process for all three models.
7. int8 quantization (detection first) once fp16 + throttling headroom is confirmed insufficient.
