# Facial Recognition — Feasibility Evaluation

Evaluation against the Vision Demo context ([PLAN.md](../PLAN.md)): 4 concurrent streams, RTX 3090 24GB, shared GPU with entity/fire/weapon YOLO models and zone logic.

**Verdict: Feasible for demo** using **Option B** (InsightFace + ONNX Runtime GPU + ByteTrack), with staged integration and measured budgets. Not feasible to run all models at full per-frame rate on every stream without batching and tracker gating.

---

## 1. Stack recommendation (test pipeline)

| Stage | Spec target | Test pipeline choice | Rationale |
|-------|-------------|----------------------|-----------|
| Deploy path | Option B for demo | **Option B** | Faster iteration, Python-native, fits FastAPI server; DeepStream deferred |
| Person detector | Light YOLO | **YOLO11n** (`data/models/yolo11n.pt`) | Batched person detection across 4 streams |
| Tracker | ByteTrack or NvDCF | **ByteTrack** | Per-camera track IDs; gates recognition calls |
| Face + embed | SCRFD + ArcFace | **InsightFace `buffalo_l`** | SCRFD-10G + w600k_r50 embeddings |
| Runtime | FP16 TensorRT | **ONNX Runtime CUDA** for face; optional YOLO TensorRT (`--trt`) | ORT ships faster for demo |
| Gallery | cosine similarity | **In-memory + JSON file** | Demo scope; no vector DB |

Implementation: `server/inference/face_recognition.py` (CLI: `run`, `build-gallery`).

---

## 2. Shared GPU budget (24GB RTX 3090)

Estimated VRAM when co-resident with main demo pipeline:

| Component | FP16 / shared instance | Notes |
|-----------|------------------------|-------|
| YOLO entity (yolo11m) | ~1.5–2.5 GB | One instance, batched 4 streams |
| YOLO fire + weapon | ~1–2 GB combined | Smaller custom weights |
| YOLO11n person (face pipeline) | ~0.3–0.5 GB | Shared with face module |
| SCRFD-10G + buffalo_l | ~0.8–1.5 GB | Single shared InsightFace pack |
| Activations (4×1080p batch) | ~2–4 GB | Depends on batch size and det_size |
| Decode / frame buffers | ~1–2 GB | NVDEC reduces CPU copy pressure |
| **Total (rough)** | **~6–12 GB** | Leaves headroom on 3090 if batching bounded |

**Conclusion:** VRAM feasible with 10G + buffalo_l, shared instances, FP16, tracker-gated recognition.

---

## 3. Throughput feasibility (4 × 15–30 FPS)

| Factor | Impact |
|--------|--------|
| Tracker gating | Recognition every N frames (default 5) cuts embed cost 5–15× |
| Batched YOLO | Mux frames from 4 streams → one detector forward pass |
| Threaded decode/encode | Overlaps CPU I/O with GPU compute |
| Adaptive throttle (`--adaptive`) | Raises rec interval under load |

**Conclusion:** 15 FPS/tile annotated demo is **achievable** with throttling; 30 FPS/tile on all 4 with full YOLO suite + face at full rate is **unlikely** without prioritization.

---

## 4. Integration with main demo (PLAN.md)

- Face pipeline will run as an **optional module** invoked from `server/inference/pipeline.py` when enabled in `.env` (future pass).
- Alerts: `face.recognized`, `face.unknown` alongside zone/fire/weapon alerts.

---

## 5. Gaps and validation checklist

| Item | Status | Action |
|------|--------|--------|
| End-to-end 4-stream FPS + VRAM | Open | `python -m server.inference.face_recognition run` on 3090 |
| SCRFD-10G vs 34G on workplace footage | Open | Manual benchmark |
| Co-run with entity/fire/weapon YOLO | Open | Joint benchmark after main pipeline exists |
| TensorRT FP16 for YOLO | Partial | `--trt` flag in face_recognition CLI |

---

## 6. Run the pipeline

```bash
cd demo-app
pip install -e ".[facial]"

# Build gallery from unlabelled enrollment video
python -m server.inference.face_recognition build-gallery --video enroll.mp4

# Run 4-camera inference (auto-discovers *.mp4 under data/facial_rec/samples/)
python -m server.inference.face_recognition run \
  --gallery data/facial_rec/gallery_built/gallery.json
```

InsightFace `buffalo_l` downloads automatically to `~/.insightface/models/` on first run. YOLO11n downloads to `data/models/yolo11n.pt` on first run if missing.

See also: [PIPELINE_OPTIMIZATIONS.md](PIPELINE_OPTIMIZATIONS.md).
