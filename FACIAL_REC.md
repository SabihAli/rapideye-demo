# Facial Recognition Pipeline

Sub-module spec for real-time facial detection + recognition across 4 simultaneous video streams on a single RTX 3090 (24GB), tuned for workplace-distance detections (subjects not always close or front-facing to camera).

## 1. Scope

- **Target load:** 4 concurrent video streams, 15–30 FPS each, single RTX 3090.
- **Use case:** workplace/office environment, moderate camera-to-subject distance, non-cooperative capture (people not posing for the camera).
- **Deliverable stage:** demo-ready. No custom training or fine-tuning in this phase — off-the-shelf pretrained weights only.
- **VRAM constraint:** this module shares the 24GB card with the rest of the project (other models/services). Every choice below defaults to the lower-VRAM option unless it costs meaningful accuracy — this module should not assume it owns the whole GPU.

## 2. Pipeline Stages

```
Video Streams (x4)
      |
   Decode (NVDEC)
      |
   Batch / Mux
      |
   Face Detector (SCRFD)
      |
   Tracker (assigns stable ID per face across frames)
      |
   Recognition Backbone (ArcFace-trained embeddings)
      |
   Embedding Match (cosine sim vs gallery)
      |
   Output / Alert
```

### 2.1 Detection — SCRFD

- Test both **SCRFD-10GF** and **SCRFD-34GF**; benchmark on real workplace footage before locking one in.
- **Default assumption for this build: SCRFD-10GF.** It's the smaller/lighter model (fewer parameters, less compute and VRAM per inference) and SCRFD as a family is already built to redistribute compute efficiently rather than just scaling up — 10GF should cover typical workplace distances fine.
- 34GF is the stronger candidate specifically on WIDER FACE's "hard" subset (very small/tiny faces), but costs more VRAM and compute per frame. Only move up to 34GF if 10GF's small-face recall is visibly insufficient on our own footage — don't default to the bigger model preemptively.
- Decision point: run both through the full 4-stream pipeline, compare FPS, VRAM footprint, and small-face recall side by side. Given the shared-GPU constraint, 10GF wins ties.

### 2.2 Tracking — lightweight tracker between detector and recognizer

- Options: DeepStream NvDCF (if using DeepStream) or ByteTrack (if using a custom ONNXRuntime/TensorRT pipeline).
- Purpose: assign a stable ID to each detected face across consecutive frames so the recognition backbone only runs once per tracked identity (or at a reduced interval), not once per frame per face.
- This is the main lever that keeps 4-stream real-time performance comfortable on a single 3090 — without it, recognition cost scales with every face in every frame.

### 2.3 Recognition — buffalo_l (no fine-tuning)

- Use InsightFace's pretrained server-tier pack as-is:
  - **buffalo_l** (w600k_r50) — **default for this build.** Strong general-purpose accuracy, best-supported, and lighter on VRAM than antelopev2.
  - **antelopev2** (glintr100) — larger/stronger alternative, better on profile/off-angle and age-gap cases, but bigger model = more VRAM and slightly slower. Only fall back to this if buffalo_l's off-angle accuracy is visibly inadequate on real footage — not a default choice given the shared-GPU constraint.
- No AdaFace fine-tuning or custom margin training in this phase — this is a demo, not a production-hardened deployment. Standard ArcFace-trained embeddings are the practical choice for "works out of the box."
- Note for later: if recognition accuracy on distant/off-angle faces proves insufficient post-demo, AdaFace-style quality-adaptive margin training is the documented next step — flagged here for future scope, not part of current build.

### 2.4 Precision / Runtime

- **FP16 TensorRT engines** for both detector and recognizer — roughly half the VRAM footprint of FP32 with negligible accuracy loss, and the 3090's Tensor Cores handle FP16 natively. This is the baseline for this build, not just a performance nicety.
- **INT8 with calibration** is the next lever if VRAM needs to come down further once the rest of the project's GPU footprint is known — smaller weights and activations again, at some calibration effort and a small accuracy cost. Worth testing once other services on the GPU are in the picture, not required for the initial demo.
- Engine build: export ONNX -> TensorRT engine per model, cached per GPU (standard DeepStream/TensorRT workflow).
- **Shared model instance across streams:** run one detector engine and one recognizer engine, batching frames from all 4 streams through them, rather than loading 4 separate copies of each model. This is the single biggest VRAM lever available — model weights get loaded once regardless of stream count, only activation/batch memory scales with streams.
- Set explicit TensorRT workspace size limits and cap max batch size at what 4 streams actually need (4, or 4x a small per-stream sub-batch) rather than leaving it unbounded.

## 3. VRAM Budget Strategy

This module is one piece of a larger project sharing the same 24GB card, so it should stay lean rather than use whatever's available. Levers, in order of impact:

1. **One shared model instance per stage, not per stream.** Detector and recognizer weights loaded once; 4 streams batched through the same engines. Avoids 4x duplication of model weights.
2. **FP16 by default** for both detector and recognizer engines (roughly half the VRAM of FP32).
3. **Smaller model variant by default** — SCRFD-10GF over 34GF, buffalo_l over antelopev2 — unless benchmarking on real footage shows the smaller model is actually failing on accuracy, not just "slightly worse."
4. **Tracker keeps recognition calls low**, which keeps batch sizes and activation memory down, not just compute — fewer faces going through the recognizer per frame means smaller intermediate tensors too.
5. **Bounded TensorRT workspace/batch size** rather than default/unbounded settings, sized to what 4 streams actually need.
6. **INT8 held in reserve** as the next step down if the above isn't enough once the rest of the project's GPU usage is known.

Rough expectation: SCRFD-10GF + buffalo_l in FP16, batched, should sit in the low single-digit GB range for weights + working memory — leaving the large majority of the 24GB free for the rest of the project. Exact figure needs to be measured on the actual pipeline, not assumed.

## 4. Expected Performance

- All 4 streams comfortably real-time (15–30 FPS each).
- Substantial VRAM and compute headroom left on the 24GB card after running detector + tracker + recognizer across 4 streams — room to add a person detector, additional camera streams, or run recognition at full frame rate instead of relying solely on the tracker to reduce calls.
- Reference throughput (single-model, isolated, 640x480, batch=1, RTX 3090): buffalo_l ~450 FPS, buffalo_m ~900 FPS. Treat as directional only — actual end-to-end pipeline throughput (with decode, batching, tracking, 4-stream I/O) needs to be measured directly, not assumed from this figure.

## 5. Deployment Options

- **Option A — NVIDIA DeepStream + TensorRT:** standard production path for multi-stream video AI, native TensorRT integration, built-in stream muxing/batching, NvDCF tracker included. More setup overhead, but purpose-built for exactly this (many concurrent streams, single GPU).
- **Option B — Custom pipeline (InsightFace + ONNXRuntime-TensorRT + ByteTrack):** lighter to stand up for a demo, more Python-native, easier to iterate on quickly. Less mature multi-stream batching than DeepStream out of the box.
- Demo-stage recommendation: Option B for speed of iteration, with a note that Option A is the path if this module needs to scale beyond 4 streams or move toward production later.

## 6. Open Items / To Validate

- [ ] Benchmark SCRFD-10GF vs 34GF on actual workplace camera footage (FPS + small-face recall).
- [ ] Confirm tracker choice (NvDCF vs ByteTrack) based on which deployment option (A/B) is chosen.
- [ ] Measure actual end-to-end 4-stream FPS and VRAM usage — reference throughput numbers above are single-model estimates, not pipeline measurements.
- [ ] Decide buffalo_l vs antelopev2 based on accuracy on off-angle/distant test footage.
- [ ] Revisit AdaFace fine-tuning if demo-stage accuracy on distant faces is insufficient (future scope, not current).
- [ ] Measure actual VRAM footprint of SCRFD-10GF + buffalo_l (shared instances, FP16, 4-stream batching) to confirm the low-single-digit-GB estimate and set a real budget for the rest of the project.
- [ ] Only escalate to SCRFD-34GF / antelopev2 / INT8-instead-of-FP16 tradeoffs if the lighter defaults measurably fail on accuracy — not preemptively.

## 7. Summary Decisions Log

| # | Decision | Status |
|---|----------|--------|
| 1 | Test SCRFD-10GF and SCRFD-34GF | Approved — both to be benchmarked |
| 2 | Tracker between detector and recognizer | Approved |
| 3 | No fine-tuning; use pretrained buffalo_l/antelopev2 (ArcFace) | Approved — demo-ready scope |
| 4 | FP16 TensorRT engines | Approved |
| 5 | Expected 4-stream real-time performance with VRAM headroom | Approved |
| 6 | Minimize VRAM usage (shared module in bigger project): lighter model defaults, shared engine instances, bounded batch/workspace | Approved |
