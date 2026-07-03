# Usage — Vision Demo

Command reference for the demo-app Python environment and the facial recognition pipeline.

For project overview and directory layout, see [README.md](README.md).

---

## Setup

### Recommended (pinned GPU stack)

Tested on Python 3.13, Linux x86_64, NVIDIA RTX 3090 (driver CUDA 13.0).

```bash
cd demo-app
python -m venv facialvenv
source facialvenv/bin/activate          # Linux/macOS
# facialvenv\Scripts\activate           # Windows

pip install -r requirements.txt
```

`requirements.txt` pins torch to the **CUDA 12** build (`+cu126`) so YOLO and InsightFace can share one GPU process without cuDNN/NCCL conflicts. Read the header comments in that file before changing versions.

### Editable install (dashboard + tests)

For the full demo package with FastAPI server and pytest:

```bash
pip install -e ".[facial,dev]"
```

### CPU-only (no GPU)

```bash
pip install -r requirements-facial-cpu.txt
```

Run inference with `--face-cpu` and `--gpu -1`.

### Headless servers (no live window)

Swap `opencv-python` for `opencv-python-headless` in `requirements.txt` if you do not need `--show`.

---

## Facial recognition CLI

Entry points (equivalent):

```bash
python -m server.inference.face_recognition <command> [options]
face-recognition <command> [options]    # after pip install -e .
```

Help:

```bash
python -m server.inference.face_recognition --help
python -m server.inference.face_recognition run --help
python -m server.inference.face_recognition build-gallery --help
```

---

## 1. Build gallery (`build-gallery`)

Enroll identities before running inference. Output defaults to `data/facial_rec/gallery_built/gallery.json`.

### From an unlabelled enrollment video (clustering)

Detects faces, clusters embeddings, auto-names identities `Person_1`, `Person_2`, … Edit `display_name` in the JSON or use QA crops under `<output>/<id>/` to assign real names.

```bash
python -m server.inference.face_recognition build-gallery \
  --video path/to/enroll.mp4 \
  --output data/facial_rec/gallery_built
```

Tune clustering:

```bash
python -m server.inference.face_recognition build-gallery \
  --video enroll.mp4 \
  --sample-every 5 \
  --min-samples 5 \
  --cluster-threshold 0.5 \
  --merge-threshold 0.6 \
  --name-prefix Person \
  --match-threshold 0.4
```

### From ChokePoint-style labelled dataset

Expects:

- `data/facial_rec/samples/<SEQUENCE>/` — frame images + `bg_img.txt`
- `data/facial_rec/gallery/<SEQUENCE>.xml` — person IDs and eye landmarks

One sequence:

```bash
python -m server.inference.face_recognition build-gallery \
  --sequence P1E_S2_C1 \
  --top-k 5 \
  --min-quality 30
```

All sequences that have matching XML:

```bash
python -m server.inference.face_recognition build-gallery
```

Enroll from one sequence only (avoid gallery leakage when probing another):

```bash
python -m server.inference.face_recognition build-gallery \
  --enroll-sequence P1E_S2_C1
```

---

## 2. Run inference (`run`)

4-camera batched pipeline: **YOLO11n** person detection → **ByteTrack** → **InsightFace buffalo_l** on person crops → cosine match against gallery.

Writes annotated MP4s to `data/facial_rec/output/annotated_<name>.mp4`.

### Auto-discover inputs

Places up to **4** `*.mp4` files under `data/facial_rec/samples/` (recursive). No `--inputs` needed.

```bash
python -m server.inference.face_recognition run \
  --gallery data/facial_rec/gallery_built/gallery.json
```

### Explicit inputs (up to 4 videos)

```bash
python -m server.inference.face_recognition run \
  --inputs cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
  --gallery data/facial_rec/gallery_built/gallery.json \
  --output-dir data/facial_rec/output
```

### Performance tuning

```bash
python -m server.inference.face_recognition run \
  --inputs cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
  --gallery data/facial_rec/gallery_built/gallery.json \
  --target-fps 20 \
  --rec-interval 5 \
  --imgsz 640 \
  --adaptive \
  --log-gpu \
  --gpu-decode
```

TensorRT YOLO engine (optional; falls back to PyTorch fp16 if export fails):

```bash
python -m server.inference.face_recognition run \
  --inputs cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
  --gallery data/facial_rec/gallery_built/gallery.json \
  --trt
```

### Detect + track only (no gallery matching)

```bash
python -m server.inference.face_recognition run \
  --inputs probe.mp4 \
  --detect-only
```

### Live preview window

Requires `opencv-python` (not headless). Press **q** to quit.

```bash
python -m server.inference.face_recognition run \
  --inputs cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
  --gallery data/facial_rec/gallery_built/gallery.json \
  --show
```

### CPU-only face recognition

Use when InsightFace cannot share the GPU with torch:

```bash
python -m server.inference.face_recognition run \
  --inputs cam1.mp4 \
  --gallery data/facial_rec/gallery_built/gallery.json \
  --face-cpu \
  --gpu -1
```

### Smoke test (few frames)

```bash
python -m server.inference.face_recognition run \
  --inputs cam1.mp4 \
  --gallery data/facial_rec/gallery_built/gallery.json \
  --max-frames 30
```

---

## Common flags

### `run`

| Flag | Default | Description |
|------|---------|-------------|
| `--inputs` | auto-discover | Up to 4 input video paths |
| `--gallery` / `-g` | `data/facial_rec/gallery_built/gallery.json` | Gallery JSON or image folder |
| `--output-dir` | `data/facial_rec/output` | Annotated MP4 output directory |
| `--yolo-model` | `data/models/yolo11n.pt` | YOLO weights (auto-downloaded if missing) |
| `--model` | `buffalo_l` | InsightFace model pack |
| `--det-size` | `640` | SCRFD input size |
| `--threshold` | `0.4` | Cosine match threshold |
| `--target-fps` | `20` | Downsample each input to this FPS |
| `--rec-interval` | `5` | Re-recognize every N frames per track |
| `--min-person-box` | `0` | Skip face-rec if person box height &lt; N px |
| `--track-buffer` | `30` | ByteTrack frames to keep lost tracks |
| `--person-conf` | `0.25` | YOLO person confidence |
| `--imgsz` | `640` | YOLO inference size |
| `--no-half` | fp16 on | Disable fp16 for YOLO |
| `--trt` | off | Build/reuse TensorRT engine for YOLO |
| `--adaptive` | off | Raise rec interval when falling behind |
| `--log-gpu` | off | Print GPU memory during run |
| `--gpu-decode` | off | Request NVDEC hardware decode |
| `--face-cpu` | off | Run InsightFace on CPU |
| `--gpu` | `0` | GPU device id (`-1` for CPU YOLO) |
| `--detect-only` | off | Tracking only, no identity labels |
| `--show` | off | Live 2×2 mosaic window |
| `--max-frames` | unlimited | Cap frames per video |

### `build-gallery`

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | — | Unlabelled enrollment video (clustering mode) |
| `--sequence` | all with XML | One ChokePoint sequence |
| `--enroll-sequence` | — | Single sequence for enrollment |
| `--output` | `data/facial_rec/gallery_built` | Gallery output directory |
| `--data-root` | `data/facial_rec` | Dataset root for XML mode |
| `--top-k` | `5` | Best frames per person to embed |
| `--min-quality` | `30` | Min quality score (XML mode) |
| `--sample-every` | `5` | Frame stride (video mode) |
| `--min-samples` | `5` | Min detections per cluster (video mode) |
| `--cluster-threshold` | `0.5` | Cosine sim to join cluster |
| `--merge-threshold` | `0.6` | Cosine sim to merge clusters |
| `--match-threshold` | `0.4` | Threshold written to gallery.json |

---

## Model weights (not in git)

| Model | Location | How it arrives |
|-------|----------|----------------|
| YOLO11n | `data/models/yolo11n.pt` | Ultralytics downloads on first `run` |
| InsightFace buffalo_l | `~/.insightface/models/` | InsightFace downloads on first use |
| YOLO TensorRT engine | `data/models/yolo11n.engine` | Built locally with `--trt` (gitignored) |

---

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not gpu"
```

GPU integration tests (when added) will be marked `@pytest.mark.gpu`.

---

## Full demo dashboard (future)

```bash
cp .env.example .env
# Set CAMERA_1_URL … CAMERA_4_URL

pip install -e ".[facial]"
cd web && npm install && cd ..
./scripts/start.sh
```

Open `http://localhost:5173` (dev) or `http://localhost:8000` (API + static build).

Facial recognition is **not** wired into the live dashboard yet; use the `run` subcommand above for pipeline validation.

---

## Further reading

- [docs/FACIAL_REC_FEASIBILITY.md](docs/FACIAL_REC_FEASIBILITY.md) — GPU budget and feasibility
- [docs/PIPELINE_OPTIMIZATIONS.md](docs/PIPELINE_OPTIMIZATIONS.md) — performance levers
- [PLAN.md](PLAN.md) — dashboard design spec
