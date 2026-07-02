# Vision Demo

Client-server security demo dashboard — **isolated from production RapidEye**.

- **Server:** camera ingest + GPU inference (FastAPI, Python)
- **Client:** local web UI for 4 live annotated camera tiles
- **Design spec:** [PLAN.md](PLAN.md)

## Separation from RapidEye

This folder is a self-contained subproject:

- Own Python virtualenv, Node dependencies, and runtime data under `data/`
- **No imports** from `vision-service/` or other RapidEye microservices
- Not deployed as part of the main platform

## Status

- **Dashboard:** initialized — implementation follows [PLAN.md](PLAN.md) phases.
- **Facial recognition:** feasibility evaluated — see [FACIAL_REC_FEASIBILITY.md](FACIAL_REC_FEASIBILITY.md). Test pipeline in `server/facial_rec/`.

## Prerequisites (target dev machine)

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04+ |
| GPU | NVIDIA RTX 3090 (CUDA 12.x) |
| Python | 3.10+ |
| Node.js | 20+ |
| System | `ffmpeg`, `yt-dlp` |

## Quick start (after implementation)

```bash
cd demo-app
cp .env.example .env
# Edit CAMERA_1_URL … CAMERA_4_URL (YouTube, RTSP, or HLS)

python -m venv .venv
source .venv/bin/activate
pip install -e .

cd web && npm install && cd ..
./scripts/start.sh
```

Open `http://localhost:5173` (dev) or `http://localhost:8000` (API + static build).

## Configuration

Copy `.env.example` to `.env`. Four independent stream URLs:

```
CAMERA_1_URL=...
CAMERA_2_URL=...
CAMERA_3_URL=...
CAMERA_4_URL=...
```

## Facial recognition test pipeline

Spec: [FACIAL_REC.md](FACIAL_REC.md) · Feasibility: [FACIAL_REC_FEASIBILITY.md](FACIAL_REC_FEASIBILITY.md)

**Standalone video test** (single file, no server imports):

```bash
pip install opencv-python-headless insightface onnxruntime-gpu numpy

# 1. Build gallery from ChokePoint-style XML annotations (your dataset layout)
python build_gallery_from_dataset.py --sequence P1E_S2_C1

# 2. Run annotated video (gallery.json or folder)
python facial_rec_video_test.py --input data/facial_rec/samples/P1E_S2_C1 --output data/facial_rec/output/annotated.mp4 --gallery data/facial_rec/gallery_built/gallery.json
```

Note: `--input` is the **frame folder** or a video file. Output is always an annotated MP4.

Draws track IDs, recognized names (or `Unknown`), similarity scores, and processing FPS on the output video.

Stack: **InsightFace buffalo_l** (SCRFD-10G + ArcFace) · IoU tracker · cosine gallery from folder.

Legacy modular code under `server/facial_rec/` is for future dashboard integration; use `facial_rec_video_test.py` for pipeline validation.

## Directory layout

```
demo-app/
  PLAN.md                    # dashboard design
  FACIAL_REC.md              # face pipeline spec
  FACIAL_REC_FEASIBILITY.md  # feasibility evaluation
  server/
    facial_rec/              # face test pipeline
  web/                       # React dashboard (to be implemented)
  scripts/
  data/
```

## License

Internal demo — not for production use.
