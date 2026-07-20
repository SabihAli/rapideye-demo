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

- **Facial recognition:** feasibility evaluated — see [docs/FACIAL_REC_FEASIBILITY.md](docs/FACIAL_REC_FEASIBILITY.md). Pipeline in `server/inference/face_recognition.py`.



## Prerequisites (target dev machine)



| Component | Version |

|-----------|---------|

| OS | Ubuntu 22.04+ |

| GPU | NVIDIA RTX 3090 (CUDA 12.x) |

| Python | 3.10+ |

| Node.js | 20+ |

| System | `ffmpeg`, `yt-dlp` |



## Quick start (after implementation)

See [USAGE.md](USAGE.md) for install and facial recognition commands.

```bash
cd demo-app
cp .env.example .env
# Edit CAMERA_1_URL … CAMERA_4_URL (YouTube, RTSP, or HLS)

python -m venv facialvenv
source facialvenv/bin/activate
pip install -r requirements.txt

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



## Facial recognition pipeline

**Commands and flags:** [USAGE.md](USAGE.md)

Feasibility: [docs/FACIAL_REC_FEASIBILITY.md](docs/FACIAL_REC_FEASIBILITY.md) · Optimizations: [docs/PIPELINE_OPTIMIZATIONS.md](docs/PIPELINE_OPTIMIZATIONS.md)

Quick example:

```bash
pip install -r requirements.txt
python -m server.inference.face_recognition build-gallery --video enroll.mp4
python -m server.inference.face_recognition run --gallery data/facial_rec/gallery_built/gallery.json
```



## Directory layout



```

demo-app/

  USAGE.md                  # install + CLI command reference

  requirements.txt          # pinned GPU stack for face pipeline

  PLAN.md

  docs/

    FACIAL_REC_FEASIBILITY.md

    PIPELINE_OPTIMIZATIONS.md

  server/

    inference/

      face_recognition.py   # 4-camera face pipeline + gallery builder

  tests/

  web/

  scripts/

  data/

    models/                 # yolo11n.pt (gitignored, auto-downloaded)

    facial_rec/

      samples/              # place up to 4 *.mp4 for auto-discovery

      gallery_built/

      output/

```



## License



Internal demo — not for production use.


