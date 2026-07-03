# How to Run RapidEye Demo

Step-by-step guide to run the full stack: **FastAPI backend** (GPU inference + WebSockets) and **React frontend** (live dashboard).

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Node.js | 20+ |
| npm | 9+ |
| NVIDIA GPU + driver | Required for GPU inference (e.g. RTX 3090 / 4070) |
| Optional | `make` (GNU Make on Windows, or use manual commands below) |

Place video files in `assets/`:

```
assets/camera_1.mp4
assets/camera_2.mp4
assets/camera_3.mp4
assets/camera_4.mp4
```

Place custom model weights in `data/models/` (fire + weapon — see step 3).

---

## 1. Open the project

```bash
cd rapideye-demo
```

---

## 2. Configure `.env`

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Key settings in `.env`:

```env
# Camera sources (local MP4 by default)
CAMERA_1_URL=assets/camera_1.mp4
CAMERA_2_URL=assets/camera_2.mp4
CAMERA_3_URL=assets/camera_3.mp4
CAMERA_4_URL=assets/camera_4.mp4

# Model paths (relative to project root)
MODEL_FIRE=data/models/fire_yolo.pt
MODEL_WEAPON=data/models/weapon_yolo.pt

# Inference toggles
ENABLE_ENTITY_DETECTION=false   # entity/COCO detection OFF (fire + weapon only)
USE_CUDA=true                   # run models on GPU
CUDA_DEVICE=0                   # GPU index (usually 0)

BASE_FPS=15
API_HOST=0.0.0.0
API_PORT=8000
```

**Current default:** only **fire/smoke** and **weapon** detection run. Entity (person/vehicle/COCO) detection is disabled to reduce load and avoid downloading large YOLO11 weights.

To re-enable entity detection later, set `ENABLE_ENTITY_DETECTION=true` and ensure `MODEL_ENTITY` weights exist.

---

## 3. Model weights

The backend expects these files (create `data/models/` if missing):

| File | Purpose |
|------|---------|
| `data/models/fire_yolo.pt` | Fire/smoke YOLOv5 weights |
| `data/models/weapon_yolo.pt` | Weapon YOLOv8 weights |

Entity weights (`yolo11m.pt` etc.) are **not needed** while `ENABLE_ENTITY_DETECTION=false`.

---

## 4. Python virtual environment + backend deps

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[dev]"
```

**Or:**

```bash
make install
```

---

## 5. Install CUDA PyTorch (GPU required)

The default `pip install` may pull a **CPU-only** PyTorch build (`2.x.x+cpu`). For GPU inference you need the CUDA build.

With venv activated:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Expected output should show a `+cu124` (or similar) version and `CUDA: True` with your GPU name.

If `CUDA: False` and version ends with `+cpu`, repeat the force-reinstall step above.

---

## 6. Install frontend dependencies

```bash
cd web
npm install
cd ..
```

**Or:**

```bash
make install-web
```

---

## 7. Start the backend (Terminal 1)

With venv activated:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Without activating venv:

```bash
# Linux/macOS
.venv/bin/python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

# Windows
.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

**Or:**

```bash
make backend
```

On startup you should see logs like:

```
[YoloRunner] Entity detection disabled (ENABLE_ENTITY_DETECTION=false)
[YoloRunner] Using CUDA device 0: NVIDIA GeForce ...
[YoloRunner] Loading fire/smoke model from: ...
[YoloRunner] Loading weapons model from: ...
```

Verify: [http://localhost:8000/api/health](http://localhost:8000/api/health)

Health response should include:

- `gpu_available: true`
- `gpu_device: "NVIDIA ..."`
- `models_loaded`: `["Fire/Smoke (YOLOv5)", "Weapons (YOLOv8)"]` (no Entity model)

---

## 8. Start the frontend (Terminal 2)

```bash
cd web
npm run dev
```

**Or from project root:**

```bash
make frontend
```

Open [http://localhost:5173](http://localhost:5173)

The Vite dev server proxies `/api` and `/ws` to the backend on port 8000.

---

## 9. Use the dashboard

1. **Live feeds** — 2×2 grid via WebSocket `/ws/streams/{1-4}`; frames are pre-annotated on the backend.
2. **Start / Reset Demo** — calls `POST /api/demo/start` (rewinds videos, clears alerts).
3. **Alerts panel** — polls `GET /api/alerts`; clip links play recorded MP4s.
4. **Zone Management** — draw one polygon per camera; saved via `PUT /api/zones/{camera_id}`.

Video tiles use **letterboxing** (`object-contain`) so the full frame is visible without cropping.

---

## Quick reference (Make)

```bash
make env          # create .env from example
make install      # venv + pip install backend
make install-web  # npm install in web/
make backend      # run API on :8000
make frontend     # run Vite on :5173
make dev          # print two-terminal instructions
make test         # run pytest
```

**Linux/macOS — both servers in one terminal:**

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Backend unreachable` in UI | Start backend on port 8000 first |
| Black tiles / "Waiting for stream" | Wait for models to load; check `/api/health` — all 4 streams should be `is_active: true` |
| `CUDA: False` / CPU-only PyTorch | Run step 5 (`--force-reinstall` with `cu124` index URL) |
| `gpu_available: false` in health | Same as above; confirm `USE_CUDA=true` in `.env` |
| YOLO11 / entity model downloading | Set `ENABLE_ENTITY_DETECTION=false` in `.env` and restart backend |
| Fire/weapon model warnings | Add weights to `data/models/` and match paths in `.env` |
| Videos look cropped/zoomed | Restart frontend after pull — tiles use `object-contain` |
| WebSocket fails | Ensure backend is up; or set `VITE_WS_BASE_URL=ws://localhost:8000` in `web/.env.local` |
| Missing videos | Place `camera_1.mp4` … `camera_4.mp4` in `assets/` or update `.env` URLs |

---

## Production-like build (optional)

```bash
cd web && npm run build
```

Serve `web/dist` via FastAPI static files or any static host. Set `VITE_API_BASE_URL` / `VITE_WS_BASE_URL` at build time if API is on a different origin.
