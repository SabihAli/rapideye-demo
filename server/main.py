import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from server.config import settings
from server.ingest.stream_manager import stream_manager
from server.recording.clip_writer import clip_writer
from server.inference.pipeline import inference_pipeline
from server.inference.yolo_runner import yolo_runner
from server.inference.face_pipeline import face_pipeline_service
from server.schemas.status import SystemHealth

from server.api.routes_zones import router as zones_router
from server.api.routes_alerts import router as alerts_router
from server.api.routes_recordings import router as recordings_router
from server.api.routes_camera_recordings import router as camera_recordings_router
from server.api.routes_model_switches import router as model_switches_router
from server.api.routes_cameras import router as cameras_router
from server.api.ws_streams import router as ws_router
from server.db.database import init_db, close_db
from server.recording.camera_recorder import camera_recorder
from server.inference.camera_registry import camera_registry

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Sequence
    print("[main] Starting RapidEye demo backend services...")
    init_db()
    # 1. Start stream ingestion decoders for all persisted cameras
    camera_registry.load_and_start_all()
    # 2. Start the recording clip writer queue
    clip_writer.start()
    # 3. Start the main inference and distribution pipeline
    inference_pipeline.start(clip_writer_ref=clip_writer)
    
    yield
    
    # Shutdown Sequence
    print("[main] Stopping RapidEye demo backend services...")
    inference_pipeline.stop()
    clip_writer.stop()
    camera_recorder.stop_all()
    stream_manager.stop_all()
    close_db()

app = FastAPI(
    title="RapidEye Security Demo API",
    description="REST and WebSocket API for real-time video stream ingestion, GPU inference, and zone alerts.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(zones_router)
app.include_router(alerts_router)
app.include_router(recordings_router)
app.include_router(model_switches_router)
app.include_router(camera_recordings_router)
app.include_router(cameras_router)
app.include_router(ws_router)

@app.get("/api/health", response_model=SystemHealth, tags=["System"])
async def get_health():
    """
    Returns system status including GPU CUDA availability, loaded AI models,
    and individual statuses for each currently-registered camera (see
    /api/cameras to add/remove cameras; up to 4 at a time).
    """
    return SystemHealth(
        gpu_available=torch.cuda.is_available() and yolo_runner.device.type == "cuda",
        gpu_device=yolo_runner.gpu_name,
        models_loaded=yolo_runner.loaded_models_names + face_pipeline_service.loaded_model_names,
        streams=stream_manager.get_streams_status()
    )

@app.get("/api/streams/status", tags=["System"])
async def get_streams_status():
    """
    Returns the real-time decoding FPS and error statuses of each currently-registered camera.
    """
    return stream_manager.get_streams_status()


# Helper for resolving paths in both development and PyInstaller bundled environments
import sys
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller _MEIPASS bundle."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_path)

# Serve static React UI if web/dist exists (for desktop .exe bundle and production web build)
web_dist_path = get_resource_path("web/dist")
if os.path.exists(web_dist_path):
    assets_dir = os.path.join(web_dist_path, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="static-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = os.path.join(web_dist_path, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(web_dist_path, "index.html"))

