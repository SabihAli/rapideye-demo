import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from server.config import settings
from server.ingest.stream_manager import stream_manager
from server.recording.clip_writer import clip_writer
from server.inference.pipeline import inference_pipeline
from server.inference.scheduler import fps_scheduler
from server.inference.yolo_runner import yolo_runner
from server.schemas.status import SystemHealth

from server.api.routes_zones import router as zones_router
from server.api.routes_alerts import router as alerts_router
from server.api.routes_recordings import router as recordings_router
from server.api.routes_camera_recordings import router as camera_recordings_router
from server.api.ws_streams import router as ws_router
from server.db.database import init_db, close_db
from server.recording.camera_recorder import camera_recorder

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Sequence
    print("[main] Starting RapidEye demo backend services...")
    init_db()
    # 1. Start stream ingestion decoders
    stream_manager.start_all()
    # 2. Start the recording clip writer queue
    clip_writer.start()
    # 3. Start the main inference and distribution pipeline
    inference_pipeline.start(clip_writer_ref=clip_writer)
    # 4. Start the adaptive scheduler
    fps_scheduler.start(stream_manager_ref=stream_manager)
    
    yield
    
    # Shutdown Sequence
    print("[main] Stopping RapidEye demo backend services...")
    fps_scheduler.stop()
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
app.include_router(camera_recordings_router)
app.include_router(ws_router)

@app.get("/api/health", response_model=SystemHealth, tags=["System"])
async def get_health():
    """
    Returns system status including GPU CUDA availability, loaded AI models,
    and individual statuses for the 4 camera feeds.
    """
    return SystemHealth(
        gpu_available=torch.cuda.is_available() and yolo_runner.device.type == "cuda",
        gpu_device=yolo_runner.gpu_name,
        models_loaded=yolo_runner.loaded_models_names,
        streams=stream_manager.get_streams_status()
    )

@app.get("/api/streams/status", tags=["System"])
async def get_streams_status():
    """
    Returns the real-time decoding FPS and error statuses of all 4 cameras.
    """
    return stream_manager.get_streams_status()

@app.post("/api/demo/start", tags=["System"])
async def start_demo():
    """
    Starts or restarts the security demo. Rewinds all hardcoded video assets
    to frame 0 and clears the alert event logs from memory and disk.
    """
    try:
        # Rewind decoders
        stream_manager.reset_all()
        # Stop any active manual recordings
        camera_recorder.stop_all()
        # Clear alert history
        inference_pipeline.clear_alerts()
        return {"status": "success", "message": "Demo started/reset successfully"}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
