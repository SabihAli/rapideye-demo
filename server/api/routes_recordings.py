from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse
from server.config import settings

router = APIRouter(prefix="/api/recordings", tags=["Recordings"])

@router.get("/{alert_id}")
async def get_recording(
    alert_id: str = Path(..., description="Unique UUID of the alert recording")
):
    """
    Streams or downloads the compiled 20-second MP4 video clip corresponding
    to the given alert ID for browser inline playback.
    """
    video_path = settings.recordings_dir / f"{alert_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Alert recording not found")
        
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"{alert_id}.mp4"
    )
