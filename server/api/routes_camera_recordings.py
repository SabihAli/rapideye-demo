from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse

from server.recording.camera_recorder import camera_recorder
from server.schemas.camera_recordings import CameraRecording, CameraRecordingStatus

router = APIRouter(tags=["Camera Recordings"])


@router.post(
    "/api/cameras/{camera_id}/recordings/start",
    response_model=CameraRecording,
)
async def start_camera_recording(
    camera_id: int = Path(..., ge=1, le=4, description="Camera ID (1-4)"),
):
    """Start saving live frames from the given camera to disk."""
    try:
        return camera_recorder.start_recording(camera_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/api/cameras/{camera_id}/recordings/stop",
    response_model=CameraRecording,
)
async def stop_camera_recording(
    camera_id: int = Path(..., ge=1, le=4, description="Camera ID (1-4)"),
):
    """Stop the active recording for the given camera and finalize the MP4."""
    try:
        return camera_recorder.stop_recording(camera_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/api/cameras/{camera_id}/recordings/status",
    response_model=CameraRecordingStatus,
)
async def get_camera_recording_status(
    camera_id: int = Path(..., ge=1, le=4, description="Camera ID (1-4)"),
):
    """Return whether the camera is currently recording."""
    recording = camera_recorder.get_status(camera_id)
    return CameraRecordingStatus(
        camera_id=camera_id,
        is_recording=recording is not None,
        recording=recording,
    )


@router.get(
    "/api/cameras/{camera_id}/recordings",
    response_model=List[CameraRecording],
)
async def list_camera_recordings(
    camera_id: int = Path(..., ge=1, le=4, description="Camera ID (1-4)"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List saved recordings for one camera."""
    return camera_recorder.list_recordings(camera_id=camera_id, limit=limit, offset=offset)


@router.get("/api/camera-recordings", response_model=List[CameraRecording])
async def list_all_camera_recordings(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List saved recordings across all cameras."""
    return camera_recorder.list_recordings(camera_id=None, limit=limit, offset=offset)


@router.get("/api/camera-recordings/{recording_id}")
async def get_camera_recording_file(
    recording_id: str = Path(..., description="Recording UUID"),
):
    """Stream or download a saved camera recording MP4."""
    recording = camera_recorder.get_recording(recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    if recording.status == "recording":
        raise HTTPException(status_code=409, detail="Recording is still in progress")

    if recording.status == "failed":
        raise HTTPException(status_code=404, detail="Recording file is unavailable")

    video_path = camera_recorder.get_recording_file_path(recording_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Recording file not found on disk")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"camera_{recording.camera_id}_{recording_id}.mp4",
        # Default is "attachment", which forces a download instead of
        # letting the browser's <video> element play it inline.
        content_disposition_type="inline",
    )
