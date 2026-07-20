import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi import Path as PathParam

from server.config import settings
from server.inference.camera_registry import CameraLimitError, CameraNotFoundError, camera_registry
from server.schemas.cameras import CameraOut

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])

_UPLOAD_CHUNK_BYTES = 1024 * 1024


@router.get("", response_model=List[CameraOut])
async def list_cameras():
    """Returns all currently-registered cameras (up to 4) with live stream status."""
    return camera_registry.list_cameras()


@router.post("", response_model=CameraOut, status_code=201)
async def add_camera(
    name: str = Form(..., description="Display name for the camera."),
    source_type: str = Form(..., description="'url' or 'file'."),
    url: Optional[str] = Form(None, description="Stream URL, required when source_type is 'url'."),
    file: Optional[UploadFile] = File(None, description="Video file, required when source_type is 'file'."),
):
    """
    Registers a new camera and starts decoding it immediately. Rejects the
    request if 4 cameras are already registered, if the request is
    malformed, or (for file uploads) if the file's extension or size is
    outside the configured limits.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Camera name must not be empty.")

    if source_type not in ("url", "file"):
        raise HTTPException(status_code=400, detail="source_type must be 'url' or 'file'.")

    if source_type == "url":
        if file is not None:
            raise HTTPException(status_code=400, detail="Do not provide a file when source_type is 'url'.")
        if not url or not url.strip():
            raise HTTPException(status_code=400, detail="url is required when source_type is 'url'.")
        return _add_camera_or_409(name=name, source_type="url", source_value=url.strip())

    # source_type == "file"
    if url is not None:
        raise HTTPException(status_code=400, detail="Do not provide a url when source_type is 'file'.")
    if file is None:
        raise HTTPException(status_code=400, detail="file is required when source_type is 'file'.")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.camera_upload_allowed_ext:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(settings.camera_upload_allowed_ext)}.",
        )

    dest = settings.videos_uploads_dir / f"upload_{uuid.uuid4().hex}{ext}"
    await _save_upload_with_limit(file, dest, settings.camera_upload_max_bytes)
    try:
        return _add_camera_or_409(
            name=name,
            source_type="file",
            source_value=str(dest),
            original_filename=file.filename,
        )
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise


def _add_camera_or_409(**kwargs) -> CameraOut:
    try:
        return camera_registry.add_camera(**kwargs)
    except CameraLimitError as e:
        raise HTTPException(status_code=409, detail=str(e))


async def _save_upload_with_limit(file: UploadFile, dest: Path, max_bytes: int) -> None:
    """Streams the upload to disk, aborting (and deleting the partial file)
    if it exceeds max_bytes — enforced on bytes actually read, not on the
    client-supplied Content-Length header."""
    size = 0
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {max_bytes // (1024 * 1024)}MB limit.",
                )
            out.write(chunk)


@router.delete("/{camera_id}", status_code=204)
async def remove_camera(
    camera_id: int = PathParam(..., description="Camera slot index (1-4)", ge=1, le=4),
):
    """Stops the camera's stream and permanently deletes its zone config,
    model-switch config, and (for uploaded files) the stored video."""
    try:
        camera_registry.remove_camera(camera_id)
    except CameraNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
