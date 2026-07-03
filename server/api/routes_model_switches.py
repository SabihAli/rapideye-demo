from typing import List

from fastapi import APIRouter, HTTPException, Path

from server.inference.model_switches import model_switch_manager
from server.schemas.model_switches import CameraModelSwitches

router = APIRouter(prefix="/api/cameras", tags=["Model Switches"])


@router.get("/model-switches", response_model=List[CameraModelSwitches])
async def get_all_model_switches():
    """Returns per-camera inference model toggles for all configured cameras."""
    try:
        return model_switch_manager.get_all_switches()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}/model-switches", response_model=CameraModelSwitches)
async def get_model_switches(
    camera_id: int = Path(..., description="Camera ID index (1-4)", ge=1, le=4),
):
    """Returns inference model toggles for a single camera."""
    try:
        return model_switch_manager.get_switches(camera_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{camera_id}/model-switches", response_model=CameraModelSwitches)
async def update_model_switches(
    camera_id: int = Path(..., description="Camera ID index (1-4)", ge=1, le=4),
    config: CameraModelSwitches = None,
):
    """
    Updates which models run for a camera. Disabled models are skipped during
    inference to reduce GPU load.
    """
    if config is None:
        raise HTTPException(status_code=400, detail="Missing request body")

    if config.camera_id != camera_id:
        raise HTTPException(
            status_code=400,
            detail="Camera ID in path does not match request body",
        )

    try:
        model_switch_manager.save_switches(camera_id, config)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
