from fastapi import APIRouter, HTTPException, Path
from server.schemas.zones import ZoneConfig
from server.inference.zone_engine import zone_engine

router = APIRouter(prefix="/api/zones", tags=["Zones"])

@router.get("/{camera_id}", response_model=ZoneConfig)
async def get_zone(
    camera_id: int = Path(..., description="Camera ID index (1-4)", ge=1, le=4)
):
    """
    Retrieves the current active zone configuration (including polygon vertices 
    and alert classes) for the specified camera.
    """
    try:
        return zone_engine.get_zone(camera_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{camera_id}", response_model=ZoneConfig)
async def update_zone(
    camera_id: int = Path(..., description="Camera ID index (1-4)", ge=1, le=4),
    config: ZoneConfig = None
):
    """
    Updates or overwrites the active zone configuration (polygon coordinates and 
    alert classes) for the specified camera. Stored coordinates must be normalized.
    """
    if config is None:
        raise HTTPException(status_code=400, detail="Missing request body")
    
    if config.camera_id != camera_id:
        raise HTTPException(status_code=400, detail="Camera ID in path does not match request body")

    try:
        zone_engine.save_zone(camera_id, config)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
