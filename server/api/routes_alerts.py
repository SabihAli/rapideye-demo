from fastapi import APIRouter, Query
from typing import List
from server.schemas.alerts import AlertEvent
from server.inference.pipeline import inference_pipeline

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])

@router.get("", response_model=List[AlertEvent])
async def get_alerts(
    limit: int = Query(50, description="Maximum number of alerts to return", ge=1, le=100),
    offset: int = Query(0, description="Number of alerts to skip", ge=0)
):
    """
    Retrieves the paginated list of security alert events detected by the system,
    sorted chronologically in descending order (newest first).
    """
    return inference_pipeline.get_alerts(limit=limit, offset=offset)
