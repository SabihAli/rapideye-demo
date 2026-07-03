"""GPU inference modules for the Vision Demo."""

from server.inference.face_pipeline import FacePipelineService, face_pipeline_service
from server.inference.face_recognition import (
    DEFAULT_YOLO_MODEL,
    FaceAnalysisApp,
    FaceGallery,
    PersonDetector,
    discover_sample_videos,
    process_videos_batched,
)

__all__ = [
    "DEFAULT_YOLO_MODEL",
    "FaceAnalysisApp",
    "FaceGallery",
    "FacePipelineService",
    "PersonDetector",
    "discover_sample_videos",
    "face_pipeline_service",
    "process_videos_batched",
]
