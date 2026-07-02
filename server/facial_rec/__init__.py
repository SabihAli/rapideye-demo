"""Facial recognition test pipeline — see FACIAL_REC.md and FACIAL_REC_FEASIBILITY.md."""

from server.facial_rec.config import FacialRecConfig
from server.facial_rec.pipeline import FacialRecTestPipeline
from server.facial_rec.schemas import FaceDetection, FaceTrack, FrameResult, MatchResult

__all__ = [
    "FacialRecConfig",
    "FacialRecTestPipeline",
    "FaceDetection",
    "FaceTrack",
    "FrameResult",
    "MatchResult",
]
