from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FaceDetection:
    bbox: tuple[float, float, float, float]
    score: float
    landmarks: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class FaceTrack:
    track_id: int
    stream_id: int
    bbox: tuple[float, float, float, float]
    score: float
    frames_since_recognition: int = 0
    embedding: list[float] | None = None


@dataclass
class MatchResult:
    identity_id: str | None
    display_name: str | None
    similarity: float
    known: bool


@dataclass
class FrameResult:
    stream_id: int
    frame_index: int
    detections: list[FaceDetection]
    tracks: list[FaceTrack]
    matches: list[MatchResult]
    detect_ms: float
    recognize_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
