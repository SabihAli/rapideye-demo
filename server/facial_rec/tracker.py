from __future__ import annotations

from dataclasses import dataclass, field

from server.facial_rec.schemas import FaceDetection, FaceTrack


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _TrackState:
    track_id: int
    stream_id: int
    bbox: tuple[float, float, float, float]
    score: float
    age: int = 0
    missed: int = 0
    frames_since_recognition: int = 0
    embedding: list[float] | None = None


class FaceTracker:
    """Lightweight IoU tracker — ByteTrack-compatible interface for demo tests."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[tuple[int, int], _TrackState] = {}

    def update(self, stream_id: int, detections: list[FaceDetection]) -> list[FaceTrack]:
        key_prefix = stream_id
        active_keys = {k for k in self._tracks if k[0] == key_prefix}
        matched_keys: set[tuple[int, int]] = set()
        outputs: list[FaceTrack] = []

        for det in detections:
            best_key: tuple[int, int] | None = None
            best_iou = 0.0
            for key in active_keys - matched_keys:
                state = self._tracks[key]
                iou = _iou(state.bbox, det.bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_key = key
            if best_key is not None:
                state = self._tracks[best_key]
                state.bbox = det.bbox
                state.score = det.score
                state.age += 1
                state.missed = 0
                state.frames_since_recognition += 1
                matched_keys.add(best_key)
                outputs.append(self._to_face_track(state))
            else:
                track_id = self._next_id
                self._next_id += 1
                key = (key_prefix, track_id)
                state = _TrackState(
                    track_id=track_id,
                    stream_id=stream_id,
                    bbox=det.bbox,
                    score=det.score,
                    frames_since_recognition=0,
                )
                self._tracks[key] = state
                matched_keys.add(key)
                outputs.append(self._to_face_track(state))

        for key in active_keys - matched_keys:
            state = self._tracks[key]
            state.missed += 1
            state.frames_since_recognition += 1
            if state.missed <= self.max_missed:
                outputs.append(self._to_face_track(state))
            else:
                del self._tracks[key]

        return outputs

    def set_embedding(self, stream_id: int, track_id: int, embedding: list[float]) -> None:
        key = (stream_id, track_id)
        if key in self._tracks:
            self._tracks[key].embedding = embedding
            self._tracks[key].frames_since_recognition = 0

    @staticmethod
    def _to_face_track(state: _TrackState) -> FaceTrack:
        return FaceTrack(
            track_id=state.track_id,
            stream_id=state.stream_id,
            bbox=state.bbox,
            score=state.score,
            frames_since_recognition=state.frames_since_recognition,
            embedding=state.embedding,
        )
