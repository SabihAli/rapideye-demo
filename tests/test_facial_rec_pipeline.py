import numpy as np

from server.facial_rec.gallery import FaceGallery
from server.facial_rec.pipeline import FacialRecTestPipeline
from server.facial_rec.schemas import FaceDetection
from server.facial_rec.tracker import FaceTracker


class _MockBackend:
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        h, w = image_bgr.shape[:2]
        return [
            FaceDetection(
                bbox=(w * 0.3, h * 0.2, w * 0.7, h * 0.8),
                score=0.95,
                landmarks=[(w * 0.4, h * 0.4), (w * 0.6, h * 0.4)],
            )
        ]

    def embed(self, image_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_gallery_match_threshold(tmp_path):
    gallery = FaceGallery(tmp_path / "gallery.json", threshold=0.9)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    gallery.enroll("u1", "Alice", emb)
    identity_id, name, sim = gallery.match(emb)
    assert identity_id == "u1"
    assert name == "Alice"
    assert sim >= 0.9


def test_tracker_assigns_stable_id():
    tracker = FaceTracker()
    det = FaceDetection(bbox=(10, 10, 50, 50), score=0.9)
    t1 = tracker.update(0, [det])
    t2 = tracker.update(0, [FaceDetection(bbox=(12, 12, 52, 52), score=0.88)])
    assert len(t1) == 1
    assert len(t2) == 1
    assert t1[0].track_id == t2[0].track_id


def test_pipeline_with_mock_backend(tmp_path):
    config_gallery = tmp_path / "gallery.json"
    from server.facial_rec.config import FacialRecConfig

    cfg = FacialRecConfig(gallery_path=config_gallery, recognition_interval=1)
    mock = _MockBackend()
    pipeline = FacialRecTestPipeline(config=cfg, backend=mock)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pipeline.enroll_from_frame(frame, "u1", "Alice")
    result = pipeline.process_frame(frame, stream_id=0)
    assert len(result.detections) == 1
    assert len(result.tracks) >= 1
    assert result.matches
    assert result.matches[0].known is True
    assert result.matches[0].display_name == "Alice"


def test_multistream_independent_track_ids():
    tracker = FaceTracker()
    det = FaceDetection(bbox=(10, 10, 50, 50), score=0.9)
    s0 = tracker.update(0, [det])
    s1 = tracker.update(1, [det])
    assert s0[0].track_id != s1[0].track_id
    assert s0[0].stream_id == 0
    assert s1[0].stream_id == 1
