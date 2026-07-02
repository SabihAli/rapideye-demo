from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from server.facial_rec.schemas import FaceDetection


class FaceDetectorBackend(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]: ...


class FaceRecognizerBackend(Protocol):
    def embed(self, image_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray: ...


class InsightFaceBackend:
    """SCRFD-10G + buffalo_l via InsightFace (Option B)."""

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    ):
        from insightface.app import FaceAnalysis

        self.det_size = det_size
        self._app = FaceAnalysis(name=model_pack, providers=list(providers))
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        faces = self._app.get(image_bgr)
        out: list[FaceDetection] = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.tolist()
            kps = face.kps.tolist() if face.kps is not None else []
            landmarks = [(float(x), float(y)) for x, y in kps]
            out.append(
                FaceDetection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(face.det_score),
                    landmarks=landmarks,
                )
            )
        return out

    def embed(self, image_bgr: np.ndarray, detection: FaceDetection) -> np.ndarray:
        faces = self._app.get(image_bgr)
        if not faces:
            raise RuntimeError("No face available for embedding")
        best = max(faces, key=lambda f: float(f.det_score))
        emb = np.asarray(best.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / max(norm, 1e-12)

    def detect_and_embed(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        faces = self._app.get(image_bgr)
        results = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.tolist()
            emb = np.asarray(face.embedding, dtype=np.float32)
            emb /= max(np.linalg.norm(emb), 1e-12)
            results.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "score": float(face.det_score),
                    "embedding": emb,
                }
            )
        return results


def create_backend(
    model_pack: str,
    det_size: tuple[int, int],
    ctx_id: int,
    providers: tuple[str, ...],
) -> InsightFaceBackend:
    return InsightFaceBackend(
        model_pack=model_pack,
        det_size=det_size,
        ctx_id=ctx_id,
        providers=providers,
    )
