from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FacialRecConfig:
    """Option B defaults from FACIAL_REC.md."""

    model_pack: str = "buffalo_l"
    det_size: tuple[int, int] = (640, 640)
    recognition_interval: int = 5
    match_threshold: float = 0.4
    gallery_path: Path = Path("data/facial_rec/gallery.json")
    ctx_id: int = 0
    providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

    @classmethod
    def from_env(cls) -> FacialRecConfig:
        import os

        det = int(os.getenv("FACE_DET_SIZE", "640"))
        return cls(
            model_pack=os.getenv("FACE_MODEL_PACK", "buffalo_l"),
            det_size=(det, det),
            recognition_interval=int(os.getenv("FACE_REC_INTERVAL", "5")),
            match_threshold=float(os.getenv("FACE_MATCH_THRESHOLD", "0.4")),
            gallery_path=Path(os.getenv("FACE_GALLERY_PATH", "data/facial_rec/gallery.json")),
            ctx_id=int(os.getenv("FACE_GPU_ID", "0")),
        )
