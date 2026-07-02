from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaceGallery:
    """Cosine-similarity gallery for demo enrollment."""

    def __init__(self, path: Path, threshold: float = 0.4):
        self.path = path
        self.threshold = threshold
        self._entries: list[dict] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._entries = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._entries = data.get("identities", [])
        self.threshold = float(data.get("threshold", self.threshold))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"threshold": self.threshold, "identities": self._entries}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def enroll(self, identity_id: str, display_name: str, embedding: np.ndarray) -> None:
        vec = embedding.astype(np.float32)
        vec /= max(np.linalg.norm(vec), 1e-12)
        self._entries = [e for e in self._entries if e["identity_id"] != identity_id]
        self._entries.append(
            {
                "identity_id": identity_id,
                "display_name": display_name,
                "embedding": vec.tolist(),
            }
        )
        self.save()

    def match(self, embedding: np.ndarray) -> tuple[str | None, str | None, float]:
        if not self._entries:
            return None, None, 0.0
        query = embedding.astype(np.float32)
        query /= max(np.linalg.norm(query), 1e-12)
        best_id: str | None = None
        best_name: str | None = None
        best_sim = -1.0
        for entry in self._entries:
            ref = np.array(entry["embedding"], dtype=np.float32)
            sim = float(np.dot(query, ref))
            if sim > best_sim:
                best_sim = sim
                best_id = entry["identity_id"]
                best_name = entry["display_name"]
        if best_sim < self.threshold:
            return None, None, best_sim
        return best_id, best_name, best_sim
