"""
Per-camera motion gating for the inference pipeline.

Person and fire detection only run while a camera's motion gate is open.
The gate compares a downscaled grayscale frame against an exponential
moving-average background; if the fraction of changed pixels exceeds
``motion_area_ratio`` the gate opens, and it stays open for
``motion_cooldown_sec`` after the last motion so tracks can settle/close.

This costs well under 1 ms per frame on CPU — cheap enough to run on every
frame of all 4 cameras, and it removes entire detector batches in the common
no-activity case.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import cv2
import numpy as np

from server.config import settings


class MotionGate:
    def __init__(self) -> None:
        self._background: Dict[int, np.ndarray] = {}
        self._last_motion: Dict[int, float] = {}
        self.last_ratio: Dict[int, float] = {}

    def update(self, camera_id: int, frame_bgr: np.ndarray) -> bool:
        """Feed the latest frame; returns True if the gate is open (run inference)."""
        if not settings.motion_enabled:
            return True

        now = time.time()
        small = self._downscale_gray(frame_bgr)
        background = self._background.get(camera_id)

        if background is None or background.shape != small.shape:
            # First frame (or resolution change): treat as motion so the
            # pipeline starts detecting immediately while the EMA settles.
            self._background[camera_id] = small.astype(np.float32)
            self._last_motion[camera_id] = now
            self.last_ratio[camera_id] = 1.0
            return True

        delta = cv2.absdiff(small, background.astype(np.uint8))
        changed = delta >= settings.motion_pixel_delta
        ratio = float(np.count_nonzero(changed)) / changed.size
        self.last_ratio[camera_id] = ratio

        # EMA background update; changed regions blend slower so a person
        # standing still isn't absorbed instantly.
        cv2.accumulateWeighted(small.astype(np.float32), background, 0.05)

        if ratio >= settings.motion_area_ratio:
            self._last_motion[camera_id] = now
            return True
        return (now - self._last_motion.get(camera_id, 0.0)) < settings.motion_cooldown_sec

    def reset(self, camera_id: Optional[int] = None) -> None:
        if camera_id is None:
            self._background.clear()
            self._last_motion.clear()
            self.last_ratio.clear()
        else:
            self._background.pop(camera_id, None)
            self._last_motion.pop(camera_id, None)
            self.last_ratio.pop(camera_id, None)

    @staticmethod
    def _downscale_gray(frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        target_w = settings.motion_downscale_width
        target_h = max(1, round(h * target_w / max(w, 1)))
        small = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)


motion_gate = MotionGate()
