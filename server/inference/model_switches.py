import json
import threading
from pathlib import Path
from typing import Dict, List

from server.config import settings
from server.schemas.model_switches import CameraModelSwitches


class ModelSwitchManager:
    """
    Per-camera toggles for which inference models run on each feed.
    Persisted to disk so settings survive restarts.
    """

    def __init__(self):
        self._switches: Dict[int, CameraModelSwitches] = {}
        self._lock = threading.Lock()
        self._switches_dir = settings.data_dir / "model_switches"
        self._switches_dir.mkdir(parents=True, exist_ok=True)
        self._load_all()

    def _get_filepath(self, camera_id: int) -> Path:
        return self._switches_dir / f"camera_{camera_id}.json"

    def _default_switches(self, camera_id: int) -> CameraModelSwitches:
        presets: Dict[int, Dict[str, bool]] = {
            1: {"fire_enabled": True, "weapon_enabled": False, "face_enabled": False},
            2: {"fire_enabled": True, "weapon_enabled": False, "face_enabled": True},
            3: {"fire_enabled": False, "weapon_enabled": True, "face_enabled": False},
            4: {"fire_enabled": True, "weapon_enabled": False, "face_enabled": False},
        }
        opts = presets.get(
            camera_id,
            {"fire_enabled": False, "weapon_enabled": False, "face_enabled": False},
        )
        return CameraModelSwitches(camera_id=camera_id, **opts)

    def _load_all(self):
        for cam_id in range(1, 5):
            filepath = self._get_filepath(cam_id)
            if filepath.exists():
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        self._switches[cam_id] = CameraModelSwitches(**data)
                except Exception as e:
                    print(f"[ModelSwitches] Error loading switches for camera {cam_id}: {e}")
                    self._switches[cam_id] = self._default_switches(cam_id)
            else:
                default = self._default_switches(cam_id)
                self._switches[cam_id] = default
                self.save_switches(cam_id, default)

    def get_switches(self, camera_id: int) -> CameraModelSwitches:
        with self._lock:
            if camera_id not in self._switches:
                self._switches[camera_id] = self._default_switches(camera_id)
            return self._switches[camera_id].model_copy()

    def get_all_switches(self) -> List[CameraModelSwitches]:
        with self._lock:
            return [self._switches[cam_id].model_copy() for cam_id in range(1, 5)]

    def save_switches(self, camera_id: int, config: CameraModelSwitches):
        with self._lock:
            self._switches[camera_id] = config
            filepath = self._get_filepath(camera_id)
            try:
                with open(filepath, "w") as f:
                    json.dump(config.model_dump(), f, indent=4)
                print(f"[ModelSwitches] Saved model switches for camera {camera_id}")
            except Exception as e:
                print(f"[ModelSwitches] Failed to save switches for camera {camera_id}: {e}")


model_switch_manager = ModelSwitchManager()
