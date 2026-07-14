import time
from pathlib import Path
from typing import Dict, List, Optional

from server.db import database
from server.ingest.stream_manager import stream_manager
from server.inference.zone_engine import zone_engine
from server.inference.model_switches import model_switch_manager
from server.schemas.cameras import CameraOut
from server.schemas.zones import ZoneConfig

MAX_CAMERAS = 4


class CameraLimitError(Exception):
    """Raised when adding a camera would exceed MAX_CAMERAS."""


class CameraNotFoundError(Exception):
    """Raised when referencing a camera_id that isn't registered."""


class CameraRegistry:
    """
    Owns the set of user-added cameras (persisted in the `cameras` DB table,
    see server/db/database.py) and their lifecycle: which slot (1-4) each
    occupies, and wiring that slot's decoder up via StreamManager.

    Reads/writes the DB fresh on every call rather than caching rows in
    memory — same as the existing camera_recordings storage functions in
    database.py — so the module-level `camera_registry` singleton below has
    no import-time dependency on init_db() having already run (important for
    tests, which point init_db() at an isolated DB file).

    Zone config and model-switch toggles are keyed by the same camera_id
    slots and already default lazily (ZoneEngine, ModelSwitchManager) — this
    class resets those to default and deletes their on-disk file when a
    camera is removed, so a reused slot starts clean.
    """

    def load_and_start_all(self) -> None:
        """Starts decoders for every persisted camera. Called once at app startup."""
        for row in database.list_cameras():
            print(f"[CameraRegistry] Starting camera {row['camera_id']} ({row['name']}): {row['source_value']}")
            stream_manager.add_decoder(row["camera_id"], row["source_value"])

    @staticmethod
    def _to_camera_out(row: dict, status_by_id: Dict[int, object]) -> CameraOut:
        status = status_by_id.get(row["camera_id"])
        return CameraOut(
            camera_id=row["camera_id"],
            name=row["name"],
            source_type=row["source_type"],
            source_value=row["source_value"],
            original_filename=row["original_filename"],
            created_at=row["created_at"],
            is_active=status.is_active if status else False,
            current_fps=status.current_fps if status else 0.0,
            error_count=status.error_count if status else 0,
        )

    def list_cameras(self) -> List[CameraOut]:
        status_by_id = {s.camera_id: s for s in stream_manager.get_streams_status()}
        return [self._to_camera_out(row, status_by_id) for row in database.list_cameras()]

    @staticmethod
    def _next_free_slot(existing_ids: set) -> Optional[int]:
        for slot in range(1, MAX_CAMERAS + 1):
            if slot not in existing_ids:
                return slot
        return None

    def add_camera(
        self,
        name: str,
        source_type: str,
        source_value: str,
        original_filename: Optional[str] = None,
    ) -> CameraOut:
        existing = database.list_cameras()
        if len(existing) >= MAX_CAMERAS:
            raise CameraLimitError(f"Maximum of {MAX_CAMERAS} cameras already registered.")

        camera_id = self._next_free_slot({row["camera_id"] for row in existing})
        if camera_id is None:
            raise CameraLimitError(f"Maximum of {MAX_CAMERAS} cameras already registered.")

        row = {
            "camera_id": camera_id,
            "name": name,
            "source_type": source_type,
            "source_value": source_value,
            "original_filename": original_filename,
            "created_at": time.time(),
        }
        database.insert_camera(row)

        print(f"[CameraRegistry] Added camera {camera_id} ({name}): {source_value}")
        stream_manager.add_decoder(camera_id, source_value)

        status_by_id = {s.camera_id: s for s in stream_manager.get_streams_status()}
        return self._to_camera_out(row, status_by_id)

    def remove_camera(self, camera_id: int) -> None:
        row = database.get_camera(camera_id)
        if row is None:
            raise CameraNotFoundError(f"Camera {camera_id} is not registered.")

        stream_manager.remove_decoder(camera_id)

        if row["source_type"] == "file":
            file_path = Path(row["source_value"])
            if file_path.is_file():
                file_path.unlink()

        zone_path = zone_engine._get_filepath(camera_id)
        if zone_path.exists():
            zone_path.unlink()
        zone_engine.zones[camera_id] = ZoneConfig(camera_id=camera_id, polygon=[], alert_classes=[])

        switches_path = model_switch_manager._get_filepath(camera_id)
        if switches_path.exists():
            switches_path.unlink()
        model_switch_manager._switches[camera_id] = model_switch_manager._default_switches(camera_id)

        database.delete_camera(camera_id)
        print(f"[CameraRegistry] Removed camera {camera_id} ({row['name']})")


camera_registry = CameraRegistry()
