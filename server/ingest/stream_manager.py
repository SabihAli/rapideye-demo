from typing import Dict, List, Optional
from server.config import settings
from server.ingest.decoder import StreamDecoder
from server.schemas.status import StreamStatus

class StreamManager:
    """
    Manages the lifecycle of active camera stream decoders (up to 4 slots,
    IDs 1-4). Decoders are added/removed dynamically by CameraRegistry
    (server/inference/camera_registry.py), which owns persistence of which
    cameras exist; this class just starts/stops/tracks the decoder threads.
    """
    def __init__(self):
        self.decoders: Dict[int, StreamDecoder] = {}

    def add_decoder(self, camera_id: int, url: str) -> StreamDecoder:
        """Creates and starts a decoder for camera_id, replacing any existing one."""
        existing = self.decoders.pop(camera_id, None)
        if existing:
            existing.stop()
        decoder = StreamDecoder(
            camera_id=camera_id,
            url=url,
            base_fps=settings.base_fps
        )
        self.decoders[camera_id] = decoder
        decoder.start()
        return decoder

    def remove_decoder(self, camera_id: int) -> None:
        """Stops and drops the decoder for camera_id, if any."""
        decoder = self.decoders.pop(camera_id, None)
        if decoder:
            print(f"[StreamManager] Stopping camera {camera_id}...")
            decoder.stop()

    def stop_all(self):
        """Stops all running decoders."""
        for cam_id, decoder in list(self.decoders.items()):
            print(f"[StreamManager] Stopping camera {cam_id}...")
            decoder.stop()
        self.decoders.clear()

    def get_decoder(self, camera_id: int) -> Optional[StreamDecoder]:
        """Retrieves decoder instance by ID."""
        return self.decoders.get(camera_id)

    def get_streams_status(self) -> List[StreamStatus]:
        """Compiles status schemas for all currently-registered cameras."""
        status_list = []
        for cam_id in sorted(self.decoders.keys()):
            decoder = self.decoders[cam_id]
            status_list.append(StreamStatus(
                camera_id=cam_id,
                is_active=decoder.is_active,
                current_fps=round(decoder.fps_measure, 2),
                error_count=decoder.error_count
            ))
        return status_list

# Global stream manager instance
stream_manager = StreamManager()
