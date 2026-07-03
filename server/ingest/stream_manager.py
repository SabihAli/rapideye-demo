from typing import Dict, List, Optional
from server.config import settings
from server.ingest.decoder import StreamDecoder
from server.schemas.status import StreamStatus

class StreamManager:
    """
    Manages the lifecycle of all 4 active camera stream decoders.
    """
    def __init__(self):
        self.decoders: Dict[int, StreamDecoder] = {}

    def start_all(self):
        """Initializes and starts decoders for camera 1 to 4."""
        for cam_id in range(1, 5):
            url = settings.get_camera_url(cam_id)
            if not url:
                print(f"[StreamManager] Camera {cam_id} has no configured URL. Skipping.")
                continue
            
            print(f"[StreamManager] Starting camera {cam_id} with URL: {url}")
            decoder = StreamDecoder(
                camera_id=cam_id,
                url=url,
                base_fps=settings.base_fps
            )
            self.decoders[cam_id] = decoder
            decoder.start()

    def stop_all(self):
        """Stops all running decoders."""
        for cam_id, decoder in list(self.decoders.items()):
            print(f"[StreamManager] Stopping camera {cam_id}...")
            decoder.stop()
        self.decoders.clear()

    def reset_all(self):
        """Resets all stream decoders back to the starting frame."""
        for cam_id, decoder in self.decoders.items():
            decoder.reset()

    def get_decoder(self, camera_id: int) -> Optional[StreamDecoder]:
        """Retrieves decoder instance by ID."""
        return self.decoders.get(camera_id)

    def get_streams_status(self) -> List[StreamStatus]:
        """Compiles status schemas for all decoders."""
        status_list = []
        for cam_id in range(1, 5):
            decoder = self.decoders.get(cam_id)
            if decoder:
                status_list.append(StreamStatus(
                    camera_id=cam_id,
                    is_active=decoder.is_active,
                    current_fps=round(decoder.fps_measure, 2),
                    error_count=decoder.error_count
                ))
            else:
                status_list.append(StreamStatus(
                    camera_id=cam_id,
                    is_active=False,
                    current_fps=0.0,
                    error_count=0
                ))
        return status_list

# Global stream manager instance
stream_manager = StreamManager()
