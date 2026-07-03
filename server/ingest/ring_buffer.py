import threading
from collections import deque
from typing import List, Tuple, Any, Optional

class RingBuffer:
    """
    A thread-safe rolling ring buffer of frames and detections for pre-alert clip compilation.
    """
    def __init__(self, max_len: int = 150):
        self.max_len = max_len
        self._buffer = deque(maxlen=max_len)
        self._lock = threading.Lock()

    def append(self, timestamp: float, frame: Any, detections: Optional[List[Any]] = None):
        """Appends a new frame tuple to the ring buffer."""
        with self._lock:
            self._buffer.append((timestamp, frame, detections or []))

    def get_all(self) -> List[Tuple[float, Any, List[Any]]]:
        """Returns a snapshot copy of all items in the ring buffer."""
        with self._lock:
            return list(self._buffer)

    def clear(self):
        """Clears the ring buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
