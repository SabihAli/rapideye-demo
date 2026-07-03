import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.config import settings

_lock = threading.Lock()
_connection: Optional[sqlite3.Connection] = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS camera_recordings (
    id TEXT PRIMARY KEY,
    camera_id INTEGER NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    duration_seconds REAL,
    file_name TEXT NOT NULL,
    file_size_bytes INTEGER,
    status TEXT NOT NULL CHECK(status IN ('recording', 'completed', 'failed')),
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_camera_recordings_camera_id
    ON camera_recordings(camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_recordings_started_at
    ON camera_recordings(started_at DESC);
"""


def get_db_path() -> Path:
    return settings.data_dir / "rapideye_demo.db"


def init_db() -> None:
    global _connection
    with _lock:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _connection = sqlite3.connect(str(get_db_path()), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.executescript(_SCHEMA)
        _connection.commit()


def close_db() -> None:
    global _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None


def _conn() -> sqlite3.Connection:
    if _connection is None:
        init_db()
    assert _connection is not None
    return _connection


def insert_recording(row: Dict[str, Any]) -> None:
    with _lock:
        conn = _conn()
        conn.execute(
            """
            INSERT INTO camera_recordings (
                id, camera_id, started_at, ended_at, duration_seconds,
                file_name, file_size_bytes, status, created_at
            ) VALUES (
                :id, :camera_id, :started_at, :ended_at, :duration_seconds,
                :file_name, :file_size_bytes, :status, :created_at
            )
            """,
            row,
        )
        conn.commit()


def update_recording(recording_id: str, updates: Dict[str, Any]) -> None:
    if not updates:
        return
    columns = ", ".join(f"{key} = :{key}" for key in updates)
    params = dict(updates)
    params["id"] = recording_id
    with _lock:
        conn = _conn()
        conn.execute(
            f"UPDATE camera_recordings SET {columns} WHERE id = :id",
            params,
        )
        conn.commit()


def get_recording(recording_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _conn()
        row = conn.execute(
            "SELECT * FROM camera_recordings WHERE id = ?",
            (recording_id,),
        ).fetchone()
        return dict(row) if row else None


def list_recordings(
    camera_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    with _lock:
        conn = _conn()
        if camera_id is None:
            rows = conn.execute(
                """
                SELECT * FROM camera_recordings
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM camera_recordings
                WHERE camera_id = ?
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?
                """,
                (camera_id, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]
