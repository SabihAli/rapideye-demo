import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.api.routes_camera_recordings import router
from server.db.database import close_db, init_db
from server.recording.camera_recorder import camera_recorder


@pytest.fixture
def client(tmp_path):
    """Lightweight client — no stream decoders or inference pipeline.

    Uses an isolated DB file under pytest's tmp_path rather than the real
    production database (init_db() with no args would resolve to the live
    data/rapideye_demo.db and pollute it with test rows on every run)."""
    init_db(tmp_path / "test_recordings.db")
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as test_client:
        yield test_client
    camera_recorder.stop_all()
    close_db()


def test_camera_recording_lifecycle(client: TestClient):
    """Start, verify status, stop, and list a per-camera recording."""
    camera_id = 1

    start = client.post(f"/api/cameras/{camera_id}/recordings/start")
    assert start.status_code == 200
    started = start.json()
    assert started["camera_id"] == camera_id
    assert started["status"] == "recording"
    assert started["playback_url"].startswith("/api/camera-recordings/")

    duplicate = client.post(f"/api/cameras/{camera_id}/recordings/start")
    assert duplicate.status_code == 409

    status = client.get(f"/api/cameras/{camera_id}/recordings/status")
    assert status.status_code == 200
    status_data = status.json()
    assert status_data["is_recording"] is True
    assert status_data["recording"]["id"] == started["id"]

    stop = client.post(f"/api/cameras/{camera_id}/recordings/stop")
    assert stop.status_code == 200
    stopped = stop.json()
    assert stopped["id"] == started["id"]
    assert stopped["status"] in {"completed", "failed"}

    listing = client.get(f"/api/cameras/{camera_id}/recordings")
    assert listing.status_code == 200
    items = listing.json()
    assert any(item["id"] == started["id"] for item in items)

    all_recordings = client.get("/api/camera-recordings")
    assert all_recordings.status_code == 200
    assert any(item["id"] == started["id"] for item in all_recordings.json())

    if stopped["status"] == "completed":
        playback = client.get(stopped["playback_url"])
        assert playback.status_code == 200
        assert playback.headers["content-type"].startswith("video/mp4")

    stop_again = client.post(f"/api/cameras/{camera_id}/recordings/stop")
    assert stop_again.status_code == 409
