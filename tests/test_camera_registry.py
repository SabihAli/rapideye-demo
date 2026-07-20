import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.api.routes_cameras import router
from server.config import settings
from server.db.database import close_db, init_db
from server.ingest.stream_manager import stream_manager
from server.inference import camera_registry as camera_registry_module
from server.inference.model_switches import ModelSwitchManager
from server.inference.zone_engine import ZoneEngine

SAMPLE_URL = "assets/camera_1.mp4"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Isolated DB + uploads dir, and isolated ZoneEngine/ModelSwitchManager
    instances (same pattern as tests/test_zone_engine.py: monkeypatch the
    settings dir, then construct a fresh manager against it) swapped into
    camera_registry's module namespace so removal side-effects never touch
    real repo data under data/zones or data/model_switches."""
    init_db(tmp_path / "test_cameras.db")

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    monkeypatch.setattr(settings, "videos_uploads_dir", uploads_dir)

    zones_dir = tmp_path / "zones"
    zones_dir.mkdir()
    monkeypatch.setattr(settings, "zones_dir", zones_dir)
    monkeypatch.setattr(camera_registry_module, "zone_engine", ZoneEngine())

    switches_root = tmp_path / "switches_root"
    switches_root.mkdir()
    monkeypatch.setattr(settings, "data_dir", switches_root)
    monkeypatch.setattr(camera_registry_module, "model_switch_manager", ModelSwitchManager())

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as test_client:
        yield test_client

    stream_manager.stop_all()
    close_db()


def add_url_camera(client: TestClient, name: str = "Lobby", url: str = SAMPLE_URL):
    return client.post("/api/cameras", data={"name": name, "source_type": "url", "url": url})


def test_add_camera_via_url_assigns_slot_1(client: TestClient):
    resp = add_url_camera(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["camera_id"] == 1
    assert body["source_type"] == "url"
    assert body["source_value"] == SAMPLE_URL
    assert body["name"] == "Lobby"

    listing = client.get("/api/cameras").json()
    assert len(listing) == 1
    assert listing[0]["camera_id"] == 1


def test_add_camera_via_file_upload_saves_to_disk(client: TestClient):
    resp = client.post(
        "/api/cameras",
        data={"name": "Warehouse", "source_type": "file"},
        files={"file": ("clip.mp4", b"not a real video, just bytes", "video/mp4")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["source_type"] == "file"
    assert body["original_filename"] == "clip.mp4"

    from pathlib import Path
    saved = Path(body["source_value"])
    assert saved.exists()
    assert saved.read_bytes() == b"not a real video, just bytes"


def test_reject_unsupported_extension(client: TestClient, tmp_path):
    resp = client.post(
        "/api/cameras",
        data={"name": "Bad", "source_type": "file"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415
    assert list(settings.videos_uploads_dir.iterdir()) == []
    assert client.get("/api/cameras").json() == []


def test_reject_oversized_file(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "camera_upload_max_bytes", 10)
    resp = client.post(
        "/api/cameras",
        data={"name": "TooBig", "source_type": "file"},
        files={"file": ("clip.mp4", b"x" * 100, "video/mp4")},
    )
    assert resp.status_code == 413
    assert list(settings.videos_uploads_dir.iterdir()) == []
    assert client.get("/api/cameras").json() == []


@pytest.mark.parametrize(
    "data",
    [
        {"name": "   ", "source_type": "url", "url": SAMPLE_URL},
        {"name": "NoUrl", "source_type": "url"},
        {"name": "NoFile", "source_type": "file"},
        {"name": "Bogus", "source_type": "bogus", "url": SAMPLE_URL},
    ],
)
def test_reject_malformed_requests(client: TestClient, data):
    resp = client.post("/api/cameras", data=data)
    assert resp.status_code == 400


def test_url_and_file_both_provided_rejected(client: TestClient):
    resp = client.post(
        "/api/cameras",
        data={"name": "Both", "source_type": "url", "url": SAMPLE_URL},
        files={"file": ("clip.mp4", b"bytes", "video/mp4")},
    )
    assert resp.status_code == 400


def test_fifth_camera_rejected_with_409(client: TestClient):
    for i in range(4):
        resp = add_url_camera(client, name=f"Cam {i}")
        assert resp.status_code == 201

    resp = add_url_camera(client, name="One Too Many")
    assert resp.status_code == 409
    assert len(client.get("/api/cameras").json()) == 4


def test_remove_camera_wipes_zone_and_switch_config_and_frees_slot(client: TestClient):
    add_url_camera(client, name="Lobby")  # slot 1

    zone_engine = camera_registry_module.zone_engine
    model_switch_manager = camera_registry_module.model_switch_manager
    zone_engine.save_zone(
        1, zone_engine.zones[1].model_copy(update={"polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]})
    )
    model_switch_manager.save_switches(
        1, model_switch_manager.get_switches(1).model_copy(update={"face_enabled": True})
    )
    assert zone_engine._get_filepath(1).exists()
    assert model_switch_manager._get_filepath(1).exists()

    resp = client.delete("/api/cameras/1")
    assert resp.status_code == 204

    assert not zone_engine._get_filepath(1).exists()
    assert zone_engine.zones[1].polygon == []
    assert not model_switch_manager._get_filepath(1).exists()
    assert model_switch_manager.get_switches(1).face_enabled == (
        model_switch_manager._default_switches(1).face_enabled
    )
    assert client.get("/api/cameras").json() == []

    # Slot 1 should be reusable immediately.
    resp = add_url_camera(client, name="New Camera")
    assert resp.status_code == 201
    assert resp.json()["camera_id"] == 1


def test_remove_nonexistent_camera_404(client: TestClient):
    resp = client.delete("/api/cameras/2")
    assert resp.status_code == 404


def test_remove_file_camera_deletes_uploaded_video(client: TestClient):
    resp = client.post(
        "/api/cameras",
        data={"name": "Warehouse", "source_type": "file"},
        files={"file": ("clip.mp4", b"video bytes", "video/mp4")},
    )
    from pathlib import Path
    saved = Path(resp.json()["source_value"])
    assert saved.exists()

    client.delete(f"/api/cameras/{resp.json()['camera_id']}")
    assert not saved.exists()


def test_cameras_persist_and_restart_recreates_decoders(client: TestClient):
    add_url_camera(client, name="Cam A")
    add_url_camera(client, name="Cam B")

    stream_manager.stop_all()
    assert stream_manager.get_streams_status() == []

    camera_registry_module.camera_registry.load_and_start_all()
    statuses = stream_manager.get_streams_status()
    assert {s.camera_id for s in statuses} == {1, 2}
