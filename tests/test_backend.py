from fastapi.testclient import TestClient
from server.main import app
from server.config import settings

client = TestClient(app)

def test_health_endpoint():
    """Verifies that the /api/health GET endpoint returns valid system schema.

    `streams` now reflects however many cameras are currently registered via
    /api/cameras (0-4) rather than a hardcoded 4 — see CameraRegistry/
    StreamManager.get_streams_status()."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "gpu_available" in data
    assert "models_loaded" in data
    assert "streams" in data
    assert isinstance(data["streams"], list)

def test_streams_status_endpoint():
    """Verifies the /api/streams/status GET endpoint output structure."""
    response = client.get("/api/streams/status")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    for stream in data:
        assert "camera_id" in stream
        assert "is_active" in stream
        assert "current_fps" in stream
        assert "error_count" in stream

def test_zones_crud_endpoints():
    """Tests CRUD retrieval and updates of polygon zone configuration files."""
    camera_id = 1

    # 1. A camera with no zone file yet (or an explicitly cleared one) has
    # zero zones by default — no auto-created full-frame zone that would
    # alert on anything, anywhere in the picture.
    get_response = client.get(f"/api/zones/{camera_id}")
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["camera_id"] == camera_id
    assert get_data["polygon"] == []
    assert get_data["alert_classes"] == []

    # 2. Modify zone
    test_polygon = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
    put_payload = {
        "camera_id": camera_id,
        "polygon": test_polygon,
        "alert_classes": ["person", "fire"]
    }

    put_response = client.put(f"/api/zones/{camera_id}", json=put_payload)
    assert put_response.status_code == 200
    put_data = put_response.json()
    assert put_data["polygon"] == test_polygon
    assert "person" in put_data["alert_classes"]

    # 3. Retrieve modified zone to verify persistence
    get_again = client.get(f"/api/zones/{camera_id}")
    assert get_again.status_code == 200
    assert get_again.json()["polygon"] == test_polygon

    # 4. Clearing the zone (what the frontend's "Remove" button does) brings
    # the camera back to zero zones instead of reverting to a default.
    clear_response = client.put(
        f"/api/zones/{camera_id}",
        json={"camera_id": camera_id, "polygon": [], "alert_classes": []},
    )
    assert clear_response.status_code == 200
    assert clear_response.json()["polygon"] == []

def test_model_switches_endpoints():
    """Verifies per-camera inference model toggle CRUD."""
    camera_id = 1

    get_response = client.get(f"/api/cameras/{camera_id}/model-switches")
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["camera_id"] == camera_id
    assert "fire_enabled" in get_data
    assert "weapon_enabled" in get_data
    assert "face_enabled" in get_data

    put_payload = {
        "camera_id": camera_id,
        "fire_enabled": False,
        "weapon_enabled": True,
        "face_enabled": False,
    }
    put_response = client.put(
        f"/api/cameras/{camera_id}/model-switches",
        json=put_payload,
    )
    assert put_response.status_code == 200
    put_data = put_response.json()
    assert put_data["fire_enabled"] is False
    assert put_data["weapon_enabled"] is True

    get_again = client.get(f"/api/cameras/{camera_id}/model-switches")
    assert get_again.status_code == 200
    assert get_again.json()["fire_enabled"] is False

    all_response = client.get("/api/cameras/model-switches")
    assert all_response.status_code == 200
    all_data = all_response.json()
    assert len(all_data) == 4
    cam1 = next(item for item in all_data if item["camera_id"] == 1)
    assert cam1["fire_enabled"] is False

def test_demo_start_endpoint_removed():
    """The legacy demo start/reset endpoint should no longer exist."""
    response = client.post("/api/demo/start")
    assert response.status_code == 404
