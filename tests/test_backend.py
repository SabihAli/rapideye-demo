from fastapi.testclient import TestClient
from server.main import app
from server.config import settings

client = TestClient(app)

def test_health_endpoint():
    """Verifies that the /api/health GET endpoint returns valid system schema."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "gpu_available" in data
    assert "models_loaded" in data
    assert "streams" in data
    assert len(data["streams"]) == 4

def test_streams_status_endpoint():
    """Verifies the /api/streams/status GET endpoint output structure."""
    response = client.get("/api/streams/status")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4
    for stream in data:
        assert "camera_id" in stream
        assert "is_active" in stream
        assert "current_fps" in stream
        assert "error_count" in stream

def test_zones_crud_endpoints():
    """Tests CRUD retrieval and updates of polygon zone configuration files."""
    camera_id = 1
    
    # 1. Retrieve default zone
    get_response = client.get(f"/api/zones/{camera_id}")
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["camera_id"] == camera_id
    assert "polygon" in get_data
    assert len(get_data["polygon"]) > 0
    
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

def test_demo_start_endpoint():
    """Verifies that the /api/demo/start POST endpoint resets the demo correctly."""
    response = client.post("/api/demo/start")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "started/reset" in data["message"]
