from server.config import settings
from server.inference.yolo_runner import RawDetection
from server.inference.zone_engine import ZoneEngine


def make_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "zones_dir", tmp_path)
    return ZoneEngine()


def test_camera_with_no_zone_file_has_zero_zones(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    zone = engine.get_zone(1)
    assert zone.polygon == []
    assert zone.alert_classes == []
    # And nothing gets auto-persisted to disk for an unconfigured camera.
    assert not (tmp_path / "camera_1.json").exists()


def test_check_detections_no_zone_never_alerts(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    det = RawDetection(bbox=(100, 100, 200, 200), class_name="person", confidence=0.9)
    is_alert, triggering = engine.check_detections(1, [det], img_w=640, img_h=480)
    assert is_alert is False
    assert triggering == []


def test_check_detections_person_inside_zone_triggers(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    engine.save_zone(
        1,
        engine.zones[1].model_copy(
            update={"polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "alert_classes": ["person"]}
        ),
    )
    # Person point-of-interest is bottom-center of the box (feet), well
    # inside a full-frame zone.
    det = RawDetection(bbox=(100, 100, 200, 200), class_name="person", confidence=0.9)
    is_alert, triggering = engine.check_detections(1, [det], img_w=640, img_h=480)
    assert is_alert is True
    assert triggering == [det]


def test_check_detections_outside_polygon_does_not_trigger(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    # A small zone confined to the top-left quadrant.
    engine.save_zone(
        1,
        engine.zones[1].model_copy(
            update={"polygon": [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]], "alert_classes": ["person"]}
        ),
    )
    # Bottom-right of a 640x480 frame — well outside the top-left zone.
    det = RawDetection(bbox=(500, 400, 550, 450), class_name="person", confidence=0.9)
    is_alert, triggering = engine.check_detections(1, [det], img_w=640, img_h=480)
    assert is_alert is False
    assert triggering == []


def test_check_detections_class_not_in_alert_classes_ignored(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    engine.save_zone(
        1,
        engine.zones[1].model_copy(
            update={"polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "alert_classes": ["fire"]}
        ),
    )
    det = RawDetection(bbox=(100, 100, 200, 200), class_name="person", confidence=0.9)
    is_alert, triggering = engine.check_detections(1, [det], img_w=640, img_h=480)
    assert is_alert is False
    assert triggering == []


def test_save_and_reload_zone_round_trips(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    polygon = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
    engine.save_zone(
        2,
        engine.zones[2].model_copy(update={"polygon": polygon, "alert_classes": ["weapon"]}),
    )

    # A fresh engine instance reading the same directory should see the
    # persisted zone, same as a backend restart would.
    reloaded = ZoneEngine()
    zone = reloaded.get_zone(2)
    assert zone.polygon == polygon
    assert zone.alert_classes == ["weapon"]
