import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from server.config import settings
from server.schemas.zones import ZoneConfig
from server.inference.yolo_runner import RawDetection

class ZoneEngine:
    """
    Manages camera polygon zones and checks if detections trigger intrusions.
    Polygons and query points are evaluated in normalized [0.0, 1.0] coordinates.
    """
    def __init__(self):
        self.zones: Dict[int, ZoneConfig] = {}
        self._load_all_zones()

    def _get_filepath(self, camera_id: int) -> Path:
        return settings.zones_dir / f"camera_{camera_id}.json"

    def _load_all_zones(self):
        """Loads existing zone configuration files from disk."""
        for cam_id in range(1, 5):
            filepath = self._get_filepath(cam_id)
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self.zones[cam_id] = ZoneConfig(**data)
                except Exception as e:
                    print(f"[ZoneEngine] Error loading zone file for camera {cam_id}: {e}")
            else:
                # No zone file yet: default to an empty zone (no polygon),
                # not persisted — a camera with no user-drawn zone should
                # have zero zones, not a full-frame zone that alerts on
                # anything, anywhere in the picture. check_detections()
                # already treats an empty polygon as "no zone configured".
                self.zones[cam_id] = ZoneConfig(camera_id=cam_id, polygon=[], alert_classes=[])

    def get_zone(self, camera_id: int) -> ZoneConfig:
        """Gets the current zone configuration for a camera."""
        if camera_id not in self.zones:
            self._load_all_zones()
        return self.zones[camera_id]

    def save_zone(self, camera_id: int, config: ZoneConfig):
        """Saves a new zone configuration for a camera both in memory and on disk."""
        self.zones[camera_id] = config
        filepath = self._get_filepath(camera_id)
        try:
            with open(filepath, 'w') as f:
                # Use model_dump in pydantic v2
                json.dump(config.model_dump(), f, indent=4)
            print(f"[ZoneEngine] Saved zone configuration for camera {camera_id}")
        except Exception as e:
            print(f"[ZoneEngine] Failed to save zone file for camera {camera_id}: {e}")

    def check_detections(
        self, 
        camera_id: int, 
        detections: List[RawDetection], 
        img_w: int, 
        img_h: int
    ) -> Tuple[bool, List[RawDetection]]:
        """
        Checks if any detections trigger a zone boundary violation.
        Returns a tuple of (is_alert, list_of_triggering_detections).
        """
        zone = self.get_zone(camera_id)
        if not zone or not zone.polygon:
            return False, []

        triggering_detections: List[RawDetection] = []
        polygon_np = np.array(zone.polygon, dtype=np.float32)

        for det in detections:
            # Check if this class is mapped to alert triggers
            # Also check lowercase variations/substrings to handle model differences
            is_alert_class = False
            for ac in zone.alert_classes:
                if ac.lower() in det.class_name.lower():
                    is_alert_class = True
                    break
            
            if not is_alert_class:
                continue

            # Calculate comparison point (normalized)
            xmin, ymin, xmax, ymax = det.bbox
            
            # For human entities, use the bottom-center (where feet touch ground)
            # For fire/weapons, use the bounding box center
            if "person" in det.class_name.lower():
                px = ((xmin + xmax) / 2.0) / img_w
                py = ymax / img_h
            else:
                px = ((xmin + xmax) / 2.0) / img_w
                py = ((ymin + ymax) / 2.0) / img_h

            # Clip normalized point coordinates
            px = max(0.0, min(px, 1.0))
            py = max(0.0, min(py, 1.0))

            # Perform point-in-polygon test
            # cv2.pointPolygonTest returns positive value if point is inside, 0 if on edge, negative if outside
            dist = cv2.pointPolygonTest(polygon_np, (px, py), False)
            if dist >= 0:
                triggering_detections.append(det)

        is_alert = len(triggering_detections) > 0
        return is_alert, triggering_detections

# Global zone engine instance
zone_engine = ZoneEngine()
