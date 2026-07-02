import cv2
import numpy as np
from typing import List, Tuple, Any
from server.inference.yolo_runner import RawDetection
from server.schemas.zones import ZoneConfig

class Annotator:
    """
    Renders overlays on BGR frames: bounding boxes, labels, zones, and stats.
    Colors are aligned with visual cues: orange for fire, red for weapons/alerts, green for safe.
    """
    @staticmethod
    def draw_overlays(
        frame: Any,
        detections: List[RawDetection],
        zone_config: ZoneConfig,
        is_alert: bool,
        current_fps: float,
        latency_ms: float,
        camera_id: int
    ) -> Any:
        # Create a copy to prevent in-place modifications on source deque frames
        annotated_frame = frame.copy()
        h, w, _ = annotated_frame.shape

        # 1. Draw Zone Polygon
        if zone_config and zone_config.polygon:
            # Map normalized coordinates to pixel coordinates
            pts = np.array(
                [[int(pt[0] * w), int(pt[1] * h)] for pt in zone_config.polygon],
                dtype=np.int32
            )
            pts = pts.reshape((-1, 1, 2))
            
            # Color: Red if active alert, Green if idle
            zone_color = (0, 0, 255) if is_alert else (0, 255, 0)
            thickness = 3 if is_alert else 1
            
            # Draw polygon lines
            cv2.polylines(annotated_frame, [pts], isClosed=True, color=zone_color, thickness=thickness, lineType=cv2.LINE_AA)
            
            # Optional overlay shade for active alert
            if is_alert:
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [pts], color=(0, 0, 80))  # dark red transparent overlay
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

        # 2. Draw Bounding Boxes and Labels
        for det in detections:
            xmin, ymin, xmax, ymax = map(int, det.bbox)
            class_name = det.class_name.lower()
            conf = det.confidence
            
            # Color coding:
            # Fire/smoke -> Orange (0, 140, 255)
            # Weapons -> Red (0, 0, 255)
            # Other/Entities -> Bright cyan/green (255, 255, 0)
            if any(w in class_name for w in ["fire", "smoke"]):
                box_color = (0, 140, 255)
            elif any(w in class_name for w in ["gun", "weapon", "knife", "pistol", "handgun", "rifle"]):
                box_color = (0, 0, 255)
            else:
                box_color = (255, 220, 0)

            # Draw rectangle
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), box_color, 2, lineType=cv2.LINE_AA)

            # Draw text label
            label = f"{class_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            # Background block for text readability
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(
                annotated_frame, 
                (xmin, ymin - text_h - 6), 
                (xmin + text_w + 10, ymin), 
                box_color, 
                -1
            )
            
            # Draw text inside block
            cv2.putText(
                annotated_frame, 
                label, 
                (xmin + 5, ymin - 4), 
                font, 
                font_scale, 
                (0, 0, 0) if box_color != (0, 0, 255) else (255, 255, 255), 
                font_thickness, 
                lineType=cv2.LINE_AA
            )

        # 3. Render Status Information (Camera ID, FPS, Latency)
        info_font = cv2.FONT_HERSHEY_SIMPLEX
        info_y = 30
        
        # Draw background bar for information
        cv2.rectangle(annotated_frame, (0, 0), (w, 40), (20, 20, 20), -1)
        
        # Text string: "CAMERA X | FPS: Y | LATENCY: Z ms"
        status_text = f"CAM {camera_id} | Target FPS: {current_fps:.1f} | GPU Latency: {latency_ms:.1f}ms"
        
        # Color red-alert warning text if zone is breached
        if is_alert:
            status_text += " | !!! ZONE BREACH !!!"
            text_color = (0, 0, 255)
        else:
            text_color = (0, 255, 0)

        cv2.putText(
            annotated_frame,
            status_text,
            (15, 25),
            info_font,
            0.6,
            text_color,
            2,
            lineType=cv2.LINE_AA
        )

        return annotated_frame
