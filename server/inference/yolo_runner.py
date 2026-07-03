import torch
import cv2
from typing import List, Dict, Any, Tuple
from pathlib import Path
from ultralytics import YOLO
from server.config import settings

class RawDetection:
    """
    Internal representation of an object detection containing pixel coordinates.
    """
    def __init__(self, bbox: Tuple[float, float, float, float], class_name: str, confidence: float):
        self.bbox = bbox  # (xmin, ymin, xmax, ymax) in pixel coordinates
        self.class_name = class_name
        self.confidence = confidence

    def to_normalized(self, img_w: int, img_h: int) -> Dict[str, Any]:
        """Normalizes the bounding box coords to [0.0, 1.0] range."""
        xmin, ymin, xmax, ymax = self.bbox
        return {
            "bbox": [
                max(0.0, min(float(xmin) / img_w, 1.0)),
                max(0.0, min(float(ymin) / img_h, 1.0)),
                max(0.0, min(float(xmax) / img_w, 1.0)),
                max(0.0, min(float(ymax) / img_h, 1.0)),
            ],
            "class_name": self.class_name,
            "confidence": float(self.confidence)
        }

class YoloRunner:
    """
    Manages loading and running fire/smoke (YOLOv5) and weapon (YOLOv8) models on GPU.
    Entity (COCO) detection is optional and disabled by default.
    """
    def __init__(self):
        self.device = self._resolve_device()
        self.inference_device = self._inference_device_arg()
        self.gpu_name = self._gpu_name()
        
        self.model_entity = None
        self.model_fire = None
        self.model_weapon = None
        self.loaded_models_names = []
        
        self._load_models()

    def _resolve_device(self) -> torch.device:
        if settings.use_cuda:
            if not torch.cuda.is_available():
                print("[YoloRunner] WARNING: USE_CUDA=true but CUDA is not available.")
                print("[YoloRunner] Install GPU PyTorch, e.g.:")
                print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
                return torch.device("cpu")
            idx = settings.cuda_device
            torch.cuda.set_device(idx)
            name = torch.cuda.get_device_name(idx)
            print(f"[YoloRunner] Using CUDA device {idx}: {name}")
            return torch.device(f"cuda:{idx}")

        print("[YoloRunner] Using CPU (USE_CUDA=false)")
        return torch.device("cpu")

    def _inference_device_arg(self):
        """Ultralytics accepts int GPU index or 'cpu'."""
        if self.device.type == "cuda":
            return self.device.index if self.device.index is not None else 0
        return "cpu"

    def _gpu_name(self) -> str | None:
        if self.device.type != "cuda":
            return None
        idx = self.device.index if self.device.index is not None else 0
        return torch.cuda.get_device_name(idx)

    def _load_models(self):
        # 1. Entity model (optional — disabled by default)
        if settings.enable_entity_detection:
            try:
                entity_path = settings.models_dir / settings.model_entity
                if not entity_path.parent.exists():
                    entity_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"[YoloRunner] Loading entity model from: {entity_path}")
                self.model_entity = YOLO(str(entity_path))
                self.model_entity.to(self.device)
                self.loaded_models_names.append("Entity (YOLO11)")
            except Exception as e:
                print(f"[YoloRunner] ERROR loading Entity Model: {e}")
        else:
            print("[YoloRunner] Entity detection disabled (ENABLE_ENTITY_DETECTION=false)")

        # 2. Load Fire/Smoke Model (YOLOv5 custom)
        try:
            fire_path = Path(settings.model_fire)
            if not fire_path.is_absolute():
                fire_path = settings.project_root / fire_path

            if fire_path.exists():
                print(f"[YoloRunner] Loading fire/smoke model from: {fire_path}")
                self.model_fire = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=str(fire_path),
                    trust_repo=True
                ).to(self.device)
                self.model_fire.conf = settings.conf_fire
                if self.device.type == "cuda":
                    self.model_fire.eval()
                self.loaded_models_names.append("Fire/Smoke (YOLOv5)")
            else:
                print(f"[YoloRunner] WARNING: Fire/Smoke model not found at: {fire_path}")
        except Exception as e:
            print(f"[YoloRunner] ERROR loading Fire/Smoke Model: {e}")

        # 3. Load Weapons Model (YOLOv8 custom)
        try:
            weapon_path = Path(settings.model_weapon)
            if not weapon_path.is_absolute():
                weapon_path = settings.project_root / weapon_path

            if weapon_path.exists():
                print(f"[YoloRunner] Loading weapons model from: {weapon_path}")
                self.model_weapon = YOLO(str(weapon_path))
                self.model_weapon.to(self.device)
                self.loaded_models_names.append("Weapons (YOLOv8)")
            else:
                print(f"[YoloRunner] WARNING: Weapons model not found at: {weapon_path}")
        except Exception as e:
            print(f"[YoloRunner] ERROR loading Weapons Model: {e}")

    def run_inference(self, frame: Any) -> List[RawDetection]:
        """
        Runs fire/smoke and weapon models on the provided BGR frame.
        Entity detection runs only when ENABLE_ENTITY_DETECTION=true.
        """
        detections: List[RawDetection] = []
        if frame is None:
            return detections

        if settings.enable_entity_detection and self.model_entity:
            try:
                results = self.model_entity(
                    frame,
                    conf=settings.conf_entity,
                    device=self.inference_device,
                    verbose=False,
                )
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    names = self.model_entity.names
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        class_name = names.get(cls_id, "entity")
                        detections.append(RawDetection(
                            bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                            class_name=class_name,
                            confidence=conf
                        ))
            except Exception as e:
                print(f"[YoloRunner] Entity model inference error: {e}")

        if self.model_fire:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.inference_mode():
                    results = self.model_fire(frame_rgb)

                if hasattr(results, "xyxy") and len(results.xyxy) > 0:
                    xyxy_tensor = results.xyxy[0]
                    names = self.model_fire.names
                    for det in xyxy_tensor:
                        xmin, ymin, xmax, ymax, conf, cls_id = det.tolist()
                        class_name = names[int(cls_id)] if int(cls_id) < len(names) else "fire/smoke"
                        detections.append(RawDetection(
                            bbox=(xmin, ymin, xmax, ymax),
                            class_name=class_name,
                            confidence=conf
                        ))
            except Exception as e:
                print(f"[YoloRunner] Fire/Smoke model inference error: {e}")

        if self.model_weapon:
            try:
                results = self.model_weapon(
                    frame,
                    conf=settings.conf_weapon,
                    device=self.inference_device,
                    verbose=False,
                )
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    names = self.model_weapon.names
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        class_name = names.get(cls_id, "weapon")
                        detections.append(RawDetection(
                            bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                            class_name=class_name,
                            confidence=conf
                        ))
            except Exception as e:
                print(f"[YoloRunner] Weapons model inference error: {e}")

        return detections

# Global runner instance
yolo_runner = YoloRunner()
