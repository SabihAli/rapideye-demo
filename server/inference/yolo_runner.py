import torch
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from ultralytics import YOLO
from server.config import settings

class RawDetection:
    """
    Internal representation of an object detection containing pixel coordinates.
  Optional ``track_id`` / ``identity`` / ``similarity`` are set for person tracks
    with facial recognition labels.
    """
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        class_name: str,
        confidence: float,
        *,
        track_id: Optional[int] = None,
        identity: Optional[str] = None,
        similarity: Optional[float] = None,
    ):
        self.bbox = bbox  # (xmin, ymin, xmax, ymax) in pixel coordinates
        self.class_name = class_name
        self.confidence = confidence
        self.track_id = track_id
        self.identity = identity
        self.similarity = similarity

    def to_normalized(self, img_w: int, img_h: int) -> Dict[str, Any]:
        """Normalizes the bounding box coords to [0.0, 1.0] range."""
        xmin, ymin, xmax, ymax = self.bbox
        payload: Dict[str, Any] = {
            "bbox": [
                max(0.0, min(float(xmin) / img_w, 1.0)),
                max(0.0, min(float(ymin) / img_h, 1.0)),
                max(0.0, min(float(xmax) / img_w, 1.0)),
                max(0.0, min(float(ymax) / img_h, 1.0)),
            ],
            "class_name": self.class_name,
            "confidence": float(self.confidence),
        }
        if self.track_id is not None:
            payload["track_id"] = self.track_id
        if self.identity is not None:
            payload["identity"] = self.identity
            payload["is_unknown"] = self.identity == "Unknown"
        if self.similarity is not None:
            payload["similarity"] = float(self.similarity)
        return payload

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

    def run_inference(
        self,
        frame: Any,
        *,
        fire_enabled: bool = True,
        weapon_enabled: bool = True,
    ) -> List[RawDetection]:
        """
        Runs enabled models on a single BGR frame.
        Entity detection requires ENABLE_ENTITY_DETECTION=true at startup.
        Facial recognition is handled separately by face_pipeline_service.
        """
        fire = self.run_fire_batch([frame], fire_mask=[fire_enabled])[0]
        weapon = self.run_weapon_batch([frame], weapon_mask=[weapon_enabled])[0]
        entity = self.run_entity_batch(
            [frame],
            entity_mask=[settings.enable_entity_detection],
        )[0]
        return fire + weapon + entity

    def run_fire_batch(
        self,
        frames: List[Any],
        fire_mask: List[bool] | None = None,
    ) -> List[List[RawDetection]]:
        """Batch fire/smoke inference. Skips frames where mask entry is False."""
        n = len(frames)
        if n == 0:
            return []
        mask = fire_mask if fire_mask is not None else [True] * n
        results: List[List[RawDetection]] = [[] for _ in range(n)]

        if not self.model_fire or not any(mask):
            return results

        active_idx = [i for i in range(n) if mask[i] and frames[i] is not None]
        if not active_idx:
            return results

        try:
            active_frames = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in active_idx]
            with torch.inference_mode():
                batch_results = self.model_fire(active_frames)

            # YOLOv5 hub returns one Results object; xyxy[i] holds detections per batch image.
            for local_i, frame_i in enumerate(active_idx):
                results[frame_i] = self._parse_yolov5_result(batch_results, image_index=local_i)
        except Exception as e:
            print(f"[YoloRunner] Fire/Smoke batch inference error: {e}")

        return results

    def run_weapon_batch(
        self,
        frames: List[Any],
        weapon_mask: List[bool] | None = None,
    ) -> List[List[RawDetection]]:
        """Batch weapon inference via Ultralytics YOLO."""
        n = len(frames)
        if n == 0:
            return []
        mask = weapon_mask if weapon_mask is not None else [True] * n
        results: List[List[RawDetection]] = [[] for _ in range(n)]

        if not self.model_weapon or not any(mask):
            return results

        active_idx = [i for i in range(n) if mask[i] and frames[i] is not None]
        if not active_idx:
            return results

        try:
            active_frames = [frames[i] for i in active_idx]
            with torch.inference_mode():
                yolo_results = self.model_weapon(
                    active_frames,
                    conf=settings.conf_weapon,
                    imgsz=settings.weapon_imgsz,
                    half=settings.weapon_half and self.device.type == "cuda",
                    device=self.inference_device,
                    verbose=False,
                )
            if not isinstance(yolo_results, (list, tuple)):
                yolo_results = [yolo_results]

            for local_i, frame_i in enumerate(active_idx):
                result = yolo_results[local_i]
                results[frame_i] = self._parse_ultralytics_boxes(result)
        except Exception as e:
            print(f"[YoloRunner] Weapons batch inference error: {e}")

        return results

    def run_entity_batch(
        self,
        frames: List[Any],
        entity_mask: List[bool] | None = None,
    ) -> List[List[RawDetection]]:
        """Batch COCO entity inference (optional, env-gated)."""
        n = len(frames)
        if n == 0:
            return []
        mask = entity_mask if entity_mask is not None else [True] * n
        results: List[List[RawDetection]] = [[] for _ in range(n)]

        if not settings.enable_entity_detection or not self.model_entity or not any(mask):
            return results

        active_idx = [i for i in range(n) if mask[i] and frames[i] is not None]
        if not active_idx:
            return results

        try:
            active_frames = [frames[i] for i in active_idx]
            yolo_results = self.model_entity(
                active_frames,
                conf=settings.conf_entity,
                device=self.inference_device,
                verbose=False,
            )
            if not isinstance(yolo_results, (list, tuple)):
                yolo_results = [yolo_results]

            for local_i, frame_i in enumerate(active_idx):
                result = yolo_results[local_i]
                parsed = self._parse_ultralytics_boxes(result)
                results[frame_i] = [
                    det for det in parsed if det.class_name.lower() == "person"
                ]
        except Exception as e:
            print(f"[YoloRunner] Entity batch inference error: {e}")

        return results

    def _parse_yolov5_result(self, results: Any, image_index: int = 0) -> List[RawDetection]:
        detections: List[RawDetection] = []
        if not hasattr(results, "xyxy") or len(results.xyxy) <= image_index:
            return detections

        names = self.model_fire.names
        for det in results.xyxy[image_index]:
            xmin, ymin, xmax, ymax, conf, cls_id = det.tolist()
            class_name = names[int(cls_id)] if int(cls_id) < len(names) else "fire/smoke"
            class_key = class_name.lower()
            if not any(token in class_key for token in ("fire", "smoke", "flame")):
                continue
            detections.append(
                RawDetection(
                    bbox=(xmin, ymin, xmax, ymax),
                    class_name=class_name,
                    confidence=conf,
                )
            )
        return detections

    def _parse_ultralytics_boxes(self, result: Any) -> List[RawDetection]:
        detections: List[RawDetection] = []
        if result is None or not hasattr(result, "boxes") or result.boxes is None:
            return detections

        names = result.names
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, "object")
            detections.append(
                RawDetection(
                    bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    class_name=class_name,
                    confidence=conf,
                )
            )
        return detections

    @staticmethod
    def merge_detection_lists(*parts: List[RawDetection]) -> List[RawDetection]:
        merged: List[RawDetection] = []
        for part in parts:
            merged.extend(part)
        return merged

# Global runner instance
yolo_runner = YoloRunner()
