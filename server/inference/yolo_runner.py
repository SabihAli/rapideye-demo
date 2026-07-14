from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from server.config import settings
from server.inference.ort_models import OrtYoloDetector, nms_xyxy


@dataclass
class RawDetection:
    """
    Internal representation of an object detection containing pixel coordinates.
    Optional ``track_id`` / ``identity`` / ``similarity`` are set for person tracks
    with facial recognition labels.
    """

    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    identity: Optional[str] = None
    similarity: Optional[float] = None

    def to_normalized(self, img_w: int, img_h: int) -> Dict[str, Any]:
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


@dataclass
class PersonCropRequest:
    camera_id: int
    frame: Any
    bbox: Tuple[int, int, int, int]
    track_id: Optional[int] = None


class YoloRunner:
    """
    ORT-backed detector runner used by the live pipeline.

    Person and fire run on full frames; weapon runs only on padded person crops.
    """

    def __init__(self) -> None:
        self.device = self._resolve_device()
        self.gpu_name = f"cuda:{settings.cuda_device}" if self.device.type == "cuda" else None

        self.person_detector: Optional[OrtYoloDetector] = None
        self.fire_detector: Optional[OrtYoloDetector] = None
        self.weapon_detector: Optional[OrtYoloDetector] = None
        self.loaded_models_names: List[str] = []
        self._load_models()

    @staticmethod
    def _resolve_device() -> SimpleNamespace:
        return SimpleNamespace(
            type="cuda" if settings.use_cuda else "cpu",
            index=settings.cuda_device if settings.use_cuda else None,
        )

    def _load_models(self) -> None:
        try:
            self.person_detector = self._load_detector_with_fallback(
                key="person",
                conf=settings.person_conf,
                imgsz=settings.person_imgsz,
                keep_classes=[0],
            )
            self.loaded_models_names.append(f"Person ORT ({self.person_detector.active_provider})")
        except Exception as exc:
            print(f"[YoloRunner] Failed to load person detector: {exc}")

        try:
            self.fire_detector = self._load_detector_with_fallback(
                key="fire",
                conf=settings.conf_fire,
                imgsz=settings.fire_imgsz,
            )
            self.loaded_models_names.append(f"Fire ORT ({self.fire_detector.active_provider})")
        except Exception as exc:
            print(f"[YoloRunner] Failed to load fire detector: {exc}")

        try:
            self.weapon_detector = self._load_detector_with_fallback(
                key="weapon",
                conf=settings.conf_weapon,
                imgsz=settings.weapon_imgsz,
            )
            self.loaded_models_names.append(f"Weapon ORT ({self.weapon_detector.active_provider})")
        except Exception as exc:
            print(f"[YoloRunner] Failed to load weapon detector: {exc}")

    def _load_detector_with_fallback(
        self,
        *,
        key: str,
        conf: float,
        imgsz: int,
        keep_classes: Optional[List[int]] = None,
    ) -> OrtYoloDetector:
        preferred = settings.detector_onnx(key)
        detector = OrtYoloDetector(
            preferred,
            name=key,
            conf=conf,
            imgsz=imgsz,
            keep_classes=keep_classes,
        )
        detector.warmup()
        fallback_reason = self._needs_fp32_fallback(detector, key, preferred)
        if fallback_reason is None:
            return detector

        fp_path = settings.onnx_dir / f"{key}.onnx"
        if preferred != fp_path and fp_path.is_file():
            print(f"[YoloRunner] {key}: {fallback_reason}, falling back to {fp_path.name}")
            detector = OrtYoloDetector(
                fp_path,
                name=f"{key}_fp32",
                conf=conf,
                imgsz=imgsz,
                keep_classes=keep_classes,
            )
            detector.warmup()
            return detector
        return detector

    def _needs_fp32_fallback(self, detector: OrtYoloDetector, key: str, preferred_path: Path) -> Optional[str]:
        """Returns a reason string if `detector` should be swapped for the fp32
        model, else None.

        ORT's CUDA EP has no real int8 tensor-core kernels for QDQ graphs — it
        just executes the Quantize/DequantizeLinear nodes as literal ops
        around fp32 compute, which is pure overhead with no speed benefit
        (often *slower* than fp32 outright, as measured: person int8-on-CUDA
        ran at ~980ms/batch vs a normal few tens of ms). Int8 only pays off
        on TensorRT, so if the int8 model didn't land on TensorRT, always
        fall back — regardless of what the sample-frame check below finds.
        This is checked before (and independently of) the sample-frame check,
        which only catches wrong/broken outputs, not "technically works but
        silently much slower".
        """
        if preferred_path.name.endswith(".int8.onnx") and detector.active_provider != "TensorrtExecutionProvider":
            return f"int8 model didn't land on TensorRT ({detector.active_provider})"

        if key == "weapon":
            return None
        sample = self._sample_frame_for_key(key)
        if sample is None:
            return None
        try:
            dets = detector.infer([sample])[0]
            return None if len(dets) > 0 else "int8 sanity check found no detections"
        except Exception as exc:
            return f"sanity check raised ({exc})"

    @staticmethod
    def _sample_frame_for_key(key: str) -> Optional[np.ndarray]:
        candidates = [
            settings.assets_dir / "camera_2.mp4",
            settings.assets_dir / "camera_4.mp4",
            settings.assets_dir / "camera_3.mp4",
            settings.assets_dir / "camera_1.mp4",
        ]
        if key == "fire":
            candidates = [
                settings.assets_dir / "camera_4.mp4",
                settings.assets_dir / "camera_1.mp4",
                settings.assets_dir / "camera_2.mp4",
                settings.assets_dir / "camera_3.mp4",
            ]
        for video_path in candidates:
            if not video_path.is_file():
                continue
            cap = cv2.VideoCapture(str(video_path))
            for frame_idx in (150, 60, 30, 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    cap.release()
                    return frame
            cap.release()
        return None

    def run_person_batch(
        self,
        frames: List[Any],
        person_mask: Optional[List[bool]] = None,
    ) -> List[List[RawDetection]]:
        return self._run_full_frame_batch(frames, person_mask, self.person_detector)

    def run_fire_batch(
        self,
        frames: List[Any],
        fire_mask: Optional[List[bool]] = None,
    ) -> List[List[RawDetection]]:
        return self._run_full_frame_batch(frames, fire_mask, self.fire_detector)

    def _run_full_frame_batch(
        self,
        frames: List[Any],
        mask: Optional[List[bool]],
        detector: Optional[OrtYoloDetector],
    ) -> List[List[RawDetection]]:
        count = len(frames)
        if count == 0:
            return []
        active_mask = mask if mask is not None else [True] * count
        outputs: List[List[RawDetection]] = [[] for _ in range(count)]
        if detector is None or not any(active_mask):
            return outputs

        active_idx = [i for i, (enabled, frame) in enumerate(zip(active_mask, frames)) if enabled and frame is not None]
        if not active_idx:
            return outputs

        arrays = detector.infer([frames[i] for i in active_idx])
        for local_i, frame_i in enumerate(active_idx):
            outputs[frame_i] = self._decode_array(arrays[local_i], detector)
        return outputs

    def run_weapon_batch_on_crops(
        self,
        requests: List[PersonCropRequest],
    ) -> Dict[int, List[RawDetection]]:
        grouped: Dict[int, List[RawDetection]] = {}
        if not requests or self.weapon_detector is None:
            return grouped

        crops: List[np.ndarray] = []
        metas: List[Tuple[int, Tuple[int, int, int, int]]] = []

        for req in requests:
            crop_box = self._padded_bbox(req.frame, req.bbox)
            if crop_box is None:
                continue
            x1, y1, x2, y2 = crop_box
            crop = req.frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            metas.append((req.camera_id, crop_box))

        if not crops:
            return grouped

        arrays = self.weapon_detector.infer(crops)
        for dets, (camera_id, crop_box) in zip(arrays, metas):
            x_off, y_off, _, _ = crop_box
            mapped: List[RawDetection] = []
            for row in dets:
                x1, y1, x2, y2, conf, cls_id = row.tolist()
                mapped.append(
                    RawDetection(
                        bbox=(x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off),
                        class_name=self.weapon_detector.class_name(int(cls_id)),
                        confidence=float(conf),
                    )
                )
            if mapped:
                grouped.setdefault(camera_id, []).extend(mapped)

        return {camera_id: self._dedupe(group) for camera_id, group in grouped.items()}

    @staticmethod
    def _decode_array(dets: np.ndarray, detector: OrtYoloDetector) -> List[RawDetection]:
        out: List[RawDetection] = []
        for row in dets:
            x1, y1, x2, y2, conf, cls_id = row.tolist()
            out.append(
                RawDetection(
                    bbox=(x1, y1, x2, y2),
                    class_name=detector.class_name(int(cls_id)),
                    confidence=float(conf),
                )
            )
        return out

    @staticmethod
    def _padded_bbox(
        frame: Any,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * settings.weapon_crop_padding)
        pad_y = int((y2 - y1) * settings.weapon_crop_padding)
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)
        if (px2 - px1) < settings.weapon_min_crop_px or (py2 - py1) < settings.weapon_min_crop_px:
            return None
        return (px1, py1, px2, py2)

    @staticmethod
    def _dedupe(dets: List[RawDetection]) -> List[RawDetection]:
        if len(dets) <= 1:
            return dets
        boxes = np.asarray([det.bbox for det in dets], dtype=np.float32)
        scores = np.asarray([det.confidence for det in dets], dtype=np.float32)
        cls_ids = np.asarray([hash(det.class_name) % 1000 for det in dets], dtype=np.float32)
        keep = nms_xyxy(boxes, scores, cls_ids, 0.5)
        return [dets[int(i)] for i in keep.tolist()]

    def run_inference(
        self,
        frame: Any,
        *,
        fire_enabled: bool = True,
        weapon_enabled: bool = False,
    ) -> List[RawDetection]:
        fire = self.run_fire_batch([frame], fire_mask=[fire_enabled])[0]
        person = self.run_person_batch([frame], person_mask=[True])[0]
        detections = self.merge_detection_lists(fire, person)
        if weapon_enabled and person:
            weapon_hits = self.run_weapon_batch_on_crops(
                [PersonCropRequest(camera_id=1, frame=frame, bbox=tuple(map(int, det.bbox))) for det in person]
            )
            detections.extend(weapon_hits.get(1, []))
        return detections

    @staticmethod
    def merge_detection_lists(*parts: List[RawDetection]) -> List[RawDetection]:
        merged: List[RawDetection] = []
        for part in parts:
            merged.extend(part)
        return merged


yolo_runner = YoloRunner()
