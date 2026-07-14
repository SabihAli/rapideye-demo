"""
ONNX Runtime GPU inference layer for all detection models.

Every model (person / fire / weapon YOLOs, and InsightFace via its own
sessions) runs through onnxruntime-gpu with the provider chain:

    TensorRT EP (fp16; int8 for QDQ-quantized detectors, engine cached on disk)
      -> CUDA EP
        -> CPU EP

Models are exported/quantized by ``scripts/export_models.py`` into
``data/models/onnx/``. See docs/PIPELINE_OPTIMIZATIONS.md sections 2, 7.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from server.config import settings

_TRT_AVAILABLE: bool | None = None


def tensorrt_available() -> bool:
    """Import the tensorrt wheel so libnvinfer.so.10 is loaded into the process;
    ORT's TensorRT EP resolves its libnvinfer dependency from the already-loaded
    library (the pip wheels are not on the system linker path)."""
    global _TRT_AVAILABLE
    if _TRT_AVAILABLE is None:
        try:
            import tensorrt  # noqa: F401

            _TRT_AVAILABLE = True
        except Exception as exc:
            print(f"[OrtModels] TensorRT unavailable ({exc}); using CUDA EP.")
            _TRT_AVAILABLE = False
    return _TRT_AVAILABLE


def build_providers(
    cache_name: str,
    input_name: str = "images",
    imgsz: int = 640,
    max_batch: int | None = None,
    int8: bool = False,
    with_trt: bool = True,
) -> list:
    """Provider chain for an InferenceSession, TRT first when enabled."""
    providers: list = []
    if settings.use_cuda:
        if with_trt and settings.trt_enable and tensorrt_available():
            cache_dir = settings.trt_cache_dir / cache_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            max_b = max_batch or settings.ort_max_batch
            shape = f"3x{imgsz}x{imgsz}"
            trt_options: Dict[str, Any] = {
                "device_id": settings.cuda_device,
                "trt_fp16_enable": settings.trt_fp16,
                "trt_int8_enable": int8,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir),
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": str(cache_dir),
                "trt_max_workspace_size": 2 << 30,
                # Explicit optimization profile so batch 1..max_b reuses one engine
                # instead of rebuilding per new batch size.
                "trt_profile_min_shapes": f"{input_name}:1x{shape}",
                "trt_profile_opt_shapes": f"{input_name}:{min(4, max_b)}x{shape}",
                "trt_profile_max_shapes": f"{input_name}:{max_b}x{shape}",
            }
            providers.append(("TensorrtExecutionProvider", trt_options))
        providers.append(("CUDAExecutionProvider", {"device_id": settings.cuda_device}))
    providers.append("CPUExecutionProvider")
    return providers


def face_providers() -> list:
    """Provider chain for InsightFace sessions (fp16 TRT, never int8).
    InsightFace resizes to a fixed det_size and ArcFace input is 112x112, but
    input names/shapes differ per model, so no explicit TRT profiles here —
    dynamic-shape engines are cached per shape instead."""
    providers: list = []
    if settings.use_cuda:
        if settings.trt_enable and tensorrt_available():
            cache_dir = settings.trt_cache_dir / "insightface"
            cache_dir.mkdir(parents=True, exist_ok=True)
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": settings.cuda_device,
                        "trt_fp16_enable": settings.trt_fp16,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache_dir),
                        "trt_timing_cache_enable": True,
                        "trt_timing_cache_path": str(cache_dir),
                        "trt_max_workspace_size": 2 << 30,
                    },
                )
            )
        providers.append(("CUDAExecutionProvider", {"device_id": settings.cuda_device}))
    providers.append("CPUExecutionProvider")
    return providers


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize with unchanged aspect ratio and gray padding (YOLO-style).

    Returns (padded_rgb_image, gain, pad_x, pad_y)."""
    h, w = img.shape[:2]
    gain = min(size / h, size / w)
    new_w, new_h = round(w * gain), round(h * gain)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, gain, left, top


class OrtYoloDetector:
    """Batched YOLO detector on ONNX Runtime.

    Handles both output layouts:
      - ``v8``: (B, 4+nc, N) — YOLOv8/YOLO11, xywh + per-class scores
      - ``v5``: (B, N, 5+nc) — YOLOv5, xywh + objectness + per-class scores

    ``infer(images)`` accepts arbitrary-size BGR images (full frames or crops)
    and returns per-image float32 arrays of shape (n, 6):
    ``[x1, y1, x2, y2, conf, cls_id]`` in the input image's pixel coordinates.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        name: str,
        conf: float,
        imgsz: int = 640,
        iou: float = 0.45,
        keep_classes: Sequence[int] | None = None,
        max_batch: int | None = None,
    ):
        import onnxruntime as ort

        self.path = Path(onnx_path)
        self.name = name
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.keep_classes = set(keep_classes) if keep_classes else None
        self.max_batch = max_batch or settings.ort_max_batch

        meta = self._load_sidecar(self.path)
        self.layout: str = meta.get("layout", "v8")
        self.names: Dict[int, str] = {int(k): v for k, v in meta.get("names", {}).items()}

        is_int8 = ".int8." in self.path.name
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        input_name = meta.get("input_name", "images")
        providers = build_providers(
            cache_name=f"{name}_{'int8' if is_int8 else 'fp'}",
            input_name=input_name,
            imgsz=imgsz,
            max_batch=self.max_batch,
            int8=is_int8,
        )
        try:
            self.session = ort.InferenceSession(
                str(self.path), sess_options, providers=providers
            )
        except Exception as exc:
            print(f"[OrtModels] {name}: TRT session failed ({exc}); retrying CUDA/CPU.")
            self.session = ort.InferenceSession(
                str(self.path),
                sess_options,
                providers=build_providers(cache_name=name, with_trt=False),
            )
        self.input_name = self.session.get_inputs()[0].name
        self.active_provider = self.session.get_providers()[0]
        print(
            f"[OrtModels] {name}: {self.path.name} on {self.active_provider} "
            f"(imgsz={imgsz}, layout={self.layout}, int8={is_int8})"
        )

    @staticmethod
    def _load_sidecar(onnx_path: Path) -> dict:
        """Class names/layout metadata written by scripts/export_models.py."""
        base = onnx_path.name.replace(".int8.onnx", "").replace(".onnx", "")
        sidecar = onnx_path.parent / f"{base}.meta.json"
        if sidecar.is_file():
            return json.loads(sidecar.read_text(encoding="utf-8"))
        print(f"[OrtModels] WARNING: no sidecar {sidecar.name}; assuming v8 layout.")
        return {}

    def warmup(self) -> None:
        """Build/load TRT engines up-front so the first real tick isn't slow."""
        for b in (1, min(4, self.max_batch)):
            blob = np.zeros((b, 3, self.imgsz, self.imgsz), dtype=np.float32)
            self._run(blob)

    def _preprocess(self, images: List[np.ndarray]) -> tuple[np.ndarray, list]:
        blobs, metas = [], []
        for img in images:
            padded, gain, pad_x, pad_y = letterbox(img, self.imgsz)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            blobs.append(rgb.transpose(2, 0, 1))
            metas.append((gain, pad_x, pad_y, img.shape[1], img.shape[0]))
        blob = np.ascontiguousarray(np.stack(blobs), dtype=np.float32) / 255.0
        return blob, metas

    def infer(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Run batched inference; chunks internally at max_batch."""
        if not images:
            return []
        out: List[np.ndarray] = []
        for start in range(0, len(images), self.max_batch):
            chunk = images[start : start + self.max_batch]
            blob, metas = self._preprocess(chunk)
            preds = self._run(blob)
            for i, meta in enumerate(metas):
                out.append(self._decode(preds[i], meta))
        return out

    def _run(self, blob: np.ndarray) -> np.ndarray:
        """Run one chunk via IO Binding instead of session.run(numpy_dict).

        `bind_cpu_input` + `bind_output` reuse the session's own device
        allocator/copy path through the binding object rather than
        re-validating and re-boxing a fresh feed dict into OrtValues on every
        call — session.run(None, {...}) redoes that setup work each tick even
        though the input/output signature never changes between calls.
        """
        binding = self.session.io_binding()
        binding.bind_cpu_input(self.input_name, blob)
        for output in self.session.get_outputs():
            binding.bind_output(output.name)
        self.session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()[0]

    def _decode(self, pred: np.ndarray, meta: tuple) -> np.ndarray:
        if self.layout == "v8e2e":
            return self._decode_e2e(pred, meta)

        gain, pad_x, pad_y, orig_w, orig_h = meta
        if self.layout == "v5":
            # (N, 5+nc): xywh, obj, cls scores
            boxes_xywh = pred[:, :4]
            scores_all = pred[:, 5:] * pred[:, 4:5]
        else:
            # (4+nc, N): xywh rows then class rows
            pred = pred.T
            boxes_xywh = pred[:, :4]
            scores_all = pred[:, 4:]

        cls_ids = scores_all.argmax(axis=1)
        confs = scores_all[np.arange(len(cls_ids)), cls_ids]
        mask = confs >= self.conf
        if self.keep_classes is not None:
            mask &= np.isin(cls_ids, list(self.keep_classes))
        if not mask.any():
            return np.zeros((0, 6), dtype=np.float32)

        boxes_xywh, confs, cls_ids = boxes_xywh[mask], confs[mask], cls_ids[mask]
        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:4]
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        # letterbox -> original image coords
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / gain
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / gain
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, orig_w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, orig_h)

        keep = nms_xyxy(xyxy, confs, cls_ids, self.iou)
        dets = np.concatenate(
            [xyxy[keep], confs[keep, None], cls_ids[keep, None].astype(np.float32)],
            axis=1,
        ).astype(np.float32)
        return dets

    def _decode_e2e(self, pred: np.ndarray, meta: tuple) -> np.ndarray:
        """End-to-end/NMS-free architectures (YOLOv10, YOLO26, ...) export a
        fixed-size (N, 6) [x1, y1, x2, y2, conf, cls] tensor, already NMS'd
        and sorted by confidence descending, in letterboxed input pixel
        coordinates. Just threshold/filter and remap to source coords — no
        argmax or NMS needed (or wanted: re-running NMS on already-suppressed
        boxes is redundant work at best)."""
        gain, pad_x, pad_y, orig_w, orig_h = meta
        confs = pred[:, 4]
        cls_ids = pred[:, 5].astype(np.int64)
        mask = confs >= self.conf
        if self.keep_classes is not None:
            mask &= np.isin(cls_ids, list(self.keep_classes))
        if not mask.any():
            return np.zeros((0, 6), dtype=np.float32)

        xyxy = pred[mask, :4].copy()
        confs = confs[mask]
        cls_ids = cls_ids[mask].astype(np.float32)

        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / gain
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / gain
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, orig_w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, orig_h)

        return np.concatenate([xyxy, confs[:, None], cls_ids[:, None]], axis=1).astype(np.float32)

    # Weapon checkpoints have been swapped a few times during this project
    # (weapons_yolov8.pt: "guns", epoch20.pt/best_mgd.pt: "pistol", ...) with
    # each bringing its own raw class names. Normalize the handgun class to a
    # single consistent label here rather than hand-editing meta.json every
    # time the checkpoint changes.
    _CLASS_NAME_ALIASES = {"guns": "handgun", "gun": "handgun", "pistol": "handgun"}

    def class_name(self, cls_id: int) -> str:
        raw = self.names.get(int(cls_id), f"class_{int(cls_id)}")
        return self._CLASS_NAME_ALIASES.get(raw.lower(), raw)


class OrtReidEncoder:
    """Batched appearance-embedding (ReID) encoder for BoT-SORT tracker fusion.

    Output feeds directly into ByteTrackAdapter's ``feats=`` param (see
    ultralytics ``BYTETracker.update``/``BOTSORT.get_dists``) to fuse
    appearance with IOU/motion in the association step itself — it is never
    matched against a named gallery or exposed outside the tracker layer.
    Deliberately not routed through ultralytics' own ``AutoBackend``-based
    ``ReID`` loader (``ultralytics/trackers/utils/reid.py``): this project's
    whole inference stack is onnxruntime-gpu + TensorRT (see module
    docstring), so this class mirrors ``OrtYoloDetector``'s loading/IO-binding
    pattern instead of adding a second inference backend.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        name: str = "reid",
        imgsz: int = 224,
        max_batch: int | None = None,
    ):
        import onnxruntime as ort

        self.path = Path(onnx_path)
        if not self.path.is_file():
            self._download(self.path)

        self.name = name
        self.imgsz = imgsz
        self.max_batch = max_batch or settings.ort_max_batch

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = build_providers(
            cache_name=name,
            input_name="images",
            imgsz=imgsz,
            max_batch=self.max_batch,
        )
        try:
            self.session = ort.InferenceSession(str(self.path), sess_options, providers=providers)
        except Exception as exc:
            print(f"[OrtModels] {name}: TRT session failed ({exc}); retrying CUDA/CPU.")
            self.session = ort.InferenceSession(
                str(self.path), sess_options, providers=build_providers(cache_name=name, with_trt=False)
            )
        self.input_name = self.session.get_inputs()[0].name
        self.active_provider = self.session.get_providers()[0]
        print(f"[OrtModels] {name}: {self.path.name} on {self.active_provider} (imgsz={imgsz})")

    @staticmethod
    def _download(dest: Path) -> None:
        """``dest``'s filename must match one of ultralytics' published ReID
        release assets (e.g. ``yolo26n-reid.onnx``, see ``REID_ASSETS`` in
        ``ultralytics/trackers/utils/reid.py``) — the name is what's matched
        against the GitHub release, not the ``settings.model_reid`` value in
        isolation, so callers must resolve the path with that filename."""
        from ultralytics.utils.downloads import attempt_download_asset

        dest.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(dest)

    def warmup(self) -> None:
        """Build/load TRT engines up-front so the first real tick isn't slow."""
        for b in (1, min(4, self.max_batch)):
            blob = np.zeros((b, 3, self.imgsz, self.imgsz), dtype=np.float32)
            self._run(blob)

    def _preprocess(self, crops: List[np.ndarray]) -> np.ndarray:
        # Plain resize (no letterbox padding) + /255, matching ultralytics'
        # own ReID preprocessing (utils/reid.py's _crops_to_tensor) rather
        # than OrtYoloDetector's letterbox: crops are already tight person
        # boxes, not full frames needing aspect-preserving padding.
        blobs = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            blobs.append(resized.transpose(2, 0, 1))
        return np.ascontiguousarray(np.stack(blobs), dtype=np.float32) / 255.0

    def encode(self, crops: List[np.ndarray]) -> np.ndarray:
        """Returns (N, D) L2-normalized float32 embeddings; chunks internally at max_batch."""
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        chunks: List[np.ndarray] = []
        for start in range(0, len(crops), self.max_batch):
            blob = self._preprocess(crops[start : start + self.max_batch])
            chunks.append(self._run(blob))
        out = np.concatenate(chunks, axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def _run(self, blob: np.ndarray) -> np.ndarray:
        binding = self.session.io_binding()
        binding.bind_cpu_input(self.input_name, blob)
        for output in self.session.get_outputs():
            binding.bind_output(output.name)
        self.session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()[0]


def nms_xyxy(
    boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, iou_thresh: float
) -> np.ndarray:
    """Class-aware NMS via OpenCV (offsets boxes per class so classes don't suppress
    each other). Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    offset = cls_ids.astype(np.float32) * 7680.0
    shifted = boxes + offset[:, None]
    xywh = np.concatenate([shifted[:, :2], shifted[:, 2:] - shifted[:, :2]], axis=1)
    keep = cv2.dnn.NMSBoxes(
        xywh.tolist(), scores.astype(float).tolist(), 0.0, iou_thresh
    )
    return np.asarray(keep, dtype=np.int64).reshape(-1)
