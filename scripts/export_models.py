"""
One-time model export + quantization for the ONNX Runtime pipeline.

For each detector (person yolo11n, weapon yolov8, fire yolov5):
  1. Export the .pt checkpoint to ONNX with a dynamic batch axis
     -> data/models/onnx/<key>.onnx
  2. Write a <key>.meta.json sidecar (class names, output layout, input name).
  3. Statically quantize to int8 QDQ, calibrated on frames sampled from the
     real camera footage in assets/camera_*.mp4 (not stock datasets — see
     docs/PIPELINE_OPTIMIZATIONS.md section 7)
     -> data/models/onnx/<key>.int8.onnx

Face models (InsightFace) are intentionally NOT quantized to int8; they run
TRT fp16 (identity matching is the most int8-accuracy-sensitive stage).

Usage:
    .venv/bin/python scripts/export_models.py            # export + int8
    .venv/bin/python scripts/export_models.py --no-int8  # export only
    .venv/bin/python scripts/export_models.py --only fire
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.config import settings  # noqa: E402
from server.inference.ort_models import letterbox  # noqa: E402

IMGSZ = 640
OPSET = 17


def _write_meta(key: str, names: dict, layout: str) -> None:
    meta = {
        "names": {int(k): str(v) for k, v in dict(names).items()},
        "layout": layout,
        "input_name": "images",
        "imgsz": IMGSZ,
    }
    path = settings.onnx_dir / f"{key}.meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  meta -> {path.name} ({len(meta['names'])} classes, layout={layout})")


def export_ultralytics(pt_path: Path, key: str) -> Path:
    """Export a YOLOv8/v11/v26-family checkpoint.

    Most Ultralytics detectors export the raw (4+nc, N) per-anchor layout
    ("v8"): our own argmax + NMS decodes it. Newer end-to-end/NMS-free
    architectures (YOLOv10, YOLO26, ...) instead export a fixed-size,
    already-NMS'd (N, 6) [x1,y1,x2,y2,conf,cls] tensor ("v8e2e") — decoding
    that with the v8 path silently produces garbage (transposes detection
    rows as if they were per-class score channels), so the layout has to be
    recorded correctly at export time via the Detect head's `end2end` flag.
    """
    from ultralytics import YOLO

    out_path = settings.onnx_dir / f"{key}.onnx"
    print(f"[export] {key}: {pt_path.name} -> {out_path.name}")
    model = YOLO(str(pt_path))
    is_end2end = bool(getattr(model.model.model[-1], "end2end", False))
    exported = model.export(
        format="onnx", imgsz=IMGSZ, dynamic=True, opset=OPSET, simplify=False, device="cpu"
    )
    Path(exported).replace(out_path)
    layout = "v8e2e" if is_end2end else "v8"
    if is_end2end:
        print(f"  {key}: end-to-end (NMS-free) architecture detected -> layout={layout}")
    _write_meta(key, model.names, layout=layout)
    return out_path


def export_yolov5(pt_path: Path, key: str) -> Path:
    """Export the fire/smoke YOLOv5 checkpoint (v5 output layout) via torch."""
    import torch

    out_path = settings.onnx_dir / f"{key}.onnx"
    print(f"[export] {key}: {pt_path.name} -> {out_path.name} (yolov5)")
    hub_model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=str(pt_path), trust_repo=True
    )
    names = getattr(hub_model, "names", {})

    m = hub_model
    while hasattr(m, "model") and m.__class__.__name__ not in ("DetectionModel", "Model"):
        m = m.model
    m = m.float().cpu().eval()
    detect = m.model[-1]
    detect.inplace = False
    if hasattr(detect, "dynamic"):
        detect.dynamic = False
    if hasattr(detect, "export"):
        detect.export = False  # keep grid-decoded (bs, N, 5+nc) output

    class _V5Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)[0]

    wrapper = _V5Export(m).eval()
    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)
    kwargs = dict(
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
        opset_version=OPSET,
    )
    try:
        torch.onnx.export(wrapper, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(wrapper, dummy, str(out_path), **kwargs)

    _write_meta(key, names, layout="v5")
    return out_path


class VideoCalibrationReader:
    """Feeds letterboxed frames from the real camera videos to the quantizer."""

    def __init__(self, videos: list[Path], per_video: int = 32, stride: int = 15):
        self._blobs: list[np.ndarray] = []
        for video in videos:
            cap = cv2.VideoCapture(str(video))
            grabbed = 0
            idx = 0
            while cap.isOpened() and grabbed < per_video:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    padded, _, _, _ = letterbox(frame, IMGSZ)
                    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                    self._blobs.append(np.ascontiguousarray(blob))
                    grabbed += 1
                idx += 1
            cap.release()
            print(f"  calibration: {video.name} -> {grabbed} frames")
        self._iter = iter(self._blobs)

    def get_next(self):
        blob = next(self._iter, None)
        return None if blob is None else {"images": blob}

    def rewind(self):
        self._iter = iter(self._blobs)


def quantize_int8(onnx_path: Path, key: str, calib_videos: list[Path]) -> Path:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    out_path = settings.onnx_dir / f"{key}.int8.onnx"
    print(f"[int8] {key}: calibrating on real footage...")

    pre_path = settings.onnx_dir / f"{key}.pre.onnx"
    model_input = onnx_path
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(str(onnx_path), str(pre_path), skip_symbolic_shape=False)
        model_input = pre_path
    except Exception as exc:
        print(f"  quant_pre_process skipped ({exc})")

    reader = VideoCalibrationReader(calib_videos)
    if not reader._blobs:
        raise RuntimeError("No calibration frames found — check assets/camera_*.mp4")

    quantize_static(
        str(model_input),
        str(out_path),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        # Only wrap the compute-heavy ops (this is where int8 actually buys
        # speed). Left at the ORT default, the quantizer also wraps the tiny
        # scalar/rank-0 Constants in the YOLO Detect head (DFL projection,
        # stride constants) with QDQ pairs. Those get an ONNX-spec-default
        # `axis=1` on their DequantizeLinear node, which TensorRT's parser
        # rejects for a 0-D tensor ("Axis must be in range [0, nbDims (0))]")
        # and aborts the whole engine build. Excluding non-Conv/Gemm/MatMul
        # ops from quantization keeps the Detect head in fp32 (cheap anyway)
        # and avoids the crash entirely.
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        extra_options={
            # TensorRT requires symmetric int8 quantization
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            # TensorRT's DequantizeLayer only accepts activation inputs, not a
            # weight-like initializer such as a Conv bias — quantizing bias to
            # int32 + DequantizeLinear (ORT's default) makes TRT's parser
            # reject the node ("only activation types allowed as input to
            # this layer"). Leave bias in fp32; TRT's int8 Conv kernel adds
            # it directly into the int32 accumulator, no extra DQ needed.
            "QuantizeBias": False,
        },
    )
    if pre_path.exists():
        pre_path.unlink()
    print(f"  int8 -> {out_path.name}")
    return out_path


def resolve_pt(setting_value: str) -> Path:
    p = Path(setting_value)
    if not p.is_absolute():
        p = settings.project_root / p
    if not p.is_file():
        # model_person is a bare filename under models_dir
        alt = settings.models_dir / setting_value
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"Checkpoint not found: {setting_value}")
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-int8", action="store_true", help="Skip int8 quantization")
    parser.add_argument(
        "--only", choices=["person", "fire", "weapon"], default=None, help="Export one model"
    )
    args = parser.parse_args()

    settings.onnx_dir.mkdir(parents=True, exist_ok=True)
    calib_videos = sorted(settings.assets_dir.glob("camera_*.mp4"))

    jobs = {
        "person": (resolve_pt(settings.model_person), export_ultralytics),
        "weapon": (resolve_pt(settings.model_weapon), export_ultralytics),
        "fire": (resolve_pt(settings.model_fire), export_yolov5),
    }
    if args.only:
        jobs = {args.only: jobs[args.only]}

    for key, (pt_path, exporter) in jobs.items():
        onnx_path = exporter(pt_path, key)
        if not args.no_int8:
            try:
                quantize_int8(onnx_path, key, calib_videos)
            except Exception as exc:
                print(f"[int8] {key} FAILED ({exc}); fp path will be used at runtime.")

    print("Done. Runtime resolves int8 variants automatically via settings.detector_onnx().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
