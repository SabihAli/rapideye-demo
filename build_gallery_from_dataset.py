#!/usr/bin/env python3
"""
Build a face recognition gallery from ChokePoint-style datasets in data/facial_rec.

Your layout:
  data/facial_rec/samples/<SEQUENCE>/00000000.jpg ...
  data/facial_rec/samples/<SEQUENCE>/bg_img.txt   # background-only frames (skip)
  data/facial_rec/gallery/<SEQUENCE>.xml          # person id + eye landmarks per frame

Strategy (does not change detection/recognition models):
  1. Parse XML ground-truth person IDs (reliable labels — no clustering).
  2. Exclude bg_img.txt frames (no faces).
  3. Score each (person, frame) by face size, sharpness, and frontal pose.
  4. Keep top-K frames per person, crop via eye landmarks, embed with InsightFace.
  5. Mean-pool embeddings per person → one robust prototype per identity.

Outputs:
  data/facial_rec/gallery_built/<person_id>/enroll_*.jpg  # visual QA
  data/facial_rec/gallery_built/gallery.json              # for facial_rec_video_test.py

Usage:
  pip install opencv-python-headless insightface onnxruntime-gpu numpy
  python build_gallery_from_dataset.py
  python build_gallery_from_dataset.py --sequence P1E_S2_C1 --top-k 5
  python facial_rec_video_test.py -i ... --gallery data/facial_rec/gallery_built/gallery.json
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DATA_ROOT = Path("data/facial_rec")
DEFAULT_OUT = DEFAULT_DATA_ROOT / "gallery_built"


@dataclass
class EyeAnnotation:
    frame_id: str
    person_id: str
    left_eye: tuple[int, int]
    right_eye: tuple[int, int]


def parse_chokepoint_xml(xml_path: Path) -> list[EyeAnnotation]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations: list[EyeAnnotation] = []
    for frame_el in root.findall("frame"):
        frame_id = frame_el.attrib.get("number", "")
        for person_el in frame_el.findall("person"):
            person_id = person_el.attrib.get("id", "")
            left = person_el.find("leftEye")
            right = person_el.find("rightEye")
            if left is None or right is None:
                continue
            annotations.append(
                EyeAnnotation(
                    frame_id=frame_id,
                    person_id=person_id,
                    left_eye=(int(left.attrib["x"]), int(left.attrib["y"])),
                    right_eye=(int(right.attrib["x"]), int(right.attrib["y"])),
                )
            )
    return annotations


def load_bg_frames(sequence_dir: Path) -> set[str]:
    bg_file = sequence_dir / "bg_img.txt"
    if not bg_file.exists():
        return set()
    lines = bg_file.read_text(encoding="utf-8").strip().splitlines()
    return {line.strip() for line in lines if line.strip()}


def frame_filename(frame_id: str) -> str:
    return f"{frame_id}.jpg" if not frame_id.endswith(".jpg") else frame_id


def eye_bbox(
    left: tuple[int, int],
    right: tuple[int, int],
    img_w: int,
    img_h: int,
    scale: float = 2.8,
) -> tuple[int, int, int, int]:
    """Estimate a square face crop from eye positions (ChokePoint XML has eyes only)."""
    lx, ly = left
    rx, ry = right
    cx = (lx + rx) / 2
    cy = (ly + ry) / 2
    eye_dist = max(np.hypot(rx - lx, ry - ly), 1.0)
    half = eye_dist * scale
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half * 1.1))
    x2 = int(min(img_w, cx + half))
    y2 = int(min(img_h, cy + half * 1.2))
    return x1, y1, x2, y2


def sharpness_score(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_score(ann: EyeAnnotation, crop_bgr: np.ndarray) -> float:
    lx, ly = ann.left_eye
    rx, ry = ann.right_eye
    eye_dist = np.hypot(rx - lx, ry - ly)
    frontal = 1.0 / (1.0 + abs(ly - ry))
    sharp = sharpness_score(crop_bgr)
    size = eye_dist
    return float(size * 0.5 + sharp * 0.003 + frontal * 20.0)


def load_insightface(model_pack: str, det_size: int, gpu: int):
    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name=model_pack, providers=providers)
    app.prepare(ctx_id=gpu, det_size=(det_size, det_size))
    return app


def embed_crop(app, crop_bgr: np.ndarray) -> np.ndarray | None:
    """Embed a pre-cropped face region; falls back to full SCRFD on crop."""
    faces = app.get(crop_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: float(f.det_score))
    emb = np.asarray(face.embedding, dtype=np.float32)
    return emb / max(np.linalg.norm(emb), 1e-12)


def build_for_sequence(
    sequence: str,
    data_root: Path,
    out_dir: Path,
    app,
    top_k: int,
    min_quality: float,
) -> dict:
    seq_dir = data_root / "samples" / sequence
    xml_path = data_root / "gallery" / f"{sequence}.xml"
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
    if not xml_path.is_file():
        raise FileNotFoundError(f"Annotation XML not found: {xml_path}")

    bg_frames = load_bg_frames(seq_dir)
    annotations = parse_chokepoint_xml(xml_path)

    by_person: dict[str, list[tuple[float, EyeAnnotation, np.ndarray]]] = {}

    for ann in annotations:
        fname = frame_filename(ann.frame_id)
        if fname in bg_frames:
            continue
        img_path = seq_dir / fname
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = eye_bbox(ann.left_eye, ann.right_eye, w, h)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        crop = img[y1:y2, x1:x2]
        score = quality_score(ann, crop)
        if score < min_quality:
            continue
        by_person.setdefault(ann.person_id, []).append((score, ann, crop))

    identities: list[dict] = []
    seq_out = out_dir / sequence
    seq_out.mkdir(parents=True, exist_ok=True)

    for person_id, candidates in sorted(by_person.items()):
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:top_k]
        embeddings: list[np.ndarray] = []
        person_dir = seq_out / person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        for rank, (score, ann, crop) in enumerate(selected):
            out_img = person_dir / f"enroll_{rank:02d}_f{ann.frame_id}.jpg"
            cv2.imwrite(str(out_img), crop)
            emb = embed_crop(app, crop)
            if emb is not None:
                embeddings.append(emb)
            print(f"  {person_id} rank={rank} frame={ann.frame_id} quality={score:.1f}")

        if not embeddings:
            print(f"  Warning: no embeddings for person {person_id}", file=sys.stderr)
            continue
        prototype = np.mean(np.stack(embeddings, axis=0), axis=0)
        prototype /= max(np.linalg.norm(prototype), 1e-12)
        identities.append(
            {
                "identity_id": person_id,
                "display_name": f"Person_{person_id}",
                "sequence": sequence,
                "num_enrollment_crops": len(embeddings),
                "embedding": prototype.tolist(),
            }
        )

    return {
        "sequence": sequence,
        "identities": identities,
        "threshold": 0.4,
        "source": "chokepoint_xml_auto",
    }


def discover_sequences(data_root: Path) -> list[str]:
    samples = data_root / "samples"
    if not samples.is_dir():
        return []
    sequences = []
    for seq_dir in sorted(samples.iterdir()):
        if not seq_dir.is_dir():
            continue
        xml = data_root / "gallery" / f"{seq_dir.name}.xml"
        if xml.is_file():
            sequences.append(seq_dir.name)
    return sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build face gallery from ChokePoint-style dataset")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sequence", default=None, help="One sequence (default: all with XML)")
    p.add_argument("--top-k", type=int, default=5, help="Best frames per person to embed")
    p.add_argument("--min-quality", type=float, default=30.0, help="Min quality score threshold")
    p.add_argument("--model", default="buffalo_l")
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--enroll-sequence",
        default=None,
        help="Build gallery only from this sequence; useful when probing another (avoids leakage)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sequences = [args.sequence] if args.sequence else discover_sequences(args.data_root)
    if args.enroll_sequence:
        sequences = [args.enroll_sequence]
    if not sequences:
        print("No sequences found. Expected samples/<SEQ>/ + gallery/<SEQ>.xml", file=sys.stderr)
        return 1

    print(f"Loading InsightFace {args.model}...")
    app = load_insightface(args.model, args.det_size, args.gpu)

    all_identities: list[dict] = []
    args.output.mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        print(f"Building gallery from {seq}...")
        result = build_for_sequence(
            seq, args.data_root, args.output, app, args.top_k, args.min_quality
        )
        all_identities.extend(result["identities"])
        per_seq_json = args.output / f"gallery_{seq}.json"
        per_seq_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  Wrote {per_seq_json} ({len(result['identities'])} identities)")

    combined = {
        "threshold": 0.4,
        "source": "chokepoint_xml_auto",
        "identities": all_identities,
    }
    gallery_json = args.output / "gallery.json"
    gallery_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nGallery ready: {gallery_json} ({len(all_identities)} identities)")
    print("Run video test:")
    print(f"  python facial_rec_video_test.py -i <video> --gallery {gallery_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
