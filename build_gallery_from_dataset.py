#!/usr/bin/env python3
"""
Build a face recognition gallery from data/facial_rec — two modes.

MODE A — raw unlabelled video (no ground truth needed):
  Scan a video, detect+embed every face, cluster the embeddings into distinct
  people (unsupervised), and mean-pool each cluster into one prototype. Identities
  are auto-named Person_1, Person_2, ... — edit display_name in gallery.json (or use
  the saved QA crops to recognise who's who) to assign real names.

    python build_gallery_from_dataset.py --video enroll.mp4 --output data/facial_rec/gallery_built
    python facial_rec_video_test.py -i probe.mp4 \
        --gallery data/facial_rec/gallery_built/gallery.json -o output.mp4

MODE B — ChokePoint-style labelled dataset:
  data/facial_rec/samples/<SEQUENCE>/00000000.jpg ...
  data/facial_rec/samples/<SEQUENCE>/bg_img.txt   # background-only frames (skip)
  data/facial_rec/gallery/<SEQUENCE>.xml          # person id + eye landmarks per frame

  Strategy (does not change detection/recognition models):
    1. Parse XML ground-truth person IDs (reliable labels — no clustering).
    2. Exclude bg_img.txt frames (no faces).
    3. Score each (person, frame) by face size, sharpness, and frontal pose.
    4. Keep top-K frames per person, crop via eye landmarks, embed with InsightFace.
    5. Mean-pool embeddings per person → one robust prototype per identity.

Outputs (both modes):
  <output>/<identity_id>/enroll_*.jpg   # visual QA crops
  <output>/gallery.json                 # for facial_rec_video_test.py

Usage:
  pip install opencv-python-headless insightface onnxruntime-gpu numpy
  python build_gallery_from_dataset.py --video enroll.mp4
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


# ---------------------------------------------------------------------------
# Raw video mode — build a gallery from an UNLABELLED video via clustering
# ---------------------------------------------------------------------------


@dataclass
class FaceRecord:
    frame_idx: int
    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: np.ndarray  # L2-normalized
    crop: np.ndarray


def collect_faces_from_video(
    app,
    video_path: Path,
    sample_every: int,
    max_frames: int | None,
    min_det_score: float,
    min_face_px: int,
) -> list[FaceRecord]:
    """Detect + embed every face in the video (subsampled by sample_every)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    records: list[FaceRecord] = []
    frame_idx = 0
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue
            h, w = frame.shape[:2]
            for face in app.get(frame):
                det = float(face.det_score)
                if det < min_det_score:
                    continue
                x1, y1, x2, y2 = face.bbox.astype(int).tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < min_face_px or y2 - y1 < min_face_px:
                    continue
                emb = np.asarray(face.embedding, dtype=np.float32)
                emb /= max(np.linalg.norm(emb), 1e-12)
                records.append(
                    FaceRecord(
                        frame_idx=frame_idx,
                        bbox=(x1, y1, x2, y2),
                        det_score=det,
                        embedding=emb,
                        crop=frame[y1:y2, x1:x2].copy(),
                    )
                )
            frame_idx += 1
            if frame_idx % (sample_every * 50) == 0:
                print(f"  scanned {frame_idx} frames, {len(records)} face crops")
    finally:
        cap.release()

    print(f"Collected {len(records)} face crops from {frame_idx} frames")
    return records


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / max(np.linalg.norm(vec), 1e-12)


def cluster_faces(
    records: list[FaceRecord],
    join_threshold: float,
    merge_threshold: float,
) -> list[dict]:
    """Greedy leader clustering on cosine similarity, then a centroid merge pass.

    No external deps: assign each face to the nearest existing cluster centroid if
    similarity >= join_threshold, else start a new cluster; then fold together any
    two clusters whose centroids are near-duplicates (handles pose/lighting drift).
    """
    clusters: list[dict] = []
    for idx, rec in enumerate(records):
        best_j, best_sim = -1, -1.0
        for j, c in enumerate(clusters):
            sim = float(np.dot(rec.embedding, c["centroid"]))
            if sim > best_sim:
                best_sim, best_j = sim, j
        if best_j >= 0 and best_sim >= join_threshold:
            c = clusters[best_j]
            c["members"].append(idx)
            c["sum"] += rec.embedding
            c["centroid"] = _normalize(c["sum"])
        else:
            clusters.append(
                {"members": [idx], "sum": rec.embedding.copy(), "centroid": rec.embedding.copy()}
            )

    merged = True
    while merged:
        merged = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if float(np.dot(clusters[a]["centroid"], clusters[b]["centroid"])) >= merge_threshold:
                    clusters[a]["members"].extend(clusters[b]["members"])
                    clusters[a]["sum"] += clusters[b]["sum"]
                    clusters[a]["centroid"] = _normalize(clusters[a]["sum"])
                    del clusters[b]
                    merged = True
                    break
            if merged:
                break
    return clusters


def build_gallery_from_video(
    app,
    video_path: Path,
    out_dir: Path,
    *,
    sample_every: int,
    max_frames: int | None,
    min_det_score: float,
    min_face_px: int,
    join_threshold: float,
    merge_threshold: float,
    min_samples: int,
    top_k: int,
    name_prefix: str,
    match_threshold: float,
) -> dict:
    records = collect_faces_from_video(
        app, video_path, sample_every, max_frames, min_det_score, min_face_px
    )
    if not records:
        raise RuntimeError("No faces detected in enrollment video — check the clip or lower --min-det-score.")

    clusters = cluster_faces(records, join_threshold, merge_threshold)
    kept = [c for c in clusters if len(c["members"]) >= min_samples]
    kept.sort(key=lambda c: len(c["members"]), reverse=True)
    print(
        f"Clustered into {len(clusters)} groups; "
        f"{len(kept)} identities pass min-samples={min_samples}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    identities: list[dict] = []
    for rank, c in enumerate(kept, start=1):
        identity_id = f"id{rank:03d}"
        display_name = f"{name_prefix}_{rank}"
        prototype = _normalize(c["sum"])

        # Save the highest-scoring crops for visual QA + renaming.
        members = sorted((records[i] for i in c["members"]), key=lambda r: r.det_score, reverse=True)
        person_dir = out_dir / identity_id
        person_dir.mkdir(parents=True, exist_ok=True)
        for k, rec in enumerate(members[:top_k]):
            cv2.imwrite(str(person_dir / f"enroll_{k:02d}_f{rec.frame_idx}.jpg"), rec.crop)

        identities.append(
            {
                "identity_id": identity_id,
                "display_name": display_name,
                "num_enrollment_crops": len(members),
                "embedding": prototype.tolist(),
            }
        )
        print(f"  {display_name}: {len(members)} crops (id={identity_id})")

    return {
        "threshold": match_threshold,
        "source": "unlabelled_video_cluster",
        "enrollment_video": str(video_path),
        "identities": identities,
    }


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

    # Raw video mode (unlabelled) — enable by passing --video
    v = p.add_argument_group("raw video mode (unlabelled --video)")
    v.add_argument("--video", type=Path, default=None, help="Raw video to cluster faces from (no labels needed)")
    v.add_argument("--sample-every", type=int, default=5, help="Process every Nth frame")
    v.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames")
    v.add_argument("--min-det-score", type=float, default=0.5, help="Drop weak face detections")
    v.add_argument("--min-face", type=int, default=40, help="Min face box size in px")
    v.add_argument("--cluster-threshold", type=float, default=0.5, help="Cosine sim to join a cluster")
    v.add_argument("--merge-threshold", type=float, default=0.6, help="Cosine sim to merge two clusters")
    v.add_argument("--min-samples", type=int, default=5, help="Min detections to keep an identity")
    v.add_argument("--name-prefix", default="Person", help="Auto display-name prefix (Person_1, ...)")
    v.add_argument("--match-threshold", type=float, default=0.4, help="Match threshold written into gallery.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.video is not None:
        if not args.video.is_file():
            print(f"Video not found: {args.video}", file=sys.stderr)
            return 1
        print(f"Loading InsightFace {args.model}...")
        app = load_insightface(args.model, args.det_size, args.gpu)
        print(f"Building gallery from unlabelled video {args.video}...")
        result = build_gallery_from_video(
            app,
            args.video,
            args.output,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            min_det_score=args.min_det_score,
            min_face_px=args.min_face,
            join_threshold=args.cluster_threshold,
            merge_threshold=args.merge_threshold,
            min_samples=args.min_samples,
            top_k=args.top_k,
            name_prefix=args.name_prefix,
            match_threshold=args.match_threshold,
        )
        gallery_json = args.output / "gallery.json"
        gallery_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nGallery ready: {gallery_json} ({len(result['identities'])} identities)")
        print(f"QA crops per identity under {args.output}/<id>/ — edit display_name in the JSON for real names.")
        print("Run inference on another raw video with the same people:")
        print(f"  python facial_rec_video_test.py -i probe.mp4 --gallery {gallery_json} -o output.mp4")
        return 0

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
