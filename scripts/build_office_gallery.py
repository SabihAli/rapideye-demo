"""Build the "office" facial-rec gallery from assets/gallery_assets/<name>/*.jpg.

Each subfolder of --images-dir is one identity (folder name = display name).
Every image in a folder is embedded with InsightFace and averaged into a
single prototype vector, same as the ChokePoint XML enrollment path in
server/inference/face_recognition.py.

Usage:
    .venv/bin/python scripts/build_office_gallery.py

Writes:
    data/facial_rec/gallery_built/gallery_office.json  (labelled copy)
    data/facial_rec/gallery_built/gallery_P1E_S2_C1.json          (live gallery, used
                                                           by FacePipelineService)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from server.config import settings
from server.inference.face_recognition import FaceAnalysisApp, write_gallery_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build(images_dir: Path, app: FaceAnalysisApp) -> dict:
    identities: list[dict] = []
    for person_dir in sorted(images_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        embeddings: list[np.ndarray] = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [skip] unreadable: {img_path}")
                continue
            faces = app.analyze(img)
            if not faces:
                print(f"  [skip] no face detected: {img_path}")
                continue
            face = max(faces, key=lambda f: f.det_score)
            emb = face.embedding.astype(np.float32)
            embeddings.append(emb / max(np.linalg.norm(emb), 1e-12))

        if not embeddings:
            print(f"  [warn] no usable faces for {person_dir.name}, skipping identity")
            continue

        prototype = np.mean(np.stack(embeddings, axis=0), axis=0)
        prototype /= max(np.linalg.norm(prototype), 1e-12)
        identities.append(
            {
                "identity_id": person_dir.name,
                "display_name": person_dir.name,
                "num_enrollment_crops": len(embeddings),
                "embedding": prototype.tolist(),
            }
        )
        print(f"  [ok] {person_dir.name}: {len(embeddings)}/{len(images)} images enrolled")

    return {
        "identities": identities,
        "threshold": settings.facial_match_threshold,
        "source": "office_image_folders",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=settings.project_root / "assets" / "gallery_assets",
        help="Folder of <identity_name>/*.jpg subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.facial_gallery_dir,
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id (-1 for CPU)")
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        print(f"Images dir not found: {args.images_dir}")
        return 1

    gpu = args.gpu if settings.use_cuda else -1
    app = FaceAnalysisApp(
        model_pack=settings.insightface_model,
        det_size=(settings.insightface_det_size, settings.insightface_det_size),
        ctx_id=gpu,
    )

    print(f"Enrolling identities from {args.images_dir}")
    result = build(args.images_dir, app)
    if not result["identities"]:
        print("No identities enrolled — aborting, gallery files left untouched.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    office_path = write_gallery_json(result, args.output_dir, filename=settings.facial_gallery_office_filename)
    live_path = write_gallery_json(result, args.output_dir, filename=settings.facial_gallery_filename)
    print(f"\nGallery ready: {office_path} ({len(result['identities'])} identities)")
    print(f"Live gallery updated: {live_path}  <- FacePipelineService reads this path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
