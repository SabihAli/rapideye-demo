"""Unit tests for server.inference.face_recognition (no GPU required by default)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from server.inference.face_recognition import (
    MAX_CAMERA_INPUTS,
    FaceGallery,
    cluster_faces,
    discover_sample_videos,
    parse_chokepoint_xml,
    recognize_persons_batch,
    write_gallery_json,
)


def _unit_vector(dim: int, idx: int = 0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx % dim] = 1.0
    return v


class TestFaceGallery:
    def test_match_batch_returns_unknown_below_threshold(self):
        gallery = FaceGallery(threshold=0.9)
        gallery.enroll("a", "Alice", _unit_vector(4, 0))
        query = _unit_vector(4, 1)
        name, sim = gallery.match_batch(query[None, :])[0]
        assert name == "Unknown"
        assert sim < 0.9

    def test_match_batch_finds_enrolled_identity(self):
        gallery = FaceGallery(threshold=0.5)
        emb = _unit_vector(8, 2)
        gallery.enroll("bob", "Bob", emb)
        name, sim = gallery.match(emb)
        assert name == "Bob"
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_enroll_replaces_same_identity_id(self):
        gallery = FaceGallery()
        gallery.enroll("x", "Old", _unit_vector(4, 0))
        gallery.enroll("x", "New", _unit_vector(4, 1))
        assert len(gallery.entries) == 1
        assert gallery.entries[0].display_name == "New"


class TestDiscoverSampleVideos:
    def test_discovers_mp4_recursively(self, tmp_path: Path):
        (tmp_path / "a.mp4").write_bytes(b"")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "b.mp4").write_bytes(b"")
        (tmp_path / "readme.txt").write_text("skip")

        found = discover_sample_videos(tmp_path, max_inputs=MAX_CAMERA_INPUTS)
        assert [p.name for p in found] == ["a.mp4", "b.mp4"]

    def test_respects_max_inputs(self, tmp_path: Path):
        for i in range(6):
            (tmp_path / f"cam{i}.mp4").write_bytes(b"")
        found = discover_sample_videos(tmp_path, max_inputs=4)
        assert len(found) == 4

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert discover_sample_videos(tmp_path / "missing") == []


class TestClusterFaces:
    def test_clusters_similar_embeddings(self):
        from server.inference.face_recognition import FaceRecord

        base = _unit_vector(4, 0)
        records = [
            FaceRecord(0, (0, 0, 10, 10), 0.9, base, np.zeros((10, 10, 3), np.uint8)),
            FaceRecord(1, (0, 0, 10, 10), 0.9, base.copy(), np.zeros((10, 10, 3), np.uint8)),
            FaceRecord(2, (0, 0, 10, 10), 0.9, _unit_vector(4, 1), np.zeros((10, 10, 3), np.uint8)),
        ]
        clusters = cluster_faces(records, join_threshold=0.99, merge_threshold=0.99)
        assert len(clusters) == 2
        sizes = sorted(len(c["members"]) for c in clusters)
        assert sizes == [1, 2]


class TestParseChokepointXml:
    def test_parses_eye_annotations(self, tmp_path: Path):
        xml = tmp_path / "seq.xml"
        xml.write_text(
            """<?xml version="1.0"?>
            <dataset>
              <frame number="00000001">
                <person id="42">
                  <leftEye x="10" y="20"/>
                  <rightEye x="30" y="22"/>
                </person>
              </frame>
            </dataset>""",
            encoding="utf-8",
        )
        anns = parse_chokepoint_xml(xml)
        assert len(anns) == 1
        assert anns[0].person_id == "42"
        assert anns[0].left_eye == (10, 20)
        assert anns[0].right_eye == (30, 22)


class TestRecognizePersonsBatch:
    def test_batch_matches_multiple_embeddings(self):
        gallery = FaceGallery(threshold=0.5)
        gallery.enroll("a", "Alice", _unit_vector(8, 0))
        gallery.enroll("b", "Bob", _unit_vector(8, 1))

        app = MagicMock()
        face_a = MagicMock()
        face_a.det_score = 0.9
        face_a.embedding = _unit_vector(8, 0)
        face_b = MagicMock()
        face_b.det_score = 0.9
        face_b.embedding = _unit_vector(8, 1)
        app.analyze.side_effect = [[face_a], [face_b]]

        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        results = recognize_persons_batch(
            [(frame, (10, 10, 60, 60)), (frame, (20, 20, 70, 70))],
            app,
            gallery,
        )
        assert results[0][0] == "Alice"
        assert results[1][0] == "Bob"


class TestWriteGalleryJson:
    def test_writes_json_file(self, tmp_path: Path):
        payload = {"threshold": 0.4, "identities": []}
        path = write_gallery_json(payload, tmp_path)
        assert path.is_file()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["threshold"] == 0.4
