"""Manifest and checkpoint-safety regression tests; no external dataset downloads."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.agents.perception.train_models import validate_manifest, _safe_path


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_manifest_requires_real_data_and_three_splits(tmp_path):
    for index, item in enumerate(("train.jpg", "val.jpg", "test.jpg")):
        Image.new("RGB", (8, 8), color=(index * 80, 10, 50)).save(tmp_path / item)
    records = [
        {"id": "i1", "split": "train", "group_id": "src1", "positive_id": "cat",
         "text": "a cat", "vision": "train.jpg", "label": 0},
        {"id": "i2", "split": "validation", "group_id": "src2", "positive_id": "cat",
         "text": "cat", "vision": "val.jpg", "label": 0},
        {"id": "i3", "split": "test", "group_id": "src3", "positive_id": "dog",
         "text": "dog", "vision": "test.jpg", "label": 1},
    ]
    path = _write(tmp_path / "manifest.jsonl", records)
    rows, summary = validate_manifest(path, tmp_path, ("text", "vision"))
    assert len(rows) == 3
    assert summary["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert len(summary["manifest_sha256"]) == 64
    records[1]["group_id"] = "src1"
    _write(path, records)
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(path, tmp_path, ("text", "vision"))


def test_rejects_outside_root_and_duplicate_ids(tmp_path):
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(tmp_path / "img.jpg")
    with pytest.raises(ValueError):
        _safe_path(tmp_path, "../another-place")
    with pytest.raises(ValueError):
        _safe_path(tmp_path, "/tmp/secret.png")
    data = [
        {"id": "dupe", "split": "train", "text": "one", "vision": "img.jpg"},
        {"id": "dupe", "split": "validation", "text": "two", "vision": "img.jpg"},
        {"id": "unique", "split": "test", "text": "three", "vision": "img.jpg"},
    ]
    with pytest.raises(ValueError):
        validate_manifest(_write(tmp_path / "manifest.jsonl", data), tmp_path, ("text", "vision"))


def test_missing_dataset_is_not_silently_generated(tmp_path):
    with pytest.raises(ValueError, match="real local"):
        validate_manifest(tmp_path / "missing.jsonl", tmp_path, ("audio", "text"))

def test_same_bytes_cross_split_rejected_even_with_different_paths(tmp_path):
    original = tmp_path / "original.jpg"
    Image.new("RGB", (8, 8), color=(100, 40, 30)).save(original)
    (tmp_path / "copy.jpg").write_bytes(original.read_bytes())
    other = tmp_path / "unique.jpg"
    Image.new("RGB", (8, 8), color=(200, 40, 30)).save(other)
    rows = [
        {"id": "t1", "group_id": "src1", "split": "train",
         "text": "cat", "vision": "original.jpg"},
        {"id": "v1", "group_id": "src2", "split": "validation",
         "text": "cat", "vision": "copy.jpg"},
        {"id": "e1", "group_id": "src3", "split": "test",
         "text": "dog", "vision": "unique.jpg"},
    ]
    with pytest.raises(ValueError, match="Byte-identical"):
        validate_manifest(_write(tmp_path / "manifest.jsonl", rows),
                          tmp_path, ("text", "vision"))
