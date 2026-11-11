"""Manifest and checkpoint-safety regression tests; no external dataset downloads."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.perception.train_models import validate_manifest, _safe_path


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_manifest_requires_real_data_and_three_splits(tmp_path):
    for item in ("train.jpg", "val.jpg", "test.jpg"):
        (tmp_path / item).write_bytes(b"fixture")
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
    (tmp_path / "img.jpg").write_bytes(b"fixture")
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
