"""Targeted regression tests for canonical perception data handling."""
from __future__ import annotations

import torch
import pytest

from src.agents.perception.data_loader import MultiModalDataLoader, MultiModalDataset
from src.agents.perception.utils.perception_errors import PerceptionShapeError


def test_variable_text_lengths_collate_with_attention_mask():
    samples = [
        {"text": {"input_ids": torch.tensor([1, 2, 3]),
                  "attention_mask": torch.tensor([1, 1, 1])}, "metadata": {"id": "a"}},
        {"text": {"input_ids": torch.tensor([1, 2]),
                  "attention_mask": torch.tensor([1, 1])}, "metadata": {"id": "b"}},
    ]
    result = MultiModalDataLoader.collate_fn(samples)
    assert result["text"]["input_ids"].shape == (2, 3)
    assert result["text"]["attention_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert len(result["metadata"]) == 2


def test_reject_misaligned_sparse_batch_instead_of_silent_drop():
    samples = [
        {"text": {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])}},
        {"vision": torch.zeros(3, 4, 4)},
    ]
    with pytest.raises(PerceptionShapeError):
        MultiModalDataLoader.collate_fn(samples)


def test_image_train_and_infer_preprocessing_match(tmp_path):
    from PIL import Image
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (24, 30), color=(80, 120, 200)).save(image_path)
    training = MultiModalDataset(mode="train")
    inference = MultiModalDataset(mode="inference")
    assert "vision" in training.transforms and "vision" in inference.transforms
    with Image.open(image_path) as image:
        left = training.transforms["vision"](image)
        right = inference.transforms["vision"](image)
    assert torch.allclose(left, right, atol=0.0, rtol=0.0)
