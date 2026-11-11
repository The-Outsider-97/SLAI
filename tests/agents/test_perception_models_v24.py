"""Focused, CPU-only contract tests for Perception v2.4 trainable model views."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.agents.perception.perception_contracts import Modality, ModalityRepresentation
from src.agents.perception.perception_fusion import PerceptionFusion
from src.agents.perception.perception_models import MultimodalPerception, MultimodalContrastiveEncoder
from src.agents.perception.perception_objectives import PerceptionObjectives
from src.agents.perception.utils.perception_errors import ModalityInputError


class TinyPipeline(nn.Module):
    def __init__(self, modality: Modality, in_dim: int, out_dim: int):
        super().__init__()
        self.modality = modality
        self.embed_dim = out_dim
        self.projection = nn.Linear(in_dim, out_dim)

    def encode(self, payload):
        return ModalityRepresentation(modality=self.modality,
                                      pooled=self.projection(payload))


def build_models():
    torch.manual_seed(9)
    modalities = {
        Modality.TEXT: TinyPipeline(Modality.TEXT, 3, 5),
        Modality.VISION: TinyPipeline(Modality.VISION, 4, 7),
    }
    fusion = PerceptionFusion(
        input_dims={Modality.TEXT: 5, Modality.VISION: 7},
        output_dim=6, fusion_method="concat", use_attention=True, num_heads=2, dropout=0.0,
    )
    objectives = PerceptionObjectives(
        input_dims={Modality.TEXT: 5, Modality.VISION: 7},
        contrastive_projection_dim=3, symmetric_contrastive=True,
    )
    heads = nn.ModuleDict({"classification": nn.Linear(6, 2)})
    return (MultimodalPerception(modalities, fusion, heads),
            MultimodalContrastiveEncoder(modalities, objectives))


def test_missing_modality_and_fused_task_learning():
    perception, _ = build_models()
    batch = {Modality.TEXT: torch.randn(4, 3)}
    fused = perception(batch)
    assert tuple(fused.pooled.shape) == (4, 6)
    assert fused.presence_mask.shape == (4, 2)
    assert int(fused.presence_mask.sum()) == 4
    with pytest.raises(ModalityInputError):
        perception({})
    logits = perception(batch, downstream_head="classification")
    assert logits.shape == (4, 2)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
    loss.backward()
    assert perception.fusion.concat_projection[0].weight.grad is not None
    assert perception.modalities["text"].projection.weight.grad is not None


def test_contrastive_embeddings_and_multi_positive_loss():
    _, contrastive = build_models()
    inputs = {
        Modality.TEXT: torch.randn(4, 3),
        Modality.VISION: torch.randn(4, 4),
    }
    vectors = contrastive(inputs)
    assert set(vectors) == {Modality.TEXT, Modality.VISION}
    for item in vectors.values():
        assert item.shape == (4, 3)
        assert torch.allclose(item.norm(dim=1), torch.ones(4), atol=1e-5)
    similarities = contrastive.similarity(vectors[Modality.TEXT], vectors[Modality.VISION])
    assert similarities.shape == (4, 4)
    reps = {m: contrastive.modalities[m.value].encode(data) for m, data in inputs.items()}
    naive = contrastive.contrastive_loss(reps[Modality.TEXT], reps[Modality.VISION])
    groups = ["a", "a", "b", "c"]
    grouped = contrastive.contrastive_loss(reps[Modality.TEXT], reps[Modality.VISION], group_ids=groups)
    assert grouped.value <= naive.value + 1e-5
    grouped.value.backward()
    assert contrastive.objectives.contrastive_projectors["text"].weight.grad is not None
    with pytest.raises(Exception):
        contrastive.contrastive_loss(reps[Modality.TEXT], reps[Modality.VISION], group_ids=["bad"])


def test_positive_pairs_can_be_learned_without_dummy_checkpoints():
    # Synthetic numerical fixture validates optimization, not real-world quality.
    torch.manual_seed(21)
    objective = PerceptionObjectives(
        input_dims={Modality.TEXT: 4, Modality.VISION: 4},
        contrastive_projection_dim=4, contrastive_temperature=0.2,
        symmetric_contrastive=True,
    )
    for proj in objective.contrastive_projectors.values():
        nn.init.eye_(proj.weight)
        nn.init.zeros_(proj.bias)
    left = torch.eye(4).repeat(2, 1)
    right = torch.roll(left, shifts=1, dims=1)
    a = ModalityRepresentation(Modality.TEXT, left)
    b = ModalityRepresentation(Modality.VISION, right)
    optim = torch.optim.AdamW(objective.parameters(), lr=0.05)
    initial = float(objective.contrastive(a, b).value.detach())
    for _ in range(50):
        optim.zero_grad()
        loss = objective.contrastive(a, b).value
        loss.backward()
        optim.step()
    final = float(objective.contrastive(a, b).value.detach())
    assert final < initial, (initial, final)
