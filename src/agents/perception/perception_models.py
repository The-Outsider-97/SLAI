"""Trainable model views over the canonical Perception subsystem.

Neither view creates encoders, projectors, fusion layers, or an optimizer.
All learned parameters are owned by the existing modality pipelines, fusion and
PerceptionObjectives; views are deliberately held in an ordinary dict on the
PerceptionAgent to avoid duplicate checkpoint registration.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modalities.base import BasePerceptionModality
from .perception_contracts import FusedRepresentation, Modality, ModalityRepresentation, ObjectiveLoss
from .perception_fusion import PerceptionFusion
from .perception_objectives import PerceptionObjectives
from .utils.perception_errors import (
    ModalityInputError, PerceptionDimensionError, PerceptionTrainingError,
    UnsupportedModalityOperationError,
)


class MultimodalPerception(nn.Module):
    """Shared-parameter task-oriented multimodal fusion and interpretation.

    Accepts raw modality payloads or existing ModalityRepresentation objects.
    Missing modalities are handled by PerceptionFusion's canonical presence mask.
    The optional named downstream head is owned by the agent/trainer.
    """

    MODEL_NAME = "MultimodalPerception"

    def __init__(
        self,
        modalities: Mapping[Union[str, Modality], BasePerceptionModality],
        fusion: PerceptionFusion,
        task_heads: Optional[nn.ModuleDict] = None,
    ) -> None:
        super().__init__()
        self.modalities = nn.ModuleDict({Modality.parse(k).value: v for k, v in modalities.items()})
        self.fusion = fusion
        self.task_heads = task_heads if task_heads is not None else nn.ModuleDict()

    def encode(self, inputs: Mapping[Union[str, Modality], Any]) -> dict[Modality, ModalityRepresentation]:
        if not isinstance(inputs, Mapping) or not inputs:
            raise ModalityInputError("MultimodalPerception requires a non-empty modality mapping.")
        representations: dict[Modality, ModalityRepresentation] = {}
        for raw_modality, payload in inputs.items():
            modality = Modality.parse(raw_modality)
            if payload is None:
                continue
            if modality.value not in self.modalities:
                raise UnsupportedModalityOperationError(
                    f"Modality {modality.value!r} is not registered with MultimodalPerception."
                )
            if isinstance(payload, ModalityRepresentation):
                if payload.modality != modality:
                    raise ModalityInputError("Representation modality disagrees with input key.")
                representations[modality] = payload
            else:
                representations[modality] = self.modalities[modality.value].encode(payload)
        if not representations:
            raise ModalityInputError("At least one non-null modality is required.")
        return representations

    def forward(
        self,
        inputs: Mapping[Union[str, Modality], Any],
        *,
        downstream_head: Optional[str] = None,
    ) -> Union[FusedRepresentation, torch.Tensor]:
        fused = self.fusion(self.encode(inputs))
        if downstream_head is None:
            return fused
        if downstream_head not in self.task_heads:
            raise PerceptionTrainingError(
                "Unknown downstream head.", details={"head": downstream_head}
            )
        return self.task_heads[downstream_head](fused.pooled)


class MultimodalContrastiveEncoder(nn.Module):
    """Shared-parameter, L2-normalized cross-modal embedding model.

    Reuses PerceptionObjectives' trained projection heads and loss mathematics;
    its embeddings are comparable across only those modality pairs on which
    alignment training and independent retrieval evaluation have succeeded.
    """

    MODEL_NAME = "MultimodalContrastiveEncoder"

    def __init__(
        self,
        modalities: Mapping[Union[str, Modality], BasePerceptionModality],
        objectives: PerceptionObjectives,
    ) -> None:
        super().__init__()
        self.modalities = nn.ModuleDict({Modality.parse(k).value: v for k, v in modalities.items()})
        self.objectives = objectives

    def forward(
        self,
        inputs: Mapping[Union[str, Modality], Any],
    ) -> dict[Modality, torch.Tensor]:
        if not isinstance(inputs, Mapping) or not inputs:
            raise ModalityInputError("Contrastive inference requires a modality mapping.")
        encoded: dict[Modality, torch.Tensor] = {}
        for raw_modality, payload in inputs.items():
            modality = Modality.parse(raw_modality)
            if payload is None:
                continue
            if modality.value not in self.modalities:
                raise UnsupportedModalityOperationError(
                    f"Modality {modality.value!r} is unavailable for contrastive encoding."
                )
            if isinstance(payload, ModalityRepresentation):
                if payload.modality != modality:
                    raise ModalityInputError("Representation modality disagrees with input key.")
                representation = payload
            else:
                representation = self.modalities[modality.value].encode(payload)
            encoded[modality] = self.objectives.embed(representation)
        if not encoded:
            raise ModalityInputError("At least one non-null modality is required.")
        return encoded

    @staticmethod
    def similarity(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.ndim != 2 or second.ndim != 2 or first.size(1) != second.size(1):
            raise PerceptionDimensionError("Similarity expects (N,D) and (M,D) tensors with matching D.")
        if first.device != second.device:
            raise PerceptionDimensionError("Similarity embeddings must share a device.")
        if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
            raise ModalityInputError("Similarity inputs must be finite.")
        return F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T

    def contrastive_loss(
        self,
        first: ModalityRepresentation,
        second: ModalityRepresentation,
        *,
        group_ids: Optional[list[str]] = None,
    ) -> ObjectiveLoss:
        return self.objectives.contrastive(first, second, symmetric=True, group_ids=group_ids)


__all__ = ["MultimodalPerception", "MultimodalContrastiveEncoder"]
