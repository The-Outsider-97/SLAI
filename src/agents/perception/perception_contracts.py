"""Typed runtime contracts for the SLAI perception subsystem.

The contracts normalize the boundary between modality-specific encoders,
multimodal fusion, learning objectives, and the perception trainer.  They do
not load configuration or construct models.  Their purpose is to make tensor
shape, modality identity, masking, and result semantics explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from .utils.perception_errors import (
    PerceptionContractError,
    PerceptionDimensionError,
    PerceptionShapeError,
)
from .utils.perception_helpers import tensor_summary


class Modality(str, Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"

    @classmethod
    def parse(cls, value: Union["Modality", str]) -> "Modality":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise PerceptionContractError(
                f"Unsupported modality: {value!r}.",
                details={"value": value, "supported": [item.value for item in cls]},
                cause=exc,
            ) from exc


class PoolingStrategy(str, Enum):
    CLS = "cls"
    MEAN = "mean"
    MASKED_MEAN = "masked_mean"

    @classmethod
    def parse(cls, value: Union["PoolingStrategy", str]) -> "PoolingStrategy":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise PerceptionContractError(
                f"Unsupported pooling strategy: {value!r}.",
                details={"value": value, "supported": [item.value for item in cls]},
                cause=exc,
            ) from exc


def _metadata_copy(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(metadata or {})


@dataclass(frozen=True)
class ModalityRepresentation:
    """Canonical representation emitted by one modality pipeline.

    ``pooled`` is always ``(B,D)`` and is the stable cross-modal interface.
    ``sequence`` is optional and, when present, must be ``(B,L,D)``.
    """

    modality: Union[Modality, str]
    pooled: torch.Tensor
    sequence: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    source_shape: Optional[Tuple[int, ...]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modality = Modality.parse(self.modality)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

        if not isinstance(self.pooled, torch.Tensor) or self.pooled.dim() != 2:
            raise PerceptionShapeError(
                "ModalityRepresentation.pooled must have shape (B, D).",
                component="perception_contracts",
                details={"pooled": tensor_summary(self.pooled) if isinstance(self.pooled, torch.Tensor) else type(self.pooled).__name__},
            )

        if self.sequence is not None:
            if not isinstance(self.sequence, torch.Tensor) or self.sequence.dim() != 3:
                raise PerceptionShapeError(
                    "ModalityRepresentation.sequence must have shape (B, L, D).",
                    component="perception_contracts",
                    details={"sequence": tensor_summary(self.sequence) if isinstance(self.sequence, torch.Tensor) else type(self.sequence).__name__},
                )
            if self.sequence.size(0) != self.pooled.size(0):
                raise PerceptionDimensionError(
                    "Sequence and pooled batch sizes must match.",
                    component="perception_contracts",
                    details={"sequence_batch": self.sequence.size(0), "pooled_batch": self.pooled.size(0)},
                )
            if self.sequence.size(-1) != self.pooled.size(-1):
                raise PerceptionDimensionError(
                    "Sequence and pooled embedding dimensions must match.",
                    component="perception_contracts",
                    details={"sequence_dim": self.sequence.size(-1), "pooled_dim": self.pooled.size(-1)},
                )

        if self.attention_mask is not None:
            if self.sequence is None:
                raise PerceptionContractError(
                    "attention_mask requires a sequence representation.",
                    details={"modality": modality.value},
                )
            if self.attention_mask.dim() != 2:
                raise PerceptionShapeError(
                    "attention_mask must have shape (B, L).",
                    component="perception_contracts",
                    details={"mask": tensor_summary(self.attention_mask)},
                )
            expected = (self.sequence.size(0), self.sequence.size(1))
            if tuple(self.attention_mask.shape) != expected:
                raise PerceptionDimensionError(
                    "attention_mask does not match the sequence dimensions.",
                    component="perception_contracts",
                    details={"mask_shape": list(self.attention_mask.shape), "expected": list(expected)},
                )

    @property
    def batch_size(self) -> int:
        return int(self.pooled.size(0))

    @property
    def embedding_dim(self) -> int:
        return int(self.pooled.size(-1))

    def to(self, device: Union[str, torch.device]) -> "ModalityRepresentation":
        return ModalityRepresentation(
            modality=self.modality,
            pooled=self.pooled.to(device),
            sequence=self.sequence.to(device) if self.sequence is not None else None,
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            source_shape=self.source_shape,
            metadata=self.metadata,
        )

    def detach(self, *, cpu: bool = False) -> "ModalityRepresentation":
        def _detach(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            result = tensor.detach()
            return result.cpu() if cpu else result

        return ModalityRepresentation(
            modality=self.modality,
            pooled=_detach(self.pooled),
            sequence=_detach(self.sequence),
            attention_mask=_detach(self.attention_mask),
            source_shape=self.source_shape,
            metadata=self.metadata,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "pooled": tensor_summary(self.pooled),
            "sequence": tensor_summary(self.sequence),
            "attention_mask": tensor_summary(self.attention_mask),
            "source_shape": list(self.source_shape) if self.source_shape is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FusedRepresentation:
    """Fixed-width multimodal representation produced by ``PerceptionFusion``."""

    pooled: torch.Tensor
    tokens: torch.Tensor
    modalities: Tuple[Union[Modality, str], ...]
    presence_mask: torch.Tensor
    modality_embeddings: Mapping[Union[Modality, str], torch.Tensor] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modalities = tuple(Modality.parse(item) for item in self.modalities)
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))
        normalized_embeddings = {
            Modality.parse(key): value for key, value in self.modality_embeddings.items()
        }
        object.__setattr__(self, "modality_embeddings", normalized_embeddings)

        if self.pooled.dim() != 2:
            raise PerceptionShapeError(
                "FusedRepresentation.pooled must have shape (B, D).",
                component="perception_contracts",
                details={"pooled": tensor_summary(self.pooled)},
            )
        if self.tokens.dim() != 3:
            raise PerceptionShapeError(
                "FusedRepresentation.tokens must have shape (B, M, D).",
                component="perception_contracts",
                details={"tokens": tensor_summary(self.tokens)},
            )
        if self.presence_mask.dim() != 2:
            raise PerceptionShapeError(
                "FusedRepresentation.presence_mask must have shape (B, M).",
                component="perception_contracts",
                details={"presence_mask": tensor_summary(self.presence_mask)},
            )
        if self.tokens.size(0) != self.pooled.size(0):
            raise PerceptionDimensionError(
                "Fused tokens and pooled batch sizes must match.",
                component="perception_contracts",
            )
        if self.tokens.size(-1) != self.pooled.size(-1):
            raise PerceptionDimensionError(
                "Fused tokens and pooled embedding dimensions must match.",
                component="perception_contracts",
            )
        if self.tokens.size(1) != len(modalities):
            raise PerceptionDimensionError(
                "The modality slot count must match fused token width.",
                component="perception_contracts",
                details={"slots": self.tokens.size(1), "modalities": len(modalities)},
            )
        if tuple(self.presence_mask.shape) != tuple(self.tokens.shape[:2]):
            raise PerceptionDimensionError(
                "presence_mask must match the fused token batch and modality dimensions.",
                component="perception_contracts",
            )

    @property
    def batch_size(self) -> int:
        return int(self.pooled.size(0))

    @property
    def embedding_dim(self) -> int:
        return int(self.pooled.size(-1))

    def summary(self) -> Dict[str, Any]:
        return {
            "pooled": tensor_summary(self.pooled),
            "tokens": tensor_summary(self.tokens),
            "modalities": [item.value for item in self.modalities],
            "presence_mask": tensor_summary(self.presence_mask),
            "present_modalities": [
                item.value
                for index, item in enumerate(self.modalities)
                if bool(self.presence_mask[:, index].any().item())
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MaskedPrediction:
    """Prediction/target/mask contract for masked-modality objectives."""

    modality: Union[Modality, str]
    predictions: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    representation: Optional[ModalityRepresentation] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modality = Modality.parse(self.modality)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))
        if self.mask.dtype != torch.bool:
            object.__setattr__(self, "mask", self.mask.to(dtype=torch.bool))

        if self.predictions.dim() < 2 or self.targets.dim() < 2 or self.mask.dim() != 2:
            raise PerceptionShapeError(
                "MaskedPrediction requires batched predictions/targets and a (B,L) mask.",
                component="perception_contracts",
                details={
                    "predictions": tensor_summary(self.predictions),
                    "targets": tensor_summary(self.targets),
                    "mask": tensor_summary(self.mask),
                },
            )
        batch = self.targets.size(0)
        if self.predictions.size(0) != batch or self.mask.size(0) != batch:
            raise PerceptionDimensionError(
                "Masked prediction batch dimensions must match.",
                component="perception_contracts",
            )
        if self.targets.size(1) != self.mask.size(1) or self.predictions.size(1) != self.mask.size(1):
            raise PerceptionDimensionError(
                "Masked prediction sequence/patch dimensions must match the mask.",
                component="perception_contracts",
                details={
                    "prediction_length": self.predictions.size(1),
                    "target_length": self.targets.size(1),
                    "mask_length": self.mask.size(1),
                },
            )


@dataclass(frozen=True)
class ObjectiveLoss:
    """Named scalar objective with optional component breakdown."""

    name: str
    value: torch.Tensor
    components: Mapping[str, torch.Tensor] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.value, torch.Tensor) or self.value.numel() != 1:
            raise PerceptionContractError(
                "ObjectiveLoss.value must be a scalar torch.Tensor.",
                details={"value": tensor_summary(self.value) if isinstance(self.value, torch.Tensor) else type(self.value).__name__},
            )
        object.__setattr__(self, "components", dict(self.components))
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    def detached_metrics(self) -> Dict[str, float]:
        metrics = {self.name: float(self.value.detach().item())}
        for key, value in self.components.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[key] = float(value.detach().item())
        return metrics


@dataclass(frozen=True)
class TrainingStepResult:
    """Serializable summary of one optimizer step."""

    objective: str
    loss: float
    global_step: int
    grad_norm: Optional[float] = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "loss": float(self.loss),
            "global_step": int(self.global_step),
            "grad_norm": None if self.grad_norm is None else float(self.grad_norm),
            "metrics": {str(key): float(value) for key, value in self.metrics.items()},
            "skipped": bool(self.skipped),
        }


__all__ = [
    "Modality",
    "PoolingStrategy",
    "ModalityRepresentation",
    "FusedRepresentation",
    "MaskedPrediction",
    "ObjectiveLoss",
    "TrainingStepResult",
]
