"""Learning objectives for the SLAI perception subsystem.

This module is the single owner of perception loss mathematics.  Modality
pipelines prepare predictions/targets and representations; the objective layer
computes masked reconstruction, cross-modal contrastive alignment, and temporal
coherence without owning optimizer state or agent orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.config_loader import get_config_section, load_global_config
from .utils.perception_helpers import *
from .utils.perception_errors import *
from .perception_contracts import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Perception Objectives")
printer = PrettyPrinter()


class PerceptionObjectives(nn.Module):
    """Masked reconstruction, contrastive alignment, and temporal objectives."""

    VALID_TEMPORAL_LOSSES: Tuple[str, ...] = ("mse", "contrastive", "hybrid")

    def __init__(
        self,
        *,
        input_dims: Optional[Mapping[Union[Modality, str], int]] = None,
        contrastive_projection_dim: int = 256,
        contrastive_temperature: float = 0.07,
        symmetric_contrastive: bool = False,
        temporal_loss_type: str = "hybrid",
        temporal_max_scale: int = 3,
        temporal_temperature: float = 0.1,
        temporal_mse_weight: float = 1.0,
        temporal_contrastive_weight: float = 1.0,
    ) -> None:
        super().__init__()
        resolved_dims = input_dims or {
            Modality.TEXT: 512,
            Modality.VISION: 512,
            Modality.AUDIO: 512,
        }
        self.input_dims = {Modality.parse(key): int(value) for key, value in resolved_dims.items()}
        if not self.input_dims or any(value <= 0 for value in self.input_dims.values()):
            raise PerceptionConfigurationError(
                "Objective input dimensions must be positive.",
                component="perception_objectives",
                details={"input_dims": self.input_dims},
            )

        self.contrastive_projection_dim = int(contrastive_projection_dim)
        if self.contrastive_projection_dim <= 0:
            raise PerceptionConfigurationError(
                "contrastive_projection_dim must be positive.",
                component="perception_objectives",
            )
        self.contrastive_temperature = float(contrastive_temperature)
        if self.contrastive_temperature <= 0.0:
            raise PerceptionConfigurationError(
                "contrastive_temperature must be > 0.",
                component="perception_objectives",
            )
        self.symmetric_contrastive = bool(symmetric_contrastive)

        self.temporal_loss_type = str(temporal_loss_type).strip().lower()
        ensure_one_of(
            self.temporal_loss_type,
            self.VALID_TEMPORAL_LOSSES,
            "temporal_loss_type",
            component="perception_objectives",
        )
        self.temporal_max_scale = int(temporal_max_scale)
        if self.temporal_max_scale < 1:
            raise PerceptionConfigurationError(
                "temporal_max_scale must be >= 1.",
                component="perception_objectives",
            )
        self.temporal_temperature = float(temporal_temperature)
        if self.temporal_temperature <= 0.0:
            raise PerceptionConfigurationError(
                "temporal_temperature must be > 0.",
                component="perception_objectives",
            )
        self.temporal_mse_weight = float(temporal_mse_weight)
        self.temporal_contrastive_weight = float(temporal_contrastive_weight)
        if self.temporal_mse_weight < 0.0 or self.temporal_contrastive_weight < 0.0:
            raise PerceptionConfigurationError(
                "Temporal objective weights must be non-negative.",
                component="perception_objectives",
            )

        self.contrastive_projectors = nn.ModuleDict(
            {
                modality.value: nn.Linear(input_dim, self.contrastive_projection_dim)
                for modality, input_dim in self.input_dims.items()
            }
        )

    def masked_reconstruction(self, prediction: MaskedPrediction) -> ObjectiveLoss:
        """Compute MLM cross-entropy or masked patch/frame MSE."""

        if not isinstance(prediction, MaskedPrediction):
            raise PerceptionObjectiveError(
                "masked_reconstruction expects a MaskedPrediction contract.",
                details={"actual_type": type(prediction).__name__},
            )

        mask = prediction.mask
        if not bool(mask.any().item()):
            zero = differentiable_zero(prediction.predictions)
            return ObjectiveLoss(
                name=f"masked_{prediction.modality.value}",
                value=zero,
                components={"reconstruction": zero},
                metadata={"masked_count": 0},
            )

        if prediction.modality is Modality.TEXT:
            if prediction.targets.dim() != 2 or prediction.predictions.dim() != 3:
                raise PerceptionDimensionError(
                    "Text masked reconstruction expects targets (B,L) and logits (B,L,V).",
                    component="perception_objectives",
                    details={
                        "prediction_shape": list(prediction.predictions.shape),
                        "target_shape": list(prediction.targets.shape),
                    },
                )
            masked_logits = prediction.predictions[mask]
            masked_targets = prediction.targets[mask].long()
            loss = F.cross_entropy(masked_logits, masked_targets)
        else:
            if tuple(prediction.predictions.shape) != tuple(prediction.targets.shape):
                raise PerceptionDimensionError(
                    "Vision/audio masked predictions and targets must have identical shapes.",
                    component="perception_objectives",
                    details={
                        "prediction_shape": list(prediction.predictions.shape),
                        "target_shape": list(prediction.targets.shape),
                    },
                )
            expanded = mask.unsqueeze(-1).expand_as(prediction.targets)
            loss = F.mse_loss(
                prediction.predictions[expanded],
                prediction.targets[expanded],
            )

        ensure_finite_tensor(loss, "masked_reconstruction_loss", component="perception_objectives")
        return ObjectiveLoss(
            name=f"masked_{prediction.modality.value}",
            value=loss,
            components={"reconstruction": loss},
            metadata={"masked_count": int(mask.sum().item())},
        )

    def _project(self, representation: ModalityRepresentation) -> torch.Tensor:
        modality = representation.modality
        if modality not in self.input_dims:
            raise PerceptionObjectiveError(
                f"No contrastive projector is configured for {modality.value}.",
                details={"configured": [item.value for item in self.input_dims]},
            )
        expected = self.input_dims[modality]
        if representation.embedding_dim != expected:
            raise PerceptionDimensionError(
                "Representation dimension does not match the contrastive projector.",
                component="perception_objectives",
                details={"modality": modality.value, "actual": representation.embedding_dim, "expected": expected},
            )
        return self.contrastive_projectors[modality.value](representation.pooled)

    def contrastive(
        self,
        first: ModalityRepresentation,
        second: ModalityRepresentation,
        *,
        symmetric: Optional[bool] = None,
    ) -> ObjectiveLoss:
        """Compute paired cross-modal InfoNCE over batch-aligned examples."""

        if first.modality == second.modality:
            raise PerceptionObjectiveError(
                "Cross-modal contrastive learning requires two different modalities.",
                details={"modality": first.modality.value},
            )
        if first.batch_size != second.batch_size:
            raise PerceptionDimensionError(
                "Contrastive representations must have the same batch size.",
                component="perception_objectives",
                details={"first_batch": first.batch_size, "second_batch": second.batch_size},
            )
        if first.batch_size < 2:
            raise PerceptionObjectiveError(
                "Contrastive learning requires batch_size >= 2 to provide negatives.",
                details={"batch_size": first.batch_size},
            )

        first_projected = F.normalize(self._project(first), p=2, dim=-1)
        second_projected = F.normalize(self._project(second), p=2, dim=-1)
        logits = torch.matmul(first_projected, second_projected.transpose(0, 1))
        logits = logits / self.contrastive_temperature
        labels = torch.arange(first.batch_size, device=logits.device)

        forward_loss = F.cross_entropy(logits, labels)
        use_symmetric = self.symmetric_contrastive if symmetric is None else bool(symmetric)
        if use_symmetric:
            reverse_loss = F.cross_entropy(logits.transpose(0, 1), labels)
            loss = 0.5 * (forward_loss + reverse_loss)
            components = {
                "forward_contrastive": forward_loss,
                "reverse_contrastive": reverse_loss,
            }
        else:
            loss = forward_loss
            components = {"forward_contrastive": forward_loss}

        ensure_finite_tensor(loss, "contrastive_loss", component="perception_objectives")
        return ObjectiveLoss(
            name=f"contrastive_{first.modality.value}_{second.modality.value}",
            value=loss,
            components=components,
            metadata={"symmetric": use_symmetric, "batch_size": first.batch_size},
        )

    def temporal_coherence(
        self,
        sequence_embeddings: torch.Tensor,
        *,
        loss_type: Optional[str] = None,
    ) -> ObjectiveLoss:
        """Compute multi-scale smoothness and adjacent-pair contrastive coherence.

        ``sequence_embeddings`` must have shape ``(B,T,D)``.  The contrastive
        component treats each adjacent pair as a positive and all other aligned
        adjacent pairs in the flattened batch/time set as negatives.
        """

        if not isinstance(sequence_embeddings, torch.Tensor) or sequence_embeddings.dim() != 3:
            raise PerceptionObjectiveError(
                "Temporal coherence expects sequence embeddings with shape (B,T,D).",
                details={"shape": list(sequence_embeddings.shape) if isinstance(sequence_embeddings, torch.Tensor) else None},
            )
        active_type = self.temporal_loss_type if loss_type is None else str(loss_type).strip().lower()
        ensure_one_of(
            active_type,
            self.VALID_TEMPORAL_LOSSES,
            "loss_type",
            component="perception_objectives",
        )

        _, timesteps, _ = sequence_embeddings.shape
        if timesteps < 2:
            zero = differentiable_zero(sequence_embeddings)
            return ObjectiveLoss(
                name="temporal_coherence",
                value=zero,
                components={"temporal_zero": zero},
                metadata={"timesteps": timesteps, "loss_type": active_type},
            )

        components: Dict[str, torch.Tensor] = {}
        total = differentiable_zero(sequence_embeddings)

        if active_type in ("mse", "hybrid"):
            scale_losses = []
            upper = min(self.temporal_max_scale, timesteps - 1)
            for scale in range(1, upper + 1):
                difference = sequence_embeddings[:, scale:, :] - sequence_embeddings[:, :-scale, :]
                scale_losses.append(difference.pow(2).mean())
            mse_loss = torch.stack(scale_losses).mean()
            components["temporal_mse"] = mse_loss
            total = total + self.temporal_mse_weight * mse_loss

        if active_type in ("contrastive", "hybrid"):
            anchors = sequence_embeddings[:, :-1, :].reshape(-1, sequence_embeddings.size(-1))
            positives = sequence_embeddings[:, 1:, :].reshape(-1, sequence_embeddings.size(-1))
            pair_count = anchors.size(0)
            if pair_count >= 2:
                anchors = F.normalize(anchors, p=2, dim=-1)
                positives = F.normalize(positives, p=2, dim=-1)
                logits = torch.matmul(anchors, positives.transpose(0, 1)) / self.temporal_temperature
                labels = torch.arange(pair_count, device=logits.device)
                contrastive_loss = F.cross_entropy(logits, labels)
            else:
                contrastive_loss = differentiable_zero(sequence_embeddings)
            components["temporal_contrastive"] = contrastive_loss
            total = total + self.temporal_contrastive_weight * contrastive_loss

        ensure_finite_tensor(total, "temporal_coherence_loss", component="perception_objectives")
        return ObjectiveLoss(
            name="temporal_coherence",
            value=total,
            components=components,
            metadata={"timesteps": timesteps, "loss_type": active_type},
        )

    def combine(
        self,
        losses: Mapping[str, Union[ObjectiveLoss, torch.Tensor]],
        *,
        weights: Optional[Mapping[str, float]] = None,
        name: str = "combined_perception",
    ) -> ObjectiveLoss:
        """Combine named scalar losses without detaching their autograd graphs."""

        if not losses:
            raise PerceptionObjectiveError("At least one loss is required for combination.")

        components: Dict[str, torch.Tensor] = {}
        total: Optional[torch.Tensor] = None
        for key, item in losses.items():
            value = item.value if isinstance(item, ObjectiveLoss) else item
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise PerceptionObjectiveError(
                    "Combined losses must be scalar tensors or ObjectiveLoss values.",
                    details={"loss": key, "actual_type": type(value).__name__},
                )
            weight = 1.0 if weights is None else float(weights.get(key, 1.0))
            if weight < 0.0:
                raise PerceptionObjectiveError(
                    "Objective weights must be non-negative.",
                    details={"loss": key, "weight": weight},
                )
            weighted = value * weight
            total = weighted if total is None else total + weighted
            components[str(key)] = value

        assert total is not None
        ensure_finite_tensor(total, "combined_loss", component="perception_objectives")
        return ObjectiveLoss(
            name=name,
            value=total,
            components=components,
            metadata={"weights": dict(weights or {})},
        )


__all__ = ["PerceptionObjectives"]
