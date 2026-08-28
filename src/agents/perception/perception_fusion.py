"""Single-owner multimodal fusion for SLAI perception.

``PerceptionFusion`` replaces the competing fusion paths currently distributed
across the v2.2 perception agent, data-loader helpers, and task heads.  It
projects modality-specific pooled representations into one common dimension,
optionally performs modality-level self-attention, and emits a fixed-width
``FusedRepresentation`` even when one or more modalities are absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .perception_contracts import FusedRepresentation, Modality, ModalityRepresentation
from .utils.perception_errors import (
    PerceptionConfigurationError,
    PerceptionDimensionError,
    PerceptionFusionError,
    ensure_one_of,
)


class PerceptionFusion(nn.Module):
    """Project and fuse text, vision, and audio representations."""

    VALID_METHODS: Tuple[str, ...] = ("concat", "mean", "sum", "max")

    def __init__(
        self,
        *,
        input_dims: Optional[Mapping[Union[Modality, str], int]] = None,
        output_dim: int = 512,
        fusion_method: str = "concat",
        use_attention: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1,
        modality_order: Sequence[Union[Modality, str]] = (
            Modality.TEXT,
            Modality.VISION,
            Modality.AUDIO,
        ),
    ) -> None:
        super().__init__()

        resolved_dims = input_dims or {
            Modality.TEXT: output_dim,
            Modality.VISION: output_dim,
            Modality.AUDIO: output_dim,
        }
        normalized_dims: Dict[Modality, int] = {
            Modality.parse(key): int(value) for key, value in resolved_dims.items()
        }
        if not normalized_dims or any(value <= 0 for value in normalized_dims.values()):
            raise PerceptionConfigurationError(
                "Fusion input dimensions must be a non-empty mapping of positive integers.",
                component="perception_fusion",
                details={"input_dims": normalized_dims},
            )

        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise PerceptionConfigurationError(
                "output_dim must be positive.",
                component="perception_fusion",
                details={"output_dim": output_dim},
            )

        self.fusion_method = str(fusion_method).strip().lower()
        ensure_one_of(
            self.fusion_method,
            self.VALID_METHODS,
            "fusion_method",
            component="perception_fusion",
        )
        self.use_attention = bool(use_attention)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout <= 1.0:
            raise PerceptionConfigurationError(
                "Fusion dropout must be in [0,1].",
                component="perception_fusion",
                details={"dropout": dropout},
            )

        ordered = []
        seen = set()
        for item in modality_order:
            modality = Modality.parse(item)
            if modality in normalized_dims and modality not in seen:
                ordered.append(modality)
                seen.add(modality)
        for modality in normalized_dims:
            if modality not in seen:
                ordered.append(modality)
                seen.add(modality)
        self.modality_order = tuple(ordered)
        self.input_dims = normalized_dims

        self.projectors = nn.ModuleDict(
            {
                modality.value: nn.Linear(normalized_dims[modality], self.output_dim)
                for modality in self.modality_order
            }
        )
        self.project_norms = nn.ModuleDict(
            {
                modality.value: nn.LayerNorm(self.output_dim)
                for modality in self.modality_order
            }
        )

        if self.use_attention:
            if self.output_dim % self.num_heads != 0:
                raise PerceptionConfigurationError(
                    "Fusion output_dim must be divisible by num_heads when attention is enabled.",
                    component="perception_fusion",
                    details={"output_dim": self.output_dim, "num_heads": self.num_heads},
                )
            self.attention = nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(self.output_dim)
        else:
            self.attention = None
            self.attention_norm = None

        self.concat_projection = nn.Sequential(
            nn.Linear(self.output_dim * len(self.modality_order), self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def _normalize_representations(
        self,
        representations: Mapping[Union[Modality, str], ModalityRepresentation],
    ) -> Dict[Modality, ModalityRepresentation]:
        if not representations:
            raise PerceptionFusionError("At least one modality representation is required.")

        normalized: Dict[Modality, ModalityRepresentation] = {}
        batch_size: Optional[int] = None
        device: Optional[torch.device] = None
        for raw_key, representation in representations.items():
            key = Modality.parse(raw_key)
            if not isinstance(representation, ModalityRepresentation):
                raise PerceptionFusionError(
                    "Fusion inputs must be ModalityRepresentation instances.",
                    details={"modality": key.value, "actual_type": type(representation).__name__},
                )
            if representation.modality != key:
                raise PerceptionFusionError(
                    "Fusion mapping key and representation modality disagree.",
                    details={"key": key.value, "representation": representation.modality.value},
                )
            if key not in self.input_dims:
                raise PerceptionFusionError(
                    f"Fusion was not configured for modality '{key.value}'.",
                    details={"configured": [item.value for item in self.modality_order]},
                )
            expected_dim = self.input_dims[key]
            if representation.embedding_dim != expected_dim:
                raise PerceptionDimensionError(
                    "Modality representation dimension does not match its fusion projector.",
                    component="perception_fusion",
                    details={
                        "modality": key.value,
                        "actual": representation.embedding_dim,
                        "expected": expected_dim,
                    },
                )
            if batch_size is None:
                batch_size = representation.batch_size
                device = representation.pooled.device
            elif representation.batch_size != batch_size:
                raise PerceptionDimensionError(
                    "All fused modalities must have the same batch size.",
                    component="perception_fusion",
                    details={"expected": batch_size, "actual": representation.batch_size, "modality": key.value},
                )
            elif representation.pooled.device != device:
                raise PerceptionFusionError(
                    "All fused modalities must be on the same device.",
                    details={"expected_device": str(device), "actual_device": str(representation.pooled.device)},
                )
            normalized[key] = representation
        return normalized

    def forward(
        self,
        representations: Mapping[Union[Modality, str], ModalityRepresentation],
    ) -> FusedRepresentation:
        normalized = self._normalize_representations(representations)
        first = next(iter(normalized.values()))
        batch_size = first.batch_size
        device = first.pooled.device
        dtype = first.pooled.dtype

        slots = []
        presence_columns = []
        projected_present: Dict[Modality, torch.Tensor] = {}
        for modality in self.modality_order:
            representation = normalized.get(modality)
            if representation is None:
                slots.append(torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype))
                presence_columns.append(torch.zeros(batch_size, device=device, dtype=torch.bool))
                continue

            projected = self.projectors[modality.value](representation.pooled)
            projected = self.project_norms[modality.value](projected)
            projected_present[modality] = projected
            slots.append(projected)
            presence_columns.append(torch.ones(batch_size, device=device, dtype=torch.bool))

        tokens = torch.stack(slots, dim=1)
        presence_mask = torch.stack(presence_columns, dim=1)

        if self.attention is not None:
            attended, _ = self.attention(
                tokens,
                tokens,
                tokens,
                key_padding_mask=~presence_mask,
                need_weights=False,
            )
            tokens = self.attention_norm(tokens + attended)
            tokens = tokens * presence_mask.unsqueeze(-1).to(dtype=tokens.dtype)

        if self.fusion_method == "concat":
            pooled = self.concat_projection(tokens.reshape(batch_size, -1))
        elif self.fusion_method == "sum":
            pooled = tokens.sum(dim=1)
        elif self.fusion_method == "mean":
            denominator = presence_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=tokens.dtype)
            pooled = tokens.sum(dim=1) / denominator
        elif self.fusion_method == "max":
            fill_value = torch.finfo(tokens.dtype).min
            masked = tokens.masked_fill(~presence_mask.unsqueeze(-1), fill_value)
            pooled = masked.max(dim=1).values
        else:  # validated at construction; defensive guard for corrupted state
            raise PerceptionFusionError(
                f"Unsupported fusion method: {self.fusion_method!r}.",
            )

        pooled = self.output_norm(pooled)
        return FusedRepresentation(
            pooled=pooled,
            tokens=tokens,
            modalities=self.modality_order,
            presence_mask=presence_mask,
            modality_embeddings=projected_present,
            metadata={
                "fusion_method": self.fusion_method,
                "use_attention": self.use_attention,
                "present_modalities": [item.value for item in normalized],
            },
        )


__all__ = ["PerceptionFusion"]
