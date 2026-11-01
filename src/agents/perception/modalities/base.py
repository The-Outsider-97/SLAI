"""Base modality pipeline contract for SLAI perception.

A modality pipeline is an internal ``nn.Module`` boundary, not an SLAI agent.
It owns one modality's encoder and optional reconstructor/decoder, normalizes
that modality's outputs to ``ModalityRepresentation``, and exposes hooks used by
masked-modality and temporal training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn

from ..perception_contracts import MaskedPrediction, Modality, ModalityRepresentation, PoolingStrategy
from ..utils.perception_errors import *
from ..utils.perception_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Base Perception Modality")
printer = PrettyPrinter()


class BasePerceptionModality(nn.Module, ABC):
    """Common internal interface implemented by text, vision, and audio pipelines."""

    def __init__(self, *,
        modality: Union[Modality, str], encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.modality = Modality.parse(modality)
        self.config = dict(config or {})
        self.encoder = encoder
        self.decoder = decoder

        inferred_device = module_device(encoder, fallback="cpu")
        self.device = resolve_torch_device(device or inferred_device)
        self.encoder.to(self.device)
        if self.decoder is not None:
            self.decoder.to(self.device)

        self.embed_dim = int(getattr(self.encoder, "embed_dim", 0) or 0)
        if self.embed_dim <= 0:
            raise PerceptionConfigurationError(
                "The modality encoder must expose a positive 'embed_dim'.",
                component=f"{self.modality.value}_perception",
                details={"encoder_type": type(self.encoder).__name__},
            )

        self.pooling = PoolingStrategy.parse(self.config.get("pooling", "cls"))

        logger.info(f"Base perception modality initialized with {self.pooling}")

    @property
    def decoder_enabled(self) -> bool:
        return self.decoder is not None

    def _normalize_style_id(self, style_id: Optional[Any], batch_size: int) -> Optional[torch.Tensor]:
        printer.status("STYLE ID", "Normalizing...", "info")
        return normalize_style_id(style_id, batch_size=batch_size, device=self.device)

    def _build_representation(
        self,
        encoded: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        source_shape: Optional[tuple[int, ...]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        pooling: Optional[Union[PoolingStrategy, str]] = None,
    ) -> ModalityRepresentation:
        printer.status("REPRESENTATION", "Building representation...", "info")
        if encoded.dim() not in (2, 3):
            raise PerceptionConfigurationError(
                "Encoder output must be rank 2 or rank 3 to satisfy the perception contract.",
                component=f"{self.modality.value}_perception",
                details={"shape": list(encoded.shape), "encoder_type": type(self.encoder).__name__},
            )
        ensure_last_dimension(encoded, self.embed_dim, "encoded", component=f"{self.modality.value}_perception")

        strategy = PoolingStrategy.parse(pooling or self.pooling)
        normalized_mask = None
        if encoded.dim() == 3 and attention_mask is not None:
            normalized_mask = normalize_attention_mask(
                attention_mask,
                batch_size=encoded.size(0),
                seq_len=encoded.size(1),
                device=encoded.device,
            )

        pooled = pool_encoded(encoded, strategy=strategy.value, attention_mask=normalized_mask)
        sequence = encoded if encoded.dim() == 3 else None

        merged_metadata = {
            "encoder": type(self.encoder).__name__,
            "encoder_type": getattr(self.encoder, "encoder_type", None),
            "pooling": strategy.value,
        }
        merged_metadata.update(dict(metadata or {}))

        return ModalityRepresentation(
            modality=self.modality,
            pooled=pooled,
            sequence=sequence,
            attention_mask=normalized_mask,
            source_shape=source_shape,
            metadata=merged_metadata,
        )

    def require_decoder(self) -> nn.Module:
        if self.decoder is None:
            raise UnsupportedModalityOperationError(
                f"{self.modality.value} decoding/reconstruction is disabled for this pipeline.",
                component=f"{self.modality.value}_perception",
                remediation="Construct the modality pipeline with an explicit compatible decoder or enable_decoder=True after the lower-level decoder contract is validated.",
            )
        return self.decoder

    def forward(self, payload: Any, **kwargs: Any) -> ModalityRepresentation:
        return self.encode(payload, **kwargs)

    @abstractmethod
    def encode(self, payload: Any, **kwargs: Any) -> ModalityRepresentation:
        """Encode one batch into the canonical representation contract."""
        raise NotImplementedError

    @abstractmethod
    def masked_prediction(
        self,
        payload: Any,
        *,
        mask_ratio: float = 0.15,
        **kwargs: Any,
    ) -> MaskedPrediction:
        """Produce prediction/target/mask tensors for masked-modality learning."""
        raise NotImplementedError

    def encode_temporal(self, sequence_data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return ``(B,T,D)`` temporal embeddings when a modality supports sequences."""
        raise UnsupportedModalityOperationError(
            f"Temporal encoding is not supported for {self.modality.value}.",
            component=f"{self.modality.value}_perception",
        )

    @abstractmethod
    def reconstruct(self, representation: Union[ModalityRepresentation, torch.Tensor], **kwargs: Any) -> Any:
        """Reconstruct or decode from a latent representation when enabled."""
        raise NotImplementedError


__all__ = ["BasePerceptionModality"]
