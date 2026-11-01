"""Vision perception pipeline for SLAI.

The component composes the current ``main`` branch ``VisionEncoder`` and
optional ``VisionDecoder`` while enforcing the common perception representation
contract.  Transformer-backed vision encoding is explicitly configured to
return hidden representations because downstream perception/fusion requires
embedding tensors rather than the encoder's task-head output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoders.vision_decoder import VisionDecoder
from ..encoders.vision_encoder import VisionEncoder
from ..perception_contracts import *
from ..utils.perception_errors import *
from ..utils.perception_helpers import *
from .base import BasePerceptionModality
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Vision Perception")
printer = PrettyPrinter()


class VisionPerception(BasePerceptionModality):
    """Vertical vision-perception component: validate -> encode -> reconstruct."""

    def __init__(self, *,
        encoder: Optional[VisionEncoder] = None,
        decoder: Optional[VisionDecoder] = None,
        enable_decoder: bool = False,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        active_encoder = encoder or VisionEncoder()

        # Current main initializes VisionEncoder.transformer.return_hidden from
        # vision_encoder.return_hidden (default False).  Perception requires the
        # hidden embedding sequence, so the wrapper makes that contract explicit.
        if getattr(active_encoder, "encoder_type", None) == "transformer":
            if hasattr(active_encoder, "return_hidden"):
                active_encoder.return_hidden = True
            transformer = getattr(active_encoder, "transformer", None)
            if transformer is not None and hasattr(transformer, "return_hidden"):
                transformer.return_hidden = True

        active_decoder = decoder
        if active_decoder is None and enable_decoder:
            active_decoder = VisionDecoder()

        super().__init__(
            modality=Modality.VISION,
            encoder=active_encoder,
            decoder=active_decoder,
            config=config,
            device=device,
        )

        self.in_channels = int(getattr(self.encoder, "in_channels", 0) or 0)
        self.patch_size = int(getattr(self.encoder, "patch_size", 0) or 0)
        if self.in_channels <= 0 or self.patch_size <= 0:
            raise PerceptionConfigurationError(
                "Vision encoder must expose positive in_channels and patch_size.",
                component="vision_perception",
                details={"in_channels": self.in_channels, "patch_size": self.patch_size},
            )

        self.patch_dim = self.in_channels * (self.patch_size ** 2)
        self.prediction_head = nn.Linear(self.embed_dim, self.patch_dim).to(self.device)
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, device=self.device))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def _prepare_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, torch.Tensor):
            prepared: Dict[str, Any] = {"pixel_values": payload}
        elif isinstance(payload, Mapping):
            prepared = canonicalize_payload_aliases(Modality.VISION, payload)
        else:
            raise ModalityInputError(
                "Vision input must be a tensor or mapping containing pixel_values.",
                component="vision_perception",
                details={"input_type": type(payload).__name__},
            )

        if "pixel_values" not in prepared:
            raise ModalityInputError(
                "Vision payload does not contain canonical 'pixel_values'.",
                component="vision_perception",
                details={"keys": sorted(prepared.keys())},
            )

        pixels = prepared["pixel_values"]
        if not isinstance(pixels, torch.Tensor):
            pixels = torch.as_tensor(pixels)
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        if pixels.dim() != 4:
            raise ModalityInputError(
                "Vision pixel_values must have shape (B,C,H,W). Use encode_temporal for video tensors.",
                component="vision_perception",
                details={"shape": list(pixels.shape)},
            )
        if pixels.size(1) != self.in_channels:
            raise PerceptionDimensionError(
                "Vision channel count does not match the encoder contract.",
                component="vision_perception",
                details={"channels": pixels.size(1), "expected": self.in_channels},
            )
        if not torch.is_floating_point(pixels):
            pixels = pixels.float()
        prepared["pixel_values"] = pixels.to(self.device)
        return prepared

    def encode(
        self,
        payload: Any,
        *,
        style_id: Optional[Any] = None,
        pooling: Optional[str] = None,
    ) -> ModalityRepresentation:
        prepared = self._prepare_payload(payload)
        pixels = prepared["pixel_values"]
        style = self._normalize_style_id(
            prepared.get("style_id", style_id),
            pixels.size(0),
        )
        try:
            encoded = self.encoder(pixels, style_id=style)
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityEncodingError,
                "Vision encoder failed.",
                component="vision_perception",
                details={"input_shape": list(pixels.shape)},
            ) from exc

        return self._build_representation(
            encoded,
            source_shape=tuple(pixels.shape),
            metadata={"height": int(pixels.size(-2)), "width": int(pixels.size(-1))},
            pooling=pooling,
        )

    def _require_transformer_masking_contract(self) -> None:
        if getattr(self.encoder, "encoder_type", None) != "transformer":
            raise UnsupportedModalityOperationError(
                "Masked patch prediction currently requires the transformer's patch/projection contract.",
                component="vision_perception",
                details={"encoder_type": getattr(self.encoder, "encoder_type", None)},
            )
        required = ("extract_patches", "projection", "cls_token", "transformer")
        missing = [name for name in required if not hasattr(self.encoder, name)]
        if missing:
            raise PerceptionConfigurationError(
                "Vision transformer is missing required masked-modeling components.",
                component="vision_perception",
                details={"missing": missing},
            )

    def masked_prediction(
        self,
        payload: Any,
        *,
        mask_ratio: float = 0.15,
        style_id: Optional[Any] = None,
        **_: Any,
    ) -> MaskedPrediction:
        if not 0.0 <= float(mask_ratio) <= 1.0:
            raise ModalityInputError(
                "mask_ratio must be in [0, 1].",
                component="vision_perception",
                details={"mask_ratio": mask_ratio},
            )
        self._require_transformer_masking_contract()
        prepared = self._prepare_payload(payload)
        pixels = prepared["pixel_values"]
        style = self._normalize_style_id(
            prepared.get("style_id", style_id),
            pixels.size(0),
        )

        try:
            raw_patches = self.encoder.extract_patches(pixels)
            embeddings = torch.matmul(raw_patches, self.encoder.projection)
            if self.encoder.training and float(getattr(self.encoder, "dropout_rate", 0.0)) > 0.0:
                embeddings = F.dropout(
                    embeddings,
                    p=float(self.encoder.dropout_rate),
                    training=True,
                )

            batch_size, patch_count, _ = embeddings.shape
            mask = torch.rand(batch_size, patch_count, device=self.device) < float(mask_ratio)
            masked_embeddings = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.view(1, 1, -1).expand(batch_size, patch_count, -1),
                embeddings,
            )
            cls = self.encoder.cls_token.expand(batch_size, -1, -1)
            transformer_input = torch.cat([cls, masked_embeddings], dim=1)

            position_embed = getattr(self.encoder, "position_embed", None)
            if position_embed is not None:
                if position_embed.size(1) < transformer_input.size(1):
                    raise PerceptionDimensionError(
                        "Vision positional embedding table is shorter than the masked sequence.",
                        component="vision_perception",
                        details={
                            "position_capacity": position_embed.size(1),
                            "sequence_length": transformer_input.size(1),
                        },
                        remediation="Increase/interpolate positional embeddings in VisionEncoder before masked pretraining.",
                    )
                transformer_input = transformer_input + position_embed[:, : transformer_input.size(1), :]

            encoded = self.encoder.transformer(transformer_input, style_id=style)
            if encoded.dim() != 3 or encoded.size(1) != patch_count + 1:
                raise PerceptionDimensionError(
                    "Vision transformer did not return the expected hidden sequence.",
                    component="vision_perception",
                    details={"shape": list(encoded.shape), "expected_length": patch_count + 1},
                )
            predictions = self.prediction_head(encoded[:, 1:, :])
        except Exception as exc:
            if isinstance(exc, (PerceptionConfigurationError, PerceptionDimensionError)):
                raise
            raise wrap_exception(
                exc,
                ModalityEncodingError,
                "Masked vision prediction failed.",
                component="vision_perception",
            ) from exc

        representation = self._build_representation(
            encoded,
            source_shape=tuple(pixels.shape),
            metadata={"patch_count": patch_count, "masked_patches": int(mask.sum().item())},
        )
        return MaskedPrediction(
            modality=Modality.VISION,
            predictions=predictions,
            targets=raw_patches,
            mask=mask,
            representation=representation,
        )

    def encode_temporal( # type: ignore
        self,
        sequence_data: torch.Tensor,
        *,
        style_id: Optional[Any] = None,
    ) -> torch.Tensor:
        sequence_data = require_tensor(sequence_data, "sequence_data", component="vision_perception")
        if sequence_data.dim() != 5:
            raise ModalityInputError(
                "Vision temporal input must have shape (B,T,C,H,W).",
                component="vision_perception",
                details={"shape": list(sequence_data.shape)},
            )
        batch_size, timesteps = sequence_data.shape[:2]
        flat = sequence_data.reshape(-1, *sequence_data.shape[2:])
        style = None
        if style_id is not None:
            base_style = self._normalize_style_id(style_id, batch_size)
            assert base_style is not None
            style = base_style.repeat_interleave(timesteps)
        representation = self.encode(flat, style_id=style)
        return representation.pooled.reshape(batch_size, timesteps, self.embed_dim)

    def reconstruct(
        self,
        representation: Union[ModalityRepresentation, torch.Tensor],
        *,
        style_id: Optional[Any] = None,
        orig_shape: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        decoder = self.require_decoder()
        if isinstance(representation, ModalityRepresentation):
            if orig_shape is None and representation.source_shape is not None and len(representation.source_shape) >= 4:
                orig_shape = (
                    int(representation.source_shape[-2]),
                    int(representation.source_shape[-1]),
                )
            latent = representation.sequence if getattr(decoder, "decoder_type", "transformer") == "transformer" else representation.pooled
            if latent is None:
                latent = representation.pooled.unsqueeze(1)
        else:
            latent = require_tensor(representation, "representation", component="vision_perception")

        style = self._normalize_style_id(style_id, latent.size(0))
        try:
            return decoder(latent, style_id=style, orig_shape=orig_shape)
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityDecodingError,
                "Vision decoder failed.",
                component="vision_perception",
                details={"latent_shape": list(latent.shape)},
            ) from exc


__all__ = ["VisionPerception"]
