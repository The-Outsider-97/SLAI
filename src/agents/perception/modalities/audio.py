"""Audio perception pipeline for SLAI.

The component composes the current ``main`` branch ``AudioEncoder`` and optional
``AudioDecoder`` behind the common modality contract.  Decoder construction is
opt-in because the current main configuration can give the encoder and decoder
different channel counts; when decoding is enabled this wrapper validates that
contract explicitly instead of silently reconstructing the wrong geometry.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoders.audio_decoder import AudioDecoder
from ..encoders.audio_encoder import AudioEncoder
from ..perception_contracts import *
from ..utils.perception_errors import *
from ..utils.perception_helpers import *
from .base import BasePerceptionModality
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Audio Percetion]")
printer = PrettyPrinter()


class AudioPerception(BasePerceptionModality):
    """Vertical audio-perception component: validate -> encode -> reconstruct."""

    def __init__(
        self,
        *,
        encoder: Optional[AudioEncoder] = None,
        decoder: Optional[AudioDecoder] = None,
        enable_decoder: bool = False,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        active_encoder = encoder or AudioEncoder()

        if getattr(active_encoder, "encoder_type", None) == "transformer":
            if hasattr(active_encoder, "return_hidden"):
                active_encoder.return_hidden = True
            transformer = getattr(active_encoder, "transformer", None)
            if transformer is not None and hasattr(transformer, "return_hidden"):
                transformer.return_hidden = True

        active_decoder = decoder
        if active_decoder is None and enable_decoder:
            active_decoder = AudioDecoder()

        super().__init__(
            modality=Modality.AUDIO,
            encoder=active_encoder,
            decoder=active_decoder,
            config=config,
            device=device,
        )

        self.in_channels = int(getattr(self.encoder, "in_channels", 0) or 0)
        self.patch_size = int(getattr(self.encoder, "patch_size", 0) or 0)
        if self.in_channels <= 0 or self.patch_size <= 0:
            raise PerceptionConfigurationError(
                "Audio encoder must expose positive in_channels and patch_size.",
                component="audio_perception",
                details={"in_channels": self.in_channels, "patch_size": self.patch_size},
            )

        self.patch_dim = self.in_channels * self.patch_size
        self.prediction_head = nn.Linear(self.embed_dim, self.patch_dim).to(self.device)
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, device=self.device))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        if self.decoder is not None:
            decoder_channels = int(getattr(self.decoder, "in_channels", self.in_channels))
            if decoder_channels != self.in_channels:
                raise PerceptionConfigurationError(
                    "Audio encoder and decoder channel contracts do not match.",
                    component="audio_perception",
                    details={
                        "encoder_in_channels": self.in_channels,
                        "decoder_in_channels": decoder_channels,
                    },
                    remediation="Align AudioDecoder.in_channels with audio_encoder.in_channels before enabling audio reconstruction.",
                )

    def _prepare_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, torch.Tensor):
            prepared: Dict[str, Any] = {"audio_values": payload}
        elif isinstance(payload, Mapping):
            prepared = canonicalize_payload_aliases(Modality.AUDIO, payload)
        else:
            raise ModalityInputError(
                "Audio input must be a tensor or mapping containing audio_values/waveform.",
                component="audio_perception",
                details={"input_type": type(payload).__name__},
            )

        if "audio_values" not in prepared:
            raise ModalityInputError(
                "Audio payload does not contain canonical 'audio_values'.",
                component="audio_perception",
                details={"keys": sorted(prepared.keys())},
            )

        audio = prepared["audio_values"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            # Preserve AudioEncoder.main semantics: rank-2 means (B,T) mono.
            audio = audio.unsqueeze(1)
        if audio.dim() != 3:
            raise ModalityInputError(
                "Audio values must have shape (B,T) or (B,C,T). Use encode_temporal for temporal batches.",
                component="audio_perception",
                details={"shape": list(audio.shape)},
            )
        if audio.size(1) != self.in_channels:
            raise PerceptionDimensionError(
                "Audio channel count does not match the encoder contract.",
                component="audio_perception",
                details={"channels": audio.size(1), "expected": self.in_channels},
            )
        if not torch.is_floating_point(audio):
            audio = audio.float()
        prepared["audio_values"] = audio.to(self.device)
        return prepared

    def encode(
        self,
        payload: Any,
        *,
        style_id: Optional[Any] = None,
        pooling: Optional[str] = None,
    ) -> ModalityRepresentation:
        prepared = self._prepare_payload(payload)
        audio = prepared["audio_values"]
        style = self._normalize_style_id(
            prepared.get("style_id", style_id),
            audio.size(0),
        )
        try:
            encoded = self.encoder(audio, style_id=style)
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityEncodingError,
                "Audio encoder failed.",
                component="audio_perception",
                details={"input_shape": list(audio.shape)},
            ) from exc

        return self._build_representation(
            encoded,
            source_shape=tuple(audio.shape),
            metadata={"sample_length": int(audio.size(-1))},
            pooling=pooling,
        )

    def _require_transformer_masking_contract(self) -> None:
        if getattr(self.encoder, "encoder_type", None) != "transformer":
            raise UnsupportedModalityOperationError(
                "Masked audio-patch prediction currently requires the transformer patch/projection contract.",
                component="audio_perception",
                details={"encoder_type": getattr(self.encoder, "encoder_type", None)},
            )
        required = ("extract_patches", "projection", "cls_token", "transformer")
        missing = [name for name in required if not hasattr(self.encoder, name)]
        if missing:
            raise PerceptionConfigurationError(
                "Audio transformer is missing required masked-modeling components.",
                component="audio_perception",
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
                component="audio_perception",
                details={"mask_ratio": mask_ratio},
            )
        self._require_transformer_masking_contract()
        prepared = self._prepare_payload(payload)
        audio = prepared["audio_values"]
        style = self._normalize_style_id(
            prepared.get("style_id", style_id),
            audio.size(0),
        )

        try:
            raw_patches = self.encoder.extract_patches(audio)
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
                        "Audio positional embedding table is shorter than the masked sequence.",
                        component="audio_perception",
                        details={
                            "position_capacity": position_embed.size(1),
                            "sequence_length": transformer_input.size(1),
                        },
                        remediation="Increase the AudioEncoder positional capacity before masked pretraining.",
                    )
                transformer_input = transformer_input + position_embed[:, : transformer_input.size(1), :]

            encoded = self.encoder.transformer(transformer_input, style_id=style)
            if encoded.dim() != 3 or encoded.size(1) != patch_count + 1:
                raise PerceptionDimensionError(
                    "Audio transformer did not return the expected hidden sequence.",
                    component="audio_perception",
                    details={"shape": list(encoded.shape), "expected_length": patch_count + 1},
                )
            predictions = self.prediction_head(encoded[:, 1:, :])
        except Exception as exc:
            if isinstance(exc, (PerceptionConfigurationError, PerceptionDimensionError)):
                raise
            raise wrap_exception(
                exc,
                ModalityEncodingError,
                "Masked audio prediction failed.",
                component="audio_perception",
            ) from exc

        representation = self._build_representation(
            encoded,
            source_shape=tuple(audio.shape),
            metadata={"patch_count": patch_count, "masked_patches": int(mask.sum().item())},
        )
        return MaskedPrediction(
            modality=Modality.AUDIO,
            predictions=predictions,
            targets=raw_patches,
            mask=mask,
            representation=representation,
        )

    def encode_temporal(
        self,
        sequence_data: torch.Tensor,
        *,
        style_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        sequence_data = require_tensor(sequence_data, "sequence_data", component="audio_perception")
        if sequence_data.dim() not in (3, 4):
            raise ModalityInputError(
                "Audio temporal input must have shape (B,T,S) or (B,T,C,S).",
                component="audio_perception",
                details={"shape": list(sequence_data.shape)},
            )
        batch_size, timesteps = sequence_data.shape[:2]
        if sequence_data.dim() == 3:
            flat = sequence_data.reshape(batch_size * timesteps, sequence_data.size(-1))
        else:
            flat = sequence_data.reshape(batch_size * timesteps, *sequence_data.shape[2:])

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
        **kwargs: Any,
    ) -> torch.Tensor:
        decoder = self.require_decoder()
        if isinstance(representation, ModalityRepresentation):
            latent = representation.sequence
            if latent is None:
                latent = representation.pooled.unsqueeze(1)
        else:
            latent = require_tensor(representation, "representation", component="audio_perception")
            if latent.dim() == 2:
                latent = latent.unsqueeze(1)
        if latent.dim() != 3 or latent.size(-1) != self.embed_dim:
            raise PerceptionDimensionError(
                "Audio decoder input must have shape (B,L,embed_dim).",
                component="audio_perception",
                details={"shape": list(latent.shape), "embed_dim": self.embed_dim},
            )
        style = self._normalize_style_id(style_id, latent.size(0))
        try:
            return decoder(latent, style_id=style)
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityDecodingError,
                "Audio decoder failed.",
                component="audio_perception",
                details={"latent_shape": list(latent.shape)},
            ) from exc


__all__ = ["AudioPerception"]
