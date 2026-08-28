"""Shared contract for SLAI audio decoder fallback backends.

Fallback decoders are internal ``nn.Module`` components owned by
``AudioDecoder``.  They are not independently routable agents and do not own
optimizer, checkpoint, shared-memory, or perception-lifecycle state.

The base class centralizes only the behavior that is genuinely common to audio
reconstruction backends:

- configuration resolution;
- device resolution;
- latent-shape validation;
- target-length normalization;
- waveform contract validation;
- optional output bounding;
- freeze/unfreeze convenience helpers.

Backend-specific synthesis remains the responsibility of concrete subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.config_loader import get_config_section, load_global_config
from ...utils.perception_errors import *
from ...utils.perception_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Base Decoder")
printer = PrettyPrinter()


class BaseDecoder(nn.Module, ABC):
    """Common audio-reconstruction contract for fallback decoder backends.

    Parameters
    ----------
    decoder_type:
        Stable backend identifier used in error metadata and diagnostics.
    config_section:
        Optional backend-specific section in ``perception_config.yaml``.
    config:
        Optional complete perception configuration mapping.  When omitted, the
        existing perception config loader is used.
    decoder_config:
        Optional explicit ``audio_decoder`` section.  Supplying it is useful
        when ``AudioDecoder`` has already resolved configuration and should
        pass the exact same snapshot into a fallback backend.
    """

    VALID_OUTPUT_ACTIVATIONS = ("none", "tanh")

    def __init__(
        self,
        *,
        decoder_type: str,
        config_section: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        decoder_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()

        normalized_type = str(decoder_type).strip().lower()
        if not normalized_type:
            raise InvalidPerceptionConfigurationError(
                "decoder_type must be a non-empty string.",
                component="audio_decoder_fallback",
            )

        self.decoder_type = normalized_type
        self.component = f"audio_{normalized_type}_decoder"
        self.config_section = config_section
        self._config_injected = config is not None

        self.config = self._resolve_root_config(config)
        self.audio_config = self._resolve_section("audio_encoder")
        self.decoder_config = self._resolve_decoder_config(decoder_config)
        self.backend_config = (
            self._resolve_section(config_section)
            if config_section
            else {}
        )

        self.embed_dim = int(
            self.backend_config.get(
                "embed_dim",
                self.decoder_config.get(
                    "embed_dim",
                    self.config.get("embed_dim", 0),
                ),
            )
        )
        self.in_channels = int(
            self.backend_config.get(
                "in_channels",
                self.decoder_config.get(
                    "in_channels",
                    self.audio_config.get("in_channels", 1),
                ),
            )
        )
        self.audio_length = int(
            self.backend_config.get(
                "audio_length",
                self.decoder_config.get(
                    "audio_length",
                    self.audio_config.get("audio_length", 0),
                ),
            )
        )

        self.dropout_rate = float(
            self.backend_config.get(
                "dropout_rate",
                self.decoder_config.get(
                    "dropout_rate",
                    self.config.get("dropout_rate", 0.0),
                ),
            )
        )
        self.initializer = str(
            self.backend_config.get(
                "initializer",
                self.decoder_config.get(
                    "initializer",
                    self.config.get("initializer", "xavier_uniform"),
                ),
            )
        ).strip().lower()

        requested_device = self.backend_config.get(
            "device",
            self.decoder_config.get(
                "device",
                self.config.get("device", "cpu"),
            ),
        )
        self.device = resolve_torch_device(requested_device)

        output_activation = self.backend_config.get(
            "output_activation",
            self.decoder_config.get("output_activation", "none"),
        )
        self.output_activation = (
            "none"
            if output_activation is None
            else str(output_activation).strip().lower()
        )

        self._validate_base_config()

    @staticmethod
    def _resolve_root_config(
        config: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if config is None:
            loaded = load_global_config()
            if not isinstance(loaded, Mapping):
                raise InvalidPerceptionConfigurationError(
                    "Perception config loader must return a mapping.",
                    component="audio_decoder_fallback",
                    details={"actual_type": type(loaded).__name__},
                )
            return dict(loaded)

        if not isinstance(config, Mapping):
            raise InvalidPerceptionTypeError(
                "config must be a mapping when supplied.",
                component="audio_decoder_fallback",
                details={"actual_type": type(config).__name__},
            )
        return dict(config)

    def _resolve_section(self, name: Optional[str]) -> dict[str, Any]:
        if not name:
            return {}

        if name in self.config:
            value = self.config.get(name)
        elif self._config_injected:
            # An explicit config is an authoritative snapshot.  Do not mix it
            # with ambient loader state when a section is intentionally absent.
            value = None
        else:
            # Preserve compatibility with the existing config loader for callers
            # that did not inject a configuration snapshot.
            try:
                value = get_config_section(name)
            except Exception:
                value = None

        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise InvalidPerceptionConfigurationError(
                f"Configuration section '{name}' must be a mapping.",
                component=self.component,
                details={"section": name, "actual_type": type(value).__name__},
            )
        return dict(value)

    def _resolve_decoder_config(
        self,
        decoder_config: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if decoder_config is None:
            return self._resolve_section("audio_decoder")
        if not isinstance(decoder_config, Mapping):
            raise InvalidPerceptionTypeError(
                "decoder_config must be a mapping when supplied.",
                component=self.component,
                details={"actual_type": type(decoder_config).__name__},
            )
        return dict(decoder_config)

    def _validate_base_config(self) -> None:
        if self.embed_dim <= 0:
            raise InvalidPerceptionConfigurationError(
                "Audio fallback decoder requires embed_dim > 0.",
                component=self.component,
                details={"embed_dim": self.embed_dim},
            )
        if self.in_channels <= 0:
            raise InvalidPerceptionConfigurationError(
                "Audio fallback decoder requires in_channels > 0.",
                component=self.component,
                details={"in_channels": self.in_channels},
            )
        if self.audio_length <= 0:
            raise InvalidPerceptionConfigurationError(
                "Audio fallback decoder requires audio_length > 0.",
                component=self.component,
                details={"audio_length": self.audio_length},
            )
        if not 0.0 <= self.dropout_rate < 1.0:
            raise InvalidPerceptionConfigurationError(
                "dropout_rate must be in [0, 1).",
                component=self.component,
                details={"dropout_rate": self.dropout_rate},
            )

        ensure_one_of(
            self.output_activation,
            self.VALID_OUTPUT_ACTIVATIONS,
            "output_activation",
            component=self.component,
            exc_type=InvalidPerceptionConfigurationError,
        )

    def _live_device(self) -> torch.device:
        """Return the actual module device after any later ``module.to(...)`` call."""

        return module_device(self, fallback=self.device)

    def _prepare_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Normalize decoder input to ``(B, L, D)`` and validate its contract."""

        latent = require_tensor(latent, "latent", component=self.component)
        ensure_tensor_rank(latent, (2, 3), "latent", component=self.component)

        if not torch.is_floating_point(latent):
            raise InvalidPerceptionTypeError(
                "Audio decoder latent representations must use a floating dtype.",
                component=self.component,
                details={"dtype": str(latent.dtype)},
            )

        if latent.dim() == 2:
            latent = latent.unsqueeze(1)

        ensure_last_dimension(
            latent,
            self.embed_dim,
            "latent",
            component=self.component,
        )

        if latent.size(0) <= 0 or latent.size(1) <= 0:
            raise PerceptionShapeError(
                "Audio decoder latent batch and sequence dimensions must be non-empty.",
                component=self.component,
                details={"shape": list(latent.shape)},
            )

        return latent.to(device=self._live_device())

    def _resolve_target_length(self, target_length: Optional[int]) -> int:
        length = self.audio_length if target_length is None else int(target_length)
        if length <= 0:
            raise InvalidPerceptionValueError(
                "target_length must be > 0.",
                component=self.component,
                details={"target_length": target_length},
            )
        return length

    @staticmethod
    def _match_output_length(
        waveform: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Right-trim or right-pad a waveform without fabricating interpolation."""

        current = int(waveform.size(-1))
        if current == target_length:
            return waveform
        if current > target_length:
            return waveform[..., :target_length]
        return F.pad(waveform, (0, target_length - current))

    def _apply_output_activation(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(waveform)
        return waveform

    def _finalize_waveform(
        self,
        waveform: torch.Tensor,
        *,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Enforce the canonical audio decoder output ``(B, C, T)`` contract."""

        waveform = require_tensor(
            waveform,
            "waveform",
            component=self.component,
        )
        ensure_tensor_rank(
            waveform,
            3,
            "waveform",
            component=self.component,
        )

        if waveform.size(1) != self.in_channels:
            raise PerceptionDimensionError(
                "Decoded waveform channel count does not match the decoder contract.",
                component=self.component,
                details={
                    "shape": list(waveform.shape),
                    "channels": int(waveform.size(1)),
                    "expected_channels": self.in_channels,
                },
            )

        resolved_length = self._resolve_target_length(target_length)
        waveform = self._match_output_length(waveform, resolved_length)
        waveform = self._apply_output_activation(waveform)

        if not bool(torch.isfinite(waveform).all().item()):
            raise ModalityDecodingError(
                "Decoded waveform contains NaN or infinite values.",
                component=self.component,
                details={
                    "shape": list(waveform.shape),
                    "dtype": str(waveform.dtype),
                },
            )

        return waveform

    def freeze(self) -> "BaseDecoder":
        """Freeze every trainable parameter in this fallback backend."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        return self

    def unfreeze(self) -> "BaseDecoder":
        """Unfreeze every parameter in this fallback backend."""

        for parameter in self.parameters():
            parameter.requires_grad = True
        return self

    @abstractmethod
    def forward(
        self,
        latent: torch.Tensor,
        *,
        target_length: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode a latent representation into ``(B, C, T)`` audio."""

        raise NotImplementedError


__all__ = ["BaseDecoder"]
