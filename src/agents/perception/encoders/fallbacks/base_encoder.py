"""Shared contract for SLAI perception fallback encoders.

Fallback encoders are internal ``nn.Module`` components owned by the modality
encoders.  They are not agents and do not own optimizer, checkpoint,
PerceptionMemory, SharedMemory, or AgentFactory lifecycle state.

The base class centralizes only behavior that is genuinely common to the
fallback encoder backends:

- perception/model configuration resolution;
- modality-specific encoder-section resolution;
- device and dtype-safe input validation;
- output-representation validation;
- configured output activation;
- common parameter initialization;
- freeze/unfreeze convenience helpers.

Signal- or image-specific feature extraction remains the responsibility of the
concrete encoder implementation.
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

logger = get_logger("Base Encoder")
printer = PrettyPrinter()


class BaseEncoder(nn.Module, ABC):
    """Common contract for non-transformer perception encoder backends.

    Parameters
    ----------
    encoder_type:
        Stable backend identifier, e.g. ``"cnn"`` or ``"mfcc"``.
    modality:
        Perception modality owned by the concrete backend.  The shared CNN
        backend supports ``"audio"`` and ``"vision"``; MFCC is audio-only.
    config_section:
        Existing backend-specific section in ``perception_config.yaml``.
    config:
        Optional complete perception configuration snapshot.  When supplied,
        it is treated as authoritative and is not silently mixed with ambient
        loader state.
    encoder_config:
        Optional already-resolved modality encoder section.  This is the
        preferred integration path from ``AudioEncoder``/``VisionEncoder``.
    """

    VALID_MODALITIES = ("audio", "vision")
    VALID_OUTPUT_ACTIVATIONS = ("none", "sigmoid", "tanh")

    def __init__(
        self,
        *,
        encoder_type: str,
        modality: str,
        config_section: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        encoder_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()

        normalized_type = str(encoder_type).strip().lower()
        normalized_modality = str(modality).strip().lower()

        if not normalized_type:
            raise InvalidPerceptionConfigurationError(
                "encoder_type must be a non-empty string.",
                component="perception_encoder_fallback",
            )
        ensure_one_of(
            normalized_modality,
            self.VALID_MODALITIES,
            "modality",
            component="perception_encoder_fallback",
            exc_type=InvalidPerceptionConfigurationError,
        )

        self.encoder_type = normalized_type
        self.modality = normalized_modality
        self.component = f"{self.modality}_{self.encoder_type}_encoder"
        self.config_section = config_section
        self._config_injected = config is not None

        self.config = self._resolve_root_config(config)
        self.encoder_config = self._resolve_encoder_config(encoder_config)
        self.backend_config = (
            self._resolve_section(config_section)
            if config_section
            else {}
        )

        default_channels = 1 if self.modality == "audio" else 3

        self.embed_dim = int(
            self.backend_config.get(
                "embed_dim",
                self.encoder_config.get(
                    "embed_dim",
                    self.config.get("embed_dim", 0),
                ),
            )
        )
        self.in_channels = int(
            self.backend_config.get(
                "in_channels",
                self.encoder_config.get(
                    "in_channels",
                    self.config.get("in_channels", default_channels),
                ),
            )
        )
        self.dropout_rate = float(
            self.backend_config.get(
                "dropout_rate",
                self.encoder_config.get(
                    "dropout_rate",
                    self.config.get("dropout_rate", 0.0),
                ),
            )
        )
        self.initializer = str(
            self.backend_config.get(
                "initializer",
                self.encoder_config.get(
                    "initializer",
                    self.config.get("initializer", "xavier_uniform"),
                ),
            )
        ).strip().lower()

        requested_device = self.backend_config.get(
            "device",
            self.encoder_config.get(
                "device",
                self.config.get("device", "cpu"),
            ),
        )
        self.device = resolve_torch_device(requested_device)

        output_activation = self.backend_config.get(
            "output_activation",
            self.encoder_config.get("output_activation", "none"),
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
                    component="perception_encoder_fallback",
                    details={"actual_type": type(loaded).__name__},
                )
            return dict(loaded)

        if not isinstance(config, Mapping):
            raise InvalidPerceptionTypeError(
                "config must be a mapping when supplied.",
                component="perception_encoder_fallback",
                details={"actual_type": type(config).__name__},
            )
        return dict(config)

    def _resolve_section(self, name: Optional[str]) -> dict[str, Any]:
        if not name:
            return {}

        if name in self.config:
            value = self.config.get(name)
        elif self._config_injected:
            value = None
        else:
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

    def _resolve_encoder_config(
        self,
        encoder_config: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if encoder_config is None:
            return self._resolve_section(f"{self.modality}_encoder")
        if not isinstance(encoder_config, Mapping):
            raise InvalidPerceptionTypeError(
                "encoder_config must be a mapping when supplied.",
                component=self.component,
                details={"actual_type": type(encoder_config).__name__},
            )
        return dict(encoder_config)

    def _validate_base_config(self) -> None:
        if self.embed_dim <= 0:
            raise InvalidPerceptionConfigurationError(
                "Fallback encoder requires embed_dim > 0.",
                component=self.component,
                details={"embed_dim": self.embed_dim},
            )
        if self.in_channels <= 0:
            raise InvalidPerceptionConfigurationError(
                "Fallback encoder requires in_channels > 0.",
                component=self.component,
                details={"in_channels": self.in_channels},
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

    def _live_dtype(self) -> torch.dtype:
        """Return the active floating dtype used by trainable backend state."""

        for parameter in self.parameters(recurse=True):
            if torch.is_floating_point(parameter):
                return parameter.dtype
        for buffer in self.buffers(recurse=True):
            if torch.is_floating_point(buffer):
                return buffer.dtype
        return torch.float32

    def _prepare_input(self, value: torch.Tensor) -> torch.Tensor:
        """Validate and normalize raw modality input without changing semantics."""

        value = require_tensor(value, "input", component=self.component)

        if self.modality == "audio":
            if value.dim() == 2:
                value = value.unsqueeze(1)
            if value.dim() != 3:
                raise PerceptionShapeError(
                    "Audio encoder input must have shape (B,T) or (B,C,T).",
                    component=self.component,
                    details={"shape": list(value.shape)},
                )
            if value.size(1) != self.in_channels:
                raise PerceptionDimensionError(
                    "Audio input channel count does not match the encoder contract.",
                    component=self.component,
                    details={
                        "channels": int(value.size(1)),
                        "expected": self.in_channels,
                    },
                )
        else:
            if value.dim() == 3:
                value = value.unsqueeze(0)
            if value.dim() != 4:
                raise PerceptionShapeError(
                    "Vision encoder input must have shape (C,H,W) or (B,C,H,W).",
                    component=self.component,
                    details={"shape": list(value.shape)},
                )
            if value.size(1) != self.in_channels:
                raise PerceptionDimensionError(
                    "Vision input channel count does not match the encoder contract.",
                    component=self.component,
                    details={
                        "channels": int(value.size(1)),
                        "expected": self.in_channels,
                    },
                )

        if not torch.is_floating_point(value):
            value = value.float()

        if not bool(torch.isfinite(value).all().item()):
            raise ModalityEncodingError(
                "Encoder input contains NaN or infinite values.",
                component=self.component,
                details={"shape": list(value.shape), "dtype": str(value.dtype)},
            )

        return value.to(
            device=self._live_device(),
            dtype=self._live_dtype(),
        )

    def _finalize_encoded(self, encoded: torch.Tensor) -> torch.Tensor:
        """Validate the canonical fallback output contract.

        Both rank-2 ``(B,D)`` and rank-3 ``(B,L,D)`` outputs are permitted
        because the existing ``BasePerceptionModality`` contract accepts both.
        """

        encoded = require_tensor(encoded, "encoded", component=self.component)
        if encoded.dim() not in (2, 3):
            raise PerceptionShapeError(
                "Fallback encoder output must have shape (B,D) or (B,L,D).",
                component=self.component,
                details={"shape": list(encoded.shape)},
            )

        ensure_last_dimension(
            encoded,
            self.embed_dim,
            "encoded",
            component=self.component,
        )

        if self.output_activation == "sigmoid":
            encoded = torch.sigmoid(encoded)
        elif self.output_activation == "tanh":
            encoded = torch.tanh(encoded)

        if not bool(torch.isfinite(encoded).all().item()):
            raise ModalityEncodingError(
                "Encoder output contains NaN or infinite values.",
                component=self.component,
                details={"shape": list(encoded.shape), "dtype": str(encoded.dtype)},
            )

        return encoded

    def _initialize_module(self, module: nn.Module) -> None:
        """Initialize trainable Conv/Linear modules using the SLAI initializer policy."""

        if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            return

        if self.initializer in {"he", "he_normal"}:
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        elif self.initializer == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif self.initializer == "lecun":
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            nn.init.normal_(module.weight, mean=0.0, std=fan_in ** -0.5)
        elif self.initializer == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        else:
            raise InvalidPerceptionConfigurationError(
                "Unsupported fallback-encoder initializer.",
                component=self.component,
                details={
                    "initializer": self.initializer,
                    "supported": [
                        "xavier_uniform",
                        "xavier_normal",
                        "he",
                        "he_normal",
                        "lecun",
                    ],
                },
            )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def freeze_feature_extractor(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode one batch into the perception embedding contract."""
        raise NotImplementedError


__all__ = ["BaseEncoder"]