"""CNN fallback encoder for SLAI audio and vision perception.

The implementation consolidates the CNN paths that were previously embedded
inside ``AudioEncoder`` and ``VisionEncoder`` while preserving their current
representation semantics:

- audio: Conv1d stack -> global temporal pooling -> projection -> ``(B,1,D)``;
- vision: Conv2d stack -> spatial pyramid pooling -> projection -> ``(B,D)``.

The backend remains an ordinary child ``nn.Module`` of the owning modality
encoder.  It does not own optimizer, persistence, cache, or agent lifecycle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ...utils.config_loader import load_global_config, get_config_section
from ...utils.perception_errors import *
from ...utils.perception_helpers import *
from .base_encoder import BaseEncoder
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("CNN Encoder")
printer = PrettyPrinter()


class CNNEncoder(BaseEncoder):
    """Shared CNN feature encoder for audio and vision fallback paths."""

    def __init__(
        self,
        *,
        modality: str = "audio",
        config: Optional[Mapping[str, Any]] = None,
        encoder_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            encoder_type="cnn",
            modality=modality,
            config_section="cnn",
            config=config,
            encoder_config=encoder_config,
        )

        self.normalize_input = bool(
            self.encoder_config.get("normalize_input", self.modality == "audio")
        )
        self.normalization_epsilon = float(
            self.encoder_config.get("normalization_epsilon", 1e-8)
        )
        if self.normalization_epsilon <= 0.0:
            raise InvalidPerceptionConfigurationError(
                "normalization_epsilon must be > 0.",
                component=self.component,
                details={"normalization_epsilon": self.normalization_epsilon},
            )

        if self.modality == "audio":
            self._init_audio_backend()
        else:
            self._init_vision_backend()

        self.apply(self._initialize_module)
        self.to(self.device)

    # ------------------------------------------------------------------
    # Audio CNN
    # ------------------------------------------------------------------

    def _init_audio_backend(self) -> None:
        out_channels = self._integer_sequence(
            self.backend_config.get("out_channels", (64, 128, 256)),
            "cnn.out_channels",
        )
        kernel_sizes = self._integer_sequence(
            self.backend_config.get("kernel_sizes", (3, 3, 3)),
            "cnn.kernel_sizes",
        )
        strides = self._integer_sequence(
            self.backend_config.get("strides", (2, 2, 2)),
            "cnn.strides",
        )

        if not (len(out_channels) == len(kernel_sizes) == len(strides)):
            raise InvalidPerceptionConfigurationError(
                "Audio CNN out_channels, kernel_sizes, and strides must have equal length.",
                component=self.component,
                details={
                    "out_channels": out_channels,
                    "kernel_sizes": kernel_sizes,
                    "strides": strides,
                },
            )
        if not out_channels:
            raise InvalidPerceptionConfigurationError(
                "Audio CNN requires at least one convolutional stage.",
                component=self.component,
            )

        layers: list[nn.Module] = []
        current_channels = self.in_channels
        for output_channels, kernel_size, stride in zip(
            out_channels,
            kernel_sizes,
            strides,
        ):
            layers.extend(
                [
                    nn.Conv1d(
                        current_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(output_channels),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            current_channels = output_channels

        self.backbone = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(current_channels, self.embed_dim)

        self.audio_out_channels = tuple(out_channels)
        self.audio_kernel_sizes = tuple(kernel_sizes)
        self.audio_strides = tuple(strides)

    # ------------------------------------------------------------------
    # Vision CNN
    # ------------------------------------------------------------------

    def _init_vision_backend(self) -> None:
        filters = self.backend_config.get("filters")
        if filters is None:
            raise InvalidPerceptionConfigurationError(
                "Vision CNN requires cnn.filters.",
                component=self.component,
                remediation=(
                    "Retain the existing cnn.filters configuration used by VisionEncoder, "
                    "or pass an equivalent configuration snapshot."
                ),
            )
        if not isinstance(filters, Sequence) or isinstance(filters, (str, bytes)):
            raise InvalidPerceptionTypeError(
                "cnn.filters must be a sequence of [kernel_h, kernel_w, out_channels].",
                component=self.component,
                details={"actual_type": type(filters).__name__},
            )

        parsed_filters: list[tuple[int, int, int]] = []
        for index, spec in enumerate(filters):
            if (
                not isinstance(spec, Sequence)
                or isinstance(spec, (str, bytes))
                or len(spec) != 3
            ):
                raise InvalidPerceptionConfigurationError(
                    "Each cnn.filters entry must contain [kernel_h, kernel_w, out_channels].",
                    component=self.component,
                    details={"index": index, "value": spec},
                )
            kernel_h, kernel_w, output_channels = (int(value) for value in spec)
            if min(kernel_h, kernel_w, output_channels) <= 0:
                raise InvalidPerceptionConfigurationError(
                    "Vision CNN filter dimensions must be positive.",
                    component=self.component,
                    details={"index": index, "value": list(spec)},
                )
            parsed_filters.append((kernel_h, kernel_w, output_channels))

        if not parsed_filters:
            raise InvalidPerceptionConfigurationError(
                "Vision CNN requires at least one filter stage.",
                component=self.component,
            )

        spp_levels = self._integer_sequence(
            self.backend_config.get("spp_levels", (1, 2, 4)),
            "cnn.spp_levels",
        )
        if not spp_levels:
            raise InvalidPerceptionConfigurationError(
                "Vision CNN requires at least one spatial-pyramid level.",
                component=self.component,
            )

        layers: list[nn.Module] = []
        current_channels = self.in_channels
        for index, (kernel_h, kernel_w, output_channels) in enumerate(parsed_filters):
            # Preserve the current VisionEncoder geometry: stride 4 in the first
            # convolution, stride 1 afterwards, and max-pooling after the first
            # two convolutional stages.
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        output_channels,
                        kernel_size=(kernel_h, kernel_w),
                        stride=4 if index == 0 else 1,
                        padding=2,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
            if index < 2:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            current_channels = output_channels

        self.backbone = nn.Sequential(*layers)
        self.spp_levels = tuple(spp_levels)
        pooled_features = current_channels * sum(level * level for level in self.spp_levels)
        self.projection = nn.Linear(pooled_features, self.embed_dim)
        self.vision_filters = tuple(parsed_filters)

    # ------------------------------------------------------------------
    # Shared execution
    # ------------------------------------------------------------------

    def _integer_sequence(self, value: Any, name: str) -> list[int]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise InvalidPerceptionTypeError(
                f"{name} must be a sequence of positive integers.",
                component=self.component,
                details={"actual_type": type(value).__name__},
            )
        parsed = [int(item) for item in value]
        if any(item <= 0 for item in parsed):
            raise InvalidPerceptionConfigurationError(
                f"{name} must contain only positive integers.",
                component=self.component,
                details={"value": parsed},
            )
        return parsed

    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return waveform

        mean = waveform.mean(dim=-1, keepdim=True)
        # ``unbiased=False`` avoids NaN for a one-sample sequence while
        # preserving ordinary per-channel standardization for real waveforms.
        std = waveform.std(dim=-1, keepdim=True, unbiased=False)
        return (waveform - mean) / std.clamp_min(self.normalization_epsilon)

    def _spatial_pyramid_pool(self, features: torch.Tensor) -> torch.Tensor:
        pooled = [
            F.adaptive_max_pool2d(features, output_size=(level, level)).flatten(1)
            for level in self.spp_levels
        ]
        return torch.cat(pooled, dim=1)

    def forward( # type: ignore
        self,
        x: torch.Tensor,
        *,
        style_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio or vision input using the configured CNN backend.

        ``style_id`` is accepted for call-surface compatibility with the
        transformer façade.  The CNN backend does not own style-conditioning
        parameters, so the value is intentionally not consumed here.
        """

        del style_id
        prepared = self._prepare_input(x)

        try:
            if self.modality == "audio":
                prepared = self._normalize_audio(prepared)
                features = self.backbone(prepared)
                pooled = self.global_pool(features).squeeze(-1)
                encoded = self.projection(pooled).unsqueeze(1)
            else:
                features = self.backbone(prepared)
                pooled = self._spatial_pyramid_pool(features)
                encoded = self.projection(pooled)

            return self._finalize_encoded(encoded)
        except Exception as exc:
            if isinstance(
                exc,
                (
                    InvalidPerceptionConfigurationError,
                    InvalidPerceptionTypeError,
                    ModalityEncodingError,
                ),
            ):
                raise
            raise wrap_exception(
                exc,
                ModalityEncodingError,
                "CNN fallback encoding failed.",
                component=self.component,
                details={"input_shape": list(prepared.shape)},
            ) from exc


__all__ = ["CNNEncoder"]