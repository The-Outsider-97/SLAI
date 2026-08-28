"""Learned convolutional fallback decoder for SLAI audio reconstruction.

This backend is intentionally described as a *learned reconstructor*, not as an
exact inverse of ``AudioEncoder``'s CNN path.  The current CNN encoder performs
global temporal pooling, which discards timing information; therefore exact
sample-level inversion is mathematically impossible from that representation
alone.  ``CNNDecoder`` instead learns a waveform synthesis mapping from the
available latent representation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ...utils.config_loader import load_global_config, get_config_section
from ...utils.perception_errors import *
from ...utils.perception_helpers import *
from .base_decoder import BaseDecoder
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("CNN Decoder")
printer = PrettyPrinter()


class _UpsampleConvBlock(nn.Module):
    """Interpolation + convolution block used by ``CNNDecoder``.

    Interpolation followed by convolution is used instead of unconstrained
    transposed-convolution geometry so target-size behavior remains explicit and
    predictable while reducing common checkerboard-style upsampling artifacts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        scale_factor: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2
        self.scale_factor = int(scale_factor)
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="linear",
            align_corners=False,
        )
        return self.block(x)


class CNNDecoder(BaseDecoder):
    """Efficient learned convolutional waveform reconstructor.

    Input
    -----
    ``latent`` may be ``(B, D)`` or ``(B, L, D)``.  Sequence latents preserve
    available temporal structure.  Pooled/single-token latents are augmented
    with deterministic positional features before hierarchical upsampling so
    the network is not forced to synthesize a time-varying waveform from a
    perfectly constant temporal seed.

    Configuration
    -------------
    Optional ``cnn_decoder`` keys:

    ``base_channels``
        Width of the initial synthesis representation.  Defaults to
        ``min(embed_dim, 256)`` as an efficiency cap.
    ``min_channels``
        Lower bound for later synthesis stages.  Defaults to 32 (or enough to
        remain wider than the requested output channel count).
    ``upsample_factors``
        Positive integer scale factors.  Defaults to ``[4, 4, 4, 4]``.
    ``kernel_size``
        Odd convolution kernel width.  Defaults to 5.
    ``dropout_rate``
        Optional backend-specific dropout override.
    ``output_activation``
        ``none`` or ``tanh``; handled by ``BaseDecoder``.
    """

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        decoder_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            decoder_type="cnn",
            config_section="cnn_decoder",
            config=config,
            decoder_config=decoder_config,
        )

        self.base_channels = int(
            self.backend_config.get(
                "base_channels",
                min(self.embed_dim, 256),
            )
        )
        self.min_channels = int(
            self.backend_config.get(
                "min_channels",
                max(32, self.in_channels * 4),
            )
        )
        self.kernel_size = int(self.backend_config.get("kernel_size", 5))

        raw_factors = self.backend_config.get(
            "upsample_factors",
            (4, 4, 4, 4),
        )
        if not isinstance(raw_factors, Sequence) or isinstance(
            raw_factors,
            (str, bytes),
        ):
            raise InvalidPerceptionConfigurationError(
                "cnn_decoder.upsample_factors must be a sequence of integers.",
                component=self.component,
                details={"actual_type": type(raw_factors).__name__},
            )
        self.upsample_factors = tuple(int(value) for value in raw_factors)

        self._validate_cnn_config()

        self.input_projection = nn.Linear(self.embed_dim, self.base_channels)
        self.position_projection = nn.Conv1d(
            3,
            self.base_channels,
            kernel_size=1,
            bias=False,
        )

        stage_channels = self._build_stage_channels()
        blocks = []
        current_channels = self.base_channels
        for factor, out_channels in zip(
            self.upsample_factors,
            stage_channels,
        ):
            blocks.append(
                _UpsampleConvBlock(
                    current_channels,
                    out_channels,
                    scale_factor=factor,
                    kernel_size=self.kernel_size,
                    dropout=self.dropout_rate,
                )
            )
            current_channels = out_channels
        self.upsample_blocks = nn.ModuleList(blocks)

        self.output_norm = nn.GroupNorm(1, current_channels)
        self.output_projection = nn.Conv1d(
            current_channels,
            self.in_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=True,
        )

        self._reset_parameters()
        self.to(self.device)

    def _validate_cnn_config(self) -> None:
        if self.base_channels <= 0:
            raise InvalidPerceptionConfigurationError(
                "cnn_decoder.base_channels must be > 0.",
                component=self.component,
                details={"base_channels": self.base_channels},
            )
        if self.min_channels <= 0:
            raise InvalidPerceptionConfigurationError(
                "cnn_decoder.min_channels must be > 0.",
                component=self.component,
                details={"min_channels": self.min_channels},
            )
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise InvalidPerceptionConfigurationError(
                "cnn_decoder.kernel_size must be a positive odd integer.",
                component=self.component,
                details={"kernel_size": self.kernel_size},
            )
        if not self.upsample_factors:
            raise InvalidPerceptionConfigurationError(
                "cnn_decoder.upsample_factors must not be empty.",
                component=self.component,
            )
        if any(value <= 1 for value in self.upsample_factors):
            raise InvalidPerceptionConfigurationError(
                "Every cnn_decoder upsample factor must be > 1.",
                component=self.component,
                details={"upsample_factors": self.upsample_factors},
            )

    def _build_stage_channels(self) -> tuple[int, ...]:
        channels = []
        for stage_index in range(len(self.upsample_factors)):
            reduced = self.base_channels // (2 ** (stage_index + 1))
            channels.append(max(self.min_channels, reduced))
        return tuple(channels)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if self.initializer in {"he", "he_normal"}:
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_in",
                        nonlinearity="relu",
                    )
                elif self.initializer == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _position_features(
        batch_size: int,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.linspace(
            -1.0,
            1.0,
            steps=length,
            device=device,
            dtype=dtype,
        )
        features = torch.stack(
            (
                position,
                torch.sin(math.pi * position),
                torch.cos(math.pi * position),
            ),
            dim=0,
        )
        return features.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        latent: torch.Tensor,
        *,
        target_length: Optional[int] = None,
        style_id: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Decode latent representations into a waveform.

        ``style_id`` is accepted for ``AudioDecoder`` call-surface compatibility
        but intentionally ignored: this fallback has no independent style
        embedding owner.  Style information already present in the latent
        representation remains available to the learned decoder.
        """

        del style_id

        latent = self._prepare_latent(latent)
        resolved_length = self._resolve_target_length(target_length)

        try:
            batch_size = latent.size(0)
            total_scale = math.prod(self.upsample_factors)
            seed_length = max(1, math.ceil(resolved_length / total_scale))

            features = self.input_projection(latent).transpose(1, 2)
            if features.size(-1) != seed_length:
                features = F.interpolate(
                    features,
                    size=seed_length,
                    mode="linear",
                    align_corners=False,
                )

            position = self._position_features(
                batch_size,
                seed_length,
                device=features.device,
                dtype=features.dtype,
            )
            features = features + self.position_projection(position)

            for block in self.upsample_blocks:
                features = block(features)

            # The hierarchical schedule intentionally need not multiply to the
            # exact requested length; exact geometry is resolved explicitly here.
            if features.size(-1) != resolved_length:
                features = F.interpolate(
                    features,
                    size=resolved_length,
                    mode="linear",
                    align_corners=False,
                )

            waveform = self.output_projection(
                F.gelu(self.output_norm(features))
            )
            return self._finalize_waveform(
                waveform,
                target_length=resolved_length,
            )
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityDecodingError,
                "CNN fallback audio reconstruction failed.",
                component=self.component,
                details={
                    "latent_shape": list(latent.shape),
                    "target_length": resolved_length,
                },
            ) from exc


__all__ = ["CNNDecoder"]