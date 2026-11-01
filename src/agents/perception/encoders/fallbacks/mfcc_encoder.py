"""Approximate inverse-MFCC fallback decoder for SLAI audio reconstruction.

MFCCs are deliberately lossy: Mel-band aggregation, logarithmic compression,
DCT truncation, and loss of STFT phase remove information required for exact
waveform inversion.  This module therefore exposes an explicit *approximate*
reconstruction path:

    latent -> learned MFCC estimate
           -> truncated inverse DCT
           -> approximate Mel-energy inversion
           -> approximate power spectrum
           -> magnitude spectrum
           -> phase estimation (Griffin-Lim or zero phase)
           -> waveform

The implementation mirrors the current ``AudioEncoder`` MFCC analysis
parameters (frame length, frame step, Mel filterbank, DCT normalization, and
Hamming window) so analysis/synthesis assumptions remain aligned.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Mapping
from typing import Any, Optional

from ...utils.config_loader import load_global_config, get_config_section
from ...utils.perception_errors import *
from ...utils.perception_helpers import *
from .base_encoder import BaseEncoder
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Base Encoder")
printer = PrettyPrinter()


class MFCCEncoder(BaseEncoder):
    """Audio MFCC analysis backend with a trainable embedding projection."""

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        encoder_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            encoder_type="mfcc",
            modality="audio",
            config_section="mfcc",
            config=config,
            encoder_config=encoder_config,
        )

        self.sample_rate = int(self.backend_config.get("sample_rate", 16000))
        self.n_mfcc = int(self.backend_config.get("n_mfcc", 13))
        self.frame_length_ms = float(self.backend_config.get("frame_length_ms", 25.0))
        self.frame_step_ms = float(self.backend_config.get("frame_step_ms", 10.0))
        self.n_filters = int(self.backend_config.get("n_filters", 40))
        self.low_freq = float(self.backend_config.get("low_freq", 0.0))
        self.high_freq = float(
            self.backend_config.get("high_freq", self.sample_rate / 2.0)
        )
        self.log_epsilon = float(self.backend_config.get("log_epsilon", 1e-6))

        self.frame_length = int(
            round(self.frame_length_ms * self.sample_rate / 1000.0)
        )
        self.frame_step = int(
            round(self.frame_step_ms * self.sample_rate / 1000.0)
        )

        self._validate_mfcc_config()

        self.register_buffer("mel_filters", self._build_mel_filterbank())
        self.register_buffer("dct_matrix", self._build_dct_matrix().transpose(0, 1).contiguous())
        self.register_buffer("analysis_window", torch.hamming_window(self.frame_length))

        self.projection = nn.Sequential(
            nn.Linear(self.n_mfcc, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )
        self.projection.apply(self._initialize_module)
        self.to(self.device)

    def _validate_mfcc_config(self) -> None:
        if self.sample_rate <= 0:
            raise InvalidPerceptionConfigurationError(
                "mfcc.sample_rate must be > 0.",
                component=self.component,
                details={"sample_rate": self.sample_rate},
            )
        if self.n_mfcc <= 0 or self.n_filters <= 0:
            raise InvalidPerceptionConfigurationError(
                "mfcc.n_mfcc and mfcc.n_filters must be > 0.",
                component=self.component,
                details={"n_mfcc": self.n_mfcc, "n_filters": self.n_filters},
            )
        if self.n_mfcc > self.n_filters:
            raise InvalidPerceptionConfigurationError(
                "mfcc.n_mfcc must not exceed mfcc.n_filters.",
                component=self.component,
                details={"n_mfcc": self.n_mfcc, "n_filters": self.n_filters},
                remediation=(
                    "Use no more cepstral coefficients than available Mel bands. "
                    "The current SLAI configuration uses n_mfcc=13 and n_filters=40."
                ),
            )
        if self.frame_length <= 0 or self.frame_step <= 0:
            raise InvalidPerceptionConfigurationError(
                "MFCC frame length and frame step must resolve to positive sample counts.",
                component=self.component,
                details={
                    "frame_length": self.frame_length,
                    "frame_step": self.frame_step,
                },
            )
        if not 0.0 <= self.low_freq < self.high_freq:
            raise InvalidPerceptionConfigurationError(
                "MFCC frequency bounds must satisfy 0 <= low_freq < high_freq.",
                component=self.component,
                details={"low_freq": self.low_freq, "high_freq": self.high_freq},
            )
        nyquist = self.sample_rate / 2.0
        if self.high_freq > nyquist:
            raise InvalidPerceptionConfigurationError(
                "mfcc.high_freq cannot exceed the Nyquist frequency.",
                component=self.component,
                details={"high_freq": self.high_freq, "nyquist": nyquist},
            )
        if self.log_epsilon <= 0.0:
            raise InvalidPerceptionConfigurationError(
                "mfcc.log_epsilon must be > 0.",
                component=self.component,
                details={"log_epsilon": self.log_epsilon},
            )

    def _build_mel_filterbank(self) -> torch.Tensor:
        """Build the triangular Mel filterbank used by SLAI MFCC analysis."""

        def hz_to_mel(value: torch.Tensor) -> torch.Tensor:
            return 2595.0 * torch.log10(1.0 + value / 700.0)

        def mel_to_hz(value: torch.Tensor) -> torch.Tensor:
            return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)

        low = torch.tensor(self.low_freq, dtype=torch.float32)
        high = torch.tensor(self.high_freq, dtype=torch.float32)
        mel_points = torch.linspace(
            hz_to_mel(low),
            hz_to_mel(high),
            self.n_filters + 2,
        )
        hz_points = mel_to_hz(mel_points)

        frequency_bins = self.frame_length // 2 + 1
        bins = torch.floor(
            (self.frame_length + 1) * hz_points / self.sample_rate
        ).to(dtype=torch.long)
        bins = bins.clamp(0, frequency_bins - 1)

        filters = torch.zeros(
            self.n_filters,
            frequency_bins,
            dtype=torch.float32,
        )

        for filter_index in range(self.n_filters):
            left = int(bins[filter_index].item())
            center = int(bins[filter_index + 1].item())
            right = int(bins[filter_index + 2].item())

            if left < center:
                numerator = torch.arange(left, center, dtype=torch.float32) - left
                filters[filter_index, left:center] = numerator / (center - left)
            if center < right:
                numerator = right - torch.arange(center, right, dtype=torch.float32)
                filters[filter_index, center:right] = numerator / (right - center)

        denominator = filters.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(filters.dtype).eps
        )
        return filters / denominator

    def _build_dct_matrix(self) -> torch.Tensor:
        """Build the encoder-compatible orthonormal DCT-II matrix.

        Returns
        -------
        torch.Tensor
            Shape ``(n_mfcc, n_filters)``.  The registered ``dct_matrix`` is
            its transpose ``(n_filters, n_mfcc)`` so MFCC extraction remains
            ``log_mel @ dct_matrix``, matching the existing AudioEncoder.
        """

        n = self.n_filters
        k = torch.arange(self.n_mfcc, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(n, dtype=torch.float32).unsqueeze(0)
        dct = torch.cos(
            math.pi * k * (2.0 * j + 1.0) / (2.0 * n)
        ) * math.sqrt(2.0 / n)
        dct[0] *= 1.0 / math.sqrt(2.0)
        return dct

    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract raw MFCC coefficients with shape ``(B,frames,n_mfcc)``."""

        waveform = self._prepare_input(waveform)

        # The existing AudioEncoder MFCC path averages channels before analysis.
        # Preserve that semantic explicitly rather than treating channels as
        # independent MFCC streams.
        if waveform.size(1) > 1:
            waveform = waveform.mean(dim=1)
        else:
            waveform = waveform[:, 0, :]

        if waveform.size(-1) < self.frame_length:
            waveform = F.pad(waveform, (0, self.frame_length - waveform.size(-1)))

        try:
            frames = waveform.unfold(
                dimension=1,
                size=self.frame_length,
                step=self.frame_step,
            )
            if frames.size(1) <= 0:
                raise InvalidPerceptionValueError(
                    "MFCC extraction produced no analysis frames.",
                    component=self.component,
                    details={
                        "waveform_length": int(waveform.size(-1)),
                        "frame_length": self.frame_length,
                        "frame_step": self.frame_step,
                    },
                )

            window = self.analysis_window.to(
                device=waveform.device,
                dtype=waveform.dtype,
            )
            windowed = frames * window
            spectrum = torch.fft.rfft(windowed, n=self.frame_length, dim=-1)
            power_spectrum = spectrum.abs().square()

            mel_filters = self.mel_filters.to(
                device=power_spectrum.device,
                dtype=power_spectrum.dtype,
            )
            mel_energies = torch.matmul(power_spectrum, mel_filters.transpose(0, 1))
            log_mel = torch.log(mel_energies.clamp_min(self.log_epsilon))

            dct_matrix = self.dct_matrix.to(
                device=log_mel.device,
                dtype=log_mel.dtype,
            )
            mfcc = torch.matmul(log_mel, dct_matrix)

            if not bool(torch.isfinite(mfcc).all().item()):
                raise ModalityEncodingError(
                    "MFCC extraction produced NaN or infinite coefficients.",
                    component=self.component,
                    details={"shape": list(mfcc.shape)},
                )
            return mfcc
        except Exception as exc:
            if isinstance(
                exc,
                (
                    InvalidPerceptionValueError,
                    ModalityEncodingError,
                    PerceptionDimensionError,
                ),
            ):
                raise
            raise wrap_exception(
                exc,
                ModalityEncodingError,
                "MFCC feature extraction failed.",
                component=self.component,
                details={"waveform_shape": list(waveform.shape)},
            ) from exc

    def analysis_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the registered Mel filterbank and DCT matrix.

        This is primarily useful when constructing a matched ``MFCCDecoder``;
        the returned tensors remain owned by this module and callers should not
        mutate them in-place.
        """

        return self.mel_filters, self.dct_matrix

    def forward( # type: ignore
        self,
        x: torch.Tensor,
        *,
        style_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode waveform audio to an MFCC-derived embedding sequence."""

        del style_id
        mfcc = self.extract_mfcc(x)
        try:
            encoded = self.projection(mfcc)
            return self._finalize_encoded(encoded)
        except Exception as exc:
            if isinstance(exc, ModalityEncodingError):
                raise
            raise wrap_exception(
                exc,
                ModalityEncodingError,
                "MFCC embedding projection failed.",
                component=self.component,
                details={"mfcc_shape": list(mfcc.shape)},
            ) from exc


__all__ = ["MFCCEncoder"]