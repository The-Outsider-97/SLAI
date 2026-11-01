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

from collections.abc import Mapping
from typing import Any, Optional

from ...utils.config_loader import load_global_config, get_config_section
from ...utils.perception_errors import *
from ...utils.perception_helpers import *
from .base_decoder import BaseDecoder
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("MFCC Decoder")
printer = PrettyPrinter()


class MFCCDecoder(BaseDecoder):
    """Approximate waveform reconstruction from embedded or raw MFCC features.

    ``forward()`` accepts SLAI latent embeddings ``(B, L, embed_dim)`` and uses
    a trainable projection to estimate MFCCs.  ``decode_mfcc()`` is provided for
    callers that already have raw MFCC tensors ``(B, frames, n_mfcc)`` and want
    to bypass the learned latent projection.
    """

    VALID_PHASE_INITIALIZATIONS = ("zeros", "random")

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        decoder_config: Optional[Mapping[str, Any]] = None,
        mel_filters: Optional[torch.Tensor] = None,
        dct_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            decoder_type="mfcc",
            config_section="mfcc_decoder",
            config=config,
            decoder_config=decoder_config,
        )

        self.mfcc_config = self._resolve_section("mfcc")

        self.sample_rate = int(self.mfcc_config.get("sample_rate", 16000))
        self.n_mfcc = int(self.mfcc_config.get("n_mfcc", 13))
        self.frame_length_ms = float(
            self.mfcc_config.get("frame_length_ms", 25)
        )
        self.frame_step_ms = float(
            self.mfcc_config.get("frame_step_ms", 10)
        )
        self.n_filters = int(self.mfcc_config.get("n_filters", 40))
        self.low_freq = float(self.mfcc_config.get("low_freq", 0.0))
        self.high_freq = float(
            self.mfcc_config.get(
                "high_freq",
                self.sample_rate / 2.0,
            )
        )

        self.frame_length = int(
            round(self.frame_length_ms * self.sample_rate / 1000.0)
        )
        self.frame_step = int(
            round(self.frame_step_ms * self.sample_rate / 1000.0)
        )

        self.log_epsilon = float(
            self.backend_config.get("log_epsilon", 1e-6)
        )
        self.pinv_rcond = float(
            self.backend_config.get("pinv_rcond", 1e-6)
        )
        self.griffin_lim_iterations = int(
            self.backend_config.get("griffin_lim_iterations", 16)
        )
        self.phase_initialization = str(
            self.backend_config.get("phase_initialization", "zeros")
        ).strip().lower()
        self.replicate_channels = bool(
            self.backend_config.get("replicate_channels", False)
        )

        self._validate_mfcc_config()

        resolved_mel = self._resolve_mel_filters(mel_filters)
        resolved_dct = self._resolve_dct_matrix(dct_matrix)

        # Analysis matrices are buffers: they are part of model state and move
        # with ``module.to(device)`` but are not trainable parameters.
        self.register_buffer("mel_filters", resolved_mel)
        self.register_buffer("dct_matrix", resolved_dct)

        # Encoder relation:
        #   mel = power @ mel_filters.T
        # Therefore the least-squares reconstruction is:
        #   power_hat = mel @ pinv(mel_filters.T)
        self.register_buffer(
            "mel_inverse",
            torch.linalg.pinv(
                resolved_mel.transpose(0, 1),
                rcond=self.pinv_rcond,
            ),
        )

        # Encoder relation:
        #   mfcc = log_mel @ dct_matrix
        # with dct_matrix shape (n_filters, n_mfcc).  Truncation makes the
        # inverse underdetermined, so use the least-squares pseudoinverse.
        self.register_buffer(
            "dct_inverse",
            torch.linalg.pinv(
                resolved_dct,
                rcond=self.pinv_rcond,
            ),
        )

        self.register_buffer(
            "analysis_window",
            torch.hamming_window(self.frame_length),
        )

        self.latent_to_mfcc = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.n_mfcc),
        )
        self._reset_parameters()
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
                details={
                    "n_mfcc": self.n_mfcc,
                    "n_filters": self.n_filters,
                },
            )
        if self.n_mfcc > self.n_filters:
            raise InvalidPerceptionConfigurationError(
                "mfcc.n_mfcc must not exceed mfcc.n_filters for this decoder.",
                component=self.component,
                details={
                    "n_mfcc": self.n_mfcc,
                    "n_filters": self.n_filters,
                },
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
                details={
                    "low_freq": self.low_freq,
                    "high_freq": self.high_freq,
                },
            )
        nyquist = self.sample_rate / 2.0
        if self.high_freq > nyquist:
            raise InvalidPerceptionConfigurationError(
                "mfcc.high_freq cannot exceed the Nyquist frequency.",
                component=self.component,
                details={
                    "high_freq": self.high_freq,
                    "nyquist": nyquist,
                },
            )
        if self.log_epsilon <= 0.0:
            raise InvalidPerceptionConfigurationError(
                "mfcc_decoder.log_epsilon must be > 0.",
                component=self.component,
                details={"log_epsilon": self.log_epsilon},
            )
        if self.pinv_rcond <= 0.0:
            raise InvalidPerceptionConfigurationError(
                "mfcc_decoder.pinv_rcond must be > 0.",
                component=self.component,
                details={"pinv_rcond": self.pinv_rcond},
            )
        if self.griffin_lim_iterations < 0:
            raise InvalidPerceptionConfigurationError(
                "mfcc_decoder.griffin_lim_iterations must be >= 0.",
                component=self.component,
                details={
                    "griffin_lim_iterations": self.griffin_lim_iterations,
                },
            )

        ensure_one_of(
            self.phase_initialization,
            self.VALID_PHASE_INITIALIZATIONS,
            "phase_initialization",
            component=self.component,
            exc_type=InvalidPerceptionConfigurationError,
        )

        if self.in_channels != 1 and not self.replicate_channels:
            raise InvalidPerceptionConfigurationError(
                "The current MFCC analysis path is mono; multi-channel reconstruction "
                "requires explicit replicate_channels=true.",
                component=self.component,
                details={"in_channels": self.in_channels},
                remediation=(
                    "Use in_channels=1 for MFCC reconstruction, or explicitly enable "
                    "channel replication with the understanding that independent channel "
                    "information was not present in the MFCC representation."
                ),
            )

    def _resolve_mel_filters(
        self,
        provided: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if provided is None:
            return self._build_mel_filterbank()

        provided = require_tensor(
            provided,
            "mel_filters",
            component=self.component,
        ).detach().to(dtype=torch.float32, device="cpu")
        expected = (self.n_filters, self.frame_length // 2 + 1)
        if tuple(provided.shape) != expected:
            raise PerceptionDimensionError(
                "Provided Mel filterbank shape does not match MFCC configuration.",
                component=self.component,
                details={
                    "shape": list(provided.shape),
                    "expected": list(expected),
                },
            )
        return provided.contiguous()

    def _resolve_dct_matrix(
        self,
        provided: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if provided is None:
            return self._build_dct_matrix().transpose(0, 1).contiguous()

        provided = require_tensor(
            provided,
            "dct_matrix",
            component=self.component,
        ).detach().to(dtype=torch.float32, device="cpu")
        expected = (self.n_filters, self.n_mfcc)
        if tuple(provided.shape) != expected:
            raise PerceptionDimensionError(
                "Provided DCT matrix shape does not match MFCC configuration.",
                component=self.component,
                details={
                    "shape": list(provided.shape),
                    "expected": list(expected),
                },
            )
        return provided.contiguous()

    def _build_mel_filterbank(self) -> torch.Tensor:
        """Build the same triangular Mel filterbank convention as AudioEncoder."""

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
        bins = torch.floor(
            (self.frame_length + 1) * hz_points / self.sample_rate
        ).to(dtype=torch.long)

        frequency_bins = self.frame_length // 2 + 1
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

        # AudioEncoder normalizes each triangular band by its total weight.
        # Clamp only protects degenerate edge filters from divide-by-zero.
        denominator = filters.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(filters.dtype).eps
        )
        return filters / denominator

    def _build_dct_matrix(self) -> torch.Tensor:
        """Build the encoder-compatible orthonormal DCT-II coefficient matrix."""

        n = self.n_filters
        k = torch.arange(self.n_mfcc, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(n, dtype=torch.float32).unsqueeze(0)
        dct = torch.cos(
            math.pi * k * (2.0 * j + 1.0) / (2.0 * n)
        ) * math.sqrt(2.0 / n)
        dct[0] *= 1.0 / math.sqrt(2.0)
        return dct

    def _reset_parameters(self) -> None:
        projection = self.latent_to_mfcc[-1]
        if self.initializer in {"he", "he_normal"}:
            nn.init.kaiming_normal_(
                projection.weight,
                mode="fan_in",
                nonlinearity="linear",
            )
        elif self.initializer == "xavier_normal":
            nn.init.xavier_normal_(projection.weight)
        else:
            nn.init.xavier_uniform_(projection.weight)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)

    def _mfcc_to_magnitude(self, mfcc: torch.Tensor) -> torch.Tensor:
        """Return approximate STFT magnitude ``(B, frames, frequency_bins)``."""

        log_mel = torch.matmul(mfcc, self.dct_inverse)

        # Avoid overflow without imposing arbitrary application-level clipping:
        # cap only at the largest representable finite exponential for the dtype.
        finfo = torch.finfo(log_mel.dtype)
        max_log = math.log(finfo.max) - 2.0
        mel_energies = torch.exp(log_mel.clamp(max=max_log)) - self.log_epsilon
        mel_energies = mel_energies.clamp_min(0.0)

        power = torch.matmul(mel_energies, self.mel_inverse)
        power = power.clamp_min(0.0)
        return torch.sqrt(power)

    def _initial_phase(self, magnitude: torch.Tensor) -> torch.Tensor:
        if self.phase_initialization == "zeros":
            return torch.ones_like(magnitude, dtype=torch.complex64)

        phase = torch.rand_like(magnitude) * (2.0 * math.pi)
        return torch.polar(torch.ones_like(magnitude), phase)

    def _griffin_lim(
        self,
        magnitude: torch.Tensor,
        *,
        iterations: int,
    ) -> torch.Tensor:
        """Estimate a waveform from an STFT magnitude using Griffin-Lim."""

        # torch.stft/istft use (B, frequency, frames).
        magnitude = magnitude.transpose(1, 2).contiguous()
        frames = int(magnitude.size(-1))
        if frames <= 0:
            raise InvalidPerceptionValueError(
                "MFCC reconstruction requires at least one frame.",
                component=self.component,
            )

        synthesis_length = self.frame_length + (frames - 1) * self.frame_step
        window = self.analysis_window.to(
            device=magnitude.device,
            dtype=magnitude.dtype,
        )

        phase = self._initial_phase(magnitude)
        complex_spectrum = magnitude.to(phase.dtype) * phase

        # ``iterations == 0`` intentionally means deterministic one-pass
        # zero/random-phase synthesis, useful for low-cost training paths.
        for _ in range(iterations):
            waveform = torch.istft(
                complex_spectrum,
                n_fft=self.frame_length,
                hop_length=self.frame_step,
                win_length=self.frame_length,
                window=window,
                center=False,
                normalized=False,
                onesided=True,
                length=synthesis_length,
            )
            rebuilt = torch.stft(
                waveform,
                n_fft=self.frame_length,
                hop_length=self.frame_step,
                win_length=self.frame_length,
                window=window,
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )

            phase = rebuilt / rebuilt.abs().clamp_min(self.log_epsilon)
            complex_spectrum = magnitude.to(phase.dtype) * phase

        return torch.istft(
            complex_spectrum,
            n_fft=self.frame_length,
            hop_length=self.frame_step,
            win_length=self.frame_length,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            length=synthesis_length,
        )

    def decode_mfcc(
        self,
        mfcc: torch.Tensor,
        *,
        target_length: Optional[int] = None,
        iterations: Optional[int] = None,
    ) -> torch.Tensor:
        """Approximately reconstruct waveform audio from raw MFCC coefficients."""

        mfcc = require_tensor(mfcc, "mfcc", component=self.component)
        ensure_tensor_rank(mfcc, 3, "mfcc", component=self.component)
        ensure_last_dimension(
            mfcc,
            self.n_mfcc,
            "mfcc",
            component=self.component,
        )
        if not torch.is_floating_point(mfcc):
            mfcc = mfcc.float()

        mfcc = mfcc.to(
            device=self.dct_inverse.device,
            dtype=self.dct_inverse.dtype,
        )
        resolved_length = self._resolve_target_length(target_length)
        active_iterations = (
            self.griffin_lim_iterations
            if iterations is None
            else int(iterations)
        )
        if active_iterations < 0:
            raise InvalidPerceptionValueError(
                "iterations must be >= 0.",
                component=self.component,
                details={"iterations": active_iterations},
            )

        try:
            magnitude = self._mfcc_to_magnitude(mfcc)
            mono = self._griffin_lim(
                magnitude,
                iterations=active_iterations,
            ).unsqueeze(1)

            if self.in_channels > 1:
                # This is deliberately opt-in because the encoder's MFCC path
                # collapses channels to mono before analysis.
                mono = mono.expand(-1, self.in_channels, -1).contiguous()

            return self._finalize_waveform(
                mono,
                target_length=resolved_length,
            )
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityDecodingError,
                "Inverse-MFCC fallback reconstruction failed.",
                component=self.component,
                details={
                    "mfcc_shape": list(mfcc.shape),
                    "target_length": resolved_length,
                    "griffin_lim_iterations": active_iterations,
                },
            ) from exc

    def forward(
        self,
        latent: torch.Tensor,
        *,
        target_length: Optional[int] = None,
        style_id: Optional[torch.Tensor] = None,
        iterations: Optional[int] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Estimate MFCCs from SLAI latent embeddings and reconstruct waveform."""

        del style_id

        latent = self._prepare_latent(latent)
        try:
            mfcc = self.latent_to_mfcc(latent)
            return self.decode_mfcc(
                mfcc,
                target_length=target_length,
                iterations=iterations,
            )
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityDecodingError,
                "MFCC fallback latent decoding failed.",
                component=self.component,
                details={"latent_shape": list(latent.shape)},
            ) from exc


__all__ = ["MFCCDecoder"]