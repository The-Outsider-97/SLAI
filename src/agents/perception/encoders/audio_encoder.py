"""Audio encoder façade for SLAI perception.

The façade owns backend selection and the Transformer-specific patch path.
Concrete non-transformer backends are delegated to ``encoders.fallbacks`` so
CNN/MFCC feature extraction is registered once and can be checkpointed through
the normal PyTorch module tree.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional

from ...base.modules.activation_engine import he_init
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import *
from ..utils.perception_errors import *
from ..utils.perception_helpers import *
from ..modules.transformer import Transformer
from .fallbacks import CNNEncoder, MFCCEncoder
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Audio Encoder")
printer = PrettyPrinter()


class AudioEncoder(nn.Module):
    """Audio feature encoder with Transformer, CNN, and MFCC backends."""

    SUPPORTED_ENCODERS = ("transformer", "cnn", "mfcc")

    def __init__(self) -> None:
        super().__init__()
        self._init_configs()
        self._validate_configs()
        self._init_components()
        logger.info(
            "AudioEncoder initialized: type=%s, patch_size=%s, embed_dim=%s, in_channels=%s",
            self.encoder_type,
            self.patch_size,
            self.embed_dim,
            self.in_channels,
        )

    def _init_configs(self) -> None:
        self.config = load_global_config()
        self.audio_config = get_config_section("audio_encoder") or {}

        self.embed_dim = int(self.config.get("embed_dim", 512))
        self.encoder_type = str(
            self.audio_config.get(
                "encoder_type",
                self.config.get("encoder_type", "transformer"),
            )
        ).strip().lower()
        self.device = resolve_torch_device(self.config.get("device", "cpu"))
        self.max_position_embeddings = int(
            self.config.get("max_position_embeddings", 5000)
        )
        self.dropout_rate = float(self.config.get("dropout_rate", 0.1))

        self.in_channels = int(self.audio_config.get("in_channels", 1))
        self.audio_length = int(self.audio_config.get("audio_length", 16000))
        self.patch_size = int(self.audio_config.get("patch_size", 400))
        self.positional_encoding = str(
            self.audio_config.get("positional_encoding", "sinusoidal")
        ).strip().lower()
        self.dynamic_patching = bool(
            self.audio_config.get(
                "dynamic_patching",
                self.config.get("dynamic_patching", True),
            )
        )
        self.patch_overlap = float(self.audio_config.get("patch_overlap", 0.0))
        self.normalize_input = bool(self.audio_config.get("normalize_input", True))
        self.return_hidden = bool(self.audio_config.get("return_hidden", True))

        self.fallback_encoder: Optional[nn.Module] = None

    def _validate_configs(self) -> None:
        if self.embed_dim <= 0:
            raise InvalidPerceptionConfigurationError(
                "embed_dim must be > 0.",
                component="audio_encoder",
                details={"embed_dim": self.embed_dim},
            )
        if self.in_channels <= 0:
            raise InvalidPerceptionConfigurationError(
                "audio_encoder.in_channels must be > 0.",
                component="audio_encoder",
                details={"in_channels": self.in_channels},
            )
        if self.patch_size <= 0:
            raise InvalidPerceptionConfigurationError(
                "audio_encoder.patch_size must be > 0.",
                component="audio_encoder",
                details={"patch_size": self.patch_size},
            )
        if self.max_position_embeddings <= 0:
            raise InvalidPerceptionConfigurationError(
                "max_position_embeddings must be > 0.",
                component="audio_encoder",
            )
        if not 0.0 <= self.patch_overlap < 1.0:
            raise InvalidPerceptionConfigurationError(
                "audio_encoder.patch_overlap must be in [0,1).",
                component="audio_encoder",
                details={"patch_overlap": self.patch_overlap},
            )
        if self.encoder_type not in self.SUPPORTED_ENCODERS:
            raise UnsupportedPerceptionOptionError(
                "Unsupported audio encoder backend.",
                component="audio_encoder",
                details={
                    "encoder_type": self.encoder_type,
                    "supported": list(self.SUPPORTED_ENCODERS),
                },
            )

    def _init_components(self) -> None:
        if self.encoder_type == "transformer":
            self._init_transformer_encoder()
        elif self.encoder_type == "cnn":
            self.fallback_encoder = CNNEncoder(
                modality="audio",
                config=self.config,
                encoder_config=self.audio_config,
            )
        elif self.encoder_type == "mfcc":
            self.fallback_encoder = MFCCEncoder(
                config=self.config,
                encoder_config=self.audio_config,
            )

    def _init_transformer_encoder(self) -> None:
        projection_input = self.patch_size * self.in_channels
        self.projection = Parameter(
            he_init(
                (projection_input, self.embed_dim),
                fan_in=projection_input,
                device=str(self.device),
            )
        )

        if self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding()
        elif self.positional_encoding == "rotary":
            self.position_embed = None
        else:
            self.position_embed = Parameter(
                torch.randn(
                    1,
                    self.max_position_embeddings,
                    self.embed_dim,
                    device=self.device,
                )
                * 0.02
            )

        self.cls_token = Parameter(
            torch.randn(1, 1, self.embed_dim, device=self.device) * 0.02
        )
        self.transformer = Transformer(
            causal=False,
            enable_cross_attention=False,
            return_hidden=True,
        )

    def _init_sinusoidal_encoding(self) -> Parameter:
        pe = torch.zeros(
            1,
            self.max_position_embeddings,
            self.embed_dim,
            device=self.device,
        )
        position = torch.arange(
            0,
            self.max_position_embeddings,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0,
                self.embed_dim,
                2,
                dtype=torch.float32,
                device=self.device,
            )
            * (-math.log(10000.0) / self.embed_dim)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return Parameter(pe, requires_grad=False)

    def _prepare_transformer_input(self, x: torch.Tensor) -> torch.Tensor:
        x = require_tensor(x, "x", component="audio_encoder")
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ModalityInputError(
                "Audio input must have shape (B,T) or (B,C,T).",
                component="audio_encoder",
                details={"shape": list(x.shape)},
            )
        if x.size(1) != self.in_channels:
            raise PerceptionDimensionError(
                "Audio channel count does not match audio_encoder.in_channels.",
                component="audio_encoder",
                details={"channels": int(x.size(1)), "expected": self.in_channels},
            )
        if not torch.is_floating_point(x):
            x = x.float()
        x = x.to(self.device)

        if self.normalize_input:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
            x = (x - mean) / std
        return x

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ``(B,C,T)`` waveform data into flattened audio patches."""

        if x.dim() != 3:
            raise ModalityInputError(
                "extract_patches expects shape (B,C,T).",
                component="audio_encoder",
                details={"shape": list(x.shape)},
            )

        batch_size, channels, length = x.shape
        if channels != self.in_channels:
            raise PerceptionDimensionError(
                "Audio patch input channel count is invalid.",
                component="audio_encoder",
                details={"channels": channels, "expected": self.in_channels},
            )

        if self.dynamic_patching:
            pad_length = (self.patch_size - (length % self.patch_size)) % self.patch_size
            if pad_length:
                x = F.pad(x, (0, pad_length))

        if x.size(-1) < self.patch_size:
            raise PerceptionDimensionError(
                "Audio input is shorter than one patch and dynamic padding is disabled.",
                component="audio_encoder",
                details={"length": int(x.size(-1)), "patch_size": self.patch_size},
            )

        stride = self.patch_size
        if self.patch_overlap > 0.0:
            stride = max(1, int(round(self.patch_size * (1.0 - self.patch_overlap))))

        patches = x.unfold(2, self.patch_size, stride)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return patches.reshape(batch_size, -1, self.in_channels * self.patch_size)

    def _forward_transformer(
        self,
        x: torch.Tensor,
        style_id: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self._prepare_transformer_input(x)
        x = torch.matmul(self.extract_patches(x), self.projection)

        if self.training and self.dropout_rate > 0.0:
            x = F.dropout(x, p=self.dropout_rate, training=True)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.position_embed is not None:
            if x.size(1) > self.position_embed.size(1):
                raise PerceptionDimensionError(
                    "Audio sequence exceeds positional embedding capacity.",
                    component="audio_encoder",
                    details={
                        "sequence_length": int(x.size(1)),
                        "position_capacity": int(self.position_embed.size(1)),
                    },
                )
            x = x + self.position_embed[:, : x.size(1), :]

        return self.transformer(x, style_id=style_id)

    def forward(
        self,
        x: torch.Tensor,
        style_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            x = x[0]

        if self.encoder_type == "transformer":
            return self._forward_transformer(x, style_id)

        if self.fallback_encoder is None:
            raise RuntimeError("Configured audio fallback encoder was not initialized.")
        return self.fallback_encoder(x, style_id=style_id)

    def load_pretrained(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.encoder_type != "transformer":
            logger.warning("load_pretrained is only defined for the Transformer audio encoder.")
            return

        if "conv_proj" in weights:
            candidate = weights["conv_proj"].reshape(weights["conv_proj"].shape[0], -1).T
            if candidate.shape == self.projection.shape:
                self.projection.data.copy_(candidate.to(self.projection.device))
            else:
                logger.warning("Audio projection shape mismatch; skipping conv_proj.")

        if "cls_token" in weights and weights["cls_token"].shape == self.cls_token.shape:
            self.cls_token.data.copy_(weights["cls_token"].to(self.cls_token.device))

        if (
            "pos_embed" in weights
            and self.position_embed is not None
            and weights["pos_embed"].shape == self.position_embed.shape
        ):
            self.position_embed.data.copy_(weights["pos_embed"].to(self.position_embed.device))

        prefix = "transformer."
        transformer_weights = {
            key[len(prefix) :]: value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def freeze_feature_extractor(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        logger.info("Audio feature extractor frozen")

    def unfreeze_feature_extractor(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
        logger.info("Audio feature extractor unfrozen")

    def get_hidden_states(self) -> Optional[List[torch.Tensor]]:
        if self.encoder_type != "transformer":
            return None
        getter = getattr(self.transformer, "get_hidden_states", None)
        return getter() if callable(getter) else None


__all__ = ["AudioEncoder"]


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Testing Audio Encoder ===\n")

    # Create test inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio = torch.randn(2, 1, 16000).to(device)

    # Test transformer encoder
    print("Testing transformer encoder...")
    config = load_global_config()
    config['device'] = str(device)
    config['encoder_type'] = "transformer"
    config['dynamic_patching'] = True
    config['dropout_rate'] = 0.1
    encoder = AudioEncoder().to(device)
    output = encoder(audio)
    print("Transformer output shape:", output.shape)

    # Test MFCC encoder
    print("\nTesting MFCC encoder...")
    config['encoder_type'] = "mfcc"
    mfcc_encoder = AudioEncoder().to(device)
    mfcc_output = mfcc_encoder(audio)
    print("MFCC output shape:", mfcc_output.shape)

    # Test CNN encoder
    print("\nTesting CNN encoder...")
    config['encoder_type'] = "cnn"
    cnn_encoder = AudioEncoder().to(device)
    cnn_output = cnn_encoder(audio)
    print("CNN output shape:", cnn_output.shape)

    print("\n=== Audio Encoder tests passed ===")