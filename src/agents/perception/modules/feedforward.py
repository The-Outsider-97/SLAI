"""
FeedForward module for the Perception Agent's subsystem
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Mapping
from typing import Any, Dict, Tuple

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.perception_errors import *
from ..utils.perception_helpers import *
from ...base.modules.activation_engine import get_activation, he_init, lecun_normal, xavier_uniform, xavier_normal
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("FeedForward")
printer = PrettyPrinter()


class FeedForward(nn.Module):
    """Position-wise Transformer feed-forward network.

    Normalization, residual connections, and multimodal/context fusion are
    deliberately owned by the surrounding Transformer layer.  This module is
    therefore a pure ``D -> FF -> D`` transformation.
    """
    def __init__(self) -> None:
        super().__init__()

        self.config = load_global_config()
        self.ff_config = get_config_section("feedforward") or {}

        self.embed_dim = int(self.config.get("embed_dim", 512))
        self.ff_dim = int(self.config.get("ff_dim", self.embed_dim * 4))
        self.dropout_rate = float(self.config.get("dropout_rate", 0.1))
        self.activation_name = str(self.config.get("activation", "gelu"))
        self.initializer = str(self.config.get("initializer", "xavier_uniform"))
        self.device = resolve_torch_device(self.config.get("device", "cpu"))
        self.use_bias = bool(self.ff_config.get("use_bias", True))

        self._validate_config()
        self.activation = get_activation(self.activation_name)

        # These matrix shapes intentionally preserve the existing SLAI
        # checkpoint/pretrained-weight convention used by this module:
        #   x @ w1 -> hidden, hidden @ w2 -> output.
        self.w1 = nn.Parameter(self._initialized_tensor((self.embed_dim, self.ff_dim)))
        self.w2 = nn.Parameter(self._initialized_tensor((self.ff_dim, self.embed_dim)))

        if self.use_bias:
            self.b1 = nn.Parameter(torch.zeros(self.ff_dim, device=self.device))
            self.b2 = nn.Parameter(torch.zeros(self.embed_dim, device=self.device))
        else:
            self.register_parameter("b1", None)
            self.register_parameter("b2", None)

        logger.info(
            "FeedForward initialized: embed_dim=%s, ff_dim=%s, "
            "activation=%s, dropout=%.4f, use_bias=%s",
            self.embed_dim,
            self.ff_dim,
            self.activation_name,
            self.dropout_rate,
            self.use_bias,
        )

    def _validate_config(self) -> None:
        if self.embed_dim <= 0:
            raise PerceptionConfigurationError(
                "embed_dim must be positive.",
                component="feedforward",
                details={"embed_dim": self.embed_dim},
            )
        if self.ff_dim <= 0:
            raise PerceptionConfigurationError(
                "ff_dim must be positive.",
                component="feedforward",
                details={"ff_dim": self.ff_dim},
            )
        if not 0.0 <= self.dropout_rate < 1.0:
            raise PerceptionConfigurationError(
                "dropout_rate must be in [0,1).",
                component="feedforward",
                details={
                    "dropout_rate": self.dropout_rate
                },
            )

    def _initialized_tensor(
        self,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        init_map = {
            "he": he_init,
            "he_normal": he_init,
            "lecun": lecun_normal,
            "xavier_uniform": xavier_uniform,
            "xavier_normal": xavier_normal,
        }
        init_fn = init_map.get(self.initializer, xavier_uniform)
        tensor = init_fn(shape, device=self.device)
        return tensor.to(device=self.device, dtype=torch.float32)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the position-wise feed-forward transformation."""
        if not isinstance(x, torch.Tensor):
            raise PerceptionDimensionError(
                "FeedForward input must be a torch.Tensor.",
                component="feedforward",
                details={
                    "actual_type": type(x).__name__
                },
            )
        if x.dim() < 2:
            raise PerceptionDimensionError(
                "FeedForward input must have rank >= 2.",
                component="feedforward",
                details={"shape": list(x.shape)},
            )
        if x.size(-1) != self.embed_dim:
            raise PerceptionDimensionError(
                "FeedForward input dimension does not match embed_dim.",
                component="feedforward",
                details={
                    "actual": x.size(-1),
                    "expected": self.embed_dim,
                    "shape": list(x.shape),
                },
            )

        hidden = torch.matmul(x, self.w1)
        if self.b1 is not None:
            hidden = hidden + self.b1

        hidden = self.activation(hidden)

        if self.dropout_rate > 0.0:
            hidden = F.dropout(
                hidden,
                p=self.dropout_rate,
                training=self.training,
            )

        output = torch.matmul(hidden, self.w2)
        if self.b2 is not None:
            output = output + self.b2

        return output

    @staticmethod
    def _coerce_weight(
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        """Adapt standard Linear weight orientation to SLAI's matmul layout.

        ``nn.Linear`` stores weights as ``(out_features, in_features)`` while
        this module stores matrices for direct right-multiplication as
        ``(in_features, out_features)``.
        """
        if not isinstance(source, torch.Tensor):
            raise PerceptionStateError(
                f"Pretrained {name} must be a tensor.",
                component="feedforward",
                details={
                    "actual_type": type(source).__name__
                },
            )

        if tuple(source.shape) == tuple(target.shape):
            return source

        transposed = source.transpose(-2, -1)
        if tuple(transposed.shape) == tuple(target.shape):
            return transposed

        raise PerceptionDimensionError(
            f"Pretrained {name} shape is incompatible.",
            component="feedforward",
            details={
                "source_shape": list(source.shape),
                "target_shape": list(target.shape),
            },
        )

    @staticmethod
    def _copy_parameter(
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            target.copy_(source.to(device=target.device, dtype=target.dtype))

    def load_pretrained(
        self,
        weights: Mapping[str, torch.Tensor],
        prefix: str = "",
        *,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Load a BERT/HuggingFace-style feed-forward block conservatively.

        Supported source names:
          ``intermediate.dense.weight``
          ``intermediate.dense.bias``
          ``output.dense.weight``
          ``output.dense.bias``

        Unknown source parameters are not interpreted heuristically.
        """
        if not isinstance(weights, Mapping):
            raise PerceptionStateError(
                "weights must be a mapping.",
                component="feedforward",
                details={
                    "actual_type": type(weights).__name__
                },
            )

        keys = {
            "w1": f"{prefix}intermediate.dense.weight",
            "b1": f"{prefix}intermediate.dense.bias",
            "w2": f"{prefix}output.dense.weight",
            "b2": f"{prefix}output.dense.bias",
        }

        loaded = []
        missing = []

        for name, source_key in keys.items():
            target = getattr(self, name)

            # Biases are legitimately absent when use_bias=False.
            if target is None:
                continue

            source = weights.get(source_key)
            if source is None:
                missing.append(source_key)
                continue

            if name in ("w1", "w2"):
                source = self._coerce_weight(
                    source,
                    target,
                    name=name,
                )
            elif tuple(source.shape) != tuple(target.shape):
                raise PerceptionDimensionError(
                    f"Pretrained {name} shape is incompatible.",
                    component="feedforward",
                    details={
                        "source_shape": list(source.shape),
                        "target_shape": list(target.shape),
                    },
                )

            self._copy_parameter(target, source)
            loaded.append(source_key)

        if strict and missing:
            raise PerceptionStateError(
                "Required pretrained feed-forward parameters are missing.",
                component="feedforward",
                details={"missing_keys": missing},
            )

        return {
            "loaded_keys": tuple(loaded),
            "missing_keys": tuple(missing),
        }


__all__ = ["FeedForward"]


if __name__ == "__main__":
    print("\n=== Running FeedForward ===\n")
    model = FeedForward()
    print("Initialized FeedForward module:")
    print(model)

    x = torch.randn(4, 128, 512)
    print(f"\nInput shape: {x.shape}")

    model.train()
    print("\nMode: Training")
    y_train = model(x)
    print("Forward output (train):", y_train.shape)

    model.eval()
    print("\nMode: Evaluation")
    y_eval = model(x)
    print("Forward output (eval):", y_eval.shape)

    print("\n=== Successfully Ran FeedForward ===\n")