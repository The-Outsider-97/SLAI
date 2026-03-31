"""
Activation Engine – Factory and utilities for activation functions and weight initializations.

Provides a unified way to instantiate activation functions by name and a set of
common weight initialisation methods (He, Xavier, etc.) for use in neural networks.
"""

import math
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor

# -------------------------------------------------------------------------
# Activation function registry
# -------------------------------------------------------------------------
_ACTIVATION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "linear": nn.Identity,          # no activation
    "swish": nn.SiLU,               # SiLU = Swish
    "gelu": nn.GELU,
    "mish": nn.Mish,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Return an activation module by name.

    Args:
        name: Activation name (case‑insensitive). Supported: relu, leaky_relu, elu,
              sigmoid, tanh, softmax, linear, swish, gelu, mish.
        **kwargs: Additional arguments passed to the activation constructor.

    Returns:
        An instance of nn.Module that implements the activation.

    Raises:
        ValueError: If the name is not supported.
    """
    name_lower = name.lower()
    if name_lower not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation: {name}. Supported: {list(_ACTIVATION_REGISTRY.keys())}")
    return _ACTIVATION_REGISTRY[name_lower](**kwargs)


# -------------------------------------------------------------------------
# Weight initialisation utilities
# -------------------------------------------------------------------------
def he_init(shape: tuple, fan_in: Optional[int] = None, nonlinearity: str = "relu", device: str = "cpu") -> Tensor:
    """
    He (Kaiming) initialisation for ReLU‑like activations.

    Args:
        shape: Desired tensor shape.
        fan_in: Number of input units. If None, inferred from shape (first dimension).
        nonlinearity: Type of nonlinearity (used to determine gain).
        device: Target device.

    Returns:
        Tensor initialised with He normal distribution.
    """
    if fan_in is None:
        fan_in = shape[0] if len(shape) <= 2 else torch.prod(torch.tensor(shape[:-1])).item()
    gain = math.sqrt(2.0) if nonlinearity.lower() == "relu" else 1.0
    std = gain / math.sqrt(fan_in)
    return torch.randn(*shape, device=device) * std


def lecun_normal(shape: tuple, device: str = "cpu") -> Tensor:
    """
    LeCun normal initialisation (used for SELU, etc.).

    Args:
        shape: Desired tensor shape.
        device: Target device.

    Returns:
        Tensor initialised with normal distribution with std = 1/sqrt(fan_in).
    """
    fan_in = shape[0] if len(shape) <= 2 else torch.prod(torch.tensor(shape[:-1])).item()
    std = math.sqrt(1.0 / fan_in)
    return torch.normal(0, std, size=shape, device=device)


def xavier_uniform(shape: tuple, gain: float = 1.0, device: str = "cpu") -> Tensor:
    """
    Xavier uniform initialisation.

    Args:
        shape: Desired tensor shape.
        gain: Scaling factor (e.g., 5/3 for tanh, 1 for linear).
        device: Target device.

    Returns:
        Tensor initialised uniformly in [−a, a] where a = gain * sqrt(6/(fan_in+fan_out)).
    """
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*shape))
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return torch.empty(*shape, device=device).uniform_(-a, a)


def xavier_normal(shape: tuple, gain: float = 1.0, device: str = "cpu") -> Tensor:
    """
    Xavier normal initialisation.

    Args:
        shape: Desired tensor shape.
        gain: Scaling factor.
        device: Target device.

    Returns:
        Tensor initialised with normal distribution with std = gain * sqrt(2/(fan_in+fan_out)).
    """
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*shape))
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return torch.normal(0, std, size=shape, device=device)


# -------------------------------------------------------------------------
# (Optional) Legacy activation classes – kept for backward compatibility
# -------------------------------------------------------------------------
class Activation(nn.Module):
    """Base class for activation functions (for compatibility with old code)."""

    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError


# If you need to keep the old class names (ReLU, etc.), you can alias them:
ReLU = nn.ReLU
LeakyReLU = nn.LeakyReLU
ELU = nn.ELU
Sigmoid = nn.Sigmoid
Tanh = nn.Tanh
Linear = nn.Identity
Softmax = nn.Softmax
Swish = nn.SiLU
SiLU = nn.SiLU
GELU = nn.GELU
Mish = nn.Mish


# -------------------------------------------------------------------------
# Test / demo
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing activation engine...")
    x = torch.randn(2, 3)
    for name in ["relu", "sigmoid", "tanh", "softmax", "linear"]:
        act = get_activation(name, dim=-1)  # softmax needs dim
        y = act(x)
        print(f"{name}: input shape {x.shape} -> output shape {y.shape}")
    print("All tests passed.")
