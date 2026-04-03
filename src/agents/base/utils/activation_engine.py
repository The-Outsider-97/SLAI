"""
Activation Engine – Factory and utilities for activation functions and weight initializations.

Provides a unified way to instantiate activation functions by name and a set of
common weight initialisation methods (He, Xavier, etc.) for use in neural networks.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Optional, Type


class Activation(nn.Module):
    """Base class for compatibility activations with analytic derivatives."""

    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return torch.relu(z)

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        grad = (z > 0).to(z.dtype)
        return grad if grad_output is None else grad_output * grad


class LeakyReLU(Activation):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, z: Tensor) -> Tensor:
        return F.leaky_relu(z, negative_slope=self.negative_slope)

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        grad = torch.ones_like(z)
        grad[z < 0] = self.negative_slope
        return grad if grad_output is None else grad_output * grad


class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, z: Tensor) -> Tensor:
        return F.elu(z, alpha=self.alpha)

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        grad = torch.where(z > 0, torch.ones_like(z), self.alpha * torch.exp(z))
        return grad if grad_output is None else grad_output * grad


class Sigmoid(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return torch.sigmoid(z)

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        s = self.forward(z)
        grad = s * (1.0 - s)
        return grad if grad_output is None else grad_output * grad


class Tanh(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return torch.tanh(z)

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        t = torch.tanh(z)
        grad = 1.0 - t * t
        return grad if grad_output is None else grad_output * grad


class Linear(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return z

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        return torch.ones_like(z) if grad_output is None else grad_output


class Softmax(Activation):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, z: Tensor) -> Tensor:
        shifted = z - torch.max(z, dim=self.dim, keepdim=True).values
        exp_z = torch.exp(shifted)
        return exp_z / exp_z.sum(dim=self.dim, keepdim=True)

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        probs = self.forward(z)
        if grad_output is None:
            raise NotImplementedError('Softmax.backward requires grad_output.')
        dot = torch.sum(grad_output * probs, dim=self.dim, keepdim=True)
        return probs * (grad_output - dot)


class Swish(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return z * torch.sigmoid(z)

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        s = torch.sigmoid(z)
        grad = s + z * s * (1.0 - s)
        return grad if grad_output is None else grad_output * grad


class SiLU(Swish):
    pass


class GELU(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return F.gelu(z)

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        inner = sqrt_2_over_pi * (z + 0.044715 * z.pow(3))
        tanh_inner = torch.tanh(inner)
        sech2 = 1.0 - tanh_inner.pow(2)
        grad = 0.5 * (1.0 + tanh_inner) + 0.5 * z * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * z.pow(2))
        return grad if grad_output is None else grad_output * grad


class Mish(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return z * torch.tanh(F.softplus(z))

    def backward(self, z: Tensor, grad_output: Optional[Tensor] = None) -> Tensor:
        sp = F.softplus(z)
        tsp = torch.tanh(sp)
        sig = torch.sigmoid(z)
        grad = tsp + z * sig * (1.0 - tsp * tsp)
        return grad if grad_output is None else grad_output * grad


_ACTIVATION_REGISTRY: Dict[str, Type[nn.Module]] = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'elu': ELU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax,
    'linear': Linear,
    'identity': Linear,
    'swish': Swish,
    'gelu': GELU,
    'mish': Mish,
    'silu': SiLU,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    name_lower = name.lower()
    if name_lower not in _ACTIVATION_REGISTRY:
        raise ValueError(f'Unknown activation: {name}. Supported: {list(_ACTIVATION_REGISTRY.keys())}')
    if name_lower == 'softmax' and 'dim' not in kwargs:
        kwargs['dim'] = -1
    return _ACTIVATION_REGISTRY[name_lower](**kwargs)


def gelu_tensor(x: Tensor) -> Tensor:
    return F.gelu(x)


def swish_tensor(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def mish_tensor(x: Tensor) -> Tensor:
    return x * torch.tanh(F.softplus(x))


def sigmoid_tensor(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def he_init(shape: tuple, fan_in: Optional[int] = None, nonlinearity: str = 'relu', device: str = 'cpu') -> Tensor:
    if fan_in is None:
        fan_in = shape[0] if len(shape) <= 2 else torch.prod(torch.tensor(shape[:-1])).item()
    gain = math.sqrt(2.0) if nonlinearity.lower() == 'relu' else 1.0
    std = gain / math.sqrt(fan_in)
    return torch.randn(*shape, device=device) * std


def lecun_normal(shape: tuple, device: str = 'cpu') -> Tensor:
    fan_in = shape[0] if len(shape) <= 2 else torch.prod(torch.tensor(shape[:-1])).item()
    std = math.sqrt(1.0 / fan_in)
    return torch.normal(0, std, size=shape, device=device)


def xavier_uniform(shape: tuple, gain: float = 1.0, device: str = 'cpu') -> Tensor:
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*shape))
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return torch.empty(*shape, device=device).uniform_(-a, a)


def xavier_normal(shape: tuple, gain: float = 1.0, device: str = 'cpu') -> Tensor:
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*shape))
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return torch.normal(0, std, size=shape, device=device)


if __name__ == '__main__':
    print('Testing activation engine...')
    x = torch.randn(2, 3)
    for name in ['relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'gelu', 'mish']:
        act = get_activation(name)
        y = act(x)
        print(f'{name}: input shape {x.shape} -> output shape {y.shape}')

    z = torch.randn(4, 5)
    upstream = torch.randn(4, 5)
    assert torch.allclose(Linear().backward(z), torch.ones_like(z))
    assert ReLU().backward(z).shape == z.shape
    assert Sigmoid().backward(z).shape == z.shape
    assert Tanh().backward(z).shape == z.shape
    assert Softmax(dim=-1).backward(z, upstream).shape == z.shape
    print('All tests passed.')
