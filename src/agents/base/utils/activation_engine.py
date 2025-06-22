
import torch
import math

from typing import Tuple, Optional
from torch import Tensor

# --- Activation Functions Base Class ---
class Activation:
    """Base class for activation functions."""
    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError
    def backward(self, z_or_a: Tensor) -> Tensor: 
        raise NotImplementedError

# --- Tensor Functional Versions ---
def relu_tensor(z: Tensor) -> Tensor:
    return torch.relu(z)

def leaky_relu_tensor(z: Tensor, alpha: float = 0.01) -> Tensor:
    return torch.where(z > 0, z, alpha * z)

def elu_tensor(z: Tensor, alpha: float = 1.0) -> Tensor:
    return torch.where(z > 0, z, alpha * (torch.exp(z) - 1))

def swish_tensor(z: Tensor) -> Tensor:
    return z * torch.sigmoid(z)

def gelu_tensor(z: Tensor) -> Tensor:
    return 0.5 * z * (1 + torch.tanh(math.sqrt(2/math.pi) * 
                     (z + 0.044715 * torch.pow(z, 3))))

def mish_tensor(z: Tensor) -> Tensor:
    return z * torch.tanh(torch.nn.functional.softplus(z))

def sigmoid_tensor(z: Tensor) -> Tensor:
    return torch.sigmoid(z)

def tanh_tensor(z: Tensor) -> Tensor:
    return torch.tanh(z)

def softmax_tensor(z: Tensor, dim: int = -1) -> Tensor:
    return torch.softmax(z, dim=dim)

def he_init(shape: Tuple[int, ...], 
            fan_in: Optional[int] = None,
            mode: str = 'fan_in',
            nonlinearity: str = 'relu', 
            device='cpu') -> torch.Tensor:
    if fan_in is None:
        num_input_fmaps = shape[0] if len(shape) <= 2 else torch.prod(torch.tensor(shape[:-1])).item()
        fan_in = num_input_fmaps
    gain = math.sqrt(2.0) if nonlinearity == 'relu' else 1.0
    std = gain / math.sqrt(fan_in)
    return torch.randn(*shape, device=device) * std

def lecun_normal(shape: Tuple[int, ...], device='cpu') -> torch.Tensor:
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*shape))
    sigma = math.sqrt(1. / fan_in)
    return torch.normal(0, sigma, size=shape, device=device)

def xavier_uniform(shape: Tuple[int, ...], 
                   gain: float = 1.0, 
                   device='cpu') -> torch.Tensor:
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*shape))
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return torch.empty(*shape, device=device).uniform_(-a, a)

# --- Class Implementations ---
class ReLU(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return relu_tensor(z)
    
    def backward(self, z: Tensor) -> Tensor:
        return (z > 0).type_as(z)

class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        
    def forward(self, z: Tensor) -> Tensor:
        return leaky_relu_tensor(z, self.alpha)
    
    def backward(self, z: Tensor) -> Tensor:
        return torch.where(z > 0, 1.0, self.alpha)

class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def forward(self, z: Tensor) -> Tensor:
        return elu_tensor(z, self.alpha)
    
    def backward(self, z: Tensor) -> Tensor:
        return torch.where(z > 0, 1.0, self.alpha * torch.exp(z))

class Swish(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return swish_tensor(z)
    
    def backward(self, z: Tensor) -> Tensor:
        sig = sigmoid_tensor(z)
        return sig + z * sig * (1 - sig)

class SiLU(Swish):  # Alias for Swish
    pass

class GELU(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return gelu_tensor(z)
    
    def backward(self, z: Tensor) -> Tensor:
        # Derivative approximation
        sqrt_2 = math.sqrt(2)
        sqrt_pi = math.sqrt(math.pi)
        cdf = 0.5 * (1 + torch.erf(z / sqrt_2))
        pdf = torch.exp(-0.5 * z * z) / (sqrt_2 * sqrt_pi)
        return cdf + z * pdf

class Mish(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return mish_tensor(z)
    
    def backward(self, z: Tensor) -> Tensor:
        sp = torch.nn.functional.softplus(z)
        tanh_sp = torch.tanh(sp)
        grad_tanh = 1 - tanh_sp**2
        grad_sp = torch.sigmoid(z)
        return tanh_sp + z * grad_tanh * grad_sp

class Sigmoid(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return sigmoid_tensor(z)
    
    def backward(self, z: Tensor) -> Tensor:
        s = self.forward(z)
        return s * (1 - s)

class Tanh(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return tanh_tensor(z)
    
    def backward(self, z: Tensor) -> Tensor:
        t = self.forward(z)
        return 1 - t**2

class Linear(Activation):
    def forward(self, z: Tensor) -> Tensor:
        return z
    
    def backward(self, z: Tensor) -> Tensor:
        return torch.ones_like(z)
