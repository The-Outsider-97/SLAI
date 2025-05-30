
import torch

# --- Activation Functions (Copied/adapted from neural_network.py for self-containment) ---
class Activation:
    """Base class for activation functions."""
    def forward(self, z):
        raise NotImplementedError
    def backward(self, z_or_a): # z for most, a (activated output) for Softmax grad w.r.t z
        raise NotImplementedError

class ReLU(Activation):
    """Rectified Linear Unit activation."""
    def forward(self, z):
        return torch.maximum(torch.tensor(0.0, device=z.device, dtype=z.dtype), z)
    def backward(self, z): # Derivative w.r.t. z
        return (z > 0).type_as(z)

class Sigmoid(Activation):
    """Sigmoid activation function."""
    def forward(self, z):
        return 1 / (1 + torch.exp(-z))
    def backward(self, z): # Derivative w.r.t. z
        s = self.forward(z)
        return s * (1 - s)

class Tanh(Activation):
    """Hyperbolic Tangent (Tanh) activation function."""
    def forward(self, z):
        return torch.tanh(z)
    def backward(self, z): # Derivative w.r.t. z
        t = torch.tanh(z)
        return 1 - t**2

class Linear(Activation):
    """Linear activation function (identity)."""
    def forward(self, z):
        return z
    def backward(self, z): # Derivative w.r.t. z
        return torch.ones_like(z)
