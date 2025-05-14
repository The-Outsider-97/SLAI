import torch 
import math

from typing import Optional, Tuple

from logs.logger import get_logger

logger = get_logger("Common")

class Parameter:
    def __init__(self, data: torch.Tensor, requires_grad: bool = True, name: Optional[str] = None):
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        self.data = data.clone().detach()
        self.requires_grad = requires_grad
        self.grad = torch.zeros_like(data) if requires_grad else None
        self.name = name or "UnnamedParameter"

    def zero_grad(self) -> None:
        """Reset the gradient to zero, if gradient tracking is enabled."""
        if self.requires_grad and self.grad is not None:
            self.grad.zero_()

    def step(self, lr: float):
        """
        Apply a simple gradient descent step. Note: for demonstration/testing only.
        Args:
            lr (float): Learning rate.
        """
        if self.requires_grad and self.grad is not None:
            self.data -= lr * self.grad

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return (f"Parameter(name={self.name}, shape={self.shape}, "
                f"dtype={self.data.dtype}, requires_grad={self.requires_grad})")

class TensorOps:
    # --------------------------
    # Normalization Operations
    # --------------------------
    @staticmethod
    def layer_norm(x: torch.Tensor,
                   eps: float = 1e-5,
                   gamma: Optional[torch.Tensor] = None,
                   beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced layer normalization with optional affine transformation
        Args:
            x: Input tensor (..., features)
            eps: Numerical stability term
            gamma: Scale parameter (features,)
            beta: Shift parameter (features,)
        """
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x = (x - mean) / torch.sqrt(var + eps)
        
        if gamma is not None:
            x *= gamma
        if beta is not None:
            x += beta
        return x

    @staticmethod
    def instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Instance normalization for 4D tensors (B, C, H, W)"""
        mean = x.mean(axis=(2, 3), keepdims=True)
        var = x.var(axis=(2, 3), keepdims=True)
        return (x - mean) / torch.sqrt(var + eps)

    # --------------------------
    # Activation Functions
    # --------------------------
    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        """Gaussian Error Linear Unit"""
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * 
                         (x + 0.044715 * x**3)))

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        """Sigmoid Linear Unit (Swish)"""
        return x * TensorOps.sigmoid(x)

    @staticmethod
    def mish(x: torch.Tensor) -> torch.Tensor:
        """Mish: Self Regularized Non-Monotonic Activation"""
        return x * torch.tanh(torch.log1p(torch.exp(x)))

    # --------------------------
    # Initialization Methods
    # --------------------------
    @staticmethod
    def he_init(shape: Tuple[int], 
              fan_in: Optional[int] = None,
              mode: str = 'fan_in',
              nonlinearity: str = 'relu', device='cpu') -> torch.Tensor:
        """
        Kaiming initialization with configurable mode/nonlinearity
        Args:
            shape: Output shape
            fan_in: Input dimension (defaults to shape[0])
            mode: 'fan_in' (default) or 'fan_out'
            nonlinearity: 'relu' (default), 'leaky_relu', etc
        """
        fan = fan_in or shape[0]
        gain = math.sqrt(2.0) if nonlinearity == 'relu' else 1.0
        std = gain / math.sqrt(fan)
        return torch.randn(*shape, device=device) * std

    @staticmethod
    def lecun_normal(shape: Tuple[int]) -> torch.Tensor:
        """LeCun normal initialization (Variance scaling)"""
        scale = 1.0 / math.sqrt(shape[0])
        return torch.normal(0, scale, size=shape)

    @staticmethod
    def xavier_uniform(shape: Tuple[int], gain: float = 1.0) -> torch.Tensor:
        """Xavier/Glorot uniform initialization"""
        limit = gain * math.sqrt(6.0 / sum(shape[:2]))
        return torch.empty(*shape).uniform_(-limit, limit)

    # --------------------------
    # Tensor Operations
    # --------------------------
    @staticmethod
    def interpolate(x: torch.Tensor, size: Tuple[int], mode: str = 'bilinear') -> torch.Tensor:
        """2D interpolation (nearest/bilinear)"""
        from scipy.ndimage import zoom
        factors = (1, 1) + tuple(s / xs for s, xs in zip(size, x.shape[-2:]))
        return zoom(x, factors, order=0 if mode == 'nearest' else 1)

    @staticmethod
    def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverted dropout with mask caching"""
        if not training or p == 0:
            return x, torch.ones_like(x)
        
        mask = (torch.rand_like(x) > p).float() / (1 - p)
        return x * mask, mask

    @staticmethod
    def attention_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create boolean attention mask from sequence lengths"""
        batch_size = lengths.shape[0]
        return torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)

    # --------------------------
    # Utility Functions
    # --------------------------
    @staticmethod
    def pad_sequence(x: torch.Tensor, max_len: int, axis: int = 1) -> torch.Tensor:
        pad_size = max_len - x.shape[axis]
        if pad_size <= 0:
            return x
        pad_dims = [(0, 0)] * x.dim()
        pad_dims[axis] = (0, pad_size)
        pad_dims_flat = [dim for pair in reversed(pad_dims) for dim in pair]
        return torch.nn.functional.pad(x, pad_dims_flat)

    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable sigmoid"""
        return torch.where(x >= 0,
                      1 / (1 + torch.exp(-x)),
                      torch.exp(x) / (1 + torch.exp(x)))

if __name__ == "__main__":
    print("\n=== Running Common ===\n")

    # Create a real tensor for testing
    data = torch.randn(3, 4)  # Example tensor of shape (3, 4)
    common01 = Parameter(data)
    common02 = TensorOps()

    w = Parameter(torch.randn(2, 2), name="weights")
    print(w)
    
    # Simulate gradient
    w.grad = torch.ones_like(w.data)
    
    # Update step
    w.step(lr=0.1)
    
    # Reset gradient
    w.zero_grad()

    print(common01)
    print("\n=== Successfully Ran Common ===\n")
