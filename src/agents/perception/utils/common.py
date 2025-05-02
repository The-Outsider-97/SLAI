import numpy as np
import math
from typing import Optional, Tuple

class Parameter:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)
        
    def zero_grad(self) -> None:
        """Reset gradients to zero"""
        self.grad.fill(0)
        
    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape
    
    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, dtype={self.data.dtype})"

class TensorOps:
    # --------------------------
    # Normalization Operations
    # --------------------------
    @staticmethod
    def layer_norm(x: np.ndarray, 
                 eps: float = 1e-5,
                 gamma: Optional[np.ndarray] = None,
                 beta: Optional[np.ndarray] = None) -> np.ndarray:
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
        x = (x - mean) / np.sqrt(var + eps)
        
        if gamma is not None:
            x *= gamma
        if beta is not None:
            x += beta
        return x

    @staticmethod
    def instance_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Instance normalization for 4D tensors (B, C, H, W)"""
        mean = x.mean(axis=(2, 3), keepdims=True)
        var = x.var(axis=(2, 3), keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    # --------------------------
    # Activation Functions
    # --------------------------
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit"""
        return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * 
                         (x + 0.044715 * x**3)))

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """Sigmoid Linear Unit (Swish)"""
        return x * TensorOps.sigmoid(x)

    @staticmethod
    def mish(x: np.ndarray) -> np.ndarray:
        """Mish: Self Regularized Non-Monotonic Activation"""
        return x * np.tanh(np.log(1 + np.exp(x)))

    # --------------------------
    # Initialization Methods
    # --------------------------
    @staticmethod
    def he_init(shape: Tuple[int], 
              fan_in: Optional[int] = None,
              mode: str = 'fan_in',
              nonlinearity: str = 'relu') -> np.ndarray:
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
        return np.random.randn(*shape) * std

    @staticmethod
    def lecun_normal(shape: Tuple[int]) -> np.ndarray:
        """LeCun normal initialization (Variance scaling)"""
        scale = 1.0 / math.sqrt(shape[0])
        return np.random.normal(0, scale, size=shape)

    @staticmethod
    def xavier_uniform(shape: Tuple[int], 
                     gain: float = 1.0) -> np.ndarray:
        """Xavier/Glorot uniform initialization"""
        limit = gain * math.sqrt(6.0 / sum(shape[:2]))
        return np.random.uniform(-limit, limit, size=shape)

    # --------------------------
    # Tensor Operations
    # --------------------------
    @staticmethod
    def interpolate(x: np.ndarray, 
                  size: Tuple[int], 
                  mode: str = 'bilinear') -> np.ndarray:
        """2D interpolation (nearest/bilinear)"""
        from scipy.ndimage import zoom
        factors = (1, 1) + tuple(s / xs for s, xs in zip(size, x.shape[-2:]))
        return zoom(x, factors, order=0 if mode == 'nearest' else 1)

    @staticmethod
    def dropout(x: np.ndarray, 
              p: float = 0.5,
              training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Inverted dropout with mask caching"""
        if not training or p == 0:
            return x, np.ones_like(x)
        
        mask = (np.random.rand(*x.shape) > p) / (1 - p)
        return x * mask, mask

    @staticmethod
    def attention_mask(lengths: np.ndarray, 
                      max_len: int) -> np.ndarray:
        """Create boolean attention mask from sequence lengths"""
        batch_size = lengths.shape[0]
        return np.arange(max_len) < lengths[:, None]

    # --------------------------
    # Utility Functions
    # --------------------------
    @staticmethod
    def pad_sequence(x: np.ndarray, 
                   max_len: int, 
                   axis: int = 1) -> np.ndarray:
        """Pad sequences to fixed length"""
        pad_size = max_len - x.shape[axis]
        if pad_size <= 0:
            return x
        pads = [(0, 0)] * x.ndim
        pads[axis] = (0, pad_size)
        return np.pad(x, pads)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid"""
        return np.where(x >= 0,
                      1 / (1 + np.exp(-x)),
                      np.exp(x) / (1 + np.exp(x)))
