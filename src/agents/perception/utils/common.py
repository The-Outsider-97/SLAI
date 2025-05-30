import torch 
import math
import yaml

from typing import Optional, Tuple, Dict, Any

from logs.logger import get_logger

logger = get_logger("Common")

DEFAULT_CONFIG_PATH = "src/agents/perception/configs/perception_config.yaml"

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in {config_path}: {e}")
        return {}

class Parameter(torch.nn.Parameter):
    def __new__(cls, data: torch.Tensor, requires_grad=True, name: Optional[str] = None):
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        param = super().__new__(cls, data.clone().detach(), requires_grad=requires_grad)
        return param
    def __init__(self, data: torch.Tensor, requires_grad = True, name: Optional[str] = None):
        if self.requires_grad and self.grad is None:
            pass

        #self.data = data.clone().detach()
        #self.requires_grad = requires_grad
        #self.grad = torch.zeros_like(data) if requires_grad else None
        #self.name = name or "UnnamedParameter"

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
            with torch.no_grad():
                self.data -= lr * self.grad

    #@property
    #def shape(self) -> Tuple[int, ...]:
    #    return self.data.shape

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        if "Parameter containing:\n" in base_repr:
            parts = base_repr.split('\n', 1)
            return f"Parameter(name={self.name}, containing:\n{parts[1]}"
        else:
            idx = base_repr.find('(')
            if idx != -1:
                return f"{base_repr[:idx+1]}name={self.name}, {base_repr[idx+1:]}"
            return f"Parameter(name={self.name}, {base_repr})"


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
        var = x.var(axis=-1, keepdims=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + eps)
        
        if gamma is not None:
            x *= gamma
        if beta is not None:
            x += beta
        return x

    @staticmethod
    def instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Instance normalization for 4D tensors (B, C, H, W)"""
        if x.ndim != 4:
            raise ValueError("Instance norm expects a 4D tensor (B, C, H, W)")
        mean = x.mean(axis=(2, 3), keepdims=True)
        var = x.var(axis=(2, 3), keepdims=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + eps)

    # --------------------------
    # Activation Functions
    # --------------------------
    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        """Gaussian Error Linear Unit"""
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * 
                         (x + 0.044715 * torch.pow(x**3))))

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        """Sigmoid Linear Unit (Swish)"""
        return x * TensorOps.sigmoid(x)

    @staticmethod
    def mish(x: torch.Tensor) -> torch.Tensor:
        """Mish: Self Regularized Non-Monotonic Activation"""
        return x * torch.tanh(torch.nn.functional.softplus(x))

    # --------------------------
    # Initialization Methods
    # --------------------------
    @staticmethod
    def he_init(shape: Tuple[int, ...], 
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
        if fan_in is None:
            num_input_fmaps = shape[0] if len(shape) <= 2 else torch.prod(torch.tensor(shape[:-1])).item()
            fan_in = num_input_fmaps
        gain = math.sqrt(2.0) if nonlinearity == 'relu' else 1.0
        std = gain / math.sqrt(fan_in)
        return torch.randn(*shape, device=device) * std

    @staticmethod
    def lecun_normal(shape: Tuple[int, ...], device='cpu') -> torch.Tensor:
        """LeCun normal initialization (Variance scaling)"""
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*shape))
        sigma = math.sqrt(1. / fan_in)
        return torch.normal(0, sigma, size=shape, device=device)

    @staticmethod
    def xavier_uniform(shape: Tuple[int, ...], gain: float = 1.0, device='cpu') -> torch.Tensor:
        """Xavier/Glorot uniform initialization"""
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*shape))
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std
        return torch.empty(*shape, device=device).uniform_(-a, a)

    # --------------------------
    # Tensor Operations
    # --------------------------
    @staticmethod
    def interpolate(x: torch.Tensor, size: Tuple[int, int], mode: str = 'bilinear', align_corners: Optional[bool] = None) -> torch.Tensor: # size is Tuple[int,int] for H,W
        """2D interpolation using torch.nn.functional.interpolate"""
        if x.ndim not in [2,3,4]:
            logger.warning(f"Interpolate input dim {x.ndim} not standard. Assuming last 2 dims are H,W.")
        original_ndim = x.ndim

        if original_ndim == 2: # H, W
            x = x.unsqueeze(0).unsqueeze(0) # 1, 1, H, W
        elif original_ndim == 3: # C, H, W
            x = x.unsqueeze(0) # 1, C, H, W
        
        if x.ndim != 4: # Should be N, C, H, W
            logger.error(f"Interpolation input must be 4D (N,C,H,W) or be reshapable to it. Got {x.shape}")
            raise ValueError("Interpolation input dimension issue.")

        output = torch.nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
        
        if original_ndim == 2:
            output = output.squeeze(0).squeeze(0)
        elif original_ndim == 3:
            output = output.squeeze(0)
            
        return output

    @staticmethod
    def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverted dropout with mask caching"""
        if not training or p == 0:
            return x, torch.ones_like(x, device=x.device)
        
        mask = (torch.rand_like(x, device=x.device) > p).float() / (1 - p)
        return x * mask, mask

    @staticmethod
    def attention_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create boolean attention mask from sequence lengths"""
        batch_size = lengths.shape[0]
        return torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths.unsqueeze(1)

    # --------------------------
    # Utility Functions
    # --------------------------
    @staticmethod
    def pad_sequence(x: torch.Tensor, max_len: int, axis: int = 1, value: float = 0) -> torch.Tensor:
        pad_size = max_len - x.shape[axis]
        if pad_size <= 0:
            return x
        pad_dims = [(0, 0)] * x.dim()
        pt_pad_dims = [0] * (2 * x.dim())
        pad_idx_in_pt_tuple = 2 * (x.dim() - 1 - axis)
        pt_pad_dims[pad_idx_in_pt_tuple + 1] = pad_size
        return torch.nn.functional.pad(x, pt_pad_dims, value=value)

    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable sigmoid"""
        return torch.sigmoid(x)

if __name__ == "__main__":
    print("\n=== Running Common ===\n")

    # Create a real tensor for testing
    data_tensor = torch.randn(3, 4, device='cpu')  # Example tensor of shape (3, 4)
    common01 = Parameter(data_tensor, name="my_param") # Pass the tensor directly

    # Test TensorOps methods
    # Layer Norm
    test_ln = torch.randn(2, 5, 10) # B, T, D
    gamma_ln = torch.ones(10)
    beta_ln = torch.zeros(10)
    out_ln = TensorOps.layer_norm(test_ln, gamma=gamma_ln, beta=beta_ln)
    print(f"Layer Norm Output mean: {out_ln.mean().item():.4f}, std: {out_ln.std().item():.4f}")
    assert torch.allclose(out_ln.mean(dim=-1), torch.zeros_like(out_ln.mean(dim=-1)), atol=1e-6)
    std_unbiased = out_ln.std(dim=-1, unbiased=False)
    assert torch.allclose(std_unbiased, torch.ones_like(std_unbiased), atol=1e-5)


    # He Init
    he_weights = TensorOps.he_init((10, 5), fan_in=5, nonlinearity='relu')
    print(f"He Init sample: {he_weights[0,0].item():.4f}")

    # Interpolate (example with image-like data)
    test_interp_img = torch.randn(1, 3, 32, 32) # N, C, H, W
    out_interp_img = TensorOps.interpolate(test_interp_img, size=(16,16))
    print(f"Interpolate img output shape: {out_interp_img.shape}")
    assert out_interp_img.shape == (1, 3, 16, 16)

    # Pad sequence
    test_pad = torch.randn(2, 5, 3) # B, T, D
    out_pad = TensorOps.pad_sequence(test_pad, max_len=10, axis=1)
    print(f"Pad sequence output shape: {out_pad.shape}")
    assert out_pad.shape == (2, 10, 3)


    common02 = TensorOps() # This is just a namespace for static methods

    w = Parameter(torch.randn(2, 2), name="weights")
    print(w)

    # Simulate gradient
    if w.requires_grad:
        # To assign a gradient, it needs to be part of a computation graph or manually set
        w.grad = torch.ones_like(w.data) # This is okay for your custom step

    # Update step
    w.step(lr=0.1)
    print(f"Weights after step: {w.data}")

    # Reset gradient
    w.zero_grad()
    if w.grad is not None:
        print(f"Gradient after zero_grad: {w.grad}")
    else:
        print("Gradient is None after zero_grad (expected if not part of graph yet)")


    print(common01)
    print("\n=== Successfully Ran Common ===\n")
