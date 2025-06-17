import torch 
import math
import yaml

from typing import Optional, Tuple, Dict, Any, Union

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.base.utils.activation_engine import (gelu_tensor, swish_tensor, mish_tensor, sigmoid_tensor,
    he_init as he_init_engine, lecun_normal as lecun_normal_engine, xavier_uniform as xavier_uniform_engine)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Common")
printer = PrettyPrinter

class Parameter(torch.nn.Parameter):
    def __new__(cls, data: torch.Tensor, requires_grad=True, name: Optional[str] = None):
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        param = super().__new__(cls, data.clone().detach(), requires_grad=requires_grad)
        return param
    def __init__(self, data: torch.Tensor, requires_grad: bool = True, name: Optional[str] = None):
        self.config = load_global_config()
        self.param_config = get_config_section('parameter')
        self.data = data.clone().detach().requires_grad_(requires_grad)
        if self.requires_grad and self.grad is None:
            pass
        self.requires_grad = requires_grad
        self._momentum_buffer = None  

    def zero_grad(self) -> None:
        """Reset the gradient to zero, if gradient tracking is enabled."""
        if self.requires_grad and self.grad is not None:
            self.grad.zero_()

    def step(self, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        """
        Apply in-place parameter update with optional momentum and L2 regularization.

        Args:
            lr (float): Learning rate.
            momentum (float): Momentum factor (0 disables momentum).
            weight_decay (float): L2 regularization factor.
        """
        if not self.requires_grad or self.grad is None:
            return

        with torch.no_grad():
            # Compute effective gradient with L2 penalty
            grad = self.grad + weight_decay * self.data if weight_decay > 0 else self.grad

            # Initialize or update momentum buffer
            if momentum > 0.0:
                if self._momentum_buffer is None:
                    self._momentum_buffer = torch.clone(grad).detach()
                else:
                    self._momentum_buffer.mul_(momentum).add_(grad)

                update = self._momentum_buffer
            else:
                update = grad

            # Apply parameter update
            self.data -= lr * update

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
    def layer_norm(x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced Layer Normalization with support for:
        - Custom normalized dimensions
        - Optional affine transformation
        - Automatic dimension detection
        - Improved numerical stability

        Args:
            x: Input tensor of shape (..., D)
            normalized_shape: Dimensions to normalize (default: last dimension)
            eps: Small value to avoid division by zero
            gamma: Scale parameter (must match normalized_shape)
            beta: Shift parameter (must match normalized_shape)
            elementwise_affine: Apply learnable affine transformation

        Returns:
            Normalized tensor of same shape as input
        """
        config = load_global_config()
        tensor_config = get_config_section('tensor_ops')
        eps = tensor_config.get('eps')
        elementwise_affine = tensor_config.get('elementwise_affine')

        normalized_shape = None
        gamma: Optional[torch.Tensor] = None
        beta: Optional[torch.Tensor] = None

        # Auto-detect normalized shape if not provided
        if normalized_shape is None:
            normalized_shape = (x.size(-1),)
        
        # Validate normalized shape
        if any(s <= 0 for s in normalized_shape):
            raise ValueError(f"normalized_shape must be positive integers, got {normalized_shape}")
        
        # Calculate dimensions to reduce
        dims = tuple(range(-len(normalized_shape), 0))
        
        # Compute statistics with improved numerical stability
        mean = x.mean(dim=dims, keepdim=True)
        variance = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # Normalize using stable computation
        x_normalized = (x - mean) / torch.sqrt(variance + eps)
        
        # Apply affine transformation if enabled
        if elementwise_affine:
            if gamma is None:
                gamma = torch.ones(normalized_shape, dtype=x.dtype, device=x.device)
            if beta is None:
                beta = torch.zeros(normalized_shape, dtype=x.dtype, device=x.device)
            
            # Validate parameter shapes
            if gamma.shape != normalized_shape:
                raise ValueError(f"gamma shape {gamma.shape} != normalized_shape {normalized_shape}")
            if beta.shape != normalized_shape:
                raise ValueError(f"beta shape {beta.shape} != normalized_shape {normalized_shape}")
            
            x_normalized = x_normalized * gamma + beta
        
        return x_normalized

    @staticmethod
    def instance_norm(x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced Instance Normalization supporting:
        - 3D (temporal), 4D (spatial), and 5D (volumetric) inputs
        - Optional affine transformation
        - Running statistics for inference
        - Configurable momentum

        Args:
            x: Input tensor (B, C, ...)
            eps: Small value to avoid division by zero
            gamma: Scale parameter (C,)
            beta: Shift parameter (C,)
            affine: Apply learnable affine parameters
            track_running_stats: Maintain running mean/variance
            momentum: Momentum for running stats update
            training: Training mode flag

        Returns:
            Normalized tensor of same shape as input
        """
        tensor_config = get_config_section('tensor_ops')
        eps = tensor_config.get('eps')
        affine = tensor_config.get('affine')
        track_running_stats = tensor_config.get('track_running_stats')
        momentum = tensor_config.get('momentum')
        training = tensor_config.get('training')

        gamma: Optional[torch.Tensor] = None
        beta: Optional[torch.Tensor] = None

        # Validate input dimensions
        if x.dim() not in (3, 4, 5):
            raise ValueError(f"InstanceNorm requires 3D, 4D or 5D input (got {x.dim()}D)")
        
        # Initialize running stats if needed
        if track_running_stats:
            running_mean = torch.zeros(x.size(1), device=x.device, dtype=x.dtype)
            running_var = torch.ones(x.size(1), device=x.device, dtype=x.dtype)
        else:
            running_mean = running_var = None
        
        # Calculate spatial dimensions to reduce
        spatial_dims = tuple(range(2, x.dim()))
        
        # Compute statistics
        if training or not track_running_stats:
            mean = x.mean(dim=spatial_dims, keepdim=True)
            var = x.var(dim=spatial_dims, keepdim=True, unbiased=False)
            
            # Update running stats if tracking
            if track_running_stats:
                with torch.no_grad():
                    running_mean = (1 - momentum) * running_mean + momentum * mean.squeeze()
                    running_var = (1 - momentum) * running_var + momentum * var.squeeze()
        else:
            mean = running_mean.view(1, -1, *((1,) * (x.dim() - 2)))
            var = running_var.view(1, -1, *((1,) * (x.dim() - 2)))
        
        # Normalize with stability term
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        
        # Apply affine transformation
        if affine:
            if gamma is None:
                gamma = torch.ones(x.size(1), device=x.device, dtype=x.dtype)
            if beta is None:
                beta = torch.zeros(x.size(1), device=x.device, dtype=x.dtype)
            
            # Reshape parameters for broadcasting
            view_shape = (1, -1) + (1,) * (x.dim() - 2)
            gamma = gamma.view(*view_shape)
            beta = beta.view(*view_shape)
            
            x_normalized = x_normalized * gamma + beta
        
        return x_normalized

    # --------------------------
    # Activation Functions
    # --------------------------
    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        return gelu_tensor(x)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return swish_tensor(x)

    @staticmethod
    def mish(x: torch.Tensor) -> torch.Tensor:
        return mish_tensor(x)

    # --------------------------
    # Initialization Methods
    # --------------------------
    @staticmethod
    def he_init(shape: Tuple[int, ...], 
              fan_in: Optional[int] = None,
              mode: str = 'fan_in',
              nonlinearity: str = 'relu', device='cpu') -> torch.Tensor:
        return he_init_engine(shape, fan_in, mode, nonlinearity, device)

    @staticmethod
    def lecun_normal(shape: Tuple[int, ...], device='cpu') -> torch.Tensor:
        return lecun_normal_engine(shape, device)

    @staticmethod
    def xavier_uniform(shape: Tuple[int, ...], gain: float = 1.0, device='cpu') -> torch.Tensor:
        return xavier_uniform_engine(shape, gain, device)

    # --------------------------
    # Tensor Operations
    # --------------------------
    @staticmethod
    def interpolate(x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced multi-dimensional interpolation supporting:
        - 1D (temporal), 2D (spatial), and 3D (volumetric) data
        - Size-based or scale-factor-based resizing
        - Multiple interpolation modes with auto-dimensionality detection
        - Antialiasing for high-quality downsampling
        - Automatic dimension handling for non-standard inputs

        Args:
            x: Input tensor (any dimension 1-5)
            size: Output spatial size (int or tuple)
            scale_factor: Multiplier for spatial size (float or tuple)
            mode: Interpolation mode ('nearest', 'linear', 'bilinear', 
                  'bicubic', 'trilinear', 'area')
            align_corners: Geometric alignment of corners
            recompute_scale_factor: Recompute scale factor for output
            antialias: Apply antialiasing when downsampling (2D/3D only)

        Returns:
            Interpolated tensor with same dimensions as input
        """
        tensor_config = get_config_section('tensor_ops')
        mode = tensor_config.get('mode')
        antialias = tensor_config.get('antialias')
        scale_factor = tensor_config.get('scale_factor')
        size = tensor_config.get('size', None)
        align_corners = tensor_config.get('align_corners', None)
        recompute_scale_factor = tensor_config.get('recompute_scale_factor', None)

        # Validate input
        if size is None and scale_factor is None:
            raise ValueError("Either size or scale_factor must be specified")
        if x.numel() == 0:
            raise ValueError("Input tensor is empty")
            
        # Determine dimensionality based on mode
        mode_to_dim = {
            'nearest': None,  # Flexible
            'linear': 3,      # 1D
            'bilinear': 4,    # 2D
            'bicubic': 4,     # 2D
            'trilinear': 5,   # 3D
            'area': None      # Flexible
        }
        
        target_dim = mode_to_dim.get(mode)
        if target_dim and x.dim() not in (target_dim, target_dim - 1, target_dim - 2):
            logger.warning(f"Mode '{mode}' typically expects {target_dim}D input, got {x.dim()}D")

        # Record original shape and compute target shape
        original_shape = x.shape
        reshaped = x
        
        # Handle 1D/2D/3D inputs by adding batch/channel dimensions
        if x.dim() == 1:  # (T) -> (1, 1, T)
            reshaped = x.reshape(1, 1, -1)
        elif x.dim() == 2:  # (H, W) -> (1, 1, H, W)
            reshaped = x.reshape(1, 1, *x.shape)
        elif x.dim() == 3:  # (C, H, W) -> (1, C, H, W)
            reshaped = x.unsqueeze(0)
        
        # Do not derive size if scale_factor is explicitly set
        if size is None and scale_factor is None:
            raise ValueError("Either size or scale_factor must be specified")
        
        # Handle antialiasing
        kwargs = {}
        if antialias and mode in ['bilinear', 'bicubic', 'trilinear']:
            if any(s_in > s_out for s_in, s_out in zip(reshaped.shape[2:], size)):
                kwargs['antialias'] = True
            else:
                logger.info("Antialiasing skipped (upsampling operation)")

        # Perform interpolation
        interpolated = torch.nn.functional.interpolate(
            reshaped,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            **kwargs
        )
        
        # Restore original dimensions
        if x.dim() == 1:
            return interpolated.squeeze(0).squeeze(0)
        elif x.dim() == 2:
            return interpolated.squeeze(0).squeeze(0)
        elif x.dim() == 3:
            return interpolated.squeeze(0)
        return interpolated

    @staticmethod
    def dropout(x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Enhanced dropout with advanced features:
        - Inverted dropout implementation
        - Optional in-place computation
        - Custom random generator support
        - Mask caching and retrieval options
        - Gradient scaling in eval mode (for model approximation)

        Args:
            x: Input tensor
            p: Dropout probability (0 = no dropout)
            training: Training mode flag
            inplace: Modify input tensor in-place
            return_mask: Return dropout mask with output
            generator: Custom random number generator

        Returns:
            Output tensor (and mask if return_mask=True)
        """
        tensor_config = get_config_section('tensor_ops')
        p = tensor_config.get('p')
        training = tensor_config.get('training')
        inplace = tensor_config.get('inplace')
        return_mask = tensor_config.get('return_mask')
        generator = tensor_config.get('generator', None)

        if not training or p < 1e-8:
            if return_mask:
                return (x, torch.ones_like(x)) if not inplace else (x, torch.ones_like(x))
            return x
        
        # Validate probability
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be 0-1, got {p}")
            
        # Create mask with custom generator
        if generator is not None:
            mask = (torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)) > p
        else:
            mask = (torch.rand_like(x)) > p
        
        # Apply inverted dropout scaling
        if p < 1.0:
            mask = mask.float() / (1 - p)
        
        # Apply dropout
        if inplace:
            x.mul_(mask)
            result = x
        else:
            result = x * mask
        
        return (result, mask) if return_mask else result

    @staticmethod
    def attention_mask(lengths: torch.Tensor, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        """
        Enhanced attention mask generator supporting:
        - Padding masks (from sequence lengths)
        - Causal masks (for autoregressive decoding)
        - Key-query masks (for encoder-decoder attention)
        - Multi-dimensional masking (2D/3D)

        Args:
            lengths: Query sequence lengths (batch_size)
            max_len: Maximum query sequence length
            key_lengths: Key sequence lengths (batch_size)
            max_key_len: Maximum key sequence length
            causal: Apply causal masking (prevents attending to future positions)
            device: Output device
            dtype: Output data type

        Returns:
            Attention mask tensor (batch_size, [1], max_len, [max_key_len])
        """
        tensor_config = get_config_section('tensor_ops')
        causal = tensor_config.get('causal')
        max_len = tensor_config.get('max_len', None)
        key_lengths = tensor_config.get('key_lengths', None)
        max_key_len = tensor_config.get('max_key_len', None)
        device = tensor_config.get('device', None)

        # Validate inputs
        if device is None:
            device = lengths.device
            
        if max_len is None:
            max_len = int(lengths.max().item())
            
        # Create padding mask for queries
        batch_size = lengths.size(0)
        query_mask = (torch.arange(max_len, device=device).expand(batch_size, max_len)) < lengths.unsqueeze(1)
        
        # Handle key lengths if provided
        if key_lengths is not None:
            if max_key_len is None:
                max_key_len = int(key_lengths.max().item())
                
            # Create padding mask for keys
            key_mask = (torch.arange(max_key_len, device=device).expand(batch_size, max_key_len)) < key_lengths.unsqueeze(1)
            
            # Combine query and key masks
            mask = query_mask.unsqueeze(2) & key_mask.unsqueeze(1)
            
            # Apply causal masking if requested
            if causal:
                causal_mask = torch.tril(torch.ones(max_len, max_key_len, device=device, dtype=dtype))
                mask = mask & causal_mask.unsqueeze(0)
                
            return mask.to(dtype)
        
        # Handle causal masking for self-attention
        if causal:
            # Create causal mask (lower triangular)
            causal_mask = torch.tril(torch.ones(max_len, max_len, device=device, dtype=dtype))
            # Combine with padding mask
            mask = query_mask.unsqueeze(1) & causal_mask.unsqueeze(0)
            return mask.to(dtype)
        
        # Return basic padding mask (batch_size, max_len)
        return query_mask.to(dtype)

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
        """Numerically stable sigmoid."""
        return sigmoid_tensor(x)
    
    @staticmethod
    def to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Casts tensor to specified dtype if not already."""
        return x if x.dtype == dtype else x.to(dtype=dtype)

    @staticmethod
    def to_device(x: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
        """Moves tensor to specified device if not already."""
        return x if x.device == device else x.to(device=device)

    @staticmethod
    def unsqueeze_n(x: torch.Tensor, n: int) -> torch.Tensor:
        """Applies `unsqueeze(0)` `n` times to add leading singleton dims."""
        for _ in range(n):
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def squeeze_all(x: torch.Tensor) -> torch.Tensor:
        """Removes all singleton dimensions from tensor."""
        return x.squeeze()

    @staticmethod
    def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        """
        Creates a boolean mask from sequence lengths.
        Args:
            lengths (Tensor): (batch_size,)
            max_len (int): Optional max sequence length.
        Returns:
            mask (Tensor): (batch_size, max_len)
        """
        if max_len is None:
            max_len = int(lengths.max().item())
        return torch.arange(max_len, device=lengths.device).expand(lengths.size(0), max_len) < lengths.unsqueeze(1)

    @staticmethod
    def reverse_sequence(x: torch.Tensor, lengths: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Reverses sequences in a batch according to their lengths.
        Useful in RNNs or signal analysis.
        Args:
            x (Tensor): (batch, time, ...)
            lengths (Tensor): (batch,)
            dim (int): Sequence dimension
        """
        idx = torch.arange(x.size(dim), device=x.device).expand(len(lengths), x.size(dim))
        mask = idx < lengths.unsqueeze(1)
        rev_idx = torch.flip(idx, dims=[1])
        rev_x = torch.gather(x, dim, rev_idx.unsqueeze(-1).expand_as(x))
        return torch.where(mask.unsqueeze(-1), rev_x, x)

    @staticmethod
    def get_shape_info(x: torch.Tensor) -> Dict[str, Any]:
        """Returns metadata summary of a tensor."""
        return {
            'shape': tuple(x.shape),
            'dtype': str(x.dtype),
            'device': str(x.device),
            'requires_grad': x.requires_grad,
            'numel': x.numel(),
            'dim': x.dim()
        }


if __name__ == "__main__":
    print("\n=== Running Common ===\n")
    printer.status("TEST", "Starting Common tests", "info")
    data = torch.tensor([1.0, 2.7, 4.2, 46.0, 32.0])

    param = Parameter(data=data)
    t_ops = TensorOps()
    print(param)
    print(t_ops)

    print("\n* * * * * Phase 2 - Tensor * * * * *\n")
    X = torch.tensor([[[5.0]]], dtype=torch.float32)
    lengths = torch.tensor([3, 5, 2], dtype=torch.long)

    printer.pretty("Layer", t_ops.layer_norm(x=X), "success")
    printer.pretty("Instance", t_ops.instance_norm(x=X), "success")
    printer.pretty("int", t_ops.interpolate(x=X), "success")
    printer.pretty("drop", t_ops.dropout(x=X), "success")
    printer.pretty("mask", t_ops.attention_mask(lengths=lengths), "success")

    print("\n* * * * * Phase 3 - Utility * * * * *\n")
    
    printer.pretty("to_dtype", t_ops.to_dtype(X, torch.float64), "success")
    printer.pretty("unsqueeze", t_ops.unsqueeze_n(X, 2), "success")
    printer.pretty("squeeze_all", t_ops.squeeze_all(torch.tensor([[[1.0]]])), "success")
    printer.pretty("mask_seq", t_ops.sequence_mask(lengths), "success")
    printer.pretty("shape_info", t_ops.get_shape_info(X), "success")
    print("\n=== Successfully Ran Common ===\n")
