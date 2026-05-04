import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple, Dict, Any, Union
from pathlib import Path

from .config_loader import load_global_config, get_config_section
from ...base.modules.activation_engine import gelu_tensor, swish_tensor, mish_tensor, sigmoid_tensor
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Common")
printer = PrettyPrinter


class Parameter(torch.nn.Parameter):
    """
    Custom parameter class with additional metadata and update methods.
    Subclasses torch.nn.Parameter to maintain autograd compatibility.
    """
    def __new__(cls, data: torch.Tensor, requires_grad=True, name: Optional[str] = None):
        param = super().__new__(cls, data, requires_grad=requires_grad)
        param._name = name
        param._momentum_buffer = None
        return param

    @property
    def name(self):
        return getattr(self, '_name', None)

    def zero_grad(self) -> None:
        """Reset the gradient to zero if gradient tracking is enabled."""
        if self.requires_grad and self.grad is not None:
            self.grad.zero_()

    def step(self, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        """Apply in-place parameter update with momentum and L2 regularization."""
        if not self.requires_grad or self.grad is None:
            return

        with torch.no_grad():
            # Compute effective gradient
            grad = self.grad + weight_decay * self.data if weight_decay > 0 else self.grad

            # Update momentum buffer
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
        name = self.name or ''
        if "Parameter containing:\n" in base_repr:
            parts = base_repr.split('\n', 1)
            return f"Parameter(name={name}, containing:\n{parts[1]}"
        else:
            idx = base_repr.find('(')
            if idx != -1:
                return f"{base_repr[:idx+1]}name={name}, {base_repr[idx+1:]}"
            return f"Parameter(name={name}, {base_repr})"


class TensorOps:
    """
    A collection of stateless tensor utilities.
    All methods are static and accept explicit parameters for clarity and
    performance, avoiding repeated config lookups.
    """

    # ----------------------------------------------------------------------
    # Normalization (custom implementations for minimal PyTorch dependency)
    # ----------------------------------------------------------------------
    @staticmethod
    def layer_norm(x: torch.Tensor,
                   normalized_shape: Optional[Tuple[int, ...]] = None,
                   eps: float = 1e-5) -> torch.Tensor:
        """
        Layer normalization. If normalized_shape is not given, it is taken
        as the last dimension of x.
        This custom implementation does not include affine parameters (gamma/beta);
        they can be applied separately if needed.
        """
        if normalized_shape is None:
            normalized_shape = (x.size(-1),)
        dims = tuple(range(-len(normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + eps)

    @staticmethod
    def instance_norm(x: torch.Tensor,
                      eps: float = 1e-5,
                      affine: bool = True,
                      track_running_stats: bool = False) -> torch.Tensor:
        """
        Instance normalization for 3D/4D/5D inputs.
        This custom implementation handles batch, channel, and spatial dimensions.
        Affine parameters are optional (not stored here).
        """
        if x.dim() not in (3, 4, 5):
            raise ValueError(f"InstanceNorm expects 3D, 4D or 5D input, got {x.dim()}D")

        spatial_dims = tuple(range(2, x.dim()))
        mean = x.mean(dim=spatial_dims, keepdim=True)
        var = x.var(dim=spatial_dims, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + eps)

        if affine:
            # Affine parameters are not stored; they must be applied externally.
            logger.warning("Affine parameters are not applied in this custom instance_norm. "
                           "Apply gamma/beta separately.")
        return x_norm

    # ----------------------------------------------------------------------
    # Activation functions (using activation_engine)
    # ----------------------------------------------------------------------
    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        return gelu_tensor(x)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return swish_tensor(x)

    @staticmethod
    def mish(x: torch.Tensor) -> torch.Tensor:
        return mish_tensor(x)

    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        return sigmoid_tensor(x)

    # ----------------------------------------------------------------------
    # Interpolation (PyTorch is still needed; it's a core operation)
    # ----------------------------------------------------------------------
    @staticmethod
    def interpolate(x: torch.Tensor,
                    size: Optional[Union[int, Tuple[int, ...]]] = None,
                    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
                    mode: str = 'bilinear',
                    align_corners: Optional[bool] = None,
                    antialias: bool = False) -> torch.Tensor:
        """
        Multi‑dimensional interpolation. Adds batch/channel dimensions if needed.
        """
        if size is None and scale_factor is None:
            raise ValueError("Either size or scale_factor must be specified")

        original_shape = x.shape
        input_dim = x.dim()

        if input_dim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif input_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif input_dim == 3:
            x = x.unsqueeze(0)

        kwargs = {'antialias': antialias} if antialias else {}
        x = F.interpolate(x, size=size, scale_factor=scale_factor,
                          mode=mode, align_corners=align_corners, **kwargs)

        if input_dim == 1:
            x = x.squeeze(0).squeeze(0)
        elif input_dim == 2:
            x = x.squeeze(0).squeeze(0)
        elif input_dim == 3:
            x = x.squeeze(0)
        return x

    # ----------------------------------------------------------------------
    # Dropout (custom implementation)
    # ----------------------------------------------------------------------
    @staticmethod
    def dropout(x: torch.Tensor,
                p: float = 0.5,
                training: bool = True,
                inplace: bool = False,
                return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Dropout with optional mask return. Inverted scaling is applied.
        """
        if not training or p < 1e-8:
            if return_mask:
                return x, torch.ones_like(x)
            return x

        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be 0-1, got {p}")

        mask = (torch.rand_like(x) > p).float() / (1 - p)
        if inplace:
            x.mul_(mask)
            result = x
        else:
            result = x * mask

        return (result, mask) if return_mask else result

    # ----------------------------------------------------------------------
    # Attention masks (custom)
    # ----------------------------------------------------------------------
    @staticmethod
    def attention_mask(lengths: torch.Tensor,
                       max_len: Optional[int] = None,
                       key_lengths: Optional[torch.Tensor] = None,
                       max_key_len: Optional[int] = None,
                       causal: bool = False,
                       dtype: torch.dtype = torch.bool) -> torch.Tensor:
        """
        Create padding and/or causal attention masks.
        """
        device = lengths.device
        if max_len is None:
            max_len = int(lengths.max().item())

        batch_size = lengths.size(0)
        query_mask = (torch.arange(max_len, device=device).expand(batch_size, max_len)
                      < lengths.unsqueeze(1))

        if key_lengths is not None:
            if max_key_len is None:
                max_key_len = int(key_lengths.max().item())
            key_mask = (torch.arange(max_key_len, device=device).expand(batch_size, max_key_len)
                        < key_lengths.unsqueeze(1))
            mask = query_mask.unsqueeze(2) & key_mask.unsqueeze(1)
            if causal:
                causal_mask = torch.tril(torch.ones(max_len, max_key_len, device=device, dtype=dtype))
                mask = mask & causal_mask.unsqueeze(0)
        else:
            if causal:
                causal_mask = torch.tril(torch.ones(max_len, max_len, device=device, dtype=dtype))
                mask = query_mask.unsqueeze(1) & causal_mask.unsqueeze(0)
            else:
                mask = query_mask

        return mask.to(dtype)

    # ----------------------------------------------------------------------
    # Utility functions
    # ----------------------------------------------------------------------
    @staticmethod
    def pad_sequence(x: torch.Tensor, max_len: int, axis: int = 1, value: float = 0) -> torch.Tensor:
        pad_size = max_len - x.shape[axis]
        if pad_size <= 0:
            return x
        pad_dims = [(0, 0)] * x.dim()
        pt_pad_dims = [0] * (2 * x.dim())
        pad_idx = 2 * (x.dim() - 1 - axis)
        pt_pad_dims[pad_idx + 1] = pad_size
        return F.pad(x, pt_pad_dims, value=value)

    @staticmethod
    def to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return x.to(dtype) if x.dtype != dtype else x

    @staticmethod
    def to_device(x: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
        return x.to(device) if x.device != device else x

    @staticmethod
    def unsqueeze_n(x: torch.Tensor, n: int) -> torch.Tensor:
        for _ in range(n):
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def squeeze_all(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()

    @staticmethod
    def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        if max_len is None:
            max_len = int(lengths.max().item())
        return torch.arange(max_len, device=lengths.device).expand(lengths.size(0), max_len) < lengths.unsqueeze(1)

    @staticmethod
    def reverse_sequence(x: torch.Tensor, lengths: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Reverse sequences in a batch according to their lengths.
        """
        idx = torch.arange(x.size(dim), device=x.device).expand(len(lengths), x.size(dim))
        mask = idx < lengths.unsqueeze(1)
        rev_idx = torch.flip(idx, dims=[1])
        rev_x = torch.gather(x, dim, rev_idx.unsqueeze(-1).expand_as(x))
        return torch.where(mask.unsqueeze(-1), rev_x, x)

    @staticmethod
    def get_shape_info(x: torch.Tensor) -> Dict[str, Any]:
        return {
            'shape': tuple(x.shape),
            'dtype': str(x.dtype),
            'device': str(x.device),
            'requires_grad': x.requires_grad,
            'numel': x.numel(),
            'dim': x.dim(),
        }


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Common Tests ===\n")
    t_ops = TensorOps()
    X = torch.randn(2, 3, 4, 5)  # (B, C, H, W)

    # Test layer_norm
    normed = t_ops.layer_norm(X)
    print("layer_norm shape:", normed.shape)

    # Test instance_norm
    inst_normed = t_ops.instance_norm(X)
    print("instance_norm shape:", inst_normed.shape)

    # Test interpolate
    upsampled = t_ops.interpolate(X, scale_factor=2.0, mode='bilinear')
    print("interpolate shape:", upsampled.shape)

    # Test dropout
    dropped, mask = t_ops.dropout(X, p=0.5, return_mask=True)
    print("dropout shape:", dropped.shape)

    # Test attention mask
    lengths = torch.tensor([3, 5, 2])
    mask = t_ops.attention_mask(lengths, causal=True)
    print("attention mask shape:", mask.shape)

    print("\n=== Successfully Ran Common ===\n")