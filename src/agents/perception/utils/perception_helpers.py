"""Shared helper primitives for the SLAI perception subsystem.

The helpers in this module are dependency-direction leaves for perception code.
They centralize tensor normalization, modality naming, pooling, device handling,
and optimizer parameter bookkeeping without owning model policy, training
objectives, persistence, or agent lifecycle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perception_errors import *


VALID_MODALITIES: Tuple[str, ...] = ("text", "vision", "audio")
MODALITY_INPUT_KEYS: Dict[str, str] = {
    "text": "input_ids",
    "vision": "pixel_values",
    "audio": "audio_values",
}
MODALITY_INPUT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "text": ("tokens", "token_ids"),
    "vision": ("video_frames", "frames", "frame_sequence", "image", "images"),
    "audio": ("waveform", "microphone_buffer", "mic_buffer", "audio_stream"),
}


def normalize_modality_name(value: Any) -> str:
    """Normalize and validate a perception modality identifier."""

    if hasattr(value, "value"):
        value = getattr(value, "value")
    normalized = str(value).strip().lower()
    if normalized not in VALID_MODALITIES:
        raise UnsupportedModalityError(
            f"Unsupported perception modality: {value!r}.",
            details={"modality": value, "supported": VALID_MODALITIES},
        )
    return normalized


def input_key_for_modality(modality: Any) -> str:
    return MODALITY_INPUT_KEYS[normalize_modality_name(modality)]


def canonicalize_payload_aliases(modality: Any, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of a modality payload with its canonical input key populated."""

    normalized = normalize_modality_name(modality)
    result = dict(payload)
    canonical_key = MODALITY_INPUT_KEYS[normalized]
    if canonical_key in result:
        return result

    for alias in MODALITY_INPUT_ALIASES[normalized]:
        if alias in result:
            result[canonical_key] = result[alias]
            break
    return result


def resolve_torch_device(
    device: Optional[Union[str, torch.device]] = None,
    *,
    fallback_cuda_to_cpu: bool = True,
) -> torch.device:
    """Resolve a torch device without silently selecting unavailable CUDA."""

    if isinstance(device, torch.device):
        resolved = device
    else:
        setting = "auto" if device is None else str(device).strip().lower()
        if setting == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(setting)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        if fallback_cuda_to_cpu:
            return torch.device("cpu")
        raise InvalidPerceptionValueError(
            "CUDA was requested but is not available.",
            component="perception_helpers",
            details={"requested_device": str(resolved)},
        )
    return resolved


def module_device(module: nn.Module, fallback: Union[str, torch.device] = "cpu") -> torch.device:
    """Return the device of the first parameter/buffer in a module."""

    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device(fallback)


def require_tensor(value: Any, name: str, *, component: str = "perception") -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise InvalidPerceptionTypeError(
            f"'{name}' must be a torch.Tensor, got {type(value).__name__}.",
            component=component,
            details={"field": name, "actual_type": type(value).__name__},
        )
    return value


def ensure_tensor_rank(
    tensor: torch.Tensor,
    expected: Union[int, Sequence[int]],
    name: str,
    *,
    component: str = "perception",
) -> torch.Tensor:
    tensor = require_tensor(tensor, name, component=component)
    allowed = (expected,) if isinstance(expected, int) else tuple(int(item) for item in expected)
    if tensor.dim() not in allowed:
        raise PerceptionShapeError(
            f"'{name}' must have rank in {allowed}, got rank {tensor.dim()}.",
            component=component,
            details={"field": name, "shape": list(tensor.shape), "allowed_ranks": allowed},
        )
    return tensor


def ensure_last_dimension(
    tensor: torch.Tensor,
    expected: int,
    name: str,
    *,
    component: str = "perception",
) -> torch.Tensor:
    if tensor.size(-1) != int(expected):
        raise PerceptionDimensionError(
            f"'{name}' has last dimension {tensor.size(-1)}, expected {expected}.",
            component=component,
            details={"field": name, "shape": list(tensor.shape), "expected": int(expected)},
        )
    return tensor


def ensure_finite_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    component: str = "perception",
) -> torch.Tensor:
    tensor = require_tensor(tensor, name, component=component)
    if not bool(torch.isfinite(tensor).all().item()):
        raise NonFiniteLossError(
            f"'{name}' contains NaN or infinite values.",
            component=component,
            details={"field": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
        )
    return tensor


def normalize_style_id(
    style_id: Optional[Any],
    *,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize optional style IDs to a batch-aligned long tensor."""

    if style_id is None:
        return None
    if not isinstance(style_id, torch.Tensor):
        style_id = torch.as_tensor(style_id, dtype=torch.long)
    style_id = style_id.to(device=device, dtype=torch.long)
    if style_id.dim() == 0:
        style_id = style_id.expand(batch_size)
    elif style_id.dim() == 2 and style_id.size(1) == 1:
        style_id = style_id.squeeze(1)
    if style_id.dim() != 1 or style_id.size(0) != batch_size:
        raise ModalityInputError(
            "style_id must be scalar or have shape (batch,).",
            details={"shape": list(style_id.shape), "batch_size": batch_size},
        )
    return style_id


def normalize_attention_mask(
    mask: Optional[Any],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize an attention mask to boolean shape ``(B, L)``."""

    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=device)
    if mask.dim() == 1 and batch_size == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() != 2 or tuple(mask.shape) != (batch_size, seq_len):
        raise PerceptionShapeError(
            "attention_mask must have shape (batch, sequence_length).",
            component="perception_helpers",
            details={
                "shape": list(mask.shape),
                "expected": [batch_size, seq_len],
            },
        )
    return mask.to(dtype=torch.bool)


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``(B,L,D)`` using a ``(B,L)`` validity mask."""

    ensure_tensor_rank(sequence, 3, "sequence", component="perception_helpers")
    mask = normalize_attention_mask(
        mask,
        batch_size=sequence.size(0),
        seq_len=sequence.size(1),
        device=sequence.device,
    )
    assert mask is not None
    weights = mask.to(dtype=sequence.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denominator


def pool_encoded(
    encoded: torch.Tensor,
    *,
    strategy: str = "cls",
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool an encoder output to a stable ``(B,D)`` representation."""

    encoded = require_tensor(encoded, "encoded", component="perception_helpers")
    if encoded.dim() == 2:
        return encoded
    ensure_tensor_rank(encoded, 3, "encoded", component="perception_helpers")

    normalized_strategy = str(strategy).strip().lower()
    ensure_one_of(
        normalized_strategy,
        ("cls", "mean", "masked_mean"),
        "pooling_strategy",
        component="perception_helpers",
    )
    if normalized_strategy == "cls":
        return encoded[:, 0, :]
    if normalized_strategy == "mean":
        return encoded.mean(dim=1)
    if attention_mask is None:
        raise InvalidPerceptionValueError(
            "masked_mean pooling requires an attention mask.",
            component="perception_helpers",
        )
    return masked_mean(encoded, attention_mask)


def differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    """Create a scalar zero attached to ``reference``'s autograd graph."""

    reference = require_tensor(reference, "reference", component="perception_helpers")
    return reference.sum() * 0.0


def tensor_summary(tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def detach_tree(value: Any, *, cpu: bool = False) -> Any:
    """Detach tensors recursively while preserving container structure."""

    if isinstance(value, torch.Tensor):
        result = value.detach()
        return result.cpu() if cpu else result
    if isinstance(value, Mapping):
        return {key: detach_tree(item, cpu=cpu) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_tree(item, cpu=cpu) for item in value)
    if isinstance(value, list):
        return [detach_tree(item, cpu=cpu) for item in value]
    return value


def _iter_parameters(source: Any) -> Iterator[nn.Parameter]:
    if source is None:
        return
    if isinstance(source, nn.Parameter):
        yield source
        return
    if isinstance(source, nn.Module):
        yield from source.parameters()
        return
    if isinstance(source, Mapping):
        for item in source.values():
            yield from _iter_parameters(item)
        return
    if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        for item in source:
            yield from _iter_parameters(item)
        return
    raise InvalidPerceptionTypeError(
        "Unsupported parameter source.",
        component="perception_helpers",
        details={"source_type": type(source).__name__},
    )


def collect_unique_trainable_parameters(*sources: Any) -> List[nn.Parameter]:
    """Collect trainable parameters exactly once across shared modules."""

    parameters: List[nn.Parameter] = []
    seen: set[int] = set()
    for source in sources:
        for parameter in _iter_parameters(source):
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            parameters.append(parameter)
    return parameters


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    }


def add_missing_parameters_to_optimizer(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
) -> int:
    """Add newly-created trainable parameters to an existing optimizer once."""

    existing = optimizer_parameter_ids(optimizer)
    missing = [
        parameter
        for parameter in parameters
        if parameter.requires_grad and id(parameter) not in existing
    ]
    if not missing:
        return 0
    optimizer.add_param_group({"params": missing})
    return len(missing)


# ===========================
# Attention helper functions
# ===========================
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


__all__ = [
    "VALID_MODALITIES",
    "MODALITY_INPUT_KEYS",
    "MODALITY_INPUT_ALIASES",
    "normalize_modality_name",
    "input_key_for_modality",
    "canonicalize_payload_aliases",
    "resolve_torch_device",
    "module_device",
    "require_tensor",
    "ensure_tensor_rank",
    "ensure_last_dimension",
    "ensure_finite_tensor",
    "normalize_style_id",
    "normalize_attention_mask",
    "masked_mean",
    "pool_encoded",
    "differentiable_zero",
    "tensor_summary",
    "detach_tree",
    "collect_unique_trainable_parameters",
    "optimizer_parameter_ids",
    "add_missing_parameters_to_optimizer",
    "exists",
    "default",
    "l2norm",
]
