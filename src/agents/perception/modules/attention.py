"""
Attention module for the Perception Agent's subsystem
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.perception_errors import *
from ..utils.perception_helpers import *
from ...base.modules.activation_engine import he_init, lecun_normal, xavier_uniform, xavier_normal
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Attention")
printer = PrettyPrinter()

# ===========================
# Core attention functions (custom)
# ===========================
def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
            raise PerceptionShapeError(
                f"{name} must have shape (B,H,L,D).",
                component="attention",
                details={"shape": list(tensor.shape) if isinstance(tensor, torch.Tensor) else None},
            )

    if q.size(0) != k.size(0) or k.size(0) != v.size(0):
        raise PerceptionDimensionError("Q/K/V batch sizes must match.", component="attention")
    if k.size(-2) != v.size(-2):
        raise PerceptionDimensionError("K/V sequence lengths must match.", component="attention")
    if q.size(-1) != k.size(-1) or k.size(-1) != v.size(-1):
        raise PerceptionDimensionError("Q/K/V head dimensions must match.", component="attention")
    if k.size(1) != v.size(1) or k.size(1) not in (1, q.size(1)):
        raise PerceptionDimensionError(
            "K/V head count must be 1 or equal to the query head count.",
            component="attention",
            details={"q_heads": q.size(1), "kv_heads": k.size(1)},
        )


def _normalize_mask(
    mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize a keep-mask to a shape broadcastable to (B,H,Q,K)."""
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=device).bool()

    if mask.dim() == 2:
        if tuple(mask.shape) != (batch_size, key_length):
            raise PerceptionShapeError(
                "2D attention mask must have shape (B,K).",
                component="attention",
                details={"shape": list(mask.shape), "expected": [batch_size, key_length]},
            )
        return mask[:, None, None, :]

    if mask.dim() == 3:
        if mask.size(0) != batch_size or mask.size(-1) != key_length or mask.size(1) not in (1, query_length):
            raise PerceptionShapeError(
                "3D attention mask must have shape (B,1,K) or (B,Q,K).",
                component="attention",
                details={"shape": list(mask.shape)},
            )
        return mask[:, None, :, :]

    if mask.dim() == 4:
        if mask.size(0) != batch_size or mask.size(-1) != key_length or mask.size(-2) not in (1, query_length):
            raise PerceptionShapeError(
                "4D attention mask must be broadcastable to (B,H,Q,K).",
                component="attention",
                details={"shape": list(mask.shape)},
            )
        return mask

    raise PerceptionShapeError(
        "Attention mask must have rank 2, 3, or 4.",
        component="attention",
        details={"shape": list(mask.shape)},
    )


def _normalize_bias(
    bias: Optional[torch.Tensor],
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if bias is None:
        return None

    bias = torch.as_tensor(bias, device=device, dtype=dtype)
    if bias.dim() == 2:
        if tuple(bias.shape) != (query_length, key_length):
            raise PerceptionShapeError(
                "2D attention bias must have shape (Q,K).",
                component="attention",
                details={"shape": list(bias.shape)},
            )
        return bias[None, None, :, :]

    if bias.dim() == 3:
        if bias.size(0) not in (1, batch_size) or bias.size(-1) != key_length or bias.size(-2) not in (1, query_length):
            raise PerceptionShapeError(
                "3D attention bias must be broadcastable to (B,Q,K).",
                component="attention",
                details={"shape": list(bias.shape)},
            )
        return bias[:, None, :, :]

    if bias.dim() == 4:
        if bias.size(0) not in (1, batch_size) or bias.size(-1) != key_length or bias.size(-2) not in (1, query_length):
            raise PerceptionShapeError(
                "4D attention bias must be broadcastable to (B,H,Q,K).",
                component="attention",
                details={"shape": list(bias.shape)},
            )
        return bias

    raise PerceptionShapeError(
        "Attention bias must have rank 2, 3, or 4.",
        component="attention",
        details={"shape": list(bias.shape)},
    )


def _causal_mask(
    query_length: int,
    key_length: int,
    *,
    device: torch.device,
    query_offset: int = 0,
    key_offset: int = 0,
    total_query_length: Optional[int] = None,
    total_key_length: Optional[int] = None,
) -> torch.Tensor:
    total_q = query_length if total_query_length is None else total_query_length
    total_k = key_length if total_key_length is None else total_key_length
    query_base = max(total_k - total_q, 0)

    q_pos = torch.arange(query_length, device=device) + query_offset + query_base
    k_pos = torch.arange(key_length, device=device) + key_offset
    return k_pos.unsqueeze(0) > q_pos.unsqueeze(1)


def _masked_softmax(scores: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(scores, dim=-1)
    return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    attn_bias: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Exact scaled dot-product attention with boolean keep-mask semantics."""
    _validate_qkv(q, k, v)
    if not 0.0 <= float(dropout) < 1.0:
        raise PerceptionConfigurationError(
            "Attention dropout must be in [0,1).",
            component="attention",
            details={"dropout": dropout},
        )

    q_len, k_len = q.size(-2), k.size(-2)
    keep_mask = _normalize_mask(
        mask,
        batch_size=q.size(0),
        query_length=q_len,
        key_length=k_len,
        device=q.device,
    )
    bias = _normalize_bias(
        attn_bias,
        batch_size=q.size(0),
        query_length=q_len,
        key_length=k_len,
        device=q.device,
        dtype=q.dtype,
    )

    scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * (q.size(-1) ** -0.5)
    if bias is not None:
        scores = scores + bias
    if keep_mask is not None:
        scores = scores.masked_fill(~keep_mask, -torch.inf)
    if causal:
        blocked = _causal_mask(q_len, k_len, device=q.device)
        scores = scores.masked_fill(blocked[None, None], -torch.inf)

    weights = _masked_softmax(scores)
    used_weights = F.dropout(weights, p=float(dropout), training=True) if training and dropout > 0 else weights
    return torch.einsum("b h i j, b h j d -> b h i d", used_weights, v)


def memory_efficient_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    attn_bias: Optional[torch.Tensor] = None,
    q_bucket_size: int = 512,
    k_bucket_size: int = 1024,
    eps: float = 1e-8,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Exact chunked attention using an online softmax over key chunks.

    When dropout is active, query chunking is used while keys stay intact so
    dropout is applied after global softmax normalization.
    """
    _validate_qkv(q, k, v)
    if q_bucket_size <= 0 or k_bucket_size <= 0 or eps <= 0:
        raise PerceptionConfigurationError(
            "q_bucket_size, k_bucket_size, and eps must be positive.",
            component="attention",
        )

    q_len, k_len = q.size(-2), k.size(-2)
    keep_mask = _normalize_mask(
        mask,
        batch_size=q.size(0),
        query_length=q_len,
        key_length=k_len,
        device=q.device,
    )
    bias = _normalize_bias(
        attn_bias,
        batch_size=q.size(0),
        query_length=q_len,
        key_length=k_len,
        device=q.device,
        dtype=q.dtype,
    )

    def slice_qk(value, qs, qe, ks, ke):
        if value is None:
            return None
        q_slice = slice(None) if value.size(-2) == 1 else slice(qs, qe)
        k_slice = slice(None) if value.size(-1) == 1 else slice(ks, ke)
        return value[..., q_slice, k_slice]

    scale = q.size(-1) ** -0.5
    outputs = []

    # Exact attention-dropout semantics require a globally normalized softmax.
    if training and dropout > 0:
        for qs in range(0, q_len, q_bucket_size):
            qe = min(qs + q_bucket_size, q_len)
            q_chunk = q[..., qs:qe, :]
            scores = torch.einsum("b h i d, b h j d -> b h i j", q_chunk, k) * scale

            mask_chunk = slice_qk(keep_mask, qs, qe, 0, k_len)
            bias_chunk = slice_qk(bias, qs, qe, 0, k_len)
            if bias_chunk is not None:
                scores = scores + bias_chunk
            if mask_chunk is not None:
                scores = scores.masked_fill(~mask_chunk, -torch.inf)
            if causal:
                blocked = _causal_mask(
                    qe - qs,
                    k_len,
                    device=q.device,
                    query_offset=qs,
                    total_query_length=q_len,
                    total_key_length=k_len,
                )
                scores = scores.masked_fill(blocked[None, None], -torch.inf)

            weights = _masked_softmax(scores)
            weights = F.dropout(weights, p=float(dropout), training=True)
            outputs.append(torch.einsum("b h i j, b h j d -> b h i d", weights, v))
        return torch.cat(outputs, dim=-2)

    # Online log-sum-exp softmax across K chunks.
    for qs in range(0, q_len, q_bucket_size):
        qe = min(qs + q_bucket_size, q_len)
        q_chunk = q[..., qs:qe, :]
        chunk_q_len = qe - qs

        running_max = torch.full(
            (q.size(0), q.size(1), chunk_q_len, 1),
            -torch.inf,
            device=q.device,
            dtype=q.dtype,
        )
        running_sum = torch.zeros_like(running_max)
        running_value = torch.zeros(
            (q.size(0), q.size(1), chunk_q_len, v.size(-1)),
            device=v.device,
            dtype=v.dtype,
        )

        for ks in range(0, k_len, k_bucket_size):
            ke = min(ks + k_bucket_size, k_len)
            k_chunk = k[..., ks:ke, :]
            v_chunk = v[..., ks:ke, :]
            scores = torch.einsum("b h i d, b h j d -> b h i j", q_chunk, k_chunk) * scale

            mask_chunk = slice_qk(keep_mask, qs, qe, ks, ke)
            bias_chunk = slice_qk(bias, qs, qe, ks, ke)
            if bias_chunk is not None:
                scores = scores + bias_chunk
            if mask_chunk is not None:
                scores = scores.masked_fill(~mask_chunk, -torch.inf)
            if causal:
                blocked = _causal_mask(
                    chunk_q_len,
                    ke - ks,
                    device=q.device,
                    query_offset=qs,
                    key_offset=ks,
                    total_query_length=q_len,
                    total_key_length=k_len,
                )
                scores = scores.masked_fill(blocked[None, None], -torch.inf)

            block_max = scores.amax(dim=-1, keepdim=True)
            safe_block_max = torch.where(torch.isfinite(block_max), block_max, torch.zeros_like(block_max))
            exp_scores = torch.exp(scores - safe_block_max)
            exp_scores = torch.nan_to_num(exp_scores, nan=0.0, posinf=0.0, neginf=0.0)

            block_sum = exp_scores.sum(dim=-1, keepdim=True)
            block_value = torch.einsum("b h i j, b h j d -> b h i d", exp_scores, v_chunk)

            new_max = torch.maximum(running_max, block_max)
            safe_new_max = torch.where(torch.isfinite(new_max), new_max, torch.zeros_like(new_max))
            old_scale = torch.where(
                torch.isfinite(running_max),
                torch.exp(running_max - safe_new_max),
                torch.zeros_like(running_max),
            )
            new_scale = torch.where(
                torch.isfinite(block_max),
                torch.exp(block_max - safe_new_max),
                torch.zeros_like(block_max),
            )

            running_value = running_value * old_scale.to(running_value.dtype) + block_value * new_scale.to(block_value.dtype)
            running_sum = running_sum * old_scale + block_sum * new_scale
            running_max = new_max

        out = running_value / running_sum.clamp_min(float(eps)).to(running_value.dtype)
        out = torch.where(running_sum > 0, out, torch.zeros_like(out))
        outputs.append(out)

    return torch.cat(outputs, dim=-2)


def _init_linear(layer: nn.Linear, initializer: str) -> None:
    init_map = {
        "he": he_init,
        "he_normal": he_init,
        "lecun": lecun_normal,
        "xavier_uniform": xavier_uniform,
        "xavier_normal": xavier_normal,
    }
    init_fn = init_map.get(initializer, xavier_uniform)
    initialized = init_fn(tuple(layer.weight.shape), device=layer.weight.device)
    with torch.no_grad():
        layer.weight.copy_(initialized.to(dtype=layer.weight.dtype, device=layer.weight.device))
        if layer.bias is not None:
            layer.bias.zero_()


# ===========================
# Attention Modules
# ===========================
class BaseAttention(nn.Module):
    """Multi-head attention with explicit construction-time causality."""

    def __init__(self, *, causal: Optional[bool] = None) -> None:
        super().__init__()
        self.config = load_global_config()
        self.attention_config = get_config_section("attention") or {}

        self.embed_dim = int(self.config.get("embed_dim", 512))
        self.num_heads = int(self.config.get("num_heads", 8))
        self.dropout_rate = float(self.config.get("dropout_rate", 0.1))
        self.initializer = str(self.config.get("initializer", "xavier_uniform"))
        self.device = resolve_torch_device(self.config.get("device", "cpu"))

        if self.embed_dim <= 0 or self.num_heads <= 0 or self.embed_dim % self.num_heads != 0:
            raise PerceptionConfigurationError(
                "embed_dim must be positive and divisible by num_heads.",
                component="attention",
                details={"embed_dim": self.embed_dim, "num_heads": self.num_heads},
            )
        if not 0.0 <= self.dropout_rate < 1.0:
            raise PerceptionConfigurationError(
                "dropout_rate must be in [0,1).",
                component="attention",
                details={"dropout_rate": self.dropout_rate},
            )

        self.head_dim = self.embed_dim // self.num_heads
        configured_head_dim = int(self.attention_config.get("dim_head", self.head_dim))
        if configured_head_dim != self.head_dim:
            raise PerceptionConfigurationError(
                "attention.dim_head must equal embed_dim / num_heads.",
                component="attention",
                details={"configured": configured_head_dim, "derived": self.head_dim},
            )

        self.q_bucket_size = int(self.attention_config.get("q_bucket_size", 512))
        self.k_bucket_size = int(self.attention_config.get("k_bucket_size", 1024))
        self.memory_efficient = bool(self.attention_config.get("memory_efficient", False))
        self.causal = bool(self.config.get("causal", False)) if causal is None else bool(causal)

        self.to_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.to_kv = nn.Linear(self.embed_dim, self.embed_dim * 2, bias=False)
        self.to_out = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.output_attentions = False
        BaseAttention._init_weights(self)
        self.to(self.device)

    def _init_weights(self) -> None:
        for layer in (self.to_q, self.to_kv, self.to_out):
            _init_linear(layer, self.initializer)

    def _project_qkv(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)
        return q, k, v

    def _dense_with_weights(self, q, k, v, mask=None, attn_bias=None):
        q_len, k_len = q.size(-2), k.size(-2)
        keep_mask = _normalize_mask(mask, batch_size=q.size(0), query_length=q_len, key_length=k_len, device=q.device)
        bias = _normalize_bias(attn_bias, batch_size=q.size(0), query_length=q_len, key_length=k_len, device=q.device, dtype=q.dtype)

        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * (q.size(-1) ** -0.5)
        if bias is not None:
            scores = scores + bias
        if keep_mask is not None:
            scores = scores.masked_fill(~keep_mask, -torch.inf)
        if self.causal:
            scores = scores.masked_fill(_causal_mask(q_len, k_len, device=q.device)[None, None], -torch.inf)

        weights = _masked_softmax(scores)
        used_weights = F.dropout(weights, p=self.dropout_rate, training=True) if self.training and self.dropout_rate > 0 else weights
        return torch.einsum("b h i j, b h j d -> b h i d", used_weights, v), weights

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        memory_efficient: Optional[bool] = None,
        q_bucket_size: Optional[int] = None,
        k_bucket_size: Optional[int] = None,
        **kwargs
    ):
        if x.dim() != 3 or x.size(-1) != self.embed_dim:
            raise PerceptionShapeError(
                "Attention input must have shape (B,L,embed_dim).",
                component="attention",
                details={"shape": list(x.shape), "embed_dim": self.embed_dim},
            )
        context = x if context is None else context
        if context.dim() != 3 or context.size(0) != x.size(0) or context.size(-1) != self.embed_dim:
            raise PerceptionShapeError(
                "Attention context must have shape (B,S,embed_dim) with the same batch size.",
                component="attention",
                details={"input_shape": list(x.shape), "context_shape": list(context.shape)},
            )

        q, k, v = self._project_qkv(x, context)
        if self.output_attentions:
            out, weights = self._dense_with_weights(q, k, v, mask=mask, attn_bias=attn_bias)
        else:
            use_chunked = self.memory_efficient if memory_efficient is None else bool(memory_efficient)
            fn = memory_efficient_attention if use_chunked else scaled_dot_product_attention
            kwargs = dict(
                mask=mask,
                causal=self.causal,
                attn_bias=attn_bias,
                dropout=self.dropout_rate,
                training=self.training,
            )
            if use_chunked:
                kwargs.update(
                    q_bucket_size=self.q_bucket_size if q_bucket_size is None else int(q_bucket_size),
                    k_bucket_size=self.k_bucket_size if k_bucket_size is None else int(k_bucket_size),
                    eps=float(self.attention_config.get("epsilon", 1e-8)),
                )
            out = fn(q, k, v, **kwargs)
            weights = None

        out = self.to_out(rearrange(out, "b h n d -> b n (h d)"))
        return (out, weights) if self.output_attentions else out

    @staticmethod
    def build_attention_mask(
        input_ids,
        pad_token_id,
        masked_token_id=None,
        is_masked_training: bool = False,
        device: str = "cpu",
    ) -> torch.Tensor:
        input_ids = torch.as_tensor(input_ids, device=device)
        mask = input_ids.ne(pad_token_id)
        if is_masked_training and masked_token_id is not None:
            mask = mask & input_ids.ne(masked_token_id)
        return mask


class CosineAttention(BaseAttention):
    """Cosine attention with a learned positive logit scale."""

    def __init__(self, seq_len: int, *, causal: Optional[bool] = None) -> None:
        if int(seq_len) < 2:
            raise PerceptionConfigurationError(
                "CosineAttention requires seq_len >= 2.",
                component="attention",
                details={"seq_len": seq_len},
            )
        super().__init__(causal=causal)
        scale_init = -math.log(math.log2(int(seq_len) ** 2 - int(seq_len)))
        self.scale = nn.Parameter(
            torch.full(
                (1, self.num_heads, 1, 1),
                float(scale_init),
                device=self.device,
            )
        )

    def forward(self, x, context=None, mask=None, attn_bias=None, **_):
        context = x if context is None else context
        q, k, v = self._project_qkv(x, context)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        q_len, k_len = q.size(-2), k.size(-2)
        keep_mask = _normalize_mask(mask, batch_size=q.size(0), query_length=q_len, key_length=k_len, device=q.device)
        bias = _normalize_bias(attn_bias, batch_size=q.size(0), query_length=q_len, key_length=k_len, device=q.device, dtype=q.dtype)

        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale.exp()
        if bias is not None:
            scores = scores + bias
        if keep_mask is not None:
            scores = scores.masked_fill(~keep_mask, -torch.inf)
        if self.causal:
            scores = scores.masked_fill(_causal_mask(q_len, k_len, device=q.device)[None, None], -torch.inf)

        weights = _masked_softmax(scores)
        used_weights = F.dropout(weights, p=self.dropout_rate, training=True) if self.training and self.dropout_rate > 0 else weights
        out = torch.einsum("b h i j, b h j d -> b h i d", used_weights, v)
        out = self.to_out(rearrange(out, "b h n d -> b n (h d)"))
        return (out, weights) if self.output_attentions else out


class EfficientAttention(BaseAttention):
    """Non-causal kernelized linear attention using positive random features.

    It intentionally does not claim exact Performer/FAVOR+ equivalence.
    """

    def __init__(self, *, causal: Optional[bool] = None) -> None:
        super().__init__(causal=causal)
        if self.causal:
            raise UnsupportedPerceptionOptionError(
                "EfficientAttention does not implement causal linear attention.",
                component="attention",
                remediation="Use BaseAttention or MultiQueryAttention for causal decoding.",
            )

        self.epsilon = float(self.attention_config.get("epsilon", 1e-8))
        self.num_features = int(self.attention_config.get("num_features", 256))
        if self.epsilon <= 0 or self.num_features <= 0:
            raise PerceptionConfigurationError(
                "EfficientAttention epsilon and num_features must be positive.",
                component="attention",
            )

        projection = torch.randn(
            self.head_dim,
            self.num_features,
            dtype=self.to_q.weight.dtype,
            device=self.device,
        )
        projection = F.normalize(projection, p=2, dim=0) * math.sqrt(self.head_dim)
        self.register_buffer("projection_matrix", projection, persistent=True)

    def _positive_features(self, x: torch.Tensor) -> torch.Tensor:
        projection = self.projection_matrix.to(device=x.device, dtype=x.dtype)
        projected = torch.einsum("b h s d, d f -> b h s f", x, projection)
        squared_norm = 0.5 * x.square().sum(dim=-1, keepdim=True)
        exponent = torch.clamp(projected - squared_norm, min=-30.0, max=30.0)
        return torch.exp(exponent) / math.sqrt(self.num_features)

    def forward(self, x, context=None, mask=None, attn_bias=None, **_):
        if attn_bias is not None:
            raise UnsupportedPerceptionOptionError(
                "EfficientAttention does not support arbitrary additive attention bias.",
                component="attention",
            )

        context = x if context is None else context
        q, k, v = self._project_qkv(x, context)
        scale = self.head_dim ** -0.25
        qf = self._positive_features(q * scale)
        kf = self._positive_features(k * scale)

        keep_mask = _normalize_mask(
            mask,
            batch_size=q.size(0),
            query_length=q.size(-2),
            key_length=k.size(-2),
            device=q.device,
        )
        if keep_mask is not None:
            if keep_mask.size(-2) != 1:
                raise UnsupportedPerceptionOptionError(
                    "EfficientAttention supports key-validity masks only.",
                    component="attention",
                    remediation="Use a (B,K)/(B,1,K) mask or BaseAttention.",
                )
            key_mask = keep_mask[..., 0, :].unsqueeze(-1)
            kf = kf * key_mask.to(kf.dtype)
            v = v * key_mask.to(v.dtype)

        kv = torch.einsum("b h s f, b h s d -> b h f d", kf, v)
        key_sum = kf.sum(dim=-2)
        denominator = torch.einsum("b h s f, b h f -> b h s", qf, key_sum).unsqueeze(-1)
        numerator = torch.einsum("b h s f, b h f d -> b h s d", qf, kv)
        out = numerator / denominator.clamp_min(self.epsilon)
        out = torch.where(denominator > 0, out, torch.zeros_like(out))
        out = self.to_out(rearrange(out, "b h s d -> b s (h d)"))
        return (out, None) if self.output_attentions else out


class MultiQueryAttention(BaseAttention):
    """Multi-query attention with shared K/V projections across query heads."""

    def __init__(self, *, causal: Optional[bool] = None) -> None:
        super().__init__(causal=causal)
        del self.to_kv

        self.to_k = nn.Linear(self.embed_dim, self.head_dim, bias=False).to(self.device)
        self.to_v = nn.Linear(self.embed_dim, self.head_dim, bias=False).to(self.device)
        _init_linear(self.to_k, self.initializer)
        _init_linear(self.to_v, self.initializer)

        self.rotary_emb = (
            RotaryEmbedding(dim=self.head_dim)
            if self.attention_config.get("positional_encoding") == "rotary"
            else None
        )

    def forward(self, x, context=None, mask=None, attn_bias=None, memory_efficient=None, q_bucket_size=None, k_bucket_size=None):
        context = x if context is None else context
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)
        k = self.to_k(context).unsqueeze(1)
        v = self.to_v(context).unsqueeze(1)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # ``expand`` keeps shared K/V storage while making the head dimension
        # explicit for the common attention kernels.
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        if self.output_attentions:
            out, weights = self._dense_with_weights(q, k, v, mask=mask, attn_bias=attn_bias)
        else:
            use_chunked = self.memory_efficient if memory_efficient is None else bool(memory_efficient)
            fn = memory_efficient_attention if use_chunked else scaled_dot_product_attention
            kwargs = dict(mask=mask, causal=self.causal, attn_bias=attn_bias, dropout=self.dropout_rate, training=self.training)
            if use_chunked:
                kwargs.update(
                    q_bucket_size=self.q_bucket_size if q_bucket_size is None else int(q_bucket_size),
                    k_bucket_size=self.k_bucket_size if k_bucket_size is None else int(k_bucket_size),
                    eps=float(self.attention_config.get("epsilon", 1e-8)),
                )
            out = fn(q, k, v, **kwargs)
            weights = None

        out = self.to_out(rearrange(out, "b h n d -> b n (h d)"))
        return (out, weights) if self.output_attentions else out


class CrossAttention(BaseAttention):
    """Non-causal cross-attention over explicit context states."""

    def __init__(self) -> None:
        super().__init__(causal=False)
        del self.to_kv

        self.to_k_enc = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(self.device)
        self.to_v_enc = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(self.device)
        _init_linear(self.to_k_enc, self.initializer)
        _init_linear(self.to_v_enc, self.initializer)

    def forward(self, x, context=None, mask=None, attn_bias=None, memory_efficient=None, q_bucket_size=None, k_bucket_size=None):
        if context is None:
            raise PerceptionDimensionError("CrossAttention requires explicit context.", component="attention")
        if x.dim() != 3 or context.dim() != 3 or x.size(0) != context.size(0):
            raise PerceptionShapeError(
                "CrossAttention expects x=(B,Q,D) and context=(B,K,D).",
                component="attention",
                details={"x_shape": list(x.shape), "context_shape": list(context.shape)},
            )
        if x.size(-1) != self.embed_dim or context.size(-1) != self.embed_dim:
            raise PerceptionDimensionError(
                "CrossAttention embedding dimension mismatch.",
                component="attention",
                details={"expected": self.embed_dim, "x_dim": x.size(-1), "context_dim": context.size(-1)},
            )

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)
        k = rearrange(self.to_k_enc(context), "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)
        v = rearrange(self.to_v_enc(context), "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)

        if self.output_attentions:
            out, weights = self._dense_with_weights(q, k, v, mask=mask, attn_bias=attn_bias)
        else:
            use_chunked = self.memory_efficient if memory_efficient is None else bool(memory_efficient)
            fn = memory_efficient_attention if use_chunked else scaled_dot_product_attention
            kwargs = dict(mask=mask, causal=False, attn_bias=attn_bias, dropout=self.dropout_rate, training=self.training)
            if use_chunked:
                kwargs.update(
                    q_bucket_size=self.q_bucket_size if q_bucket_size is None else int(q_bucket_size),
                    k_bucket_size=self.k_bucket_size if k_bucket_size is None else int(k_bucket_size),
                    eps=float(self.attention_config.get("epsilon", 1e-8)),
                )
            out = fn(q, k, v, **kwargs)
            weights = None

        out = self.to_out(rearrange(out, "b h n d -> b n (h d)"))
        return (out, weights) if self.output_attentions else out


__all__ = [
    "scaled_dot_product_attention",
    "memory_efficient_attention",
    "BaseAttention",
    "CosineAttention",
    "EfficientAttention",
    "MultiQueryAttention",
    "CrossAttention",
]


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Attention ===\n")
    printer.status("TEST", "Starting Attention tests", "info")

    x = torch.randn(4, 128, 512)
    length = 128

    base = BaseAttention()
    base._init_weights()  # Explicit init for base
    cosine = CosineAttention(seq_len=length)
    efficient = EfficientAttention()
    mqa = MultiQueryAttention()
    cross = CrossAttention()

    printer.pretty("BaseAttention", base, "success" if base == "success" else "error")
    printer.pretty("CosineAttention", cosine, "success" if cosine == "success" else "error")
    printer.pretty("EfficientAttention", efficient, "success" if efficient == "success" else "error")
    printer.pretty("MultiQueryAttention", mqa, "success" if mqa == "success" else "error")
    printer.pretty("CrossAttention", cross, "success" if cross == "success" else "error")

    print("\n=== Successfully Ran Attention ===\n")