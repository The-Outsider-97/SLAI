import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Optional, Tuple, Union
from einops import rearrange, reduce
from rotary_embedding_torch import RotaryEmbedding

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import TensorOps, Parameter
from ...base.utils.activation_engine import (
    get_activation,
    he_init, lecun_normal, xavier_uniform, xavier_normal
)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Attention")
printer = PrettyPrinter

# ===========================
# Helper functions
# ===========================
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

# ===========================
# Core attention functions (custom)
# ===========================
def scaled_dot_product_attention(q, k, v, mask=None, causal=False, attn_bias=None):
    """Standard scaled dot‑product attention."""
    scale = q.shape[-1] ** -0.5
    q = q * scale
    sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias):
        sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        if mask.ndim == 2:
            mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones(i, j, device=q.device, dtype=torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, mask_value)

    attn = sim.softmax(dim=-1)
    return torch.einsum('b h i j, b h j d -> b h i d', attn, v)


def memory_efficient_attention(q, k, v, mask=None, causal=False, attn_bias=None,
                               q_bucket_size=512, k_bucket_size=1024, eps=1e-8,
                               dropout=0.0, training=False):
    """
    Memory‑efficient attention with chunking and optional checkpointing.
    """
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # Determine if we need gradient checkpointing
    needs_backward = any(t.requires_grad for t in (q, k, v))

    # Define chunk processing function
    def summarize_qkv_chunk(q_chunk, k_chunk, v_chunk, mask_chunk, attn_bias_chunk,
                            causal, q_start, k_start, dropout):
        q_chunk_size = q_chunk.shape[-2]
        k_chunk_size = k_chunk.shape[-2]
        weight = torch.einsum('b h i d, b h j d -> b h i j', q_chunk, k_chunk)

        if exists(attn_bias_chunk):
            weight = weight + attn_bias_chunk

        mask_value = -torch.finfo(weight.dtype).max

        if exists(mask_chunk):
            mask_chunk = rearrange(mask_chunk, 'b j -> b 1 1 j')
            weight = weight.masked_fill(~mask_chunk, mask_value)

        if causal and q_start < (k_start + k_chunk_size - 1):
            causal_mask = torch.ones((q_chunk_size, k_chunk_size), device=weight.device,
                                     dtype=torch.bool).triu(q_start - k_start + 1)
            weight = weight.masked_fill(causal_mask, mask_value)

        weight_max = weight.amax(dim=-1, keepdim=True).detach()
        weight = weight - weight_max
        exp_weight = weight.exp()
        if dropout > 0:
            exp_weight = F.dropout(exp_weight, p=dropout)

        weighted_value = torch.einsum('b h i j, b h j d -> b h i d', exp_weight, v_chunk)
        return exp_weight.sum(dim=-1), weighted_value, rearrange(weight_max, '... 1 -> ...')

    checkpointed_summarize = partial(torch.utils.checkpoint.checkpoint, summarize_qkv_chunk)
    summarize_fn = checkpointed_summarize if needs_backward else summarize_qkv_chunk

    # Split into chunks
    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks = k.split(k_bucket_size, dim=-2)
    v_chunks = v.split(k_bucket_size, dim=-2)
    mask_chunks = mask.split(k_bucket_size, dim=-1) if exists(mask) else [None] * len(k_chunks)

    if exists(attn_bias):
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim=-2)
        attn_bias_chunks = [b.split(k_bucket_size, dim=-1) for b in attn_bias_chunks]

    out = []
    for q_idx, q_chunk in enumerate(q_chunks):
        exp_weights, weighted_values, weight_maxes = [], [], []
        q_start = q_idx * q_bucket_size

        for k_idx, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            k_start = k_idx * k_bucket_size

            # Skip if causal mask would hide all
            if causal and k_start > (q_start + q_chunk.shape[-2] - 1):
                continue

            attn_bias_chunk = attn_bias_chunks[q_idx][k_idx] if exists(attn_bias) else None
            current_dropout = dropout if training else 0.

            exp_w, wv, wm = summarize_fn(
                q_chunk, k_chunk, v_chunk, mask_chunk, attn_bias_chunk,
                causal, q_start, k_start, current_dropout
            )
            exp_weights.append(exp_w)
            weighted_values.append(wv)
            weight_maxes.append(wm)

        weight_maxes = torch.stack(weight_maxes, dim=-1)
        weighted_values = torch.stack(weighted_values, dim=-1)
        exp_weights = torch.stack(exp_weights, dim=-1)

        global_max = weight_maxes.amax(dim=-1, keepdim=True)
        renorm = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm
        weighted_values = weighted_values * rearrange(renorm, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim=-1)
        all_weights = exp_weights.sum(dim=-1)

        out.append(all_values / (rearrange(all_weights, '... -> ... 1') + eps))

    return torch.cat(out, dim=-2)


# ===========================
# Attention Modules
# ===========================
class BaseAttention(nn.Module):
    """Base class for all attention variants with common projections."""
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.embed_dim = self.config.get('embed_dim')
        self.num_heads = self.config.get('num_heads')
        self.dropout_rate = self.config.get('dropout_rate')
        self.causal = self.config.get('causal')
        self.initializer = self.config.get('initializer', 'xavier_uniform')
        self.device = self.config.get('device', 'cpu')

        self.attention_config = get_config_section('attention')
        self.dim_head = self.attention_config.get('dim_head', self.embed_dim // self.num_heads)
        self.q_bucket_size = self.attention_config.get('q_bucket_size', 512)
        self.k_bucket_size = self.attention_config.get('k_bucket_size', 1024)
        self.memory_efficient = self.attention_config.get('memory_efficient', False)

        # Ensure embed_dim is divisible by num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads

        # Projections (to be overridden by children if needed)
        self.to_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.to_kv = nn.Linear(self.embed_dim, self.embed_dim * 2, bias=False)
        self.to_out = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.output_attentions = False

    def _init_weights(self):
        """Initialize base weights using the configured initializer."""
        init_map = {
            'he': he_init,
            'lecun': lecun_normal,
            'xavier_uniform': xavier_uniform,
            'xavier_normal': xavier_normal,
        }
        init_fn = init_map.get(self.initializer, xavier_uniform)

        for layer in [self.to_q, self.to_kv, self.to_out]:
            weight_shape = layer.weight.shape
            with torch.no_grad():
                layer.weight.data = init_fn(weight_shape, device=self.device)

    def forward(self, x, context=None, mask=None, attn_bias=None,
                memory_efficient=None, q_bucket_size=None, k_bucket_size=None):
        # Handle optional parameters
        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)
        context = default(context, x)
    
        # Project inputs
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
    
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
    
        if self.output_attentions:
            # Use the custom function that returns both output and weights
            out, attn_weights = self._attention_with_weights(q, k, v, mask, attn_bias, self.causal)
        else:
            # Choose attention mechanism based on memory_efficient flag
            attn_fn = memory_efficient_attention if memory_efficient else scaled_dot_product_attention
            out = attn_fn(q, k, v, mask=mask, attn_bias=attn_bias, causal=self.causal)
            attn_weights = None
    
        # Combine heads and project
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
    
        if self.output_attentions:
            return out, attn_weights
        return out

    def _attention(self, q, k, v, mask, attn_bias, causal):
        # Standard scaled dot‑product attention (no weights returned)
        scale = q.shape[-1] ** -0.5
        q = q * scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        if attn_bias is not None:
            sim = sim + attn_bias
        mask_value = -torch.finfo(sim.dtype).max
        if mask is not None:
            if mask.ndim == 2:
                mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)
        if causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device=q.device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)
        attn = sim.softmax(dim=-1)
        return torch.einsum('b h i j, b h j d -> b h i d', attn, v)

    def _attention_with_weights(self, q, k, v, mask, attn_bias, causal):
        # Same as above but returns (output, attention_weights)
        scale = q.shape[-1] ** -0.5
        q = q * scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        if attn_bias is not None:
            sim = sim + attn_bias
        mask_value = -torch.finfo(sim.dtype).max
        if mask is not None:
            if mask.ndim == 2:
                mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)
        if causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device=q.device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out, attn

    def _split_heads(self, x):
        """Split last dimension into multiple heads."""
        batch_size, seq_len, embed_dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        """Combine heads back into single embedding."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.embed_dim)

    @staticmethod
    def build_attention_mask(input_ids, pad_token_id, masked_token_id=None, is_masked_training=False, device='cpu'):
        """Build attention mask from token IDs."""
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=device)
        else:
            input_ids = input_ids.to(device)

        base_mask = (input_ids != pad_token_id)
        if is_masked_training and masked_token_id is not None:
            masked_positions = (input_ids == masked_token_id)
            base_mask = base_mask & (~masked_positions)

        # Expand to (batch, 1, 1, seq) for broadcasting
        return base_mask.unsqueeze(1).unsqueeze(2)


class CosineAttention(BaseAttention):
    """Attention with cosine similarity and learnable scaling."""
    def __init__(self, seq_len):
        super().__init__()
        scale_init = -math.log(math.log2(seq_len ** 2 - seq_len))
        self.scale = Parameter(torch.full((1, self.num_heads, 1, 1), scale_init))
        self._init_weights()  # Base has all layers, so this works

    def forward(self, x, context=None, mask=None, attn_bias=None,
                memory_efficient=None, q_bucket_size=None, k_bucket_size=None):
        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)

        # Cosine attention: normalize q and k, apply scale
        q, k = map(l2norm, (q, k))
        q = q * self.scale.exp()

        attn_fn = memory_efficient_attention if memory_efficient else scaled_dot_product_attention
        out = attn_fn(q, k, v, mask=mask, attn_bias=attn_bias, causal=self.causal)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        if self.output_attentions:
            # For cosine attention, we can compute weights similarly, but for simplicity return None
            return out, None
        return out


class EfficientAttention(BaseAttention):
    """Linear‑time attention using random feature maps (Performer style)."""
    def __init__(self):
        super().__init__()
        self.epsilon = self.attention_config.get('epsilon', 1e-8)
        self.num_features = self.attention_config.get('num_features', 256)
        self.kernel_fn = self._positive_random_features

        # Random projection matrix (dim_head → num_features)
        self.register_buffer("projection_matrix", self._create_random_projection())

        # Projections to lower‑rank space (per head)
        self.query_proj = nn.Linear(self.head_dim, self.num_features, bias=False)
        self.key_proj = nn.Linear(self.head_dim, self.num_features, bias=False)
        self.value_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Final projection (after mixing heads)
        self.final_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Initialize all weights (including base and extra)
        self._init_weights()

    def _init_weights(self):
        # First, initialize base layers (to_q, to_kv, to_out)
        super()._init_weights()

        # Then initialize the extra layers
        init_map = {
            'he': he_init,
            'lecun': lecun_normal,
            'xavier_uniform': xavier_uniform,
            'xavier_normal': xavier_normal,
        }
        init_fn = init_map.get(self.initializer, xavier_uniform)

        for layer in [self.query_proj, self.key_proj, self.value_proj, self.final_proj]:
            weight_shape = layer.weight.shape
            with torch.no_grad():
                layer.weight.data = init_fn(weight_shape, device=self.device)

    def _create_random_projection(self):
        """Generate an orthogonal random matrix for kernel mapping."""
        ortho = torch.randn(self.num_features, self.head_dim)
        ortho, _ = torch.linalg.qr(ortho)
        return ortho.T

    def _positive_random_features(self, x):
        """Apply FAVOR+ kernel: exp(-||x||^2/2) * exp(-||y||^2/2) * exp(x·y)."""
        x_proj = torch.einsum('b h s d, d f -> b h s f', x, self.projection_matrix)
        return torch.exp(-x_proj ** 2 / 2)

    def forward(self, x, context=None, mask=None, attn_bias=None,
                memory_efficient=None, q_bucket_size=None, k_bucket_size=None):
        context = default(context, x)

        # Get base projections (queries, keys, values)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape to (batch, heads, seq, head_dim)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)

        # Apply kernel transformation per head
        q_kernel = self.kernel_fn(q)   # (batch, heads, seq, num_features)
        k_kernel = self.kernel_fn(k)   # (batch, heads, seq, num_features)

        # Masking: set masked positions to zero
        if exists(mask):
            if mask.ndim == 4:
                mask = mask.squeeze(1).squeeze(1)  # (batch, seq)
            elif mask.ndim == 3 and mask.size(1) == 1:
                mask = mask.squeeze(1)            # (batch, seq)
            if mask.ndim == 2:
                # Convert to boolean if it's not already
                mask_bool = mask.to(torch.bool) if mask.dtype != torch.bool else mask
                mask_expanded = mask_bool.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq, 1)
                k_kernel = k_kernel.masked_fill(~mask_expanded, 0.0)
                v = v.masked_fill(~mask_expanded, 0.0)

        # Compute KV contraction
        kv = torch.einsum('b h s f, b h s d -> b h f d', k_kernel, v)

        # Denominator: sum of kernel values for each head
        z = 1 / (torch.einsum('b h s f, b h f -> b h s', q_kernel, k_kernel.sum(dim=2)) + self.epsilon)
        z = z.unsqueeze(-1)  # (batch, heads, seq, 1)

        # Output = q_kernel @ kv, scaled by 1/z
        out = torch.einsum('b h s f, b h f d -> b h s d', q_kernel, kv) * z

        # Combine heads and project
        out = rearrange(out, 'b h s d -> b s (h d)')
        if self.output_attentions:
            return out, None
        return out


class MultiQueryAttention(BaseAttention):
    """Multi‑query attention: keys and values are shared across heads."""
    def __init__(self):
        super().__init__()
        # Remove the inherited to_kv
        delattr(self, 'to_kv')

        # Create separate key and value projections (single head)
        self.to_k = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.embed_dim, self.head_dim, bias=False)

        # Rotary embeddings (optional)
        if self.attention_config.get('positional_encoding') == 'rotary':
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        else:
            self.rotary_emb = None

        # Initialize all weights (including base and new layers)
        self._init_weights()

    def _init_weights(self):
        # Initialize base layers (to_q, to_out) – note to_kv is gone
        init_map = {
            'he': he_init,
            'lecun': lecun_normal,
            'xavier_uniform': xavier_uniform,
            'xavier_normal': xavier_normal,
        }
        init_fn = init_map.get(self.initializer, xavier_uniform)

        for layer in [self.to_q, self.to_out]:
            weight_shape = layer.weight.shape
            with torch.no_grad():
                layer.weight.data = init_fn(weight_shape, device=self.device)

        # Initialize the extra layers (to_k, to_v)
        for layer in [self.to_k, self.to_v]:
            weight_shape = layer.weight.shape
            with torch.no_grad():
                layer.weight.data = init_fn(weight_shape, device=self.device)

    def forward(self, x, context=None, mask=None, attn_bias=None,
                memory_efficient=None, q_bucket_size=None, k_bucket_size=None):
        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)
        context = default(context, x)

        # Queries: multi‑head
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)

        # Keys/values: single head
        k = self.to_k(context)  # (batch, seq, head_dim)
        v = self.to_v(context)  # (batch, seq, head_dim)
        # Add head dimension for broadcasting
        k = k.unsqueeze(1)      # (batch, 1, seq, head_dim)
        v = v.unsqueeze(1)      # (batch, 1, seq, head_dim)

        # Apply rotary embeddings if present
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)

        attn_fn = memory_efficient_attention if memory_efficient else scaled_dot_product_attention
        out = attn_fn(q, k, v, mask=mask, attn_bias=attn_bias, causal=self.causal)

        out = rearrange(out, 'b h n d -> b n (h d)')
        if self.output_attentions:
            return out, None
        return out


class CrossAttention(BaseAttention):
    """Cross‑attention with separate key/value projections for encoder."""
    def __init__(self):
        super().__init__()
        # Remove inherited to_kv
        delattr(self, 'to_kv')

        # Encoder projections
        self.to_k_enc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.to_v_enc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Initialize all weights (including base and new layers)
        self._init_weights()

    def _init_weights(self):
        # Initialize base layers (to_q, to_out) – note to_kv is gone
        init_map = {
            'he': he_init,
            'lecun': lecun_normal,
            'xavier_uniform': xavier_uniform,
            'xavier_normal': xavier_normal,
        }
        init_fn = init_map.get(self.initializer, xavier_uniform)

        for layer in [self.to_q, self.to_out]:
            weight_shape = layer.weight.shape
            with torch.no_grad():
                layer.weight.data = init_fn(weight_shape, device=self.device)

        # Initialize the extra layers (to_k_enc, to_v_enc)
        for layer in [self.to_k_enc, self.to_v_enc]:
            weight_shape = layer.weight.shape
            with torch.no_grad():
                layer.weight.data = init_fn(weight_shape, device=self.device)

    def forward(self, x, context, mask=None, attn_bias=None,
                memory_efficient=None, q_bucket_size=None, k_bucket_size=None):
        if context is None:
            raise ValueError("CrossAttention requires explicit context input")

        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)

        # Queries from decoder, keys/values from encoder
        q = self.to_q(x)
        k = self.to_k_enc(context)
        v = self.to_v_enc(context)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)

        attn_fn = memory_efficient_attention if memory_efficient else scaled_dot_product_attention
        out = attn_fn(q, k, v, mask=mask, attn_bias=attn_bias, causal=self.causal)

        out = rearrange(out, 'b h n d -> b n (h d)')
        if self.output_attentions:
            return out, None
        return out


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

    printer.pretty("BaseAttention", base, "success")
    printer.pretty("CosineAttention", cosine, "success")
    printer.pretty("EfficientAttention", efficient, "success")
    printer.pretty("MultiQueryAttention", mqa, "success")
    printer.pretty("CrossAttention", cross, "success")

    print("\n=== Successfully Ran Attention ===\n")