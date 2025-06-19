import torch
import math
import yaml
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding 
from functools import partial
from torch import nn, einsum
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from pathlib import Path

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.base.utils.common import TensorOps, Parameter
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Attention")
printer = PrettyPrinter

# ===========================
# helper functions
# ===========================
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

# ===========================
# Attention Mechanisms
# ===========================
def scaled_dot_product_attention(q, k, v, mask=None, causal=False, attn_bias=None):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    sim = einsum('b h i d, b h j d -> b h i j', q, k)

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
    return einsum('b h i j, b h j d -> b h i d', attn, v)

def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices, dropout):
    q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = *qk_start_indices, q.shape[-2], k.shape[-2], q.device

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias_chunk):
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype=torch.bool, device=device).triu(q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    weight_max = weight.amax(dim=-1, keepdim=True).detach()
    weight = weight - weight_max
    exp_weight = weight.exp()
    
    if dropout > 0:
        exp_weight = F.dropout(exp_weight, p=dropout)
        
    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)
    return exp_weight.sum(dim=-1), weighted_value, rearrange(weight_max, '... 1 -> ...')

checkpointed_summarize = partial(checkpoint, summarize_qkv_chunk)

def memory_efficient_attention(q, k, v, mask=None, causal=False, attn_bias=None, 
                              q_bucket_size=512, k_bucket_size=1024, eps=1e-8, 
                              dropout=0., training=False):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # Determine if we need gradient checkpointing
    needs_backward = any(t.requires_grad for t in (q, k, v))
    summarize_fn = checkpointed_summarize if needs_backward else summarize_qkv_chunk

    # Chunk tensors for memory efficiency
    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks = k.split(k_bucket_size, dim=-2)
    v_chunks = v.split(k_bucket_size, dim=-2)
    mask_chunks = mask.split(k_bucket_size, dim=-1) if exists(mask) else [None] * len(k_chunks)

    if exists(attn_bias):
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim=-2)
        attn_bias_chunks = [b.split(k_bucket_size, dim=-1) for b in attn_bias_chunks]

    # Process chunks
    out = []
    for q_idx, q_chunk in enumerate(q_chunks):
        exp_weights, weighted_values, weight_maxes = [], [], []

        for k_idx, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            q_start = q_idx * q_bucket_size
            k_start = k_idx * k_bucket_size

            # Skip computation if masked by causality
            if causal and k_start > (q_start + q_chunk.shape[-2] - 1):
                continue

            attn_bias_chunk = attn_bias_chunks[q_idx][k_idx] if exists(attn_bias) else None
            current_dropout = dropout if training else 0.

            # Compute attention for chunk
            exp_w, wv, wm = summarize_fn(
                q_chunk, k_chunk, v_chunk, mask_chunk, 
                attn_bias_chunk, causal, (q_start, k_start), 
                current_dropout
            )
            exp_weights.append(exp_w)
            weighted_values.append(wv)
            weight_maxes.append(wm)

        # Combine results across chunks
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
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.embed_dim = self.config.get('embed_dim')
        self.num_heads = self.config.get('num_heads')
        self.initializer = self.config.get('initializer')
        self.device = self.config.get('device')
        self.dropout_rate = self.config.get('dropout_rate')
        self.causal = self.config.get('causal')

        self.attention_config = get_config_section('attention')
        self.dim_head = self.attention_config.get('dim_head')
        self.q_bucket_size = self.attention_config.get('q_bucket_size')
        self.k_bucket_size = self.attention_config.get('k_bucket_size')
        self.memory_efficient = self.attention_config.get('memory_efficient')
        self.head_dim = self.embed_dim // self.num_heads
        inner_dim = self.num_heads * self.dim_head

        self.heads = self.num_heads
        self.dropout = self.dropout_rate

        self.to_q = nn.Linear( self.embed_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear( self.embed_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim,  self.embed_dim, bias=False)

        # Initialize parameters with proper scaling
        init_fn = getattr(TensorOps, self.initializer) # f"{self.initializer}_init")
        self.q_proj = torch.nn.Parameter(init_fn((self.embed_dim, self.embed_dim), self.embed_dim, device=self.device))
        self.k_proj = torch.nn.Parameter(init_fn((self.embed_dim, self.embed_dim), self.embed_dim, device=self.device))
        self.v_proj = torch.nn.Parameter(init_fn((self.embed_dim, self.embed_dim), self.embed_dim, device=self.device))
        self.out_proj = torch.nn.Parameter(init_fn((self.embed_dim, self.embed_dim), self.embed_dim, device=self.device))
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self._cache = {}
        self.observers = []

        logger.info(f"Base Attention is successfully initialized")

    def load_from_dict(self, weights_dict, prefix=''):
        """Load pretrained weights from dictionary"""
        # Map weights to current module's parameters
        mapping = {
            f'{prefix}attention.self.query.weight': 'to_q.weight',
            f'{prefix}attention.self.key.weight': 'to_kv.weight',
            f'{prefix}attention.self.value.weight': 'to_kv.weight',
            f'{prefix}attention.output.dense.weight': 'to_out.weight'
        }
        
        for hf_key, our_key in mapping.items():
            if hf_key in weights_dict:
                parts = our_key.split('.')
                module = self
                for part in parts[:-1]:
                    module = getattr(module, part)
                param = getattr(module, parts[-1])
                param.data = weights_dict[hf_key].data.clone().to(param.device)

    def add_observer(self, observer):
        self.observers.append(observer)

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
        
        # Rearrange for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        # Select attention mechanism
        attn_fn = memory_efficient_attention if memory_efficient else scaled_dot_product_attention
        out = attn_fn(
            q, k, v, 
            mask=mask, 
            attn_bias=attn_bias, 
            causal=self.causal
        )
        
        # Combine heads and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the last dimension (embedding) into multiple heads.
        
        Input shape: (batch_size, seq_len, embed_dim)
        Output shape: (batch_size, num_heads, seq_len, head_dim)
    
        This is necessary for multi-head attention, allowing each head
        to attend independently over the input.
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"
    
        # Reshape to split embed_dim into (num_heads, head_dim)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
    
        # Rearrange dimensions to: (batch, num_heads, seq_len, head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines multi-head tensor into a single embedding dimension.
    
        Input shape: (batch_size, num_heads, seq_len, head_dim)
        Output shape: (batch_size, seq_len, embed_dim)
    
        Reverses the operation from _split_heads().
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert num_heads == self.num_heads and head_dim == self.head_dim, "Shape mismatch in combine_heads"
    
        # Rearrange back to (batch, seq_len, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
    
        # Flatten last two dimensions to (embed_dim)
        return x.view(batch_size, seq_len, self.embed_dim)

    @staticmethod
    def build_attention_mask(input_ids, pad_token_id, masked_token_id=None, is_masked_training=False, device='cpu'):
        """
        Constructs attention masks.
    
        Args:
            input_ids (np.ndarray): shape [batch, seq]
            pad_token_id (int): ID used for [PAD]
            masked_token_id (int, optional): ID used for [MASK] during masked LM.
            is_masked_training (bool): if True, will block masked positions from attention.
    
        Returns:
            np.ndarray: attention mask of shape [batch, 1, 1, seq]
        """
        # Base mask: 1 for real tokens, 0 for padding
        if not isinstance(input_ids, torch.Tensor):
             input_ids = torch.tensor(input_ids, device=device)
        else:
             input_ids = input_ids.to(device)

        # Base mask: True for real tokens, False for padding
        base_mask = (input_ids != pad_token_id)

        if is_masked_training and masked_token_id is not None:
            masked_positions = (input_ids == masked_token_id)
            base_mask = base_mask & (~masked_positions)

        # Expand to [batch, 1, 1, seq] (compatible with attention scores B,H,S,S)
        attention_mask = base_mask.unsqueeze(1).unsqueeze(2) # Add head and query seq dims
        return attention_mask

class CosineAttention(BaseAttention):
    def __init__(self, seq_len):
        super().__init__()
        # Learned scale parameter with log initialization
        scale_init = -math.log(math.log2(seq_len ** 2 - seq_len))
        self.scale = Parameter(torch.full((1, self.num_heads, 1, 1), scale_init))

    def load_from_dict(self, weights_dict, prefix=''):
        super().load_from_dict(weights_dict, prefix)

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
        
        # Rearrange for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        # Apply cosine attention modifications
        q, k = map(l2norm, (q, k))
        q = q * self.scale.exp()
        
        # Select attention mechanism
        attn_fn = memory_efficient_attention if memory_efficient else scaled_dot_product_attention
        out = attn_fn(
            q, k, v, 
            mask=mask, 
            attn_bias=attn_bias, 
            causal=self.causal,
        )
        
        # Combine heads and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class EfficientAttention(BaseAttention):
    """
    Linear-time approximation of attention using kernelized feature maps.
    Implements the Performer-style mechanism.
    """
    def __init__(self):
        super().__init__()
        self.epsilon = self.attention_config.get('epsilon')
        self.num_features = self.attention_config.get('num_features')
        self.kernel_fn = self._positive_random_features

        # Projections for query, key, value into lower-rank space
        self.query_proj = nn.Linear(self.embed_dim, self.num_features, bias=False)
        self.key_proj = nn.Linear(self.embed_dim, self.num_features, bias=False)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Final projection to map back to embed space
        self.final_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Random projection matrix for kernel mapping (dim_head → num_features)
        self.register_buffer("projection_matrix", self._create_random_projection())

    def _positive_random_features(self, x):
        """
        Applies FAVOR+ random feature kernel transformation.
        Ensures positivity and approximates exp(QKᵀ) without full pairwise computation.
        """
        x_proj = torch.matmul(x, self.projection_matrix)  # [B, H, T, F]
        return torch.exp(-x_proj**2 / 2)

    def _create_random_projection(self):
        """
        Generates an orthogonal random projection matrix using QR decomposition.
        Shape: [dim_head, num_features]
        """
        ortho = torch.randn((self.num_features, self.dim_head))
        ortho, _ = torch.linalg.qr(ortho)
        return ortho.T

    def load_from_dict(self, weights_dict, prefix=''):
        """Handle weight conversion for efficient attention"""
        # Convert weights to standard attention format
        standard_weights = {}
        for key in weights_dict:
            if 'query' in key:
                standard_weights[key.replace('query', 'self.query')] = weights_dict[key]
            elif 'key' in key:
                standard_weights[key.replace('key', 'self.key')] = weights_dict[key]
            elif 'value' in key:
                standard_weights[key.replace('value', 'self.value')] = weights_dict[key]
        super().load_from_dict(standard_weights, prefix)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.query_proj(x)  # [B, T, F]
        k = self.key_proj(x)
        v = self.value_proj(x)

        q = self.kernel_fn(q)
        k = self.kernel_fn(k)

        # Key-Value contraction
        kv = torch.einsum('bnd,bne->bde', k, v)  # [B, F, E]

        # Normalization: compute inverse denominator (softmax-style)
        z = 1 / (torch.einsum('bnd,bd->bn', q, k.sum(dim=1)) + self.epsilon).unsqueeze(-1)  # [B, T, 1]

        # Efficient attention output: dot product in feature space, rescaled
        attn_output = torch.einsum('bnd,bde->bne', q, kv) * z  # [B, T, E]
        return self.final_proj(attn_output)

class MultiQueryAttention(BaseAttention):
    """
    Memory-efficient attention variant that shares key/value projections 
    across all attention heads. Reduces memory usage and computational cost.
    """
    def __init__(self):
        super().__init__()

        if self.attention_config.get("positional_encoding") == "rotary":
            self.rotary_emb = RotaryEmbedding(dim=self.dim_head)
        else:
            self.rotary_emb = None

        # Override projections for multi-query setup
        del self.to_kv  # Remove standard key-value projection
        
        # Create separate key and value projections
        self.to_k = nn.Linear(self.embed_dim, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.embed_dim, self.dim_head, bias=False)
        
        # Initialize new projections
        init_fn = getattr(TensorOps, self.initializer)
        for layer in [self.to_k, self.to_v]:
            weight = init_fn(
                layer.weight.shape, 
                self.embed_dim,
                device=self.device
            )
            layer.weight = Parameter(weight)


    def forward(self, x, context=None, mask=None, attn_bias=None,
                memory_efficient=None, q_bucket_size=None, k_bucket_size=None):
        # Handle optional parameters
        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)
        context = default(context, x)

        # Project inputs - queries are head-specific, keys/values are shared
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Rearrange tensors for attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n d -> b 1 n d')  # Add head dimension for broadcasting
        v = rearrange(v, 'b n d -> b 1 n d')  # Add head dimension for broadcasting
        
        # Select attention mechanism
        attn_fn = memory_efficient_attention if memory_efficient else scaled_dot_product_attention
        out = attn_fn(
            q, k, v, 
            mask=mask, 
            attn_bias=attn_bias, 
            causal=self.causal
        )

        # After computing q, k
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)  # Apply RoPE
        
        # Combine heads and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(BaseAttention):
    """
    Cross-attention layer for encoder-decoder architectures. 
    Processes queries from one modality against keys/values from another.
    """
    def __init__(self):
        super().__init__()
        # Separate projections for encoder context
        self.to_k_enc = nn.Linear(self.embed_dim, self.heads * self.dim_head, bias=False)
        self.to_v_enc = nn.Linear(self.embed_dim, self.heads * self.dim_head, bias=False)
        
        # Initialize encoder projections
        init_fn = getattr(TensorOps, self.initializer)
        for layer in [self.to_k_enc, self.to_v_enc]:
            weight = init_fn(
                layer.weight.shape, 
                self.embed_dim,
                device=self.device
            )
            layer.weight = Parameter(weight)
            #else:
            #    layer.weight = init_fn(
            #        layer.weight.shape, 
            #        device=self.device
            #    )

    def load_from_dict(self, weights_dict, prefix=''):
        # Additional handling for encoder-specific weights
        mapping = {
            f'{prefix}crossattention.key.weight': 'to_k_enc.weight',
            f'{prefix}crossattention.value.weight': 'to_v_enc.weight'
        }
        for hf_key, our_key in mapping.items():
            if hf_key in weights_dict:
                parts = our_key.split('.')
                module = self
                for part in parts[:-1]:
                    module = getattr(module, part)
                param = getattr(module, parts[-1])
                param.data = weights_dict[hf_key].data.clone().to(param.device)
        super().load_from_dict(weights_dict, prefix)

    def forward(self, x, context, mask=None, attn_bias=None,
                memory_efficient=None, q_bucket_size=None, k_bucket_size=None):
        """Requires explicit context tensor from encoder"""
        if context is None:
            raise ValueError("CrossAttention requires context input")
        
        # Handle optional parameters
        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)
        
        # Project inputs - queries from decoder, keys/values from encoder
        q = self.to_q(x)
        k = self.to_k_enc(context)
        v = self.to_v_enc(context)
        
        # Rearrange for multi-head attention
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), 
            (q, k, v)
        )
        
        # Select attention mechanism
        attn_fn = memory_efficient_attention if memory_efficient else scaled_dot_product_attention
        out = attn_fn(
            q, k, v, 
            mask=mask, 
            attn_bias=attn_bias, 
            causal=self.causal
        )
        
        # Combine heads and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

if __name__ == "__main__":
    print("\n=== Running Attention ===\n")
    printer.status("TEST", "Starting Attention tests", "info")
    length=16
    x = torch.randn(4, 128, 512)

    base = BaseAttention()
    cosine = CosineAttention(seq_len=length)
    model = EfficientAttention()
    mqa = MultiQueryAttention()
    cross = CrossAttention()

    print(base)
    print(cosine)
    print(model)
    print(mqa)
    print(cross)

    print("\n* * * * * Phase 2 - Base * * * * *\n")
    context=x.clone() 
    mask=None
    attn_bias=None
    memory_efficient=None
    q_bucket_size=None
    k_bucket_size=None

    forward1 = base.forward(
        x=x,
        mask=mask,
        attn_bias=attn_bias,
        memory_efficient=memory_efficient,
        q_bucket_size=q_bucket_size,
        k_bucket_size=k_bucket_size
    )

    printer.pretty("BASE", forward1, "success")
    printer.pretty("SPLIT", base._split_heads(x=x), "success")
    split = base._split_heads(x=x)  
    combined = base._combine_heads(x=split) 
    printer.pretty("COMBINED", combined, "success")

    print("\n* * * * * Phase 3 - Cosine * * * * *\n")
    forward2 = cosine.forward(
        x=x,
        mask=mask,
        attn_bias=attn_bias,
        memory_efficient=memory_efficient,
        q_bucket_size=q_bucket_size,
        k_bucket_size=k_bucket_size
    )

    printer.pretty("COSINE", forward2, "success")

    print("\n* * * * * Phase 4 - Efficient * * * * *\n")
    x_heads = rearrange(x, 'b t (h d) -> b h t d', h=model.heads)  # [B, H, T, D]

    printer.pretty("create", model._create_random_projection(), "success")
    printer.pretty("positive", model._positive_random_features(x=x_heads), "success")
    printer.pretty("Efficient", model.forward(x=x), "success")

    print("\n* * * * * Phase 5 - Multi Query * * * * *\n")
    forward3 = mqa.forward(
        x=x,
        mask=mask,
        attn_bias=attn_bias,
        memory_efficient=memory_efficient,
        q_bucket_size=q_bucket_size,
        k_bucket_size=k_bucket_size
    )

    printer.pretty("MQA", forward3, "success")

    print("\n* * * * * Phase 6 - Cross * * * * *\n")
    forward4 = cross.forward(
        x=x,
        mask=mask,
        context=context,
        attn_bias=attn_bias,
        memory_efficient=memory_efficient,
        q_bucket_size=q_bucket_size,
        k_bucket_size=k_bucket_size
    )

    printer.pretty("MQA", forward4, "success")

    print("\n=== Successfully Ran Attention ===\n")
