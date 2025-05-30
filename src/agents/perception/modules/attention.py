import torch
import math
import yaml

from pathlib import Path

from src.agents.perception.utils.common import TensorOps, Parameter
from logs.logger import get_logger

logger = get_logger("Attention")

CONFIG_PATH = "src/agents/perception/configs/perception_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class EfficientAttention(torch.nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        cfg = config.get('attention', {})
        self.dropout_rate = cfg.get('dropout_rate', 0.1)
        transformer_cfg = config['transformer']
        self.embed_dim = transformer_cfg['embed_dim']
        self.num_heads = transformer_cfg['num_heads']
        self.head_dim = self.embed_dim // self.num_heads
        self.device = device
        self.dropout_rate = cfg.get('dropout_rate', 0.1)
        self.initializer = cfg.get('initializer', 'xavier_uniform')
        
        # Initialize parameters with proper scaling
        init_fn = getattr(TensorOps, self.initializer) # f"{self.initializer}_init")
        self.q_proj = torch.nn.Parameter(init_fn((self.embed_dim, self.embed_dim), self.embed_dim, device=self.device))
        self.k_proj = torch.nn.Parameter(init_fn((self.embed_dim, self.embed_dim), self.embed_dim, device=self.device))
        self.v_proj = torch.nn.Parameter(init_fn((self.embed_dim, self.embed_dim), self.embed_dim, device=self.device))
        self.out_proj = torch.nn.Parameter(init_fn((self.embed_dim, self.embed_dim), self.embed_dim, device=self.device))
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self._cache = {}
        self.observers = []

        logger.info(f"Attention is successfully initialized with:\n- {torch.nn.Module}")

    def add_observer(self, observer):
        self.observers.append(observer)

    def forward(self, x, context=None, mask=None, causal=False):
        x = x.to(self.device)
        if context is None:
            context = x
        else:
            context = context.to(self.device) # Ensure context is on device

        q = torch.matmul(x, self.q_proj.data)
        k = torch.matmul(context, self.k_proj.data)
        v = torch.matmul(context, self.v_proj.data)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(self.head_dim)

        batch_size, num_heads, seq_len, _ = attn_scores.shape

        if causal:
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        if mask is not None:
            mask = mask.to(self.device).bool()  # Enforce boolean type
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))

        context_vec = torch.einsum('bhij,bhjd->bhid', attn_probs, v)
        context_vec = self._combine_heads(context_vec)
        output = torch.matmul(context_vec, self.out_proj.data)

        # Cache PyTorch tensors
        self._cache = {'q': q, 'k': k, 'v': v, 'attn_probs': attn_probs}
        self._cache['x'] = x
        self._cache['context'] = context
        self._cache['context_vec'] = context_vec # Cache this for backward pass

        if self.observers:
            for observer in self.observers:
                # Pass tensor instead of numpy array
                observer.log_attention(attn_probs.detach().cpu())
        return output

    def backward(self, dout):
        """Manual backpropagation using PyTorch tensors"""
        # Ensure gradient is on the correct device
        dout = dout.to(self.device)

        # Retrieve cached tensors from forward pass
        q = self._cache['q'] # (batch, n_heads, seq_len, head_dim)
        k = self._cache['k']
        v = self._cache['v']
        attn_probs = self._cache['attn_probs'] # (batch, n_heads, seq_len, seq_len)
        context_vec_combined = self._cache['context_vec'] # (batch, seq_len, embed_dim)
        x_orig = self._cache['x'] # (batch, seq_len, embed_dim)
        context_orig = self._cache['context'] # (batch, ctx_len, embed_dim)

        batch_size, num_heads, seq_len, head_dim = q.shape
        embed_dim = self.embed_dim
        ctx_len = k.shape[2] # Context sequence length

        # --- Gradient Calculation ---

        # 1. Gradient w.r.t. output projection (d_out_proj) and context_vec
        d_context_vec = torch.matmul(dout, self.out_proj.data.T) # (batch, seq_len, embed_dim)
        context_vec_reshaped = context_vec_combined.reshape(-1, embed_dim)
        dout_reshaped = dout.reshape(-1, embed_dim)
        d_out_proj = torch.matmul(context_vec_reshaped.T, dout_reshaped) # (embed_dim, embed_dim)

        # Accumulate gradient (assuming Parameter class handles it)
        if self.out_proj.grad is None: self.out_proj.grad = torch.zeros_like(self.out_proj.data)
        self.out_proj.grad += d_out_proj

        # 2. Gradient w.r.t. combined heads context_vec -> split heads context_vec (d_context_vec_split)
        # This reverses the _combine_heads operation
        # d_context_vec shape: (batch, seq_len, embed_dim)
        # Need shape: (batch, n_heads, seq_len, head_dim)
        d_context_vec_split = d_context_vec.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)


        # 3. Gradient w.r.t. attention probabilities (d_attn_probs) and V (d_v_split)
        # d_attn_probs = einsum('bhid,bhjd->bhij', d_context_vec_split, v)
        d_attn_probs = torch.einsum('bhid,bhjd->bhij', d_context_vec_split, v) # (b, h, seq, ctx_len)
        # d_v = einsum('bhij,bhid->bhjd', attn_probs, d_context_vec_split)
        d_v_split = torch.einsum('bhij,bhid->bhjd', attn_probs, d_context_vec_split) # (b, h, ctx_len, head_dim)


        # 4. Gradient w.r.t. attention scores (d_scores) through softmax
        # dL/ds_ij = dL/dp_ik * dp_ik/ds_ij
        # dp_ik/ds_ij = p_ik * (delta_kj - p_ij)
        # Sum over k: dL/ds_ij = sum_k [ dL/dp_ik * p_ik * (delta_kj - p_ij) ]
        # dL/ds_ij = p_ij * (dL/dp_ij - sum_k [ dL/dp_ik * p_ik ])
        # Let S = sum_k [dL/dp_ik * p_ik] (sum over key/ctx dimension)
        # dL/ds_ij = p_ij * (dL/dp_ij - S_i) where S_i is sum for query i
        sum_term = torch.einsum('bhij,bhij->bhi', attn_probs, d_attn_probs) # (b, h, seq)
        d_scores = attn_probs * (d_attn_probs - sum_term.unsqueeze(-1)) # (b, h, seq, ctx_len)
        d_scores /= math.sqrt(self.head_dim) # Apply scaling factor derivative


        # 5. Gradients w.r.t. Q (d_q_split) and K (d_k_split)
        # d_q = einsum('bhij,bhjd->bhid', d_scores, k)
        d_q_split = torch.einsum('bhij,bhjd->bhid', d_scores, k) # (b, h, seq, head_dim)
        # d_k = einsum('bhij,bhid->bhjd', d_scores, q)
        d_k_split = torch.einsum('bhij,bhid->bhjd', d_scores, q) # (b, h, ctx_len, head_dim)


        # 6. Combine head gradients for Q, K, V
        # Need shape (batch, seq_len/ctx_len, embed_dim)
        d_q = self._combine_heads(d_q_split) # (b, seq, embed)
        d_k = self._combine_heads(d_k_split) # (b, ctx_len, embed)
        d_v = self._combine_heads(d_v_split) # (b, ctx_len, embed)


        # 7. Gradients w.r.t. projection weights (d_q_proj, d_k_proj, d_v_proj)
        # d_q_proj = matmul(x_orig.T, d_q) summed over batch
        d_q_proj = torch.matmul(x_orig.transpose(1, 2), d_q).sum(dim=0) # (embed, embed)
        d_k_proj = torch.matmul(context_orig.transpose(1, 2), d_k).sum(dim=0) # (embed, embed)
        d_v_proj = torch.matmul(context_orig.transpose(1, 2), d_v).sum(dim=0) # (embed, embed)

        # Accumulate gradients
        if self.q_proj.grad is None: self.q_proj.grad = torch.zeros_like(self.q_proj.data)
        self.q_proj.grad += d_q_proj
        if self.k_proj.grad is None: self.k_proj.grad = torch.zeros_like(self.k_proj.data)
        self.k_proj.grad += d_k_proj
        if self.v_proj.grad is None: self.v_proj.grad = torch.zeros_like(self.v_proj.data)
        self.v_proj.grad += d_v_proj

        # 8. Gradient w.r.t. input x (d_x) and context (d_context)
        d_x = torch.matmul(d_q, self.q_proj.data.T)
        d_context = torch.matmul(d_k, self.k_proj.data.T) + torch.matmul(d_v, self.v_proj.data.T)

        if self._cache['context'] is self._cache['x']:
             d_x += d_context
             return d_x
        else:
             # Return gradients separately if needed, or combined if appropriate
             # Returning d_context might be needed depending on the model structure
             return d_x # Or return d_x, d_context

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

if __name__ == "__main__":
    print("\n=== Running Common ===\n")

    # Test Parameter class with basic initialization
    data = torch.randn(3, 4)
    param = Parameter(data, name="test_param")

    print(f"Initial: {param}")
    print(f"Data:\n{param.data}")
    print(f"Grad:\n{param.grad}")

    # Simulate a gradient
    param.grad = torch.ones_like(param.data) * 0.5
    print(f"\nAfter Gradient Simulation:\nGrad:\n{param.grad}")

    # Apply a gradient descent step
    lr = 0.1
    param.step(lr)
    print(f"\nAfter Step(lr={lr}):\nData:\n{param.data}")

    # Reset the gradients
    param.zero_grad()
    print(f"\nAfter zero_grad():\nGrad:\n{param.grad}")

    # Move to another device (if available)
    if torch.cuda.is_available():
        param.to("cuda")
        print(f"\nMoved to CUDA:\n{param}")

    print("\n=== Successfully Ran Common ===\n")
