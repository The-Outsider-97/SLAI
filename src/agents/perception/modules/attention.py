import numpy as np
import math

from src.agents.perception.utils.common import TensorOps, Parameter

class EfficientAttention:
    def __init__(self, embed_dim=512, num_heads=8):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Initialize parameters with proper scaling
        self.q_proj = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        self.k_proj = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        self.v_proj = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        self.out_proj = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        
        self._cache = {}

    def forward(self, x, context=None):
        if context is None:
            context = x
            
        q = np.matmul(x, self.q_proj.data)
        k = np.matmul(context, self.k_proj.data)
        v = np.matmul(context, self.v_proj.data)
        
        # Efficient batch-aware computation using einsum
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        attn_scores = np.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(self.head_dim)
        attn_probs = self._softmax(attn_scores)
        
        # Memory-efficient attention computation
        context_vec = np.einsum('bhij,bhjd->bhid', attn_probs, v)
        context_vec = self._combine_heads(context_vec)
        output = np.matmul(context_vec, self.out_proj.data)
        
        # Store intermediates for backward pass
        self._cache = {'q': q, 'k': k, 'v': v, 'attn_probs': attn_probs}
        self._cache['x'] = x
        self._cache['context'] = context 

        return output

    def backward(self, dout):
        # Retrieve cached tensors from forward pass
        q = self._cache['q']
        k = self._cache['k']
        v = self._cache['v']
        attn_probs = self._cache['attn_probs']
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Gradient for output projection
        d_context_vec = np.matmul(dout, self.out_proj.data.T)
        d_out_proj = np.matmul(
            self._cache['context_vec'].transpose(0,2,1,3).reshape(-1, self.embed_dim).T,
            dout.reshape(-1, self.embed_dim)
        )
        self.out_proj.grad += d_out_proj

        # Gradient through attention combination
        d_attn_probs = np.einsum('bhid,bhjd->bhij', d_context_vec, v)
        d_v = np.einsum('bhij,bhid->bhjd', attn_probs, d_context_vec)

        # Gradient through softmax
        d_scores = attn_probs * (d_attn_probs - np.einsum('bhij,bhij->bhi', attn_probs, d_attn_probs)[..., None])
        d_scores /= math.sqrt(self.head_dim)

        # Gradients for Q and K
        d_q = np.einsum('bhij,bhjd->bhid', d_scores, k)
        d_k = np.einsum('bhij,bhid->bhjd', d_scores, q)

        # Combine heads and calculate parameter gradients
        d_q = self._combine_heads(d_q.transpose(0,2,1,3))
        d_k = self._combine_heads(d_k.transpose(0,2,1,3))
        d_v = self._combine_heads(d_v.transpose(0,2,1,3))

        self.q_proj.grad += np.matmul(self._cache['x'].transpose(0,2,1), d_q).sum(axis=0)
        self.k_proj.grad += np.matmul(self._cache['context'].transpose(0,2,1), d_k).sum(axis=0)
        self.v_proj.grad += np.matmul(self._cache['context'].transpose(0,2,1), d_v).sum(axis=0)

        return np.matmul(d_q, self.q_proj.data.T) + \
               np.matmul(d_k, self.k_proj.data.T) + \
               np.matmul(d_v, self.v_proj.data.T)

    def _split_heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(0,2,1,3)

    def _combine_heads(self, x):
        return x.transpose(0,2,1,3).reshape(x.shape[0], x.shape[2], -1)

    def _softmax(self, x):
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
