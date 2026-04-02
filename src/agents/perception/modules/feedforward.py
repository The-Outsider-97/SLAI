import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import TensorOps, Parameter
from ...base.utils.activation_engine import (
    get_activation,
    he_init, lecun_normal, xavier_uniform, xavier_normal
)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("FeedForward")
printer = PrettyPrinter


class FeedForward(nn.Module):
    """Enhanced position‑wise feed‑forward network with configurable components."""
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.embed_dim = self.config.get('embed_dim')
        self.initializer = self.config.get('initializer', 'xavier_uniform')
        self.device = self.config.get('device', 'cpu')
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        self.ff_dim = self.config.get('ff_dim', self.embed_dim * 4)
        self.activation_name = self.config.get('activation', 'relu')
        self.norm_type = self.config.get('norm_type', 'layernorm')

        self.ff_config = get_config_section('feedforward')
        self.use_bias = self.ff_config.get('use_bias', True)
        self.use_residual = self.ff_config.get('use_residual', True)
        self.fusion_type = self.ff_config.get('fusion_type', None)
        self.context_dim = self.ff_config.get('context_dim', self.embed_dim)

        # Activation from engine
        self.activation = get_activation(self.activation_name)

        # Normalization layer
        self._init_normalization()

        # Fusion parameters (if needed)
        self._init_fusion()

        # Linear layers
        self._init_parameters()

        logger.info(f"FeedForward initialized: embed_dim={self.embed_dim}, ff_dim={self.ff_dim}, "
                    f"activation={self.activation_name}, use_bias={self.use_bias}, "
                    f"residual={self.use_residual}, norm={self.norm_type}, fusion={self.fusion_type}")

    def _init_normalization(self):
        """Initialize the normalization layer based on config."""
        if self.norm_type == 'layernorm':
            self.norm = nn.LayerNorm(self.embed_dim, eps=1e-5)
        elif self.norm_type == 'instancenorm':
            # InstanceNorm expects (batch, channels, seq) – we'll adapt in forward
            self.norm = nn.InstanceNorm1d(self.embed_dim, affine=True)
        elif self.norm_type is None or self.norm_type == 'none':
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}")

    def _init_fusion(self):
        """Initialize parameters for multi‑modal fusion."""
        self.film_gamma = None
        self.film_beta = None
        self.context_proj = None

        if self.fusion_type == 'concat':
            # First linear layer input dimension will be increased
            self.fused_embed_dim = self.embed_dim + self.context_dim
        elif self.fusion_type == 'film':
            # Feature‑wise Linear Modulation: learnable scales and shifts
            self.film_gamma = Parameter(torch.ones(1, self.embed_dim, device=self.device))
            self.film_beta = Parameter(torch.zeros(1, self.embed_dim, device=self.device))
            # Project context to match embedding dimension if needed
            if self.context_dim != self.embed_dim:
                self.context_proj = nn.Linear(self.context_dim, self.embed_dim, bias=False)
                self._init_layer(self.context_proj)
        elif self.fusion_type == 'add':
            # No extra parameters
            pass
        elif self.fusion_type is not None:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

    def _init_layer(self, layer):
        """Helper to initialize a linear layer with the configured initializer."""
        init_map = {
            'he': he_init,
            'lecun': lecun_normal,
            'xavier_uniform': xavier_uniform,
            'xavier_normal': xavier_normal,
        }
        init_fn = init_map.get(self.initializer, xavier_uniform)
        weight_shape = layer.weight.shape
        with torch.no_grad():
            layer.weight.data = init_fn(weight_shape, device=self.device)

    def _init_parameters(self):
        """Initialize weights and biases for linear layers."""
        # Input dimension for first layer (may be increased by concat fusion)
        input_dim = self.embed_dim
        if self.fusion_type == 'concat':
            input_dim = self.fused_embed_dim

        init_map = {
            'he': he_init,
            'lecun': lecun_normal,
            'xavier_uniform': xavier_uniform,
            'xavier_normal': xavier_normal,
        }
        init_fn = init_map.get(self.initializer, xavier_uniform)

        # First linear transformation
        w1_shape = (input_dim, self.ff_dim)
        self.w1 = Parameter(init_fn(w1_shape, device=self.device))
        self.b1 = Parameter(torch.zeros(self.ff_dim, device=self.device)) if self.use_bias else None

        # Second linear transformation
        w2_shape = (self.ff_dim, self.embed_dim)
        self.w2 = Parameter(init_fn(w2_shape, device=self.device))
        self.b2 = Parameter(torch.zeros(self.embed_dim, device=self.device)) if self.use_bias else None

    def _apply_fusion(self, x, context):
        """Apply multi‑modal fusion (modifies x in place or returns new tensor)."""
        if self.fusion_type == 'add':
            # Simple addition (context must have same shape as x)
            if context.shape != x.shape:
                # Project context to match if needed
                if not hasattr(self, '_add_proj'):
                    self._add_proj = nn.Linear(context.shape[-1], self.embed_dim, bias=False).to(x.device)
                    self._init_layer(self._add_proj)
                context = self._add_proj(context)
            return x + context

        elif self.fusion_type == 'concat':
            # Concatenate along feature dimension
            # Ensure context has same batch and sequence dimensions as x
            if context.dim() == 2:  # (batch, features) -> add sequence dim
                context = context.unsqueeze(1).expand(-1, x.size(1), -1)
            elif context.dim() == 3 and context.size(1) == 1:
                context = context.expand(-1, x.size(1), -1)
            # If context dims still don't match, project
            if context.size(-1) != self.context_dim:
                context = context[..., :self.context_dim]  # truncate or pad? For simplicity, truncate
            return torch.cat([x, context], dim=-1)

        elif self.fusion_type == 'film':
            # Feature‑wise Linear Modulation
            # Context may be global (batch, features) or sequence (batch, seq, features)
            if context.dim() == 2:
                context = context.unsqueeze(1)  # (batch, 1, features)
            # Project context to embedding dimension if needed
            if self.context_proj is not None:
                context = self.context_proj(context)
            gamma = self.film_gamma * context
            beta = self.film_beta * context
            return x * gamma + beta

        else:
            return x

    def forward(self, x, context=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            context: Optional context tensor for multi‑modal fusion.
        """
        residual = x  # Save for residual connection

        # Apply normalization
        if self.norm is not None:
            if isinstance(self.norm, nn.InstanceNorm1d):
                # InstanceNorm expects (batch, channels, seq)
                x = x.permute(0, 2, 1)
                x = self.norm(x)
                x = x.permute(0, 2, 1)
            else:
                x = self.norm(x)

        # Apply multi‑modal fusion
        if context is not None:
            x = self._apply_fusion(x, context)

        # First linear projection
        h = torch.matmul(x, self.w1)
        if self.use_bias and self.b1 is not None:
            h += self.b1

        # Activation
        h_act = self.activation(h)

        # Dropout
        if self.training and self.dropout_rate > 0:
            h_act = F.dropout(h_act, p=self.dropout_rate, training=self.training)

        # Second linear projection
        out = torch.matmul(h_act, self.w2)
        if self.use_bias and self.b2 is not None:
            out += self.b2

        # Residual connection
        if self.use_residual:
            out = residual + out

        return out

    def load_pretrained(self, weights, prefix=''):
        """Load pretrained weights from a dictionary (e.g., from HuggingFace)."""
        # Map keys
        w1_key = f'{prefix}intermediate.dense.weight'
        b1_key = f'{prefix}intermediate.dense.bias'
        w2_key = f'{prefix}output.dense.weight'
        b2_key = f'{prefix}output.dense.bias'

        if w1_key in weights:
            self.w1.data = weights[w1_key].to(self.device)
        if b1_key in weights and self.use_bias and self.b1 is not None:
            self.b1.data = weights[b1_key].to(self.device)

        if w2_key in weights:
            self.w2.data = weights[w2_key].to(self.device)
        if b2_key in weights and self.use_bias and self.b2 is not None:
            self.b2.data = weights[b2_key].to(self.device)


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running FeedForward ===\n")
    model = FeedForward()
    print("Initialized FeedForward module:")
    print(model)

    x = torch.randn(4, 128, 512)
    print(f"\nInput shape: {x.shape}")

    model.train()
    print("\nMode: Training")
    y_train = model(x)
    print("Forward output (train):", y_train.shape)

    model.eval()
    print("\nMode: Evaluation")
    y_eval = model(x)
    print("Forward output (eval):", y_eval.shape)

    print("\n=== Successfully Ran FeedForward ===\n")