import math
import numpy as np
from src.agents.perception.utils.common import TensorOps, Parameter  
from src.agents.perception.modules.attention import EfficientAttention  
from src.agents.perception.modules.feedforward import FeedForward  

class Transformer:
    def __init__(self, num_layers=6, embed_dim=512, num_heads=8, ff_dim=2048):
        self.layers = [
            {
                'attention': EfficientAttention(embed_dim, num_heads),
                'ff': FeedForward(embed_dim, ff_dim),
                'norm1': Parameter(np.ones(embed_dim)),
                'norm2': Parameter(np.ones(embed_dim)),
            }
            for _ in range(num_layers) 
        ]
        self.positional_encoding = self._init_positional_encoding(embed_dim)

    def parameters(self):
        params = [self.positional_encoding]
        for layer in self.layers:
            params.extend([
                layer['attention'].q_proj,
                layer['attention'].k_proj,
                layer['attention'].v_proj,
                layer['attention'].out_proj,
                layer['ff'].w1,
                layer['ff'].b1,
                layer['ff'].w2,
                layer['ff'].b2,
                layer['norm1'],
                layer['norm2']
            ])
        return params

    def _init_positional_encoding(self, d_model, max_len=5000):
        """Sinusoidal positional encoding (Vaswani et al. 2017)"""
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return Parameter(pe.T)  # (d_model, max_len)

    def forward(self, x):
        """Implements transformer encoder forward pass"""
        seq_len = x.shape[1]
        x += self.positional_encoding.data[:, :seq_len]
        
        for layer in self.layers:
            # Self-attention sublayer
            residual = x
            x = TensorOps.layer_norm(x + layer['norm1'].data)
            x = layer['attention'].forward(x)
            x = residual + x
            
            # Feed-forward sublayer
            residual = x
            x = TensorOps.layer_norm(x + layer['norm2'].data)
            x = layer['ff'].forward(x)
            x = residual + x
            
        return TensorOps.layer_norm(x)

    def backward(self, dout):
        """Backpropagation through transformer layers"""
        grads = []
        for layer in reversed(self.layers):
            # Backward through FFN
            d_ff = layer['ff'].backward(dout)
            d_norm2 = d_ff * (1 + layer['norm2'].grad)
            dout += d_norm2
            
            # Backward through attention
            d_attn = layer['attention'].backward(dout)
            d_norm1 = d_attn * (1 + layer['norm1'].grad)
            dout += d_norm1
            
        # Positional encoding gradients (non-trainable in original paper)
        return dout

    def load_pretrained(self, weights):
        """Load weights in Hugging Face-style format"""
        for i, layer in enumerate(self.layers):
            prefix = f'encoder.layer.{i}.'
            layer['attention'].q_proj.data = weights[f'{prefix}attention.self.query.weight']
            layer['attention'].k_proj.data = weights[f'{prefix}attention.self.key.weight']
            layer['attention'].v_proj.data = weights[f'{prefix}attention.self.value.weight']
            layer['attention'].out_proj.data = weights[f'{prefix}attention.output.dense.weight']
            layer['norm1'].data = weights[f'{prefix}attention.output.LayerNorm.weight']
            layer['ff'].w1.data = weights[f'{prefix}intermediate.dense.weight']
            layer['ff'].w2.data = weights[f'{prefix}output.dense.weight']
            layer['norm2'].data = weights[f'{prefix}output.LayerNorm.weight']
