import math
import numpy as np

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer  

class AudioEncoder:
    def __init__(self,
                 audio_length=16000,
                 patch_size=400,
                 embed_dim=512,
                 in_channels=1,
                 num_layers=6,
                 dropout_rate=0.1,
                 positional_encoding="learned"):
        self.patch_size = patch_size
        self.num_patches = audio_length // patch_size
        #self.in_channels = 1  # Mono audio input
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.training = True
        
        # Convolutional projection initialization (1D equivalent)
        self.projection = Parameter(
            TensorOps.he_init((patch_size * in_channels, embed_dim), patch_size * in_channels))

        # Positional embeddings
        self.positional_encoding = positional_encoding
        if self.positional_encoding == "learned":
            self.position_embed = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)
        elif self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding(max_len=5000)

        self.cls_token = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)
        self.transformer = Transformer(num_layers=num_layers, embed_dim=embed_dim)        
        self._cache = {}

    def _init_sinusoidal_encoding(self, max_len=5000):
        pe = np.zeros((max_len, self.embed_dim))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return Parameter(pe[np.newaxis, :, :])

    def extract_patches(self, x):
        """Convert waveform to patched representation with padding"""
        if x.ndim == 2:
            x = x[:, np.newaxis, :]  # Add channel dim: (B, C, L)
        batch, channels, length = x.shape
 
        # Pad if necessary
        remainder = length % self.patch_size
        if remainder != 0:
            pad_size = self.patch_size - remainder
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_size)))

        # Reshape into non-overlapping patches
        num_patches = x.shape[2] // self.patch_size
        x = x.reshape(batch, channels, num_patches, self.patch_size)
        return x.transpose(0, 2, 1, 3).reshape(batch, num_patches, -1)

    def load_pretrained(self, weights):
        """Handle 1D conv, transformer, and positional weights"""
        if 'conv_proj' in weights:
            # Convert (embed_dim, in_channels, kernel_size) → (in_channels*kernel_size, embed_dim)
            w = weights['conv_proj'].reshape(weights['conv_proj'].shape[0], -1).T
            self.projection.data = w
        
        self.cls_token.data = weights.get('cls_token', self.cls_token.data)
        self.position_embed.data = weights.get('pos_embed', self.position_embed.data)
        
        # Load transformer weights
        transformer_weights = {
            k.split('transformer_')[-1]: v 
            for k, v in weights.items() 
            if k.startswith('transformer_')
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def forward(self, x, style_id=0):
        """Process audio with dropout and dynamic patching"""
        x = self.extract_patches(x)
        self._cache['input_shape'] = x.shape
        
        # Project patches
        x = np.matmul(x, self.projection.data)
        
        # Apply dropout
        if self.training and self.dropout_rate > 0:
            mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
            x *= mask
        
        # Add CLS token
        cls_tokens = np.tile(self.cls_token.data, (x.shape[0], 1, 1))
        x = np.concatenate((cls_tokens, x), axis=1)
        
        # Positional embeddings
        if self.positional_encoding == "sinusoidal":
            seq_len = x.shape[1]
            x += self.position_embed.data[:, :seq_len, :]
        else:
            x += self.position_embed.data[:, :x.shape[1]]
        
        # Transformer processing
        x = self.transformer.forward(x, style_id)
        self._cache['pre_projection'] = x
        return x

    def backward(self, dout):
        """Backprop through encoder"""
        d_x = self.transformer.backward(dout)
        d_x = d_x[:, 1:, :]  # Remove CLS token
        
        # Gradient for projection
        d_proj = np.matmul(
            self._cache['input_shape'].transpose(0, 2, 1), 
            d_x.reshape(-1, self.embed_dim))
        self.projection.grad += d_proj.sum(axis=0)
        
        return np.matmul(d_x, self.projection.data.T)

    def parameters(self):
        return [self.projection, self.cls_token, self.position_embed] + self.transformer.parameters()

    def train(self):
        self.training = True
        self.transformer.training = True

    def eval(self):
        self.training = False
        self.transformer.training = False
