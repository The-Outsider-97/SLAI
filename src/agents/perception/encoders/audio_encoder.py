import math
import numpy as np

from src.agents.perception import TensorOps, Parameter, Transformer


class AudioEncoder:
    def __init__(self, audio_length=16000, patch_size=400, embed_dim=512):
        self.patch_size = patch_size
        self.num_patches = audio_length // patch_size
        self.in_channels = 1  # Mono audio input
        
        # Convolutional projection initialization (1D equivalent)
        self.projection = Parameter(
            TensorOps.he_init((patch_size * self.in_channels, embed_dim), 
            patch_size * self.in_channels)
        )
        self.cls_token = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)
        self.position_embed = Parameter(
            np.random.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        self.transformer = Transformer(num_layers=6, embed_dim=embed_dim)
        
        self._cache = {}

    def extract_patches(self, x):
        """Convert waveform to patched representation"""
        if x.ndim == 2:
            x = x[:, np.newaxis, :]
        # x shape: (batch, channels, length)
        batch, channels, length = x.shape
        assert length == self.num_patches * self.patch_size, \
            "Input length must be divisible by patch size"
        
        # Reshape into non-overlapping patches
        x = x.reshape(batch, channels, self.num_patches, self.patch_size)
        return x.transpose(0, 2, 1, 3).reshape(batch, self.num_patches, -1)

    def load_pretrained(self, weights):
        """Load pretrained weights in audio transformer format"""
        # Handle 1D convolutional projection conversion
        if 'conv_proj' in weights:
            # Convert (embed_dim, in_channels, kernel_size) to linear projection
            self.projection.data = weights['conv_proj'].squeeze().T
        
        # Load standard parameters
        self.cls_token.data = weights.get('cls_token', self.cls_token.data)
        self.position_embed.data = weights.get('pos_embed', self.position_embed.data)
        
        # Load transformer weights
        if any(k.startswith('transformer_') for k in weights):
            self.transformer.load_pretrained({
                k.split('transformer_')[-1]: v 
                for k, v in weights.items() 
                if k.startswith('transformer_')
            })

    def forward(self, x):
        """Process audio input through encoder"""
        # Patch extraction and projection
        x = self.extract_patches(x)
        self._cache['input_shape'] = x.shape
        x = np.matmul(x, self.projection.data)
        
        # Add classification token
        cls_tokens = np.tile(self.cls_token.data, (x.shape[0], 1, 1))
        x = np.concatenate((cls_tokens, x), axis=1)
        
        # Add positional embeddings
        x += self.position_embed.data[:, :x.shape[1]]
        
        # Transformer processing
        x = self.transformer.forward(x)
        self._cache['pre_projection'] = x
        return x

    def backward(self, dout):
        """Backpropagate gradients through audio encoder"""
        # Backward through transformer
        d_x = self.transformer.backward(dout)
        
        # Remove CLS token gradient
        d_x = d_x[:, 1:, :]
        
        # Gradient for projection matrix
        input_patches = self._cache['input_shape'][0]
        d_proj = np.matmul(
            self._cache['input_shape'].transpose(0, 2, 1), 
            d_x.reshape(-1, d_x.shape[-1])
        )
        self.projection.grad += d_proj
        
        # Gradient for input (not used but maintained for completeness)
        d_input = np.matmul(d_x, self.projection.data.T)
        return d_input.reshape(self._cache['input_shape'])

    def parameters(self):
        return [self.projection, self.cls_token, self.position_embed] + self.transformer.parameters()
