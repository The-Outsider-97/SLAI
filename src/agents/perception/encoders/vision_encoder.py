import numpy as np
import math

from collections import OrderedDict

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer


class VisionEncoder:
    def __init__(self, img_size=224, patch_size=16, embed_dim=512):
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Proper initialization for convolutional projection
        self.projection = Parameter(
            TensorOps.he_init((3 * patch_size**2, embed_dim), 3*patch_size**2)
        )
        self.cls_token = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)
        self.position_embed = Parameter(np.random.randn(1, self.num_patches+1, embed_dim) * 0.02)
        
        self.transformer = Transformer(num_layers=6, embed_dim=embed_dim)

    def extract_patches(self, x):
        # Batch-aware patch extraction using reshape
        b, c, h, w = x.shape
        x = x.reshape(b, c, h//self.patch_size, self.patch_size, w//self.patch_size, self.patch_size)
        x = x.transpose(0,2,4,1,3,5).reshape(b, -1, c*self.patch_size**2)
        return x

    def load_pretrained(self, weights):
        """
        Load pretrained weights in Vision Transformer format.
        weights: Dict containing keys:
            - conv_proj: (patch_h, patch_w, in_ch, embed_dim)
            - cls_token: (1, 1, embed_dim)
            - pos_embed: (1, num_patches + 1, embed_dim)
            - transformer_*: transformer weights
        """
        if 'conv_proj' in weights:
            # (patch_h, patch_w, in_ch, embed_dim) → (in_ch, patch_h, patch_w, embed_dim)
            w = weights['conv_proj'].transpose(2, 0, 1, 3)
            w = w.reshape(-1, w.shape[-1])  # flatten to match linear shape
            self.projection.data = w
    
        self.cls_token.data = weights.get('cls_token', self.cls_token.data)
        self.position_embed.data = weights.get('pos_embed', self.position_embed.data)
    
        if any(k.startswith('transformer_') for k in weights):
            self.transformer.load_pretrained({
                k.split('transformer_')[-1]: v
                for k, v in weights.items()
                if k.startswith('transformer_')
            })

    def forward(self, x):
        x = self.extract_patches(x)
        self._patches = x.copy()
        x = np.matmul(x, self.projection.data)
        cls_tokens = np.tile(self.cls_token.data, (x.shape[0], 1, 1))
        x = np.concatenate((cls_tokens, x), axis=1)
        x += self.position_embed.data
        x = self.transformer.forward(x)
        return x

    def backward(self, dout):
        d_x = self.transformer.backward(dout)
    
        # Remove CLS token gradient before backpropagating to patch projection
        d_patch_tokens = d_x[:, 1:, :]  # skip CLS token
    
        # Gradient w.r.t. projection
        d_proj = np.matmul(self._patches.transpose(0, 2, 1), d_patch_tokens)
        self.projection.grad += d_proj.sum(axis=0)
    
        # Gradient for input patches (not used directly)
        return np.matmul(d_patch_tokens, self.projection.data.T)

    def backward(self, dout):
        d_x = self.transformer.backward(dout)
        d_proj = np.matmul(self._patches.transpose(0, 2, 1), d_x[:, 1:, :])  # skip cls token
        self.projection.grad += d_proj.sum(axis=0)

    def parameters(self):
        return [self.projection, self.cls_token, self.position_embed] + self.transformer.parameters()
