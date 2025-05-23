
import numpy as np
import math
from typing import Dict, Any

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer

class VisionEncoder:
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=512,
        in_channels=3,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
        positional_encoding="learned",
        dynamic_patching=True,
        encoder_type: str = "transformer",
        cnn_config: Dict[str, Any] = None
    ):
        self.encoder_type = encoder_type

        if self.encoder_type == "transformer":
            self.patch_size = patch_size
            self.in_channels = in_channels
            self.embed_dim = embed_dim
            self.dropout_rate = dropout_rate
            self.dynamic_patching = dynamic_patching
            self.training = True

            # Calculate initial number of patches
            self.base_num_patches = (img_size // patch_size) ** 2
            
            # Projection layer (convolutional equivalent)
            self.projection = Parameter(
                TensorOps.he_init((in_channels * patch_size**2, embed_dim), 
                                in_channels * patch_size**2)
            )
            
            # Positional encoding system
            self.positional_encoding = positional_encoding
            if positional_encoding == "learned":
                self.position_embed = Parameter(
                    np.random.randn(1, self.base_num_patches + 1, embed_dim) * 0.02
                )
            elif positional_encoding == "sinusoidal":
                self.position_embed = self._init_sinusoidal_encoding()
            
            self.cls_token = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)
            self.transformer = Transformer(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads
            )
            self._cache = {}

        elif self.encoder_type == "cnn":
            # CNN-based feature extractor implementation
            cnn_config = cnn_config or {}
            self.input_size = cnn_config.get('input_size', (224, 224))
            self.channels = cnn_config.get('channels', 3)
            self.filters = cnn_config.get('filters', [
                (11, 11, 96),    # Conv1: 11x11, 96 filters
                (5, 5, 256),     # Conv2: 5x5, 256 filters
                (3, 3, 384)      # Conv3: 3x3, 384 filters
            ])
            
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

    def _init_sinusoidal_encoding(self):
        """Sinusoidal positional encoding for vision patches"""
        num_patches = self.base_num_patches
        position = np.arange(num_patches + 1)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * 
                   -(math.log(10000.0) / self.embed_dim))
        pe = np.zeros((num_patches + 1, self.embed_dim))
        pe[1:, 0::2] = np.sin(position[1:] * div_term)  # Skip CLS token
        pe[1:, 1::2] = np.cos(position[1:] * div_term)
        return Parameter(pe[np.newaxis, :, :])

    def extract_patches(self, x):
        """Handle dynamic input sizes with optional padding"""
        b, c, h, w = x.shape
        
        # Calculate actual patches
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        
        if self.dynamic_patching:
            # Pad if not divisible
            pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
            pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
            x = np.pad(x, ((0,0), (0,0), 
                      (0,pad_h), (0,pad_w)))
            h_patches = (h + pad_h) // self.patch_size
            w_patches = (w + pad_w) // self.patch_size
        
        # Reshape into patches
        x = x.reshape(b, c, h_patches, self.patch_size, w_patches, self.patch_size)
        x = x.transpose(0, 2, 4, 1, 3, 5).reshape(b, -1, c*self.patch_size**2)
        return x

    def load_pretrained(self, weights):
        """Handle multiple weight formats including HF-style"""
        # Projection weights
        if 'conv_proj' in weights:
            # Convert (embed_dim, in_ch, ph, pw) → (in_ch*ph*pw, embed_dim)
            w = weights['conv_proj'].reshape(
                weights['conv_proj'].shape[0], -1).T
            self.projection.data = w
        
        # Positional embeddings
        if 'pos_embed' in weights:
            if weights['pos_embed'].shape[1] == self.base_num_patches + 1:
                self.position_embed.data = weights['pos_embed']
            else:
                self._interpolate_positional_embeddings(weights['pos_embed'])
        
        # CLS token
        self.cls_token.data = weights.get('cls_token', self.cls_token.data)
        
        # Transformer weights
        transformer_weights = {
            k.split('transformer_')[-1]: v 
            for k, v in weights.items() 
            if k.startswith('transformer_')
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def _interpolate_positional_embeddings(self, new_pe):
        """Handle positional embedding size mismatches"""
        old_num_patches = new_pe.shape[1] - 1  # Exclude CLS token
        new_num_patches = self.base_num_patches
        
        # Extract patch embeddings (exclude CLS token)
        old_pe = new_pe[:, 1:, :]
        old_pe = old_pe.reshape(1, int(math.sqrt(old_num_patches)),
                              int(math.sqrt(old_num_patches)), self.embed_dim)
        
        # Interpolate using bilinear
        new_pe = TensorOps.interpolate(
            old_pe, 
            size=(int(math.sqrt(new_num_patches)), int(math.sqrt(new_num_patches))),
            mode='bilinear'
        )
        new_pe = new_pe.reshape(1, new_num_patches, self.embed_dim)
        self.position_embed.data = np.concatenate(
            [self.position_embed.data[:, :1, :], new_pe], 
            axis=1
        )

    def _cnn_forward(self, image: np.ndarray) -> np.ndarray:
        """CNN-based feature extraction pipeline"""
        features = image
        channel_axis = -1
        
        # Conv1 Layer
        features = self._conv2d(features, self.filters[0], stride=4, padding=2)
        features = self._relu(features)
        features = self._max_pool(features, 3, stride=2)
        
        # Conv2 Layer
        features = self._conv2d(features, self.filters[1], padding=2)
        features = self._relu(features)
        features = self._max_pool(features, 3, stride=2)
        
        # Conv3 Layer
        features = self._conv2d(features, self.filters[2])
        features = self._relu(features)
        
        # Spatial Pyramid Pooling
        features = self._spatial_pyramid_pooling(features)
        
        return features

    def _conv2d(self, x, filters, stride=1, padding=0):
        """2D Convolution implementation"""
        H, W, C_in = x.shape
        Fh, Fw, C_out = filters.shape[:3]
        
        H_out = (H - Fh + 2*padding) // stride + 1
        W_out = (W - Fw + 2*padding) // stride + 1
        
        output = np.zeros((H_out, W_out, C_out))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i*stride - padding
                w_start = j*stride - padding
                receptive_field = x[h_start:h_start+Fh, w_start:w_start+Fw, :]
                output[i,j] = np.sum(receptive_field * filters, axis=(0,1,2))
        return output

    def _spatial_pyramid_pooling(self, features):
        """Spatial pyramid pooling implementation"""
        levels = [1, 2, 4]
        pooled = []
        for level in levels:
            H, W = features.shape[:2]
            bin_h = H // level
            bin_w = W // level
            for i in range(level):
                for j in range(level):
                    pool_region = features[i*bin_h:(i+1)*bin_h, 
                                        j*bin_w:(j+1)*bin_w]
                    pooled.append(np.max(pool_region, axis=(0,1)))
        return np.concatenate(pooled)

    def _relu(self, x):
        """ReLU activation implementation"""
        return np.maximum(0, x)

    def _max_pool(self, x, pool_size, stride):
        """Max pooling implementation"""
        H, W, C = x.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        
        output = np.zeros((H_out, W_out, C))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i*stride
                w_start = j*stride
                pool_region = x[h_start:h_start+pool_size, 
                              w_start:w_start+pool_size, :]
                output[i,j] = np.max(pool_region, axis=(0,1))
        return output

    def forward(self, x, style_id=0):
        if self.encoder_type == "transformer":
            # Original transformer processing
            x = self.extract_patches(x)
            self._cache['input_shape'] = x.shape
            
            x = np.matmul(x, self.projection.data)
            
            if self.training and self.dropout_rate > 0:
                mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
                x *= mask
            
            cls_tokens = np.tile(self.cls_token.data, (x.shape[0], 1, 1))
            x = np.concatenate((cls_tokens, x), axis=1)
            
            if self.positional_encoding == "sinusoidal":
                x += self.position_embed.data[:, :x.shape[1]]
            else:
                x += self.position_embed.data[:, :x.shape[1]]
            
            x = self.transformer.forward(x, style_id)
            return x
            
        elif self.encoder_type == "cnn":
            # Process with CNN pipeline
            return self._cnn_forward(x)
            
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def backward(self, dout):
        """Backprop through encoder"""
        d_x = self.transformer.backward(dout)
        d_patch_tokens = d_x[:, 1:, :]  # Skip CLS token
        
        # Gradient for projection
        d_proj = np.matmul(
            self._cache['input_shape'].transpose(0, 2, 1),
            d_patch_tokens.reshape(-1, self.embed_dim))
        self.projection.grad += d_proj.sum(axis=0)
        
        return np.matmul(d_patch_tokens, self.projection.data.T)

    def parameters(self):
        return [self.projection, self.cls_token, self.position_embed] + \
               self.transformer.parameters()

    def train(self):
        self.training = True
        self.transformer.training = True

    def eval(self):
        self.training = False
        self.transformer.training = False
