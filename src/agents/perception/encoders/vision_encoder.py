import math
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger

logger = get_logger("Vision Encoder")

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

class VisionEncoder:
    def __init__(self, config):
        self.config = config
        vision_cfg = config['vision_encoder']
        transformer_shared_cfg = config['transformer']
        
        # Extract vision-specific parameters
        self.img_size = vision_cfg['img_size']
        self.patch_size = vision_cfg['patch_size']
        self.patch_size = vision_cfg['patch_size'] if vision_cfg['encoder_type'] == 'transformer' else None
        self.in_channels = vision_cfg['in_channels']
        self.encoder_type = vision_cfg['encoder_type']
        self.dynamic_patching = vision_cfg['dynamic_patching']
        self.positional_encoding = vision_cfg['positional_encoding']
        self.dropout_rate = vision_cfg['dropout_rate']
        
        # Shared transformer parameters
        self.embed_dim = transformer_shared_cfg['embed_dim']
        self.num_heads = transformer_shared_cfg['num_heads']
        self.num_layers = transformer_shared_cfg['num_layers']

        self._cache = {}
        if self.encoder_type == "transformer":
            # Initialize transformer components using shared config
            self.projection = Parameter(
                TensorOps.he_init(
                    (self.in_channels * self.patch_size**2, self.embed_dim),
                    self.in_channels * self.patch_size**2
                )
            )
            
            # Positional encoding
            self.base_num_patches = (self.img_size // self.patch_size) ** 2
            if self.positional_encoding == "learned":
                self.position_embed = Parameter(
                    torch.randn(1, self.base_num_patches + 1, self.embed_dim) * 0.02
                )
            
            self.cls_token = Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
            self.transformer = Transformer(config)

        elif self.encoder_type == "cnn":
            self.cnn_config = vision_cfg['cnn']
            self._init_cnn_components()
            
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

    def _init_cnn_components(self):
        """Initialize CNN-specific parameters from config"""
        cnn_cfg = self.cnn_config
        self.input_size = cnn_cfg['input_size']
        self.channels = cnn_cfg['channels']
        self.filters = [tuple(f) for f in cnn_cfg['filters']]

    def _init_sinusoidal_encoding(self, max_len, embed_dim, device):
        if self.positional_encoding == "sinusoidal":
             pe = torch.zeros(1, max_len, embed_dim, device=device)
             position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
             div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
             pe[0, :, 0::2] = torch.sin(position * div_term)
             pe[0, :, 1::2] = torch.cos(position * div_term)
             # Make it a non-learnable buffer or Parameter(..., requires_grad=False)
             return Parameter(pe, requires_grad=False)
        elif self.positional_encoding == "learned":
             return Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        else:
             raise ValueError("Positional encoding must be 'sinusoidal' or 'learned'")

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
            x = F.pad(x, (0, pad_w, 0, pad_h))
            h_patches = (h + pad_h) // self.patch_size
            w_patches = (w + pad_w) // self.patch_size
        
        # Reshape into patches
        x = x.reshape(b, c, h_patches, self.patch_size, w_patches, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c*self.patch_size**2)
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
        old_num_patches = new_pe.shape[1] - 1
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
        self.position_embed.data = torch.concatenate(
            [self.position_embed.data[:, :1, :], new_pe], 
            axis=1
        )

    def _cnn_forward(self, image: torch.Tensor) -> torch.Tensor:
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

        output = torch.zeros((H_out, W_out, C_out))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i*stride - padding
                w_start = j*stride - padding
                receptive_field = x[h_start:h_start+Fh, w_start:w_start+Fw, :]
                output[i,j] = torch.sum(receptive_field * filters, axis=(0,1,2))
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
                    pooled.append(torch.max(pool_region, axis=(0,1)))
        return torch.concatenate(pooled)

    def _relu(self, x):
        """ReLU activation implementation"""
        return torch.maximum(0, x)

    def _max_pool(self, x, pool_size, stride):
        """Max pooling implementation"""
        H, W, C = x.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1

        output = torch.zeros((H_out, W_out, C))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i*stride
                w_start = j*stride
                pool_region = x[h_start:h_start+pool_size, 
                              w_start:w_start+pool_size, :]
                output[i,j] = torch.max(pool_region, axis=(0,1))
        return output

    def forward(self, x, style_id=0):
        if self.encoder_type == "transformer":
            # Original transformer processing
            x = self.extract_patches(x)
            self._cache['input_shape'] = x.shape
            self._cache['x'] = x.clone()

            x = torch.matmul(x, self.projection.data)

            if self.training and self.dropout_rate > 0:
                mask = (torch.rand(*x.shape) > self.dropout_rate).to(torch.float32)
                x *= mask

            cls_tokens = torch.tile(self.cls_token.data, (x.shape[0], 1, 1))
            x = torch.concatenate((cls_tokens, x), axis=1)

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
        # Gradient for projection
        x = self._cache['x']
        d_patch_tokens = d_x[:, 1:, :]
        d_proj = torch.einsum('bni,bno->io', x, d_patch_tokens)
        
        if self.projection.grad is None:
            self.projection.grad = torch.zeros_like(self.projection.data)
        
        self.projection.grad += d_proj.sum(axis=0)
        return torch.matmul(d_patch_tokens, self.projection.data.T)

    def parameters(self):
        return [self.projection, self.cls_token, self.position_embed] + \
               self.transformer.parameters()

    def train(self):
        self.training = True
        self.transformer.training = True

    def eval(self):
        self.training = False
        self.transformer.training = False

if __name__ == "__main__":
    print("\n=== Testing Vision Encoder ===\n")
    config = load_config()

    # Test Transformer-based Vision Encoder
    print("\n--- Testing Transformer Encoder ---")
    vision_transformer = VisionEncoder(config)
    print("Initialized Vision Transformer:", vision_transformer)

    # Create dummy input (batch_size, channels, height, width)
    dummy_input = torch.randn(2, config['vision_encoder']['in_channels'], 
                            config['vision_encoder']['img_size'], 
                            config['vision_encoder']['img_size'])
    print(f"\nInput shape: {dummy_input.shape}")

    # Test forward pass
    vision_transformer.train()
    print("\n[Transformer] Training mode:")
    output = vision_transformer.forward(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")

    # Test backward pass
    dummy_grad = torch.randn_like(output)
    dx = vision_transformer.backward(dummy_grad)
    print(f"\nBackward pass gradient shape: {dx.shape}")

    # Test evaluation mode
    vision_transformer.eval()
    with torch.no_grad():
        print("\n[Transformer] Evaluation mode:")
        eval_output = vision_transformer.forward(dummy_input)
        print(f"Eval output mean: {eval_output.mean().item():.4f}")

    # Print parameter counts
    total_params = sum(p.data.numel() for p in vision_transformer.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test CNN-based Vision Encoder
    print("\n\n--- Testing CNN Encoder ---")
    cnn_config = get_merged_config({
        'vision_encoder': {
            'encoder_type': 'cnn',
            'img_size': config['vision_encoder']['img_size'],
            'in_channels': config['vision_encoder']['in_channels'],
            'cnn': config['vision_encoder']['cnn']
            }
    })
    
    try:
        vision_cnn = VisionEncoder(cnn_config)
        print("Initialized CNN Encoder:", vision_cnn)
        
        # Create CNN-style input (batch, height, width, channels)
        cnn_input = torch.randn(1,  # Works with batch_size=1 due to CNN implementation
                              cnn_config['vision_encoder']['img_size'],
                              cnn_config['vision_encoder']['img_size'],
                              cnn_config['vision_encoder']['in_channels'])
        print(f"\nCNN Input shape: {cnn_input.shape}")

        # Test CNN forward pass
        vision_cnn.train()
        print("\n[CNN] Training mode:")
        cnn_output = vision_cnn.forward(cnn_input)
        print(f"CNN Output shape: {cnn_output.shape}")
        
    except Exception as e:
        print("\n⚠️ CNN Implementation Warning:")
        print(f"Error during CNN forward pass: {str(e)}")
        print("Note: The CNN implementation might require adjustments for batch processing")

    print("\n=== Vision Encoder Tests Completed ===")
