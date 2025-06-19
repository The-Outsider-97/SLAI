import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.base.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Vision Encoder")
printer = PrettyPrinter

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._init_components()
        logger.info(f"VisionEncoder initialized: type={self.encoder_type}, "
                   f"patch_size={self.patch_size}, embed_dim={self.embed_dim}")

    def _init_configs(self):
        """Load and validate all configurations"""
        self.config = load_global_config()
        self.embed_dim = self.config.get('embed_dim')
        self.in_channels = self.config.get('in_channels')

        self.vision_config = get_config_section('vision_encoder')
        self.cnn_config = get_config_section('cnn')
        
        # Core parameters
        self.encoder_type = self.config.get('encoder_type')
        self.device = self.config.get('device')
        self.dropout_rate = self.config.get('dropout_rate')
        self.dynamic_patching = self.config.get('dynamic_patching')
        self.positional_encoding = self.vision_config.get('positional_encoding', 'learned')
        self.max_position_embeddings = self.config.get('max_position_embeddings')
        
        # Vision-specific parameters
        self.img_size = self.vision_config.get('img_size')
        self.patch_size = self.vision_config.get('patch_size')
        
        # Transformer parameters
        self.num_layers = self.config.get('num_layers', self.config['num_layers'])
        self.num_heads = self.config.get('num_heads', self.config['num_heads'])
        self.num_styles = self.config.get('num_styles', self.config['num_styles'])
        self.ff_dim = self.config.get('ff_dim', self.config['ff_dim'])
        
        # Initialize cache
        self._cache = {}

    def _init_components(self):
        """Initialize encoder-specific components"""
        if self.encoder_type == "transformer":
            self._init_transformer_encoder()
        elif self.encoder_type == "cnn":
            self._init_cnn_encoder()
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def _init_transformer_encoder(self):
        """Initialize transformer-based encoder components"""
        # Calculate base patch dimensions
        self.base_num_patches = (self.img_size // self.patch_size) ** 2
        
        # Projection layer
        in_dim =self.in_channels * self.patch_size ** 2 #  self.embed_dim
        projection_tensor_data = TensorOps.xavier_uniform((in_dim, self.embed_dim), device=self.device)
        # logger.debug(f"VisionEncoder._init_transformer_encoder: TensorOps.xavier_uniform created tensor of shape: {projection_tensor_data.shape}") # Optional debug
        
        self.projection = Parameter(
            projection_tensor_data, 
            name="projection"
        )
        
        # Positional encoding
        pe_length = self.base_num_patches + 1 
        if self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding(pe_length, self.embed_dim)
        else:
            self.position_embed = Parameter(
                torch.randn(1, pe_length, self.embed_dim, device=self.device) * 0.02,
                name="position_embed"
            )
        
        # CLS token
        self.cls_token = Parameter(
            torch.randn(1, 1, self.embed_dim, device=self.device) * 0.02,
            name="cls_token"
        )
        
        # Transformer backbone
        self.transformer = Transformer()

    def _init_cnn_encoder(self):
        """Initialize CNN-based encoder components"""
        filters = self.cnn_config.get('filters', [])
        if not filters:
            raise ValueError("CNN configuration must include 'filters' list")
        
        self.conv_layers = nn.ModuleList()
        in_channels = self.in_channels
        
        # Create Conv2d, ReLU, and MaxPool layers
        for i, f in enumerate(filters):
            kernel_h, kernel_w, out_channels = f
            conv = nn.Conv2d(
                in_channels, 
                out_channels,
                kernel_size=(kernel_h, kernel_w),
                stride=4 if i == 0 else 1,
                padding=2
            )
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.ReLU())
            
            # Add MaxPool after first two conv layers
            if i < 2:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            
            in_channels = out_channels
        
        # Spatial Pyramid Pooling levels
        self.spp_levels = [1, 2, 4]

    def _init_sinusoidal_encoding(self, max_len: int, embed_dim: int) -> Parameter:
        """Create sinusoidal positional encoding (non-trainable)"""
        pe = torch.zeros(1, max_len, embed_dim, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return Parameter(pe, requires_grad=False, name="sinusoidal_pe")

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image into patches with dynamic padding"""
        printer.status("VISION", "Converting image into patches with dynamic padding", "info")

        b, c, h, w = x.shape
        
        # Calculate actual patches
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        
        # Apply dynamic padding if needed
        if self.dynamic_patching:
            pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
            pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h))
            h_patches = (h + pad_h) // self.patch_size
            w_patches = (w + pad_w) // self.patch_size
        
        # Extract patches
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        return x.view(b, h_patches * w_patches, -1)

    def forward(self, x: torch.Tensor, style_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process image input through selected encoder"""
        printer.status("VISION", "Processing audio input through selected encoder", "info")

        if self.encoder_type == "transformer":
            return self._forward_transformer(x, style_id)
        elif self.encoder_type == "cnn":
            return self._forward_cnn(x)
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def _forward_transformer(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        printer.status("VISION", "Forward transformer", "info")

        # Extract and project patches
        x_patched = self.extract_patches(x)
        x = torch.matmul(x_patched, self.projection)
        
        # Apply dropout during training
        if self.training and self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        if self.position_embed.shape[1] < x.shape[1]:
            raise ValueError(
                f"Positional embedding size ({self.position_embed.shape[1]}) "
                f"smaller than sequence length ({x.shape[1]})"
            )
        x = x + self.position_embed[:, :x.size(1)]
        
        # Process through transformer
        return self.transformer(x, style_id)

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """CNN-based feature extraction with SPP"""
        printer.status("VISION", "Extracting CNN-based feature", "info")

        # Process through CNN layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Apply Spatial Pyramid Pooling
        return self._spatial_pyramid_pooling(x)

    def _spatial_pyramid_pooling(self, x: torch.Tensor, 
        bin_sizes: List[int] = [1, 2, 4]) -> torch.Tensor:
        """
        Spatial Pyramid Pooling to create fixed-length representation.
        
        Args:
            x: Input feature maps (B, C, H, W)
            bin_sizes: List of grid dimensions to use
        
        Returns:
            Fixed-size vector (B, C * sum(bin_sizes[i]^2))
        """
        printer.status("VISION", "Spatial Pyramid Pooling to create fixed-length representation", "info")

        features = []
        b, c, h, w = x.size()
        
        for bin_size in bin_sizes:
            # Calculate bin dimensions
            bin_h = int(np.ceil(h / bin_size))
            bin_w = int(np.ceil(w / bin_size))
            
            # Create pooling windows
            for i in range(bin_size):
                for j in range(bin_size):
                    # Calculate window boundaries
                    h_start = min(i * bin_h, h)
                    h_end = min((i + 1) * bin_h, h)
                    w_start = min(j * bin_w, w)
                    w_end = min((j + 1) * bin_w, w)
                    
                    # Skip empty windows
                    if h_end <= h_start or w_end <= w_start:
                        continue
                    
                    # Extract window and max pool
                    window = x[:, :, h_start:h_end, w_start:w_end]
                    pooled = F.adaptive_max_pool2d(window, (1, 1))
                    features.append(pooled.view(b, c))
        
        # Concatenate all features
        return torch.cat(features, dim=1)

    def load_pretrained(self, weights: Dict[str, torch.Tensor]):
        """Load pretrained weights for transformer encoder"""
        printer.status("VISION", "Loading pretrained weights", "info")

        if self.encoder_type != "transformer":
            logger.warning("load_pretrained only applicable for transformer encoder")
            return
        
        # Projection weights
        if 'conv_proj' in weights:
            w = weights['conv_proj'].view(weights['conv_proj'].size(0), -1).t()
            self.projection.data.copy_(w)
        
        # Positional embeddings
        if 'pos_embed' in weights:
            loaded_pe = weights['pos_embed']
            if loaded_pe.size(1) == self.position_embed.size(1):
                self.position_embed.data.copy_(loaded_pe)
            else:
                self._interpolate_positional_embeddings(loaded_pe)
        
        # CLS token
        if 'cls_token' in weights:
            self.cls_token.data.copy_(weights['cls_token'])
        
        # Transformer weights
        transformer_weights = {k[len('transformer.'):]: v 
                              for k, v in weights.items() 
                              if k.startswith('transformer.')}
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def _interpolate_positional_embeddings(self, new_pe: torch.Tensor):
        """Handle positional embedding size mismatches"""
        printer.status("VISION", "Handling positional embedding size mismatches", "info")

        old_num_patches = new_pe.size(1) - 1
        new_num_patches = self.base_num_patches
        
        # Extract existing embeddings (exclude CLS token)
        old_pe = new_pe[:, 1:, :]
        old_pe = old_pe.view(1, int(math.sqrt(old_num_patches)),
                             int(math.sqrt(old_num_patches)), 
                             self.embed_dim).permute(0, 3, 1, 2)
        
        # Interpolate
        new_pe = F.interpolate(
            old_pe,
            size=(int(math.sqrt(new_num_patches)), int(math.sqrt(new_num_patches))),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1).view(1, new_num_patches, self.embed_dim)
        
        # Combine with CLS token
        self.position_embed.data = torch.cat(
            [new_pe[:, :1, :], new_pe], 
            dim=1
        )

    def train(self, mode: bool = True):
        printer.status("VISION", "Setting training mode", "info")

        super().train(mode)
        if self.encoder_type == "transformer":
            self.transformer.train(mode)
        return self

    def eval(self):
        printer.status("VISION", "Setting evaluation mode", "info")

        return self.train(False)

if __name__ == "__main__":
    print("\n=== Testing Vision Encoder ===")
    
    # Override config for testing
    config = load_global_config()
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    config['in_channels'] = 3  # Standard RGB input
    
    # Create test input
    img_size = config['vision_encoder']['img_size']
    test_image = torch.randn(4, config['in_channels'], img_size, img_size).to(config['device'])
    
    # Test transformer-based encoder
    config['encoder_type'] = "transformer"
    transformer_encoder = VisionEncoder().to(config['device'])
    print(f"\nEncoder type: {transformer_encoder.encoder_type}")
    output = transformer_encoder(test_image)
    print("Output shape:", output.shape)
    
    # Test CNN-based encoder
    config['encoder_type'] = "cnn"
    cnn_encoder = VisionEncoder().to(config['device'])
    print(f"\nEncoder type: {cnn_encoder.encoder_type}")
    cnn_output = cnn_encoder(test_image)
    print("CNN output shape:", cnn_output.shape)
    
    print("\n=== VisionEncoder tests passed ===")
