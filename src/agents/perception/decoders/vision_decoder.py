import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Vision Decoder")
printer = PrettyPrinter

class VisionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._init_components()
        logger.info(f"VisionDecoder initialized: type={self.decoder_type}, "
                   f"patch_size={self.patch_size}, embed_dim={self.embed_dim}")

    def _init_configs(self):
        """Load and validate all configurations"""
        self.config = load_global_config()
        self.vision_config = get_config_section('vision_encoder')
        self.cnn_config = get_config_section('cnn')
        
        # Core parameters
        self.embed_dim = self.config['embed_dim']
        self.in_channels = self.config['in_channels']
        self.decoder_type = self.config['encoder_type']
        self.device = self.config['device']
        self.dropout_rate = self.config['dropout_rate']
        self.positional_encoding = self.vision_config.get('positional_encoding')
        self.max_position_embeddings = self.config['max_position_embeddings']
        
        # Vision-specific parameters
        self.img_size = self.vision_config['img_size']
        self.patch_size = self.vision_config['patch_size']
        self.output_activation = self.vision_config.get('output_activation')
        
        # Transformer parameters
        self.num_layers = self.config.get('num_layers', self.config['num_layers'])
        self.num_heads = self.config.get('num_heads', self.config['num_heads'])
        self.num_styles = self.config.get('num_styles', self.config['num_styles'])
        self.ff_dim = self.config.get('ff_dim', self.config['ff_dim'])
        
        # Calculate base dimensions
        self.base_num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_dim = self.in_channels * self.patch_size ** 2

    def _init_components(self):
        """Initialize decoder-specific components"""
        if self.decoder_type == "transformer":
            self._init_transformer_decoder()
        elif self.decoder_type == "cnn":
            self._init_cnn_decoder()
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")

    def _init_transformer_decoder(self):
        """Initialize transformer-based decoder components"""
        # Positional encoding
        if self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding(self.base_num_patches + 1, self.embed_dim)
        else:  # learned
            self.position_embed = Parameter(
                torch.randn(1, self.base_num_patches + 1, self.embed_dim, device=self.device) * 0.02,
                name="position_embed"
            )
        
        # Transformer backbone
        self.transformer = Transformer()
        
        # Projection to patch space
        self.projection = nn.Linear(self.embed_dim, self.patch_dim)
        
        # Output activation
        self.output_act = self._get_output_activation()

    def _init_cnn_decoder(self):
        """Initialize CNN-based decoder components"""
        filters = self.cnn_config.get('filters', [])
        if not filters:
            raise ValueError("CNN configuration must include 'filters' list")
        
        # Reverse filter order for decoder
        reversed_filters = list(reversed(filters))
        
        # Calculate SPP dimension
        spp_dim = sum(ch * (bin_size ** 2) for bin_size in [1, 2, 4] 
                     for _, _, ch in filters[:3])
        
        # Initial projection
        self.initial_proj = nn.Linear(self.embed_dim, spp_dim)
        
        # Transposed convolution layers
        self.deconv_layers = nn.ModuleList()
        in_channels = reversed_filters[0][2]  # Last filter's output channels
        
        # Add inverse SPP (if needed)
        self.inv_spp = self._init_inverse_spp()
        
        # Create deconvolution layers
        for i, f in enumerate(reversed_filters):
            kernel_h, kernel_w, out_channels = f
            is_final = (i == len(reversed_filters) - 1)
            
            deconv = nn.ConvTranspose2d(
                in_channels,
                self.in_channels if is_final else out_channels,
                kernel_size=(kernel_h, kernel_w),
                stride=4 if i == len(reversed_filters)-1 else 1,
                padding=2,
                output_padding=1 if i == len(reversed_filters)-1 else 0
            )
            self.deconv_layers.append(deconv)
            
            # Add ReLU except for final layer
            if not is_final:
                self.deconv_layers.append(nn.ReLU())
                
            in_channels = out_channels
        
        # Output activation
        self.output_act = self._get_output_activation()

    def _init_inverse_spp(self) -> nn.Module:
        """Create inverse SPP module for CNN decoder"""
        return nn.Sequential(
            nn.Linear(self.embed_dim, 256 * 14 * 14),  # Adjust dimensions as needed
            nn.Unflatten(1, (256, 14, 14))
        )

    def _init_sinusoidal_encoding(self, max_len: int, embed_dim: int) -> Parameter:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(1, max_len, embed_dim, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return Parameter(pe, requires_grad=False, name="sinusoidal_pe")

    def _get_output_activation(self):
        """Get output activation function based on config"""
        activations = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'none': nn.Identity()
        }
        return activations.get(self.output_activation, nn.Sigmoid())

    def assemble_patches(self, patches: torch.Tensor, orig_shape: Tuple[int, int]) -> torch.Tensor:
        """Reconstruct image from patches"""
        printer.status("VISION", "Reconstructing image from patches", "info")
        
        b, num_patches, _ = patches.shape
        c = self.in_channels
        patch_size = self.patch_size
        h, w = orig_shape
        
        # Calculate grid dimensions
        grid_h = int(math.sqrt(num_patches))
        grid_w = int(math.sqrt(num_patches))
        
        # Reshape to (B, grid_h, grid_w, c, patch_size, patch_size)
        x = patches.view(b, grid_h, grid_w, c, patch_size, patch_size)
        
        # Permute and reshape to (B, c, grid_h * patch_size, grid_w * patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, grid_h * patch_size, grid_w * patch_size)
        
        # Crop to original dimensions
        return x[:, :, :h, :w]

    def forward(self, x: torch.Tensor, style_id: Optional[torch.Tensor] = None, 
               orig_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Reconstruct image from encoded representations"""
        printer.status("VISION", "Reconstructing image from encoded representations", "info")
        
        if orig_shape is None:
            orig_shape = (self.img_size, self.img_size)
            
        if self.decoder_type == "transformer":
            return self._forward_transformer(x, style_id, orig_shape)
        elif self.decoder_type == "cnn":
            return self._forward_cnn(x, orig_shape)
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")

    def _forward_transformer(self, x: torch.Tensor, style_id: torch.Tensor, 
                            orig_shape: Tuple[int, int]) -> torch.Tensor:
        printer.status("VISION", "Transformer decoder forward pass", "info")
        
        # Add positional embeddings
        seq_len = x.size(1)
        if self.position_embed.shape[1] < seq_len:
            raise ValueError(
                f"Positional embedding size ({self.position_embed.shape[1]}) "
                f"smaller than sequence length ({seq_len})"
            )
        x = x + self.position_embed[:, :seq_len]
        
        # Process through transformer
        x = self.transformer(x, style_id)
        
        # Remove CLS token
        x = x[:, 1:, :]
        
        # Project to patch space
        patches = self.projection(x)
        
        # Reconstruct image
        image = self.assemble_patches(patches, orig_shape)
        return self.output_act(image)

    def _forward_cnn(self, x: torch.Tensor, orig_shape: Tuple[int, int]) -> torch.Tensor:
        """CNN-based image reconstruction"""
        printer.status("VISION", "CNN decoder forward pass", "info")
        
        # Initial projection
        x = self.initial_proj(x)
        
        # Inverse SPP
        x = self.inv_spp(x)
        
        # Process through deconvolution layers
        for layer in self.deconv_layers:
            x = layer(x)
        
        # Resize to original dimensions
        if x.shape[2:] != orig_shape:
            x = F.interpolate(x, size=orig_shape, mode='bilinear', align_corners=False)
        
        return self.output_act(x)

    def load_pretrained(self, weights: Dict[str, torch.Tensor]):
        """Load pretrained weights for transformer decoder"""
        printer.status("VISION", "Loading pretrained decoder weights", "info")
        
        if self.decoder_type != "transformer":
            logger.warning("load_pretrained only applicable for transformer decoder")
            return
        
        # Positional embeddings
        if 'pos_embed' in weights:
            self.position_embed.data.copy_(weights['pos_embed'])
        
        # Projection weights
        if 'decoder_proj' in weights:
            self.projection.weight.data.copy_(weights['decoder_proj.weight'])
            self.projection.bias.data.copy_(weights['decoder_proj.bias'])
        
        # Transformer weights
        transformer_weights = {k[len('decoder_transformer.'):]: v 
                              for k, v in weights.items() 
                              if k.startswith('decoder_transformer.')}
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def train(self, mode: bool = True):
        printer.status("VISION", "Setting decoder training mode", "info")
        super().train(mode)
        if self.decoder_type == "transformer":
            self.transformer.train(mode)
        return self

    def eval(self):
        printer.status("VISION", "Setting decoder evaluation mode", "info")
        return self.train(False)

if __name__ == "__main__":
    print("\n=== Testing Vision Decoder ===")
    
    # Override config for testing
    config = load_global_config()
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test transformer-based decoder
    config['encoder_type'] = "transformer"
    transformer_decoder = VisionDecoder().to(config['device'])
    print(f"\nDecoder type: {transformer_decoder.decoder_type}")
    
    # Create test input with CORRECT sequence length
    seq_length = transformer_decoder.base_num_patches + 1
    test_latent = torch.randn(4, seq_length, config['embed_dim']).to(config['device'])
    
    # Add dummy style_id (required by transformer)
    style_id = torch.zeros(4, dtype=torch.long).to(config['device'])
    
    output = transformer_decoder(test_latent, style_id)
    print("Output shape:", output.shape)
    
    # Test CNN-based decoder
    config['encoder_type'] = "cnn"
    cnn_decoder = VisionDecoder().to(config['device'])
    print(f"\nDecoder type: {cnn_decoder.decoder_type}")
    
    # Create test input for CNN (single vector per sample)
    test_latent_cnn = torch.randn(4, config['embed_dim']).to(config['device'])
    cnn_output = cnn_decoder(test_latent_cnn)
    print("CNN output shape:", cnn_output.shape)
    
    print("\n=== VisionDecoder tests passed ===")
