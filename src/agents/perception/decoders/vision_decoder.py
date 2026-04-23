import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import TensorOps, Parameter
from ..modules.transformer import Transformer
from ..perception_memory import PerceptionMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Vision Decoder")
printer = PrettyPrinter

class VisionDecoder(nn.Module):
    """
    Vision decoder that reconstructs images from latent representations.
    Supports transformer and CNN decoders.
    """
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._validate_configs()
        self._init_components()
        logger.info(f"VisionDecoder initialized: type={self.decoder_type}, "
                    f"patch_size={self.patch_size}, embed_dim={self.embed_dim}")

    def _init_configs(self):
        """Load all configurations from global and section."""
        self.config = load_global_config()
        self.vision_config = get_config_section('vision_encoder')
        self.cnn_config = get_config_section('cnn') if 'cnn' in self.config else {}
        self.decoder_config = get_config_section('vision_decoder') if 'vision_decoder' in self.config else {}

        # Core parameters
        self.embed_dim = self.config.get('embed_dim')
        self.in_channels = self.config.get('in_channels', 3)
        self.decoder_type = self.config.get('decoder_type', self.config.get('encoder_type', 'transformer'))
        self.device = self.config.get('device', 'cpu')
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        self.positional_encoding = self.vision_config.get('positional_encoding', 'learned')
        self.max_position_embeddings = self.config.get('max_position_embeddings', 5000)

        # Vision‑specific parameters
        self.img_size = self.vision_config.get('img_size', 224)
        self.patch_size = self.vision_config.get('patch_size', 16)
        self.output_activation = self.vision_config.get('output_activation', 'sigmoid')
        self.dynamic_patching = self.config.get('dynamic_patching', True)

        # Transformer parameters
        self.num_layers = self.config.get('num_layers', 4)
        self.num_heads = self.config.get('num_heads', 8)
        self.num_styles = self.config.get('num_styles', 14)
        self.ff_dim = self.config.get('ff_dim', 2048)

        # Decoder‑specific parameters
        self.return_hidden = self.decoder_config.get('return_hidden', False)
        self.use_checkpointing = self.decoder_config.get('use_gradient_checkpointing', True)

        # Calculate derived dimensions
        self.base_num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_dim = self.in_channels * self.patch_size ** 2

        # Attention maps cache
        self.attention_maps = {}

    def _validate_configs(self):
        """Validate critical parameters."""
        if not self.embed_dim:
            raise ValueError("embed_dim must be specified")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.decoder_type not in ['transformer', 'cnn']:
            logger.warning(f"Unknown decoder type '{self.decoder_type}', falling back to 'transformer'")
            self.decoder_type = 'transformer'

    def _init_components(self):
        """Initialize decoder components based on type."""
        if self.decoder_type == "transformer":
            self._init_transformer_decoder()
        elif self.decoder_type == "cnn":
            self._init_cnn_decoder()
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")

        # Memory for gradient checkpointing
        self.memory = PerceptionMemory(enable_checkpointing=self.use_checkpointing)

    def _init_transformer_decoder(self):
        """Initialize transformer‑based decoder."""
        # Positional encoding
        num_positions = self.base_num_patches + 1  # +1 for CLS token
        if self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding(num_positions, self.embed_dim)
        elif self.positional_encoding == "rotary":
            self.position_embed = None
            logger.info("Rotary positional encoding handled by attention layers")
        else:  # learned
            self.position_embed = Parameter(
                torch.randn(1, num_positions, self.embed_dim, device=self.device) * 0.02
            )

        # Transformer backbone
        self.transformer = Transformer()
        self.transformer.return_hidden = True  # Always return full sequence
        self.transformer.causal = False        # Vision decoder doesn't need causal masking

        # Disable feedforward fusion (no context addition)
        for layer in self.transformer.layers:
            if 'ff' in layer and hasattr(layer['ff'], 'fusion_type'):
                layer['ff'].fusion_type = None

        # Projection to patch space
        self.projection = nn.Linear(self.embed_dim, self.patch_dim, bias=True)

        # Output activation
        self.output_act = self._get_output_activation()

    def _init_sinusoidal_encoding(self, max_len: int, embed_dim: int) -> Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(1, max_len, embed_dim, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=self.device).float() *
                             (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return Parameter(pe, requires_grad=False)

    def _init_cnn_decoder(self):
        """Initialize CNN‑based decoder with configurable upsampling layers."""
        cnn_decoder_config = self.cnn_config.get('decoder', {})

        # Base spatial size after initial projection (e.g., 8x8)
        self.decoder_base_size = cnn_decoder_config.get('base_size', 8)
        # Number of channels at the base feature map
        self.decoder_base_channels = cnn_decoder_config.get('base_channels', 256)

        # Upsampling layers: list of dicts with out_channels, kernel_size, stride, padding, output_padding
        up_layers = cnn_decoder_config.get('up_layers', [
            {'out_channels': 128, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'output_padding': 0},
            {'out_channels': 64, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'output_padding': 0},
            {'out_channels': 32, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'output_padding': 0},
            {'out_channels': self.in_channels, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'output_padding': 0},
        ])

        # Compute target spatial size after upsampling
        target_size = self.decoder_base_size
        for layer in up_layers:
            # Output = (input - 1) * stride - 2*padding + kernel_size + output_padding
            target_size = (target_size - 1) * layer['stride'] - 2 * layer['padding'] + layer['kernel_size'] + layer.get('output_padding', 0)

        if target_size != self.img_size:
            logger.warning(f"CNN decoder upsampling results in size {target_size}, but expected {self.img_size}. "
                           "Adjusting final interpolation.")
            self._resize_to_target = True
        else:
            self._resize_to_target = False

        # Linear projection from embed_dim to base feature vector
        base_features = self.decoder_base_channels * self.decoder_base_size * self.decoder_base_size
        self.initial_proj = nn.Linear(self.embed_dim, base_features)

        # Build transposed conv layers
        self.up_layers = nn.ModuleList()
        in_channels = self.decoder_base_channels
        for layer_cfg in up_layers:
            conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=layer_cfg['out_channels'],
                kernel_size=layer_cfg['kernel_size'],
                stride=layer_cfg['stride'],
                padding=layer_cfg['padding'],
                output_padding=layer_cfg.get('output_padding', 0),
                bias=False
            )
            self.up_layers.append(conv)
            # Add BatchNorm and ReLU except for the last layer
            if layer_cfg['out_channels'] != self.in_channels:
                self.up_layers.append(nn.BatchNorm2d(layer_cfg['out_channels']))
                self.up_layers.append(nn.ReLU(inplace=True))
            in_channels = layer_cfg['out_channels']

        # Output activation
        self.output_act = self._get_output_activation()

        # Weight initialization
        self._init_cnn_weights()

    def _init_cnn_weights(self):
        """Initialize CNN decoder weights."""
        for module in self.up_layers:
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.initial_proj.weight, 0, 0.02)
        if self.initial_proj.bias is not None:
            nn.init.constant_(self.initial_proj.bias, 0)

    def _get_output_activation(self):
        """Get output activation function."""
        activations = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'none': nn.Identity()
        }
        return activations.get(self.output_activation, nn.Sigmoid())

    def assemble_patches(self, patches: torch.Tensor, orig_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Reconstruct image from patches (excluding CLS token).

        Args:
            patches: Tensor (B, num_patches, patch_dim)
            orig_shape: Original (height, width)

        Returns:
            Reconstructed image (B, C, H, W)
        """
        batch_size, num_patches, _ = patches.shape
        c = self.in_channels
        patch_size = self.patch_size
        h, w = orig_shape

        # Calculate grid dimensions (assumes square grid)
        grid_size = int(math.sqrt(num_patches))
        if grid_size ** 2 != num_patches:
            # Fallback: use 1D to 2D with padding
            logger.warning(f"Number of patches ({num_patches}) not a perfect square; adjusting grid.")
            grid_size = int(math.ceil(math.sqrt(num_patches)))
            pad = grid_size ** 2 - num_patches
            if pad > 0:
                patches = F.pad(patches, (0, 0, 0, pad))

        # Reshape to (B, grid, grid, C, patch_size, patch_size)
        patches = patches.view(batch_size, grid_size, grid_size, c, patch_size, patch_size)

        # Permute and combine
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        reconstructed = patches.view(batch_size, c, grid_size * patch_size, grid_size * patch_size)

        # Crop to original size
        return reconstructed[:, :, :h, :w]

    def forward(
        self,
        x: torch.Tensor,
        style_id: Optional[torch.Tensor] = None,
        orig_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Reconstruct image from latent representation.

        Args:
            x: Latent tensor (batch, seq_len, embed_dim) for transformer,
               or (batch, embed_dim) for CNN
            style_id: Optional style IDs (batch,)
            orig_shape: Original image dimensions (height, width)

        Returns:
            Reconstructed image (batch, channels, height, width)
        """
        if orig_shape is None:
            orig_shape = (self.img_size, self.img_size)

        if self.decoder_type == "transformer":
            return self._forward_transformer(x, style_id, orig_shape)
        elif self.decoder_type == "cnn":
            return self._forward_cnn(x, orig_shape)
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")

    def _forward_transformer(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor,
        orig_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Transformer decoder forward pass."""
        # Add positional embeddings
        seq_len = x.size(1)
        if self.position_embed is not None:
            if seq_len > self.position_embed.size(1):
                logger.warning(f"Sequence length {seq_len} exceeds positional embedding size "
                               f"{self.position_embed.size(1)}; truncating.")
                x = x[:, :self.position_embed.size(1), :]
                seq_len = x.size(1)
            x = x + self.position_embed[:, :seq_len, :]

        # Process through transformer
        x = self.transformer(x, style_id=style_id)

        # Remove CLS token (first token) if present (assumes CLS token is first)
        if x.size(1) > self.base_num_patches:
            x = x[:, 1:, :]  # (B, num_patches, D)

        # Project to patch space
        patches = self.projection(x)  # (B, num_patches, patch_dim)

        # Reconstruct image
        image = self.assemble_patches(patches, orig_shape)
        return self.output_act(image)

    def _forward_cnn(self, x: torch.Tensor, orig_shape: Tuple[int, int]) -> torch.Tensor:
        """CNN decoder forward pass."""
        # Project latent vector to base feature vector
        x = self.initial_proj(x)  # (B, base_features)

        # Reshape to (B, C, H, W)
        x = x.view(-1, self.decoder_base_channels, self.decoder_base_size, self.decoder_base_size)

        # Apply upsampling layers
        for layer in self.up_layers:
            x = layer(x)

        # Resize to original dimensions if needed
        if self._resize_to_target:
            x = F.interpolate(x, size=orig_shape, mode='bilinear', align_corners=False)

        return self.output_act(x)

    def register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        if self.decoder_type != "transformer":
            logger.warning("Attention hooks only supported for transformer decoder")
            return

        self.transformer.output_attentions = True
        self.attention_maps = {}

        def hook_fn(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.attention_maps[layer_idx] = attn_weights.detach().cpu()
            return hook

        for idx, layer in enumerate(self.transformer.layers):
            if 'attention' in layer:
                attn_module = layer['attention']
                attn_module.output_attentions = True
                attn_module.register_forward_hook(hook_fn(idx))
                logger.info(f"Registered attention hook for layer {idx}")

    def freeze_layers(self, layer_indices: Optional[List[int]] = None):
        """Freeze transformer layers or all parameters."""
        if self.decoder_type != "transformer":
            logger.warning("Freezing layers only supported for transformer decoder")
            return
        if layer_indices is None:
            for param in self.parameters():
                param.requires_grad = False
            logger.info("All decoder layers frozen")
        else:
            self.transformer.freeze_layers(layer_indices)

    def unfreeze_layers(self, layer_indices: Optional[List[int]] = None):
        """Unfreeze transformer layers or all parameters."""
        if self.decoder_type != "transformer":
            logger.warning("Unfreezing layers only supported for transformer decoder")
            return
        if layer_indices is None:
            for param in self.parameters():
                param.requires_grad = True
            logger.info("All decoder layers unfrozen")
        else:
            self.transformer.unfreeze_layers(layer_indices)

    def load_pretrained(self, weights: Dict[str, torch.Tensor]):
        """Load pretrained weights."""
        if self.decoder_type != "transformer":
            logger.warning("load_pretrained only supported for transformer decoder")
            return

        # Positional embeddings
        if 'pos_embed' in weights:
            self.position_embed.data.copy_(weights['pos_embed'].to(self.device))

        # Projection weights
        if 'decoder_proj.weight' in weights:
            self.projection.weight.data.copy_(weights['decoder_proj.weight'].to(self.device))
            if 'decoder_proj.bias' in weights:
                self.projection.bias.data.copy_(weights['decoder_proj.bias'].to(self.device))

        # Transformer weights
        prefix = 'decoder_transformer.'
        transformer_weights = {
            k[len(prefix):]: v
            for k, v in weights.items()
            if k.startswith(prefix)
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'transformer'):
            self.transformer.train(mode)
        return self

    def eval(self):
        return self.train(False)


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Testing Vision Decoder ===\n")

    # Override config for testing
    config = load_global_config()
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # Test transformer decoder
    print("Testing transformer decoder...")
    config['encoder_type'] = "transformer"
    transformer_decoder = VisionDecoder().to(config['device'])
    seq_len = transformer_decoder.base_num_patches + 1
    test_latent = torch.randn(2, seq_len, config['embed_dim']).to(config['device'])
    style_id = torch.zeros(2, dtype=torch.long).to(config['device'])
    output = transformer_decoder(test_latent, style_id)
    print("Transformer output shape:", output.shape)

    # Test CNN decoder
    print("\nTesting CNN decoder...")
    config['encoder_type'] = "cnn"
    cnn_decoder = VisionDecoder().to(config['device'])
    test_latent_cnn = torch.randn(2, config['embed_dim']).to(config['device'])
    cnn_output = cnn_decoder(test_latent_cnn)
    print("CNN output shape:", cnn_output.shape)

    print("\n=== VisionDecoder tests passed ===\n")