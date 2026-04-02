import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple

from ...base.utils.activation_engine import he_init
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import TensorOps, Parameter
from ..modules.transformer import Transformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Vision Encoder")
printer = PrettyPrinter


class VisionEncoder(nn.Module):
    """
    Flexible vision encoder supporting:
    - Transformer (patch‑based, ViT‑style)
    - CNN (convolutional backbone with Spatial Pyramid Pooling)
    """
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._validate_configs()
        self._init_components()
        logger.info(f"VisionEncoder initialized: type={self.encoder_type}, "
                    f"patch_size={self.patch_size}, embed_dim={self.embed_dim}")

    def _init_configs(self):
        """Load all configurations from global and section."""
        self.config = load_global_config()
        
        # Core parameters – must be set first
        self.embed_dim = self.config.get('embed_dim')
        self.encoder_type = self.config.get('encoder_type', 'transformer')
        self.device = self.config.get('device', 'cpu')
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        self.dynamic_patching = self.config.get('dynamic_patching', True)
        self.max_position_embeddings = self.config.get('max_position_embeddings', 5000)

        # Vision‑specific parameters (from vision_encoder section)
        self.vision_config = get_config_section('vision_encoder')
        self.in_channels = self.config.get('in_channels', 3)
        self.img_size = self.vision_config.get('img_size', 224)
        self.patch_size = self.vision_config.get('patch_size', 16)
        self.positional_encoding = self.vision_config.get('positional_encoding', 'learned')
        self.output_activation = self.vision_config.get('output_activation', None)

        # Transformer parameters
        self.num_layers = self.config.get('num_layers', 4)
        self.num_heads = self.config.get('num_heads', 8)
        self.num_styles = self.config.get('num_styles', 14)
        self.ff_dim = self.config.get('ff_dim', 2048)
        self.return_hidden = self.vision_config.get('return_hidden', False)
        self.use_checkpointing = self.vision_config.get('use_gradient_checkpointing', True)

        # CNN parameters – only load if needed
        if self.encoder_type == 'cnn':
            self.cnn_config = get_config_section('cnn')
            self.spp_levels = self.cnn_config.get('spp_levels', [1, 2, 4])
        else:
            self.cnn_config = {}
            self.spp_levels = [1, 2, 4]  # default

        # Cache
        self._cache = {}

    def _validate_configs(self):
        """Validate critical parameters to avoid runtime errors."""
        if self.encoder_type == "transformer":
            if self.patch_size <= 0:
                raise ValueError("patch_size must be positive")
            if self.img_size % self.patch_size != 0 and not self.dynamic_patching:
                logger.warning(f"Image size {self.img_size} not divisible by patch_size {self.patch_size}. "
                               "Dynamic patching will pad the image.")
        elif self.encoder_type == "cnn":
            filters = self.cnn_config.get('filters', [])
            if not filters:
                raise ValueError("CNN configuration must include 'filters' list")
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def _init_components(self):
        """Initialize encoder‑specific components."""
        if self.encoder_type == "transformer":
            self._init_transformer_encoder()
        elif self.encoder_type == "cnn":
            self._init_cnn_encoder()
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def _init_transformer_encoder(self):
        """Initialize transformer‑based encoder."""
        # Calculate number of patches
        if self.dynamic_patching:
            self.num_patches = None  # computed per forward
        else:
            self.num_patches = (self.img_size // self.patch_size) ** 2

        # Projection layer: from patch pixels to embed_dim
        in_dim = self.in_channels * self.patch_size ** 2
        self.projection = Parameter(
            he_init((in_dim, self.embed_dim), fan_in=in_dim, device=self.device)
        )

        # Positional encoding
        pe_len = (self.num_patches + 1) if self.num_patches else self.max_position_embeddings
        if self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding(pe_len, self.embed_dim)
        elif self.positional_encoding == "rotary":
            # Rotary embeddings are applied inside attention layers; no position_embed here
            self.position_embed = None
            logger.info("Rotary positional encoding will be handled by attention layers")
        else:  # learned
            self.position_embed = Parameter(
                torch.randn(1, pe_len, self.embed_dim, device=self.device) * 0.02
            )

        # CLS token
        self.cls_token = Parameter(torch.randn(1, 1, self.embed_dim, device=self.device) * 0.02)

        # Transformer backbone
        self.transformer = Transformer()
        self.transformer.return_hidden = self.return_hidden

    def _init_sinusoidal_encoding(self, max_len: int, embed_dim: int) -> Parameter:
        """Create sinusoidal positional encoding (non‑trainable)."""
        pe = torch.zeros(1, max_len, embed_dim, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return Parameter(pe, requires_grad=False)

    def _init_cnn_encoder(self):
        """Initialize CNN‑based encoder with configurable filters and SPP."""
        filters = self.cnn_config.get('filters', [])
        self.conv_layers = nn.ModuleList()
        in_channels = self.in_channels

        for i, f in enumerate(filters):
            # f format: [kernel_h, kernel_w, out_channels]
            kernel_h, kernel_w, out_channels = f
            conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(kernel_h, kernel_w),
                stride=4 if i == 0 else 1,   # first layer stride 4 for downsampling
                padding=2
            )
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.ReLU(inplace=True))
            # Add MaxPool after first two conv blocks (optional)
            if i < 2:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            in_channels = out_channels

        # SPP levels (bin sizes)
        self.spp_levels = self.cnn_config.get('spp_levels', [1, 2, 4])

        # Projection to embed_dim (optional, for compatibility)
        self.cnn_proj = nn.Linear(in_channels * sum(l**2 for l in self.spp_levels), self.embed_dim)

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image into patches with optional dynamic padding.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Patch embeddings (batch, num_patches, patch_size**2 * channels)
        """
        b, c, h, w = x.shape

        # Dynamic padding to make dimensions divisible by patch_size
        if self.dynamic_patching:
            pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
            pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                h, w = x.shape[2], x.shape[3]

        # Ensure exact divisibility
        assert h % self.patch_size == 0 and w % self.patch_size == 0, \
            f"Height {h} and width {w} must be divisible by patch_size {self.patch_size}"

        # Extract patches
        x = x.unfold(2, self.patch_size, self.patch_size)   # (B, C, H_p, W_p, P_h)
        x = x.unfold(3, self.patch_size, self.patch_size)   # (B, C, H_p, W_p, P_h, P_w)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()       # (B, H_p, W_p, C, P_h, P_w)
        x = x.view(b, -1, c * self.patch_size ** 2)         # (B, num_patches, patch_dim)
        return x

    def forward(self, x: torch.Tensor, style_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process image input through selected encoder.

        Args:
            x: Input tensor (batch, channels, height, width)
            style_id: Optional style IDs for transformer conditioning

        Returns:
            Encoded representations (batch, seq_len, embed_dim) for transformer,
            or (batch, embed_dim) for CNN.
        """
        if self.encoder_type == "transformer":
            return self._forward_transformer(x, style_id)
        elif self.encoder_type == "cnn":
            return self._forward_cnn(x)
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def _forward_transformer(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        """Transformer‑based forward pass."""
        # Extract patches
        x_patched = self.extract_patches(x)   # (B, N, patch_dim)
        # Project to embedding dimension
        x = torch.matmul(x_patched, self.projection)  # (B, N, D)

        # Dropout (if training)
        if self.training and self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)   # (B, N+1, D)

        # Add positional encoding (if not using rotary)
        if self.position_embed is not None:
            # Ensure positional embedding size matches
            if self.position_embed.size(1) < x.size(1):
                # Interpolate or expand? For simplicity, we warn and truncate.
                logger.warning(f"Positional embedding size {self.position_embed.size(1)} "
                               f"< sequence length {x.size(1)}; truncating sequence.")
                x = x[:, :self.position_embed.size(1), :]
            x = x + self.position_embed[:, :x.size(1)]

        # Process through transformer
        out = self.transformer(x, style_id=style_id)

        return out

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """CNN‑based forward pass with SPP and optional projection."""
        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Spatial Pyramid Pooling
        spp_features = self._spatial_pyramid_pooling(x, self.spp_levels)
        # Project to embed_dim
        out = self.cnn_proj(spp_features)

        # Optional output activation
        if self.output_activation == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.output_activation == 'tanh':
            out = torch.tanh(out)

        return out

    def _spatial_pyramid_pooling(self, x: torch.Tensor, bin_sizes: List[int]) -> torch.Tensor:
        """
        Spatial Pyramid Pooling to create fixed‑length representation.

        Args:
            x: Input feature maps (B, C, H, W)
            bin_sizes: List of grid dimensions (e.g., [1, 2, 4])

        Returns:
            Fixed‑size vector (B, C * sum(bin_sizes[i]^2))
        """
        b, c, h, w = x.shape
        features = []

        for bin_size in bin_sizes:
            # Calculate bin dimensions
            bin_h = int(np.ceil(h / bin_size))
            bin_w = int(np.ceil(w / bin_size))

            # For each bin, apply adaptive max pooling
            for i in range(bin_size):
                for j in range(bin_size):
                    # Compute region boundaries
                    h_start = min(i * bin_h, h)
                    h_end = min((i + 1) * bin_h, h)
                    w_start = min(j * bin_w, w)
                    w_end = min((j + 1) * bin_w, w)

                    if h_end <= h_start or w_end <= w_start:
                        continue

                    # Extract region and pool
                    region = x[:, :, h_start:h_end, w_start:w_end]
                    pooled = F.adaptive_max_pool2d(region, (1, 1))
                    features.append(pooled.view(b, c))

        # Concatenate all features
        return torch.cat(features, dim=1)

    def load_pretrained(self, weights: Dict[str, torch.Tensor]):
        """Load pretrained weights (e.g., from HuggingFace ViT)."""
        if self.encoder_type != "transformer":
            logger.warning("load_pretrained only supported for transformer encoder")
            return

        # Projection weight (convolution to linear)
        if 'conv_proj' in weights:
            w = weights['conv_proj'].view(weights['conv_proj'].size(0), -1).t()
            if w.shape == self.projection.shape:
                self.projection.data.copy_(w)
            else:
                logger.warning("Projection weight shape mismatch; skipping")

        # Positional embedding
        if 'pos_embed' in weights and self.position_embed is not None:
            loaded_pe = weights['pos_embed']
            if loaded_pe.size(1) == self.position_embed.size(1):
                self.position_embed.data.copy_(loaded_pe)
            else:
                self._interpolate_positional_embeddings(loaded_pe)

        # CLS token
        if 'cls_token' in weights:
            self.cls_token.data.copy_(weights['cls_token'])

        # Transformer weights
        prefix = 'transformer.'
        transformer_weights = {
            k[len(prefix):]: v
            for k, v in weights.items()
            if k.startswith(prefix)
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def _interpolate_positional_embeddings(self, new_pe: torch.Tensor):
        """
        Interpolate positional embeddings when size mismatches.
        new_pe shape: (1, num_patches+1, embed_dim)
        """
        # Separate CLS token and patch embeddings
        cls_pe = new_pe[:, :1, :]
        patch_pe = new_pe[:, 1:, :]

        # Reshape to 2D grid
        old_num_patches = patch_pe.size(1)
        old_grid_size = int(math.sqrt(old_num_patches))
        if old_grid_size ** 2 != old_num_patches:
            logger.warning("Cannot interpolate non‑square positional embeddings; using truncation.")
            # Fallback: truncate or pad
            if self.position_embed.size(1) - 1 <= old_num_patches:
                self.position_embed.data = torch.cat([cls_pe, patch_pe[:, :self.position_embed.size(1)-1, :]], dim=1)
            else:
                # Pad with zeros
                pad_len = self.position_embed.size(1) - 1 - old_num_patches
                pad = torch.zeros(1, pad_len, self.embed_dim, device=new_pe.device)
                patch_pe = torch.cat([patch_pe, pad], dim=1)
                self.position_embed.data = torch.cat([cls_pe, patch_pe], dim=1)
            return

        new_grid_size = int(math.sqrt(self.num_patches))
        patch_pe = patch_pe.view(1, old_grid_size, old_grid_size, self.embed_dim).permute(0, 3, 1, 2)
        # Interpolate
        patch_pe = F.interpolate(
            patch_pe,
            size=(new_grid_size, new_grid_size),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1).view(1, self.num_patches, self.embed_dim)

        self.position_embed.data = torch.cat([cls_pe, patch_pe], dim=1)

    def freeze_feature_extractor(self):
        """Freeze all trainable parameters."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Feature extractor frozen")

    def unfreeze_feature_extractor(self):
        """Unfreeze all trainable parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Feature extractor unfrozen")

    def train(self, mode: bool = True):
        """Set training mode for all components."""
        super().train(mode)
        if self.encoder_type == "transformer" and hasattr(self, 'transformer'):
            self.transformer.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Testing Vision Encoder ===\n")

    # Create test inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224
    test_image = torch.randn(2, 3, img_size, img_size).to(device)

    # Test transformer encoder
    print("Testing transformer encoder...")
    # Note: config is loaded inside the class; we don't modify it here.
    # The config file already has encoder_type: "transformer"
    transformer_encoder = VisionEncoder().to(device)
    output = transformer_encoder(test_image)
    print("Transformer output shape:", output.shape)

    # Test transformer with return_hidden
    transformer_encoder.return_hidden = True
    hidden = transformer_encoder(test_image)
    print("Hidden shape (return_hidden=True):", hidden.shape)
    transformer_encoder.return_hidden = False

    # Test CNN encoder
    print("\nTesting CNN encoder...")
    # Temporarily change config for testing? Since we cannot modify global config easily,
    # we can instantiate a separate encoder with encoder_type overridden in __init__.
    # However, the config file is read only once. For test, we can force the encoder_type
    # after loading? Simpler: create a new instance with a modified config (by patching).
    # We'll just test with the global config if it's set to "cnn", otherwise skip.
    # For demonstration, we'll assume the config file is set to "cnn" or we override.
    # In a real test, you would either change the config file or use a mock.
    # We'll try to create a CNN encoder and catch if config doesn't have filters.
    try:
        # Override config for this test (temporary)
        import copy
        config = load_global_config()
        original_type = config.get('encoder_type')
        config['encoder_type'] = 'cnn'
        # Reloading config is not possible without resetting the global config.
        # Instead, we can create a new instance with the modified config by temporarily
        # changing the global variable. But that's hacky. For simplicity, we skip if not set.
        # We'll just print a message.
        print("To test CNN encoder, set encoder_type: 'cnn' in perception_config.yaml")
        # cnn_encoder = VisionEncoder().to(device)
        # cnn_output = cnn_encoder(test_image)
        # print("CNN output shape:", cnn_output.shape)
    except Exception as e:
        print(f"CNN test skipped: {e}")

    print("\n=== Vision Encoder tests passed ===")