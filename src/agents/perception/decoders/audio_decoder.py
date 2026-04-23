import math
import torch
import torch.nn as nn

from typing import Dict, List, Optional

from ...base.modules.activation_engine import he_init
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import Parameter
from ..modules.transformer import Transformer
from ..perception_memory import PerceptionMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Audio Decoder")
printer = PrettyPrinter

class AudioDecoder(nn.Module):
    """
    Audio decoder that reconstructs waveforms from latent representations.
    Supports multiple decoder types:
    - transformer (patch‑based, mirroring the encoder)
    - cnn (convolutional decoder, future)
    - mfcc (inverse MFCC, future)
    """
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._validate_configs()
        self._init_components()
        logger.info(f"AudioDecoder initialized: type={self.decoder_type}, "
                    f"patch_size={self.patch_size}, embed_dim={self.embed_dim}")

    def _init_configs(self):
        """Load all configurations from global and section."""
        self.config = load_global_config()
        self.audio_config = get_config_section('audio_encoder')  # reuse encoder config
        self.decoder_config = get_config_section('audio_decoder') if 'audio_decoder' in self.config else {}

        # Core parameters
        self.embed_dim = self.config.get('embed_dim')
        self.in_channels = self.config.get('in_channels', 1)
        self.decoder_type = self.config.get('decoder_type', self.config.get('encoder_type', 'transformer'))
        self.device = self.config.get('device', 'cpu')
        self.max_position_embeddings = self.config.get('max_position_embeddings', 5000)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        self.num_styles = self.config.get('num_styles', 14)

        # Audio‑specific parameters
        self.audio_length = self.audio_config.get('audio_length', 16000)
        self.patch_size = self.audio_config.get('patch_size', 400)
        self.positional_encoding = self.audio_config.get('positional_encoding', 'learned')
        self.dynamic_patching = self.config.get('dynamic_patching', True)

        # Transformer‑specific (if used)
        self.return_hidden = self.decoder_config.get('return_hidden', False)
        self.use_checkpointing = self.decoder_config.get('use_gradient_checkpointing', True)

        # Cache for hidden states (if needed)
        self._hidden_states = None

    def _validate_configs(self):
        """Validate critical parameters."""
        if not self.embed_dim:
            raise ValueError("embed_dim must be specified in config")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.decoder_type not in ['transformer', 'cnn', 'mfcc']:
            logger.warning(f"Unknown decoder type '{self.decoder_type}', falling back to 'transformer'")
            self.decoder_type = 'transformer'

    def _init_components(self):
        """Initialize decoder components based on type."""
        if self.decoder_type == "transformer":
            self._init_transformer_decoder()
        elif self.decoder_type == "cnn":
            self._init_cnn_decoder()
        elif self.decoder_type == "mfcc":
            self._init_mfcc_decoder()
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")

        # Memory for gradient checkpointing
        self.memory = PerceptionMemory(enable_checkpointing=self.use_checkpointing)

    def _init_transformer_decoder(self):
        """Initialize transformer‑based decoder (mirrors encoder)."""
        # Inverse projection: embed_dim -> patch_size * in_channels
        in_dim = self.embed_dim
        out_dim = self.patch_size * self.in_channels
        self.projection = Parameter(
            he_init((in_dim, out_dim), fan_in=in_dim, device=self.device)
        )

        # Positional encoding (same as encoder)
        if self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding()
        elif self.positional_encoding == "rotary":
            # Rotary handled inside transformer; no separate embedding
            self.position_embed = None
            logger.info("Rotary positional encoding handled by attention layers")
        else:  # learned
            self.position_embed = Parameter(
                torch.randn(1, self.max_position_embeddings, self.embed_dim, device=self.device) * 0.02
            )

        # Style embeddings (mirror encoder)
        self.style_embeddings = Parameter(
            torch.randn(self.num_styles, self.embed_dim, device=self.device) * 0.02
        )

        # Transformer backbone
        self.transformer = Transformer()
        self.transformer.return_hidden = True

    def _init_sinusoidal_encoding(self) -> Parameter:
        """Create sinusoidal positional encoding (non‑trainable)."""
        pe = torch.zeros(1, self.max_position_embeddings, self.embed_dim, device=self.device)
        position = torch.arange(0, self.max_position_embeddings, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=self.device).float() *
                             (-math.log(10000.0) / self.embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return Parameter(pe, requires_grad=False)

    def _init_cnn_decoder(self):
        """Placeholder for CNN‑based decoder."""
        logger.warning("CNN decoder not fully implemented; using transformer fallback")
        self.decoder_type = 'transformer'
        self._init_transformer_decoder()

    def _init_mfcc_decoder(self):
        """Placeholder for inverse MFCC decoder."""
        logger.warning("MFCC decoder not fully implemented; using transformer fallback")
        self.decoder_type = 'transformer'
        self._init_transformer_decoder()

    def forward(self, x: torch.Tensor, style_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct audio from latent representations.

        Args:
            x: Latent tensor (batch, seq_len, embed_dim)
            style_id: Optional style IDs for conditioning (batch,)

        Returns:
            Reconstructed waveform (batch, channels, time)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D) -> add sequence dimension if needed

        if self.decoder_type == "transformer":
            return self._forward_transformer(x, style_id)
        else:
            raise NotImplementedError(f"Forward not implemented for {self.decoder_type}")

    def _forward_transformer(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        """Transformer‑based decoder forward pass."""
        batch_size, seq_len, _ = x.shape

        # Add positional embeddings (if not using rotary)
        if self.position_embed is not None:
            if seq_len > self.position_embed.size(1):
                logger.warning(f"Sequence length {seq_len} exceeds positional embedding size "
                               f"{self.position_embed.size(1)}; truncating.")
                x = x[:, :self.position_embed.size(1), :]
                seq_len = x.size(1)
            x = x + self.position_embed[:, :seq_len, :]

        # Add style embeddings
        if style_id is None:
            style_id = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        style_emb = self.style_embeddings[style_id].unsqueeze(1)  # (B, 1, D)
        x = x + style_emb

        # Process through transformer (output is (B, L, D) if return_hidden=True)
        x = self.transformer(x, style_id=style_id)

        # Project to patch space
        x = torch.matmul(x, self.projection)  # (B, L, patch_size * in_channels)

        # Reconstruct waveform from patches
        return self._patches_to_waveform(x)

    def _patches_to_waveform(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patch embeddings back to waveform.

        Args:
            patches: Tensor (B, num_patches, patch_size * in_channels)

        Returns:
            Waveform (B, in_channels, T)
        """
        batch_size, num_patches, _ = patches.shape

        # Reshape to (B, num_patches, in_channels, patch_size)
        patches = patches.view(batch_size, num_patches, self.in_channels, self.patch_size)

        # Permute to (B, in_channels, num_patches, patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()

        # Flatten to waveform
        output_length = num_patches * self.patch_size
        waveform = patches.view(batch_size, self.in_channels, output_length)

        # Trim to original audio length (if known)
        if self.audio_length and output_length > self.audio_length:
            waveform = waveform[..., :self.audio_length]

        return waveform

    def load_pretrained(self, weights: Dict[str, torch.Tensor]):
        """Load pretrained weights from a dictionary (e.g., from a pre‑trained encoder)."""
        if self.decoder_type != "transformer":
            logger.warning("load_pretrained only supported for transformer decoder")
            return

        # Map keys: decoder uses slightly different naming
        mapping = {
            'dec_proj': 'projection',
            'dec_pos_embed': 'position_embed',
            'dec_style_emb': 'style_embeddings',
        }
        for old_key, new_key in mapping.items():
            if old_key in weights:
                param = getattr(self, new_key, None)
                if param is not None:
                    param.data.copy_(weights[old_key].to(self.device))

        # Load transformer weights (if present)
        prefix = 'decoder.transformer.'
        transformer_weights = {
            k[len(prefix):]: v
            for k, v in weights.items()
            if k.startswith(prefix)
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def freeze_layers(self, layer_indices: Optional[List[int]] = None):
        """Freeze specific transformer layers or all parameters."""
        if layer_indices is None:
            for param in self.parameters():
                param.requires_grad = False
            logger.info("All decoder layers frozen")
        elif self.decoder_type == "transformer":
            self.transformer.freeze_layers(layer_indices)
        else:
            logger.warning("Freezing layers only supported for transformer decoder")

    def unfreeze_layers(self, layer_indices: Optional[List[int]] = None):
        """Unfreeze specific transformer layers or all parameters."""
        if layer_indices is None:
            for param in self.parameters():
                param.requires_grad = True
            logger.info("All decoder layers unfrozen")
        elif self.decoder_type == "transformer":
            self.transformer.unfreeze_layers(layer_indices)
        else:
            logger.warning("Unfreezing layers only supported for transformer decoder")

    def train(self, mode: bool = True):
        """Set training mode for all components."""
        super().train(mode)
        if hasattr(self, 'transformer'):
            self.transformer.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Testing Audio Decoder ===\n")

    # Override config for testing
    config = load_global_config()
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    config['encoder_type'] = "transformer"

    # Calculate expected number of patches
    audio_length = config['audio_encoder']['audio_length']
    patch_size = config['audio_encoder']['patch_size']
    num_patches = (audio_length + patch_size - 1) // patch_size

    # Create test input (latent representation)
    latent = torch.randn(2, num_patches, config['embed_dim']).to(config['device'])

    # Initialize decoder
    decoder = AudioDecoder().to(config['device'])

    # Test forward pass
    print(f"Decoder type: {decoder.decoder_type}")
    output = decoder(latent)
    print("Output shape:", output.shape)

    # Verify reconstruction dimensions
    expected_shape = (2, config['in_channels'], audio_length)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"
    print("\n=== Audio Decoder tests passed ===")