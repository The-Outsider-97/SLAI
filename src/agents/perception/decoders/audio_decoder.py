import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional, Tuple

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Audio Decoder")
printer = PrettyPrinter

class AudioDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._init_components()
        logger.info(f"AudioDecoder initialized: type={self.decoder_type}, "
                   f"patch_size={self.patch_size}, embed_dim={self.embed_dim}")

    def _init_configs(self):
        """Load and validate all configurations"""
        self.config = load_global_config()
        self.audio_config = get_config_section('audio_encoder')
        
        # Core parameters
        self.embed_dim = self.config['embed_dim']
        self.in_channels = self.config['in_channels']
        self.decoder_type = self.config['encoder_type']  # Mirror encoder type
        self.device = self.config['device']
        self.max_position_embeddings = self.config['max_position_embeddings']
        self.dropout_rate = self.config['dropout_rate']
        
        # Audio-specific parameters
        self.audio_length = self.audio_config['audio_length']
        self.patch_size = self.audio_config['patch_size']
        self.positional_encoding = self.audio_config['positional_encoding']
        self.dynamic_patching = self.config['dynamic_patching']

    def _init_components(self):
        """Initialize decoder-specific components"""
        if self.decoder_type == "transformer":
            self._init_transformer_decoder()
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")

    def _init_transformer_decoder(self):
        """Initialize transformer-based decoder components"""
        # Projection layer - inverse of encoder projection
        self.projection = Parameter(
            TensorOps.he_init(
                (self.embed_dim, self.patch_size * self.in_channels),
                fan_in=self.embed_dim,
                device=self.device
            )
        )
        
        # Positional encoding (same as encoder)
        if self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding()
        else:  # learned
            self.position_embed = Parameter(
                torch.randn(1, self.max_position_embeddings, self.embed_dim, 
                          device=self.device) * 0.02
            )
        
        # Transformer backbone (with return_hidden enabled)
        self.transformer = Transformer()
        self.transformer.return_hidden = True  # Force hidden state return
        
        # Style embeddings (mirror encoder)
        self.style_embeddings = Parameter(
            torch.randn(self.transformer.num_styles, self.embed_dim) * 0.02
        )

    def _init_sinusoidal_encoding(self) -> Parameter:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(1, self.max_position_embeddings, self.embed_dim, 
                       device=self.device)
        position = torch.arange(0, self.max_position_embeddings, dtype=torch.float,
                             device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() *
                         (-math.log(10000.0) / self.embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor, style_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct audio from latent representations
        Args:
            x: Latent tensor of shape (B, L, D)
            style_id: Optional style IDs for conditioning
        Returns:
            Reconstructed waveform (B, C, T)
        """
        printer.status("AUDIO", "Reconstruct audio from latent representations", "info")

        if self.decoder_type == "transformer":
            return self._forward_transformer(x, style_id)
        else:
            raise NotImplementedError

    def _forward_transformer(
        self, 
        x: torch.Tensor, 
        style_id: torch.Tensor
    ) -> torch.Tensor:
        """Transformer decoder forward pass"""
        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.position_embed[:, :seq_len]
        
        # Process through transformer
        x = self.transformer(x, style_id)  # (B, L, D)
        
        # Project to patch space
        x = torch.matmul(x, self.projection)  # (B, L, P*C)
        
        # Reconstruct waveform from patches
        return self._patches_to_waveform(x)

    def _patches_to_waveform(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patch embeddings back to waveform
        Args:
            patches: Patch tensor (B, num_patches, patch_size*C)
        Returns:
            Reconstructed waveform (B, C, T)
        """
        printer.status("AUDIO", "Converting patch embeddings", "info")

        batch_size, num_patches, _ = patches.shape
        
        # Reshape to (B, num_patches, C, patch_size)
        patches = patches.view(
            batch_size, 
            num_patches, 
            self.in_channels, 
            self.patch_size
        )
        
        # Permute to (B, C, num_patches, patch_size)
        patches = patches.permute(0, 2, 1, 3)
        
        # Calculate output length
        output_length = num_patches * self.patch_size
        
        # Reconstruct waveform
        waveform = patches.reshape(batch_size, self.in_channels, output_length)
        
        # Trim to original audio length
        return waveform[..., :self.audio_length]

    def load_pretrained(self, weights: Dict[str, torch.Tensor]):
        """Load pretrained weights"""
        printer.status("AUDIO", "Loading pretrained weights", "info")

        if self.decoder_type == "transformer":
            # Load projection weights
            if 'dec_proj' in weights:
                self.projection.data.copy_(weights['dec_proj'])
            
            # Load special tokens
            self.position_embed.data.copy_(
                weights.get('dec_pos_embed', self.position_embed.data)
            )
            self.style_embeddings.data.copy_(
                weights.get('dec_style_emb', self.style_embeddings.data)
            )
            
            # Load transformer weights
            prefix = 'decoder.transformer.'
            transformer_weights = {
                k[len(prefix):]: v 
                for k, v in weights.items() 
                if k.startswith(prefix)
            }
            self.transformer.load_pretrained(transformer_weights)

    def train(self, mode: bool = True):
        """Set training mode"""
        super().train(mode)
        self.transformer.train(mode)
        return self

    def eval(self):
        """Set evaluation mode"""
        return self.train(False)

if __name__ == "__main__":
    print("\n=== Testing Audio Decoder ===")
    
    # Override config for testing
    config = load_global_config()
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    config['encoder_type'] = "transformer"
    
    # Calculate expected number of patches
    audio_length = config['audio_encoder']['audio_length']
    patch_size = config['audio_encoder']['patch_size']
    num_patches = (audio_length + patch_size - 1) // patch_size
    
    # Create test input (latent representation)
    latent = torch.randn(4, num_patches, config['embed_dim']).to(config['device'])
    
    # Initialize decoder
    decoder = AudioDecoder().to(config['device'])
    
    # Test forward pass
    print(f"\nDecoder type: {decoder.decoder_type}")
    output = decoder(latent)
    print("Output shape:", output.shape)
    
    # Verify reconstruction dimensions
    expected_shape = (4, config['in_channels'], audio_length)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"
    print("\n=== AudioDecoder tests passed ===")
