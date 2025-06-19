import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.base.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Audio Encoder")
printer = PrettyPrinter

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._init_components()
        logger.info(f"AudioEncoder initialized: type={self.encoder_type}, "
                   f"patch_size={self.patch_size}, embed_dim={self.embed_dim},"
                   f"in_channels={self.in_channels}")

    def _init_configs(self):
        """Load and validate all configurations"""
        self.config = load_global_config()
        self.audio_config = get_config_section('audio_encoder')
        self.mfcc_config = get_config_section('mfcc')
        
        # Core parameters
        self.embed_dim = self.config.get('embed_dim')
        self.encoder_type = self.config.get('encoder_type')
        self.device = self.config.get('device')
        self.max_position_embeddings = self.config.get('max_position_embeddings')
        self.dropout_rate = self.config.get('dropout_rate')
        
        # Audio-specific parameters
        self.in_channels = self.audio_config.get('in_channels')
        self.audio_length = self.audio_config.get('audio_length')
        self.patch_size = self.audio_config.get('patch_size')
        self.positional_encoding = self.audio_config.get('positional_encoding')
        self.dynamic_patching = self.config.get('dynamic_patching')
        
        # MFCC parameters (conditionally load if "mfcc" is part of the encoder_type)
        if "mfcc" in str(self.encoder_type).lower():
            self.sample_rate = self.mfcc_config.get('sample_rate')
            self.n_mfcc = self.mfcc_config.get('n_mfcc')
            self.frame_length_ms = self.mfcc_config.get('frame_length_ms') # Store ms for clarity
            self.frame_step_ms = self.mfcc_config.get('frame_step_ms')   # Store ms for clarity
            
            if self.sample_rate and self.frame_length_ms and self.frame_step_ms: # Check if necessary mfcc params exist
                self.frame_length = int(self.frame_length_ms * self.sample_rate / 1000)
                self.frame_step = int(self.frame_step_ms * self.sample_rate / 1000)
            else: # Handle missing parameters for MFCC if MFCC is chosen
                if "mfcc" in str(self.encoder_type).lower(): # Only warn if MFCC type is explicitly chosen
                    logger.warning("MFCC encoder type selected, but sample_rate, frame_length_ms, or frame_step_ms missing in mfcc_config.")
                self.frame_length = 0 # or some default
                self.frame_step = 0   # or some default
            
            self.n_filters = self.mfcc_config.get('n_filters')
            self.low_freq = self.mfcc_config.get('low_freq')
            self.high_freq = self.mfcc_config.get('high_freq')

    def _init_components(self):
        """Initialize encoder-specific components"""
        if self.encoder_type == "transformer":
            self._init_transformer_encoder()
        elif self.encoder_type == "mfcc":
            self._init_mfcc_encoder()
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def _init_transformer_encoder(self):
        """Initialize transformer-based encoder components"""
        # Projection layer - now using standard tensor initialization
        self.projection = nn.Parameter(
            TensorOps.he_init(
                (self.patch_size * self.in_channels, self.embed_dim),
                fan_in=self.patch_size * self.in_channels,
                device=self.device
            )
        )
        
        # Positional encoding
        if self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding()
        else:  # learned
            self.position_embed = nn.Parameter(
                torch.randn(1, self.max_position_embeddings, self.embed_dim, 
                          device=self.device) * 0.02
            )
        
        # Special tokens - now using standard Parameter
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.embed_dim, device=self.device) * 0.02
        )
        
        # Transformer backbone
        self.transformer = Transformer()

    def _init_mfcc_encoder(self):
        """Initialize MFCC-based encoder components"""
        printer.status("AUDIO", "Initializing MFCC-based encoder", "info")

        # Precomputed buffers - register directly instead of setting as attributes
        self.register_buffer("mel_filters", self._create_mel_filterbank())
        self.register_buffer("dct_matrix", self._create_dct_matrix().t())  # Transpose to (n_filters, n_mfcc)
        
        # Projection layer
        self.mfcc_proj = nn.Sequential(
            nn.Linear(self.n_mfcc, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim)
        )

    def _init_sinusoidal_encoding(self) -> Parameter:
        """Create sinusoidal positional encoding (non-trainable)"""
        printer.status("AUDIO", "Creating sinusoidal positional encoding", "info")

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
        Process audio input through selected encoder
        Args:
            x: Input tensor of shape (B, C, T)
            style_id: Optional style IDs for transformer conditioning
        Returns:
            Encoded representations (B, L, D)
        """
        printer.status("AUDIO", "Processing audio input through selected encoder", "info")

        if isinstance(x, tuple):
            x = x[0]  # or raise an error
    
        if self.encoder_type == "transformer":
            return self._forward_transformer(x, style_id)
        return self._forward_mfcc(x)

    def _forward_transformer(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        printer.status("AUDIO", "Forward transformer", "info")

        # Patch extraction and projection
        x = self.extract_patches(x)  # (B, N, P*C)
        x = torch.matmul(x, self.projection)  # (B, N, D)
        
        # Apply dropout during training - handle tuple return
        if self.training and self.dropout_rate > 0:
            dropped = TensorOps.dropout(x)
            x = dropped[0] if isinstance(dropped, tuple) else dropped
        
        # Add CLS token and positional embeddings
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.position_embed[:, :x.size(1)]
        
        # Process through transformer
        return self.transformer(x, style_id)

    def _forward_mfcc(self, x: torch.Tensor) -> torch.Tensor:
        """MFCC feature extraction and projection"""
        printer.status("AUDIO", "MFCC feature extraction", "info")

        mfcc = self._extract_mfcc(x)  # (B, T, n_mfcc)
        return self.mfcc_proj(mfcc)  # (B, T, D)

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform into patches
        Args:
            x: Input tensor (B, C, T)
        Returns:
            Patch embeddings (B, num_patches, patch_size*C)
        """
        printer.status("AUDIO", "Converting waveform into patches", "info")

        if x.ndim == 2:
            x = x.unsqueeze(1)  # Add channel dim: (B, 1, T)
        
        b, c, t = x.shape
        if self.dynamic_patching:
            pad = (self.patch_size - (t % self.patch_size)) % self.patch_size
            x = F.pad(x, (0, pad))
        
        # Unfold into patches
        x = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, N, P)
        x = x.permute(0, 2, 1, 3)  # (B, N, C, P)
        return x.reshape(b, -1, c * self.patch_size)  # (B, N, C*P)

    def _extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Vectorized MFCC extraction
        Args:
            waveform: Input audio (B, C, T)
        Returns:
            MFCC features (B, frames, n_mfcc)
        """
        printer.status("AUDIO", "Vectorizing MFCC extraction", "info")

        # Handle multi-channel input
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)  # Mix to mono: (B, T)
        
        # Frame extraction
        frames = waveform.unfold(1, self.frame_length, self.frame_step)  # (B, N, L)
        
        # Windowing
        window = torch.hamming_window(self.frame_length, device=waveform.device)
        windowed = frames * window
        
        # Power spectrum
        fft = torch.fft.rfft(windowed, dim=-1)
        power_spectrum = torch.abs(fft) ** 2
        
        # Mel filterbank
        mel_energies = torch.matmul(power_spectrum, self.mel_filters.t())
        return torch.matmul(torch.log(mel_energies + 1e-6), self.dct_matrix)

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create Mel-scale filterbank"""
        printer.status("AUDIO", "Creating Mel-scale filterbank", "info")
    
        # Convert frequencies to tensors
        low_freq = torch.tensor(self.low_freq, device=self.device)
        high_freq = torch.tensor(self.high_freq, device=self.device)
        
        # Frequency to Mel conversion
        low_mel = 2595 * torch.log10(1 + low_freq / 700)
        high_mel = 2595 * torch.log10(1 + high_freq / 700)
        
        # Mel points and Hz conversion
        mel_points = torch.linspace(low_mel, high_mel, self.n_filters + 2, device=self.device)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = torch.floor((self.frame_length + 1) * hz_points / self.sample_rate)
        
        # Build filters
        filters = torch.zeros(self.n_filters, self.frame_length // 2 + 1, device=self.device)
        for i in range(1, self.n_filters + 1):
            left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]
            left_idx, center_idx, right_idx = map(int, (left, center, right))
            
            # Rising slope
            if center_idx > left_idx:
                filters[i-1, left_idx:center_idx] = torch.linspace(0, 1, center_idx - left_idx, device=self.device)
            
            # Falling slope
            if right_idx > center_idx:
                filters[i-1, center_idx:right_idx] = torch.linspace(1, 0, right_idx - center_idx, device=self.device)
        
        # Normalize filters
        return filters / filters.sum(dim=1, keepdim=True)

    def _create_dct_matrix(self) -> torch.Tensor:
        """Create DCT matrix for MFCC computation"""
        printer.status("AUDIO", "Creating DCT matrix", "info")

        n = self.n_filters
        k = torch.arange(self.n_mfcc, device=self.device)[:, None]
        j = torch.arange(n, device=self.device)
        return torch.cos(math.pi * k * (2 * j + 1) / (2 * n)) * math.sqrt(2 / n)

    def load_pretrained(self, weights: Dict[str, torch.Tensor]):
        """Load pretrained weights"""
        printer.status("AUDIO", "Loading pretrained weights", "info")

        if self.encoder_type == "transformer":
            # Load projection weights
            if 'conv_proj' in weights:
                w = weights['conv_proj'].reshape(weights['conv_proj'].shape[0], -1).T
                self.projection.data.copy_(w)
            
            # Load special tokens
            self.cls_token.data.copy_(weights.get('cls_token', self.cls_token.data))
            self.position_embed.data.copy_(weights.get('pos_embed', self.position_embed.data))
            
            # Load transformer weights
            prefix = 'transformer.'
            transformer_weights = {
                k[len(prefix):]: v 
                for k, v in weights.items() 
                if k.startswith(prefix)
            }
            self.transformer.load_pretrained(transformer_weights)

    def train(self, mode: bool = True):
        """Set training mode"""
        printer.status("AUDIO", "Setting training mode", "info")

        super().train(mode)
        if self.encoder_type == "transformer":
            self.transformer.train(mode)
        return self

    def eval(self):
        """Set evaluation mode"""
        printer.status("AUDIO", "Setting evaluation mode", "info")

        return self.train(False)

if __name__ == "__main__":
    print("\n=== Testing Audio Encoder ===")

    # Override config for testing
    config = load_global_config()
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    config['encoder_type'] = "transformer"
    config['dynamic_patching'] = True
    
    # Create test input
    audio = torch.randn(4, 1, 16000).to(config['device'])
    
    # Initialize encoder
    encoder = AudioEncoder().to(config['device'])
    
    # Test forward pass
    print(f"\nEncoder type: {encoder.encoder_type}")
    output = encoder(audio)
    print("Output shape:", output.shape)
    
    # Test MFCC mode
    config['encoder_type'] = "mfcc"
    mfcc_encoder = AudioEncoder().to(config['device'])
    print(f"\nEncoder type: {mfcc_encoder.encoder_type}")
    mfcc_output = mfcc_encoder(audio)
    print("MFCC output shape:", mfcc_output.shape)
    print("\n=== AudioEncoder tests passed ===")
