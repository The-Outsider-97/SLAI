import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional, Tuple, List, Union

from ...base.utils.activation_engine import he_init
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import TensorOps, Parameter
from ..modules.transformer import Transformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Audio Encoder")
printer = PrettyPrinter


class AudioEncoder(nn.Module):
    """
    Flexible audio encoder supporting multiple backends:
    - Transformer (patch‑based)
    - MFCC (Mel‑frequency cepstral coefficients)
    - CNN (1D convolutional encoder)
    - Wav2Vec2 (placeholder for pre‑trained models)
    """
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._validate_configs()
        self._init_components()
        logger.info(f"AudioEncoder initialized: type={self.encoder_type}, "
                    f"patch_size={self.patch_size}, embed_dim={self.embed_dim}, "
                    f"in_channels={self.in_channels}")

    def _init_configs(self):
        """Load all configurations from global and section."""
        self.config = load_global_config()
        self.audio_config = get_config_section('audio_encoder')
        self.mfcc_config = get_config_section('mfcc')

        # Core parameters
        self.embed_dim = self.config.get('embed_dim')
        self.encoder_type = self.config.get('encoder_type', 'transformer')
        self.device = self.config.get('device', 'cpu')
        self.max_position_embeddings = self.config.get('max_position_embeddings', 5000)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)

        # Audio‑specific parameters
        self.in_channels = self.audio_config.get('in_channels', 1)
        self.audio_length = self.audio_config.get('audio_length', 16000)
        self.patch_size = self.audio_config.get('patch_size', 400)
        self.positional_encoding = self.audio_config.get('positional_encoding', 'sinusoidal')
        self.dynamic_patching = self.config.get('dynamic_patching', True)
        self.patch_overlap = self.audio_config.get('patch_overlap', 0.0)  # 0 = no overlap
        self.normalize_input = self.audio_config.get('normalize_input', True)
        self.return_hidden = self.audio_config.get('return_hidden', False)

        # MFCC parameters (only if needed)
        if "mfcc" in self.encoder_type.lower():
            self.sample_rate = self.mfcc_config.get('sample_rate', 16000)
            self.n_mfcc = self.mfcc_config.get('n_mfcc', 13)
            self.frame_length_ms = self.mfcc_config.get('frame_length_ms', 25)
            self.frame_step_ms = self.mfcc_config.get('frame_step_ms', 10)
            self.n_filters = self.mfcc_config.get('n_filters', 40)
            self.low_freq = self.mfcc_config.get('low_freq', 0)
            self.high_freq = self.mfcc_config.get('high_freq', self.sample_rate // 2)

            # Convert ms to samples
            self.frame_length = int(self.frame_length_ms * self.sample_rate / 1000)
            self.frame_step = int(self.frame_step_ms * self.sample_rate / 1000)

        # CNN parameters (if using CNN encoder)
        self.cnn_config = get_config_section('cnn') if self.encoder_type == 'cnn' else {}

    def _validate_configs(self):
        """Validate critical parameters to avoid runtime errors."""
        if self.encoder_type == "transformer":
            if self.patch_size <= 0:
                raise ValueError("patch_size must be positive for transformer encoder")
            if self.max_position_embeddings <= 0:
                raise ValueError("max_position_embeddings must be positive")
        elif self.encoder_type == "mfcc":
            if self.sample_rate <= 0:
                raise ValueError("sample_rate must be positive")
            if self.frame_length <= 0 or self.frame_step <= 0:
                raise ValueError("frame_length and frame_step must be positive")
            if self.n_mfcc > self.n_filters:
                logger.warning(f"n_mfcc ({self.n_mfcc}) > n_filters ({self.n_filters}), DCT will pad zeros")
        elif self.encoder_type not in ['cnn', 'wav2vec2']:
            logger.warning(f"Unknown encoder type '{self.encoder_type}', using transformer as fallback")
            self.encoder_type = 'transformer'

    def _init_components(self):
        """Initialize encoder‑specific components."""
        if self.encoder_type == "transformer":
            self._init_transformer_encoder()
        elif self.encoder_type == "mfcc":
            self._init_mfcc_encoder()
        elif self.encoder_type == "cnn":
            self._init_cnn_encoder()
        elif self.encoder_type == "wav2vec2":
            self._init_wav2vec2_encoder()
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

        # Common feature normalizer (if enabled)
        if self.normalize_input:
            self.norm = nn.LayerNorm(self.embed_dim)  # applied after projection

    def _init_transformer_encoder(self):
        """Initialize transformer‑based encoder."""
        # Projection from patches to embedding dimension
        self.projection = Parameter(
            he_init(
                (self.patch_size * self.in_channels, self.embed_dim),
                fan_in=self.patch_size * self.in_channels,
                device=self.device
            )
        )

        # Positional encoding
        if self.positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding()
        elif self.positional_encoding == "rotary":
            # Placeholder for rotary embeddings (implemented inside attention)
            self.position_embed = None
            logger.info("Rotary positional encoding will be handled by attention layers")
        else:  # learned
            self.position_embed = Parameter(
                torch.randn(1, self.max_position_embeddings, self.embed_dim, device=self.device) * 0.02
            )

        # CLS token
        self.cls_token = Parameter(torch.randn(1, 1, self.embed_dim, device=self.device) * 0.02)

        # Transformer backbone
        self.transformer = Transformer()

        # Store transformer layers for hidden states
        self._hidden_states = None

    def _init_sinusoidal_encoding(self) -> Parameter:
        """Create sinusoidal positional encoding (non‑trainable)."""
        pe = torch.zeros(1, self.max_position_embeddings, self.embed_dim, device=self.device)
        position = torch.arange(0, self.max_position_embeddings, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() *
                             (-math.log(10000.0) / self.embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return Parameter(pe, requires_grad=False)

    def _init_mfcc_encoder(self):
        """Initialize MFCC‑based encoder."""
        # Precompute Mel filterbank and DCT matrix (buffers)
        self.register_buffer("mel_filters", self._create_mel_filterbank())
        self.register_buffer("dct_matrix", self._create_dct_matrix().t())  # (n_filters, n_mfcc)

        # Projection to embedding dimension
        self.mfcc_proj = nn.Sequential(
            nn.Linear(self.n_mfcc, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim)
        )

    def _init_cnn_encoder(self):
        """Initialize CNN‑based encoder (example with configurable layers)."""
        # Example: simple 1D CNN
        cnn_layers = []
        in_channels = self.in_channels
        out_channels = self.cnn_config.get('out_channels', [64, 128, 256])
        kernel_sizes = self.cnn_config.get('kernel_sizes', [3, 3, 3])
        strides = self.cnn_config.get('strides', [2, 2, 2])

        for i, (oc, ks, st) in enumerate(zip(out_channels, kernel_sizes, strides)):
            cnn_layers.extend([
                nn.Conv1d(in_channels, oc, kernel_size=ks, stride=st, padding=ks//2),
                nn.BatchNorm1d(oc),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = oc

        self.cnn_encoder = nn.Sequential(*cnn_layers)

        # Adaptive pooling to fixed length
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Projection to embedding dimension
        self.cnn_proj = nn.Linear(in_channels, self.embed_dim)

    def _init_wav2vec2_encoder(self):
        """Placeholder for pre‑trained Wav2Vec2 model."""
        # In production, you'd load a model from transformers library
        # We'll create a dummy placeholder for now.
        logger.warning("Wav2Vec2 encoder is not fully implemented; using transformer fallback")
        self.encoder_type = 'transformer'
        self._init_transformer_encoder()

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create Mel‑scale filterbank (tensor on self.device)."""
        # Convert frequencies to tensors
        low_freq_t = torch.tensor(self.low_freq, device=self.device, dtype=torch.float)
        high_freq_t = torch.tensor(self.high_freq, device=self.device, dtype=torch.float)
    
        # Convert frequencies to Mel
        low_mel = 2595 * torch.log10(1 + low_freq_t / 700)
        high_mel = 2595 * torch.log10(1 + high_freq_t / 700)
    
        # Mel points and Hz conversion
        mel_points = torch.linspace(low_mel, high_mel, self.n_filters + 2, device=self.device)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = torch.floor((self.frame_length + 1) * hz_points / self.sample_rate).long()
    
        # Build filters
        n_fft_bins = self.frame_length // 2 + 1
        filters = torch.zeros(self.n_filters, n_fft_bins, device=self.device)
        for i in range(1, self.n_filters + 1):
            left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]
            if left < center:
                # Rising slope
                filters[i-1, left:center] = torch.linspace(0, 1, center - left, device=self.device)
            if center < right:
                # Falling slope
                filters[i-1, center:right] = torch.linspace(1, 0, right - center, device=self.device)
    
        # Normalize by energy
        filters = filters / filters.sum(dim=1, keepdim=True)
        return filters

    def _create_dct_matrix(self) -> torch.Tensor:
        """Create DCT matrix for MFCC computation (type‑II)."""
        n = self.n_filters
        k = torch.arange(self.n_mfcc, device=self.device)[:, None]
        j = torch.arange(n, device=self.device)
        dct = torch.cos(math.pi * k * (2 * j + 1) / (2 * n)) * math.sqrt(2 / n)
        # First coefficient scaling (optional)
        dct[0] *= 1 / math.sqrt(2)
        return dct

    def forward(self, x: torch.Tensor, style_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process audio input through selected encoder.

        Args:
            x: Input tensor (batch, channels, time) or (batch, time) for mono
            style_id: Optional style IDs for transformer conditioning

        Returns:
            Encoded representations (batch, seq_len, embed_dim)
        """
        if isinstance(x, tuple):
            x = x[0]  # handle data loader returning tuples

        # Ensure at least 3D (B, C, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        # Optional input normalization
        if self.normalize_input and self.encoder_type != 'mfcc':
            # Per‑channel mean/std (over time)
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-8
            x = (x - mean) / std

        if self.encoder_type == "transformer":
            return self._forward_transformer(x, style_id)
        elif self.encoder_type == "mfcc":
            return self._forward_mfcc(x)
        elif self.encoder_type == "cnn":
            return self._forward_cnn(x)
        else:
            raise ValueError(f"Encoder type {self.encoder_type} not implemented in forward")

    def _forward_transformer(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        """Transformer‑based forward pass."""
        # Extract patches
        x = self.extract_patches(x)  # (B, N, P*C)

        # Project to embedding dimension
        x = torch.matmul(x, self.projection)  # (B, N, D)

        # Dropout (if training)
        if self.training and self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding (if not using rotary)
        if self.position_embed is not None:
            x = x + self.position_embed[:, :x.size(1)]

        # Process through transformer
        if self.return_hidden:
            # Get hidden states from transformer if we want them
            # For simplicity, we'll just return the output (transformer already returns final)
            # Extend Transformer to return list of hidden states if needed.
            output = self.transformer(x, style_id)
            # Optionally store hidden states (would require modifying Transformer)
            return output
        else:
            return self.transformer(x, style_id)

    def _forward_mfcc(self, x: torch.Tensor) -> torch.Tensor:
        """MFCC‑based forward pass."""
        # Extract MFCC features
        mfcc = self._extract_mfcc(x)  # (B, frames, n_mfcc)

        # Project to embedding dimension
        return self.mfcc_proj(mfcc)

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """CNN‑based forward pass."""
        # x: (B, C, T)
        out = self.cnn_encoder(x)  # (B, C', T')
        # Global average pooling over time
        out = self.global_pool(out)  # (B, C', 1)
        out = out.squeeze(-1)        # (B, C')
        # Project to embedding dimension
        return self.cnn_proj(out).unsqueeze(1)  # (B, 1, D) for sequence compatibility

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform into patches.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Patch embeddings (B, num_patches, patch_size * C)
        """
        B, C, T = x.shape

        # Dynamic patching: pad to multiple of patch_size
        if self.dynamic_patching:
            pad_len = (self.patch_size - (T % self.patch_size)) % self.patch_size
            if pad_len > 0:
                x = F.pad(x, (0, pad_len))
                T = x.shape[2]

        # Calculate number of patches with optional overlap
        if self.patch_overlap > 0:
            stride = int(self.patch_size * (1 - self.patch_overlap))
            if stride <= 0:
                stride = 1
            patches = x.unfold(2, self.patch_size, stride)  # (B, C, N, P)
        else:
            patches = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, N, P)

        # Rearrange to (B, N, C, P)
        patches = patches.permute(0, 2, 1, 3)  # (B, N, C, P)
        # Flatten channel and patch dimensions
        return patches.reshape(B, -1, C * self.patch_size)

    def _extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features from waveform.

        Args:
            waveform: (B, C, T) or (B, T)

        Returns:
            MFCC features (B, frames, n_mfcc)
        """
        # Convert to mono if multi‑channel
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)  # (B, T)

        # Frame extraction
        frames = waveform.unfold(1, self.frame_length, self.frame_step)  # (B, N, L)

        # Windowing
        window = torch.hamming_window(self.frame_length, device=waveform.device)
        windowed = frames * window

        # Power spectrum (magnitude squared)
        fft = torch.fft.rfft(windowed, dim=-1)
        power_spectrum = fft.abs().square()  # (B, N, freq_bins)

        # Mel filterbank
        mel_energies = torch.matmul(power_spectrum, self.mel_filters.t())  # (B, N, n_filters)
        log_mel = torch.log(mel_energies + 1e-6)

        # DCT to get MFCCs
        mfcc = torch.matmul(log_mel, self.dct_matrix)  # (B, N, n_mfcc)
        return mfcc

    def load_pretrained(self, weights: Dict[str, torch.Tensor]):
        """Load pretrained weights for transformer encoder."""
        if self.encoder_type != "transformer":
            logger.warning("load_pretrained only supported for transformer encoder")
            return

        # Load projection weight (if shape matches)
        if 'conv_proj' in weights:
            w = weights['conv_proj'].reshape(weights['conv_proj'].shape[0], -1).T
            if w.shape == self.projection.shape:
                self.projection.data.copy_(w)
            else:
                logger.warning("Projection weight shape mismatch; skipping")

        # Load CLS token
        if 'cls_token' in weights:
            self.cls_token.data.copy_(weights['cls_token'])

        # Load positional embedding (if learned)
        if 'pos_embed' in weights and self.position_embed is not None:
            self.position_embed.data.copy_(weights['pos_embed'])

        # Load transformer weights
        prefix = 'transformer.'
        transformer_weights = {
            k[len(prefix):]: v
            for k, v in weights.items()
            if k.startswith(prefix)
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def train(self, mode: bool = True):
        """Set training mode for all components."""
        super().train(mode)
        if hasattr(self, 'transformer'):
            self.transformer.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def freeze_feature_extractor(self):
        """Freeze the feature extraction part (for fine‑tuning)."""
        if self.encoder_type == "transformer":
            for param in self.projection.parameters():
                param.requires_grad = False
            if self.position_embed is not None and self.position_embed.requires_grad:
                self.position_embed.requires_grad = False
            # Optionally freeze transformer layers
            # self.transformer.freeze_layers()  # if needed
        elif self.encoder_type == "mfcc":
            # MFCC has no trainable parameters except the projection
            for param in self.mfcc_proj.parameters():
                param.requires_grad = False
        elif self.encoder_type == "cnn":
            for param in self.cnn_encoder.parameters():
                param.requires_grad = False
            for param in self.cnn_proj.parameters():
                param.requires_grad = False
        logger.info("Feature extractor frozen")

    def unfreeze_feature_extractor(self):
        """Unfreeze feature extraction part."""
        if self.encoder_type == "transformer":
            for param in self.projection.parameters():
                param.requires_grad = True
            if self.position_embed is not None and not self.position_embed.requires_grad:
                self.position_embed.requires_grad = True
        elif self.encoder_type == "mfcc":
            for param in self.mfcc_proj.parameters():
                param.requires_grad = True
        elif self.encoder_type == "cnn":
            for param in self.cnn_encoder.parameters():
                param.requires_grad = True
            for param in self.cnn_proj.parameters():
                param.requires_grad = True
        logger.info("Feature extractor unfrozen")

    def get_hidden_states(self) -> Optional[List[torch.Tensor]]:
        """Return hidden states from transformer if available."""
        if hasattr(self.transformer, 'get_hidden_states'):
            return self.transformer.get_hidden_states()
        return None


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Testing Audio Encoder ===\n")

    # Create test inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio = torch.randn(2, 1, 16000).to(device)

    # Test transformer encoder
    print("Testing transformer encoder...")
    config = load_global_config()
    config['device'] = str(device)
    config['encoder_type'] = "transformer"
    config['dynamic_patching'] = True
    config['dropout_rate'] = 0.1
    encoder = AudioEncoder().to(device)
    output = encoder(audio)
    print("Transformer output shape:", output.shape)

    # Test MFCC encoder
    print("\nTesting MFCC encoder...")
    config['encoder_type'] = "mfcc"
    mfcc_encoder = AudioEncoder().to(device)
    mfcc_output = mfcc_encoder(audio)
    print("MFCC output shape:", mfcc_output.shape)

    # Test CNN encoder
    print("\nTesting CNN encoder...")
    config['encoder_type'] = "cnn"
    cnn_encoder = AudioEncoder().to(device)
    cnn_output = cnn_encoder(audio)
    print("CNN output shape:", cnn_output.shape)

    print("\n=== Audio Encoder tests passed ===")
