import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger

logger = get_logger("Audio Encoder")

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

class AudioEncoder(nn.Module):
    def __init__(self, config, device: str = 'cpu'):
        super().__init__()
        self._cache = {}
        audio_cfg = config['audio_encoder']
        transformer_cfg = config['transformer']
        
        # Audio-specific parameters
        self.encoder_type = audio_cfg['encoder_type']
        self.audio_length = audio_cfg['audio_length']
        self.in_channels = audio_cfg['in_channels']
        self.device = device

        # Shared transformer parameters
        self.embed_dim = transformer_cfg['embed_dim']
        
        if self.encoder_type == "transformer":
            self.patch_size = audio_cfg['patch_size']
            self.dynamic_patching = audio_cfg['dynamic_patching']
            self.positional_encoding = audio_cfg['positional_encoding']
            self.dropout_rate = audio_cfg['dropout_rate']
            
            # Projection layer
            self.projection = Parameter(
                TensorOps.he_init(
                    (self.patch_size * self.in_channels, self.embed_dim),
                    self.patch_size * self.in_channels,
                    device=device
                )
            )
            
            # Positional encoding
            if self.positional_encoding == "sinusoidal":
                self.position_embed = self._init_sinusoidal_encoding()
            else:  # learned
                self.position_embed = Parameter(
                    torch.randn(1, transformer_cfg['max_position_embeddings'], 
                              self.embed_dim, device=device) * 0.02
                )
            
            self.cls_token = Parameter(torch.randn(1, 1, self.embed_dim, device=device) * 0.02)
            self.transformer = Transformer(config)
            
        elif self.encoder_type == "mfcc":
            mfcc_cfg = audio_cfg['mfcc']
            self.n_mfcc = mfcc_cfg['n_mfcc']
            self.sample_rate = mfcc_cfg['sample_rate']
            self.frame_length = int(mfcc_cfg['frame_length_ms'] * self.sample_rate / 1000)
            self.frame_step = int(mfcc_cfg['frame_step_ms'] * self.sample_rate / 1000)
            
            # Project MFCC features to embed_dim
            self.mfcc_proj = nn.Linear(self.n_mfcc, self.embed_dim)
            self.mel_filters = self._create_mel_filterbank(mfcc_cfg)

    def _init_sinusoidal_encoding(self):
        pe = torch.zeros(1, self.max_position_embeddings, self.embed_dim, device=self.device)
        position = torch.arange(0, self.max_position_embeddings, dtype=torch.float, 
                              device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * 
                           (-math.log(10000.0) / self.embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return Parameter(pe, requires_grad=False)

    def extract_patches(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Add channel dim if missing
        b, c, l = x.shape
        
        if self.dynamic_patching:
            pad = (self.patch_size - (l % self.patch_size)) % self.patch_size
            x = F.pad(x, (0, pad))
        
        x = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, num_patches, patch_size)
        x = x.permute(0, 2, 1, 3).reshape(b, -1, self.patch_size * c)
        return x

    def forward(self, x, style_id=0):
        if self.encoder_type == "transformer":
            x = self.extract_patches(x)
            self._cache['input_shape'] = x.shape
            
            x = torch.matmul(x, self.projection.data)
            
            if self.training and self.dropout_rate > 0:
                mask = (torch.rand(x.shape, device=self.device) > self.dropout_rate).float()
                x *= mask
            
            cls_tokens = self.cls_token.data.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            seq_len = x.size(1)
            x += self.position_embed.data[:, :seq_len]
            
            return self.transformer.forward(x, style_id)
        
        elif self.encoder_type == "mfcc":
            mfcc = self._extract_mfcc(x)  # (batch, seq_len, n_mfcc)
            return self.mfcc_proj(mfcc)  # Project to embed_dim
            
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def backward(self, dout):
        """Backprop through encoder"""
        d_x = self.transformer.backward(dout)
        d_x = d_x[:, 1:, :]  # Remove CLS token
        
        # Gradient for projection
        d_proj = torch.matmul(
            self._cache['input_shape'].transpose(0, 2, 1), 
            d_x.reshape(-1, self.embed_dim))
        self.projection.grad += d_proj.sum(axis=0)
        
        return torch.matmul(d_x, self.projection.data.T)

    def _extract_mfcc(self, waveform):
        """Extract MFCC features with proper batching"""
        batch_size = waveform.shape[0]
        frames = []
        
        for i in range(batch_size):
            # Process each sample in batch
            frames.append(self._mfcc_forward(waveform[i]))
        
        return torch.stack(frames)  # (batch, seq_len, n_mfcc)

    def _mfcc_forward(self, waveform):
        """MFCC feature extraction for single sample"""
        T = waveform.shape[-1]
        N = 1 + (T - self.frame_length) // self.frame_step
        mfccs = []
        
        for i in range(N):
            frame = waveform[..., i*self.frame_step : i*self.frame_step+self.frame_length]
            
            # Windowing
            window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(self.frame_length, 
                                    device=self.device) / (self.frame_length - 1))
            windowed = frame * window
            
            # Power spectrum
            spectrum = torch.abs(torch.fft.rfft(windowed)) ** 2
            
            # Mel filterbank
            filter_energies = torch.matmul(spectrum, self.mel_filters.T)
            
            # Log compression and DCT
            log_energies = torch.log(filter_energies + 1e-6)
            mfcc = torch.matmul(log_energies, self._dct_matrix())
            
            mfccs.append(mfcc)
        
        return torch.stack(mfccs)  # (seq_len, n_mfcc)

    def _dct_matrix(self):
        """Create DCT-II matrix for MFCC computation"""
        n_filters = self.mel_filters.shape[0]
        dct = torch.zeros((self.n_mfcc, n_filters), device=self.device)
        
        for i in range(self.n_mfcc):
            for j in range(n_filters):
                dct[i,j] = math.cos(math.pi * i * (j + 0.5) / n_filters)
                
        return dct

    def parameters(self):
        if self.encoder_type == "transformer":
            return [self.projection, self.cls_token, self.position_embed] + self.transformer.parameters()
        else:
            return list(self.mfcc_proj.parameters())

    def _create_mel_filterbank(self, config: Dict[str, Any]) -> torch.Tensor:
        """Create Mel-scale filter bank"""
        n_filters = config.get('n_filters', 40)
        low_freq = config.get('low_freq', 0)
        high_freq = config.get('high_freq', self.sample_rate//2)
        
        # Convert frequencies to Mel scale
        low_mel = 2595 * torch.log10(1 + low_freq/700)
        high_mel = 2595 * torch.log10(1 + high_freq/700)
        
        # Create filter points
        mel_points = torch.linspace(low_mel, high_mel, n_filters + 2)
        hz_points = 700 * (10**(mel_points/2595) - 1)
        bin_points = torch.floor((self.frame_length + 1) * hz_points / self.sample_rate)
        
        # Create filter banks
        filters = torch.zeros((n_filters, self.frame_length//2 + 1))
        for i in range(1, n_filters + 1):
            left = bin_points[i-1]
            center = bin_points[i]
            right = bin_points[i+1]
            
            for j in range(int(left), int(center)):
                filters[i-1, j] = (j - left) / (center - left)
            for j in range(int(center), int(right)):
                filters[i-1, j] = (right - j) / (right - center)
        
        return filters

    def load_pretrained(self, weights):
        """Handle 1D conv, transformer, and positional weights"""
        if 'conv_proj' in weights:
            # Convert (embed_dim, in_channels, kernel_size) → (in_channels*kernel_size, embed_dim)
            w = weights['conv_proj'].reshape(weights['conv_proj'].shape[0], -1).T
            self.projection.data = w
        
        self.cls_token.data = weights.get('cls_token', self.cls_token.data)
        self.position_embed.data = weights.get('pos_embed', self.position_embed.data)
        
        # Load transformer weights
        transformer_weights = {
            k.split('transformer_')[-1]: v 
            for k, v in weights.items() 
            if k.startswith('transformer_')
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def _mfcc_forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """MFCC feature extraction pipeline"""
        T = len(waveform)
        L = self.frame_length
        S = self.frame_step
        N = 1 + (T - L) // S
        
        mfccs = []
        for i in range(N):
            # Frame extraction
            frame = waveform[i*S : i*S+L]
            
            # Pre-emphasis
            emphasized = torch.append(frame[0], frame[1:] - 0.97 * frame[:-1])
            
            # Windowing
            window = 0.54 - 0.46 * torch.cos(2 * torch.pi * torch.arange(L) / (L - 1))
            windowed = emphasized * window
            
            # Power spectrum
            spectrum = torch.abs(torch.fft.rfft(windowed)) ** 2
            
            # Mel filterbank
            filter_energies = torch.dot(spectrum, self.mel_filters.T)
            
            # Log compression
            log_energies = torch.log(filter_energies + 1e-6)
            
            # DCT
            mfcc = torch.dot(log_energies, self._dct_matrix(self.n_mfcc, self.mel_filters.shape[0]))
            mfccs.append(mfcc)
        
        return torch.array(mfccs)

    @staticmethod
    def _dct_matrix(n_filters: int, n_coefficients: int) -> torch.Tensor:
        """Create DCT-II matrix for MFCC computation"""
        dct_matrix = torch.zeros((n_filters, n_coefficients))
        for i in range(n_filters):
            for j in range(n_coefficients):
                dct_matrix[i, j] = torch.cos(torch.pi * i * (j + 0.5) / n_coefficients)
        return dct_matrix

    def parameters(self):
        return [self.projection, self.cls_token, self.position_embed] + self.transformer.parameters()

    def train(self):
        self.training = True
        self.transformer.training = True

    def eval(self):
        self.training = False
        self.transformer.training = False

if __name__ == "__main__":
    print("\n=== Running Audio Encoder ===\n")
    import torch

    # Load configuration
    config = get_merged_config()

    # Create dummy audio input: (batch_size, in_channels, audio_length)
    batch_size = 2
    audio_cfg = config['audio_encoder']
    in_channels = audio_cfg['in_channels']
    audio_length = audio_cfg['audio_length']
    dummy_audio = torch.randn(batch_size, in_channels, audio_length)

    # Instantiate encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = AudioEncoder(config, device=device).to(device)
    dummy_audio = dummy_audio.to(device)

    # Forward pass
    output = encoder(dummy_audio)

    # Print output shape
    if isinstance(output, tuple):
        output = output[0]
    print("AudioEncoder output shape:", output.shape)
