import math
import numpy as np
from typing import Dict, Any

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer  

class AudioEncoder:
    def __init__(self,
                 audio_length=16000,
                 patch_size=400,
                 embed_dim=100,
                 in_channels=1,
                 num_layers=6,
                 dropout_rate=0.1,
                 positional_encoding="learned",
                 encoder_type: str = "transformer",
                 mfcc_config: Dict[str, Any] = None):
        self.encoder_type = encoder_type
        if self.encoder_type == "transformer":
            self.patch_size = patch_size
            self.num_patches = audio_length // patch_size
            #self.in_channels = 1  # Mono audio input
            self.in_channels = in_channels
            self.embed_dim = embed_dim
            self.dropout_rate = dropout_rate
            self.training = True
            
            # Convolutional projection initialization (1D equivalent)
            self.projection = Parameter(
                TensorOps.he_init((patch_size * in_channels, embed_dim), patch_size * in_channels))

            # Positional embeddings
            self.positional_encoding = positional_encoding
            if self.positional_encoding == "learned":
                self.position_embed = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)
            elif self.positional_encoding == "sinusoidal":
                self.position_embed = self._init_sinusoidal_encoding(max_len=5000)

            self.cls_token = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)
            self.transformer = Transformer(num_layers=num_layers, embed_dim=embed_dim)        
            self._cache = {}

        elif self.encoder_type == "mfcc":
            # MFCC-specific initialization
            mfcc_config = mfcc_config or {}
            self.sample_rate = mfcc_config.get('sample_rate', 16000)
            self.n_mfcc = mfcc_config.get('n_mfcc', 13)
            self.frame_length = int(mfcc_config.get('frame_length_ms', 25) * self.sample_rate // 1000)
            self.frame_step = int(mfcc_config.get('frame_step_ms', 10) * self.sample_rate // 1000)
            self.mel_filters = self._create_mel_filterbank(mfcc_config)

        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

    def _init_sinusoidal_encoding(self, max_len=5000):
        pe = np.zeros((max_len, self.embed_dim))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return Parameter(pe[np.newaxis, :, :])

    def _create_mel_filterbank(self, config: Dict[str, Any]) -> np.ndarray:
        """Create Mel-scale filter bank"""
        n_filters = config.get('n_filters', 40)
        low_freq = config.get('low_freq', 0)
        high_freq = config.get('high_freq', self.sample_rate//2)
        
        # Convert frequencies to Mel scale
        low_mel = 2595 * np.log10(1 + low_freq/700)
        high_mel = 2595 * np.log10(1 + high_freq/700)
        
        # Create filter points
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_points = 700 * (10**(mel_points/2595) - 1)
        bin_points = np.floor((self.frame_length + 1) * hz_points / self.sample_rate)
        
        # Create filter banks
        filters = np.zeros((n_filters, self.frame_length//2 + 1))
        for i in range(1, n_filters + 1):
            left = bin_points[i-1]
            center = bin_points[i]
            right = bin_points[i+1]
            
            for j in range(int(left), int(center)):
                filters[i-1, j] = (j - left) / (center - left)
            for j in range(int(center), int(right)):
                filters[i-1, j] = (right - j) / (right - center)
        
        return filters

    def extract_patches(self, x):
        """Convert waveform to patched representation with padding"""
        if x.ndim == 2:
            x = x[:, np.newaxis, :]  # Add channel dim: (B, C, L)
        batch, channels, length = x.shape
 
        # Pad if necessary
        remainder = length % self.patch_size
        if remainder != 0:
            pad_size = self.patch_size - remainder
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_size)))

        # Reshape into non-overlapping patches
        num_patches = x.shape[2] // self.patch_size
        x = x.reshape(batch, channels, num_patches, self.patch_size)
        return x.transpose(0, 2, 1, 3).reshape(batch, num_patches, -1)

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

    def _mfcc_forward(self, waveform: np.ndarray) -> np.ndarray:
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
            emphasized = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
            
            # Windowing
            window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(L) / (L - 1))
            windowed = emphasized * window
            
            # Power spectrum
            spectrum = np.abs(np.fft.rfft(windowed)) ** 2
            
            # Mel filterbank
            filter_energies = np.dot(spectrum, self.mel_filters.T)
            
            # Log compression
            log_energies = np.log(filter_energies + 1e-6)
            
            # DCT
            mfcc = np.dot(log_energies, self._dct_matrix(self.n_mfcc, self.mel_filters.shape[0]))
            mfccs.append(mfcc)
        
        return np.array(mfccs)

    @staticmethod
    def _dct_matrix(n_filters: int, n_coefficients: int) -> np.ndarray:
        """Create DCT-II matrix for MFCC computation"""
        dct_matrix = np.zeros((n_filters, n_coefficients))
        for i in range(n_filters):
            for j in range(n_coefficients):
                dct_matrix[i, j] = np.cos(np.pi * i * (j + 0.5) / n_coefficients)
        return dct_matrix

    def forward(self, x, style_id=0):
        if self.encoder_type == "transformer":
            # Original transformer processing
            x = self.extract_patches(x)
            self._cache['input_shape'] = x.shape
            
            x = np.matmul(x, self.projection.data)
            
            if self.training and self.dropout_rate > 0:
                mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
                x *= mask
            
            cls_tokens = np.tile(self.cls_token.data, (x.shape[0], 1, 1))
            x = np.concatenate((cls_tokens, x), axis=1)
            
            if self.positional_encoding == "sinusoidal":
                seq_len = x.shape[1]
                x += self.position_embed.data[:, :seq_len, :]
            else:
                x += self.position_embed.data[:, :x.shape[1]]
            
            x = self.transformer.forward(x, style_id)
            self._cache['pre_projection'] = x
            return x
            
        elif self.encoder_type == "mfcc":
            # Process with MFCC pipeline
            return self._mfcc_forward(x)
            
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def backward(self, dout):
        """Backprop through encoder"""
        d_x = self.transformer.backward(dout)
        d_x = d_x[:, 1:, :]  # Remove CLS token
        
        # Gradient for projection
        d_proj = np.matmul(
            self._cache['input_shape'].transpose(0, 2, 1), 
            d_x.reshape(-1, self.embed_dim))
        self.projection.grad += d_proj.sum(axis=0)
        
        return np.matmul(d_x, self.projection.data.T)

    def parameters(self):
        return [self.projection, self.cls_token, self.position_embed] + self.transformer.parameters()

    def train(self):
        self.training = True
        self.transformer.training = True

    def eval(self):
        self.training = False
        self.transformer.training = False
