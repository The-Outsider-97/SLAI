import math
import json, yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding
from typing import List, Tuple, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Text Encoder")
printer = PrettyPrinter

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._init_components()
        self.transformer = Transformer()
        self.transformer.return_hidden = True 
        logger.info(f"TextEncoder initialized: "
                   f"vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, "
                   f"num_layers={self.num_layers}, num_heads={self.num_heads}")

    def _init_configs(self):
        """Load and validate configurations from global and section-specific settings"""
        self.config = load_global_config()
        self.training = self.config.get('training')

        self.text_config = get_config_section('text_encoder')
        self.max_gen_length = self.text_config.get('max_gen_length')
        
        # Core parameters
        self.vocab_size = self.config['vocab_size']
        self.embed_dim = self.config['embed_dim']
        self.num_layers = self.config['num_layers']
        self.num_heads = self.config['num_heads']
        self.ff_dim = self.config['ff_dim']
        self.num_styles = self.config['num_styles']
        self.max_position_embeddings = self.config['max_position_embeddings']
        self.positional_encoding = self.config['positional_encoding']
        self.dropout_rate = self.config['dropout_rate']
        self.device = self.config['device']
        self.initializer = self.config['initializer']
        
        # Text-specific parameters
        self.transformer_config = self.text_config.get('transformer', {})
        self.return_hidden = self.transformer_config.get('return_hidden', False)

        self.attention_maps = []
        
        # Validate critical parameters
        if not self.embed_dim:
            raise ValueError("embed_dim must be specified in config")
        if not self.vocab_size:
            raise ValueError("vocab_size must be specified in config")

    def _init_components(self):
        """Initialize all subcomponents of the encoder"""
        self.style_embeddings = nn.Embedding(self.num_styles, self.embed_dim).to(self.device)

        # Token embeddings
        self.token_embeddings = self._init_embeddings()
        
        # Positional encoding
        self.position_embeddings = self._init_positional_encoding()
        
        # Transformer backbone
        self.transformer = Transformer()
        self.transformer.return_hidden = self.return_hidden

    def _init_embeddings(self) -> Parameter:
        """Initialize token embeddings with proper initialization scheme"""
        printer.status("TEXT", "Initializing token embeddings", "info")

        embeddings = nn.Parameter(torch.empty(self.vocab_size, self.embed_dim, device=self.device))
        
        if self.initializer == "xavier_uniform":
            nn.init.xavier_uniform_(embeddings)
        elif self.initializer == "he_normal":
            nn.init.kaiming_normal_(embeddings)
        else:
            nn.init.normal_(embeddings, mean=0.0, std=0.02)
        
        return Parameter(embeddings, requires_grad=True)

    def _init_positional_encoding(self) -> Parameter:
        """Initialize positional embeddings based on configuration"""
        printer.status("TEXT", "Initializing positional embeddings", "info")

        if self.positional_encoding == "rotary":
            self.rotary_emb = RotaryEmbedding(dim=self.embed_dim // self.num_heads)
            return None

        if self.positional_encoding == "sinusoidal":
            pe = self._sinusoidal_encoding()
            return Parameter(pe, requires_grad=False)
        else:  # learned
            pe = torch.randn(1, self.max_position_embeddings, self.embed_dim, 
                           device=self.device) * 0.02
            return Parameter(pe, requires_grad=True)

    def _sinusoidal_encoding(self) -> torch.Tensor:
        """Generate sinusoidal positional encodings"""
        printer.status("TEXT", "Generating sinusoidal positional encodings", "info")

        pe = torch.zeros(1, self.max_position_embeddings, self.embed_dim, device=self.device)
        position = torch.arange(0, self.max_position_embeddings, 
                             dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() *
            (-math.log(10000.0) / self.embed_dim
        ))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, input_ids: torch.Tensor, attention_mask=None,
        style_id: Optional[torch.Tensor] = None, output_type: str = "sequence") -> torch.Tensor:
        """
        Forward pass for text encoding
        Args:
            input_ids: Token indices tensor [batch_size, seq_len]
            style_id: Optional style conditioning tensor [batch_size]
        Returns:
            Contextual embeddings [batch_size, seq_len, embed_dim]
        """
        printer.status("TEXT", "Forward pass", "info")

        batch_size, seq_len = input_ids.shape
        
        # Truncate sequences longer than model supports
        if seq_len > self.max_position_embeddings:
            input_ids = input_ids[:, :self.max_position_embeddings]
            seq_len = self.max_position_embeddings
            logger.warning(f"Input sequence truncated to {seq_len} tokens")

        # Embed tokens [batch_size, seq_len] => [batch_size, seq_len, embed_dim]
        token_embeds = self.token_embeddings[input_ids]
        
        # Add positional embeddings
        position_embeds = self.position_embeddings[:, :seq_len, :]
        embeddings = token_embeds + position_embeds
        
        # Apply input dropout during training
        if self.training and self.dropout_rate > 0:
            embeddings = TensorOps.dropout(embeddings)

        # Handle style embeddings
        if style_id is not None:
            if not isinstance(style_id, torch.Tensor):
                style_id = torch.tensor(style_id, device=input_ids.device)
            style_emb = self.style_embeddings(style_id).unsqueeze(1)
            style_emb = style_emb.expand(-1, seq_len, -1)
            embeddings = embeddings + style_emb

        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            extended_mask = (1.0 - extended_mask) * -1e4
        else:
            extended_mask = None
        
        # Process through transformer
        encoded = self.transformer(embeddings, style_id=style_id)
        if output_type == "cls":
            return encoded[:, 0]  # [batch_size, embed_dim]
        elif output_type == "mean":
            return encoded.mean(dim=1)  # [batch_size, embed_dim]
        
        return encoded  # [batch_size, seq_len, embed_dim]

    def load_pretrained_embeddings(self, weights: torch.Tensor):
        """Load pretrained embeddings while respecting vocab size"""
        printer.status("TEXT", "Loading pretrained embeddings", "info")

        if weights.shape[0] != self.vocab_size:
            logger.warning(f"Pretrained vocab size {weights.shape[0]} "
                          f"doesn't match model vocab {self.vocab_size}")
        
        min_size = min(self.vocab_size, weights.shape[0])
        self.token_embeddings.data[:min_size] = weights[:min_size].to(self.device)

    def _register_hooks(self):
        """Attach forward hooks to all self-attention layers to capture attention weights."""
        printer.status("TEXT", "Registering attention hooks", "info")
    
        self.attention_maps = {}
    
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Assume output is (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) == 2:
                    attn_weights = output[1]
                    self.attention_maps[layer_idx] = attn_weights.detach()
            return hook_fn
    
        for idx, layer in enumerate(self.transformer.encoder_layers):
            layer.self_attn.register_forward_hook(make_hook(idx))

if __name__ == "__main__":
    print("\n=== Testing Text Encoder ===\n")
    
    # Override config for testing
    config = load_global_config()
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize encoder
    encoder = TextEncoder().to(config['device'])
    
    # Create test input
    input_ids = torch.randint(0, config['vocab_size'], (4, 128)).to(config['device'])
    style_ids = torch.randint(0, config['num_styles'], (4,)).to(config['device'])
    
    # Test forward pass
    print("Testing forward pass...")
    output = encoder(input_ids, style_id=style_ids)
    print(f"Output shape: {output.shape}")
    
    # Test sequence truncation
    print("\nTesting sequence truncation...")
    long_input = torch.randint(0, config['vocab_size'], (2, 6000)).to(config['device'])
    output_long = encoder(long_input)
    print(f"Truncated output shape: {output_long.shape}")
    
    # Test pretrained loading
    print("\nTesting pretrained embedding loading...")
    dummy_weights = torch.randn(50000, config['embed_dim'])
    encoder.load_pretrained_embeddings(dummy_weights)
    
    print("\n=== TextEncoder tests passed ===\n")
