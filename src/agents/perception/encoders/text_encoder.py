import math
import torch
import json, yaml
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, List
from rotary_embedding_torch import RotaryEmbedding
from pathlib import Path

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import TensorOps, Parameter
from ..modules.transformer import Transformer
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Text Encoder")
printer = PrettyPrinter

class TextEncoder(nn.Module):
    """
    Text encoder with token embeddings, positional encoding, and a transformer backbone.
    Supports style conditioning and various output types (sequence, CLS, mean).
    """
    def __init__(self):
        super().__init__()
        self._init_configs()
        self._validate_configs()
        self._init_components()
        logger.info(f"TextEncoder initialized: vocab_size={self.vocab_size}, "
                    f"embed_dim={self.embed_dim}, num_layers={self.num_layers}")

    def _init_configs(self):
        """Load all configurations from global and section."""
        self.config = load_global_config()
        self.text_config = get_config_section('text_encoder')

        # Core parameters
        self.vocab_size = self.config.get('vocab_size')
        self.embed_dim = self.config.get('embed_dim')
        self.num_layers = self.config.get('num_layers')
        self.num_heads = self.config.get('num_heads')
        self.ff_dim = self.config.get('ff_dim')
        self.num_styles = self.config.get('num_styles')
        self.max_position_embeddings = self.config.get('max_position_embeddings', 5000)
        self.positional_encoding = self.config.get('positional_encoding', 'sinusoidal')
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        self.device = self.config.get('device', 'cpu')
        self.initializer = self.config.get('initializer', 'xavier_uniform')
        self.training = self.config.get('training', True)

        # Text‑specific parameters
        self.max_gen_length = self.text_config.get('max_gen_length', 64)
        self.return_hidden = self.text_config.get('return_hidden', False)

        # Cache for attention maps (if hooks are registered)
        self.attention_maps = {}

    def _validate_configs(self):
        """Validate critical parameters to avoid runtime errors."""
        if not self.embed_dim:
            raise ValueError("embed_dim must be specified in config")
        if not self.vocab_size:
            raise ValueError("vocab_size must be specified in config")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")

    def _init_components(self):
        """Initialize all subcomponents."""
        # Token embeddings
        self.token_embeddings = self._init_embeddings()

        # Positional encoding
        self.position_embeddings, self.rotary_emb = self._init_positional_encoding()

        # Style embeddings (optional, but kept for compatibility)
        self.style_embeddings = nn.Embedding(self.num_styles, self.embed_dim)

        # Transformer backbone
        self.transformer = Transformer()
        self.transformer.return_hidden = True   # <-- always get full sequence
        self.transformer.output_attentions = False  # will be set by register_attention_hooks

    def _init_embeddings(self) -> Parameter:
        """Initialize token embeddings with the configured initializer."""
        embeddings = torch.empty(self.vocab_size, self.embed_dim, device=self.device)

        if self.initializer == "xavier_uniform":
            nn.init.xavier_uniform_(embeddings)
        elif self.initializer == "he_normal":
            nn.init.kaiming_normal_(embeddings, nonlinearity='relu')
        else:
            nn.init.normal_(embeddings, mean=0.0, std=0.02)

        return Parameter(embeddings, requires_grad=True)

    def _init_positional_encoding(self):
        """Return positional embeddings (or None) and optional rotary embedding."""
        if self.positional_encoding == "rotary":
            rotary_emb = RotaryEmbedding(dim=self.embed_dim // self.num_heads)
            return None, rotary_emb

        if self.positional_encoding == "sinusoidal":
            pe = self._sinusoidal_encoding()
            return Parameter(pe, requires_grad=False), None
        else:  # learned
            pe = torch.randn(1, self.max_position_embeddings, self.embed_dim, device=self.device) * 0.02
            return Parameter(pe, requires_grad=True), None

    def _sinusoidal_encoding(self) -> torch.Tensor:
        """Generate sinusoidal positional encodings."""
        pe = torch.zeros(1, self.max_position_embeddings, self.embed_dim, device=self.device)
        position = torch.arange(0, self.max_position_embeddings, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=self.device).float() *
            (-math.log(10000.0) / self.embed_dim)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None,
        output_type: str = "sequence"
    ) -> torch.Tensor:
        """
        Forward pass for text encoding.

        Args:
            input_ids: Token indices (batch, seq_len)
            attention_mask: Mask for padding (1 for real, 0 for pad)
            style_id: Optional style IDs (batch,)
            output_type: One of "sequence", "cls", or "mean"

        Returns:
            If output_type == "sequence": (batch, seq_len, embed_dim)
            If output_type == "cls": (batch, embed_dim) – CLS token output
            If output_type == "mean": (batch, embed_dim) – mean of sequence
        """
        batch_size, seq_len = input_ids.shape

        # Truncate long sequences
        if seq_len > self.max_position_embeddings:
            input_ids = input_ids[:, :self.max_position_embeddings]
            seq_len = self.max_position_embeddings
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_position_embeddings]
            logger.warning(f"Input sequence truncated to {seq_len} tokens")

        # Token embeddings
        embeddings = self.token_embeddings[input_ids]  # (B, L, D)

        # Add positional embeddings (if not using rotary)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings[:, :seq_len, :]

        # Apply rotary embeddings (if used) – this is applied inside attention layers,
        # so we don't need to modify embeddings here. The transformer will handle it.

        # Input dropout
        if self.training and self.dropout_rate > 0:
            embeddings = F.dropout(embeddings, p=self.dropout_rate, training=self.training)

        # Style conditioning (add to embeddings)
        if style_id is not None:
            if not isinstance(style_id, torch.Tensor):
                style_id = torch.tensor(style_id, device=input_ids.device)
            style_emb = self.style_embeddings(style_id).unsqueeze(1)  # (B, 1, D)
            embeddings = embeddings + style_emb

        # Prepare mask for transformer (if provided)
        # The transformer expects context_mask or attention_mask. We'll pass as attention_mask.
        transformer_mask = None
        if attention_mask is not None:
            # Ensure attention_mask is (batch, seq_len) and values are 0/1
            transformer_mask = attention_mask

        # Process through transformer
        encoded = self.transformer(
            embeddings,
            style_id=style_id,
            attention_mask=transformer_mask
        )

        # Output selection
        if output_type == "cls":
            return encoded[:, 0, :]  # (B, D)
        elif output_type == "mean":
            return encoded.mean(dim=1)  # (B, D)
        else:  # sequence
            return encoded  # (B, L, D)

    def load_pretrained_embeddings(self, weights: torch.Tensor):
        """Load pretrained token embeddings, matching vocab size."""
        if weights.shape[0] != self.vocab_size:
            logger.warning(f"Pretrained vocab size {weights.shape[0]} != model vocab {self.vocab_size}; "
                           "will load common part only.")
        min_size = min(self.vocab_size, weights.shape[0])
        with torch.no_grad():
            self.token_embeddings.data[:min_size] = weights[:min_size].to(self.device)
        logger.info(f"Loaded {min_size} pretrained embeddings.")

    def register_attention_hooks(self):
        """
        Register forward hooks on all attention layers to capture attention weights.
        After calling this, attention weights for each layer will be stored in
        self.attention_maps[layer_idx] after each forward pass.
        """
        # Enable attention output in the transformer
        self.transformer.output_attentions = True
        self.transformer.return_hidden = True
        
        # Clear previous maps
        self.attention_maps = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.attention_maps[layer_idx] = attn_weights.detach().cpu()
                    else:
                        # Some attention types (e.g., EfficientAttention) don't produce weights
                        logger.debug(f"Attention layer {layer_idx} returned None weights (expected for some attention types)")
                else:
                    logger.warning(f"Attention layer {layer_idx} did not return a tuple of (output, weights)")
            return hook
        
        # Iterate over transformer layers and attach hooks
        for idx, layer in enumerate(self.transformer.layers):
            if 'attention' in layer:
                attn_module = layer['attention']
                # Ensure the attention module has output_attentions flag
                attn_module.output_attentions = True
                attn_module.register_forward_hook(hook_fn(idx))
                logger.info(f"Registered attention hook for layer {idx}")
            else:
                logger.warning(f"Layer {idx} has no 'attention' key")

    def freeze_feature_extractor(self):
        """Freeze all parameters except the task head (if any)."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Feature extractor frozen")

    def unfreeze_feature_extractor(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Feature extractor unfrozen")

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
    print("\n=== Testing Text Encoder ===\n")

    # Create test inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_global_config()
    config['device'] = str(device)

    # Initialize encoder
    encoder = TextEncoder().to(device)

    # Test forward pass
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)
    style_ids = torch.randint(0, config['num_styles'], (batch_size,)).to(device)

    encoder.register_attention_hooks()
    output = encoder(input_ids, style_id=style_ids)
    # Now encoder.attention_maps contains attention weights for each layer
    for layer_idx, attn_weights in encoder.attention_maps.items():
        print(f"Layer {layer_idx} attention shape: {attn_weights.shape}")

    print("Testing forward pass (sequence)...")
    output = encoder(input_ids, style_id=style_ids)
    print(f"Output shape (sequence): {output.shape}")

    print("Testing forward pass (CLS)...")
    cls_output = encoder(input_ids, style_id=style_ids, output_type="cls")
    print(f"Output shape (CLS): {cls_output.shape}")

    print("Testing forward pass (mean)...")
    mean_output = encoder(input_ids, style_id=style_ids, output_type="mean")
    print(f"Output shape (mean): {mean_output.shape}")

    # Test attention mask
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    attention_mask[:, seq_len//2:] = 0  # mask second half
    output_masked = encoder(input_ids, attention_mask=attention_mask, style_id=style_ids)
    print(f"Output shape with mask: {output_masked.shape}")

    # Test truncation
    print("\nTesting sequence truncation...")
    long_input = torch.randint(0, config['vocab_size'], (2, 6000)).to(device)
    truncated_output = encoder(long_input)
    print(f"Truncated output shape: {truncated_output.shape}")

    # Test pretrained embedding loading
    print("\nTesting pretrained embedding loading...")
    dummy_weights = torch.randn(config['vocab_size'], config['embed_dim'])
    encoder.load_pretrained_embeddings(dummy_weights)

    print("\n=== TextEncoder tests passed ===")