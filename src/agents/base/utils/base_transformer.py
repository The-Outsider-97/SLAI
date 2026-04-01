import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from src.agents.base.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Base Transformer")
printer = PrettyPrinter


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs (batch‑first).
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaseTransformer(nn.Module):
    """Core transformer module with encoder‑decoder architecture."""
    def __init__(self):
        super().__init__()
        config = load_global_config()
        transform_config = get_config_section('base_transformer')

        # Model dimensions
        self.d_model = transform_config.get('d_model', 512)
        self.nhead = transform_config.get('nhead', 8)
        self.num_encoder_layers = transform_config.get('num_encoder_layers', 6)
        self.num_decoder_layers = transform_config.get('num_decoder_layers', 6)
        self.dim_feedforward = transform_config.get('dim_feedforward', 2048)
        self.dropout = transform_config.get('dropout', 0.1)
        self.activation = transform_config.get('activation', 'relu')
        self.layer_norm_eps = transform_config.get('layer_norm_eps', 1e-5)
        self.batch_first = transform_config.get('batch_first', True)
        self.norm_first = transform_config.get('norm_first', False)

        # Vocabulary sizes
        self.src_vocab_size = config.get('src_vocab_size', 30000)
        self.tgt_vocab_size = config.get('tgt_vocab_size', 30000)

        # Inference defaults (loaded from config)
        inference_config = get_config_section('inference')
        self.default_max_len = inference_config.get('max_len', 50)
        self.default_sos_token = inference_config.get('sos_token', 1)
        self.default_eos_token = inference_config.get('eos_token', 2)
        self.default_temperature = inference_config.get('temperature', 1.0)

        # Embedding layers
        self.src_embed = nn.Embedding(self.src_vocab_size, self.d_model)
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, self.d_model)

        # Positional encoding (shared for encoder and decoder)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=5000, dropout=self.dropout)

        # Core transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=self.batch_first,
            norm_first=self.norm_first
        )

        # Output projection
        self.fc_out = nn.Linear(self.d_model, self.tgt_vocab_size)

        self._init_weights()

        printer.status("INIT", f"BaseTransformer initialized with d_model={self.d_model}", "success")

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode source sequence."""
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        return self.transformer.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence using encoder output."""
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Full encoder‑decoder forward pass."""
        memory = self.encode(src, src_mask, src_key_padding_mask)
        output = self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder attention."""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model state and config to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'src_vocab_size': self.src_vocab_size,
                'tgt_vocab_size': self.tgt_vocab_size,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_encoder_layers': self.num_encoder_layers,
                'num_decoder_layers': self.num_decoder_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'activation': self.activation,
                'layer_norm_eps': self.layer_norm_eps,
                'batch_first': self.batch_first,
                'norm_first': self.norm_first
            }
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'BaseTransformer':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        if device is not None:
            model = model.to(device)
        logger.info(f"Model loaded from {path}")
        return model

    def inference(
        self,
        src: torch.Tensor,
        max_len: Optional[int] = None,
        sos_token: Optional[int] = None,
        eos_token: Optional[int] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate sequence from source.

        Args:
            src: Source tensor (batch, src_len)
            max_len: Maximum generation length (default: self.default_max_len)
            sos_token: Start‑of‑sequence token (default: self.default_sos_token)
            eos_token: End‑of‑sequence token (default: self.default_eos_token)
            temperature: Sampling temperature (default: 1.0)

        Returns:
            Generated token sequence (batch, seq_len)
        """
        max_len = max_len or self.default_max_len
        sos_token = sos_token or self.default_sos_token
        eos_token = eos_token or self.default_eos_token

        self.eval()
        with torch.no_grad():
            memory = self.encode(src)
            batch_size = src.size(0)
            generated = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=src.device)

            for _ in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(src.device)
                output = self.decode(generated, memory, tgt_mask=tgt_mask)
                logits = self.fc_out(output[:, -1:, :]) / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.view(batch_size, -1), 1)

                if (next_token == eos_token).all():
                    break
                generated = torch.cat([generated, next_token], dim=1)

        return generated


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Testing Base Transformer ===\n")
    from torch.utils.data import DataLoader, TensorDataset

    # Initialization
    transformer = BaseTransformer()

    # Dummy data
    src_tensor = torch.randint(1, 100, (2, 10))  # (batch=2, seq=10)
    tgt_tensor = torch.randint(1, 100, (2, 10))

    # Define a dummy dataloader
    dataset = TensorDataset(src_tensor, tgt_tensor)
    dataloader = DataLoader(dataset, batch_size=2)

    # Forward pass
    output = transformer(src_tensor, tgt_tensor)
    print("Output shape:", output.shape)

    # Saving and loading
    transformer.save("test_model.pth")
    # loaded = BaseTransformer.load("test_model.pth")

    # Inference
    src_sample = torch.randint(1, 100, (1, 5))
    generated = transformer.inference(src_sample)
    print("Generated sequence shape:", generated.shape)

    print("=== Successfully Ran Base Transformer ===")
