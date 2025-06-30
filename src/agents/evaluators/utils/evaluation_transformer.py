
import math
import torch
import torch.nn as nn

from src.agents.evaluators.utils.config_loader import load_global_config
from src.agents.base.utils.base_transformer import BaseTransformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Evaluation Transformer")
printer = PrettyPrinter

class EvaluationTransformer(BaseTransformer, nn.Module):
    """Specialized transformer for evaluation tasks with flexible input dimensions"""
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 512, **kwargs):
        BaseTransformer.__init__(self)
        nn.Module.__init__(self)
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(d_model, input_dim)
        
        # Precompute positional encoding
        self.register_buffer('pe', self._generate_positional_encoding(d_model, max_len=seq_len))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_positional_encoding(self, d_model, max_len=5000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection: (B, S, D_in) -> (B, S, D_model)
        x = self.input_proj(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        
        # Encoder processing
        memory = self.encoder(x)
        return self.output_proj(memory)