import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Base Transformer")
printer = Prettyprinter

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
  
        self.config = load_global_config()
        self.pe_config = get_config_section('positional_encoding')

        self.max_len = self.pe_config.get('max_len', 5000)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BaseTransformer(nn.Module):
    """Core transformer module with encoder-decoder architecture"""
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.be_config = get_config_section('base_transformer')

        self.src_vocab_size = self.pe_config.get('src_vocab_size')
        self.tgt_vocab_size = self.pe_config.get('tgt_vocab_size')
        self.d_model = self.pe_config.get('d_model', '512')
        self.nhead = self.pe_config.get('nhead', '8')
        self.num_encoder_layers = self.pe_config.get('num_encoder_layers', '6')
        self.num_decoder_layers = self.pe_config.get('num_decoder_layers', '6')
        self.dim_feedforward = self.pe_config.get('dim_feedforward', '2048')
        self.dropout = self.pe_config.get('dropout', '0.1')
        self.activation = self.pe_config.get('activation', 'relu')
        self.layer_norm_eps = self.pe_config.get('layer_norm_eps', '0.00001')
        self.batch_first = self.pe_config.get('batch_first', True)
                   
        # Embedding layers
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        # Core transformer module
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=True  # Pre-LayerNorm configuration
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

        # Initialize parameters
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self,
               src: torch.Tensor,
               src_mask: torch.Tensor = None,
               src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode source sequence
        
        Args:
            src: Input sequence tensor (S, N, E) or (N, S, E) if batch_first
            src_mask: Source sequence mask
            src_key_padding_mask: Source padding mask
            
        Returns:
            Encoded memory tensor
        """
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

    def decode(self,
               tgt: torch.Tensor,
               memory: torch.Tensor,
               tgt_mask: torch.Tensor = None,
               memory_mask: torch.Tensor = None,
               tgt_key_padding_mask: torch.Tensor = None,
               memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Decode target sequence using encoder output
        
        Args:
            tgt: Target sequence tensor
            memory: Encoder output from encode()
            tgt_mask: Target sequence mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
            
        Returns:
            Decoded output tensor
        """
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Full encoder-decoder forward pass
        
        Args:
            src: Source sequence (N, S) if batch_first
            tgt: Target sequence (N, T) if batch_first
            ... (masks as described in encode/decode)
            
        Returns:
            Output logits tensor (N, T, tgt_vocab_size)
        """
        memory = self.encode(src, src_mask, src_key_padding_mask)
        output = self.decode(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask
        )
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder attention"""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def parameter_count(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save model state to file"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'src_vocab_size': self.src_embed.num_embeddings,
                'tgt_vocab_size': self.tgt_embed.num_embeddings,
                'd_model': self.d_model,
                'nhead': self.transformer.nhead,
                'num_encoder_layers': self.transformer.num_encoder_layers,
                'num_decoder_layers': self.transformer.num_decoder_layers,
                'dim_feedforward': self.transformer.dim_feedforward,
                'dropout': self.transformer.dropout,
                'activation': self.transformer.activation,
                'layer_norm_eps': self.transformer.layer_norm_eps
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'BaseTransformer':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device)

    def configure_optimizer(self,
                            lr: float = 1e-4,
                            betas: tuple = (0.9, 0.98),
                            eps: float = 1e-9,
                            weight_decay: float = 1e-4) -> torch.optim.Adam:
        """Return Adam optimizer with recommended transformer settings"""
        return torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

    def training_step(self,
                      batch: tuple,
                      optimizer: torch.optim.Optimizer,
                      criterion: nn.Module,
                      clip_grad: float = 1.0) -> float:
        """Single training step with gradient clipping
        
        Args:
            batch: (src, tgt) tensors
            optimizer: Pre-configured optimizer
            criterion: Loss function
            clip_grad: Gradient clipping value
            
        Returns:
            Loss value
        """
        src, tgt = batch
        src, tgt_in = src[:, :-1], tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_in.size(1)).to(src.device)

        # Forward pass
        outputs = self(src, tgt_in, tgt_mask=tgt_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_out.contiguous().view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
        optimizer.step()

        return loss.item()

    def inference(self,
                  src: torch.Tensor,
                  max_len: int = 50,
                  sos_token: int = 1,
                  eos_token: int = 2,
                  temperature: float = 1.0) -> torch.Tensor:
        """Generate sequence from source
        
        Args:
            src: Source tensor (1, S)
            max_len: Maximum generation length
            sos_token: Start-of-sequence token
            eos_token: End-of-sequence token
            temperature: Sampling temperature
            
        Returns:
            Generated token sequence (1, L)
        """
        self.eval()
        with torch.no_grad():
            memory = self.encode(src)
            generated = torch.ones(1, 1).fill_(sos_token).long().to(src.device)
            
            for _ in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(src.device)
                output = self.decode(generated, memory, tgt_mask=tgt_mask)
                logits = self.fc_out(output[:, -1:, :]) / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.view(-1), 1)
                
                if next_token.item() == eos_token:
                    break
                    
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                
        return generated
