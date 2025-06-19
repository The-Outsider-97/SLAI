import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Text Decoder")
printer = PrettyPrinter

class TextDecoder(nn.Module):
    def __init__(self, encoder: Optional[nn.Module] = None):
        super().__init__()
        self._init_configs()
        self._init_components(encoder)
        logger.info(f"TextDecoder initialized: "
                   f"vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, "
                   f"num_layers={self.num_layers}, num_heads={self.num_heads}")

    def _init_configs(self):
        """Load and validate configurations from global settings"""
        self.config = load_global_config()
        
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
        self.beam_size = self.config.get('beam_size', 5)
        
        # Decoder-specific parameters
        self.tie_embeddings = self.config.get('tie_embeddings', True)
        self.layer_norm_eps = self.config.get('layer_norm_eps', 1e-5)
        self.temperature = self.config.get('temperature', 1.0)
        self.top_k = self.config.get('top_k', 0)
        self.top_p = self.config.get('top_p', 0.0)
        self.repetition_penalty = self.config.get('repetition_penalty', 1.2)

    def _init_components(self, encoder: Optional[nn.Module] = None):
        """Initialize all subcomponents of the decoder"""
        # Token embeddings (shared with encoder if specified)
        if encoder and self.tie_embeddings:
            self.token_embeddings = encoder.token_embeddings
            logger.info("Sharing token embeddings with encoder")
        else:
            self.token_embeddings = self._init_embeddings()

        # Positional encoding
        self.position_embeddings = self._init_positional_encoding()

        # Style embeddings
        self.style_embeddings = nn.Embedding(self.num_styles, self.embed_dim).to(self.device)

        # Transformer backbone
        self.transformer = self._init_transformer()

        # Output projection
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size)
        if self.tie_embeddings:
            self.output_proj.weight = self.token_embeddings  # Remove .data
        else:
            # Wrap non-shared embeddings in nn.Parameter
            self.output_proj.weight = nn.Parameter(self.token_embeddings.data.clone())

    def _init_embeddings(self) -> Parameter:
        """Initialize token embeddings with proper initialization scheme"""
        printer.status("DECODER", "Initializing token embeddings", "info")
        embeddings = torch.empty(self.vocab_size, self.embed_dim, device=self.device)
        
        if self.initializer == "xavier_uniform":
            nn.init.xavier_uniform_(embeddings)
        elif self.initializer == "he_normal":
            nn.init.kaiming_normal_(embeddings)
        else:
            nn.init.normal_(embeddings, mean=0.0, std=0.02)
        
        return Parameter(embeddings, requires_grad=True)

    def _init_positional_encoding(self) -> Parameter:
        """Initialize positional embeddings based on configuration"""
        printer.status("DECODER", "Initializing positional embeddings", "info")
        if self.positional_encoding == "sinusoidal":
            pe = self._sinusoidal_encoding()
            return Parameter(pe, requires_grad=False)
        else:  # learned
            pe = torch.randn(1, self.max_position_embeddings, self.embed_dim, 
                           device=self.device) * 0.02
            return Parameter(pe, requires_grad=True)

    def _init_transformer(self) -> nn.Module:
        """Initialize transformer with decoder-specific configuration"""
        printer.status("DECODER", "Initializing transformer", "info")
        # Create a copy of the transformer config
        transformer_config = self.config.copy()
        
        # Modify for decoder-specific behavior
        transformer_config['causal'] = True  # Enable causal masking
        transformer_config['return_hidden'] = True  # Always return logits
        
        # Initialize transformer
        transformer = Transformer()
        transformer.return_hidden = True
        return transformer

    def _sinusoidal_encoding(self) -> torch.Tensor:
        """Generate sinusoidal positional encodings"""
        printer.status("DECODER", "Generating sinusoidal positional encodings", "info")
        pe = torch.zeros(1, self.max_position_embeddings, self.embed_dim, device=self.device)
        position = torch.arange(0, self.max_position_embeddings, 
                             dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() *
            (-math.log(10000.0) / self.embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, 
                input_ids: torch.Tensor, 
                memory: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                style_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for text decoding
        Args:
            input_ids: Token indices tensor [batch_size, tgt_seq_len]
            memory: Encoder output [batch_size, src_seq_len, embed_dim]
            attention_mask: Attention mask for memory
            style_id: Optional style conditioning tensor [batch_size]
        Returns:
            Logits [batch_size, tgt_seq_len, vocab_size]
        """
        printer.status("DECODER", "Forward pass", "info")
        batch_size, tgt_seq_len = input_ids.shape
        src_seq_len = memory.size(1)
        
        # Truncate sequences longer than model supports
        if tgt_seq_len > self.max_position_embeddings:
            input_ids = input_ids[:, :self.max_position_embeddings]
            tgt_seq_len = self.max_position_embeddings
            logger.warning(f"Target sequence truncated to {tgt_seq_len} tokens")
        
        # Embed tokens [batch_size, tgt_seq_len] => [batch_size, tgt_seq_len, embed_dim]
        token_embeds = self.token_embeddings[input_ids]
        
        # Add positional embeddings
        position_embeds = self.position_embeddings[:, :tgt_seq_len, :]
        embeddings = token_embeds + position_embeds
        
        # Apply input dropout during training
        if self.training and self.dropout_rate > 0:
            embeddings = TensorOps.dropout(embeddings)
        
        # Add style embeddings if provided
        if style_id is not None:
            style_emb = self.style_embeddings(style_id).unsqueeze(1)  # [B, 1, D]
            style_emb = style_emb.expand(-1, tgt_seq_len, -1)         # [B, tgt_seq_len, D]
            embeddings = embeddings + style_emb
        
        # Prepare memory mask
        memory_mask = None
        if attention_mask is not None:
            # Create [batch_size, tgt_seq_len, src_seq_len] mask
            memory_mask = attention_mask.unsqueeze(1).expand(-1, tgt_seq_len, -1)
        
        # Process through transformer
        decoded = self.transformer(
            x=embeddings, 
            context=memory, 
            context_mask=memory_mask,
            style_id=style_id
        )
        print("Transformer output shape:", decoded.shape)  # Should be [4, 32, 512]
        
        # Project to vocabulary
        logits = self.output_proj(decoded)
        return logits

    def inference(self,
                  memory: torch.Tensor,
                  attention_mask: Optional[torch.Tensor] = None,
                  style_id: Optional[torch.Tensor] = None,
                  strategy: str = "greedy") -> torch.Tensor:
        """
        Autoregressive generation of text
        Args:
            memory: Encoder output [batch_size, src_seq_len, embed_dim]
            attention_mask: Attention mask for memory
            style_id: Optional style conditioning tensor [batch_size]
            strategy: Generation strategy ("greedy", "sampling", "beam")
        Returns:
            Generated token IDs [batch_size, gen_seq_len]
        """
        printer.status("DECODER", f"Generating text with strategy: {strategy}", "info")
        batch_size = memory.size(0)
        device = memory.device
        finished_mask = None
        
        # Initialize with SOS token (assuming token 1 is SOS)
        input_ids = torch.ones(batch_size, 1, dtype=torch.long, device=device) * 1
        
        # Generation loop
        for _ in range(self.max_gen_length):
            # Get logits for current sequence
            logits = self.forward(input_ids, memory, attention_mask, style_id)
            
            # Get next token logits (last in sequence)
            next_logits = logits[:, -1, :] / self.temperature
            
            # Apply repetition penalty
            if self.repetition_penalty != 1.0:
                for idx in range(batch_size):
                    for token_id in input_ids[idx]:
                        if next_logits[idx, token_id] > 0:
                            next_logits[idx, token_id] /= self.repetition_penalty
                        else:
                            next_logits[idx, token_id] *= self.repetition_penalty
            
            # Apply top-k/top-p filtering
            if self.top_k > 0:
                next_logits = self._top_k_filtering(next_logits)
            if self.top_p > 0:
                next_logits = self._top_p_filtering(next_logits)
            
            # Convert to probabilities
            probs = F.softmax(next_logits, dim=-1)
            
            # Select next token
            if strategy == "greedy":
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:  # sampling
                next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if all sequences have EOS (assuming token 2 is EOS)
            # Track which sequences have finished
            if finished_mask is None:
                finished_mask = (next_token == 2)
            else:
                finished_mask |= (next_token == 2)
            
            if finished_mask.all():
                break
        
        return input_ids

    def _top_k_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """Filter logits to only the top k options"""
        values, _ = torch.topk(logits, self.top_k)
        min_values = values[:, -1].unsqueeze(1)
        return torch.where(logits < min_values, torch.full_like(logits, -float('inf')), logits)

    def _top_p_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        # Shift indices to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create mask to set filtered logits to -inf
        for idx in range(logits.size(0)):
            indices_to_remove = sorted_indices_to_remove[idx]
            logits[idx, sorted_indices[idx, indices_to_remove]] = -float('inf')
        
        return logits

    def beam_search(self,
                    memory: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    style_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Beam search for text generation
        Args:
            memory: Encoder output [batch_size, src_seq_len, embed_dim]
            attention_mask: Attention mask for memory
            style_id: Optional style conditioning tensor [batch_size]
        Returns:
            Generated token IDs [batch_size, gen_seq_len]
        """
        printer.status("DECODER", "Running beam search", "info")
        if style_id is not None:
            style_id = style_id.unsqueeze(1).expand(-1, self.beam_size).reshape(-1)
        
        batch_size = memory.size(0)
        device = memory.device
        
        # Initialize beams (SOS token)
        beams = torch.ones(batch_size, self.beam_size, 1, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
        
        # Expand memory for beam search
        memory = memory.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
        memory = memory.view(batch_size * self.beam_size, -1, memory.size(-1))
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.beam_size, 1)
            attention_mask = attention_mask.view(batch_size * self.beam_size, -1)
        
        # Generation loop
        for step in range(self.max_gen_length):
            # Flatten batch and beams
            flat_beams = beams.view(batch_size * self.beam_size, -1)
            
            # Get logits for current sequence
            logits = self.forward(flat_beams, memory, attention_mask, style_id)
            next_logits = logits[:, -1, :] / self.temperature
            
            # Convert to scores
            vocab_size = next_logits.size(-1)
            scores = F.log_softmax(next_logits, dim=-1)
            
            # Add to beam scores
            scores = beam_scores.view(-1, 1) + scores
            
            # Reshape to [batch_size, beam_size * vocab_size]
            scores = scores.view(batch_size, self.beam_size * vocab_size)
            
            # Select top-k beams
            top_scores, top_indices = torch.topk(scores, self.beam_size, dim=-1)
            
            # Determine beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update beams
            new_beams = []
            for i in range(batch_size):
                batch_beams = beams[i].index_select(0, beam_indices[i])
                new_beam = torch.cat([batch_beams, token_indices[i].unsqueeze(-1)], dim=-1)
                new_beams.append(new_beam)
            
            beams = torch.stack(new_beams)
            beam_scores = top_scores
            
            # Check for EOS
            if (token_indices == 2).all():
                break
        
        # Select best beam for each batch
        best_beams = beams[:, 0, :]
        return best_beams

if __name__ == "__main__":
    print("\n=== Testing Text Decoder ===\n")
    
    # Override config for testing
    config = load_global_config()
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize encoder and decoder
    from src.agents.perception.encoders.text_encoder import TextEncoder
    encoder = TextEncoder().to(config['device'])
    decoder = TextDecoder(encoder).to(config['device'])
    
    # Create test inputs
    input_ids = torch.randint(0, config['vocab_size'], (4, 128)).to(config['device'])
    style_ids = torch.randint(0, config['num_styles'], (4,)).to(config['device'])
    
    # Test encoder
    print("Testing encoder...")
    memory = encoder(input_ids, style_id=style_ids)
    print(f"Memory shape: {memory.shape}")
    
    # Test decoder forward pass
    print("\nTesting decoder forward pass...")
    target_ids = torch.randint(0, config['vocab_size'], (4, 32)).to(config['device'])
    logits = decoder(target_ids, memory, style_id=style_ids)
    print(f"Logits shape: {logits.shape}")
    
    # Test autoregressive generation
    print("\nTesting autoregressive generation...")
    generated_ids = decoder.inference(memory, style_id=style_ids, strategy="greedy")
    print(f"Generated IDs shape: {generated_ids.shape}")
    
    # Test beam search
    print("\nTesting beam search...")
    beam_ids = decoder.beam_search(memory, style_id=style_ids)
    print(f"Beam search IDs shape: {beam_ids.shape}")
    
    print("\n=== TextDecoder tests passed ===\n")
