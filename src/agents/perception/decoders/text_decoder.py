import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List, Dict

from ...base.utils.activation_engine import he_init
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import TensorOps, Parameter
from ..modules.transformer import Transformer
from ..perception_memory import PerceptionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Text Decoder")
printer = PrettyPrinter

class TextDecoder(nn.Module):
    """
    Text decoder for autoregressive generation.
    Supports greedy, sampling, and beam search.
    """
    def __init__(self, encoder: Optional[nn.Module] = None):
        super().__init__()
        self._init_configs()
        self._validate_configs()
        self._init_components(encoder)
        logger.info(f"TextDecoder initialized: vocab_size={self.vocab_size}, "
                    f"embed_dim={self.embed_dim}, num_layers={self.num_layers}")

    def _init_configs(self):
        """Load configurations from global and section."""
        self.config = load_global_config()
        self.text_decoder_config = get_config_section('text_decoder') if 'text_decoder' in self.config else {}
        self.text_encoder_config = get_config_section('text_encoder')

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

        # Decoder-specific parameters
        self.max_gen_length = self.text_decoder_config.get('max_gen_length',
                                                            self.text_encoder_config.get('max_gen_length', 64))
        self.beam_size = self.text_decoder_config.get('beam_size', 5)
        self.tie_embeddings = self.text_decoder_config.get('tie_embeddings', True)
        self.temperature = self.text_decoder_config.get('temperature', 1.0)
        self.top_k = self.text_decoder_config.get('top_k', 0)
        self.top_p = self.text_decoder_config.get('top_p', 0.0)
        self.repetition_penalty = self.text_decoder_config.get('repetition_penalty', 1.2)
        self.length_penalty = self.text_decoder_config.get('length_penalty', 0.6)
        self.sos_token_id = self.text_decoder_config.get('sos_token_id', 1)
        self.eos_token_id = self.text_decoder_config.get('eos_token_id', 2)
        self.return_hidden = self.text_decoder_config.get('return_hidden', False)
        self.use_checkpointing = self.text_decoder_config.get('use_gradient_checkpointing', True)

        # Attention maps cache
        self.attention_maps = {}

    def _validate_configs(self):
        """Validate critical parameters."""
        if not self.embed_dim:
            raise ValueError("embed_dim must be specified")
        if not self.vocab_size:
            raise ValueError("vocab_size must be specified")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")

    def _init_components(self, encoder: Optional[nn.Module] = None):
        """Initialize all decoder components."""
        # Token embeddings (shared with encoder if possible)
        if encoder is not None and self.tie_embeddings and hasattr(encoder, 'token_embeddings'):
            self.token_embeddings = encoder.token_embeddings
            logger.info("Sharing token embeddings with encoder")
        else:
            self.token_embeddings = self._init_embeddings()

        # Positional encoding
        self.position_embeddings = self._init_positional_encoding()

        # Style embeddings
        self.style_embeddings = nn.Embedding(self.num_styles, self.embed_dim)

        # Transformer backbone
        self.transformer = Transformer()
        self.transformer.return_hidden = True  # Always return full sequence
        self.transformer.causal = True         # Enable causal masking for autoregressive decoding

        # Disable feedforward fusion (decoder should not add encoder memory to FFN)
        for layer in self.transformer.layers:
            if 'ff' in layer and hasattr(layer['ff'], 'fusion_type'):
                layer['ff'].fusion_type = None

        # Output projection
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        if self.tie_embeddings:
            self.output_proj.weight = self.token_embeddings
        else:
            self.output_proj.weight = nn.Parameter(self.token_embeddings.data.clone())

        # Memory for gradient checkpointing
        self.memory = PerceptionMemory(enable_checkpointing=self.use_checkpointing)

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

    def _init_positional_encoding(self) -> Parameter:
        """Initialize positional embeddings."""
        if self.positional_encoding == "sinusoidal":
            pe = self._sinusoidal_encoding()
            return Parameter(pe, requires_grad=False)
        else:  # learned
            pe = torch.randn(1, self.max_position_embeddings, self.embed_dim, device=self.device) * 0.02
            return Parameter(pe, requires_grad=True)

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
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for text decoding.

        Args:
            input_ids: Token indices (batch, tgt_seq_len)
            memory: Encoder output (batch, src_seq_len, embed_dim)
            memory_mask: Padding mask for memory (batch, src_seq_len) – 1 for real, 0 for pad
            style_id: Optional style IDs (batch,)

        Returns:
            Logits (batch, tgt_seq_len, vocab_size)
        """
        batch_size, tgt_seq_len = input_ids.shape

        # Truncate long sequences
        if tgt_seq_len > self.max_position_embeddings:
            input_ids = input_ids[:, :self.max_position_embeddings]
            tgt_seq_len = self.max_position_embeddings
            logger.warning(f"Target sequence truncated to {tgt_seq_len} tokens")

        # Token embeddings
        embeddings = self.token_embeddings[input_ids]  # (B, L, D)

        # Add positional embeddings
        embeddings = embeddings + self.position_embeddings[:, :tgt_seq_len, :]

        # Input dropout
        if self.training and self.dropout_rate > 0:
            embeddings = F.dropout(embeddings, p=self.dropout_rate, training=self.training)

        # Style conditioning
        if style_id is not None:
            if not isinstance(style_id, torch.Tensor):
                style_id = torch.tensor(style_id, device=input_ids.device)
            style_emb = self.style_embeddings(style_id).unsqueeze(1)  # (B, 1, D)
            embeddings = embeddings + style_emb

        # Process through transformer (causal attention is enabled internally)
        output = self.transformer(
            x=embeddings,
            context=memory,
            context_mask=memory_mask,
            style_id=style_id
        )

        # Project to vocabulary
        logits = self.output_proj(output)  # (B, L, vocab_size)

        return logits

    def inference(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None,
        strategy: str = "greedy"
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            memory: Encoder output (batch, src_seq_len, embed_dim)
            memory_mask: Padding mask for memory
            style_id: Optional style IDs (batch,)
            strategy: One of "greedy", "sampling", "beam"

        Returns:
            Generated token IDs (batch, seq_len)
        """
        if strategy == "beam":
            return self._beam_search(memory, memory_mask, style_id)
        else:
            return self._greedy_or_sampling(memory, memory_mask, style_id, strategy)

    def _greedy_or_sampling(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        style_id: Optional[torch.Tensor],
        strategy: str
    ) -> torch.Tensor:
        """Greedy or sampling generation."""
        batch_size = memory.size(0)
        device = memory.device

        # Initialize with SOS token
        input_ids = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.max_gen_length):
            # Get logits for current sequence
            logits = self.forward(input_ids, memory, memory_mask, style_id)  # (B, L, V)
            next_logits = logits[:, -1, :]  # (B, V)

            # Apply repetition penalty
            if self.repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in input_ids[i]:
                        if next_logits[i, token_id] > 0:
                            next_logits[i, token_id] /= self.repetition_penalty
                        else:
                            next_logits[i, token_id] *= self.repetition_penalty

            # Temperature scaling
            if self.temperature != 1.0:
                next_logits = next_logits / self.temperature

            # Top‑k / top‑p filtering
            if self.top_k > 0:
                next_logits = self._top_k_filtering(next_logits)
            if self.top_p > 0.0:
                next_logits = self._top_p_filtering(next_logits)

            probs = F.softmax(next_logits, dim=-1)

            # Select next token
            if strategy == "greedy":
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:  # sampling
                next_token = torch.multinomial(probs, 1)

            # Mark finished sequences
            finished = finished | (next_token.squeeze(-1) == self.eos_token_id)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if all finished
            if finished.all():
                break

        return input_ids

    def _beam_search(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        style_id: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Beam search generation."""
        batch_size = memory.size(0)
        device = memory.device

        # Expand memory and mask for beam search
        memory = memory.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
        memory = memory.view(batch_size * self.beam_size, -1, memory.size(-1))
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1).repeat(1, self.beam_size, 1)
            memory_mask = memory_mask.view(batch_size * self.beam_size, -1)
        if style_id is not None:
            style_id = style_id.unsqueeze(1).repeat(1, self.beam_size).view(-1)

        # Initialize beams
        beams = torch.full((batch_size, self.beam_size, 1), self.sos_token_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, self.beam_size, device=device)

        for step in range(self.max_gen_length):
            # Flatten batch and beams
            flat_beams = beams.view(batch_size * self.beam_size, -1)

            # Get logits for all beams
            logits = self.forward(flat_beams, memory, memory_mask, style_id)  # (B*beam, L, V)
            next_logits = logits[:, -1, :]  # (B*beam, V)

            # Temperature scaling
            if self.temperature != 1.0:
                next_logits = next_logits / self.temperature

            # Top‑k / top‑p filtering
            if self.top_k > 0:
                next_logits = self._top_k_filtering(next_logits)
            if self.top_p > 0.0:
                next_logits = self._top_p_filtering(next_logits)

            # Compute log probabilities
            log_probs = F.log_softmax(next_logits, dim=-1)  # (B*beam, V)

            # Add beam scores
            scores = beam_scores.view(-1, 1) + log_probs  # (B*beam, V)

            # Reshape to (batch, beam * vocab)
            scores = scores.view(batch_size, self.beam_size * self.vocab_size)

            # Select top‑k beams
            top_scores, top_indices = torch.topk(scores, self.beam_size, dim=-1)

            # Extract beam and token indices
            beam_indices = top_indices // self.vocab_size
            token_indices = top_indices % self.vocab_size

            # Update beams
            new_beams = []
            for i in range(batch_size):
                # Gather previous beams
                prev_beams = beams[i].index_select(0, beam_indices[i])
                new_beam = torch.cat([prev_beams, token_indices[i].unsqueeze(-1)], dim=-1)
                new_beams.append(new_beam)
            beams = torch.stack(new_beams)  # (batch, beam, step+1)
            beam_scores = top_scores

            # Check for EOS
            # (Simplified: we stop if all top‑1 tokens are EOS; full beam search would handle EOS early)
            if (token_indices[:, 0] == self.eos_token_id).all():
                break

        # Return best beam for each batch (first beam, ignoring EOS after)
        best_beams = beams[:, 0, :]  # (batch, seq_len)
        return best_beams

    def _top_k_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """Keep only top k logits."""
        if self.top_k <= 0:
            return logits
        values, _ = torch.topk(logits, self.top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(logits < min_values, torch.full_like(logits, -float('inf')), logits)

    def _top_p_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """Keep only tokens with cumulative probability > top_p."""
        if self.top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter back to original indices
        for i in range(logits.size(0)):
            logits[i, sorted_indices[i, sorted_indices_to_remove[i]]] = -float('inf')
        return logits

    def register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        self.transformer.output_attentions = True
        self.attention_maps = {}

        def hook_fn(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.attention_maps[layer_idx] = attn_weights.detach().cpu()
            return hook

        for idx, layer in enumerate(self.transformer.layers):
            if 'attention' in layer:
                attn_module = layer['attention']
                attn_module.output_attentions = True
                attn_module.register_forward_hook(hook_fn(idx))
                logger.info(f"Registered attention hook for layer {idx}")

    def freeze_layers(self, layer_indices: Optional[List[int]] = None):
        """Freeze transformer layers."""
        if layer_indices is None:
            for param in self.parameters():
                param.requires_grad = False
            logger.info("All decoder layers frozen")
        else:
            self.transformer.freeze_layers(layer_indices)

    def unfreeze_layers(self, layer_indices: Optional[List[int]] = None):
        """Unfreeze transformer layers."""
        if layer_indices is None:
            for param in self.parameters():
                param.requires_grad = True
            logger.info("All decoder layers unfrozen")
        else:
            self.transformer.unfreeze_layers(layer_indices)

    def load_pretrained(self, weights: Dict[str, torch.Tensor]):
        """Load pretrained weights from a dictionary."""
        # Map keys (e.g., from a pre‑trained model)
        mapping = {
            'decoder.token_embeddings': 'token_embeddings',
            'decoder.position_embeddings': 'position_embeddings',
            'decoder.style_embeddings': 'style_embeddings',
            'decoder.output_proj': 'output_proj',
        }
        for old_key, new_key in mapping.items():
            if old_key in weights:
                param = getattr(self, new_key, None)
                if param is not None:
                    param.data.copy_(weights[old_key].to(self.device))

        # Load transformer weights
        prefix = 'decoder.transformer.'
        transformer_weights = {
            k[len(prefix):]: v
            for k, v in weights.items()
            if k.startswith(prefix)
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'transformer'):
            self.transformer.train(mode)
        return self

    def eval(self):
        return self.train(False)


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Testing Text Decoder ===\n")

    # Override config for testing
    config = load_global_config()
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # Create encoder and decoder
    from ..encoders.text_encoder import TextEncoder
    encoder = TextEncoder().to(config['device'])
    decoder = TextDecoder(encoder).to(config['device'])

    # Test data
    batch_size, src_len, tgt_len = 2, 128, 32
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, src_len)).to(config['device'])
    style_ids = torch.randint(0, config['num_styles'], (batch_size,)).to(config['device'])

    # Encode
    memory = encoder(input_ids, style_id=style_ids)
    print(f"Memory shape: {memory.shape}")

    # Decoder forward
    target_ids = torch.randint(0, config['vocab_size'], (batch_size, tgt_len)).to(config['device'])
    logits = decoder(target_ids, memory, style_id=style_ids)
    print(f"Logits shape: {logits.shape}")

    # Greedy generation
    generated = decoder.inference(memory, style_id=style_ids, strategy="greedy")
    print(f"Greedy generation shape: {generated.shape}")

    # Sampling generation
    generated_sample = decoder.inference(memory, style_id=style_ids, strategy="sampling")
    print(f"Sampling generation shape: {generated_sample.shape}")

    # Beam search
    if config['device'] != 'cpu':
        # Beam search is slower; only test on GPU
        generated_beam = decoder.inference(memory, style_id=style_ids, strategy="beam")
        print(f"Beam search generation shape: {generated_beam.shape}")

    print("\n=== TextDecoder tests passed ===\n")