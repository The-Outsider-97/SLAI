
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional, Dict, Any

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from src.agents.base.utils.base_transformer import BaseTransformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Language Transformer")
printer = PrettyPrinter

class LanguageTransformer(BaseTransformer):
    """Transformer specialized for language agent tasks with beam search support"""
    def __init__(self):
        """
        Initialize with optional language-specific configuration
        
        Args:
            lang_config: Additional configuration for language tasks
        """
        super().__init__()
        self.config = load_global_config()
        self.lang_config = get_config_section('language_transformer')
        self.beam_width = self.lang_config.get('beam_width')
        self.max_len = self.lang_config.get('max_len')
        self.sos_token = self.lang_config.get('sos_token')
        self.eos_token = self.lang_config.get('eos_token')
        self.norm_first = self.lang_config.get('norm_first')
        self.temperature = self.lang_config.get('temperature')
        self.length_penalty = self.lang_config.get('length_penalty')

        self._init_language_params()

    def _init_language_params(self) -> None:
        """Initialize language-specific parameters"""
        # Set language-specific dropout if provided
        if 'dropout' in self.lang_config:
            self.dropout = self.lang_config['dropout']
            # Update dropout layers
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.p = self.dropout

    def beam_search_decode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Beam search decoding for improved text generation
        
        Args:
            src: Source tensor (1, S)
            beam_width: Number of beams to maintain
            max_len: Maximum generation length
            sos_token: Start token ID
            eos_token: End token ID
            temperature: Sampling temperature
            length_penalty: Penalty for longer sequences (>0 encourages longer sequences)
            
        Returns:
            Best sequence found (1, L)
        """
        self.eval()
        with torch.no_grad():
            # Encode source once
            memory = self.encode(src)
            batch_size = src.size(0)
            device = src.device

            # Initialize beams (sequence, score)
            beams = [([self.sos_token], 0.0)] * batch_size
            final_beams = [[] for _ in range(batch_size)]

            for step in range(self.max_len):
                all_candidates = []
                for beam_idx, (tokens, score) in enumerate(beams):
                    # Skip completed beams
                    if tokens[-1] == self.eos_token:
                        final_beams[beam_idx].append((tokens, score))
                        continue

                    # Prepare decoder input
                    tgt = torch.tensor(tokens, device=device).unsqueeze(0)
                    tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)

                    # Decode step
                    output = self.decode(tgt, memory, tgt_mask=tgt_mask)
                    logits = self.fc_out(output[:, -1, :]) / self.temperature
                    probs = F.softmax(logits, dim=-1).squeeze(0)

                    # Get top candidates
                    top_probs, top_indices = torch.topk(probs, self.beam_width)
                    for i in range(self.beam_width):
                        token = top_indices[i].item()
                        new_score = score + torch.log(top_probs[i]).item()
                        new_tokens = tokens + [token]
                        all_candidates.append((new_tokens, new_score))

                # No new candidates means we're done
                if not all_candidates:
                    break

                # Select top beams
                all_candidates.sort(key=lambda x: x[1] / (len(x[0])**self.length_penalty), reverse=True)
                beams = all_candidates[:self.beam_width]

            # Combine final and active beams
            for beam_list in final_beams:
                beam_list.extend(beams)
            final_beams = [sorted(b, key=lambda x: x[1] / (len(x[0])**self.length_penalty), reverse=True)[0][0] 
                          for b in final_beams]

            return torch.tensor(final_beams, device=device)

    def language_forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Language-optimized forward pass with automatic mask generation
        
        Args:
            src: Source sequence tensor
            tgt: Target sequence tensor
            src_mask: Optional source mask
            tgt_mask: Optional target mask
            
        Returns:
            Transformer output
        """
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            
        return super().forward(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_mask=src_mask
        )

    def initialize_language_params(self) -> None:
        """Custom initialization for language tasks"""
        for name, param in self.named_parameters():
            if 'embed' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'weight' in name and 'norm' not in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def adapt_for_language_task(self, task_type: str) -> None:
        """
        Modify transformer for specific language tasks
        
        Args:
            task_type: One of ['generation', 'classification', 'translation']
        """
        if task_type == 'generation':
            # Increase dropout for regularization
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.p = min(0.3, module.p + 0.1)
                    
        elif task_type == 'classification':
            # Freeze embedding layers
            for param in self.src_embed.parameters():
                param.requires_grad = False
            for param in self.tgt_embed.parameters():
                param.requires_grad = False

    def save_language_model(self, path: str, lang_metadata: Optional[Dict] = None) -> None:
        """
        Save with language-specific metadata
        
        Args:
            path: Save path
            lang_metadata: Additional language metadata
        """
        state = {
            'state_dict': self.state_dict(),
            'config': {
                'src_vocab_size': self.src_embed.num_embeddings,
                'tgt_vocab_size': self.tgt_embed.num_embeddings,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_encoder_layers': self.num_encoder_layers,
                'num_decoder_layers': self.num_decoder_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'activation': self.activation,
                'layer_norm_eps': self.layer_norm_eps,
                'lang_config': self.lang_config
            },
            'lang_metadata': lang_metadata or {}
        }
        torch.save(state, path)

    @classmethod
    def load_language_model(cls, path: str, device: torch.device = None) -> 'LanguageTransformer':
        """
        Load language transformer from checkpoint
        
        Args:
            path: Model path
            device: Target device
            
        Returns:
            Loaded LanguageTransformer instance
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'].get('lang_config', {}))
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device)

    def calculate_perplexity(self, src: torch.Tensor, tgt: torch.Tensor) -> float:
        """
        Calculate perplexity for language evaluation
        
        Args:
            src: Source sequence
            tgt: Target sequence
            
        Returns:
            Perplexity score
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(src, tgt[:, :-1])
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1),
                ignore_index=0
            )
            return math.exp(loss.item())
        

if __name__ == "__main__":
    print("\n=== Running Language Transformer ===\n")
    printer.status("Init", "Language Transformer initialized", "success")

    model = LanguageTransformer()

    # Dummy input tensors
    src_tensor = torch.randint(1, 100, (1, 10))  # (batch_size, src_seq_len)
    tgt_val = torch.randint(1, 100, (1, 11))     # (batch_size, tgt_seq_len), one longer than src for teacher forcing
    
    # Perform beam search decoding
    best_sequence = model.beam_search_decode(src_tensor)
    print("Best Sequence:", best_sequence)
    
    # Calculate perplexity
    perplexity = model.calculate_perplexity(src_tensor, tgt_val)
    print("Perplexity:", perplexity)

    # Save with language metadata
    model.save_language_model(
        "model.pth",
        lang_metadata={'lang': 'en', 'task': 'generation'}
    )
    #printer.pretty("Tokenizer", tokenizer.tokenize(text=text2), "success")
    print("\n=== Successfully Ran Language Transformer ===\n")
