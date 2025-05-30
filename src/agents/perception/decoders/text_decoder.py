
import yaml, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger

logger = get_logger("Text Decoder")

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

class TextDecoder(torch.nn.Module):
    def __init__(self, text_encoder, tokenizer, config):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.embed_dim = config['transformer']['embed_dim']
        
        # Replace Generator's output_head with decoder parameters
        self.output_head = Parameter(
            TensorOps.he_init((self.embed_dim, self.vocab_size), 
            fan_in=self.embed_dim,
            device=text_encoder.device
        )
        self.repetition_penalty = 1.2  # From generator config
        self._cache = {}

    def forward(self, hidden_states):
        """For teacher-forcing training (replaces Generator's step-by-step)"""
        logits = torch.matmul(hidden_states, self.output_head.data)
        return logits  # Shape: [batch, seq_len, vocab_size]

    def generate(self, prompt, max_length=50, temperature=1.0, top_k=0, top_p=0.0):
        encoded = self.tokenizer.encode(prompt)
        input_ids = encoded['input_ids'].tolist()
        attention_mask = encoded['attention_mask'].tolist()
        
        generated = input_ids.copy()

        for _ in range(max_length):
            # Prepare input
            input_tensor = torch.tensor(generated).unsqueeze(0)
            attention_tensor = torch.tensor(attention_mask, dtype=torch.bool).unsqueeze(0)
            # attention_tensor = torch.tensor(attention_mask).unsqueeze(0)
            # embeddings = self.text_encoder.embedding(input_tensor)

            # Run transformer in causal mode
            hidden_states = self.text_encoder.forward(
                x=torch.index_select(
                    self.text_encoder.embedding.data, 
                    0, 
                    input_tensor.squeeze(0)  # Remove batch dimension
                ).unsqueeze(0),  # Restore batch dimension
                style_id=0,
                # attention_mask=torch.ones_like(input_tensor),
                attention_mask=torch.ones_like(input_tensor, dtype=torch.bool),
                causal=True
            )

            # Get last token hidden state
            last_hidden = hidden_states[:, -1, :]  # shape: (1, hidden_dim)

            # Project to logits
            logits = torch.matmul(last_hidden, self.output_head.data).flatten()
            logits = logits / temperature
            probs = softmax(logits)

            # Apply top-k or top-p filtering
            if top_k > 0:
                probs = top_k_filter(probs, k=top_k)
            if top_p > 0.0:
                probs = top_p_filter(probs, p=top_p)

            # Sample next token
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token_id)
            attention_mask.append(1)

            # Stop if SEP token reached
            if next_token_id == self.tokenizer.sep_token_id:
                break

        # Decode final sequence
        return self.tokenizer.decode(generated)

    def beam_search_generate(self, prompt, max_length=50, beam_width=5, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.2):
        encoded = self.tokenizer.encode(prompt)
        input_ids = encoded['input_ids'].tolist()
        attention_mask = encoded['attention_mask'].tolist()

        beams = [(input_ids.copy(), 0.0)]  # (sequence, cumulative log-prob)
    
        for _ in range(max_length):
            all_candidates = []
    
            for seq, log_prob in beams:
                input_tensor = torch.tensor(seq).unsqueeze(0)
                hidden_states = self.text_encoder.forward(
                    x=torch.index_select(self.text_encoder.embedding.data, 0, input_tensor),
                    style_id=0,
                    # attention_mask=torch.ones_like(input_tensor),
                    attention_mask=torch.ones_like(input_tensor, dtype=torch.bool),
                    causal=True
                )
                last_hidden = hidden_states[:, -1, :]
                logits = torch.matmul(last_hidden, self.output_head.data).flatten() / temperature
    
                # Apply repetition penalty
                for token_id in set(seq):
                    logits[token_id] /= repetition_penalty
    
                probs = softmax(logits)
                if top_k > 0:
                    probs = top_k_filter(probs, top_k)
                if top_p > 0.0:
                    probs = top_p_filter(probs, top_p)
    
                top_indices = torch.topk(probs, beam_width).indices
    
                for idx in top_indices:
                    new_seq = seq + [idx.item()]
                    new_log_prob = log_prob + torch.log(probs[idx] + 1e-12).item()
                    all_candidates.append((new_seq, new_log_prob))
    
            # Keep top beams
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
            # Check for SEP token in top beam
            if any(seq[-1] == self.tokenizer.sep_token_id for seq, _ in beams):
                break
    
        best_sequence, best_log_prob = beams[0]
        return self.tokenizer.decode(best_sequence), best_log_prob


    def _generate_impl(self, prompt, max_length, beam_width=1, 
                      temperature=1.0, top_k=0, top_p=0.0):
        # Copy logic from Generator.generate() but:
        # 1. Replace `self.text_encoder.transformer` with `self.text_encoder`
        # 2. Use `self.output_head` instead of `generator.output_head`
        # 3. Use `self.tokenizer` instead of `generator.tokenizer`
        
        # [Keep original Generator code here, replacing 'self' references]
        # Return (generated_text, log_prob) if beam_width > 1 else generated_text

class Generator:
    def beam_search_generate(self, prompt, max_length=50, beam_width=5, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.2):
