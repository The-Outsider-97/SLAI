
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

    def generate(self, prompt, max_length=50, **kwargs):
        """Mirror Generator.generate() but as decoder method"""
        return self._generate_impl(prompt, max_length, beam_width=1, **kwargs)

    def beam_search_generate(self, prompt, max_length=50, beam_width=5, **kwargs):
        """Mirror Generator.beam_search_generate()"""
        return self._generate_impl(prompt, max_length, beam_width, **kwargs)

    def _generate_impl(self, prompt, max_length, beam_width=1, 
                      temperature=1.0, top_k=0, top_p=0.0):
        # Copy logic from Generator.generate() but:
        # 1. Replace `self.text_encoder.transformer` with `self.text_encoder`
        # 2. Use `self.output_head` instead of `generator.output_head`
        # 3. Use `self.tokenizer` instead of `generator.tokenizer`
        
        # [Keep original Generator code here, replacing 'self' references]
        # Return (generated_text, log_prob) if beam_width > 1 else generated_text
