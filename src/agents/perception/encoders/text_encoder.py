import math
import json
import numpy as np

from typing import List

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer  

class TextEncoder:
    def __init__(
        self,
        vocab_size=50257,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
        positional_encoding="learned",
        max_seq_len=512
    ):
        encoder = TextEncoder()
        with open("data/embeddings/glove.6B.100d.json") as f:
            glove_data = json.load(f)
        vocab = list(glove_data.keys())  # Or your actual vocabulary list
        encoder.load_glove_embeddings("data/embeddings/glove.6B.100d.json", vocab)
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.training = True

        # Token embeddings
        self.embedding = Parameter(np.random.randn(vocab_size, embed_dim) * 0.02)
        
        # Positional embeddings
        self.positional_encoding = positional_encoding
        if positional_encoding == "learned":
            self.position_embed = Parameter(TensorOps.he_init((1, max_seq_len, embed_dim), embed_dim))
        elif positional_encoding == "sinusoidal":
            self.position_embed = self._init_sinusoidal_encoding(max_seq_len, embed_dim)
        
        self.transformer = Transformer(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self._cache = {}

    def _init_sinusoidal_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return Parameter(pe[np.newaxis, :, :])  # Add batch dimension

    def load_pretrained(self, weights):
        """Handle multiple weight formats (HF-style, custom, partial)"""
        # Token embeddings
        if 'token_embedding' in weights:
            self.embedding.data = weights['token_embedding']
        elif 'word_embeddings.weight' in weights:  # HF compatibility
            self.embedding.data = weights['word_embeddings.weight']
        
        # Positional embeddings
        if 'position_embedding' in weights:
            self.position_embed.data = weights['position_embedding']
        elif 'position_embeddings.weight' in weights:  # HF compatibility
            self.position_embed.data = weights['position_embeddings.weight'][np.newaxis]
        
        # Transformer weights
        transformer_weights = {
            k.split('transformer_')[-1]: v 
            for k, v in weights.items() 
            if k.startswith('transformer_')
        }
        if transformer_weights:
            self.transformer.load_pretrained(transformer_weights)

    def load_glove_embeddings(self, glove_path: str, vocab: List[str]):
        """Load GloVe vectors and assign to the embedding matrix"""
        import json
        with open(glove_path, 'r') as f:
            glove_data = json.load(f)
        
        for idx, word in enumerate(vocab):
            if word in glove_data:
                self.embedding.data[idx] = np.array(glove_data[word])

    def forward(self, x, style_id=0):
        """Forward pass with dropout and dynamic sequence handling"""
        self._tokens = x.copy()
        seq_len = x.shape[1]
        
        # Embed tokens
        embed = np.take(self.embedding.data, x, axis=0)
        
        # Add positional embeddings
        if self.positional_encoding == "sinusoidal":
            embed += self.position_embed.data[:, :seq_len, :]
        else:
            embed += self.position_embed.data[:, :seq_len, :]
        
        # Apply dropout
        if self.training and self.dropout_rate > 0:
            mask = (np.random.rand(*embed.shape) > self.dropout_rate).astype(np.float32)
            embed *= mask
        
        # Transformer processing
        embed = self.transformer.forward(embed, style_id)
        return embed

    def backward(self, dout):
        """Backprop through encoder"""
        d_embed = self.transformer.backward(dout)
        
        # Gradient for token embeddings
        np.add.at(self.embedding.grad, self._tokens, d_embed)
        return d_embed  # For chaining gradients if needed

    def parameters(self):
        return [self.embedding, self.position_embed] + self.transformer.parameters()

    def train(self):
        self.training = True
        self.transformer.training = True

    def eval(self):
        self.training = False
        self.transformer.training = False
