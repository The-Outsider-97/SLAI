import math
import numpy as np

from src.agents.perception import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer  


class TextEncoder:
    def __init__(self, vocab_size=50257, embed_dim=512):
        self.embedding = Parameter(np.random.randn(vocab_size, embed_dim) * 0.02)
        self.position_embed = Parameter(
            TensorOps.he_init((1, 512, embed_dim), embed_dim))
        self.transformer = Transformer(num_layers=6, embed_dim=embed_dim)

    def load_pretrained(self, weights):
        self.embedding.data = weights['token_embedding']
        self.position_embed.data = weights['position_embedding']

    def forward(self, x):
        self._tokens = x.copy()
        embed = np.take(self.embedding.data, x, axis=0) + self.position_embed.data[:, :x.shape[1]]
        embed = self.transformer.forward(embed)
        return embed

    def backward(self, dout):
        d_embed = self.transformer.backward(dout)
        for i in range(self._tokens.shape[0]):
            for j in range(self._tokens.shape[1]):
                self.embedding.grad[self._tokens[i, j]] += d_embed[i, j]
        np.add.at(self.embedding.grad, self._tokens, d_embed)

    def parameters(self):
        return [self.embedding, self.position_embed] + self.transformer.parameters()
