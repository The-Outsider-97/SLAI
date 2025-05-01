import math
import json
import numpy as np

from typing import List, Tuple
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger

logger = get_logger(__name__)

#==========================
# Embeddings using GloVe
#==========================
EMBEDDING_PATH = "data/embeddings/glove.6B.100d.json"
MAX_SYNONYMS = 5
MAX_RELATED = 5

def load_embeddings(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Embedding file {path} not found!")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_from_embeddings(word: str, embedding_lookup: dict, topn=10) -> Tuple[List[str], List[str]]:
    if word not in embedding_lookup:
        return [], []
    word_vec = np.array(embedding_lookup[word]).reshape(1, -1)

    # Ensure all vectors are numpy arrays
    valid_embeddings = {
        other: np.array(vec)
        for other, vec in embedding_lookup.items()
        if other != word and isinstance(vec, list) and len(vec) == word_vec.shape[1] # Basic check
    }
    if not valid_embeddings:
        return [], []

    other_words = list(valid_embeddings.keys())
    other_vecs = np.array(list(valid_embeddings.values()))

    try:
        similarities = cosine_similarity(word_vec, other_vecs)[0]
        # Create word-score pairs
        scores = {other_words[i]: similarities[i] for i in range(len(other_words))}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        synonyms = [w for w, score in ranked[:MAX_SYNONYMS] if score > 0.6] # Add threshold?
        related = [w for w, score in ranked[MAX_SYNONYMS : MAX_SYNONYMS + MAX_RELATED] if score > 0.4] # Add threshold?

        return synonyms, related
    except Exception as e:
        logger.error(f"Error in cosine similarity for '{word}': {e}")
        return [], []

class EmbeddingManager:
    def __init__(self, path):
        self.embeddings = load_embeddings(path)

    def get_similar(self, word, topn=10):
        return generate_from_embeddings(word, self.embeddings, topn)
# =======================

class TextEncoder:
    def __init__(
        self,
        vocab_size=50257,
        embed_dim=100,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
        positional_encoding="learned",
        max_seq_len=512
    ):
        with open("data/embeddings/glove.6B.100d.json", encoding="utf-8") as f:
            glove_data = json.load(f)

        # Token embeddings
        self.embedding = Parameter(np.random.randn(vocab_size, embed_dim) * 0.02)

        vocab = list(glove_data.keys())
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        self.unk_token_id = self.vocab.get("<unk>", 0)
        self.load_glove_embeddings("data/embeddings/glove.6B.100d.json", vocab)
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.training = True
 
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
        with open(glove_path, 'r', encoding="utf-8") as f:
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
