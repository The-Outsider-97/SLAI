import math
import json
import torch
import torch.nn as nn

from typing import List, Tuple, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer, TaskHead
from src.agents.perception.modules.tokenizer import Tokenizer
from logs.logger import get_logger

logger = get_logger(__name__)

#==========================
# Embeddings using GloVe
#==========================
# EMBEDDING_PATH = "data/embeddings/glove.6B.200d.json"
# MAX_SYNONYMS = 5
# MAX_RELATED = 5

#def load_embeddings(path: str) -> dict:
#    if not Path(path).exists():
#        raise FileNotFoundError(f"Embedding file {path} not found!")
#    with open(path, "r", encoding="utf-8") as f:
#        return json.load(f)

#def generate_from_embeddings(word: str, embedding_lookup: dict, topn=10) -> Tuple[List[str], List[str]]:
#    if word not in embedding_lookup:
#        return [], []
#    word_vec = torch.tensor(embedding_lookup[word]).reshape(1, -1)

    # Ensure all vectors are numpy arrays
#    valid_embeddings = {
#        other: torch.tensor(vec)
#        for other, vec in embedding_lookup.items()
#        if other != word and isinstance(vec, list) and len(vec) == word_vec.shape[1] # Basic check
#    }
#    if not valid_embeddings:
#        return [], []

#    other_words = list(valid_embeddings.keys())
#    other_vecs = torch.tensor(list(valid_embeddings.values()))

#    try:
#        similarities = cosine_similarity(word_vec, other_vecs)[0]
#        # Create word-score pairs
#        scores = {other_words[i]: similarities[i] for i in range(len(other_words))}
#        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

#        synonyms = [w for w, score in ranked[:MAX_SYNONYMS] if score > 0.6] # Add threshold?
#        related = [w for w, score in ranked[MAX_SYNONYMS : MAX_SYNONYMS + MAX_RELATED] if score > 0.4] # Add threshold?

#        return synonyms, related
#    except Exception as e:
#        logger.error(f"Error in cosine similarity for '{word}': {e}")
#        return [], []

#class EmbeddingManager:
#    def __init__(self, path):
#        self.embeddings = load_embeddings(path)

#    def get_similar(self, word, topn=10):
#        return generate_from_embeddings(word, self.embeddings, topn)
# =======================

class TextEncoder(torch.nn.Module):
    def __init__(self,
                 
                 vocab_size: int, # Get from tokenizer
                 embed_dim: int, # Model's embedding dim
                 num_layers: int,
                 num_heads: int,
                 ff_dim: int,
                 num_styles: int, # Pass through if needed
                 dropout_rate: float = 0.1,
                 positional_encoding: str = "sinusoidal", # or "learned"
                 max_length: int = 512, # Needed for positional encoding size
                 device: str = 'cpu', # Added device
                 tokenizer=None
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.positional_encoding = positional_encoding
        self.device = device # Store device
        self.training = True # Added training state

        # Token embeddings
        self.embedding = nn.Parameter(torch.randn(vocab_size, embed_dim, device=self.device) * 0.02)
        self.position_embed = self._init_sinusoidal_encoding(max_length, embed_dim, device)
        #self.tokenizer = Tokenizer(vocab_path=self.vocab_path, max_length=self.max_seq_length)
        self.tokenizer = tokenizer
        

        # Initialize Transformer layers
        self.transformer = Transformer(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_styles=num_styles,
            # Ensure Transformer and its submodules get the device if they need it
        )
        # Move transformer parameters to device if not handled internally
        for param in self.transformer.parameters():
             if param is not None and hasattr(param, 'data'):
                 param.data = param.data.to(self.device)

        self._tokens = None # Cache for backward pass

    def _init_sinusoidal_encoding(self, max_len, embed_dim, device):
        if self.positional_encoding == "sinusoidal":
             pe = torch.zeros(1, max_len, embed_dim, device=device)
             position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
             div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
             pe[0, :, 0::2] = torch.sin(position * div_term)
             pe[0, :, 1::2] = torch.cos(position * div_term)
             # Make it a non-learnable buffer or Parameter(..., requires_grad=False)
             param = Parameter(pe)
             param.requires_grad = False
             return param
        elif self.positional_encoding == "learned":
             return Parameter(torch.randn(1, max_len, embed_dim) * 0.02) # Add batch dim
        else:
             raise ValueError("Positional encoding must be 'sinusoidal' or 'learned'")

    def forward(self, x, style_id=None, task_head=None):
        """Forward pass with dropout and dynamic sequence handling"""
        hidden_states = self.transformer.forward(x)
        if task_head:
            return task_head.forward(hidden_states)
        self._tokens = x.copy()
        seq_len = x.shape[1]
        
        # Embed tokens
        embed = nn.take(self.embedding.data, x, axis=0)
        
        # Add positional embeddings
        if self.positional_encoding == "sinusoidal":
            embed += self.position_embed.data[:, :seq_len, :]
        else:
            embed += self.position_embed.data[:, :seq_len, :]
        
        # Apply dropout
        if self.training and self.dropout_rate > 0:
            mask = (torch.rand(embed.shape, device=embed.device) > self.dropout_rate).float()
            embed *= mask
        
        # Transformer processing
        embed = self.transformer.forward(embed, style_id)
        return embed, hidden_states

    def backward(self, dout):
        """Backprop through encoder"""
        d_embed = self.transformer.backward(dout)
        
        # Gradient for token embeddings
        nn.add.at(self.embedding.grad, self._tokens, d_embed)
        return d_embed  # For chaining gradients if needed

    def parameters(self):
        return [self.embedding, self.position_embed] + self.transformer.parameters()

    def train(self):
        self.training = True
        self.transformer.training = True

    def eval(self):
        self.training = False
        self.transformer.training = False

    # def load_pretrained(self, weights):
    #    """Handle multiple weight formats (HF-style, custom, partial)"""
    #    # Token embeddings
    #    if 'token_embedding' in weights:
    #        self.embedding.data = weights['token_embedding']
    #    elif 'word_embeddings.weight' in weights:  # HF compatibility
    #        self.embedding.data = weights['word_embeddings.weight']
    #    
    #    # Positional embeddings
    #    if 'position_embedding' in weights:
    #        self.position_embed.data = weights['position_embedding']
    #    elif 'position_embeddings.weight' in weights:  # HF compatibility
    #        self.position_embed.data = weights['position_embeddings.weight'][nn.newaxis]
        
    #    # Transformer weights
    #    transformer_weights = {
    #        k.split('transformer_')[-1]: v 
    #        for k, v in weights.items() 
    #        if k.startswith('transformer_')
    #    }
    #    if transformer_weights:
    #        self.transformer.load_pretrained(transformer_weights)

    #def load_glove_embeddings(self, glove_path: str, vocab: List[str]):
    #    """Load GloVe vectors and assign to the embedding matrix"""
    #    import json
    #    with open(glove_path, 'r', encoding="utf-8") as f:
    #        glove_data = json.load(f)
        
    #    for idx, word in enumerate(vocab):
    #        if word in glove_data:
    #            self.embedding.data[idx] = torch.tensor(glove_data[word])
