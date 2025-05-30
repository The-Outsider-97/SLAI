import math
import json, yaml
import torch
import torch.nn as nn

from typing import List, Tuple, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
#from src.agents.perception.modules.tokenizer import Tokenizer
from logs.logger import get_logger

logger = get_logger("Text Encoder")

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
    def __init__(self, config, device: str = 'cpu', tokenizer=None):
        super().__init__()
        transformer_cfg = config['transformer']
        tokenizer_cfg = config['tokenizer']

        # Extract parameters from config
        self.vocab_size = tokenizer_cfg['vocab_size']
        self.embed_dim = transformer_cfg['embed_dim']
        self.num_layers = transformer_cfg['num_layers']
        self.num_heads = transformer_cfg['num_heads']
        self.ff_dim = transformer_cfg['ff_dim']
        self.num_styles = transformer_cfg['num_styles']
        self.max_length = transformer_cfg['max_position_embeddings']
        self.positional_encoding = transformer_cfg['positional_encoding']
        self.dropout_rate = transformer_cfg['dropout_rate']
        self.device = device
        self.tokenizer = tokenizer

        # Initialize components
        self.embedding = nn.Parameter(torch.randn(self.vocab_size, self.embed_dim, device=device) * 0.02)
        self.position_embed = self._init_positional_encoding()
        self.transformer = Transformer(config)

    def _init_positional_encoding(self):
        if self.positional_encoding == "sinusoidal":
            pe = torch.zeros(1, self.max_length, self.embed_dim, device=self.device)
            position = torch.arange(0, self.max_length, dtype=torch.float, device=self.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            return Parameter(pe, requires_grad=False)
        else:  # learned
            return Parameter(torch.randn(1, self.max_length, self.embed_dim, device=self.device) * 0.02)

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
        self._tokens = x.clone()
        seq_len = x.shape[1]

        embed = self.embedding[x]  # [batch_size, seq_len, embed_dim]

        embed += self.position_embed.data[:, :seq_len, :]

        # Apply dropout
        if self.training and self.dropout_rate > 0:
            mask = (torch.rand(embed.shape, device=embed.device) > self.dropout_rate).float()
            embed *= mask
    
        # Transformer processing
        hidden_states = self.transformer.forward(embed, style_id=style_id)
    
        if task_head:
            return task_head.forward(hidden_states)
    
        return hidden_states, hidden_states

    def backward(self, dout):
        """Backprop through encoder"""
        d_embed = self.transformer.backward(dout)  # Shape: [batch_size, seq_len, embed_dim]
    
        # Create a gradient tensor for the embedding matrix
        if self.embedding.grad is None:
            self.embedding.grad = torch.zeros_like(self.embedding.data)
    
        # Accumulate gradients per token
        for i in range(self._tokens.shape[0]):  # batch
            for j in range(self._tokens.shape[1]):  # sequence
                token_id = self._tokens[i, j]
                self.embedding.grad[token_id] += d_embed[i, j]
    
        return d_embed

    def parameters(self):
        return [self.embedding, self.position_embed] + list(self.transformer.parameters())

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

if __name__ == "__main__":
    print("\n=== Running Text Encoder ===\n")

    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize text encoder
    text_encoder = TextEncoder(config=config, device=device)
    text_encoder.to(device)
    text_encoder.train()

    # Create dummy token input (batch_size=2, sequence_length=10)
    dummy_input = torch.randint(0, text_encoder.vocab_size, (2, 10), device=device)

    # Run forward pass
    print("[Forward] Training mode:")
    output, hidden_states = text_encoder(dummy_input, style_id=0)
    print(f"Output shape: {output.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Output mean: {output.mean().item():.4f}")

    # Run backward pass with dummy gradient
    dummy_grad = torch.randn_like(output)
    grad = text_encoder.backward(dummy_grad)
    print(f"Backward gradient shape: {grad.shape}")

    # Switch to evaluation mode
    text_encoder.eval()
    print("\n[Forward] Evaluation mode:")
    with torch.no_grad():
        output_eval, _ = text_encoder(dummy_input, style_id=0)
        print(f"Eval output shape: {output_eval.shape}")
        print(f"Eval output mean: {output_eval.mean().item():.4f}")

    # Print parameter summary
    total_params = sum(p.data.numel() for p in text_encoder.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}")

    print("\n=== Successfully Ran Text Encoder ===\n")
