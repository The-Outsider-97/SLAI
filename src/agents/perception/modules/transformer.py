import math
import torch
import yaml

from abc import ABC, abstractmethod

from src.agents.perception.utils.common import TensorOps, Parameter  
from src.agents.perception.modules.attention import EfficientAttention  
from src.agents.perception.modules.feedforward import FeedForward  
from logs.logger import get_logger

logger = get_logger("Transformer")

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

class Transformer:
    def __init__(self, config):
        cfg = config.get('transformer', {})
        self.dropout_rate = cfg.get('dropout_rate', 0.1)
        self.embed_dim = cfg['embed_dim']
        self.layers = [
            {
                'attention': EfficientAttention(config),
                'ff': FeedForward(config),
                'norm1': Parameter(torch.ones(cfg['embed_dim'])),
                'norm2': Parameter(torch.ones(cfg['embed_dim'])),
            }
            for _ in range(cfg['num_layers'])
        ]
        self.positional_encoding = self._init_positional_encoding(
            cfg['embed_dim'], 
            cfg['max_position_embeddings']
        )
        self.style_embeddings = Parameter(
            torch.randn(cfg['num_styles'], cfg['embed_dim']) * 0.02)

    def parameters(self):
        params = [self.positional_encoding]
        for layer in self.layers:
            params.extend([
                layer['attention'].q_proj,
                layer['attention'].k_proj,
                layer['attention'].v_proj,
                layer['attention'].out_proj,
                layer['ff'].w1,
                layer['ff'].b1,
                layer['ff'].w2,
                layer['ff'].b2,
                layer['norm1'],
                layer['norm2']
            ])
        return params

    def _init_positional_encoding(self, d_model, max_len=5000):
        """Sinusoidal positional encoding (Vaswani et al. 2017)"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return Parameter(pe)  # (max_len, d_model)

    def forward(self, x, style_id, attention_mask=None, causal=False):
        """Implements transformer encoder forward pass"""
        seq_len = x.shape[1]
        x = x + self.positional_encoding.data.unsqueeze(0)[:, :seq_len, :]

        style_emb = self.style_embeddings.data[style_id].unsqueeze(0)

        for layer in self.layers:
            residual = x
            x = TensorOps.layer_norm(x + layer['norm1'].data)
            x = layer['attention'].forward(x, mask=attention_mask, causal=causal)
            x = residual + x

            residual = x
            x = TensorOps.layer_norm(x + layer['norm2'].data)
            x = x + style_emb.unsqueeze(1)
            x = layer['ff'].forward(x)
            x = residual + x

        return TensorOps.layer_norm(x)

    def backward(self, dout):
        """Backpropagation through transformer layers"""
        grads = []
        for layer in reversed(self.layers):
            # Backward through FFN
            d_ff = layer['ff'].backward(dout)
            norm2_grad = layer['norm2'].grad if layer['norm2'].grad is not None else 0
            d_norm2 = d_ff * (1 + norm2_grad)
            dout += d_norm2
            
            # Backward through attention
            d_attn = layer['attention'].backward(dout)
            norm1_grad = layer['norm1'].grad if layer['norm1'].grad is not None else 0
            d_norm1 = d_attn * (1 + norm1_grad)
            dout += d_norm1
            
        # Positional encoding gradients (non-trainable in original paper)
        return dout

    def load_pretrained(self, weights):
        """Load weights in Hugging Face-style format"""
        for i, layer in enumerate(self.layers):
            prefix = f'encoder.layer.{i}.'
            layer['attention'].q_proj.data = weights[f'{prefix}attention.self.query.weight']
            layer['attention'].k_proj.data = weights[f'{prefix}attention.self.key.weight']
            layer['attention'].v_proj.data = weights[f'{prefix}attention.self.value.weight']
            layer['attention'].out_proj.data = weights[f'{prefix}attention.output.dense.weight']
            layer['norm1'].data = weights[f'{prefix}attention.output.LayerNorm.weight']
            layer['ff'].w1.data = weights[f'{prefix}intermediate.dense.weight']
            layer['ff'].w2.data = weights[f'{prefix}output.dense.weight']
            layer['norm2'].data = weights[f'{prefix}output.LayerNorm.weight']

def softmax(logits):
    logits = logits - torch.max(logits)  # for numerical stability
    exp_logits = torch.exp(logits)
    return exp_logits / torch.sum(exp_logits)

def top_p_filter(probs, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)  # descending order
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    # Mask tokens beyond the top-p cumulative threshold
    cutoff_index = torch.searchsorted(cumulative_probs, p)
    mask = torch.zeros_like(probs)
    mask[sorted_indices[:cutoff_index + 1]] = 1

    filtered_probs = probs * mask
    filtered_probs = filtered_probs / torch.sum(filtered_probs)
    return filtered_probs

def top_k_filter(probs, k=10):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    mask = torch.zeros_like(probs)
    mask[sorted_indices[:k]] = 1

    filtered_probs = probs * mask
    filtered_probs = filtered_probs / torch.sum(filtered_probs)
    return filtered_probs

class Generator:
    def __init__(self, text_encoder, tokenizer, vocab_size, hidden_dim):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.output_head = Parameter(torch.randn(hidden_dim, vocab_size) * 0.02)

    def generate(self, prompt, max_length=50, temperature=1.0, top_k=0, top_p=0.0):
        encoded = self.tokenizer.encode(prompt)
        input_ids = encoded['input_ids'].tolist()
        attention_mask = encoded['attention_mask'].tolist()
        
        generated = input_ids.copy()

        for _ in range(max_length):
            # Prepare input
            input_tensor = torch.tensor(generated).unsqueeze(0)
            attention_tensor = torch.tensor(attention_mask).unsqueeze(0)

            # Run transformer in causal mode
            hidden_states = self.text_encoder.transformer.forward(
                x=torch.index_select(self.text_encoder.embedding.data, 0, input_tensor),
                style_id=0,
                attention_mask=attention_tensor,
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
                hidden_states = self.text_encoder.transformer.forward(
                    x=torch.index_select(self.text_encoder.embedding.data, 0, input_tensor),
                    style_id=0,
                    attention_mask=torch.ones_like(input_tensor),
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

class TaskHead(ABC):
    """Enhanced abstract base class for task-specific heads with common utilities"""
    def __init__(self, 
                 dropout_rate: float = 0.1, 
                 layer_norm_eps: float = 1e-5,
                 device: str = 'cpu'):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.device = device
        self.training = True
        
        # Common components
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.layer_norm = lambda x: TensorOps.layer_norm(x, eps=layer_norm_eps)

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass must be implemented by subclasses"""
        pass

    def parameters(self) -> list:
        """Return all trainable parameters"""
        params = []
        for attr in vars(self).values():
            if isinstance(attr, Parameter):
                params.append(attr)
            elif hasattr(attr, 'parameters'):  # For nested modules
                params.extend(attr.parameters())
        return params

    def to(self, device: str) -> None:
        """Move all parameters to specified device"""
        self.device = device
        for param in self.parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)

    def train(self) -> None:
        """Set to training mode"""
        self.training = True

    def eval(self) -> None:
        """Set to evaluation mode"""
        self.training = False

    def freeze_parameters(self) -> None:
        """Disable gradient calculation for all parameters"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """Enable gradient calculation for all parameters"""
        for param in self.parameters():
            param.requires_grad = True

    def _init_weights(self, module: Parameter, initializer: str = 'he') -> None:
        """Apply weight initialization to a parameter"""
        if initializer == 'he':
            std = math.sqrt(2.0 / module.data.shape[0])
            module.data.normal_(mean=0, std=std)
        elif initializer == 'xavier':
            torch.nn.init.xavier_uniform_(module.data)

class ClassificationHead(TaskHead):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.dense = Parameter(torch.randn(hidden_dim, num_classes) * 0.02)

    def forward(self, hidden_states):
        cls_output = hidden_states[:, 0, :]
        return torch.matmul(cls_output, self.dense.data)

class RegressionHead(TaskHead):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = Parameter(torch.randn(hidden_dim, 1) * 0.02)

    def forward(self, hidden_states):
        pooled = torch.mean(hidden_states, dim=1)
        return torch.matmul(pooled, self.dense.data)

class Seq2SeqHead(TaskHead):
    """Sequence-to-sequence task head with decoder and generator"""
    def __init__(self, 
                 text_encoder, 
                 tokenizer, 
                 config,
                 device: str = 'cpu'):
        seq2seq_cfg = config['task_heads']['seq2seq']
        decoder_cfg = config['task_heads']['seq2seq']['decoder']

        super().__init__(
            dropout_rate=seq2seq_cfg['dropout_rate'],
            device=device)
        self.vocab_size = tokenizer.vocab_size
        self.hidden_dim = text_encoder.embed_dim
        decoder_config = {
            'transformer': {
                'num_layers': decoder_cfg['num_layers'],
                'embed_dim': decoder_cfg['embed_dim'],
                'num_heads': decoder_cfg['num_heads'],
                'ff_dim': decoder_cfg['ff_dim'],
                'num_styles': decoder_cfg['num_styles'],
                'max_position_embeddings': decoder_cfg['max_position_embeddings']
            },
            'attention': config['attention'],
            'feedforward': config['feedforward']
        }
        self.decoder = Transformer(decoder_config)
        self.generator = Generator(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim)
        self.proj = Parameter(torch.randn(self.hidden_dim, self.hidden_dim) * 0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Decoder processing
        decoder_out = self.decoder.forward(
            hidden_states.to(self.device), 
            style_id=0, 
            causal=True
        )
        
        # Projection layer
        projected = torch.matmul(decoder_out, self.proj.data)
        
        # Generator processing
        logits = self.generator.output_head.data @ projected.transpose(-1, -2)
        return self.dropout(logits) if self.training else logits

class MultiModalClassificationHead(TaskHead):
    def __init__(self, hidden_dims, num_classes, fusion_method='concat'):
        super().__init__()
        """
        Args:
            hidden_dims: List of hidden dimensions for each modality.
            num_classes: Number of output classes.
            fusion_method: Method to fuse modalities ('concat', 'mean', etc.).
        """
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            total_dim = sum(hidden_dims)
        elif fusion_method == 'mean':
            total_dim = hidden_dims[0]  # Assumes all dims are the same for mean
        else:
            raise ValueError(f"Fusion method '{fusion_method}' not supported.")

        self.dense = Parameter(torch.randn(total_dim, num_classes) * 0.02)

    def forward(self, hidden_states_list):
        """
        Args:
            hidden_states_list: List of hidden states for each modality.
                Each tensor has shape (batch_size, seq_len, hidden_dim).
        """
        if self.fusion_method == 'concat':
            # Concatenate along the last dimension (hidden_dim)
            fused_states = torch.cat(hidden_states_list, dim=-1)
        elif self.fusion_method == 'mean':
            # Average the hidden states
            fused_states = torch.mean(torch.stack(hidden_states_list), dim=0)
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported.")

        # Use the first element in the sequence
        cls_output = fused_states[:, 0, :]
        return torch.matmul(cls_output, self.dense.data)

if __name__ == "__main__":
    print("\n=== Running Transformer ===\n")
    config = load_config()

    embed_dim = 200
    vocab_size = 50016
    transformer = Transformer(config)

    from src.agents.perception.modules.tokenizer import Tokenizer
    tokenizer = Tokenizer(config)

    generator = Generator(
        text_encoder=transformer,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        hidden_dim=embed_dim
    )
    task_heads = [
        ClassificationHead(hidden_dim=embed_dim, num_classes=10),
        RegressionHead(hidden_dim=embed_dim),
        Seq2SeqHead(
            text_encoder=transformer,
            tokenizer=tokenizer,
            config=config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ),
        MultiModalClassificationHead(
            hidden_dims=[embed_dim, 64],  # Example multimodal setup
            num_classes=10,
            fusion_method='concat'
        )
    ]

    for head in task_heads:
        head.to('cuda' if torch.cuda.is_available() else 'cpu')

    print("Enhanced TaskHeads initialized successfully with:")
    print(f"- Device awareness\n- Parameter freezing\n- Dropout/LayerNorm\n- Weight initialization\n")
    print("Transformer and Generator initialized successfully.")
    print(transformer)
    print(generator)
    print(task_heads)
    print("\n=== Successfully Ran Transformer ===\n")
