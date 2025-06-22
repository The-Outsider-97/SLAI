

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.utils.common import TensorOps, Parameter
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Task Heads")
printer = PrettyPrinter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class TaskHead(torch.nn.Module):
    """Enhanced abstract base class for task-specific heads with common utilities"""
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.device = self.config.get('device')
        self.training = self.config.get('training')
        self.dropout_rate = self.config.get('dropout_rate')
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.trans_config = get_config_section('task_heads')
        self.layer_norm_eps = self.trans_config.get('layer_norm_eps')
        self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_norm:
            x = self.layer_norm(x)
        x = self.dropout(x)
        return x

    def parameters(self) -> list:
        """Return all trainable parameters using PyTorch's native parameter tracking"""
        params = []
        for name, module in self.named_modules():
            if name == '':  # Skip self
                continue
            params += list(module.parameters())
        return params

    def to(self, device: str) -> None:
        """Move all parameters to specified device using PyTorch's native to()"""
        super().to(device)  # This handles module movement
        self.device = device

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
    def __init__(self, hidden_dim):
        super().__init__()
        self.config = load_global_config()
        self.dropout_rate = self.config.get('dropout_rate')
        self.num_layers = self.config.get('num_layers')
        self.activation = self.config.get('activation')
        self.norm_type = self.config.get('norm_type')
        self.num_classes = self.config.get('num_classes')

        self.ch_config = get_config_section('classification')

        self.hidden_dim = hidden_dim
        self.layer_norm = torch.nn.LayerNorm(hidden_dim, eps=self.layer_norm_eps)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, self.num_classes)
        )

        # Activation setup
        if isinstance(self.activation, str):
            self.activation = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid()
            }.get(self.activation.lower(), nn.ReLU())
        else:
            self.activation = self.activation

        # Optional normalization
        if self.norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim, eps=self.layer_norm_eps)
        elif self.norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = None

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # Classifier MLP
        layers = []
        input_dim = hidden_dim
        for i in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(self.dropout_rate)
            ])
            input_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(input_dim, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize across last dimension (B, ..., D)
        if self.norm:
            if isinstance(self.norm, nn.BatchNorm1d):
                # Flatten B x T x D → (B*T) x D for BatchNorm
                B, T, D = x.shape
                x = x.view(B * T, D)
                x = self.norm(x)
                x = x.view(B, T, D)
            else:
                x = self.norm(x)

        x = self.dropout(x)

        # Pool if more than 2 dimensions (i.e., sequence)
        if x.ndim > 2:
            x = x.mean(dim=1)  # Average pooling across sequence dimension

        return self.classifier(x)

class RegressionHead(TaskHead):
    def __init__(self, hidden_dim):
        super().__init__()
        self.config = load_global_config()
        self.dropout_rate = self.config.get('dropout_rate')
        self.num_layers = self.config.get('num_layers')
        self.activation = self.config.get('activation')
        self.norm_type = self.config.get('norm_type')
        self.dropout_rate = self.config.get('dropout_rate')

        self.rh_config = get_config_section('regression')
        self.use_skip = self.rh_config.get('use_skip')
        self.output_dim = self.rh_config.get('output_dim')
        
        # Initialize normalization
        if self.norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim, eps=self.layer_norm_eps)
        elif self.norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = None
        
        # Build MLP layers
        layers = []
        current_dim = hidden_dim
        
        for i in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self._get_activation(self.activation),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Final regression layer
        layers.append(nn.Linear(current_dim, self.output_dim))
        
        # Add skip connection if enabled
        if self.use_skip and hidden_dim != self.output_dim:
            self.skip_proj = nn.Linear(hidden_dim, self.output_dim)
        else:
            self.skip_proj = None
            
        self.mlp = nn.Sequential(*layers)
        
        # Output activation (sigmoid for bounded outputs, none for unbounded)
        self.output_activation = self.rh_config.get('output_activation')
        if self.output_activation == 'sigmoid':
            self.act_out = nn.Sigmoid()
        elif self.output_activation == 'tanh':
            self.act_out = nn.Tanh()
        else:
            self.act_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pool sequence dimension if needed
        if x.ndim > 2:
            x = x.mean(dim=1)
        
        # Apply normalization
        residual = x
        if self.norm:
            if isinstance(self.norm, nn.BatchNorm1d):
                x = self.norm(x)
            else:
                x = self.norm(x)
        
        # Process through MLP
        x = self.mlp(x)
        
        # Add skip connection
        if self.use_skip:
            if self.skip_proj:
                residual = self.skip_proj(residual)
            x = x + residual
        
        # Apply output activation if specified
        if self.act_out:
            x = self.act_out(x)
            
        return x

    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'swish': nn.SiLU()
        }
        return activations.get(name.lower(), nn.ReLU())

class Seq2SeqHead(TaskHead):
    """Sequence-to-sequence task head with decoder and generator"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.config = load_global_config()
        self.dropout_rate = self.config.get('dropout_rate')
        self.embed_dim = self.config.get('embed_dim')
        self.num_heads = self.config.get('num_heads')
        self.ff_dim = self.config.get('ff_dim')
        self.vocab_size = self.config.get('vocab_size')
        self.activation = self.config.get('activation')

        self.s2s_config = get_config_section('seq2seq')
        self.context_dim = self.s2s_config.get('context_dim', hidden_dim)
        self.decoder_layers = self.s2s_config.get('decoder_layers')
        self.tie_weights = self.s2s_config.get('tie_weights')

        self.max_length = self.embed_dim
        
        # Decoder components
        self.decoder = nn.ModuleList()
        for _ in range(self.decoder_layers):
            self.decoder.append(nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_dim,
                dropout=self.dropout_rate,
                activation=self.activation,
                batch_first=True
            ))
        
        # Output projection
        self.generator = nn.Linear(hidden_dim, self.vocab_size)
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, self.max_length)
        
        # Weight tying
        if self.tie_weights:
            self.embedding.weight = self.generator.weight

    def forward(
        self, 
        encoder_states: torch.Tensor, 
        tgt_tokens: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Prepare decoder inputs
        if tgt_tokens is None:
            # Inference mode - start with SOS token
            batch_size = encoder_states.size(0)
            tgt_tokens = torch.full((batch_size, 1), self.tokenizer.sos_token_id, 
                                   device=encoder_states.device)
        
        # Embed target tokens
        tgt = self.embedding(tgt_tokens)
        tgt = self.positional_encoding(tgt)
        
        # Process through decoder layers
        decoder_output = tgt
        for layer in self.decoder:
            decoder_output = layer(
                tgt=decoder_output,
                memory=encoder_states,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )
        
        # Generate logits
        logits = self.generator(decoder_output)
        return logits

    def generate(
        self, 
        encoder_states: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """Autoregressive sequence generation"""
        max_len = max_length or self.max_length
        batch_size = encoder_states.size(0)
        device = encoder_states.device
        
        # Initialize with SOS token
        tokens = torch.full((batch_size, 1), self.tokenizer.sos_token_id, device=device)
        
        for _ in range(max_len):
            # Get logits for current sequence
            logits = self.forward(encoder_states, tokens)[:, -1, :]
            
            # Apply temperature scaling
            logits = logits / temperature
            
            # Apply top-k/top-p filtering
            if top_k > 0:
                logits = top_k_logits(logits, top_k)
            if top_p < 1.0:
                logits = top_p_logits(logits, top_p)
                
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=-1)
            
            # Stop if all sequences generated EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
                
        return tokens

class MultiModalClassificationHead(TaskHead):
    def __init__(self, hidden_dims: List[int]):
        super().__init__()
        """
        Args:
            hidden_dims: List of hidden dimensions for each modality.
            num_classes: Number of output classes.
            fusion_method: Method to fuse modalities ('concat', 'mean', etc.).
        """
        self.config = load_global_config()
        self.embed_dim = self.config.get('embed_dim')
        self.num_layers = self.config.get('num_layers')
        self.dropout_rate = self.config.get('dropout_rate')
        self.activation = self.config.get('activation')
        self.num_classes = self.config.get('num_classes')

        self.mm_config = get_config_section('multimodal')
        self.fusion_method = self.mm_config.get('fusion_method')
        self.use_attention = self.mm_config.get('use_attention')

        self.project_dim = self.embed_dim
        
        # Validate input dimensions
        if len(hidden_dims) < 2:
            raise ValueError("MultiModalClassificationHead requires at least two modalities")
        
        # Projection layers for each modality
        self.projections = nn.ModuleList()
        for dim in hidden_dims:
            self.projections.append(nn.Sequential(
                nn.Linear(dim, self.project_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ))
        
        # Fusion mechanism
        if self.fusion_method == 'concat':
            fused_dim = self.project_dim * len(hidden_dims)
        elif self.fusion_method in ['sum', 'mean', 'max']:
            fused_dim = self.project_dim
        elif self.fusion_method == 'bilinear':
            fused_dim = self.project_dim
            self.bilinear_weights = Parameter(
                torch.randn(self.project_dim, self.project_dim, len(hidden_dims)))
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # Attention fusion
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.project_dim,
                num_heads=4,
                dropout=self.dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(self.project_dim)
            fused_dim = self.project_dim  # Attention outputs single vector per modality
        
        # Classifier MLP
        layers = []
        current_dim = fused_dim
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, self.project_dim),
                self._get_activation(self.activation),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = self.project_dim
        
        layers.append(nn.Linear(current_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, modality_tensors: List[torch.Tensor]) -> torch.Tensor:
        # Project all modalities to common space
        projected = []
        for i, tensor in enumerate(modality_tensors):
            # Handle sequence inputs (use first token)
            if tensor.ndim == 3:
                tensor = tensor[:, 0, :]
            projected.append(self.projections[i](tensor))
        
        # Apply attention fusion
        if self.use_attention:
            # Stack projected features (B, N, D)
            stacked = torch.stack(projected, dim=1)
            
            # Compute self-attention
            attn_output, _ = self.attention(
                stacked, stacked, stacked
            )
            fused = self.attention_norm(attn_output.mean(dim=1))
        else:
            # Apply fusion method
            if self.fusion_method == 'concat':
                fused = torch.cat(projected, dim=-1)
            elif self.fusion_method == 'sum':
                fused = torch.stack(projected, dim=0).sum(dim=0)
            elif self.fusion_method == 'mean':
                fused = torch.stack(projected, dim=0).mean(dim=0)
            elif self.fusion_method == 'max':
                fused = torch.stack(projected, dim=0).max(dim=0).values
            elif self.fusion_method == 'bilinear':
                bilinear_out = []
                for i in range(len(projected)):
                    for j in range(i+1, len(projected)):
                        bilinear = torch.einsum(
                            'bd,de,be->b', 
                            projected[i], 
                            self.bilinear_weights[:, :, i], 
                            projected[j]
                        )
                        bilinear_out.append(bilinear)
                fused = torch.stack(bilinear_out, dim=1).mean(dim=1)
        
        # Classify fused representation
        return self.classifier(fused)

    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'swish': nn.SiLU()
        }
        return activations.get(name.lower(), nn.ReLU())

# To be later enhanced
class MultiTaskHead(TaskHead):
    def __init__(self, task_configs):
        super().__init__()
        self.heads = nn.ModuleDict()
        for config in task_configs:
            task_type = config["type"]
            if task_type == "classification":
                self.heads[config["name"]] = ClassificationHead(config["hidden_dim"])
            elif task_type == "regression":
                self.heads[config["name"]] = RegressionHead(config["hidden_dim"])
            # Add other task types as needed
    
    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}

if __name__ == "__main__":
    print("\n=== Running Perception TaskHead Test ===\n")
    printer.status("Init", "Perception TaskHead initialized", "success")
    main = TaskHead()

    print(main)
    print("\n* * * * * Phase 2 - Child heads * * * * *\n")
    hidden_dim=512
    classification = ClassificationHead(hidden_dim=hidden_dim)
    regression = RegressionHead(hidden_dim=hidden_dim)
    seq2seq = Seq2SeqHead(hidden_dim=hidden_dim)
    multimodal = MultiModalClassificationHead(hidden_dims=[512, 64])

    printer.pretty("CHILD1", classification, "success")
    printer.pretty("CHILD2", regression, "success")
    printer.pretty("CHILD3", seq2seq, "success")
    printer.pretty("CHILD4", multimodal, "success")

    print("\n=== Successfully Ran Perception TaskHead ===\n")
