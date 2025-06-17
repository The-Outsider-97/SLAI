

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.utils.common import TensorOps, Parameter
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Task Heads")
printer = PrettyPrinter

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
        raise x

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

        self.ch_config = get_config_section('classification')
        self.num_classes = self.ch_config.get('num_classes')

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

        self.rh_config = get_config_section('regression')
        # self.num_classes = self.rh_config.get('num_classes')

        hidden_dim = hidden_dim
        self.layer_norm = torch.nn.LayerNorm(hidden_dim, eps=self.layer_norm_eps)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()  # For [0,1] outputs
        )

    def forward(self, hidden_states):
        x = torch.mean(hidden_states, dim=1) # Pooling
        x = super().forward(x) # Base processing
        
        return self.regressor(x)

class Seq2SeqHead(TaskHead):
    """Sequence-to-sequence task head with decoder and generator"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.config = load_global_config()
        self.dropout_rate = self.config.get('dropout_rate')
        self.embed_dim = self.config.get('embed_dim')
        self.num_heads = self.config.get('num_heads')
        self.ff_dim = self.config.get('ff_dim')
        self.num_styles = self.config.get('num_styles')
        self.max_position_embeddings = self.config.get('max_position_embeddings')

        self.s2sh_config = get_config_section('seq2seq')
        # self.decoder = self.s2sh_config.get('num_classes')

        self.decoder = None
        self.generator = None
        hidden_dim = hidden_dim
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states):
        # Decoder processing
        decoder_out = self.decoder(hidden_states)
        
        # Projection
        projected = self.proj(decoder_out)
        
        # Generate logits
        logits = self.generator(projected)
        return logits

class MultiModalClassificationHead(TaskHead):
    def __init__(self):
        super().__init__()
        """
        Args:
            hidden_dims: List of hidden dimensions for each modality.
            num_classes: Number of output classes.
            fusion_method: Method to fuse modalities ('concat', 'mean', etc.).
        """
        self.config = load_global_config()
        self.multi_config = get_config_section('multimodal')
        self.hidden_dims = self.multi_config.get('hidden_dims')
        self.fusion_method = self.multi_config.get('fusion_method')

        num_classes = None

        self.fusion_method = self.fusion_method
        if self.fusion_method == 'concat':
            total_dim = sum(self.hidden_dims)
        elif self.fusion_method == 'mean':
            total_dim = self.hidden_dims[0]  # Assumes all dims are the same for mean
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported.")

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
