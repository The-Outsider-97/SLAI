import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Dict, Any

from .config_loader import load_global_config, get_config_section
from ...base.modules.activation_engine import get_activation
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Task Heads")
printer = PrettyPrinter


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, :x.size(1), :].to(x.device)


class TaskHead(nn.Module):
    """
    Enhanced abstract base class for task-specific heads.
    Provides common utilities: dropout, device handling, and parameter freezing.
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.device = self.config.get('device', 'cpu')
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Load task_heads config for layer_norm_eps
        self.taskheads_config = get_config_section('task_heads')
        self.layer_norm_eps = self.taskheads_config.get('layer_norm_eps', 1e-5)
        self.layer_norm = None  # To be set by child classes if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_norm:
            x = self.layer_norm(x)
        x = self.dropout(x)
        return x

    def freeze_parameters(self) -> None:
        """Disable gradient calculation for all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """Enable gradient calculation for all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class ClassificationHead(TaskHead):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ch_config = get_config_section('classification')
        self.num_classes = self.ch_config.get('num_classes', self.config.get('num_classes', 10))
        self.num_layers = self.ch_config.get('num_layers', 2)
        self.activation_name = self.ch_config.get('activation', 'relu')
        self.norm_type = self.ch_config.get('norm_type', 'layernorm')

        # Activation (from activation_engine)
        self.activation = get_activation(self.activation_name)

        # Normalization
        if self.norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim, eps=self.layer_norm_eps)
        elif self.norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = None

        # Build classifier MLP
        layers = []
        input_dim = hidden_dim
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(self.dropout_rate)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D (batch, features) and 3D (batch, sequence, features) inputs
        if x.dim() == 2:
            # Already 2D: apply norm directly
            if self.norm:
                if isinstance(self.norm, nn.BatchNorm1d):
                    x = self.norm(x)
                else:
                    x = self.norm(x)
            x = self.dropout(x)
            return self.classifier(x)

        # For 3D input (batch, seq, features)
        if self.norm:
            if isinstance(self.norm, nn.BatchNorm1d):
                # Flatten (B, T, D) -> (B*T, D) for BatchNorm
                B, T, D = x.shape
                x = x.view(B * T, D)
                x = self.norm(x)
                x = x.view(B, T, D)
            else:
                x = self.norm(x)

        x = self.dropout(x)

        # Pool sequence dimension
        x = x.mean(dim=1)

        return self.classifier(x)


class RegressionHead(TaskHead):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.rh_config = get_config_section('regression')
        self.num_layers = self.rh_config.get('num_layers', 2)
        self.activation_name = self.rh_config.get('activation', 'relu')
        self.norm_type = self.rh_config.get('norm_type', 'layernorm')
        self.use_skip = self.rh_config.get('use_skip', True)
        self.output_dim = self.rh_config.get('output_dim', 1)
        self.output_activation = self.rh_config.get('output_activation', None)

        # Activation
        self.activation = get_activation(self.activation_name)

        # Normalization
        if self.norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim, eps=self.layer_norm_eps)
        elif self.norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = None

        # Build MLP
        layers = []
        current_dim = hidden_dim
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self.activation,
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

        # Skip connection projection
        if self.use_skip and hidden_dim != self.output_dim:
            self.skip_proj = nn.Linear(hidden_dim, self.output_dim)
        else:
            self.skip_proj = None

        # Output activation
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

        # MLP
        x = self.mlp(x)

        # Skip connection
        if self.use_skip:
            if self.skip_proj:
                residual = self.skip_proj(residual)
            x = x + residual

        # Output activation
        if self.act_out:
            x = self.act_out(x)

        return x


class Seq2SeqHead(TaskHead):
    def __init__(self, hidden_dim: int, sos_token_id: int = 1, eos_token_id: int = 2):
        super().__init__()
        self.s2s_config = get_config_section('seq2seq')
        self.decoder_layers = self.s2s_config.get('decoder_layers', 1)
        self.tie_weights = self.s2s_config.get('tie_weights', True)
        self.max_length = self.s2s_config.get('max_length', self.config.get('max_position_embeddings', 5000))

        # Global config values
        self.embed_dim = self.config.get('embed_dim', 512)
        self.num_heads = self.config.get('num_heads', 8)
        self.ff_dim = self.config.get('ff_dim', 2048)
        self.vocab_size = self.config.get('vocab_size', 50000)
        self.activation_name = self.config.get('activation', 'relu')

        # Token IDs
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

        # Activation (for transformer)
        self.activation = get_activation(self.activation_name)

        # Decoder layers (PyTorch's TransformerDecoderLayer is used; it's standard)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_dim,
                dropout=self.dropout_rate,
                activation=self.activation_name,  # string; PyTorch expects 'relu' or 'gelu'
                batch_first=True
            )
            for _ in range(self.decoder_layers)
        ])

        # Embedding and positional encoding
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, self.max_length)

        # Output projection
        self.generator = nn.Linear(hidden_dim, self.vocab_size)

        # Weight tying
        if self.tie_weights:
            self.embedding.weight = self.generator.weight

    def forward(self,
                encoder_states: torch.Tensor,
                tgt_tokens: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tgt_tokens is None:
            batch_size = encoder_states.size(0)
            tgt_tokens = torch.full((batch_size, 1), self.sos_token_id,
                                    device=encoder_states.device, dtype=torch.long)

        tgt = self.embedding(tgt_tokens)
        tgt = self.positional_encoding(tgt)

        decoder_output = tgt
        for layer in self.decoder:
            decoder_output = layer(
                tgt=decoder_output,
                memory=encoder_states,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )

        return self.generator(decoder_output)

    def generate(self,
                 encoder_states: torch.Tensor,
                 max_length: Optional[int] = None,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 1.0) -> torch.Tensor:
        max_len = max_length or self.max_length
        batch_size = encoder_states.size(0)
        device = encoder_states.device

        tokens = torch.full((batch_size, 1), self.sos_token_id, device=device, dtype=torch.long)

        for _ in range(max_len):
            logits = self.forward(encoder_states, tokens)[:, -1, :]  # (batch, vocab)

            logits = logits / temperature

            if top_k > 0:
                logits = self._top_k_logits(logits, top_k)
            if top_p < 1.0:
                logits = self._top_p_logits(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=-1)

            if (next_token == self.eos_token_id).all():
                break

        return tokens

    @staticmethod
    def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
        if k == 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)

    @staticmethod
    def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
        if p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('Inf'))
        return logits


class MultiModalClassificationHead(TaskHead):
    def __init__(self, hidden_dims: List[int]):
        super().__init__()
        self.mm_config = get_config_section('multimodal')
        self.num_classes = self.mm_config.get('num_classes', self.config.get('num_classes', 10))
        self.num_layers = self.mm_config.get('num_layers', 2)
        self.fusion_method = self.mm_config.get('fusion_method', 'concat')
        self.use_attention = self.mm_config.get('use_attention', False)
        self.activation_name = self.mm_config.get('activation', 'relu')

        self.project_dim = self.config.get('embed_dim', 512)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)

        # Projection layers
        self.projections = nn.ModuleList()
        for dim in hidden_dims:
            self.projections.append(nn.Sequential(
                nn.Linear(dim, self.project_dim),
                nn.ReLU(),  # ReLU is standard for projection; could be made configurable
                nn.Dropout(self.dropout_rate)
            ))

        # Attention fusion (if used)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.project_dim,
                num_heads=4,
                dropout=self.dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(self.project_dim)

        # Determine fused dimension
        if self.fusion_method == 'concat':
            fused_dim = self.project_dim * len(hidden_dims)
        elif self.fusion_method in ['sum', 'mean', 'max']:
            fused_dim = self.project_dim
        elif self.fusion_method == 'bilinear':
            fused_dim = self.project_dim
            # Bilinear weights: one matrix per pair? Simpler: single weight matrix.
            self.bilinear_weight = nn.Parameter(torch.randn(self.project_dim, self.project_dim))
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # Classifier MLP
        self.activation = get_activation(self.activation_name)
        layers = []
        current_dim = fused_dim
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, self.project_dim),
                self.activation,
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = self.project_dim
        layers.append(nn.Linear(current_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, modality_tensors: List[torch.Tensor]) -> torch.Tensor:
        # Project each modality to common space
        projected = []
        for i, tensor in enumerate(modality_tensors):
            if tensor.ndim == 3:
                tensor = tensor[:, 0, :]  # use first token
            projected.append(self.projections[i](tensor))

        # Apply attention fusion
        if self.use_attention:
            stacked = torch.stack(projected, dim=1)  # (B, N, D)
            attn_output, _ = self.attention(stacked, stacked, stacked)
            fused = self.attention_norm(attn_output.mean(dim=1))
        else:
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
                        bilinear = torch.einsum('bd,de,be->b', projected[i], self.bilinear_weight, projected[j])
                        bilinear_out.append(bilinear)
                fused = torch.stack(bilinear_out, dim=1).mean(dim=1)

        return self.classifier(fused)


class MultiTaskHead(TaskHead):
    """
    Multi‑task head that wraps several task heads.
    Expects task_configs list of dicts, each with:
        - name: str
        - type: 'classification' or 'regression'
        - hidden_dim: int
        - [optional] additional parameters for the head
    """
    def __init__(self, task_configs: List[Dict[str, Any]]):
        super().__init__()
        self.heads = nn.ModuleDict()
        for cfg in task_configs:
            name = cfg['name']
            task_type = cfg['type']
            hidden_dim = cfg['hidden_dim']
            if task_type == 'classification':
                self.heads[name] = ClassificationHead(hidden_dim)
            elif task_type == 'regression':
                self.heads[name] = RegressionHead(hidden_dim)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Perception TaskHead Test ===\n")
    printer.status("Init", "Perception TaskHead initialized", "success")

    hidden_dim = 512
    classification = ClassificationHead(hidden_dim)
    regression = RegressionHead(hidden_dim)
    seq2seq = Seq2SeqHead(hidden_dim)
    multimodal = MultiModalClassificationHead([512, 64])

    printer.pretty("ClassificationHead", classification, "success")
    printer.pretty("RegressionHead", regression, "success")
    printer.pretty("Seq2SeqHead", seq2seq, "success")
    printer.pretty("MultiModalClassificationHead", multimodal, "success")

    print("\n=== Successfully Ran Perception TaskHead ===\n")