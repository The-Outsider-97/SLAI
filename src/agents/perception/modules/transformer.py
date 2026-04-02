import math
import torch
import torch.nn as nn

from typing import Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.taskheads import (
    TaskHead, ClassificationHead, MultiTaskHead,
    RegressionHead, Seq2SeqHead, MultiModalClassificationHead
)
from ..utils.common import TensorOps, Parameter
from .attention import (
    EfficientAttention, BaseAttention,
    CosineAttention, MultiQueryAttention, CrossAttention
)
from .feedforward import FeedForward
from ..perception_memory import PerceptionMemory
from ...base.utils.base_transformer import BaseTransformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Transformer")
printer = PrettyPrinter


class Transformer(BaseTransformer):
    """
    Enhanced transformer with custom attention, feedforward, task heads,
    and production‑ready features like gradient checkpointing and layer freezing.
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.embed_dim = self.config.get('embed_dim')
        self.num_heads = self.config.get('num_heads')
        self.dropout_rate = self.config.get('dropout_rate')
        self.num_layers = self.config.get('num_layers')
        self.num_styles = self.config.get('num_styles')
        self.max_position_embeddings = self.config.get('max_position_embeddings')
        self.trans_config = get_config_section('transformer')
        self.return_hidden = self.trans_config.get('return_hidden', False)
        self.use_checkpointing = self.trans_config.get('use_gradient_checkpointing', True)

        self.memory = PerceptionMemory(enable_checkpointing=self.use_checkpointing)

        # Style embeddings (additional token‑level style control)
        self.style_embeddings = Parameter(
            torch.randn(self.num_styles, self.embed_dim) * 0.02
        )

        # Build custom layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            attn = self.select_attention(
                input_shape=(1, self.max_position_embeddings, self.embed_dim),
                task_type="classification"
            )
            self.layers.append(nn.ModuleDict({
                'attention': attn,
                'ff': FeedForward(),
                'norm1': nn.LayerNorm(self.embed_dim),
                'norm2': nn.LayerNorm(self.embed_dim),
            }))

        # Task head
        self.task_head = self.select_taskhead(task_type=self.config.get("task_type", "classification"))

        # Initialize custom weights (but base already initialised some; we may override)
        self._init_custom_weights()

        logger.info(f"Transformer initialized with {self.num_layers} layers, "
                    f"checkpointing={'on' if self.use_checkpointing else 'off'}")

    def _init_custom_weights(self):
        """Apply custom weight initialization to all modules (avoids bias for embeddings)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def select_attention(self, input_shape, task_type=None, context=None):
        """
        Attention selector based on sequence length, device, and task type.
        Uses config flags for bucket sizes and memory efficiency.
        """
        seq_len = input_shape[1]
        device = self.config.get('device', 'cpu')
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        max_seq_len = self.max_position_embeddings

        # Approximate memory requirement
        approx_mem_req = (seq_len ** 2) * 4 * 4  # bytes (rough)
        device_mem = torch.cuda.get_device_properties(device).total_memory if device_type == 'cuda' else 4e9

        # Cross‑attention if context is provided
        if context is not None:
            logger.debug("Selecting CrossAttention due to context presence")
            return CrossAttention()

        # Memory‑efficient for long sequences
        if seq_len > 1024 or approx_mem_req > device_mem * 0.3:
            logger.debug(f"Selecting EfficientAttention for long sequence (len={seq_len})")
            return EfficientAttention()

        # Task‑specific choices
        task_map = {
            "similarity": CosineAttention(seq_len=seq_len),
            "classification": MultiQueryAttention() if seq_len <= 512 else BaseAttention(),
            "regression": BaseAttention(),
            "seq2seq": MultiQueryAttention(),
            "multimodal": CosineAttention(seq_len=seq_len)
        }
        if task_type in task_map:
            logger.debug(f"Selecting {type(task_map[task_type]).__name__} for task '{task_type}'")
            return task_map[task_type]

        # Hardware‑optimized
        if device_type == 'cuda' and seq_len > 512:
            logger.debug("Selecting EfficientAttention for CUDA device")
            return EfficientAttention()

        # Default
        logger.debug("Selecting BaseAttention as default")
        return BaseAttention()

    def select_taskhead(self, task_type=None, **kwargs):
        """Select the appropriate task head."""
        task_type = task_type or self.config.get("task_type", "classification")
        task_config = self.config.get("task_heads", {}).get(task_type, {})

        registry = {
            "classification": (ClassificationHead, {"hidden_dim": self.embed_dim}),
            "regression": (RegressionHead, {"hidden_dim": self.embed_dim}),
            "seq2seq": (Seq2SeqHead, {"hidden_dim": self.embed_dim}),
            "multimodal": (MultiModalClassificationHead, {"hidden_dims": [self.embed_dim, self.embed_dim//2]}),
            "multitask": (MultiTaskHead, {"task_configs": task_config.get("tasks", [])})
        }

        if task_type in registry:
            head_class, default_params = registry[task_type]
            params = default_params.copy()
            params.update(task_config.get("params", {}))
            params.update(kwargs)
            logger.info(f"Initializing {head_class.__name__} for task '{task_type}'")
            return head_class(**params)

        # Fallback
        logger.error(f"Unknown task type: {task_type}. Using base TaskHead.")
        return TaskHead()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            context: Optional context for cross‑attention
            context_mask: Attention mask for context
            style_id: Style embedding IDs (batch,)
            attention_mask: Alias for context_mask

        Returns:
            Output tensor (batch, seq_len, embed_dim) if return_hidden, else task head output.
        """
        if context_mask is None and attention_mask is not None:
            context_mask = attention_mask

        # Add positional encoding (from base class)
        if hasattr(self, 'pos_encoder'):
            x = self.pos_encoder(x)
        else:
            # Fallback (should not happen)
            logger.warning("No positional encoder found; using identity.")

        # Add style embeddings
        if style_id is None:
            style_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        style_emb = self.style_embeddings[style_id].unsqueeze(1)  # (batch, 1, embed_dim)
        x = x + style_emb

        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            # Attention block
            residual = x
            x = layer['norm1'](x)
            attn_out = self.memory.run_checkpointed(
                layer['attention'],
                x,
                context,
                context_mask
            )
            # If attention returns (output, weights), take output
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = residual + nn.functional.dropout(attn_out, p=self.dropout_rate, training=self.training)

            # Cache attention output (optional, for debugging)
            if self.training:
                self.memory.cache_item(
                    tensor=x,
                    key=f"layer_{i}_attn_out",
                    tags=["layer_output", f"layer_{i}"],
                    metadata={"block": "attention"}
                )

            # Feedforward block
            residual = x
            x = layer['norm2'](x)
            ff_out = self.memory.run_checkpointed(layer['ff'], x, context)
            x = residual + nn.functional.dropout(ff_out, p=self.dropout_rate, training=self.training)

            if self.training:
                self.memory.cache_item(
                    tensor=x,
                    key=f"layer_{i}_ff_out",
                    tags=["layer_output", f"layer_{i}"],
                    metadata={"block": "feedforward"}
                )

        if self.return_hidden:
            return x
        else:
            return self.task_head(x)

    def freeze_layers(self, layer_indices: Optional[list] = None):
        """
        Freeze specific layers (or all layers) to prevent gradient updates.
        Useful for fine‑tuning.
        """
        if layer_indices is None:
            # Freeze all custom layers
            for param in self.layers.parameters():
                param.requires_grad = False
            for param in self.style_embeddings.parameters():
                param.requires_grad = False
            for param in self.task_head.parameters():
                param.requires_grad = False
            logger.info("All custom layers frozen.")
        else:
            for idx in layer_indices:
                if 0 <= idx < len(self.layers):
                    for param in self.layers[idx].parameters():
                        param.requires_grad = False
                    logger.info(f"Layer {idx} frozen.")
                else:
                    logger.warning(f"Layer index {idx} out of range.")

    def unfreeze_layers(self, layer_indices: Optional[list] = None):
        """Unfreeze specific layers (or all layers)."""
        if layer_indices is None:
            for param in self.layers.parameters():
                param.requires_grad = True
            for param in self.style_embeddings.parameters():
                param.requires_grad = True
            for param in self.task_head.parameters():
                param.requires_grad = True
            logger.info("All custom layers unfrozen.")
        else:
            for idx in layer_indices:
                if 0 <= idx < len(self.layers):
                    for param in self.layers[idx].parameters():
                        param.requires_grad = True
                    logger.info(f"Layer {idx} unfrozen.")
                else:
                    logger.warning(f"Layer index {idx} out of range.")

    def load_pretrained(self, weights: dict):
        """Load pretrained weights from a dictionary (e.g., from HuggingFace)."""
        for i, layer in enumerate(self.layers):
            # Try different key patterns for attention
            attn_prefixes = [
                f'encoder.layer.{i}.',
                f'layers.{i}.',
                f'transformer.layer_{i}.'
            ]
            for prefix in attn_prefixes:
                try:
                    layer['attention'].load_from_dict(weights, prefix)
                    break
                except KeyError:
                    continue

            # Load layer norms
            norm_patterns = {
                'norm1': [
                    f'{prefix}attention.output.LayerNorm.weight',
                    f'{prefix}attention.output.layernorm.weight',
                    f'{prefix}attn_norm.weight'
                ],
                'norm2': [
                    f'{prefix}output.LayerNorm.weight',
                    f'{prefix}ffn_norm.weight'
                ]
            }
            for norm_name, patterns in norm_patterns.items():
                for pattern in patterns:
                    if pattern in weights:
                        layer[norm_name].weight.data = weights[pattern]
                        break
                for pattern in patterns:
                    bias_pattern = pattern.replace('.weight', '.bias')
                    if bias_pattern in weights:
                        layer[norm_name].bias.data = weights[bias_pattern]
                        break


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Transformer ===\n")
    model = Transformer()
    print(f"Model initialized with {model.num_layers} layers")

    x = torch.randn(4, 128, model.embed_dim)
    style_id = torch.tensor([0, 1, 2, 3])
    print(f"\nInput shape: {x.shape}")
    output = model(x, style_id=style_id)
    print("Output shape:", output.shape)

    # Test with return_hidden
    model.return_hidden = True
    hidden = model(x, style_id=style_id)
    print("Hidden shape:", hidden.shape)

    # Test freezing
    model.freeze_layers([0, 1])
    model.unfreeze_layers([2])

    print("\n=== Successfully Ran Transformer ===\n")