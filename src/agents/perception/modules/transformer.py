import math
import torch
import yaml
import torch.nn as nn

from typing import Optional
from abc import ABC, abstractmethod

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.utils.taskheads import (TaskHead, ClassificationHead, MultiTaskHead,
                                RegressionHead, Seq2SeqHead, MultiModalClassificationHead)
from src.agents.base.utils.common import TensorOps, Parameter
from src.agents.perception.modules.attention import (EfficientAttention, BaseAttention,
                                        CosineAttention, MultiQueryAttention, CrossAttention)
from src.agents.perception.modules.feedforward import FeedForward
from src.agents.perception.perception_memory import PerceptionMemory
from src.agents.base.utils.base_transformer import BaseTransformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Transformer")
printer = PrettyPrinter

class Transformer(BaseTransformer, nn.Module):
    def __init__(self):
        BaseTransformer.__init__(self)
        nn.Module.__init__(self)
        self.config = load_global_config()
        self.training = self.config.get('training')
        self.embed_dim = self.config.get('embed_dim')
        self.num_heads = self.config.get('num_heads')
        self.dropout_rate = self.config.get('dropout_rate')
        self.num_layers = self.config.get('num_layers')
        self.num_styles = self.config.get('num_styles')
        self.max_position_embeddings = self.config.get('max_position_embeddings')

        self.trans_config = get_config_section('transformer')
        self.return_hidden = self.trans_config.get('return_hidden')

        # Override return_hidden if specified
        self.return_hidden = True

        self.memory = PerceptionMemory()

        # Choose attention type from config
        self.attn_selector = lambda input_shape, task_type=None, context=None: self.select_attention(
            input_shape=input_shape, task_type=task_type, context=context
        )

        # Choose taskhead type from config
        self.task_head = self.select_taskhead(task_type=self.config.get("task_type", "classification"))

        # Initialize modules
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            attn = self.attn_selector(input_shape=(1, self.max_position_embeddings, self.embed_dim), task_type="classification")
            self.layers.append(nn.ModuleDict({
                'attention': attn,
                'ff': FeedForward(),
                'norm1': nn.LayerNorm(self.embed_dim),
                'norm2': nn.LayerNorm(self.embed_dim),
            }))
        
        # Positional and style embeddings
        self.positional_encoding = Parameter(
            self._init_positional_encoding(
                self.embed_dim,
                self.max_position_embeddings
            ), 
            requires_grad=True
        )
        self.style_embeddings = Parameter(
            torch.randn(self.num_styles, self.embed_dim) * 0.02
        )
        
        # Apply weight initialization to our custom components
        self._init_custom_weights()
        logger.info(f"Transformer initialized with {self.num_layers} layers")

    def _init_positional_encoding(self, d_model, max_len=5000):
        """Sinusoidal positional encoding (Vaswani et al. 2017)"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_custom_weights(self):
        """Apply weight initialization to custom components"""
        for module in self.modules():
            # Skip base class modules
            if module is self:
                continue
                
            # Initialize our custom layers
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def select_attention(self, input_shape, task_type=None, context=None):
        """
        Attention selector implements:
        - Memory-aware selection
        - Device capability checks
        - Sequence length thresholds
        - Fallback mechanisms
        - Task-specific optimizations
        """
        # Get sequence length and device info
        seq_len = input_shape[1]
        device = self.config.get('device')
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        max_seq_len = self.max_position_embeddings

        # Calculate memory requirements
        approx_mem_req = (seq_len ** 2) * 4 * 4  # Approx memory in bytes
        device_mem = torch.cuda.get_device_properties(device).total_memory if 'cuda' in device else 4e9  # 4GB for CPU

        # Priority 1: Cross-attention when context is provided
        if context is not None:
            logger.debug("Selecting CrossAttention due to context presence")
            return CrossAttention()

        # Priority 2: Memory-efficient attention for long sequences
        if seq_len > 1024 or approx_mem_req > device_mem * 0.3:
            logger.debug(f"Selecting EfficientAttention for long sequence (length={seq_len})")
            return EfficientAttention()

        # Priority 3: Task-specific attention
        task_specific_map = {
            "similarity": CosineAttention(seq_len=seq_len),
            "classification": MultiQueryAttention() if seq_len <= 512 else BaseAttention(),
            "regression": BaseAttention(),
            "seq2seq": MultiQueryAttention(),
            "multimodal": CosineAttention(seq_len=seq_len)
        }

        if task_type in task_specific_map:
            logger.debug(f"Selecting {type(task_specific_map[task_type]).__name__} for task '{task_type}'")
            return task_specific_map[task_type]

        # Priority 4: Hardware-optimized selection
        if device_type == 'cuda' and seq_len > 512:
            logger.debug("Selecting MemoryEfficientAttention for CUDA device")
            return EfficientAttention()

        # Fallback to BaseAttention
        logger.debug("Selecting BaseAttention as default")
        return BaseAttention()    

    def set_task_head(self, task_type):
        self.task_head = self.select_taskhead(task_type)

    def select_taskhead(self, task_type=None, **kwargs):
        """
        Ttask head selector implements:
        - Multi-task support
        - Dynamic parameter configuration
        - Fallback mechanisms
        - Custom head registration
        """
        task_type = task_type or self.config.get("task_type", "classification")
        task_config = self.config.get("task_heads", {}).get(task_type, {})
        
        # Task head registry with default parameters
        task_head_registry = {
            "classification": {
                "class": ClassificationHead,
                "params": {"hidden_dim": self.embed_dim}
            },
            "regression": {
                "class": RegressionHead,
                "params": {"hidden_dim": self.embed_dim}
            },
            "seq2seq": {
                "class": Seq2SeqHead,
                "params": {"hidden_dim": self.embed_dim}
            },
            "multimodal": {
                "class": MultiModalClassificationHead,
                "params": {"hidden_dims": [self.embed_dim, self.embed_dim//2]}
            },
            "multitask": {
                "class": MultiTaskHead,
                "params": {"task_configs": task_config.get("tasks", [])}
            }
        }
        
        # Get task head specification
        head_spec = task_head_registry.get(task_type)
        
        if not head_spec:
            # Try partial match
            for key in task_head_registry:
                if key in task_type:
                    head_spec = task_head_registry[key]
                    logger.warning(f"Using partial match for task type '{task_type}' -> '{key}'")
                    break
        
        # Fallback to base TaskHead if no match found
        if not head_spec:
            logger.error(f"Unknown task type: {task_type}. Using base TaskHead")
            return TaskHead()
        
        # Merge parameters: default -> config overrides -> runtime kwargs
        params = head_spec["params"].copy()
        params.update(task_config.get("params", {}))
        params.update(kwargs)  # Add runtime parameters
        
        # Instantiate the task head
        logger.info(f"Initializing {head_spec['class'].__name__} for task '{task_type}'")
        return head_spec["class"](**params)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None,
                style_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add positional and style embeddings
        seq_len = x.shape[1]
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)

        # Handle style embeddings
        if style_id is None:
            style_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Get style embeddings and reshape for broadcasting
        style_emb = self.style_embeddings[style_id]  # Shape: (B, embed_dim)
        style_emb = style_emb.unsqueeze(1)  # Shape: (B, 1, embed_dim)
        x = x + style_emb
    
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            # Attention block
            residual = x
            x = layer['norm1'](x)
            x_attn = self.memory.run_checkpointed(layer['attention'], x)
            x = residual + nn.Dropout(self.dropout_rate)(x_attn)

            # Cache intermediate attention output
            self.memory(
                input=(f"layer_{i}_out", x),
                operation='update',
                tags=["layer_output", f"layer_{i}"]
            )

            # Feedforward block
            residual = x
            x = layer['norm2'](x)
            x_ff = self.memory.run_checkpointed(layer['ff'], x)
            x = residual + nn.Dropout(self.dropout_rate)(x_ff)

            # Cache final layer output
            self.memory(
                input=(f"layer_{i}_out", x),
                operation='update',
                tags=["layer_output", f"layer_{i}"]
            )

        if self.return_hidden:
            return x
        else:
            return self.task_head(x)

    def load_pretrained(self, weights):
        """Load weights with flexible key patterns"""
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
                    break  # Stop if successful
                except KeyError:
                    continue
            
            # Try different patterns for LayerNorm
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

if __name__ == "__main__":
    print("\n=== Running Transformer ===\n")
    model = Transformer()
    print(f"Model initialized with {model.num_layers} layers")

    x = torch.randn(4, 128, model.embed_dim)
    style_id = torch.tensor([0, 1, 2, 3])
    
    print(f"\nInput shape: {x.shape}")
    output = model(x, style_id)
    print("Output shape:", output.shape)

    print("\n* * * * * Phase 2 * * * * *\n")
    X = torch.tensor([[[5.0]]], dtype=torch.float32)
    style_id=torch.randint(0, model.num_styles, (X.shape[0],))

    printer.pretty("TEST2", model.forward(x=X, style_id=style_id))
    print("\n=== Successfully Ran Transformer ===\n")
