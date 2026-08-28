"""
Transformer module for the Perception Agent's subsystem
"""

from __future__ import annotations

import torch
import torch.nn as nn

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Union
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.perception_errors import *
from ..utils.perception_helpers import *
from ..perception_memory import *
from .attention import *
from .feedforward import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Transformer")
printer = PrettyPrinter()


def _build_attention(
    attention_type: str,
    *,
    causal: bool,
    max_sequence_length: int,
) -> nn.Module:
    """Construct a deterministic attention implementation.

    Attention policy is resolved once at model construction.  Parameterized
    attention modules are never swapped at runtime.
    """
    kind = str(attention_type).strip().lower()

    if kind == "base":
        return BaseAttention(causal=causal)

    if kind in {"multi_query", "multi-query", "mqa"}:
        return MultiQueryAttention(causal=causal)

    if kind == "cosine":
        return CosineAttention(
            seq_len=max_sequence_length,
            causal=causal,
        )

    if kind == "efficient":
        if causal:
            raise UnsupportedPerceptionOptionError(
                "EfficientAttention cannot be used for causal decoding.",
                component="transformer",
                remediation=(
                    "Use attention.type='base' or 'multi_query' for a causal "
                    "Transformer."
                ),
            )
        return EfficientAttention(causal=False)

    raise UnsupportedPerceptionOptionError(
        f"Unsupported attention type: {attention_type!r}.",
        component="transformer",
        details={
            "supported": (
                "base",
                "multi_query",
                "cosine",
                "efficient",
            )
        },
    )


class _TransformerLayer(nn.Module):
    """One Pre-LN Transformer encoder/decoder layer.

    Encoder mode:
        self-attention -> FFN

    Decoder mode:
        causal self-attention -> cross-attention -> FFN

    The layer is the sole owner of normalization and residual connections.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        dropout: float,
        self_attention: nn.Module,
        feedforward: nn.Module,
        cross_attention: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise PerceptionConfigurationError(
                "Transformer layer embed_dim must be positive.",
                component="transformer",
                details={"embed_dim": embed_dim},
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise PerceptionConfigurationError(
                "Transformer layer dropout must be in [0,1).",
                component="transformer",
                details={"dropout": dropout},
            )

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward

        self.norm_self = nn.LayerNorm(embed_dim)
        self.norm_cross = (
            nn.LayerNorm(embed_dim)
            if cross_attention is not None
            else None
        )
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(float(dropout))

    # ------------------------------------------------------------------
    # Compatibility aliases
    # ------------------------------------------------------------------
    # Existing SLAI encoder hook code accesses transformer layers using the old
    # ModuleDict keys.  Preserve that read-only surface while the internal layer
    # becomes a proper module with explicit responsibilities.
    def __contains__(self, key: str) -> bool:
        return key in {
            "attention",
            "cross_attention",
            "ff",
            "norm1",
            "norm2",
        }

    def __getitem__(self, key: str) -> nn.Module:
        mapping = {
            "attention": self.self_attention,
            "cross_attention": self.cross_attention,
            "ff": self.feedforward,
            "norm1": self.norm_self,
            "norm2": self.norm_ff,
        }
        if key not in mapping or mapping[key] is None:
            raise KeyError(key)
        return mapping[key]  # type: ignore[return-value]

    @staticmethod
    def _attention_output(value: Any) -> torch.Tensor:
        output = value[0] if isinstance(value, tuple) else value
        if not isinstance(output, torch.Tensor):
            raise PerceptionContractError(
                "Attention module returned a non-tensor output.",
                component="transformer",
                details={"actual_type": type(output).__name__},
            )
        return output

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Pre-LN self-attention.
        residual = x
        normalized = self.norm_self(x)
        attended = self.self_attention(
            normalized,
            mask=attention_mask,
        )
        x = residual + self.dropout(
            self._attention_output(attended)
        )

        # 2. Optional encoder-memory cross-attention.
        if self.cross_attention is not None:
            if context is None:
                raise PerceptionContractError(
                    "Cross-attention is enabled but no context was provided.",
                    component="transformer",
                    remediation=(
                        "Provide encoder/context states or construct the "
                        "Transformer with enable_cross_attention=False."
                    ),
                )

            residual = x
            assert self.norm_cross is not None
            normalized = self.norm_cross(x)
            attended = self.cross_attention(
                normalized,
                context=context,
                mask=context_mask,
            )
            x = residual + self.dropout(
                self._attention_output(attended)
            )

        # 3. Position-wise FFN. The FFN itself has no residual/norm.
        residual = x
        normalized = self.norm_ff(x)
        x = residual + self.dropout(
            self.feedforward(normalized)
        )

        return x


class Transformer(nn.Module):
    """Perception Transformer backbone.

    This is intentionally a standalone ``nn.Module`` rather than a subclass of
    ``BaseTransformer``.  It owns only the custom perception stack and does not
    instantiate a second unused encoder/decoder Transformer.

    ``Transformer`` always returns hidden states with shape ``(B,L,D)``.
    Downstream task heads belong to PerceptionTrainer/PerceptionAgent.
    """

    def __init__(
        self,
        *,
        causal: bool = False,
        enable_cross_attention: bool = False,
        attention_type: Optional[str] = None,
        return_hidden: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.config = load_global_config()
        self.trans_config = get_config_section("transformer") or {}
        self.attention_config = get_config_section("attention") or {}

        self.embed_dim = int(self.config.get("embed_dim", 512))
        self.num_heads = int(self.config.get("num_heads", 8))
        self.num_layers = int(self.config.get("num_layers", 4))
        self.num_styles = int(self.config.get("num_styles", 0))
        self.dropout_rate = float(self.config.get("dropout_rate", 0.1))
        self.max_position_embeddings = int(self.config.get("max_position_embeddings", 5000))
        self.device = resolve_torch_device(self.config.get("device", "cpu"))
        self.use_checkpointing = bool(self.trans_config.get("use_gradient_checkpointing", True))
        
        # Optional transient runtime cache.
        # This is NOT durable checkpoint persistence.
        self.enable_runtime_cache = bool(self.trans_config.get("enable_runtime_cache", False))
        
        self.memory = (
            PerceptionMemory(enable_checkpointing=False)
            if self.enable_runtime_cache
            else None
        )

        self._validate_config()

        self._causal = bool(causal)
        self.enable_cross_attention = bool(
            enable_cross_attention
        )

        self.attention_type = (
            str(attention_type).strip().lower()
            if attention_type is not None
            else str(
                self.attention_config.get(
                    "type",
                    "base",
                )
            ).strip().lower()
        )

        # The legacy flag remains as a compatibility attribute, but there is no
        # task head in this module anymore. Hidden states are therefore the only
        # valid Transformer output.
        requested_return_hidden = (
            bool(return_hidden)
            if return_hidden is not None
            else bool(
                self.trans_config.get(
                    "return_hidden",
                    True,
                )
            )
        )
        if not requested_return_hidden:
            logger.debug(
                "transformer.return_hidden=False is ignored: the perception "
                "Transformer no longer owns task heads and always returns hidden states."
            )
        self.return_hidden = True

        self.use_checkpointing = bool(
            self.trans_config.get(
                "use_gradient_checkpointing",
                True,
            )
        )
        self.preserve_rng_state = bool(
            self.trans_config.get(
                "preserve_rng_state",
                True,
            )
        )

        # Style conditioning remains here because current VisionEncoder and
        # AudioEncoder delegate style_id directly to this backbone.
        self.style_embeddings = nn.Parameter(
            torch.empty(
                self.num_styles,
                self.embed_dim,
                device=self.device,
            )
        )
        if self.num_styles > 0:
            nn.init.normal_(
                self.style_embeddings,
                mean=0.0,
                std=0.02,
            )

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self_attention = _build_attention(
                self.attention_type,
                causal=self._causal,
                max_sequence_length=(
                    self.max_position_embeddings
                ),
            )
            cross_attention = (
                CrossAttention()
                if self.enable_cross_attention
                else None
            )

            self.layers.append(
                _TransformerLayer(
                    embed_dim=self.embed_dim,
                    dropout=self.dropout_rate,
                    self_attention=self_attention,
                    cross_attention=cross_attention,
                    feedforward=FeedForward(),
                )
            )

        self._output_attentions = False
        self._initialize_layer_norms()
        self.to(self.device)

        logger.info(
            "Transformer initialized: layers=%s, attention=%s, causal=%s, "
            "cross_attention=%s, checkpointing=%s",
            self.num_layers,
            self.attention_type,
            self._causal,
            self.enable_cross_attention,
            self.use_checkpointing,
        )

    def _validate_config(self) -> None:
        if self.embed_dim <= 0:
            raise PerceptionConfigurationError(
                "embed_dim must be positive.",
                component="transformer",
                details={"embed_dim": self.embed_dim},
            )
        if self.num_heads <= 0:
            raise PerceptionConfigurationError(
                "num_heads must be positive.",
                component="transformer",
                details={"num_heads": self.num_heads},
            )
        if self.embed_dim % self.num_heads != 0:
            raise PerceptionConfigurationError(
                "embed_dim must be divisible by num_heads.",
                component="transformer",
                details={
                    "embed_dim": self.embed_dim,
                    "num_heads": self.num_heads,
                },
            )
        if self.num_layers <= 0:
            raise PerceptionConfigurationError(
                "num_layers must be positive.",
                component="transformer",
                details={"num_layers": self.num_layers},
            )
        if self.num_styles < 0:
            raise PerceptionConfigurationError(
                "num_styles must be >= 0.",
                component="transformer",
                details={"num_styles": self.num_styles},
            )
        if self.max_position_embeddings <= 0:
            raise PerceptionConfigurationError(
                "max_position_embeddings must be positive.",
                component="transformer",
                details={
                    "max_position_embeddings":
                        self.max_position_embeddings
                },
            )
        if not 0.0 <= self.dropout_rate < 1.0:
            raise PerceptionConfigurationError(
                "dropout_rate must be in [0,1).",
                component="transformer",
                details={
                    "dropout_rate": self.dropout_rate
                },
            )

    def _initialize_layer_norms(self) -> None:
        # Attention and FFN modules initialize their own parameters.  Do not
        # globally reinitialize them here; the previous implementation did so
        # and silently overrode the configured initializer.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                with torch.no_grad():
                    module.weight.fill_(1.0)
                    module.bias.zero_()

    # ------------------------------------------------------------------
    # Runtime properties
    # ------------------------------------------------------------------
    @property
    def causal(self) -> bool:
        return self._causal

    @causal.setter
    def causal(self, value: bool) -> None:
        new_value = bool(value)

        if new_value:
            for layer in getattr(
                self,
                "layers",
                (),
            ):
                if isinstance(
                    layer.self_attention,
                    EfficientAttention,
                ):
                    raise UnsupportedPerceptionOptionError(
                        "EfficientAttention cannot be switched to causal mode.",
                        component="transformer",
                        remediation=(
                            "Construct the Transformer with attention_type='base' "
                            "or 'multi_query'."
                        ),
                    )

        self._causal = new_value
        for layer in getattr(
            self,
            "layers",
            (),
        ):
            if hasattr(
                layer.self_attention,
                "causal",
            ):
                layer.self_attention.causal = (
                    new_value
                )

    @property
    def output_attentions(self) -> bool:
        return self._output_attentions

    @output_attentions.setter
    def output_attentions(
        self,
        value: bool,
    ) -> None:
        enabled = bool(value)
        self._output_attentions = enabled

        for layer in getattr(
            self,
            "layers",
            (),
        ):
            if hasattr(
                layer.self_attention,
                "output_attentions",
            ):
                layer.self_attention.output_attentions = (
                    enabled
                )

            if (
                layer.cross_attention is not None
                and hasattr(
                    layer.cross_attention,
                    "output_attentions",
                )
            ):
                layer.cross_attention.output_attentions = (
                    enabled
                )

    # ------------------------------------------------------------------
    # Input normalization / validation
    # ------------------------------------------------------------------
    def _normalize_style_id(
        self,
        style_id: Optional[Any],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if style_id is None:
            return None

        if self.num_styles <= 0:
            raise PerceptionContractError(
                "style_id was provided but num_styles is 0.",
                component="transformer",
            )

        if not isinstance(style_id, torch.Tensor):
            style_id = torch.as_tensor(
                style_id,
                dtype=torch.long,
                device=device,
            )
        else:
            style_id = style_id.to(
                device=device,
                dtype=torch.long,
            )

        if style_id.dim() == 0:
            style_id = style_id.expand(
                batch_size
            )
        elif (
            style_id.dim() == 2
            and style_id.size(1) == 1
        ):
            style_id = style_id.squeeze(1)

        if (
            style_id.dim() != 1
            or style_id.size(0) != batch_size
        ):
            raise PerceptionShapeError(
                "style_id must be scalar or have shape (B,).",
                component="transformer",
                details={
                    "shape": list(style_id.shape),
                    "batch_size": batch_size,
                },
            )

        if bool(
            (
                (style_id < 0)
                | (style_id >= self.num_styles)
            ).any().item()
        ):
            raise PerceptionRangeError(
                "style_id contains an out-of-range value.",
                component="transformer",
                details={
                    "num_styles": self.num_styles,
                    "minimum": int(
                        style_id.min().item()
                    ),
                    "maximum": int(
                        style_id.max().item()
                    ),
                },
            )

        return style_id

    def _validate_inputs(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> None:
        if not isinstance(
            x,
            torch.Tensor,
        ) or x.dim() != 3:
            raise PerceptionShapeError(
                "Transformer input must have shape (B,L,D).",
                component="transformer",
                details={
                    "shape":
                        list(x.shape)
                        if isinstance(
                            x,
                            torch.Tensor,
                        )
                        else None
                },
            )

        if x.size(-1) != self.embed_dim:
            raise PerceptionDimensionError(
                "Transformer input dimension does not match embed_dim.",
                component="transformer",
                details={
                    "actual": x.size(-1),
                    "expected": self.embed_dim,
                },
            )

        if x.size(1) <= 0:
            raise PerceptionShapeError(
                "Transformer sequence length must be positive.",
                component="transformer",
            )

        if context is not None:
            if (
                not isinstance(context, torch.Tensor)
                or context.dim() != 3
            ):
                raise PerceptionShapeError(
                    "Transformer context must have shape (B,S,D).",
                    component="transformer",
                    details={
                        "shape":
                            list(context.shape)
                            if isinstance(context, torch.Tensor,)
                            else None
                    },
                )

            if (
                context.size(0) != x.size(0)
                or context.size(-1)
                != self.embed_dim
            ):
                raise PerceptionDimensionError(
                    "Transformer context is incompatible with the input.",
                    component="transformer",
                    details={
                        "input_shape": list(x.shape),
                        "context_shape": list(context.shape),
                        "embed_dim": self.embed_dim,
                    },
                )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _run_layer(
        self,
        layer: _TransformerLayer,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        should_checkpoint = (
            self.use_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and not self.output_attentions
        )

        if not should_checkpoint:
            return layer(
                x,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
            )

        if context is None:
            return torch_checkpoint(
                lambda hidden: layer(
                    hidden,
                    attention_mask=attention_mask,
                    context=None,
                    context_mask=None,
                ),
                x,
                use_reentrant=False,
                preserve_rng_state=(
                    self.preserve_rng_state
                ),
            )

        return torch_checkpoint(
            lambda hidden, memory: layer(
                hidden,
                attention_mask=attention_mask,
                context=memory,
                context_mask=context_mask,
            ),
            x,
            context,
            use_reentrant=False,
            preserve_rng_state=(
                self.preserve_rng_state
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[
            torch.Tensor
        ] = None,
        style_id: Optional[
            torch.Tensor
        ] = None,
        attention_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        """Return final hidden states with shape ``(B,L,D)``.

        ``attention_mask`` applies to self-attention keys.
        ``context_mask`` applies only to cross-attention memory.

        A temporary compatibility shim recognizes the current AudioEncoder call
        ``transformer(x, style_id)`` when the second positional tensor is rank
        0/1.  Rank-3 second arguments remain explicit cross-attention context.
        """
        if (
            context is not None
            and style_id is None
            and isinstance(context, torch.Tensor)
            and context.dim() <= 1
        ):
            style_id = context
            context = None

        self._validate_inputs(x, context)

        if (
            context is not None
            and not self.enable_cross_attention
        ):
            raise PerceptionContractError(
                "Context was provided to a Transformer without cross-attention.",
                component="transformer",
                remediation=(
                    "Construct the decoder backbone with "
                    "enable_cross_attention=True."
                ),
            )

        if (
            self.enable_cross_attention
            and context is None
        ):
            raise PerceptionContractError(
                "This Transformer requires cross-attention context.",
                component="transformer",
                remediation=(
                    "Provide encoder/context states or construct with "
                    "enable_cross_attention=False."
                ),
            )

        style = self._normalize_style_id(style_id, batch_size=x.size(0), device=x.device)
        if style is not None:
            style_embedding = (self.style_embeddings[style].unsqueeze(1))
            x = x + style_embedding

        for layer_index, layer in enumerate(self.layers):
            x = self._run_layer(
                layer,
                x,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
            )
        
            self._cache_layer_output(
                x,
                layer_index,
            )
        
        return x

        return x

    

    def _cache_layer_output(self, tensor: torch.Tensor, layer_index: int) -> None:
        if self.memory is None:
            return
    
        self.memory.cache_item(
            tensor=tensor,
            key=f"transformer_layer_{layer_index}",
            tags=[
                "transformer",
                "layer_output",
                f"layer_{layer_index}",
            ],
            metadata={
                "layer_index": layer_index,
                "causal": self.causal,
                "cross_attention": self.enable_cross_attention,
                "training": self.training,
            },
        )

    # ------------------------------------------------------------------
    # Fine-tuning controls
    # ------------------------------------------------------------------
    def _normalize_layer_indices(
        self,
        layer_indices: Sequence[int],
    ) -> tuple[int, ...]:
        normalized = tuple(
            dict.fromkeys(
                int(index)
                for index in layer_indices
            )
        )
        invalid = [
            index
            for index in normalized
            if index < 0
            or index >= len(self.layers)
        ]
        if invalid:
            raise PerceptionRangeError(
                "Transformer layer index is out of range.",
                component="transformer",
                details={
                    "invalid_indices": invalid,
                    "num_layers":
                        len(self.layers),
                },
            )
        return normalized

    def freeze_layers(
        self,
        layer_indices: Optional[
            Sequence[int]
        ] = None,
    ) -> None:
        """Freeze selected layers, or the full backbone when omitted."""
        if layer_indices is None:
            for parameter in (
                self.layers.parameters()
            ):
                parameter.requires_grad = (
                    False
                )
            self.style_embeddings.requires_grad_(
                False
            )
            return

        for index in self._normalize_layer_indices(
            layer_indices
        ):
            for parameter in (
                self.layers[
                    index
                ].parameters()
            ):
                parameter.requires_grad = (
                    False
                )

    def unfreeze_layers(
        self,
        layer_indices: Optional[
            Sequence[int]
        ] = None,
    ) -> None:
        """Unfreeze selected layers, or the full backbone when omitted."""
        if layer_indices is None:
            for parameter in (
                self.layers.parameters()
            ):
                parameter.requires_grad = (
                    True
                )
            self.style_embeddings.requires_grad_(
                True
            )
            return

        for index in self._normalize_layer_indices(
            layer_indices
        ):
            for parameter in (
                self.layers[
                    index
                ].parameters()
            ):
                parameter.requires_grad = (
                    True
                )

    # ------------------------------------------------------------------
    # Conservative state import
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_legacy_key(
        key: str,
    ) -> str:
        normalized = str(key)

        for prefix in (
            "module.",
            "transformer.",
        ):
            if normalized.startswith(
                prefix
            ):
                normalized = normalized[
                    len(prefix):
                ]

        # Deterministic migration from the old ModuleDict field names.
        replacements = (
            (".attention.", ".self_attention."),
            (".ff.", ".feedforward."),
            (".norm1.", ".norm_self."),
            (".norm2.", ".norm_ff."),
        )
        for old, new in replacements:
            normalized = (
                normalized.replace(
                    old,
                    new,
                )
            )

        return normalized

    def load_pretrained(
        self,
        weights: Mapping[
            str,
            torch.Tensor
        ],
        *,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Load only name- and shape-compatible Transformer state.

        No architecture-guessing or tensor reshaping is performed here.
        """
        if not isinstance(
            weights,
            Mapping,
        ):
            raise PerceptionStateError(
                "weights must be a mapping.",
                component="transformer",
                details={
                    "actual_type":
                        type(weights).__name__
                },
            )

        current = self.state_dict()
        prepared: Dict[
            str,
            torch.Tensor
        ] = {}
        unexpected = []
        shape_mismatches = []

        for raw_key, tensor in (
            weights.items()
        ):
            key = self._normalize_legacy_key(
                raw_key
            )

            # Task heads belonged to the previous Transformer design and are
            # intentionally not restored into the backbone.
            if (
                key.startswith("task_head.")
                or key.startswith("transformer.")
            ):
                continue

            if key not in current:
                unexpected.append(
                    raw_key
                )
                continue

            if not isinstance(
                tensor,
                torch.Tensor,
            ):
                unexpected.append(
                    raw_key
                )
                continue

            if tuple(
                tensor.shape
            ) != tuple(
                current[key].shape
            ):
                shape_mismatches.append({
                    "key": raw_key,
                    "source_shape":
                        list(tensor.shape),
                    "target_shape":
                        list(
                            current[key].shape
                        ),
                })
                continue

            prepared[key] = tensor

        missing = sorted(
            set(current)
            - set(prepared)
        )

        if strict and (
            missing
            or unexpected
            or shape_mismatches
        ):
            raise PerceptionStateError(
                "Pretrained Transformer state is not strictly compatible.",
                component="transformer",
                details={
                    "missing_keys": missing,
                    "unexpected_keys":
                        unexpected,
                    "shape_mismatches":
                        shape_mismatches,
                },
            )

        result = self.load_state_dict(
            prepared,
            strict=False,
        )

        return {
            "loaded_keys": tuple(
                sorted(prepared)
            ),
            "missing_keys": tuple(
                sorted(
                    result.missing_keys
                )
            ),
            "unexpected_keys": tuple(
                sorted(
                    set(unexpected)
                    | set(
                        result.unexpected_keys
                    )
                )
            ),
            "shape_mismatches":
                tuple(shape_mismatches),
        }


__all__ = [
    "Transformer",
    "_TransformerLayer",
]


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

    # print("\n===* * * Test 2: Selector * * *===\n")
    # input_shape = [1, 128, model.embed_dim]

    # attention = model.select_attention(input_shape=input_shape)
    # taskhead = model.select_taskhead()

    # printer.pretty("ATTENTION", attention.__class__.__name__, "success" if attention.__class__.__name__ == "success" else "error")
    # printer.pretty("TASKHEAD", taskhead, "success" if taskhead == "success" else "error")

    print("\n===* * * Test 3: Transformer Layer * * *===\n")
    # Instantiate attention and feedforward without arguments (as in __init__)
    dummy_attn = BaseAttention()
    dummy_ff = FeedForward()

    # Instantiate _TransformerLayer
    layer = _TransformerLayer(
        embed_dim=model.embed_dim,
        dropout=model.dropout_rate,
        self_attention=dummy_attn,
        feedforward=dummy_ff,
        cross_attention=None
    )

    test_x = torch.randn(4, 128, model.embed_dim)
    output_layer = layer(test_x)
    print(f"Output shape from _TransformerLayer: {output_layer.shape}")

    printer.pretty("TRANSFORMER LAYER", "Forward pass successful", "success")
    
    print("\n=== Successfully Ran Transformer ===\n")