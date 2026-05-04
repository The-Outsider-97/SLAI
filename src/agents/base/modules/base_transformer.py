"""
Base transformer subsystem for the Base Agent stack.

This module provides the production-grade transformer backbone used by the
base agent subsystem for sequence-to-sequence modeling, representation
learning, autoregressive generation, and encoder-decoder experimentation.
It standardizes model configuration, validation, mask construction,
checkpointing, optimizer creation, inference, runtime statistics, and
optional memory integration.

Design goals:
- keep the core architecture faithful to the standard transformer pattern
- validate configuration aggressively and fail with structured base errors
- reuse shared helper primitives instead of duplicating utility logic
- support both training-time and inference-time workflows cleanly
- provide deterministic, serializable metadata for checkpoints and diagnostics
- remain extensible without forcing higher-level agents to reimplement basics
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
from collections.abc import Mapping
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_errors import *
from ..utils.base_helpers import *
from ..base_memory import BaseMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Base Transformer")
printer = PrettyPrinter


@dataclass(frozen=True)
class TransformerConfig:
    """Resolved transformer configuration."""

    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    activation: str
    layer_norm_eps: float
    batch_first: bool
    norm_first: bool
    max_position_embeddings: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    tie_embeddings: bool
    tie_output_projection: bool
    enable_memory_integration: bool
    memory_namespace: str
    record_history: bool
    history_limit: int
    checkpoint_pretty: bool
    label_smoothing: float
    clip_grad: float
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    inference_max_len: int
    inference_temperature: float
    inference_do_sample: bool
    inference_top_k: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
            "layer_norm_eps": self.layer_norm_eps,
            "batch_first": self.batch_first,
            "norm_first": self.norm_first,
            "max_position_embeddings": self.max_position_embeddings,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "tie_embeddings": self.tie_embeddings,
            "tie_output_projection": self.tie_output_projection,
            "enable_memory_integration": self.enable_memory_integration,
            "memory_namespace": self.memory_namespace,
            "record_history": self.record_history,
            "history_limit": self.history_limit,
            "checkpoint_pretty": self.checkpoint_pretty,
            "label_smoothing": self.label_smoothing,
            "clip_grad": self.clip_grad,
            "lr": self.lr,
            "betas": list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "inference_max_len": self.inference_max_len,
            "inference_temperature": self.inference_temperature,
            "inference_do_sample": self.inference_do_sample,
            "inference_top_k": self.inference_top_k,
        }


@dataclass(frozen=True)
class TransformerRunStats:
    """High-level runtime statistics for the transformer instance."""

    request_id: str
    created_at: str
    trainable_parameters: int
    total_parameters: int
    d_model: int
    nhead: int
    encoder_layers: int
    decoder_layers: int
    history_length: int
    memory_enabled: bool
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "created_at": self.created_at,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "history_length": self.history_length,
            "memory_enabled": self.memory_enabled,
            "device": self.device,
        }


@dataclass
class GenerationOutput:
    """Structured generation result."""

    sequences: torch.Tensor
    logits_history: Optional[List[torch.Tensor]] = None
    token_scores: Optional[List[torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequences": to_json_safe(self.sequences.detach().cpu().tolist()),
            "logits_history": None if self.logits_history is None else [to_json_safe(t.detach().cpu().tolist()) for t in self.logits_history],
            "token_scores": None if self.token_scores is None else [to_json_safe(t.detach().cpu().tolist()) for t in self.token_scores],
            "metadata": to_json_safe(self.metadata),
        }


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first or sequence-first inputs."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, batch_first: bool = True) -> None:
        super().__init__()
        ensure_numeric_range(d_model, "d_model", minimum=1, config={"component": "PositionalEncoding"}, error_cls=BaseValidationError)
        ensure_numeric_range(max_len, "max_len", minimum=1, config={"component": "PositionalEncoding"}, error_cls=BaseValidationError)
        ensure_numeric_range(dropout, "dropout", minimum=0.0, maximum=1.0, config={"component": "PositionalEncoding"}, error_cls=BaseValidationError)

        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.batch_first = bool(batch_first)
        self.dropout = nn.Dropout(p=float(dropout))

        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if self.batch_first:
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        else:
            pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise BaseValidationError(
                "PositionalEncoding expects a 3D tensor.",
                {"component": "PositionalEncoding"},
                component="PositionalEncoding",
                operation="forward",
                context={"received_shape": list(x.shape)},
            )
    
        seq_len = x.size(1) if self.batch_first else x.size(0)
        if seq_len > self.max_len:
            raise BaseValidationError(
                "Sequence length exceeds the configured positional encoding capacity.",
                {"component": "PositionalEncoding"},
                component="PositionalEncoding",
                operation="forward",
                context={"seq_len": seq_len, "max_len": self.max_len},
            )
    
        # Cast the buffer to a tensor to satisfy the type checker
        pe = cast(torch.Tensor, self.pe)
        positional = pe[:, :seq_len] if self.batch_first else pe[:seq_len]
        return self.dropout(x + positional.to(device=x.device, dtype=x.dtype))


class BaseTransformer(nn.Module):
    """Production-grade encoder-decoder transformer for the base subsystem."""

    VALID_ACTIVATIONS: Tuple[str, ...] = ("relu", "gelu")

    def __init__(self, **overrides: Any) -> None:
        super().__init__()

        self.global_config = load_global_config()
        self.transformer_config = get_config_section("base_transformer") or {}
        self.inference_config = get_config_section("inference") or {}
        self.forward_config = get_config_section("forward") or {}
        self._request_id = generate_request_id("transformer", include_timestamp=True)
        self._created_at = utc_now_iso()
        self._history: Deque[Dict[str, Any]] = deque(
            maxlen=coerce_int(self.transformer_config.get("history_limit", 200), default=200, minimum=1)
        )

        self.config = self._resolve_config(overrides)
        self.shared_memory: Optional[BaseMemory] = None
        if self.config.enable_memory_integration:
            self.shared_memory = BaseMemory()

        self.src_embed = nn.Embedding(self.config.src_vocab_size, self.config.d_model)
        self.tgt_embed = nn.Embedding(self.config.tgt_vocab_size, self.config.d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=self.config.d_model,
            max_len=self.config.max_position_embeddings,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.transformer = nn.Transformer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            layer_norm_eps=self.config.layer_norm_eps,
            batch_first=self.config.batch_first,
            norm_first=self.config.norm_first,
        )
        self.output_projection = nn.Linear(self.config.d_model, self.config.tgt_vocab_size, bias=False)

        self._tie_weights_if_configured()
        self._init_weights()
        self._record_history("init", details=self.config.to_dict())
        self._sync_memory_state("initialized")

        printer.status(
            "INIT",
            f"Base Transformer initialized with d_model={self.config.d_model}, nhead={self.config.nhead}",
            "success",
        )

    # ------------------------------------------------------------------
    # Configuration / validation
    # ------------------------------------------------------------------
    def _resolve_config(self, overrides: Mapping[str, Any]) -> TransformerConfig:
        merged = deep_merge_dicts(self.transformer_config, dict(overrides or {}))
        global_src_vocab = self.global_config.get("src_vocab_size", 30000)
        global_tgt_vocab = self.global_config.get("tgt_vocab_size", 30000)

        activation = str(merged.get("activation", "relu")).strip().lower()
        ensure_one_of(
            activation,
            self.VALID_ACTIVATIONS,
            "activation",
            config=merged,
            error_cls=BaseConfigurationError,
            component="BaseTransformer",
            operation="configuration",
        )

        betas = self._resolve_betas(merged.get("betas", (0.9, 0.98)))

        config = TransformerConfig(
            src_vocab_size=coerce_int(merged.get("src_vocab_size", global_src_vocab), default=global_src_vocab, minimum=2),
            tgt_vocab_size=coerce_int(merged.get("tgt_vocab_size", global_tgt_vocab), default=global_tgt_vocab, minimum=2),
            d_model=coerce_int(merged.get("d_model", 512), default=512, minimum=2),
            nhead=coerce_int(merged.get("nhead", 8), default=8, minimum=1),
            num_encoder_layers=coerce_int(merged.get("num_encoder_layers", 6), default=6, minimum=1),
            num_decoder_layers=coerce_int(merged.get("num_decoder_layers", 6), default=6, minimum=1),
            dim_feedforward=coerce_int(merged.get("dim_feedforward", 2048), default=2048, minimum=4),
            dropout=coerce_float(merged.get("dropout", 0.1), default=0.1, minimum=0.0, maximum=1.0),
            activation=activation,
            layer_norm_eps=coerce_float(merged.get("layer_norm_eps", 1e-5), default=1e-5, minimum=1e-12),
            batch_first=coerce_bool(merged.get("batch_first", True), default=True),
            norm_first=coerce_bool(merged.get("norm_first", False), default=False),
            max_position_embeddings=coerce_int(merged.get("max_len", merged.get("max_position_embeddings", 5000)), default=5000, minimum=2),
            pad_token_id=coerce_int(merged.get("pad_token_id", 0), default=0, minimum=0),
            bos_token_id=coerce_int(merged.get("bos_token_id", self.inference_config.get("sos_token", 1)), default=1, minimum=0),
            eos_token_id=coerce_int(merged.get("eos_token_id", self.inference_config.get("eos_token", 2)), default=2, minimum=0),
            tie_embeddings=coerce_bool(merged.get("tie_embeddings", False), default=False),
            tie_output_projection=coerce_bool(merged.get("tie_output_projection", True), default=True),
            enable_memory_integration=coerce_bool(merged.get("enable_memory_integration", False), default=False),
            memory_namespace=normalize_identifier(merged.get("memory_namespace", "base_transformer"), lowercase=True, separator="_", max_length=120),
            record_history=coerce_bool(merged.get("record_history", True), default=True),
            history_limit=coerce_int(merged.get("history_limit", 200), default=200, minimum=1),
            checkpoint_pretty=coerce_bool(merged.get("checkpoint_pretty", True), default=True),
            label_smoothing=coerce_float(merged.get("label_smoothing", 0.0), default=0.0, minimum=0.0, maximum=1.0),
            clip_grad=coerce_float(merged.get("clip_grad", 1.0), default=1.0, minimum=0.0),
            lr=coerce_float(merged.get("lr", 1e-4), default=1e-4, minimum=0.0),
            betas=betas,
            eps=coerce_float(merged.get("eps", 1e-9), default=1e-9, minimum=1e-12),
            weight_decay=coerce_float(merged.get("weight_decay", 1e-4), default=1e-4, minimum=0.0),
            inference_max_len=coerce_int(merged.get("inference_max_len", self.inference_config.get("max_len", 50)), default=50, minimum=1),
            inference_temperature=coerce_float(merged.get("inference_temperature", self.inference_config.get("temperature", 1.0)), default=1.0, minimum=1e-6),
            inference_do_sample=coerce_bool(merged.get("inference_do_sample", False), default=False),
            inference_top_k=coerce_int(merged.get("inference_top_k", 0), default=0, minimum=0),
        )

        self._validate_config(config)
        return config

    def _resolve_betas(self, raw: Any) -> Tuple[float, float]:
        if isinstance(raw, str):
            parsed = parse_delimited_text(raw)
        elif isinstance(raw, Sequence):
            parsed = list(raw)
        else:
            parsed = [0.9, 0.98]

        numeric = [coerce_float(item, default=0.0) for item in parsed]
        numeric = [value for value in numeric if 0.0 <= value < 1.0]
        if len(numeric) < 2:
            return (0.9, 0.98)
        return (numeric[0], numeric[1])

    def _validate_config(self, config: TransformerConfig) -> None:
        if config.d_model % config.nhead != 0:
            raise BaseConfigurationError(
                "d_model must be divisible by nhead.",
                config.to_dict(),
                component="BaseTransformer",
                operation="configuration",
                context={"d_model": config.d_model, "nhead": config.nhead},
            )

        if config.tie_embeddings and config.src_vocab_size != config.tgt_vocab_size:
            raise BaseConfigurationError(
                "tie_embeddings requires src_vocab_size and tgt_vocab_size to match.",
                config.to_dict(),
                component="BaseTransformer",
                operation="configuration",
                context={
                    "src_vocab_size": config.src_vocab_size,
                    "tgt_vocab_size": config.tgt_vocab_size,
                },
            )

    # ------------------------------------------------------------------
    # Memory / history
    # ------------------------------------------------------------------
    def _record_history(self, action: str, details: Optional[Mapping[str, Any]] = None) -> None:
        if not self.config.record_history:
            return
        self._history.append(
            {
                "timestamp": utc_now_iso(),
                "action": action,
                "details": to_json_safe(details or {}),
            }
        )

    def _sync_memory_state(self, action: str) -> None:
        if self.shared_memory is None:
            return
        try:
            self.shared_memory.put(
                key=f"transformer:{self._request_id}",
                value=drop_none_values(
                    {
                        "request_id": self._request_id,
                        "created_at": self._created_at,
                        "action": action,
                        "config": self.config.to_dict(),
                        "stats": self.stats().to_dict(),
                    },
                    recursive=True,
                ),
                namespace=self.config.memory_namespace,
                metadata={"component": "BaseTransformer", "action": action},
                tags=["transformer", action],
                persistent=False,
            )
        except Exception as exc:
            logger.warning(f"BaseTransformer memory sync skipped due to error: {exc}")

    # ------------------------------------------------------------------
    # Weights / embeddings
    # ------------------------------------------------------------------
    def _tie_weights_if_configured(self) -> None:
        if self.config.tie_embeddings:
            self.tgt_embed.weight = self.src_embed.weight
        if self.config.tie_output_projection:
            if self.config.tie_embeddings and self.config.src_vocab_size == self.config.tgt_vocab_size:
                self.output_projection.weight = self.src_embed.weight
            elif self.config.tgt_vocab_size == self.tgt_embed.num_embeddings:
                self.output_projection.weight = self.tgt_embed.weight

    def _init_weights(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def _scale_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings * math.sqrt(float(self.config.d_model))

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> None:
        ensure_type(
            tensor,
            torch.Tensor,
            name,
            config=self.config.to_dict(),
            error_cls=BaseValidationError,
            component="BaseTransformer",
            operation="tensor_validation",
        )
        if tensor.dim() != 2:
            raise BaseValidationError(
                f"'{name}' must be rank-2 with shape (batch, seq_len) when batch_first=True.",
                self.config.to_dict(),
                component="BaseTransformer",
                operation="tensor_validation",
                context={"name": name, "shape": list(tensor.shape)},
            )
        if tensor.size(-1) > self.config.max_position_embeddings:
            raise BaseValidationError(
                f"'{name}' sequence length exceeds max_position_embeddings.",
                self.config.to_dict(),
                component="BaseTransformer",
                operation="tensor_validation",
                context={
                    "name": name,
                    "shape": list(tensor.shape),
                    "max_position_embeddings": self.config.max_position_embeddings,
                },
            )

    def _validate_forward_inputs(self, src: torch.Tensor, tgt: torch.Tensor) -> None:
        self._validate_tensor(src, "src")
        self._validate_tensor(tgt, "tgt")
        if src.device != tgt.device:
            raise BaseValidationError(
                "src and tgt must be on the same device.",
                self.config.to_dict(),
                component="BaseTransformer",
                operation="forward",
                context={"src_device": str(src.device), "tgt_device": str(tgt.device)},
            )
        if src.size(0) != tgt.size(0):
            raise BaseValidationError(
                "src and tgt batch sizes must match.",
                self.config.to_dict(),
                component="BaseTransformer",
                operation="forward",
                context={"src_batch": int(src.size(0)), "tgt_batch": int(tgt.size(0))},
            )

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def generate_square_subsequent_mask(
        self,
        size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ensure_numeric_range(size, "size", minimum=1, config=self.config.to_dict(), error_cls=BaseValidationError)
        mask = torch.triu(
            torch.full((int(size), int(size)), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        return mask

    def make_padding_mask(self, token_ids: torch.Tensor, pad_token_id: Optional[int] = None) -> torch.Tensor:
        self._validate_tensor(token_ids, "token_ids")
        pad_value = self.config.pad_token_id if pad_token_id is None else int(pad_token_id)
        return token_ids.eq(pad_value)

    # ------------------------------------------------------------------
    # Core encode/decode/forward
    # ------------------------------------------------------------------
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_tensor(src, "src")
        src_emb = self._scale_embeddings(self.src_embed(src))
        src_emb = self.positional_encoding(src_emb)
        return self.transformer.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_tensor(tgt, "tgt")
        ensure_type(memory, torch.Tensor, "memory", config=self.config.to_dict(), error_cls=BaseValidationError)
        tgt_emb = self._scale_embeddings(self.tgt_embed(tgt))
        tgt_emb = self.positional_encoding(tgt_emb)
        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_forward_inputs(src, tgt)
        memory = self.encode(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        decoder_output = self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.output_projection(decoder_output)
        self._record_history(
            "forward",
            details={
                "src_shape": list(src.shape),
                "tgt_shape": list(tgt.shape),
                "logits_shape": list(logits.shape),
            },
        )
        if return_dict:
            return {
                "logits": logits,
                "memory": memory,
                "decoder_output": decoder_output,
            }
        return logits

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        ensure_type(logits, torch.Tensor, "logits", config=self.config.to_dict(), error_cls=BaseValidationError)
        ensure_type(targets, torch.Tensor, "targets", config=self.config.to_dict(), error_cls=BaseValidationError)

        if logits.dim() != 3:
            raise BaseValidationError(
                "logits must have shape (batch, seq_len, vocab).",
                self.config.to_dict(),
                component="BaseTransformer",
                operation="compute_loss",
                context={"logits_shape": list(logits.shape)},
            )
        if targets.dim() != 2:
            raise BaseValidationError(
                "targets must have shape (batch, seq_len).",
                self.config.to_dict(),
                component="BaseTransformer",
                operation="compute_loss",
                context={"targets_shape": list(targets.shape)},
            )

        if logits.shape[:2] != targets.shape:
            raise BaseValidationError(
                "logits and targets sequence dimensions must match.",
                self.config.to_dict(),
                component="BaseTransformer",
                operation="compute_loss",
                context={"logits_shape": list(logits.shape), "targets_shape": list(targets.shape)},
            )

        effective_ignore_index = self.config.pad_token_id if ignore_index is None else int(ignore_index)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=effective_ignore_index,
            reduction=reduction,
            label_smoothing=self.config.label_smoothing,
        )

    def configure_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

    def clip_gradients(self, optimizer: Optional[torch.optim.Optimizer] = None) -> float:
        del optimizer
        if self.config.clip_grad <= 0:
            return 0.0
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.clip_grad)
        return float(norm.detach().cpu().item() if isinstance(norm, torch.Tensor) else norm)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _sample_next_token(self, logits: torch.Tensor, *, temperature: float, do_sample: bool, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            raise BaseValidationError(
                "temperature must be > 0.",
                self.config.to_dict(),
                component="BaseTransformer",
                operation="generation",
                context={"temperature": temperature},
            )

        step_logits = logits / temperature
        if top_k > 0 and top_k < step_logits.size(-1):
            top_values, top_indices = torch.topk(step_logits, k=top_k, dim=-1)
            filtered = torch.full_like(step_logits, float("-inf"))
            filtered.scatter_(dim=-1, index=top_indices, src=top_values)
            step_logits = filtered

        if do_sample:
            probs = F.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_score = torch.gather(probs, dim=-1, index=next_token)
        else:
            next_score, next_token = torch.max(step_logits, dim=-1, keepdim=True)
        return next_token, next_score

    def inference(
        self,
        src: torch.Tensor,
        *,
        max_len: Optional[int] = None,
        sos_token: Optional[int] = None,
        eos_token: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        capture_logits: bool = False,
    ) -> Union[torch.Tensor, GenerationOutput]:
        self._validate_tensor(src, "src")

        effective_max_len = self.config.inference_max_len if max_len is None else int(max_len)
        effective_sos = self.config.bos_token_id if sos_token is None else int(sos_token)
        effective_eos = self.config.eos_token_id if eos_token is None else int(eos_token)
        effective_temperature = self.config.inference_temperature if temperature is None else float(temperature)
        effective_do_sample = self.config.inference_do_sample if do_sample is None else bool(do_sample)
        effective_top_k = self.config.inference_top_k if top_k is None else int(top_k)

        ensure_numeric_range(effective_max_len, "max_len", minimum=1, config=self.config.to_dict(), error_cls=BaseValidationError)

        was_training = self.training
        self.eval()
        logits_history: List[torch.Tensor] = []
        token_scores: List[torch.Tensor] = []

        with torch.no_grad():
            memory = self.encode(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            batch_size = src.size(0)
            generated = torch.full((batch_size, 1), effective_sos, dtype=torch.long, device=src.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

            for _ in range(effective_max_len):
                tgt_mask = self.generate_square_subsequent_mask(generated.size(1), device=src.device)
                decoder_output = self.decode(generated, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
                logits = self.output_projection(decoder_output[:, -1:, :])
                step_logits = logits[:, -1, :]
                next_token, next_score = self._sample_next_token(
                    step_logits,
                    temperature=effective_temperature,
                    do_sample=effective_do_sample,
                    top_k=effective_top_k,
                )

                next_token = next_token.masked_fill(finished.unsqueeze(1), effective_eos)
                generated = torch.cat([generated, next_token], dim=1)
                finished = finished | next_token.squeeze(1).eq(effective_eos)

                if capture_logits:
                    logits_history.append(step_logits.detach().cpu())
                token_scores.append(next_score.detach().cpu())

                if finished.all():
                    break

        if was_training:
            self.train()

        self._record_history(
            "inference",
            details={
                "src_shape": list(src.shape),
                "generated_shape": list(generated.shape),
                "max_len": effective_max_len,
                "do_sample": effective_do_sample,
                "top_k": effective_top_k,
            },
        )

        if return_dict:
            return GenerationOutput(
                sequences=generated,
                logits_history=logits_history if capture_logits else None,
                token_scores=token_scores,
                metadata={
                    "max_len": effective_max_len,
                    "sos_token": effective_sos,
                    "eos_token": effective_eos,
                    "temperature": effective_temperature,
                    "do_sample": effective_do_sample,
                    "top_k": effective_top_k,
                },
            )
        return generated

    # ------------------------------------------------------------------
    # Checkpointing / export
    # ------------------------------------------------------------------
    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def stats(self) -> TransformerRunStats:
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")
        return TransformerRunStats(
            request_id=self._request_id,
            created_at=self._created_at,
            trainable_parameters=self.parameter_count(trainable_only=True),
            total_parameters=self.parameter_count(trainable_only=False),
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            encoder_layers=self.config.num_encoder_layers,
            decoder_layers=self.config.num_decoder_layers,
            history_length=len(self._history),
            memory_enabled=self.shared_memory is not None,
            device=str(device),
        )

    def to_dict(self, *, include_history: bool = True) -> Dict[str, Any]:
        payload = {
            "request_id": self._request_id,
            "created_at": self._created_at,
            "config": self.config.to_dict(),
            "stats": self.stats().to_dict(),
        }
        if include_history:
            payload["history"] = list(self._history)
        return to_json_safe(payload)

    def save(
        self,
        path: Union[str, Path],
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "state_dict": self.state_dict(),
                "config": self.config.to_dict(),
                "metadata": {
                    "request_id": self._request_id,
                    "created_at": self._created_at,
                    "saved_at": utc_now_iso(),
                    "stats": self.stats().to_dict(),
                    "extra": to_json_safe(dict(extra_metadata or {})),
                },
            }
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(checkpoint, target)
            self._record_history("save", details={"path": str(target)})
            logger.info(f"BaseTransformer checkpoint saved to {target}")
            return str(target)
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Failed to save BaseTransformer checkpoint.",
                config=self.config.to_dict(),
                component="BaseTransformer",
                operation="save",
                context={"path": str(target)},
            ) from exc

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "BaseTransformer":
        target = Path(path)
        if not target.exists():
            raise BaseIOError(
                "BaseTransformer checkpoint path does not exist.",
                {"component": "BaseTransformer"},
                component="BaseTransformer",
                operation="load",
                context={"path": str(target)},
            )

        try:
            checkpoint = torch.load(target, map_location=device)
            config_payload = dict(checkpoint.get("config") or {})
            if overrides:
                config_payload = deep_merge_dicts(config_payload, dict(overrides))
            model = cls(**config_payload)
            model.load_state_dict(checkpoint["state_dict"], strict=strict)
            if device is not None:
                model = model.to(device)
            model._record_history("load", details={"path": str(target), "strict": strict})
            logger.info(f"BaseTransformer checkpoint loaded from {target}")
            return model
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Failed to load BaseTransformer checkpoint.",
                config={"path": str(target)},
                component="BaseTransformer",
                operation="load",
                context={"path": str(target), "strict": strict},
            ) from exc


if __name__ == "__main__":
    print("\n=== Running Base Transformer ===\n")
    printer.status("TEST", "Base Transformer initialized", "info")

    transformer = BaseTransformer(
        d_model=64,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        src_vocab_size=128,
        tgt_vocab_size=128,
        max_len=128,
        inference_max_len=12,
        enable_memory_integration=False,
    )

    src = torch.randint(3, 100, (2, 10), dtype=torch.long)
    tgt = torch.randint(3, 100, (2, 9), dtype=torch.long)

    src_padding_mask = transformer.make_padding_mask(src)
    tgt_padding_mask = transformer.make_padding_mask(tgt)
    tgt_mask = transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)

    forward_output = transformer(
        src,
        tgt,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_padding_mask,
        tgt_key_padding_mask=tgt_padding_mask,
        memory_key_padding_mask=src_padding_mask,
        return_dict=True,
    )

    loss = transformer.compute_loss(forward_output["logits"], tgt)
    loss.backward()
    grad_norm = transformer.clip_gradients()
    optimizer = transformer.configure_optimizer()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    generation = transformer.inference(
        src[:1],
        max_len=8,
        do_sample=False,
        return_dict=True,
        capture_logits=True,
    )

    checkpoint_path = Path("/mnt/data/base_transformer_test_checkpoint.pt")
    saved_path = transformer.save(checkpoint_path, optimizer=optimizer, extra_metadata={"test": True})
    reloaded = BaseTransformer.load(
        saved_path,
        device=torch.device("cpu"),
        overrides={
            "d_model": 64,
            "nhead": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 128,
            "src_vocab_size": 128,
            "tgt_vocab_size": 128,
            "max_len": 128,
            "inference_max_len": 12,
            "enable_memory_integration": False,
        },
    )

    printer.pretty("FORWARD_SHAPES", {
        "src": list(src.shape),
        "tgt": list(tgt.shape),
        "logits": list(forward_output["logits"].shape),
        "memory": list(forward_output["memory"].shape),
        "decoder_output": list(forward_output["decoder_output"].shape),
    }, "success")
    printer.pretty("LOSS_AND_GRAD", {"loss": float(loss.detach().cpu().item()), "grad_norm": float(grad_norm)}, "success")
    printer.pretty("GENERATION", generation.to_dict(), "success") # type: ignore
    printer.pretty("STATS", transformer.stats().to_dict(), "success")
    printer.pretty("RELOADED_STATS", reloaded.stats().to_dict(), "success")
    printer.pretty("CHECKPOINT_PATH", {"path": saved_path}, "success")

    print("\n=== Test ran successfully ===\n")
