"""
Language Transformer Module

Core Function:
Provides the language subsystem with a production-ready transformer layer built
on top of the shared BaseTransformer. The language transformer specializes the
base encoder-decoder backbone for language-agent workflows: generation,
beam-search decoding, representation extraction, sequence scoring, perplexity
evaluation, task adaptation, checkpoint metadata, diagnostics, and integration
with tokenizer-driven pipelines.

Responsibilities:
- Inherit the BaseTransformer contract and reuse its validated architecture.
- Resolve language-specific generation and adaptation config from language_config.yaml.
- Provide robust greedy/sample/beam generation without relying on stale base attributes.
- Produce structured generation results, sequence scores, embeddings, and diagnostics.
- Support checkpoint save/load with language metadata and base-transformer config.
- Use language helpers and language errors instead of duplicating utility logic.

Why it matters:
The transformer is the model-facing backbone for language generation and
representation learning. A production language transformer must be deterministic,
configurable, inspectable, serializable, error-aware, and compatible with the
base transformer while adding language-specific behavior without weakening the
shared base contract.
"""

from __future__ import annotations

import math

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config_loader import load_global_config, get_config_section
from ...base.modules.base_transformer import BaseTransformer, GenerationOutput
from ..utils.language_error import *
from ..utils.language_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Language Transformer")
printer = PrettyPrinter

TensorLike = Union[torch.Tensor, Sequence[int], Sequence[Sequence[int]]]


@dataclass(frozen=True)
class BeamCandidate:
    """Single beam-search hypothesis."""

    tokens: Tuple[int, ...]
    score: float
    finished: bool = False
    token_scores: Tuple[float, ...] = ()

    @property
    def length(self) -> int:
        return len(self.tokens)

    def normalized_score(self, *, length_penalty: float = 1.0) -> float:
        if self.length <= 0:
            return self.score
        penalty = max(1e-6, float(self.length) ** max(0.0, float(length_penalty)))
        return self.score / penalty

    def extend(self, token_id: int, token_logprob: float, *, eos_token_id: int) -> "BeamCandidate":
        return BeamCandidate(
            tokens=tuple(list(self.tokens) + [int(token_id)]),
            score=float(self.score + token_logprob),
            finished=int(token_id) == int(eos_token_id),
            token_scores=tuple(list(self.token_scores) + [float(token_logprob)]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "score": self.score,
            "finished": self.finished,
            "token_scores": list(self.token_scores),
            "length": self.length,
        }


@dataclass(frozen=True)
class BeamSearchOutput:
    """Structured beam-search result."""

    sequences: torch.Tensor
    beam_scores: torch.Tensor
    beams: Tuple[Tuple[BeamCandidate, ...], ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequences": self.sequences.detach().cpu().tolist(),
            "beam_scores": self.beam_scores.detach().cpu().tolist(),
            "beams": [[beam.to_dict() for beam in item] for item in self.beams],
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class SequenceScore:
    """Sequence likelihood/perplexity details."""

    loss: float
    perplexity: float
    token_count: int
    mean_log_probability: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "token_count": self.token_count,
            "mean_log_probability": self.mean_log_probability,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class EmbeddingOutput:
    """Encoder representation output for downstream language modules."""

    embeddings: torch.Tensor
    strategy: str
    attention_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.embeddings.shape),
            "strategy": self.strategy,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class TaskAdaptationResult:
    """Summary of task-specific model adaptation."""

    task_type: str
    frozen_parameters: int
    trainable_parameters: int
    dropout: Optional[float] = None
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LanguageTransformerStats:
    """Language-specific runtime snapshot."""

    request_id: str
    version: str
    task_type: str
    base_stats: Dict[str, Any]
    beam_width: int
    max_generation_length: int
    temperature: float
    length_penalty: float
    diagnostics_count: int
    history_length: int
    current_device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LanguageTransformer(BaseTransformer):
    """
    Transformer specialized for language-agent tasks.

    The class inherits BaseTransformer and keeps all architecture, validation,
    masking, optimizer, checkpoint, and forward-pass behavior compatible with
    the base stack. It adds language-specific decoding, embeddings, scoring,
    task adaptation, and metadata handling.
    """

    VERSION = "2.0"
    CHECKPOINT_FORMAT = "language-transformer-checkpoint"
    VALID_TASKS: Tuple[str, ...] = (
        "generation",
        "classification",
        "translation",
        "summarization",
        "dialogue",
        "embedding",
        "reranking",
    )

    def __init__(self, **overrides: Any) -> None:
        global_config = load_global_config()
        lang_config = get_config_section("language_transformer") or {}
        base_overrides = self._resolve_base_overrides(lang_config, overrides)

        super().__init__(**base_overrides)

        self.language_global_config = global_config
        self.lang_config = ensure_mapping(lang_config, field_name="language_transformer", allow_none=True)
        self.version = ensure_text(self.lang_config.get("version", self.VERSION))
        self.task_type = normalize_identifier_component(
            self.lang_config.get("task_type", "generation"),
            default="generation",
            lowercase=True,
        )
        if self.task_type not in self.VALID_TASKS:
            raise ConfigurationLanguageError(
                ConfigurationIssue(
                    code=LanguageErrorCode.CONFIG_VALUE_INVALID,
                    message="Unsupported language transformer task type.",
                    module="LanguageTransformer",
                    details={"task_type": self.task_type, "valid_tasks": list(self.VALID_TASKS)},
                ),
                recoverable=False,
            )

        generation_config = ensure_mapping(self.lang_config.get("generation", {}), field_name="language_transformer.generation", allow_none=True)
        beam_config = ensure_mapping(self.lang_config.get("beam_search", {}), field_name="language_transformer.beam_search", allow_none=True)
        eval_config = ensure_mapping(self.lang_config.get("evaluation", {}), field_name="language_transformer.evaluation", allow_none=True)
        checkpoint_config = ensure_mapping(self.lang_config.get("checkpointing", {}), field_name="language_transformer.checkpointing", allow_none=True)

        self.beam_width = coerce_int(beam_config.get("beam_width", self.lang_config.get("beam_width", 5)), default=5, minimum=1)
        self.max_generation_length = coerce_int(
            generation_config.get("max_length", self.lang_config.get("max_len", self.config.inference_max_len)),
            default=self.config.inference_max_len,
            minimum=1,
            maximum=self.config.max_position_embeddings,
        )
        self.min_generation_length = coerce_int(generation_config.get("min_length", 0), default=0, minimum=0)
        self.temperature = coerce_float(
            generation_config.get("temperature", self.lang_config.get("temperature", self.config.inference_temperature)),
            default=self.config.inference_temperature,
            minimum=1e-6,
        )
        self.length_penalty = coerce_float(
            beam_config.get("length_penalty", self.lang_config.get("length_penalty", 0.6)),
            default=0.6,
            minimum=0.0,
        )
        self.top_k = coerce_int(generation_config.get("top_k", self.config.inference_top_k), default=self.config.inference_top_k, minimum=0)
        self.do_sample = coerce_bool(generation_config.get("do_sample", self.config.inference_do_sample), default=self.config.inference_do_sample)
        self.repetition_penalty = coerce_float(generation_config.get("repetition_penalty", 1.0), default=1.0, minimum=1e-6)
        self.no_repeat_ngram_size = coerce_int(generation_config.get("no_repeat_ngram_size", 0), default=0, minimum=0)
        self.pad_token_id = coerce_int(generation_config.get("pad_token_id", self.config.pad_token_id), default=self.config.pad_token_id, minimum=0)
        self.bos_token_id = coerce_int(
            generation_config.get("bos_token_id", self.lang_config.get("sos_token", self.config.bos_token_id)),
            default=self.config.bos_token_id,
            minimum=0,
        )
        self.eos_token_id = coerce_int(
            generation_config.get("eos_token_id", self.lang_config.get("eos_token", self.config.eos_token_id)),
            default=self.config.eos_token_id,
            minimum=0,
        )
        self.ignore_index = coerce_int(eval_config.get("ignore_index", self.config.pad_token_id), default=self.config.pad_token_id, minimum=-100)
        self.checkpoint_dir = ensure_text(checkpoint_config.get("default_dir", "src/agents/language/checkpoints"))
        self.strict_checkpoint_loading = coerce_bool(checkpoint_config.get("strict", True), default=True)

        self.diagnostics = LanguageDiagnostics()
        self.language_history: Deque[Dict[str, Any]] = deque(maxlen=coerce_int(self.lang_config.get("history_limit", 200), default=200, minimum=1))
        self._apply_language_runtime_settings()
        self._record_language_event("init", task_type=self.task_type, config=self.language_config_snapshot())

        printer.status(
            "INIT",
            f"Language Transformer initialized task={self.task_type}, d_model={self.config.d_model}, vocab={self.config.tgt_vocab_size}",
            "success",
        )

    @staticmethod
    def _resolve_base_overrides(lang_config: Mapping[str, Any], explicit_overrides: Mapping[str, Any]) -> Dict[str, Any]:
        base_keys = {
            "src_vocab_size", "tgt_vocab_size", "d_model", "nhead", "num_encoder_layers",
            "num_decoder_layers", "dim_feedforward", "dropout", "activation", "layer_norm_eps",
            "batch_first", "norm_first", "max_position_embeddings", "max_len", "pad_token_id",
            "bos_token_id", "eos_token_id", "tie_embeddings", "tie_output_projection",
            "enable_memory_integration", "memory_namespace", "record_history", "history_limit",
            "checkpoint_pretty", "label_smoothing", "clip_grad", "lr", "betas", "eps",
            "weight_decay", "inference_max_len", "inference_temperature", "inference_do_sample",
            "inference_top_k",
        }
        configured = ensure_mapping(lang_config.get("base_overrides", {}), field_name="language_transformer.base_overrides", allow_none=True)
        for key in base_keys:
            if key in lang_config:
                configured[key] = lang_config[key]
        configured.update(dict(explicit_overrides or {}))
        return configured

    def _apply_language_runtime_settings(self) -> None:
        dropout = self.lang_config.get("dropout")
        if dropout is not None:
            self.set_dropout(coerce_float(dropout, default=self.config.dropout, minimum=0.0, maximum=1.0))

        initialization = ensure_mapping(self.lang_config.get("initialization", {}), field_name="language_transformer.initialization", allow_none=True)
        if coerce_bool(initialization.get("apply_language_initialization", False), default=False):
            self.initialize_language_params(
                embedding_std=coerce_float(initialization.get("embedding_std", 0.02), default=0.02, minimum=1e-8),
                output_std=coerce_float(initialization.get("output_std", 0.02), default=0.02, minimum=1e-8),
            )


    def _restore_language_runtime_config(self, payload: Mapping[str, Any]) -> None:
        """Restore language-specific runtime settings saved in a checkpoint."""

        self.version = ensure_text(payload.get("version", self.version))
        self.task_type = normalize_identifier_component(payload.get("task_type", self.task_type), default="generation", lowercase=True)
        self.beam_width = coerce_int(payload.get("beam_width", self.beam_width), default=self.beam_width, minimum=1)
        self.max_generation_length = coerce_int(
            payload.get("max_generation_length", self.max_generation_length),
            default=self.max_generation_length,
            minimum=1,
            maximum=self.config.max_position_embeddings,
        )
        self.min_generation_length = coerce_int(payload.get("min_generation_length", self.min_generation_length), default=self.min_generation_length, minimum=0)
        self.temperature = coerce_float(payload.get("temperature", self.temperature), default=self.temperature, minimum=1e-6)
        self.length_penalty = coerce_float(payload.get("length_penalty", self.length_penalty), default=self.length_penalty, minimum=0.0)
        self.top_k = coerce_int(payload.get("top_k", self.top_k), default=self.top_k, minimum=0)
        self.do_sample = coerce_bool(payload.get("do_sample", self.do_sample), default=self.do_sample)
        self.repetition_penalty = coerce_float(payload.get("repetition_penalty", self.repetition_penalty), default=self.repetition_penalty, minimum=1e-6)
        self.no_repeat_ngram_size = coerce_int(payload.get("no_repeat_ngram_size", self.no_repeat_ngram_size), default=self.no_repeat_ngram_size, minimum=0)
        self.pad_token_id = coerce_int(payload.get("pad_token_id", self.pad_token_id), default=self.pad_token_id, minimum=0)
        self.bos_token_id = coerce_int(payload.get("bos_token_id", self.bos_token_id), default=self.bos_token_id, minimum=0)
        self.eos_token_id = coerce_int(payload.get("eos_token_id", self.eos_token_id), default=self.eos_token_id, minimum=0)

    def language_config_snapshot(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "task_type": self.task_type,
            "beam_width": self.beam_width,
            "max_generation_length": self.max_generation_length,
            "min_generation_length": self.min_generation_length,
            "temperature": self.temperature,
            "length_penalty": self.length_penalty,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    def _add_issue(self, issue: Union[LanguageIssue, LanguageError]) -> None:
        self.diagnostics.add(issue)

    def _record_language_event(self, action: str, **payload: Any) -> None:
        event = {"timestamp": utc_timestamp(), "action": action, "payload": json_safe(payload)}
        self.language_history.append(event)
        record_history = getattr(self, "_record_history", None)
        if callable(record_history):
            record_history(f"language_{action}", details=payload)

    def _model_error(self, message: str, *, code: LanguageErrorCode, details: Optional[Mapping[str, Any]] = None) -> ModelLanguageError:
        return ModelLanguageError(
            ModelIssue(
                code=code,
                message=message,
                severity=Severity.ERROR,
                module="LanguageTransformer",
                details=dict(details or {}),
            ),
            recoverable=True,
        )

    def _coerce_token_tensor(self, value: TensorLike, *, field_name: str, device: Optional[torch.device] = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach() if not value.requires_grad else value
        else:
            tensor = torch.tensor(value, dtype=torch.long)
        if tensor.dtype != torch.long:
            tensor = tensor.long()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise self._model_error(
                f"{field_name} must be rank-2 token IDs with shape (batch, sequence).",
                code=LanguageErrorCode.MODEL_INFERENCE_FAILED,
                details={"field_name": field_name, "shape": list(tensor.shape)},
            )
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _safe_exp(self, value: float) -> float:
        return math.exp(min(50.0, max(-50.0, float(value))))

    def language_forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Language-aware forward pass with automatic causal target mask."""

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        output = super().forward(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_dict=return_dict,
        )
        self._record_language_event("forward", src_shape=list(src.shape), tgt_shape=list(tgt.shape), return_dict=return_dict)
        return output

    def encode_representations(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        normalize: bool = False,
        return_tokens: bool = False,
    ) -> EmbeddingOutput:
        """Encode input IDs and optionally pool them into sentence embeddings."""

        input_ids = self._coerce_token_tensor(input_ids, field_name="input_ids", device=self._device())
        key_padding_mask = None
        if attention_mask is not None:
            attention_mask = self._coerce_token_tensor(attention_mask, field_name="attention_mask", device=input_ids.device)
            key_padding_mask = attention_mask.eq(0)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            memory = self.encode(input_ids, src_key_padding_mask=key_padding_mask)
            strategy = normalize_identifier_component(pooling, default="mean", lowercase=True)
            if return_tokens or strategy == "token":
                embeddings = memory
                strategy = "token"
            elif strategy in {"cls", "first"}:
                embeddings = memory[:, 0, :]
                strategy = "first"
            elif strategy == "max":
                if key_padding_mask is not None:
                    masked = memory.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
                    embeddings = masked.max(dim=1).values
                    embeddings = torch.where(torch.isinf(embeddings), torch.zeros_like(embeddings), embeddings)
                else:
                    embeddings = memory.max(dim=1).values
            else:
                if key_padding_mask is not None:
                    valid = (~key_padding_mask).unsqueeze(-1).to(memory.dtype)
                    denom = valid.sum(dim=1).clamp_min(1.0)
                    embeddings = (memory * valid).sum(dim=1) / denom
                else:
                    embeddings = memory.mean(dim=1)
                strategy = "mean"

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        if was_training:
            self.train()

        self._record_language_event("encode_representations", shape=list(embeddings.shape), pooling=strategy)
        return EmbeddingOutput(
            embeddings=embeddings,
            strategy=strategy,
            attention_mask=attention_mask,
            metadata={"normalized": normalize, "input_shape": list(input_ids.shape)},
        )

    def _apply_generation_constraints(
        self,
        logits: torch.Tensor,
        tokens: Sequence[int],
        *,
        step_index: int,
        min_length: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        constrained = logits.clone()
        if step_index < min_length and 0 <= eos_token_id < constrained.size(-1):
            constrained[..., eos_token_id] = float("-inf")

        if self.repetition_penalty != 1.0:
            for token_id in set(int(item) for item in tokens):
                if 0 <= token_id < constrained.size(-1):
                    value = constrained[..., token_id]
                    constrained[..., token_id] = torch.where(value < 0, value * self.repetition_penalty, value / self.repetition_penalty)

        if self.no_repeat_ngram_size > 0 and len(tokens) + 1 >= self.no_repeat_ngram_size:
            banned = self._banned_ngram_tokens(tokens, self.no_repeat_ngram_size)
            for token_id in banned:
                if 0 <= token_id < constrained.size(-1):
                    constrained[..., token_id] = float("-inf")
        return constrained

    @staticmethod
    def _banned_ngram_tokens(tokens: Sequence[int], ngram_size: int) -> List[int]:
        if ngram_size <= 0 or len(tokens) < ngram_size - 1:
            return []
        prefix = tuple(tokens[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
        banned: List[int] = []
        for index in range(0, len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[index:index + ngram_size])
            if ngram_size == 1 or ngram[:-1] == prefix:
                banned.append(int(ngram[-1]))
        return banned

    def beam_search_decode(
        self,
        src: torch.Tensor,
        *,
        beam_width: Optional[int] = None,
        max_len: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        temperature: Optional[float] = None,
        length_penalty: Optional[float] = None,
        min_length: Optional[int] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, BeamSearchOutput]:
        """Beam-search decoding built on BaseTransformer.encode/decode/output_projection."""

        src = self._coerce_token_tensor(src, field_name="src", device=self._device())
        effective_beam_width = coerce_int(beam_width if beam_width is not None else self.beam_width, default=self.beam_width, minimum=1)
        effective_max_len = coerce_int(max_len if max_len is not None else self.max_generation_length, default=self.max_generation_length, minimum=1)
        effective_bos = self.bos_token_id if bos_token_id is None else int(bos_token_id)
        effective_eos = self.eos_token_id if eos_token_id is None else int(eos_token_id)
        effective_temperature = coerce_float(temperature if temperature is not None else self.temperature, default=self.temperature, minimum=1e-6)
        effective_length_penalty = coerce_float(length_penalty if length_penalty is not None else self.length_penalty, default=self.length_penalty, minimum=0.0)
        effective_min_length = coerce_int(min_length if min_length is not None else self.min_generation_length, default=self.min_generation_length, minimum=0)

        was_training = self.training
        self.eval()
        start_ms = monotonic_ms()
        best_sequences: List[List[int]] = []
        best_scores: List[float] = []
        all_final_beams: List[Tuple[BeamCandidate, ...]] = []

        with torch.no_grad():
            memory = self.encode(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            batch_size = int(src.size(0))
            vocab_size = int(self.config.tgt_vocab_size)
            top_k = min(effective_beam_width, vocab_size)

            for batch_index in range(batch_size):
                memory_item = memory[batch_index:batch_index + 1]
                padding_item = src_key_padding_mask[batch_index:batch_index + 1] if src_key_padding_mask is not None else None
                beams: List[BeamCandidate] = [BeamCandidate(tokens=(effective_bos,), score=0.0, finished=False)]

                for step_index in range(effective_max_len):
                    candidates: List[BeamCandidate] = []
                    for beam in beams:
                        if beam.finished:
                            candidates.append(beam)
                            continue
                        tgt = torch.tensor([list(beam.tokens)], dtype=torch.long, device=src.device)
                        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=src.device)
                        decoder_output = self.decode(tgt, memory_item, tgt_mask=tgt_mask, memory_key_padding_mask=padding_item)
                        logits = self.output_projection(decoder_output[:, -1, :])
                        logits = logits / effective_temperature
                        logits = self._apply_generation_constraints(
                            logits,
                            beam.tokens,
                            step_index=step_index,
                            min_length=effective_min_length,
                            eos_token_id=effective_eos,
                        )
                        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                        top_scores, top_indices = torch.topk(log_probs, k=top_k)
                        for score, token_id in zip(top_scores.tolist(), top_indices.tolist()):
                            candidates.append(beam.extend(int(token_id), float(score), eos_token_id=effective_eos))

                    if not candidates:
                        break
                    candidates.sort(key=lambda item: item.normalized_score(length_penalty=effective_length_penalty), reverse=True)
                    beams = candidates[:effective_beam_width]
                    if all(beam.finished for beam in beams):
                        break

                beams.sort(key=lambda item: item.normalized_score(length_penalty=effective_length_penalty), reverse=True)
                best = beams[0]
                best_sequences.append(list(best.tokens))
                best_scores.append(float(best.normalized_score(length_penalty=effective_length_penalty)))
                all_final_beams.append(tuple(beams))

        max_sequence_len = max(len(item) for item in best_sequences) if best_sequences else 0
        padded = [item + [self.pad_token_id] * (max_sequence_len - len(item)) for item in best_sequences]
        sequences = torch.tensor(padded, dtype=torch.long, device=src.device)
        scores = torch.tensor(best_scores, dtype=torch.float32, device=src.device)

        if was_training:
            self.train()

        self._record_language_event(
            "beam_search",
            batch_size=int(src.size(0)),
            beam_width=effective_beam_width,
            max_len=effective_max_len,
            elapsed_ms=elapsed_ms(start_ms),
        )

        if return_dict:
            return BeamSearchOutput(
                sequences=sequences,
                beam_scores=scores,
                beams=tuple(all_final_beams),
                metadata={
                    "beam_width": effective_beam_width,
                    "max_len": effective_max_len,
                    "bos_token_id": effective_bos,
                    "eos_token_id": effective_eos,
                    "temperature": effective_temperature,
                    "length_penalty": effective_length_penalty,
                    "min_length": effective_min_length,
                },
            )
        return sequences

    def generate(
        self,
        src: torch.Tensor,
        *,
        strategy: str = "beam",
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, BeamSearchOutput, GenerationOutput]:
        """Generate from source IDs using beam, greedy, or sampling strategy."""

        normalized_strategy = normalize_identifier_component(strategy, default="beam", lowercase=True)
        if normalized_strategy in {"beam", "beam_search"}:
            return self.beam_search_decode(src, return_dict=return_dict, **kwargs)
        if normalized_strategy in {"greedy", "sample", "sampling"}:
            do_sample = normalized_strategy in {"sample", "sampling"}
            return super().inference(
                src,
                max_len=kwargs.get("max_len", self.max_generation_length),
                sos_token=kwargs.get("bos_token_id", self.bos_token_id),
                eos_token=kwargs.get("eos_token_id", self.eos_token_id),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=kwargs.get("do_sample", do_sample),
                top_k=kwargs.get("top_k", self.top_k),
                return_dict=return_dict,
                capture_logits=kwargs.get("capture_logits", False),
            )
        raise self._model_error(
            "Unsupported generation strategy.",
            code=LanguageErrorCode.MODEL_INFERENCE_FAILED,
            details={"strategy": strategy, "supported": ["beam", "greedy", "sample"]},
        )

    def sequence_score(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        *,
        ignore_index: Optional[int] = None,
    ) -> SequenceScore:
        """Return loss, perplexity, token count, and mean log-probability."""

        src = self._coerce_token_tensor(src, field_name="src", device=self._device())
        tgt = self._coerce_token_tensor(tgt, field_name="tgt", device=src.device)
        if tgt.size(1) < 2:
            raise self._model_error(
                "Target sequence must contain at least two tokens for language scoring.",
                code=LanguageErrorCode.MODEL_INFERENCE_FAILED,
                details={"tgt_shape": list(tgt.shape)},
            )

        was_training = self.training
        self.eval()
        with torch.no_grad():
            decoder_input = tgt[:, :-1]
            targets = tgt[:, 1:]
            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1), device=decoder_input.device)
            logits = self.language_forward(src, decoder_input, tgt_mask=tgt_mask)
            effective_ignore = self.ignore_index if ignore_index is None else int(ignore_index)
            output = self.language_forward(src, decoder_input, tgt_mask=tgt_mask)
            if isinstance(output, dict):
                logits = output["logits"]
            elif isinstance(output, tuple):
                # BaseTransformer.forward returns (logits, memory) when return_dict=False
                logits = output[0]
            else:
                logits = output
            
            # Now logits is a tensor, safe to reshape
            loss_tensor = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=effective_ignore,
                reduction="mean",
            )
            valid_mask = targets.ne(effective_ignore)
            token_count = int(valid_mask.sum().detach().cpu().item())
            loss = float(loss_tensor.detach().cpu().item())
            perplexity = self._safe_exp(loss)
            mean_log_probability = -loss

        if was_training:
            self.train()

        self._record_language_event("sequence_score", loss=loss, perplexity=perplexity, token_count=token_count)
        return SequenceScore(
            loss=loss,
            perplexity=perplexity,
            token_count=token_count,
            mean_log_probability=mean_log_probability,
            metadata={"ignore_index": effective_ignore, "src_shape": list(src.shape), "tgt_shape": list(tgt.shape)},
        )

    def calculate_perplexity(self, src: torch.Tensor, tgt: torch.Tensor, *, ignore_index: Optional[int] = None) -> float:
        """Calculate perplexity for a source/target language modeling pair."""

        return self.sequence_score(src, tgt, ignore_index=ignore_index).perplexity

    def set_dropout(self, dropout: float) -> None:
        value = coerce_float(dropout, default=self.config.dropout, minimum=0.0, maximum=1.0)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = value
        self._record_language_event("set_dropout", dropout=value)

    def freeze_embeddings(self) -> int:
        frozen = 0
        for module in (self.src_embed, self.tgt_embed):
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameter.requires_grad = False
                    frozen += parameter.numel()
        self._record_language_event("freeze_embeddings", frozen_parameters=frozen)
        return frozen

    def unfreeze_all(self) -> int:
        unfrozen = 0
        for parameter in self.parameters():
            if not parameter.requires_grad:
                parameter.requires_grad = True
                unfrozen += parameter.numel()
        self._record_language_event("unfreeze_all", unfrozen_parameters=unfrozen)
        return unfrozen

    def initialize_language_params(self, *, embedding_std: float = 0.02, output_std: float = 0.02) -> None:
        """Apply language-model-style initialization to embeddings/projection."""

        with torch.no_grad():
            nn.init.normal_(self.src_embed.weight, mean=0.0, std=float(embedding_std))
            nn.init.normal_(self.tgt_embed.weight, mean=0.0, std=float(embedding_std))
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=float(output_std))
            for name, parameter in self.named_parameters():
                if "bias" in name and parameter.requires_grad:
                    nn.init.zeros_(parameter)
        self._record_language_event("initialize_language_params", embedding_std=embedding_std, output_std=output_std)

    def adapt_for_language_task(self, task_type: str) -> TaskAdaptationResult:
        """Adjust runtime training behavior for a supported language task."""

        task = normalize_identifier_component(task_type, default="generation", lowercase=True)
        if task not in self.VALID_TASKS:
            raise ConfigurationLanguageError(
                ConfigurationIssue(
                    code=LanguageErrorCode.CONFIG_VALUE_INVALID,
                    message="Unsupported language task adaptation requested.",
                    module="LanguageTransformer",
                    details={"task_type": task, "valid_tasks": list(self.VALID_TASKS)},
                ),
                recoverable=False,
            )

        notes: List[str] = []
        frozen = 0
        dropout_value: Optional[float] = None
        if task in {"classification", "reranking", "embedding"}:
            frozen = self.freeze_embeddings()
            notes.append("embedding parameters frozen")
        else:
            self.unfreeze_all()
            notes.append("all parameters trainable")

        if task in {"generation", "dialogue", "summarization"}:
            dropout_value = min(0.3, max(self.config.dropout, self.config.dropout + 0.05))
            self.set_dropout(dropout_value)
            notes.append("dropout increased for generative regularization")
        elif task in {"embedding", "reranking"}:
            dropout_value = min(0.15, self.config.dropout)
            self.set_dropout(dropout_value)
            notes.append("dropout constrained for representation stability")

        self.task_type = task
        result = TaskAdaptationResult(
            task_type=task,
            frozen_parameters=frozen,
            trainable_parameters=self.parameter_count(trainable_only=True),
            dropout=dropout_value,
            notes=tuple(notes),
        )
        self._record_language_event("adapt_for_language_task", result=result.to_dict())
        return result

    def save_language_model(
        self,
        path: Union[str, Path],
        *,
        lang_metadata: Optional[Mapping[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> str:
        """Save model weights with language-specific metadata."""

        target = resolve_path(path, field_name="path")
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.version,
            "saved_at": utc_timestamp(),
            "state_dict": self.state_dict(),
            "base_config": self.config.to_dict(),
            "language_config": self.language_config_snapshot(),
            "stats": self.stats().to_dict(),
            "metadata": json_safe(dict(lang_metadata or {})),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        try:
            torch.save(checkpoint, target)
        except Exception as exc:
            raise ModelLanguageError(
                ModelIssue(
                    code=LanguageErrorCode.MODEL_LOAD_FAILED,
                    message="Failed to save LanguageTransformer checkpoint.",
                    severity=Severity.ERROR,
                    module="LanguageTransformer",
                    details={"path": str(target), "exception": str(exc)},
                ),
                recoverable=False,
                cause=exc,
            ) from exc
        self._record_language_event("save_language_model", path=str(target))
        logger.info("LanguageTransformer checkpoint saved to %s", target)
        return str(target)

    @classmethod
    def load_language_model(
        cls,
        path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        strict: Optional[bool] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "LanguageTransformer":
        """Load a LanguageTransformer checkpoint."""

        target = resolve_path(path, must_exist=True, field_name="path")
        try:
            checkpoint = torch.load(target, map_location=device)
            base_config = ensure_mapping(checkpoint.get("base_config", checkpoint.get("config", {})), field_name="checkpoint.base_config", allow_none=True)
            if overrides:
                base_config.update(dict(overrides))
            model = cls(**base_config)
            language_config = ensure_mapping(checkpoint.get("language_config", {}), field_name="checkpoint.language_config", allow_none=True)
            if language_config:
                model._restore_language_runtime_config(language_config)
            model.load_state_dict(checkpoint["state_dict"], strict=model.strict_checkpoint_loading if strict is None else bool(strict))
            if device is not None:
                model = model.to(device)
            model._record_language_event("load_language_model", path=str(target))
            logger.info("LanguageTransformer checkpoint loaded from %s", target)
            return model
        except Exception as exc:
            raise ModelLanguageError(
                ModelIssue(
                    code=LanguageErrorCode.MODEL_LOAD_FAILED,
                    message="Failed to load LanguageTransformer checkpoint.",
                    severity=Severity.ERROR,
                    module="LanguageTransformer",
                    details={"path": str(target), "exception": str(exc)},
                ),
                recoverable=False,
                cause=exc,
            ) from exc

    def stats(self) -> LanguageTransformerStats:  # type: ignore[override]
        base_stats = super().stats().to_dict()
        diagnostics = getattr(self, "diagnostics", None)
        history = getattr(self, "language_history", ())
        return LanguageTransformerStats(
            request_id=base_stats.get("request_id", generate_correlation_id("lang_transformer")),
            version=getattr(self, "version", self.VERSION),
            task_type=getattr(self, "task_type", "generation"),
            base_stats=base_stats,
            beam_width=getattr(self, "beam_width", 1),
            max_generation_length=getattr(self, "max_generation_length", self.config.inference_max_len),
            temperature=getattr(self, "temperature", self.config.inference_temperature),
            length_penalty=getattr(self, "length_penalty", 0.0),
            diagnostics_count=len(getattr(diagnostics, "issues", [])),
            history_length=len(history),
            current_device=str(self._device()),
        )

    def recent_language_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        count = coerce_int(limit, default=20, minimum=1)
        return list(self.language_history)[-count:]

    def to_dict(self, *, include_history: bool = True) -> Dict[str, Any]:  # type: ignore[override]
        payload = {
            "class": self.__class__.__name__,
            "version": self.version,
            "checkpoint_format": self.CHECKPOINT_FORMAT,
            "task_type": self.task_type,
            "base_config": self.config.to_dict(),
            "language_config": self.language_config_snapshot(),
            "stats": self.stats().to_dict(),
            "diagnostics": self.diagnostics.to_list(),
        }
        if include_history:
            payload["language_history"] = list(self.language_history)
        return json_safe(payload)

    def __repr__(self) -> str:
        return (
            f"<LanguageTransformer version='{self.version}' task='{self.task_type}' "
            f"d_model={self.config.d_model} vocab={self.config.tgt_vocab_size} device='{self._device()}'>"
        )


if __name__ == "__main__":
    print("\n=== Running Language Transformer ===\n")
    printer.status("TEST", "Language Transformer initialized", "info")

    runtime_config = get_config_section("language_transformer") or {}
    test_overrides = ensure_mapping(runtime_config.get("test_overrides", {}), field_name="language_transformer.test_overrides", allow_none=True)
    if not test_overrides:
        test_overrides = {
            "src_vocab_size": 128,
            "tgt_vocab_size": 128,
            "d_model": 64,
            "nhead": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "max_len": 128,
            "inference_max_len": 12,
            "enable_memory_integration": False,
            "tie_output_projection": False,
        }

    model = LanguageTransformer(**test_overrides)
    device = torch.device("cpu")
    model = model.to(device)

    src = torch.randint(3, min(100, model.config.src_vocab_size), (2, 10), dtype=torch.long, device=device)
    tgt = torch.randint(3, min(100, model.config.tgt_vocab_size), (2, 9), dtype=torch.long, device=device)

    src_padding_mask = model.make_padding_mask(src)
    tgt_padding_mask = model.make_padding_mask(tgt)
    tgt_mask = model.generate_square_subsequent_mask(tgt.size(1), device=device)

    forward_output = model.language_forward(
        src,
        tgt,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_padding_mask,
        tgt_key_padding_mask=tgt_padding_mask,
        memory_key_padding_mask=src_padding_mask,
        return_dict=True,
    )

    loss = model.compute_loss(forward_output["logits"], tgt) # type: ignore
    loss.backward()
    grad_norm = model.clip_gradients()
    optimizer = model.configure_optimizer()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    beam_output = model.beam_search_decode(src[:1], beam_width=3, max_len=8, return_dict=True)
    greedy_output = model.generate(src[:1], strategy="greedy", max_len=8, return_dict=True, capture_logits=True)
    representation = model.encode_representations(src, pooling="mean", normalize=True)
    score = model.sequence_score(src, tgt)
    adaptation = model.adapt_for_language_task("generation")

    checkpoint_path = Path("/mnt/data/language_transformer_test_checkpoint.pt")
    saved_path = model.save_language_model(checkpoint_path, optimizer=optimizer, lang_metadata={"test": True, "task": "generation"})
    reloaded = LanguageTransformer.load_language_model(saved_path, device=device, overrides=test_overrides)

    printer.pretty("FORWARD_SHAPES", {
        "src": list(src.shape),
        "tgt": list(tgt.shape),
        "logits": list(forward_output["logits"].shape), # type: ignore
        "memory": list(forward_output["memory"].shape), # type: ignore
        "decoder_output": list(forward_output["decoder_output"].shape), # type: ignore
    }, "success")
    printer.pretty("LOSS_AND_GRAD", {"loss": float(loss.detach().cpu().item()), "grad_norm": float(grad_norm)}, "success")
    printer.pretty("BEAM_OUTPUT", beam_output.to_dict(), "success") # type: ignore
    printer.pretty("GREEDY_OUTPUT", greedy_output.to_dict() if hasattr(greedy_output, "to_dict") else {"sequences": greedy_output.detach().cpu().tolist()}, "success") # type: ignore
    printer.pretty("REPRESENTATION", representation.to_dict(), "success")
    printer.pretty("SEQUENCE_SCORE", score.to_dict(), "success")
    printer.pretty("ADAPTATION", adaptation.to_dict(), "success")
    printer.pretty("STATS", model.stats().to_dict(), "success")
    printer.pretty("RELOADED_STATS", reloaded.stats().to_dict(), "success")
    printer.pretty("CHECKPOINT_PATH", {"path": saved_path}, "success")

    print("\n=== Test ran successfully ===\n")