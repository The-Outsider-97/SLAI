from __future__ import annotations

"""
LANTRA Runtime / Inference Adapter

Core function
-------------
Bind SLAI's trained LANTRA LanguageTransformer checkpoint to the existing
LanguageTokenizer and expose stable inference contracts for the seven LANTRA
tasks without duplicating tokenizer, transformer, checkpoint, or model logic.

Ownership
---------
LanguageTokenizer
    Owns BPE tokenization and BPE resources.

LanguageTransformer
    Owns the encoder-decoder architecture, generation, representation extraction,
    sequence scoring, and checkpoint loading.

LantraRuntime
    Owns inference-time binding of tokenizer + trained checkpoint, exact task
    prompt serialization, device placement, runtime validation, batching for
    representation tasks, and task-facing inference APIs.

NLGEngine
    Owns template/neural/hybrid response-generation policy.

LanguageAgent
    Owns orchestration and decides whether LANTRA is enabled for the agent.

Important
---------
This module deliberately does not train, fine-tune, mutate task-specific model
parameters, perform retrieval, or duplicate KnowledgeAgent search. It consumes
the already-trained LANTRA checkpoint and can embed/rerank caller-supplied
candidates.
"""

import threading
import time as time_module
import torch

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from .modules.language_tokenizer import LanguageTokenizer
from .modules.language_transformer import LanguageTransformer
from .utils.config_loader import get_config_section
from .utils.language_error import *
from .utils.language_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]


logger = get_logger("LANTRA Runtime")
printer = PrettyPrinter()

TextSequence = Sequence[str]
DialogueInput = Union[str, Sequence[Mapping[str, Any]]]


@dataclass(frozen=True)
class LantraTextResult:
    task: str
    text: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "text": self.text,
            "latency_ms": self.latency_ms,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class LantraClassificationResult:
    label: str
    mode: str
    latency_ms: float
    candidate_losses: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "mode": self.mode,
            "latency_ms": self.latency_ms,
            "candidate_losses": dict(self.candidate_losses),
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class LantraRankedCandidate:
    index: int
    text: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LantraRerankResult:
    query: str
    candidates: Tuple[LantraRankedCandidate, ...]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "candidates": [item.to_dict() for item in self.candidates],
            "latency_ms": self.latency_ms,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class LantraRuntimeStats:
    version: str
    ready: bool
    checkpoint_path: str
    device: str
    model_parameters: int
    d_model: int
    src_vocab_size: int
    tgt_vocab_size: int
    tokenizer_vocab_size: int
    generation_calls: int
    classification_calls: int
    embedding_calls: int
    rerank_calls: int
    total_inference_calls: int
    total_inference_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LantraRuntime:
    """Inference-only runtime for the trained LANTRA small language model."""

    VERSION = "1.0"
    SUPPORTED_SEQ2SEQ_TASKS: Tuple[str, ...] = (
        "generation",
        "classification",
        "translation",
        "summarization",
        "dialogue",
    )

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        configured = get_config_section("lantra_runtime") or {}
        self.config: Dict[str, Any] = dict(configured)
        if config:
            self.config.update(dict(config))

        self.version = ensure_text(self.config.get("version", self.VERSION))
        self.checkpoint_path = ensure_text(
            self.config.get(
                "checkpoint_path",
                "src/agents/language/checkpoints/lantra/lantra_20260919T142819Z.pt",
            )
        )
        self.device_policy = ensure_text(self.config.get("device", "auto")).strip().lower()
        self.strict_checkpoint_loading = coerce_bool(
            self.config.get("strict_checkpoint_loading", True),
            default=True,
        )
        self.require_trained_tokenizer = coerce_bool(
            self.config.get("require_trained_tokenizer", True),
            default=True,
        )
        self.verify_vocab_compatibility = coerce_bool(
            self.config.get("verify_vocab_compatibility", True),
            default=True,
        )

        inference = ensure_mapping(
            self.config.get("inference", {}),
            field_name="lantra_runtime.inference",
            allow_none=True,
        )
        self.generation_strategy = ensure_text(
            inference.get("generation_strategy", "greedy")
        ).strip().lower()
        if self.generation_strategy not in {"greedy", "beam", "beam_search", "sample", "sampling"}:
            self._raise_config(
                "Unsupported LANTRA generation strategy.",
                details={"generation_strategy": self.generation_strategy},
            )

        self.source_max_length = coerce_int(
            inference.get("source_max_length", 384),
            default=384,
            minimum=8,
        )
        self.target_max_length = coerce_int(
            inference.get("target_max_length", 160),
            default=160,
            minimum=2,
        )
        self.embedding_max_length = coerce_int(
            inference.get("embedding_max_length", 192),
            default=192,
            minimum=8,
        )
        self.embedding_pooling = ensure_text(
            inference.get("embedding_pooling", "mean")
        ).strip().lower()
        self.normalize_embeddings = coerce_bool(
            inference.get("normalize_embeddings", True),
            default=True,
        )
        self.temperature = coerce_float(
            inference.get("temperature", 1.0),
            default=1.0,
            minimum=1e-6,
        )
        self.top_k = coerce_int(
            inference.get("top_k", 0),
            default=0,
            minimum=0,
        )
        self.default_rerank_top_k = coerce_int(
            inference.get("rerank_top_k", 10),
            default=10,
            minimum=1,
        )

        self._lock = threading.RLock()
        self._loaded_at: Optional[float] = None
        self._generation_calls = 0
        self._classification_calls = 0
        self._embedding_calls = 0
        self._rerank_calls = 0
        self._total_inference_calls = 0
        self._total_inference_ms = 0.0

        self.device = self._resolve_device(self.device_policy)
        self.tokenizer: LanguageTokenizer
        self.model: LanguageTransformer
        self._load_runtime()

    # ------------------------------------------------------------------
    # Initialization and validation
    # ------------------------------------------------------------------
    def _resolve_device(self, policy: str) -> torch.device:
        value = (policy or "auto").strip().lower()
        if value == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        try:
            device = torch.device(value)
        except Exception as exc:
            self._raise_config(
                "Invalid LANTRA device configuration.",
                details={"device": value, "exception": str(exc)},
                cause=exc,
            )
            raise AssertionError("unreachable")

        if device.type == "cuda" and not torch.cuda.is_available():
            self._raise_config(
                "LANTRA is configured for CUDA, but CUDA is not available.",
                details={"device": value},
            )
        if device.type == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                self._raise_config(
                    "LANTRA is configured for MPS, but MPS is not available.",
                    details={"device": value},
                )
        return device

    def _load_runtime(self) -> None:
        checkpoint = Path(self.checkpoint_path).expanduser()
        if not checkpoint.exists():
            self._raise_model(
                "LANTRA checkpoint was not found.",
                code=LanguageErrorCode.MODEL_UNAVAILABLE,
                details={"checkpoint_path": str(checkpoint)},
            )

        started = time_module.perf_counter()
        try:
            tokenizer = LanguageTokenizer()
            if self.require_trained_tokenizer and not bool(getattr(tokenizer, "is_trained", False)):
                self._raise_model(
                    "LANTRA requires a trained LanguageTokenizer.",
                    code=LanguageErrorCode.MODEL_UNAVAILABLE,
                    details={
                        "bpe_model_path": str(getattr(tokenizer, "bpe_model_file", "")),
                        "bpe_vocab_path": str(getattr(tokenizer, "bpe_vocab_file", "")),
                    },
                )

            model = LanguageTransformer.load_language_model(
                checkpoint,
                device=self.device,
                strict=self.strict_checkpoint_loading,
            )
            model.eval()

            self.tokenizer = tokenizer
            self.model = model
            self._validate_model_tokenizer_contract()
            self._validate_sequence_limits()
            self._loaded_at = time_module.time()
        except (ConfigurationLanguageError, ModelLanguageError):
            raise
        except Exception as exc:
            self._raise_model(
                "Failed to initialize LANTRA runtime.",
                code=LanguageErrorCode.MODEL_LOAD_FAILED,
                details={
                    "checkpoint_path": str(checkpoint),
                    "device": str(self.device),
                    "exception": str(exc),
                },
                cause=exc,
            )

        elapsed_ms = (time_module.perf_counter() - started) * 1000.0
        logger.info(
            "LANTRA runtime loaded checkpoint=%s device=%s tokenizer_vocab=%s d_model=%s in %.3f ms",
            checkpoint,
            self.device,
            len(self.tokenizer.vocab),
            self.model.config.d_model,
            elapsed_ms,
        )
        printer.status(
            "INIT",
            f"LANTRA runtime ready: {checkpoint.name} on {self.device}",
            "success",
        )

    def _validate_model_tokenizer_contract(self) -> None:
        if not self.verify_vocab_compatibility:
            return

        tokenizer_vocab = len(self.tokenizer.vocab)
        src_vocab = int(self.model.config.src_vocab_size)
        tgt_vocab = int(self.model.config.tgt_vocab_size)
        problems: List[str] = []

        if tokenizer_vocab != src_vocab:
            problems.append(
                f"tokenizer vocab ({tokenizer_vocab}) != model src vocab ({src_vocab})"
            )
        if tokenizer_vocab != tgt_vocab:
            problems.append(
                f"tokenizer vocab ({tokenizer_vocab}) != model tgt vocab ({tgt_vocab})"
            )

        special_pairs = {
            "pad_token_id": (
                int(self.tokenizer.pad_token_id),
                int(self.model.pad_token_id),
            ),
            "bos_token_id": (
                int(self.tokenizer.bos_token_id),
                int(self.model.bos_token_id),
            ),
            "eos_token_id": (
                int(self.tokenizer.eos_token_id),
                int(self.model.eos_token_id),
            ),
        }
        for name, (tokenizer_value, model_value) in special_pairs.items():
            if tokenizer_value != model_value:
                problems.append(
                    f"{name}: tokenizer={tokenizer_value}, model={model_value}"
                )

        if problems:
            self._raise_model(
                "LANTRA tokenizer/model contract mismatch.",
                code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                details={
                    "problems": problems,
                    "checkpoint_path": self.checkpoint_path,
                },
            )

    def _validate_sequence_limits(self) -> None:
        max_positions = int(self.model.config.max_position_embeddings)
        configured = {
            "source_max_length": self.source_max_length,
            "target_max_length": self.target_max_length,
            "embedding_max_length": self.embedding_max_length,
        }
        too_large = {
            key: value
            for key, value in configured.items()
            if value > max_positions
        }
        if too_large:
            self._raise_config(
                "LANTRA inference sequence limit exceeds model positional capacity.",
                details={
                    "max_position_embeddings": max_positions,
                    "configured": configured,
                    "invalid": too_large,
                },
            )

    @property
    def ready(self) -> bool:
        return (
            hasattr(self, "model")
            and hasattr(self, "tokenizer")
            and self.model is not None
            and self.tokenizer is not None
            and bool(getattr(self.tokenizer, "is_trained", False))
        )

    # ------------------------------------------------------------------
    # Exact LANTRA task prompt contracts
    # ------------------------------------------------------------------
    @staticmethod
    def _generation_source(prompt: str) -> str:
        return f"task: generation\nprompt:\n{prompt}\nresponse:"

    @staticmethod
    def _classification_source(text: str) -> str:
        return f"task: classification\ntext:\n{text}\nlabel:"

    @staticmethod
    def _translation_source(
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        return (
            "task: translation\n"
            f"source_language: {source_language}\n"
            f"target_language: {target_language}\n"
            f"source:\n{text}\ntranslation:"
        )

    @staticmethod
    def _summarization_source(text: str) -> str:
        return f"task: summarization\ndocument:\n{text}\nsummary:"

    @classmethod
    def _dialogue_source(cls, history: DialogueInput) -> str:
        conversation = cls._format_dialogue_history(history)
        return f"task: dialogue\nconversation:\n{conversation}\nassistant:"

    @staticmethod
    def _embedding_source(text: str) -> str:
        return f"task: embedding\ntext:\n{text}"

    @staticmethod
    def _rerank_query_source(query: str) -> str:
        return f"task: reranking\nquery:\n{query}"

    @staticmethod
    def _rerank_document_source(document: str) -> str:
        return f"task: reranking\ndocument:\n{document}"

    @staticmethod
    def _format_dialogue_history(history: DialogueInput) -> str:
        if isinstance(history, str):
            value = history.strip()
            if not value:
                raise ValueError("Dialogue history cannot be empty.")
            return value

        if not isinstance(history, Sequence):
            raise TypeError(
                "Dialogue history must be a string or sequence of role/content mappings."
            )

        turns: List[str] = []
        for index, turn in enumerate(history):
            if not isinstance(turn, Mapping):
                raise TypeError(
                    f"Dialogue history item {index} must be a mapping."
                )
            role = ensure_text(turn.get("role", "unknown")).strip().lower() or "unknown"
            content = ensure_text(
                turn.get("content", turn.get("text", turn.get("message", "")))
            ).strip()
            if not content:
                raise ValueError(
                    f"Dialogue history item {index} has no content."
                )
            turns.append(f"{role}: {content}")

        if not turns:
            raise ValueError("Dialogue history cannot be empty.")
        return "\n".join(turns)

    # ------------------------------------------------------------------
    # Token/model adapters
    # ------------------------------------------------------------------
    def _encode_source(
        self,
        text: str,
        *,
        max_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        payload = self.tokenizer.encode(
            ensure_text(text),
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
            return_token_metadata=False,
            return_tokens=False,
            return_tensors="pt",
        )
        ids = payload.get("input_ids")
        mask = payload.get("attention_mask")
        if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
            self._raise_model(
                "LanguageTokenizer.encode did not return tensor input_ids and attention_mask.",
                code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
            )
        ids_tensor = cast(torch.Tensor, ids)
        mask_tensor = cast(torch.Tensor, mask)
        if ids_tensor.dim() != 1 or mask_tensor.dim() != 1:
            self._raise_model(
                "Unexpected LANTRA tokenizer tensor shape.",
                code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                details={
                    "input_ids_shape": list(ids_tensor.shape),
                    "attention_mask_shape": list(mask_tensor.shape),
                },
            )
        return (
            ids_tensor.unsqueeze(0).to(self.device),
            mask_tensor.unsqueeze(0).to(self.device),
        )

    def _generate_source(
        self,
        source_text: str,
        *,
        task: str,
        max_length: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> LantraTextResult:
        if not self.ready:
            self._raise_model(
                "LANTRA runtime is not ready.",
                code=LanguageErrorCode.MODEL_UNAVAILABLE,
            )

        started = time_module.perf_counter()
        effective_strategy = ensure_text(
            strategy or self.generation_strategy
        ).strip().lower()
        effective_max_length = coerce_int(
            max_length if max_length is not None else self.target_max_length,
            default=self.target_max_length,
            minimum=2,
            maximum=int(self.model.config.max_position_embeddings),
        )

        with self._lock, torch.inference_mode():
            src, _mask = self._encode_source(
                source_text,
                max_length=self.source_max_length,
            )
            generated = self.model.generate(
                src,
                strategy=effective_strategy,
                max_len=effective_max_length,
                temperature=self.temperature,
                top_k=self.top_k,
                return_dict=False,
            )
            if not isinstance(generated, torch.Tensor):
                sequences = getattr(generated, "sequences", None)
                if not isinstance(sequences, torch.Tensor):
                    self._raise_model(
                        "LanguageTransformer.generate returned an unsupported output type.",
                        code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                        details={"type": type(generated).__name__},
                    )
                    raise TypeError("LanguageTransformer.generate returned no tensor sequence.")
                generated = sequences

            if not isinstance(generated, torch.Tensor):
                raise TypeError("LanguageTransformer.generate returned no tensor sequence.")

            if generated.dim() != 2 or generated.size(0) < 1:
                self._raise_model(
                    "LanguageTransformer.generate returned an invalid sequence shape.",
                    code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                    details={"shape": list(generated.shape)},
                )

            text = self.tokenizer.decode(
                generated[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

        latency_ms = (time_module.perf_counter() - started) * 1000.0
        self._generation_calls += 1
        self._record_call(latency_ms)
        return LantraTextResult(
            task=task,
            text=text,
            latency_ms=latency_ms,
            metadata={
                "strategy": effective_strategy,
                "device": str(self.device),
                "max_length": effective_max_length,
            },
        )

    def _embedding_batch(
        self,
        texts: Sequence[str],
        *,
        prefix_builder: Any,
    ) -> torch.Tensor:
        values = [ensure_text(text).strip() for text in texts]
        if not values or any(not value for value in values):
            raise ValueError("Embedding inputs must contain non-empty text.")

        encoded_ids: List[torch.Tensor] = []
        encoded_masks: List[torch.Tensor] = []
        for value in values:
            ids, mask = self._encode_source(
                prefix_builder(value),
                max_length=self.embedding_max_length,
            )
            encoded_ids.append(ids.squeeze(0))
            encoded_masks.append(mask.squeeze(0))

        max_len = max(int(item.size(0)) for item in encoded_ids)
        batch_size = len(encoded_ids)
        ids_batch = torch.full(
            (batch_size, max_len),
            fill_value=int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        mask_batch = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=self.device,
        )

        for row, (ids, mask) in enumerate(zip(encoded_ids, encoded_masks)):
            length = int(ids.size(0))
            ids_batch[row, :length] = ids
            mask_batch[row, :length] = mask

        output = self.model.encode_representations(
            ids_batch,
            attention_mask=mask_batch,
            pooling=self.embedding_pooling,
            normalize=self.normalize_embeddings,
        )
        return output.embeddings

    # ------------------------------------------------------------------
    # Public seven-task inference API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_length: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> LantraTextResult:
        value = ensure_text(prompt).strip()
        if not value:
            raise ValueError("Generation prompt cannot be empty.")
        return self._generate_source(
            self._generation_source(value),
            task="generation",
            max_length=max_length,
            strategy=strategy,
        )

    def summarize(
        self,
        text: str,
        *,
        max_length: Optional[int] = None,
    ) -> LantraTextResult:
        value = ensure_text(text).strip()
        if not value:
            raise ValueError("Summarization input cannot be empty.")
        return self._generate_source(
            self._summarization_source(value),
            task="summarization",
            max_length=max_length,
        )

    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        max_length: Optional[int] = None,
    ) -> LantraTextResult:
        value = ensure_text(text).strip()
        source = ensure_text(source_language).strip()
        target = ensure_text(target_language).strip()
        if not value:
            raise ValueError("Translation input cannot be empty.")
        if not source or not target:
            raise ValueError(
                "source_language and target_language are required for LANTRA translation."
            )
        return self._generate_source(
            self._translation_source(value, source, target),
            task="translation",
            max_length=max_length,
        )

    def dialogue(
        self,
        history: DialogueInput,
        *,
        max_length: Optional[int] = None,
    ) -> LantraTextResult:
        return self._generate_source(
            self._dialogue_source(history),
            task="dialogue",
            max_length=max_length,
        )

    def classify(
        self,
        text: str,
        *,
        labels: Optional[Sequence[str]] = None,
        max_length: Optional[int] = None,
    ) -> LantraClassificationResult:
        value = ensure_text(text).strip()
        if not value:
            raise ValueError("Classification input cannot be empty.")

        started = time_module.perf_counter()
        source_text = self._classification_source(value)

        if labels:
            candidates = []
            seen = set()
            for raw_label in labels:
                label = ensure_text(raw_label).strip()
                if label and label not in seen:
                    candidates.append(label)
                    seen.add(label)
            if not candidates:
                raise ValueError("Classification labels cannot be empty.")

            losses: Dict[str, float] = {}
            with self._lock, torch.inference_mode():
                src, _src_mask = self._encode_source(
                    source_text,
                    max_length=self.source_max_length,
                )
                for label in candidates:
                    target, _target_mask = self._encode_source(
                        label,
                        max_length=self.target_max_length,
                    )
                    score = self.model.sequence_score(
                        src,
                        target,
                        ignore_index=int(self.tokenizer.pad_token_id),
                    )
                    losses[label] = float(score.loss)

            selected = min(losses, key=losses.get) # type: ignore
            mode = "candidate_sequence_score"
        else:
            generated = self._generate_source(
                source_text,
                task="classification",
                max_length=max_length,
            )
            selected = generated.text.strip()
            losses = {}
            mode = "generated_label"

        latency_ms = (time_module.perf_counter() - started) * 1000.0
        self._classification_calls += 1
        if labels:
            self._record_call(latency_ms)

        return LantraClassificationResult(
            label=selected,
            mode=mode,
            latency_ms=latency_ms,
            candidate_losses=losses,
            metadata={"device": str(self.device)},
        )

    def embed(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        values = [text] if isinstance(text, str) else list(text)
        started = time_module.perf_counter()
        with self._lock, torch.inference_mode():
            embeddings = self._embedding_batch(values, prefix_builder=self._embedding_source)
        latency_ms = (time_module.perf_counter() - started) * 1000.0
        self._embedding_calls += 1
        self._record_call(latency_ms)
        return embeddings

    def rerank(self, query: str, candidates: Sequence[str], *, top_k: Optional[int] = None) -> LantraRerankResult:
        query_text = ensure_text(query).strip()
        documents = [ensure_text(item).strip() for item in candidates]
        documents = [item for item in documents if item]
        if not query_text:
            raise ValueError("Reranking query cannot be empty.")
        if not documents:
            raise ValueError("Reranking requires at least one non-empty candidate.")

        started = time_module.perf_counter()
        with self._lock, torch.inference_mode():
            query_embedding = self._embedding_batch([query_text], prefix_builder=self._rerank_query_source)
            document_embeddings = self._embedding_batch(documents, prefix_builder=self._rerank_document_source)
            # Training uses normalized representations and dot-product cosine
            # similarity. Keep inference mathematically identical.
            scores = torch.matmul(
                document_embeddings,
                query_embedding[0].unsqueeze(-1),
            ).squeeze(-1)

        requested_top_k = self.default_rerank_top_k if top_k is None else int(top_k)
        effective_top_k = max(1, min(requested_top_k, len(documents)))
        ranked_indices = torch.argsort(scores, descending=True).tolist()[:effective_top_k]

        ranked = tuple(
            LantraRankedCandidate(
                index=int(index),
                text=documents[int(index)],
                score=float(scores[int(index)].detach().cpu().item()),
            )
            for index in ranked_indices
        )

        latency_ms = (time_module.perf_counter() - started) * 1000.0
        self._rerank_calls += 1
        self._record_call(latency_ms)
        return LantraRerankResult(
            query=query_text,
            candidates=ranked,
            latency_ms=latency_ms,
            metadata={
                "device": str(self.device),
                "candidate_count": len(documents),
                "top_k": effective_top_k,
            },
        )

    # ------------------------------------------------------------------
    # NLGEngine neural-generator contract
    # ------------------------------------------------------------------
    def nlg_generate(self, prompt: str, frame: Any, context: Mapping[str, Any]) -> str:
        """
        Adapter matching NLGEngine.NeuralGenerator:
            Callable[[str, LinguisticFrame, Mapping[str, Any]], str]

        NLGEngine owns prompt construction and neural/template/hybrid policy.
        LANTRA owns the model-facing `task: generation` serialization.
        """
        del frame, context  # NLGEngine already serialized these into `prompt`.
        return self.generate(prompt).text

    # ------------------------------------------------------------------
    # Lifecycle / observability
    # ------------------------------------------------------------------
    def reload(self, checkpoint_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        with self._lock:
            if checkpoint_path is not None:
                self.checkpoint_path = str(checkpoint_path)
            self._load_runtime()
        return self.health_check()

    def _record_call(self, latency_ms: float) -> None:
        self._total_inference_calls += 1
        self._total_inference_ms += float(latency_ms)

    def stats(self) -> LantraRuntimeStats:
        parameter_count = 0
        if hasattr(self, "model"):
            counter = cast(Any, getattr(self.model, "parameter_count", None))
            if callable(counter):
                try:
                    parameter_count = int(cast(Any, counter(trainable_only=False)))
                except TypeError:
                    parameter_count = int(cast(Any, counter()))
            else:
                parameter_count = sum(
                    int(parameter.numel())
                    for parameter in self.model.parameters()
                )

        model_config = getattr(getattr(self, "model", None), "config", None)
        return LantraRuntimeStats(
            version=self.version,
            ready=self.ready,
            checkpoint_path=self.checkpoint_path,
            device=str(self.device),
            model_parameters=parameter_count,
            d_model=int(getattr(model_config, "d_model", 0) or 0),
            src_vocab_size=int(getattr(model_config, "src_vocab_size", 0) or 0),
            tgt_vocab_size=int(getattr(model_config, "tgt_vocab_size", 0) or 0),
            tokenizer_vocab_size=len(getattr(self.tokenizer, "vocab", {}))
            if hasattr(self, "tokenizer")
            else 0,
            generation_calls=self._generation_calls,
            classification_calls=self._classification_calls,
            embedding_calls=self._embedding_calls,
            rerank_calls=self._rerank_calls,
            total_inference_calls=self._total_inference_calls,
            total_inference_ms=round(self._total_inference_ms, 3),
        )

    def health_check(self) -> Dict[str, Any]:
        stats = self.stats().to_dict()
        stats.update(
            {
                "health": "healthy" if self.ready else "degraded",
                "loaded_at": self._loaded_at,
                "tokenizer_trained": bool(
                    getattr(getattr(self, "tokenizer", None), "is_trained", False)
                ),
                "generation_strategy": self.generation_strategy,
                "source_max_length": self.source_max_length,
                "target_max_length": self.target_max_length,
                "embedding_max_length": self.embedding_max_length,
                "embedding_pooling": self.embedding_pooling,
                "normalize_embeddings": self.normalize_embeddings,
            }
        )
        return stats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "version": self.version,
            "health": self.health_check(),
            "config": json_safe(self.config),
        }

    # ------------------------------------------------------------------
    # Existing language-error hierarchy reuse
    # ------------------------------------------------------------------
    @staticmethod
    def _raise_config(
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        issue = ConfigurationIssue(
            code=LanguageErrorCode.CONFIG_VALUE_INVALID,
            message=message,
            severity=Severity.ERROR,
            module="LantraRuntime",
            recoverable=False,
            details=dict(details or {}),
        )
        raise ConfigurationLanguageError(
            issue,
            recoverable=False,
            cause=cause,
        )

    @staticmethod
    def _raise_model(
        message: str,
        *,
        code: LanguageErrorCode = LanguageErrorCode.MODEL_INFERENCE_FAILED,
        details: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        issue = ModelIssue(
            code=code,
            message=message,
            severity=Severity.ERROR,
            module="LantraRuntime",
            recoverable=False,
            details=dict(details or {}),
        )
        raise ModelLanguageError(
            issue,
            recoverable=False,
            cause=cause,
        )


__all__ = [
    "LantraTextResult",
    "LantraClassificationResult",
    "LantraRankedCandidate",
    "LantraRerankResult",
    "LantraRuntimeStats",
    "LantraRuntime",
]
