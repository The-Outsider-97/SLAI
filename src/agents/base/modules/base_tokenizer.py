from __future__ import annotations

import json
import torch
import unicodedata
import regex as re

from dataclasses import dataclass, field
from collections import Counter, deque
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Deque, Dict, List, Optional, Tuple, Union, cast

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_errors import *
from ..utils.base_helpers import *
from ..base_memory import BaseMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Base Tokenizer")
printer = PrettyPrinter()


@dataclass(frozen=True)
class TokenizerStats:
    """Operational summary for the tokenizer."""

    request_id: str
    vocab_size: int
    special_token_count: int
    trained: bool
    training_runs: int
    encode_calls: int
    decode_calls: int
    tokenize_calls: int
    added_tokens: int
    history_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "vocab_size": self.vocab_size,
            "special_token_count": self.special_token_count,
            "trained": self.trained,
            "training_runs": self.training_runs,
            "encode_calls": self.encode_calls,
            "decode_calls": self.decode_calls,
            "tokenize_calls": self.tokenize_calls,
            "added_tokens": self.added_tokens,
            "history_length": self.history_length,
        }


@dataclass
class TrainingSummary:
    """Deterministic training summary for vocabulary-building runs."""

    corpus_size: int
    min_frequency: int
    max_vocab_size: int
    learned_tokens: int
    total_unique_observed_tokens: int
    total_observed_tokens: int
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_size": self.corpus_size,
            "min_frequency": self.min_frequency,
            "max_vocab_size": self.max_vocab_size,
            "learned_tokens": self.learned_tokens,
            "total_unique_observed_tokens": self.total_unique_observed_tokens,
            "total_observed_tokens": self.total_observed_tokens,
            "timestamp": self.timestamp,
        }


class BaseTokenizer:
    """
    Core tokenizer for the base subsystem.

    This implementation provides a deterministic, Unicode-aware tokenization
    pipeline with configurable normalization, vocabulary management,
    encode/decode helpers, persistence, bounded operational history, and
    optional memory integration through BaseMemory.

    The tokenizer is intentionally generic and lightweight so it can serve as a
    baseline text-processing component for experimentation, infrastructure
    tooling, preprocessing, and small model-serving flows without requiring a
    heavyweight external tokenizer runtime.
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.bt_config = get_config_section("base_tokenizer") or {}
        self.encode_config = get_config_section("encode") or {}
        self.decode_config = get_config_section("decode") or {}

        self._lock_request_id = generate_request_id("tokenizer", include_timestamp=True)
        self._created_at = utc_now_iso()

        self.src_vocab_size = coerce_int(
            self.bt_config.get("src_vocab_size", self.config.get("src_vocab_size", 30000)),
            default=30000,
            minimum=5,
        )
        self.max_vocab_size = coerce_int(
            self.bt_config.get("max_vocab_size", self.src_vocab_size),
            default=self.src_vocab_size,
            minimum=5,
        )
        self.min_frequency = coerce_int(
            self.bt_config.get("min_frequency", 2),
            default=2,
            minimum=1,
        )
        self.lowercase = coerce_bool(self.bt_config.get("lowercase", True), default=True)
        self.remove_control_characters = coerce_bool(
            self.bt_config.get("remove_control_characters", True),
            default=True,
        )
        self.squeeze_whitespace = coerce_bool(
            self.bt_config.get("squeeze_whitespace", True),
            default=True,
        )
        self.track_token_counts = coerce_bool(
            self.bt_config.get("track_token_counts", True),
            default=True,
        )
        self.allow_dynamic_vocab_growth = coerce_bool(
            self.bt_config.get("allow_dynamic_vocab_growth", True),
            default=True,
        )
        self.record_history = coerce_bool(
            self.bt_config.get("record_history", True),
            default=True,
        )
        self.history_limit = coerce_int(
            self.bt_config.get("history_limit", 200),
            default=200,
            minimum=10,
        )
        self.enable_memory_integration = coerce_bool(
            self.bt_config.get("enable_memory_integration", False),
            default=False,
        )
        self.memory_namespace = normalize_identifier(
            self.bt_config.get("memory_namespace", "base_tokenizer"),
            lowercase=True,
            separator="_",
            max_length=120,
        )
        self.snapshot_pretty = coerce_bool(
            self.bt_config.get("snapshot_pretty", True),
            default=True,
        )
        self.default_max_length = coerce_int(
            self.bt_config.get("default_max_length", self.encode_config.get("max_length", 128)),
            default=128,
            minimum=1,
        )
        self.default_padding = ensure_text(
            self.bt_config.get("default_padding", self.encode_config.get("padding", "max_length"))
        )
        self.default_truncation = coerce_bool(
            self.bt_config.get("default_truncation", self.encode_config.get("truncation", True)),
            default=True,
        )
        self.default_add_special_tokens = coerce_bool(
            self.bt_config.get("default_add_special_tokens", self.encode_config.get("add_special_tokens", True)),
            default=True,
        )
        self.default_return_tensors = self.bt_config.get(
            "default_return_tensors",
            self.encode_config.get("return_tensors", None),
        )
        self.return_attention_mask = coerce_bool(
            self.bt_config.get("return_attention_mask", True),
            default=True,
        )
        self.return_special_tokens_mask = coerce_bool(
            self.bt_config.get("return_special_tokens_mask", False),
            default=False,
        )
        self.return_tokens = coerce_bool(
            self.bt_config.get("return_tokens", False),
            default=False,
        )
        self.default_skip_special_tokens = coerce_bool(
            self.decode_config.get("skip_special_tokens", True),
            default=True,
        )
        self.default_clean_up_tokenization_spaces = coerce_bool(
            self.decode_config.get("clean_up_tokenization_spaces", True),
            default=True,
        )

        normalization_form = ensure_text(self.bt_config.get("unicode_normalization_form", "NFKC")).upper()
        ensure_one_of(
            normalization_form,
            ["NFC", "NFD", "NFKC", "NFKD"],
            "unicode_normalization_form",
            config=self.bt_config,
            error_cls=BaseConfigurationError,
        )
        self.unicode_normalization_form = normalization_form

        self.bpe_model_path = self._path_or_none(self.bt_config.get("bpe_model_path", self.config.get("bpe_model_path", "")))
        self.bpe_vocab_path = self._path_or_none(self.bt_config.get("bpe_vocab_path", self.config.get("bpe_vocab_path", "")))

        self.pad_token = ensure_non_empty_string(
            self.bt_config.get("pad_token", "[PAD]"),
            "pad_token",
            config=self.bt_config,
            error_cls=BaseConfigurationError,
        )
        self.unk_token = ensure_non_empty_string(
            self.bt_config.get("unk_token", "[UNK]"),
            "unk_token",
            config=self.bt_config,
            error_cls=BaseConfigurationError,
        )
        self.bos_token = ensure_non_empty_string(
            self.bt_config.get("bos_token", "[BOS]"),
            "bos_token",
            config=self.bt_config,
            error_cls=BaseConfigurationError,
        )
        self.eos_token = ensure_non_empty_string(
            self.bt_config.get("eos_token", "[EOS]"),
            "eos_token",
            config=self.bt_config,
            error_cls=BaseConfigurationError,
        )
        self.mask_token = ensure_non_empty_string(
            self.bt_config.get("mask_token", "[MASK]"),
            "mask_token",
            config=self.bt_config,
            error_cls=BaseConfigurationError,
        )

        parsed_special_tokens = parse_delimited_text(self.bt_config.get("special_tokens", []), unique=True)
        self.special_tokens = self._build_special_tokens(parsed_special_tokens)
        self.special_token_set = set(self.special_tokens)

        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.token_counts: Counter[str] = Counter()
        self.is_trained = False
        self.training_runs = 0
        self.encode_calls = 0
        self.decode_calls = 0
        self.tokenize_calls = 0
        self.added_tokens_count = 0
        self.last_training_summary: Optional[TrainingSummary] = None
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)
        self.shared_memory: Optional[BaseMemory] = None

        self._reinitialize_vocab()
        self._rebuild_token_pattern()
        self._init_memory_integration()
        self._record_event("init", vocab_size=len(self.vocab), special_tokens=self.special_tokens)

        printer.status("INIT", f"Base Tokenizer initialized with vocab size: {len(self.vocab)}", "success")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _path_or_none(self, value: Any) -> Optional[Path]:
        if value in (None, "", "none", "None"):
            return None
        return Path(ensure_text(value))

    def _build_special_tokens(self, configured_tokens: Sequence[str]) -> List[str]:
        ordered: List[str] = []
        for token in list(configured_tokens) + [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.mask_token,
        ]:
            normalized = ensure_non_empty_string(
                token,
                "special_token",
                config=self.bt_config,
                error_cls=BaseConfigurationError,
            )
            if normalized not in ordered:
                ordered.append(normalized)
        return ordered

    def _reinitialize_vocab(self) -> None:
        self.vocab = {token: index for index, token in enumerate(self.special_tokens)}
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}

    def _rebuild_token_pattern(self) -> None:
        escaped_special_tokens = sorted(
            (re.escape(token) for token in self.special_tokens),
            key=len,
            reverse=True
        )
        special_group = "|".join(escaped_special_tokens) if escaped_special_tokens else r"(?!x)x"
    
        # Try to use the 'regex' library if available (supports \p{...})
        try:
            # Check if we are actually using the 'regex' module
            if hasattr(re, 'UNICODE') and 'regex' in str(re):
                # Build pattern with Unicode property escapes
                pattern = (
                    rf"(?:{special_group})|"
                    r"(?:\p{L}+(?:['’]\p{L}+)*)|"
                    r"(?:\p{N}+(?:[\.,:/-]\p{N}+)*)|"
                    r"(?:[\p{Extended_Pictographic}\p{Emoji_Presentation}])|"
                    r"(?:[^\s])"
                )
                self._token_pattern = re.compile(pattern)
                return
        except Exception:
            pass
    
        # Fallback for standard 're' – no Unicode property escapes
        # Word: sequence of Unicode letters (including accented) and apostrophe
        # Number: digits and internal punctuation . , : / -
        # Emoji: a very broad range; for simplicity we capture any non‑space character
        #        that is not matched by the previous rules.
        word_pattern = r"(?:\w+(?:['’]\w+)*)"          # \w with re.UNICODE matches letters/digits/underscore
        number_pattern = r"(?:\d+(?:[\.,:/-]\d+)*)"   # only ASCII digits (Unicode digits would need \d with re.UNICODE)
        # is not a space. In practice, this works for most emojis.
        emoji_pattern = r"(?:[^\u0000-\uFFFF\s])"
        other_pattern = r"(?:[^\s])"                  # any remaining non‑space character
    
        pattern = (
            rf"(?:{special_group})|"
            rf"{word_pattern}|"
            rf"{number_pattern}|"
            rf"{emoji_pattern}|"
            rf"{other_pattern}"
        )
        self._token_pattern = re.compile(pattern, re.UNICODE)

    def _init_memory_integration(self) -> None:
        if not self.enable_memory_integration:
            return
        try:
            self.shared_memory = BaseMemory()
        except Exception as exc:
            wrapped = BaseInitializationError.wrap(
                exc,
                message="BaseTokenizer failed to initialize optional BaseMemory integration.",
                config=self.bt_config,
                component="BaseTokenizer",
                operation="__init__",
            )
            logger.warning(wrapped.message)
            self.shared_memory = None

    def _record_event(self, action: str, **payload: Any) -> None:
        if not self.record_history:
            return
        event = {
            "timestamp": utc_now_iso(),
            "action": action,
            "payload": drop_none_values(to_json_safe(payload), recursive=True, drop_empty=False),
        }
        self._history.append(event)
        self._sync_memory_state(last_event=event)

    def _sync_memory_state(self, *, last_event: Optional[Mapping[str, Any]] = None) -> None:
        if self.shared_memory is None:
            return
        try:
            self.shared_memory.put(
                key="tokenizer_state",
                value={
                    "request_id": self._lock_request_id,
                    "created_at": self._created_at,
                    "trained": self.is_trained,
                    "vocab_size": len(self.vocab),
                    "training_runs": self.training_runs,
                    "encode_calls": self.encode_calls,
                    "decode_calls": self.decode_calls,
                    "tokenize_calls": self.tokenize_calls,
                    "last_training_summary": self.last_training_summary.to_dict() if self.last_training_summary else None,
                    "last_event": to_json_safe(last_event) if last_event else None,
                },
                namespace=self.memory_namespace,
                metadata={"component": "BaseTokenizer", "request_id": self._lock_request_id},
                tags=["tokenizer", "state"],
                persistent=False,
            )
        except Exception as exc:
            logger.debug(f"Failed to synchronize tokenizer state to memory: {safe_repr(exc)}")

    def _normalize_input_text(self, text: str) -> str:
        ensure_type(text, str, "text", config=self.bt_config, error_cls=BaseValidationError)
        normalization_form = cast(Literal['NFC', 'NFD', 'NFKC', 'NFKD'], self.unicode_normalization_form)
        normalized = unicodedata.normalize(normalization_form, text)
        if self.remove_control_characters:
            normalized = re.sub(r"[\u0000-\u001F\u007F]", "", normalized)
        if self.squeeze_whitespace:
            normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _coerce_max_length(self, value: Any) -> Optional[int]:
        if value in (None, "", "none", "None", False):
            return None
        result = coerce_int(value, default=self.default_max_length, minimum=1)
        return result

    def _resolve_padding(self, value: Any) -> Optional[str]:
        if value in (None, False, "none", "None"):
            return None
        padding = ensure_text(value).strip().lower()
        ensure_one_of(
            padding,
            ["max_length", "longest", "do_not_pad"],
            "padding",
            config=self.bt_config,
            error_cls=BaseValidationError,
        )
        return None if padding == "do_not_pad" else padding

    def _truncate_tokens(self, tokens: List[str], max_length: Optional[int], add_special_tokens: bool) -> List[str]:
        if max_length is None:
            return tokens
        reserved = 2 if add_special_tokens else 0
        limit = max(max_length - reserved, 0)
        return tokens[:limit]

    def _apply_padding(self, token_ids: List[int], attention_mask: List[int], target_length: Optional[int]) -> Tuple[List[int], List[int]]:
        if target_length is None:
            return token_ids, attention_mask
        if len(token_ids) >= target_length:
            return token_ids[:target_length], attention_mask[:target_length]
        pad_id = self.token_to_id(self.pad_token)
        missing = target_length - len(token_ids)
        return token_ids + [pad_id] * missing, attention_mask + [0] * missing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(
        self,
        corpus: Sequence[str],
        min_freq: Optional[int] = None,
        max_vocab_size: Optional[int] = None,
        reset_existing: bool = True,
    ) -> TrainingSummary:
        """Train a deterministic frequency-based vocabulary from a text corpus."""
        ensure_sequence(corpus, "corpus", config=self.bt_config, error_cls=BaseValidationError)

        effective_min_freq = coerce_int(min_freq if min_freq is not None else self.min_frequency, self.min_frequency, minimum=1)
        effective_max_vocab_size = coerce_int(
            max_vocab_size if max_vocab_size is not None else self.max_vocab_size,
            self.max_vocab_size,
            minimum=len(self.special_tokens),
        )

        if reset_existing:
            self._reinitialize_vocab()

        observed_counts: Counter[str] = Counter()
        for entry in corpus:
            ensure_type(entry, str, "corpus_entry", config=self.bt_config, error_cls=BaseValidationError)
            tokens = self.tokenize(entry)
            observed_counts.update(tokens)

        sorted_candidates = sorted(
            (
                (token, count)
                for token, count in observed_counts.items()
                if count >= effective_min_freq and token not in self.special_token_set
            ),
            key=lambda item: (-item[1], item[0]),
        )

        allowed_new_tokens = max(effective_max_vocab_size - len(self.special_tokens), 0)
        learned = 0
        for token, _count in sorted_candidates[:allowed_new_tokens]:
            if token not in self.vocab:
                next_id = len(self.vocab)
                self.vocab[token] = next_id
                self.inverse_vocab[next_id] = token
                learned += 1

        if self.track_token_counts:
            self.token_counts = observed_counts

        self.is_trained = True
        self.training_runs += 1
        self.src_vocab_size = len(self.vocab)

        summary = TrainingSummary(
            corpus_size=len(corpus),
            min_frequency=effective_min_freq,
            max_vocab_size=effective_max_vocab_size,
            learned_tokens=learned,
            total_unique_observed_tokens=len(observed_counts),
            total_observed_tokens=sum(observed_counts.values()),
        )
        self.last_training_summary = summary
        self._record_event("train", summary=summary.to_dict())

        printer.status("TRAIN", f"Base Tokenizer trained with vocab size: {self.src_vocab_size}", "success")
        return summary

    def tokenize(self, text: str, **_: Any) -> List[str]:
        """Tokenize text into words, numbers, special tokens, symbols, and emoji."""
        normalized = self._normalize_input_text(text)
        matches = self._token_pattern.findall(normalized)
        tokens: List[str] = []
        for token in matches:
            if token in self.special_token_set:
                tokens.append(token)
            else:
                tokens.append(token.lower() if self.lowercase else token)
        self.tokenize_calls += 1
        self._record_event("tokenize", text_preview=truncate_string(text, 120), token_count=len(tokens))
        return tokens

    def encode(self, text: str, **kwargs: Any) -> Dict[str, Union[List[int], torch.Tensor, List[str]]]:
        """Encode text into token IDs and attention metadata."""
        ensure_type(text, str, "text", config=self.bt_config, error_cls=BaseValidationError)

        max_length = self._coerce_max_length(kwargs.get("max_length", self.default_max_length))
        padding = self._resolve_padding(kwargs.get("padding", self.default_padding))
        truncation = coerce_bool(kwargs.get("truncation", self.default_truncation), self.default_truncation)
        add_special_tokens = coerce_bool(
            kwargs.get("add_special_tokens", self.default_add_special_tokens),
            self.default_add_special_tokens,
        )
        return_tensors = kwargs.get("return_tensors", self.default_return_tensors)
        include_attention_mask = coerce_bool(
            kwargs.get("return_attention_mask", self.return_attention_mask),
            self.return_attention_mask,
        )
        include_special_tokens_mask = coerce_bool(
            kwargs.get("return_special_tokens_mask", self.return_special_tokens_mask),
            self.return_special_tokens_mask,
        )
        include_tokens = coerce_bool(
            kwargs.get("return_tokens", self.return_tokens),
            self.return_tokens,
        )

        ensure_one_of(
            return_tensors if return_tensors is not None else None,
            [None, "pt"],
            "return_tensors",
            config=self.bt_config,
            error_cls=BaseValidationError,
        )

        tokens = self.tokenize(text)
        if truncation:
            tokens = self._truncate_tokens(tokens, max_length, add_special_tokens)

        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]

        token_ids = [self.token_to_id(token) for token in tokens]
        attention_mask = [1] * len(token_ids)
        special_tokens_mask = [1 if token in self.special_token_set else 0 for token in tokens]

        if padding == "max_length":
            target_length = max_length
        else:
            target_length = None

        token_ids, attention_mask = self._apply_padding(token_ids, attention_mask, target_length)
        if target_length is not None and include_special_tokens_mask:
            special_tokens_mask = special_tokens_mask + [1 if self.pad_token in self.special_token_set else 0] * max(0, target_length - len(special_tokens_mask))
            special_tokens_mask = special_tokens_mask[:target_length]

        payload: Dict[str, Union[List[int], torch.Tensor, List[str]]] = {"input_ids": token_ids}
        if include_attention_mask:
            payload["attention_mask"] = attention_mask
        if include_special_tokens_mask:
            payload["special_tokens_mask"] = special_tokens_mask
        if include_tokens:
            payload["tokens"] = tokens

        if return_tensors == "pt":
            for key, value in list(payload.items()):
                if key != "tokens":
                    payload[key] = torch.tensor(value, dtype=torch.long)

        self.encode_calls += 1
        self._record_event(
            "encode",
            text_preview=truncate_string(text, 120),
            token_count=len(tokens),
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
        )
        return payload

    def encode_batch(self, texts: Sequence[str], **kwargs: Any) -> Dict[str, Union[List[List[int]], torch.Tensor, List[List[str]]]]:
        """Encode a batch of texts using the same options for each item."""
        ensure_sequence(texts, "texts", config=self.bt_config, error_cls=BaseValidationError)
        encoded_items = [self.encode(text, **kwargs) for text in texts]

        batch: Dict[str, Any] = {}
        for key in encoded_items[0].keys() if encoded_items else []:
            values = [item[key] for item in encoded_items]
            if kwargs.get("return_tensors", self.default_return_tensors) == "pt" and key != "tokens":
                batch[key] = torch.stack(values)  # type: ignore[arg-type]
            else:
                batch[key] = values
        return batch

    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs: Any) -> str:
        """Convert token IDs back to text."""
        skip_special_tokens = coerce_bool(
            kwargs.get("skip_special_tokens", self.default_skip_special_tokens),
            self.default_skip_special_tokens,
        )
        clean_up_tokenization_spaces = coerce_bool(
            kwargs.get("clean_up_tokenization_spaces", self.default_clean_up_tokenization_spaces),
            self.default_clean_up_tokenization_spaces,
        )

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        ensure_sequence(token_ids, "token_ids", config=self.bt_config, error_cls=BaseValidationError)

        tokens = [self.id_to_token(int(token_id)) for token_id in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_token_set]

        text = self.detokenize(tokens)
        if clean_up_tokenization_spaces:
            text = self.clean_text(text)

        self.decode_calls += 1
        self._record_event("decode", token_count=len(token_ids), output_preview=truncate_string(text, 120))
        return text

    def decode_batch(self, sequences: Sequence[Union[List[int], torch.Tensor]], **kwargs: Any) -> List[str]:
        """Decode a batch of token ID sequences."""
        ensure_sequence(sequences, "sequences", config=self.bt_config, error_cls=BaseValidationError)
        return [self.decode(sequence, **kwargs) for sequence in sequences]

    def detokenize(self, tokens: Sequence[str]) -> str:
        """Reconstruct text from tokens using conservative spacing rules."""
        ensure_sequence(tokens, "tokens", config=self.bt_config, error_cls=BaseValidationError, allow_str=False)
        if not tokens:
            return ""

        output_parts: List[str] = []
        no_leading_space = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "'", '"'}
        opening_tokens = {"(", "[", "{", "$", "£", "€", "¥", "₹", '"'}

        for token in tokens:
            current = ensure_text(token)
            if not output_parts:
                output_parts.append(current)
                continue

            previous = output_parts[-1]
            if current in no_leading_space:
                output_parts[-1] = previous + current
            elif previous in opening_tokens:
                output_parts[-1] = previous + current
            elif current in {"-", "/"} and previous and previous[-1].isdigit():
                output_parts[-1] = previous + current
            else:
                output_parts.append(current)

        return " ".join(output_parts)

    def clean_text(self, text: str) -> str:
        """Normalize detokenized output for readability."""
        ensure_type(text, str, "text", config=self.bt_config, error_cls=BaseValidationError)
        normalization_form = cast(Literal['NFC', 'NFD', 'NFKC', 'NFKD'], self.unicode_normalization_form)
        cleaned = unicodedata.normalize(normalization_form, text)
        cleaned = cleaned.replace("</w>", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([\.,!?;:%\)\]\}])", r"\1", cleaned)
        cleaned = re.sub(r"([\(\[\{])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s+([\)\]\}])", r"\1", cleaned)
        cleaned = re.sub(r"\s*([-/])\s*", r"\1", cleaned)
        cleaned = re.sub(r"\b(\w+)\s+'\s*s\b", r"\1's", cleaned)
        cleaned = re.sub(r"\b(\w+)\s+'\b", r"\1'", cleaned)
        return cleaned.strip()

    def token_to_id(self, token: str) -> int:
        """Convert a token to its integer ID."""
        ensure_type(token, str, "token", config=self.bt_config, error_cls=BaseValidationError)
        return int(self.vocab.get(token, self.vocab[self.unk_token]))

    def id_to_token(self, token_id: int) -> str:
        """Convert an integer ID back into a token."""
        ensure_type(token_id, (int,), "token_id", config=self.bt_config, error_cls=BaseValidationError)
        return self.inverse_vocab.get(int(token_id), self.unk_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id(self.unk_token)

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id(self.eos_token)

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id(self.mask_token)

    def add_tokens(self, tokens: Union[str, Sequence[str]], special: bool = False) -> int:
        """Add new tokens to the vocabulary."""
        values = [tokens] if isinstance(tokens, str) else list(tokens)
        new_count = 0
        for token in values:
            normalized = ensure_non_empty_string(token, "token", config=self.bt_config, error_cls=BaseValidationError)
            if normalized not in self.vocab:
                next_id = len(self.vocab)
                self.vocab[normalized] = next_id
                self.inverse_vocab[next_id] = normalized
                new_count += 1
                self.added_tokens_count += 1
            if special and normalized not in self.special_token_set:
                self.special_tokens.append(normalized)
                self.special_token_set.add(normalized)
        if special:
            self._rebuild_token_pattern()
        if new_count:
            self._record_event("add_tokens", count=new_count, special=special)
        return new_count

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def get_special_tokens(self) -> Dict[str, Union[str, List[str]]]:
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
            "special_tokens": list(self.special_tokens),
        }

    def stats(self) -> TokenizerStats:
        return TokenizerStats(
            request_id=self._lock_request_id,
            vocab_size=len(self.vocab),
            special_token_count=len(self.special_tokens),
            trained=self.is_trained,
            training_runs=self.training_runs,
            encode_calls=self.encode_calls,
            decode_calls=self.decode_calls,
            tokenize_calls=self.tokenize_calls,
            added_tokens=self.added_tokens_count,
            history_length=len(self._history),
        )

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        count = coerce_int(limit, default=20, minimum=1)
        return list(self._history)[-count:]

    def save(self, directory: Union[str, Path], name: str = "tokenizer") -> List[str]:
        """Persist tokenizer state to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        config_file = target_dir / f"{name}_config.json"
        vocab_file = target_dir / f"{name}_vocab.json"
        counts_file = target_dir / f"{name}_token_counts.json"

        config_to_save = {
            "tokenizer_class": self.__class__.__name__,
            "request_id": self._lock_request_id,
            "created_at": self._created_at,
            "src_vocab_size": self.src_vocab_size,
            "max_vocab_size": self.max_vocab_size,
            "min_frequency": self.min_frequency,
            "lowercase": self.lowercase,
            "unicode_normalization_form": self.unicode_normalization_form,
            "remove_control_characters": self.remove_control_characters,
            "squeeze_whitespace": self.squeeze_whitespace,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
            "special_tokens": list(self.special_tokens),
            "is_trained": self.is_trained,
            "training_runs": self.training_runs,
            "stats": self.stats().to_dict(),
            "last_training_summary": self.last_training_summary.to_dict() if self.last_training_summary else None,
        }

        try:
            config_file.write_text(json_dumps(config_to_save, pretty=self.snapshot_pretty), encoding="utf-8")
            vocab_file.write_text(json_dumps(self.vocab, pretty=self.snapshot_pretty), encoding="utf-8")
            counts_file.write_text(json_dumps(dict(self.token_counts), pretty=self.snapshot_pretty), encoding="utf-8")
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Failed to save BaseTokenizer state.",
                config=self.bt_config,
                component="BaseTokenizer",
                operation="save",
                context={"directory": str(target_dir), "name": name},
            ) from exc

        self._record_event("save", directory=str(target_dir), name=name)
        printer.status("SAVE", f"BaseTokenizer saved to {target_dir}", "success")
        return [str(config_file), str(vocab_file), str(counts_file)]

    @classmethod
    def load(cls, directory: Union[str, Path], name: str = "tokenizer") -> "BaseTokenizer":
        """Load tokenizer state from disk."""
        target_dir = Path(directory)
        config_file = target_dir / f"{name}_config.json"
        vocab_file = target_dir / f"{name}_vocab.json"
        counts_file = target_dir / f"{name}_token_counts.json"

        if not config_file.exists() or not vocab_file.exists():
            raise BaseIOError(
                "Tokenizer save files are incomplete or missing.",
                None,
                component="BaseTokenizer",
                operation="load",
                context={
                    "config_file": str(config_file),
                    "vocab_file": str(vocab_file),
                    "counts_file": str(counts_file),
                },
            )

        try:
            config = json_loads(config_file.read_text(encoding="utf-8"), default={})
            vocab = json_loads(vocab_file.read_text(encoding="utf-8"), default={})
            counts = json_loads(counts_file.read_text(encoding="utf-8"), default={}) if counts_file.exists() else {}
            ensure_mapping(config, "config", error_cls=BaseValidationError)
            ensure_mapping(vocab, "vocab", error_cls=BaseValidationError)
            ensure_mapping(counts, "counts", error_cls=BaseValidationError)
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Failed to load BaseTokenizer state.",
                component="BaseTokenizer",
                operation="load",
                context={"directory": str(target_dir), "name": name},
            ) from exc

        tokenizer = cls()
        tokenizer.src_vocab_size = coerce_int(config.get("src_vocab_size", 30000), default=30000, minimum=5)
        tokenizer.max_vocab_size = coerce_int(config.get("max_vocab_size", tokenizer.src_vocab_size), default=tokenizer.src_vocab_size, minimum=5)
        tokenizer.min_frequency = coerce_int(config.get("min_frequency", 2), default=2, minimum=1)
        tokenizer.lowercase = coerce_bool(config.get("lowercase", tokenizer.lowercase), tokenizer.lowercase)
        tokenizer.unicode_normalization_form = ensure_text(config.get("unicode_normalization_form", tokenizer.unicode_normalization_form)).upper()
        tokenizer.remove_control_characters = coerce_bool(config.get("remove_control_characters", tokenizer.remove_control_characters), tokenizer.remove_control_characters)
        tokenizer.squeeze_whitespace = coerce_bool(config.get("squeeze_whitespace", tokenizer.squeeze_whitespace), tokenizer.squeeze_whitespace)
        tokenizer.pad_token = ensure_text(config.get("pad_token", tokenizer.pad_token))
        tokenizer.unk_token = ensure_text(config.get("unk_token", tokenizer.unk_token))
        tokenizer.bos_token = ensure_text(config.get("bos_token", tokenizer.bos_token))
        tokenizer.eos_token = ensure_text(config.get("eos_token", tokenizer.eos_token))
        tokenizer.mask_token = ensure_text(config.get("mask_token", tokenizer.mask_token))
        tokenizer.special_tokens = tokenizer._build_special_tokens(config.get("special_tokens", tokenizer.special_tokens))
        tokenizer.special_token_set = set(tokenizer.special_tokens)
        tokenizer.vocab = {ensure_text(token): int(index) for token, index in dict(vocab).items()}
        tokenizer.inverse_vocab = {index: token for token, index in tokenizer.vocab.items()}
        tokenizer.token_counts = Counter({ensure_text(token): int(count) for token, count in dict(counts).items()})
        tokenizer.is_trained = coerce_bool(config.get("is_trained", False), False)
        tokenizer.training_runs = coerce_int(config.get("training_runs", 0), default=0, minimum=0)
        if isinstance(config.get("last_training_summary"), Mapping):
            summary = dict(config["last_training_summary"])
            tokenizer.last_training_summary = TrainingSummary(
                corpus_size=coerce_int(summary.get("corpus_size", 0), default=0, minimum=0),
                min_frequency=coerce_int(summary.get("min_frequency", tokenizer.min_frequency), default=tokenizer.min_frequency, minimum=1),
                max_vocab_size=coerce_int(summary.get("max_vocab_size", tokenizer.max_vocab_size), default=tokenizer.max_vocab_size, minimum=5),
                learned_tokens=coerce_int(summary.get("learned_tokens", 0), default=0, minimum=0),
                total_unique_observed_tokens=coerce_int(summary.get("total_unique_observed_tokens", 0), default=0, minimum=0),
                total_observed_tokens=coerce_int(summary.get("total_observed_tokens", 0), default=0, minimum=0),
                timestamp=ensure_text(summary.get("timestamp", utc_now_iso())),
            )
        tokenizer._rebuild_token_pattern()
        tokenizer._record_event("load", directory=str(target_dir), name=name)

        printer.status("LOAD", f"BaseTokenizer loaded from {target_dir}", "success")
        return tokenizer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self._lock_request_id,
            "created_at": self._created_at,
            "trained": self.is_trained,
            "stats": self.stats().to_dict(),
            "special_tokens": self.get_special_tokens(),
            "vocab_size": len(self.vocab),
            "last_training_summary": self.last_training_summary.to_dict() if self.last_training_summary else None,
        }

    def __call__(self, text: str, **kwargs: Any) -> Dict[str, Union[List[int], torch.Tensor, List[str]]]:
        return self.encode(text, **kwargs)

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and token in self.vocab

    def __repr__(self) -> str:
        return (
            f"<BaseTokenizer request_id='{self._lock_request_id}' "
            f"vocab_size={len(self.vocab)} trained={self.is_trained}>"
        )


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Base Tokenizer ===\n")
    printer.status("TEST", "Base Tokenizer initialized", "info")

    tokenizer = BaseTokenizer()
    print(tokenizer)

    corpus = [
        "Life is worth living with friends like you!",
        "Tokenization should handle emojis 🙂, prices like $19.99, and dates like 2026-04-23.",
        "Don't split contractions oddly, and keep special tokens like [MASK] intact.",
    ]
    summary = tokenizer.train(corpus, min_freq=1)
    printer.pretty("TRAIN_SUMMARY", summary.to_dict(), "success")

    text = "Life is worth living with friends like you! [MASK] costs $19.99 🙂"
    tokens = tokenizer.tokenize(text)
    encoded = tokenizer.encode(text, return_tokens=True, return_special_tokens_mask=True)
    decoded = tokenizer.decode(encoded["input_ids"]) # type: ignore
    cleaned = tokenizer.clean_text(decoded)

    printer.pretty("TOKENS", tokens, "success")
    printer.pretty("ENCODED", encoded, "success")
    printer.pretty("DECODED", {"text": decoded}, "success")
    printer.pretty("CLEANED", {"text": cleaned}, "success")
    printer.pretty("STATS", tokenizer.stats().to_dict(), "success")
    printer.pretty("RECENT_HISTORY", tokenizer.recent_history(), "success")

    save_paths = tokenizer.save("/tmp/base_tokenizer_artifacts", name="base_tokenizer_test")
    reloaded = BaseTokenizer.load("/tmp/base_tokenizer_artifacts", name="base_tokenizer_test")
    printer.pretty("SAVE_PATHS", save_paths, "success")
    printer.pretty("RELOADED", reloaded.to_dict(), "success")

    print("\n=== Test ran successfully ===\n")
