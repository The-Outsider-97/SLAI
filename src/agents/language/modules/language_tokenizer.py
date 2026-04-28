"""
Language Tokenizer Module

Core Function:
Provides the language subsystem with a production-ready tokenizer built on top
of the shared BaseTokenizer. The tokenizer supports deterministic word-level
pre-tokenization, Byte Pair Encoding (BPE) subword training/application,
encode/decode interoperability with model-facing pipelines, token span metadata,
resource persistence, diagnostics, and config-driven behavior.

Responsibilities:
- Inherit the base tokenizer contract while specializing it for language-agent use.
- Normalize and pre-tokenize text without losing language-facing token boundaries.
- Train, load, save, and apply a BPE model with stable merge ranks.
- Produce subword tokens, IDs, attention masks, special-token masks, and offsets.
- Preserve token metadata for NLP, grammar, NLU, dialogue context, and NLG modules.
- Use language helpers and language errors rather than duplicating generic logic.

Why it matters:
Tokenization is the boundary between raw text and every downstream language
module. A production tokenizer must be deterministic, inspectable, serializable,
span-aware, error-aware, and expandable without forcing grammar/NLP components
to treat model subwords as linguistic words.
"""

from __future__ import annotations

import json

import regex as regex_re
import torch

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from ..utils.config_loader import load_global_config, get_config_section
from ...base.modules.base_tokenizer import BaseTokenizer, TrainingSummary
from ..utils.language_error import *
from ..utils.language_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Language Tokenizer")
printer = PrettyPrinter()

Pair = Tuple[str, str]
TokenSpan = Tuple[int, int]


@dataclass(frozen=True)
class BPEToken:
    """Subword token plus source alignment metadata."""

    token: str
    token_id: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    word_index: Optional[int] = None
    is_special: bool = False
    is_unknown: bool = False
    is_continuation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> Optional[TokenSpan]:
        if self.start_char is None or self.end_char is None:
            return None
        return (self.start_char, self.end_char)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self), drop_empty=True)


@dataclass(frozen=True)
class PreToken:
    """Word/punctuation-level token produced before BPE segmentation."""

    text: str
    start_char: int
    end_char: int
    index: int
    kind: str = "text"

    @property
    def span(self) -> TokenSpan:
        return (self.start_char, self.end_char)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BPETrainingSummary:
    """Deterministic summary for BPE training runs."""

    corpus_size: int
    normalized_corpus_size: int
    min_frequency: int
    target_vocab_size: int
    initial_vocab_size: int
    final_vocab_size: int
    learned_merges: int
    observed_words: int
    observed_word_types: int
    stopped_reason: str
    duration_ms: float
    created_at: str = field(default_factory=lambda: utc_timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LanguageTokenizerStats:
    """Operational snapshot for the language tokenizer."""

    request_id: str
    version: str
    algorithm: str
    trained: bool
    vocab_size: int
    special_token_count: int
    merge_count: int
    token_counts_size: int
    tokenize_calls: int
    encode_calls: int
    decode_calls: int
    training_runs: int
    unknown_token_count: int
    diagnostics_count: int
    history_length: int
    model_path: Optional[str] = None
    vocab_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizationResult:
    """Detailed tokenizer result for NLP/grammar/NLU integration."""

    original_text: str
    normalized_text: str
    tokens: Tuple[BPEToken, ...]
    pre_tokens: Tuple[PreToken, ...] = ()
    issues: Tuple[LanguageIssue, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_texts(self) -> List[str]:
        return [token.token for token in self.tokens]

    @property
    def token_ids(self) -> List[int]:
        return [token.token_id for token in self.tokens]

    @property
    def ok(self) -> bool:
        return not any(issue.is_blocking for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "tokens": [token.to_dict() for token in self.tokens],
            "pre_tokens": [token.to_dict() for token in self.pre_tokens],
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": json_safe(self.metadata),
        }


class LanguageTokenizer(BaseTokenizer):
    """
    BPE tokenizer specialized for the language agent.

    The class inherits BaseTokenizer so it remains compatible with the base
    encode/decode contract. It adds a language-specific BPE layer, span-aware
    tokenization, structured diagnostics, config-backed persistence, and model
    metadata without reimplementing generic helper functions.
    """

    VERSION = "2.1"
    MODEL_FORMAT = "language-bpe-tokenizer"

    def __init__(self) -> None:
        super().__init__()
        self.config = load_global_config()
        self.lt_config = get_config_section("language_tokenizer") or {}

        self.version = ensure_text(self.lt_config.get("version", self.VERSION))
        self.algorithm = ensure_text(self.lt_config.get("algorithm", "bpe")).lower()
        self.end_of_word_suffix = ensure_text(self.lt_config.get("end_of_word_suffix", "</w>"))
        self.continuation_prefix = ensure_text(self.lt_config.get("continuation_prefix", ""))
        self.unknown_policy = ensure_text(self.lt_config.get("unknown_policy", "char_fallback")).lower()
        self.target_vocab_size = coerce_int(
            self.lt_config.get("target_vocab_size", self.config.get("src_vocab_size", getattr(self, "max_vocab_size", 30000))),
            default=getattr(self, "max_vocab_size", 30000),
            minimum=len(self.special_tokens) + 1,
        )
        self.max_merges = coerce_int(self.lt_config.get("max_merges", self.target_vocab_size), default=self.target_vocab_size, minimum=0)
        self.min_frequency = coerce_int(self.lt_config.get("min_frequency", 2), default=2, minimum=1)
        self.max_input_chars = coerce_int(self.lt_config.get("max_input_chars", 200_000), default=200_000, minimum=1)
        self.max_token_chars = coerce_int(self.lt_config.get("max_token_chars", 256), default=256, minimum=1)
        self.add_end_of_word = coerce_bool(self.lt_config.get("add_end_of_word", True), default=True)
        self.preserve_punctuation_tokens = coerce_bool(self.lt_config.get("preserve_punctuation_tokens", True), default=True)
        self.fallback_to_char_tokenization = coerce_bool(self.lt_config.get("fallback_to_char_tokenization", True), default=True)
        self.allow_dynamic_vocab_growth = coerce_bool(
            self.lt_config.get("allow_dynamic_vocab_growth", getattr(self, "allow_dynamic_vocab_growth", False)),
            default=False,
        )
        self.track_offsets = coerce_bool(self.lt_config.get("track_offsets", True), default=True)
        self.emit_diagnostics = coerce_bool(self.lt_config.get("emit_diagnostics", True), default=True)
        self.save_pretty = coerce_bool(self.lt_config.get("save_pretty", True), default=True)
        self.strict_loading = coerce_bool(self.lt_config.get("strict_loading", False), default=False)
        self.lowercase = coerce_bool(self.lt_config.get("lowercase", getattr(self, "lowercase", True)), default=True)
        self.normalization_rules = ensure_mapping(
            self.lt_config.get("normalization_rules", {"lowercase": self.lowercase, "form": "NFKC"}),
            field_name="language_tokenizer.normalization_rules",
            allow_none=True,
        )

        self.bpe_model_file = self._resolve_optional_path(
            self.lt_config.get("bpe_model_path", self.config.get("bpe_model_path")),
            field_name="bpe_model_path",
        )
        self.bpe_vocab_file = self._resolve_optional_path(
            self.lt_config.get("bpe_vocab_path", self.config.get("bpe_vocab_path")),
            field_name="bpe_vocab_path",
        )
        self.artifact_dir = self._resolve_optional_path(
            self.lt_config.get("artifact_dir", "src/agents/language/artifacts/tokenizer"),
            field_name="artifact_dir",
        )

        self.merges: Dict[Pair, str] = {}
        self.ordered_merges: List[Pair] = []
        self.merge_ranks: Dict[Pair, int] = {}
        self.diagnostics = LanguageDiagnostics()
        self.training_history: Deque[Dict[str, Any]] = deque(maxlen=coerce_int(self.lt_config.get("training_history_limit", 20), default=20, minimum=1))
        self.unknown_token_count = 0
        self.bpe_last_training_summary: Optional[BPETrainingSummary] = None
        self._tokenize_history: Deque[Dict[str, Any]] = deque(maxlen=coerce_int(self.lt_config.get("history_limit", 200), default=200, minimum=1))
        self._pre_tokenize_pattern = self._compile_pre_tokenize_pattern()

        self._bootstrap_special_vocab()
        self._load_configured_resources()
        self._refresh_merge_ranks()
        self.is_trained = bool(self.vocab and self.ordered_merges)
        self.src_vocab_size = len(self.vocab)
        self.max_vocab_size = max(getattr(self, "max_vocab_size", 0), self.target_vocab_size, len(self.vocab))

        if not self.is_trained:
            self._add_issue(
                TokenizationIssue(
                    code=LanguageErrorCode.TOKENIZER_NOT_TRAINED,
                    message="LanguageTokenizer initialized without a trained BPE merge table.",
                    severity=Severity.WARNING,
                    module="LanguageTokenizer",
                    recoverable=True,
                    details={
                        "bpe_model_path": str(self.bpe_model_file) if self.bpe_model_file else None,
                        "bpe_vocab_path": str(self.bpe_vocab_file) if self.bpe_vocab_file else None,
                    },
                )
            )
            printer.status("INIT", "Language Tokenizer initialized without trained BPE resources", "warning")
        else:
            printer.status("INIT", f"Language Tokenizer loaded BPE resources: vocab={len(self.vocab)}, merges={len(self.ordered_merges)}", "success")

    # ------------------------------------------------------------------
    # Config and resource loading
    # ------------------------------------------------------------------
    def _resolve_optional_path(self, value: Any, *, field_name: str) -> Optional[Path]:
        if value in (None, "", "none", "None"):
            return None
        return resolve_path(ensure_text(value), field_name=field_name)

    def _compile_pre_tokenize_pattern(self) -> Any:
        pattern = ensure_text(
            self.lt_config.get(
                "pre_tokenize_pattern",
                r"(?:\[[A-Z_]+\])|(?:\p{L}+(?:['’]\p{L}+)*)|(?:\p{N}+(?:[\.,:/-]\p{N}+)*)|(?:[\p{Extended_Pictographic}\p{Emoji_Presentation}])|(?:[^\s])",
            )
        )
        try:
            return regex_re.compile(pattern)
        except regex_re.error as exc:
            logger.warning(
                "Unicode property escapes were not accepted by the regex engine. "
                "Falling back to the portable pre-tokenization pattern. Error: %s",
                exc,
            )
            fallback = r"(?:\[[A-Z_]+\])|(?:\w+(?:['’]\w+)*)|(?:\d+(?:[\.,:/-]\d+)*)|(?:[^\s])"
            return regex_re.compile(fallback, regex_re.UNICODE)

    def _bootstrap_special_vocab(self) -> None:
        ordered: List[str] = []
        for token in self.special_tokens:
            token_text = require_non_empty_string(token, "special_token")
            if token_text not in ordered:
                ordered.append(token_text)
        self.special_tokens = ordered
        self.special_token_set = set(ordered)
        self.vocab = {token: index for index, token in enumerate(self.special_tokens)}
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}

    def _load_configured_resources(self) -> None:
        loaded_model = False
        loaded_vocab = False

        if self.bpe_model_file and self.bpe_model_file.exists():
            self._load_bpe_model(self.bpe_model_file)
            loaded_model = True
        elif self.bpe_model_file:
            self._handle_missing_resource(self.bpe_model_file, kind="model")

        if self.bpe_vocab_file and self.bpe_vocab_file.exists():
            self._load_bpe_vocab(self.bpe_vocab_file)
            loaded_vocab = True
        elif self.bpe_vocab_file:
            self._handle_missing_resource(self.bpe_vocab_file, kind="vocab")

        if loaded_model and not loaded_vocab:
            self._ensure_vocab_contains_model_tokens()

        self._validate_vocab_integrity()

    def _handle_missing_resource(self, path: Path, *, kind: str) -> None:
        code = LanguageErrorCode.TOKENIZER_MODEL_MISSING if kind == "model" else LanguageErrorCode.TOKENIZER_VOCAB_MISSING
        issue = TokenizationIssue(
            code=code,
            message=f"Configured BPE {kind} file was not found.",
            severity=Severity.WARNING,
            module="LanguageTokenizer",
            recoverable=True,
            details={"path": str(path)},
        )
        self._add_issue(issue)
        logger.warning(issue.to_json())
        if self.strict_loading:
            raise ResourceLanguageError(issue, recoverable=False)

    def _load_bpe_model(self, path: Union[str, Path]) -> None:
        model_path = Path(path)
        data = load_json_file(model_path)
        if not isinstance(data, Mapping):
            raise ResourceLanguageError(
                ResourceIssue(
                    code=LanguageErrorCode.RESOURCE_FORMAT_INVALID,
                    message="BPE model file must contain a JSON object.",
                    module="LanguageTokenizer",
                    details={"path": str(model_path)},
                ),
                recoverable=False,
            )

        raw_merges = data.get("ordered_merges", data.get("merges", []))
        merges: List[Pair] = []
        for item in ensure_list(raw_merges):
            pair = self._coerce_merge_pair(item)
            if pair is not None:
                merges.append(pair)

        self.ordered_merges = dedupe_preserve_order(merges)
        self.merges = {pair: pair[0] + pair[1] for pair in self.ordered_merges}
        self._refresh_merge_ranks()

        specials = data.get("special_tokens")
        if isinstance(specials, Sequence) and not isinstance(specials, (str, bytes, bytearray)):
            normalized_specials = [ensure_text(token) for token in specials if ensure_text(token)]
            if normalized_specials:
                self.add_tokens(normalized_specials, special=True)

        normalization = data.get("normalization", data.get("normalization_rules"))
        if isinstance(normalization, Mapping):
            self.normalization_rules = merge_mappings(self.normalization_rules, normalization)

        self.end_of_word_suffix = ensure_text(data.get("end_of_word_suffix", self.end_of_word_suffix))
        logger.info("Loaded BPE model from %s with %s merges", model_path, len(self.ordered_merges))

    def _load_bpe_vocab(self, path: Union[str, Path]) -> None:
        vocab_path = Path(path)
        data = load_json_file(vocab_path)
        if isinstance(data, Mapping) and "vocab" in data and isinstance(data["vocab"], Mapping):
            raw_vocab = data["vocab"]
        else:
            raw_vocab = data

        if not isinstance(raw_vocab, Mapping):
            raise ResourceLanguageError(
                ResourceIssue(
                    code=LanguageErrorCode.RESOURCE_FORMAT_INVALID,
                    message="BPE vocab file must contain a token-to-id mapping.",
                    module="LanguageTokenizer",
                    details={"path": str(vocab_path)},
                ),
                recoverable=False,
            )

        parsed_vocab: Dict[str, int] = {}
        for token, token_id in raw_vocab.items():
            token_text = ensure_text(token)
            parsed_vocab[token_text] = coerce_int(token_id, default=len(parsed_vocab), minimum=0)

        for token in self.special_tokens:
            parsed_vocab.setdefault(token, len(parsed_vocab))

        parsed_vocab = self._compact_vocab_ids(parsed_vocab)
        self.vocab = parsed_vocab
        self.inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}
        self.special_token_set = set(self.special_tokens)
        logger.info("Loaded BPE vocab from %s with %s tokens", vocab_path, len(self.vocab))

    def _coerce_merge_pair(self, item: Any) -> Optional[Pair]:
        if isinstance(item, str):
            parts = item.split()
            if len(parts) == 2:
                return (parts[0], parts[1])
            return None
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) == 2:
            return (ensure_text(item[0]), ensure_text(item[1]))
        return None

    def _compact_vocab_ids(self, vocab: Mapping[str, int]) -> Dict[str, int]:
        ordered_tokens = sorted(vocab.items(), key=lambda item: (int(item[1]), item[0]))
        compact: Dict[str, int] = {}
        for token, _old_id in ordered_tokens:
            if token not in compact:
                compact[token] = len(compact)
        return compact

    def _ensure_vocab_contains_model_tokens(self) -> None:
        for token in self.special_tokens:
            self._add_token_to_vocab(token)
        for pair in self.ordered_merges:
            self._add_token_to_vocab(pair[0])
            self._add_token_to_vocab(pair[1])
            self._add_token_to_vocab(pair[0] + pair[1])
        self._add_token_to_vocab(self.end_of_word_suffix)

    def _validate_vocab_integrity(self) -> None:
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.mask_token]:
            if token not in self.vocab:
                self._add_token_to_vocab(token)
                self._add_issue(
                    TokenizationIssue(
                        code=LanguageErrorCode.TOKENIZER_VOCAB_MISSING,
                        message="Required special token was missing from the vocabulary and was added.",
                        severity=Severity.WARNING,
                        module="LanguageTokenizer",
                        recoverable=True,
                        details={"token": token},
                    )
                )
        self.inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}

    def _refresh_merge_ranks(self) -> None:
        self.merge_ranks = {pair: index for index, pair in enumerate(self.ordered_merges)}

    # ------------------------------------------------------------------
    # Diagnostics and low-level utilities
    # ------------------------------------------------------------------
    def _add_issue(self, issue: Union[LanguageIssue, LanguageError]) -> None:
        if not self.emit_diagnostics:
            return
        self.diagnostics.add(issue)

    def _record_tokenization_event(self, action: str, **payload: Any) -> None:
        self._tokenize_history.append(
            {
                "timestamp": utc_timestamp(),
                "action": action,
                "payload": json_safe(payload),
            }
        )
        record_event = getattr(self, "_record_event", None)
        if callable(record_event):
            record_event(action, **payload)

    def _add_token_to_vocab(self, token: str) -> int:
        token_text = require_non_empty_string(token, "token")
        if token_text not in self.vocab:
            token_id = len(self.vocab)
            self.vocab[token_text] = token_id
            self.inverse_vocab[token_id] = token_text
            if hasattr(self, "added_tokens_count"):
                self.added_tokens_count += 1
        return self.vocab[token_text]

    def _normalize_text(self, text: str) -> str:
        raw = ensure_text(text)
        if len(raw) > self.max_input_chars:
            raise TokenizationError(
                TokenizationIssue(
                    code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                    message="Input text exceeds the configured maximum length for tokenization.",
                    severity=Severity.ERROR,
                    module="LanguageTokenizer",
                    details={"max_input_chars": self.max_input_chars, "received_chars": len(raw)},
                ),
                recoverable=True,
            )
        return normalize_text(
            raw,
            lowercase=coerce_bool(self.normalization_rules.get("lowercase", self.lowercase), default=self.lowercase),
            casefold=coerce_bool(self.normalization_rules.get("casefold", False), default=False),
            unicode_form=ensure_text(self.normalization_rules.get("form", self.normalization_rules.get("unicode_form", "NFKC"))),
            normalize_quote_chars=coerce_bool(self.normalization_rules.get("normalize_quotes", False), default=False),
            normalize_dash_chars=coerce_bool(self.normalization_rules.get("normalize_dashes", False), default=False),
            collapse_whitespace=coerce_bool(self.normalization_rules.get("collapse_whitespace", True), default=True),
            remove_control_chars=coerce_bool(self.normalization_rules.get("remove_control_chars", True), default=True),
        )

    def _token_kind(self, text: str) -> str:
        if text in self.special_token_set:
            return "special"
        if is_punctuation(text):
            return "punct"
        if is_numeric_token(text):
            return "number"
        if is_word_like(text):
            return "word"
        return "symbol"

    # ------------------------------------------------------------------
    # Pre-tokenization and BPE application
    # ------------------------------------------------------------------
    def pre_tokenize(self, text: str, *, normalize: bool = True) -> List[PreToken]:
        source = self._normalize_text(text) if normalize else ensure_text(text)
        pre_tokens: List[PreToken] = []
        for index, match in enumerate(self._pre_tokenize_pattern.finditer(source)):
            value = match.group(0)
            if not value or value.isspace():
                continue
            pre_tokens.append(
                PreToken(
                    text=value,
                    start_char=int(match.start()),
                    end_char=int(match.end()),
                    index=index,
                    kind=self._token_kind(value),
                )
            )
        return pre_tokens

    def _word_to_initial_symbols(self, token: str) -> List[str]:
        if token in self.special_token_set:
            return [token]
        if self.preserve_punctuation_tokens and is_punctuation(token):
            return [token]
        symbols = list(token[: self.max_token_chars])
        if self.add_end_of_word:
            symbols.append(self.end_of_word_suffix)
        return symbols

    def _apply_bpe_to_symbols(self, symbols: Sequence[str]) -> List[str]:
        if not symbols:
            return []
        if not self.merge_ranks:
            return list(symbols)

        tokens = list(symbols)
        while len(tokens) > 1:
            candidate: Optional[Pair] = None
            candidate_rank: Optional[int] = None
            for index in range(len(tokens) - 1):
                pair = (tokens[index], tokens[index + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (candidate_rank is None or rank < candidate_rank):
                    candidate = pair
                    candidate_rank = rank
            if candidate is None:
                break
            merged = self.merges[candidate]
            next_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == candidate:
                    next_tokens.append(merged)
                    i += 2
                else:
                    next_tokens.append(tokens[i])
                    i += 1
            tokens = next_tokens
        return tokens

    def _decompose_unknown_subword(self, subword: str) -> List[str]:
        if subword in self.vocab:
            return [subword]
        if subword == self.end_of_word_suffix:
            return [subword] if subword in self.vocab else []
        if self.unknown_policy == "unk":
            self.unknown_token_count += 1
            return [self.unk_token]
        if self.unknown_policy == "dynamic" and self.allow_dynamic_vocab_growth:
            self._add_token_to_vocab(subword)
            return [subword]
        if self.unknown_policy in {"char", "char_fallback", "byte_fallback"} or self.fallback_to_char_tokenization:
            pieces: List[str] = []
            for char in subword:
                if char in self.vocab:
                    pieces.append(char)
                elif self.allow_dynamic_vocab_growth:
                    self._add_token_to_vocab(char)
                    pieces.append(char)
                else:
                    pieces.append(self.unk_token)
                    self.unknown_token_count += 1
            return pieces or [self.unk_token]
        self.unknown_token_count += 1
        return [self.unk_token]

    def _tokenize_pre_token(self, pre_token: PreToken) -> List[BPEToken]:
        if pre_token.text in self.special_token_set:
            token_id = self.token_to_id(pre_token.text)
            return [
                BPEToken(
                    token=pre_token.text,
                    token_id=token_id,
                    start_char=pre_token.start_char,
                    end_char=pre_token.end_char,
                    word_index=pre_token.index,
                    is_special=True,
                )
            ]

        initial_symbols = self._word_to_initial_symbols(pre_token.text)
        bpe_symbols = self._apply_bpe_to_symbols(initial_symbols)
        if not self.ordered_merges and self.fallback_to_char_tokenization:
            bpe_symbols = initial_symbols

        output: List[BPEToken] = []
        cursor = pre_token.start_char
        for symbol_index, symbol in enumerate(bpe_symbols):
            pieces = self._decompose_unknown_subword(symbol)
            for piece in pieces:
                if piece == self.end_of_word_suffix:
                    continue
                token_id = self.token_to_id(piece)
                unknown = piece == self.unk_token and symbol not in self.vocab
                visible_piece = piece.replace(self.end_of_word_suffix, "")
                if visible_piece:
                    start = cursor
                    width = min(len(visible_piece), max(0, pre_token.end_char - cursor))
                    end = cursor + width
                    cursor = end
                else:
                    start = pre_token.end_char
                    end = pre_token.end_char
                output.append(
                    BPEToken(
                        token=piece,
                        token_id=token_id,
                        start_char=start if self.track_offsets else None,
                        end_char=end if self.track_offsets else None,
                        word_index=pre_token.index,
                        is_special=False,
                        is_unknown=unknown,
                        is_continuation=symbol_index > 0,
                        metadata={"pre_token_kind": pre_token.kind},
                    )
                )
        return output

    # ------------------------------------------------------------------
    # Public tokenization API
    # ------------------------------------------------------------------
    def tokenize_detailed(self, text: str, **_: Any) -> TokenizationResult:
        original = ensure_text(text)
        normalized = self._normalize_text(original)
        pre_tokens = self.pre_tokenize(normalized, normalize=False)
        bpe_tokens: List[BPEToken] = []
        for pre_token in pre_tokens:
            bpe_tokens.extend(self._tokenize_pre_token(pre_token))

        self.tokenize_calls += 1
        self._record_tokenization_event(
            "language_tokenize",
            text_preview=truncate_text(original, self.lt_config.get("text_preview_length", 160)),
            pre_token_count=len(pre_tokens),
            token_count=len(bpe_tokens),
            trained=self.is_trained,
        )
        return TokenizationResult(
            original_text=original,
            normalized_text=normalized,
            tokens=tuple(bpe_tokens),
            pre_tokens=tuple(pre_tokens),
            issues=tuple(self.diagnostics.issues),
            metadata={
                "algorithm": self.algorithm,
                "trained": self.is_trained,
                "merge_count": len(self.ordered_merges),
                "vocab_size": len(self.vocab),
            },
        )

    def tokenize(self, text: str, **kwargs: Any) -> List[str]:
        """Convert text to BPE subword tokens."""
        return self.tokenize_detailed(text, **kwargs).token_texts

    def tokenize_words(self, text: str, *, normalize: bool = True) -> List[str]:
        """Return language-facing pre-tokens, not BPE subwords."""
        return [token.text for token in self.pre_tokenize(text, normalize=normalize)]

    def tokenize_with_offsets(self, text: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Return BPE tokens with character offsets and metadata."""
        return [token.to_dict() for token in self.tokenize_detailed(text, **kwargs).tokens]

    def encode(self, text: str, **kwargs: Any) -> Dict[str, Union[List[int], torch.Tensor, List[str], List[Tuple[int, int]], List[Dict[str, Any]]]]:  # type: ignore[override]
        max_length = self._coerce_max_length(kwargs.get("max_length", getattr(self, "default_max_length", None)))
        padding = self._resolve_padding(kwargs.get("padding", getattr(self, "default_padding", None)))
        truncation = coerce_bool(kwargs.get("truncation", getattr(self, "default_truncation", True)), default=True)
        add_special_tokens = coerce_bool(kwargs.get("add_special_tokens", getattr(self, "default_add_special_tokens", True)), default=True)
        return_tensors = kwargs.get("return_tensors", getattr(self, "default_return_tensors", None))
        return_tokens = coerce_bool(kwargs.get("return_tokens", getattr(self, "return_tokens", False)), default=False)
        return_offsets_mapping = coerce_bool(kwargs.get("return_offsets_mapping", False), default=False)
        return_token_metadata = coerce_bool(kwargs.get("return_token_metadata", False), default=False)
        return_attention_mask = coerce_bool(kwargs.get("return_attention_mask", getattr(self, "return_attention_mask", True)), default=True)
        return_special_tokens_mask = coerce_bool(kwargs.get("return_special_tokens_mask", getattr(self, "return_special_tokens_mask", False)), default=False)

        if return_tensors not in (None, "pt"):
            raise TokenizationError(
                TokenizationIssue(
                    code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                    message="LanguageTokenizer only supports return_tensors=None or 'pt'.",
                    severity=Severity.ERROR,
                    module="LanguageTokenizer",
                    details={"return_tensors": return_tensors},
                ),
                recoverable=True,
            )

        detailed = self.tokenize_detailed(text)
        bpe_tokens = list(detailed.tokens)
        if truncation and max_length is not None:
            reserved = 2 if add_special_tokens else 0
            bpe_tokens = bpe_tokens[: max(0, max_length - reserved)]

        tokens = [token.token for token in bpe_tokens]
        offsets: List[Tuple[int, int]] = [token.span or (0, 0) for token in bpe_tokens]
        token_metadata: List[Dict[str, Any]] = [token.to_dict() for token in bpe_tokens]

        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
            offsets = [(0, 0)] + offsets + [(0, 0)]
            token_metadata = [
                BPEToken(self.bos_token, self.bos_token_id, is_special=True).to_dict(),
                *token_metadata,
                BPEToken(self.eos_token, self.eos_token_id, is_special=True).to_dict(),
            ]

        input_ids = [self.token_to_id(token) for token in tokens]
        attention_mask = [1] * len(input_ids)
        special_tokens_mask = [1 if token in self.special_token_set else 0 for token in tokens]

        target_length = max_length if padding == "max_length" else None
        if target_length is not None:
            pad_id = self.pad_token_id
            if len(input_ids) < target_length:
                missing = target_length - len(input_ids)
                input_ids += [pad_id] * missing
                attention_mask += [0] * missing
                special_tokens_mask += [1] * missing
                offsets += [(0, 0)] * missing
                token_metadata += [BPEToken(self.pad_token, pad_id, is_special=True).to_dict()] * missing
            else:
                input_ids = input_ids[:target_length]
                attention_mask = attention_mask[:target_length]
                special_tokens_mask = special_tokens_mask[:target_length]
                offsets = offsets[:target_length]
                token_metadata = token_metadata[:target_length]

        payload: Dict[str, Any] = {"input_ids": input_ids}
        if return_attention_mask:
            payload["attention_mask"] = attention_mask
        if return_special_tokens_mask:
            payload["special_tokens_mask"] = special_tokens_mask
        if return_tokens:
            payload["tokens"] = tokens[: len(input_ids)]
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets[: len(input_ids)]
        if return_token_metadata:
            payload["token_metadata"] = token_metadata[: len(input_ids)]

        if return_tensors == "pt":
            for key in ("input_ids", "attention_mask", "special_tokens_mask"):
                if key in payload:
                    payload[key] = torch.tensor(payload[key], dtype=torch.long)

        self.encode_calls += 1
        self._record_tokenization_event(
            "language_encode",
            token_count=len(input_ids),
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
        )
        return payload

    def detokenize(self, tokens: Sequence[str]) -> str:
        values = [ensure_text(token) for token in ensure_sequence(tokens, field_name="tokens", allow_none=True)]
        if not values:
            return ""
        words: List[str] = []
        current = ""
        for token in values:
            if token in self.special_token_set:
                continue
            if token == self.end_of_word_suffix:
                if current:
                    words.append(current)
                    current = ""
                continue
            if token.endswith(self.end_of_word_suffix):
                current += token[: -len(self.end_of_word_suffix)]
                words.append(current)
                current = ""
                continue
            if is_punctuation(token):
                if current:
                    words.append(current)
                    current = ""
                words.append(token)
                continue
            current += token
        if current:
            words.append(current)
        return normalize_spacing_around_punctuation(tokens_to_text(words))

    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs: Any) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        skip_special_tokens = coerce_bool(kwargs.get("skip_special_tokens", getattr(self, "default_skip_special_tokens", True)), default=True)
        clean_up = coerce_bool(kwargs.get("clean_up_tokenization_spaces", getattr(self, "default_clean_up_tokenization_spaces", True)), default=True)
        ids = [int(item) for item in ensure_sequence(token_ids, field_name="token_ids", allow_none=True)]
        tokens = [self.id_to_token(token_id) for token_id in ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_token_set]
        text = self.detokenize(tokens)
        if clean_up:
            text = self.clean_text(text)
        self.decode_calls += 1
        self._record_tokenization_event("language_decode", token_count=len(ids), output_preview=truncate_text(text, 160))
        return text

    def clean_text(self, text: str) -> str:
        return normalize_spacing_around_punctuation(normalize_whitespace(text))

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_pair_stats(word_freqs: Mapping[Tuple[str, ...], int]) -> Counter[Pair]:
        """Count adjacent symbol-pair frequencies for BPE training."""
        counts: Counter[Pair] = Counter()
        for symbols, frequency in word_freqs.items():
            if len(symbols) < 2:
                continue
            freq = int(frequency)
            for index in range(len(symbols) - 1):
                counts[(symbols[index], symbols[index + 1])] += freq
        return counts

    @staticmethod
    def _merge_pair(
        word_freqs: Mapping[Tuple[str, ...], int],
        pair_to_merge: Pair,
        new_token: str,
    ) -> Dict[Tuple[str, ...], int]:
        """Apply one BPE merge to the training corpus representation."""
        merged_word_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
        for symbols, frequency in word_freqs.items():
            next_symbols: List[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and symbols[index] == pair_to_merge[0]
                    and symbols[index + 1] == pair_to_merge[1]
                ):
                    next_symbols.append(new_token)
                    index += 2
                else:
                    next_symbols.append(symbols[index])
                    index += 1
            merged_word_freqs[tuple(next_symbols)] += int(frequency)
        return dict(merged_word_freqs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        corpus: Sequence[str],
        min_freq: Optional[int] = None,
        max_vocab_size: Optional[int] = None,
        reset_existing: bool = True,
        **kwargs: Any,
    ) -> TrainingSummary:
        """Train the BPE tokenizer on a corpus and return a BaseTokenizer-compatible summary."""
        force_retrain = coerce_bool(kwargs.pop("force_retrain", False), default=False)
        effective_reset = reset_existing or force_retrain

        if self.is_trained and not effective_reset:
            logger.info("LanguageTokenizer is already trained. Use force_retrain=True to retrain.")
            if isinstance(self.last_training_summary, TrainingSummary):
                return self.last_training_summary
            return TrainingSummary(
                corpus_size=len(corpus),
                min_frequency=self.min_frequency,
                max_vocab_size=self.target_vocab_size,
                learned_tokens=0,
                total_unique_observed_tokens=0,
                total_observed_tokens=0,
            )

        start_ms = monotonic_ms()
        effective_min_freq = coerce_int(min_freq if min_freq is not None else self.min_frequency, default=self.min_frequency, minimum=1)
        target_size = coerce_int(max_vocab_size if max_vocab_size is not None else self.target_vocab_size, default=self.target_vocab_size, minimum=len(self.special_tokens) + 1)

        if effective_reset:
            self._bootstrap_special_vocab()
            self.merges.clear()
            self.ordered_merges.clear()
            self.merge_ranks.clear()

        printer.status("TRAIN", "Starting LanguageTokenizer BPE training", "info")

        word_counts: Counter[str] = Counter()
        normalized_count = 0
        for item in corpus:
            text = ensure_text(item)
            if not text:
                continue
            normalized = self._normalize_text(text)
            normalized_count += 1
            for pre_token in self.pre_tokenize(normalized, normalize=False):
                if pre_token.kind in {"word", "number", "symbol", "punct"}:
                    word_counts[pre_token.text] += 1

        self.token_counts = word_counts
        bpe_word_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
        alphabet: set[str] = set()
        for word, freq in word_counts.items():
            symbols = tuple(self._word_to_initial_symbols(word))
            bpe_word_freqs[symbols] += int(freq)
            alphabet.update(symbols)

        for token in sorted(alphabet):
            self._add_token_to_vocab(token)

        initial_vocab_size = len(self.vocab)
        merges_budget = max(0, min(self.max_merges, target_size - len(self.vocab)))
        stopped_reason = "merge_budget_exhausted"

        for merge_index in range(merges_budget):
            pair_counts = self._get_pair_stats(bpe_word_freqs)
            if not pair_counts:
                stopped_reason = "no_pairs_remaining"
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < effective_min_freq:
                stopped_reason = "min_frequency_reached"
                break
            new_token = best_pair[0] + best_pair[1]
            if new_token in self.vocab and best_pair in self.merges:
                stopped_reason = "duplicate_merge_reached"
                break
            self.ordered_merges.append(best_pair)
            self.merges[best_pair] = new_token
            self._add_token_to_vocab(new_token)
            bpe_word_freqs = self._merge_pair(bpe_word_freqs, best_pair, new_token)
            if (merge_index + 1) % 250 == 0:
                printer.status("TRAIN", f"Learned {merge_index + 1} BPE merges", "info")

        self._refresh_merge_ranks()
        self.is_trained = bool(self.ordered_merges)
        self.training_runs = getattr(self, "training_runs", 0) + 1
        self.src_vocab_size = len(self.vocab)
        duration = elapsed_ms(start_ms)

        bpe_summary = BPETrainingSummary(
            corpus_size=len(corpus),
            normalized_corpus_size=normalized_count,
            min_frequency=effective_min_freq,
            target_vocab_size=target_size,
            initial_vocab_size=initial_vocab_size,
            final_vocab_size=len(self.vocab),
            learned_merges=len(self.ordered_merges),
            observed_words=sum(word_counts.values()),
            observed_word_types=len(word_counts),
            stopped_reason=stopped_reason,
            duration_ms=duration,
        )
        self.bpe_last_training_summary = bpe_summary
        self.training_history.append(bpe_summary.to_dict())
        self._record_tokenization_event("language_train", summary=bpe_summary.to_dict())

        base_summary = TrainingSummary(
            corpus_size=bpe_summary.corpus_size,
            min_frequency=bpe_summary.min_frequency,
            max_vocab_size=bpe_summary.target_vocab_size,
            learned_tokens=bpe_summary.learned_merges,
            total_unique_observed_tokens=bpe_summary.observed_word_types,
            total_observed_tokens=bpe_summary.observed_words,
            timestamp=bpe_summary.created_at,
        )
        self.last_training_summary = base_summary

        printer.status("TRAIN", f"LanguageTokenizer training complete: vocab={len(self.vocab)}, merges={len(self.ordered_merges)}", "success")
        return base_summary

    # ------------------------------------------------------------------
    # Vocabulary and persistence
    # ------------------------------------------------------------------
    def token_to_id(self, token: str) -> int:
        token_text = ensure_text(token)
        if token_text in self.vocab:
            return int(self.vocab[token_text])
        if self.allow_dynamic_vocab_growth and self.unknown_policy == "dynamic":
            return self._add_token_to_vocab(token_text)
        self.unknown_token_count += 1
        return int(self.vocab.get(self.unk_token, 0))

    def id_to_token(self, token_id: int) -> str:
        return self.inverse_vocab.get(int(token_id), self.unk_token)

    def add_tokens(self, tokens: Union[str, Sequence[str]], special: bool = False) -> int:
        values = [tokens] if isinstance(tokens, str) else list(tokens)
        added = 0
        for value in values:
            token = require_non_empty_string(value, "token")
            if token not in self.vocab:
                self._add_token_to_vocab(token)
                added += 1
            if special and token not in self.special_token_set:
                self.special_tokens.append(token)
                self.special_token_set.add(token)
        if special:
            self._pre_tokenize_pattern = self._compile_pre_tokenize_pattern()
        return added

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def model_payload(self) -> Dict[str, Any]:
        return {
            "format": self.MODEL_FORMAT,
            "version": self.version,
            "algorithm": self.algorithm,
            "created_at": utc_timestamp(),
            "end_of_word_suffix": self.end_of_word_suffix,
            "continuation_prefix": self.continuation_prefix,
            "ordered_merges": [list(pair) for pair in self.ordered_merges],
            "merges": [list(pair) for pair in self.ordered_merges],
            "special_tokens": list(self.special_tokens),
            "normalization_rules": json_safe(self.normalization_rules),
            "training_summary": self.bpe_last_training_summary.to_dict() if self.bpe_last_training_summary else None,
            "base_training_summary": self.last_training_summary.to_dict() if isinstance(self.last_training_summary, TrainingSummary) else None,
        }

    def save(self, directory: Union[str, Path], name: str = "language_tokenizer") -> List[str]:
        target_dir = resolve_path(directory, field_name="directory")
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = safe_filename(name, default="language_tokenizer")
        config_path = target_dir / f"{safe_name}_config.json"
        model_path = target_dir / f"{safe_name}_model.json"
        vocab_path = target_dir / f"{safe_name}_vocab.json"
        counts_path = target_dir / f"{safe_name}_token_counts.json"

        config_payload = {
            "tokenizer_class": self.__class__.__name__,
            "version": self.version,
            "model_format": self.MODEL_FORMAT,
            "stats": self.stats().to_dict(),
            "settings": {
                "algorithm": self.algorithm,
                "target_vocab_size": self.target_vocab_size,
                "min_frequency": self.min_frequency,
                "max_merges": self.max_merges,
                "end_of_word_suffix": self.end_of_word_suffix,
                "unknown_policy": self.unknown_policy,
                "normalization_rules": json_safe(self.normalization_rules),
            },
            "diagnostics": self.diagnostics.to_list(),
        }
        save_json_file(config_path, config_payload, pretty=self.save_pretty)
        save_json_file(model_path, self.model_payload(), pretty=self.save_pretty)
        save_json_file(vocab_path, {"vocab": self.vocab}, pretty=self.save_pretty)
        save_json_file(counts_path, dict(self.token_counts), pretty=self.save_pretty)
        self._record_tokenization_event("language_save", directory=str(target_dir), name=safe_name)
        printer.status("SAVE", f"LanguageTokenizer saved to {target_dir}", "success")
        return [str(config_path), str(model_path), str(vocab_path), str(counts_path)]

    @classmethod
    def load(cls, directory: Union[str, Path], name: str = "language_tokenizer") -> "LanguageTokenizer":
        target_dir = resolve_path(directory, must_exist=True, field_name="directory")
        safe_name = safe_filename(name, default="language_tokenizer")
        config_path = target_dir / f"{safe_name}_config.json"
        model_path = target_dir / f"{safe_name}_model.json"
        vocab_path = target_dir / f"{safe_name}_vocab.json"
        counts_path = target_dir / f"{safe_name}_token_counts.json"
        if not config_path.exists() or not model_path.exists() or not vocab_path.exists():
            raise ResourceLanguageError(
                ResourceIssue(
                    code=LanguageErrorCode.RESOURCE_MISSING,
                    message="LanguageTokenizer save files are incomplete or missing.",
                    module="LanguageTokenizer",
                    details={
                        "config_path": str(config_path),
                        "model_path": str(model_path),
                        "vocab_path": str(vocab_path),
                    },
                ),
                recoverable=False,
            )
        tokenizer = cls()
        tokenizer._load_bpe_model(model_path)
        tokenizer._load_bpe_vocab(vocab_path)
        counts = load_json_file(counts_path) if counts_path.exists() else {}
        tokenizer.token_counts = Counter({ensure_text(k): int(v) for k, v in dict(counts or {}).items()})
        tokenizer._refresh_merge_ranks()
        tokenizer.is_trained = bool(tokenizer.ordered_merges and tokenizer.vocab)
        tokenizer.src_vocab_size = len(tokenizer.vocab)
        tokenizer._record_tokenization_event("language_load", directory=str(target_dir), name=safe_name)
        printer.status("LOAD", f"LanguageTokenizer loaded from {target_dir}", "success")
        return tokenizer

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> LanguageTokenizerStats:  # type: ignore[override]
        return LanguageTokenizerStats(
            request_id=getattr(self, "_lock_request_id", generate_correlation_id("lang_tok")),
            version=self.version,
            algorithm=self.algorithm,
            trained=self.is_trained,
            vocab_size=len(self.vocab),
            special_token_count=len(self.special_tokens),
            merge_count=len(self.ordered_merges),
            token_counts_size=len(self.token_counts),
            tokenize_calls=getattr(self, "tokenize_calls", 0),
            encode_calls=getattr(self, "encode_calls", 0),
            decode_calls=getattr(self, "decode_calls", 0),
            training_runs=getattr(self, "training_runs", 0),
            unknown_token_count=self.unknown_token_count,
            diagnostics_count=len(self.diagnostics.issues),
            history_length=len(self._tokenize_history),
            model_path=str(self.bpe_model_file) if self.bpe_model_file else None,
            vocab_path=str(self.bpe_vocab_file) if self.bpe_vocab_file else None,
        )

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        count = coerce_int(limit, default=20, minimum=1)
        return list(self._tokenize_history)[-count:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokenizer_class": self.__class__.__name__,
            "version": self.version,
            "model_format": self.MODEL_FORMAT,
            "stats": self.stats().to_dict(),
            "special_tokens": list(self.special_tokens),
            "vocab_size": len(self.vocab),
            "merge_count": len(self.ordered_merges),
            "last_training_summary": self.last_training_summary.to_dict() if isinstance(self.last_training_summary, TrainingSummary) else None,
            "bpe_last_training_summary": self.bpe_last_training_summary.to_dict() if self.bpe_last_training_summary else None,
            "diagnostics": self.diagnostics.to_list(),
        }

    def __repr__(self) -> str:
        return f"<LanguageTokenizer version='{self.version}' vocab_size={len(self.vocab)} merges={len(self.ordered_merges)} trained={self.is_trained}>"


if __name__ == "__main__":
    print("\n=== Running Language Tokenizer ===\n")
    printer.status("TEST", "Language Tokenizer initialized", "info")

    tokenizer = LanguageTokenizer()
    corpus = [
        "Language tokenization should preserve useful linguistic boundaries.",
        "Do not split contractions oddly; keep punctuation and numbers like 19.99 stable.",
        "A production tokenizer needs spans, diagnostics, BPE merges, and clean decode behavior.",
        "The language agent uses tokens for NLP, grammar, NLU, context, and NLG.",
    ]

    summary = tokenizer.train(corpus, min_freq=1, force_retrain=True)
    printer.pretty("TRAINING_SUMMARY", summary.to_dict(), "success")

    sample = "Do not oversimplify tokenization - keep spans, $19.99, and symbols intact!"
    detailed = tokenizer.tokenize_detailed(sample)
    encoded = tokenizer.encode(
        sample,
        max_length=48,
        padding="max_length",
        return_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    decoded = tokenizer.decode(encoded["input_ids"])  # type: ignore[arg-type]

    printer.pretty("TOKENS", detailed.to_dict(), "success")
    printer.pretty("ENCODED", encoded, "success")
    printer.pretty("DECODED", {"text": decoded}, "success")
    printer.pretty("STATS", tokenizer.stats().to_dict(), "success")
    printer.pretty("DIAGNOSTICS", tokenizer.diagnostics.to_list(), "info")

    save_paths = tokenizer.save("/tmp/language_tokenizer_artifacts", name="language_tokenizer_test")
    loaded = LanguageTokenizer.load("/tmp/language_tokenizer_artifacts", name="language_tokenizer_test")
    roundtrip = loaded.decode(loaded.encode(sample)["input_ids"])  # type: ignore[arg-type]

    printer.pretty("SAVE_PATHS", save_paths, "success")
    printer.pretty("LOADED", loaded.to_dict(), "success")
    printer.pretty("ROUNDTRIP", {"text": roundtrip}, "success")

    print("\n=== Test ran successfully ===\n")
