"""
Production-grade shared helpers for the language subsystem.

This module centralizes reusable primitives for SLAI's language stack:
orthography correction, tokenization, NLP feature extraction, dependency
analysis, grammar checking, NLU parsing, dialogue context tracking, and NLG.

The helpers here are intentionally language-domain utilities. They do not own
model inference, grammar rule execution, tokenization algorithms, dialogue
memory policy, or response generation. Instead, they provide stable contracts
for normalization, spans, lexical scoring, token snapshots, frame handling,
template filling, diagnostics, serialization, redaction, identifiers, and
pipeline payload construction.

Design principles
-----------------
1. Stable language contracts: dataclasses expose predictable shapes.
2. Span safety: helpers preserve original/normalized text alignment whenever
   possible and make span failures explicit.
3. Linguistic focus: utilities are scoped to text, tokens, frames, grammar,
   NLU/NLG payloads, and language diagnostics.
4. Expansion-ready: functions are granular so future modules can reuse them
   without expanding monolithic processors.
5. Project-native diagnostics: errors and issues use language_error.py and the
   project's logger directly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time as time_module
import unicodedata
import uuid

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, TypeVar, Union, cast

from .config_loader import get_config_section, load_global_config
from .linguistic_frame import LinguisticFrame, SpeechActType
from .language_error import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Language Helpers")
printer = PrettyPrinter

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
JsonDict = Dict[str, Any]
Span = Tuple[int, int]
TokenLike = Any
SentenceLike = Any

# ---------------------------------------------------------------------------
# Defaults and patterns
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE = "en"
DEFAULT_LOCALE = "en-US"
DEFAULT_MAX_STRING_LENGTH = 4096
DEFAULT_MAX_SERIALIZATION_DEPTH = 8
DEFAULT_MAX_COLLECTION_ITEMS = 100
DEFAULT_REDACTION = "***REDACTED***"
DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_IDENTIFIER_PREFIX = "lang"

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_NON_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_ASCII_WORD_RE = re.compile(r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_ACRONYM_RE = re.compile(r"^[A-Z](?:[A-Z.]*[A-Z])?\.?$")
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"')\]]*$")
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_\.\-]*)\}")
_CAMEL_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")

PUNCT_NO_SPACE_BEFORE = set(",.;:!?%)]}»”’")
PUNCT_NO_SPACE_AFTER = set("([{¿¡«“‘$")
SENTENCE_TERMINATORS = {".", "!", "?", "…"}
OPENING_QUOTES = {"\"", "'", "“", "‘", "«"}
CLOSING_QUOTES = {"\"", "'", "”", "’", "»"}

PENN_TO_UPOS: Dict[str, str] = {
    "CC": "CCONJ", "CD": "NUM", "DT": "DET", "EX": "PRON", "FW": "X", "IN": "ADP",
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ", "LS": "X", "MD": "AUX", "NN": "NOUN",
    "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN", "PDT": "DET", "POS": "PART", "PRP": "PRON",
    "PRP$": "PRON", "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "RP": "PART", "SYM": "SYM",
    "TO": "PART", "UH": "INTJ", "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB",
    "VBP": "VERB", "VBZ": "VERB", "WDT": "DET", "WP": "PRON", "WP$": "PRON", "WRB": "ADV",
    ".": "PUNCT", ",": "PUNCT", ":": "PUNCT", "-LRB-": "PUNCT", "-RRB-": "PUNCT",
}
UPOS_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}

SENSITIVE_KEYS = {
    "password", "passwd", "secret", "token", "access_token", "refresh_token", "api_key", "apikey",
    "authorization", "auth", "private_key", "client_secret", "session", "session_id", "cookie", "set_cookie",
    "email", "phone", "address", "user_id", "username", "name", "pii",
}
SECRET_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(bearer\s+)([a-z0-9\-._~+/]+=*)"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(token\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(secret\s*[=:]\s*)([^\s,;]+)"),
)

ACRONYM_AN_INITIALS = set("AEFHILMNORSX")
SILENT_H_PREFIXES = (
    "honest", "honor", "honour", "hour", "heir", "herb",
)
LONG_U_PREFIXES = (
    "uni", "use", "user", "usual", "utensil", "utility", "utopia", "euro", "eul", "euph",
)
VOWELS = set("aeiou")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TextSpan:
    """A validated half-open character span: [start, end)."""

    start: int
    end: int
    label: Optional[str] = None
    text: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span [{self.start}, {self.end}).")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", clamp_float(self.confidence, 0.0, 1.0))

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def empty(self) -> bool:
        return self.length == 0

    def as_tuple(self) -> Span:
        return (self.start, self.end)

    def overlaps(self, other: Union["TextSpan", Span]) -> bool:
        other_span = ensure_span(other)
        return spans_overlap(self.as_tuple(), other_span)

    def contains(self, index: int) -> bool:
        return self.start <= index < self.end

    def shift(self, delta: int) -> "TextSpan":
        return TextSpan(
            start=max(0, self.start + delta),
            end=max(0, self.end + delta),
            label=self.label,
            text=self.text,
            confidence=self.confidence,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> JsonDict:
        return prune_none(asdict(self), drop_empty=True)


@dataclass(frozen=True)
class SpanEdit:
    """A replacement applied to a half-open source span."""

    source_span: Span
    replacement: str
    original: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_span(self.source_span, field_name="source_span")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", clamp_float(self.confidence, 0.0, 1.0))

    @property
    def source_length(self) -> int:
        return self.source_span[1] - self.source_span[0]

    @property
    def replacement_length(self) -> int:
        return len(self.replacement)

    @property
    def delta(self) -> int:
        return self.replacement_length - self.source_length

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["source_span"] = list(self.source_span)
        return prune_none(payload, drop_empty=True)


@dataclass(frozen=True)
class NormalizedText:
    """Normalized text plus edit and offset mapping metadata."""

    original: str
    normalized: str
    edits: Tuple[SpanEdit, ...] = ()
    original_to_normalized: Dict[int, int] = field(default_factory=dict)
    normalized_to_original: Dict[int, int] = field(default_factory=dict)
    language: str = DEFAULT_LANGUAGE
    locale: str = DEFAULT_LOCALE
    issues: Tuple[LanguageIssue, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.original != self.normalized

    def map_original_index(self, index: int, *, default: Optional[int] = None) -> Optional[int]:
        return self.original_to_normalized.get(index, default)

    def map_normalized_index(self, index: int, *, default: Optional[int] = None) -> Optional[int]:
        return self.normalized_to_original.get(index, default)

    def to_dict(self) -> JsonDict:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "changed": self.changed,
            "edits": [edit.to_dict() for edit in self.edits],
            "original_to_normalized": dict(self.original_to_normalized),
            "normalized_to_original": dict(self.normalized_to_original),
            "language": self.language,
            "locale": self.locale,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class TokenSnapshot:
    """Serializable token representation shared by NLP, grammar, and NLU."""

    text: str
    index: int
    lemma: Optional[str] = None
    pos: Optional[str] = None
    upos: Optional[str] = None
    dep: Optional[str] = None
    head: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    sentence_index: Optional[int] = None
    is_stop: bool = False
    is_punct: bool = False
    morphology: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> Optional[Span]:
        if self.start_char is None or self.end_char is None:
            return None
        return (self.start_char, self.end_char)

    def to_dict(self) -> JsonDict:
        return prune_none(asdict(self), drop_empty=True)


@dataclass(frozen=True)
class SentenceSnapshot:
    """Serializable sentence-level language analysis snapshot."""

    text: str
    index: int
    tokens: Tuple[TokenSnapshot, ...] = ()
    span: Optional[Span] = None
    sentence_type: Optional[str] = None
    confidence: Optional[float] = None
    issues: Tuple[LanguageIssue, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "text": self.text,
            "index": self.index,
            "tokens": [token.to_dict() for token in self.tokens],
            "span": list(self.span) if self.span else None,
            "sentence_type": self.sentence_type,
            "confidence": self.confidence,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class EntitySnapshot:
    """Normalized entity representation for NLU and dialogue context."""

    text: str
    label: str
    value: Any = None
    span: Optional[Span] = None
    confidence: Optional[float] = None
    normalized: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return prune_none(
            {
                "text": self.text,
                "label": self.label,
                "value": json_safe(self.value),
                "span": list(self.span) if self.span else None,
                "confidence": self.confidence,
                "normalized": self.normalized,
                "source": self.source,
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )


@dataclass(frozen=True)
class LanguagePipelinePayload:
    """Shared payload shape for expanded language modules."""

    original_text: str
    normalized_text: Optional[str] = None
    sentences: Tuple[SentenceSnapshot, ...] = ()
    tokens: Tuple[TokenSnapshot, ...] = ()
    frame: Optional[LinguisticFrame] = None
    issues: Tuple[LanguageIssue, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: generate_language_id("lang_payload"))
    created_at: str = field(default_factory=lambda: utc_timestamp())

    @property
    def text(self) -> str:
        return self.normalized_text if self.normalized_text is not None else self.original_text

    @property
    def ok(self) -> bool:
        return not any(issue.is_blocking for issue in self.issues)

    def with_issue(self, issue: Union[LanguageIssue, LanguageError]) -> "LanguagePipelinePayload":
        issue_obj = issue.issue if isinstance(issue, LanguageError) else issue
        return LanguagePipelinePayload(
            original_text=self.original_text,
            normalized_text=self.normalized_text,
            sentences=self.sentences,
            tokens=self.tokens,
            frame=self.frame,
            issues=tuple(list(self.issues) + [issue_obj]),
            metadata=dict(self.metadata),
            correlation_id=self.correlation_id,
            created_at=self.created_at,
        )

    def to_dict(self) -> JsonDict:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "text": self.text,
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "tokens": [token.to_dict() for token in self.tokens],
            "frame": frame_to_dict(self.frame),
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": json_safe(self.metadata),
            "correlation_id": self.correlation_id,
            "created_at": self.created_at,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class TemplateRenderResult:
    """Template rendering output with diagnostics."""

    text: str
    template: str
    values: Dict[str, Any] = field(default_factory=dict)
    missing_placeholders: Tuple[str, ...] = ()
    unused_values: Tuple[str, ...] = ()
    issues: Tuple[LanguageIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing_placeholders and not any(issue.is_blocking for issue in self.issues)

    def to_dict(self) -> JsonDict:
        return {
            "ok": self.ok,
            "text": self.text,
            "template": self.template,
            "values": json_safe(self.values),
            "missing_placeholders": list(self.missing_placeholders),
            "unused_values": list(self.unused_values),
            "issues": [issue.to_dict() for issue in self.issues],
        }


# ---------------------------------------------------------------------------
# Runtime config helpers
# ---------------------------------------------------------------------------
def get_language_config() -> Dict[str, Any]:
    """Return the current global language config as a plain dictionary."""

    config = load_global_config()
    return dict(config or {}) if isinstance(config, Mapping) else {}


def get_language_helper_config() -> Dict[str, Any]:
    """Return config for this helper module if present."""

    section = get_config_section("language_helpers")
    return dict(section or {}) if isinstance(section, Mapping) else {}


def config_bool(section_name: str, key: str, default: bool = False) -> bool:
    section = get_config_section(section_name) or {}
    return coerce_bool(section.get(key), default=default) if isinstance(section, Mapping) else default


def config_int(section_name: str, key: str, default: int = 0, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    section = get_config_section(section_name) or {}
    value = section.get(key) if isinstance(section, Mapping) else default
    return coerce_int(value, default=default, minimum=minimum, maximum=maximum)


def config_float(section_name: str, key: str, default: float = 0.0, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    section = get_config_section(section_name) or {}
    value = section.get(key) if isinstance(section, Mapping) else default
    return coerce_float(value, default=default, minimum=minimum, maximum=maximum)

# ---------------------------------------------------------------------------
# Time, identifiers, hashing
# ---------------------------------------------------------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(*, timespec: str = "seconds") -> str:
    return utc_now().isoformat(timespec=timespec)


def epoch_seconds() -> float:
    return time_module.time()


def monotonic_ms() -> float:
    return time_module.monotonic() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    return round(max(0.0, monotonic_ms() - float(start_ms)), 3)


def normalize_identifier_component(
    value: Any,
    *,
    default: str = "item",
    lowercase: bool = True,
    max_length: int = 120,
    separator: str = "_",
) -> str:
    text = normalize_whitespace(value)
    if lowercase:
        text = text.lower()
    text = _NON_IDENTIFIER_RE.sub(separator, text).strip(separator)
    if max_length > 0:
        text = text[:max_length].strip(separator)
    return text or default


def generate_language_id(prefix: str = DEFAULT_IDENTIFIER_PREFIX, *, length: int = 24, separator: str = "_") -> str:
    normalized_prefix = normalize_identifier_component(prefix, default=DEFAULT_IDENTIFIER_PREFIX)
    token = uuid.uuid4().hex[: max(8, min(32, int(length)))]
    return f"{normalized_prefix}{separator}{token}"


def generate_trace_id(prefix: str = "lang_trace") -> str:
    return generate_language_id(prefix, length=24)


def generate_correlation_id(prefix: str = "lang_corr") -> str:
    return generate_language_id(prefix, length=24)


def stable_hash(value: Any, *, algorithm: str = DEFAULT_HASH_ALGORITHM, length: int = 16) -> str:
    digest = hashlib.new((algorithm or DEFAULT_HASH_ALGORITHM).lower())
    digest.update(stable_json_dumps(value).encode("utf-8", errors="replace"))
    hexdigest = digest.hexdigest()
    return hexdigest[: max(1, int(length))] if length else hexdigest


def fingerprint_text(text: Any, *, length: int = 16) -> str:
    return stable_hash(normalize_for_comparison(text), length=length)

# ---------------------------------------------------------------------------
# Coercion and validation
# ---------------------------------------------------------------------------
def coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled", "disable"}:
        return False
    return default


def coerce_int(value: Any, *, default: int = 0, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not an integer value")
        result = int(float(value))
    except Exception:
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def coerce_float(value: Any, *, default: float = 0.0, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not a float value")
        result = float(value)
    except Exception:
        result = float(default)
    if math.isnan(result) or math.isinf(result):
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def coerce_probability(value: Any, *, default: float = 0.0) -> float:
    return coerce_float(value, default=default, minimum=0.0, maximum=1.0)


def clamp(value: Union[int, float], minimum: Union[int, float], maximum: Union[int, float]) -> Union[int, float]:
    return max(minimum, min(maximum, value))


def clamp_float(value: Any, minimum: float = 0.0, maximum: float = 1.0, *, default: float = 0.0) -> float:
    return coerce_float(value, default=default, minimum=minimum, maximum=maximum)


def ensure_list(value: Any, *, drop_none: bool = True) -> List[Any]:
    if value is None:
        return [] if drop_none else [None]
    if isinstance(value, list):
        items = value
    elif isinstance(value, (tuple, set, frozenset)):
        items = list(value)
    else:
        items = [value]
    return [item for item in items if item is not None] if drop_none else list(items)


def ensure_mapping(value: Any, *, field_name: str = "value", allow_none: bool = False) -> Dict[str, Any]:
    if value is None:
        if allow_none:
            return {}
        raise PipelineContractError(f"{field_name} must be a mapping, received None.", expected="Mapping", received=None)
    if isinstance(value, Mapping):
        return dict(value)
    raise PipelineContractError(
        f"{field_name} must be a mapping-like object.",
        expected="Mapping",
        received=type(value).__name__,
    )


def ensure_sequence(
    value: Any,
    *,
    field_name: str = "value",
    allow_none: bool = False,
    coerce_scalar: bool = False,
) -> Tuple[Any, ...]:
    if value is None:
        if allow_none:
            return ()
        raise PipelineContractError(f"{field_name} must be a sequence, received None.", expected="Sequence", received=None)
    if isinstance(value, (str, bytes, bytearray)):
        if coerce_scalar:
            return (value,)
        raise PipelineContractError(f"{field_name} must be a sequence, not scalar text/bytes.", expected="Sequence", received=type(value).__name__)
    if isinstance(value, Sequence):
        return tuple(value)
    if coerce_scalar:
        return (value,)
    raise PipelineContractError(f"{field_name} must be sequence-like.", expected="Sequence", received=type(value).__name__)


def require_non_empty_string(value: Any, field_name: str, *, max_length: Optional[int] = None) -> str:
    text = normalize_whitespace(value)
    if not text:
        raise PipelineContractError(f"{field_name} must be a non-empty string.", expected="non-empty string", received=value)
    if max_length is not None and max_length > 0 and len(text) > max_length:
        raise PipelineContractError(f"{field_name} exceeds maximum length {max_length}.", expected=f"<= {max_length} chars", received=len(text))
    return text


def first_non_none(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def first_truthy(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value:
            return value
    return default


def dedupe_preserve_order(values: Iterable[T], *, key: Optional[Callable[[T], Any]] = None) -> List[T]:
    seen: Set[Any] = set()
    result: List[T] = []
    for value in values:
        marker = key(value) if key else value
        marker_hashable = stable_json_dumps(marker) if isinstance(marker, (Mapping, list, tuple, set)) else marker
        if marker_hashable in seen:
            continue
        seen.add(marker_hashable)
        result.append(value)
    return result


def chunked(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size <= 0:
        raise PipelineContractError("chunk size must be > 0", expected="> 0", received=size)
    for index in range(0, len(values), size):
        yield values[index:index + size]

# ---------------------------------------------------------------------------
# Text normalization and language-specific text helpers
# ---------------------------------------------------------------------------
def ensure_text(value: Any, *, encoding: str = "utf-8", errors: str = "replace") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode(encoding, errors=errors)
    return str(value)


def strip_control_chars(value: Any) -> str:
    return _CONTROL_CHARS_RE.sub("", ensure_text(value))


def normalize_unicode(value: Any, *, form: Optional[str] = "NFKC") -> str:
    text = ensure_text(value)
    valid_forms = {"NFC", "NFD", "NFKC", "NFKD"}
    if form and form.upper() in valid_forms:
        return unicodedata.normalize(form.upper(), text)  # type: ignore
    return text


def normalize_quotes(value: Any) -> str:
    text = ensure_text(value)
    return (
        text.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"')
        .replace("‘", "'").replace("’", "'").replace("‚", "'").replace("‛", "'")
    )


def normalize_dashes(value: Any) -> str:
    return ensure_text(value).replace("—", "-").replace("–", "-").replace("−", "-").replace("‐", "-")


def normalize_whitespace(value: Any, *, preserve_newlines: bool = False) -> str:
    text = strip_control_chars(value)
    if preserve_newlines:
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text.strip()
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_newlines(value: Any, *, max_blank_lines: int = 2) -> str:
    text = ensure_text(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    blank_pattern = r"\n{" + str(max(2, max_blank_lines + 1)) + r",}"
    return re.sub(blank_pattern, "\n" * max_blank_lines, text).strip()


def normalize_spacing_around_punctuation(value: Any) -> str:
    text = normalize_whitespace(value)
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+(['\"”’])(?=\s|$)", r"\1", text)
    return text.strip()


def normalize_text(
    value: Any,
    *,
    lowercase: bool = False,
    casefold: bool = False,
    strip: bool = True,
    collapse_whitespace: bool = True,
    unicode_form: Optional[str] = "NFKC",
    normalize_quote_chars: bool = False,
    normalize_dash_chars: bool = False,
    remove_control_chars: bool = True,
) -> str:
    text = ensure_text(value)
    if remove_control_chars:
        text = strip_control_chars(text)
    if unicode_form:
        text = normalize_unicode(text, form=unicode_form)
    if normalize_quote_chars:
        text = normalize_quotes(text)
    if normalize_dash_chars:
        text = normalize_dashes(text)
    if collapse_whitespace:
        text = normalize_whitespace(text)
    elif strip:
        text = text.strip()
    if casefold:
        text = text.casefold()
    elif lowercase:
        text = text.lower()
    return text


def normalize_text_util(text: str, lowercase: bool = True, normalization_form: Optional[str] = "NFKC") -> str:
    """Compatibility helper used by the tokenizer."""

    return normalize_text(text, lowercase=lowercase, unicode_form=normalization_form)


def normalize_for_comparison(value: Any, *, keep_case: bool = False) -> str:
    return normalize_text(
        value,
        lowercase=not keep_case,
        casefold=not keep_case,
        normalize_quote_chars=True,
        normalize_dash_chars=True,
        collapse_whitespace=True,
    )


def compact_text(value: Any, *, max_length: int = DEFAULT_MAX_STRING_LENGTH) -> str:
    return truncate_text(normalize_whitespace(value), max_length=max_length)


def truncate_text(value: Any, max_length: Optional[int] = DEFAULT_MAX_STRING_LENGTH, *, suffix: str = "...") -> str:
    text = ensure_text(value)
    if max_length is None or max_length < 0 or len(text) <= max_length:
        return text
    suffix = suffix or ""
    return text[: max(0, int(max_length) - len(suffix))] + suffix


def split_camel_case(value: Any) -> str:
    text = ensure_text(value)
    text = _CAMEL_RE_1.sub(r"\1 \2", text)
    return _CAMEL_RE_2.sub(r"\1 \2", text)


def safe_filename(value: Any, *, default: str = "language-artifact", max_length: int = 120) -> str:
    text = normalize_for_comparison(split_camel_case(value), keep_case=False)
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip(".-_")
    if max_length > 0:
        text = text[:max_length].strip(".-_")
    return text or default

# ---------------------------------------------------------------------------
# Serialization, redaction, and logging safety
# ---------------------------------------------------------------------------
def safe_repr(value: Any, *, max_length: int = DEFAULT_MAX_STRING_LENGTH) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    return truncate_text(text, max_length)


def json_safe(
    value: Any,
    *,
    max_depth: int = DEFAULT_MAX_SERIALIZATION_DEPTH,
    max_items: int = DEFAULT_MAX_COLLECTION_ITEMS,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
    _depth: int = 0,
) -> Any:
    if _depth >= max_depth:
        return safe_repr(value, max_length=max_string_length)
    if value is None or isinstance(value, (bool, int, str)):
        return truncate_text(value, max_string_length) if isinstance(value, str) else value
    if isinstance(value, float):
        return str(value) if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, bytes):
        chunk = value[:max_string_length]
        try:
            return chunk.decode("utf-8")
        except UnicodeDecodeError:
            return {"encoding": "base64", "length": len(value), "truncated": len(value) > len(chunk), "data": base64.b64encode(chunk).decode("ascii")}
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, LinguisticFrame):
        return frame_to_dict(value)
    if isinstance(value, LanguageIssue):
        return value.to_dict()
    if isinstance(value, LanguageError):
        return value.to_dict()
    if isinstance(value, LanguageDiagnostics):
        return value.to_list()
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value), max_depth=max_depth, max_items=max_items, max_string_length=max_string_length, _depth=_depth + 1)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:max_items]:
            safe_key = str(json_safe(key, max_depth=max_depth, max_items=max_items, max_string_length=max_string_length, _depth=_depth + 1))
            result[safe_key] = json_safe(item, max_depth=max_depth, max_items=max_items, max_string_length=max_string_length, _depth=_depth + 1)
        if len(items) > max_items:
            result["__truncated__"] = True
            result["__remaining_items__"] = len(items) - max_items
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)
        payload = [json_safe(item, max_depth=max_depth, max_items=max_items, max_string_length=max_string_length, _depth=_depth + 1) for item in seq[:max_items]]
        if len(seq) > max_items:
            payload.append({"__truncated__": True, "__remaining_items__": len(seq) - max_items})
        return payload
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": truncate_text(str(value), max_string_length)}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return json_safe(to_dict(), max_depth=max_depth, max_items=max_items, max_string_length=max_string_length, _depth=_depth + 1)
    if hasattr(value, "__dict__"):
        return json_safe(vars(value), max_depth=max_depth, max_items=max_items, max_string_length=max_string_length, _depth=_depth + 1)
    return safe_repr(value, max_length=max_string_length)


def stable_json_dumps(value: Any, *, sort_keys: bool = True, indent: Optional[int] = None) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=sort_keys, indent=indent, separators=(",", ":") if indent is None else None)


def json_dumps(value: Any, *, sort_keys: bool = False, indent: Optional[int] = None, redact: bool = False) -> str:
    payload = json_safe(value)
    if redact:
        payload = redact_data(payload)
    return json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys, indent=indent)


def json_loads(value: Union[str, bytes, bytearray], *, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8", errors="replace")
        return json.loads(str(value))
    except Exception:
        return default


def is_sensitive_key(key: Any, *, sensitive_keys: Optional[Iterable[str]] = None) -> bool:
    key_text = str(key or "").strip().lower().replace("-", "_")
    active = set(SENSITIVE_KEYS)
    active.update(str(item).strip().lower().replace("-", "_") for item in (sensitive_keys or ()) if str(item).strip())
    return key_text in active or any(token in key_text for token in ("password", "secret", "token", "credential", "private_key", "cookie"))


def redact_text(value: Any, *, replacement: str = DEFAULT_REDACTION) -> str:
    redacted = ensure_text(value)
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}{replacement}", redacted)
    return redacted


def redact_sensitive_value(value: Any, *, replacement: str = DEFAULT_REDACTION, preserve_length: bool = False) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): redact_sensitive_value(item, replacement=replacement, preserve_length=preserve_length) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_sensitive_value(item, replacement=replacement, preserve_length=preserve_length) for item in value]
    if isinstance(value, bytes):
        return f"{replacement}[bytes:{len(value)}]" if preserve_length else replacement
    if preserve_length:
        return f"{replacement}[len:{len(str(value))}]"
    return replacement


def redact_data(value: Any, *, replacement: str = DEFAULT_REDACTION, sensitive_keys: Optional[Iterable[str]] = None, _key: Any = None) -> Any:
    if is_sensitive_key(_key, sensitive_keys=sensitive_keys):
        return redact_sensitive_value(value, replacement=replacement)
    if isinstance(value, Mapping):
        return {str(key): redact_data(item, replacement=replacement, sensitive_keys=sensitive_keys, _key=key) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_data(item, replacement=replacement, sensitive_keys=sensitive_keys, _key=_key) for item in value]
    if isinstance(value, str):
        return redact_text(value, replacement=replacement)
    return value


def sanitize_for_logging(value: Any, *, redact: bool = True) -> Any:
    payload = json_safe(value)
    return redact_data(payload) if redact else payload


def log_payload(label: str, payload: Any, *, level: int = 20, redact: bool = True) -> None:
    logger.log(level, "%s | %s", label, stable_json_dumps(sanitize_for_logging(payload, redact=redact)))

# ---------------------------------------------------------------------------
# Mapping and collection utilities
# ---------------------------------------------------------------------------
def merge_mappings(*mappings: Optional[Mapping[str, Any]], deep: bool = True) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for mapping in mappings:
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise PipelineContractError("merge_mappings expects mappings only", expected="Mapping", received=type(mapping).__name__)
        for key, value in mapping.items():
            key_text = str(key)
            if deep and key_text in merged and isinstance(merged[key_text], Mapping) and isinstance(value, Mapping):
                merged[key_text] = merge_mappings(cast(Mapping[str, Any], merged[key_text]), value, deep=True)
            else:
                merged[key_text] = value
    return merged


def prune_none(value: Any, *, drop_empty: bool = False) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned = prune_none(item, drop_empty=drop_empty)
            if drop_empty and cleaned in ("", [], {}, ()):  # keep False/0
                continue
            result[key] = cleaned
        return result
    if isinstance(value, list):
        return [prune_none(item, drop_empty=drop_empty) for item in value if item is not None]
    if isinstance(value, tuple):
        return tuple(prune_none(item, drop_empty=drop_empty) for item in value if item is not None)
    return value


def flatten_mapping(data: Mapping[str, Any], *, parent_key: str = "", separator: str = ".", max_depth: int = 10) -> Dict[str, Any]:
    if max_depth < 0:
        raise PipelineContractError("max_depth must be >= 0", expected=">= 0", received=max_depth)
    flattened: Dict[str, Any] = {}

    def _walk(prefix: str, item: Any, depth: int) -> None:
        if depth > max_depth or not isinstance(item, Mapping):
            flattened[prefix] = item
            return
        for child_key, child_value in item.items():
            next_key = f"{prefix}{separator}{child_key}" if prefix else str(child_key)
            _walk(next_key, child_value, depth + 1)

    _walk(parent_key, data, 0)
    return flattened


def unflatten_mapping(flat_mapping: Mapping[str, Any], *, separator: str = ".") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in flat_mapping.items():
        parts = str(key).split(separator)
        current: MutableMapping[str, Any] = result
        for part in parts[:-1]:
            existing = current.setdefault(part, {})
            if not isinstance(existing, MutableMapping):
                raise PipelineContractError("Cannot unflatten mapping with conflicting scalar path", details={"key": key, "part": part})
            current = existing
        current[parts[-1]] = value
    return result

# ---------------------------------------------------------------------------
# Span and alignment helpers
# ---------------------------------------------------------------------------
def ensure_span(value: Union[TextSpan, Span, Sequence[int]], *, field_name: str = "span", allow_none: bool = False) -> Span:
    if value is None:  # type: ignore[comparison-overlap]
        if allow_none:
            return cast(Span, None)
        raise PipelineContractError(f"{field_name} must not be None.", expected="(start, end)", received=None)
    if isinstance(value, TextSpan):
        return value.as_tuple()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 2:
        raise PipelineContractError(f"{field_name} must be a two-item span.", expected="(start, end)", received=value)
    start, end = int(value[0]), int(value[1])
    if start < 0 or end < start:
        raise PipelineContractError(f"{field_name} must satisfy 0 <= start <= end.", expected="valid half-open span", received=(start, end))
    return (start, end)


def clamp_span(span: Union[TextSpan, Span], text_or_length: Union[str, int]) -> Span:
    start, end = ensure_span(span)
    length = len(text_or_length) if isinstance(text_or_length, str) else int(text_or_length)
    return (max(0, min(start, length)), max(0, min(end, length)))


def span_length(span: Union[TextSpan, Span]) -> int:
    start, end = ensure_span(span)
    return end - start


def extract_span(text: Any, span: Union[TextSpan, Span], *, clamp_to_text: bool = True) -> str:
    source = ensure_text(text)
    start, end = clamp_span(span, source) if clamp_to_text else ensure_span(span)
    return source[start:end]


def spans_overlap(a: Union[TextSpan, Span], b: Union[TextSpan, Span]) -> bool:
    a_start, a_end = ensure_span(a)
    b_start, b_end = ensure_span(b)
    return a_start < b_end and b_start < a_end


def merge_spans(spans: Iterable[Union[TextSpan, Span]], *, merge_touching: bool = True) -> List[Span]:
    ordered = sorted((ensure_span(span) for span in spans), key=lambda item: (item[0], item[1]))
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start < prev_end or (merge_touching and start == prev_end):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def shift_span(span: Union[TextSpan, Span], delta: int) -> Span:
    start, end = ensure_span(span)
    return (max(0, start + int(delta)), max(0, end + int(delta)))


def apply_span_edits(text: str, edits: Iterable[SpanEdit], *, language: str = DEFAULT_LANGUAGE, locale: str = DEFAULT_LOCALE) -> NormalizedText:
    source = ensure_text(text)
    ordered = sorted(list(edits), key=lambda edit: edit.source_span[0])
    pieces: List[str] = []
    cursor = 0
    issues: List[LanguageIssue] = []
    normalized_to_original: Dict[int, int] = {}

    for edit in ordered:
        start, end = ensure_span(edit.source_span, field_name="edit.source_span")
        if start < cursor:
            issue = OrthographyIssue(
                LanguageErrorCode.ORTH_SPAN_MAPPING_MISSING,
                "Overlapping normalization edits were provided.",
                severity=Severity.ERROR,
                source_span=(start, end),
                details={"previous_cursor": cursor, "edit": edit.to_dict()},
            )
            issues.append(issue)
            continue
        unchanged = source[cursor:start]
        for offset, _ in enumerate(unchanged):
            normalized_to_original[len("".join(pieces)) + offset] = cursor + offset
        pieces.append(unchanged)
        replacement_start = len("".join(pieces))
        pieces.append(edit.replacement)
        for offset, _ in enumerate(edit.replacement):
            normalized_to_original[replacement_start + offset] = start
        cursor = end

    tail = source[cursor:]
    tail_start = len("".join(pieces))
    for offset, _ in enumerate(tail):
        normalized_to_original[tail_start + offset] = cursor + offset
    pieces.append(tail)
    normalized = "".join(pieces)
    original_to_normalized = build_offset_map(source, normalized)
    return NormalizedText(
        original=source,
        normalized=normalized,
        edits=tuple(ordered),
        original_to_normalized=original_to_normalized,
        normalized_to_original=normalized_to_original,
        language=language,
        locale=locale,
        issues=tuple(issues),
    )


def build_offset_map(original: str, normalized: str) -> Dict[int, int]:
    """Build a best-effort monotonic character offset map between two strings."""

    mapping: Dict[int, int] = {}
    j = 0
    normalized_folded = normalized.casefold()
    for i, char in enumerate(original):
        target = char.casefold()
        while j < len(normalized_folded) and normalized_folded[j] != target:
            j += 1
        if j < len(normalized_folded):
            mapping[i] = j
            j += 1
    return mapping


def find_text_spans(text: str, pattern: Union[str, re.Pattern[str]], *, flags: int = 0, label: Optional[str] = None) -> List[TextSpan]:
    regex = re.compile(pattern, flags) if isinstance(pattern, str) else pattern
    return [TextSpan(match.start(), match.end(), label=label, text=match.group(0)) for match in regex.finditer(text)]

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
def get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def token_text(token: TokenLike) -> str:
    return ensure_text(first_non_none(get_attr_or_key(token, "text"), get_attr_or_key(token, "orth"), get_attr_or_key(token, "word"), default=""))


def token_index(token: TokenLike, *, default: int = -1) -> int:
    return coerce_int(first_non_none(get_attr_or_key(token, "index"), get_attr_or_key(token, "id"), default=default), default=default)


def token_lemma(token: TokenLike) -> Optional[str]:
    value = first_non_none(get_attr_or_key(token, "lemma"), get_attr_or_key(token, "lemma_"), default=None)
    return ensure_text(value) if value is not None else None


def token_pos(token: TokenLike) -> Optional[str]:
    value = first_non_none(get_attr_or_key(token, "upos"), get_attr_or_key(token, "pos"), get_attr_or_key(token, "pos_"), get_attr_or_key(token, "tag"), get_attr_or_key(token, "tag_"), default=None)
    return normalize_pos_tag(value) if value is not None else None


def token_dep(token: TokenLike) -> Optional[str]:
    value = first_non_none(get_attr_or_key(token, "dep"), get_attr_or_key(token, "dep_"), default=None)
    return ensure_text(value) if value is not None else None


def token_head(token: TokenLike) -> Optional[int]:
    value = first_non_none(get_attr_or_key(token, "head"), get_attr_or_key(token, "head_index"), default=None)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return coerce_int(value, default=-1)


def token_span(token: TokenLike) -> Optional[Span]:
    span = get_attr_or_key(token, "span", None)
    if span is not None:
        return ensure_span(span, allow_none=True)
    start = first_non_none(get_attr_or_key(token, "start_char"), get_attr_or_key(token, "start_char_abs"), get_attr_or_key(token, "idx"), default=None)
    end = first_non_none(get_attr_or_key(token, "end_char"), get_attr_or_key(token, "end_char_abs"), default=None)
    if start is None:
        return None
    start_i = int(start)
    if end is None:
        end_i = start_i + len(token_text(token))
    else:
        end_i = int(end)
        # GrammarProcessor currently documents end_char_abs as inclusive.
        if end_i >= start_i and end_i - start_i + 1 == len(token_text(token)):
            end_i += 1
    return ensure_span((start_i, end_i), allow_none=True)


def token_to_snapshot(token: TokenLike, *, sentence_index: Optional[int] = None, index: Optional[int] = None) -> TokenSnapshot:
    text = token_text(token)
    span = token_span(token)
    pos_raw = first_non_none(get_attr_or_key(token, "pos"), get_attr_or_key(token, "tag"), get_attr_or_key(token, "upos"), default=None)
    upos = normalize_pos_tag(pos_raw) if pos_raw is not None else None
    return TokenSnapshot(
        text=text,
        index=token_index(token, default=index if index is not None else -1),
        lemma=token_lemma(token),
        pos=ensure_text(pos_raw) if pos_raw is not None else None,
        upos=upos,
        dep=token_dep(token),
        head=token_head(token),
        start_char=span[0] if span else None,
        end_char=span[1] if span else None,
        sentence_index=sentence_index if sentence_index is not None else get_attr_or_key(token, "sentence_index", None),
        is_stop=bool(get_attr_or_key(token, "is_stop", False)),
        is_punct=bool(get_attr_or_key(token, "is_punct", False)) or is_punctuation(text),
        morphology=dict(get_attr_or_key(token, "morphology", {}) or {}),
        confidence=get_attr_or_key(token, "confidence", None),
        metadata={"source_type": type(token).__name__},
    )


def tokens_to_snapshots(tokens: Iterable[TokenLike], *, sentence_index: Optional[int] = None) -> Tuple[TokenSnapshot, ...]:
    return tuple(token_to_snapshot(token, sentence_index=sentence_index, index=i) for i, token in enumerate(tokens))


def tokens_to_text(tokens: Iterable[TokenLike], *, normalize_spacing: bool = True) -> str:
    pieces: List[str] = []
    quote_open = False
    for token in tokens:
        text = token_text(token)
        if not text:
            continue
        if not pieces:
            pieces.append(text)
            continue
        prev = pieces[-1]
        if text in PUNCT_NO_SPACE_BEFORE:
            pieces[-1] = prev + text
        elif prev and prev[-1] in PUNCT_NO_SPACE_AFTER:
            pieces[-1] = prev + text
        elif text in OPENING_QUOTES and not quote_open:
            pieces.append(text)
            quote_open = True
        elif text in CLOSING_QUOTES and quote_open:
            pieces[-1] = prev + text
            quote_open = False
        else:
            pieces.append(" " + text)
    result = "".join(pieces)
    return normalize_spacing_around_punctuation(result) if normalize_spacing else result


def infer_token_offsets(text: str, tokens: Iterable[TokenLike], *, case_sensitive: bool = False) -> List[TokenSnapshot]:
    source = ensure_text(text)
    search_source = source if case_sensitive else source.casefold()
    snapshots: List[TokenSnapshot] = []
    cursor = 0
    for i, token in enumerate(tokens):
        snap = token_to_snapshot(token, index=i)
        token_value = snap.text if case_sensitive else snap.text.casefold()
        found = search_source.find(token_value, cursor)
        if found < 0:
            snapshots.append(snap)
            continue
        end = found + len(snap.text)
        snapshots.append(TokenSnapshot(**{**snap.to_dict(), "start_char": found, "end_char": end}))
        cursor = end
    return snapshots


def sentence_from_tokens(tokens: Iterable[TokenLike], *, index: int = 0, sentence_type: Optional[str] = None) -> SentenceSnapshot:
    snapshots = tokens_to_snapshots(tokens, sentence_index=index)
    text = tokens_to_text(snapshots)
    spans = [token.span for token in snapshots if token.span]
    merged_span = (min(span[0] for span in spans), max(span[1] for span in spans)) if spans else None
    return SentenceSnapshot(
        text=text,
        index=index,
        tokens=snapshots,
        span=merged_span,
        sentence_type=sentence_type or classify_sentence_type(text, snapshots),
    )

# ---------------------------------------------------------------------------
# Linguistic classification and lexical helpers
# ---------------------------------------------------------------------------
def is_punctuation(value: Any) -> bool:
    text = ensure_text(value)
    return bool(text) and all(unicodedata.category(char).startswith("P") for char in text)


def is_word_like(value: Any) -> bool:
    text = ensure_text(value)
    return bool(_ASCII_WORD_RE.match(text))


def is_numeric_token(value: Any) -> bool:
    return bool(_NUMBER_RE.match(ensure_text(value).strip()))


def word_shape(value: Any) -> str:
    result: List[str] = []
    for char in ensure_text(value):
        if char.isupper():
            marker = "X"
        elif char.islower():
            marker = "x"
        elif char.isdigit():
            marker = "d"
        elif char.isspace():
            marker = " "
        else:
            marker = char
        if not result or result[-1] != marker:
            result.append(marker)
    return "".join(result)


def normalize_pos_tag(tag: Any, *, default: Optional[str] = None) -> Optional[str]:
    if tag is None:
        return default
    text = ensure_text(tag).strip()
    if not text:
        return default
    upper = text.upper()
    if upper in UPOS_TAGS:
        return upper
    return PENN_TO_UPOS.get(upper, default or upper)


def is_content_pos(tag: Any) -> bool:
    return normalize_pos_tag(tag) in {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}


def is_function_pos(tag: Any) -> bool:
    return normalize_pos_tag(tag) in {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}


def classify_sentence_type(text: Any, tokens: Optional[Iterable[TokenLike]] = None) -> str:
    normalized = normalize_whitespace(text)
    if not normalized:
        return "empty"
    first_token = None
    if tokens is not None:
        token_list = list(tokens)
        first_token = token_list[0] if token_list else None
    first_word = token_text(first_token).lower() if first_token is not None else normalized.split(" ", 1)[0].lower()
    if normalized.endswith("?") or first_word in {"who", "what", "where", "when", "why", "how", "do", "does", "did", "is", "are", "was", "were", "can", "could", "would", "should", "will"}:
        return "interrogative"
    if normalized.endswith("!"):
        return "exclamative"
    if first_word in {"please", "let", "make", "tell", "show", "give", "send", "create", "write", "explain"}:
        return "imperative"
    if _SENTENCE_END_RE.search(normalized):
        return "declarative"
    return "fragment"


def choose_indefinite_article(phrase: Any) -> str:
    text = normalize_whitespace(phrase)
    if not text:
        return "a"
    first = re.sub(r"^[^A-Za-z0-9]+", "", text).split(" ", 1)[0]
    if not first:
        return "a"
    stripped = first.replace(".", "")
    if _ACRONYM_RE.match(first) and stripped[:1].upper() in ACRONYM_AN_INITIALS:
        return "an"
    lowered = first.lower()
    if lowered.startswith(SILENT_H_PREFIXES):
        return "an"
    if lowered.startswith(LONG_U_PREFIXES):
        return "a"
    return "an" if lowered[0] in VOWELS else "a"


def with_indefinite_article(phrase: Any) -> str:
    text = normalize_whitespace(phrase)
    return f"{choose_indefinite_article(text)} {text}".strip()


def simple_word_tokenize(text: Any) -> List[str]:
    return _WORD_RE.findall(ensure_text(text))


def ngrams(sequence: Sequence[T], n: int) -> List[Tuple[T, ...]]:
    if n <= 0:
        raise PipelineContractError("n must be > 0", expected="> 0", received=n)
    return [tuple(sequence[index:index + n]) for index in range(0, max(0, len(sequence) - n + 1))]


def char_ngrams(text: Any, n: int = 3, *, pad: bool = False) -> List[str]:
    value = normalize_for_comparison(text)
    if pad:
        value = f"{' ' * (n - 1)}{value}{' ' * (n - 1)}"
    return ["".join(chars) for chars in ngrams(list(value), n)]


def token_ngrams(text: Any, n: int = 2) -> List[Tuple[str, ...]]:
    return ngrams(simple_word_tokenize(normalize_for_comparison(text)), n)


def jaccard_similarity(a: Iterable[Any], b: Iterable[Any]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def lexical_overlap(text_a: Any, text_b: Any) -> float:
    return jaccard_similarity(simple_word_tokenize(normalize_for_comparison(text_a)), simple_word_tokenize(normalize_for_comparison(text_b)))


def cosine_similarity_from_counts(a: Mapping[Any, Union[int, float]], b: Mapping[Any, Union[int, float]]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(float(a.get(key, 0.0)) * float(b.get(key, 0.0)) for key in keys)
    mag_a = math.sqrt(sum(float(value) ** 2 for value in a.values()))
    mag_b = math.sqrt(sum(float(value) ** 2 for value in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def bag_of_words(text: Any, *, lowercase: bool = True) -> Counter[str]:
    source = normalize_for_comparison(text) if lowercase else normalize_whitespace(text)
    return Counter(simple_word_tokenize(source))


def edit_distance(a: Any, b: Any, *, substitution_cost: int = 1) -> int:
    left = ensure_text(a)
    right = ensure_text(b)
    if left == right:
        return 0
    rows, cols = len(left) + 1, len(right) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else substitution_cost
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def normalized_edit_similarity(a: Any, b: Any) -> float:
    left = ensure_text(a)
    right = ensure_text(b)
    maximum = max(len(left), len(right))
    if maximum == 0:
        return 1.0
    return 1.0 - (edit_distance(left, right) / maximum)

# ---------------------------------------------------------------------------
# LinguisticFrame and semantic helpers
# ---------------------------------------------------------------------------
def normalize_speech_act(value: Any, *, default: SpeechActType = SpeechActType.ASSERTIVE) -> SpeechActType:
    if isinstance(value, SpeechActType):
        return value
    text = normalize_for_comparison(value)
    for member in SpeechActType:
        if text in {member.name.lower(), member.value.lower()}:
            return member
    aliases = {
        "representative": SpeechActType.ASSERTIVE,
        "representatives": SpeechActType.ASSERTIVE,
        "assertive": SpeechActType.ASSERTIVE,
        "request": SpeechActType.DIRECTIVE,
        "directive": SpeechActType.DIRECTIVE,
        "promise": SpeechActType.COMMISSIVE,
        "commissive": SpeechActType.COMMISSIVE,
        "apology": SpeechActType.EXPRESSIVE,
        "expressive": SpeechActType.EXPRESSIVE,
        "declare": SpeechActType.DECLARATION,
        "declaration": SpeechActType.DECLARATION,
    }
    return aliases.get(text, default)


def make_linguistic_frame(
    *,
    intent: Any = "unknown",
    entities: Optional[Mapping[str, Any]] = None,
    sentiment: Any = 0.0,
    modality: Any = "neutral",
    confidence: Any = 0.0,
    act_type: Any = SpeechActType.ASSERTIVE,
    propositional_content: Optional[str] = None,
    illocutionary_force: Optional[str] = None,
    perlocutionary_effect: Optional[str] = None,
) -> LinguisticFrame:
    return LinguisticFrame(
        intent=normalize_identifier_component(intent, default="unknown", lowercase=True),
        entities=dict(entities or {}),
        sentiment=coerce_float(sentiment, default=0.0, minimum=-1.0, maximum=1.0),
        modality=normalize_whitespace(modality) or "neutral",
        confidence=coerce_probability(confidence),
        act_type=normalize_speech_act(act_type),
        propositional_content=propositional_content,
        illocutionary_force=illocutionary_force,
        perlocutionary_effect=perlocutionary_effect,
    )


def frame_to_dict(frame: Optional[LinguisticFrame], *, redact: bool = False) -> Optional[JsonDict]:
    if frame is None:
        return None
    payload: JsonDict = {
        "intent": frame.intent,
        "entities": json_safe(frame.entities),
        "sentiment": frame.sentiment,
        "modality": frame.modality,
        "confidence": frame.confidence,
        "act_type": frame.act_type.value if isinstance(frame.act_type, SpeechActType) else frame.act_type,
        "propositional_content": frame.propositional_content,
        "illocutionary_force": frame.illocutionary_force,
        "perlocutionary_effect": frame.perlocutionary_effect,
    }
    return cast(JsonDict, redact_data(payload)) if redact else payload


def frame_from_mapping(data: Mapping[str, Any]) -> LinguisticFrame:
    return make_linguistic_frame(
        intent=data.get("intent", "unknown"),
        entities=data.get("entities", {}) if isinstance(data.get("entities", {}), Mapping) else {},
        sentiment=data.get("sentiment", 0.0),
        modality=data.get("modality", "neutral"),
        confidence=data.get("confidence", 0.0),
        act_type=data.get("act_type", SpeechActType.ASSERTIVE),
        propositional_content=data.get("propositional_content"),
        illocutionary_force=data.get("illocutionary_force"),
        perlocutionary_effect=data.get("perlocutionary_effect"),
    )


def validate_linguistic_frame(frame: Any, *, raise_on_error: bool = False) -> LanguageDiagnostics:
    diagnostics = LanguageDiagnostics()
    if not isinstance(frame, LinguisticFrame):
        issue = NLUIssue(LanguageErrorCode.NLU_FRAME_VALIDATION_FAILED, "Expected a LinguisticFrame instance.", severity=Severity.ERROR, details={"received": type(frame).__name__})
        diagnostics.add(issue)
        if raise_on_error:
            raise NLUError(issue)
        return diagnostics
    if not normalize_whitespace(frame.intent):
        diagnostics.add(NLUIssue(LanguageErrorCode.NLU_FRAME_VALIDATION_FAILED, "Frame intent is empty.", severity=Severity.ERROR, frame=frame))
    if not isinstance(frame.entities, Mapping):
        diagnostics.add(NLUIssue(LanguageErrorCode.NLU_FRAME_VALIDATION_FAILED, "Frame entities must be a mapping.", severity=Severity.ERROR, frame=frame))
    if not -1.0 <= float(frame.sentiment) <= 1.0:
        diagnostics.add(NLUIssue(LanguageErrorCode.NLU_FRAME_VALIDATION_FAILED, "Frame sentiment must be in [-1, 1].", severity=Severity.ERROR, frame=frame))
    if not 0.0 <= float(frame.confidence) <= 1.0:
        diagnostics.add(NLUIssue(LanguageErrorCode.NLU_FRAME_VALIDATION_FAILED, "Frame confidence must be in [0, 1].", severity=Severity.ERROR, frame=frame))
    if raise_on_error:
        diagnostics.raise_if_blocking("Invalid LinguisticFrame")
    return diagnostics


def merge_linguistic_frames(primary: Optional[LinguisticFrame], secondary: Optional[LinguisticFrame]) -> Optional[LinguisticFrame]:
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    entities = merge_mappings(secondary.entities, primary.entities)
    return make_linguistic_frame(
        intent=primary.intent or secondary.intent,
        entities=entities,
        sentiment=primary.sentiment if primary.confidence >= secondary.confidence else secondary.sentiment,
        modality=primary.modality or secondary.modality,
        confidence=max(primary.confidence, secondary.confidence),
        act_type=primary.act_type or secondary.act_type,
        propositional_content=primary.propositional_content or secondary.propositional_content,
        illocutionary_force=primary.illocutionary_force or secondary.illocutionary_force,
        perlocutionary_effect=primary.perlocutionary_effect or secondary.perlocutionary_effect,
    )

# ---------------------------------------------------------------------------
# Template and NLG helpers
# ---------------------------------------------------------------------------
def extract_placeholders(template: Any) -> Tuple[str, ...]:
    text = ensure_text(template)
    formatter_fields = [field_name for _, field_name, _, _ in Formatter().parse(text) if field_name]
    regex_fields = _PLACEHOLDER_RE.findall(text)
    return tuple(dedupe_preserve_order([*formatter_fields, *regex_fields]))


def dotted_lookup(data: Mapping[str, Any], path: str, *, default: Any = None, separator: str = ".") -> Any:
    current: Any = data
    for part in path.split(separator):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


class _FormatValue(dict):
    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        return "{" + key + "}"


def _format_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FormatValue({str(key): _format_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_format_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_format_value(item) for item in value)
    return value


class _SafeFormatMapping(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_template(template: Any, values: Optional[Mapping[str, Any]] = None, *, strict: bool = False, frame: Optional[LinguisticFrame] = None) -> TemplateRenderResult:
    source = ensure_text(template)
    value_map = dict(values or {})
    if frame is not None:
        value_map.setdefault("intent", frame.intent)
        value_map.setdefault("entities", frame.entities)
        value_map.setdefault("modality", frame.modality)
        value_map.setdefault("sentiment", frame.sentiment)
    placeholders = extract_placeholders(source)
    flat_values = flatten_mapping(value_map) if value_map else {}
    missing = tuple(name for name in placeholders if name not in value_map and name not in flat_values)
    unused = tuple(sorted(set(value_map) - {name.split(".", 1)[0] for name in placeholders}))
    issues: List[LanguageIssue] = []
    if missing:
        issue = NLGIssue(
            LanguageErrorCode.NLG_TEMPLATE_PLACEHOLDER_MISSING,
            "Template is missing values for one or more placeholders.",
            severity=Severity.ERROR if strict else Severity.WARNING,
            frame=frame,
            details={"missing_placeholders": missing, "template": source},
        )
        issues.append(issue)
        if strict:
            raise NLGFillingError("Template placeholder values are missing.", template=source, missing_fields=list(missing), frame=frame)
    resolved = _SafeFormatMapping({str(key): _format_value(item) for key, item in value_map.items()})
    for key in placeholders:
        if key not in resolved and key in flat_values:
            resolved[key] = flat_values[key]
    try:
        rendered = source.format_map(resolved)
    except Exception as exc:
        raise NLGFillingError("Template rendering failed.", template=source, entity_data=value_map, frame=frame, original_exception=exc) from exc
    return TemplateRenderResult(text=rendered, template=source, values=value_map, missing_placeholders=missing, unused_values=unused, issues=tuple(issues))


def validate_response_text(text: Any, *, min_length: int = 1, max_length: Optional[int] = None, frame: Optional[LinguisticFrame] = None) -> str:
    value = normalize_whitespace(text)
    errors: List[str] = []
    if len(value) < min_length:
        errors.append(f"response shorter than {min_length} characters")
    if max_length is not None and len(value) > max_length:
        errors.append(f"response longer than {max_length} characters")
    if errors:
        raise NLGValidationError("Generated response failed validation.", response_text=value, validation_errors=errors, frame=frame)
    return value

# ---------------------------------------------------------------------------
# Diagnostics and result helpers
# ---------------------------------------------------------------------------
def make_language_issue(
    code: Union[str, LanguageErrorCode],
    message: str,
    *,
    stage: Union[str, LanguageStage] = LanguageStage.UNKNOWN,
    severity: Union[str, Severity] = Severity.WARNING,
    category: Union[str, ErrorCategory] = ErrorCategory.UNKNOWN,
    frame: Optional[LinguisticFrame] = None,
    **kwargs: Any,
) -> LanguageIssue:
    return LanguageIssue(code=code, message=message, stage=stage, severity=severity, category=category, frame=frame, **kwargs)


def issue_from_exception(
    exc: BaseException,
    *,
    code: Union[str, LanguageErrorCode] = LanguageErrorCode.UNKNOWN,
    stage: Union[str, LanguageStage] = LanguageStage.UNKNOWN,
    category: Union[str, ErrorCategory] = ErrorCategory.UNKNOWN,
    module: Optional[str] = None,
    frame: Optional[LinguisticFrame] = None,
    severity: Union[str, Severity] = Severity.ERROR,
    include_traceback: bool = False,
) -> LanguageIssue:
    details: Dict[str, Any] = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
    if include_traceback:
        import traceback
        details["traceback"] = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return LanguageIssue(code=code, message=str(exc), stage=stage, category=category, severity=severity, module=module, frame=frame, details=details)


def diagnostics_from_issues(*issues: Union[LanguageIssue, LanguageError, Iterable[Union[LanguageIssue, LanguageError]]]) -> LanguageDiagnostics:
    diagnostics = LanguageDiagnostics()
    for item in issues:
        if isinstance(item, (LanguageIssue, LanguageError)):
            diagnostics.add(item)
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes, Mapping)):
            diagnostics.extend(item)
        else:
            raise PipelineContractError("diagnostics_from_issues received an unsupported item", expected="LanguageIssue or iterable", received=type(item).__name__)
    return diagnostics


def success_result(data: Any = None, *, frame: Optional[LinguisticFrame] = None, metadata: Optional[Mapping[str, Any]] = None) -> LanguageResult[Any]:
    return LanguageResult(data=data, frame=frame, metadata=dict(metadata or {}))


def error_result(issue: Union[LanguageIssue, LanguageError], *, data: Any = None, frame: Optional[LinguisticFrame] = None, metadata: Optional[Mapping[str, Any]] = None) -> LanguageResult[Any]:
    issue_obj = issue.issue if isinstance(issue, LanguageError) else issue
    return LanguageResult(data=data, frame=frame or issue_obj.frame, issues=[issue_obj], metadata=dict(metadata or {}))


def normalize_result(result: Any, *, frame: Optional[LinguisticFrame] = None) -> LanguageResult[Any]:
    if isinstance(result, LanguageResult):
        return result
    if isinstance(result, LanguageError):
        return error_result(result, frame=frame)
    if isinstance(result, LanguageIssue):
        return error_result(result, frame=frame)
    if isinstance(result, Mapping) and "issues" in result:
        issues = [item if isinstance(item, LanguageIssue) else make_language_issue(LanguageErrorCode.UNKNOWN, ensure_text(item)) for item in result.get("issues", [])]
        return LanguageResult(data=result.get("data"), issues=issues, metadata=dict(result.get("metadata", {}) or {}), frame=frame)
    return success_result(result, frame=frame)


def validate_pipeline_payload(payload: Any, *, require_tokens: bool = False, require_frame: bool = False) -> LanguagePipelinePayload:
    if isinstance(payload, LanguagePipelinePayload):
        if require_tokens and not payload.tokens:
            raise PipelineContractError("Pipeline payload is missing tokens.", expected="tokens", received="empty")
        if require_frame and payload.frame is None:
            raise PipelineContractError("Pipeline payload is missing LinguisticFrame.", expected="frame", received=None)
        return payload
    if isinstance(payload, Mapping):
        original_text = ensure_text(payload.get("original_text", payload.get("text", "")))
        if not original_text:
            raise PipelineContractError("Pipeline payload requires original_text or text.", expected="text", received=payload)
        tokens = tuple(token_to_snapshot(token) for token in payload.get("tokens", []) or [])
        frame_value = payload.get("frame")
        frame = frame_value if isinstance(frame_value, LinguisticFrame) else frame_from_mapping(frame_value) if isinstance(frame_value, Mapping) else None
        normalized_text = payload.get("normalized_text")
        pipeline = LanguagePipelinePayload(original_text=original_text, normalized_text=ensure_text(normalized_text) if normalized_text is not None else None, tokens=tokens, frame=frame, metadata=dict(payload.get("metadata", {}) or {}))
        return validate_pipeline_payload(pipeline, require_tokens=require_tokens, require_frame=require_frame)
    raise PipelineContractError("Unsupported pipeline payload type.", expected="LanguagePipelinePayload or mapping", received=type(payload).__name__)

# ---------------------------------------------------------------------------
# Entity and intent helpers
# ---------------------------------------------------------------------------
def normalize_entity_label(label: Any) -> str:
    return normalize_identifier_component(label, default="entity", lowercase=True)


def normalize_entity(entity: Any, *, default_label: str = "entity") -> EntitySnapshot:
    if isinstance(entity, EntitySnapshot):
        return entity
    if isinstance(entity, Mapping):
        text = ensure_text(first_non_none(entity.get("text"), entity.get("value"), default=""))
        label = normalize_entity_label(first_non_none(entity.get("label"), entity.get("type"), default=default_label))
        span = ensure_span(entity["span"], allow_none=True) if entity.get("span") is not None else None
        return EntitySnapshot(
            text=text,
            label=label,
            value=entity.get("value", text),
            span=span,
            confidence=coerce_probability(entity.get("confidence"), default=0.0) if entity.get("confidence") is not None else None,
            normalized=entity.get("normalized"),
            source=entity.get("source"),
            metadata=dict(entity.get("metadata", {}) or {}),
        )
    text = ensure_text(entity)
    return EntitySnapshot(text=text, label=normalize_entity_label(default_label), value=text)


def normalize_entities(entities: Any) -> Tuple[EntitySnapshot, ...]:
    if entities is None:
        return ()
    if isinstance(entities, Mapping):
        return tuple(normalize_entity({"label": key, "value": value, "text": value}) for key, value in entities.items())
    return tuple(normalize_entity(item) for item in ensure_list(entities))


def normalize_intent(intent: Any, *, default: str = "unknown") -> str:
    return normalize_identifier_component(intent, default=default, lowercase=True)


def rank_intents(candidates: Mapping[str, Any], *, threshold: float = 0.0, limit: Optional[int] = None) -> List[Tuple[str, float]]:
    ranked = sorted(((normalize_intent(intent), coerce_probability(score)) for intent, score in candidates.items()), key=lambda item: item[1], reverse=True)
    filtered = [item for item in ranked if item[1] >= threshold]
    return filtered[:limit] if limit else filtered


def is_ambiguous_intent(candidates: Mapping[str, Any], *, margin: float = 0.1) -> bool:
    ranked = rank_intents(candidates, limit=2)
    if len(ranked) < 2:
        return False
    return abs(ranked[0][1] - ranked[1][1]) <= margin

# ---------------------------------------------------------------------------
# File/resource helpers
# ---------------------------------------------------------------------------
def resolve_path(path: Union[str, Path], *, base_dir: Optional[Union[str, Path]] = None, must_exist: bool = False, field_name: str = "path") -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else Path(base_dir or ".") / raw
    if must_exist and not resolved.exists():
        raise ResourceLanguageError(f"{field_name} does not exist: {resolved}", details={"path": str(resolved)})
    return resolved


def read_text_file(path: Union[str, Path], *, encoding: str = "utf-8") -> str:
    resolved = resolve_path(path, must_exist=True)
    return resolved.read_text(encoding=encoding)


def write_text_file(path: Union[str, Path], content: Any, *, encoding: str = "utf-8", create_parent: bool = True) -> Path:
    resolved = Path(path)
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(ensure_text(content), encoding=encoding)
    return resolved


def load_json_file(path: Union[str, Path], *, encoding: str = "utf-8") -> Any:
    return json_loads(read_text_file(path, encoding=encoding))


def save_json_file(path: Union[str, Path], payload: Any, *, pretty: bool = True, encoding: str = "utf-8") -> Path:
    return write_text_file(path, json_dumps(payload, indent=2 if pretty else None), encoding=encoding)
