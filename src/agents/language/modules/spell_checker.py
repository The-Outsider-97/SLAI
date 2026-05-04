"""
Language Spell Checker Module

Core Function:
Provides the language subsystem with a production-ready spelling checker for
orthography correction, typo analysis, candidate generation, confidence scoring,
and text-level correction workflows.

Responsibilities:
- Load lexical resources from the structured language wordlist and config.
- Detect whether words are known while respecting case, locale, and ignore rules.
- Generate spelling candidates through bounded edit distance, typo variants,
  phonetic buckets, keyboard adjacency, and lexical metadata.
- Rank suggestions with transparent confidence components and diagnostics.
- Preserve enough metadata for OrthographyProcessor, GrammarProcessor, NLU, and
  agent-facing correction workflows.
- Use language helpers and language errors rather than duplicating generic
  normalization, path, serialization, and diagnostics primitives.

Why it matters:
Spelling correction is an early language-stage component. A production spell
checker must be deterministic, explainable, configurable, resource-aware, and
safe to expand without turning every downstream module into a correction engine.
"""

from __future__ import annotations

import math
import re
import yaml

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from metaphone import doublemetaphone # pyright: ignore[reportMissingImports]

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.language_error import *
from ..utils.language_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Spell Checker")
printer = PrettyPrinter()

Span = Tuple[int, int]
KeyboardLayout = Dict[str, Set[str]]


@dataclass(frozen=True)
class WordEntry:
    """Normalized lexical entry loaded from the structured wordlist."""

    word: str
    normalized: str
    pos: Tuple[str, ...] = ()
    synonyms: Tuple[str, ...] = ()
    related_terms: Tuple[str, ...] = ()
    frequency: float = 0.0
    source: str = "wordlist"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "word": self.word,
                "normalized": self.normalized,
                "pos": list(self.pos),
                "synonyms": list(self.synonyms),
                "related_terms": list(self.related_terms),
                "frequency": self.frequency,
                "source": self.source,
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )


@dataclass(frozen=True)
class SpellSuggestion:
    """A ranked correction candidate with transparent score components."""

    word: str
    score: float
    rank: int = 0
    edit_distance: Optional[float] = None
    edit_similarity: Optional[float] = None
    phonetic_similarity: Optional[float] = None
    keyboard_similarity: Optional[float] = None
    frequency_score: Optional[float] = None
    prefix_similarity: Optional[float] = None
    shape_similarity: Optional[float] = None
    source: str = "ranked"
    reasons: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> Tuple[str, float]:
        return (self.word, self.score)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "word": self.word,
                "score": self.score,
                "rank": self.rank,
                "edit_distance": self.edit_distance,
                "edit_similarity": self.edit_similarity,
                "phonetic_similarity": self.phonetic_similarity,
                "keyboard_similarity": self.keyboard_similarity,
                "frequency_score": self.frequency_score,
                "prefix_similarity": self.prefix_similarity,
                "shape_similarity": self.shape_similarity,
                "source": self.source,
                "reasons": list(self.reasons),
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )


@dataclass(frozen=True)
class SpellCheckResult:
    """Single-token spelling analysis result."""

    original: str
    normalized: str
    is_correct: bool
    checkable: bool = True
    best: Optional[SpellSuggestion] = None
    suggestions: Tuple[SpellSuggestion, ...] = ()
    corrected: Optional[str] = None
    confidence: float = 0.0
    issues: Tuple[LanguageIssue, ...] = ()
    span: Optional[Span] = None
    language: str = "en"
    locale: str = "en-US"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.corrected is not None and self.corrected != self.original

    @property
    def ok(self) -> bool:
        return self.is_correct or not self.checkable or bool(self.suggestions)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "ok": self.ok,
                "original": self.original,
                "normalized": self.normalized,
                "is_correct": self.is_correct,
                "checkable": self.checkable,
                "best": self.best.to_dict() if self.best else None,
                "suggestions": [suggestion.to_dict() for suggestion in self.suggestions],
                "corrected": self.corrected,
                "confidence": self.confidence,
                "issues": [issue.to_dict() for issue in self.issues],
                "span": list(self.span) if self.span else None,
                "language": self.language,
                "locale": self.locale,
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )


@dataclass(frozen=True)
class TextSpellCheckResult:
    """Text-level spelling result with span-aware corrections."""

    original_text: str
    corrected_text: str
    checks: Tuple[SpellCheckResult, ...]
    issues: Tuple[LanguageIssue, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.original_text != self.corrected_text

    @property
    def corrections(self) -> List[Dict[str, Any]]:
        return [check.to_dict() for check in self.checks if check.changed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "corrected_text": self.corrected_text,
            "changed": self.changed,
            "checks": [check.to_dict() for check in self.checks],
            "corrections": self.corrections,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class SpellCheckerStats:
    """Operational snapshot for the spell checker."""

    language: str
    locale: str
    word_count: int
    entry_count: int
    phonetic_bucket_count: int
    length_bucket_count: int
    check_calls: int
    suggest_calls: int
    text_check_calls: int
    diagnostics_count: int
    resources: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["resources"] = json_safe(self.resources)
        return payload


class SpellChecker:
    """
    Production spelling checker for the language agent subsystem.

    The class intentionally avoids inheriting from the third-party
    ``pyspellchecker.SpellChecker`` because the language subsystem needs
    structured diagnostics, project-native config, wordlist metadata, and
    transparent ranking behavior.
    """

    DEFAULT_WORD_SEED: Tuple[str, ...] = (
        "a", "able", "about", "agent", "analysis", "assistant", "because", "believe", "cache",
        "checker", "context", "correction", "diagnostic", "grammar", "language", "memory", "module",
        "production", "quality", "really", "receive", "rules", "spelling", "suggestion", "system",
        "their", "there", "token", "tokenizer", "transformer", "unfortunately", "word", "world",
    )

    DEFAULT_TYPO_PATTERNS: Tuple[Tuple[str, str, str], ...] = (
        (r"ie$", "ei", "ie-ei-swap"),
        (r"ei$", "ie", "ei-ie-swap"),
        (r"([aeiou])\1+", r"\1", "repeated-vowel-collapse"),
        (r"([a-z])\1{2,}", r"\1\1", "long-repeat-collapse"),
        (r"ph", "f", "phonetic-ph-f"),
        (r"f", "ph", "phonetic-f-ph"),
        (r"able$", "ible", "able-ible-swap"),
        (r"ible$", "able", "ible-able-swap"),
        (r"ance$", "ence", "ance-ence-swap"),
        (r"ence$", "ance", "ence-ance-swap"),
        (r"ary$", "ery", "ary-ery-swap"),
        (r"ery$", "ary", "ery-ary-swap"),
        (r"re$", "er", "re-er-swap"),
        (r"er$", "re", "er-re-swap"),
        (r"([^e])e$", r"\1", "silent-e-drop"),
    )

    def __init__(self) -> None:
        self.config = load_global_config()
        self.checker_config = get_config_section("spell_checker") or {}

        self.language = ensure_text(self.checker_config.get("language", "en"))
        self.locale = ensure_text(self.checker_config.get("locale", "en-US"))
        self.case_sensitive = coerce_bool(self.checker_config.get("case_sensitive", False), default=False)
        self.casefold = coerce_bool(self.checker_config.get("casefold", True), default=True)
        self.preserve_case = coerce_bool(self.checker_config.get("preserve_case", True), default=True)
        self.strict_loading = coerce_bool(self.checker_config.get("strict_loading", False), default=False)
        self.enable_phonetic = coerce_bool(self.checker_config.get("enable_phonetic", True), default=True)
        self.phonetic_algorithm = ensure_text(self.checker_config.get("phonetic_algorithm", "soundex")).lower()
        self.max_suggestions = coerce_int(self.checker_config.get("max_suggestions", 5), default=5, minimum=1)
        self.max_candidates = coerce_int(self.checker_config.get("max_candidates", 2500), default=2500, minimum=1)
        self.max_typo_variants = coerce_int(self.checker_config.get("max_typo_variants", 2500), default=2500, minimum=1)
        self.max_edit_distance = coerce_float(self.checker_config.get("max_edit_distance", 2.0), default=2.0, minimum=0.0)
        self.min_candidate_score = coerce_float(self.checker_config.get("min_candidate_score", 0.0), default=0.0, minimum=0.0, maximum=1.0)
        self.auto_correct_threshold = coerce_float(self.checker_config.get("auto_correct_threshold", 0.86), default=0.86, minimum=0.0, maximum=1.0)
        self.min_word_length = coerce_int(self.checker_config.get("min_word_length", 2), default=2, minimum=1)
        self.max_word_length = coerce_int(self.checker_config.get("max_word_length", 64), default=64, minimum=1)
        self.length_window = coerce_int(self.checker_config.get("candidate_length_window", 3), default=3, minimum=0)
        self.prefix_window = coerce_int(self.checker_config.get("candidate_prefix_length", 1), default=1, minimum=0, maximum=8)
        self.enable_typo_patterns = coerce_bool(self.checker_config.get("enable_typo_patterns", True), default=True)
        self.enable_keyboard = coerce_bool(self.checker_config.get("enable_keyboard", True), default=True)
        self.enable_frequency = coerce_bool(self.checker_config.get("enable_frequency", True), default=True)
        self.enable_compound_check = coerce_bool(self.checker_config.get("enable_compound_check", True), default=True)
        self.enable_hyphenated_check = coerce_bool(self.checker_config.get("enable_hyphenated_check", True), default=True)
        self.include_related_terms = coerce_bool(self.checker_config.get("include_related_terms", False), default=False)
        self.suggestion_strategies = tuple(ensure_text(item).lower() for item in ensure_list(self.checker_config.get("suggestion_strategies", [])))
        if not self.suggestion_strategies:
            self.suggestion_strategies = ("typo_patterns", "edit_distance", "phonetic", "keyboard")

        self.score_weights = self._load_score_weights()
        self.keyboard_layout = self._load_keyboard_layout()
        self.typo_patterns = self._load_typo_patterns()
        self.ignore_words = self._load_ignore_words()
        self.ignore_regexes = self._compile_ignore_patterns()

        self.entries: Dict[str, WordEntry] = {}
        self.words: Set[str] = set()
        self.surface_forms: Dict[str, str] = {}
        self.length_index: Dict[int, Set[str]] = defaultdict(set)
        self.prefix_index: Dict[str, Set[str]] = defaultdict(set)
        self.phonetic_map: Dict[str, Set[str]] = defaultdict(set)
        self.wordlist_metadata: Dict[str, Any] = {}
        self.diagnostics = LanguageDiagnostics()
        self._check_calls = 0
        self._suggest_calls = 0
        self._text_check_calls = 0

        self._load_lexical_resources()
        self._build_indexes()

        printer.status("INIT", f"Spell Checker initialized with {len(self.words)} words", "success")

    # ------------------------------------------------------------------
    # Resource loading and configuration
    # ------------------------------------------------------------------
    def _load_score_weights(self) -> Dict[str, float]:
        raw = ensure_mapping(self.checker_config.get("score_weights", {}), field_name="spell_checker.score_weights", allow_none=True)
        defaults = {
            "edit": 0.38,
            "phonetic": 0.18,
            "keyboard": 0.14,
            "frequency": 0.10,
            "prefix": 0.10,
            "shape": 0.05,
            "length": 0.05,
        }
        defaults.update({ensure_text(k): coerce_float(v, default=0.0, minimum=0.0) for k, v in raw.items()})
        total = sum(defaults.values()) or 1.0
        return {key: value / total for key, value in defaults.items()}

    def _load_keyboard_layout(self) -> KeyboardLayout:
        configured = self.checker_config.get("keyboard_layout")
        if isinstance(configured, Mapping):
            return {
                ensure_text(key).lower(): {ensure_text(value).lower() for value in ensure_list(values)}
                for key, values in configured.items()
            }

        rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        layout: KeyboardLayout = defaultdict(set)
        positions: Dict[str, Tuple[int, int]] = {}
        for row_index, row in enumerate(rows):
            for col_index, char in enumerate(row):
                positions[char] = (row_index, col_index)
        for char, (row, col) in positions.items():
            for other, (other_row, other_col) in positions.items():
                if char == other:
                    continue
                if abs(row - other_row) <= 1 and abs(col - other_col) <= 1:
                    layout[char].add(other)
        return dict(layout)

    def _load_ignore_words(self) -> Set[str]:
        values = ensure_list(self.checker_config.get("ignore_words", []))
        return {self._normalize_word(value) for value in values if ensure_text(value)}

    def _compile_ignore_patterns(self) -> Tuple[re.Pattern[str], ...]:
        patterns = ensure_list(
            self.checker_config.get(
                "ignore_patterns",
                [
                    r"^https?://",
                    r"^www\.",
                    r"^[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}$",
                    r"^[_\W]+$",
                    r"^\d+(?:[.,:/-]\d+)*$",
                    r"^\[[A-Z_]+\]$",
                ],
            )
        )
        return tuple(re.compile(ensure_text(pattern)) for pattern in patterns if ensure_text(pattern))

    def _configured_wordlist_paths(self) -> List[Path]:
        keys = ["structured_wordlist_path", "main_wordlist_path", "wordlist_path"]
        paths: List[Path] = []
        for key in keys:
            value = self.checker_config.get(key, self.config.get(key))
            if value in (None, "", "none", "None"):
                continue
            path = resolve_path(ensure_text(value), field_name=key)
            if path not in paths:
                paths.append(path)
        return paths

    def _load_lexical_resources(self) -> None:
        loaded_any = False
        for path in self._configured_wordlist_paths():
            if not path.exists():
                self._record_resource_issue(path, missing=True)
                continue
            self._load_wordlist_path(path)
            loaded_any = True

        inline_words = ensure_list(self.checker_config.get("additional_words", []))
        for word in inline_words:
            self._add_word_entry(ensure_text(word), source="config.additional_words")

        if not loaded_any and not self.entries:
            for word in self.DEFAULT_WORD_SEED:
                self._add_word_entry(word, source="fallback_seed")
            self._add_issue(
                ResourceIssue(
                    code=LanguageErrorCode.RESOURCE_MISSING,
                    message="No configured spell-checking wordlist was loaded; fallback seed words were used.",
                    severity=Severity.WARNING,
                    module="SpellChecker",
                    recoverable=True,
                )
            )
            if self.strict_loading:
                raise ResourceLanguageError(
                    ResourceIssue(
                        code=LanguageErrorCode.RESOURCE_MISSING,
                        message="SpellChecker strict loading failed because no wordlist resource was available.",
                        module="SpellChecker",
                        details={"paths": [str(path) for path in self._configured_wordlist_paths()]},
                    ),
                    recoverable=False,
                )

    def _record_resource_issue(self, path: Path, *, missing: bool) -> None:
        issue = ResourceIssue(
            code=LanguageErrorCode.RESOURCE_MISSING if missing else LanguageErrorCode.RESOURCE_LOAD_FAILED,
            message="Configured spell-checking resource was not found." if missing else "Configured spell-checking resource could not be loaded.",
            severity=Severity.WARNING,
            module="SpellChecker",
            recoverable=True,
            details={"path": str(path)},
        )
        self._add_issue(issue)
        logger.warning(issue.to_json())

    def _load_wordlist_path(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = load_json_file(path)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(read_text_file(path)) or {}
        else:
            payload = read_text_file(path).splitlines()

        if isinstance(payload, Mapping):
            self.wordlist_metadata = json_safe(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {}
            words_payload = payload.get("words", payload.get("entries", payload))
            if isinstance(words_payload, Mapping):
                for word, metadata in words_payload.items():
                    self._add_word_entry(ensure_text(word), metadata=metadata if isinstance(metadata, Mapping) else {}, source=str(path))
            elif isinstance(words_payload, Sequence) and not isinstance(words_payload, (str, bytes, bytearray)):
                for item in words_payload:
                    if isinstance(item, Mapping):
                        word = item.get("word") or item.get("text") or item.get("token")
                        if word:
                            self._add_word_entry(ensure_text(word), metadata=item, source=str(path))
                    else:
                        self._add_word_entry(ensure_text(item), source=str(path))
            else:
                raise ResourceLanguageError(
                    ResourceIssue(
                        code=LanguageErrorCode.RESOURCE_FORMAT_INVALID,
                        message="Unsupported spell-checking wordlist JSON shape.",
                        module="SpellChecker",
                        details={"path": str(path), "payload_type": type(payload).__name__},
                    ),
                    recoverable=False,
                )
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            for item in payload:
                self._add_word_entry(ensure_text(item), source=str(path))
        else:
            raise ResourceLanguageError(
                ResourceIssue(
                    code=LanguageErrorCode.RESOURCE_FORMAT_INVALID,
                    message="Unsupported spell-checking wordlist format.",
                    module="SpellChecker",
                    details={"path": str(path), "payload_type": type(payload).__name__},
                ),
                recoverable=False,
            )

    def _add_word_entry(self, word: str, *, metadata: Optional[Mapping[str, Any]] = None, source: str = "runtime") -> None:
        cleaned = self._normalize_word(word)
        if not cleaned:
            return
        meta = dict(metadata or {})
        pos = tuple(ensure_text(item).lower() for item in ensure_list(meta.get("pos", meta.get("upos", []))) if ensure_text(item))
        synonyms = tuple(self._normalize_word(item) for item in ensure_list(meta.get("synonyms", [])) if ensure_text(item))
        related = tuple(self._normalize_word(item) for item in ensure_list(meta.get("related_terms", meta.get("related", []))) if ensure_text(item))
        frequency = coerce_float(meta.get("frequency", meta.get("freq", meta.get("count", 0.0))), default=0.0, minimum=0.0)
        entry = WordEntry(
            word=ensure_text(word),
            normalized=cleaned,
            pos=pos,
            synonyms=synonyms,
            related_terms=related,
            frequency=frequency,
            source=source,
            metadata=json_safe(meta),
        )
        self.entries[cleaned] = entry
        self.words.add(cleaned)
        self.surface_forms.setdefault(cleaned, ensure_text(word))
        if self.include_related_terms:
            for related_word in synonyms + related:
                if related_word and related_word not in self.entries:
                    self.entries[related_word] = WordEntry(word=related_word, normalized=related_word, source=f"{source}:related")
                    self.words.add(related_word)
                    self.surface_forms.setdefault(related_word, related_word)

    def _load_typo_patterns(self) -> Tuple[Tuple[re.Pattern[str], str, str], ...]:
        configured_patterns = ensure_list(self.checker_config.get("typo_patterns", []))
        patterns: List[Tuple[str, str, str]] = list(self.DEFAULT_TYPO_PATTERNS)
        path_value = self.checker_config.get("typo_patterns_path")
        if path_value not in (None, "", "none", "None"):
            path = resolve_path(ensure_text(path_value), field_name="typo_patterns_path")
            if path.exists():
                payload = load_json_file(path) if path.suffix.lower() == ".json" else yaml.safe_load(read_text_file(path))
                configured_patterns.extend(ensure_list(payload))
        for item in configured_patterns:
            if isinstance(item, Mapping):
                pattern = ensure_text(item.get("pattern", ""))
                replacement = ensure_text(item.get("replacement", item.get("repl", "")))
                name = ensure_text(item.get("name", stable_hash({"pattern": pattern, "replacement": replacement}, length=8)))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) >= 2:
                pattern = ensure_text(item[0])
                replacement = ensure_text(item[1])
                name = ensure_text(item[2]) if len(item) >= 3 else stable_hash({"pattern": pattern, "replacement": replacement}, length=8)
            else:
                continue
            if pattern:
                patterns.append((pattern, replacement, name))
        compiled: List[Tuple[re.Pattern[str], str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for pattern, replacement, name in patterns:
            marker = (pattern, replacement)
            if marker in seen:
                continue
            seen.add(marker)
            compiled.append((re.compile(pattern, re.IGNORECASE), replacement, name))
        return tuple(compiled)

    def _build_indexes(self) -> None:
        self.length_index.clear()
        self.prefix_index.clear()
        self.phonetic_map.clear()
        for word in self.words:
            self.length_index[len(word)].add(word)
            if self.prefix_window > 0:
                self.prefix_index[word[: self.prefix_window]].add(word)
            if self.enable_phonetic:
                key = self._get_phonetic_key(word)
                if key:
                    self.phonetic_map[key].add(word)

    # ------------------------------------------------------------------
    # Normalization, filtering, and lookup
    # ------------------------------------------------------------------
    def _normalize_word(self, word: Any) -> str:
        return normalize_text(
            word,
            lowercase=not self.case_sensitive,
            casefold=self.casefold and not self.case_sensitive,
            unicode_form="NFKC",
            normalize_quote_chars=True,
            normalize_dash_chars=True,
            collapse_whitespace=True,
            remove_control_chars=True,
        ).strip(" \t\r\n")

    def _is_checkable(self, word: str) -> bool:
        text = ensure_text(word).strip()
        normalized = self._normalize_word(text)
        if not normalized or normalized in self.ignore_words:
            return False
        if len(normalized) < self.min_word_length or len(normalized) > self.max_word_length:
            return False
        if any(pattern.search(text) or pattern.search(normalized) for pattern in self.ignore_regexes):
            return False
        return any(char.isalpha() for char in normalized)

    def is_correct(self, word: Any) -> bool:
        normalized = self._normalize_word(word)
        if not normalized:
            return False
        if normalized in self.words:
            return True
        if self.enable_hyphenated_check and "-" in normalized:
            parts = [part for part in normalized.split("-") if part]
            return bool(parts) and all(part in self.words for part in parts)
        if self.enable_compound_check and " " in normalized:
            parts = [part for part in normalized.split() if part]
            return bool(parts) and all(part in self.words for part in parts)
        return False

    def word_probability(self, word: Any) -> float:
        normalized = self._normalize_word(word)
        entry = self.entries.get(normalized)
        if entry is None:
            return 0.0
        if entry.frequency > 0:
            max_frequency = max((candidate.frequency for candidate in self.entries.values()), default=entry.frequency)
            return clamp_float(entry.frequency / max(max_frequency, 1.0), 0.0, 1.0)
        # Resource has no frequencies; use a conservative lexical prior.
        length_penalty = min(len(normalized), 24) / 24.0
        return clamp_float(0.35 + (0.15 * length_penalty), 0.0, 0.5)

    # ------------------------------------------------------------------
    # Candidate generation and scoring
    # ------------------------------------------------------------------
    def _candidate_pool(self, normalized: str) -> Set[str]:
        candidates: Set[str] = set()
        length = len(normalized)
        for candidate_length in range(max(1, length - self.length_window), length + self.length_window + 1):
            candidates.update(self.length_index.get(candidate_length, set()))
        if self.prefix_window > 0 and normalized[: self.prefix_window]:
            candidates.update(self.prefix_index.get(normalized[: self.prefix_window], set()))
        if "phonetic" in self.suggestion_strategies and self.enable_phonetic:
            candidates.update(self.phonetic_map.get(self._get_phonetic_key(normalized), set()))
        if "typo_patterns" in self.suggestion_strategies and self.enable_typo_patterns:
            candidates.update(variant for variant in self.generate_typo_variants(normalized) if variant in self.words)
        if not candidates:
            # Last resort stays bounded by length buckets rather than scanning every word blindly.
            for candidate_length in range(max(1, length - self.length_window - 1), length + self.length_window + 2):
                candidates.update(self.length_index.get(candidate_length, set()))
        candidates.discard(normalized)
        if len(candidates) > self.max_candidates:
            candidates = set(sorted(candidates, key=lambda item: (abs(len(item) - len(normalized)), item))[: self.max_candidates])
        return candidates

    def suggest(self, word: Any, max_suggestions: Optional[int] = None) -> List[str]:
        self._suggest_calls += 1
        return [suggestion.word for suggestion in self._ranked_suggestions(word, max_suggestions=max_suggestions)]

    def suggest_with_scores(self, word: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        self._suggest_calls += 1
        return [suggestion.to_tuple() for suggestion in self._ranked_suggestions(word, max_suggestions=max_suggestions)]

    def _ranked_suggestions(self, word: Any, *, max_suggestions: Optional[int] = None) -> Tuple[SpellSuggestion, ...]:
        original = ensure_text(word)
        normalized = self._normalize_word(original)
        if not normalized or not self._is_checkable(original):
            return ()
        if normalized in self.words:
            return (SpellSuggestion(word=self._restore_case(original, self.surface_forms.get(normalized, normalized)), score=1.0, rank=1, source="exact", reasons=("known-word",)),)

        suggestions: List[SpellSuggestion] = []
        for candidate in self._candidate_pool(normalized):
            suggestion = self._score_candidate(normalized, candidate)
            if suggestion.score >= self.min_candidate_score:
                suggestions.append(suggestion)

        suggestions.sort(key=lambda item: (-item.score, item.edit_distance if item.edit_distance is not None else math.inf, item.word))
        limit = max_suggestions or self.max_suggestions
        ranked: List[SpellSuggestion] = []
        for index, suggestion in enumerate(dedupe_preserve_order(suggestions, key=lambda item: item.word)[:limit], start=1):
            surface = self._restore_case(original, self.surface_forms.get(suggestion.word, suggestion.word))
            ranked.append(
                SpellSuggestion(
                    word=surface,
                    score=suggestion.score,
                    rank=index,
                    edit_distance=suggestion.edit_distance,
                    edit_similarity=suggestion.edit_similarity,
                    phonetic_similarity=suggestion.phonetic_similarity,
                    keyboard_similarity=suggestion.keyboard_similarity,
                    frequency_score=suggestion.frequency_score,
                    prefix_similarity=suggestion.prefix_similarity,
                    shape_similarity=suggestion.shape_similarity,
                    source=suggestion.source,
                    reasons=suggestion.reasons,
                    metadata=suggestion.metadata,
                )
            )
        return tuple(ranked)

    def _score_candidate(self, original: str, candidate: str) -> SpellSuggestion:
        distance = self._weighted_edit_distance(original, candidate)
        max_len = max(len(original), len(candidate), 1)
        edit_sim = clamp_float(1.0 - (distance / max(max_len, self.max_edit_distance + 1.0)), 0.0, 1.0)
        phonetic_sim = self._phonetic_similarity(original, candidate) if self.enable_phonetic else 0.0
        keyboard_sim = self._keyboard_similarity(original, candidate) if self.enable_keyboard else 0.0
        frequency_score = self.word_probability(candidate) if self.enable_frequency else 0.0
        prefix_sim = self._prefix_similarity(original, candidate)
        length_sim = 1.0 - min(abs(len(original) - len(candidate)) / max_len, 1.0)
        shape_sim = 1.0 if word_shape(original) == word_shape(candidate) else 0.0
        typo_bonus = 0.05 if candidate in self.generate_typo_variants(original) else 0.0

        score = (
            edit_sim * self.score_weights["edit"]
            + phonetic_sim * self.score_weights["phonetic"]
            + keyboard_sim * self.score_weights["keyboard"]
            + frequency_score * self.score_weights["frequency"]
            + prefix_sim * self.score_weights["prefix"]
            + shape_sim * self.score_weights["shape"]
            + length_sim * self.score_weights["length"]
            + typo_bonus
        )
        reasons: List[str] = []
        if edit_sim >= 0.75:
            reasons.append("close-edit-distance")
        if phonetic_sim >= 0.9:
            reasons.append("phonetic-match")
        if keyboard_sim >= 0.7:
            reasons.append("keyboard-neighbor")
        if frequency_score > 0:
            reasons.append("known-frequency-prior")
        return SpellSuggestion(
            word=candidate,
            score=clamp_float(score, 0.0, 1.0),
            edit_distance=round(distance, 4),
            edit_similarity=round(edit_sim, 4),
            phonetic_similarity=round(phonetic_sim, 4),
            keyboard_similarity=round(keyboard_sim, 4),
            frequency_score=round(frequency_score, 4),
            prefix_similarity=round(prefix_sim, 4),
            shape_similarity=round(shape_sim, 4),
            source="ranked",
            reasons=tuple(reasons),
            metadata={"normalized_candidate": candidate},
        )

    def _prefix_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        shared = 0
        for char_a, char_b in zip(a, b):
            if char_a != char_b:
                break
            shared += 1
        return clamp_float(shared / max(min(len(a), len(b)), 1), 0.0, 1.0)

    def _weighted_edit_distance(self, original: str, candidate: str) -> float:
        insertion_cost = coerce_float(self.checker_config.get("insertion_cost", 1.0), default=1.0, minimum=0.0)
        deletion_cost = coerce_float(self.checker_config.get("deletion_cost", 1.0), default=1.0, minimum=0.0)
        default_sub_cost = coerce_float(self.checker_config.get("default_substitution_cost", 1.5), default=1.5, minimum=0.0)
        a = original if self.case_sensitive else original.lower()
        b = candidate if self.case_sensitive else candidate.lower()
        rows = len(a) + 1
        cols = len(b) + 1
        dp = [[0.0] * cols for _ in range(rows)]
        for i in range(1, rows):
            dp[i][0] = dp[i - 1][0] + deletion_cost
        for j in range(1, cols):
            dp[0][j] = dp[0][j - 1] + insertion_cost
        for i in range(1, rows):
            for j in range(1, cols):
                if a[i - 1] == b[j - 1]:
                    sub_cost = 0.0
                elif self.enable_keyboard and b[j - 1] in self.keyboard_layout.get(a[i - 1], set()):
                    sub_cost = min(default_sub_cost, 0.55)
                else:
                    sub_cost = default_sub_cost
                dp[i][j] = min(
                    dp[i - 1][j] + deletion_cost,
                    dp[i][j - 1] + insertion_cost,
                    dp[i - 1][j - 1] + sub_cost,
                )
        return dp[-1][-1]

    def _keyboard_similarity(self, word: str, candidate: str) -> float:
        if not word or not candidate:
            return 0.0
        comparable = min(len(word), len(candidate))
        if comparable == 0:
            return 0.0
        score = 0.0
        for left, right in zip(word[:comparable], candidate[:comparable]):
            if left == right:
                score += 1.0
            elif right in self.keyboard_layout.get(left, set()):
                score += 0.65
        length_penalty = 1.0 - min(abs(len(word) - len(candidate)) / max(len(word), len(candidate), 1), 1.0)
        return clamp_float((score / comparable) * length_penalty, 0.0, 1.0)

    def _phonetic_similarity(self, word: str, candidate: str) -> float:
        key_a = self._get_phonetic_key(word)
        key_b = self._get_phonetic_key(candidate)
        if not key_a or not key_b:
            return 0.0
        if key_a == key_b:
            return 1.0
        if key_a[0] == key_b[0]:
            shared = sum(1 for a, b in zip(key_a, key_b) if a == b)
            return clamp_float(shared / max(len(key_a), len(key_b), 1), 0.0, 0.85)
        return 0.0

    def _get_phonetic_key(self, word: Any) -> str:
        normalized = self._normalize_word(word)
        if not normalized:
            return ""
        if self.phonetic_algorithm == "metaphone":
            primary, secondary = doublemetaphone(normalized)
            return primary or secondary or self._soundex(normalized)
        return self._soundex(normalized)

    def _soundex(self, word: Any) -> str:
        text = re.sub(r"[^A-Za-z]", "", ensure_text(word)).upper()
        if not text:
            return "0000"
        first = text[0]
        mapping = {
            "B": "1", "F": "1", "P": "1", "V": "1",
            "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
            "D": "3", "T": "3", "L": "4", "M": "5", "N": "5", "R": "6",
        }
        digits: List[str] = []
        previous = mapping.get(first, "")
        for char in text[1:]:
            code = mapping.get(char, "")
            if code and code != previous:
                digits.append(code)
            previous = code
        return (first + "".join(digits) + "000")[:4]

    def generate_typo_variants(self, word: str) -> Set[str]:
        normalized = self._normalize_word(word)
        variants: Set[str] = set()

        def add(value: str) -> None:
            value = self._normalize_word(value)
            if value and value != normalized and len(variants) < self.max_typo_variants:
                variants.add(value)

        if not normalized:
            return variants

        for pattern, replacement, _name in self.typo_patterns:
            add(pattern.sub(replacement, normalized))

        for index in range(len(normalized) - 1):
            chars = list(normalized)
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
            add("".join(chars))

        for index in range(len(normalized)):
            add(normalized[:index] + normalized[index + 1:])
            add(normalized[: index + 1] + normalized[index] + normalized[index + 1:])

        if self.enable_keyboard:
            for index, char in enumerate(normalized):
                for nearby in sorted(self.keyboard_layout.get(char, set())):
                    add(normalized[:index] + nearby + normalized[index + 1:])

        vowels = ensure_text(self.checker_config.get("insertion_characters", "aeiourstln")).lower()
        for index in range(len(normalized) + 1):
            for char in vowels:
                add(normalized[:index] + char + normalized[index:])

        return variants

    # ------------------------------------------------------------------
    # Public checking and correction API
    # ------------------------------------------------------------------
    def check(self, word: Any, *, span: Optional[Span] = None, max_suggestions: Optional[int] = None) -> SpellCheckResult:
        self._check_calls += 1
        original = ensure_text(word)
        normalized = self._normalize_word(original)
        checkable = self._is_checkable(original)
        issues: List[LanguageIssue] = []
        if not checkable:
            return SpellCheckResult(
                original=original,
                normalized=normalized,
                is_correct=True,
                checkable=False,
                span=span,
                language=self.language,
                locale=self.locale,
                metadata={"reason": "not-checkable"},
            )
        if self.is_correct(original):
            return SpellCheckResult(
                original=original,
                normalized=normalized,
                is_correct=True,
                checkable=True,
                corrected=original,
                confidence=1.0,
                suggestions=(),
                span=span,
                language=self.language,
                locale=self.locale,
                metadata={"reason": "known-word"},
            )

        suggestions = self._ranked_suggestions(original, max_suggestions=max_suggestions)
        best = suggestions[0] if suggestions else None
        corrected = best.word if best and best.score >= self.auto_correct_threshold else None
        confidence = best.score if best else 0.0
        issue = OrthographyIssue(
            code=LanguageErrorCode.ORTH_UNKNOWN_WORD,
            message="Unknown or misspelled word detected.",
            severity=Severity.WARNING,
            module="SpellChecker",
            source_text=original,
            source_span=span,
            suggestion=best.word if best else None,
            confidence=confidence,
            recoverable=True,
            details={"suggestion_count": len(suggestions), "normalized": normalized},
        )
        issues.append(issue)
        self._add_issue(issue)
        return SpellCheckResult(
            original=original,
            normalized=normalized,
            is_correct=False,
            checkable=True,
            best=best,
            suggestions=suggestions,
            corrected=corrected,
            confidence=confidence,
            issues=tuple(issues),
            span=span,
            language=self.language,
            locale=self.locale,
        )

    def correct(self, word: Any, *, threshold: Optional[float] = None) -> str:
        result = self.check(word)
        effective_threshold = self.auto_correct_threshold if threshold is None else clamp_float(threshold, 0.0, 1.0)
        if result.best and result.best.score >= effective_threshold:
            return result.best.word
        return ensure_text(word)

    def check_text(self, text: Any, *, auto_correct: bool = False, threshold: Optional[float] = None) -> TextSpellCheckResult:
        self._text_check_calls += 1
        original_text = ensure_text(text)
        checks: List[SpellCheckResult] = []
        issues: List[LanguageIssue] = []
        replacements: List[Tuple[Span, str]] = []
        word_pattern = re.compile(r"\b[\w][\w'’-]*\b", re.UNICODE)
        effective_threshold = self.auto_correct_threshold if threshold is None else clamp_float(threshold, 0.0, 1.0)

        for match in word_pattern.finditer(original_text):
            span = (int(match.start()), int(match.end()))
            result = self.check(match.group(0), span=span)
            checks.append(result)
            issues.extend(result.issues)
            if auto_correct and result.best and result.best.score >= effective_threshold:
                replacements.append((span, result.best.word))

        corrected_text = original_text
        for (start, end), replacement in sorted(replacements, key=lambda item: item[0][0], reverse=True):
            corrected_text = corrected_text[:start] + replacement + corrected_text[end:]

        return TextSpellCheckResult(
            original_text=original_text,
            corrected_text=corrected_text,
            checks=tuple(checks),
            issues=tuple(issues),
            metadata={"auto_correct": auto_correct, "threshold": effective_threshold, "checked_tokens": len(checks)},
        )

    def batch_check(self, words: Iterable[Any], *, max_suggestions: Optional[int] = None) -> List[SpellCheckResult]:
        return [self.check(word, max_suggestions=max_suggestions) for word in words]

    def add_words(self, words: Iterable[Any], *, source: str = "runtime") -> int:
        before = len(self.words)
        for word in words:
            self._add_word_entry(ensure_text(word), source=source)
        self._build_indexes()
        return len(self.words) - before

    def remove_words(self, words: Iterable[Any]) -> int:
        removed = 0
        for word in words:
            normalized = self._normalize_word(word)
            if normalized in self.words:
                self.words.remove(normalized)
                self.entries.pop(normalized, None)
                self.surface_forms.pop(normalized, None)
                removed += 1
        if removed:
            self._build_indexes()
        return removed

    # ------------------------------------------------------------------
    # Diagnostics, stats, and serialization
    # ------------------------------------------------------------------
    def _restore_case(self, original: str, suggestion: str) -> str:
        if not self.preserve_case:
            return suggestion
        if original.isupper():
            return suggestion.upper()
        if original.istitle():
            return suggestion.title()
        if len(original) > 1 and original[0].isupper() and original[1:].islower():
            return suggestion.capitalize()
        return suggestion

    def _add_issue(self, issue: Union[LanguageIssue, LanguageError]) -> None:
        self.diagnostics.add(issue)

    def stats(self) -> SpellCheckerStats:
        return SpellCheckerStats(
            language=self.language,
            locale=self.locale,
            word_count=len(self.words),
            entry_count=len(self.entries),
            phonetic_bucket_count=len(self.phonetic_map),
            length_bucket_count=len(self.length_index),
            check_calls=self._check_calls,
            suggest_calls=self._suggest_calls,
            text_check_calls=self._text_check_calls,
            diagnostics_count=len(self.diagnostics.issues),
            resources={
                "wordlist_paths": [str(path) for path in self._configured_wordlist_paths()],
                "wordlist_metadata": self.wordlist_metadata,
                "phonetic_algorithm": self.phonetic_algorithm,
                "strategy_count": len(self.suggestion_strategies),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "language": self.language,
            "locale": self.locale,
            "stats": self.stats().to_dict(),
            "config": json_safe(self.checker_config),
            "diagnostics": self.diagnostics.to_list(),
        }

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and self.is_correct(word)

    def __len__(self) -> int:
        return len(self.words)

    def __repr__(self) -> str:
        return f"<SpellChecker language='{self.language}' locale='{self.locale}' words={len(self.words)}>"


if __name__ == "__main__":
    print("\n=== Running Spell Checker ===\n")
    printer.status("TEST", "Spell Checker initialized", "info")

    checker = SpellChecker()

    sample_words = ["Unfortonally", "recieve", "langauge", "tokenizer", "production", "queeen"]
    results = [checker.check(word).to_dict() for word in sample_words]
    suggestions = checker.suggest_with_scores("Unfortonally", max_suggestions=5)
    typo_variants = sorted(list(checker.generate_typo_variants("queeen")))[:25]
    corrected = checker.correct("Unfortonally", threshold=0.70)
    text_result = checker.check_text(
        "Unfortonally, this langauge module should recieve production-ready checks.",
        auto_correct=True,
        threshold=0.70,
    )

    printer.pretty("CHECK_RESULTS", results, "success")
    printer.pretty("SUGGESTIONS", suggestions, "success")
    printer.pretty("TYPO_VARIANTS", typo_variants, "success")
    printer.pretty("CORRECTED_WORD", {"word": corrected}, "success")
    printer.pretty("TEXT_RESULT", text_result.to_dict(), "success")
    printer.pretty("SOUNDEX", {"killer": checker._soundex("killer"), "killing": checker._soundex("killing")}, "success")
    printer.pretty("STATS", checker.stats().to_dict(), "success")

    print("\n=== Test ran successfully ===\n")
