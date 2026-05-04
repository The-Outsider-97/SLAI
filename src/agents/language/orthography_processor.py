"""
Orthography Processor Module

Core Function:
Normalizes and corrects user-facing text before it enters the language pipeline.
The processor performs Unicode cleanup, quote/dash normalization, contraction
expansion, locale-aware spelling variants, spell-check backed typo correction,
compound handling, and span-aware edit tracking.

Responsibilities:
- Preserve the current agent contract: ``batch_process(text) -> str`` and
  ``correct(word, context=None) -> str``.
- Provide a richer ``process_text`` result for newer pipeline stages that need
  edits, spans, diagnostics, and confidence metadata.
- Use the production SpellChecker module for lexical checks and candidate
  ranking instead of duplicating spelling logic here.
- Keep locale and normalization policy config-driven through language_config.yaml.
- Record enough metadata for NLP, grammar, NLU, dialogue context, and NLG to
  reason about what changed and why.

Why it matters:
Orthography is the first language-stage transformation after input safety. If it
silently corrupts spans, punctuation, casing, locale, or contractions, every
later module receives a distorted view of the user input. A production
orthography processor must therefore be conservative, deterministic,
inspectable, and compatible with both simple string workflows and structured
pipeline workflows.
"""

from __future__ import annotations

import re
import yaml

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.language_error import *
from .utils.language_helpers import *
from .modules.spell_checker import SpellChecker
from .language_memory import LanguageMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Orthography Processor")
printer = PrettyPrinter()

Span = Tuple[int, int]


@dataclass(frozen=True)
class OrthographyToken:
    """Span-aware token inspected by the orthography processor."""

    text: str
    start_char: int
    end_char: int
    index: int
    kind: str = "word"
    normalized: Optional[str] = None
    checkable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> Span:
        return (self.start_char, self.end_char)

    @property
    def lower(self) -> str:
        return self.text.lower()

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self), drop_empty=True)


@dataclass(frozen=True)
class OrthographyEdit:
    """Single text edit emitted by orthography normalization or correction."""

    original: str
    replacement: str
    source_span: Span
    edit_type: str
    confidence: float = 1.0
    rule_id: Optional[str] = None
    suggestion_source: Optional[str] = None
    issues: Tuple[Any, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.original != self.replacement

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "original": self.original,
                "replacement": self.replacement,
                "source_span": list(self.source_span),
                "edit_type": self.edit_type,
                "confidence": self.confidence,
                "rule_id": self.rule_id,
                "suggestion_source": self.suggestion_source,
                "issues": [issue.to_dict() if hasattr(issue, "to_dict") else json_safe(issue) for issue in self.issues],
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )


@dataclass(frozen=True)
class OrthographyProcessingResult:
    """Structured text-level orthography result."""

    original_text: str
    normalized_text: str
    corrected_text: str
    tokens: Tuple[OrthographyToken, ...]
    edits: Tuple[OrthographyEdit, ...] = ()
    issues: Tuple[Any, ...] = ()
    locale: str = "en-US"
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.original_text != self.corrected_text

    @property
    def corrections(self) -> List[Dict[str, Any]]:
        return [edit.to_dict() for edit in self.edits if edit.changed]

    @property
    def ok(self) -> bool:
        return not any(getattr(issue, "is_blocking", False) for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "changed": self.changed,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "corrected_text": self.corrected_text,
            "tokens": [token.to_dict() for token in self.tokens],
            "edits": [edit.to_dict() for edit in self.edits],
            "corrections": self.corrections,
            "issues": [issue.to_dict() if hasattr(issue, "to_dict") else json_safe(issue) for issue in self.issues],
            "locale": self.locale,
            "language": self.language,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class OrthographyProcessorStats:
    """Runtime snapshot for observability and tests."""

    language: str
    locale: str
    process_calls: int
    correct_calls: int
    normalize_calls: int
    contraction_expansions: int
    spelling_corrections: int
    locale_normalizations: int
    compound_edits: int
    diagnostics_count: int
    history_length: int
    resources: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["resources"] = json_safe(self.resources)
        return payload


class OrthographyProcessor:
    """
    Production orthography processor for the language agent pipeline.

    The class is deliberately conservative: it does not rewrite arbitrary text
    unless the configured rules or the SpellChecker confidence threshold justify
    the edit. It also keeps the simple legacy API used by LanguageAgent while
    exposing structured results for newer integrations.
    """

    DEFAULT_CONTRACTIONS: Dict[str, str] = {
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "I would",
        "i'll": "I will",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "where's": "where is",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
    }

    DEFAULT_LOCALE_VARIANTS: Dict[str, Dict[str, str]] = {
        "en-US": {
            "colour": "color",
            "colours": "colors",
            "flavour": "flavor",
            "flavours": "flavors",
            "honour": "honor",
            "honours": "honors",
            "humour": "humor",
            "labour": "labor",
            "neighbour": "neighbor",
            "neighbours": "neighbors",
            "programme": "program",
            "theatre": "theater",
            "centre": "center",
            "metre": "meter",
            "litre": "liter",
            "realise": "realize",
            "realised": "realized",
            "realising": "realizing",
            "organise": "organize",
            "organised": "organized",
            "analysed": "analyzed",
            "cancelled": "canceled",
            "travelling": "traveling",
        },
        "en-GB": {
            "color": "colour",
            "colors": "colours",
            "flavor": "flavour",
            "flavors": "flavours",
            "honor": "honour",
            "honors": "honours",
            "humor": "humour",
            "labor": "labour",
            "neighbor": "neighbour",
            "neighbors": "neighbours",
            "program": "programme",
            "theater": "theatre",
            "center": "centre",
            "meter": "metre",
            "liter": "litre",
            "realize": "realise",
            "realized": "realised",
            "realizing": "realising",
            "organize": "organise",
            "organized": "organised",
            "analyzed": "analysed",
            "canceled": "cancelled",
            "traveling": "travelling",
        },
    }

    def __init__(self) -> None:
        self.config = load_global_config()
        self.op_config = get_config_section("orthography_processor") or {}

        self.version = ensure_text(self.op_config.get("version", "2.0"))
        self.language = ensure_text(self.op_config.get("language", "en"))
        self.locale = self._validate_locale()
        self.enabled = coerce_bool(self.op_config.get("enabled", True), default=True)
        self.enable_auto_correction = coerce_bool(self.op_config.get("enable_auto_correction", True), default=True)
        self.auto_correction_confidence = coerce_float(self.op_config.get("auto_correction_confidence", 0.86), default=0.86, minimum=0.0, maximum=1.0)
        self.low_confidence_threshold = coerce_float(self.op_config.get("low_confidence_threshold", 0.62), default=0.62, minimum=0.0, maximum=1.0)
        self.enable_contraction_expansion = coerce_bool(self.op_config.get("enable_contraction_expansion", True), default=True)
        self.enable_compound_handling = coerce_bool(self.op_config.get("enable_compound_handling", True), default=True)
        self.enable_locale_normalization = coerce_bool(self.op_config.get("enable_locale_normalization", True), default=True)
        self.enable_spelling_correction = coerce_bool(self.op_config.get("enable_spelling_correction", True), default=True)
        self.preserve_case = coerce_bool(self.op_config.get("preserve_case", True), default=True)
        self.preserve_whitespace = coerce_bool(self.op_config.get("preserve_whitespace", True), default=True)
        self.normalize_unicode = coerce_bool(self.op_config.get("normalize_unicode", True), default=True)
        self.normalize_quotes = coerce_bool(self.op_config.get("normalize_quotes", True), default=True)
        self.normalize_dashes = coerce_bool(self.op_config.get("normalize_dashes", True), default=True)
        self.collapse_whitespace = coerce_bool(self.op_config.get("collapse_whitespace", False), default=False)
        self.remove_control_chars = coerce_bool(self.op_config.get("remove_control_chars", True), default=True)
        self.record_history = coerce_bool(self.op_config.get("record_history", True), default=True)
        self.log_errors = coerce_bool(self.op_config.get("log_errors", True), default=True)
        self.strict_loading = coerce_bool(self.op_config.get("strict_loading", False), default=False)
        self.max_context_window = coerce_int(self.op_config.get("max_context_window", 3), default=3, minimum=0)
        self.max_input_chars = coerce_int(self.op_config.get("max_input_chars", 200_000), default=200_000, minimum=1)
        self.max_token_length = coerce_int(self.op_config.get("max_token_length", 96), default=96, minimum=1)
        self.max_suggestions = coerce_int(self.op_config.get("max_suggestions", 5), default=5, minimum=1)

        self.normalization_map_path = self.op_config.get("normalization_map_path")
        self.contraction_map_path = self.op_config.get("contraction_map_path")
        self.locale_variants_path = self.op_config.get("locale_variants_path")
        self.protected_terms_path = self.op_config.get("protected_terms_path")

        self.token_pattern = re.compile(
            ensure_text(
                self.op_config.get(
                    "token_pattern",
                    r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*|\d+(?:[.,:/-]\d+)*|\S",
                )
            ),
            re.UNICODE,
        )
        self.word_like_pattern = re.compile(
            ensure_text(self.op_config.get("word_like_pattern", r"^[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*$")),
            re.UNICODE,
        )
        self.ignore_patterns = tuple(
            re.compile(ensure_text(pattern))
            for pattern in ensure_list(
                self.op_config.get(
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
            if ensure_text(pattern)
        )

        self.spellchecker = SpellChecker()
        self.memory: Optional[LanguageMemory] = LanguageMemory() if coerce_bool(self.op_config.get("enable_memory", False), default=False) else None
        self.diagnostics = LanguageDiagnostics()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=coerce_int(self.op_config.get("history_limit", 200), default=200, minimum=1))

        self.normalization_map = self._load_normalization_map()
        self.contraction_map = self._load_contraction_map()
        self.locale_variants = self._load_locale_variants()
        self.protected_terms = self._load_protected_terms()
        self._stats = Counter()

        logger.info("Orthography Processor initialized with locale=%s", self.locale)
        printer.status("INIT", "Orthography Processor initialized", "success")

    # ------------------------------------------------------------------
    # Configuration and resources
    # ------------------------------------------------------------------
    def _validate_locale(self) -> str:
        configured_locale = ensure_text(self.op_config.get("locale", self.config.get("locale", "en-US")))
        allowed = [ensure_text(item) for item in ensure_list(self.op_config.get("allowed_locales", ["en-US", "en-GB"]))]
        if configured_locale in allowed:
            return configured_locale
        issue = OrthographyIssue(
            code=LanguageErrorCode.ORTH_LOCALE_CONFLICT,
            message="Configured orthography locale is not in allowed_locales; using fallback locale.",
            severity=Severity.WARNING,
            module="OrthographyProcessor",
            recoverable=True,
            details={"configured_locale": configured_locale, "allowed_locales": allowed, "fallback": "en-US"},
        )
        # diagnostics is not initialized yet, so log directly.
        logger.warning(issue.to_json())
        return "en-US"

    def _load_mapping_file(self, path_value: Any, *, root_key: Optional[str] = None) -> Dict[str, Any]:
        if path_value in (None, "", "none", "None"):
            return {}
        path = resolve_path(ensure_text(path_value), field_name=root_key or "mapping_path")
        if not path.exists():
            issue = ResourceIssue(
                code=LanguageErrorCode.RESOURCE_MISSING,
                message="Configured orthography resource file was not found.",
                severity=Severity.WARNING,
                module="OrthographyProcessor",
                recoverable=True,
                details={"path": str(path), "root_key": root_key},
            )
            logger.warning(issue.to_json())
            if self.strict_loading:
                raise ResourceLanguageError(issue, recoverable=False)
            return {}
        payload = load_json_file(path) if path.suffix.lower() == ".json" else yaml.safe_load(read_text_file(path)) or {}
        if root_key and isinstance(payload, Mapping):
            payload = payload.get(root_key, payload)
        return dict(payload or {}) if isinstance(payload, Mapping) else {}

    def _load_normalization_map(self) -> Dict[str, str]:
        values: Dict[str, str] = {}
        values.update({ensure_text(k).lower(): ensure_text(v) for k, v in self._load_mapping_file(self.normalization_map_path, root_key="normalization_rules").items()})
        inline = ensure_mapping(self.op_config.get("normalization_map", {}), field_name="orthography_processor.normalization_map", allow_none=True)
        values.update({ensure_text(k).lower(): ensure_text(v) for k, v in inline.items()})
        return values

    def _load_contraction_map(self) -> Dict[str, str]:
        values: Dict[str, str] = dict(self.DEFAULT_CONTRACTIONS)
        values.update({ensure_text(k).lower(): ensure_text(v) for k, v in self._load_mapping_file(self.contraction_map_path, root_key="contractions").items()})
        inline = ensure_mapping(self.op_config.get("contraction_map", {}), field_name="orthography_processor.contraction_map", allow_none=True)
        values.update({ensure_text(k).lower(): ensure_text(v) for k, v in inline.items()})
        return values

    def _load_locale_variants(self) -> Dict[str, Dict[str, str]]:
        variants: Dict[str, Dict[str, str]] = {locale: dict(mapping) for locale, mapping in self.DEFAULT_LOCALE_VARIANTS.items()}
        resource = self._load_mapping_file(self.locale_variants_path, root_key="locale_variants")
        inline = ensure_mapping(self.op_config.get("locale_variants", {}), field_name="orthography_processor.locale_variants", allow_none=True)
        for source in (resource, inline):
            for locale, mapping in source.items():
                if isinstance(mapping, Mapping):
                    variants.setdefault(ensure_text(locale), {}).update({ensure_text(k).lower(): ensure_text(v) for k, v in mapping.items()})
        return variants

    def _load_protected_terms(self) -> set[str]:
        protected = {self._normalize_lookup(item) for item in ensure_list(self.op_config.get("protected_terms", [])) if ensure_text(item)}
        resource = self._load_mapping_file(self.protected_terms_path, root_key="protected_terms")
        if isinstance(resource, Mapping):
            for value in resource.values():
                protected.update(self._normalize_lookup(item) for item in ensure_list(value) if ensure_text(item))
        return {item for item in protected if item}

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _normalize_lookup(self, text: Any) -> str:
        return normalize_text(
            text,
            lowercase=True,
            casefold=True,
            unicode_form="NFKC",
            normalize_quote_chars=True,
            normalize_dash_chars=True,
            collapse_whitespace=True,
            remove_control_chars=True,
        )

    def _normalize_text_surface(self, text: Any) -> str:
        value = ensure_text(text)
        if len(value) > self.max_input_chars:
            raise OrthographyError(
                OrthographyIssue(
                    code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                    message="Input text exceeds configured orthography maximum length.",
                    severity=Severity.ERROR,
                    module="OrthographyProcessor",
                    recoverable=True,
                    details={"max_input_chars": self.max_input_chars, "received_chars": len(value)},
                ),
                recoverable=True,
            )
        return normalize_text(
            value,
            lowercase=False,
            unicode_form="NFKC" if self.normalize_unicode else None,
            normalize_quote_chars=self.normalize_quotes,
            normalize_dash_chars=self.normalize_dashes,
            collapse_whitespace=self.collapse_whitespace and not self.preserve_whitespace,
            remove_control_chars=self.remove_control_chars,
        )

    def _restore_case(self, original: str, replacement: str) -> str:
        if not self.preserve_case or not replacement:
            return replacement
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.title()
        if len(original) > 1 and original[0].isupper() and original[1:].islower():
            return replacement[0].upper() + replacement[1:] if replacement else replacement
        return replacement

    def _token_kind(self, text: str) -> str:
        if any(pattern.search(text) for pattern in self.ignore_patterns):
            return "ignored"
        if self.word_like_pattern.match(text):
            return "word"
        if text.isnumeric():
            return "number"
        if len(text) == 1 and not text.isalnum():
            return "punct"
        return "symbol"

    def _is_checkable_token(self, token: OrthographyToken) -> bool:
        if token.kind != "word":
            return False
        normalized = self._normalize_lookup(token.text)
        if not normalized or normalized in self.protected_terms:
            return False
        if len(normalized) > self.max_token_length:
            return False
        return not any(pattern.search(token.text) or pattern.search(normalized) for pattern in self.ignore_patterns)

    def _iter_tokens(self, text: str) -> Tuple[OrthographyToken, ...]:
        tokens: List[OrthographyToken] = []
        for index, match in enumerate(self.token_pattern.finditer(text)):
            raw = match.group(0)
            kind = self._token_kind(raw)
            token = OrthographyToken(
                text=raw,
                start_char=int(match.start()),
                end_char=int(match.end()),
                index=index,
                kind=kind,
                normalized=self._normalize_lookup(raw) if kind == "word" else raw,
                checkable=kind == "word",
            )
            tokens.append(token)
        return tuple(tokens)

    # ------------------------------------------------------------------
    # Correction logic
    # ------------------------------------------------------------------
    def normalize(self, word: str) -> str:
        """Compatibility method: normalize one token without running full text processing."""
        self._stats["normalize_calls"] += 1
        token = ensure_text(word)
        replacement, _edit = self._normalize_token_value(OrthographyToken(token, 0, len(token), 0, kind=self._token_kind(token)))
        return replacement

    def expand_contractions(self, word: str) -> str:
        """Compatibility method: expand a single contraction if configured."""
        normalized = self._normalize_lookup(word)
        expanded = self.contraction_map.get(normalized, ensure_text(word))
        return self._restore_case(ensure_text(word), expanded)

    def correct(self, word: str, context: Optional[List[str]] = None) -> str:
        """Compatibility method: correct one token using optional surrounding context."""
        self._stats["correct_calls"] += 1
        token_text = ensure_text(word)
        token = OrthographyToken(token_text, 0, len(token_text), 0, kind=self._token_kind(token_text), normalized=self._normalize_lookup(token_text))
        replacement, _edit = self._process_token(token, context=context or [], auto_correct=self.enable_auto_correction)
        return replacement

    def batch_process(self, text: str) -> str:
        """Compatibility method used by LanguageAgent: return only corrected text."""
        return self.process_text(text, auto_correct=self.enable_auto_correction).corrected_text

    def process_text(
        self,
        text: Any,
        *,
        context: Optional[Sequence[str]] = None,
        auto_correct: Optional[bool] = None,
        return_result: bool = True,
    ) -> OrthographyProcessingResult:
        """Run the full span-aware orthography pipeline."""
        del return_result
        self._stats["process_calls"] += 1
        original = ensure_text(text)
        normalized_text = self._normalize_text_surface(original)
        tokens = self._iter_tokens(normalized_text)
        edits: List[OrthographyEdit] = []
        issues: List[Any] = []
        corrections: List[Tuple[Span, str]] = []
        effective_auto_correct = self.enable_auto_correction if auto_correct is None else bool(auto_correct)

        for token in tokens:
            local_context = self._context_window(tokens, token.index, external_context=context)
            replacement, edit = self._process_token(token, context=local_context, auto_correct=effective_auto_correct)
            if edit is not None:
                edits.append(edit)
                issues.extend(edit.issues)
                corrections.append((token.span, replacement))

        corrected_text = normalized_text
        for (start, end), replacement in sorted(corrections, key=lambda item: item[0][0], reverse=True):
            corrected_text = corrected_text[:start] + replacement + corrected_text[end:]
        corrected_text = self._finalize_text(corrected_text)

        result = OrthographyProcessingResult(
            original_text=original,
            normalized_text=normalized_text,
            corrected_text=corrected_text,
            tokens=tokens,
            edits=tuple(edits),
            issues=tuple(issues),
            locale=self.locale,
            language=self.language,
            metadata={
                "auto_correct": effective_auto_correct,
                "context_terms": list(context or []),
                "token_count": len(tokens),
                "edit_count": len(edits),
            },
        )
        self._record("process_text", text_preview=truncate_text(original, 160), corrected_preview=truncate_text(corrected_text, 160), edits=len(edits))
        self._remember_result(result)
        return result

    def _process_token(self, token: OrthographyToken, *, context: Sequence[str], auto_correct: bool) -> Tuple[str, Optional[OrthographyEdit]]:
        if not self.enabled or not self._is_checkable_token(token):
            return token.text, None

        current = token.text
        applied_types: List[str] = []
        applied_rules: List[str] = []
        confidence = 1.0
        issues: List[Any] = []
        metadata: Dict[str, Any] = {}

        normalized_replacement, normalized_edit = self._normalize_token_value(token)
        if normalized_edit is not None and normalized_replacement != current:
            current = normalized_replacement
            applied_types.append("normalization")
            applied_rules.append("orth.normalization.map")
            confidence = min(confidence, normalized_edit.confidence)

        if self.enable_contraction_expansion:
            expanded = self.contraction_map.get(self._normalize_lookup(current))
            if expanded and expanded != current:
                current = self._restore_case(token.text, expanded)
                applied_types.append("contraction")
                applied_rules.append("orth.contraction.expand")
                confidence = min(confidence, 0.99)
                self._stats["contraction_expansions"] += 1

        if self.enable_locale_normalization:
            locale_replacement = self._locale_variant(current)
            if locale_replacement and locale_replacement != current:
                current = self._restore_case(token.text, locale_replacement)
                applied_types.append("locale")
                applied_rules.append(f"orth.locale.{self.locale}")
                confidence = min(confidence, 0.95)
                self._stats["locale_normalizations"] += 1

        # Spelling correction is applied after deterministic normalization so
        # words like ``cant`` can normalize to ``can't`` and then expand safely.
        if self.enable_spelling_correction and current == token.text:
            spell_replacement, spell_edit = self._spellcheck_token(token, context=context, auto_correct=auto_correct)
            if spell_edit is not None:
                return spell_replacement, spell_edit

        if self.enable_compound_handling and current == token.text:
            compound_replacement = self._compound_replacement(token.text)
            if compound_replacement and compound_replacement != token.text:
                current = compound_replacement
                applied_types.append("compound")
                applied_rules.append("orth.compound.split")
                confidence = min(confidence, 0.78)
                self._stats["compound_edits"] += 1

        if applied_types:
            return current, self._make_edit(
                token,
                current,
                "+".join(applied_types),
                confidence,
                rule_id="+".join(applied_rules),
                issues=tuple(issues),
                metadata=metadata,
            )
        return token.text, None

    def _normalize_token_value(self, token: OrthographyToken) -> Tuple[str, Optional[OrthographyEdit]]:
        normalized_lookup = self._normalize_lookup(token.text)
        mapped = self.normalization_map.get(normalized_lookup)
        if mapped and mapped != token.text:
            replacement = self._restore_case(token.text, mapped)
            return replacement, self._make_edit(token, replacement, "normalization", 0.96, rule_id="orth.normalization.map")
        return token.text, None

    def _locale_variant(self, token: str) -> Optional[str]:
        normalized = self._normalize_lookup(token)
        return self.locale_variants.get(self.locale, {}).get(normalized)

    def _spellcheck_token(self, token: OrthographyToken, *, context: Sequence[str], auto_correct: bool) -> Tuple[str, Optional[OrthographyEdit]]:
        result = self.spellchecker.check(token.text, span=token.span, max_suggestions=self.max_suggestions)
        if getattr(result, "is_correct", False) or not getattr(result, "checkable", True):
            return token.text, None
        best = getattr(result, "best", None)
        confidence = float(getattr(best, "score", 0.0) if best else getattr(result, "confidence", 0.0))
        suggestion = ensure_text(getattr(best, "word", "")) if best else ""
        if context and suggestion:
            confidence = self._context_adjusted_confidence(token.text, suggestion, confidence, context)
        issues = tuple(getattr(result, "issues", ()) or ())

        if auto_correct and suggestion and confidence >= self.auto_correction_confidence:
            replacement = self._restore_case(token.text, suggestion)
            self._stats["spelling_corrections"] += 1
            if self.log_errors:
                logger.info("Orthography correction accepted: %s -> %s (confidence=%.3f)", token.text, replacement, confidence)
            return replacement, self._make_edit(
                token,
                replacement,
                "spelling",
                confidence,
                rule_id="orth.spelling.auto_correct",
                suggestion_source=getattr(best, "source", "spellchecker") if best else "spellchecker",
                issues=issues,
                metadata={"suggestions": [item.to_dict() for item in getattr(result, "suggestions", ())]},
            )

        issue = OrthographyIssue(
            code=LanguageErrorCode.ORTH_CORRECTION_LOW_CONFIDENCE if suggestion else LanguageErrorCode.ORTH_UNKNOWN_WORD,
            message="Spelling correction was not applied because confidence was below the auto-correction threshold." if suggestion else "Unknown word detected with no safe correction.",
            severity=Severity.INFO if suggestion else Severity.WARNING,
            module="OrthographyProcessor",
            source_text=token.text,
            source_span=token.span,
            suggestion=suggestion or None,
            confidence=confidence,
            recoverable=True,
            details={"threshold": self.auto_correction_confidence, "context": list(context)},
        )
        self._add_issue(issue)
        return token.text, self._make_edit(token, token.text, "spelling_review", confidence, rule_id="orth.spelling.review", issues=issues + (issue,))

    def _compound_replacement(self, token: str) -> Optional[str]:
        normalized = self._normalize_lookup(token)
        if len(normalized) < coerce_int(self.op_config.get("min_compound_length", 8), default=8, minimum=2):
            return None
        if self.spellchecker.is_correct(normalized):
            return None
        for split in range(3, len(normalized) - 2):
            left, right = normalized[:split], normalized[split:]
            if self.spellchecker.is_correct(left) and self.spellchecker.is_correct(right):
                return self._restore_case(token, f"{left} {right}")
        return None

    def _context_adjusted_confidence(self, original: str, suggestion: str, base_confidence: float, context: Sequence[str]) -> float:
        context_terms = [self._normalize_lookup(item) for item in context if ensure_text(item)]
        if not context_terms:
            return base_confidence
        suggestion_norm = self._normalize_lookup(suggestion)
        overlap = 0.0
        for term in context_terms:
            overlap = max(overlap, normalized_edit_similarity(suggestion_norm, term))
        # Context is useful but should not dominate spelling evidence.
        return clamp_float((base_confidence * 0.88) + (overlap * 0.12), 0.0, 1.0)

    def _context_window(self, tokens: Sequence[OrthographyToken], index: int, *, external_context: Optional[Sequence[str]] = None) -> List[str]:
        values = [ensure_text(item) for item in ensure_list(external_context or []) if ensure_text(item)]
        if self.max_context_window <= 0:
            return values
        start = max(0, index - self.max_context_window)
        end = min(len(tokens), index + self.max_context_window + 1)
        values.extend(token.text for token in tokens[start:end] if token.index != index and token.kind == "word")
        return values

    def _make_edit(
        self,
        token: OrthographyToken,
        replacement: str,
        edit_type: str,
        confidence: float,
        *,
        rule_id: Optional[str] = None,
        suggestion_source: Optional[str] = None,
        issues: Tuple[Any, ...] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> OrthographyEdit:
        return OrthographyEdit(
            original=token.text,
            replacement=replacement,
            source_span=token.span,
            edit_type=edit_type,
            confidence=clamp_float(confidence, 0.0, 1.0),
            rule_id=rule_id,
            suggestion_source=suggestion_source,
            issues=tuple(issues),
            metadata=dict(metadata or {}),
        )

    def _finalize_text(self, text: str) -> str:
        output = ensure_text(text)
        if self.collapse_whitespace and not self.preserve_whitespace:
            output = normalize_whitespace(output)
        if coerce_bool(self.op_config.get("normalize_spacing_around_punctuation", True), default=True):
            output = normalize_spacing_around_punctuation(output)
        return output

    def _confidence_check(self, original: str, corrected: str) -> bool:
        """Compatibility method retained for older callers."""
        if not corrected or original == corrected:
            return True
        edit_similarity_score = normalized_edit_similarity(original, corrected)
        phonetic_similarity = 0.0
        if hasattr(self.spellchecker, "_phonetic_similarity"):
            phonetic_similarity = self.spellchecker._phonetic_similarity(original, corrected)
        combined = (edit_similarity_score * 0.72) + (phonetic_similarity * 0.28)
        return combined >= self.auto_correction_confidence

    # ------------------------------------------------------------------
    # Diagnostics, memory, history, stats
    # ------------------------------------------------------------------
    def _add_issue(self, issue: Any) -> None:
        self.diagnostics.add(issue)

    def _record(self, action: str, **payload: Any) -> None:
        if not self.record_history:
            return
        self.history.append({"timestamp": utc_timestamp(), "action": action, "payload": json_safe(payload)})

    def _remember_result(self, result: OrthographyProcessingResult) -> None:
        if self.memory is None or not result.changed:
            return
        self.memory.remember(
            kind="note",
            key="orthography:last_correction",
            text=result.corrected_text,
            value=result.to_dict(),
            tags=("orthography", "correction"),
            confidence=1.0,
            salience=0.35,
            source="OrthographyProcessor",
        )

    def diagnostics_result(self) -> Any:
        return self.diagnostics

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        count = coerce_int(limit, default=20, minimum=1)
        return list(self.history)[-count:]

    def stats(self) -> OrthographyProcessorStats:
        return OrthographyProcessorStats(
            language=self.language,
            locale=self.locale,
            process_calls=int(self._stats["process_calls"]),
            correct_calls=int(self._stats["correct_calls"]),
            normalize_calls=int(self._stats["normalize_calls"]),
            contraction_expansions=int(self._stats["contraction_expansions"]),
            spelling_corrections=int(self._stats["spelling_corrections"]),
            locale_normalizations=int(self._stats["locale_normalizations"]),
            compound_edits=int(self._stats["compound_edits"]),
            diagnostics_count=len(self.diagnostics.issues),
            history_length=len(self.history),
            resources={
                "normalization_map_path": str(self.normalization_map_path) if self.normalization_map_path else None,
                "contraction_map_path": str(self.contraction_map_path) if self.contraction_map_path else None,
                "locale_variants_path": str(self.locale_variants_path) if self.locale_variants_path else None,
                "protected_terms": len(self.protected_terms),
                "normalization_entries": len(self.normalization_map),
                "contraction_entries": len(self.contraction_map),
                "locale_variant_entries": len(self.locale_variants.get(self.locale, {})),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "version": self.version,
            "language": self.language,
            "locale": self.locale,
            "stats": self.stats().to_dict(),
            "diagnostics": self.diagnostics.to_list(),
            "config": json_safe(self.op_config),
        }

    def __repr__(self) -> str:
        return f"<OrthographyProcessor version='{self.version}' locale='{self.locale}' auto_correct={self.enable_auto_correction}>"


if __name__ == "__main__":
    print("\n=== Running Orthography Processor ===\n")
    printer.status("TEST", "Orthography Processor initialized", "info")

    processor = OrthographyProcessor()

    sample_words = [
        "cant",
        "can't",
        "beleive",
        "recieve",
        "seperate",
        "programme",
        "theatre",
        "Unfortonally",
        "keyboard",
    ]

    word_results = []
    for sample in sample_words:
        word_results.append({"input": sample, "output": processor.correct(sample, context=["language", "module", "production"])})

    sample_text = "I can't beleive their programme in the theatre will recieve seperate reviews tommorow."
    result = processor.process_text(sample_text, auto_correct=True)
    compatibility_text = processor.batch_process(sample_text)

    printer.pretty("WORD_RESULTS", word_results, "success")
    printer.pretty("PROCESS_RESULT", result.to_dict(), "success")
    printer.pretty("BATCH_PROCESS", {"text": compatibility_text}, "success")
    printer.pretty("NORMALIZE", {"colour": processor.normalize("colour"), "centre": processor.normalize("centre")}, "success")
    printer.pretty("CONTRACTION", {"can't": processor.expand_contractions("can't")}, "success")
    printer.pretty("STATS", processor.stats().to_dict(), "success")
    printer.pretty("HISTORY", processor.recent_history(), "info")

    print("\n=== Test ran successfully ===\n")
