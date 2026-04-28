"""
Grammar Processor Module

Core Function:
Analyzes syntax and sentence structure after orthography and NLP processing.
It validates grammar, detects recoverable user-facing language issues, and
returns structured diagnostics that can help NLU disambiguation and NLG produce
clearer responses.

Responsibilities:
- Accept dependency-aware InputToken sentences from the agent pipeline.
- Provide adapters for NLP tokens and rule-engine dependency relations.
- Reconstruct readable sentence text without damaging punctuation spacing.
- Classify sentence type and detect fragments, missing subjects, and root issues.
- Apply grammar checks such as subject-verb agreement, article usage,
  determiner-number agreement, negation/auxiliary form issues, and repetition.
- Emit production diagnostics through language_error.py while preserving the
  existing GrammarAnalysisResult surface used by LanguageAgent.
- Reuse language_helpers.py and Rules instead of duplicating normalization,
  span, POS, serialization, article, or morphology helper logic.

Why it matters:
Grammar sits between structural NLP and semantic NLU. It should not block the
entire agent for minor language problems, but it must preserve enough detail for
correction suggestions, user-facing feedback, diagnostics, telemetry, and future
expansion into deeper syntax and style checks.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .modules.rules import Rules, DependencyRelation
from .utils.language_error import * # type: ignore
from .utils.language_helpers import * # type: ignore
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Grammar Processor")
printer = PrettyPrinter()

Span = Tuple[int, int]
SentenceIndex = int
TokenIndex = int


class GrammarSeverity(str, Enum):
    """Local severity labels retained for legacy GrammarIssue compatibility."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class InputToken:
    """
    Dependency-aware token expected by GrammarProcessor.

    The legacy required fields are preserved. Additional fields let expanded NLP
    modules pass morphology, UPOS/XPOS, confidence, and metadata without changing
    the current LanguageAgent adapter.
    """

    text: str
    lemma: str
    pos: str
    index: int
    head: int
    dep: str
    start_char_abs: int
    end_char_abs: int
    upos: Optional[str] = None
    xpos: Optional[str] = None
    morphology: Dict[str, Any] = field(default_factory=dict)
    sentence_index: int = 0
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = ensure_text(self.text)
        self.lemma = ensure_text(self.lemma or self.text).lower()
        self.pos = ensure_text(self.pos or "X").upper()
        self.upos = normalize_pos_tag(self.upos or self.pos, default="X") or "X"
        self.dep = ensure_text(self.dep or "dep")
        self.index = int(self.index)
        self.head = int(self.head)
        self.start_char_abs = int(self.start_char_abs)
        self.end_char_abs = int(self.end_char_abs)
        if self.confidence is not None:
            self.confidence = clamp_float(self.confidence, 0.0, 1.0)

    @property
    def lower(self) -> str:
        return self.text.lower()

    @property
    def normalized_dep(self) -> str:
        dep = self.dep.lower().strip()
        if dep == "root":
            return "root"
        if dep == "nsubjpass":
            return "nsubj:pass"
        return dep

    @property
    def span_closed(self) -> Span:
        start = max(0, self.start_char_abs)
        end = max(start, self.end_char_abs)
        return (start, end)

    @property
    def span_half_open(self) -> Span:
        start, end = self.span_closed
        return (start, end + 1)

    @property
    def is_punct(self) -> bool:
        return self.upos == "PUNCT" or is_punctuation(self.text)

    @property
    def is_content(self) -> bool:
        return self.upos in {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "NUM"}

    def morph(self, key: str, default: Any = None) -> Any:
        return self.morphology.get(key, default)

    def to_rule_token_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "lemma": self.lemma,
            "pos": self.pos,
            "upos": self.upos,
            "xpos": self.xpos,
            "id": self.index,
            "index": self.index,
            "head": self.head,
            "dep": self.dep,
            "start_char": self.start_char_abs,
            "end_char": self.end_char_abs + 1,
            "morphology": dict(self.morphology),
        }

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self), drop_empty=True)


@dataclass
class DiagnosticGrammarIssue(LanguageIssue):
    """Grammar-specific diagnostic issue for structured grammar feedback."""
    
    def __init__(
        self,
        code: Union[str, LanguageErrorCode],
        message: str,
        *,
        stage: Union[str, LanguageStage] = LanguageStage.GRAMMAR,
        category: Union[str, ErrorCategory] = ErrorCategory.LINGUISTIC_ANALYSIS,
        severity: Union[str, Severity] = Severity.WARNING,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("stage", stage)
        kwargs.setdefault("category", category)
        kwargs.setdefault("severity", severity)
        super().__init__(code=code, message=message, **kwargs)

@dataclass
class GrammarIssue:
    """
    Legacy-compatible grammar issue record with production metadata.

    Existing callers can still read `.description`, `.severity`, `.suggestion`,
    `.source_text_char_span`, and `.source_sentence_token_indices_span`.
    """

    description: str
    source_text_char_span: Span
    source_sentence_token_indices_span: Span
    severity: str = "warning"
    suggestion: Optional[str] = None
    code: str = LanguageErrorCode.GRAMMAR_RULE_COVERAGE.value
    rule_id: Optional[str] = None
    confidence: float = 0.75
    sentence_index: int = 0
    token_indices: Tuple[int, ...] = ()
    autofixable: bool = False
    replacement: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        return self.severity.lower() in {"error", "critical"}

    def to_diagnostic(self, *, source_text: Optional[str] = None, language: str = "en", locale: Optional[str] = None) -> DiagnosticGrammarIssue:
        start, inclusive_end = self.source_text_char_span
        half_open = (max(0, int(start)), max(0, int(inclusive_end)) + 1)
        severity = Severity.ERROR if self.severity.lower() == "error" else Severity.CRITICAL if self.severity.lower() == "critical" else Severity.INFO if self.severity.lower() == "info" else Severity.WARNING
        return DiagnosticGrammarIssue(
            code=self.code,
            message=self.description,
            severity=severity,
            module="GrammarProcessor",
            source_text=source_text,
            source_span=half_open,
            token_span=self.source_sentence_token_indices_span,
            sentence_index=self.sentence_index,
            token_indices=self.token_indices or tuple(range(self.source_sentence_token_indices_span[0], self.source_sentence_token_indices_span[1] + 1)),
            suggestion=self.suggestion,
            confidence=self.confidence,
            rule_id=self.rule_id,
            language=language,
            locale=locale,
            autofixable=self.autofixable,
            recoverable=True,
            details={**self.details, "replacement": self.replacement},
        )

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "description": self.description,
                "source_text_char_span": list(self.source_text_char_span),
                "source_sentence_token_indices_span": list(self.source_sentence_token_indices_span),
                "severity": self.severity,
                "suggestion": self.suggestion,
                "code": self.code,
                "rule_id": self.rule_id,
                "confidence": self.confidence,
                "sentence_index": self.sentence_index,
                "token_indices": list(self.token_indices),
                "autofixable": self.autofixable,
                "replacement": self.replacement,
                "details": json_safe(self.details),
            },
            drop_empty=True,
        )


@dataclass(frozen=True)
class SentenceGrammarAnalysis:
    """Structured sentence-level grammar result."""

    text: str
    type: str
    issues: Tuple[GrammarIssue, ...]
    original_sentence_token_count: int
    sentence_index: int = 0
    token_count: int = 0
    root_indices: Tuple[int, ...] = ()
    dependency_count: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_grammatical(self) -> bool:
        return not any(issue.is_blocking for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "issues": list(self.issues),
            "issue_dicts": [issue.to_dict() for issue in self.issues],
            "original_sentence_token_count": self.original_sentence_token_count,
            "sentence_index": self.sentence_index,
            "token_count": self.token_count,
            "root_indices": list(self.root_indices),
            "dependency_count": self.dependency_count,
            "confidence": self.confidence,
            "is_grammatical": self.is_grammatical,
            "metadata": json_safe(self.metadata),
        }


@dataclass
class GrammarAnalysisResult:
    """Grammar analysis result preserved for LanguageAgent compatibility."""

    original_text_snippet: str
    is_grammatical: bool
    sentence_analyses: List[Dict[str, Any]]
    issues: List[GrammarIssue] = field(default_factory=list)
    diagnostics: List[LanguageIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_id: str = field(default_factory=lambda: generate_language_id("grammar_analysis"))
    created_at: str = field(default_factory=lambda: utc_timestamp())

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.is_blocking)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "created_at": self.created_at,
            "original_text_snippet": self.original_text_snippet,
            "is_grammatical": self.is_grammatical,
            "sentence_analyses": [
                {
                    **{key: value for key, value in sentence.items() if key != "issues"},
                    "issues": [issue.to_dict() if hasattr(issue, "to_dict") else json_safe(issue) for issue in sentence.get("issues", [])],
                }
                for sentence in self.sentence_analyses
            ],
            "issues": [issue.to_dict() for issue in self.issues],
            "diagnostics": [issue.to_dict() for issue in self.diagnostics],
            "issue_count": self.issue_count,
            "error_count": self.error_count,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class GrammarProcessorStats:
    """Runtime stats for observability and test validation."""

    version: str
    analyses_run: int
    total_sentences: int
    total_tokens: int
    total_issues: int
    diagnostics_count: int
    history_length: int
    enabled_checks: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "analyses_run": self.analyses_run,
            "total_sentences": self.total_sentences,
            "total_tokens": self.total_tokens,
            "total_issues": self.total_issues,
            "diagnostics_count": self.diagnostics_count,
            "history_length": self.history_length,
            "enabled_checks": list(self.enabled_checks),
        }


class GrammarProcessor:
    """
    Production grammar analyzer for the language pipeline.

    The processor is intentionally conservative: it reports high-confidence
    grammar problems and recoverable warnings while preserving compatibility
    with the current LanguageAgent pipeline.
    """

    VERSION = "2.0"
    SUBJECT_DEPS = {"nsubj", "nsubj:pass", "nsubjpass", "csubj", "expl"}
    VERBAL_UPOS = {"VERB", "AUX"}
    NOMINAL_UPOS = {"NOUN", "PROPN", "PRON"}
    CONTENT_UPOS = {"NOUN", "PROPN", "PRON", "VERB", "AUX", "ADJ", "ADV", "NUM"}

    def __init__(self) -> None:
        self.config = load_global_config()
        self.wordlist_path = self.config.get("main_wordlist_path")
        self.grammar_config = get_config_section("grammar_processor") or {}
        self.version = ensure_text(self.grammar_config.get("version", self.VERSION))
        self.language = ensure_text(self.grammar_config.get("language", "en"))
        self.locale = ensure_text(self.grammar_config.get("locale", "en-US"))
        self.pos_map = self.grammar_config.get("pos_map_path")

        self.max_sentences = coerce_int(self.grammar_config.get("max_sentences", 128), default=128, minimum=1)
        self.max_tokens_per_sentence = coerce_int(self.grammar_config.get("max_tokens_per_sentence", 256), default=256, minimum=1)
        self.strict_input_contract = coerce_bool(self.grammar_config.get("strict_input_contract", False), default=False)
        self.enable_dependency_enrichment = coerce_bool(self.grammar_config.get("enable_dependency_enrichment", True), default=True)
        self.enable_subject_verb_agreement = coerce_bool(self.grammar_config.get("enable_subject_verb_agreement", True), default=True)
        self.enable_article_usage = coerce_bool(self.grammar_config.get("enable_article_usage", True), default=True)
        self.enable_sentence_completeness = coerce_bool(self.grammar_config.get("enable_sentence_completeness", True), default=True)
        self.enable_dependency_integrity = coerce_bool(self.grammar_config.get("enable_dependency_integrity", True), default=True)
        self.enable_determiner_number = coerce_bool(self.grammar_config.get("enable_determiner_number", True), default=True)
        self.enable_negation_auxiliary = coerce_bool(self.grammar_config.get("enable_negation_auxiliary", True), default=True)
        self.enable_repetition_check = coerce_bool(self.grammar_config.get("enable_repetition_check", True), default=True)
        self.enable_fragment_detection = coerce_bool(self.grammar_config.get("enable_fragment_detection", True), default=True)
        self.allow_singular_they = coerce_bool(self.grammar_config.get("allow_singular_they", True), default=True)
        self.treat_fragments_as_errors = coerce_bool(self.grammar_config.get("treat_fragments_as_errors", False), default=False)
        self.minimum_fragment_content_tokens = coerce_int(self.grammar_config.get("minimum_fragment_content_tokens", 2), default=2, minimum=1)
        self.default_issue_confidence = coerce_float(self.grammar_config.get("default_issue_confidence", 0.78), default=0.78, minimum=0.0, maximum=1.0)
        self.record_history = coerce_bool(self.grammar_config.get("record_history", True), default=True)
        self.history_limit = coerce_int(self.grammar_config.get("history_limit", 200), default=200, minimum=1)

        self.pronoun_features = self._load_pronoun_features()
        self.plural_exceptions = set(ensure_text(item).lower() for item in ensure_list(self.grammar_config.get("plural_exceptions", ["series", "species", "news", "mathematics", "physics", "economics"])))
        self.plural_only_nouns = set(ensure_text(item).lower() for item in ensure_list(self.grammar_config.get("plural_only_nouns", ["scissors", "trousers", "pants", "clothes"])))
        self.mass_nouns = set(ensure_text(item).lower() for item in ensure_list(self.grammar_config.get("mass_nouns", ["information", "advice", "equipment", "furniture", "research", "software", "knowledge"])))
        self.article_ignore_pos = set(ensure_text(item).upper() for item in ensure_list(self.grammar_config.get("article_ignore_pos", ["PUNCT", "SYM"])))
        self.auxiliaries_requiring_base = set(ensure_text(item).lower() for item in ensure_list(self.grammar_config.get("auxiliaries_requiring_base", ["do", "does", "did", "can", "could", "may", "might", "must", "shall", "should", "will", "would"])))

        self.rule_engine = Rules()
        self.diagnostics = LanguageDiagnostics()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)
        self.analyses_run = 0
        self.total_sentences = 0
        self.total_tokens = 0
        self.total_issues = 0

        logger.info("Grammar Processor initialized with production diagnostics.")
        printer.status("INIT", "Grammar Processor initialized", "success")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_text(self, sentences: Sequence[Any], full_text_snippet: Optional[str] = None) -> GrammarAnalysisResult:
        """
        Analyze sentence-token input from the language pipeline.

        `sentences` may be:
        - List[List[InputToken]] from the current LanguageAgent.
        - List[InputToken] for a single sentence.
        - List[List[dict/object]] where token-like objects expose text/POS/dep.
        """
        start_ms = monotonic_ms()
        normalized_sentences = self._coerce_sentences(sentences, full_text_snippet=full_text_snippet)
        if not normalized_sentences:
            logger.warning("Received empty sentence list for grammar analysis.")
            return GrammarAnalysisResult(
                original_text_snippet=full_text_snippet or "",
                is_grammatical=True,
                sentence_analyses=[],
                metadata={"duration_ms": elapsed_ms(start_ms), "empty": True},
            )

        if len(normalized_sentences) > self.max_sentences:
            raise GrammarError(
                DiagnosticGrammarIssue(
                    code=LanguageErrorCode.GRAMMAR_INPUT_CONTRACT,
                    message="Too many sentences were provided to GrammarProcessor.",
                    severity=Severity.ERROR,
                    module="GrammarProcessor",
                    details={"max_sentences": self.max_sentences, "received": len(normalized_sentences)},
                ),
                recoverable=True,
            )

        all_issues: List[GrammarIssue] = []
        diagnostics = LanguageDiagnostics()
        sentence_analyses: List[Dict[str, Any]] = []

        for sentence_index, raw_sentence_tokens in enumerate(normalized_sentences):
            sentence_tokens = self._prepare_sentence(raw_sentence_tokens, sentence_index=sentence_index, full_text=full_text_snippet)
            if not sentence_tokens:
                continue
            sentence_text = self._reconstruct_sentence_text(sentence_tokens)
            sentence_type = self._classify_sentence_type(sentence_tokens)
            root_indices = tuple(token.index for token in sentence_tokens if token.normalized_dep == "root")

            current_issues: List[GrammarIssue] = []
            for check in self._enabled_checks():
                current_issues.extend(check(sentence_tokens, sentence_text, sentence_type, sentence_index))

            for issue in current_issues:
                diagnostics.add(issue.to_diagnostic(source_text=full_text_snippet or sentence_text, language=self.language, locale=self.locale))

            all_issues.extend(current_issues)
            analysis = SentenceGrammarAnalysis(
                text=sentence_text,
                type=sentence_type,
                issues=tuple(current_issues),
                original_sentence_token_count=len(sentence_tokens),
                sentence_index=sentence_index,
                token_count=len(sentence_tokens),
                root_indices=root_indices,
                dependency_count=sum(1 for token in sentence_tokens if token.dep),
                confidence=self._sentence_confidence(sentence_tokens, current_issues),
                metadata={
                    "content_token_count": sum(1 for token in sentence_tokens if token.is_content),
                    "has_explicit_subject": self._has_subject(sentence_tokens),
                    "has_finite_verb": self._has_finite_verb(sentence_tokens),
                },
            )
            sentence_analyses.append(analysis.to_dict())

        self.diagnostics.extend(diagnostics.issues)
        self.analyses_run += 1
        self.total_sentences += len(sentence_analyses)
        self.total_tokens += sum(item.get("token_count", 0) for item in sentence_analyses)
        self.total_issues += len(all_issues)

        display_text = full_text_snippet or self._build_display_text(normalized_sentences)
        result = GrammarAnalysisResult(
            original_text_snippet=display_text or "N/A",
            is_grammatical=not any(issue.is_blocking for issue in all_issues),
            sentence_analyses=sentence_analyses,
            issues=all_issues,
            diagnostics=list(diagnostics.issues),
            metadata={
                "duration_ms": elapsed_ms(start_ms),
                "sentence_count": len(sentence_analyses),
                "token_count": self.total_tokens,
                "enabled_checks": [check.__name__ for check in self._enabled_checks()],
                "version": self.version,
            },
        )
        self._record("analyze_text", result=result.to_dict())
        return result

    def analyze_tokens(
        self,
        tokens: Sequence[Any],
        dependencies: Optional[Sequence[DependencyRelation]] = None,
        *,
        full_text_snippet: Optional[str] = None,
        sentence_index: int = 0,
    ) -> GrammarAnalysisResult:
        """Adapter for callers that have NLP tokens and optional dependency relations."""
        input_tokens = self.build_input_tokens(tokens, dependencies=dependencies, full_text=full_text_snippet, sentence_index=sentence_index)
        return self.analyze_text([input_tokens], full_text_snippet=full_text_snippet)

    def build_input_tokens(
        self,
        tokens: Sequence[Any],
        dependencies: Optional[Sequence[DependencyRelation]] = None,
        *,
        full_text: Optional[str] = None,
        sentence_index: int = 0,
    ) -> List[InputToken]:
        """Build GrammarProcessor InputTokens from NLP tokens and rule dependencies."""
        snapshots = infer_token_offsets(full_text or tokens_to_text(tokens), tokens) if full_text else [self._token_to_snapshot_like(token, i) for i, token in enumerate(tokens)]
        relation_by_dependent: Dict[int, DependencyRelation] = {}
        for relation in dependencies or []:
            relation_by_dependent[int(relation.dependent_index)] = relation

        input_tokens: List[InputToken] = []
        for position, token in enumerate(snapshots):
            text = token_text(token)
            index = token_index(token, default=position)
            relation = relation_by_dependent.get(index)
            dep = relation.relation if relation else token_dep(token) or ("root" if self._looks_like_root_token(token, position) else "dep")
            head = int(relation.head_index) if relation else token_head(token)
            if head is None:
                head = index if dep == "root" else index
            span = token_span(token)
            if span is None:
                span = (0, max(0, len(text)))
            input_tokens.append(
                InputToken(
                    text=text,
                    lemma=token_lemma(token) or text.lower(),
                    pos=token_pos(token) or self._infer_pos(text),
                    index=index,
                    head=head,
                    dep=dep,
                    start_char_abs=span[0],
                    end_char_abs=max(span[0], span[1] - 1),
                    sentence_index=sentence_index,
                    morphology=dict(get_attr_or_key(token, "morphology", {}) or {}),
                    metadata={"source_type": type(token).__name__},
                )
            )
        return input_tokens

    def diagnostics_result(self) -> LanguageResult[Dict[str, Any]]:
        return LanguageResult(data=self.stats().to_dict(), issues=list(self.diagnostics.issues), metadata={"module": "GrammarProcessor"})

    def stats(self) -> GrammarProcessorStats:
        return GrammarProcessorStats(
            version=self.version,
            analyses_run=self.analyses_run,
            total_sentences=self.total_sentences,
            total_tokens=self.total_tokens,
            total_issues=self.total_issues,
            diagnostics_count=len(self.diagnostics.issues),
            history_length=len(self.history),
            enabled_checks=tuple(check.__name__ for check in self._enabled_checks()),
        )

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    def _coerce_sentences(self, sentences: Sequence[Any], *, full_text_snippet: Optional[str] = None) -> List[List[Any]]:
        if sentences is None:
            return []
        if isinstance(sentences, (str, bytes, bytearray)):
            raise PipelineContractError(
                "GrammarProcessor requires dependency-aware tokens, not raw text.",
                expected="Sequence[Sequence[InputToken]] or Sequence[InputToken]",
                received=type(sentences).__name__,
                details={"text_preview": ensure_text(sentences)[:160]},
            )
        seq = list(ensure_sequence(sentences, field_name="sentences", allow_none=True))
        if not seq:
            return []
        if self._looks_like_token(seq[0]):
            return [seq]
        normalized: List[List[Any]] = []
        for item in seq:
            if item is None:
                continue
            if isinstance(item, (str, bytes, bytearray)):
                raise PipelineContractError(
                    "GrammarProcessor received a raw string inside sentence input.",
                    expected="Sequence[InputToken]",
                    received=type(item).__name__,
                )
            normalized.append(list(ensure_sequence(item, field_name="sentence", allow_none=True)))
        return normalized

    def _prepare_sentence(self, tokens: Sequence[Any], *, sentence_index: int, full_text: Optional[str]) -> List[InputToken]:
        if len(tokens) > self.max_tokens_per_sentence:
            raise GrammarError(
                DiagnosticGrammarIssue(
                    code=LanguageErrorCode.GRAMMAR_INPUT_CONTRACT,
                    message="Too many tokens were provided for one sentence.",
                    severity=Severity.ERROR,
                    module="GrammarProcessor",
                    details={"max_tokens_per_sentence": self.max_tokens_per_sentence, "received": len(tokens)},
                ),
                recoverable=True,
            )
        prepared = [self._coerce_token(token, position=position, sentence_index=sentence_index, full_text=full_text) for position, token in enumerate(tokens)]
        if self.enable_dependency_enrichment:
            prepared = self._enrich_dependencies(prepared)
        return prepared

    def _coerce_token(self, token: Any, *, position: int, sentence_index: int, full_text: Optional[str]) -> InputToken:
        if isinstance(token, InputToken):
            token.sentence_index = sentence_index
            return token
        text = token_text(token)
        if not text:
            if self.strict_input_contract:
                raise PipelineContractError("Grammar token is missing text.", expected="token.text", received=json_safe(token))
            text = ""
        lemma = token_lemma(token) or text.lower()
        pos = token_pos(token) or self._infer_pos(text)
        index = token_index(token, default=position)
        dep = token_dep(token) or get_attr_or_key(token, "relation", None) or "dep"
        head = token_head(token)
        if head is None:
            head = index if str(dep).lower() == "root" else index
        span = token_span(token)
        if span is None:
            span = self._estimate_span(text, position, full_text)
        morphology = get_attr_or_key(token, "morphology", get_attr_or_key(token, "feats", {}))
        return InputToken(
            text=text,
            lemma=lemma,
            pos=pos or "X",
            index=index,
            head=head,
            dep=dep,
            start_char_abs=span[0],
            end_char_abs=max(span[0], span[1] - 1),
            upos=get_attr_or_key(token, "upos", pos),
            xpos=get_attr_or_key(token, "xpos", get_attr_or_key(token, "tag", None)),
            morphology=dict(morphology or {}) if isinstance(morphology, Mapping) else {},
            sentence_index=sentence_index,
            confidence=get_attr_or_key(token, "confidence", None),
            metadata={"source_type": type(token).__name__, "raw": json_safe(token)},
        )

    def _token_to_snapshot_like(self, token: Any, position: int) -> Any:
        return token

    def _looks_like_token(self, value: Any) -> bool:
        if isinstance(value, InputToken):
            return True
        if isinstance(value, Mapping):
            return any(key in value for key in ("text", "token", "word", "lemma", "pos", "upos"))
        return any(hasattr(value, attr) for attr in ("text", "lemma", "pos", "upos"))

    def _looks_like_root_token(self, token: Any, position: int) -> bool:
        dep = token_dep(token)
        if dep and dep.lower() == "root":
            return True
        return position == 0 and (token_pos(token) in self.VERBAL_UPOS)

    def _estimate_span(self, text: str, position: int, full_text: Optional[str]) -> Span:
        if not full_text:
            start = position
            return (start, start + len(text))
        found = full_text.find(text)
        if found >= 0:
            return (found, found + len(text))
        start = min(len(full_text), position)
        return (start, min(len(full_text), start + len(text)))

    def _infer_pos(self, text: str) -> str:
        value = ensure_text(text)
        lower = value.lower()
        if not value:
            return "X"
        if is_punctuation(value):
            return "PUNCT"
        if is_numeric_token(value):
            return "NUM"
        if lower in {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}:
            return "PRON"
        if lower in {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"}:
            return "DET"
        if lower in self.auxiliaries_requiring_base or lower in {"am", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had"}:
            return "AUX"
        return "NOUN" if value.isalpha() else "X"

    def _enrich_dependencies(self, tokens: List[InputToken]) -> List[InputToken]:
        root_count = sum(1 for token in tokens if token.normalized_dep == "root")
        generic_count = sum(1 for token in tokens if token.normalized_dep in {"dep", ""})
        if root_count == 1 and generic_count == 0:
            return tokens
        rule_result = self.rule_engine.apply([token.to_rule_token_dict() for token in tokens])
        relation_map: Dict[int, DependencyRelation] = {int(rel.dependent_index): rel for rel in rule_result.relations}
        updated: List[InputToken] = []
        for token in tokens:
            relation = relation_map.get(token.index)
            if relation and (token.normalized_dep in {"dep", ""} or relation.relation == "root"):
                updated.append(
                    InputToken(
                        **{
                            **token.to_dict(),
                            "head": int(relation.head_index) if relation.head != "ROOT" else token.index,
                            "dep": relation.relation,
                            "metadata": {**token.metadata, "dependency_enriched": True, "dependency_rule_id": relation.rule_id},
                            "confidence": relation.confidence,
                        }
                    )
                )
            else:
                updated.append(token)
        return updated

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    def _enabled_checks(self) -> List[Any]:
        checks: List[Any] = []
        if self.enable_dependency_integrity:
            checks.append(self._check_dependency_integrity)
        if self.enable_sentence_completeness:
            checks.append(self._check_sentence_completeness)
        if self.enable_subject_verb_agreement:
            checks.append(self._check_subject_verb_agreement)
        if self.enable_article_usage:
            checks.append(self._check_article_usage)
        if self.enable_determiner_number:
            checks.append(self._check_determiner_number)
        if self.enable_negation_auxiliary:
            checks.append(self._check_negation_auxiliary)
        if self.enable_repetition_check:
            checks.append(self._check_repetition)
        return checks

    def _check_dependency_integrity(self, sentence_tokens: List[InputToken], sentence_text: Optional[str] = None, sentence_type: Optional[str] = None, sentence_index: int = 0) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        indices = {token.index for token in sentence_tokens}
        roots = [token for token in sentence_tokens if token.normalized_dep == "root"]
        if not roots and any(token.upos in self.VERBAL_UPOS for token in sentence_tokens):
            issues.append(
                self._make_issue(
                    code=LanguageErrorCode.DEP_NO_ROOT,
                    description="No dependency root was found for a sentence with verbal content.",
                    tokens=sentence_tokens,
                    severity="warning",
                    suggestion="Re-run dependency parsing or mark the main predicate as root.",
                    rule_id="grammar.dep.no_root",
                    confidence=0.7,
                    sentence_index=sentence_index,
                )
            )
        if len(roots) > 1:
            issues.append(
                self._make_issue(
                    code=LanguageErrorCode.DEP_MULTIPLE_ROOTS,
                    description="Multiple dependency roots were found in the same sentence.",
                    tokens=roots,
                    severity="warning",
                    suggestion="Review sentence segmentation or dependency attachment.",
                    rule_id="grammar.dep.multiple_roots",
                    confidence=0.72,
                    sentence_index=sentence_index,
                )
            )
        for token in sentence_tokens:
            if token.normalized_dep != "root" and token.head not in indices:
                issues.append(
                    self._make_issue(
                        code=LanguageErrorCode.GRAMMAR_INPUT_CONTRACT,
                        description=f"Token '{token.text}' references a missing dependency head index {token.head}.",
                        tokens=[token],
                        severity="warning",
                        suggestion="Ensure dependency heads use the same token index scheme as grammar input tokens.",
                        rule_id="grammar.dep.missing_head",
                        confidence=0.8,
                        sentence_index=sentence_index,
                        details={"missing_head": token.head},
                    )
                )
        seen = Counter((token.head, token.normalized_dep, token.index) for token in sentence_tokens)
        for key, count in seen.items():
            if count > 1:
                offenders = [token for token in sentence_tokens if (token.head, token.normalized_dep, token.index) == key]
                issues.append(
                    self._make_issue(
                        code=LanguageErrorCode.DEP_DUPLICATE_RELATION,
                        description="Duplicate dependency relation metadata was detected.",
                        tokens=offenders,
                        severity="info",
                        suggestion="Deduplicate dependency edges before grammar analysis.",
                        rule_id="grammar.dep.duplicate_relation",
                        confidence=0.62,
                        sentence_index=sentence_index,
                    )
                )
        return issues

    def _check_sentence_completeness(self, sentence_tokens: List[InputToken], sentence_text: Optional[str] = None, sentence_type: Optional[str] = None, sentence_index: int = 0) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        content_tokens = [token for token in sentence_tokens if token.upos in self.CONTENT_UPOS]
        has_finite = self._has_finite_verb(sentence_tokens)
        has_subject = self._has_subject(sentence_tokens)
        resolved_sentence_type = sentence_type or self._classify_sentence_type(sentence_tokens)
        if self.enable_fragment_detection and resolved_sentence_type == "fragment" and len(content_tokens) >= self.minimum_fragment_content_tokens and not has_finite:
            issues.append(
                self._make_issue(
                    code=LanguageErrorCode.GRAMMAR_FRAGMENT,
                    description="This appears to be a sentence fragment without a finite predicate.",
                    tokens=content_tokens,
                    severity="error" if self.treat_fragments_as_errors else "warning",
                    suggestion="Add a finite verb or merge the fragment with a complete sentence.",
                    rule_id="grammar.syntax.fragment",
                    confidence=0.72,
                    sentence_index=sentence_index,
                    autofixable=False,
                )
            )
        if resolved_sentence_type == "declarative" and has_finite and not has_subject and not self._is_imperative_like(sentence_tokens):
            main_verb = self._main_verb(sentence_tokens)
            issues.append(
                self._make_issue(
                    code=LanguageErrorCode.GRAMMAR_MISSING_SUBJECT,
                    description="The sentence has a finite predicate but no clear subject.",
                    tokens=[main_verb] if main_verb else content_tokens[:1],
                    severity="warning",
                    suggestion="Add an explicit subject unless this is intended as an imperative.",
                    rule_id="grammar.syntax.missing_subject",
                    confidence=0.68,
                    sentence_index=sentence_index,
                )
            )
        if resolved_sentence_type == "interrogative" and not has_finite and len(content_tokens) > 1:
            issues.append(
                self._make_issue(
                    code=LanguageErrorCode.GRAMMAR_SENTENCE_TYPE_AMBIGUOUS,
                    description="This looks like a question but lacks a clear finite verb.",
                    tokens=content_tokens,
                    severity="warning",
                    suggestion="Add an auxiliary or main verb to make the question explicit.",
                    rule_id="grammar.sentence_type.interrogative_no_verb",
                    confidence=0.66,
                    sentence_index=sentence_index,
                )
            )
        return issues

    def _check_subject_verb_agreement(self, sentence_tokens: List[InputToken], sentence_text: Optional[str] = None, sentence_type: Optional[str] = None, sentence_index: int = 0) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        for verb in sentence_tokens:
            if verb.upos not in self.VERBAL_UPOS:
                continue
            if verb.normalized_dep in {"aux", "aux:pass", "cop", "mark"} and not self._acts_as_predicate(verb, sentence_tokens):
                continue
            if not self._looks_present_finite(verb, sentence_tokens):
                continue
            subjects = self._subjects_for_verb(sentence_tokens, verb)
            for subject in subjects:
                number, person = self._get_token_number_and_person(subject)
                expected_forms = self._expected_present_forms(verb.lemma or verb.lower, number, person)
                if not expected_forms:
                    continue
                current = verb.lower
                if current in expected_forms:
                    continue
                if self._is_contraction_equivalent(current, expected_forms):
                    continue
                suggestion = self._preserve_case(verb.text, sorted(expected_forms)[0])
                issues.append(
                    self._make_issue(
                        code=LanguageErrorCode.GRAMMAR_SUBJECT_VERB_AGREEMENT,
                        description=(
                            f"Potential subject-verb agreement error: subject '{subject.text}' "
                            f"({number}/{person}) does not agree with verb '{verb.text}'."
                        ),
                        tokens=[subject, verb],
                        severity="error",
                        suggestion=f"Use '{suggestion}' instead of '{verb.text}'.",
                        rule_id="grammar.agreement.subject_verb.present",
                        confidence=0.86,
                        sentence_index=sentence_index,
                        replacement=suggestion,
                        autofixable=True,
                        details={
                            "subject": subject.text,
                            "verb": verb.text,
                            "lemma": verb.lemma,
                            "expected_forms": sorted(expected_forms),
                            "number": number,
                            "person": person,
                        },
                    )
                )
        return issues

    def _check_article_usage(self, sentence_tokens: List[InputToken], sentence_text: Optional[str] = None, sentence_type: Optional[str] = None, sentence_index: int = 0) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        for token in sentence_tokens:
            if token.lower not in {"a", "an"} or token.upos not in {"DET", "X"}:
                continue
            phrase_tokens = self._article_head_phrase(sentence_tokens, token)
            if not phrase_tokens:
                continue
            phrase = tokens_to_text(phrase_tokens)
            expected = choose_indefinite_article(phrase)
            if token.lower == expected:
                continue
            replacement = self._preserve_case(token.text, expected)
            issues.append(
                self._make_issue(
                    code=LanguageErrorCode.GRAMMAR_ARTICLE_USAGE,
                    description=f"Incorrect article: '{token.text}' is unlikely before the sound in '{phrase}'.",
                    tokens=[token, *phrase_tokens[:1]],
                    severity="error",
                    suggestion=f"Use '{replacement}' before '{phrase}'.",
                    rule_id="grammar.article.indefinite_sound",
                    confidence=0.9,
                    sentence_index=sentence_index,
                    replacement=replacement,
                    autofixable=True,
                    details={"article": token.text, "head_phrase": phrase, "expected_article": expected},
                )
            )
        return issues

    def _check_determiner_number(self, sentence_tokens: List[InputToken], sentence_text: Optional[str] = None, sentence_type: Optional[str] = None, sentence_index: int = 0) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        singular_dets = {"this", "that", "a", "an"}
        plural_dets = {"these", "those", "many", "few", "several"}
        for token in sentence_tokens:
            if token.upos != "DET" or token.lower not in singular_dets | plural_dets:
                continue
            head = self._head_for_token(sentence_tokens, token) or self._next_nominal(sentence_tokens, token.index)
            if not head or head.upos not in {"NOUN", "PROPN"}:
                continue
            number, _person = self._get_token_number_and_person(head)
            if token.lower in singular_dets and number == "plural":
                replacement = "these" if token.lower == "this" else "those" if token.lower == "that" else None
                issues.append(
                    self._make_issue(
                        code=LanguageErrorCode.GRAMMAR_RULE_COVERAGE,
                        description=f"Determiner '{token.text}' may not agree in number with plural noun '{head.text}'.",
                        tokens=[token, head],
                        severity="warning",
                        suggestion=f"Consider '{replacement}' or a singular noun form." if replacement else "Use a plural-compatible determiner or singular noun form.",
                        rule_id="grammar.agreement.determiner_number",
                        confidence=0.72,
                        sentence_index=sentence_index,
                        replacement=replacement,
                        autofixable=bool(replacement),
                    )
                )
            if token.lower in plural_dets and number == "singular" and head.lower not in self.mass_nouns:
                replacement = "this" if token.lower == "these" else "that" if token.lower == "those" else None
                issues.append(
                    self._make_issue(
                        code=LanguageErrorCode.GRAMMAR_RULE_COVERAGE,
                        description=f"Determiner '{token.text}' may not agree in number with singular noun '{head.text}'.",
                        tokens=[token, head],
                        severity="warning",
                        suggestion=f"Consider '{replacement}' or a plural noun form." if replacement else "Use a singular-compatible determiner or plural noun form.",
                        rule_id="grammar.agreement.determiner_number",
                        confidence=0.7,
                        sentence_index=sentence_index,
                        replacement=replacement,
                        autofixable=bool(replacement),
                    )
                )
        return issues

    def _check_negation_auxiliary(self, sentence_tokens: List[InputToken], sentence_text: Optional[str] = None, sentence_type: Optional[str] = None, sentence_index: int = 0) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        for idx, token in enumerate(sentence_tokens[:-1]):
            if token.lower not in self.auxiliaries_requiring_base:
                continue
            following = self._nearest_after(sentence_tokens, token.index, lambda item: item.upos == "VERB")
            if following is None:
                continue
            if following.lower == following.lemma or following.morph("VerbForm") == "Inf":
                continue
            # After modal/do-support, the lexical verb should normally be bare/base.
            if following.lower.endswith(("s", "ed", "ing")) and following.lemma and following.lemma != following.lower:
                replacement = self._preserve_case(following.text, following.lemma)
                issues.append(
                    self._make_issue(
                        code=LanguageErrorCode.GRAMMAR_RULE_COVERAGE,
                        description=f"After auxiliary '{token.text}', the verb '{following.text}' should usually be in base form.",
                        tokens=[token, following],
                        severity="warning",
                        suggestion=f"Use '{replacement}' after '{token.text}'.",
                        rule_id="grammar.auxiliary.base_form",
                        confidence=0.76,
                        sentence_index=sentence_index,
                        replacement=replacement,
                        autofixable=True,
                    )
                )
        return issues

    def _check_repetition(self, sentence_tokens: List[InputToken], sentence_text: Optional[str] = None, sentence_type: Optional[str] = None, sentence_index: int = 0) -> List[GrammarIssue]:
        issues: List[GrammarIssue] = []
        ignored = set(ensure_text(item).lower() for item in ensure_list(self.grammar_config.get("allowed_repetitions", ["very", "no", "ha"])))
        for prev, current in zip(sentence_tokens, sentence_tokens[1:]):
            if prev.is_punct or current.is_punct:
                continue
            if prev.lower == current.lower and current.lower not in ignored:
                issues.append(
                    self._make_issue(
                        code=LanguageErrorCode.GRAMMAR_RULE_COVERAGE,
                        description=f"Repeated word '{current.text}' may be accidental.",
                        tokens=[prev, current],
                        severity="warning",
                        suggestion=f"Remove one instance of '{current.text}' if the repetition is unintended.",
                        rule_id="grammar.style.repeated_word",
                        confidence=0.82,
                        sentence_index=sentence_index,
                        replacement=current.text,
                        autofixable=False,
                    )
                )
        return issues

    # Legacy method names retained for current callers/tests.
    def _check_subject_verb_agreement_legacy(self, sentence_tokens: List[InputToken]) -> List[GrammarIssue]:
        return self._check_subject_verb_agreement(sentence_tokens, self._reconstruct_sentence_text(sentence_tokens), self._classify_sentence_type(sentence_tokens), 0)

    def _check_article_usage_legacy(self, sentence_tokens: List[InputToken]) -> List[GrammarIssue]:
        return self._check_article_usage(sentence_tokens, self._reconstruct_sentence_text(sentence_tokens), self._classify_sentence_type(sentence_tokens), 0)

    # ------------------------------------------------------------------
    # Linguistic helpers
    # ------------------------------------------------------------------
    def _reconstruct_sentence_text(self, sentence_tokens: List[InputToken]) -> str:
        return tokens_to_text(sentence_tokens)

    def _classify_sentence_type(self, sentence_tokens: List[InputToken]) -> str:
        if not sentence_tokens:
            return "empty"
        text = self._reconstruct_sentence_text(sentence_tokens)
        helper_type = classify_sentence_type(text, sentence_tokens)
        if helper_type == "fragment" and self._is_imperative_like(sentence_tokens):
            return "imperative"
        return helper_type

    def _sentence_confidence(self, tokens: List[InputToken], issues: List[GrammarIssue]) -> float:
        if not tokens:
            return 1.0
        token_conf = [token.confidence for token in tokens if token.confidence is not None]
        base = sum(token_conf) / len(token_conf) if token_conf else 0.82
        penalty = min(0.5, 0.08 * len([issue for issue in issues if issue.severity in {"error", "critical"}]) + 0.03 * len(issues))
        return clamp_float(base - penalty, 0.0, 1.0)

    def _has_subject(self, tokens: List[InputToken]) -> bool:
        return any(token.normalized_dep in self.SUBJECT_DEPS for token in tokens)

    def _has_finite_verb(self, tokens: List[InputToken]) -> bool:
        return any(token.upos in self.VERBAL_UPOS and self._looks_finite(token, tokens) for token in tokens)

    def _main_verb(self, tokens: List[InputToken]) -> Optional[InputToken]:
        roots = [token for token in tokens if token.normalized_dep == "root" and token.upos in self.VERBAL_UPOS]
        if roots:
            return roots[0]
        for token in tokens:
            if token.upos == "VERB":
                return token
        for token in tokens:
            if token.upos == "AUX":
                return token
        return None

    def _subjects_for_verb(self, tokens: List[InputToken], verb: InputToken) -> List[InputToken]:
        subjects = [token for token in tokens if token.head == verb.index and token.normalized_dep in self.SUBJECT_DEPS and token.upos in self.NOMINAL_UPOS]
        if subjects:
            return subjects
        if verb.normalized_dep == "root":
            preceding = [token for token in tokens if token.index < verb.index and token.upos in self.NOMINAL_UPOS]
            if preceding:
                return [preceding[-1]]
        return []

    def _looks_finite(self, token: InputToken, tokens: List[InputToken]) -> bool:
        if token.upos not in self.VERBAL_UPOS:
            return False
        verb_form = ensure_text(token.morph("VerbForm", "")).lower()
        if verb_form in {"inf", "ger", "part"}:
            return False
        previous = self._previous_token(tokens, token.index)
        if previous and previous.lower == "to" and previous.upos in {"PART", "ADP"}:
            return False
        if token.lower.endswith("ing") and token.lemma != "be":
            return False
        return True

    def _looks_present_finite(self, token: InputToken, tokens: List[InputToken]) -> bool:
        if not self._looks_finite(token, tokens):
            return False
        tense = ensure_text(token.morph("Tense", "")).lower()
        if tense == "past":
            return False
        if token.lemma == "be":
            return token.lower in {"am", "is", "are", "be"}
        if token.lower in {"was", "were", "had", "did"}:
            return False
        if token.lower.endswith("ed") and token.lemma != token.lower:
            return False
        return True

    def _acts_as_predicate(self, token: InputToken, tokens: List[InputToken]) -> bool:
        if token.normalized_dep == "root":
            return True
        return any(item.head == token.index and item.normalized_dep in self.SUBJECT_DEPS for item in tokens)

    def _is_imperative_like(self, tokens: List[InputToken]) -> bool:
        if not tokens:
            return False
        first_content = next((token for token in tokens if not token.is_punct), None)
        if not first_content:
            return False
        if first_content.lower in {"please", "kindly"}:
            next_content = next((token for token in tokens if token.index > first_content.index and token.upos in self.VERBAL_UPOS), None)
            return bool(next_content and not self._has_subject(tokens))
        return first_content.upos == "VERB" and not self._has_subject(tokens) and first_content.lower == first_content.lemma

    def _get_token_number_and_person(self, token: InputToken) -> Tuple[str, str]:
        lower = token.lower
        configured = self.pronoun_features.get(lower)
        if configured:
            return configured["number"], configured["person"]
        number = ensure_text(token.morph("Number", "")).lower()
        person_raw = ensure_text(token.morph("Person", "")).lower()
        if number in {"sing", "singular"}:
            resolved_number = "singular"
        elif number in {"plur", "plural"}:
            resolved_number = "plural"
        else:
            resolved_number = self._infer_nominal_number(token)
        if person_raw in {"1", "first", "1st"}:
            person = "1st"
        elif person_raw in {"2", "second", "2nd"}:
            person = "2nd"
        elif person_raw in {"3", "third", "3rd"}:
            person = "3rd"
        else:
            person = "3rd"
        return resolved_number, person

    def _infer_nominal_number(self, token: InputToken) -> str:
        lower = token.lower
        if lower in self.plural_only_nouns:
            return "plural"
        if lower in self.mass_nouns or lower in self.plural_exceptions:
            return "singular"
        if token.upos == "PROPN":
            return "singular"
        if token.upos == "PRON":
            return self.pronoun_features.get(lower, {"number": "singular"})["number"]
        if token.xpos in {"NNS", "NNPS"}:
            return "plural"
        if token.upos == "NOUN" and lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
            return "plural"
        return "singular"

    def _expected_present_forms(self, lemma: str, subject_number: str, subject_person: str) -> set[str]:
        base = normalize_text(lemma, lowercase=True)
        if not base:
            return set()
        if base == "be":
            if subject_number == "singular" and subject_person == "1st":
                return {"am"}
            if subject_number == "singular" and subject_person == "3rd":
                return {"is"}
            return {"are"}
        if subject_number == "singular" and subject_person == "3rd":
            return set(self.rule_engine.inflect_verb(base, tense="present", number="singular", person=3))
        return set(self.rule_engine.inflect_verb(base, tense="present", number="plural", person=1))

    def _suggest_verb_form(self, verb_lemma: str, subject_number: Optional[str], subject_person: Optional[str], verb_tense: str = "present") -> str:
        if verb_tense.lower() == "present":
            return next(iter(self._expected_present_forms(verb_lemma, subject_number or "singular", subject_person or "3rd")), verb_lemma)
        return verb_lemma

    def _is_contraction_equivalent(self, current: str, expected_forms: set[str]) -> bool:
        contraction_map = {
            "don't": "do",
            "doesn't": "does",
            "can't": "can",
            "won't": "will",
            "isn't": "is",
            "aren't": "are",
            "haven't": "have",
            "hasn't": "has",
        }
        return contraction_map.get(current) in expected_forms

    def _article_head_phrase(self, tokens: List[InputToken], article: InputToken) -> List[InputToken]:
        head = self._head_for_token(tokens, article)
        if head and head.upos not in self.article_ignore_pos:
            start = article.index + 1
            phrase = [token for token in tokens if start <= token.index <= head.index and token.upos not in self.article_ignore_pos]
            return phrase or [head]
        phrase: List[InputToken] = []
        for token in tokens:
            if token.index <= article.index:
                continue
            if token.upos in self.article_ignore_pos:
                continue
            if token.upos in {"ADJ", "ADV", "NOUN", "PROPN", "NUM", "X"}:
                phrase.append(token)
                if token.upos in {"NOUN", "PROPN", "NUM", "X"}:
                    break
            else:
                break
        return phrase

    def _head_for_token(self, tokens: List[InputToken], token: InputToken) -> Optional[InputToken]:
        return next((item for item in tokens if item.index == token.head), None)

    def _next_nominal(self, tokens: List[InputToken], start_index: int) -> Optional[InputToken]:
        return self._nearest_after(tokens, start_index, lambda token: token.upos in {"NOUN", "PROPN", "PRON"})

    def _nearest_after(self, tokens: List[InputToken], start_index: int, predicate: Any) -> Optional[InputToken]:
        for token in sorted(tokens, key=lambda item: item.index):
            if token.index > start_index and predicate(token):
                return token
        return None

    def _previous_token(self, tokens: List[InputToken], index: int) -> Optional[InputToken]:
        previous = [token for token in tokens if token.index < index]
        return sorted(previous, key=lambda item: item.index)[-1] if previous else None

    def _preserve_case(self, original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original[:1].isupper():
            return replacement.capitalize()
        return replacement

    def _make_issue(
        self,
        *,
        code: Union[str, LanguageErrorCode],
        description: str,
        tokens: Sequence[InputToken],
        severity: str,
        suggestion: Optional[str],
        rule_id: str,
        confidence: float,
        sentence_index: int,
        replacement: Optional[str] = None,
        autofixable: bool = False,
        details: Optional[Mapping[str, Any]] = None,
    ) -> GrammarIssue:
        token_list = list(tokens)
        if token_list:
            start = min(token.start_char_abs for token in token_list)
            end = max(token.end_char_abs for token in token_list)
            start_idx = min(token.index for token in token_list)
            end_idx = max(token.index for token in token_list)
            token_indices = tuple(token.index for token in token_list)
        else:
            start = end = start_idx = end_idx = 0
            token_indices = ()
        return GrammarIssue(
            description=description,
            source_text_char_span=(max(0, start), max(0, end)),
            source_sentence_token_indices_span=(start_idx, end_idx),
            severity=severity,
            suggestion=suggestion,
            code=code.value if isinstance(code, LanguageErrorCode) else ensure_text(code),
            rule_id=rule_id,
            confidence=clamp_float(confidence, 0.0, 1.0, default=self.default_issue_confidence),
            sentence_index=sentence_index,
            token_indices=token_indices,
            autofixable=autofixable,
            replacement=replacement,
            details=dict(details or {}),
        )

    def _load_pronoun_features(self) -> Dict[str, Dict[str, str]]:
        defaults: Dict[str, Dict[str, str]] = {
            "i": {"number": "singular", "person": "1st"},
            "me": {"number": "singular", "person": "1st"},
            "we": {"number": "plural", "person": "1st"},
            "us": {"number": "plural", "person": "1st"},
            "you": {"number": "plural" if self.allow_singular_they else "singular", "person": "2nd"},
            "he": {"number": "singular", "person": "3rd"},
            "him": {"number": "singular", "person": "3rd"},
            "she": {"number": "singular", "person": "3rd"},
            "her": {"number": "singular", "person": "3rd"},
            "it": {"number": "singular", "person": "3rd"},
            "they": {"number": "plural", "person": "3rd"},
            "them": {"number": "plural", "person": "3rd"},
        }
        configured = ensure_mapping(self.grammar_config.get("pronoun_features", {}), field_name="pronoun_features", allow_none=True)
        for pronoun, payload in configured.items():
            if isinstance(payload, Mapping):
                defaults[ensure_text(pronoun).lower()] = {
                    "number": ensure_text(payload.get("number", "singular")),
                    "person": ensure_text(payload.get("person", "3rd")),
                }
        return defaults

    def _build_display_text(self, sentences: List[List[Any]]) -> str:
        if not sentences:
            return ""
        text = " ".join(self._reconstruct_sentence_text([self._coerce_token(token, position=i, sentence_index=s_idx, full_text=None) for i, token in enumerate(sentence)]) for s_idx, sentence in enumerate(sentences[:3]))
        return normalize_spacing_around_punctuation(text + ("..." if len(sentences) > 3 else ""))

    def _record(self, action: str, **payload: Any) -> None:
        if not self.record_history:
            return
        self.history.append({"timestamp": utc_timestamp(), "action": action, "payload": json_safe(payload)})

    def __repr__(self) -> str:
        return f"<GrammarProcessor version='{self.version}' analyses={self.analyses_run} diagnostics={len(self.diagnostics.issues)}>"


if __name__ == "__main__":
    print("\n=== Running Grammar Processor ===\n")
    printer.status("TEST", "Grammar Processor initialized", "info")

    processor = GrammarProcessor()

    sentences = [[
        InputToken(text="He", lemma="he", pos="PRON", index=0, head=1, dep="nsubj", start_char_abs=0, end_char_abs=1),
        InputToken(text="go", lemma="go", pos="VERB", index=1, head=1, dep="root", start_char_abs=3, end_char_abs=4),
        InputToken(text="to", lemma="to", pos="ADP", index=2, head=4, dep="case", start_char_abs=6, end_char_abs=7),
        InputToken(text="a", lemma="a", pos="DET", index=3, head=4, dep="det", start_char_abs=9, end_char_abs=9),
        InputToken(text="office", lemma="office", pos="NOUN", index=4, head=1, dep="obl", start_char_abs=11, end_char_abs=16),
        InputToken(text=".", lemma=".", pos="PUNCT", index=5, head=1, dep="punct", start_char_abs=17, end_char_abs=17),
    ]]

    result = processor.analyze_text(sentences, full_text_snippet="He go to a office.")
    printer.pretty("ANALYSIS", result.to_dict(), "success")
    printer.pretty("STATS", processor.stats().to_dict(), "success")

    nlp_like_tokens = [
        {"text": "They", "lemma": "they", "pos": "PRON", "index": 0},
        {"text": "is", "lemma": "be", "pos": "AUX", "index": 1},
        {"text": "ready", "lemma": "ready", "pos": "ADJ", "index": 2},
        {"text": "!", "lemma": "!", "pos": "PUNCT", "index": 3},
    ]
    adapter_result = processor.analyze_tokens(nlp_like_tokens, full_text_snippet="They is ready!")
    printer.pretty("ADAPTER_ANALYSIS", adapter_result.to_dict(), "success")

    print("\n=== Test ran successfully ===\n")
