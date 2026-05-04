"""
Production diagnostics and exceptions for the language stack.

This module is intentionally language-focused. It separates recoverable language
issues from runtime language errors, preserves existing NLG exception names, and
supports LinguisticFrame as first-class semantic context for NLU, dialogue, and
NLG failures.
"""

from __future__ import annotations

import json
import traceback

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union

from .linguistic_frame import LinguisticFrame, SpeechActType
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Language Error")
printer = PrettyPrinter


E = TypeVar("E", bound=Enum)
T = TypeVar("T")
Span = Tuple[int, int]


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LanguageStage(str, Enum):
    ORTHOGRAPHY = "orthography"
    TOKENIZATION = "tokenization"
    NLP = "nlp"
    DEPENDENCY = "dependency"
    GRAMMAR = "grammar"
    NLU = "nlu"
    CONTEXT = "context"
    NLG = "nlg"
    CACHE = "cache"
    CONFIG = "config"
    RESOURCE = "resource"
    MODEL = "model"
    PIPELINE = "pipeline"
    UNKNOWN = "unknown"


class ErrorCategory(str, Enum):
    USER_INPUT = "user_input"
    LINGUISTIC_ANALYSIS = "linguistic_analysis"
    PIPELINE_CONTRACT = "pipeline_contract"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    MODEL = "model"
    CACHE = "cache"
    STATE = "state"
    GENERATION = "generation"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class LanguageErrorCode(str, Enum):
    ORTH_UNKNOWN_WORD = "ORTH.SPELLING.UNKNOWN_WORD"
    ORTH_CORRECTION_LOW_CONFIDENCE = "ORTH.SPELLING.LOW_CONFIDENCE_CORRECTION"
    ORTH_CORRECTION_FAILED = "ORTH.SPELLING.CORRECTION_FAILED"
    ORTH_NORMALIZATION_FAILED = "ORTH.NORMALIZATION.FAILED"
    ORTH_LOCALE_CONFLICT = "ORTH.NORMALIZATION.LOCALE_CONFLICT"
    ORTH_CONTRACTION_EXPANSION_FAILED = "ORTH.CONTRACTION.EXPANSION_FAILED"
    ORTH_COMPOUND_HANDLING_FAILED = "ORTH.COMPOUND.HANDLING_FAILED"
    ORTH_SPAN_MAPPING_MISSING = "ORTH.SPAN.MAPPING_MISSING"

    TOKENIZER_MODEL_MISSING = "TOKENIZER.MODEL.MISSING"
    TOKENIZER_VOCAB_MISSING = "TOKENIZER.VOCAB.MISSING"
    TOKENIZER_NOT_TRAINED = "TOKENIZER.MODEL.NOT_TRAINED"
    TOKEN_BOUNDARY_LOSS = "TOKEN.SPAN.BOUNDARY_LOSS"
    TOKEN_SUBWORD_USED_AS_WORD = "TOKEN.BPE.SUBWORD_USED_AS_WORD"
    TOKEN_ALIGNMENT_FAILED = "TOKEN.ALIGNMENT.FAILED"

    NLP_POS_UNKNOWN = "NLP.POS.UNKNOWN_TAG"
    NLP_POS_SCHEMA_MISMATCH = "NLP.POS.SCHEMA_MISMATCH"
    NLP_LEMMA_FAILED = "NLP.LEMMA.FAILED"
    NLP_TOKEN_SPAN_MISSING = "NLP.TOKEN.SPAN_MISSING"
    DEP_NO_ROOT = "NLP.DEPENDENCY.NO_ROOT"
    DEP_MULTIPLE_ROOTS = "NLP.DEPENDENCY.MULTIPLE_ROOTS"
    DEP_DUPLICATE_RELATION = "NLP.DEPENDENCY.DUPLICATE_RELATION"
    DEP_LOW_CONFIDENCE_ATTACHMENT = "NLP.DEPENDENCY.LOW_CONFIDENCE_ATTACHMENT"
    DEP_RULE_RUNTIME_ERROR = "NLP.DEPENDENCY.RULE_RUNTIME_ERROR"
    DEP_UNSUPPORTED_CONSTRUCTION = "NLP.DEPENDENCY.UNSUPPORTED_CONSTRUCTION"

    GRAMMAR_INPUT_CONTRACT = "GRAMMAR.INPUT.CONTRACT_MISMATCH"
    GRAMMAR_SUBJECT_VERB_AGREEMENT = "GRAMMAR.AGREEMENT.SUBJECT_VERB"
    GRAMMAR_ARTICLE_USAGE = "GRAMMAR.ARTICLE.USAGE"
    GRAMMAR_SENTENCE_TYPE_AMBIGUOUS = "GRAMMAR.SENTENCE.TYPE_AMBIGUOUS"
    GRAMMAR_FRAGMENT = "GRAMMAR.SYNTAX.FRAGMENT"
    GRAMMAR_MISSING_SUBJECT = "GRAMMAR.SYNTAX.MISSING_SUBJECT"
    GRAMMAR_RULE_COVERAGE = "GRAMMAR.RULE.COVERAGE_WARNING"
    GRAMMAR_SUGGESTION_FAILED = "GRAMMAR.SUGGESTION.FAILED"

    NLU_INTENT_NO_MATCH = "NLU.INTENT.NO_MATCH"
    NLU_INTENT_AMBIGUOUS = "NLU.INTENT.AMBIGUOUS"
    NLU_ENTITY_EXTRACTION_FAILED = "NLU.ENTITY.EXTRACTION_FAILED"
    NLU_ENTITY_VALIDATION_FAILED = "NLU.ENTITY.VALIDATION_FAILED"
    NLU_MODALITY_UNKNOWN = "NLU.MODALITY.UNKNOWN"
    NLU_SENTIMENT_LEXICON_MISSING = "NLU.SENTIMENT.LEXICON_MISSING"
    NLU_FRAME_VALIDATION_FAILED = "NLU.FRAME.VALIDATION_FAILED"
    NLU_WORDLIST_INVALID = "NLU.WORDLIST.INVALID"

    CONTEXT_ROLE_MISMATCH = "CONTEXT.ROLE.MISMATCH"
    CONTEXT_FOLLOWUP_AMBIGUOUS = "CONTEXT.FOLLOWUP.AMBIGUOUS_REFERENCE"
    CONTEXT_SLOT_MISSING = "CONTEXT.SLOT.MISSING_REQUIRED"
    CONTEXT_SUMMARY_FAILED = "CONTEXT.SUMMARY.FAILED"
    CONTEXT_TOPIC_SHIFT_AMBIGUOUS = "CONTEXT.TOPIC_SHIFT.AMBIGUOUS"
    CONTEXT_STATE_SERIALIZATION_FAILED = "CONTEXT.STATE.SERIALIZATION_FAILED"

    NLG_TEMPLATE_NOT_FOUND = "NLG.TEMPLATE.NOT_FOUND"
    NLG_TEMPLATE_SCHEMA_INVALID = "NLG.TEMPLATE.SCHEMA_INVALID"
    NLG_TEMPLATE_FILLING_FAILED = "NLG.TEMPLATE.FILLING_FAILED"
    NLG_TEMPLATE_PLACEHOLDER_MISSING = "NLG.TEMPLATE.PLACEHOLDER_MISSING"
    NLG_STYLE_CONFIG_INVALID = "NLG.STYLE.CONFIG_INVALID"
    NLG_RESPONSE_VALIDATION_FAILED = "NLG.RESPONSE.VALIDATION_FAILED"
    NLG_GENERATION_FAILED = "NLG.GENERATION.FAILED"
    NLG_FALLBACK_USED = "NLG.GENERATION.FALLBACK_USED"
    NLG_SPEECH_ACT_REALIZATION_FAILED = "NLG.SPEECH_ACT.REALIZATION_FAILED"

    CONFIG_SECTION_MISSING = "CONFIG.SECTION.MISSING"
    CONFIG_SCHEMA_INVALID = "CONFIG.SCHEMA.INVALID"
    CONFIG_VALUE_INVALID = "CONFIG.VALUE.INVALID"
    RESOURCE_MISSING = "RESOURCE.MISSING"
    RESOURCE_LOAD_FAILED = "RESOURCE.LOAD_FAILED"
    RESOURCE_FORMAT_INVALID = "RESOURCE.FORMAT.INVALID"
    CACHE_MISS = "CACHE.MISS"
    CACHE_TYPE_MISMATCH = "CACHE.TYPE_MISMATCH"
    CACHE_SERIALIZATION_FAILED = "CACHE.SERIALIZATION_FAILED"
    MODEL_UNAVAILABLE = "MODEL.UNAVAILABLE"
    MODEL_LOAD_FAILED = "MODEL.LOAD_FAILED"
    MODEL_INFERENCE_FAILED = "MODEL.INFERENCE_FAILED"
    PIPELINE_CONTRACT_MISMATCH = "PIPELINE.CONTRACT.MISMATCH"
    PIPELINE_STAGE_FAILED = "PIPELINE.STAGE.FAILED"
    UNKNOWN = "LANGUAGE.UNKNOWN"


_ENUM_VALUE_MAP: Dict[Type[Enum], Dict[Any, Enum]] = {
    Severity: {member.value: member for member in Severity},
    LanguageStage: {member.value: member for member in LanguageStage},
    ErrorCategory: {member.value: member for member in ErrorCategory},
}
_CODE_VALUE_MAP: Dict[str, LanguageErrorCode] = {member.value: member for member in LanguageErrorCode}

def _coerce_enum(value: Union[str, E, None], enum_cls: Type[E], default: E) -> E:
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    raw_value = value.value if isinstance(value, Enum) else value
    try:
        return enum_cls(raw_value)
    except ValueError:
        logger.warning("Unknown %s value: %r. Using default %s.", enum_cls.__name__, raw_value, default.value)
        return default


def _coerce_code(code: Union[str, LanguageErrorCode, None]) -> str:
    if code is None:
        return LanguageErrorCode.UNKNOWN.value
    if isinstance(code, LanguageErrorCode):
        return code.value
    if isinstance(code, Enum):
        return str(code.value)
    return str(code)


def _validate_span(span: Optional[Span], field_name: str) -> Optional[Span]:
    if span is None:
        return None
    if not isinstance(span, tuple) or len(span) != 2:
        raise ValueError(f"{field_name} must be a tuple of two integers, got {span!r}")
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{field_name} values must be integers, got {span!r}")
    if start < 0 or end < start:
        raise ValueError(f"{field_name} must satisfy 0 <= start <= end, got {span!r}")
    return span


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, LinguisticFrame):
        return _frame_to_dict(value)
    if is_dataclass(value):
        return _to_serializable(asdict(value)) # type: ignore
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_serializable(item) for item in value)
    return repr(value)


def _frame_to_dict(frame: Optional[LinguisticFrame]) -> Optional[Dict[str, Any]]:
    if frame is None:
        return None
    act_type = frame.act_type.value if isinstance(frame.act_type, SpeechActType) else frame.act_type
    return {
        "intent": frame.intent,
        "entities": _to_serializable(frame.entities),
        "sentiment": frame.sentiment,
        "modality": frame.modality,
        "confidence": frame.confidence,
        "act_type": act_type,
        "propositional_content": frame.propositional_content,
        "illocutionary_force": frame.illocutionary_force,
        "perlocutionary_effect": frame.perlocutionary_effect,
    }


def _context_summary(context: Optional[Any]) -> Optional[Any]:
    if context is None:
        return None
    if hasattr(context, "get_context_summary"):
        return _to_serializable(context.get_context_summary())
    if isinstance(context, Mapping):
        return _to_serializable(dict(context))
    return repr(context)


@dataclass
class LanguageIssue:
    code: str
    message: str
    stage: LanguageStage
    severity: Severity
    category: ErrorCategory
    module: Optional[str] = None
    source_text: Optional[str] = None
    normalized_text: Optional[str] = None
    source_span: Optional[Span] = None
    normalized_span: Optional[Span] = None
    token_span: Optional[Span] = None
    sentence_index: Optional[int] = None
    token_indices: Optional[Tuple[int, ...]] = None
    suggestion: Optional[str] = None
    confidence: Optional[float] = None
    rule_id: Optional[str] = None
    locale: Optional[str] = None
    language: Optional[str] = "en"
    autofixable: bool = False
    recoverable: bool = True
    frame: Optional[LinguisticFrame] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        code: Union[str, LanguageErrorCode],
        message: str,
        stage: Union[str, LanguageStage] = LanguageStage.UNKNOWN,
        severity: Union[str, Severity] = Severity.WARNING,
        category: Union[str, ErrorCategory] = ErrorCategory.UNKNOWN,
        module: Optional[str] = None,
        source_text: Optional[str] = None,
        normalized_text: Optional[str] = None,
        source_span: Optional[Span] = None,
        normalized_span: Optional[Span] = None,
        token_span: Optional[Span] = None,
        sentence_index: Optional[int] = None,
        token_indices: Optional[Tuple[int, ...]] = None,
        suggestion: Optional[str] = None,
        confidence: Optional[float] = None,
        rule_id: Optional[str] = None,
        locale: Optional[str] = None,
        language: Optional[str] = "en",
        autofixable: bool = False,
        recoverable: bool = True,
        frame: Optional[LinguisticFrame] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = _coerce_code(code)
        self.message = message
        self.stage = _coerce_enum(stage, LanguageStage, LanguageStage.UNKNOWN)
        self.severity = _coerce_enum(severity, Severity, Severity.WARNING)
        self.category = _coerce_enum(category, ErrorCategory, ErrorCategory.UNKNOWN)
        self.module = module
        self.source_text = source_text
        self.normalized_text = normalized_text
        self.source_span = _validate_span(source_span, "source_span")
        self.normalized_span = _validate_span(normalized_span, "normalized_span")
        self.token_span = _validate_span(token_span, "token_span")
        self.sentence_index = sentence_index
        self.token_indices = tuple(int(i) for i in token_indices) if token_indices else None
        self.suggestion = suggestion
        self.confidence = max(0.0, min(1.0, float(confidence))) if confidence is not None else None
        self.rule_id = rule_id
        self.locale = locale
        self.language = language
        self.autofixable = autofixable
        self.recoverable = recoverable
        self.frame = frame
        self.details = _to_serializable(details or {})

    @property
    def is_blocking(self) -> bool:
        return self.severity in {Severity.ERROR, Severity.CRITICAL}

    @property
    def intent(self) -> Optional[str]:
        return self.frame.intent if self.frame else None

    def with_details(self, **details: Any) -> "LanguageIssue":
        merged = dict(self.details)
        merged.update(details)
        self.details = _to_serializable(merged)
        return self

    def with_frame(self, frame: LinguisticFrame) -> "LanguageIssue":
        self.frame = frame
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage.value,
            "severity": self.severity.value,
            "category": self.category.value,
            "module": self.module,
            "source_text": self.source_text,
            "normalized_text": self.normalized_text,
            "source_span": list(self.source_span) if self.source_span else None,
            "normalized_span": list(self.normalized_span) if self.normalized_span else None,
            "token_span": list(self.token_span) if self.token_span else None,
            "sentence_index": self.sentence_index,
            "token_indices": list(self.token_indices) if self.token_indices else None,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "rule_id": self.rule_id,
            "locale": self.locale,
            "language": self.language,
            "autofixable": self.autofixable,
            "recoverable": self.recoverable,
            "frame": _frame_to_dict(self.frame),
            "details": self.details,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        *,
        code: Union[str, LanguageErrorCode] = LanguageErrorCode.UNKNOWN,
        stage: Union[str, LanguageStage] = LanguageStage.UNKNOWN,
        category: Union[str, ErrorCategory] = ErrorCategory.UNKNOWN,
        module: Optional[str] = None,
        severity: Union[str, Severity] = Severity.ERROR,
        message: Optional[str] = None,
        frame: Optional[LinguisticFrame] = None,
        include_traceback: bool = False,
        **details: Any,
    ) -> "LanguageIssue":
        payload = dict(details)
        payload.setdefault("exception_type", type(exc).__name__)
        payload.setdefault("exception_message", str(exc))
        if include_traceback:
            payload["traceback"] = traceback.format_exception(type(exc), exc, exc.__traceback__)
        return cls(
            code=code,
            message=message or str(exc),
            stage=stage,
            category=category,
            severity=severity,
            module=module,
            frame=frame,
            details=payload,
        )


class LanguageError(Exception):
    default_code: LanguageErrorCode = LanguageErrorCode.UNKNOWN
    default_stage: LanguageStage = LanguageStage.UNKNOWN
    default_category: ErrorCategory = ErrorCategory.UNKNOWN
    default_severity: Severity = Severity.ERROR
    default_module: Optional[str] = None
    default_recoverable: bool = True

    def __init__(
        self,
        issue: Union[LanguageIssue, str],
        *,
        code: Optional[Union[str, LanguageErrorCode]] = None,
        stage: Optional[Union[str, LanguageStage]] = None,
        category: Optional[Union[str, ErrorCategory]] = None,
        severity: Optional[Union[str, Severity]] = None,
        module: Optional[str] = None,
        recoverable: Optional[bool] = None,
        cause: Optional[Exception] = None,
        frame: Optional[LinguisticFrame] = None,
        details: Optional[Dict[str, Any]] = None,
        **issue_fields: Any,
    ):
        resolved_recoverable = self.default_recoverable if recoverable is None else recoverable
        if isinstance(issue, LanguageIssue):
            self.issue = issue
            self.issue.recoverable = resolved_recoverable
            if frame is not None:
                self.issue.frame = frame
        else:
            self.issue = LanguageIssue(
                code=code or self.default_code,
                message=str(issue),
                stage=stage or self.default_stage,
                category=category or self.default_category,
                severity=severity or self.default_severity,
                module=module or self.default_module,
                recoverable=resolved_recoverable,
                frame=frame,
                details=details or {},
                **issue_fields,
            )
        self.recoverable = resolved_recoverable
        self.cause = cause
        super().__init__(self._build_message())

    @property
    def code(self) -> str:
        return self.issue.code

    @property
    def severity(self) -> Severity:
        return self.issue.severity

    @property
    def stage(self) -> LanguageStage:
        return self.issue.stage

    @property
    def frame(self) -> Optional[LinguisticFrame]:
        return self.issue.frame

    def _build_message(self) -> str:
        cause_text = f" | cause={type(self.cause).__name__}: {self.cause}" if self.cause else ""
        return f"{self.issue.code}: {self.issue.message}{cause_text}"

    def log(self, level: Optional[int] = None) -> None:
        severity_level = level or {
            Severity.INFO: 20,
            Severity.WARNING: 30,
            Severity.ERROR: 40,
            Severity.CRITICAL: 50,
        }[self.issue.severity]
        logger.log(severity_level, self.to_json())

    def to_dict(self) -> Dict[str, Any]:
        payload = self.issue.to_dict()
        payload.update(
            {
                "exception_type": type(self).__name__,
                "recoverable": self.recoverable,
                "cause_type": type(self.cause).__name__ if self.cause else None,
                "cause_message": str(self.cause) if self.cause else None,
            }
        )
        return payload

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class DomainLanguageError(LanguageError):
    domain_name: str = "language"

    def __init__(self, issue: Union[LanguageIssue, str], *, details: Optional[Dict[str, Any]] = None, **kwargs: Any):
        merged_details = dict(details or {})
        merged_details.setdefault("domain", self.domain_name)
        super().__init__(issue, details=merged_details, **kwargs)


@dataclass
class LanguageDiagnostics:
    issues: List[LanguageIssue] = field(default_factory=list)

    def add(self, issue: Union[LanguageIssue, LanguageError]) -> None:
        if isinstance(issue, LanguageError):
            self.issues.append(issue.issue)
            return
        if isinstance(issue, LanguageIssue):
            self.issues.append(issue)
            return
        raise TypeError(f"Expected LanguageIssue or LanguageError, got {type(issue).__name__}")

    def extend(self, issues: Iterable[Union[LanguageIssue, LanguageError]]) -> None:
        for issue in issues:
            self.add(issue)

    def by_stage(self, stage: Union[str, LanguageStage]) -> List[LanguageIssue]:
        stage_value = _coerce_enum(stage, LanguageStage, LanguageStage.UNKNOWN)
        return [issue for issue in self.issues if issue.stage == stage_value]

    def by_code(self, code: Union[str, LanguageErrorCode]) -> List[LanguageIssue]:
        code_value = _coerce_code(code)
        return [issue for issue in self.issues if issue.code == code_value]

    def by_severity(self, severity: Union[str, Severity]) -> List[LanguageIssue]:
        severity_value = _coerce_enum(severity, Severity, Severity.WARNING)
        return [issue for issue in self.issues if issue.severity == severity_value]

    def by_intent(self, intent: str) -> List[LanguageIssue]:
        return [issue for issue in self.issues if issue.frame and issue.frame.intent == intent]

    def has_blocking_issues(self) -> bool:
        return any(issue.is_blocking for issue in self.issues)

    def raise_if_blocking(self, message: str = "Blocking language diagnostics were produced") -> None:
        if self.has_blocking_issues():
            raise PipelineContractError(message, details={"issues": self.to_list()})

    def to_list(self) -> List[Dict[str, Any]]:
        return [issue.to_dict() for issue in self.issues]

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_list(), ensure_ascii=False, indent=indent)


@dataclass
class LanguageResult(Generic[T]):
    data: Optional[T] = None
    issues: List[LanguageIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame: Optional[LinguisticFrame] = None

    @property
    def ok(self) -> bool:
        return not any(issue.is_blocking for issue in self.issues)

    def add_issue(self, issue: Union[LanguageIssue, LanguageError]) -> None:
        self.issues.append(issue.issue if isinstance(issue, LanguageError) else issue)

    def extend_issues(self, issues: Iterable[Union[LanguageIssue, LanguageError]]) -> None:
        for issue in issues:
            self.add_issue(issue)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "data": _to_serializable(self.data),
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": _to_serializable(self.metadata),
            "frame": _frame_to_dict(self.frame),
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def make_issue(
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


class OrthographyIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.ORTHOGRAPHY)
        kwargs.setdefault("category", ErrorCategory.USER_INPUT)
        super().__init__(code=code, message=message, **kwargs)


class TokenizationIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.TOKENIZATION)
        kwargs.setdefault("category", ErrorCategory.LINGUISTIC_ANALYSIS)
        super().__init__(code=code, message=message, **kwargs)


class NLPIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.NLP)
        kwargs.setdefault("category", ErrorCategory.LINGUISTIC_ANALYSIS)
        super().__init__(code=code, message=message, **kwargs)


class DependencyIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.DEPENDENCY)
        kwargs.setdefault("category", ErrorCategory.LINGUISTIC_ANALYSIS)
        super().__init__(code=code, message=message, **kwargs)


class GrammarIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.GRAMMAR)
        kwargs.setdefault("category", ErrorCategory.USER_INPUT)
        super().__init__(code=code, message=message, **kwargs)


class NLUIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.NLU)
        kwargs.setdefault("category", ErrorCategory.LINGUISTIC_ANALYSIS)
        super().__init__(code=code, message=message, **kwargs)


class ContextIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.CONTEXT)
        kwargs.setdefault("category", ErrorCategory.STATE)
        super().__init__(code=code, message=message, **kwargs)


class NLGIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.NLG)
        kwargs.setdefault("category", ErrorCategory.GENERATION)
        super().__init__(code=code, message=message, **kwargs)


class ConfigurationIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.CONFIG)
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("severity", Severity.ERROR)
        super().__init__(code=code, message=message, **kwargs)


class ResourceIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.RESOURCE)
        kwargs.setdefault("category", ErrorCategory.RESOURCE)
        kwargs.setdefault("severity", Severity.ERROR)
        super().__init__(code=code, message=message, **kwargs)


class ModelIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.MODEL)
        kwargs.setdefault("category", ErrorCategory.MODEL)
        kwargs.setdefault("severity", Severity.ERROR)
        super().__init__(code=code, message=message, **kwargs)


class CacheIssue(LanguageIssue):
    def __init__(self, code: Union[str, LanguageErrorCode], message: str, **kwargs: Any):
        kwargs.setdefault("stage", LanguageStage.CACHE)
        kwargs.setdefault("category", ErrorCategory.CACHE)
        super().__init__(code=code, message=message, **kwargs)


class OrthographyError(DomainLanguageError):
    """Runtime failure in normalization, spelling, contraction, or span mapping."""
    domain_name = "orthography"
    default_code = LanguageErrorCode.ORTH_CORRECTION_FAILED
    default_stage = LanguageStage.ORTHOGRAPHY
    default_category = ErrorCategory.USER_INPUT
    default_module = "OrthographyProcessor"


class TokenizationError(DomainLanguageError):
    """Runtime failure in word/subword tokenization or token alignment."""
    domain_name = "tokenization"
    default_code = LanguageErrorCode.TOKEN_ALIGNMENT_FAILED
    default_stage = LanguageStage.TOKENIZATION
    default_category = ErrorCategory.LINGUISTIC_ANALYSIS
    default_module = "LanguageTokenizer"


class NLPError(DomainLanguageError):
    """Runtime failure in POS tagging, lemmatization, morphology, or NLP payloads."""
    domain_name = "nlp"
    default_code = LanguageErrorCode.NLP_LEMMA_FAILED
    default_stage = LanguageStage.NLP
    default_category = ErrorCategory.LINGUISTIC_ANALYSIS
    default_module = "NLPEngine"


class DependencyError(DomainLanguageError):
    """Runtime failure in dependency construction or dependency rule execution."""
    domain_name = "dependency"
    default_code = LanguageErrorCode.DEP_RULE_RUNTIME_ERROR
    default_stage = LanguageStage.DEPENDENCY
    default_category = ErrorCategory.LINGUISTIC_ANALYSIS
    default_module = "Rules"


class GrammarError(DomainLanguageError):
    """Runtime failure in grammar analysis or grammar-rule contracts."""
    domain_name = "grammar"
    default_code = LanguageErrorCode.GRAMMAR_INPUT_CONTRACT
    default_stage = LanguageStage.GRAMMAR
    default_category = ErrorCategory.USER_INPUT
    default_module = "GrammarProcessor"


class NLUError(DomainLanguageError):
    """Runtime failure in intent, entity, modality, sentiment, or semantic-frame analysis."""
    domain_name = "nlu"
    default_code = LanguageErrorCode.NLU_FRAME_VALIDATION_FAILED
    default_stage = LanguageStage.NLU
    default_category = ErrorCategory.LINGUISTIC_ANALYSIS
    default_module = "NLUEngine"


class ContextError(DomainLanguageError):
    """Runtime failure in dialogue state, slots, follow-up tracking, or memory."""
    domain_name = "context"
    default_code = LanguageErrorCode.CONTEXT_STATE_SERIALIZATION_FAILED
    default_stage = LanguageStage.CONTEXT
    default_category = ErrorCategory.STATE
    default_module = "DialogueContext"


class NLGError(DomainLanguageError):
    """Runtime failure in template filling, style realization, or response generation."""
    domain_name = "nlg"
    default_code = LanguageErrorCode.NLG_GENERATION_FAILED
    default_stage = LanguageStage.NLG
    default_category = ErrorCategory.GENERATION
    default_module = "NLGEngine"


class ConfigurationLanguageError(DomainLanguageError):
    """Invalid, missing, or incompatible language configuration."""
    domain_name = "configuration"
    default_code = LanguageErrorCode.CONFIG_SCHEMA_INVALID
    default_stage = LanguageStage.CONFIG
    default_category = ErrorCategory.CONFIGURATION
    default_module = "config_loader"
    default_recoverable = False


class ResourceLanguageError(DomainLanguageError):
    """Missing or malformed external language resource."""
    domain_name = "resource"
    default_code = LanguageErrorCode.RESOURCE_LOAD_FAILED
    default_stage = LanguageStage.RESOURCE
    default_category = ErrorCategory.RESOURCE
    default_module = "language_resources"
    default_recoverable = False


class ModelLanguageError(DomainLanguageError):
    """Unavailable, incompatible, or failing language model."""
    domain_name = "model"
    default_code = LanguageErrorCode.MODEL_INFERENCE_FAILED
    default_stage = LanguageStage.MODEL
    default_category = ErrorCategory.MODEL
    default_module = "LanguageTransformer"
    default_recoverable = False


class CacheLanguageError(DomainLanguageError):
    """Cache corruption, serialization, type, or contract failure."""
    domain_name = "cache"
    default_code = LanguageErrorCode.CACHE_SERIALIZATION_FAILED
    default_stage = LanguageStage.CACHE
    default_category = ErrorCategory.CACHE
    default_module = "LanguageCache"


class PipelineContractError(DomainLanguageError):
    """Mismatch between pipeline stages, payloads, or expected processor contracts."""
    domain_name = "pipeline"
    default_code = LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH
    default_stage = LanguageStage.PIPELINE
    default_category = ErrorCategory.PIPELINE_CONTRACT
    default_module = "LanguagePipeline"
    default_recoverable = False

    def __init__(self, message: str, *, expected: Any = None, received: Any = None, **kwargs: Any):
        details = dict(kwargs.pop("details", {}) or {})
        if expected is not None:
            details["expected"] = _to_serializable(expected)
        if received is not None:
            details["received"] = _to_serializable(received)
        super().__init__(message, details=details, **kwargs)


class NLGFillingError(NLGError):
    """Raised when template filling fails due to missing or malformed entities."""

    def __init__(
        self,
        message: str,
        intent: Optional[str] = None,
        template: Optional[str] = None,
        entity_data: Optional[Dict[str, Any]] = None,
        missing_fields: Optional[List[str]] = None,
        frame: Optional[LinguisticFrame] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.intent = intent
        self.template = template
        self.entity_data = entity_data
        self.missing_fields = missing_fields or []
        self.original_exception = original_exception
        issue = NLGIssue(
            code=LanguageErrorCode.NLG_TEMPLATE_FILLING_FAILED,
            message=message,
            severity=Severity.ERROR,
            module="NLGEngine",
            recoverable=True,
            frame=frame,
            details={
                "intent": intent or (frame.intent if frame else None),
                "template": template,
                "entity_data": entity_data or {},
                "missing_fields": self.missing_fields,
            },
        )
        super().__init__(issue, recoverable=True, cause=original_exception)


class NLGValidationError(NLGError):
    """Raised when generated text violates response validation constraints."""

    def __init__(
        self,
        message: str,
        response_text: Optional[str] = None,
        expected_format: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        frame: Optional[LinguisticFrame] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.response_text = response_text
        self.expected_format = expected_format
        self.validation_errors = validation_errors or []
        self.original_exception = original_exception
        issue = NLGIssue(
            code=LanguageErrorCode.NLG_RESPONSE_VALIDATION_FAILED,
            message=message,
            severity=Severity.ERROR,
            module="NLGEngine",
            source_text=response_text,
            recoverable=True,
            frame=frame,
            details={
                "response_text": response_text,
                "expected_format": expected_format,
                "validation_errors": self.validation_errors,
            },
        )
        super().__init__(issue, recoverable=True, cause=original_exception)


class TemplateNotFoundError(NLGError):
    """Raised when no response template exists for the resolved intent."""

    def __init__(
        self,
        message: str,
        intent: Optional[str] = None,
        templates: Optional[Dict[str, Any]] = None,
        available_intents: Optional[List[str]] = None,
        frame: Optional[LinguisticFrame] = None,
    ):
        self.intent = intent
        self.templates = templates
        self.available_intents = available_intents or (list(templates.keys()) if templates else [])
        issue = NLGIssue(
            code=LanguageErrorCode.NLG_TEMPLATE_NOT_FOUND,
            message=message,
            severity=Severity.ERROR,
            module="NLGEngine",
            recoverable=True,
            frame=frame,
            details={
                "intent": intent or (frame.intent if frame else None),
                "available_intents": self.available_intents,
            },
        )
        super().__init__(issue, recoverable=True)


class NLGGenerationError(NLGError):
    """Raised when neural or template-assisted generation fails."""

    def __init__(
        self,
        message: str,
        prompt: Optional[str] = None,
        frame: Optional[LinguisticFrame] = None,
        context: Optional[Any] = None,
        error_type: Optional[str] = None,
        original_exception: Optional[Exception] = None,
        fallback_attempted: bool = False,
        fallback_response: Optional[str] = None,
    ):
        self.message = message
        self.prompt = prompt
        self.context = context
        self.error_type = error_type
        self.original_exception = original_exception
        self.fallback_attempted = fallback_attempted
        self.fallback_response = fallback_response
        issue = NLGIssue(
            code=LanguageErrorCode.NLG_GENERATION_FAILED,
            message=message,
            severity=Severity.ERROR,
            module="NLGEngine",
            source_text=prompt,
            recoverable=fallback_attempted,
            frame=frame,
            details={
                "error_type": error_type or "unspecified",
                "original_exception_type": type(original_exception).__name__ if original_exception else None,
                "original_exception_message": str(original_exception) if original_exception else None,
                "prompt_length": len(prompt) if prompt else 0,
                "context_summary": _context_summary(context),
                "fallback_attempted": fallback_attempted,
                "fallback_response": fallback_response,
            },
        )
        super().__init__(issue, recoverable=fallback_attempted, cause=original_exception)
