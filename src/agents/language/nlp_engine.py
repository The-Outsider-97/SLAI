"""
Natural Language Processing Engine

Core Function:
Extracts linguistic features from normalized input to support downstream parsing,
grammar checking, semantic understanding, dialogue context, and generation.

Responsibilities:
- Produce sentence-aware, span-aware linguistic tokens for grammar and NLU.
- Keep word-level linguistic tokenization separate from BPE/model tokenization.
- Assign Universal POS tags using lexicon metadata, context, and configurable rules.
- Generate lemmas and morphology features consistently with the Rules engine.
- Apply dependency rules through the shared language Rules module.
- Extract lightweight entities and resolve simple coreference chains.
- Surface structured diagnostics without blocking the rest of the agent pipeline.

Why it matters:
These annotations form the linguistic backbone for grammar checks, intent
classification, entity extraction, context tracking, and response generation.
A production NLP layer must be deterministic, inspectable, span-preserving,
resource-aware, and compatible with the current LanguageAgent pipeline while
remaining expandable for statistical models later.
"""

from __future__ import annotations

import json
import math
import re
import time as time_module
import yaml

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.language_error import *
from .utils.language_helpers import *
from .modules.language_tokenizer import LanguageTokenizer
from .modules.rules import Rules, DependencyRelation, RuleApplicationResult
from .language_memory import LanguageMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("NLP Engine")
printer = PrettyPrinter()

Span = Tuple[int, int]
TokenId = int
SentenceId = int


# ---------------------------------------------------------------------------
# Small NLP-local utility functions
# ---------------------------------------------------------------------------
def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _lower(value: Any) -> str:
    return _text(value).casefold()


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _int(value: Any, default: int = 0, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _float(value: Any, default: float = 0.0, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return repr(value)
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)


def _dedupe(items: Iterable[Any]) -> List[Any]:
    seen: Set[str] = set()
    output: List[Any] = []
    for item in items:
        marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(item)
    return output


def _word_shape(text: str) -> str:
    shape = []
    for char in text:
        if char.isupper():
            shape.append("X")
        elif char.islower():
            shape.append("x")
        elif char.isdigit():
            shape.append("d")
        elif char.isspace():
            shape.append("s")
        else:
            shape.append(char)
    collapsed: List[str] = []
    for item in shape:
        if not collapsed or collapsed[-1] != item:
            collapsed.append(item)
    return "".join(collapsed)


def _load_json_path(path: Optional[Union[str, Path]], default: Any) -> Any:
    if path in (None, "", "none", "None"):
        return default
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml_path(path: Optional[Union[str, Path]], default: Any) -> Any:
    if path in (None, "", "none", "None"):
        return default
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or default


def _normalize_upos(label: Any) -> str:
    value = _text(label).strip().upper()
    if not value:
        return "X"
    aliases = {
        "N": "NOUN", "NN": "NOUN", "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN",
        "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",
        "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ", "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
        "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON", "DT": "DET",
        "IN": "ADP", "TO": "PART", "CC": "CCONJ", "CD": "NUM", ".": "PUNCT",
    }
    value = aliases.get(value, value)
    valid = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"}
    return value if value in valid else "X"


def _is_punctuation(text: str) -> bool:
    return bool(text) and all(not char.isalnum() and not char.isspace() for char in text)


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,:/-]\d+)*", text.strip()))

def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Convert value to int, return default if conversion fails or value is None."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """Entity mention or lightweight coreference candidate."""

    text: str
    type: str
    gender: str
    number: str
    sentence_index: int
    token_indices: Tuple[int, ...]
    mentions: List[Tuple[int, int]] = field(default_factory=list)
    coref_id: int = -1
    confidence: float = 0.7
    span: Optional[Span] = None
    normalized: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.span is not None:
            payload["span"] = list(self.span)
        return _json_safe(payload)


@dataclass(frozen=True)
class Token:
    """Word-level linguistic token used by NLP, grammar, and NLU.

    The first six fields preserve compatibility with the old Token constructor
    used by the current LanguageAgent.
    """

    text: str
    lemma: str
    pos: str
    index: int
    is_stop: bool = False
    is_punct: bool = False
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    sentence_index: int = 0
    token_id: Optional[int] = None
    upos: Optional[str] = None
    xpos: Optional[str] = None
    feats: Dict[str, Any] = field(default_factory=dict)
    dep: Optional[str] = None
    head: Optional[int] = None
    normalized: Optional[str] = None
    shape: Optional[str] = None
    is_alpha: bool = False
    is_numeric: bool = False
    is_oov: bool = False
    confidence: float = 0.75
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> int:
        return self.index if self.token_id is None else self.token_id

    @property
    def lower(self) -> str:
        return self.text.casefold()

    @property
    def span(self) -> Optional[Span]:
        if self.start_char is None or self.end_char is None:
            return None
        return (self.start_char, self.end_char)

    def to_rule_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "lemma": self.lemma,
            "upos": self.upos or self.pos,
            "pos": self.pos,
            "xpos": self.xpos,
            "id": self.index,
            "index": self.index,
            "feats": dict(self.feats),
            "dep": self.dep,
            "head": self.head,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["id"] = self.id
        payload["span"] = list(self.span) if self.span else None
        return _json_safe(payload)


@dataclass(frozen=True)
class SentenceAnalysis:
    """Sentence-level NLP analysis."""

    text: str
    index: int
    start_char: int
    end_char: int
    token_indices: Tuple[int, ...]
    sentence_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> Span:
        return (self.start_char, self.end_char)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["span"] = list(self.span)
        return _json_safe(payload)


@dataclass(frozen=True)
class NLPAnalysisResult:
    """Structured result for the full NLP pass."""

    original_text: str
    normalized_text: str
    tokens: Tuple[Token, ...]
    sentences: Tuple[SentenceAnalysis, ...] = ()
    dependencies: Tuple[DependencyRelation, ...] = ()
    entities: Tuple[Entity, ...] = ()
    coreferences: Tuple[Entity, ...] = ()
    sarcasm_score: float = 0.0
    model_tokens: Tuple[str, ...] = ()
    issues: Tuple[LanguageIssue, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.is_blocking for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "tokens": [token.to_dict() for token in self.tokens],
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "dependencies": [dep.to_dict() if hasattr(dep, "to_dict") else _json_safe(dep) for dep in self.dependencies],
            "entities": [entity.to_dict() for entity in self.entities],
            "coreferences": [entity.to_dict() for entity in self.coreferences],
            "sarcasm_score": self.sarcasm_score,
            "model_tokens": list(self.model_tokens),
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class NLPEngineStats:
    """Operational snapshot for the NLP engine."""

    language: str
    locale: str
    token_count: int
    sentence_count: int
    entity_count: int
    dependency_count: int
    process_calls: int
    analysis_calls: int
    diagnostics_count: int
    lexicon_size: int
    stopword_count: int
    history_length: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class NLPEngine:
    """Production NLP engine for the language agent subsystem."""

    DEFAULT_STOPWORDS: Set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "if", "in", "into", "is",
        "it", "of", "on", "or", "so", "the", "that", "to", "was", "were", "with", "you", "your", "i", "we",
    }

    DEFAULT_PRONOUN_FEATURES: Dict[str, Dict[str, Any]] = {
        "i": {"Person": 1, "Number": "Sing", "Case": "Nom", "Gender": "Com"},
        "me": {"Person": 1, "Number": "Sing", "Case": "Acc", "Gender": "Com"},
        "we": {"Person": 1, "Number": "Plur", "Case": "Nom", "Gender": "Com"},
        "us": {"Person": 1, "Number": "Plur", "Case": "Acc", "Gender": "Com"},
        "you": {"Person": 2, "Number": "Com", "Case": "Com", "Gender": "Com"},
        "he": {"Person": 3, "Number": "Sing", "Case": "Nom", "Gender": "Masc"},
        "him": {"Person": 3, "Number": "Sing", "Case": "Acc", "Gender": "Masc"},
        "she": {"Person": 3, "Number": "Sing", "Case": "Nom", "Gender": "Fem"},
        "her": {"Person": 3, "Number": "Sing", "Case": "Acc", "Gender": "Fem"},
        "it": {"Person": 3, "Number": "Sing", "Case": "Com", "Gender": "Neut"},
        "they": {"Person": 3, "Number": "Plur", "Case": "Nom", "Gender": "Com"},
        "them": {"Person": 3, "Number": "Plur", "Case": "Acc", "Gender": "Com"},
        "who": {"PronType": "Rel"}, "whom": {"PronType": "Rel"}, "which": {"PronType": "Rel"},
        "that": {"PronType": "Rel"}, "this": {"PronType": "Dem"}, "these": {"PronType": "Dem", "Number": "Plur"},
        "those": {"PronType": "Dem", "Number": "Plur"},
    }

    def __init__(self) -> None:
        self.config = load_global_config()
        self.nlp_config = get_config_section("nlp") or {}
        self.language = _text(self.nlp_config.get("language", "en"))
        self.locale = _text(self.nlp_config.get("locale", "en-US"))
        self.wordlist_path = self.nlp_config.get("structured_wordlist_path", self.config.get("main_wordlist_path"))
        self.stopwords_list_path = self.nlp_config.get("stopwords_list_path")
        self.irregular_nouns_path = self.nlp_config.get("irregular_nouns_path")
        self.pos_patterns_path = self.nlp_config.get("pos_patterns_path")
        self.sentiment_lexicon_path = self.nlp_config.get("sentiment_lexicon_path")
        self.entity_patterns_path = self.nlp_config.get("entity_patterns_path")

        self.max_input_chars = _int(self.nlp_config.get("max_input_chars", 100_000), 100_000, minimum=1)
        self.max_tokens = _int(self.nlp_config.get("max_tokens", 2_000), 2_000, minimum=1)
        self.max_sentence_tokens = _int(self.nlp_config.get("max_sentence_tokens", 256), 256, minimum=1)
        self.track_model_tokens = _bool(self.nlp_config.get("track_model_tokens", True), True)
        self.enable_dependency_rules = _bool(self.nlp_config.get("enable_dependency_rules", True), True)
        self.enable_entity_extraction = _bool(self.nlp_config.get("enable_entity_extraction", True), True)
        self.enable_coreference = _bool(self.nlp_config.get("enable_coreference", True), True)
        self.enable_sarcasm_detection = _bool(self.nlp_config.get("enable_sarcasm_detection", True), True)
        self.enable_memory = _bool(self.nlp_config.get("enable_memory", False), False)
        self.record_history = _bool(self.nlp_config.get("record_history", True), True)
        self.history_limit = _int(self.nlp_config.get("history_limit", 200), 200, minimum=1)
        self.default_unknown_pos = _normalize_upos(self.nlp_config.get("default_unknown_pos", "X"))
        self.default_word_pos = _normalize_upos(self.nlp_config.get("default_word_pos", "NOUN"))

        self.token_pattern = re.compile(
            _text(
                self.nlp_config.get(
                    "token_pattern",
                    r"\[[A-Z_]+\]|[^\W\d_]+(?:['’][^\W\d_]+)?|\d+(?:[.,:/-]\d+)*|[^\w\s]",
                )
            ),
            re.UNICODE,
        )
        self.sentence_end_pattern = re.compile(_text(self.nlp_config.get("sentence_end_pattern", r"[.!?]+(?:[\"')\]]+)?")))

        self.wordlist = self._load_wordlist(self.wordlist_path)
        self._stopwords = self._load_stopwords(self.stopwords_list_path)
        self.irregular_nouns = self._load_irregular_nouns(self.irregular_nouns_path)
        self._pos_patterns = self._load_pos_patterns(self.pos_patterns_path)
        self.entity_patterns = self._load_entity_patterns(self.entity_patterns_path)
        self.sentiment_lexicon = self._load_sentiment_lexicon(self.sentiment_lexicon_path)
        self.lexical_sets = self._load_lexical_sets()

        self.tokenizer = LanguageTokenizer()
        self.rule_engine = Rules()
        self.memory = LanguageMemory() if self.enable_memory else None
        self.diagnostics = LanguageDiagnostics()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)
        self._process_calls = 0
        self._analysis_calls = 0
        self._last_result: Optional[NLPAnalysisResult] = None

        logger.info("NLP Engine initialized with %s lexical entries", len(self.wordlist))
        printer.status("INIT", "NLP Engine initialized", "success")

    # ------------------------------------------------------------------
    # Compatibility properties
    # ------------------------------------------------------------------
    @property
    def stopwords(self) -> Set[str]:
        return self._stopwords

    @stopwords.setter
    def stopwords(self, value: Iterable[str]) -> None:
        self._stopwords = {_lower(item) for item in value}

    @property
    def pos_patterns(self) -> List[Tuple[re.Pattern[str], str]]:
        return self._pos_patterns

    @pos_patterns.setter
    def pos_patterns(self, value: Iterable[Tuple[re.Pattern[str], str]]) -> None:
        self._pos_patterns = list(value)

    # ------------------------------------------------------------------
    # Public analysis APIs
    # ------------------------------------------------------------------
    def process_text(self, text: str) -> List[Token]:
        """Compatibility API: return a flat list of linguistic tokens."""

        self._process_calls += 1
        return list(self.analyze_text(text).tokens)

    def analyze_text(self, text: str, *, include_dependencies: Optional[bool] = None) -> NLPAnalysisResult:
        """Run sentence/token/POS/lemma/entity/dependency analysis."""

        self._analysis_calls += 1
        original_text = _text(text)
        if len(original_text) > self.max_input_chars:
            issue = NLPIssue(
                code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                message="Input text exceeds the configured NLP maximum length.",
                severity=Severity.ERROR,
                module="NLPEngine",
                details={"max_input_chars": self.max_input_chars, "received": len(original_text)},
            )
            self._add_issue(issue)
            raise NLPError(issue, recoverable=True)

        normalized_text = self._normalize_text(original_text)
        pre_tokens = self._pre_tokenize(normalized_text)
        if len(pre_tokens) > self.max_tokens:
            issue = NLPIssue(
                code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                message="NLP token count exceeds configured maximum.",
                severity=Severity.ERROR,
                module="NLPEngine",
                details={"max_tokens": self.max_tokens, "received": len(pre_tokens)},
            )
            self._add_issue(issue)
            raise NLPError(issue, recoverable=True)

        sentences = self._segment_sentences(normalized_text, pre_tokens)
        token_objects = self._annotate_tokens(pre_tokens, sentences)

        run_dependencies = self.enable_dependency_rules if include_dependencies is None else bool(include_dependencies)
        dependencies: Tuple[DependencyRelation, ...] = ()
        dependency_issues: Tuple[LanguageIssue, ...] = ()
        if run_dependencies:
            dependency_result = self.apply_dependency_rules_detailed(token_objects)
            dependencies = tuple(dependency_result.relations)
            dependency_issues = tuple(issue for issue in dependency_result.issues if isinstance(issue, LanguageIssue))
            token_objects = tuple(self._attach_dependency_heads(list(token_objects), list(dependencies)))
        else:
            token_objects = tuple(token_objects)

        entities: Tuple[Entity, ...] = ()
        if self.enable_entity_extraction:
            entities = tuple(self.extract_entities(list(token_objects)))

        coreferences: Tuple[Entity, ...] = ()
        if self.enable_coreference:
            sentence_token_lists = self._tokens_by_sentence(token_objects)
            coreferences = tuple(self.resolve_coreferences(sentence_token_lists))

        sarcasm_score = self.detect_sarcasm(list(token_objects)) if self.enable_sarcasm_detection else 0.0
        model_tokens = tuple(self._model_tokenize(normalized_text)) if self.track_model_tokens else ()

        issues = tuple(_dedupe([*self.diagnostics.issues, *dependency_issues]))
        result = NLPAnalysisResult(
            original_text=original_text,
            normalized_text=normalized_text,
            tokens=tuple(token_objects),
            sentences=tuple(sentences),
            dependencies=dependencies,
            entities=entities,
            coreferences=coreferences,
            sarcasm_score=sarcasm_score,
            model_tokens=model_tokens,
            issues=issues,
            metadata={
                "language": self.language,
                "locale": self.locale,
                "tokenizer": type(self.tokenizer).__name__,
                "rule_engine": type(self.rule_engine).__name__,
                "analyzed_at": time_module.time(),
            },
        )
        self._last_result = result
        self._record("analyze_text", result={"token_count": len(token_objects), "sentence_count": len(sentences)})
        self._remember_result(result)
        return result

    def analyze_sentences(self, text: str) -> List[List[Token]]:
        """Return tokens grouped by sentence for grammar/coreference callers."""

        return self._tokens_by_sentence(self.analyze_text(text).tokens)

    def apply_dependency_rules(self, tokens: List[Token]) -> List[DependencyRelation]:
        """Compatibility API used by LanguageAgent."""

        return list(self.apply_dependency_rules_detailed(tokens).relations)

    def apply_dependency_rules_detailed(self, tokens: Sequence[Token]) -> RuleApplicationResult:
        """Apply the shared Rules engine and return structured relation output."""

        token_dicts = [token.to_rule_dict() if isinstance(token, Token) else _json_safe(token) for token in tokens]
        result = self.rule_engine.apply(token_dicts)
        for issue in result.issues:
            if isinstance(issue, LanguageIssue):
                self._add_issue(issue)
        self._record("apply_dependency_rules", relation_count=len(result.relations), token_count=len(tokens))
        return result

    def extract_entities(self, tokens: List[Token]) -> List[Entity]:
        """Public entity extraction API used by the language agent."""

        entities: List[Entity] = []
        for sentence_index, sentence_tokens in enumerate(self._tokens_by_sentence(tokens)):
            entities.extend(self._extract_entities(sentence_tokens, sentence_index))
        return entities

    # ------------------------------------------------------------------
    # Tokenization and sentence segmentation
    # ------------------------------------------------------------------
    def _normalize_text(self, text: str) -> str:
        normalized = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
        if _bool(self.nlp_config.get("collapse_whitespace", True), True):
            normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _pre_tokenize(self, text: str) -> List[Dict[str, Any]]:
        tokens: List[Dict[str, Any]] = []
        for index, match in enumerate(self.token_pattern.finditer(text)):
            token_text = match.group(0)
            tokens.append({
                "text": token_text,
                "index": index,
                "start_char": int(match.start()),
                "end_char": int(match.end()),
                "kind": self._token_kind(token_text),
            })
        if not tokens and text:
            issue = TokenizationIssue(
                code=LanguageErrorCode.TOKEN_BOUNDARY_LOSS,
                message="NLP tokenizer could not produce word-level tokens.",
                severity=Severity.WARNING,
                module="NLPEngine",
                source_text=text,
                recoverable=True,
            )
            self._add_issue(issue)
        return tokens

    def _token_kind(self, token: str) -> str:
        if _is_punctuation(token):
            return "punct"
        if _is_number(token):
            return "number"
        if token.startswith("[") and token.endswith("]"):
            return "special"
        if any(char.isalpha() for char in token):
            return "word"
        return "symbol"

    def _segment_sentences(self, text: str, tokens: Sequence[Mapping[str, Any]]) -> List[SentenceAnalysis]:
        if not tokens:
            return []
        sentences: List[SentenceAnalysis] = []
        start_token = 0
        sentence_index = 0
        for index, token in enumerate(tokens):
            token_text = _text(token.get("text"))
            is_end = bool(self.sentence_end_pattern.fullmatch(token_text)) or index == len(tokens) - 1
            if not is_end:
                continue
            sentence_tokens = tokens[start_token : index + 1]
            if not sentence_tokens:
                continue
            start_char = int(sentence_tokens[0].get("start_char", 0))
            end_char = int(sentence_tokens[-1].get("end_char", start_char))
            sentence_text = text[start_char:end_char]
            token_indices = tuple(int(item.get("index", i)) for i, item in enumerate(sentence_tokens, start=start_token))
            sentences.append(
                SentenceAnalysis(
                    text=sentence_text,
                    index=sentence_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_indices=token_indices,
                    sentence_type=self._classify_sentence_surface(sentence_text),
                )
            )
            sentence_index += 1
            start_token = index + 1
        return sentences

    def _classify_sentence_surface(self, text: str) -> str:
        stripped = text.strip()
        if stripped.endswith("?"):
            return "interrogative"
        if stripped.endswith("!"):
            return "exclamatory"
        if stripped.endswith((".", "…")):
            return "declarative"
        return "fragment"

    def _model_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        if hasattr(self.tokenizer, "tokenize"):
            return [str(item) for item in self.tokenizer.tokenize(text)]
        return [item["text"] for item in self._pre_tokenize(text)]

    # ------------------------------------------------------------------
    # POS, lemma, morphology
    # ------------------------------------------------------------------
    def _annotate_tokens(self, pre_tokens: Sequence[Mapping[str, Any]], sentences: Sequence[SentenceAnalysis]) -> Tuple[Token, ...]:
        sentence_lookup: Dict[int, int] = {}
        for sentence in sentences:
            for token_index in sentence.token_indices:
                sentence_lookup[int(token_index)] = sentence.index

        tokens: List[Token] = []
        for index, raw in enumerate(pre_tokens):
            text = _text(raw.get("text"))
            previous_token = tokens[-1] if tokens else None
            next_text = _text(pre_tokens[index + 1].get("text")) if index + 1 < len(pre_tokens) else ""
            pos_tag, pos_confidence = self._get_pos_tag_detailed(text, previous=previous_token, next_text=next_text)
            lemma = self._get_lemma(text, pos_tag)
            feats = self._infer_morphology(text, lemma, pos_tag, previous=previous_token, next_text=next_text)
            normalized = text.casefold()
            entry = self.wordlist.get(normalized)
            # Safe extraction of token index
            raw_index = raw.get("index")
            token_index = int(raw_index) if raw_index is not None else index
            token = Token(
                text=text,
                lemma=lemma,
                pos=pos_tag,
                index=token_index,
                is_stop=normalized in self.stopwords,
                is_punct=pos_tag == "PUNCT",
                start_char = _safe_int(raw.get("start_char")),
                end_char = _safe_int(raw.get("end_char")),
                sentence_index=sentence_lookup.get(int(raw.get("index", index)), 0),
                upos=pos_tag,
                feats=feats,
                normalized=normalized,
                shape=_word_shape(text),
                is_alpha=text.isalpha(),
                is_numeric=_is_number(text),
                is_oov=bool(text.isalpha() and normalized not in self.wordlist),
                confidence=pos_confidence,
                metadata={"lexicon_entry": _json_safe(entry) if isinstance(entry, Mapping) else None, "token_kind": raw.get("kind")},
            )
            tokens.append(token)
        return tuple(tokens)

    def _get_pos_tag(self, word: str) -> str:
        return self._get_pos_tag_detailed(word)[0]

    def _get_pos_tag_detailed(self, word: str, *, previous: Optional[Token] = None, next_text: str = "") -> Tuple[str, float]:
        value = _text(word)
        lower = value.casefold()
        next_lower = next_text.casefold()
        if not value:
            return "X", 0.0
        if _is_punctuation(value):
            return "PUNCT", 0.99
        if _is_number(value):
            return "NUM", 0.95
        if lower in self.lexical_sets["interjections"]:
            return "INTJ", 0.92
        if lower in self.lexical_sets["coordinating_conjunctions"]:
            return "CCONJ", 0.95
        if lower in self.lexical_sets["subordinating_conjunctions"]:
            return "SCONJ", 0.88
        if lower in self.lexical_sets["determiners"]:
            return "DET", 0.95
        if lower in self.lexical_sets["pronouns"]:
            return "PRON", 0.95
        if lower in self.lexical_sets["auxiliaries"] or lower in self.lexical_sets["modals"]:
            return "AUX", 0.95
        if lower == "to" and self._looks_like_verb(next_lower):
            return "PART", 0.92
        if lower in self.lexical_sets["prepositions"]:
            return "ADP", 0.90
        if lower in self.lexical_sets["negators"]:
            return "PART", 0.90

        entry = self.wordlist.get(lower)
        if isinstance(entry, Mapping):
            labels = [_normalize_upos(label) for label in _list(entry.get("upos", entry.get("pos", [])))]
            labels = [label for label in labels if label != "X"]
            if labels:
                return self._choose_pos_from_lexicon(labels, value, previous=previous, next_text=next_text), 0.86

        for pattern, tag in self.pos_patterns:
            if pattern.fullmatch(value):
                return _normalize_upos(tag), 0.78

        heuristic = self._heuristic_pos(value, previous=previous, next_text=next_text)
        if heuristic == "X":
            issue = NLPIssue(
                code=LanguageErrorCode.NLP_POS_UNKNOWN,
                message="Unable to assign a confident POS tag.",
                severity=Severity.INFO,
                module="NLPEngine",
                source_text=value,
                recoverable=True,
            )
            self._add_issue(issue)
            return self.default_unknown_pos, 0.35
        return heuristic, 0.62

    def _choose_pos_from_lexicon(self, labels: Sequence[str], word: str, *, previous: Optional[Token], next_text: str) -> str:
        unique = _dedupe(labels)
        if len(unique) == 1:
            return unique[0]
        lower = word.casefold()
        if previous and previous.pos in {"DET", "ADJ"} and "NOUN" in unique:
            return "NOUN"
        if previous and previous.text.casefold() == "to" and "VERB" in unique:
            return "VERB"
        if lower.endswith("ly") and "ADV" in unique:
            return "ADV"
        if word[:1].isupper() and "PROPN" in unique:
            return "PROPN"
        preference = ["AUX", "VERB", "NOUN", "PROPN", "ADJ", "ADV", "PRON", "DET", "ADP"]
        for label in preference:
            if label in unique:
                return label
        return unique[0]

    def _heuristic_pos(self, word: str, *, previous: Optional[Token], next_text: str) -> str:
        lower = word.casefold()
        if lower.endswith("ly"):
            return "ADV"
        if lower.endswith(("ing", "ed", "en")):
            return "VERB"
        if lower.endswith(("tion", "sion", "ment", "ness", "ity", "ism", "ship", "age", "ery")):
            return "NOUN"
        if lower.endswith(("ous", "ful", "less", "able", "ible", "al", "ive", "ic", "ish", "ary")):
            return "ADJ"
        if word[:1].isupper() and lower not in self.stopwords:
            return "PROPN"
        if previous and previous.pos in {"DET", "ADJ"} and any(char.isalpha() for char in word):
            return "NOUN"
        if self._looks_like_verb(lower):
            return "VERB"
        if word.isalpha():
            return self.default_word_pos
        return "X"

    def _looks_like_verb(self, word: str) -> bool:
        if not word:
            return False
        if word in self.lexical_sets["auxiliaries"] or word in self.lexical_sets["modals"]:
            return True
        if word in getattr(self.rule_engine, "form_to_lemmas", {}):
            return True
        entry = self.wordlist.get(word)
        if isinstance(entry, Mapping):
            labels = {str(label).lower() for label in _list(entry.get("pos", []))}
            return bool(labels.intersection({"verb", "aux", "v"}))
        return word.endswith(("ing", "ed"))

    def _get_lemma(self, word: str, pos: str) -> str:
        value = _text(word)
        if not value or pos == "PUNCT":
            return value
        lower = value.casefold()
        stripped = re.sub(r"(?:['’]s|s['’])$", "", lower)
        entry = self.wordlist.get(lower) or self.wordlist.get(stripped)
        if isinstance(entry, Mapping):
            lemma = entry.get("lemma") or entry.get("base")
            if lemma:
                return _text(lemma).casefold()

        upos = _normalize_upos(pos)
        if upos in {"VERB", "AUX"}:
            candidates = self.rule_engine.lemmatize_verb_form(lower)
            if candidates:
                return candidates[0]
            return self._regular_verb_lemma(lower)
        if upos in {"NOUN", "PROPN"}:
            if lower in self.irregular_nouns:
                return self.irregular_nouns[lower]
            return self._regular_noun_lemma(lower)
        if upos == "ADJ":
            return self._adjective_lemma(lower)
        if upos == "ADV" and lower.endswith("ly") and len(lower) > 4:
            return lower[:-2]
        return stripped or lower

    def _regular_verb_lemma(self, lower: str) -> str:
        if lower.endswith("ies") and len(lower) > 4:
            return lower[:-3] + "y"
        if lower.endswith("ing") and len(lower) > 5:
            base = lower[:-3]
            if len(base) > 2 and base[-1] == base[-2]:
                return base[:-1]
            if base + "e" in self.wordlist:
                return base + "e"
            return base
        if lower.endswith("ied") and len(lower) > 4:
            return lower[:-3] + "y"
        if lower.endswith("ed") and len(lower) > 3:
            base = lower[:-2]
            if len(base) > 2 and base[-1] == base[-2]:
                return base[:-1]
            if base + "e" in self.wordlist:
                return base + "e"
            return base
        if lower.endswith("es") and len(lower) > 3:
            if lower[:-2] in self.wordlist:
                return lower[:-2]
            return lower[:-1]
        if lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
            return lower[:-1]
        return lower

    def _regular_noun_lemma(self, lower: str) -> str:
        if lower.endswith("ies") and len(lower) > 3:
            return lower[:-3] + "y"
        if lower.endswith("ves") and len(lower) > 4:
            return lower[:-3] + "f"
        if lower.endswith("es") and len(lower) > 3 and lower[:-2] in self.wordlist:
            return lower[:-2]
        if lower.endswith("s") and not lower.endswith(("ss", "us", "is")) and len(lower) > 3:
            return lower[:-1]
        return lower

    def _adjective_lemma(self, lower: str) -> str:
        if lower.endswith("iest") and len(lower) > 5:
            return lower[:-4] + "y"
        if lower.endswith("ier") and len(lower) > 4:
            return lower[:-3] + "y"
        if lower.endswith("est") and len(lower) > 4:
            return lower[:-3]
        if lower.endswith("er") and len(lower) > 3:
            return lower[:-2]
        return lower

    def _infer_morphology(self, word: str, lemma: str, pos: str, *, previous: Optional[Token], next_text: str) -> Dict[str, Any]:
        lower = word.casefold()
        upos = _normalize_upos(pos)
        feats: Dict[str, Any] = {}
        if upos == "PRON":
            feats.update(self.DEFAULT_PRONOUN_FEATURES.get(lower, {}))
        elif upos in {"NOUN", "PROPN"}:
            feats["Number"] = "Plur" if self._looks_plural(lower) else "Sing"
            if upos == "PROPN":
                feats["NounType"] = "Prop"
        elif upos in {"VERB", "AUX"}:
            if lower.endswith("ing"):
                feats.update({"VerbForm": "Part", "Tense": "Pres", "Aspect": "Prog"})
            elif lower.endswith(("ed", "en")) or lower in getattr(self.rule_engine, "form_to_lemmas", {}):
                feats.update({"VerbForm": "Fin", "Tense": "Past"})
            elif lower.endswith("s") and lower not in {"is", "was"}:
                feats.update({"VerbForm": "Fin", "Tense": "Pres", "Person": 3, "Number": "Sing"})
            else:
                feats.update({"VerbForm": "Fin", "Tense": "Pres"})
            if upos == "AUX":
                feats["Aux"] = True
        elif upos == "DET":
            feats["Definite"] = "Def" if lower == "the" else "Ind"
        elif upos == "NUM":
            feats["NumType"] = "Card"
        return feats

    def _looks_plural(self, lower: str) -> bool:
        if lower in self.irregular_nouns:
            return True
        return lower.endswith("s") and not lower.endswith(("ss", "us", "is"))

    def _attach_dependency_heads(self, tokens: List[Token], dependencies: List[DependencyRelation]) -> List[Token]:
        relation_by_dependent: Dict[int, DependencyRelation] = {}
        for relation in dependencies:
            if relation.dependent_index not in relation_by_dependent or getattr(relation, "confidence", 0.0) > getattr(relation_by_dependent[relation.dependent_index], "confidence", 0.0):
                relation_by_dependent[relation.dependent_index] = relation
        updated: List[Token] = []
        for token in tokens:
            relation = relation_by_dependent.get(token.index)
            if relation is None:
                updated.append(token)
                continue
            head = token.index if relation.relation == "root" and relation.head == "ROOT" else relation.head_index
            updated.append(
                Token(
                    text=token.text,
                    lemma=token.lemma,
                    pos=token.pos,
                    index=token.index,
                    is_stop=token.is_stop,
                    is_punct=token.is_punct,
                    start_char=token.start_char,
                    end_char=token.end_char,
                    sentence_index=token.sentence_index,
                    token_id=token.token_id,
                    upos=token.upos,
                    xpos=token.xpos,
                    feats=token.feats,
                    dep=relation.relation,
                    head=head,
                    normalized=token.normalized,
                    shape=token.shape,
                    is_alpha=token.is_alpha,
                    is_numeric=token.is_numeric,
                    is_oov=token.is_oov,
                    confidence=token.confidence,
                    metadata={**token.metadata, "dependency": relation.to_dict() if hasattr(relation, "to_dict") else _json_safe(relation)},
                )
            )
        return updated

    # ------------------------------------------------------------------
    # Entity and coreference handling
    # ------------------------------------------------------------------
    def _extract_entities(self, tokens: List[Token], sentence_idx: int) -> List[Entity]:
        entities: List[Entity] = []
        current: List[Token] = []

        def flush() -> None:
            if not current:
                return
            indices = tuple(token.index for token in current)
            text = " ".join(token.text for token in current)
            lower = text.casefold()
            first = current[0]
            last = current[-1]
            entity_type = self._entity_type(current)
            gender, number = self._entity_gender_number(current)
            entities.append(
                Entity(
                    text=text,
                    type=entity_type,
                    gender=gender,
                    number=number,
                    sentence_index=sentence_idx,
                    token_indices=indices,
                    mentions=[(sentence_idx, index) for index in indices],
                    confidence=0.82 if entity_type != "OBJECT" else 0.62,
                    span=(first.start_char or 0, last.end_char or (first.start_char or 0) + len(text)),
                    normalized=lower,
                    metadata={"source": "nlp_chunk", "token_count": len(current)},
                )
            )
            current.clear()

        for token in tokens:
            if self._is_entity_token(token):
                current.append(token)
            else:
                flush()
                pattern_entity = self._pattern_entity(token)
                if pattern_entity:
                    entities.append(pattern_entity)
        flush()
        return entities

    def _is_entity_token(self, token: Token) -> bool:
        if token.pos == "PROPN":
            return True
        if token.pos == "PRON" and token.lower in self.DEFAULT_PRONOUN_FEATURES:
            return True
        if token.pos == "NOUN" and token.lower not in self.stopwords and len(token.text) > 2:
            return True
        return False

    def _entity_type(self, tokens: Sequence[Token]) -> str:
        text = " ".join(token.text for token in tokens)
        lower = text.casefold()
        if all(token.pos == "PRON" for token in tokens):
            return "PRONOUN"
        if any(token.pos == "PROPN" for token in tokens):
            if lower in self.lexical_sets.get("person_names", set()):
                return "PERSON"
            if any(token.text[:1].isupper() for token in tokens):
                return "PROPER_NOUN"
        if re.fullmatch(r"\d+(?:[.,:/-]\d+)*", text):
            return "NUMBER"
        return "OBJECT"

    def _entity_gender_number(self, tokens: Sequence[Token]) -> Tuple[str, str]:
        text = " ".join(token.text for token in tokens).casefold()
        if text in {"he", "him", "his"}:
            return "MASC", "SING"
        if text in {"she", "her", "hers"}:
            return "FEM", "SING"
        if text in {"they", "them", "their", "theirs", "we", "us", "our", "ours"}:
            return "COM", "PLUR"
        if text in {"it", "its"}:
            return "NEUT", "SING"
        number = "PLUR" if len(tokens) > 1 or any(token.feats.get("Number") == "Plur" for token in tokens) else "SING"
        return "UNK", number

    def _pattern_entity(self, token: Token) -> Optional[Entity]:
        for pattern_name, pattern in self.entity_patterns:
            if pattern.fullmatch(token.text):
                return Entity(
                    text=token.text,
                    type=pattern_name,
                    gender="UNK",
                    number="SING",
                    sentence_index=token.sentence_index,
                    token_indices=(token.index,),
                    mentions=[(token.sentence_index, token.index)],
                    confidence=0.9,
                    span=token.span,
                    normalized=token.normalized,
                    metadata={"source": "entity_pattern"},
                )
        return None

    def resolve_coreferences(self, sentences: List[List[Token]]) -> List[Entity]:
        all_entities: List[Entity] = []
        prior_mentions: List[Entity] = []
        next_coref_id = 0
        for sent_idx, sentence in enumerate(sentences):
            for entity in self._extract_entities(sentence, sent_idx):
                matched_id: Optional[int] = None
                if entity.type != "PRONOUN":
                    for previous in reversed(prior_mentions):
                        if previous.type != "PRONOUN" and previous.normalized == entity.normalized:
                            matched_id = previous.coref_id
                            break
                else:
                    for previous in reversed(prior_mentions):
                        if previous.type == "PRONOUN":
                            continue
                        compatible_gender = entity.gender in {"UNK", "COM"} or previous.gender in {"UNK", "COM"} or entity.gender == previous.gender
                        compatible_number = entity.number == "SING" and previous.number == "SING" or entity.number == "PLUR" and previous.number == "PLUR" or entity.number == "COM"
                        close_enough = sent_idx - previous.sentence_index <= _int(self.nlp_config.get("coref_lookback_sentences", 3), 3)
                        if compatible_gender and compatible_number and close_enough:
                            matched_id = previous.coref_id
                            break
                if matched_id is None:
                    matched_id = next_coref_id
                    next_coref_id += 1
                updated = Entity(
                    text=entity.text,
                    type=entity.type,
                    gender=entity.gender,
                    number=entity.number,
                    sentence_index=entity.sentence_index,
                    token_indices=entity.token_indices,
                    mentions=entity.mentions,
                    coref_id=matched_id,
                    confidence=entity.confidence,
                    span=entity.span,
                    normalized=entity.normalized,
                    metadata=entity.metadata,
                )
                prior_mentions.append(updated)
                all_entities.append(updated)
        return all_entities

    # ------------------------------------------------------------------
    # Sentiment and sarcasm
    # ------------------------------------------------------------------
    def detect_sarcasm(self, tokens: List[Token]) -> float:
        if not tokens:
            return 0.0
        text = " ".join(token.text.casefold() for token in tokens)
        lexicon = self.sentiment_lexicon
        positive = set(lexicon.get("positive", {})) if isinstance(lexicon.get("positive"), Mapping) else set(_list(lexicon.get("positive", [])))
        negative = set(lexicon.get("negative", {})) if isinstance(lexicon.get("negative"), Mapping) else set(_list(lexicon.get("negative", [])))
        negators = set(_list(lexicon.get("negators", self.lexical_sets["negators"])))
        intensifiers = lexicon.get("intensifiers", {}) if isinstance(lexicon.get("intensifiers"), Mapping) else {}
        sarcastic_phrases = set(_list(self.nlp_config.get("sarcasm_phrases", [
            "oh great", "big surprise", "what a joy", "as if", "yeah right", "what a shocker", "perfect just perfect"
        ])))
        score = 0.0
        sentiment = 0.0
        negation = False
        intensity = 1.0
        for token in tokens:
            lower = token.lemma.casefold()
            if lower in negators:
                negation = not negation
                continue
            if lower in intensifiers:
                intensity = _float(intensifiers.get(lower), 1.25, minimum=0.0)
                continue
            token_score = 0.0
            if lower in positive:
                token_score = 1.0
            elif lower in negative:
                token_score = -1.0
            if negation:
                token_score *= -1
                negation = False
            sentiment += token_score * intensity
            intensity = 1.0

        has_positive = any(token.lemma.casefold() in positive for token in tokens)
        has_negative = any(token.lemma.casefold() in negative for token in tokens)
        if has_positive and has_negative:
            score += 0.25
        if sentiment > 1.5 and has_negative:
            score += 0.20
        if sentiment < -1.5 and has_positive:
            score += 0.20
        for phrase in sarcastic_phrases:
            if phrase in text:
                score += 0.25
        if re.search(r"[!?]{2,}", text):
            score += 0.15
        if any(token.text.isupper() and len(token.text) > 2 for token in tokens):
            score += 0.10
        if any(token.lemma in {"totally", "completely", "absolutely", "obviously", "surely"} for token in tokens):
            score += 0.10
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    def _load_wordlist(self, path: Optional[str]) -> Dict[str, Dict[str, Any]]:
        payload = _load_json_path(path, {})
        if isinstance(payload, Mapping):
            words = payload.get("words", payload)
            if isinstance(words, Mapping):
                return {str(word).casefold(): dict(data or {}) if isinstance(data, Mapping) else {} for word, data in words.items()}
        return {}

    def _load_stopwords(self, path: Optional[str]) -> Set[str]:
        payload = _load_json_path(path, [])
        if isinstance(payload, Mapping):
            values = payload.get("stopwords", payload.get("words", []))
        else:
            values = payload
        loaded = {_lower(item) for item in _list(values) if _text(item)}
        return loaded or set(self.DEFAULT_STOPWORDS)

    def _load_irregular_nouns(self, path: Optional[str]) -> Dict[str, str]:
        defaults = {
            "children": "child", "men": "man", "women": "woman", "people": "person", "mice": "mouse",
            "geese": "goose", "teeth": "tooth", "feet": "foot", "data": "datum", "criteria": "criterion",
        }
        payload = _load_json_path(path, {})
        if isinstance(payload, Mapping):
            defaults.update({str(k).casefold(): str(v).casefold() for k, v in payload.items()})
        return defaults

    def _load_pos_patterns(self, path: Optional[str]) -> List[Tuple[re.Pattern[str], str]]:
        patterns: List[Tuple[str, str]] = []
        configured = self.nlp_config.get("pos_patterns", [])
        for item in _list(configured):
            if isinstance(item, Mapping) and item.get("pattern") and item.get("tag"):
                patterns.append((_text(item.get("pattern")), _text(item.get("tag"))))
        payload = _load_json_path(path, [])
        for item in _list(payload):
            if not isinstance(item, Mapping):
                continue
            pattern = item.get("pattern")
            if isinstance(pattern, list):
                continue
            tag = item.get("tag") or (item.get("example_tags") or [None])[0]
            if pattern and tag:
                patterns.append((_text(pattern), _text(tag)))
        compiled: List[Tuple[re.Pattern[str], str]] = []
        for pattern, tag in patterns:
            compiled.append((re.compile(pattern, re.UNICODE | re.IGNORECASE), _normalize_upos(tag)))
        return compiled

    def _load_entity_patterns(self, path: Optional[str]) -> List[Tuple[str, re.Pattern[str]]]:
        defaults = {
            "EMAIL": r"^[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}$",
            "URL": r"^https?://|^www\.",
            "DATE": r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{2,4}$",
            "TIME": r"^\d{1,2}:\d{2}(?:am|pm)?$",
        }
        configured = _load_json_path(path, {})
        if isinstance(configured, Mapping):
            defaults.update({str(k): str(v) for k, v in configured.items()})
        return [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in defaults.items()]

    def _load_sentiment_lexicon(self, path: Optional[str]) -> Dict[str, Any]:
        payload = _load_json_path(path, {})
        if isinstance(payload, Mapping):
            return dict(payload)
        return {
            "positive": ["good", "great", "excellent", "perfect", "helpful", "joy", "love"],
            "negative": ["bad", "terrible", "awful", "wrong", "hate", "difficult", "problem"],
            "negators": ["not", "never", "no", "n't"],
            "intensifiers": {"very": 1.25, "really": 1.2, "absolutely": 1.4, "totally": 1.35},
        }

    def _load_lexical_sets(self) -> Dict[str, Set[str]]:
        cfg = self.nlp_config
        return {
            "determiners": set(_lower(item) for item in _list(cfg.get("determiners", ["a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "each", "every", "some", "any", "no"]))),
            "pronouns": set(_lower(item) for item in _list(cfg.get("pronouns", list(self.DEFAULT_PRONOUN_FEATURES.keys())))),
            "auxiliaries": set(_lower(item) for item in _list(cfg.get("auxiliaries", ["be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did"]))),
            "modals": set(_lower(item) for item in _list(cfg.get("modals", ["can", "could", "may", "might", "must", "shall", "should", "will", "would"]))),
            "prepositions": set(_lower(item) for item in _list(cfg.get("prepositions", ["in", "on", "at", "by", "for", "from", "to", "with", "without", "under", "over", "of", "about", "into", "through", "after", "before", "during", "near"]))),
            "coordinating_conjunctions": set(_lower(item) for item in _list(cfg.get("coordinating_conjunctions", ["and", "or", "but", "nor", "so", "yet", "for"]))),
            "subordinating_conjunctions": set(_lower(item) for item in _list(cfg.get("subordinating_conjunctions", ["if", "because", "although", "when", "while", "since", "before", "after", "unless", "that", "whether"]))),
            "negators": set(_lower(item) for item in _list(cfg.get("negators", ["not", "never", "no", "n't"]))),
            "interjections": set(_lower(item) for item in _list(cfg.get("interjections", ["oh", "wow", "hey", "hello", "hi", "ah", "oops"]))),
            "person_names": set(_lower(item) for item in _list(cfg.get("person_names", []))),
        }

    # ------------------------------------------------------------------
    # Utilities, diagnostics, stats
    # ------------------------------------------------------------------
    def _tokens_by_sentence(self, tokens: Sequence[Token]) -> List[List[Token]]:
        grouped: Dict[int, List[Token]] = defaultdict(list)
        for token in tokens:
            grouped[token.sentence_index].append(token)
        return [grouped[index] for index in sorted(grouped)]

    def _add_issue(self, issue: Union[LanguageIssue, LanguageError]) -> None:
        self.diagnostics.add(issue)

    def _record(self, action: str, **payload: Any) -> None:
        if not self.record_history:
            return
        self.history.append({"timestamp": time_module.time(), "action": action, "payload": _json_safe(payload)})

    def _remember_result(self, result: NLPAnalysisResult) -> None:
        if self.memory is None:
            return
        try:
            if self.memory:
                self.memory.remember(
                    kind="note",
                    key="nlp:last_analysis",
                    text=f"NLP analyzed {len(result.tokens)} tokens across {len(result.sentences)} sentence(s).",
                    value=result.to_dict(),
                    tags=("nlp", "analysis"),
                    confidence=1.0,
                    source="NLPEngine",
                    salience=0.3,
                )
        except Exception as exc:
            issue = NLPIssue(
                code=LanguageErrorCode.PIPELINE_STAGE_FAILED,
                message="NLP memory integration failed.",
                severity=Severity.WARNING,
                module="NLPEngine",
                recoverable=True,
                details={"error": str(exc)},
            )
            self._add_issue(issue)
            logger.warning(issue.to_json())

    def diagnostics_result(self) -> LanguageResult[Dict[str, Any]]:
        return LanguageResult(
            data={"stats": self.stats().to_dict(), "last_result": self._last_result.to_dict() if self._last_result else None},
            issues=list(self.diagnostics.issues),
            metadata={"component": "NLPEngine"},
        )

    def stats(self) -> NLPEngineStats:
        last = self._last_result
        return NLPEngineStats(
            language=self.language,
            locale=self.locale,
            token_count=len(last.tokens) if last else 0,
            sentence_count=len(last.sentences) if last else 0,
            entity_count=len(last.entities) if last else 0,
            dependency_count=len(last.dependencies) if last else 0,
            process_calls=self._process_calls,
            analysis_calls=self._analysis_calls,
            diagnostics_count=len(self.diagnostics.issues),
            lexicon_size=len(self.wordlist),
            stopword_count=len(self.stopwords),
            history_length=len(self.history),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "stats": self.stats().to_dict(),
            "diagnostics": self.diagnostics.to_list(),
            "config": _json_safe(self.nlp_config),
        }

    def __repr__(self) -> str:
        return f"<NLPEngine language='{self.language}' locale='{self.locale}' lexicon={len(self.wordlist)} stopwords={len(self.stopwords)}>"


if __name__ == "__main__":
    print("\n=== Running NLP Engine ===\n")
    printer.status("TEST", "NLP Engine initialized", "info")

    engine = NLPEngine()

    sample = "There aren't any resources where we are going, so get packing, friend. Eric said the modules are expanding!"
    result = engine.analyze_text(sample)
    tokens = engine.process_text(sample)
    dependencies = engine.apply_dependency_rules(tokens)
    sentences = engine.analyze_sentences(sample)
    entities = engine.extract_entities(tokens)
    coreferences = engine.resolve_coreferences(sentences)
    sarcasm_tokens = engine.process_text("Oh great, another perfect failure!!")
    sarcasm_score = engine.detect_sarcasm(sarcasm_tokens)

    printer.pretty("ANALYSIS", result.to_dict(), "success")
    printer.pretty("TOKENS", [token.to_dict() for token in tokens], "success")
    printer.pretty("DEPENDENCIES", [dep.to_dict() if hasattr(dep, "to_dict") else _json_safe(dep) for dep in dependencies], "success")
    printer.pretty("ENTITIES", [entity.to_dict() for entity in entities], "success")
    printer.pretty("COREFERENCES", [entity.to_dict() for entity in coreferences], "success")
    printer.pretty("SARCASM", {"score": sarcasm_score}, "success")
    printer.pretty("STATS", engine.stats().to_dict(), "success")

    print("\n=== Test ran successfully ===\n")
