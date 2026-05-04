"""
Natural Language Understanding Engine

Core Function:
Performs Natural Language Understanding — extracting structured meaning from
normalized user input and returning a LinguisticFrame that DialogueContext and
NLGEngine can consume.

Responsibilities:
- Identify intents with deterministic pattern, trigger, keyword, entity, and optional semantic evidence.
- Extract entities with source spans, normalized values, confidence, and validation metadata.
- Compute sentiment, modality, speech-act type, lexical coverage, confidence, and alternatives.
- Integrate with the agent pipeline without secretly re-running NLP or creating another NLPEngine.
- Accept already-computed NLP tokens/dependencies from the LanguageAgent when available.
- Support a shared injected NLPEngine for standalone NLU usage without creating circular imports.
- Preserve the current LanguageAgent contract: NLUEngine(...).parse(text) returns LinguisticFrame.

Pipeline position:
    OrthographyProcessor -> NLPEngine -> GrammarProcessor -> NLUEngine -> DialogueContext -> NLGEngine

Design note:
NLU depends conceptually on NLP annotations, but it should not own a second NLP pipeline by default.
This module therefore uses dependency injection:
    NLUEngine(wordlist_instance=wordlist, nlp_engine=shared_nlp_engine)
or per-call annotations:
    nlu.parse(text, nlp_tokens=tokens, dependencies=dependencies, grammar_result=grammar_result)
If neither is provided, the module falls back to lightweight internal word-tokenization so it remains
usable in direct tests.
"""

from __future__ import annotations

import datetime as datetime_module
import json
import math
import re
import statistics
import yaml

from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

from .utils.config_loader import load_global_config, get_config_section
from .utils.linguistic_frame import LinguisticFrame, SpeechActType
from .utils.language_error import * # type: ignore
from .utils.language_helpers import * # type: ignore
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from .nlp_engine import NLPEngine, Token as NLPToken
    from .modules.rules import DependencyRelation

logger = get_logger("NLU Engine")
printer = PrettyPrinter()

Span = Tuple[int, int]
JsonMap = Dict[str, Any]
TokenLike = Any
DependencyLike = Any


# ---------------------------------------------------------------------------
# Small local coercion helpers
# ---------------------------------------------------------------------------
# These intentionally stay tiny and NLU-specific. Larger generic helpers should
# remain in language_helpers.py. The current subsystem still has mixed helper
# maturity across files, so NLU avoids depending on non-guaranteed helper names.
def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _strip_text(value: Any, default: str = "") -> str:
    return _text(value, default).strip()


def _lower(value: Any) -> str:
    return _strip_text(value).casefold()


def _clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Iterable[Any]) -> List[Any]:
    seen: Set[str] = set()
    output: List[Any] = []
    for value in values:
        marker = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else str(value)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(value)
    return output


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {k: _safe_json(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _read_structured_file(path: Union[str, Path, None]) -> Any:
    if path in (None, "", "none", "None"):
        return None
    file_path = Path(str(path))
    if not file_path.exists():
        return None
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        with file_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _word_tokenize(text: str) -> List[Tuple[str, Span]]:
    pattern = re.compile(r"\b[\w]+(?:['’][\w]+)?\b|[^\w\s]", re.UNICODE)
    return [(match.group(0), (int(match.start()), int(match.end()))) for match in pattern.finditer(text)]


def _phrase_to_regex(phrase: str) -> str:
    cleaned = _strip_text(phrase)
    if not cleaned:
        return r"$a"
    # Treat obvious regex expressions as regex. Plain language examples become phrase patterns.
    regex_markers = {"\\b", "(?:", "(?P", "[", "]", "^", "$", ".*", "\\d", "\\w", "|", "+", "*"}
    if any(marker in cleaned for marker in regex_markers):
        return cleaned
    pieces = [re.escape(part) for part in cleaned.split()]
    return r"\b" + r"\s+".join(pieces) + r"\b"


def _extract_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------
class IntentMatchSource(str, Enum):
    PATTERN = "pattern"
    TRIGGER = "trigger"
    KEYWORD = "keyword"
    EXAMPLE = "example"
    ENTITY = "entity"
    CONTEXT = "context"
    FALLBACK = "fallback"
    EMBEDDING = "embedding"


class EntitySource(str, Enum):
    REGEX = "regex"
    BUILTIN = "builtin"
    WORDLIST = "wordlist"
    NLP = "nlp"
    CONTEXT = "context"


class NLUSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class NLUIssue:
    code: str
    message: str
    severity: NLUSeverity = NLUSeverity.WARNING
    recoverable: bool = True
    details: JsonMap = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "details": _safe_json(self.details),
        }


@dataclass(frozen=True)
class NLUInputToken:
    text: str
    lemma: str
    pos: str
    index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    is_stop: bool = False
    is_punct: bool = False
    morphology: JsonMap = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)

    @property
    def lower(self) -> str:
        return self.text.casefold()

    @property
    def span(self) -> Optional[Span]:
        if self.start_char is None or self.end_char is None:
            return None
        return (self.start_char, self.end_char)

    def to_dict(self) -> JsonMap:
        return _safe_json(asdict(self))


@dataclass(frozen=True)
class WordlistEntry:
    word: str
    normalized: str
    pos: Tuple[str, ...] = ()
    synonyms: Tuple[str, ...] = ()
    related_terms: Tuple[str, ...] = ()
    sentiment: float = 0.0
    frequency: float = 0.0
    metadata: JsonMap = field(default_factory=dict)

    def has_pos(self, *labels: str) -> bool:
        wanted = {_lower(label) for label in labels if label}
        return bool(wanted.intersection({_lower(label) for label in self.pos}))

    def to_dict(self) -> JsonMap:
        return _safe_json(asdict(self))


@dataclass(frozen=True)
class IntentPattern:
    intent: str
    pattern: str
    source: IntentMatchSource = IntentMatchSource.PATTERN
    weight: float = 1.0
    priority: int = 0
    required_entities: Tuple[str, ...] = ()
    examples: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()
    act_type: Optional[str] = None
    metadata: JsonMap = field(default_factory=dict)
    flags: int = re.IGNORECASE

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)

    def to_dict(self) -> JsonMap:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload.pop("flags", None)
        return _safe_json(payload)


@dataclass(frozen=True)
class EntityPattern:
    label: str
    pattern: str
    source: EntitySource = EntitySource.REGEX
    normalizer: Optional[str] = None
    confidence: float = 0.85
    priority: int = 0
    validation: Optional[str] = None
    metadata: JsonMap = field(default_factory=dict)
    flags: int = re.IGNORECASE

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)

    def to_dict(self) -> JsonMap:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload.pop("flags", None)
        return _safe_json(payload)


@dataclass(frozen=True)
class IntentCandidate:
    intent: str
    confidence: float
    source: IntentMatchSource
    matched_text: Optional[str] = None
    pattern: Optional[str] = None
    priority: int = 0
    evidence: Tuple[str, ...] = ()
    required_entities: Tuple[str, ...] = ()
    act_type: Optional[str] = None
    metadata: JsonMap = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        payload = asdict(self)
        payload["source"] = self.source.value
        return _safe_json(payload)


@dataclass(frozen=True)
class EntityMention:
    label: str
    text: str
    value: Any
    span: Span
    confidence: float = 0.85
    source: EntitySource = EntitySource.REGEX
    normalized: Optional[str] = None
    metadata: JsonMap = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        payload = asdict(self)
        payload["source"] = self.source.value
        return _safe_json(payload)


@dataclass(frozen=True)
class NLUAnalysisResult:
    text: str
    normalized_text: str
    frame: LinguisticFrame
    tokens: Tuple[NLUInputToken, ...]
    intents: Tuple[IntentCandidate, ...]
    entities: Tuple[EntityMention, ...]
    lexical_coverage: float
    issues: Tuple[NLUIssue, ...] = ()
    dependencies: Tuple[Any, ...] = ()
    grammar_result: Optional[Any] = None
    metadata: JsonMap = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == NLUSeverity.ERROR and not issue.recoverable for issue in self.issues)

    def to_dict(self) -> JsonMap:
        return {
            "ok": self.ok,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "frame": {
                "intent": self.frame.intent,
                "entities": _safe_json(self.frame.entities),
                "sentiment": self.frame.sentiment,
                "modality": self.frame.modality,
                "confidence": self.frame.confidence,
                "act_type": self.frame.act_type.value if isinstance(self.frame.act_type, SpeechActType) else str(self.frame.act_type),
                "propositional_content": self.frame.propositional_content,
                "illocutionary_force": self.frame.illocutionary_force,
                "perlocutionary_effect": self.frame.perlocutionary_effect,
            },
            "tokens": [token.to_dict() for token in self.tokens],
            "intents": [candidate.to_dict() for candidate in self.intents],
            "entities": [entity.to_dict() for entity in self.entities],
            "lexical_coverage": self.lexical_coverage,
            "issues": [issue.to_dict() for issue in self.issues],
            "dependencies": [_safe_json(dep) for dep in self.dependencies],
            "grammar_result": _safe_json(self.grammar_result),
            "metadata": _safe_json(self.metadata),
        }


@dataclass(frozen=True)
class NLUStats:
    parse_calls: int
    pattern_count: int
    entity_pattern_count: int
    wordlist_size: int
    diagnostics_count: int
    history_length: int
    has_shared_nlp_engine: bool
    tokenizer_attached: bool
    transformer_attached: bool

    def to_dict(self) -> JsonMap:
        return _safe_json(asdict(self))


# ---------------------------------------------------------------------------
# Wordlist
# ---------------------------------------------------------------------------
class Wordlist:
    """
    Lightweight structured wordlist used by NLU.

    The previous Wordlist eagerly initialized tokenizer/transformer resources and carried a large
    amount of unrelated keyboard/spelling behavior. This version is deliberately lexical: it loads
    structured entries, supports lookup, rough lexical probability, phonetic candidates, stemming,
    and semantic similarity through synonyms/related terms. Heavy model/tokenizer components should
    be injected into NLUEngine, not owned by Wordlist.
    """

    def __init__(self, n: int = 3, *, path: Optional[Union[str, Path]] = None) -> None:
        self.config = load_global_config()
        self.nlu_config = get_config_section("nlu") or {}
        self.path = Path(str(path or self.nlu_config.get("structured_wordlist_path") or self.config.get("main_wordlist_path") or self.config.get("wordlist_path") or ""))
        self.n = int(n)
        self.data: Dict[str, JsonMap] = {}
        self.entries: Dict[str, WordlistEntry] = {}
        self.metadata: JsonMap = {}
        self.phonetic_index: Dict[str, Set[str]] = defaultdict(set)
        self.ngram_index: Dict[str, Set[str]] = defaultdict(set)
        self.lru_cache: OrderedDict[str, Optional[JsonMap]] = OrderedDict()
        self.cache_limit = int(self.nlu_config.get("wordlist_cache_size", 512) or 512)
        self._load()
        self._precompute_linguistic_data()

    @property
    def vocabulary(self) -> Set[str]:
        return set(self.entries.keys())

    @property
    def words(self) -> Set[str]:
        return self.vocabulary

    def _load(self) -> None:
        if not str(self.path) or not self.path.exists():
            logger.warning("Wordlist path missing or unavailable for NLU: %s", self.path)
            self.data = {}
            self.entries = {}
            self.metadata = {}
            return
        raw = _read_structured_file(self.path)
        if isinstance(raw, Mapping):
            self.metadata = _as_mapping(raw.get("metadata"))
            words_payload = raw.get("words", raw.get("entries", {}))
        else:
            words_payload = raw
        entries: Dict[str, WordlistEntry] = {}
        data: Dict[str, JsonMap] = {}
        if isinstance(words_payload, Mapping):
            iterator = words_payload.items()
        elif isinstance(words_payload, Sequence) and not isinstance(words_payload, (str, bytes, bytearray)):
            iterator = ((str(item), {}) for item in words_payload)
        else:
            iterator = []
        for raw_word, raw_meta in iterator:
            word = _lower(raw_word)
            if not word:
                continue
            meta = _as_mapping(raw_meta)
            entry = WordlistEntry(
                word=str(raw_word),
                normalized=word,
                pos=tuple(_lower(item) for item in _as_list(meta.get("pos", meta.get("upos", []))) if _strip_text(item)),
                synonyms=tuple(_lower(item) for item in _as_list(meta.get("synonyms", [])) if _strip_text(item)),
                related_terms=tuple(_lower(item) for item in _as_list(meta.get("related_terms", meta.get("related", []))) if _strip_text(item)),
                sentiment=_clamp(meta.get("sentiment", 0.0), -1.0, 1.0, 0.0),
                frequency=max(0.0, float(meta.get("frequency", meta.get("freq", meta.get("count", 0.0))) or 0.0)),
                metadata={k: v for k, v in meta.items() if k not in {"pos", "upos", "synonyms", "related_terms", "related", "sentiment", "frequency", "freq", "count"}},
            )
            entries[word] = entry
            data[word] = meta
        self.entries = entries
        self.data = data
        logger.info("NLU Wordlist loaded %s entries from %s", len(self.entries), self.path)

    def _precompute_linguistic_data(self) -> None:
        self.phonetic_index.clear()
        self.ngram_index.clear()
        for word in self.entries:
            self.phonetic_index[self._soundex(word)].add(word)
            for size in range(1, max(1, self.n) + 1):
                padded = f"^{word}$" if size > 1 else word
                for index in range(max(0, len(padded) - size + 1)):
                    self.ngram_index[padded[index:index + size]].add(word)

    def query(self, word: str) -> Optional[JsonMap]:
        key = _lower(word)
        if key in self.lru_cache:
            value = self.lru_cache.pop(key)
            self.lru_cache[key] = value
            return value
        entry = self.entries.get(key)
        value = entry.to_dict() if entry else None
        self.lru_cache[key] = value
        while len(self.lru_cache) > self.cache_limit:
            self.lru_cache.popitem(last=False)
        return value

    def add_word(self, word: str, metadata: Optional[JsonMap] = None) -> None:
        key = _lower(word)
        if not key:
            return
        meta = dict(metadata or {})
        self.data[key] = meta
        self.entries[key] = WordlistEntry(
            word=word,
            normalized=key,
            pos=tuple(_lower(item) for item in _as_list(meta.get("pos", [])) if _strip_text(item)),
            synonyms=tuple(_lower(item) for item in _as_list(meta.get("synonyms", [])) if _strip_text(item)),
            related_terms=tuple(_lower(item) for item in _as_list(meta.get("related_terms", [])) if _strip_text(item)),
            sentiment=_clamp(meta.get("sentiment", 0.0), -1.0, 1.0, 0.0),
            frequency=max(0.0, float(meta.get("frequency", 0.0) or 0.0)),
            metadata=meta,
        )
        self._precompute_linguistic_data()

    def stem(self, word: str) -> str:
        value = _lower(word)
        if len(value) <= 3:
            return value
        for suffix in ("ingly", "edly", "ization", "isation", "fulness", "ousness", "iveness", "tional", "less", "ness", "ment", "ing", "ies", "ied", "ed", "ly", "s"):
            if value.endswith(suffix) and len(value) > len(suffix) + 2:
                if suffix in {"ies", "ied"}:
                    return value[: -len(suffix)] + "y"
                return value[: -len(suffix)]
        return value

    def _soundex(self, word: str) -> str:
        text = re.sub(r"[^A-Za-z]", "", word).upper()
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

    def phonetic_candidates(self, word: str) -> List[str]:
        return sorted(self.phonetic_index.get(self._soundex(_lower(word)), set()))

    def semantic_similarity(self, word1: str, word2: str) -> float:
        a = _lower(word1)
        b = _lower(word2)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        entry_a = self.entries.get(a)
        entry_b = self.entries.get(b)
        related_a = set(entry_a.synonyms + entry_a.related_terms) if entry_a else set()
        related_b = set(entry_b.synonyms + entry_b.related_terms) if entry_b else set()
        if b in related_a or a in related_b:
            return 0.9
        grams_a = {a[i:i + 3] for i in range(max(1, len(a) - 2))}
        grams_b = {b[i:i + 3] for i in range(max(1, len(b) - 2))}
        if not grams_a or not grams_b:
            return 0.0
        return len(grams_a & grams_b) / len(grams_a | grams_b)

    def word_probability(self, word: str, context: Optional[List[str]] = None) -> float:
        key = _lower(word)
        entry = self.entries.get(key)
        if not entry:
            return 0.0
        if entry.frequency > 0:
            max_frequency = max((item.frequency for item in self.entries.values()), default=entry.frequency)
            return _clamp(entry.frequency / max(max_frequency, 1.0))
        score = 0.35
        if context:
            score += min(0.3, sum(self.semantic_similarity(key, item) for item in context) / max(len(context), 1))
        return _clamp(score)

    def context_suggestions(self, previous_words: List[str], limit: int = 5) -> List[Tuple[str, float]]:
        context = [_lower(item) for item in previous_words if _strip_text(item)]
        scores: List[Tuple[str, float]] = []
        for word in self.entries:
            score = self.word_probability(word, context)
            if score > 0:
                scores.append((word, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[: max(1, int(limit))]

    def validate_word(self, word: str) -> bool:
        return _lower(word) in self.entries

    def correct_typo(self, word: str) -> Tuple[str, float]:
        key = _lower(word)
        if key in self.entries:
            return word, 1.0
        candidates = self.phonetic_candidates(key)
        if not candidates:
            return word, 0.0
        best = max(candidates, key=lambda item: self.semantic_similarity(key, item))
        return best, self.semantic_similarity(key, best)

    def __contains__(self, word: str) -> bool:
        return self.validate_word(word)

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# NLU Engine
# ---------------------------------------------------------------------------
class NLUEngine:
    """
    Production NLU engine with pipeline-safe dependency boundaries.

    It does not import or instantiate NLPEngine at runtime. If NLP annotations are needed,
    pass them into parse()/analyze() or inject a shared nlp_engine instance from the agent.
    """

    VERSION = "3.1"

    def __init__(self, wordlist_instance: Optional[Wordlist] = None, *,
        nlp_engine: Optional["NLPEngine"] = None,
        tokenizer: Optional[Any] = None,
        transformer: Optional[Any] = None,
        memory: Optional[Any] = None,
    ) -> None:
        self.config = load_global_config()
        self.nlu_config = get_config_section("nlu") or {}
        self.main_wordlist_path = self.config.get("main_wordlist_path")
        self.modality_markers_path = self.nlu_config.get("modality_markers_path", self.config.get("modality_markers_path"))
        self.sentiment_lexicon_path = self.nlu_config.get("sentiment_lexicon_path")
        self.custom_intent_patterns_path = self.nlu_config.get("custom_intent_patterns_path")
        self.custom_entity_patterns_path = self.nlu_config.get("custom_entity_patterns_path")
        self.morphology_rules_path = self.nlu_config.get("morphology_rules_path")

        self.wordlist = wordlist_instance or Wordlist()
        self.nlp_engine = nlp_engine
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.memory = memory

        self.default_intent = _strip_text(self.nlu_config.get("default_intent", "unknown"), "unknown")
        self.fallback_intent = _strip_text(self.nlu_config.get("fallback_intent", self.default_intent), self.default_intent)
        self.low_confidence_threshold = _clamp(self.nlu_config.get("low_confidence_threshold", 0.45), 0.0, 1.0, 0.45)
        self.intent_margin_threshold = _clamp(self.nlu_config.get("intent_margin_threshold", 0.12), 0.0, 1.0, 0.12)
        self.min_entity_confidence = _clamp(self.nlu_config.get("min_entity_confidence", 0.35), 0.0, 1.0, 0.35)
        self.enable_wordlist_entities = bool(self.nlu_config.get("enable_wordlist_entities", True))
        self.enable_nlp_entities = bool(self.nlu_config.get("enable_nlp_entities", True))
        self.enable_context_intents = bool(self.nlu_config.get("enable_context_intents", True))
        self.max_history = int(self.nlu_config.get("history_limit", 200) or 200)

        self.diagnostics: Deque[NLUIssue] = deque(maxlen=int(self.nlu_config.get("diagnostics_limit", 500) or 500))
        self.history: Deque[JsonMap] = deque(maxlen=self.max_history)
        self._parse_calls = 0

        self.intent_patterns: Dict[str, List[str]] = {}
        self.intent_recognizers: List[Tuple[IntentPattern, re.Pattern[str]]] = []
        self.entity_patterns: Dict[str, Any] = {}
        self.entity_recognizers: List[Tuple[EntityPattern, re.Pattern[str]]] = []
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        self.modality_markers = self._load_modality_markers()

        self._load_intent_resources()
        self._load_entity_resources()

        logger.info("NLU Engine initialized: intents=%s entity_patterns=%s shared_nlp=%s", len(self.intent_patterns), len(self.entity_recognizers), bool(self.nlp_engine))
        printer.status("INIT", "NLU Engine initialized", "success")

    # ------------------------------------------------------------------
    # Dependency injection and compatibility accessors
    # ------------------------------------------------------------------
    def attach_nlp_engine(self, nlp_engine: "NLPEngine") -> None:
        """Attach the agent-owned NLPEngine without importing or constructing one here."""
        self.nlp_engine = nlp_engine

    def attach_tokenizer(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def attach_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

    def get_intents(self) -> Optional[str]:
        return self.custom_intent_patterns_path

    def get_entities(self) -> Optional[str]:
        return self.custom_entity_patterns_path

    def get_modalities(self) -> Optional[str]:
        return self.modality_markers_path

    def get_lexicons(self) -> Optional[str]:
        return self.sentiment_lexicon_path

    def get_morphologies(self) -> Optional[str]:
        return self.morphology_rules_path

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    def _add_issue(self, issue: NLUIssue) -> None:
        self.diagnostics.append(issue)
        if issue.severity == NLUSeverity.ERROR:
            logger.error("NLU issue: %s", issue.to_dict())
        elif issue.severity == NLUSeverity.WARNING:
            logger.warning("NLU issue: %s", issue.to_dict())
        else:
            logger.info("NLU issue: %s", issue.to_dict())

    def _load_intent_resources(self) -> None:
        raw = _read_structured_file(self.custom_intent_patterns_path)
        if raw is None:
            raw = self.nlu_config.get("intent_patterns") or self._default_intent_patterns()
            self._add_issue(NLUIssue("NLU.INTENT.RESOURCE_FALLBACK", "Intent patterns resource was not found; using inline/default intent patterns.", NLUSeverity.INFO))
        processed: Dict[str, List[str]] = {}
        recognizers: List[Tuple[IntentPattern, re.Pattern[str]]] = []

        if not isinstance(raw, Mapping):
            self._add_issue(NLUIssue("NLU.INTENT.INVALID_RESOURCE", "Intent patterns must be a mapping.", NLUSeverity.WARNING, details={"type": type(raw).__name__}))
            raw = self._default_intent_patterns()

        for intent, payload in raw.items():
            intent_name = _strip_text(intent)
            if not intent_name:
                continue
            patterns, meta = self._normalize_intent_payload(payload)
            processed[intent_name] = patterns
            priority = int(meta.get("priority", 0) or 0)
            weight = float(meta.get("weight", 1.0) or 1.0)
            required_entities = tuple(_strip_text(item) for item in _as_list(meta.get("required_entities")) if _strip_text(item))
            act_type = _strip_text(meta.get("act_type") or meta.get("speech_act")) or None
            keywords = tuple(_lower(item) for item in _as_list(meta.get("keywords")) if _strip_text(item))
            examples = tuple(_strip_text(item) for item in _as_list(meta.get("examples")) if _strip_text(item))
            triggers = tuple(_strip_text(item) for item in _as_list(meta.get("triggers")) if _strip_text(item))

            for source_name, values in (
                (IntentMatchSource.PATTERN, patterns),
                (IntentMatchSource.TRIGGER, triggers),
                (IntentMatchSource.KEYWORD, keywords),
                (IntentMatchSource.EXAMPLE, examples),
            ):
                for pattern_text in values:
                    regex_text = _phrase_to_regex(pattern_text) if source_name != IntentMatchSource.PATTERN else _phrase_to_regex(pattern_text)
                    recognizer = IntentPattern(
                        intent=intent_name,
                        pattern=regex_text,
                        source=source_name,
                        weight=weight,
                        priority=priority,
                        required_entities=required_entities,
                        examples=examples,
                        keywords=keywords,
                        act_type=act_type,
                        metadata={"raw": pattern_text},
                    )
                    try:
                        recognizers.append((recognizer, recognizer.compile()))
                    except re.error as exc:
                        self._add_issue(NLUIssue("NLU.INTENT.PATTERN_INVALID", "Intent regex pattern could not be compiled.", NLUSeverity.WARNING, details={"intent": intent_name, "pattern": regex_text, "error": str(exc)}))

        recognizers.sort(key=lambda item: (item[0].priority, len(item[0].pattern), item[0].weight), reverse=True)
        self.intent_patterns = processed
        self.intent_recognizers = recognizers

    def _normalize_intent_payload(self, payload: Any) -> Tuple[List[str], JsonMap]:
        if isinstance(payload, Mapping):
            patterns = [str(item) for item in _as_list(payload.get("patterns", payload.get("pattern", []))) if _strip_text(item)]
            triggers = [str(item) for item in _as_list(payload.get("triggers", [])) if _strip_text(item)]
            keywords = [str(item) for item in _as_list(payload.get("keywords", [])) if _strip_text(item)]
            examples = [str(item) for item in _as_list(payload.get("examples", [])) if _strip_text(item)]
            all_patterns = _dedupe(patterns + triggers + keywords + examples)
            return [str(item) for item in all_patterns], dict(payload)
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            return [str(item) for item in payload if _strip_text(item)], {}
        if _strip_text(payload):
            return [str(payload)], {}
        return [], {}

    def _load_entity_resources(self) -> None:
        raw = _read_structured_file(self.custom_entity_patterns_path)
        if raw is None:
            raw = self.nlu_config.get("entity_patterns") or self._default_entity_patterns()
            self._add_issue(NLUIssue("NLU.ENTITY.RESOURCE_FALLBACK", "Entity patterns resource was not found; using inline/default entity patterns.", NLUSeverity.INFO))
        if not isinstance(raw, Mapping):
            self._add_issue(NLUIssue("NLU.ENTITY.INVALID_RESOURCE", "Entity patterns must be a mapping.", NLUSeverity.WARNING, details={"type": type(raw).__name__}))
            raw = self._default_entity_patterns()

        self.entity_patterns = dict(raw)
        recognizers: List[Tuple[EntityPattern, re.Pattern[str]]] = []
        for label, payload in raw.items():
            label_text = _strip_text(label)
            if not label_text:
                continue
            payload_map = _as_mapping(payload)
            pattern_values = _as_list(payload_map.get("patterns", payload_map.get("pattern", []))) if payload_map else _as_list(payload)
            normalizer = _strip_text(payload_map.get("normalizer")) or None
            validation = _strip_text(payload_map.get("validation")) or None
            priority = int(payload_map.get("priority", 0) or 0)
            confidence = _clamp(payload_map.get("confidence", 0.85), 0.0, 1.0, 0.85)
            source = EntitySource(payload_map.get("source", EntitySource.REGEX.value)) if payload_map.get("source") in EntitySource._value2member_map_ else EntitySource.REGEX
            for pattern_text in pattern_values:
                regex_text = str(pattern_text)
                if not _strip_text(regex_text):
                    continue
                entity_pattern = EntityPattern(
                    label=label_text,
                    pattern=regex_text,
                    source=source,
                    normalizer=normalizer,
                    confidence=confidence,
                    priority=priority,
                    validation=validation,
                    metadata={"raw": _safe_json(payload)},
                )
                try:
                    recognizers.append((entity_pattern, entity_pattern.compile()))
                except re.error as exc:
                    self._add_issue(NLUIssue("NLU.ENTITY.PATTERN_INVALID", "Entity regex pattern could not be compiled.", NLUSeverity.WARNING, details={"label": label_text, "pattern": regex_text, "error": str(exc)}))
        recognizers.sort(key=lambda item: (item[0].priority, item[0].confidence), reverse=True)
        self.entity_recognizers = recognizers

    def _load_sentiment_lexicon(self) -> JsonMap:
        raw = _read_structured_file(self.sentiment_lexicon_path)
        if isinstance(raw, Mapping):
            return dict(raw)
        return {
            "positive": {"good": 0.7, "great": 0.9, "excellent": 1.0, "thanks": 0.5, "helpful": 0.6, "love": 0.9},
            "negative": {"bad": -0.7, "terrible": -1.0, "wrong": -0.6, "hate": -0.9, "problem": -0.5, "issue": -0.4},
            "negators": ["not", "never", "no", "n't"],
            "intensifiers": {"very": 1.25, "really": 1.2, "extremely": 1.5, "slightly": 0.7},
        }

    def _load_modality_markers(self) -> JsonMap:
        raw = _read_structured_file(self.modality_markers_path)
        if isinstance(raw, Mapping):
            return dict(raw)
        return {
            "interrogative": ["who", "what", "where", "when", "why", "how", "can", "could", "would", "should", "?"],
            "imperative": ["please", "show", "tell", "give", "create", "update", "delete", "find", "explain"],
            "conditional": ["if", "unless", "provided", "assuming"],
            "epistemic": ["might", "maybe", "probably", "possibly", "seems", "think", "believe"],
            "deontic": ["must", "should", "need", "ought", "required"],
            "dynamic": ["can", "able", "could"],
        }

    def _default_intent_patterns(self) -> JsonMap:
        return {
            "greeting": {"patterns": ["hello", "hi", "hey", "good morning", "good evening"], "act_type": "expressive", "priority": 1},
            "farewell": {"patterns": ["bye", "goodbye", "see you", "talk later"], "act_type": "expressive", "priority": 1},
            "gratitude": {"patterns": ["thank you", "thanks", "appreciate it"], "act_type": "expressive", "priority": 1},
            "help_request": {"patterns": ["help", "can you help", "what can you do", "i need help"], "act_type": "directive", "priority": 1},
            "time_request": {"patterns": ["what time", "current time", "tell me the time"], "act_type": "directive", "priority": 2},
            "clarification_request": {"patterns": ["what do you mean", "can you clarify", "explain that", "i don't understand"], "act_type": "directive", "priority": 2},
            "question": {"patterns": [r"^(who|what|where|when|why|how)\b", r"\?$"], "act_type": "directive", "priority": 0},
        }

    def _default_entity_patterns(self) -> JsonMap:
        return {
            "EMAIL": {"pattern": r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "confidence": 0.98, "priority": 10},
            "URL": {"pattern": r"\bhttps?://[^\s]+|\bwww\.[^\s]+", "confidence": 0.98, "priority": 10},
            "DATE_TIME": {"pattern": r"\b(?:today|tomorrow|yesterday|tonight|morning|afternoon|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b\d{4}-\d{2}-\d{2}\b", "normalizer": "temporal", "confidence": 0.85},
            "TIME": {"pattern": r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b|\b\d{1,2}\s?(?:am|pm)\b", "normalizer": "time", "confidence": 0.9},
            "NUMBER": {"pattern": r"\b[-+]?\d+(?:\.\d+)?\b", "normalizer": "number", "confidence": 0.9},
            "QUANTITY": {"pattern": r"\b[-+]?\d+(?:\.\d+)?\s?(?:kg|g|m|cm|mm|km|hours?|minutes?|seconds?|days?|weeks?|months?|years?|%|percent)\b", "normalizer": "quantity", "confidence": 0.88},
            "QUOTED_TEXT": {"pattern": r"['\"]([^'\"]{1,200})['\"]", "normalizer": "quoted", "confidence": 0.82},
        }

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------
    def parse(self, text: str, *, nlp_tokens: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        grammar_result: Optional[Any] = None,
        nlp_result: Optional[Any] = None,
        context: Optional[Any] = None,
    ) -> LinguisticFrame:
        """Return the agent-facing LinguisticFrame.

        The method accepts optional NLP artifacts so the LanguageAgent can pass the results it already
        computed. It only calls an injected shared nlp_engine if no tokens were provided.
        """
        return self.analyze(
            text,
            nlp_tokens=nlp_tokens,
            dependencies=dependencies,
            grammar_result=grammar_result,
            nlp_result=nlp_result,
            context=context,
        ).frame

    def analyze(self, text: str, *, nlp_tokens: Optional[Sequence[Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        grammar_result: Optional[Any] = None,
        nlp_result: Optional[Any] = None,
        context: Optional[Any] = None,
    ) -> NLUAnalysisResult:
        printer.status("INIT", "NLU analysis initialized", "info")
        self._parse_calls += 1
        original_text = _text(text)
        normalized_text = self._normalize_text(original_text)
        issues: List[NLUIssue] = []

        tokens = self._resolve_tokens(original_text, nlp_tokens=nlp_tokens, nlp_result=nlp_result)
        deps = tuple(dependencies or self._extract_dependencies(nlp_result))
        entities = self._extract_entities(original_text, tokens=tokens, dependencies=deps, context=context)
        intent_candidates = self._rank_intents(original_text, tokens=tokens, entities=entities, context=context, grammar_result=grammar_result)
        sentiment = self._calculate_sentiment(original_text, tokens=tokens)
        modality = self._detect_modality(original_text, tokens=tokens, grammar_result=grammar_result)
        lexical_coverage = self._lexical_coverage(tokens)
        frame = self._build_frame(
            text=original_text,
            intents=intent_candidates,
            entities=entities,
            sentiment=sentiment,
            modality=modality,
            lexical_coverage=lexical_coverage,
        )

        if frame.confidence < self.low_confidence_threshold:
            issues.append(NLUIssue("NLU.INTENT.LOW_CONFIDENCE", "Best intent confidence is below threshold.", NLUSeverity.WARNING, details={"intent": frame.intent, "confidence": frame.confidence, "threshold": self.low_confidence_threshold}))
        if len(intent_candidates) > 1 and intent_candidates[0].confidence - intent_candidates[1].confidence < self.intent_margin_threshold:
            issues.append(NLUIssue("NLU.INTENT.AMBIGUOUS", "Top intent candidates are close in confidence.", NLUSeverity.WARNING, details={"top": intent_candidates[0].to_dict(), "second": intent_candidates[1].to_dict(), "margin_threshold": self.intent_margin_threshold}))

        result = NLUAnalysisResult(
            text=original_text,
            normalized_text=normalized_text,
            frame=frame,
            tokens=tuple(tokens),
            intents=tuple(intent_candidates),
            entities=tuple(entities),
            lexical_coverage=lexical_coverage,
            issues=tuple(issues),
            dependencies=deps,
            grammar_result=grammar_result,
            metadata={
                "version": self.VERSION,
                "token_source": self._token_source_label(nlp_tokens=nlp_tokens, nlp_result=nlp_result),
                "used_shared_nlp_engine": bool(self.nlp_engine and not nlp_tokens and not nlp_result),
            },
        )
        self._record_analysis(result)
        for issue in issues:
            self._add_issue(issue)
        return result

    # ------------------------------------------------------------------
    # Token and dependency adaptation
    # ------------------------------------------------------------------
    def _resolve_tokens(self, text: str, *, nlp_tokens: Optional[Sequence[Any]], nlp_result: Optional[Any]) -> List[NLUInputToken]:
        if nlp_tokens is not None:
            return self._adapt_tokens(nlp_tokens, original_text=text)
        if nlp_result is not None:
            candidate_tokens = _extract_attr(nlp_result, "tokens", None)
            if candidate_tokens is None:
                sentences = _extract_attr(nlp_result, "sentences", None)
                if sentences:
                    flat: List[Any] = []
                    for sentence in sentences:
                        flat.extend(_extract_attr(sentence, "tokens", sentence) or [])
                    candidate_tokens = flat
            if candidate_tokens is not None:
                return self._adapt_tokens(candidate_tokens, original_text=text)
        if self.nlp_engine is not None and hasattr(self.nlp_engine, "process_text"):
            return self._adapt_tokens(self.nlp_engine.process_text(text), original_text=text)
        return self._fallback_tokens(text)

    def _adapt_tokens(self, tokens: Sequence[Any], *, original_text: str) -> List[NLUInputToken]:
        adapted: List[NLUInputToken] = []
        search_from = 0
        for index, token in enumerate(tokens):
            text = _text(_extract_attr(token, "text", _extract_attr(token, "token", "")))
            if not text:
                continue
            start = _extract_attr(token, "start_char", _extract_attr(token, "start_char_abs", None))
            end = _extract_attr(token, "end_char", _extract_attr(token, "end_char_abs", None))
            if start is None or end is None:
                found = original_text.find(text, search_from)
                if found >= 0:
                    start, end = found, found + len(text)
                    search_from = end
                else:
                    start, end = None, None
            adapted.append(
                NLUInputToken(
                    text=text,
                    lemma=_text(_extract_attr(token, "lemma", text)).casefold(),
                    pos=_text(_extract_attr(token, "pos", _extract_attr(token, "upos", "X"))).upper(),
                    index=int(_extract_attr(token, "index", index) or index),
                    start_char=int(start) if start is not None else None,
                    end_char=int(end) if end is not None else None,
                    is_stop=bool(_extract_attr(token, "is_stop", False)),
                    is_punct=bool(_extract_attr(token, "is_punct", bool(re.fullmatch(r"\W+", text)))),
                    morphology=_as_mapping(_extract_attr(token, "morphology", _extract_attr(token, "feats", {}))),
                    metadata={"source": "nlp"},
                )
            )
        return adapted

    def _fallback_tokens(self, text: str) -> List[NLUInputToken]:
        tokens: List[NLUInputToken] = []
        stopwords = {"a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for", "with"}
        for index, (token_text, span) in enumerate(_word_tokenize(text)):
            lower = token_text.casefold()
            pos = self._infer_pos(lower, token_text)
            tokens.append(
                NLUInputToken(
                    text=token_text,
                    lemma=self.wordlist.stem(lower) if hasattr(self.wordlist, "stem") else lower,
                    pos=pos,
                    index=index,
                    start_char=span[0],
                    end_char=span[1],
                    is_stop=lower in stopwords,
                    is_punct=pos == "PUNCT",
                    metadata={"source": "nlu_fallback"},
                )
            )
        return tokens

    def _infer_pos(self, lower: str, original: str) -> str:
        if re.fullmatch(r"[^\w\s]+", original):
            return "PUNCT"
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", original):
            return "NUM"
        if lower in {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}:
            return "PRON"
        if lower in {"is", "are", "was", "were", "be", "am", "been", "being", "has", "have", "had", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must"}:
            return "AUX"
        entry = self.wordlist.query(lower) if hasattr(self.wordlist, "query") else None
        pos_values = [str(item).lower() for item in _as_list((entry or {}).get("pos", []))]
        if "verb" in pos_values:
            return "VERB"
        if "adjective" in pos_values or "adj" in pos_values:
            return "ADJ"
        if "adverb" in pos_values or "adv" in pos_values:
            return "ADV"
        if "pronoun" in pos_values or "pron" in pos_values:
            return "PRON"
        if lower.endswith("ing") or lower.endswith("ed"):
            return "VERB"
        if lower.endswith("ly"):
            return "ADV"
        if original[:1].isupper() and lower not in {"i"}:
            return "PROPN"
        return "NOUN"

    def _extract_dependencies(self, nlp_result: Optional[Any]) -> Tuple[Any, ...]:
        if nlp_result is None:
            return ()
        deps = _extract_attr(nlp_result, "dependencies", None)
        if deps is not None:
            return tuple(deps)
        sentences = _extract_attr(nlp_result, "sentences", None)
        if sentences:
            collected: List[Any] = []
            for sentence in sentences:
                collected.extend(_extract_attr(sentence, "dependencies", []) or [])
            return tuple(collected)
        return ()

    def _token_source_label(self, *, nlp_tokens: Optional[Sequence[Any]], nlp_result: Optional[Any]) -> str:
        if nlp_tokens is not None:
            return "provided_tokens"
        if nlp_result is not None:
            return "provided_nlp_result"
        if self.nlp_engine is not None:
            return "shared_nlp_engine"
        return "nlu_fallback"

    # ------------------------------------------------------------------
    # Intent recognition
    # ------------------------------------------------------------------
    def _rank_intents(self, text: str, *, tokens: Sequence[NLUInputToken], entities: Sequence[EntityMention], context: Optional[Any], grammar_result: Optional[Any]) -> List[IntentCandidate]:
        text_clean = text.casefold()
        candidates: List[IntentCandidate] = []
        token_lowers = [token.lower for token in tokens if not token.is_punct]
        entity_labels = {entity.label for entity in entities}

        for pattern, compiled in self.intent_recognizers:
            for match in compiled.finditer(text):
                matched = match.group(0)
                base = self._intent_match_score(pattern, matched, text)
                missing_required = [label for label in pattern.required_entities if label not in entity_labels]
                if missing_required:
                    base *= 0.72
                candidates.append(
                    IntentCandidate(
                        intent=pattern.intent,
                        confidence=_clamp(base * pattern.weight),
                        source=pattern.source,
                        matched_text=matched,
                        pattern=pattern.pattern,
                        priority=pattern.priority,
                        evidence=(matched,),
                        required_entities=pattern.required_entities,
                        act_type=pattern.act_type,
                        metadata={"missing_required_entities": missing_required},
                    )
                )

        # Keyword overlap fallback using configured intent keywords/examples.
        for intent, patterns in self.intent_patterns.items():
            words = set()
            for pattern_text in patterns:
                words.update(part.casefold() for part in re.findall(r"\b\w+\b", pattern_text) if len(part) > 2)
            if not words:
                continue
            overlap = len(words.intersection(token_lowers))
            if overlap:
                score = min(0.75, 0.20 + overlap / max(len(words), 1))
                candidates.append(IntentCandidate(intent=intent, confidence=score, source=IntentMatchSource.KEYWORD, evidence=tuple(sorted(words.intersection(token_lowers)))))

        # Entity-driven intent hints.
        if entity_labels:
            entity_intent_hints = _as_mapping(self.nlu_config.get("entity_intent_hints", {}))
            for label in entity_labels:
                for intent in _as_list(entity_intent_hints.get(label, [])):
                    candidates.append(IntentCandidate(intent=str(intent), confidence=0.52, source=IntentMatchSource.ENTITY, evidence=(label,)))

        # Context-driven continuation support without owning DialogueContext.
        if self.enable_context_intents and context is not None:
            last_intent = None
            if hasattr(context, "get_environment_state"):
                last_intent = context.get_environment_state("last_intent") or context.get_environment_state("pending_intent")
            elif isinstance(context, Mapping):
                last_intent = context.get("last_intent") or context.get("pending_intent")
            if last_intent and self._looks_like_followup(text_clean):
                candidates.append(IntentCandidate(intent=str(last_intent), confidence=0.58, source=IntentMatchSource.CONTEXT, evidence=("followup",)))

        if not candidates:
            inferred = "question" if text.strip().endswith("?") else self.fallback_intent
            candidates.append(IntentCandidate(intent=inferred, confidence=0.28 if inferred == self.fallback_intent else 0.48, source=IntentMatchSource.FALLBACK, evidence=("fallback",)))

        merged = self._merge_intent_candidates(candidates)
        merged.sort(key=lambda item: (item.confidence, item.priority, len(" ".join(item.evidence))), reverse=True)
        return merged

    def _intent_match_score(self, pattern: IntentPattern, matched: str, text: str) -> float:
        matched_words = max(1, len(re.findall(r"\b\w+\b", matched)))
        total_words = max(1, len(re.findall(r"\b\w+\b", text)))
        length_score = min(1.0, matched_words / total_words + 0.35)
        source_bonus = {
            IntentMatchSource.PATTERN: 0.18,
            IntentMatchSource.TRIGGER: 0.14,
            IntentMatchSource.KEYWORD: 0.05,
            IntentMatchSource.EXAMPLE: 0.10,
        }.get(pattern.source, 0.0)
        priority_bonus = min(0.12, max(0, pattern.priority) * 0.02)
        return _clamp(0.45 + length_score * 0.35 + source_bonus + priority_bonus)

    def _merge_intent_candidates(self, candidates: Sequence[IntentCandidate]) -> List[IntentCandidate]:
        grouped: Dict[str, List[IntentCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.intent].append(candidate)
        merged: List[IntentCandidate] = []
        for intent, items in grouped.items():
            items_sorted = sorted(items, key=lambda item: item.confidence, reverse=True)
            best = items_sorted[0]
            extra = sum(item.confidence for item in items_sorted[1:]) * 0.15
            confidence = _clamp(best.confidence + min(0.18, extra))
            evidence = tuple(_dedupe(piece for item in items_sorted for piece in item.evidence if piece))
            merged.append(
                IntentCandidate(
                    intent=intent,
                    confidence=confidence,
                    source=best.source,
                    matched_text=best.matched_text,
                    pattern=best.pattern,
                    priority=max(item.priority for item in items_sorted),
                    evidence=evidence,
                    required_entities=best.required_entities,
                    act_type=best.act_type,
                    metadata={"sources": [item.source.value for item in items_sorted], "candidate_count": len(items_sorted)},
                )
            )
        return merged

    def _looks_like_followup(self, text: str) -> bool:
        return bool(re.match(r"^(and|also|what about|how about|then|so|but|yes|no|okay|ok|sure)\b", text)) or any(word in text.split() for word in {"it", "that", "those", "these", "they", "them"})

    # ------------------------------------------------------------------
    # Entity extraction and validation
    # ------------------------------------------------------------------
    def _extract_entities(self, text: str, *, tokens: Sequence[NLUInputToken], dependencies: Sequence[Any], context: Optional[Any]) -> List[EntityMention]:
        entities: List[EntityMention] = []
        for pattern, compiled in self.entity_recognizers:
            for match in compiled.finditer(text):
                raw = match.group(1) if match.groups() else match.group(0)
                span = (int(match.start(1) if match.groups() else match.start()), int(match.end(1) if match.groups() else match.end()))
                value = self._normalize_entity_value(raw, pattern.normalizer or pattern.validation or pattern.label)
                if not self._validate_entity_value(value, pattern.validation or pattern.label):
                    continue
                confidence = _clamp(pattern.confidence)
                if confidence < self.min_entity_confidence:
                    continue
                entities.append(
                    EntityMention(
                        label=pattern.label,
                        text=raw,
                        value=value,
                        span=span,
                        confidence=confidence,
                        source=pattern.source,
                        normalized=_text(value),
                        metadata={"pattern": pattern.pattern, "normalizer": pattern.normalizer, "validation": pattern.validation},
                    )
                )

        if self.enable_nlp_entities:
            entities.extend(self._entities_from_nlp_tokens(tokens))
        if self.enable_wordlist_entities:
            entities.extend(self._entities_from_wordlist(tokens))
        return self._dedupe_entities(entities)

    def _entities_from_nlp_tokens(self, tokens: Sequence[NLUInputToken]) -> List[EntityMention]:
        entities: List[EntityMention] = []
        current: List[NLUInputToken] = []
        for token in list(tokens) + [NLUInputToken("", "", "PUNCT", len(tokens), metadata={"sentinel": True})]:
            if token.pos in {"PROPN"} or (current and token.pos in {"NOUN", "PROPN"} and not token.is_punct):
                current.append(token)
                continue
            if current:
                text = " ".join(item.text for item in current)
                start = current[0].start_char if current[0].start_char is not None else 0
                end = current[-1].end_char if current[-1].end_char is not None else start + len(text)
                entities.append(EntityMention("MENTION", text, text, (int(start), int(end)), 0.58, EntitySource.NLP, text.casefold(), {"pos_sequence": [item.pos for item in current]}))
                current = []
        return entities

    def _entities_from_wordlist(self, tokens: Sequence[NLUInputToken]) -> List[EntityMention]:
        output: List[EntityMention] = []
        entity_pos = {"proper_noun", "name", "location", "organization", "person", "noun"}
        for token in tokens:
            if token.is_punct or token.is_stop or not token.text.strip():
                continue
            entry = self.wordlist.query(token.text) if hasattr(self.wordlist, "query") else None
            if not entry:
                continue
            pos_values = {_lower(item) for item in _as_list(entry.get("pos", []))}
            if token.pos == "PROPN" or pos_values.intersection(entity_pos):
                span = token.span or (0, len(token.text))
                output.append(EntityMention("TERM", token.text, token.lemma or token.text, span, 0.50, EntitySource.WORDLIST, token.lemma or token.lower, {"pos": list(pos_values)}))
        return output

    def _dedupe_entities(self, entities: Sequence[EntityMention]) -> List[EntityMention]:
        best: Dict[Tuple[str, Span, str], EntityMention] = {}
        for entity in entities:
            key = (entity.label, entity.span, _text(entity.value))
            if key not in best or entity.confidence > best[key].confidence:
                best[key] = entity
        return sorted(best.values(), key=lambda item: (item.span[0], -item.confidence, item.label))

    def _normalize_entity_value(self, value: Any, normalizer: str) -> Any:
        text = _strip_text(value)
        mode = _lower(normalizer)
        if mode in {"number", "quantity"}:
            number_match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
            if number_match:
                number_text = number_match.group(0)
                number_value: Union[int, float] = float(number_text) if "." in number_text else int(number_text)
                if mode == "quantity":
                    unit = text[number_match.end():].strip()
                    return {"value": number_value, "unit": unit or None, "text": text}
                return number_value
        if mode in {"temporal", "date_time", "time"}:
            return {"text": text, "type": mode}
        if mode == "quoted":
            return text.strip("'\"")
        if mode == "boolean":
            lowered = text.casefold()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        return text

    def _validate_entity_value(self, value: Any, validator: str) -> bool:
        mode = _lower(validator)
        if mode in {"number"}:
            return isinstance(value, (int, float))
        if mode == "quantity":
            return isinstance(value, Mapping) and isinstance(value.get("value"), (int, float))
        if mode in {"email", "EMAIL".casefold()}:
            return bool(re.fullmatch(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", _text(value)))
        if mode in {"url", "URL".casefold()}:
            return _text(value).startswith(("http://", "https://", "www."))
        return True

    def _entities_to_frame_payload(self, entities: Sequence[EntityMention]) -> JsonMap:
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for entity in entities:
            grouped[entity.label].append(entity.value)
            grouped[entity.label.lower()].append(entity.value)
        payload: JsonMap = {}
        for label, values in grouped.items():
            unique = _dedupe(values)
            payload[label] = unique[0] if len(unique) == 1 else unique
        return payload

    # ------------------------------------------------------------------
    # Sentiment, modality, speech act, confidence
    # ------------------------------------------------------------------
    def _calculate_sentiment(self, text: str, *, tokens: Sequence[NLUInputToken]) -> float:
        positive = _as_mapping(self.sentiment_lexicon.get("positive"))
        negative = _as_mapping(self.sentiment_lexicon.get("negative"))
        negators = {_lower(item) for item in _as_list(self.sentiment_lexicon.get("negators"))}
        intensifiers = _as_mapping(self.sentiment_lexicon.get("intensifiers"))
        score_sum = 0.0
        weight_sum = 0.0
        negate_next = False
        intensity = 1.0
        for token in tokens or self._fallback_tokens(text):
            word = token.lower
            lemma = _lower(token.lemma)
            if word in negators or lemma in negators:
                negate_next = not negate_next
                continue
            if word in intensifiers:
                intensity *= float(intensifiers[word])
                continue
            raw = float(positive.get(word, positive.get(lemma, 0.0)) or 0.0) + float(negative.get(word, negative.get(lemma, 0.0)) or 0.0)
            # Wordlist sentiment can supplement external lexicon.
            entry = self.wordlist.query(lemma) if hasattr(self.wordlist, "query") else None
            if raw == 0.0 and entry:
                raw = float(entry.get("sentiment", 0.0) or 0.0)
            if raw != 0.0:
                if negate_next:
                    raw *= -1.0
                    negate_next = False
                raw *= intensity
                score_sum += raw
                weight_sum += abs(raw)
                intensity = 1.0
        if weight_sum == 0.0:
            return 0.0
        return _clamp(score_sum / weight_sum, -1.0, 1.0, 0.0)

    def _detect_modality(self, text: str, *, tokens: Sequence[NLUInputToken], grammar_result: Optional[Any]) -> str:
        stripped = text.strip()
        lower_text = stripped.casefold()
        token_lowers = [token.lower for token in tokens]
        if stripped.endswith("?"):
            return "interrogative"
        first_word = token_lowers[0] if token_lowers else ""
        if first_word in {_lower(item) for item in _as_list(self.modality_markers.get("imperative"))}:
            return "imperative"
        for modality in ("conditional", "deontic", "epistemic", "dynamic", "interrogative"):
            markers = {_lower(item) for item in _as_list(self.modality_markers.get(modality)) if _strip_text(item) and item != "?"}
            if any(re.search(r"\b" + re.escape(marker) + r"\b", lower_text) for marker in markers):
                return modality
        if first_word in {"who", "what", "where", "when", "why", "how"}:
            return "interrogative"
        return "declarative"

    def _speech_act_for(self, intent: str, modality: str, best: Optional[IntentCandidate]) -> SpeechActType:
        configured = _lower(best.act_type if best else None)
        if configured:
            mapping = {
                "assertive": SpeechActType.ASSERTIVE,
                "representative": SpeechActType.ASSERTIVE,
                "directive": SpeechActType.DIRECTIVE,
                "commissive": SpeechActType.COMMISSIVE,
                "expressive": SpeechActType.EXPRESSIVE,
                "declaration": SpeechActType.DECLARATION,
                "declarative": SpeechActType.DECLARATION,
            }
            if configured in mapping:
                return mapping[configured]
        intent_lower = intent.casefold()
        if modality in {"interrogative", "imperative"} or any(marker in intent_lower for marker in ("request", "question", "help", "find", "create", "update", "delete", "clarification")):
            return SpeechActType.DIRECTIVE
        if any(marker in intent_lower for marker in ("thanks", "gratitude", "greeting", "farewell", "apology")):
            return SpeechActType.EXPRESSIVE
        return SpeechActType.ASSERTIVE

    def _lexical_coverage(self, tokens: Sequence[NLUInputToken]) -> float:
        lexical = [token for token in tokens if not token.is_punct and token.text.strip()]
        if not lexical:
            return 0.0
        known = 0
        for token in lexical:
            if token.lower in self.wordlist or (token.lemma and token.lemma in self.wordlist):
                known += 1
        return known / len(lexical)

    def _overall_confidence(self, *, best_intent: IntentCandidate, entities: Sequence[EntityMention], sentiment: float, lexical_coverage: float) -> float:
        entity_bonus = min(0.12, 0.03 * len(entities))
        coverage_bonus = min(0.12, lexical_coverage * 0.12)
        sentiment_bonus = min(0.04, abs(sentiment) * 0.04)
        return _clamp(best_intent.confidence + entity_bonus + coverage_bonus + sentiment_bonus)

    def _build_frame(self, *, text: str, intents: Sequence[IntentCandidate], entities: Sequence[EntityMention], sentiment: float, modality: str, lexical_coverage: float) -> LinguisticFrame:
        best = intents[0] if intents else IntentCandidate(self.default_intent, 0.0, IntentMatchSource.FALLBACK)
        confidence = self._overall_confidence(best_intent=best, entities=entities, sentiment=sentiment, lexical_coverage=lexical_coverage)
        act_type = self._speech_act_for(best.intent, modality, best)
        return LinguisticFrame(
            intent=best.intent,
            entities=self._entities_to_frame_payload(entities),
            sentiment=sentiment,
            modality=modality,
            confidence=confidence,
            act_type=act_type,
            propositional_content=text,
            illocutionary_force=self._illocutionary_force(best.intent, modality, act_type),
        )

    def _illocutionary_force(self, intent: str, modality: str, act_type: SpeechActType) -> str:
        if act_type == SpeechActType.DIRECTIVE:
            return "request_information" if modality == "interrogative" else "request_action"
        if act_type == SpeechActType.EXPRESSIVE:
            return "express_attitude"
        if act_type == SpeechActType.COMMISSIVE:
            return "commit_action"
        if act_type == SpeechActType.DECLARATION:
            return "declare_state"
        return "inform"

    # ------------------------------------------------------------------
    # Compatibility helpers from the old NLUEngine
    # ------------------------------------------------------------------
    def _match_intent_by_pattern(self, text: str) -> List[Tuple[str, str, float]]:
        candidates = self._rank_intents(text, tokens=self._fallback_tokens(text), entities=(), context=None, grammar_result=None)
        return [(item.intent, item.matched_text or " ".join(item.evidence), item.confidence) for item in candidates]

    def _tokenize(self, text: str) -> List[str]:
        return [token for token, _span in _word_tokenize(text)]

    def _calculate_confidence(self, matches: List[Tuple[str, str, float]]) -> float:
        if not matches:
            return 0.0
        return _clamp(max(float(match[2]) for match in matches))

    def _calculate_overall_confidence(self, text: str, frame: LinguisticFrame) -> float:
        tokens = self._fallback_tokens(text)
        coverage = self._lexical_coverage(tokens)
        base = _clamp(frame.confidence)
        return _clamp(base + min(0.12, coverage * 0.12) + min(0.12, 0.03 * len(frame.entities or {})))

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", _text(text)).strip()

    def _record_analysis(self, result: NLUAnalysisResult) -> None:
        self.history.append({
            "timestamp": datetime_module.datetime.utcnow().isoformat() + "Z",
            "intent": result.frame.intent,
            "confidence": result.frame.confidence,
            "entity_count": len(result.entities),
            "issue_count": len(result.issues),
            "text_preview": result.text[:160],
        })
        if self.memory is not None and hasattr(self.memory, "add_intent"):
            try:
                self.memory.add_intent(result.frame.intent, confidence=result.frame.confidence, frame=result.frame, source="nlu")
            except Exception as exc:
                logger.warning("Could not store NLU intent in language memory: %s", exc)

    def stats(self) -> NLUStats:
        return NLUStats(
            parse_calls=self._parse_calls,
            pattern_count=len(self.intent_recognizers),
            entity_pattern_count=len(self.entity_recognizers),
            wordlist_size=len(self.wordlist) if hasattr(self.wordlist, "__len__") else 0,
            diagnostics_count=len(self.diagnostics),
            history_length=len(self.history),
            has_shared_nlp_engine=self.nlp_engine is not None,
            tokenizer_attached=self.tokenizer is not None,
            transformer_attached=self.transformer is not None,
        )

    def diagnostics_result(self) -> List[JsonMap]:
        return [issue.to_dict() for issue in self.diagnostics]

    def to_dict(self) -> JsonMap:
        return {
            "component": self.__class__.__name__,
            "version": self.VERSION,
            "stats": self.stats().to_dict(),
            "intent_patterns": self.intent_patterns,
            "entity_pattern_count": len(self.entity_recognizers),
            "diagnostics": self.diagnostics_result(),
        }


# ---------------------------------------------------------------------------
# Enhanced NLU
# ---------------------------------------------------------------------------
class EnhancedNLU(NLUEngine):
    """Extended analysis facade that preserves the old class name without undefined parser dependencies."""

    def __init__(self, wordlist_instance: Optional[Wordlist] = None, **kwargs: Any) -> None:
        super().__init__(wordlist_instance=wordlist_instance, **kwargs)
        self.context_history: Optional[Deque[Any]] = None
        logger.info("EnhancedNLU initialized.")

    def set_context_history(self, history: Deque[Any]) -> None:
        self.context_history = history

    def analyze_text_fully(self, text: str, *, nlp_tokens: Optional[Sequence[Any]] = None, context: Optional[Any] = None,
                           dependencies: Optional[Sequence[Any]] = None, grammar_result: Optional[Any] = None,
                           nlp_result: Optional[Any] = None) -> Dict[str, Any]:
        result = self.analyze(
            text,
            nlp_tokens=nlp_tokens,
            dependencies=dependencies,
            grammar_result=grammar_result,
            nlp_result=nlp_result,
            context=context,
        )
        payload = result.to_dict()
        payload["psycholinguistic"] = self._psycholinguistic_features(result.tokens)
        payload["intent_alternatives"] = [candidate.to_dict() for candidate in result.intents[1:]]
        return payload

    def _psycholinguistic_features(self, tokens: Sequence[NLUInputToken]) -> JsonMap:
        lexical = [token.text for token in tokens if not token.is_punct]
        unique = {item.casefold() for item in lexical}
        lengths = [len(item) for item in lexical if item]
        return {
            "token_count": len(lexical),
            "type_count": len(unique),
            "type_token_ratio": 0.0 if not lexical else len(unique) / len(lexical),
            "mean_token_length": 0.0 if not lengths else statistics.mean(lengths),
        }


if __name__ == "__main__":
    print("\n=== Running NLU Engine ===\n")
    printer.status("TEST", "NLU Engine initialized", "info")

    wordlist = Wordlist()
    engine = NLUEngine(wordlist_instance=wordlist)

    samples = [
        "Hello, can you help me find the current time?",
        "Please create a reminder for tomorrow at 09:30.",
        "I really love how helpful this language agent is.",
        "What do you mean by 'shared NLP engine'?",
    ]

    for sample in samples:
        result = engine.analyze(sample)
        printer.pretty("NLU_RESULT", result.to_dict(), "success")

    enhanced = EnhancedNLU(wordlist_instance=wordlist)
    full = enhanced.analyze_text_fully("Could you explain this tomorrow?")
    printer.pretty("ENHANCED_NLU", full, "success")
    printer.pretty("STATS", engine.stats().to_dict(), "success")

    print("\n=== Test ran successfully ===\n")
