"""Natural Language Understanding Engine for SLAI v2.3.

Deterministic NLU remains authoritative for normal cases. When and only when
intent recognition is ambiguous or below the configured confidence threshold,
NLUEngine requests semantic evidence from the injected LANTRA runtime. There is
no independent semantic-evidence module and no feature flag that can silently
bypass LANTRA for an ambiguous case when a ready runtime is attached.
"""
from __future__ import annotations

import datetime as datetime_module
import json
import re
import statistics
import yaml

from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

from .utils.config_loader import load_global_config, get_config_section
from .utils.linguistic_frame import LinguisticFrame, SpeechActType
from .utils.language_error import *
from .utils.language_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from .nlp_engine import NLPEngine

logger = get_logger("NLU Engine")
printer = PrettyPrinter()

Span = Tuple[int, int]
JsonMap = Dict[str, Any]


def _text(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


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
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Iterable[Any]) -> List[Any]:
    seen: Set[str] = set()
    output: List[Any] = []
    for value in values:
        marker = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else str(value)
        if marker not in seen:
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
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {} if file_path.suffix.lower() in {".yaml", ".yml"} else json.load(handle)


def _word_tokenize(text: str) -> List[Tuple[str, Span]]:
    pattern = re.compile(r"\b[\w]+(?:['’][\w]+)?\b|[^\w\s]", re.UNICODE)
    return [(m.group(0), (int(m.start()), int(m.end()))) for m in pattern.finditer(text)]


def _phrase_to_regex(phrase: str) -> str:
    cleaned = _strip_text(phrase)
    if not cleaned:
        return r"$a"
    regex_markers = {"\\b", "(?:", "(?P", "[", "]", "^", "$", ".*", "\\d", "\\w", "|", "+", "*"}
    if any(marker in cleaned for marker in regex_markers):
        return cleaned
    return r"\b" + r"\s+".join(re.escape(part) for part in cleaned.split()) + r"\b"


def _extract_attr(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


class IntentMatchSource(str, Enum):
    PATTERN = "pattern"
    TRIGGER = "trigger"
    KEYWORD = "keyword"
    EXAMPLE = "example"
    ENTITY = "entity"
    CONTEXT = "context"
    FALLBACK = "fallback"
    EMBEDDING = "embedding"
    LANTRA = "lantra"


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
        return None if self.start_char is None or self.end_char is None else (self.start_char, self.end_char)

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
            "ok": self.ok, "text": self.text, "normalized_text": self.normalized_text,
            "frame": {"intent": self.frame.intent, "entities": _safe_json(self.frame.entities), "sentiment": self.frame.sentiment, "modality": self.frame.modality, "confidence": self.frame.confidence, "act_type": self.frame.act_type.value if isinstance(self.frame.act_type, SpeechActType) else str(self.frame.act_type), "propositional_content": self.frame.propositional_content, "illocutionary_force": self.frame.illocutionary_force, "perlocutionary_effect": self.frame.perlocutionary_effect},
            "tokens": [t.to_dict() for t in self.tokens], "intents": [i.to_dict() for i in self.intents], "entities": [e.to_dict() for e in self.entities], "lexical_coverage": self.lexical_coverage,
            "issues": [i.to_dict() for i in self.issues], "dependencies": [_safe_json(d) for d in self.dependencies], "grammar_result": _safe_json(self.grammar_result), "metadata": _safe_json(self.metadata),
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
    semantic_runtime_attached: bool

    def to_dict(self) -> JsonMap:
        return _safe_json(asdict(self))


class Wordlist:
    def __init__(self, n: int = 3, *, path: Optional[Union[str, Path]] = None) -> None:
        self.config = load_global_config()
        self.nlu_config = get_config_section("nlu") or {}
        self.path = Path(str(path or self.nlu_config.get("structured_wordlist_path") or self.config.get("main_wordlist_path") or self.config.get("wordlist_path") or ""))
        self.n = int(n)
        self.data: Dict[str, JsonMap] = {}
        self.entries: Dict[str, WordlistEntry] = {}
        self.metadata: JsonMap = {}
        self.lru_cache: OrderedDict[str, Optional[JsonMap]] = OrderedDict()
        self.cache_limit = int(self.nlu_config.get("wordlist_cache_size", 512) or 512)
        self._load()

    def _load(self) -> None:
        if not str(self.path) or not self.path.exists():
            logger.warning("Wordlist path missing or unavailable for NLU: %s", self.path)
            return
        raw = _read_structured_file(self.path)
        words_payload = raw.get("words", raw.get("entries", {})) if isinstance(raw, Mapping) else raw
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
            entry = WordlistEntry(word=str(raw_word), normalized=word, pos=tuple(_lower(x) for x in _as_list(meta.get("pos", meta.get("upos", []))) if _strip_text(x)), synonyms=tuple(_lower(x) for x in _as_list(meta.get("synonyms", [])) if _strip_text(x)), related_terms=tuple(_lower(x) for x in _as_list(meta.get("related_terms", meta.get("related", []))) if _strip_text(x)), sentiment=_clamp(meta.get("sentiment", 0.0), -1.0, 1.0, 0.0), frequency=max(0.0, float(meta.get("frequency", meta.get("freq", meta.get("count", 0.0))) or 0.0)), metadata=meta)
            self.entries[word] = entry
            self.data[word] = meta

    @property
    def vocabulary(self) -> Set[str]:
        return set(self.entries)

    @property
    def words(self) -> Set[str]:
        return self.vocabulary

    def query(self, word: str) -> Optional[JsonMap]:
        key = _lower(word)
        if key in self.lru_cache:
            value = self.lru_cache.pop(key); self.lru_cache[key] = value; return value
        entry = self.entries.get(key)
        value = entry.to_dict() if entry else None
        self.lru_cache[key] = value
        while len(self.lru_cache) > self.cache_limit:
            self.lru_cache.popitem(last=False)
        return value

    def stem(self, word: str) -> str:
        value = _lower(word)
        if len(value) <= 3:
            return value
        for suffix in ("ingly", "edly", "ization", "isation", "fulness", "ousness", "iveness", "tional", "less", "ness", "ment", "ing", "ies", "ied", "ed", "ly", "s"):
            if value.endswith(suffix) and len(value) > len(suffix) + 2:
                return value[:-len(suffix)] + ("y" if suffix in {"ies", "ied"} else "")
        return value

    def semantic_similarity(self, word1: str, word2: str) -> float:
        a, b = _lower(word1), _lower(word2)
        if not a or not b: return 0.0
        if a == b: return 1.0
        ea, eb = self.entries.get(a), self.entries.get(b)
        ra = set(ea.synonyms + ea.related_terms) if ea else set()
        rb = set(eb.synonyms + eb.related_terms) if eb else set()
        if b in ra or a in rb: return 0.9
        ga = {a[i:i+3] for i in range(max(1, len(a)-2))}; gb = {b[i:i+3] for i in range(max(1, len(b)-2))}
        return len(ga & gb) / len(ga | gb) if ga and gb else 0.0

    def add_word(self, word: str, metadata: Optional[JsonMap] = None) -> None:
        key = _lower(word)
        if not key:
            return
        meta = dict(metadata or {})
        self.data[key] = meta
        self.entries[key] = WordlistEntry(
            word=word, normalized=key,
            pos=tuple(_lower(item) for item in _as_list(meta.get("pos", [])) if _strip_text(item)),
            synonyms=tuple(_lower(item) for item in _as_list(meta.get("synonyms", [])) if _strip_text(item)),
            related_terms=tuple(_lower(item) for item in _as_list(meta.get("related_terms", [])) if _strip_text(item)),
            sentiment=_clamp(meta.get("sentiment", 0.0), -1.0, 1.0, 0.0),
            frequency=max(0.0, float(meta.get("frequency", 0.0) or 0.0)), metadata=meta,
        )
        self.lru_cache.pop(key, None)

    @staticmethod
    def _soundex(word: str) -> str:
        text = re.sub(r"[^A-Za-z]", "", word).upper()
        if not text:
            return "0000"
        first = text[0]
        mapping = {"B":"1","F":"1","P":"1","V":"1","C":"2","G":"2","J":"2","K":"2","Q":"2","S":"2","X":"2","Z":"2","D":"3","T":"3","L":"4","M":"5","N":"5","R":"6"}
        digits: List[str] = []
        previous = mapping.get(first, "")
        for char in text[1:]:
            code = mapping.get(char, "")
            if code and code != previous:
                digits.append(code)
            previous = code
        return (first + "".join(digits) + "000")[:4]

    def phonetic_candidates(self, word: str) -> List[str]:
        code = self._soundex(_lower(word))
        return sorted(candidate for candidate in self.entries if self._soundex(candidate) == code)

    def word_probability(self, word: str, context: Optional[List[str]] = None) -> float:
        entry = self.entries.get(_lower(word))
        if entry is None:
            return 0.0
        if entry.frequency > 0.0:
            maximum = max((item.frequency for item in self.entries.values()), default=entry.frequency)
            return _clamp(entry.frequency / max(maximum, 1.0))
        score = 0.35
        if context:
            score += min(0.30, sum(self.semantic_similarity(entry.normalized, item) for item in context) / max(len(context), 1))
        return _clamp(score)

    def context_suggestions(self, previous_words: List[str], limit: int = 5) -> List[Tuple[str, float]]:
        scored = [(word, self.word_probability(word, previous_words)) for word in self.entries]
        scored = [item for item in scored if item[1] > 0.0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, int(limit))]

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


class NLUEngine:
    VERSION = "3.2"

    def __init__(self, wordlist_instance: Optional[Wordlist] = None, *, nlp_engine: Optional["NLPEngine"] = None, tokenizer: Optional[Any] = None, transformer: Optional[Any] = None, memory: Optional[Any] = None, semantic_runtime: Optional[Any] = None) -> None:
        self.config = load_global_config()
        self.nlu_config = get_config_section("nlu") or {}
        self.wordlist = wordlist_instance or Wordlist()
        self.nlp_engine = nlp_engine
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.memory = memory
        self.semantic_runtime = semantic_runtime
        self.default_intent = _strip_text(self.nlu_config.get("default_intent", "unknown"), "unknown")
        self.fallback_intent = _strip_text(self.nlu_config.get("fallback_intent", self.default_intent), self.default_intent)
        self.low_confidence_threshold = _clamp(self.nlu_config.get("low_confidence_threshold", 0.45), 0.0, 1.0, 0.45)
        self.intent_margin_threshold = _clamp(self.nlu_config.get("intent_margin_threshold", 0.12), 0.0, 1.0, 0.12)
        # These tune acceptance of LANTRA evidence. There is intentionally no
        # enabled/disabled flag: ambiguous NLU consults an attached ready runtime.
        self.semantic_min_loss_margin = max(0.0, float(self.nlu_config.get("semantic_min_loss_margin", 0.15) or 0.15))
        self.semantic_confidence_floor = _clamp(self.nlu_config.get("semantic_confidence_floor", 0.60), 0.0, 1.0, 0.60)
        self.semantic_max_labels = max(2, int(self.nlu_config.get("semantic_max_labels", 32) or 32))
        self.min_entity_confidence = _clamp(self.nlu_config.get("min_entity_confidence", 0.35), 0.0, 1.0, 0.35)
        self.enable_wordlist_entities = bool(self.nlu_config.get("enable_wordlist_entities", True))
        self.enable_nlp_entities = bool(self.nlu_config.get("enable_nlp_entities", True))
        self.enable_context_intents = bool(self.nlu_config.get("enable_context_intents", True))
        self.diagnostics: Deque[NLUIssue] = deque(maxlen=int(self.nlu_config.get("diagnostics_limit", 500) or 500))
        self.history: Deque[JsonMap] = deque(maxlen=int(self.nlu_config.get("history_limit", 200) or 200))
        self._parse_calls = 0
        self.intent_patterns: Dict[str, List[str]] = {}
        self.intent_recognizers: List[Tuple[IntentPattern, re.Pattern[str]]] = []
        self.entity_patterns: Dict[str, Any] = {}
        self.entity_recognizers: List[Tuple[EntityPattern, re.Pattern[str]]] = []
        self.custom_intent_patterns_path = self.nlu_config.get("custom_intent_patterns_path")
        self.custom_entity_patterns_path = self.nlu_config.get("custom_entity_patterns_path")
        self.sentiment_lexicon_path = self.nlu_config.get("sentiment_lexicon_path")
        self.modality_markers_path = self.nlu_config.get("modality_markers_path")
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        self.modality_markers = self._load_modality_markers()
        self._load_intent_resources()
        self._load_entity_resources()
        logger.info("NLU Engine initialized: intents=%s semantic_runtime=%s", len(self.intent_patterns), bool(self.semantic_runtime))

    def attach_nlp_engine(self, nlp_engine: "NLPEngine") -> None:
        self.nlp_engine = nlp_engine

    def attach_semantic_runtime(self, runtime: Any) -> None:
        self.semantic_runtime = runtime

    def _add_issue(self, issue: NLUIssue) -> None:
        self.diagnostics.append(issue)
        if issue.severity == NLUSeverity.ERROR: logger.error("NLU issue: %s", issue.to_dict())
        elif issue.severity == NLUSeverity.WARNING: logger.warning("NLU issue: %s", issue.to_dict())

    def _load_intent_resources(self) -> None:
        raw = _read_structured_file(self.custom_intent_patterns_path) or self.nlu_config.get("intent_patterns") or self._default_intent_patterns()
        if not isinstance(raw, Mapping): raw = self._default_intent_patterns()
        for intent, payload in raw.items():
            intent_name = _strip_text(intent)
            meta = _as_mapping(payload)
            values = _as_list(meta.get("patterns", meta.get("pattern", payload))) if isinstance(payload, Mapping) else _as_list(payload)
            self.intent_patterns[intent_name] = [str(v) for v in values if _strip_text(v)]
            priority = int(meta.get("priority", 0) or 0); weight = float(meta.get("weight", 1.0) or 1.0); act_type = _strip_text(meta.get("act_type") or meta.get("speech_act")) or None
            streams = [(IntentMatchSource.PATTERN, values), (IntentMatchSource.TRIGGER, _as_list(meta.get("triggers"))), (IntentMatchSource.KEYWORD, _as_list(meta.get("keywords"))), (IntentMatchSource.EXAMPLE, _as_list(meta.get("examples")))]
            for source, stream in streams:
                for value in stream:
                    if not _strip_text(value): continue
                    item = IntentPattern(intent=intent_name, pattern=_phrase_to_regex(str(value)), source=source, weight=weight, priority=priority, act_type=act_type)
                    try: self.intent_recognizers.append((item, item.compile()))
                    except re.error as exc: self._add_issue(NLUIssue("NLU.INTENT.PATTERN_INVALID", str(exc)))
        self.intent_recognizers.sort(key=lambda x: (x[0].priority, len(x[0].pattern), x[0].weight), reverse=True)

    def _load_entity_resources(self) -> None:
        raw = _read_structured_file(self.custom_entity_patterns_path) or self.nlu_config.get("entity_patterns") or self._default_entity_patterns()
        if not isinstance(raw, Mapping): raw = self._default_entity_patterns()
        self.entity_patterns = dict(raw)
        for label, payload in raw.items():
            meta = _as_mapping(payload)
            values = _as_list(meta.get("patterns", meta.get("pattern", []))) if meta else _as_list(payload)
            for value in values:
                if not _strip_text(value): continue
                item = EntityPattern(label=str(label), pattern=str(value), normalizer=_strip_text(meta.get("normalizer")) or None, confidence=_clamp(meta.get("confidence", .85), 0, 1, .85), priority=int(meta.get("priority", 0) or 0), validation=_strip_text(meta.get("validation")) or None)
                try: self.entity_recognizers.append((item, item.compile()))
                except re.error as exc: self._add_issue(NLUIssue("NLU.ENTITY.PATTERN_INVALID", str(exc)))

    def _load_sentiment_lexicon(self) -> JsonMap:
        raw = _read_structured_file(getattr(self, "sentiment_lexicon_path", None))
        return dict(raw) if isinstance(raw, Mapping) else {"positive": {"good": .7, "great": .9, "excellent": 1.0, "thanks": .5, "helpful": .6, "love": .9}, "negative": {"bad": -.7, "terrible": -1.0, "wrong": -.6, "hate": -.9, "problem": -.5}, "negators": ["not", "never", "no", "n't"], "intensifiers": {"very": 1.25, "really": 1.2, "extremely": 1.5}}

    def _load_modality_markers(self) -> JsonMap:
        raw = _read_structured_file(getattr(self, "modality_markers_path", None))
        return dict(raw) if isinstance(raw, Mapping) else {"imperative": ["please", "show", "tell", "give", "create", "update", "delete", "find", "explain"], "conditional": ["if", "unless"], "epistemic": ["might", "maybe", "probably"], "deontic": ["must", "should", "need"], "dynamic": ["can", "could"]}

    @staticmethod
    def _default_intent_patterns() -> JsonMap:
        return {"greeting": {"patterns": ["hello", "hi", "hey"], "act_type": "expressive", "priority": 1}, "farewell": {"patterns": ["bye", "goodbye"], "act_type": "expressive", "priority": 1}, "gratitude": {"patterns": ["thank you", "thanks"], "act_type": "expressive", "priority": 1}, "help_request": {"patterns": ["help", "can you help", "i need help"], "act_type": "directive", "priority": 1}, "time_request": {"patterns": ["what time", "current time", "tell me the time"], "act_type": "directive", "priority": 2}, "clarification_request": {"patterns": ["what do you mean", "can you clarify", "i don't understand"], "act_type": "directive", "priority": 2}, "question": {"patterns": [r"^(who|what|where|when|why|how)\b", r"\?$"], "act_type": "directive"}}

    @staticmethod
    def _default_entity_patterns() -> JsonMap:
        return {"EMAIL": {"pattern": r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "confidence": .98}, "URL": {"pattern": r"\bhttps?://[^\s]+|\bwww\.[^\s]+", "confidence": .98}, "DATE_TIME": {"pattern": r"\b(?:today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b\d{4}-\d{2}-\d{2}\b", "normalizer": "temporal"}, "TIME": {"pattern": r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b", "normalizer": "time"}, "NUMBER": {"pattern": r"\b[-+]?\d+(?:\.\d+)?\b", "normalizer": "number"}}

    def parse(self, text: str, *, nlp_tokens: Optional[Sequence[Any]] = None, dependencies: Optional[Sequence[Any]] = None, grammar_result: Optional[Any] = None, nlp_result: Optional[Any] = None, context: Optional[Any] = None) -> LinguisticFrame:
        return self.analyze(text, nlp_tokens=nlp_tokens, dependencies=dependencies, grammar_result=grammar_result, nlp_result=nlp_result, context=context).frame

    def analyze(self, text: str, *, nlp_tokens: Optional[Sequence[Any]] = None, dependencies: Optional[Sequence[Any]] = None, grammar_result: Optional[Any] = None, nlp_result: Optional[Any] = None, context: Optional[Any] = None) -> NLUAnalysisResult:
        self._parse_calls += 1
        original = _text(text); normalized = re.sub(r"\s+", " ", original).strip()
        tokens = self._resolve_tokens(original, nlp_tokens, nlp_result)
        deps = tuple(dependencies or ())
        entities = self._extract_entities(original, tokens)
        intents = self._rank_intents(original, tokens, entities, context)
        sentiment = self._calculate_sentiment(tokens)
        modality = self._detect_modality(original, tokens)
        coverage = self._lexical_coverage(tokens)
        frame = self._build_frame(original, intents, entities, sentiment, modality, coverage)
        issues: List[NLUIssue] = []
        low = frame.confidence < self.low_confidence_threshold
        ambiguous = len(intents) > 1 and intents[0].confidence - intents[1].confidence < self.intent_margin_threshold
        if low: issues.append(NLUIssue("NLU.INTENT.LOW_CONFIDENCE", "Best intent confidence is below threshold.", details={"intent": frame.intent, "confidence": frame.confidence}))
        if ambiguous: issues.append(NLUIssue("NLU.INTENT.AMBIGUOUS", "Top intent candidates are close in confidence.", details={"top": intents[0].to_dict(), "second": intents[1].to_dict()}))

        semantic_meta: Dict[str, Any] = {"required": bool(low or ambiguous), "applied": False}
        if low or ambiguous:
            frame, semantic_meta = self._apply_lantra_semantic_evidence(original, frame, intents)

        result = NLUAnalysisResult(text=original, normalized_text=normalized, frame=frame, tokens=tuple(tokens), intents=tuple(intents), entities=tuple(entities), lexical_coverage=coverage, issues=tuple(issues), dependencies=deps, grammar_result=grammar_result, metadata={"version": self.VERSION, "lantra_semantic_evidence": semantic_meta})
        self.history.append({"timestamp": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(), "intent": frame.intent, "confidence": frame.confidence, "semantic_evidence": semantic_meta})
        for issue in issues: self._add_issue(issue)
        return result

    def _apply_lantra_semantic_evidence(self, text: str, frame: LinguisticFrame, intents: Sequence[IntentCandidate]) -> Tuple[LinguisticFrame, Dict[str, Any]]:
        runtime = self.semantic_runtime
        evidence: Dict[str, Any] = {"required": True, "applied": False, "provider": "LANTRA"}
        if runtime is None or not bool(getattr(runtime, "ready", False)):
            evidence["reason"] = "lantra_unavailable"
            return frame, evidence
        labels: List[str] = []
        for candidate in intents:
            if candidate.intent not in labels: labels.append(candidate.intent)
        for label in self.intent_patterns:
            if label not in labels: labels.append(label)
            if len(labels) >= self.semantic_max_labels: break
        if len(labels) < 2:
            evidence["reason"] = "insufficient_candidate_labels"
            return frame, evidence
        try:
            result = runtime.classify(text, labels=labels)
            losses = dict(getattr(result, "candidate_losses", {}) or {})
            ranked = sorted(losses.items(), key=lambda item: float(item[1]))
            evidence.update({"labels": labels, "candidate_losses": losses, "selected": getattr(result, "label", None)})
            if len(ranked) < 2:
                evidence["reason"] = "insufficient_loss_evidence"
                return frame, evidence
            best_label, best_loss = ranked[0]; second_loss = ranked[1][1]
            margin = float(second_loss) - float(best_loss)
            evidence["loss_margin"] = margin
            if margin < self.semantic_min_loss_margin:
                evidence["reason"] = "semantic_margin_below_threshold"
                return frame, evidence
            selected_candidate = next((item for item in intents if item.intent == best_label), None)
            act_type = self._speech_act_for(best_label, frame.modality, selected_candidate)
            resolved = LinguisticFrame(intent=best_label, entities=dict(frame.entities or {}), sentiment=frame.sentiment, modality=frame.modality, confidence=max(float(frame.confidence), self.semantic_confidence_floor), act_type=act_type, propositional_content=frame.propositional_content, illocutionary_force=self._illocutionary_force(best_label, frame.modality, act_type), perlocutionary_effect=frame.perlocutionary_effect)
            evidence["applied"] = True
            evidence["reason"] = "semantic_evidence_resolved_ambiguity"
            return resolved, evidence
        except Exception as exc:
            logger.warning("LANTRA semantic evidence failed: %s", exc)
            evidence.update({"reason": "lantra_error", "error_type": type(exc).__name__, "error": str(exc)})
            return frame, evidence

    def _resolve_tokens(self, text: str, nlp_tokens: Optional[Sequence[Any]], nlp_result: Optional[Any]) -> List[NLUInputToken]:
        raw = nlp_tokens
        if raw is None and nlp_result is not None: raw = _extract_attr(nlp_result, "tokens", None)
        if raw is None and self.nlp_engine is not None and hasattr(self.nlp_engine, "process_text"): raw = self.nlp_engine.process_text(text)
        if raw is None:
            return [NLUInputToken(t, self.wordlist.stem(t), "PUNCT" if re.fullmatch(r"\W+", t) else "NOUN", i, s[0], s[1], is_punct=bool(re.fullmatch(r"\W+", t))) for i, (t, s) in enumerate(_word_tokenize(text))]
        output = []
        cursor = 0
        for i, token in enumerate(raw):
            t = _text(_extract_attr(token, "text", token)); start = _extract_attr(token, "start_char", None); end = _extract_attr(token, "end_char", None)
            if start is None: start = text.find(t, cursor); start = cursor if start < 0 else start
            if end is None: end = start + len(t)
            cursor = end
            output.append(NLUInputToken(t, _text(_extract_attr(token, "lemma", t)).casefold(), _text(_extract_attr(token, "pos", _extract_attr(token, "upos", "X"))).upper(), int(_extract_attr(token, "index", i) or i), int(start), int(end), bool(_extract_attr(token, "is_stop", False)), bool(_extract_attr(token, "is_punct", False))))
        return output

    def _rank_intents(self, text: str, tokens: Sequence[NLUInputToken], entities: Sequence[EntityMention], context: Optional[Any]) -> List[IntentCandidate]:
        candidates: List[IntentCandidate] = []
        for pattern, compiled in self.intent_recognizers:
            for match in compiled.finditer(text):
                words = max(1, len(re.findall(r"\b\w+\b", match.group(0)))); total = max(1, len(re.findall(r"\b\w+\b", text)))
                score = _clamp((.45 + min(1.0, words / total + .35) * .35 + min(.12, max(0, pattern.priority) * .02)) * pattern.weight)
                candidates.append(IntentCandidate(pattern.intent, score, pattern.source, matched_text=match.group(0), priority=pattern.priority, evidence=(match.group(0),), act_type=pattern.act_type))
        if self.enable_context_intents and context is not None and self._looks_like_followup(text.casefold()):
            getter = getattr(context, "get_environment_state", None)
            last = getter("last_intent") if callable(getter) else context.get("last_intent") if isinstance(context, Mapping) else None
            if last: candidates.append(IntentCandidate(str(last), .58, IntentMatchSource.CONTEXT, evidence=("followup",)))
        if not candidates:
            inferred = "question" if text.strip().endswith("?") else self.fallback_intent
            candidates.append(IntentCandidate(inferred, .48 if inferred == "question" else .28, IntentMatchSource.FALLBACK, evidence=("fallback",)))
        grouped: Dict[str, List[IntentCandidate]] = defaultdict(list)
        for c in candidates: grouped[c.intent].append(c)
        merged = []
        for intent, items in grouped.items():
            items.sort(key=lambda x: x.confidence, reverse=True); best = items[0]
            merged.append(IntentCandidate(intent, _clamp(best.confidence + min(.18, sum(x.confidence for x in items[1:]) * .15)), best.source, best.matched_text, best.pattern, max(x.priority for x in items), tuple(_dedupe(e for x in items for e in x.evidence)), act_type=best.act_type))
        return sorted(merged, key=lambda x: (x.confidence, x.priority), reverse=True)

    @staticmethod
    def _looks_like_followup(text: str) -> bool:
        return bool(re.match(r"^(and|also|what about|how about|then|so|but|yes|no|okay|ok|sure)\b", text)) or any(w in text.split() for w in {"it", "that", "those", "these", "they", "them"})

    def _extract_entities(self, text: str, tokens: Sequence[NLUInputToken]) -> List[EntityMention]:
        output: List[EntityMention] = []
        for pattern, compiled in self.entity_recognizers:
            for match in compiled.finditer(text):
                raw = match.group(1) if match.groups() else match.group(0); span = (match.start(1) if match.groups() else match.start(), match.end(1) if match.groups() else match.end())
                value: Any = raw
                if _lower(pattern.normalizer) == "number":
                    try: value = float(raw) if "." in raw else int(raw)
                    except ValueError: pass
                output.append(EntityMention(pattern.label, raw, value, (int(span[0]), int(span[1])), pattern.confidence, pattern.source, _text(value)))
        if self.enable_nlp_entities:
            for token in tokens:
                if token.pos == "PROPN" and token.span:
                    output.append(EntityMention("MENTION", token.text, token.text, token.span, .58, EntitySource.NLP, token.text.casefold()))
        best: Dict[Tuple[str, Span, str], EntityMention] = {}
        for e in output:
            key = (e.label, e.span, _text(e.value)); best[key] = e if key not in best or e.confidence > best[key].confidence else best[key]
        return sorted(best.values(), key=lambda e: (e.span[0], -e.confidence))

    def _calculate_sentiment(self, tokens: Sequence[NLUInputToken]) -> float:
        positive = _as_mapping(self.sentiment_lexicon.get("positive")); negative = _as_mapping(self.sentiment_lexicon.get("negative")); score = 0.0; weight = 0.0
        for token in tokens:
            raw = float(positive.get(token.lower, 0.0) or 0.0) + float(negative.get(token.lower, 0.0) or 0.0)
            score += raw; weight += abs(raw)
        return _clamp(score / weight, -1, 1, 0) if weight else 0.0

    def _detect_modality(self, text: str, tokens: Sequence[NLUInputToken]) -> str:
        if text.strip().endswith("?"): return "interrogative"
        first = tokens[0].lower if tokens else ""
        if first in {_lower(x) for x in _as_list(self.modality_markers.get("imperative"))}: return "imperative"
        return "declarative"

    def _lexical_coverage(self, tokens: Sequence[NLUInputToken]) -> float:
        lexical = [t for t in tokens if not t.is_punct]
        if not lexical: return 0.0
        return sum(1 for t in lexical if t.lower in self.wordlist or t.lemma in self.wordlist) / len(lexical)

    def _build_frame(self, text: str, intents: Sequence[IntentCandidate], entities: Sequence[EntityMention], sentiment: float, modality: str, coverage: float) -> LinguisticFrame:
        best = intents[0]
        confidence = _clamp(best.confidence + min(.12, .03 * len(entities)) + min(.12, coverage * .12))
        ambiguous = len(intents) > 1 and best.confidence - intents[1].confidence < self.intent_margin_threshold
        if best.confidence < self.low_confidence_threshold or ambiguous: confidence = min(confidence, self.low_confidence_threshold - 1e-6)
        act = self._speech_act_for(best.intent, modality, best)
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for e in entities: grouped[e.label].append(e.value); grouped[e.label.lower()].append(e.value)
        payload = {k: (_dedupe(v)[0] if len(_dedupe(v)) == 1 else _dedupe(v)) for k, v in grouped.items()}
        return LinguisticFrame(intent=best.intent, entities=payload, sentiment=sentiment, modality=modality, confidence=confidence, act_type=act, propositional_content=text, illocutionary_force=self._illocutionary_force(best.intent, modality, act))

    def _speech_act_for(self, intent: str, modality: str, best: Optional[IntentCandidate]) -> SpeechActType:
        configured = _lower(best.act_type if best else None)
        mapping = {"assertive": SpeechActType.ASSERTIVE, "directive": SpeechActType.DIRECTIVE, "commissive": SpeechActType.COMMISSIVE, "expressive": SpeechActType.EXPRESSIVE, "declaration": SpeechActType.DECLARATION}
        if configured in mapping: return mapping[configured]
        if modality in {"interrogative", "imperative"} or any(m in intent.casefold() for m in ("request", "question", "help", "clarification")): return SpeechActType.DIRECTIVE
        if any(m in intent.casefold() for m in ("gratitude", "greeting", "farewell")): return SpeechActType.EXPRESSIVE
        return SpeechActType.ASSERTIVE

    @staticmethod
    def _illocutionary_force(intent: str, modality: str, act_type: SpeechActType) -> str:
        if act_type == SpeechActType.DIRECTIVE: return "request_information" if modality == "interrogative" else "request_action"
        if act_type == SpeechActType.EXPRESSIVE: return "express_attitude"
        return "inform"

    def stats(self) -> NLUStats:
        return NLUStats(self._parse_calls, len(self.intent_recognizers), len(self.entity_recognizers), len(self.wordlist), len(self.diagnostics), len(self.history), self.nlp_engine is not None, self.semantic_runtime is not None)

    def diagnostics_result(self) -> List[JsonMap]:
        return [issue.to_dict() for issue in self.diagnostics]

    def to_dict(self) -> JsonMap:
        return {"component": self.__class__.__name__, "version": self.VERSION, "stats": self.stats().to_dict(), "intent_patterns": self.intent_patterns, "diagnostics": self.diagnostics_result()}


class EnhancedNLU(NLUEngine):
    def analyze_text_fully(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        result = self.analyze(text, **kwargs)
        payload = result.to_dict()
        lexical = [token.text for token in result.tokens if not token.is_punct]
        payload["psycholinguistic"] = {"token_count": len(lexical), "type_count": len({x.casefold() for x in lexical}), "type_token_ratio": len({x.casefold() for x in lexical}) / len(lexical) if lexical else 0.0, "mean_token_length": statistics.mean([len(x) for x in lexical]) if lexical else 0.0}
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


__all__ = ["NLUEngine", "EnhancedNLU", "Wordlist", "NLUAnalysisResult", "IntentCandidate", "EntityMention", "NLUIssue"]


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
