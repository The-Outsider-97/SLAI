"""
Language Rules Module

Core Function:
Provides the language subsystem with production-ready rule primitives for
morphology and lightweight dependency analysis. The module is intentionally
rule-based and deterministic: it does not try to replace a statistical parser,
but it gives the NLP, grammar, NLU, and NLG layers a consistent fallback when a
trained parser is unavailable or when linguistic forms need to be generated.

Responsibilities:
- Load lexical metadata from the structured language wordlist.
- Infer regular English verb forms algorithmically instead of hardcoding them.
- Keep genuinely non-derivable verb exceptions in language_config.yaml, not in
  Python code and not in a separate irregular_verbs.json dependency.
- Normalize token dictionaries/objects into a stable internal representation.
- Apply dependency rules with rule IDs, confidence, deduplication, and issues.
- Preserve compatibility with existing callers that use Rules._apply_rules().

Why it matters:
Rules sit between raw token features and downstream grammar/NLU decisions. A
production rules module must be inspectable, configurable, deterministic,
lexicon-aware, and conservative enough to expand without creating duplicate or
contradictory dependency relations.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.language_error import *
from ..utils.language_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Language Rules")
printer = PrettyPrinter()

TokenIndex = int
Span = Tuple[int, int]
RelationKey = Tuple[str, int, int]


class RuleConfidence(str, Enum):
    """Human-readable confidence bands for heuristic rule output."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class LexicalEntry:
    """Structured wordlist entry normalized for rule lookup."""

    word: str
    pos: Tuple[str, ...] = ()
    synonyms: Tuple[str, ...] = ()
    related_terms: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def lower(self) -> str:
        return self.word.lower()

    def has_pos(self, *labels: str) -> bool:
        active = {label.lower() for label in labels if label}
        return bool(active.intersection({item.lower() for item in self.pos}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "pos": list(self.pos),
            "synonyms": list(self.synonyms),
            "related_terms": list(self.related_terms),
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class VerbInflection:
    """Resolved verb inflection set for one lemma."""

    lemma: str
    present_singular: Tuple[str, ...]
    present_plural: Tuple[str, ...]
    past_simple: Tuple[str, ...]
    past_participle: Tuple[str, ...]
    present_participle: Tuple[str, ...]
    source: str = "inferred"
    confidence: float = 0.85
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_forms(self) -> Tuple[str, ...]:
        return tuple(
            dedupe_preserve_order(
                [
                    self.lemma,
                    *self.present_singular,
                    *self.present_plural,
                    *self.past_simple,
                    *self.past_participle,
                    *self.present_participle,
                ]
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lemma": self.lemma,
            "present_singular": list(self.present_singular),
            "present_plural": list(self.present_plural),
            "past_simple": list(self.past_simple),
            "past_participle": list(self.past_participle),
            "present_participle": list(self.present_participle),
            "source": self.source,
            "confidence": self.confidence,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class RuleToken:
    """Internal token representation consumed by dependency rules."""

    text: str
    index: int
    id: int
    upos: str
    lemma: Optional[str] = None
    xpos: Optional[str] = None
    feats: Dict[str, Any] = field(default_factory=dict)
    dep: Optional[str] = None
    head: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def lower(self) -> str:
        return self.text.lower()

    @property
    def span(self) -> Optional[Span]:
        if self.start_char is None or self.end_char is None:
            return None
        return (self.start_char, self.end_char)

    def has_upos(self, *labels: str) -> bool:
        return self.upos in {label.upper() for label in labels}

    def feature(self, key: str, default: Any = None) -> Any:
        return self.feats.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self), drop_empty=True)


@dataclass
class DependencyRelation:
    """Dependency relation emitted by the rule engine."""

    head: str
    head_index: int
    relation: str
    dependent: str
    dependent_index: int
    rule_id: str = "rules.unknown"
    confidence: float = 0.75
    source_span: Optional[Span] = None
    dependent_span: Optional[Span] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> RelationKey:
        return (self.relation, self.head_index, self.dependent_index)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["source_span"] = list(self.source_span) if self.source_span else None
        payload["dependent_span"] = list(self.dependent_span) if self.dependent_span else None
        return prune_none(payload, drop_empty=True)


@dataclass(frozen=True)
class RuleApplicationResult:
    """Structured output for dependency rule application."""

    relations: Tuple[DependencyRelation, ...]
    root_index: Optional[int] = None
    tokens: Tuple[RuleToken, ...] = ()
    issues: Tuple[Any, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(getattr(issue, "is_blocking", False) for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "root_index": self.root_index,
            "relations": [relation.to_dict() for relation in self.relations],
            "tokens": [token.to_dict() for token in self.tokens],
            "issues": [issue.to_dict() if hasattr(issue, "to_dict") else json_safe(issue) for issue in self.issues],
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class LanguageRulesStats:
    """Operational snapshot for the Rules engine."""

    lexicon_size: int
    verb_lemma_count: int
    verb_form_count: int
    diagnostics_count: int
    history_length: int
    structured_wordlist_path: Optional[str]
    enabled_dependency_rules: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _RelationCollector:
    """Deduplicating relation collector that preserves strongest evidence."""

    def __init__(self, *, deduplicate: bool = True) -> None:
        self.deduplicate = deduplicate
        self._relations: Dict[RelationKey, DependencyRelation] = {}
        self._ordered: List[DependencyRelation] = []

    def add(self, relation: DependencyRelation) -> None:
        if not self.deduplicate:
            self._ordered.append(relation)
            return
        current = self._relations.get(relation.key)
        if current is None:
            self._relations[relation.key] = relation
            self._ordered.append(relation)
            return
        if relation.confidence > current.confidence:
            self._relations[relation.key] = relation
            for index, item in enumerate(self._ordered):
                if item.key == relation.key:
                    self._ordered[index] = relation
                    break

    def relations(self) -> List[DependencyRelation]:
        return list(self._ordered)


class Rules:
    """
    Production language rule engine.

    The class keeps the old public surface (`irregular_singular_forms`,
    `irregular_plural_forms`, and `_apply_rules`) but backs those properties
    with config-driven morphology instead of hardcoded Python dictionaries or a
    separate irregular_verbs.json file.
    """

    DEFAULT_VERB_EXCEPTION_FORMS: Dict[str, Dict[str, Any]] = {
        "be": {
            "present_singular": ["is"],
            "present_plural": ["are"],
            "past_simple": ["was", "were"],
            "past_participle": ["been"],
            "present_participle": ["being"],
        },
        "have": {
            "present_singular": ["has"],
            "present_plural": ["have"],
            "past_simple": ["had"],
            "past_participle": ["had"],
            "present_participle": ["having"],
        },
        "do": {
            "present_singular": ["does"],
            "present_plural": ["do"],
            "past_simple": ["did"],
            "past_participle": ["done"],
            "present_participle": ["doing"],
        },
        "go": {
            "present_singular": ["goes"],
            "present_plural": ["go"],
            "past_simple": ["went"],
            "past_participle": ["gone"],
            "present_participle": ["going"],
        },
    }

    def __init__(self) -> None:
        self.config = load_global_config()
        self.rules_config = get_config_section("language_rules") or {}
        self.lexicon: Dict[str, LexicalEntry] = {}
        self.verb_forms: Dict[str, VerbInflection] = {}
        self.form_to_lemmas: Dict[str, Set[str]] = defaultdict(set)
        self.diagnostics = LanguageDiagnostics()
        self.history: Deque[Dict[str, Any]] = deque(
            maxlen=coerce_int(self.rules_config.get("history_limit", 200), default=200, minimum=1)
        )

        self.enable_dependency_rules = coerce_bool(self.rules_config.get("enable_dependency_rules", True), default=True)
        self.enable_morphology = coerce_bool(self.rules_config.get("enable_morphology", True), default=True)
        self.enable_wordlist_morphology = coerce_bool(self.rules_config.get("enable_wordlist_morphology", True), default=True)
        self.enable_regular_inference = coerce_bool(self.rules_config.get("enable_regular_inference", True), default=True)
        self.enable_relation_deduplication = coerce_bool(self.rules_config.get("enable_relation_deduplication", True), default=True)
        self.default_relation_confidence = coerce_float(self.rules_config.get("default_relation_confidence", 0.74), default=0.74, minimum=0.0, maximum=1.0)
        self.low_confidence_threshold = coerce_float(self.rules_config.get("low_confidence_threshold", 0.45), default=0.45, minimum=0.0, maximum=1.0)
        self.max_tokens = coerce_int(self.rules_config.get("max_tokens", 512), default=512, minimum=1)

        self.structured_wordlist_path = self._resolve_wordlist_path()
        self._load_structured_wordlist()
        self._build_morphology_index()
        self._load_dependency_lexicons()
        self._record("init", stats=self.stats().to_dict())
        printer.status("INIT", "Language Rules initialized", "success")

    # ------------------------------------------------------------------
    # Compatibility properties
    # ------------------------------------------------------------------
    @property
    def irregular_singular_forms(self) -> Dict[str, str]:
        """Compatibility map: lemma -> preferred 3rd person singular form."""

        return {
            lemma: forms.present_singular[0]
            for lemma, forms in self.verb_forms.items()
            if forms.present_singular and forms.present_singular[0] != lemma
        }

    @property
    def irregular_plural_forms(self) -> Dict[str, str]:
        """Compatibility map: lemma -> preferred non-3rd-person present form."""

        return {
            lemma: forms.present_plural[0]
            for lemma, forms in self.verb_forms.items()
            if forms.present_plural
        }

    # ------------------------------------------------------------------
    # Config, diagnostics, and lexicon loading
    # ------------------------------------------------------------------
    def _resolve_wordlist_path(self) -> Optional[Path]:
        configured = self.rules_config.get(
            "structured_wordlist_path",
            self.config.get("main_wordlist_path", self.config.get("wordlist_path")),
        )
        if configured in (None, "", "none", "None"):
            return None
        return Path(ensure_text(configured))

    def _make_issue(
        self,
        *,
        code: Any,
        message: str,
        severity: Any = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        severity = severity or Severity.WARNING
        return DependencyIssue(
            code=code,
            message=message,
            severity=severity,
            module="LanguageRules",
            recoverable=True,
            details=dict(details or {}),
        )

    def _add_issue(self, issue: Any) -> None:
        self.diagnostics.add(issue)

    def _record(self, action: str, **payload: Any) -> None:
        if not coerce_bool(self.rules_config.get("record_history", True), default=True):
            return
        self.history.append({"timestamp": utc_timestamp(), "action": action, "payload": json_safe(payload)})

    def _load_structured_wordlist(self) -> None:
        if not coerce_bool(self.rules_config.get("load_structured_wordlist", True), default=True):
            return
        if self.structured_wordlist_path is None:
            self._add_issue(
                self._make_issue(
                    code=LanguageErrorCode.RESOURCE_MISSING,
                    message="No structured wordlist path was configured for Language Rules.",
                )
            )
            return
        if not self.structured_wordlist_path.exists():
            issue = ResourceIssue(
                code=LanguageErrorCode.RESOURCE_MISSING,
                message="Structured wordlist file configured for Language Rules was not found.",
                module="LanguageRules",
                details={"path": str(self.structured_wordlist_path)},
            )
            self._add_issue(issue)
            logger.warning(issue.to_json())
            return

        data = load_json_file(self.structured_wordlist_path)
        words = data.get("words", {}) if isinstance(data, Mapping) else {}
        if not isinstance(words, Mapping):
            issue = ResourceIssue(
                code=LanguageErrorCode.RESOURCE_FORMAT_INVALID,
                message="Structured wordlist must contain a 'words' mapping.",
                module="LanguageRules",
                details={"path": str(self.structured_wordlist_path)},
            )
            self._add_issue(issue)
            logger.warning(issue.to_json())
            return

        entries: Dict[str, LexicalEntry] = {}
        for raw_word, raw_entry in words.items():
            word = normalize_text(raw_word, lowercase=True, collapse_whitespace=True)
            if not word:
                continue
            entry_map = raw_entry if isinstance(raw_entry, Mapping) else {}
            entries[word] = LexicalEntry(
                word=word,
                pos=tuple(ensure_text(pos).lower() for pos in ensure_list(entry_map.get("pos"))),
                synonyms=tuple(ensure_text(item).lower() for item in ensure_list(entry_map.get("synonyms"))),
                related_terms=tuple(ensure_text(item).lower() for item in ensure_list(entry_map.get("related_terms"))),
                metadata={key: value for key, value in entry_map.items() if key not in {"pos", "synonyms", "related_terms"}},
            )
        self.lexicon = entries
        logger.info("Loaded structured wordlist for Language Rules: %s entries", len(self.lexicon))

    def _build_morphology_index(self) -> None:
        self.verb_forms.clear()
        self.form_to_lemmas.clear()
        if not self.enable_morphology:
            return

        exceptions = merge_mappings(
            self.DEFAULT_VERB_EXCEPTION_FORMS,
            ensure_mapping(self.rules_config.get("verb_exception_forms", {}), field_name="verb_exception_forms", allow_none=True),
        )
        for lemma, payload in exceptions.items():
            inflection = self._inflection_from_exception(lemma, payload)
            self._register_inflection(inflection)

        if self.enable_wordlist_morphology and self.lexicon:
            max_lemmas = coerce_int(self.rules_config.get("max_wordlist_verb_lemmas", 50000), default=50000, minimum=1)
            count = 0
            for word, entry in self.lexicon.items():
                if count >= max_lemmas:
                    break
                if self._is_verb_lemma_candidate(word, entry):
                    if word not in self.verb_forms:
                        self._register_inflection(self.infer_verb_inflection(word, source="wordlist"))
                    count += 1

    def _load_dependency_lexicons(self) -> None:
        cfg = self.rules_config
        self.possessive_pronouns = set(ensure_list(cfg.get("possessive_pronouns", ["my", "your", "his", "her", "its", "our", "their"])))
        self.copula_verbs = set(ensure_list(cfg.get("copula_verbs", ["be", "am", "is", "are", "was", "were", "been", "being", "seem", "seems", "seemed", "appear", "appears", "appeared"])))
        self.passive_aux_verbs = set(ensure_list(cfg.get("passive_aux_verbs", ["be", "am", "is", "are", "was", "were", "been", "being"])))
        self.modal_aux_verbs = set(ensure_list(cfg.get("modal_aux_verbs", ["can", "could", "may", "might", "must", "shall", "should", "will", "would"])))
        self.do_aux_verbs = set(ensure_list(cfg.get("do_aux_verbs", ["do", "does", "did"])))
        self.negation_particles = set(ensure_list(cfg.get("negation_particles", ["not", "n't", "never", "no"])))
        self.relative_pronouns = set(ensure_list(cfg.get("relative_pronouns", ["who", "whom", "whose", "which", "that"])))
        self.adv_sconjs = set(ensure_list(cfg.get("adverbial_subordinators", ["when", "while", "before", "after", "since", "until", "because", "if", "unless", "although", "as", "though"])))
        self.ccomp_sconjs = set(ensure_list(cfg.get("clausal_complement_markers", ["that", "if", "whether"])))
        self.common_particles = set(ensure_list(cfg.get("common_particles", ["up", "down", "in", "out", "on", "off", "away", "back", "over", "through"])))
        self.discourse_words = set(ensure_list(cfg.get("discourse_words", ["well", "so", "however", "anyway", "actually", "please", "yes", "no"])))
        self.temporal_nouns = set(ensure_list(cfg.get("temporal_nouns", ["today", "yesterday", "tomorrow", "morning", "afternoon", "evening", "night", "week", "month", "year", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])))
        self.temporal_prepositions = set(ensure_list(cfg.get("temporal_prepositions", ["on", "in", "at", "during", "before", "after", "since", "until"])))
        self.locative_prepositions = set(ensure_list(cfg.get("locative_prepositions", ["in", "on", "at", "near", "under", "over", "beside", "by", "inside", "outside"])))
        self.predeterminers = set(ensure_list(cfg.get("predeterminers", ["all", "both", "half", "such"])))
        self.fixed_expressions = {tuple(ensure_text(part).lower() for part in item) for item in ensure_list(cfg.get("fixed_expressions", [["of", "course"], ["at", "least"], ["in", "fact"], ["such", "as"], ["for", "example"]])) if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))}

    def _is_verb_lemma_candidate(self, word: str, entry: LexicalEntry) -> bool:
        if entry.has_pos("verb", "VERB", "aux", "AUX"):
            return True
        if not coerce_bool(self.rules_config.get("infer_verbs_from_wordlist_when_pos_missing", False), default=False):
            return False
        if len(word) < 2 or not word.isalpha():
            return False
        blocked_suffixes = tuple(ensure_list(self.rules_config.get("blocked_verb_candidate_suffixes", ["ness", "tion", "ment", "ity", "ism"])))
        return not word.endswith(blocked_suffixes)

    # ------------------------------------------------------------------
    # Morphology
    # ------------------------------------------------------------------
    def _inflection_from_exception(self, lemma: str, payload: Any) -> VerbInflection:
        data = payload if isinstance(payload, Mapping) else {}
        lemma_text = normalize_text(lemma, lowercase=True)
        return VerbInflection(
            lemma=lemma_text,
            present_singular=tuple(normalize_text(item, lowercase=True) for item in ensure_list(data.get("present_singular", []))),
            present_plural=tuple(normalize_text(item, lowercase=True) for item in ensure_list(data.get("present_plural", [lemma_text]))),
            past_simple=tuple(normalize_text(item, lowercase=True) for item in ensure_list(data.get("past_simple", []))),
            past_participle=tuple(normalize_text(item, lowercase=True) for item in ensure_list(data.get("past_participle", []))),
            present_participle=tuple(normalize_text(item, lowercase=True) for item in ensure_list(data.get("present_participle", []))),
            source="config_exception",
            confidence=coerce_float(data.get("confidence", 0.99), default=0.99, minimum=0.0, maximum=1.0),
            metadata={key: value for key, value in data.items() if key not in {"present_singular", "present_plural", "past_simple", "past_participle", "present_participle", "confidence"}},
        )

    def _register_inflection(self, inflection: VerbInflection) -> None:
        self.verb_forms[inflection.lemma] = inflection
        for form in inflection.all_forms:
            if form:
                self.form_to_lemmas[form].add(inflection.lemma)

    def infer_verb_inflection(self, lemma: str, *, source: str = "inferred") -> VerbInflection:
        base = normalize_text(lemma, lowercase=True)
        if not base:
            raise ValueError("Cannot infer verb forms for an empty lemma.")
        return VerbInflection(
            lemma=base,
            present_singular=(self.third_person_singular(base),),
            present_plural=(base,),
            past_simple=tuple(self.regular_past_forms(base)),
            past_participle=tuple(self.regular_past_forms(base)),
            present_participle=(self.present_participle(base),),
            source=source,
            confidence=0.82 if source == "wordlist" else 0.75,
        )

    def third_person_singular(self, lemma: str) -> str:
        base = normalize_text(lemma, lowercase=True)
        if base in self.verb_forms and self.verb_forms[base].present_singular:
            return self.verb_forms[base].present_singular[0]
        if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
            return base[:-1] + "ies"
        if base.endswith(("s", "x", "z", "ch", "sh", "o")):
            return base + "es"
        return base + "s"

    def present_plural(self, lemma: str) -> str:
        base = normalize_text(lemma, lowercase=True)
        if base in self.verb_forms and self.verb_forms[base].present_plural:
            return self.verb_forms[base].present_plural[0]
        return base

    def regular_past_forms(self, lemma: str) -> List[str]:
        base = normalize_text(lemma, lowercase=True)
        if base.endswith("e"):
            return [base + "d"]
        if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
            return [base[:-1] + "ied"]
        if self._should_double_final_consonant(base):
            return [base + base[-1] + "ed"]
        return [base + "ed"]

    def present_participle(self, lemma: str) -> str:
        base = normalize_text(lemma, lowercase=True)
        if base in self.verb_forms and self.verb_forms[base].present_participle:
            return self.verb_forms[base].present_participle[0]
        if base.endswith("ie"):
            return base[:-2] + "ying"
        if base.endswith("e") and not base.endswith(("ee", "ye", "oe")):
            return base[:-1] + "ing"
        if self._should_double_final_consonant(base):
            return base + base[-1] + "ing"
        return base + "ing"

    def _should_double_final_consonant(self, word: str) -> bool:
        if len(word) < 3 or not word[-1].isalpha():
            return False
        if word[-1] in "wxy" or word[-1] in "aeiou":
            return False
        return word[-3] not in "aeiou" and word[-2] in "aeiou"

    def inflect_verb(self, lemma: str, *, tense: str = "present", number: str = "singular", person: int = 3,
                     aspect: Optional[str] = None, form: Optional[str] = None) -> List[str]:
        base = normalize_text(lemma, lowercase=True)
        inflection = self.verb_forms.get(base) or self.infer_verb_inflection(base)
        if form in {"gerund", "present_participle"} or aspect == "progressive":
            return list(inflection.present_participle)
        if tense in {"past", "past_simple"}:
            return list(inflection.past_simple)
        if form in {"past_participle", "participle"} or aspect == "perfect":
            return list(inflection.past_participle)
        if tense == "present" and number == "singular" and int(person) == 3:
            return list(inflection.present_singular)
        return list(inflection.present_plural)

    def lemmatize_verb_form(self, form: str) -> List[str]:
        normalized = normalize_text(form, lowercase=True)
        if normalized in self.form_to_lemmas:
            return sorted(self.form_to_lemmas[normalized])
        candidates: List[str] = []
        if normalized.endswith("ies") and len(normalized) > 3:
            candidates.append(normalized[:-3] + "y")
        if normalized.endswith("es") and len(normalized) > 2:
            candidates.append(normalized[:-2])
        if normalized.endswith("s") and len(normalized) > 1:
            candidates.append(normalized[:-1])
        if normalized.endswith("ied") and len(normalized) > 3:
            candidates.append(normalized[:-3] + "y")
        if normalized.endswith("ed") and len(normalized) > 2:
            candidates.extend([normalized[:-2], normalized[:-1]])
        if normalized.endswith("ing") and len(normalized) > 4:
            stem = normalized[:-3]
            candidates.extend([stem, stem + "e"])
        return dedupe_preserve_order([candidate for candidate in candidates if candidate])

    def lookup_word(self, word: str) -> Optional[LexicalEntry]:
        return self.lexicon.get(normalize_text(word, lowercase=True))

    def word_has_pos(self, word: str, *pos_labels: str) -> bool:
        entry = self.lookup_word(word)
        return bool(entry and entry.has_pos(*pos_labels))

    # ------------------------------------------------------------------
    # Token normalization
    # ------------------------------------------------------------------
    def normalize_tokens(self, tokens: Sequence[Any]) -> List[RuleToken]:
        if len(tokens) > self.max_tokens:
            raise DependencyError(
                DependencyIssue(
                    code=LanguageErrorCode.PIPELINE_CONTRACT_MISMATCH,
                    message="Too many tokens were provided to the Language Rules engine.",
                    severity=Severity.ERROR,
                    module="LanguageRules",
                    details={"max_tokens": self.max_tokens, "received": len(tokens)},
                ),
                recoverable=True,
            )
        normalized: List[RuleToken] = []
        for position, token in enumerate(tokens):
            normalized.append(self._normalize_token(token, position))
        return normalized

    def _normalize_token(self, token: Any, position: int) -> RuleToken:
        text = ensure_text(get_attr_or_key(token, "text", get_attr_or_key(token, "token", "")))
        lemma = get_attr_or_key(token, "lemma", None)
        raw_upos = get_attr_or_key(token, "upos", get_attr_or_key(token, "pos", None))
        if raw_upos is None:
            raw_upos = self._infer_upos(text)  # Always returns str
        upos = normalize_pos_tag(ensure_text(raw_upos))  # Returns str, never None
        # For safety, fallback to "X" (should never be needed)
        if upos is None:
            upos = "X"
        token_id = coerce_int(get_attr_or_key(token, "id", get_attr_or_key(token, "index", position + 1)), default=position + 1, minimum=0)
        feats = get_attr_or_key(token, "feats", get_attr_or_key(token, "morphology", {}))
        feats_map = dict(feats or {}) if isinstance(feats, Mapping) else {}
        return RuleToken(
            text=text,
            index=position,
            id=token_id,
            upos=upos,
            lemma=ensure_text(lemma).lower() if lemma is not None else None,
            xpos=get_attr_or_key(token, "xpos", None),
            feats=feats_map,
            dep=get_attr_or_key(token, "dep", None),
            head=get_attr_or_key(token, "head", None),
            start_char=get_attr_or_key(token, "start_char", get_attr_or_key(token, "start_char_abs", None)),
            end_char=get_attr_or_key(token, "end_char", get_attr_or_key(token, "end_char_abs", None)),
            metadata={"original": json_safe(token)},
        )

    def _infer_upos(self, text: str) -> str:
        """Infer a universal POS tag for a word when no annotation is available."""
        value = ensure_text(text)
        lower = value.lower()
        if not value:
            return "X"
        if is_punctuation(value):
            return "PUNCT"
        if is_numeric_token(value):
            return "NUM"
        if lower in self.modal_aux_verbs or lower in self.copula_verbs or lower in self.do_aux_verbs:
            return "AUX"
        if lower in self.negation_particles:
            return "PART"
        if lower in self.adv_sconjs or lower in self.ccomp_sconjs:
            return "SCONJ"
        entry = self.lookup_word(lower)
        if entry:
            if entry.has_pos("verb"):
                return "VERB"
            if entry.has_pos("adjective", "adj"):
                return "ADJ"
            if entry.has_pos("adverb", "adv"):
                return "ADV"
            if entry.has_pos("pronoun", "pron"):
                return "PRON"
            if entry.has_pos("preposition", "adp"):
                return "ADP"
            if entry.has_pos("determiner", "det"):
                return "DET"
        return "NOUN" if value.isalpha() else "X"

    # ------------------------------------------------------------------
    # Dependency application
    # ------------------------------------------------------------------
    def apply(self, tokens: Sequence[Any]) -> RuleApplicationResult:
        rule_tokens = self.normalize_tokens(tokens)
        collector = _RelationCollector(deduplicate=self.enable_relation_deduplication)
        issues: List[Any] = []
        root_index = self._select_root(rule_tokens)

        if root_index is None:
            issue = DependencyIssue(
                code=LanguageErrorCode.DEP_NO_ROOT,
                message="No plausible dependency root could be selected.",
                severity=Severity.WARNING,
                module="LanguageRules",
                recoverable=True,
                details={"token_count": len(rule_tokens)},
            )
            issues.append(issue)
            self._add_issue(issue)
        else:
            root = rule_tokens[root_index]
            collector.add(self._relation("ROOT", 0, "root", root, "dep.root", 0.95))

        if self.enable_dependency_rules and rule_tokens:
            self._apply_nominal_rules(rule_tokens, collector, root_index)
            self._apply_verbal_rules(rule_tokens, collector, root_index)
            self._apply_clausal_rules(rule_tokens, collector, root_index)
            self._apply_coordination_rules(rule_tokens, collector)
            self._apply_phrase_rules(rule_tokens, collector, root_index)
            self._apply_punctuation_rules(rule_tokens, collector, root_index)
            self._apply_discourse_rules(rule_tokens, collector, root_index)

        relations = tuple(collector.relations())
        self._record("apply", token_count=len(rule_tokens), relation_count=len(relations), root_index=root_index)
        return RuleApplicationResult(
            relations=relations,
            root_index=root_index,
            tokens=tuple(rule_tokens),
            issues=tuple(issues),
            metadata={"deduplicated": self.enable_relation_deduplication, "rule_engine": "LanguageRules"},
        )

    def _apply_rules(self, tokens: List[Dict[str, Any]]) -> List[DependencyRelation]:
        """Compatibility method used by existing NLP modules."""

        return list(self.apply(tokens).relations)

    def _select_root(self, tokens: Sequence[RuleToken]) -> Optional[int]:
        if not tokens:
            return None
        for index, token in enumerate(tokens):
            if token.upos == "VERB":
                return index
        for index, token in enumerate(tokens):
            if token.upos == "AUX":
                return index
        for index, token in enumerate(tokens):
            if token.upos in {"NOUN", "PROPN", "PRON", "ADJ", "ADV"}:
                return index
        return 0 if tokens else None

    def _relation(self, head: Union[str, RuleToken], head_index: int, relation: str, dependent: RuleToken,
                  rule_id: str, confidence: float, **metadata: Any) -> DependencyRelation:
        head_text = head.text if isinstance(head, RuleToken) else ensure_text(head)
        actual_head_index = head.id if isinstance(head, RuleToken) else int(head_index)
        return DependencyRelation(
            head=head_text,
            head_index=actual_head_index,
            relation=relation,
            dependent=dependent.text,
            dependent_index=dependent.id,
            rule_id=rule_id,
            confidence=clamp_float(confidence, 0.0, 1.0),
            source_span=head.span if isinstance(head, RuleToken) else None,
            dependent_span=dependent.span,
            metadata=json_safe(metadata),
        )

    def _nearest_previous(self, tokens: Sequence[RuleToken], start: int, upos: Iterable[str]) -> Optional[RuleToken]:
        labels = set(upos)
        for index in range(start - 1, -1, -1):
            if tokens[index].upos in labels:
                return tokens[index]
        return None

    def _nearest_next(self, tokens: Sequence[RuleToken], start: int, upos: Iterable[str]) -> Optional[RuleToken]:
        labels = set(upos)
        for index in range(start + 1, len(tokens)):
            if tokens[index].upos in labels:
                return tokens[index]
        return None

    def _has_relation(self, collector: _RelationCollector, relation: str, dependent_id: int) -> bool:
        return any(item.relation == relation and item.dependent_index == dependent_id for item in collector.relations())

    def _apply_nominal_rules(self, tokens: Sequence[RuleToken], collector: _RelationCollector, root_index: Optional[int]) -> None:
        for index, token in enumerate(tokens):
            nxt = tokens[index + 1] if index + 1 < len(tokens) else None
            prv = tokens[index - 1] if index > 0 else None
            if nxt and token.upos == "DET" and nxt.upos in {"NOUN", "PROPN", "PRON"}:
                collector.add(self._relation(nxt, nxt.id, "det", token, "dep.det", 0.9))
            if nxt and token.upos == "PRON" and token.lower in self.possessive_pronouns and nxt.upos in {"NOUN", "PROPN"}:
                collector.add(self._relation(nxt, nxt.id, "det:poss", token, "dep.det_poss", 0.9))
            if nxt and token.lower in self.predeterminers and token.upos in {"DET", "ADV"} and nxt.upos in {"DET", "NOUN", "PROPN", "PRON"}:
                collector.add(self._relation(nxt, nxt.id, "det", token, "dep.predet", 0.82))
            if nxt and token.upos == "ADJ" and nxt.upos in {"NOUN", "PROPN"}:
                collector.add(self._relation(nxt, nxt.id, "amod", token, "dep.amod.pre", 0.86))
            if prv and token.upos == "ADJ" and prv.upos in {"NOUN", "PROPN"}:
                collector.add(self._relation(prv, prv.id, "amod", token, "dep.amod.post", 0.68))
            if nxt and token.upos == "NUM" and nxt.upos in {"NOUN", "PROPN"}:
                collector.add(self._relation(nxt, nxt.id, "nummod", token, "dep.nummod.pre", 0.86))
            if prv and token.upos == "NUM" and prv.upos in {"NOUN", "PROPN"}:
                collector.add(self._relation(prv, prv.id, "nummod", token, "dep.nummod.post", 0.7))
            if nxt and token.upos in {"NOUN", "PROPN"} and nxt.upos == "NOUN":
                collector.add(self._relation(nxt, nxt.id, "compound", token, "dep.compound", 0.73))
            if nxt and token.upos == "PROPN" and nxt.upos == "PROPN":
                collector.add(self._relation(token, token.id, "flat", nxt, "dep.flat", 0.76))

        for index in range(len(tokens) - 2):
            first, mid, last = tokens[index], tokens[index + 1], tokens[index + 2]
            if first.upos in {"NOUN", "PROPN"} and mid.text == "'s" and mid.upos in {"PART", "PUNCT"} and last.upos in {"NOUN", "PROPN"}:
                collector.add(self._relation(last, last.id, "nmod:poss", first, "dep.possessive", 0.82))
                collector.add(self._relation(first, first.id, "case", mid, "dep.possessive.case", 0.85))
            if first.upos in {"NOUN", "PROPN"} and mid.lower == "of" and mid.upos == "ADP" and last.upos in {"NOUN", "PROPN", "PRON"}:
                collector.add(self._relation(first, first.id, "nmod", last, "dep.nmod.of", 0.82))
                collector.add(self._relation(last, last.id, "case", mid, "dep.nmod.of.case", 0.9))

    def _apply_verbal_rules(self, tokens: Sequence[RuleToken], collector: _RelationCollector, root_index: Optional[int]) -> None:
        if root_index is not None:
            root = tokens[root_index]
            subject = self._nearest_previous(tokens, root_index, {"NOUN", "PROPN", "PRON"})
            if subject:
                relation = "nsubj:pass" if self._is_passive_context(tokens, root_index) else "nsubj"
                collector.add(self._relation(root, root.id, relation, subject, f"dep.{relation}.pre", 0.82))
            elif root_index + 1 < len(tokens) and tokens[root_index + 1].upos in {"NOUN", "PROPN", "PRON"}:
                collector.add(self._relation(root, root.id, "nsubj", tokens[root_index + 1], "dep.nsubj.post", 0.68))

            obj = self._nearest_next(tokens, root_index, {"NOUN", "PROPN", "PRON"})
            if obj and obj.id != getattr(subject, "id", None):
                collector.add(self._relation(root, root.id, "obj", obj, "dep.obj", 0.74))
                nxt_index = obj.index + 1
                if nxt_index < len(tokens) and tokens[nxt_index].upos in {"NOUN", "PROPN", "PRON"}:
                    collector.add(self._relation(root, root.id, "iobj", obj, "dep.iobj", 0.62))
                    collector.add(self._relation(root, root.id, "obj", tokens[nxt_index], "dep.obj.after_iobj", 0.69))

        for index, token in enumerate(tokens[:-1]):
            nxt = tokens[index + 1]
            if token.upos == "AUX" and nxt.upos == "VERB":
                relation = "aux:pass" if token.lower in self.passive_aux_verbs and self._looks_participle(nxt) else "aux"
                confidence = 0.86 if relation == "aux" else 0.78
                collector.add(self._relation(nxt, nxt.id, relation, token, f"dep.{relation}", confidence))
            if token.upos == "AUX" and token.lower in self.copula_verbs and nxt.upos in {"NOUN", "ADJ", "PRON", "PROPN", "ADV", "NUM"}:
                collector.add(self._relation(nxt, nxt.id, "cop", token, "dep.cop", 0.84))
            if token.lower in self.negation_particles and token.upos in {"ADV", "PART", "DET"}:
                head = nxt if nxt.upos in {"VERB", "AUX", "ADJ", "ADV"} else self._nearest_previous(tokens, index, {"VERB", "AUX", "ADJ", "ADV"})
                if head:
                    collector.add(self._relation(head, head.id, "advmod", token, "dep.neg", 0.82, polarity="negative"))
            if token.upos == "ADV" and nxt.upos in {"VERB", "AUX", "ADJ", "ADV"}:
                collector.add(self._relation(nxt, nxt.id, "advmod", token, "dep.advmod", 0.78))
            if token.upos == "VERB" and nxt.upos in {"PART", "ADP"} and nxt.lower in self.common_particles:
                collector.add(self._relation(token, token.id, "compound:prt", nxt, "dep.compound_prt", 0.8))

    def _looks_participle(self, token: RuleToken) -> bool:
        if token.feature("VerbForm") == "Part" or token.xpos in {"VBN"}:
            return True
        return token.lower.endswith(("ed", "en", "wn", "lt", "nt", "pt", "ught"))

    def _is_passive_context(self, tokens: Sequence[RuleToken], verb_index: int) -> bool:
        if verb_index <= 0:
            return False
        previous = tokens[verb_index - 1]
        return previous.upos == "AUX" and previous.lower in self.passive_aux_verbs and self._looks_participle(tokens[verb_index])

    def _apply_clausal_rules(self, tokens: Sequence[RuleToken], collector: _RelationCollector, root_index: Optional[int]) -> None:
        for index in range(len(tokens) - 2):
            first, second, third = tokens[index], tokens[index + 1], tokens[index + 2]
            if first.upos == "VERB" and second.lower == "to" and second.upos == "PART" and third.upos == "VERB":
                collector.add(self._relation(first, first.id, "xcomp", third, "dep.xcomp.to", 0.82))
                collector.add(self._relation(third, third.id, "mark", second, "dep.xcomp.mark", 0.9))
            if first.upos in {"NOUN", "PROPN"} and second.lower == "to" and second.upos == "PART" and third.upos == "VERB":
                collector.add(self._relation(first, first.id, "acl", third, "dep.acl.to", 0.72))
                collector.add(self._relation(third, third.id, "mark", second, "dep.acl.mark", 0.88))
            if first.upos == "SCONJ" and second.upos in {"PRON", "NOUN", "PROPN", "ADV"} and third.upos == "VERB":
                if root_index is not None:
                    collector.add(self._relation(tokens[root_index], tokens[root_index].id, "advcl", third, "dep.advcl.subj", 0.66))
                collector.add(self._relation(third, third.id, "mark", first, "dep.advcl.mark", 0.85))

        if root_index is not None:
            root = tokens[root_index]
            for index, token in enumerate(tokens):
                nxt = tokens[index + 1] if index + 1 < len(tokens) else None
                if token.upos == "SCONJ" and token.lower in self.ccomp_sconjs and nxt and nxt.upos == "VERB":
                    relation = "ccomp" if token.lower in {"that", "if", "whether"} else "advcl"
                    collector.add(self._relation(root, root.id, relation, nxt, f"dep.{relation}.marker", 0.7))
                    collector.add(self._relation(nxt, nxt.id, "mark", token, f"dep.{relation}.mark", 0.86))
                if token.upos == "SCONJ" and token.lower in self.adv_sconjs and nxt and nxt.upos == "VERB":
                    collector.add(self._relation(root, root.id, "advcl", nxt, "dep.advcl", 0.72))
                    collector.add(self._relation(nxt, nxt.id, "mark", token, "dep.advcl.mark", 0.86))

        for index in range(1, len(tokens) - 1):
            token = tokens[index]
            prev = tokens[index - 1]
            nxt = tokens[index + 1]
            if token.upos == "PRON" and token.lower in self.relative_pronouns and prev.upos in {"NOUN", "PROPN"} and nxt.upos == "VERB":
                collector.add(self._relation(prev, prev.id, "acl:relcl", nxt, "dep.acl_relcl", 0.76))
                collector.add(self._relation(nxt, nxt.id, "nsubj", token, "dep.acl_relcl.pron", 0.64))

    def _apply_coordination_rules(self, tokens: Sequence[RuleToken], collector: _RelationCollector) -> None:
        for index in range(1, len(tokens) - 1):
            token = tokens[index]
            prev = tokens[index - 1]
            nxt = tokens[index + 1]
            if token.upos == "CCONJ" and prev.upos == nxt.upos and prev.upos in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}:
                collector.add(self._relation(prev, prev.id, "cc", token, "dep.cc", 0.88))
                collector.add(self._relation(prev, prev.id, "conj", nxt, "dep.conj", 0.82))
            if token.text == "," and prev.upos == nxt.upos and prev.upos in {"NOUN", "PROPN", "ADJ", "ADV"}:
                collector.add(self._relation(prev, prev.id, "list", nxt, "dep.list.comma", 0.62))

    def _apply_phrase_rules(self, tokens: Sequence[RuleToken], collector: _RelationCollector, root_index: Optional[int]) -> None:
        for index in range(len(tokens) - 1):
            first, second = tokens[index], tokens[index + 1]
            if (first.lower, second.lower) in self.fixed_expressions:
                collector.add(self._relation(first, first.id, "fixed", second, "dep.fixed", 0.9))
            if first.upos == "NOUN" and second.upos == "PROPN":
                collector.add(self._relation(first, first.id, "appos", second, "dep.appos.title", 0.58))
            if first.lower in {"there", "it"} and first.upos == "PRON":
                nxt = self._nearest_next(tokens, index, {"VERB", "AUX"})
                if nxt:
                    collector.add(self._relation(nxt, nxt.id, "expl", first, "dep.expl", 0.72))

        for index in range(len(tokens) - 2):
            first, mid, last = tokens[index], tokens[index + 1], tokens[index + 2]
            if first.upos in {"NOUN", "PROPN"} and mid.text == "," and last.upos in {"NOUN", "PROPN"}:
                collector.add(self._relation(first, first.id, "appos", last, "dep.appos.comma", 0.74))
            if first.upos == "VERB" and mid.lower == "by" and mid.upos == "ADP" and last.upos in {"NOUN", "PROPN", "PRON"}:
                if self._is_passive_context(tokens, first.index):
                    collector.add(self._relation(first, first.id, "obl:agent", last, "dep.obl_agent", 0.78))
                    collector.add(self._relation(last, last.id, "case", mid, "dep.obl_agent.case", 0.9))
            if mid.upos == "ADP" and last.upos in {"NOUN", "PROPN", "PRON"}:
                if root_index is not None and mid.lower in self.temporal_prepositions | self.locative_prepositions:
                    relation = "obl:tmod" if last.lower in self.temporal_nouns or mid.lower in self.temporal_prepositions else "obl"
                    collector.add(self._relation(tokens[root_index], tokens[root_index].id, relation, last, "dep.obl.prep", 0.68))
                    collector.add(self._relation(last, last.id, "case", mid, "dep.obl.case", 0.86))

    def _apply_punctuation_rules(self, tokens: Sequence[RuleToken], collector: _RelationCollector, root_index: Optional[int]) -> None:
        root = tokens[root_index] if root_index is not None else None
        for index, token in enumerate(tokens):
            if token.upos != "PUNCT":
                continue
            if token.text in {".", "?", "!", "…"} and root is not None:
                collector.add(self._relation(root, root.id, "punct", token, "dep.punct.final", 0.92))
            elif index > 0:
                collector.add(self._relation(tokens[index - 1], tokens[index - 1].id, "punct", token, "dep.punct.local", 0.78))

    def _apply_discourse_rules(self, tokens: Sequence[RuleToken], collector: _RelationCollector, root_index: Optional[int]) -> None:
        root = tokens[root_index] if root_index is not None else None
        for index, token in enumerate(tokens):
            if root and (token.upos == "INTJ" or token.lower in self.discourse_words):
                collector.add(self._relation(root, root.id, "discourse", token, "dep.discourse", 0.68))
            if root and token.upos in {"NOUN", "PROPN"}:
                if index == 0 and index + 1 < len(tokens) and tokens[index + 1].text == ",":
                    collector.add(self._relation(root, root.id, "vocative", token, "dep.vocative.initial", 0.72))
                if index >= 2 and tokens[index - 1].text == ",":
                    collector.add(self._relation(root, root.id, "vocative", token, "dep.vocative.final", 0.64))
        for index in range(len(tokens) - 1):
            if tokens[index].lower == tokens[index + 1].lower and tokens[index].upos == tokens[index + 1].upos:
                collector.add(self._relation(tokens[index], tokens[index].id, "reparandum", tokens[index + 1], "dep.reparandum", 0.71))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> LanguageRulesStats:
        return LanguageRulesStats(
            lexicon_size=len(self.lexicon),
            verb_lemma_count=len(self.verb_forms),
            verb_form_count=len(self.form_to_lemmas),
            diagnostics_count=len(self.diagnostics.issues),
            history_length=len(self.history),
            structured_wordlist_path=str(self.structured_wordlist_path) if self.structured_wordlist_path else None,
            enabled_dependency_rules=1 if self.enable_dependency_rules else 0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.stats().to_dict(),
            "diagnostics": self.diagnostics.to_list(),
            "sample_verb_forms": {lemma: self.verb_forms[lemma].to_dict() for lemma in list(self.verb_forms)[:10]},
            "history": list(self.history),
        }


if __name__ == "__main__":
    print("\n=== Running Language Rules ===\n")
    printer.status("TEST", "Language Rules initialized", "info")

    rules = Rules()

    samples = ["try", "echo", "play", "stop", "be", "have", "go"]
    inflections = {
        lemma: {
            "present_3sg": rules.inflect_verb(lemma, tense="present", number="singular", person=3),
            "present_plural": rules.inflect_verb(lemma, tense="present", number="plural", person=1),
            "past": rules.inflect_verb(lemma, tense="past"),
            "progressive": rules.inflect_verb(lemma, aspect="progressive"),
        }
        for lemma in samples
    }

    test_tokens = [
        {"id": 1, "text": "The", "upos": "DET"},
        {"id": 2, "text": "quick", "upos": "ADJ"},
        {"id": 3, "text": "fox", "upos": "NOUN"},
        {"id": 4, "text": "will", "upos": "AUX"},
        {"id": 5, "text": "jump", "upos": "VERB"},
        {"id": 6, "text": "over", "upos": "ADP"},
        {"id": 7, "text": "the", "upos": "DET"},
        {"id": 8, "text": "wall", "upos": "NOUN"},
        {"id": 9, "text": ".", "upos": "PUNCT"},
    ]
    result = rules.apply(test_tokens)

    printer.pretty("INFLECTIONS", inflections, "success")
    printer.pretty("RELATIONS", [relation.to_dict() for relation in result.relations], "success")
    printer.pretty("RESULT", result.to_dict(), "success")
    printer.pretty("STATS", rules.stats().to_dict(), "success")
    printer.pretty("DIAGNOSTICS", rules.diagnostics.to_list(), "info")

    print("\n=== Test ran successfully ===\n")
