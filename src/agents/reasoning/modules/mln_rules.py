from __future__ import annotations

"""
Production-ready Markov Logic Network (MLN)-style soft rule registry.

This module is intentionally kept compatible with ``ValidationEngine``:
``mln_rules`` remains a list of dictionaries where every dictionary exposes a
callable ``lambda_rule(kb, confidence_threshold) -> bool``. Internally the
module now uses typed rule definitions, indexed knowledge access, structured
violations, config-driven thresholds, and subsystem error classes.

Design goals:
- Keep local imports and existing config handling intact.
- Reuse reasoning helpers for fact normalization and confidence handling.
- Avoid duplicating validation-engine responsibilities while giving it a safer
  and richer MLN rule layer.
- Keep rules deterministic, bounded, explainable, and testable.
"""

import datetime
import math
import time

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("MLN Rules")
printer = PrettyPrinter()

Fact = Tuple[str, str, str]
KnowledgeBase = Mapping[Tuple[Any, Any, Any], Any]
RuleEvaluator = Callable[["KnowledgeIndex", float, "MLNRule"], List["MLNRuleViolation"]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _load_mln_config() -> Dict[str, Any]:
    """Load MLN configuration without changing the existing config pattern."""
    config = load_global_config()
    validation_cfg = get_config_section("validation", config)
    mln_cfg = get_config_section("mln_rules", config)

    defaults = {
        "default_confidence_threshold": validation_cfg.get("mln_rule_confidence_threshold", 0.7),
        "default_weight": config.get("markov_logic_weight", 0.7),
        "enabled": True,
        "case_sensitive": True,
        "strict_confidence": False,
        "max_rule_evaluation_seconds": 0.5,
        "max_reported_violations_per_rule": 8,
        "symmetry_tolerance": 0.8,
        "weak_support_threshold": 0.5,
        "room_temperature_min_celsius": 5.0,
        "room_temperature_max_celsius": 40.0,
        "flightless_bird_exceptions": ["Penguin", "Ostrich", "Emu", "Kiwi", "Cassowary"],
        "city_state_exceptions": ["Monaco", "Singapore", "VaticanCity"],
        "exclusive_states": {
            "state_is": [["Liquid", "Solid"], ["Liquid", "Gas"], ["Solid", "Gas"]],
            "has_property": [["Transparent", "Opaque"], ["Even", "Odd"]],
            "has_number": [["Singular", "Plural"]],
            "is_status": [["Possible", "Impossible"]],
        },
        "rule_weights": {},
        "disabled_rules": [],
    }
    defaults.update(mln_cfg)
    return defaults


MLN_CONFIG: Dict[str, Any] = _load_mln_config()


def _config_float(key: str, fallback: float) -> float:
    try:
        value = MLN_CONFIG.get(key, fallback)
        if value is None:
            return float(fallback)
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return float(fallback)
        return value
    except (TypeError, ValueError):
        return float(fallback)


def _config_bool(key: str, fallback: bool) -> bool:
    value = MLN_CONFIG.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


DEFAULT_CONFIDENCE_THRESHOLD: float = clamp_confidence(
    _config_float("default_confidence_threshold", 0.7)
)
DEFAULT_RULE_WEIGHT: float = clamp_confidence(_config_float("default_weight", 0.7))
CASE_SENSITIVE: bool = _config_bool("case_sensitive", True)
STRICT_CONFIDENCE: bool = _config_bool("strict_confidence", False)
MAX_RULE_EVALUATION_SECONDS: float = max(0.01, _config_float("max_rule_evaluation_seconds", 0.5))
MAX_REPORTED_VIOLATIONS_PER_RULE: int = bounded_iterations(
    MLN_CONFIG.get("max_reported_violations_per_rule", 8), minimum=1, maximum=256
)
SYMMETRY_TOLERANCE: float = clamp_confidence(_config_float("symmetry_tolerance", 0.8))
WEAK_SUPPORT_THRESHOLD: float = clamp_confidence(_config_float("weak_support_threshold", 0.5))
ROOM_TEMP_MIN_C: float = _config_float("room_temperature_min_celsius", 5.0)
ROOM_TEMP_MAX_C: float = _config_float("room_temperature_max_celsius", 40.0)
DISABLED_RULES: Set[str] = {str(rule_id) for rule_id in MLN_CONFIG.get("disabled_rules", []) or []}
RULE_WEIGHTS: Dict[str, float] = {
    str(rule_id): clamp_confidence(weight)
    for rule_id, weight in (MLN_CONFIG.get("rule_weights", {}) or {}).items()
}


# ---------------------------------------------------------------------------
# Structured rule output
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLNRuleViolation:
    """A structured MLN violation emitted by one soft rule."""

    rule_id: str
    description: str
    category: str
    severity: str
    weight: float
    facts: Tuple[Fact, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def compact_message(self) -> Tuple[str, str]:
        """Return the legacy tuple shape expected by older validation output."""
        return (f"MLN_VIOLATION ({self.rule_id})", self.description)

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-safe diagnostic payload."""
        return json_safe_reasoning_state(
            {
                "rule_id": self.rule_id,
                "description": self.description,
                "category": self.category,
                "severity": self.severity,
                "weight": self.weight,
                "confidence": self.confidence,
                "facts": [fact_to_string(fact, separator="|") for fact in self.facts],
                "details": self.details,
            }
        )


@dataclass(frozen=True)
class MLNRule:
    """Typed MLN rule metadata and evaluator."""

    id: str
    description: str
    category: str
    evaluator: RuleEvaluator
    weight: float = DEFAULT_RULE_WEIGHT
    severity: str = "medium"
    tags: Tuple[str, ...] = field(default_factory=tuple)
    example_violation: Dict[Fact, float] = field(default_factory=dict)
    enabled: bool = True

    def effective_weight(self) -> float:
        return RULE_WEIGHTS.get(self.id, clamp_confidence(self.weight))

    def is_enabled(self) -> bool:
        return self.enabled and self.id not in DISABLED_RULES and _config_bool("enabled", True)

    # FIX: widened parameter type to accept KnowledgeIndex directly
    def check(self, kb: Union[KnowledgeBase, "KnowledgeIndex"], min_confidence: Optional[float] = None) -> List[MLNRuleViolation]:
        """Evaluate this rule and return structured violations."""
        threshold = DEFAULT_CONFIDENCE_THRESHOLD if min_confidence is None else clamp_confidence(min_confidence)
        index = kb if isinstance(kb, KnowledgeIndex) else KnowledgeIndex(kb)

        if not self.is_enabled():
            return []

        started = time.time()
        try:
            violations = self.evaluator(index, threshold, self)
        except ReasoningError:
            raise
        except Exception as exc:
            raise RuleExecutionError(
                f"MLN rule {self.id} failed during evaluation",
                cause=exc,
                context={"rule_id": self.id, "description": self.description},
            ) from exc

        elapsed = time.time() - started
        if elapsed > MAX_RULE_EVALUATION_SECONDS:
            raise ReasoningTimeoutError(
                f"MLN rule {self.id} exceeded evaluation budget",
                context={
                    "rule_id": self.id,
                    "elapsed_seconds": elapsed,
                    "limit_seconds": MAX_RULE_EVALUATION_SECONDS,
                },
            )

        return violations[:MAX_REPORTED_VIOLATIONS_PER_RULE]

    def has_violation(self, kb: KnowledgeBase, min_confidence: Optional[float] = None) -> bool:
        """Compatibility helper for ValidationEngine's boolean MLN checks."""
        try:
            return bool(self.check(kb, min_confidence))
        except ReasoningError as exc:
            logger.warning(f"{self.id} evaluation skipped: {exc}")
            return False

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Return the legacy dictionary shape used by ValidationEngine."""
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "weight": self.effective_weight(),
            "severity": self.severity,
            "tags": list(self.tags),
            "lambda_rule": self.has_violation,
            "structured_rule": self,
            "example_violation": dict(self.example_violation),
        }


# ---------------------------------------------------------------------------
# Knowledge indexing and reusable query helpers
# ---------------------------------------------------------------------------
class KnowledgeIndex:
    """Indexed, normalized view over a confidence-weighted knowledge base."""

    def __init__(self, kb: Optional[KnowledgeBase] = None, *, case_sensitive: Optional[bool] = None) -> None:
        self.case_sensitive = CASE_SENSITIVE if case_sensitive is None else bool(case_sensitive)
        self.facts: Dict[Fact, float] = {}
        self.by_subject: Dict[str, List[Tuple[Fact, float]]] = defaultdict(list)
        self.by_predicate: Dict[str, List[Tuple[Fact, float]]] = defaultdict(list)
        self.by_object: Dict[str, List[Tuple[Fact, float]]] = defaultdict(list)
        self.by_sp: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
        self._lookup: Dict[Fact, float] = {}
        self._casefold_lookup: Dict[Tuple[str, str, str], Tuple[Fact, float]] = {}
        self._load(kb or {})

    def _load(self, kb: KnowledgeBase) -> None:
        if not isinstance(kb, Mapping):
            raise KnowledgeBaseError(
                "MLN rules require a mapping of fact tuples to confidence values",
                context={"kb_type": type(kb).__name__},
            )

        for raw_fact, raw_confidence in kb.items():
            try:
                fact = normalize_fact(raw_fact)
                confidence = self._normalize_confidence(raw_confidence, fact=fact)
            except ReasoningError:
                if STRICT_CONFIDENCE:
                    raise
                logger.warning(f"Skipping invalid MLN fact: {raw_fact!r}")
                continue
            except Exception as exc:
                if STRICT_CONFIDENCE:
                    raise FactNormalizationError(
                        "Unable to normalize fact for MLN indexing",
                        cause=exc,
                        context={"fact": repr(raw_fact)},
                    ) from exc
                logger.warning(f"Skipping invalid MLN fact {raw_fact!r}: {exc}")
                continue

            current = self.facts.get(fact, 0.0)
            if confidence > current:
                self.facts[fact] = confidence

        self._rebuild_indexes()

    @staticmethod
    def _normalize_confidence(raw_confidence: Any, *, fact: Fact) -> float:
        try:
            if STRICT_CONFIDENCE:
                # FIX: parameter name corrected from 'name' to 'label'
                return assert_confidence(raw_confidence, label=f"confidence:{fact}")
            return clamp_confidence(raw_confidence)
        except ReasoningError:
            raise
        except Exception as exc:
            raise ConfidenceBoundsError(
                "Fact confidence must be numeric and bounded",
                cause=exc,
                context={"fact": fact, "confidence": raw_confidence},
            ) from exc

    def _rebuild_indexes(self) -> None:
        for fact, confidence in self.facts.items():
            s, p, o = fact
            self.by_subject[s].append((fact, confidence))
            self.by_predicate[p].append((fact, confidence))
            self.by_object[o].append((fact, confidence))
            self.by_sp[(s, p)].append((o, confidence))
            self._lookup[fact] = confidence
            self._casefold_lookup[self._fold_fact(fact)] = (fact, confidence)

    @staticmethod
    def _fold(value: Any) -> str:
        return str(value).casefold()

    @classmethod
    def _fold_fact(cls, fact: Fact) -> Tuple[str, str, str]:
        return (cls._fold(fact[0]), cls._fold(fact[1]), cls._fold(fact[2]))

    def __len__(self) -> int:
        return len(self.facts)

    def __contains__(self, fact: Any) -> bool:
        try:
            return self.exists(*normalize_fact(fact), min_confidence=0.0)
        except ReasoningError:
            return False

    def items(self) -> Iterable[Tuple[Fact, float]]:
        return self.facts.items()

    def get(self, fact: Tuple[Any, Any, Any], default: float = 0.0) -> float:
        try:
            normalized = normalize_fact(fact)
        except ReasoningError:
            return default
        return self.confidence(*normalized, default=default)

    def confidence(self, subject: Any, predicate: Any, obj: Any, *, default: float = 0.0) -> float:
        fact = normalize_fact((subject, predicate, obj))
        if self.case_sensitive:
            return self._lookup.get(fact, default)
        folded = self._fold_fact(fact)
        return self._casefold_lookup.get(folded, (fact, default))[1]

    def exists(self, subject: Any, predicate: Any, obj: Any, min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD) -> bool:
        threshold = clamp_confidence(min_confidence)
        return self.confidence(subject, predicate, obj) >= threshold

    def values(self, subject: Any, predicate: Any, min_confidence: float = 0.0) -> List[str]:
        threshold = clamp_confidence(min_confidence)
        s = normalize_token(subject)
        p = normalize_token(predicate)
        values = [obj for obj, conf in self.by_sp.get((s, p), []) if conf >= threshold]
        if values or self.case_sensitive:
            return values

        folded_s = self._fold(s)
        folded_p = self._fold(p)
        return [
            obj
            for (subj, pred), objs in self.by_sp.items()
            if self._fold(subj) == folded_s and self._fold(pred) == folded_p
            for obj, conf in objs
            if conf >= threshold
        ]

    def first_value(self, subject: Any, predicate: Any, min_confidence: float = 0.0) -> Optional[str]:
        values = self.values(subject, predicate, min_confidence)
        return values[0] if values else None

    def facts_by_predicate(self, predicate: Any, min_confidence: float = 0.0) -> List[Tuple[Fact, float]]:
        threshold = clamp_confidence(min_confidence)
        p = normalize_token(predicate)
        facts = [(fact, conf) for fact, conf in self.by_predicate.get(p, []) if conf >= threshold]
        if facts or self.case_sensitive:
            return facts
        folded_p = self._fold(p)
        return [
            (fact, conf)
            for pred, pred_facts in self.by_predicate.items()
            if self._fold(pred) == folded_p
            for fact, conf in pred_facts
            if conf >= threshold
        ]

    def facts_for_subject(self, subject: Any, min_confidence: float = 0.0) -> List[Tuple[Fact, float]]:
        threshold = clamp_confidence(min_confidence)
        s = normalize_token(subject)
        facts = [(fact, conf) for fact, conf in self.by_subject.get(s, []) if conf >= threshold]
        if facts or self.case_sensitive:
            return facts
        folded_s = self._fold(s)
        return [
            (fact, conf)
            for subj, subj_facts in self.by_subject.items()
            if self._fold(subj) == folded_s
            for fact, conf in subj_facts
            if conf >= threshold
        ]

    def subjects(self, predicate: Any, obj: Any, min_confidence: float = 0.0) -> Set[str]:
        threshold = clamp_confidence(min_confidence)
        p = normalize_token(predicate)
        o = normalize_token(obj)
        return {
            s
            for (s, pred, value), conf in self.facts.items()
            if conf >= threshold and self._match(pred, p) and self._match(value, o)
        }

    def objects_for_predicate(self, predicate: Any, min_confidence: float = 0.0) -> Set[str]:
        return {fact[2] for fact, _ in self.facts_by_predicate(predicate, min_confidence)}

    def _match(self, left: Any, right: Any) -> bool:
        return str(left) == str(right) if self.case_sensitive else self._fold(left) == self._fold(right)


# Backward-compatible top-level helpers ------------------------------------------------
def build_knowledge_index(kb: KnowledgeBase) -> KnowledgeIndex:
    """Build a normalized MLN knowledge index."""
    return KnowledgeIndex(kb)


def fact_exists(kb: KnowledgeBase, s: Any, p: Any, o: Any, min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD) -> bool:
    """Check whether a fact exists in the KB with at least ``min_confidence``."""
    index = kb if isinstance(kb, KnowledgeIndex) else KnowledgeIndex(kb)
    return index.exists(s, p, o, min_confidence)


def get_fact_value(kb: KnowledgeBase, s: Any, p: Any, min_confidence: float = 0.0) -> Optional[str]:
    """Return the first object for ``(subject, predicate, *)`` above threshold."""
    index = kb if isinstance(kb, KnowledgeIndex) else KnowledgeIndex(kb)
    return index.first_value(s, p, min_confidence)


def get_fact_values(kb: KnowledgeBase, s: Any, p: Any, min_confidence: float = 0.0) -> List[str]:
    """Return all objects for ``(subject, predicate, *)`` above threshold."""
    index = kb if isinstance(kb, KnowledgeIndex) else KnowledgeIndex(kb)
    return index.values(subject=s, predicate=p, min_confidence=min_confidence) # type: ignore


# ---------------------------------------------------------------------------
# Internal evaluator helpers
# ---------------------------------------------------------------------------
def _violation(rule: MLNRule, facts: Sequence[Fact], *, confidence: float = 1.0, details: Optional[Dict[str, Any]] = None) -> MLNRuleViolation:
    return MLNRuleViolation(
        rule_id=rule.id,
        description=rule.description,
        category=rule.category,
        severity=rule.severity,
        weight=rule.effective_weight(),
        facts=tuple(facts),
        confidence=clamp_confidence(confidence),
        details=details or {},
    )


def _fact_conf(index: KnowledgeIndex, fact: Fact) -> float:
    return index.confidence(*fact)


def _combined_confidence(index: KnowledgeIndex, facts: Sequence[Fact]) -> float:
    if not facts:
        return 1.0
    return min(clamp_confidence(_fact_conf(index, fact)) for fact in facts)


def _to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(str(value).strip())
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _parse_date(value: Any) -> Optional[datetime.datetime]:
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time.min)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    try:
        if "T" in text:
            return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return datetime.datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        logger.debug(f"Unable to parse MLN date value: {value!r}")
        return None


def _has_any_value(index: KnowledgeIndex, subject: str, predicate: str, min_confidence: float) -> bool:
    return bool(index.values(subject, predicate, min_confidence))


def _values_count(index: KnowledgeIndex, subject: str, predicate: str, min_confidence: float) -> int:
    return len(set(index.values(subject, predicate, min_confidence)))


def _is_hierarchical_location(index: KnowledgeIndex, place_a: str, place_b: str) -> bool:
    return (
        index.exists(place_a, "is_part_of", place_b, WEAK_SUPPORT_THRESHOLD)
        or index.exists(place_b, "is_part_of", place_a, WEAK_SUPPORT_THRESHOLD)
        or index.exists(place_a, "located_in", place_b, WEAK_SUPPORT_THRESHOLD)
        or index.exists(place_b, "located_in", place_a, WEAK_SUPPORT_THRESHOLD)
    )


def _not_form(value: Any) -> str:
    text = normalize_token(value)
    return text if text.startswith("not_") else f"not_{text}"


# ---------------------------------------------------------------------------
# Rule evaluators
# ---------------------------------------------------------------------------
def _eval_mutually_exclusive_pair(
    index: KnowledgeIndex,
    threshold: float,
    rule: MLNRule,
    predicate: str,
    left: str,
    right: str,
) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    left_subjects = index.subjects(predicate, left, threshold)
    right_subjects = index.subjects(predicate, right, threshold)
    for subject in sorted(left_subjects & right_subjects):
        facts = ((subject, predicate, left), (subject, predicate, right))
        violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts)))
    return violations


def _r001_alive_dead(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    return _eval_mutually_exclusive_pair(index, threshold, rule, "is_alive", "True", "False") + [
        _violation(rule, ((subject, "is_alive", "True"), (subject, "is_dead", "True")), confidence=min(index.confidence(subject, "is_alive", "True"), index.confidence(subject, "is_dead", "True")))
        for subject in sorted(index.subjects("is_alive", "True", threshold) & index.subjects("is_dead", "True", threshold))
    ]


def _r002_liquid_solid(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    return _eval_mutually_exclusive_pair(index, threshold, rule, "state_is", "Liquid", "Solid")


def _r003_true_false(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    true_facts = index.facts_by_predicate("is_color", threshold)

    # Generic true/false facts are represented as (subject, predicate, True/False).
    for (subject, predicate, obj), conf in list(index.items()):
        if conf < threshold or obj != "True":
            continue
        opposite = (subject, predicate, "False")
        if index.exists(*opposite, min_confidence=threshold):
            violations.append(_violation(rule, ((subject, predicate, obj), opposite), confidence=min(conf, index.confidence(*opposite))))

    # Legacy compatibility for 4-token facts accidentally collapsed into object text.
    for (fact, conf) in true_facts:
        subject, predicate, obj = fact
        if conf >= threshold and str(obj).endswith(" True"):
            false_obj = str(obj).removesuffix(" True") + " False"
            if index.exists(subject, predicate, false_obj, threshold):
                violations.append(_violation(rule, (fact, (subject, predicate, false_obj))))
    return violations


def _r004_exclusive_locations(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    subjects = {fact[0] for fact, _ in index.facts_by_predicate("located_in", threshold)}
    for subject in sorted(subjects):
        places = sorted(set(index.values(subject, "located_in", threshold)))
        for i, place_a in enumerate(places):
            for place_b in places[i + 1 :]:
                if place_a == place_b or _is_hierarchical_location(index, place_a, place_b):
                    continue
                facts = ((subject, "located_in", place_a), (subject, "located_in", place_b))
                violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts)))
    return violations


def _antisymmetric(predicate: str) -> RuleEvaluator:
    def evaluator(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
        violations: List[MLNRuleViolation] = []
        for (subject, _, obj), conf in index.facts_by_predicate(predicate, threshold):
            if subject == obj:
                continue
            reverse = (obj, predicate, subject)
            if index.exists(*reverse, min_confidence=threshold):
                facts = ((subject, predicate, obj), reverse)
                # Avoid duplicate A/B and B/A reports.
                if str(subject) <= str(obj):
                    violations.append(_violation(rule, facts, confidence=min(conf, index.confidence(*reverse))))
        return violations

    return evaluator


def _r006_marriage_symmetry(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    required_threshold = clamp_confidence(threshold * SYMMETRY_TOLERANCE)
    for (subject, _, partner), conf in index.facts_by_predicate("is_married_to", threshold):
        if not index.exists(partner, "is_married_to", subject, required_threshold):
            violations.append(
                _violation(
                    rule,
                    ((subject, "is_married_to", partner),),
                    confidence=conf,
                    details={"missing_expected_fact": (partner, "is_married_to", subject), "required_threshold": required_threshold},
                )
            )
    return violations


def _r007_multiple_spouses(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    subjects = {fact[0] for fact, _ in index.facts_by_predicate("is_married_to", threshold)}
    for subject in sorted(subjects):
        spouses = sorted(set(index.values(subject, "is_married_to", threshold)))
        if len(spouses) > 1:
            facts = tuple((subject, "is_married_to", spouse) for spouse in spouses)
            violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts), details={"spouses": spouses}))
    return violations


def _r008_taxonomy_negative(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    mammal_animal = ("Mammal", "is_a", "Animal")
    if not index.exists(*mammal_animal, min_confidence=threshold):
        return []
    for subject in sorted(index.subjects("is_a", "Mammal", threshold)):
        negative = (subject, "is_a", "not_Animal")
        if index.exists(*negative, min_confidence=threshold):
            facts = ((subject, "is_a", "Mammal"), mammal_animal, negative)
            violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts)))
    return violations


def _r011_birth_after_death(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    subjects = {fact[0] for fact, _ in index.facts_by_predicate("has_birth_date", threshold)}
    for subject in sorted(subjects):
        birth = index.first_value(subject, "has_birth_date", threshold)
        death = index.first_value(subject, "has_death_date", threshold)
        birth_dt = _parse_date(birth)
        death_dt = _parse_date(death)
        if birth_dt and death_dt and birth_dt > death_dt:
            facts = ((subject, "has_birth_date", str(birth)), (subject, "has_death_date", str(death)))
            violations.append(_violation(rule, facts, details={"birth": str(birth), "death": str(death)}))
    return violations


def _r012_event_end_before_start(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    subjects = {fact[0] for fact, _ in index.facts_by_predicate("starts_at_time", threshold)}
    for event in sorted(subjects):
        start = index.first_value(event, "starts_at_time", threshold)
        end = index.first_value(event, "ends_at_time", threshold)
        start_dt = _parse_date(start)
        end_dt = _parse_date(end)
        if start_dt and end_dt and start_dt > end_dt:
            facts = ((event, "starts_at_time", str(start)), (event, "ends_at_time", str(end)))
            violations.append(_violation(rule, facts, details={"start": str(start), "end": str(end)}))
    return violations


def _r014_human_immortal(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    for subject in sorted(index.subjects("is_a", "Human", threshold)):
        immortal = (subject, "is_immortal", "True")
        if index.exists(*immortal, min_confidence=threshold):
            facts = ((subject, "is_a", "Human"), immortal)
            violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts)))
    return violations


def _r015_bird_cannot_fly_exception(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    exceptions = {str(value) for value in MLN_CONFIG.get("flightless_bird_exceptions", []) or []}
    violations: List[MLNRuleViolation] = []
    for subject in sorted(index.subjects("is_a", "Bird", threshold)):
        if any(index.exists(subject, "is_a", exception, WEAK_SUPPORT_THRESHOLD) for exception in exceptions):
            continue
        cannot_fly = (subject, "can_fly", "False")
        if index.exists(*cannot_fly, min_confidence=threshold):
            facts = ((subject, "is_a", "Bird"), cannot_fly)
            violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts), details={"exceptions": sorted(exceptions)}))
    return violations


def _r016_water_room_temperature_state(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    for subject in sorted(index.subjects("is_substance", "Water", threshold)):
        for temp_value in index.values(subject, "has_temperature_celsius", threshold):
            temp = _to_float(temp_value)
            if temp is None or not (ROOM_TEMP_MIN_C < temp < ROOM_TEMP_MAX_C):
                continue
            for impossible_state in ("Solid", "Gas"):
                state_fact = (subject, "state_is", impossible_state)
                if index.exists(*state_fact, min_confidence=threshold):
                    facts = ((subject, "is_substance", "Water"), (subject, "has_temperature_celsius", str(temp_value)), state_fact)
                    violations.append(
                        _violation(
                            rule,
                            facts,
                            confidence=_combined_confidence(index, facts),
                            details={"temperature_celsius": temp, "expected_state": "Liquid"},
                        )
                    )
    return violations


def _r018_causal_chain_contradiction(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    causal_facts = index.facts_by_predicate("causes", threshold)
    by_cause: Dict[str, Set[str]] = defaultdict(set)
    for (cause, _, effect), _ in causal_facts:
        by_cause[cause].add(effect)
    for cause_a, effects in sorted(by_cause.items()):
        for effect_b in sorted(effects):
            for effect_c in sorted(by_cause.get(effect_b, set())):
                contradiction = (cause_a, "causes", _not_form(effect_c))
                if index.exists(*contradiction, min_confidence=threshold):
                    facts = ((cause_a, "causes", effect_b), (effect_b, "causes", effect_c), contradiction)
                    violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts)))
    return violations


def _r022_country_inside_city(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    exceptions = {str(value) for value in MLN_CONFIG.get("city_state_exceptions", []) or []}
    violations: List[MLNRuleViolation] = []
    cities = sorted(index.subjects("is_a", "City", threshold))
    countries = sorted(index.subjects("is_a", "Country", threshold))
    for country in countries:
        if country in exceptions or index.exists(country, "is_a", "CityState", WEAK_SUPPORT_THRESHOLD):
            continue
        for city in cities:
            fact = (country, "located_in", city)
            if index.exists(*fact, min_confidence=threshold):
                facts = ((country, "is_a", "Country"), (city, "is_a", "City"), fact)
                violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts)))
    return violations


def _r024_company_works_for_employee(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    persons = index.subjects("is_a", "Person", threshold)
    companies = index.subjects("is_a", "Company", threshold)
    for company in sorted(companies):
        for person in sorted(persons):
            fact = (company, "works_for", person)
            if index.exists(*fact, min_confidence=threshold):
                facts = ((person, "is_a", "Person"), (company, "is_a", "Company"), fact)
                violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts)))
    return violations


def _r025_greater_than_transitivity(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    greater: Dict[str, Set[str]] = defaultdict(set)
    for (left, _, right), _ in index.facts_by_predicate("is_greater_than", threshold):
        greater[left].add(right)
    for x, ys in sorted(greater.items()):
        for y in sorted(ys):
            for z in sorted(greater.get(y, set())):
                contradiction = (x, "is_less_than", z)
                if index.exists(*contradiction, min_confidence=threshold):
                    facts = ((x, "is_greater_than", y), (y, "is_greater_than", z), contradiction)
                    violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts)))
    return violations


def _r027_bachelor_constraints(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    for subject in sorted(index.subjects("is_a", "Bachelor", threshold)):
        if _has_any_value(index, subject, "is_married_to", threshold):
            spouse_facts = tuple((subject, "is_married_to", spouse) for spouse in index.values(subject, "is_married_to", threshold))
            violations.append(_violation(rule, ((subject, "is_a", "Bachelor"), *spouse_facts), details={"constraint": "unmarried"}))
        female = (subject, "has_gender", "Female")
        if index.exists(*female, min_confidence=threshold):
            violations.append(_violation(rule, ((subject, "is_a", "Bachelor"), female), details={"constraint": "male"}))
    return violations


def _r028_required_absent_but_achieved(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    for (subject, _, requirement), _ in index.facts_by_predicate("requires", threshold):
        absent = (requirement, "is_present", "False")
        achieved = (subject, "is_achieved", "True")
        if index.exists(*absent, min_confidence=threshold) and index.exists(*achieved, min_confidence=threshold):
            facts = ((subject, "requires", requirement), absent, achieved)
            violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts)))
    return violations


def _r031_generic_not_contradiction(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    for (subject, predicate, obj), conf in index.items():
        if conf < threshold:
            continue
        text = str(obj)
        if text.startswith("not_"):
            positive = (subject, predicate, text[4:])
            negative = (subject, predicate, text)
        else:
            positive = (subject, predicate, text)
            negative = (subject, predicate, f"not_{text}")
        if positive != negative and index.exists(*positive, min_confidence=threshold) and index.exists(*negative, min_confidence=threshold):
            ordered = tuple(sorted((positive, negative)))
            if ordered[0] == positive:
                violations.append(_violation(rule, ordered, confidence=_combined_confidence(index, ordered)))
    return violations


def _r032_self_relation(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    disallowed = {
        "is_parent_of",
        "is_child_of",
        "is_part_of",
        "is_inside",
        "occurs_before",
        "is_greater_than",
        "is_less_than",
        "causes",
        "requires",
    }
    violations: List[MLNRuleViolation] = []
    for predicate in sorted(disallowed):
        for fact, conf in index.facts_by_predicate(predicate, threshold):
            subject, _, obj = fact
            if subject == obj:
                violations.append(_violation(rule, (fact,), confidence=conf, details={"predicate": predicate}))
    return violations


def _r033_greater_less_inverse(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    for (left, _, right), conf in index.facts_by_predicate("is_greater_than", threshold):
        wrong_same_direction = (left, "is_less_than", right)
        if index.exists(*wrong_same_direction, min_confidence=threshold):
            facts = ((left, "is_greater_than", right), wrong_same_direction)
            violations.append(_violation(rule, facts, confidence=min(conf, index.confidence(*wrong_same_direction))))
    return violations


def _r034_cycle_in_transitive_relations(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    predicates = ["is_part_of", "is_inside", "occurs_before", "is_greater_than", "is_less_than"]
    violations: List[MLNRuleViolation] = []
    for predicate in predicates:
        graph: Dict[str, Set[str]] = defaultdict(set)
        edge_fact: Dict[Tuple[str, str], Fact] = {}
        for (left, _, right), _ in index.facts_by_predicate(predicate, threshold):
            graph[left].add(right)
            edge_fact[(left, right)] = (left, predicate, right)

        for start in sorted(graph):
            stack: List[Tuple[str, List[str]]] = [(start, [start])]
            seen: Set[Tuple[str, ...]] = set()
            while stack:
                node, path = stack.pop()
                path_key = tuple(path)
                if path_key in seen or len(path) > 6:
                    continue
                seen.add(path_key)
                for nxt in graph.get(node, set()):
                    if nxt == start and len(path) > 1:
                        cycle_nodes = path + [nxt]
                        facts = tuple(edge_fact[(cycle_nodes[i], cycle_nodes[i + 1])] for i in range(len(cycle_nodes) - 1))
                        violations.append(_violation(rule, facts, confidence=_combined_confidence(index, facts), details={"predicate": predicate, "cycle": cycle_nodes}))
                    elif nxt not in path:
                        stack.append((nxt, path + [nxt]))
    return violations[:MAX_REPORTED_VIOLATIONS_PER_RULE]


def _r035_numeric_probability_bounds(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    probability_predicates = {"has_probability", "probability", "confidence", "likelihood"}
    violations: List[MLNRuleViolation] = []
    for predicate in probability_predicates:
        for fact, conf in index.facts_by_predicate(predicate, 0.0):
            value = _to_float(fact[2])
            if value is None:
                continue
            if value < 0.0 or value > 1.0:
                violations.append(_violation(rule, (fact,), confidence=max(conf, threshold), details={"value": value, "valid_range": [0.0, 1.0]}))
    return violations


def _r036_duplicate_exclusive_config(index: KnowledgeIndex, threshold: float, rule: MLNRule) -> List[MLNRuleViolation]:
    violations: List[MLNRuleViolation] = []
    exclusive_states = MLN_CONFIG.get("exclusive_states", {}) or {}
    if not isinstance(exclusive_states, Mapping):
        return []
    for predicate, pairs in exclusive_states.items():
        if not isinstance(pairs, Sequence):
            continue
        for pair in pairs:
            if not isinstance(pair, Sequence) or len(pair) != 2:
                continue
            left, right = str(pair[0]), str(pair[1])
            violations.extend(_eval_mutually_exclusive_pair(index, threshold, rule, str(predicate), left, right))
    return violations


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------
def _rule(
    rule_id: str,
    description: str,
    category: str,
    evaluator: RuleEvaluator,
    *,
    weight: float = DEFAULT_RULE_WEIGHT,
    severity: str = "medium",
    tags: Sequence[str] = (),
    example_violation: Optional[Dict[Fact, float]] = None,
    enabled: bool = True,
) -> MLNRule:
    if not rule_id or not callable(evaluator):
        raise RuleDefinitionError("Invalid MLN rule registration", context={"rule_id": rule_id})
    return MLNRule(
        id=rule_id,
        description=description,
        category=category,
        evaluator=evaluator,
        weight=clamp_confidence(weight),
        severity=severity,
        tags=tuple(tags),
        example_violation=example_violation or {},
        enabled=enabled,
    )


MLN_RULE_REGISTRY: List[MLNRule] = [
    _rule("R001", "A being cannot be both alive and dead simultaneously.", "basic_contradiction", _r001_alive_dead, severity="critical", tags=("life_state",), example_violation={("Socrates", "is_alive", "True"): 0.9, ("Socrates", "is_dead", "True"): 0.95}),
    _rule("R002", "An object cannot be both a liquid and a solid at the same time and conditions.", "physical_state", _r002_liquid_solid, severity="high", tags=("state",), example_violation={("Water_Sample1", "state_is", "Liquid"): 0.8, ("Water_Sample1", "state_is", "Solid"): 0.85}),
    _rule("R003", "A statement cannot be asserted as both True and False with high confidence.", "truth_value", _r003_true_false, severity="critical", tags=("truth",), example_violation={("Sky", "is_blue", "True"): 0.9, ("Sky", "is_blue", "False"): 0.9}),
    _rule("R004", "An entity cannot be located in two distinct mutually exclusive places simultaneously.", "spatial", _r004_exclusive_locations, severity="high", tags=("location",), example_violation={("EiffelTower", "located_in", "Paris"): 1.0, ("EiffelTower", "located_in", "London"): 0.9}),
    _rule("R005", "If A is a parent of B, B cannot be a parent of A.", "relationship", _antisymmetric("is_parent_of"), severity="critical", tags=("family", "antisymmetric"), example_violation={("John", "is_parent_of", "Mary"): 0.9, ("Mary", "is_parent_of", "John"): 0.8}),
    _rule("R006", "If A is married to B, B should also be married to A.", "relationship_expectation", _r006_marriage_symmetry, severity="medium", tags=("family", "symmetric"), example_violation={("Alice", "is_married_to", "Bob"): 0.95}),
    _rule("R007", "A person cannot be married to more than one person in monogamous legal contexts.", "relationship", _r007_multiple_spouses, severity="high", tags=("family",), example_violation={("Bob", "is_married_to", "Alice"): 0.9, ("Bob", "is_married_to", "Carol"): 0.85}),
    _rule("R008", "If X is a mammal and mammals are animals, X should not be stated as not_Animal.", "taxonomy", _r008_taxonomy_negative, severity="high", tags=("ontology",), example_violation={("Dog", "is_a", "Mammal"): 1.0, ("Mammal", "is_a", "Animal"): 1.0, ("Dog", "is_a", "not_Animal"): 0.9}),
    _rule("R009", "If X is part_of Y, Y cannot be part_of X.", "mereology", _antisymmetric("is_part_of"), severity="critical", tags=("part_whole", "antisymmetric"), example_violation={("Engine", "is_part_of", "Car"): 1.0, ("Car", "is_part_of", "Engine"): 0.8}),
    _rule("R010", "If X is_heavier_than Y, Y cannot be_heavier_than X.", "comparison", _antisymmetric("is_heavier_than"), severity="high", tags=("comparison", "antisymmetric"), example_violation={("Elephant", "is_heavier_than", "Mouse"): 1.0, ("Mouse", "is_heavier_than", "Elephant"): 0.7}),
    _rule("R011", "A person's birth date cannot be after their death date.", "temporal", _r011_birth_after_death, severity="critical", tags=("time",), example_violation={("OldMan", "has_birth_date", "1900-01-01"): 1.0, ("OldMan", "has_death_date", "1890-12-31"): 0.9}),
    _rule("R012", "An event cannot end before it starts.", "temporal", _r012_event_end_before_start, severity="critical", tags=("time",), example_violation={("MeetingX", "starts_at_time", "2023-01-01T10:00:00Z"): 0.9, ("MeetingX", "ends_at_time", "2023-01-01T09:00:00Z"): 0.9}),
    _rule("R013", "If EventA occurs_before EventB, EventB cannot occur_before EventA.", "temporal", _antisymmetric("occurs_before"), severity="critical", tags=("time", "antisymmetric"), example_violation={("WW1", "occurs_before", "WW2"): 1.0, ("WW2", "occurs_before", "WW1"): 0.7}),
    _rule("R014", "Humans should not be asserted as immortal in default common-sense mode.", "commonsense", _r014_human_immortal, severity="medium", tags=("commonsense",), example_violation={("Bob", "is_a", "Human"): 0.9, ("Bob", "is_immortal", "True"): 0.8}),
    _rule("R015", "Birds typically can fly unless represented as a configured flightless exception.", "commonsense_expectation", _r015_bird_cannot_fly_exception, severity="low", tags=("default_reasoning",), example_violation={("Sparrow", "is_a", "Bird"): 1.0, ("Sparrow", "can_fly", "False"): 0.9}),
    _rule("R016", "Water should not be solid or gas at configured room-temperature range.", "physical_state", _r016_water_room_temperature_state, severity="high", tags=("temperature",), example_violation={("H2O_sample", "is_substance", "Water"): 1.0, ("H2O_sample", "has_temperature_celsius", "25"): 1.0, ("H2O_sample", "state_is", "Solid"): 0.8}),
    _rule("R017", "An object cannot be transparent and opaque simultaneously.", "property", lambda i, t, r: _eval_mutually_exclusive_pair(i, t, r, "has_property", "Transparent", "Opaque"), severity="high", tags=("property",), example_violation={("GlassPane", "has_property", "Transparent"): 0.9, ("GlassPane", "has_property", "Opaque"): 0.8}),
    _rule("R018", "If A causes B and B causes C, A should not directly cause not_C.", "causal", _r018_causal_chain_contradiction, severity="high", tags=("causal",), example_violation={("Rain", "causes", "WetGround"): 0.9, ("WetGround", "causes", "MuddyShoes"): 0.8, ("Rain", "causes", "not_MuddyShoes"): 0.7}),
    _rule("R019", "A physical object cannot be inside a container that is inside it.", "spatial", _antisymmetric("is_inside"), severity="critical", tags=("spatial", "antisymmetric"), example_violation={("BoxA", "is_inside", "BoxB"): 0.9, ("BoxB", "is_inside", "BoxA"): 0.9}),
    _rule("R020", "An entity represented as singular cannot also be represented as plural.", "linguistic", lambda i, t, r: _eval_mutually_exclusive_pair(i, t, r, "has_number", "Singular", "Plural"), severity="medium", tags=("number",), example_violation={("Cat", "has_number", "Singular"): 0.9, ("Cat", "has_number", "Plural"): 0.8}),
    _rule("R021", "Configured mutually exclusive gender labels should not both be asserted.", "attribute_consistency", lambda i, t, r: _eval_mutually_exclusive_pair(i, t, r, "has_gender", "Male", "Female"), severity="medium", tags=("attribute",), example_violation={("Alex", "has_gender", "Male"): 0.9, ("Alex", "has_gender", "Female"): 0.85}),
    _rule("R022", "A country should not be located inside a city unless represented as a city-state exception.", "geography", _r022_country_inside_city, severity="medium", tags=("location",), example_violation={("Paris", "is_a", "City"): 1.0, ("France", "is_a", "Country"): 1.0, ("France", "located_in", "Paris"): 0.7}),
    _rule("R023", "Transparent color and opaque property should not be asserted together.", "property", lambda i, t, r: [_violation(r, ((s, "has_color", "Transparent"), (s, "has_property", "Opaque")), confidence=min(i.confidence(s, "has_color", "Transparent"), i.confidence(s, "has_property", "Opaque"))) for s in sorted(i.subjects("has_color", "Transparent", t) & i.subjects("has_property", "Opaque", t))], severity="medium", tags=("property",), example_violation={("Brick", "has_color", "Transparent"): 0.8, ("Brick", "has_property", "Opaque"): 1.0}),
    _rule("R024", "A company should not work_for an employee/person.", "business", _r024_company_works_for_employee, severity="medium", tags=("organization",), example_violation={("JohnDoe", "is_a", "Person"): 1.0, ("AcmeCorp", "is_a", "Company"): 1.0, ("AcmeCorp", "works_for", "JohnDoe"): 0.7}),
    _rule("R025", "If X > Y and Y > Z, X should not be less_than Z.", "comparison", _r025_greater_than_transitivity, severity="high", tags=("comparison", "transitive"), example_violation={("A", "is_greater_than", "B"): 0.9, ("B", "is_greater_than", "C"): 0.9, ("A", "is_less_than", "C"): 0.8}),
    _rule("R026", "An action cannot be both possible and impossible.", "modal", lambda i, t, r: _eval_mutually_exclusive_pair(i, t, r, "is_status", "Possible", "Impossible"), severity="high", tags=("modal",), example_violation={("FlyToMarsTomorrow", "is_status", "Possible"): 0.7, ("FlyToMarsTomorrow", "is_status", "Impossible"): 0.9}),
    _rule("R027", "A bachelor should not be asserted as married or non-male under the default lexical rule.", "lexical", _r027_bachelor_constraints, severity="medium", tags=("lexical",), example_violation={("John", "is_a", "Bachelor"): 0.9, ("John", "is_married_to", "Jane"): 0.8}),
    _rule("R028", "If X requires Y and Y is absent, X should not be achieved.", "dependency", _r028_required_absent_but_achieved, severity="high", tags=("requirement",), example_violation={("Car", "requires", "Fuel"): 1.0, ("Fuel", "is_present", "False"): 0.9, ("Car", "is_achieved", "True"): 0.8}),
    _rule("R029", "Numbers cannot be both even and odd.", "numeric", lambda i, t, r: _eval_mutually_exclusive_pair(i, t, r, "has_property", "Even", "Odd"), severity="high", tags=("number",), example_violation={("Number7", "has_property", "Even"): 0.8, ("Number7", "has_property", "Odd"): 0.9}),
    _rule("R030", "A vegetarian should not eat meat under the default dietary rule.", "commonsense", lambda i, t, r: [_violation(r, ((s, "is_a", "Vegetarian"), (s, "eats", "Meat")), confidence=min(i.confidence(s, "is_a", "Vegetarian"), i.confidence(s, "eats", "Meat"))) for s in sorted(i.subjects("is_a", "Vegetarian", t) & i.subjects("eats", "Meat", t))], severity="medium", tags=("dietary",), example_violation={("Alice", "is_a", "Vegetarian"): 1.0, ("Alice", "eats", "Meat"): 0.7}),
    _rule("R031", "A fact should not coexist with the same fact's not_ form above threshold.", "generic_contradiction", _r031_generic_not_contradiction, severity="critical", tags=("generic", "negation"), example_violation={("Door", "is", "Open"): 0.9, ("Door", "is", "not_Open"): 0.8}),
    _rule("R032", "Irreflexive relations should not point from an entity to itself.", "relation_shape", _r032_self_relation, severity="high", tags=("irreflexive",), example_violation={("Alice", "is_parent_of", "Alice"): 0.9}),
    _rule("R033", "A greater-than fact should not share the same direction as a less-than fact.", "comparison", _r033_greater_less_inverse, severity="high", tags=("comparison",), example_violation={("A", "is_greater_than", "B"): 0.9, ("A", "is_less_than", "B"): 0.8}),
    _rule("R034", "Configured transitive/ordering relations should not contain short cycles.", "graph_consistency", _r034_cycle_in_transitive_relations, severity="critical", tags=("cycle",), example_violation={("A", "is_part_of", "B"): 0.9, ("B", "is_part_of", "C"): 0.9, ("C", "is_part_of", "A"): 0.9}),
    _rule("R035", "Probability-like fact values must stay within [0, 1].", "numeric_bounds", _r035_numeric_probability_bounds, severity="medium", tags=("probability",), example_violation={("RainTomorrow", "has_probability", "1.2"): 0.9}),
    _rule("R036", "Configured mutually exclusive predicate/object pairs should not coexist.", "configurable_exclusivity", _r036_duplicate_exclusive_config, severity="medium", tags=("configurable",), example_violation={("Task", "is_status", "Possible"): 0.9, ("Task", "is_status", "Impossible"): 0.9}),
]


# Legacy export consumed by ValidationEngine.
mln_rules: List[Dict[str, Any]] = [rule.to_legacy_dict() for rule in MLN_RULE_REGISTRY]


# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------
def get_rule(rule_id: str) -> MLNRule:
    """Return a typed MLN rule by id."""
    for rule in MLN_RULE_REGISTRY:
        if rule.id == rule_id:
            return rule
    raise RuleDefinitionError("Unknown MLN rule id", context={"rule_id": rule_id})


def validate_rule_registry(rules: Sequence[MLNRule] = MLN_RULE_REGISTRY) -> Dict[str, Any]:
    """Validate rule ids, callables, examples, and legacy adapters."""
    seen: Set[str] = set()
    duplicate_ids: List[str] = []
    invalid_examples: List[str] = []

    for rule in rules:
        if rule.id in seen:
            duplicate_ids.append(rule.id)
        seen.add(rule.id)
        if not callable(rule.evaluator):
            raise RuleDefinitionError("MLN rule evaluator is not callable", context={"rule_id": rule.id})
        if rule.example_violation and not rule.has_violation(rule.example_violation, DEFAULT_CONFIDENCE_THRESHOLD):
            invalid_examples.append(rule.id)

    if duplicate_ids:
        raise RuleDefinitionError("Duplicate MLN rule ids detected", context={"duplicates": duplicate_ids})

    return {
        "total_rules": len(rules),
        "enabled_rules": sum(1 for rule in rules if rule.is_enabled()),
        "disabled_rules": sorted(DISABLED_RULES),
        "invalid_examples": invalid_examples,
        "categories": sorted({rule.category for rule in rules}),
    }


def evaluate_mln_rules(
    kb: KnowledgeBase,
    min_confidence: Optional[float] = None,
    *,
    include_payloads: bool = True,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Evaluate all MLN rules against a knowledge base."""
    threshold = DEFAULT_CONFIDENCE_THRESHOLD if min_confidence is None else clamp_confidence(min_confidence)
    index = KnowledgeIndex(kb)
    started = time.time()
    violations: List[MLNRuleViolation] = []
    errors: List[Dict[str, Any]] = []

    for rule in MLN_RULE_REGISTRY:
        try:
            # rule.check now accepts KnowledgeIndex directly
            violations.extend(rule.check(index, threshold))
        except ReasoningError as exc:
            payload = exc.to_payload() if hasattr(exc, "to_payload") else {"message": str(exc)}
            payload["rule_id"] = rule.id
            errors.append(payload)
            if raise_on_error:
                raise

    by_rule: Dict[str, int] = defaultdict(int)
    by_category: Dict[str, int] = defaultdict(int)
    for violation in violations:
        by_rule[violation.rule_id] += 1
        by_category[violation.category] += 1

    payloads = [violation.to_payload() for violation in violations] if include_payloads else []
    return {
        "status": "success" if not errors else "partial",
        "threshold": threshold,
        "total_facts": len(index),
        "total_rules": len(MLN_RULE_REGISTRY),
        "enabled_rules": sum(1 for rule in MLN_RULE_REGISTRY if rule.is_enabled()),
        "violation_count": len(violations),
        "violations": payloads,
        "legacy_violations": [violation.compact_message() for violation in violations],
        "by_rule": dict(sorted(by_rule.items())),
        "by_category": dict(sorted(by_category.items())),
        "errors": errors,
        "execution_time": time.time() - started,
    }


def explain_rule(rule_id: str) -> Dict[str, Any]:
    """Return metadata for a configured MLN rule."""
    rule = get_rule(rule_id)
    return json_safe_reasoning_state(
        {
            "id": rule.id,
            "description": rule.description,
            "category": rule.category,
            "weight": rule.effective_weight(),
            "severity": rule.severity,
            "tags": rule.tags,
            "enabled": rule.is_enabled(),
            "example_violation": rule.example_violation,
        }
    )


def summarize_rules() -> Dict[str, Any]:
    """Return high-level registry metadata for diagnostics."""
    by_category: Dict[str, int] = defaultdict(int)
    for rule in MLN_RULE_REGISTRY:
        by_category[rule.category] += 1
    return {
        "total_rules": len(MLN_RULE_REGISTRY),
        "enabled_rules": sum(1 for rule in MLN_RULE_REGISTRY if rule.is_enabled()),
        "categories": dict(sorted(by_category.items())),
        "default_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "default_rule_weight": DEFAULT_RULE_WEIGHT,
        "case_sensitive": CASE_SENSITIVE,
    }


if __name__ == "__main__":
    print("\n=== Running MLN Rules ===\n")
    printer.status("TEST", "MLN Rules initialized", "info")

    sample_kb: Dict[Fact, float] = {
        ("Socrates", "is_alive", "True"): 0.92,
        ("Socrates", "is_dead", "True"): 0.95,
        ("Water_Sample1", "state_is", "Liquid"): 0.8,
        ("Water_Sample1", "state_is", "Solid"): 0.85,
        ("Sky", "is_blue", "True"): 0.9,
        ("Sky", "is_blue", "False"): 0.9,
        ("EiffelTower", "located_in", "Paris"): 1.0,
        ("EiffelTower", "located_in", "London"): 0.9,
        ("John", "is_parent_of", "Mary"): 0.9,
        ("Mary", "is_parent_of", "John"): 0.8,
        ("Alice", "is_married_to", "Bob"): 0.95,
        ("Bob", "is_married_to", "Alice"): 0.93,
        ("Bob", "is_married_to", "Carol"): 0.85,
        ("Dog", "is_a", "Mammal"): 1.0,
        ("Mammal", "is_a", "Animal"): 1.0,
        ("Dog", "is_a", "not_Animal"): 0.9,
        ("Engine", "is_part_of", "Car"): 1.0,
        ("Car", "is_part_of", "Engine"): 0.8,
        ("OldMan", "has_birth_date", "1900-01-01"): 1.0,
        ("OldMan", "has_death_date", "1890-12-31"): 0.9,
        ("MeetingX", "starts_at_time", "2023-01-01T10:00:00Z"): 0.9,
        ("MeetingX", "ends_at_time", "2023-01-01T09:00:00Z"): 0.9,
        ("H2O_sample", "is_substance", "Water"): 1.0,
        ("H2O_sample", "has_temperature_celsius", "25"): 1.0,
        ("H2O_sample", "state_is", "Solid"): 0.8,
        ("Rain", "causes", "WetGround"): 0.9,
        ("WetGround", "causes", "MuddyShoes"): 0.8,
        ("Rain", "causes", "not_MuddyShoes"): 0.7,
        ("Cat", "has_number", "Singular"): 0.9,
        ("Cat", "has_number", "Plural"): 0.8,
        ("France", "is_a", "Country"): 1.0,
        ("Paris", "is_a", "City"): 1.0,
        ("France", "located_in", "Paris"): 0.8,
        ("A", "is_greater_than", "B"): 0.9,
        ("B", "is_greater_than", "C"): 0.9,
        ("A", "is_less_than", "C"): 0.8,
        ("Task", "is_status", "Possible"): 0.9,
        ("Task", "is_status", "Impossible"): 0.9,
        ("John", "is_a", "Bachelor"): 0.9,
        ("John", "is_married_to", "Jane"): 0.8,
        ("Fuel", "is_present", "False"): 0.9,
        ("Car", "requires", "Fuel"): 1.0,
        ("Car", "is_achieved", "True"): 0.8,
        ("Number7", "has_property", "Even"): 0.8,
        ("Number7", "has_property", "Odd"): 0.9,
        ("Door", "is", "Open"): 0.9,
        ("Door", "is", "not_Open"): 0.8,
        ("RainTomorrow", "has_probability", "1.2"): 0.9,
    }

    registry_report = validate_rule_registry()
    print(f"Registry: {registry_report['total_rules']} rules, {registry_report['enabled_rules']} enabled")
    if registry_report["invalid_examples"]:
        raise RuleDefinitionError(
            "One or more MLN example violations failed self-validation",
            context={"invalid_examples": registry_report["invalid_examples"]},
        )

    report = evaluate_mln_rules(sample_kb, DEFAULT_CONFIDENCE_THRESHOLD)
    print(f"Detected {report['violation_count']} MLN violations across {len(report['by_rule'])} rules")
    for rule_id, count in list(report["by_rule"].items())[:10]:
        print(f" - {rule_id}: {count}")

    assert report["violation_count"] > 0
    assert any(rule["id"] == "R001" and callable(rule["lambda_rule"]) for rule in mln_rules)
    assert fact_exists(sample_kb, "Socrates", "is_dead", "True", 0.7)
    assert get_fact_value(sample_kb, "OldMan", "has_birth_date", 0.7) == "1900-01-01"

    print("\n=== Test ran successfully ===\n")
