"""Centralized helper functions for reasoning workflows.

This module is intentionally broad so every layer in the reasoning subsystem can
reuse common operations with consistent semantics:
- `reasoning_agent.py` lifecycle/orchestration glue
- symbolic rules and rule execution (`rule_engine.py`)
- reasoning strategy modules (`reasoning_types.py`, `types/`)
- probabilistic/hybrid modules (`probabilistic_models.py`,
  `hybrid_probabilistic_models.py`, `utils/model_compute.py`)
- validation and consistency checks (`validation.py`)
- memory and persistence (`reasoning_memory.py`)
"""

from __future__ import annotations

import json
import math
import random
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

from .reasoning_errors import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Helpers")
printer = PrettyPrinter()


Fact = Tuple[str, str, str]
RuleWeightMap = Dict[str, float]
RuleEntry = Tuple[str, Callable[[Dict[Fact, float]], Dict[Fact, float]], float]


# ---------------------------------------------------------------------------
# Fact normalization and text helpers
# ---------------------------------------------------------------------------
def normalize_token(token: Any) -> str:
    """Normalize an atom token to stable stripped string form."""
    return str(token).strip()


def normalize_fact(fact: Union[str, Sequence[Any]]) -> Fact:
    """Normalize a fact into canonical tuple ``(subject, predicate, object)``.

    Supports:
    - tuple/list length 3
    - ``A -> B`` => ``(A, implies, B)``
    - ``A : B`` => ``(A, is, B)``
    - token fallback ``subject predicate object...``
    - binary fallback ``A B`` => ``(A, related_to, B)``
    """
    if isinstance(fact, (tuple, list)) and len(fact) == 3:
        s, p, o = normalize_token(fact[0]), normalize_token(fact[1]), normalize_token(fact[2])
        if s and p and o:
            return (s, p, o)
        raise FactNormalizationError("Fact tuple elements must be non-empty", context={"fact": fact})

    if not isinstance(fact, str):
        raise FactNormalizationError("Fact must be string or 3-item sequence", context={"fact_type": type(fact).__name__})

    text = fact.strip()
    if not text:
        raise FactNormalizationError("Fact string cannot be empty")

    if "->" in text:
        left, right = [part.strip() for part in text.split("->", 1)]
        if left and right:
            return (left, "implies", right)
        raise FactNormalizationError("Invalid implication fact format", context={"fact": fact})

    if ":" in text:
        left, right = [part.strip() for part in text.split(":", 1)]
        if left and right:
            return (left, "is", right)
        raise FactNormalizationError("Invalid descriptor fact format", context={"fact": fact})

    tokens = text.split()
    if len(tokens) >= 3:
        return (tokens[0], tokens[1], " ".join(tokens[2:]))
    if len(tokens) == 2:
        return (tokens[0], "related_to", tokens[1])

    raise FactNormalizationError("Unable to parse fact", context={"fact": fact})


def fact_to_string(fact: Fact, *, separator: str = " ") -> str:
    """Serialize canonical fact tuple to flat display form."""
    s, p, o = normalize_fact(fact)
    return f"{s}{separator}{p}{separator}{o}"


def canonicalize_fact_case(fact: Fact, *, lower: bool = True) -> Fact:
    """Normalize fact casing for consistent indexing/comparison."""
    s, p, o = normalize_fact(fact)
    if lower:
        return (s.lower(), p.lower(), o.lower())
    return (s, p, o)


def invert_fact_object(fact: Fact, *, prefix: str = "not_") -> Fact:
    """Create inverse-object contradiction fact (s, p, not_o)."""
    s, p, o = normalize_fact(fact)
    return (s, p, f"{prefix}{o}")


# ---------------------------------------------------------------------------
# Confidence and probabilistic math helpers
# ---------------------------------------------------------------------------
def clamp_confidence(value: Any, *, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp numeric value to configured confidence interval bounds."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfidenceBoundsError("Confidence must be numeric", cause=exc, context={"value": value}) from exc
    if lower > upper:
        raise ConfidenceBoundsError("lower bound must be <= upper bound", context={"lower": lower, "upper": upper})
    return min(max(v, lower), upper)


def assert_confidence(value: Any, *, lower: float = 0.0, upper: float = 1.0, label: str = "confidence") -> float:
    """Validate confidence range without clamping (strict mode)."""
    v = float(value)
    if v < lower or v > upper:
        raise ConfidenceBoundsError(f"{label} out of bounds", context={"value": v, "lower": lower, "upper": upper})
    return v


def merge_confidence(existing: float, incoming: float) -> float:
    """Merge confidence via probabilistic OR: ``1 - (1-a)(1-b)``."""
    a = clamp_confidence(existing)
    b = clamp_confidence(incoming)
    return 1.0 - (1.0 - a) * (1.0 - b)


def weighted_confidence(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Compute weighted confidence mean with safe normalization."""
    if not values:
        raise ReasoningValidationError("values cannot be empty")
    vals = [clamp_confidence(v) for v in values]

    if weights is None:
        return sum(vals) / len(vals)

    if len(weights) != len(vals):
        raise ReasoningValidationError("weights length must match values length")
    w = [max(float(x), 0.0) for x in weights]
    total = sum(w)
    if total == 0:
        raise ReasoningValidationError("weights sum cannot be zero")
    return sum(v * wi for v, wi in zip(vals, w)) / total


def logistic_confidence(score: float, *, slope: float = 1.0, midpoint: float = 0.0) -> float:
    """Map arbitrary score to [0,1] confidence with logistic calibration."""
    z = slope * (float(score) - midpoint)
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Contradiction and consistency helpers
# ---------------------------------------------------------------------------
def detect_inverse_contradiction(
    fact: Fact,
    knowledge_base: Mapping[Fact, float],
    *,
    threshold: float,
    negation_prefix: str = "not_",
) -> bool:
    """Check contradiction against inverse-object negation semantics."""
    s, p, o = normalize_fact(fact)
    inverse = (s, p, f"{negation_prefix}{o}")
    return float(knowledge_base.get(inverse, 0.0)) > clamp_confidence(threshold)


def ensure_non_contradictory(
    fact: Fact,
    knowledge_base: Mapping[Fact, float],
    *,
    threshold: float,
    source: str = "knowledge_base",
) -> None:
    """Raise ContradictionError if fact violates contradiction policy."""
    if detect_inverse_contradiction(fact, knowledge_base, threshold=threshold):
        raise ContradictionError(
            "Fact contradicts existing knowledge",
            context={"fact": normalize_fact(fact), "threshold": threshold, "source": source},
        )


def conflict_pairs(knowledge_base: Mapping[Fact, float], *, negation_prefix: str = "not_") -> List[Tuple[Fact, Fact]]:
    """Enumerate contradiction candidate pairs present in KB."""
    pairs: List[Tuple[Fact, Fact]] = []
    for fact in knowledge_base:
        inverse = invert_fact_object(fact, prefix=negation_prefix)
        if inverse in knowledge_base:
            pairs.append((fact, inverse))
    return pairs


def redundancy_groups(knowledge_base: Mapping[Fact, float], *, margin: float = 0.05) -> Dict[Tuple[str, str], List[Fact]]:
    """Group near-equivalent facts by (subject, predicate) and confidence margin."""
    grouped: Dict[Tuple[str, str], List[Fact]] = defaultdict(list)
    for fact in knowledge_base:
        s, p, _ = fact
        grouped[(s, p)].append(fact)

    redundant: Dict[Tuple[str, str], List[Fact]] = {}
    for key, facts in grouped.items():
        if len(facts) < 2:
            continue
        vals = [knowledge_base[f] for f in facts]
        if max(vals) - min(vals) <= abs(float(margin)):
            redundant[key] = facts
    return redundant


# ---------------------------------------------------------------------------
# Rule registration/execution helpers
# ---------------------------------------------------------------------------
def validate_rule_registration(rule: Any, name: Optional[str], weight: Any) -> Tuple[str, float]:
    """Validate/normalize rule registration fields and return safe values."""
    if not callable(rule):
        raise RuleDefinitionError("Rule must be callable", context={"rule_type": type(rule).__name__})

    resolved_name = (name or getattr(rule, "__name__", "")).strip()
    if not resolved_name:
        raise RuleDefinitionError("Rule name cannot be empty")

    safe_weight = clamp_confidence(weight)
    return resolved_name, safe_weight


def rank_rules_by_weight(rules: Iterable[RuleEntry], dynamic_weights: Mapping[str, float], *, descending: bool = True) -> List[RuleEntry]:
    """Sort rules by effective weight (dynamic override, static fallback)."""
    return sorted(rules, key=lambda item: float(dynamic_weights.get(item[0], item[2])), reverse=descending)


def update_rule_weight(current: float, *, success: bool, learning_rate: float = 0.1, decay: float = 0.95) -> float:
    """Apply adaptive weight update rule for symbolic inference loops."""
    cur = clamp_confidence(current)
    lr = clamp_confidence(learning_rate)
    dc = clamp_confidence(decay)
    if success:
        return min(1.0, cur + lr * (1.0 - cur))
    return max(0.01, cur * dc)


def bounded_iterations(max_iterations: Any, *, minimum: int = 1, maximum: int = 10_000) -> int:
    """Normalize iteration budget into safe bounded integer range."""
    try:
        n = int(max_iterations)
    except (TypeError, ValueError) as exc:
        raise RuleDefinitionError("max_iterations must be an integer", cause=exc) from exc
    if minimum > maximum:
        raise RuleDefinitionError("minimum cannot exceed maximum", context={"minimum": minimum, "maximum": maximum})
    return max(minimum, min(n, maximum))


def select_top_rules(rules: Sequence[RuleEntry], dynamic_weights: Mapping[str, float], *, top_k: int) -> List[RuleEntry]:
    """Select top-k highest-weight rules for constrained inference passes."""
    k = bounded_iterations(top_k, minimum=1, maximum=max(1, len(rules)))
    return rank_rules_by_weight(rules, dynamic_weights)[:k]


def sample_rules(rules: Sequence[RuleEntry], dynamic_weights: Mapping[str, float], *, k: int) -> List[RuleEntry]:
    """Weighted rule sampling for exploration-oriented reasoning phases."""
    if not rules:
        return []
    k = max(1, min(int(k), len(rules)))
    scored = [(r, max(float(dynamic_weights.get(r[0], r[2])), 0.0)) for r in rules]
    weights = [w for _, w in scored]
    if sum(weights) == 0:
        return random.sample(list(rules), k=k)
    chosen = random.choices([r for r, _ in scored], weights=weights, k=k)
    # deduplicate while preserving order
    dedup: List[RuleEntry] = []
    seen: Set[str] = set()
    for item in chosen:
        if item[0] not in seen:
            dedup.append(item)
            seen.add(item[0])
    return dedup


# ---------------------------------------------------------------------------
# Knowledge base transformation helpers
# ---------------------------------------------------------------------------
def normalize_knowledge(knowledge: Mapping[Union[str, Sequence[Any]], Any]) -> Dict[Fact, float]:
    """Normalize arbitrary knowledge mapping keys/confidences into canonical KB."""
    normalized: Dict[Fact, float] = {}
    for raw_fact, raw_conf in (knowledge or {}).items():
        fact = normalize_fact(raw_fact)
        conf = clamp_confidence(raw_conf)
        prev = normalized.get(fact, 0.0)
        normalized[fact] = merge_confidence(prev, conf)
    return normalized


def merge_knowledge_bases(*knowledge_bases: Mapping[Fact, float]) -> Dict[Fact, float]:
    """Merge multiple KB mappings with confidence-preserving accumulation."""
    merged: Dict[Fact, float] = {}
    for kb in knowledge_bases:
        for fact, conf in kb.items():
            f = normalize_fact(fact)
            merged[f] = merge_confidence(merged.get(f, 0.0), conf)
    return merged


def knowledge_delta(previous: Mapping[Fact, float], current: Mapping[Fact, float], *, epsilon: float = 1e-9) -> Dict[str, Dict[Fact, float]]:
    """Compute inserted/updated/removed delta between two knowledge states."""
    inserted: Dict[Fact, float] = {}
    updated: Dict[Fact, float] = {}
    removed: Dict[Fact, float] = {}

    for fact, conf in current.items():
        if fact not in previous:
            inserted[fact] = conf
        elif abs(float(conf) - float(previous[fact])) > epsilon:
            updated[fact] = conf

    for fact, conf in previous.items():
        if fact not in current:
            removed[fact] = conf

    return {"inserted": inserted, "updated": updated, "removed": removed}


def filter_knowledge_by_confidence(knowledge: Mapping[Fact, float], *, min_confidence: float) -> Dict[Fact, float]:
    """Filter KB facts whose confidence is below minimum threshold."""
    floor = clamp_confidence(min_confidence)
    return {fact: float(conf) for fact, conf in knowledge.items() if float(conf) >= floor}


# ---------------------------------------------------------------------------
# Serialization/state helpers
# ---------------------------------------------------------------------------
def json_safe_reasoning_state(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert state payload into JSON-safe structure for persistence/logging."""
    safe: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            safe[key] = json_safe_reasoning_state(value)
        elif isinstance(value, (list, tuple)):
            safe[key] = [json_safe_reasoning_state(v) if isinstance(v, dict) else str(v) if isinstance(v, set) else v for v in value]
        elif isinstance(value, set):
            safe[key] = sorted(list(value))
        else:
            safe[key] = value
    return safe


def dump_knowledge_json(path: Union[str, Path], knowledge: Mapping[Fact, float]) -> None:
    """Persist knowledge base to disk with deterministic ordering."""
    out = Path(path)
    records = [
        {"subject": s, "predicate": p, "object": o, "confidence": float(c)}
        for (s, p, o), c in sorted(knowledge.items(), key=lambda item: item[0])
    ]
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        raise KnowledgePersistenceError("Failed to persist knowledge JSON", cause=exc, context={"path": str(out)}) from exc


def load_knowledge_json(path: Union[str, Path]) -> Dict[Fact, float]:
    """Load knowledge records from JSON into canonical KB mapping."""
    src = Path(path)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        raise KnowledgePersistenceError("Failed to read knowledge JSON", cause=exc, context={"path": str(src)}) from exc

    kb: Dict[Fact, float] = {}
    if not isinstance(data, list):
        raise KnowledgePersistenceError("Knowledge JSON must be a list", context={"path": str(src)})
    for item in data:
        if not isinstance(item, dict):
            continue
        fact = normalize_fact((item.get("subject", ""), item.get("predicate", ""), item.get("object", "")))
        kb[fact] = clamp_confidence(item.get("confidence", 0.0))
    return kb


# ---------------------------------------------------------------------------
# General deterministic utility helpers
# ---------------------------------------------------------------------------
def monotonic_timestamp_ms() -> int:
    """Monotonic millisecond timestamp for ordering reasoning events."""
    return int(time.monotonic_ns() // 1_000_000)


def safe_dict_update(target: MutableMapping[str, Any], patch: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deterministic sorted-key dict update for reproducible state writes."""
    for key in sorted(patch.keys()):
        target[key] = patch[key]
    return target


def freeze_kb_signature(knowledge: Mapping[Fact, float], *, precision: int = 6) -> Tuple[Tuple[Fact, float], ...]:
    """Create immutable, hashable KB signature for caching/dedup checks."""
    return tuple(sorted((normalize_fact(f), round(float(c), precision)) for f, c in knowledge.items()))


def elapsed_seconds(start_monotonic: float) -> float:
    """Compute elapsed wall-independent duration using monotonic clock."""
    return max(0.0, float(time.monotonic()) - float(start_monotonic))


# ---------------------------------------------------------------------------
# Reasoning graph/topology helpers
# ---------------------------------------------------------------------------
def build_subject_index(knowledge: Mapping[Fact, float]) -> Dict[str, List[Fact]]:
    """Build subject -> facts index for fast retrieval during rule scans."""
    index: Dict[str, List[Fact]] = defaultdict(list)
    for fact in knowledge:
        s, _, _ = normalize_fact(fact)
        index[s].append(fact)
    return dict(index)


def build_predicate_index(knowledge: Mapping[Fact, float]) -> Dict[str, List[Fact]]:
    """Build predicate -> facts index for operator-targeted reasoning."""
    index: Dict[str, List[Fact]] = defaultdict(list)
    for fact in knowledge:
        _, p, _ = normalize_fact(fact)
        index[p].append(fact)
    return dict(index)


def build_object_index(knowledge: Mapping[Fact, float]) -> Dict[str, List[Fact]]:
    """Build object -> facts index to speed reverse lookups."""
    index: Dict[str, List[Fact]] = defaultdict(list)
    for fact in knowledge:
        _, _, o = normalize_fact(fact)
        index[o].append(fact)
    return dict(index)


def adjacency_from_knowledge(knowledge: Mapping[Fact, float], *, confidence_floor: float = 0.0) -> Dict[str, Set[str]]:
    """Convert KB into directed adjacency map subject -> object."""
    floor = clamp_confidence(confidence_floor)
    graph: Dict[str, Set[str]] = defaultdict(set)
    for (s, _, o), conf in knowledge.items():
        if float(conf) >= floor:
            graph[s].add(o)
    return {k: set(v) for k, v in graph.items()}


def has_path(graph: Mapping[str, Set[str]], start: str, goal: str, *, max_depth: int = 6) -> bool:
    """Bounded DFS path existence check for transitive reasoning support."""
    if start == goal:
        return True
    limit = bounded_iterations(max_depth, minimum=1, maximum=10_000)
    stack: List[Tuple[str, int]] = [(start, 0)]
    visited: Set[str] = set()

    while stack:
        node, depth = stack.pop()
        if node in visited or depth > limit:
            continue
        visited.add(node)
        for nxt in graph.get(node, set()):
            if nxt == goal:
                return True
            stack.append((nxt, depth + 1))
    return False


# ---------------------------------------------------------------------------
# Inference bookkeeping helpers
# ---------------------------------------------------------------------------
def init_inference_stats() -> Dict[str, Any]:
    """Create canonical inference statistics payload."""
    return {
        "iterations": 0,
        "rules_evaluated": 0,
        "rules_succeeded": 0,
        "rules_failed": 0,
        "facts_added": 0,
        "facts_updated": 0,
        "started_at_ms": monotonic_timestamp_ms(),
        "finished_at_ms": None,
        "duration_seconds": None,
    }


def finalize_inference_stats(stats: MutableMapping[str, Any], *, start_monotonic: Optional[float] = None) -> MutableMapping[str, Any]:
    """Finalize timing metrics on inference statistics payload."""
    stats["finished_at_ms"] = monotonic_timestamp_ms()
    if start_monotonic is not None:
        stats["duration_seconds"] = elapsed_seconds(start_monotonic)
    return stats


def bump_stat(stats: MutableMapping[str, Any], key: str, *, amount: int = 1) -> None:
    """Increment integer counters in inference bookkeeping payloads."""
    stats[key] = int(stats.get(key, 0)) + int(amount)


def track_fact_mutation(
    stats: MutableMapping[str, Any],
    previous_conf: float,
    next_conf: float,
    *,
    epsilon: float = 1e-12,
) -> None:
    """Update stats counters based on KB mutation type."""
    if previous_conf <= epsilon and next_conf > epsilon:
        bump_stat(stats, "facts_added")
    elif abs(next_conf - previous_conf) > epsilon:
        bump_stat(stats, "facts_updated")


# ---------------------------------------------------------------------------
# Evidence and query helpers
# ---------------------------------------------------------------------------
def normalize_evidence(evidence: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize evidence dict by stripping keys and removing None entries."""
    if not evidence:
        return {}
    normalized: Dict[str, Any] = {}
    for key, value in evidence.items():
        k = str(key).strip()
        if not k or value is None:
            continue
        normalized[k] = value
    return normalized


def merge_evidence(*evidence_blobs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge evidence dictionaries from multiple sources, rightmost precedence."""
    merged: Dict[str, Any] = {}
    for blob in evidence_blobs:
        merged.update(normalize_evidence(blob))
    return merged


def normalize_query_variables(variables: Sequence[Any]) -> Tuple[str, ...]:
    """Normalize query variable list into immutable cleaned tuple."""
    if not variables:
        raise ReasoningValidationError("variables cannot be empty")
    result = tuple(str(v).strip() for v in variables if str(v).strip())
    if not result:
        raise ReasoningValidationError("variables resolved to empty after normalization")
    return result


def top_k_facts(knowledge: Mapping[Fact, float], *, k: int, descending: bool = True) -> List[Tuple[Fact, float]]:
    """Return top-k facts ordered by confidence for diagnostics/explanations."""
    limit = bounded_iterations(k, minimum=1, maximum=max(1, len(knowledge)))
    ordered = sorted(knowledge.items(), key=lambda item: float(item[1]), reverse=descending)
    return ordered[:limit]


def kb_size_metrics(knowledge: Mapping[Fact, float]) -> Dict[str, int]:
    """Compute compact KB cardinality metrics for telemetry."""
    subjects = {s for s, _, _ in knowledge.keys()}
    predicates = {p for _, p, _ in knowledge.keys()}
    objects = {o for _, _, o in knowledge.keys()}
    return {
        "facts": len(knowledge),
        "subjects": len(subjects),
        "predicates": len(predicates),
        "objects": len(objects),
    }

# ----------------------------------------------------------------------
# Configuration helpers
# ----------------------------------------------------------------------
def get_base_config_value(base_config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Retrieve a value from a base reasoning configuration dictionary.

    Args:
        base_config: The configuration dict (e.g., from get_config_section("base_reasoning")).
        key: Configuration key to look up.
        default: Value returned if key is missing.

    Returns:
        The configuration value, or default.
    """
    return base_config.get(key, default)


def log_step(message: str, level: str = "info") -> None:
    """
    Unified logging for reasoning steps using the global logger.

    Args:
        message: Log message.
        level: Log level ("info", "warning", "error", etc.).
    """
    log_func = getattr(logger, level, logger.info)
    log_func(message)


def print_reasoning(title: str, content: str) -> None:
    """
    Pretty‑print reasoning output using the global printer.

    Args:
        title: Header title.
        content: Main content to print.
    """
    printer.print_header(title)
    printer.print_content(content)
    printer.print_footer()



def identity_rule(kb: Dict[Fact, float]) -> Dict[Fact, float]:
    return {(s, p, o): conf for (s, p, o), conf in kb.items() if p == "is"}


def transitive_rule(kb: Dict[Fact, float]) -> Dict[Fact, float]:
    inferred: Dict[Fact, float] = {}
    for (a, p1, b1), c1 in kb.items():
        if p1 != "is":
            continue
        for (b2, p2, c), c2 in kb.items():
            if p2 == "is" and b1 == b2:
                inferred[(a, "is", c)] = max(inferred.get((a, "is", c), 0.0), min(c1, c2))
    return inferred
