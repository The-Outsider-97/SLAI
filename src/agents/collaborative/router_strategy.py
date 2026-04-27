from __future__ import annotations

"""
Production-grade routing strategies for SLAI's collaborative runtime.

This module owns agent ranking only. It intentionally does not execute tasks,
open reliability circuits, validate task contracts, evaluate policy rules, or
mutate registry state. Those responsibilities remain in ``task_router.py``,
``reliability.py``, ``task_contracts.py``, ``policy_engine.py``, and
``registry.py``.

Responsibilities
----------------
- Preserve the existing routing contract used by ``TaskRouter``:
  ``rank_agents(agents, stats, task_data)`` returns a list of
  ``(agent_name, agent_meta, score)`` tuples sorted best-first.
- Provide deterministic, inspectable scoring for weighted and least-loaded
  routing strategies.
- Normalize agent metadata, task payload hints, and runtime stats through the
  collaborative helper layer.
- Use collaboration errors at routing-strategy boundaries instead of emitting
  unstructured failures.
- Keep strategy configuration in ``collaborative_config.yaml`` under
  ``task_routing`` / ``task_routing.router_strategy``.

Design principles
-----------------
1. Stable public API: ``RouterScoreWeights``, ``BaseRouterStrategy``,
   ``WeightedRouterStrategy``, ``LeastLoadedRouterStrategy`` and
   ``build_router_strategy`` remain available and compatible.
2. Additive production features: diagnostics, explanations, scoring records,
   filtering, and strategy registry helpers are optional and do not change the
   tuple ranking consumed by the router.
3. Determinism: ranking uses explicit tie-breakers so equal scores are stable.
4. Separation of concerns: strategies rank eligible agents, while router and
   reliability modules decide execution and fallback behavior.
"""

import random
import threading

from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.collaboration_error import *
from .utils.collaborative_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Router Strategy")
printer = PrettyPrinter()

AgentMap = Dict[str, Dict[str, Any]]
StatsMap = Dict[str, Dict[str, Any]]
RankedTuple = Tuple[str, Dict[str, Any], float]
StrategyFactory = Callable[[Optional[Mapping[str, Any]]], "BaseRouterStrategy"]


class RouterStrategyName(str, Enum):
    """Known strategy names and aliases exposed through configuration."""

    WEIGHTED = "weighted"
    LEAST_LOADED = "least_loaded"
    ROUND_ROBIN = "round_robin"
    CAPABILITY_MATCH = "capability_match"
    RANDOM_WEIGHTED = "random_weighted"


@dataclass(frozen=True)
class RouterScoreWeights:
    """Weighted-router score coefficients.

    The original fields, ``success_rate`` and ``load_penalty``, are retained.
    Additional weights are additive and default to conservative values so old
    configs keep producing equivalent rankings unless richer stats are present.
    """

    success_rate: float = 1.0
    load_penalty: float = 0.25
    failure_penalty: float = 0.35
    capability_match: float = 0.50
    priority: float = 0.05
    reliability: float = 0.40
    latency_penalty: float = 0.001
    freshness: float = 0.05
    affinity: float = 0.25
    overload_penalty: float = 1.0

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "RouterScoreWeights":
        source = dict(config or {})
        weights = source.get("weights") if isinstance(source.get("weights"), Mapping) else {}
        merged = merge_mappings(source, weights, deep=True, drop_none=True)
        return cls(
            success_rate=coerce_float(merged.get("weight_success_rate", merged.get("success_rate")), default=1.0),
            load_penalty=coerce_float(merged.get("weight_load_penalty", merged.get("load_penalty")), default=0.25, minimum=0.0),
            failure_penalty=coerce_float(merged.get("weight_failure_penalty", merged.get("failure_penalty")), default=0.35, minimum=0.0),
            capability_match=coerce_float(merged.get("weight_capability_match", merged.get("capability_match")), default=0.50),
            priority=coerce_float(merged.get("weight_priority", merged.get("priority")), default=0.05),
            reliability=coerce_float(merged.get("weight_reliability", merged.get("reliability")), default=0.40),
            latency_penalty=coerce_float(merged.get("weight_latency_penalty", merged.get("latency_penalty")), default=0.001, minimum=0.0),
            freshness=coerce_float(merged.get("weight_freshness", merged.get("freshness")), default=0.05),
            affinity=coerce_float(merged.get("weight_affinity", merged.get("affinity")), default=0.25),
            overload_penalty=coerce_float(merged.get("weight_overload_penalty", merged.get("overload_penalty")), default=1.0, minimum=0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouterStrategyConfig:
    """Common behavior flags shared by concrete strategies."""

    strategy: str = RouterStrategyName.WEIGHTED.value
    stable_tie_breaker: str = "name"
    filter_unavailable: bool = False
    include_diagnostics: bool = True
    new_agent_success_bias: float = DEFAULT_NEW_AGENT_SUCCESS_BIAS
    max_active_tasks_per_agent: Optional[int] = None
    stale_after_seconds: Optional[float] = None
    stale_agent_penalty: float = 0.25
    unavailable_score_penalty: float = 10_000.0
    random_seed: Optional[int] = None
    audit_enabled: bool = False
    audit_key: str = "collaboration:router_strategy_events"
    audit_max_events: int = DEFAULT_MAX_AUDIT_EVENTS

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "RouterStrategyConfig":
        source = dict(config or {})
        nested = source.get("router_strategy") if isinstance(source.get("router_strategy"), Mapping) else {}
        merged = merge_mappings(source, nested, deep=True, drop_none=True)
        max_active = merged.get("max_active_tasks_per_agent")
        stale_after = merged.get("stale_after_seconds")
        return cls(
            strategy=normalize_strategy_name(merged.get("strategy", cls.strategy)),
            stable_tie_breaker=str(merged.get("stable_tie_breaker", cls.stable_tie_breaker)).strip().lower() or cls.stable_tie_breaker,
            filter_unavailable=coerce_bool(merged.get("filter_unavailable"), default=False),
            include_diagnostics=coerce_bool(merged.get("include_diagnostics"), default=True),
            new_agent_success_bias=coerce_float(merged.get("new_agent_success_bias"), default=DEFAULT_NEW_AGENT_SUCCESS_BIAS, minimum=0.0, maximum=1.0),
            max_active_tasks_per_agent=None if max_active is None else coerce_int(max_active, default=0, minimum=0),
            stale_after_seconds=None if stale_after is None else coerce_float(stale_after, default=0.0, minimum=0.0),
            stale_agent_penalty=coerce_float(merged.get("stale_agent_penalty"), default=0.25, minimum=0.0),
            unavailable_score_penalty=coerce_float(merged.get("unavailable_score_penalty"), default=10_000.0, minimum=0.0),
            random_seed=None if merged.get("random_seed") is None else coerce_int(merged.get("random_seed"), default=0),
            audit_enabled=coerce_bool(merged.get("audit_enabled"), default=False),
            audit_key=str(merged.get("audit_key", cls.audit_key)),
            audit_max_events=coerce_int(merged.get("audit_max_events"), default=DEFAULT_MAX_AUDIT_EVENTS, minimum=1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouterScoreBreakdown:
    """Inspectable score components for one agent ranking decision."""

    agent_name: str
    task_type: str = ""
    strategy: str = "base"
    score: float = 0.0
    success_rate: float = DEFAULT_NEW_AGENT_SUCCESS_BIAS
    failure_rate: float = 0.0
    active_tasks: float = 0.0
    capability_score: float = 0.0
    priority_score: float = 0.0
    reliability_score: float = 1.0
    latency_ms: float = 0.0
    freshness_score: float = 1.0
    affinity_score: float = 0.0
    availability_penalty: float = 0.0
    overload_penalty: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self), drop_empty=True)
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class RankedAgentRecord:
    """Detailed ranked-agent record produced by explanatory methods."""

    rank: int
    agent_name: str
    score: float
    agent_meta: Dict[str, Any]
    breakdown: RouterScoreBreakdown
    selected: bool = False

    def to_tuple(self) -> RankedTuple:
        return (self.agent_name, self.agent_meta, self.score)

    def to_dict(self, *, include_agent_meta: bool = True, redact: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "rank": self.rank,
            "agent_name": self.agent_name,
            "score": self.score,
            "selected": self.selected,
            "breakdown": self.breakdown.to_dict(redact=redact),
        }
        if include_agent_meta:
            payload["agent_meta"] = json_safe(self.agent_meta)
        payload = prune_none(payload, drop_empty=True)
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class RouterRankingReport:
    """Stable explanatory report for a strategy ranking pass."""

    strategy: str
    task_type: str
    ranked_agents: Tuple[Dict[str, Any], ...]
    candidate_count: int
    returned_count: int
    duration_ms: float
    correlation_id: str = field(default_factory=lambda: generate_correlation_id("route-strategy"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def selected_agent(self) -> Optional[str]:
        if not self.ranked_agents:
            return None
        return self.ranked_agents[0].get("agent_name")

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self), drop_empty=True)
        payload["selected_agent"] = self.selected_agent
        return redact_mapping(payload) if redact else payload


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def normalize_strategy_name(name: Any, *, default: str = RouterStrategyName.WEIGHTED.value) -> str:
    """Normalize strategy names and common aliases into canonical values."""

    value = normalize_whitespace(name or default).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "weighted_router": RouterStrategyName.WEIGHTED.value,
        "score": RouterStrategyName.WEIGHTED.value,
        "scored": RouterStrategyName.WEIGHTED.value,
        "leastloaded": RouterStrategyName.LEAST_LOADED.value,
        "least_load": RouterStrategyName.LEAST_LOADED.value,
        "least_busy": RouterStrategyName.LEAST_LOADED.value,
        "load": RouterStrategyName.LEAST_LOADED.value,
        "roundrobin": RouterStrategyName.ROUND_ROBIN.value,
        "rr": RouterStrategyName.ROUND_ROBIN.value,
        "capability": RouterStrategyName.CAPABILITY_MATCH.value,
        "capabilities": RouterStrategyName.CAPABILITY_MATCH.value,
        "capability_match": RouterStrategyName.CAPABILITY_MATCH.value,
        "random": RouterStrategyName.RANDOM_WEIGHTED.value,
        "randomized": RouterStrategyName.RANDOM_WEIGHTED.value,
        "random_weighted": RouterStrategyName.RANDOM_WEIGHTED.value,
    }
    return aliases.get(value, value or default)


def get_task_routing_config(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return merged routing configuration from collaborative_config.yaml and overrides."""

    runtime_config = load_global_config()
    routing_config = get_config_section("task_routing") or {}
    # Keep this explicit even if runtime_config is currently unused; it preserves
    # the module pattern used by surrounding collaborative components.
    _ = runtime_config
    return merge_mappings(routing_config, config or {}, deep=True, drop_none=True)


def extract_task_type(task_data: Optional[Mapping[str, Any]]) -> str:
    """Best-effort task type extraction for scoring diagnostics."""

    if not isinstance(task_data, Mapping):
        return ""
    for key in ("task_type", "type", "task", "operation", "capability"):
        value = task_data.get(key)
        if value is not None and normalize_whitespace(value):
            return normalize_task_type(value, allow_empty=True)
    return ""


def extract_required_capabilities(task_data: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Extract required capability hints from task data without enforcing them."""

    if not isinstance(task_data, Mapping):
        return ()
    candidates: List[Any] = []
    for key in ("required_capabilities", "capabilities", "required", "requires"):
        if key in task_data:
            value = task_data.get(key)
            if isinstance(value, (str, bytes)):
                candidates.append(value)
            elif isinstance(value, Iterable):
                candidates.extend(list(value))
            elif value is not None:
                candidates.append(value)
    task_type = extract_task_type(task_data)
    return normalize_capabilities(candidates, include_task_type=task_type or None)


def extract_preferred_agents(task_data: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Extract preferred agent names from task hints."""

    if not isinstance(task_data, Mapping):
        return ()
    values: List[Any] = []
    for key in ("preferred_agent", "preferred_agents", "agent", "agent_name"):
        if key not in task_data:
            continue
        raw = task_data.get(key)
        if isinstance(raw, (str, bytes)):
            values.append(raw)
        elif isinstance(raw, Iterable):
            values.extend(list(raw))
        elif raw is not None:
            values.append(raw)
    normalized: Dict[str, None] = {}
    for value in values:
        with_value = normalize_whitespace(value)
        if with_value:
            normalized[with_value] = None
    return tuple(normalized.keys())


def normalize_agents_for_ranking(agents: Optional[Mapping[str, Mapping[str, Any]]]) -> AgentMap:
    """Validate and copy candidate agent metadata for ranking."""

    if agents is None:
        return {}
    if not isinstance(agents, Mapping):
        raise _routing_error("agents must be a mapping of agent name to metadata", context={"received_type": type(agents).__name__})
    normalized: AgentMap = {}
    for raw_name, raw_meta in agents.items():
        name = normalize_agent_name(raw_name)
        meta = ensure_mapping(raw_meta, field_name=f"agents[{name}]", allow_none=True)
        meta.setdefault("capabilities", list(extract_agent_capabilities(meta)))
        normalized[name] = meta
    return normalized


def normalize_stats_for_ranking(stats: Optional[Mapping[str, Any]]) -> StatsMap:
    """Normalize runtime stats map for strategy scoring."""

    if not stats:
        return {}
    return normalize_agent_stats_map(stats)


def agent_is_marked_unavailable(agent_meta: Mapping[str, Any], agent_stats: Mapping[str, Any]) -> bool:
    """Return True when metadata or stats explicitly mark an agent unavailable."""

    status = str(agent_meta.get("status", agent_stats.get("status", ""))).strip().lower()
    if status in {"unavailable", "offline", "disabled", "degraded_hard", "blocked"}:
        return True
    circuit_state = str(agent_stats.get("circuit_state", agent_stats.get("state", ""))).strip().lower()
    if circuit_state == "open":
        return True
    if agent_meta.get("available") is False or agent_stats.get("available") is False:
        return True
    return False


def _routing_error(message: str, *, task_type: str = "unknown", context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> Exception:
    """Construct a collaboration routing error using the local taxonomy."""

    return make_collaboration_exception(
        "RoutingFailureError",
        message,
        context=normalize_metadata(context, drop_none=True),
        cause=cause,
    )


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------
class BaseRouterStrategy:
    name = "base"

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config_source = get_task_routing_config(config)
        self.strategy_config = RouterStrategyConfig.from_config(self.config_source)
        self._lock = threading.RLock()
        self._last_report: Optional[RouterRankingReport] = None

    def rank_agents(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Dict[str, Dict[str, Any]],
        task_data: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        raise NotImplementedError

    @property
    def last_report(self) -> Optional[RouterRankingReport]:
        return self._last_report

    def explain(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Optional[Mapping[str, Any]],
        task_data: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Return a diagnostic ranking report without changing router contract."""

        report = self.rank_agents_detailed(agents, stats or {}, task_data or {})
        return report.to_dict()

    def rank_agents_detailed(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Optional[Mapping[str, Any]],
        task_data: Optional[Mapping[str, Any]],
    ) -> RouterRankingReport:
        """Rank agents and return detailed scoring records."""

        started_ms = monotonic_ms()
        normalized_agents = self._filter_candidates(
            normalize_agents_for_ranking(agents),
            normalize_stats_for_ranking(stats),
            task_data or {},
        )
        normalized_stats = normalize_stats_for_ranking(stats)
        task = ensure_mapping(task_data, field_name="task_data", allow_none=True)
        records = self._score_records(normalized_agents, normalized_stats, task)
        ranked = self._assign_ranks(records)
        report = RouterRankingReport(
            strategy=self.name,
            task_type=extract_task_type(task),
            ranked_agents=tuple(record.to_dict(include_agent_meta=True) for record in ranked),
            candidate_count=len(agents or {}),
            returned_count=len(ranked),
            duration_ms=elapsed_ms(started_ms),
            metadata={"strategy_config": self.strategy_config.to_dict()},
        )
        self._last_report = report
        return report

    def _score_records(self, agents: AgentMap, stats: StatsMap, task_data: Dict[str, Any]) -> List[RankedAgentRecord]:
        records: List[RankedAgentRecord] = []
        for name, meta in agents.items():
            breakdown = self.score_agent(name, meta, stats.get(name, {}), task_data)
            records.append(RankedAgentRecord(rank=0, agent_name=name, score=breakdown.score, agent_meta=meta, breakdown=breakdown))
        return records

    def score_agent(self, agent_name: str, agent_meta: Mapping[str, Any], agent_stats: Mapping[str, Any], task_data: Mapping[str, Any]) -> RouterScoreBreakdown:
        """Score one agent. Concrete strategies may override this method."""

        return RouterScoreBreakdown(agent_name=agent_name, strategy=self.name, score=0.0)

    def _filter_candidates(self, agents: AgentMap, stats: StatsMap, task_data: Mapping[str, Any]) -> AgentMap:
        if not agents:
            return {}
        if not self.strategy_config.filter_unavailable and self.strategy_config.max_active_tasks_per_agent is None:
            return agents
        filtered: AgentMap = OrderedDict()
        for name, meta in agents.items():
            row = stats.get(name, {})
            if self.strategy_config.filter_unavailable and agent_is_marked_unavailable(meta, row):
                continue
            if self.strategy_config.max_active_tasks_per_agent is not None:
                active = coerce_int(row.get("active_tasks"), default=0, minimum=0)
                if active >= self.strategy_config.max_active_tasks_per_agent:
                    continue
            filtered[name] = meta
        return filtered

    def _sort_key(self, record: RankedAgentRecord) -> Tuple[Any, ...]:
        tie = self.strategy_config.stable_tie_breaker
        name_key = record.agent_name.lower()
        if tie == "priority":
            priority = coerce_float(record.agent_meta.get("priority"), default=0.0)
            return (-record.score, -priority, name_key)
        if tie == "last_seen":
            last_seen = coerce_float(record.breakdown.metadata.get("last_seen"), default=0.0)
            return (-record.score, -last_seen, name_key)
        if tie == "random":
            seeded = random.Random(stable_hash({"agent": record.agent_name, "score": record.score}, length=8))
            return (-record.score, seeded.random(), name_key)
        return (-record.score, name_key)

    def _assign_ranks(self, records: Sequence[RankedAgentRecord]) -> List[RankedAgentRecord]:
        sorted_records = sorted(records, key=self._sort_key)
        ranked: List[RankedAgentRecord] = []
        for index, record in enumerate(sorted_records, start=1):
            ranked.append(
                RankedAgentRecord(
                    rank=index,
                    agent_name=record.agent_name,
                    score=round(float(record.score), 6),
                    agent_meta=record.agent_meta,
                    breakdown=record.breakdown,
                    selected=index == 1,
                )
            )
        return ranked

    def _rank_from_records(self, records: Sequence[RankedAgentRecord]) -> List[RankedTuple]:
        return [record.to_tuple() for record in self._assign_ranks(records)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "strategy_config": self.strategy_config.to_dict(),
            "last_report": self._last_report.to_dict() if self._last_report else None,
        }


class WeightedRouterStrategy(BaseRouterStrategy):
    name = RouterStrategyName.WEIGHTED.value

    def __init__(self, weights: RouterScoreWeights | None = None, config: Optional[Mapping[str, Any]] = None):
        super().__init__(config=config)
        self.weights = weights or RouterScoreWeights.from_config(self.config_source)
        logger.info("Weighted Router Strategy initialized")

    def rank_agents(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Dict[str, Dict[str, Any]],
        task_data: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        report = self.rank_agents_detailed(agents, stats, task_data)
        return [(row["agent_name"], row["agent_meta"], row["score"]) for row in report.ranked_agents]

    def score_agent(self, agent_name: str, agent_meta: Mapping[str, Any], agent_stats: Mapping[str, Any], task_data: Mapping[str, Any]) -> RouterScoreBreakdown:
        task_type = extract_task_type(task_data)
        required_capabilities = extract_required_capabilities(task_data)
        caps = extract_agent_capabilities(agent_meta)
        successes = coerce_float(agent_stats.get("successes"), default=0.0, minimum=0.0)
        failures = coerce_float(agent_stats.get("failures"), default=0.0, minimum=0.0)
        active_tasks = coerce_float(agent_stats.get("active_tasks"), default=0.0, minimum=0.0)
        total = successes + failures
        success_rate = successes / total if total > 0 else self.strategy_config.new_agent_success_bias
        failure_rate = failures / total if total > 0 else 0.0
        capability_score = _capability_match_score(caps, required_capabilities)
        priority_score = _priority_score(agent_meta, task_data)
        reliability_score = _reliability_score(agent_stats)
        latency_ms = _latency_ms(agent_stats)
        freshness_score = _freshness_score(agent_stats, self.strategy_config)
        affinity_score = _affinity_score(agent_name, agent_meta, task_data)
        availability_penalty = self.strategy_config.unavailable_score_penalty if agent_is_marked_unavailable(agent_meta, agent_stats) else 0.0
        overload_penalty = _overload_penalty(active_tasks, agent_meta, agent_stats)

        score = (
            (self.weights.success_rate * success_rate)
            + (self.weights.capability_match * capability_score)
            + (self.weights.priority * priority_score)
            + (self.weights.reliability * reliability_score)
            + (self.weights.freshness * freshness_score)
            + (self.weights.affinity * affinity_score)
            - (self.weights.load_penalty * active_tasks)
            - (self.weights.failure_penalty * failure_rate)
            - (self.weights.latency_penalty * latency_ms)
            - (self.weights.overload_penalty * overload_penalty)
            - availability_penalty
        )
        return RouterScoreBreakdown(
            agent_name=agent_name,
            task_type=task_type,
            strategy=self.name,
            score=score,
            success_rate=round(success_rate, 6),
            failure_rate=round(failure_rate, 6),
            active_tasks=active_tasks,
            capability_score=round(capability_score, 6),
            priority_score=round(priority_score, 6),
            reliability_score=round(reliability_score, 6),
            latency_ms=latency_ms,
            freshness_score=round(freshness_score, 6),
            affinity_score=round(affinity_score, 6),
            availability_penalty=availability_penalty,
            overload_penalty=overload_penalty,
            reason="weighted score from success, capability, priority, reliability, freshness, affinity, load, failure, latency and availability components",
            metadata={
                "capabilities": list(caps),
                "required_capabilities": list(required_capabilities),
                "last_seen": agent_stats.get("last_seen"),
                "weights": self.weights.to_dict(),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["weights"] = self.weights.to_dict()
        return payload


class LeastLoadedRouterStrategy(BaseRouterStrategy):
    name = RouterStrategyName.LEAST_LOADED.value

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        super().__init__(config=config)
        logger.info("Least Loaded Router Strategy initialized")

    def rank_agents(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Dict[str, Dict[str, Any]],
        task_data: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        report = self.rank_agents_detailed(agents, stats, task_data)
        return [(row["agent_name"], row["agent_meta"], row["score"]) for row in report.ranked_agents]

    def score_agent(self, agent_name: str, agent_meta: Mapping[str, Any], agent_stats: Mapping[str, Any], task_data: Mapping[str, Any]) -> RouterScoreBreakdown:
        active_tasks = coerce_float(agent_stats.get("active_tasks"), default=0.0, minimum=0.0)
        capability_score = _capability_match_score(extract_agent_capabilities(agent_meta), extract_required_capabilities(task_data))
        availability_penalty = self.strategy_config.unavailable_score_penalty if agent_is_marked_unavailable(agent_meta, agent_stats) else 0.0
        score = (-active_tasks) + capability_score - availability_penalty
        return RouterScoreBreakdown(
            agent_name=agent_name,
            task_type=extract_task_type(task_data),
            strategy=self.name,
            score=score,
            active_tasks=active_tasks,
            capability_score=capability_score,
            availability_penalty=availability_penalty,
            reason="least active tasks, with capability and availability adjustment",
            metadata={"last_seen": agent_stats.get("last_seen")},
        )


class CapabilityMatchRouterStrategy(BaseRouterStrategy):
    """Prefer the agent that most closely matches explicit task capabilities."""

    name = RouterStrategyName.CAPABILITY_MATCH.value

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        super().__init__(config=config)
        logger.info("Capability Match Router Strategy initialized")

    def rank_agents(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Dict[str, Dict[str, Any]],
        task_data: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        report = self.rank_agents_detailed(agents, stats, task_data)
        return [(row["agent_name"], row["agent_meta"], row["score"]) for row in report.ranked_agents]

    def score_agent(self, agent_name: str, agent_meta: Mapping[str, Any], agent_stats: Mapping[str, Any], task_data: Mapping[str, Any]) -> RouterScoreBreakdown:
        caps = extract_agent_capabilities(agent_meta)
        required = extract_required_capabilities(task_data)
        capability_score = _capability_match_score(caps, required)
        success_rate = calculate_success_rate(agent_stats, default=self.strategy_config.new_agent_success_bias)
        active_tasks = coerce_float(agent_stats.get("active_tasks"), default=0.0, minimum=0.0)
        score = (2.0 * capability_score) + (0.25 * success_rate) - (0.05 * active_tasks)
        return RouterScoreBreakdown(
            agent_name=agent_name,
            task_type=extract_task_type(task_data),
            strategy=self.name,
            score=score,
            success_rate=success_rate,
            active_tasks=active_tasks,
            capability_score=capability_score,
            reason="capability coverage first, then success-rate and load tie-breaks",
            metadata={"capabilities": list(caps), "required_capabilities": list(required), "last_seen": agent_stats.get("last_seen")},
        )


class RoundRobinRouterStrategy(BaseRouterStrategy):
    """Deterministic round-robin strategy with capability-aware filtering."""

    name = RouterStrategyName.ROUND_ROBIN.value

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        super().__init__(config=config)
        self._cursors: Dict[str, int] = defaultdict(int)
        logger.info("Round Robin Router Strategy initialized")

    def rank_agents(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Dict[str, Dict[str, Any]],
        task_data: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        with self._lock:
            report = self.rank_agents_detailed(agents, stats, task_data)
            return [(row["agent_name"], row["agent_meta"], row["score"]) for row in report.ranked_agents]

    def _score_records(self, agents: AgentMap, stats: StatsMap, task_data: Dict[str, Any]) -> List[RankedAgentRecord]:
        names = sorted(agents.keys(), key=str.lower)
        if not names:
            return []
        route_key = extract_task_type(task_data) or "default"
        cursor = self._cursors[route_key] % len(names)
        ordered_names = names[cursor:] + names[:cursor]
        self._cursors[route_key] = (cursor + 1) % len(names)
        records: List[RankedAgentRecord] = []
        count = len(ordered_names)
        for index, name in enumerate(ordered_names):
            meta = agents[name]
            row = stats.get(name, {})
            capability_score = _capability_match_score(extract_agent_capabilities(meta), extract_required_capabilities(task_data))
            availability_penalty = self.strategy_config.unavailable_score_penalty if agent_is_marked_unavailable(meta, row) else 0.0
            score = (count - index) + capability_score - availability_penalty
            breakdown = RouterScoreBreakdown(
                agent_name=name,
                task_type=extract_task_type(task_data),
                strategy=self.name,
                score=score,
                active_tasks=coerce_float(row.get("active_tasks"), default=0.0, minimum=0.0),
                capability_score=capability_score,
                availability_penalty=availability_penalty,
                reason="round-robin order with capability and availability adjustment",
                metadata={"cursor": cursor, "route_key": route_key},
            )
            records.append(RankedAgentRecord(rank=0, agent_name=name, score=breakdown.score, agent_meta=meta, breakdown=breakdown))
        return records


class RandomWeightedRouterStrategy(WeightedRouterStrategy):
    """Weighted ranking with small deterministic jitter for load spreading."""

    name = RouterStrategyName.RANDOM_WEIGHTED.value

    def __init__(self, weights: RouterScoreWeights | None = None, config: Optional[Mapping[str, Any]] = None):
        super().__init__(weights=weights, config=config)
        self._random = random.Random(self.strategy_config.random_seed)
        logger.info("Random Weighted Router Strategy initialized")

    def score_agent(self, agent_name: str, agent_meta: Mapping[str, Any], agent_stats: Mapping[str, Any], task_data: Mapping[str, Any]) -> RouterScoreBreakdown:
        base = super().score_agent(agent_name, agent_meta, agent_stats, task_data)
        jitter = self._random.uniform(0.0, 0.01)
        return RouterScoreBreakdown(
            **merge_mappings(base.to_dict(redact=False), {"score": base.score + jitter, "reason": f"{base.reason}; deterministic jitter={jitter:.6f}"}, deep=True)
        )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _capability_match_score(capabilities: Iterable[Any], required_capabilities: Iterable[Any]) -> float:
    caps = {normalize_whitespace(item).lower() for item in capabilities if normalize_whitespace(item)}
    required = {normalize_whitespace(item).lower() for item in required_capabilities if normalize_whitespace(item)}
    if not required:
        return 1.0
    if not caps:
        return 0.0
    intersection = caps & required
    exact_coverage = len(intersection) / len(required)
    # Include partial text overlap for common hierarchical capability names.
    partial_hits = 0
    for required_item in required - intersection:
        if any(required_item in cap or cap in required_item for cap in caps):
            partial_hits += 1
    partial_coverage = 0.5 * (partial_hits / len(required))
    return clamp(exact_coverage + partial_coverage, 0.0, 1.0)  # type: ignore[return-value]


def _priority_score(agent_meta: Mapping[str, Any], task_data: Mapping[str, Any]) -> float:
    agent_priority = coerce_float(agent_meta.get("priority"), default=0.0)
    task_priority = coerce_float(task_data.get("priority"), default=0.0)
    return agent_priority + (0.1 * task_priority)


def _reliability_score(agent_stats: Mapping[str, Any]) -> float:
    circuit_state = str(agent_stats.get("circuit_state", agent_stats.get("state", "closed"))).strip().lower()
    if circuit_state == "open":
        return 0.0
    if circuit_state == "half_open":
        return 0.35
    availability = agent_stats.get("available")
    if availability is False:
        return 0.0
    return clamp(calculate_success_rate(agent_stats, default=DEFAULT_NEW_AGENT_SUCCESS_BIAS), 0.0, 1.0)  # type: ignore[return-value]


def _latency_ms(agent_stats: Mapping[str, Any]) -> float:
    for key in ("avg_latency_ms", "latency_ms", "mean_latency_ms", "p50_latency_ms"):
        if key in agent_stats:
            return coerce_float(agent_stats.get(key), default=0.0, minimum=0.0)
    return 0.0


def _freshness_score(agent_stats: Mapping[str, Any], config: RouterStrategyConfig) -> float:
    last_seen = agent_stats.get("last_seen")
    if last_seen is None:
        return 1.0
    age = max(0.0, epoch_seconds() - coerce_float(last_seen, default=epoch_seconds(), minimum=0.0))
    if config.stale_after_seconds is None or config.stale_after_seconds <= 0:
        return 1.0
    if age <= config.stale_after_seconds:
        return 1.0
    # A stale agent is not necessarily unroutable; registry/router reliability
    # modules may still attempt fallback. Penalize smoothly instead.
    excess = age - config.stale_after_seconds
    return max(0.0, 1.0 - (config.stale_agent_penalty * (1.0 + excess / config.stale_after_seconds)))


def _affinity_score(agent_name: str, agent_meta: Mapping[str, Any], task_data: Mapping[str, Any]) -> float:
    preferred = {item.lower() for item in extract_preferred_agents(task_data)}
    if not preferred:
        return 0.0
    names = {agent_name.lower(), str(agent_meta.get("name", "")).lower(), str(agent_meta.get("agent_name", "")).lower()}
    aliases = agent_meta.get("aliases") or agent_meta.get("alias") or ()
    if isinstance(aliases, (str, bytes)):
        names.add(str(aliases).lower())
    elif isinstance(aliases, Iterable):
        names.update(str(item).lower() for item in aliases)
    return 1.0 if preferred & {item for item in names if item} else 0.0


def _overload_penalty(active_tasks: float, agent_meta: Mapping[str, Any], agent_stats: Mapping[str, Any]) -> float:
    max_tasks = agent_meta.get("max_tasks", agent_meta.get("max_concurrent_tasks", agent_stats.get("max_tasks")))
    if max_tasks is None:
        return 0.0
    limit = coerce_float(max_tasks, default=0.0, minimum=0.0)
    if limit <= 0:
        return 0.0
    return max(0.0, active_tasks - limit)


# ---------------------------------------------------------------------------
# Strategy registry and factory
# ---------------------------------------------------------------------------
_STRATEGY_REGISTRY: Dict[str, Type[BaseRouterStrategy]] = {}


def register_router_strategy(name: str, strategy_cls: Type[BaseRouterStrategy], *, replace: bool = False) -> None:
    """Register a strategy class for factory lookup."""

    normalized = normalize_strategy_name(name)
    if not issubclass(strategy_cls, BaseRouterStrategy):
        raise _routing_error("strategy_cls must inherit BaseRouterStrategy", context={"strategy": normalized, "received": repr(strategy_cls)})
    if normalized in _STRATEGY_REGISTRY and not replace:
        raise _routing_error("router strategy already registered", context={"strategy": normalized})
    _STRATEGY_REGISTRY[normalized] = strategy_cls


def list_router_strategies() -> List[str]:
    """Return registered strategy names."""

    return sorted(_STRATEGY_REGISTRY.keys())


def build_router_strategy(name: str, config: Dict[str, Any] | None = None) -> BaseRouterStrategy:
    config = get_task_routing_config(config)
    configured_strategy = config.get("strategy")
    if isinstance(config.get("router_strategy"), Mapping):
        configured_strategy = config["router_strategy"].get("strategy", configured_strategy)
    lowered = normalize_strategy_name(name or configured_strategy or RouterStrategyName.WEIGHTED.value)

    if not _STRATEGY_REGISTRY:
        _register_default_strategies()

    strategy_cls = _STRATEGY_REGISTRY.get(lowered)
    if strategy_cls is None:
        logger.warning("Unknown router strategy '%s'; falling back to weighted.", lowered)
        strategy_cls = WeightedRouterStrategy

    if strategy_cls is WeightedRouterStrategy:
        return WeightedRouterStrategy(weights=RouterScoreWeights.from_config(config), config=config)
    if strategy_cls is RandomWeightedRouterStrategy:
        return RandomWeightedRouterStrategy(weights=RouterScoreWeights.from_config(config), config=config)
    return strategy_cls(config=config)


def _register_default_strategies() -> None:
    register_router_strategy(RouterStrategyName.WEIGHTED.value, WeightedRouterStrategy, replace=True)
    register_router_strategy(RouterStrategyName.LEAST_LOADED.value, LeastLoadedRouterStrategy, replace=True)
    register_router_strategy(RouterStrategyName.CAPABILITY_MATCH.value, CapabilityMatchRouterStrategy, replace=True)
    register_router_strategy(RouterStrategyName.ROUND_ROBIN.value, RoundRobinRouterStrategy, replace=True)
    register_router_strategy(RouterStrategyName.RANDOM_WEIGHTED.value, RandomWeightedRouterStrategy, replace=True)


_register_default_strategies()


if __name__ == "__main__":
    print("\n=== Running Router Strategy ===\n")
    printer.status("TEST", "Router Strategy initialized", "info")

    agents = {
        "TranslatorA": {
            "capabilities": ["translation", "summarization"],
            "priority": 1,
            "max_tasks": 4,
            "aliases": ["fast_translator"],
        },
        "TranslatorB": {
            "capabilities": ["translation"],
            "priority": 2,
            "max_tasks": 8,
        },
        "Analyzer": {
            "capabilities": ["analysis", "data"],
            "priority": 5,
        },
    }
    stats = {
        "TranslatorA": {"successes": 9, "failures": 1, "active_tasks": 3, "avg_latency_ms": 50, "last_seen": epoch_seconds()},
        "TranslatorB": {"successes": 2, "failures": 0, "active_tasks": 0, "avg_latency_ms": 80, "last_seen": epoch_seconds()},
        "Analyzer": {"successes": 100, "failures": 2, "active_tasks": 1, "avg_latency_ms": 10, "last_seen": epoch_seconds()},
    }
    task = {
        "task_type": "translation",
        "required_capabilities": ["translation"],
        "priority": 3,
        "preferred_agents": ["TranslatorB"],
    }

    weighted = build_router_strategy("weighted", {"weight_success_rate": 1.0, "weight_load_penalty": 0.25})
    weighted_ranked = weighted.rank_agents(agents, stats, task)
    assert weighted_ranked, "weighted strategy returned no agents"
    assert weighted_ranked[0][0] in agents, weighted_ranked
    explanation = weighted.explain(agents, stats, task)
    assert explanation["ranked_agents"], explanation
    assert explanation["selected_agent"] == weighted_ranked[0][0], explanation
    printer.status("TEST", "Weighted strategy ranking/explanation passed", "success")

    least_loaded = build_router_strategy("least_loaded", {})
    least_loaded_ranked = least_loaded.rank_agents(agents, stats, task)
    assert least_loaded_ranked[0][0] == "TranslatorB", least_loaded_ranked
    printer.status("TEST", "Least-loaded strategy ranking passed", "success")

    capability = build_router_strategy("capability_match", {})
    capability_ranked = capability.rank_agents(agents, stats, task)
    assert capability_ranked[0][0] in {"TranslatorA", "TranslatorB"}, capability_ranked
    printer.status("TEST", "Capability-match strategy ranking passed", "success")

    round_robin = build_router_strategy("round_robin", {})
    first = round_robin.rank_agents(agents, stats, task)[0][0]
    second = round_robin.rank_agents(agents, stats, task)[0][0]
    assert first != second, (first, second)
    printer.status("TEST", "Round-robin strategy cursor passed", "success")

    random_weighted = build_router_strategy("random_weighted", {"router_strategy": {"random_seed": 7}})
    randomized_ranked = random_weighted.rank_agents(agents, stats, task)
    assert randomized_ranked, randomized_ranked
    printer.status("TEST", "Random-weighted strategy ranking passed", "success")

    assert "weighted" in list_router_strategies()
    assert normalize_strategy_name("least-busy") == "least_loaded"
    assert extract_required_capabilities(task) == ("translation",)
    assert extract_preferred_agents(task) == ("TranslatorB",)

    print("\n=== Test ran successfully ===\n")
