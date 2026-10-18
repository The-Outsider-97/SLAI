"""SLAI PlanSuccessPredictor (PSP): an advisory, offline-trained whole-plan outcome model.

The existing ``HeuristicSelector`` answers "which method should this *task* use?". PSP answers a different
question: "will this complete, scheduled *plan* succeed, how long will it really take, and how likely is it to
need recovery?". It scores the finished plan (tasks, dependencies, schedule, resources, agents, environment and
history) *before* execution and reports, as calibrated probabilities:

    success         the plan reaches its goal
    replan          execution needed replanning / recovery
    resource_risk   a resource wait, contention or violation occurred
    deadline_miss   at least one task finished after its deadline
    cause_*         given a failure: resource / temporal / task failure (conditional on failure)
    duration        realised-to-planned makespan ratio of a completed plan, with an interval

PSP never generates, repairs, validates or authorises a plan. It cannot bypass ``SafetyPlanning`` or the scheduler
and it defers (``defer_to_heuristics``) when it is uncertain or the plan is unlike its training data. The
Planning Agent keeps full authority; an unavailable model returns ``{}`` and planning behaves exactly as before.

Layers
------
* ``normalize_plan`` / ``extract_features`` - one code path for ``Task`` objects and plain mappings (a
  ``Task.to_dict()`` or a scheduler payload). Only information known *before* execution is used (no target leakage).
* ``labels_from_outcome`` - the single definition of "observed outcome". Used by the trainer, the agent hook and
  the history features, so they cannot diverge.
* ``PlanSuccessModel`` - pure-Python inference of a constrained JSON checkpoint (never pickle): gradient-boosted
  tree ensembles with per-head calibration, uncertainty, out-of-distribution detection and feature attribution.
* ``PlanSuccessPredictor`` - the PlanningAgent-facing module: configuration-injected, lazily loaded, fail-open.

Modes (``plan_success_predictor.mode`` in planning_config.yaml)
    collect   record features and the observed outcome only (training-data collection, no model needed)
    shadow    additionally predict and record beside the real outcome (no effect on planning)
    advisory  additionally expose a recommendation; requires a checkpoint approved via ``train_psp.py export``

Training lives in ``train_psp.py`` at the repository root. Inference performs no network access.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section
from .utils.planning_helpers import clamp, compute_resource_margin, topological_sort
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Plan Success Predictor")
printer = PrettyPrinter()

MODEL_SCHEMA = 1
ROW_SCHEMA = 1
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = "src/agents/planning/models/plan_success_predictor/manifest.json"
DEFAULT_OBSERVATION_LOG = "data/training/planning/psp/observations.jsonl"
CONFIG_SECTION = "plan_success_predictor"
MAX_CHECKPOINT_BYTES = 40_000_000
MODES = ("collect", "shadow", "advisory")
OBSERVED = "observed_execution"  # target_source of rows that come from real executions

CAUSES = ("resource", "temporal", "task")
OUTCOMES = ("success", "replan", "resource_risk", "deadline_miss")  # observable for every executed plan
CLASSIFIERS = OUTCOMES + tuple("cause_" + c for c in CAUSES)  # cause_* are conditional on failure
REGRESSORS = ("duration_log_ratio",)  # conditional on completion: log(actual makespan / planned makespan)
TARGETS = CLASSIFIERS + REGRESSORS
UNCERTAINTY_HEADS = ("success", "replan", "resource_risk")
LOG_RATIO_LIMIT = 3.0

FEATURES: Tuple[str, ...] = (
    # structure
    "log_tasks", "primitive_share", "mean_fan_in", "max_fan_in_share", "max_fan_out_share",
    "depth_ratio", "log_depth", "width_ratio", "root_share",
    # timing
    "log_total_work", "log_critical_path", "log_parallelism", "is_scheduled", "log_planned_makespan",
    "schedule_efficiency", "log_mean_duration", "duration_cv", "max_duration_share",
    # deadlines
    "deadline_share", "deadline_slack", "late_share", "tight_deadline_share", "time_buffer_ratio",
    # resources
    "resources_known", "log_gpu_total", "log_ram_total", "gpu_peak_ratio", "ram_peak_ratio", "gpu_margin",
    "ram_margin", "concurrency_peak_ratio", "specialized_hw_share", "resource_heavy_share",
    # task risk and reliability
    "risk_mean", "risk_max", "high_risk_share", "criticality_mean", "critical_share", "retry_budget_mean",
    "zero_retry_share", "hist_task_known_share", "hist_task_success_mean", "hist_task_success_min",
    "hist_log_hazard",
    # agents and environment
    "agents_known", "log_agents_used", "agent_balance", "agent_reliability_mean", "agent_reliability_min",
    "agent_load_mean", "agent_efficiency_mean", "system_load",
    # plan-level history of the same workflow family
    "plan_hist_known", "plan_hist_log_count", "plan_hist_success_rate", "plan_hist_replan_rate",
    "plan_hist_mean_log_ratio",
    # other task properties
    "probabilistic_share", "human_share", "safety_constraint_share", "parallelizable_share", "priority_mean",
)
NF, NT = len(FEATURES), len(TARGETS)
FEATURE_SCHEMA_ID = hashlib.sha256("|".join((str(MODEL_SCHEMA),) + FEATURES + TARGETS).encode()).hexdigest()[:16]

_ABSOLUTE_EPOCH = 1_000_000_000  # same relative/absolute time split as planning_types
DEFAULT_RANKING_WEIGHTS: Dict[str, float] = {"success": 1.0, "replan": 0.25, "resource": 0.15, "duration": 0.10}
_CRITICALITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_SAFETY_DEFAULTS = {"gpu_buffer": 0.15, "ram_buffer": 0.20, "max_concurrent": 5, "time_buffer": 120.0}
_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "mode": "shadow",
    "model_path": DEFAULT_MODEL_PATH,
    "allow_experimental": True,  # honoured for collect/shadow only; advisory always needs approval
    "record_features": True,
    "record_observations": True,
    "observation_log": DEFAULT_OBSERVATION_LOG,
    "max_log_bytes": 50_000_000,
    "max_tasks": 500,
    "history_limit": 20,
    "max_families": 256,
    "top_drivers": 3,
    # Candidate ranking only; the Planning Agent keeps the final decision.
    "ranking_weights": DEFAULT_RANKING_WEIGHTS,
}


# =============================================================================
# Small numeric helpers
# =============================================================================
def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute of a ``Task``-like object or key of a mapping (scheduler payloads, ``Task.to_dict()``)."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def family_group(family: str) -> str:
    """Split group of a workflow family: all plans (and strategy variants) of a family share one split."""
    return hashlib.sha256(("psp-family:" + family).encode()).hexdigest()[:16]


# =============================================================================
# Plan normalisation
# =============================================================================
def _profile(task: Any) -> Tuple[float, float, int]:
    profile = _get(task, "resource_requirements")
    gpu = max(0.0, _finite(_get(profile, "gpu", 0.0)))
    ram = max(0.0, _finite(_get(profile, "ram", _get(profile, "memory", 0.0))))
    hardware = _get(profile, "specialized_hardware", []) or []
    return gpu, ram, len(hardware) if isinstance(hardware, (list, tuple, set)) else 0


def _dependency_ids(task: Any) -> List[str]:
    out = []
    for dep in _get(task, "dependencies", []) or []:
        out.append(str(_get(dep, "id", dep)) if not isinstance(dep, str) else dep)
    return out


def _is_primitive(task: Any) -> bool:
    kind = _get(task, "task_type", _get(task, "type"))
    if kind is None:
        return True
    return str(getattr(kind, "name", kind)).upper() in ("PRIMITIVE", "0")


def _assigned_agent(task: Any) -> Optional[str]:
    context = _get(task, "context") or {}
    agent = (context.get("assigned_agent") if isinstance(context, Mapping) else None) \
        or _get(task, "assigned_agent") or _get(task, "agent_id")
    return str(agent) if agent else None


class _PlacedTask:
    """A ``Task``-like object seen together with its scheduler placement (read-only overlay)."""

    __slots__ = ("_task", "_placement")

    def __init__(self, task: Any, placement: Dict[str, Any]) -> None:
        object.__setattr__(self, "_task", task)
        object.__setattr__(self, "_placement", placement)

    def __getattr__(self, name: str) -> Any:
        placement = object.__getattribute__(self, "_placement")
        if name in placement:
            return placement[name]
        return getattr(object.__getattribute__(self, "_task"), name)


def merge_schedule(tasks: Iterable[Any], schedule: Optional[Mapping[str, Mapping[str, Any]]]) -> List[Any]:
    """Lay a ``DeadlineAwareScheduler`` schedule over its tasks: the plan shape ``extract_features`` scores.

    The assignment supplies start, end, (agent-adjusted) duration and the assigned agent; everything else - above
    all the task's own deadline, which the scheduler defaults for tasks without one - stays the task's. A task
    without an assignment stays unscheduled. Mapping tasks come back as new dicts, objects as read-only overlays.
    """
    schedule = schedule or {}
    merged: List[Any] = []
    for index, task in enumerate(tasks):
        tid = str(_get(task, "id") or _get(task, "task_id") or "task_%d" % index)
        assignment = schedule.get(tid)
        placement: Dict[str, Any] = {}
        if isinstance(assignment, Mapping):
            for key in ("start_time", "end_time", "duration"):
                if _is_number(assignment.get(key)):
                    placement[key] = float(assignment[key])
            if assignment.get("agent_id"):
                placement["assigned_agent"] = str(assignment["agent_id"])
        merged.append({**task, **placement} if isinstance(task, Mapping) else _PlacedTask(task, placement))
    return merged


def normalize_plan(plan: Iterable[Any], *, max_tasks: int = 500) -> List[Dict[str, Any]]:
    """Plan -> list of plain task specs with *relative* times (seconds from the first scheduled start).

    Accepts ``Task`` objects, ``Task.to_dict()`` mappings and scheduler payloads alike. A time below 1e9 is already
    relative (the convention of ``planning_types``); an absolute epoch is made relative to the earliest start.
    ``start``/``end`` are ``None`` for an unscheduled task, ``deadline`` is ``0.0`` when there is none.
    A plan counts as scheduled only when every task has a window.
    """
    items = [t for t in plan if t is not None]
    if not items:
        raise ValueError("empty plan")
    if len(items) > max_tasks:
        raise ValueError("plan has %d tasks (limit %d)" % (len(items), max_tasks))
    ids = [str(_get(t, "id") or _get(t, "task_id") or "task_%d" % i) for i, t in enumerate(items)]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task ids in plan")
    known = set(ids)
    absolute_starts = [s for s in (_finite(_get(t, "start_time", 0.0)) for t in items) if s >= _ABSOLUTE_EPOCH]
    base = min(absolute_starts) if absolute_starts else 0.0

    def window(task: Any, duration: float) -> Tuple[Optional[float], Optional[float]]:
        start, end = _finite(_get(task, "start_time")), _finite(_get(task, "end_time"))
        if end <= 0.0 and start > 0.0 and duration > 0.0:
            end = start + duration
        if end <= 0.0 or start < 0.0 or end < start or (start >= _ABSOLUTE_EPOCH) != (end >= _ABSOLUTE_EPOCH):
            return None, None  # unscheduled (an unset time is 0.0, as in planning_types)
        return (start - base, end - base) if end >= _ABSOLUTE_EPOCH else (start, end)

    def deadline_offset(value: Any) -> float:
        number = _finite(value)
        if number <= 0.0:
            return 0.0
        return max(1e-3, number - base) if number >= _ABSOLUTE_EPOCH else number  # an already-passed deadline stays > 0

    specs = []
    for tid, task in zip(ids, items):
        duration = _finite(_get(task, "duration", 0.0)) or _finite(_get(task, "estimated_duration", 0.0))
        gpu, ram, hardware = _profile(task)
        start, end = window(task, duration)
        specs.append({
            "id": tid, "name": str(_get(task, "name", tid)),
            "deps": [d for d in _dependency_ids(task) if d in known and d != tid],
            "duration": max(0.0, duration), "start": start, "end": end,
            "deadline": deadline_offset(_get(task, "deadline")),
            "gpu": gpu, "ram": ram, "hw": hardware,
            "risk": clamp(_finite(_get(task, "risk_score", 0.0)), 0.0, 1.0),
            "priority": max(0.0, _finite(_get(task, "priority", 1.0))),
            "retries": max(0, int(_finite(_get(task, "max_retries", 3)))),
            "crit": _CRITICALITY.get(str(_get(task, "criticality", "medium")).lower(), 1),
            "parallelizable": bool(_get(task, "parallelizable", False)),
            "human": bool(_get(task, "human_interaction_required", False)),
            "safety": bool(_get(task, "safety_constraints", None)),
            "probabilistic": bool(_get(task, "is_probabilistic", False)),
            "primitive": _is_primitive(task), "agent": _assigned_agent(task),
        })
    return specs


# =============================================================================
# Features
# =============================================================================
def _graph(specs: Sequence[Mapping[str, Any]]) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, float]]:
    """Topological order, longest-chain level, fan-out and (unconstrained) earliest finish per task."""
    ids = [s["id"] for s in specs]
    deps = {s["id"]: s["deps"] for s in specs}
    order = topological_sort(ids, deps)  # raises CyclicDependencyError for a cyclic plan
    duration = {s["id"]: s["duration"] for s in specs}
    level: Dict[str, int] = {}
    finish: Dict[str, float] = {}
    fan_out: Counter = Counter()
    for tid in order:
        level[tid] = 1 + max((level[d] for d in deps[tid]), default=0)
        finish[tid] = max((finish[d] for d in deps[tid]), default=0.0) + duration[tid]
        fan_out.update(deps[tid])
    return order, level, dict(fan_out), finish


def _peaks(windows: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float, int]:
    """Peak concurrent (gpu, ram, task count) of scheduled windows; ends are processed before starts at a tie."""
    events = []
    for start, end, gpu, ram in windows:
        events.append((start, 1, gpu, ram))
        events.append((end, 0, -gpu, -ram))
    events.sort(key=lambda e: (e[0], e[1]))
    gpu = ram = peak_gpu = peak_ram = 0.0
    count = peak_count = 0
    for _, kind, d_gpu, d_ram in events:
        gpu, ram = gpu + d_gpu, ram + d_ram
        count += 1 if kind else -1
        peak_gpu, peak_ram, peak_count = max(peak_gpu, gpu), max(peak_ram, ram), max(peak_count, count)
    return peak_gpu, peak_ram, peak_count


def _agent_detail(agents: Any, agent_id: str) -> Optional[Any]:
    if isinstance(agents, Mapping):
        return agents.get(agent_id)
    return None


def _success_rate(entry: Any) -> Optional[float]:
    """Smoothed success rate from a rate in [0, 1], a (success, total) pair or a {"success", "total"} mapping."""
    if isinstance(entry, Mapping):
        success, total = _finite(entry.get("success")), _finite(entry.get("total"))
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        success, total = _finite(entry[0]), _finite(entry[1])
    elif _is_number(entry):
        return clamp(float(entry), 0.0, 1.0)
    else:
        return None
    return (min(success, total) + 1.0) / (total + 2.0) if total > 0 else None


def task_history_from_method_stats(method_stats: Mapping[Any, Any]) -> Dict[str, Tuple[float, float]]:
    """Aggregate PlanningAgent.method_stats ``{(task_name, method): {"success", "total"}}`` per task name."""
    totals: Dict[str, List[float]] = {}
    for key, stats in (method_stats or {}).items():
        if not isinstance(stats, Mapping):
            continue
        name = str(key[0]) if isinstance(key, tuple) and key else str(key)
        pair = totals.setdefault(name, [0.0, 0.0])
        pair[0] += max(0.0, _finite(stats.get("success")))
        pair[1] += max(0.0, _finite(stats.get("total")))
    return {name: (s, t) for name, (s, t) in totals.items() if t > 0}


def extract_features(
    plan: Iterable[Any],
    *,
    agents: Optional[Mapping[str, Any]] = None,
    resources: Any = None,
    safety_margins: Any = None,
    system_load: Optional[float] = None,
    task_history: Optional[Mapping[str, Any]] = None,
    outcome_summary: Optional[Mapping[str, float]] = None,
    max_tasks: int = _DEFAULTS["max_tasks"],
) -> Dict[str, float]:
    """Pre-execution features of one plan. Every key of ``FEATURES`` is present and finite.

    ``agents``           scheduler registry ``{id: {successes, failures, efficiency, current_load, ...}}``
    ``resources``        ``ClusterResources`` or mapping with ``gpu_total`` / ``ram_total``
    ``safety_margins``   ``SafetyMargins`` or mapping (gpu_buffer, ram_buffer, max_concurrent, time_buffer)
    ``task_history``     task name -> success rate | (success, total) | {"success", "total"}
    ``outcome_summary``  ``OutcomeHistory.summary(family)``
    """
    specs = normalize_plan(plan, max_tasks=max_tasks)
    n = len(specs)
    order, level, fan_out, earliest = _graph(specs)
    by_id = {s["id"]: s for s in specs}
    f: Dict[str, float] = {}

    # structure
    chain = max(level.values())
    width = max(Counter(level.values()).values())
    f["log_tasks"] = math.log1p(n)
    f["primitive_share"] = sum(s["primitive"] for s in specs) / n
    f["mean_fan_in"] = min(1.0, sum(len(s["deps"]) for s in specs) / n)
    f["max_fan_in_share"] = max(len(s["deps"]) for s in specs) / n
    f["max_fan_out_share"] = max(fan_out.values(), default=0) / n
    f["depth_ratio"] = chain / n
    f["log_depth"] = math.log1p(chain)
    f["width_ratio"] = width / n
    f["root_share"] = sum(not s["deps"] for s in specs) / n

    # timing
    durations = [s["duration"] for s in specs]
    total_work, mean_duration = sum(durations), _mean(durations)
    critical = max(earliest.values(), default=0.0)
    scheduled = all(s["start"] is not None for s in specs)
    planned = (max(s["end"] for s in specs) - min(s["start"] for s in specs)) if scheduled else critical
    planned = max(planned, 1e-6)
    finish = {s["id"]: (s["end"] if scheduled else earliest[s["id"]]) for s in specs}
    f["log_total_work"] = math.log1p(total_work)
    f["log_critical_path"] = math.log1p(critical)
    f["log_parallelism"] = math.log(max(1.0, total_work / critical)) if critical > 0 else 0.0
    f["is_scheduled"] = float(scheduled)
    f["log_planned_makespan"] = math.log1p(planned)
    f["schedule_efficiency"] = clamp(critical / planned, 0.0, 1.0) if scheduled else 1.0
    f["log_mean_duration"] = math.log1p(mean_duration)
    f["duration_cv"] = min(5.0, math.sqrt(_mean([(d - mean_duration) ** 2 for d in durations])) / mean_duration) if mean_duration > 0 else 0.0
    f["max_duration_share"] = max(durations) / total_work if total_work > 0 else 0.0

    # deadlines
    margins = {**_SAFETY_DEFAULTS, **{k: _get(safety_margins, k, v) for k, v in _SAFETY_DEFAULTS.items()
                                      if _get(safety_margins, k) is not None}}
    bound = [s for s in specs if s["deadline"] > 0]
    slacks = [s["deadline"] - finish[s["id"]] for s in bound]
    f["deadline_share"] = len(bound) / n
    f["deadline_slack"] = clamp(min(slacks) / planned, -1.0, 2.0) if slacks else 2.0
    f["late_share"] = sum(x < 0 for x in slacks) / n
    f["tight_deadline_share"] = sum(0 <= x < 0.25 * s["duration"] for x, s in zip(slacks, bound)) / n
    f["time_buffer_ratio"] = clamp(_finite(margins["time_buffer"]) / planned, 0.0, 5.0)

    # resources
    gpu_total = max(0.0, _finite(_get(resources, "gpu_total", _get(resources, "gpu", 0.0))))
    ram_total = max(0.0, _finite(_get(resources, "ram_total", _get(resources, "ram", 0.0))))
    known = resources is not None and (gpu_total > 0 or ram_total > 0)
    if scheduled:
        peak_gpu, peak_ram, peak_count = _peaks([(s["start"], s["end"], s["gpu"], s["ram"]) for s in specs])
    else:  # no timeline yet: the widest dependency level bounds what can run together
        per_level: Dict[int, List[Mapping[str, Any]]] = {}
        for s in specs:
            per_level.setdefault(level[s["id"]], []).append(s)
        peak_gpu = max(sum(s["gpu"] for s in group) for group in per_level.values())
        peak_ram = max(sum(s["ram"] for s in group) for group in per_level.values())
        peak_count = width

    def ratio(peak: float, total: float) -> float:
        return clamp(peak / total, 0.0, 3.0) if total > 0 else (3.0 if peak > 0 else 0.0)

    f["resources_known"] = float(known)
    f["log_gpu_total"] = math.log1p(sum(s["gpu"] for s in specs))
    f["log_ram_total"] = math.log1p(sum(s["ram"] for s in specs))
    f["gpu_peak_ratio"] = ratio(peak_gpu, gpu_total) if known else 0.0
    f["ram_peak_ratio"] = ratio(peak_ram, ram_total) if known else 0.0
    f["gpu_margin"] = compute_resource_margin(peak_gpu, gpu_total, safety_buffer=_finite(margins["gpu_buffer"])) if known else 1.0
    f["ram_margin"] = compute_resource_margin(peak_ram, ram_total, safety_buffer=_finite(margins["ram_buffer"])) if known else 1.0
    f["concurrency_peak_ratio"] = clamp(peak_count / max(1.0, _finite(margins["max_concurrent"], 5.0)), 0.0, 3.0)
    f["specialized_hw_share"] = sum(s["hw"] > 0 for s in specs) / n
    f["resource_heavy_share"] = sum(s["gpu"] > 0 or s["ram"] > 0 for s in specs) / n

    # task risk and reliability
    risks = [s["risk"] for s in specs]
    f["risk_mean"], f["risk_max"] = _mean(risks), max(risks)
    f["high_risk_share"] = sum(r > 0.7 for r in risks) / n
    f["criticality_mean"] = _mean([s["crit"] / 3.0 for s in specs])
    f["critical_share"] = sum(s["crit"] >= 2 for s in specs) / n
    f["retry_budget_mean"] = _mean([min(s["retries"], 5) / 5.0 for s in specs])
    f["zero_retry_share"] = sum(s["retries"] == 0 for s in specs) / n
    rates = [r for r in (_success_rate((task_history or {}).get(s["name"])) for s in specs) if r is not None]
    f["hist_task_known_share"] = len(rates) / n
    f["hist_task_success_mean"] = _mean(rates)
    f["hist_task_success_min"] = min(rates, default=0.0)
    f["hist_log_hazard"] = math.log1p(sum(-math.log(max(r, 0.05)) for r in rates))

    # agents and environment
    used = Counter(s["agent"] for s in specs if s["agent"])
    details = [d for d in (_agent_detail(agents, a) for a in used) if d is not None]
    f["agents_known"] = float(bool(details))
    f["log_agents_used"] = math.log1p(len(used))
    f["agent_balance"] = 1.0 - max(used.values()) / n if used else 0.0
    if details:
        reliability = [_finite(_get(d, "successes", 1)) / max(1.0, _finite(_get(d, "successes", 1)) + _finite(_get(d, "failures", 0)))
                       for d in details]
        f["agent_reliability_mean"], f["agent_reliability_min"] = _mean(reliability), min(reliability)
        f["agent_load_mean"] = _mean([clamp(_finite(_get(d, "current_load", 0.0)), 0.0, 2.0) for d in details])
        f["agent_efficiency_mean"] = _mean([clamp(_finite(_get(d, "efficiency", 1.0), 1.0), 0.0, 4.0) for d in details])
    f["system_load"] = clamp(_finite(system_load), 0.0, 1.0)

    # plan-level history
    summary = outcome_summary or {}
    count = max(0.0, _finite(summary.get("count")))
    f["plan_hist_known"] = float(count > 0)
    f["plan_hist_log_count"] = math.log1p(count)
    f["plan_hist_success_rate"] = clamp(_finite(summary.get("success_rate")), 0.0, 1.0) if count > 0 else 0.0
    f["plan_hist_replan_rate"] = clamp(_finite(summary.get("replan_rate")), 0.0, 1.0) if count > 0 else 0.0
    f["plan_hist_mean_log_ratio"] = clamp(_finite(summary.get("mean_log_ratio")), -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)

    # other
    f["probabilistic_share"] = sum(s["probabilistic"] for s in specs) / n
    f["human_share"] = sum(s["human"] for s in specs) / n
    f["safety_constraint_share"] = sum(s["safety"] for s in specs) / n
    f["parallelizable_share"] = sum(s["parallelizable"] for s in specs) / n
    f["priority_mean"] = clamp(_mean([s["priority"] for s in specs]) / 10.0, 0.0, 1.0)

    unknown = set(f) - set(FEATURES)
    if unknown:  # a programming error, not a data problem
        raise RuntimeError("PlanSuccessPredictor produced unregistered features: " + ", ".join(sorted(unknown)))
    return {name: _finite(f.get(name, 0.0)) for name in FEATURES}


# =============================================================================
# Observed outcome: the single definition used by trainer, agent hook and history
# =============================================================================
def labels_from_outcome(outcome: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Targets from an execution outcome; ``None`` marks a target that cannot be observed for this plan.

    Outcome keys (all optional except ``status``):
        status                     final status; ``SUCCESS`` means the plan reached its goal
        replans                    number of replanning / recovery episodes
        resource_events            resource waits, contention or violations
        deadline_misses            tasks that finished after their deadline
        failure_causes             subset of {"resource", "temporal", "task"} that ended a failed plan
        makespan_seconds           realised makespan (only when real task durations were measured)
        planned_makespan_seconds   makespan predicted before execution
    ``cause_*`` are defined for failed plans only; ``duration_log_ratio`` for completed plans only.
    """
    success = str(outcome.get("status", "")).strip().upper() == "SUCCESS"
    out: Dict[str, Optional[float]] = {t: None for t in TARGETS}
    out["success"] = float(success)

    def flag(key: str) -> Optional[float]:
        value = outcome.get(key)
        return float(_finite(value) > 0) if _is_number(value) else None

    out["replan"], out["resource_risk"], out["deadline_miss"] = flag("replans"), flag("resource_events"), flag("deadline_misses")
    causes = outcome.get("failure_causes")
    if not success and isinstance(causes, (list, tuple, set)) and causes:
        seen = {str(c).lower() for c in causes}
        out.update({"cause_" + c: float(c in seen) for c in CAUSES})
    makespan, planned = outcome.get("makespan_seconds"), outcome.get("planned_makespan_seconds")
    if success and _is_number(makespan) and _is_number(planned) and makespan > 0 and planned > 0:
        out["duration_log_ratio"] = clamp(math.log(makespan / planned), -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)
    return out


class OutcomeHistory:
    """Bounded per-family record of recent observed outcomes (the plan-level history features)."""

    def __init__(self, limit: int = 20, max_families: int = 256) -> None:
        self.limit, self.max_families = max(1, int(limit)), max(1, int(max_families))
        self._families: "OrderedDict[str, Deque[Dict[str, Optional[float]]]]" = OrderedDict()
        self._lock = threading.RLock()

    def add(self, family: str, labels: Mapping[str, Optional[float]]) -> None:
        with self._lock:
            records = self._families.setdefault(family, deque(maxlen=self.limit))
            self._families.move_to_end(family)
            records.append({k: labels.get(k) for k in ("success", "replan", "duration_log_ratio")})
            while len(self._families) > self.max_families:
                self._families.popitem(last=False)

    def summary(self, family: str) -> Dict[str, float]:
        with self._lock:
            records = list(self._families.get(family, ()))
        if not records:
            return {"count": 0.0}
        rate = lambda key: _mean([r[key] for r in records if r[key] is not None])  # noqa: E731
        ratios = [r["duration_log_ratio"] for r in records if r["duration_log_ratio"] is not None]
        return {"count": float(len(records)), "success_rate": rate("success"), "replan_rate": rate("replan"),
                "mean_log_ratio": _mean(ratios)}


# =============================================================================
# Rows: one executed plan with its observed outcome (used by the trainer and the agent log)
# =============================================================================
def validate_row(row: Mapping[str, Any], where: str = "row") -> None:
    if row.get("schema") != ROW_SCHEMA or row.get("feature_schema") != FEATURE_SCHEMA_ID:
        raise ValueError(f"{where}: incompatible row or feature schema (re-collect with the current PlanSuccessPredictor)")
    if not str(row.get("group") or "").strip() or not str(row.get("target_source") or "").strip():
        raise ValueError(f"{where}: group and target_source are required")
    x = row.get("x")
    if not isinstance(x, list) or len(x) != NF or not all(_is_number(v) for v in x):
        raise ValueError(f"{where}: invalid feature vector")
    targets = row.get("targets")
    if not isinstance(targets, dict) or "success" not in targets:
        raise ValueError(f"{where}: targets must include success")
    for name, value in targets.items():
        if name not in TARGETS or not _is_number(value):
            raise ValueError(f"{where}: unknown or non-numeric target {name!r}")
        if name in CLASSIFIERS and value not in (0, 1):
            raise ValueError(f"{where}: classifier target {name!r} must be 0 or 1")
    if targets["success"] == 1 and any(targets.get("cause_" + c) for c in CAUSES):
        raise ValueError(f"{where}: a successful plan has no failure cause")


def make_row(features: Mapping[str, float], outcome: Mapping[str, Any], *, family: str, plan_id: str, order: int,
             target_source: str, goal_id: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    labels = labels_from_outcome(outcome)
    row = {"schema": ROW_SCHEMA, "feature_schema": FEATURE_SCHEMA_ID,
           "id": hashlib.sha256(f"{family}|{plan_id}".encode()).hexdigest()[:24],
           "group": family_group(family), "family": family, "plan_id": plan_id, "goal_id": goal_id or plan_id,
           "order": int(order), "x": [round(float(features[name]), 6) for name in FEATURES],
           "targets": {t: round(float(v), 6) for t, v in labels.items() if v is not None},
           "target_source": target_source}
    row.update(extra)
    validate_row(row)
    return row


# =============================================================================
# Model: constrained JSON checkpoint, pure-Python inference
# =============================================================================
def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class PlanSuccessModel:
    """Immutable inference over a validated checkpoint (boosted-tree ensembles + calibration).

    Tree layout (arrays indexed by node, root = 0): ``f`` feature index (-1 = leaf), ``t`` threshold (``x <= t``
    goes left), ``l`` index of the left child (the right child is ``l + 1``), ``v`` node value (learning rate already
    applied). Internal ``v`` is the value the node would have as a leaf and is used for feature attribution.
    """

    def __init__(self, manifest_path: str | Path, *, allow_experimental: bool = False) -> None:
        path = Path(manifest_path)
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("PlanSuccessPredictor checkpoint exceeds the size limit")
        raw = path.read_bytes()
        self._init(json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest(), allow_experimental)

    @classmethod
    def from_model(cls, model: Mapping[str, Any], *, allow_experimental: bool = True) -> "PlanSuccessModel":
        """Build from an in-memory checkpoint (trainer parity checks, tests)."""
        obj = cls.__new__(cls)
        raw = json.dumps(model, sort_keys=True, allow_nan=False).encode("utf-8")
        obj._init(json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest(), allow_experimental)
        return obj

    def _init(self, model: Dict[str, Any], checksum: str, allow_experimental: bool) -> None:
        if model.get("schema") != MODEL_SCHEMA:
            raise ValueError("Incompatible PlanSuccessPredictor schema")
        if tuple(model.get("features", ())) != FEATURES or tuple(model.get("targets", ())) != TARGETS:
            raise ValueError("Incompatible PlanSuccessPredictor feature or target order")
        promotion = model.get("promotion")
        if promotion not in ("experimental", "approved"):
            raise ValueError("Unknown PlanSuccessPredictor promotion state")
        if promotion != "approved" and not allow_experimental:
            raise ValueError("PlanSuccessPredictor checkpoint is not production-approved")
        domain = model.get("domain", {})
        for key in ("min", "max"):
            if len(domain.get(key, ())) != NF or not all(_is_finite_number(v) for v in domain[key]):
                raise ValueError("Invalid PlanSuccessPredictor domain " + key)
        heads = model.get("heads") or {}
        if set(heads) != set(TARGETS):
            raise ValueError("PlanSuccessPredictor checkpoint must define exactly the registered heads")
        self._heads: Dict[str, Dict[str, Any]] = {}
        for target in TARGETS:
            self._heads[target] = self._load_head(target, heads[target])
        selection = model.get("selection") or {}
        for key in ("risky_success_below", "safe_success_at", "safe_replan_below", "max_uncertainty", "max_ood_fraction"):
            if not _is_finite_number(selection.get(key)) or not 0.0 <= selection[key] <= 1.0:
                raise ValueError("Invalid PlanSuccessPredictor selection value: " + key)
        if selection["risky_success_below"] > selection["safe_success_at"]:
            raise ValueError("PlanSuccessPredictor risky threshold exceeds the safe threshold")
        self.model, self.checksum = model, checksum
        self.version, self.promotion = str(model.get("version", "unknown")), str(promotion)
        self.selection = dict(selection)

    @staticmethod
    def _load_head(target: str, head: Mapping[str, Any]) -> Dict[str, Any]:
        kind = "regression" if target in REGRESSORS else "binary"
        if head.get("kind") != kind or not _is_finite_number(head.get("base")):
            raise ValueError("Invalid PlanSuccessPredictor head: " + target)
        members = []
        for member in head.get("members") or []:
            trees = []
            for tree in member:
                f, t, l, v = tree.get("f"), tree.get("t"), tree.get("l"), tree.get("v")
                size = len(f or ())
                if not size or not (len(t) == len(l) == len(v) == size):
                    raise ValueError("Invalid PlanSuccessPredictor tree arrays: " + target)
                for i in range(size):
                    if not (_is_finite_number(t[i]) and _is_finite_number(v[i])):
                        raise ValueError("Invalid PlanSuccessPredictor tree values: " + target)
                    if f[i] >= 0 and not (f[i] < NF and i < l[i] and l[i] + 1 < size):
                        raise ValueError("Invalid PlanSuccessPredictor tree structure: " + target)
                trees.append((f, t, l, v))
            members.append(trees)
        if not members or not all(members):
            raise ValueError("PlanSuccessPredictor head has no ensemble members: " + target)
        loaded: Dict[str, Any] = {"kind": kind, "base": float(head["base"]), "members": members}
        if kind == "binary":
            calibration = head.get("calibration")
            if not (isinstance(calibration, list) and len(calibration) == 2 and all(_is_finite_number(c) for c in calibration)):
                raise ValueError("Invalid PlanSuccessPredictor calibration: " + target)
            loaded["calibration"] = (float(calibration[0]), float(calibration[1]))
        else:
            interval = head.get("interval") or {}
            if not all(_is_finite_number(interval.get(k)) for k in ("low", "high")):
                raise ValueError("Invalid PlanSuccessPredictor duration interval")
            loaded["interval"] = (float(interval["low"]), float(interval["high"]))
        return loaded

    # -- numerics ---------------------------------------------------------------------------------
    def vector(self, features: Mapping[str, float]) -> List[float]:
        x = []
        for name in FEATURES:
            value = float(features.get(name, 0.0))
            if not math.isfinite(value):
                raise ValueError("Non-finite PlanSuccessPredictor input feature: " + name)
            x.append(value)
        return x

    @staticmethod
    def _tree(tree: Tuple[Any, ...], x: Sequence[float]) -> float:
        f, t, l, v = tree
        i = 0
        while f[i] >= 0:
            i = l[i] + (0 if x[f[i]] <= t[i] else 1)
        return v[i]

    def _member_scores(self, target: str, x: Sequence[float]) -> List[float]:
        head = self._heads[target]
        return [head["base"] + sum(self._tree(tree, x) for tree in member) for member in head["members"]]

    def ood_fraction(self, x: Sequence[float]) -> float:
        lows, highs = self.model["domain"]["min"], self.model["domain"]["max"]
        outside = 0
        for value, lo, hi in zip(x, lows, highs):
            margin = 0.1 * (hi - lo) + 1e-9
            outside += value < lo - margin or value > hi + margin
        return outside / NF

    def score(self, features: Mapping[str, float]) -> Dict[str, Any]:
        """Calibrated probabilities, duration log-ratio, ensemble disagreement and out-of-distribution share."""
        x = self.vector(features)
        probabilities, spread = {}, []
        for target in CLASSIFIERS:
            scores = self._member_scores(target, x)
            a, b = self._heads[target]["calibration"]
            probabilities[target] = _sigmoid(a * _mean(scores) + b)
            if target in UNCERTAINTY_HEADS and len(scores) > 1:
                member = [_sigmoid(s) for s in scores]
                centre = _mean(member)
                spread.append(math.sqrt(_mean([(p - centre) ** 2 for p in member])))
        duration = self._member_scores("duration_log_ratio", x)
        return {"probabilities": probabilities, "duration_log_ratio": _mean(duration),
                "uncertainty": _mean(spread), "ood_fraction": self.ood_fraction(x)}

    def explain(self, features: Mapping[str, float], target: str = "success", top: int = 3) -> List[Dict[str, Any]]:
        """Largest per-feature contributions (tree-path attribution, logit space). Positive raises the target."""
        x = self.vector(features)
        contribution = [0.0] * NF
        members = self._heads[target]["members"]
        for member in members:
            for f, t, l, v in member:
                i = 0
                while f[i] >= 0:
                    child = l[i] + (0 if x[f[i]] <= t[i] else 1)
                    contribution[f[i]] += v[child] - v[i]
                    i = child
        scale = self._heads[target].get("calibration", (1.0, 0.0))[0] / len(members)
        ranked = sorted(range(NF), key=lambda k: -abs(contribution[k]))[:max(0, top)]
        return [{"feature": FEATURES[k], "value": round(x[k], 6), "contribution": round(contribution[k] * scale, 6)}
                for k in ranked if contribution[k]]

    def assess(self, features: Mapping[str, float], *, top_drivers: int = 3,
               scored: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Risk tier and recommendation. Advisory only: it never validates, repairs or authorises a plan.

        ``scored`` is a previous ``score(features)`` result, to avoid scoring the same plan twice.
        """
        scored = scored if scored is not None else self.score(features)
        probs, sel = scored["probabilities"], self.selection
        p_success, p_replan = probs["success"], probs["replan"]
        if p_success < sel["risky_success_below"]:
            tier = "high"
        elif p_success >= sel["safe_success_at"] and p_replan <= sel["safe_replan_below"]:
            tier = "low"
        else:
            tier = "elevated"
        reliable = scored["uncertainty"] <= sel["max_uncertainty"] and scored["ood_fraction"] <= sel["max_ood_fraction"]
        recommendation = ("defer_to_heuristics" if not reliable
                          else {"low": "proceed", "elevated": "review", "high": "prefer_alternative"}[tier])
        planned = math.expm1(float(features.get("log_planned_makespan", 0.0)))
        low, high = self._heads["duration_log_ratio"]["interval"]
        centre = scored["duration_log_ratio"]
        causes = {c: round(probs["cause_" + c], 6) for c in CAUSES}
        return {
            "model_version": self.version, "model_sha256": self.checksum, "promotion": self.promotion,
            "feature_schema": FEATURE_SCHEMA_ID,
            "probabilities": {t: round(p, 6) for t, p in probs.items()},
            "success_probability": round(p_success, 6), "failure_probability": round(1.0 - p_success, 6),
            "replan_probability": round(p_replan, 6), "resource_risk_probability": round(probs["resource_risk"], 6),
            "deadline_miss_probability": round(probs["deadline_miss"], 6),
            "failure_causes": causes, "dominant_failure_cause": max(causes, key=causes.get),
            "duration": {"planned_seconds": round(planned, 3), "ratio": round(math.exp(centre), 6),
                         "expected_seconds": round(planned * math.exp(centre), 3),
                         "low_seconds": round(planned * math.exp(centre + low), 3),
                         "high_seconds": round(planned * math.exp(centre + high), 3)},
            "uncertainty": round(scored["uncertainty"], 6), "ood_fraction": round(scored["ood_fraction"], 6),
            "reliable": bool(reliable), "risk_tier": tier,
            "advice": {"recommendation": recommendation, "provisional": self.promotion != "approved",
                       "drivers": self.explain(features, "success", top_drivers) if top_drivers > 0 else [],
                       "authority": "advisory_only", "safety_checks_unchanged": True},
        }


def plan_utility(assessment: Mapping[str, Any], weights: Mapping[str, float], reference_seconds: float) -> float:
    """Single scalar used to order candidate plans: reward success, penalise recovery, resource risk and slowness."""
    expected = max(_finite(assessment["duration"]["expected_seconds"]), 1e-6)
    return (weights["success"] * assessment["success_probability"]
            - weights["replan"] * assessment["replan_probability"]
            - weights["resource"] * assessment["resource_risk_probability"]
            - weights["duration"] * math.log(expected / max(reference_seconds, 1e-6)))


# =============================================================================
# Planning Agent module
# =============================================================================
class PlanSuccessPredictor:
    """Configuration-injected, fail-open predictor owned by the Planning Agent."""

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("plan_success_predictor config must be a mapping")
        section = dict(_DEFAULTS)
        section.update(dict(config) if config is not None else get_config_section(CONFIG_SECTION, default={}))
        self.config = section
        self.enabled = bool(section["enabled"])
        self.mode = str(section["mode"]).strip().lower()
        if self.mode not in MODES:
            raise ValueError("plan_success_predictor.mode must be one of: " + ", ".join(MODES))
        for key in ("max_tasks", "history_limit", "max_families", "top_drivers", "max_log_bytes"):
            if isinstance(section[key], bool) or not isinstance(section[key], int) or section[key] < 1:
                raise ValueError("plan_success_predictor.%s must be a positive integer" % key)
        weights = {**_DEFAULTS["ranking_weights"], **dict(section["ranking_weights"] or {})}
        if not all(_is_finite_number(weights[k]) and weights[k] >= 0 for k in _DEFAULTS["ranking_weights"]):
            raise ValueError("plan_success_predictor.ranking_weights must be non-negative numbers")
        self.ranking_weights = {k: float(weights[k]) for k in _DEFAULTS["ranking_weights"]}
        self.record_features = bool(section["record_features"]) or self.mode == "collect"
        self.record_observations = bool(section["record_observations"])
        # Advisory use of an unapproved model is never allowed.
        self.allow_experimental = bool(section["allow_experimental"]) and self.mode != "advisory"
        self.model_path = self._resolve(section["model_path"])
        self.observation_log = self._resolve(section["observation_log"])
        self.history = OutcomeHistory(section["history_limit"], section["max_families"])
        self._model: Optional[PlanSuccessModel] = None
        self._load_failed = False
        self._log_full_warned = False
        self._lock = threading.RLock()
        if self.enabled:
            printer.status("INIT", "Initializing Plan Success Predictor (%s)" % self.mode, "info")

    @staticmethod
    def _resolve(value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else PROJECT_ROOT / path

    # -- model ------------------------------------------------------------------------------------
    def _get_model(self) -> Optional[PlanSuccessModel]:
        if self._model is not None or self._load_failed:
            return self._model
        with self._lock:
            if self._model is None and not self._load_failed:
                try:
                    self._model = PlanSuccessModel(self.model_path, allow_experimental=self.allow_experimental)
                    logger.info("PlanSuccessPredictor loaded | version=%s | promotion=%s", self._model.version, self._model.promotion)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logger.warning("PlanSuccessPredictor unavailable; planning unchanged (%s: %s)", type(exc).__name__, exc)
                    self._load_failed = True
            return self._model

    # -- public API -------------------------------------------------------------------------------
    def predict_plan(
        self,
        plan: Iterable[Any],
        *,
        family: Optional[str] = None,
        plan_id: Optional[str] = None,
        agents: Optional[Mapping[str, Any]] = None,
        resources: Any = None,
        safety_margins: Any = None,
        system_load: Optional[float] = None,
        task_history: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pre-execution prediction for one scheduled plan; ``{}`` when disabled or on any failure."""
        if not self.enabled:
            return {}
        try:
            family = str(family or "default")
            features = extract_features(
                list(plan), agents=agents, resources=resources, safety_margins=safety_margins, system_load=system_load,
                task_history=task_history, outcome_summary=self.history.summary(family), max_tasks=self.config["max_tasks"])
            result: Dict[str, Any] = {"mode": self.mode, "authority": "advisory_only", "feature_schema": FEATURE_SCHEMA_ID,
                                      "family": family, "plan_id": plan_id or uuid.uuid4().hex[:16]}
            if self.record_features:
                result["features"] = {k: round(v, 6) for k, v in features.items()}
            if self.mode == "collect":
                return result
            model = self._get_model()
            if model is None:
                result["status"] = "model_unavailable"
                return result
            result.update(model.assess(features, top_drivers=self.config["top_drivers"]))
            return result
        except Exception as exc:  # noqa: BLE001 - an advisory component must never break planning
            logger.warning("PlanSuccessPredictor inference failed; planning unchanged (%s: %s)", type(exc).__name__, exc)
            return {}

    def rank_plans(self, candidates: Sequence[Iterable[Any]], **context: Any) -> List[Dict[str, Any]]:
        """Order candidate plans (same goal) by utility, best first. Never drops a candidate.

        Returns ``[{"index", "utility", "prediction"}]``. Without a usable model the original order is kept and
        ``utility`` is ``None``; the caller's existing selection logic then applies unchanged.
        """
        predictions = [self.predict_plan(plan, **context) for plan in candidates]
        if not predictions or any("success_probability" not in p for p in predictions):
            return [{"index": i, "utility": None, "prediction": p} for i, p in enumerate(predictions)]
        reference = min(max(_finite(p["duration"]["expected_seconds"]), 1e-6) for p in predictions)
        ranked = [{"index": i, "utility": round(plan_utility(p, self.ranking_weights, reference), 6), "prediction": p}
                  for i, p in enumerate(predictions)]
        return sorted(ranked, key=lambda item: (-item["utility"], item["index"]))

    def observe(self, prediction: Mapping[str, Any], outcome: Mapping[str, Any], *, family: Optional[str] = None,
                plan_id: Optional[str] = None, goal_id: Optional[str] = None) -> Dict[str, Any]:
        """Attach the real outcome to a prediction: history, shadow-mode error and a retraining row."""
        if not prediction:
            return {}
        try:
            family = str(family or prediction.get("family") or "default")
            labels = labels_from_outcome(outcome)
            self.history.add(family, labels)
            out: Dict[str, Any] = {"observed": labels}
            probabilities = prediction.get("probabilities")
            if isinstance(probabilities, Mapping):
                out["absolute_error"] = {t: round(abs(float(probabilities[t]) - y), 6)
                                         for t, y in labels.items() if y is not None and t in probabilities}
            features = prediction.get("features")
            if self.record_observations and isinstance(features, Mapping):
                row = make_row(features, outcome, family=family, plan_id=str(plan_id or prediction.get("plan_id") or uuid.uuid4().hex[:16]),
                               order=int(time.time() * 1000), target_source=OBSERVED, goal_id=goal_id,
                               source_name="planning_agent", source_license="internal")
                out["logged"] = self._append(row)
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("PlanSuccessPredictor could not record the outcome (%s: %s)", type(exc).__name__, exc)
            return {}

    def _append(self, row: Mapping[str, Any]) -> bool:
        with self._lock:
            try:
                path = self.observation_log
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.stat().st_size >= self.config["max_log_bytes"]:
                    if not self._log_full_warned:
                        logger.warning("PSP observation log is full (%s); run train_psp.py collect --from-log and rotate it", path)
                        self._log_full_warned = True
                    return False
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                return True
            except OSError as exc:
                logger.warning("PSP observation log write failed (%s)", exc)
                return False

    def status(self) -> Dict[str, Any]:
        model = self._model
        return {"enabled": self.enabled, "mode": self.mode, "model_path": str(self.model_path),
                "model_loaded": model is not None, "model_load_failed": self._load_failed,
                "model_version": model.version if model else None, "promotion": model.promotion if model else None,
                "observation_log": str(self.observation_log), "feature_schema": FEATURE_SCHEMA_ID}


__all__ = ["PlanSuccessModel", "PlanSuccessPredictor", "OutcomeHistory", "extract_features", "labels_from_outcome",
           "merge_schedule", "normalize_plan", "plan_utility", "task_history_from_method_stats"]
