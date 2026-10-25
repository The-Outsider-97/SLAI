"""SLAI Orchestrator TaskRoutingPolicy (TRP): offline, read-only route predictions.

TRP answers one question for the Orchestrator: *among the routes this task is
already authorized to use, which is the best?* It never executes a task, never
grants a route, and never invents a cache hit. The Orchestrator computes the
eligible set (see :func:`eligible_routes`); TRP only ranks that set.

Safety properties enforced here (and re-checked by ``train_trp.py smoke``):

* the recommended route is always a member of the caller's eligible set;
* any uncertainty (unseen task type, out-of-support payload, unscored route,
  gain below the validated threshold) returns the deterministic baseline route;
* a valid cache hit short-circuits to ``cache`` (an exact valid entry is cheaper and as correct
  as any recomputation; ``train_trp.py evaluate`` verifies this on observed trials), so the learned
  decision is ``inline`` versus ``delegate`` only;
* checkpoints are constrained JSON (never pickle); inference is pure Python,
  performs no I/O, and makes no network requests.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

MODEL_SCHEMA = 1
ROUTES = ("inline", "cache", "delegate")
MODEL_ROUTES = ("inline", "delegate")   # routes whose outcomes are learned; cache is a validated rule
# Mirrors OrchestratorAgent.MICRO_TASK_TYPES / factory.operations.SUPPORTED_OPERATIONS;
# ``train_trp.py doctor`` asserts both stay in sync with the real modules.
MICRO_TASK_TYPES = ("normalize_payload", "sanitize_payload", "validate_schema")
SPECIALIST_OPERATIONS = ("numeric_summary", "project_records")
TASK_TYPES = MICRO_TASK_TYPES + SPECIALIST_OPERATIONS
TARGETS = ("success", "log_latency_ms", "log_peak_kib")
CONTINUOUS = ("log_items", "log_bytes", "log_width", "depth", "string_fraction", "log_max_str_len")
FEATURES = (
    "task_normalize_payload", "task_sanitize_payload", "task_validate_schema",
    "task_numeric_summary", "task_project_records", "task_other",
    *CONTINUOUS,
    "specialist_registered", "ephemeral_delegate", "cpu_utilization",
)
SOURCE_KINDS = ("controlled_trial", "production_telemetry")
# Policy choice, not a learned quantity: relative importance of the three outcomes.
DEFAULT_UTILITY = {"w_success": 0.25, "w_latency": 0.40, "w_cost": 0.35,
                   "latency_ref_ms": 1000.0, "memory_ref_kib": 65536.0}
MAX_ARTIFACT_BYTES = 16_000_000

_SAMPLE = 64         # elements inspected per payload (bounded, deterministic; keeps decisions cheap)
_NODE_BUDGET = 256   # nested containers inspected per payload
_MAX_DEPTH = 6
_HEAD_KINDS = ("constant", "linear", "logistic", "gbt_regression", "gbt_logistic")


# ---------------------------------------------------------------------------
# Decision-time features
# ---------------------------------------------------------------------------
def _stride(seq: Any) -> list:
    n = len(seq)
    step = max(1, n // _SAMPLE)
    return [seq[i] for i in range(0, n, step)][:_SAMPLE]


def _approx(value: Any, depth: int, budget: list) -> tuple[int, int]:
    """Bounded (approximate bytes, nesting depth) of one sampled element."""
    if isinstance(value, (str, bytes, bytearray)):
        return len(value) + 8, depth
    if isinstance(value, (Mapping, list, tuple)):
        if depth >= _MAX_DEPTH or budget[0] <= 0:
            return 8 * len(value), depth
        children = list(islice(value.values() if isinstance(value, Mapping) else value, 32))
        total, deepest = 8, depth + 1
        for child in children:
            budget[0] -= 1
            size, d = _approx(child, depth + 1, budget)
            total += size
            deepest = max(deepest, d)
        if children and len(value) > len(children):
            total = 8 + (total - 8) * len(value) / len(children)
        return int(total), deepest
    return 8, depth


def _measure(root: Any) -> tuple[int, int, int, float, int]:
    """(items, approx_bytes, depth, string_fraction, max_string_length); O(sample)."""
    if isinstance(root, Mapping):
        items, sample = len(root), list(islice(root.values(), _SAMPLE))
    elif isinstance(root, (list, tuple)):
        items, sample = len(root), _stride(root)
    else:
        items, sample = (0 if root is None else 1), [root]
    if not sample:
        return items, 0, 0, 0.0, 0
    budget, total, depth, strings, longest = [_NODE_BUDGET], 0, 0, 0, 0
    for element in sample:
        size, d = _approx(element, 0, budget)
        total += size
        depth = max(depth, d)
        if isinstance(element, str):
            strings += 1
            longest = max(longest, len(element))
    return items, int(total / len(sample) * items), depth, strings / len(sample), longest


def _task_view(task: Any) -> tuple[str, Any, int]:
    """Canonical (task_type, payload root, width) for an Orchestrator task mapping."""
    if not isinstance(task, Mapping):
        return "other", task, 0
    task_type = str(task.get("task_type") or "").strip().lower()
    if task_type in ("normalize_payload", "sanitize_payload"):
        return task_type, task.get("payload"), 0
    if task_type == "validate_schema":
        schema = task.get("schema")
        return task_type, task.get("data"), len(schema) if isinstance(schema, (list, tuple, set)) else 0
    spec = task.get("specialist_spec")
    operation = str(spec.get("operation") or "") if isinstance(spec, Mapping) else ""
    if operation in SPECIALIST_OPERATIONS:
        inner = task.get("task_data")
        while isinstance(inner, Mapping) and "task_data" in inner:
            inner = inner["task_data"]
        key = "values" if operation == "numeric_summary" else "records"
        root = inner.get(key) if isinstance(inner, Mapping) else inner
        fields = spec.get("fields")
        return operation, root, len(fields) if isinstance(fields, (list, tuple)) else 0
    return "other", task, 0


def routing_features(task: Any, *, specialist_registered: bool = False,
                     ephemeral_delegate: bool = False, cpu_utilization: float = 0.0) -> dict[str, float]:
    """Cheap, bounded, deterministic features computed *before* any route executes.

    ``task`` is the mapping later passed to ``OrchestratorAgent.perform_task`` (for
    specialist operations: the delegation payload carrying ``specialist_spec``). The
    same function builds training and serving features, so they cannot drift apart.
    """
    task_type, root, width = _task_view(task)
    items, nbytes, depth, string_fraction, longest = _measure(root)
    out = {name: 0.0 for name in FEATURES}
    out["task_" + (task_type if task_type in TASK_TYPES else "other")] = 1.0
    out["log_items"] = math.log1p(items)
    out["log_bytes"] = math.log1p(nbytes)
    out["log_width"] = math.log1p(width)
    out["depth"] = float(depth)
    out["string_fraction"] = float(string_fraction)
    out["log_max_str_len"] = math.log1p(longest)
    out["specialist_registered"] = float(bool(specialist_registered))
    out["ephemeral_delegate"] = float(bool(ephemeral_delegate))
    out["cpu_utilization"] = min(1.0, max(0.0, float(cpu_utilization)))
    return out


def task_type_of(features: Mapping[str, float]) -> str:
    for name in TASK_TYPES:
        if float(features.get("task_" + name, 0.0)) >= 0.5:
            return name
    return "other"


# ---------------------------------------------------------------------------
# Eligibility and the deterministic baseline (current Orchestrator behaviour)
# ---------------------------------------------------------------------------
def eligible_routes(task_type: str, *, cache_hit: bool = False, specialist_available: bool = False,
                    local_operations: Iterable[str] = ()) -> tuple[str, ...]:
    """Routes the Orchestrator may use. TRP can only choose among these.

    * ``inline``: micro-tasks, plus operations the Orchestrator explicitly authorizes locally;
    * ``cache``: only when an exact, valid entry exists (never inferred by the model);
    * ``delegate``: only for non-micro tasks with an available (or onboardable) specialist.
    """
    routes = []
    if task_type in MICRO_TASK_TYPES or task_type in set(local_operations):
        routes.append("inline")
    if cache_hit:
        routes.append("cache")
    if specialist_available and task_type not in MICRO_TASK_TYPES:
        routes.append("delegate")
    return tuple(routes)


def baseline_route(task_type: str, eligible: Iterable[str]) -> str:
    """The deterministic rule TRP must beat: valid cache, else micro-task inline, else delegate."""
    allowed = [r for r in ROUTES if r in set(eligible)]
    if not allowed:
        raise ValueError("No eligible route")
    if "cache" in allowed:
        return "cache"
    if task_type in MICRO_TASK_TYPES and "inline" in allowed:
        return "inline"
    return "delegate" if "delegate" in allowed else allowed[0]


# ---------------------------------------------------------------------------
# Utility (a decision rule over predicted outcomes, not a learned probability)
# ---------------------------------------------------------------------------
def route_utility(success: float, latency_ms: float, peak_kib: float, cfg: Mapping[str, float]) -> float:
    latency = min(1.0, math.log1p(max(0.0, latency_ms)) / math.log1p(cfg["latency_ref_ms"]))
    cost = min(1.0, math.log1p(max(0.0, peak_kib)) / math.log1p(cfg["memory_ref_kib"]))
    return cfg["w_success"] * success - cfg["w_latency"] * latency - cfg["w_cost"] * cost


def on_time_probability(log_latency: float, sd: float, deadline_ms: Optional[float]) -> float:
    if deadline_ms is None:
        return 1.0
    z = (math.log1p(max(0.0, float(deadline_ms))) - log_latency) / max(sd, 1e-3)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Observation schema shared by controlled collection and production telemetry
# ---------------------------------------------------------------------------
def routing_observation(*, group: str, case: str, route: str, task_type: str, features: Mapping[str, float],
                        eligible: Iterable[str], ok: bool, completed: bool, latency_ms: float,
                        peak_kib: Optional[float] = None, retries: int = 0, deadline_ms: Optional[float] = None,
                        source_kind: str = "controlled_trial", cold: bool = False, stress: int = 0,
                        collection_source: str = "") -> dict[str, Any]:
    """One executed route for one task. ``ok``: the task contract held (independently verified);
    ``completed``: the route returned a successful result (failed-fast runs do not train latency)."""
    return validate_observation({
        "schema": MODEL_SCHEMA, "group": group, "case": case, "route": route, "task_type": task_type,
        "x": [round(float(features.get(k, 0.0)), 8) for k in FEATURES],
        "eligible": [r for r in ROUTES if r in set(eligible)], "ok": int(bool(ok)), "completed": int(bool(completed)),
        "latency_ms": float(latency_ms), "peak_kib": None if peak_kib is None else float(peak_kib),
        "retries": int(retries), "deadline_ms": None if deadline_ms is None else float(deadline_ms),
        "source_kind": source_kind, "cold": int(bool(cold)), "stress": int(stress),
        "collection_source": collection_source})


def validate_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        if row.get("schema") != MODEL_SCHEMA:
            raise ValueError("schema")
        if row["route"] not in ROUTES or row["route"] not in row["eligible"] or row["source_kind"] not in SOURCE_KINDS:
            raise ValueError("route/eligible/source_kind")
        if not str(row["group"]) or not str(row["case"]) or row["task_type"] not in (*TASK_TYPES, "other"):
            raise ValueError("identity/task_type")
        x = row["x"]
        if len(x) != len(FEATURES) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in x):
            raise ValueError("features")
        if row["ok"] not in (0, 1) or row["completed"] not in (0, 1) or int(row["retries"]) < 0:
            raise ValueError("outcome flags")
        for key in ("latency_ms", "peak_kib", "deadline_ms"):
            value = row.get(key)
            if value is not None and not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise ValueError(key)
        if not isinstance(row["latency_ms"], (int, float)):
            raise ValueError("latency_ms")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid routing observation ({type(exc).__name__}: {exc})") from exc
    return dict(row)


# ---------------------------------------------------------------------------
# Model artifact
# ---------------------------------------------------------------------------
def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def _f32(value: float) -> float:
    """sklearn trees compare float32-cast inputs; reproduce that exactly."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _finite(*values: Any) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def validate_head(head: Any) -> None:
    if not isinstance(head, Mapping) or head.get("kind") not in _HEAD_KINDS:
        raise ValueError("TRP head has an unknown kind")
    kind, width = head["kind"], len(FEATURES)
    if kind == "constant":
        ok = _finite(head.get("value"))
    elif kind in ("linear", "logistic"):
        ok = (all(len(head.get(k, ())) == width and _finite(*head[k]) for k in ("coef", "mean", "scale"))
              and _finite(head.get("intercept")) and all(s > 0 for s in head["scale"]))
    else:
        trees = head.get("trees")
        ok = _finite(head.get("base")) and isinstance(trees, list) and bool(trees)
        for tree in trees if ok else ():
            f, t, l, r, v = (tree.get(k) for k in ("f", "t", "l", "r", "v"))
            n = len(f)
            ok = ok and n > 0 and all(len(a) == n for a in (t, l, r, v)) and _finite(*t, *v)
            for i in range(n) if ok else ():
                if f[i] >= 0 and not (f[i] < width and i < l[i] < n and i < r[i] < n):
                    ok = False  # children must follow their parent: guarantees traversal terminates
                    break
    if kind in ("logistic", "gbt_logistic") and not (len(head.get("calibration", ())) == 2 and _finite(*head["calibration"])):
        ok = False
    if not ok or not _finite(head.get("resid_sd", 0.0)) or head.get("resid_sd", 0.0) < 0:
        raise ValueError("TRP head failed numeric validation")


def head_value(head: Mapping[str, Any], x: list[float], x32: Optional[list[float]] = None) -> float:
    kind = head["kind"]
    if kind == "constant":
        return float(head["value"])
    if kind in ("linear", "logistic"):
        z = float(head["intercept"])
        for v, mean, scale, coef in zip(x, head["mean"], head["scale"], head["coef"]):
            z += (v - mean) / scale * coef
    else:
        xs = x32 if x32 is not None else [_f32(v) for v in x]
        z = float(head["base"])
        for tree in head["trees"]:
            f, t, l, r = tree["f"], tree["t"], tree["l"], tree["r"]
            i = 0
            while f[i] >= 0:
                i = l[i] if xs[f[i]] <= t[i] else r[i]
            z += tree["v"][i]
    if kind in ("logistic", "gbt_logistic"):
        a, b = head["calibration"]
        return _sigmoid(a * z + b)
    return z


class TaskRoutingPolicy:
    """Immutable numerical inference from a validated, locally stored TRP manifest."""

    def __init__(self, manifest_path: str | Path, *, allow_experimental: bool = False) -> None:
        p = Path(manifest_path)
        if p.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("TRP artifact exceeds size limit")
        raw = p.read_bytes()
        model = json.loads(raw.decode("utf-8"))
        if model.get("schema") != MODEL_SCHEMA or tuple(model.get("features", ())) != FEATURES:
            raise ValueError("Incompatible TRP schema or feature order")
        if not allow_experimental and model.get("promotion") != "approved":
            raise ValueError("TRP checkpoint is not production-approved")
        routes = model.get("routes")
        if not isinstance(routes, Mapping) or not routes or any(r not in ROUTES for r in routes):
            raise ValueError("TRP artifact has no valid routes")
        for route, heads in routes.items():
            if any(t not in heads for t in TARGETS):
                raise ValueError(f"TRP artifact is missing prediction heads for route '{route}'")
            for target in TARGETS:
                validate_head(heads[target])
            support = model.get("support", {}).get(route, {})
            bounds = support.get("bounds", {})
            if not support.get("task_types") or any(k not in bounds or not _finite(*bounds[k]) for k in CONTINUOUS):
                raise ValueError(f"TRP artifact has invalid support for route '{route}'")
        utility, policy = model.get("utility", {}), model.get("policy", {})
        if (set(utility) != set(DEFAULT_UTILITY) or not _finite(*utility.values()) or min(utility.values()) < 0
                or not _finite(policy.get("min_gain"), policy.get("support_margin")) or policy["min_gain"] < 0):
            raise ValueError("TRP utility/policy parameters are invalid")
        self.model = model
        self.checksum = hashlib.sha256(raw).hexdigest()
        self.version = str(model.get("version", "unknown"))
        self.routes = tuple(r for r in ROUTES if r in routes)

    def _vector(self, features: Mapping[str, float]) -> list[float]:
        x = [float(features.get(k, 0.0)) for k in FEATURES]
        if not all(math.isfinite(v) for v in x):
            raise ValueError("Nonfinite TRP input feature")
        return x

    def unscored_reason(self, route: str, features: Mapping[str, float]) -> Optional[str]:
        """Why this route cannot be scored for these features (None when it can)."""
        if route not in self.model["routes"]:
            return "route_not_trained"
        support = self.model["support"][route]
        if task_type_of(features) not in support["task_types"]:
            return "task_type_unseen"
        margin = float(self.model["policy"]["support_margin"])
        for name in CONTINUOUS:
            lo, hi = support["bounds"][name]
            pad = margin * (hi - lo) + 1e-6
            if not lo - pad <= float(features.get(name, 0.0)) <= hi + pad:
                return "out_of_support"
        return None

    def predict_route(self, route: str, features: Mapping[str, float], deadline_ms: Optional[float] = None) -> dict[str, float]:
        x, heads = self._vector(features), self.model["routes"][route]
        x32 = [_f32(v) for v in x]   # shared by every tree head of this route
        success = min(1.0, max(0.0, head_value(heads["success"], x, x32)))
        log_latency = min(15.0, max(0.0, head_value(heads["log_latency_ms"], x, x32)))
        log_memory = min(15.0, max(0.0, head_value(heads["log_peak_kib"], x, x32)))
        on_time = on_time_probability(log_latency, float(heads["log_latency_ms"].get("resid_sd", 0.0)), deadline_ms)
        latency, memory = math.expm1(log_latency), math.expm1(log_memory)
        return {"success_probability": success, "on_time_probability": on_time,
                "expected_latency_ms": latency, "expected_peak_kib": memory,
                "utility": route_utility(success * on_time, latency, memory, self.model["utility"])}

    def recommend(self, features: Mapping[str, float], eligible: Iterable[str], *,
                  deadline_ms: Optional[float] = None, min_gain: Optional[float] = None) -> dict[str, Any]:
        """Rank the caller's eligible routes; fall back to the deterministic baseline when unsure."""
        allowed = [r for r in ROUTES if r in set(eligible)]
        if not allowed:
            raise ValueError("No eligible route")
        task_type = task_type_of(features)
        baseline = baseline_route(task_type, allowed)
        result: dict[str, Any] = {"route": baseline, "baseline_route": baseline, "source": "baseline",
                                  "model_version": self.version, "predictions": {}, "unscored": {}}
        if len(allowed) == 1:
            result["reason"] = "single_eligible_route"
            return result
        if "cache" in allowed:
            result["reason"] = "valid_cache_hit"
            return result
        for route in allowed:
            reason = self.unscored_reason(route, features)
            if reason:
                result["unscored"][route] = reason
            else:
                result["predictions"][route] = self.predict_route(route, features, deadline_ms)
        if baseline not in result["predictions"] or len(result["predictions"]) < 2:
            result["reason"] = "baseline_or_alternatives_unscored"
            return result
        utilities = {r: p["utility"] for r, p in result["predictions"].items()}
        best = max((r for r in ROUTES if r in utilities), key=lambda r: (utilities[r], r == baseline))
        gain = utilities[best] - utilities[baseline]
        threshold = float(self.model["policy"]["min_gain"] if min_gain is None else min_gain)
        result["predicted_gain"] = gain
        if best != baseline and gain >= threshold:
            result.update(route=best, source="model", reason="model_preferred")
        else:
            result["reason"] = "model_agrees" if best == baseline else "gain_below_threshold"
        return result


__all__ = [
    "MODEL_SCHEMA", "ROUTES", "MODEL_ROUTES", "MICRO_TASK_TYPES", "SPECIALIST_OPERATIONS", "TASK_TYPES", "TARGETS", "FEATURES",
    "CONTINUOUS", "DEFAULT_UTILITY", "TaskRoutingPolicy", "head_value", "validate_head", "routing_features", "task_type_of", "eligible_routes",
    "baseline_route", "route_utility", "on_time_probability", "routing_observation", "validate_observation",
]
