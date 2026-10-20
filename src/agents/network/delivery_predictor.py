"""SLAI DeliveryPredictor: a bounded, fail-safe predictive controller for Network Agent delivery attempts.

What it does
------------
For every delivery attempt the Network Agent asks two narrow questions:

* *timeout*: how long is it worth waiting for this attempt?  (``plan``)
* *retry*:   is one more attempt likely to succeed?           (``allow_retry``)

It answers from the recent outcome history of the destination endpoint plus attempt context, using
either a trained model (constrained JSON checkpoint, pure-Python inference, no I/O) or a transparent
heuristic when no valid model is loaded.

What it can never do (deterministic overrides, enforced in :meth:`DeliveryPredictor.plan` and
:meth:`DeliveryPredictor.allow_retry`, after the model has spoken)
------------------------------------------------------------------
* It cannot make a request *wait longer* than the caller asked, nor shorten it below a floor.
* It cannot add attempts, change routes, bypass policy, or retry what policy/reliability already refused:
  it can only *veto* a retry that the reliability layer has already allowed, and only on evidence.
* It cannot veto the first attempt, and it cannot act without enough endpoint history.
* Any exception, schema mismatch, non-finite value or repeated failure degrades to the deterministic
  baseline (requested timeout, retry allowed) - never to an error in the delivery path.

Modes: ``off`` (no-op), ``shadow`` (predict and log, never change behaviour; the default), ``active``
(apply the bounded decisions; requires an ``approved`` checkpoint unless ``allow_experimental_active``).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "slai.delivery_predictor/1"
MODES = ("off", "shadow", "active")
ERROR_CLASSES = ("none", "timeout", "connection", "rate_limit", "server", "other")
HISTORY_LEN = 32
RECENT_LEN = 8
MAX_ARTIFACT_BYTES = 4_000_000

FEATURES: Tuple[str, ...] = (
    "attempt_index",          # 0 for the first attempt
    "attempts_left_frac",
    "log_requested_timeout_ms",
    "budget_ratio",           # remaining deadline / requested timeout, clipped to [0, 5]; 5 when no deadline
    "log_payload_bytes",
    "history_fill",           # observed attempts / HISTORY_LEN
    "success_rate_recent",    # smoothed, last RECENT_LEN attempts
    "success_rate_window",    # smoothed, last HISTORY_LEN attempts
    "consecutive_failures",   # / 8, clipped
    "timeout_rate_recent",
    "log_latency_ewma_ms",
    "log_latency_p95_ms",
    "log_latency_median_ms",
    "since_last_failure",     # log1p(seconds)/log1p(600); 1.0 when never failed
    "last_error_timeout",
    "last_error_connection",
    "last_error_rate_limit",
    "last_error_server",
    "is_retry",
    "idempotent",
)
FEATURE_HASH = hashlib.sha256("|".join(FEATURES).encode("utf-8")).hexdigest()[:16]

_LOG600 = math.log1p(600.0)


def classify_error(error: Optional[BaseException]) -> str:
    """Coarse error class from an exception (by class name and attributes; no imports needed)."""
    if error is None:
        return "none"
    names = {cls.__name__ for cls in type(error).__mro__}
    status = getattr(error, "status_code", None)
    if "RateLimitExceededError" in names or status == 429:
        return "rate_limit"
    if "ConnectionTimeoutError" in names or "DeliveryTimeoutError" in names or "TimeoutError" in names or "timeout" in type(error).__name__.lower():
        return "timeout"
    if "NetworkConnectionError" in names or "ConnectionError" in names:
        return "connection"
    if isinstance(status, int) and 500 <= status <= 599:
        return "server"
    if "NetworkTransportError" in names or "SendFailureError" in names or "EndpointDegradedError" in names:
        return "server"
    return "other"


# ---------------------------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DeliveryContext:
    """Everything the predictor may look at for one attempt. Contains no payload content."""

    endpoint: str
    attempt: int = 1
    max_attempts: int = 3
    requested_timeout_ms: Optional[int] = None
    remaining_ms: Optional[float] = None
    payload_bytes: int = 0
    protocol: str = "http"
    idempotent: bool = True
    last_error_class: str = "none"


@dataclass
class DeliveryDecision:
    """Result of one predictor call. ``applied`` is False in shadow/off mode and on every fallback."""

    timeout_ms: Optional[int]
    allow_retry: bool = True
    p_success: Optional[float] = None
    p_retry: Optional[float] = None
    latency_q50_ms: Optional[float] = None
    latency_q95_ms: Optional[float] = None
    latency_q99_ms: Optional[float] = None
    source: str = "disabled"         # model | heuristic | disabled | fallback:<reason>
    mode: str = "off"
    applied: bool = False
    changed_timeout: bool = False
    vetoed_retry: bool = False
    reasons: List[str] = field(default_factory=list)
    features: Optional[List[float]] = None
    requested_timeout_ms: Optional[int] = None
    overhead_us: float = 0.0

    def to_dict(self, include_features: bool = False) -> Dict[str, Any]:
        data = {
            "source": self.source, "mode": self.mode, "applied": self.applied,
            "timeout_ms": self.timeout_ms, "requested_timeout_ms": self.requested_timeout_ms,
            "changed_timeout": self.changed_timeout, "allow_retry": self.allow_retry,
            "vetoed_retry": self.vetoed_retry, "p_success": _r(self.p_success, 4), "p_retry": _r(self.p_retry, 4),
            "latency_q50_ms": _r(self.latency_q50_ms, 2), "latency_q95_ms": _r(self.latency_q95_ms, 2),
            "latency_q99_ms": _r(self.latency_q99_ms, 2), "reasons": list(self.reasons),
            "overhead_us": round(self.overhead_us, 1),
        }
        if include_features and self.features is not None:
            data["features"] = [round(v, 6) for v in self.features]
        return data


def _r(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(float(value), digits)


@dataclass
class EndpointSnapshot:
    n: int = 0
    success_rate_recent: float = 1.0
    success_rate_window: float = 1.0
    consecutive_failures: int = 0
    timeout_rate_recent: float = 0.0
    latency_ewma_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_median_ms: float = 0.0
    success_count: int = 0
    since_last_failure_s: Optional[float] = None
    last_error_class: str = "none"


class EndpointHistory:
    """Bounded outcome history of one endpoint."""

    def __init__(self) -> None:
        self.rows: Deque[Tuple[float, bool, float, str]] = deque(maxlen=HISTORY_LEN)  # (t, ok, latency_ms, error_class)
        self.latency_ewma: Optional[float] = None
        self.consecutive_failures = 0
        self.last_failure_t: Optional[float] = None

    def add(self, now: float, ok: bool, latency_ms: float, error_class: str) -> None:
        latency_ms = max(0.0, float(latency_ms))
        self.rows.append((now, bool(ok), latency_ms, error_class))
        if ok:
            self.consecutive_failures = 0
            self.latency_ewma = latency_ms if self.latency_ewma is None else 0.7 * self.latency_ewma + 0.3 * latency_ms
        else:
            self.consecutive_failures += 1
            self.last_failure_t = now

    def snapshot(self, now: float) -> EndpointSnapshot:
        rows = list(self.rows)
        n = len(rows)
        if n == 0:
            return EndpointSnapshot()
        recent = rows[-RECENT_LEN:]
        # Beta(1.8, 0.2)-style prior: an unseen endpoint is presumed healthy but a single failure is not decisive.
        sr_recent = (sum(1 for r in recent if r[1]) + 1.8) / (len(recent) + 2.0)
        sr_window = (sum(1 for r in rows if r[1]) + 1.8) / (n + 2.0)
        successes = sorted(r[2] for r in rows if r[1])
        p95 = successes[min(len(successes) - 1, int(math.ceil(0.95 * len(successes))) - 1)] if successes else 0.0
        median = successes[len(successes) // 2] if successes else 0.0
        return EndpointSnapshot(
            n=n,
            success_rate_recent=sr_recent,
            success_rate_window=sr_window,
            consecutive_failures=self.consecutive_failures,
            timeout_rate_recent=sum(1 for r in recent if (not r[1]) and r[3] == "timeout") / len(recent),
            latency_ewma_ms=self.latency_ewma or 0.0,
            latency_p95_ms=p95,
            latency_median_ms=median,
            success_count=len(successes),
            since_last_failure_s=None if self.last_failure_t is None else max(0.0, now - self.last_failure_t),
            last_error_class=rows[-1][3] if not rows[-1][1] else "none",
        )


def build_features(ctx: DeliveryContext, snap: EndpointSnapshot) -> List[float]:
    """Deterministic feature vector (order = FEATURES). Pure function of its inputs."""
    requested = float(ctx.requested_timeout_ms) if ctx.requested_timeout_ms else 5000.0
    if ctx.remaining_ms is None:
        budget_ratio = 5.0
    else:
        budget_ratio = min(5.0, max(0.0, float(ctx.remaining_ms) / max(1.0, requested)))
    err = ctx.last_error_class if ctx.last_error_class != "none" else snap.last_error_class
    since = 1.0 if snap.since_last_failure_s is None else min(1.0, math.log1p(snap.since_last_failure_s) / _LOG600)
    max_attempts = max(1, int(ctx.max_attempts))
    attempt = max(1, int(ctx.attempt))
    return [
        float(min(attempt - 1, 5)),
        max(0.0, (max_attempts - attempt) / max_attempts),
        math.log1p(requested),
        budget_ratio,
        math.log1p(max(0, int(ctx.payload_bytes))),
        min(1.0, snap.n / HISTORY_LEN),
        snap.success_rate_recent,
        snap.success_rate_window,
        min(8, snap.consecutive_failures) / 8.0,
        snap.timeout_rate_recent,
        math.log1p(snap.latency_ewma_ms),
        math.log1p(snap.latency_p95_ms),
        math.log1p(snap.latency_median_ms),
        since,
        1.0 if err == "timeout" else 0.0,
        1.0 if err == "connection" else 0.0,
        1.0 if err == "rate_limit" else 0.0,
        1.0 if err == "server" else 0.0,
        1.0 if attempt > 1 else 0.0,
        1.0 if ctx.idempotent else 0.0,
    ]


# ---------------------------------------------------------------------------------------------
# Model (constrained JSON checkpoint, pure-Python inference)
# ---------------------------------------------------------------------------------------------
class DeliveryModel:
    """Success head (logistic or gradient-boosted trees) + ridge log-latency head with empirical residual quantiles."""

    MAX_TREES = 500
    MAX_NODES_PER_TREE = 4096

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        self.manifest = dict(manifest)
        names = list(manifest.get("features") or [])
        if names != list(FEATURES) or manifest.get("feature_hash") != FEATURE_HASH:
            raise ValueError("feature schema mismatch")
        dim = len(FEATURES)
        self.mean = _vector(manifest.get("mean"), dim, "mean")
        self.scale = _vector(manifest.get("scale"), dim, "scale")
        if any(s <= 0 for s in self.scale):
            raise ValueError("non-positive scale")
        success = manifest.get("success") or {}
        latency = manifest.get("latency") or {}
        self.success_kind = str(success.get("kind", "logistic"))
        if self.success_kind == "logistic":
            self.w_success = _vector(success.get("weights"), dim, "success.weights")
            self.b_success = _finite(success.get("bias"), "success.bias")
            self.trees: List[Dict[str, List[Any]]] = []
        elif self.success_kind == "gbt_logistic":
            self.w_success, self.b_success = [], _finite(success.get("base"), "success.base")
            self.trees = _validate_trees(success.get("trees"), dim, self.MAX_TREES, self.MAX_NODES_PER_TREE)
        else:
            raise ValueError(f"unsupported success head kind: {self.success_kind}")
        retry = manifest.get("retry_success")
        self.w_retry: Optional[List[float]] = None
        self.b_retry = 0.0
        if retry is not None:
            if str(retry.get("kind", "logistic")) != "logistic":
                raise ValueError("unsupported retry head kind")
            self.w_retry = _vector(retry.get("weights"), dim, "retry_success.weights")
            self.b_retry = _finite(retry.get("bias"), "retry_success.bias")
        self.w_latency = _vector(latency.get("weights"), dim, "latency.weights")
        self.b_latency = _finite(latency.get("bias"), "latency.bias")
        quantiles = latency.get("residual_quantiles") or {}
        self.rq = {key: _finite(quantiles.get(key), f"latency.residual_quantiles.{key}") for key in ("q50", "q95", "q99")}
        self.promotion = str((manifest.get("promotion") or {}).get("status", "experimental"))

    @classmethod
    def load(cls, path: Path) -> "DeliveryModel":
        path = Path(path)
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact too large")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported schema")
        return cls(manifest)

    def predict(self, x: Sequence[float]) -> Dict[str, Optional[float]]:
        z = [(v - m) / s for v, m, s in zip(x, self.mean, self.scale)]
        if self.success_kind == "logistic":
            logit = self.b_success + sum(w * v for w, v in zip(self.w_success, z))
        else:
            logit = self.b_success
            for tree in self.trees:
                node = 0
                feature, threshold, left, right = tree["feature"], tree["threshold"], tree["left"], tree["right"]
                while feature[node] >= 0:
                    node = left[node] if x[int(feature[node])] <= threshold[node] else right[node]
                logit += tree["value"][node]
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        mu = self.b_latency + sum(w * v for w, v in zip(self.w_latency, z))
        p_retry = None
        if self.w_retry is not None:
            r_logit = self.b_retry + sum(w * v for w, v in zip(self.w_retry, z))
            p_retry = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, r_logit))))
        return {
            "p_success": p,
            "p_retry": p_retry,
            "latency_q50_ms": max(0.0, math.expm1(mu + self.rq["q50"])),
            "latency_q95_ms": max(0.0, math.expm1(mu + self.rq["q95"])),
            "latency_q99_ms": max(0.0, math.expm1(mu + self.rq["q99"])),
        }


def _validate_trees(raw: Any, dim: int, max_trees: int, max_nodes: int) -> List[Dict[str, List[Any]]]:
    """Strict structural validation so a malformed checkpoint can never loop or index out of range."""
    if not isinstance(raw, list) or not raw or len(raw) > max_trees:
        raise ValueError("trees missing or too many")
    trees: List[Dict[str, List[Any]]] = []
    for tree in raw:
        if not isinstance(tree, Mapping):
            raise ValueError("tree is not a mapping")
        arrays = {key: list(tree.get(key) or []) for key in ("feature", "threshold", "left", "right", "value")}
        n = len(arrays["feature"])
        if n == 0 or n > max_nodes or any(len(v) != n for v in arrays.values()):
            raise ValueError("tree arrays have inconsistent length")
        for index in range(n):
            f = arrays["feature"][index]
            _finite(arrays["value"][index], "tree.value")
            if f >= 0:
                if int(f) != f or int(f) >= dim:
                    raise ValueError("tree feature index out of range")
                _finite(arrays["threshold"][index], "tree.threshold")
                for side in ("left", "right"):
                    child = arrays[side][index]
                    if int(child) != child or not (index < int(child) < n):
                        raise ValueError("tree child index must point forward inside the tree")
            elif f != -1:
                raise ValueError("leaf marker must be -1")
        trees.append({
            "feature": [int(x) for x in arrays["feature"]], "left": [int(x) for x in arrays["left"]],
            "right": [int(x) for x in arrays["right"]], "threshold": [float(x) for x in arrays["threshold"]],
            "value": [float(x) for x in arrays["value"]],
        })
    return trees


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {name}")
    return number


def _vector(value: Any, dim: int, name: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != dim:
        raise ValueError(f"{name} has wrong length")
    return [_finite(v, name) for v in value]


# ---------------------------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "mode": "shadow",
    "manifest_path": "src/agents/network/models/delivery_predictor/manifest.json",
    "allow_experimental_active": False,
    "min_history": 8,               # observed attempts + successes required before any behaviour change
    "min_timeout_ms": 50,           # never shorten a timeout below this (unless the caller asked for less)
    "timeout_margin": 1.5,          # recommended timeout = margin * predicted q99 latency
    "timeout_escalation": 2.0,      # each retry after a timeout multiplies the tightened timeout
    "veto_p_success": 0.15,         # veto a retry only below this predicted success probability ...
    "veto_min_consecutive_failures": 2,   # ... and only after this many consecutive endpoint failures
    "max_endpoints": 256,
    "max_consecutive_errors": 5,    # predictor errors before it disables itself temporarily
    "error_cooldown_s": 60.0,
}


class DeliveryPredictor:
    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        model: Optional[DeliveryModel] = None,
        base_dir: Optional[Path] = None,
        clock: Callable[[], float] = time.monotonic,
        observation_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        cfg = dict(_DEFAULTS)
        cfg.update({k: v for k, v in dict(config or {}).items() if v is not None})
        self.cfg = cfg
        self.clock = clock
        self.observation_hook = observation_hook
        self.enabled = bool(cfg["enabled"])
        mode = str(cfg["mode"]).strip().lower()
        self.requested_mode = mode if mode in MODES else "shadow"
        self.min_history = max(1, int(cfg["min_history"]))
        self.min_timeout_ms = max(1, int(cfg["min_timeout_ms"]))
        self.timeout_margin = max(1.0, float(cfg["timeout_margin"]))
        self.timeout_escalation = max(1.0, float(cfg["timeout_escalation"]))
        self.veto_p = min(1.0, max(0.0, float(cfg["veto_p_success"])))
        self.veto_min_consecutive = max(1, int(cfg["veto_min_consecutive_failures"]))
        self.max_endpoints = max(1, int(cfg["max_endpoints"]))
        self.max_consecutive_errors = max(1, int(cfg["max_consecutive_errors"]))
        self.error_cooldown_s = max(0.0, float(cfg["error_cooldown_s"]))

        self._lock = threading.RLock()
        self._histories: "OrderedDict[str, EndpointHistory]" = OrderedDict()
        self._stats: Dict[str, Any] = {
            "plans": 0, "retry_checks": 0, "observations": 0, "applied_timeouts": 0, "applied_vetoes": 0,
            "would_tighten": 0, "would_veto": 0, "errors": 0, "fallbacks": 0, "overhead_us_total": 0.0,
            "by_source": {},
        }
        self._consecutive_errors = 0
        self._disabled_until = 0.0

        self.model: Optional[DeliveryModel] = None
        self.model_status = "none"
        if self.enabled and self.requested_mode != "off":
            self._load_model(model, base_dir)

        self.mode = self._effective_mode()

    # -- configuration -------------------------------------------------------------------------
    def _load_model(self, model: Optional[DeliveryModel], base_dir: Optional[Path]) -> None:
        if model is not None:
            self.model, self.model_status = model, "injected"
            return
        path = Path(str(self.cfg.get("manifest_path") or ""))
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        if not path.exists():
            self.model_status = "missing"
            return
        try:
            self.model = DeliveryModel.load(path)
            self.model_status = "loaded"
        except Exception as exc:  # noqa: BLE001 - any invalid artifact falls back to the heuristic
            self.model, self.model_status = None, f"invalid:{type(exc).__name__}:{exc}"

    def _effective_mode(self) -> str:
        if not self.enabled or self.requested_mode == "off":
            return "off"
        if self.requested_mode == "active":
            # Behaviour changes need an approved checkpoint, or an explicit opt-in for experimental/heuristic use.
            approved = self.model is not None and self.model.promotion == "approved"
            if approved or bool(self.cfg["allow_experimental_active"]):
                return "active"
        return "shadow"

    # -- history -------------------------------------------------------------------------------
    def _history(self, endpoint: str) -> EndpointHistory:
        history = self._histories.get(endpoint)
        if history is None:
            history = EndpointHistory()
            self._histories[endpoint] = history
            while len(self._histories) > self.max_endpoints:
                self._histories.popitem(last=False)
        else:
            self._histories.move_to_end(endpoint)
        return history

    def endpoint_snapshot(self, endpoint: str) -> EndpointSnapshot:
        with self._lock:
            return self._history(endpoint).snapshot(self.clock())

    def observe(
        self,
        endpoint: str,
        *,
        ok: bool,
        latency_ms: float,
        error: Optional[BaseException] = None,
        decision: Optional[DeliveryDecision] = None,
        context: Optional[DeliveryContext] = None,
    ) -> None:
        """Record the outcome of one transport attempt (and emit a training row if a hook is set)."""
        if not self.enabled or not endpoint:
            return
        try:
            error_class = "none" if ok else classify_error(error)
            with self._lock:
                self._history(endpoint).add(self.clock(), ok, latency_ms, error_class)
                self._stats["observations"] += 1
            if self.observation_hook is not None and decision is not None and decision.features is not None:
                self.observation_hook({
                    "features": list(decision.features), "success": bool(ok), "latency_ms": float(latency_ms),
                    "error_class": error_class, "endpoint": endpoint,
                    "attempt": None if context is None else context.attempt,
                    "requested_timeout_ms": decision.requested_timeout_ms, "applied_timeout_ms": decision.timeout_ms,
                    "source": decision.source, "mode": decision.mode,
                })
        except Exception:  # noqa: BLE001 - observation must never affect delivery
            self._record_error()

    # -- decisions -----------------------------------------------------------------------------
    def plan(self, ctx: DeliveryContext) -> DeliveryDecision:
        """Choose the transport timeout for one attempt. Never raises."""
        started = time.perf_counter()
        requested = None if ctx.requested_timeout_ms is None else int(ctx.requested_timeout_ms)
        passthrough = DeliveryDecision(timeout_ms=requested, requested_timeout_ms=requested, mode=self.mode)
        try:
            if not self._active_now():
                passthrough.source = "disabled" if self.mode == "off" else "fallback:cooldown"
                return self._finish(passthrough, started, plan=True)
            with self._lock:
                snap = self._history(ctx.endpoint).snapshot(self.clock())
            x = build_features(ctx, snap)
            if not all(math.isfinite(v) for v in x):
                raise ValueError("non-finite feature")
            prediction, source = self._predict(x, snap)
            decision = DeliveryDecision(
                timeout_ms=requested, requested_timeout_ms=requested, mode=self.mode, source=source,
                p_success=prediction["p_success"], p_retry=prediction.get("p_retry"), latency_q50_ms=prediction["latency_q50_ms"],
                latency_q95_ms=prediction["latency_q95_ms"], latency_q99_ms=prediction["latency_q99_ms"],
                features=x,
            )
            self._apply_timeout_policy(decision, ctx, snap)
            self._consecutive_errors = 0
            return self._finish(decision, started, plan=True)
        except Exception as exc:  # noqa: BLE001
            self._record_error()
            passthrough.source = f"fallback:{type(exc).__name__}"
            passthrough.reasons.append("predictor_error")
            return self._finish(passthrough, started, plan=True)

    def allow_retry(self, ctx: DeliveryContext) -> DeliveryDecision:
        """Last veto over a retry the reliability layer already allowed. ``ctx.attempt`` is the *next* attempt."""
        started = time.perf_counter()
        requested = None if ctx.requested_timeout_ms is None else int(ctx.requested_timeout_ms)
        passthrough = DeliveryDecision(timeout_ms=requested, requested_timeout_ms=requested, mode=self.mode)
        try:
            if not self._active_now():
                passthrough.source = "disabled" if self.mode == "off" else "fallback:cooldown"
                return self._finish(passthrough, started, plan=False)
            with self._lock:
                snap = self._history(ctx.endpoint).snapshot(self.clock())
            x = build_features(ctx, snap)
            if not all(math.isfinite(v) for v in x):
                raise ValueError("non-finite feature")
            prediction, source = self._predict(x, snap)
            decision = DeliveryDecision(
                timeout_ms=requested, requested_timeout_ms=requested, mode=self.mode, source=source,
                p_success=prediction["p_success"], p_retry=prediction.get("p_retry"), latency_q50_ms=prediction["latency_q50_ms"],
                latency_q95_ms=prediction["latency_q95_ms"], latency_q99_ms=prediction["latency_q99_ms"],
                features=x,
            )
            self._apply_retry_policy(decision, ctx, snap)
            self._consecutive_errors = 0
            return self._finish(decision, started, plan=False)
        except Exception as exc:  # noqa: BLE001
            self._record_error()
            passthrough.source = f"fallback:{type(exc).__name__}"
            passthrough.reasons.append("predictor_error")
            return self._finish(passthrough, started, plan=False)

    # -- internals -----------------------------------------------------------------------------
    def _active_now(self) -> bool:
        if self.mode == "off":
            return False
        return self.clock() >= self._disabled_until

    def _predict(self, x: Sequence[float], snap: EndpointSnapshot) -> Tuple[Dict[str, Optional[float]], str]:
        if self.model is not None:
            return self.model.predict(x), "model"
        # Heuristic fallback: smoothed recent success rate; latency from the observed success distribution.
        p95 = snap.latency_p95_ms
        return {
            "p_success": snap.success_rate_recent,
            "p_retry": None,
            "latency_q50_ms": snap.latency_median_ms,
            "latency_q95_ms": p95,
            "latency_q99_ms": max(p95, snap.latency_ewma_ms) * 1.25,
        }, "heuristic"

    def _apply_timeout_policy(self, decision: DeliveryDecision, ctx: DeliveryContext, snap: EndpointSnapshot) -> None:
        requested = decision.requested_timeout_ms
        if requested is None:
            decision.reasons.append("no_requested_timeout")
            return
        if snap.success_count < self.min_history:
            decision.reasons.append("insufficient_history")
            return
        if snap.success_rate_recent < 0.5:
            decision.reasons.append("endpoint_unreliable_keep_requested")
            return
        q99 = decision.latency_q99_ms or 0.0
        proposal = self.timeout_margin * q99
        # Escalate after a timed-out attempt so a regime shift (endpoint became slower) costs one cheap miss.
        if ctx.attempt > 1 and ctx.last_error_class == "timeout":
            proposal *= self.timeout_escalation ** (ctx.attempt - 1)
        floor = min(requested, self.min_timeout_ms)
        recommended = int(min(requested, max(floor, math.ceil(proposal))))
        if ctx.remaining_ms is not None:
            recommended = int(max(1, min(recommended, ctx.remaining_ms)))
        if recommended < requested:
            decision.reasons.append("timeout_tightened")
            with self._lock:
                self._stats["would_tighten"] += 1
                if self.mode == "active":
                    self._stats["applied_timeouts"] += 1
            if self.mode == "active":
                decision.timeout_ms = recommended
                decision.changed_timeout = True
                decision.applied = True

    def _apply_retry_policy(self, decision: DeliveryDecision, ctx: DeliveryContext, snap: EndpointSnapshot) -> None:
        if ctx.attempt <= 1:
            decision.reasons.append("first_attempt_never_vetoed")
            return
        if snap.n < self.min_history:
            decision.reasons.append("insufficient_history")
            return
        if snap.consecutive_failures < self.veto_min_consecutive:
            decision.reasons.append("failures_not_consecutive_enough")
            return
        if ctx.last_error_class not in ("timeout", "connection", "server"):
            decision.reasons.append("error_class_not_vetoable")
            return
        p_next = decision.p_retry if decision.p_retry is not None else decision.p_success
        if p_next is not None and p_next < self.veto_p:
            decision.reasons.append("retry_unlikely_to_succeed")
            with self._lock:
                self._stats["would_veto"] += 1
                if self.mode == "active":
                    self._stats["applied_vetoes"] += 1
            if self.mode == "active":
                decision.allow_retry = False
                decision.vetoed_retry = True
                decision.applied = True

    def _finish(self, decision: DeliveryDecision, started: float, *, plan: bool) -> DeliveryDecision:
        elapsed_us = (time.perf_counter() - started) * 1e6
        decision.overhead_us = elapsed_us
        with self._lock:
            self._stats["plans" if plan else "retry_checks"] += 1
            self._stats["overhead_us_total"] += elapsed_us
            by_source = self._stats["by_source"]
            by_source[decision.source] = by_source.get(decision.source, 0) + 1
            if decision.source.startswith("fallback") or decision.source == "disabled":
                self._stats["fallbacks"] += 1
        return decision

    def _record_error(self) -> None:
        with self._lock:
            self._stats["errors"] += 1
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.max_consecutive_errors:
                self._disabled_until = self.clock() + self.error_cooldown_s
                self._consecutive_errors = 0

    # -- observability -------------------------------------------------------------------------
    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
            stats["by_source"] = dict(stats["by_source"])
            calls = stats["plans"] + stats["retry_checks"]
            stats["mean_overhead_us"] = round(stats["overhead_us_total"] / calls, 2) if calls else 0.0
            stats.pop("overhead_us_total", None)
            return {
                "enabled": self.enabled,
                "requested_mode": self.requested_mode,
                "mode": self.mode,
                "model_status": self.model_status,
                "model_promotion": None if self.model is None else self.model.promotion,
                "feature_hash": FEATURE_HASH,
                "tracked_endpoints": len(self._histories),
                "cooldown_active": self.clock() < self._disabled_until,
                "stats": stats,
            }
