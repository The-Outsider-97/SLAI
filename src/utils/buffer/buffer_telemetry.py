from __future__ import annotations

import math
import time
import numpy as np  # pyright: ignore[reportMissingImports]

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Deque, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Buffer Telemetry")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Canonical telemetry signal names
# ---------------------------------------------------------------------------

PUSH_LATENCY = "push_latency_seconds"
SAMPLE_LATENCY = "sample_latency_seconds"
LOCK_WAIT = "lock_wait_seconds"
LOCK_CONTENTION = "lock_contention_count"
REJECTION_COUNT = "rejection_count"
STALE_PRUNE_COUNT = "stale_prune_count"


@dataclass
class MetricStats:
    """Online metric statistics with bounded percentile history.

    The class keeps exact running count/total/min/max values and an optional
    bounded history for recent-window percentile diagnostics. This avoids
    unbounded memory growth while still supporting p50/p95/p99 operational
    views for latency and contention metrics.
    """

    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    last: float = 0.0
    squared_total: float = 0.0
    max_samples: int = 512
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: Optional[str] = None
    samples: Deque[float] = field(default_factory=deque)

    def update(self, value: float) -> None:
        value = validate_metric_value("metric", value)
        self.count += 1
        self.total += value
        self.squared_total += value * value
        self.last = value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.updated_at = utcnow_iso()

        if self.max_samples > 0:
            self.samples.append(value)
            while len(self.samples) > self.max_samples:
                self.samples.popleft()

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        mean_square = self.squared_total / self.count
        return max(0.0, mean_square - (self.mean * self.mean))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def quantile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        return float(np.percentile(np.asarray(self.samples, dtype=np.float64), q))

    def to_dict(self, *, include_samples: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "count": int(self.count),
            "total": float(self.total),
            "mean": float(self.mean),
            "std": float(self.std),
            "min": 0.0 if self.min == float("inf") else float(self.min),
            "max": 0.0 if self.max == float("-inf") else float(self.max),
            "last": float(self.last),
            "p50": self.quantile(50),
            "p95": self.quantile(95),
            "p99": self.quantile(99),
            "history_size": len(self.samples),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_samples:
            payload["samples"] = list(self.samples)
        return payload


@dataclass(frozen=True)
class MetricSummary:
    """Immutable statistical summary for model-card/report exports."""

    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float
    count: int


@dataclass(frozen=True)
class BufferTelemetryConfig:
    """Configuration contract for buffer telemetry.

    Loaded from the existing `telemetry` section in `buffer_config.yaml`.
    User overrides follow the same shape already used by the buffer modules:
    `{"telemetry": {...}}`.
    """

    enabled: bool = True
    log_interval: int = 200
    max_observations_per_metric: int = 512
    keep_observation_history: bool = True
    slow_operation_threshold_seconds: float = 0.25
    lock_contention_threshold_seconds: float = 0.001
    rejection_rate_window: int = 1000
    emit_slow_operation_logs: bool = True
    strict_metric_validation: bool = True

    @classmethod
    def from_config(cls, user_config: Optional[Mapping[str, Any]] = None) -> "BufferTelemetryConfig":
        load_global_config()
        cfg = dict(get_config_section("telemetry") or {})
        if user_config:
            cfg.update(dict(user_config.get("telemetry", {}) if isinstance(user_config, Mapping) else {}))

        log_interval = int(cfg.get("log_interval", 200))
        max_obs = int(cfg.get("max_observations_per_metric", 512))
        slow_threshold = float(cfg.get("slow_operation_threshold_seconds", 0.25))
        contention_threshold = float(cfg.get("lock_contention_threshold_seconds", 0.001))
        rejection_window = int(cfg.get("rejection_rate_window", 1000))

        if log_interval < 0:
            raise ConfigValueError("log_interval", log_interval, "integer >= 0", section="telemetry")
        if max_obs < 0:
            raise ConfigValueError("max_observations_per_metric", max_obs, "integer >= 0", section="telemetry")
        if slow_threshold < 0:
            raise ConfigValueError("slow_operation_threshold_seconds", slow_threshold, "float >= 0", section="telemetry")
        if contention_threshold < 0:
            raise ConfigValueError("lock_contention_threshold_seconds", contention_threshold, "float >= 0", section="telemetry")
        if rejection_window <= 0:
            raise ConfigValueError("rejection_rate_window", rejection_window, "integer > 0", section="telemetry")

        return cls(
            enabled=bool(cfg.get("enabled", True)),
            log_interval=log_interval,
            max_observations_per_metric=max_obs,
            keep_observation_history=bool(cfg.get("keep_observation_history", True)),
            slow_operation_threshold_seconds=slow_threshold,
            lock_contention_threshold_seconds=contention_threshold,
            rejection_rate_window=rejection_window,
            emit_slow_operation_logs=bool(cfg.get("emit_slow_operation_logs", True)),
            strict_metric_validation=bool(cfg.get("strict_metric_validation", True)),
        )


class MetricSummarizer:
    """Replay/buffer-safe numeric summarization for sampled metrics."""

    @staticmethod
    def summarize(values: Sequence[float]) -> MetricSummary:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return MetricSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        if not np.all(np.isfinite(arr)):
            raise MetricValueError("summary_values", "non-finite sequence", reason="all values must be finite")
        return MetricSummary(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            count=int(arr.size),
        )

    @staticmethod
    def summarize_dict(metric_map: Mapping[str, Sequence[float]]) -> Dict[str, MetricSummary]:
        return {str(name): MetricSummarizer.summarize(vals) for name, vals in metric_map.items()}

    @staticmethod
    def create_model_card(metrics: Mapping[str, Any], references: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "metrics": dict(metrics),
            "references": dict(references),
            "generated_at": utcnow_iso(),
        }


class FairnessMetrics:
    """Buffer/replay-facing fairness checks for training-time guardrails."""

    @staticmethod
    def _rates_from_groups(
        metric_name: str,
        groups: Sequence[str],
        rates: Mapping[str, float],
    ) -> np.ndarray:
        if len(rates) < 2:
            raise FairnessMetricError("rates for at least 2 groups", rates, metric_name=metric_name)

        ordered = [float(rates[g]) for g in groups if g in rates]
        if len(ordered) < 2:
            ordered = [float(v) for v in rates.values()]

        arr = np.asarray(ordered, dtype=np.float64)
        if arr.size < 2 or not np.all(np.isfinite(arr)):
            raise FairnessMetricError("at least 2 finite numeric group rates", rates, metric_name=metric_name)
        return arr

    @staticmethod
    def demographic_parity(
        sensitive_groups: Sequence[str],
        positive_rates: Mapping[str, float],
        threshold: float = 0.05,
    ) -> Tuple[bool, str]:
        threshold = validate_metric_value("demographic_parity_threshold", threshold)
        arr = FairnessMetrics._rates_from_groups("demographic_parity", sensitive_groups, positive_rates)
        max_diff = float(np.max(arr) - np.min(arr))
        violation = max_diff > threshold
        msg = (
            f"Demographic parity {'VIOLATION' if violation else 'OK'} | "
            f"max_diff={max_diff:.6f}, threshold={threshold:.6f}"
        )
        return violation, msg

    @staticmethod
    def equalized_odds(
        tpr_by_group: Mapping[str, float],
        fpr_by_group: Mapping[str, float],
        threshold: float = 0.05,
    ) -> Tuple[bool, str]:
        threshold = validate_metric_value("equalized_odds_threshold", threshold)
        tprs = FairnessMetrics._rates_from_groups("equalized_odds_tpr", list(tpr_by_group.keys()), tpr_by_group)
        fprs = FairnessMetrics._rates_from_groups("equalized_odds_fpr", list(fpr_by_group.keys()), fpr_by_group)
        tpr_diff = float(np.max(tprs) - np.min(tprs))
        fpr_diff = float(np.max(fprs) - np.min(fprs))
        violation = (tpr_diff > threshold) or (fpr_diff > threshold)
        msg = (
            f"Equalized odds {'VIOLATION' if violation else 'OK'} | "
            f"tpr_diff={tpr_diff:.6f}, fpr_diff={fpr_diff:.6f}, threshold={threshold:.6f}"
        )
        return violation, msg


class BufferTelemetry:
    """Thread-safe telemetry collector for replay, reservoir, network, and n-step buffers.

    Canonical signals:
    - push latency
    - sample latency
    - lock contention / wait time
    - rejection counts and rejection rates
    - stale-prune counts
    """

    def __init__(self, component_name: str = "buffer", user_config: Optional[Mapping[str, Any]] = None):
        self.component_name = str(component_name).strip() or "buffer"
        self.config = BufferTelemetryConfig.from_config(user_config=user_config)
        self.enabled = self.config.enabled
        self.log_interval = self.config.log_interval

        self.lock = RLock()
        self.counters: Dict[str, float] = {}
        self.observations: Dict[str, MetricStats] = {}
        self.rejection_reasons: Dict[str, float] = {}
        self._operation_results: Dict[str, Deque[bool]] = {}
        self._created_at = utcnow_iso()

    # ------------------------------------------------------------------ #
    # Core metric API
    # ------------------------------------------------------------------ #
    def increment(self, name: str, amount: float = 1.0) -> None:
        if not self.enabled:
            return
        metric = validate_metric_name(name)
        value = validate_metric_value(metric, amount)
        with self.lock:
            self.counters[metric] = self.counters.get(metric, 0.0) + value
            if self.log_interval > 0 and int(self.counters[metric]) % self.log_interval == 0:
                logger.info("[%s] counter %s=%s", self.component_name, metric, self.counters[metric])

    def observe(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        metric = validate_metric_name(name)
        resolved = validate_metric_value(metric, value)
        max_samples = self.config.max_observations_per_metric if self.config.keep_observation_history else 0
        with self.lock:
            if metric not in self.observations:
                self.observations[metric] = MetricStats(max_samples=max_samples)
            self.observations[metric].update(resolved)

    @contextmanager
    def time_block(self, name: str) -> Iterator[None]:
        metric = validate_metric_name(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.observe(metric, elapsed)
            self._record_slow_operation(metric, elapsed)

    @contextmanager
    def time_operation(self, operation: str) -> Iterator[None]:
        op = validate_metric_name(operation)
        with self.time_block(f"{op}_latency_seconds"):
            yield

    @contextmanager
    def time_push(self) -> Iterator[None]:
        with self.time_block(PUSH_LATENCY):
            yield

    @contextmanager
    def time_sample(self) -> Iterator[None]:
        with self.time_block(SAMPLE_LATENCY):
            yield

    # ------------------------------------------------------------------ #
    # Canonical buffer signals
    # ------------------------------------------------------------------ #
    def record_push_latency(self, seconds: float) -> None:
        self.observe(PUSH_LATENCY, seconds)
        self._record_slow_operation(PUSH_LATENCY, float(seconds))

    def record_sample_latency(self, seconds: float) -> None:
        self.observe(SAMPLE_LATENCY, seconds)
        self._record_slow_operation(SAMPLE_LATENCY, float(seconds))

    def record_lock_wait(self, operation: str, seconds: float, *, acquired: bool = True) -> None:
        op = validate_metric_name(operation)
        waited = validate_metric_value(LOCK_WAIT, seconds)
        self.observe(f"{op}_{LOCK_WAIT}", waited)
        self.observe(LOCK_WAIT, waited)
        if waited >= self.config.lock_contention_threshold_seconds or not acquired:
            self.increment(LOCK_CONTENTION, 1)
            self.increment(f"{op}_{LOCK_CONTENTION}", 1)
        if not acquired:
            self.increment(f"{op}_lock_timeout_count", 1)

    def record_lock_contention(self, operation: str, waited_seconds: float, *, acquired: bool = True) -> None:
        self.record_lock_wait(operation, waited_seconds, acquired=acquired)

    @contextmanager
    def lock_block(
        self,
        lock: Any,
        *,
        operation: str,
        timeout_seconds: Optional[float] = None,
        raise_on_timeout: bool = True,
    ) -> Iterator[bool]:
        op = validate_metric_name(operation)
        start = time.perf_counter()
        if timeout_seconds is None:
            acquired = bool(lock.acquire())
            timeout_for_error = 0.0
        else:
            timeout_for_error = validate_metric_value(f"{op}_lock_timeout_seconds", timeout_seconds)
            acquired = bool(lock.acquire(timeout=timeout_for_error))

        waited = time.perf_counter() - start
        self.record_lock_wait(op, waited, acquired=acquired)

        if not acquired:
            if raise_on_timeout:
                raise BufferLockTimeoutError(operation=op, timeout_seconds=timeout_for_error)
            yield False
            return

        try:
            yield True
        finally:
            lock.release()

    def record_acceptance(self, operation: str = "operation") -> None:
        op = validate_metric_name(operation)
        self.increment(f"{op}_attempt_count", 1)
        self.increment(f"{op}_accepted_count", 1)
        self._remember_operation_result(op, accepted=True)

    def record_rejection(self, operation: str = "operation", reason: str = "unspecified") -> None:
        op = validate_metric_name(operation)
        normalized_reason = str(reason).strip() or "unspecified"
        self.increment(REJECTION_COUNT, 1)
        self.increment(f"{op}_attempt_count", 1)
        self.increment(f"{op}_rejected_count", 1)
        self._remember_operation_result(op, accepted=False)
        with self.lock:
            self.rejection_reasons[normalized_reason] = self.rejection_reasons.get(normalized_reason, 0.0) + 1.0

    def record_stale_prune(self, count: int, *, operation: str = "stale_prune") -> None:
        pruned = validate_metric_value(STALE_PRUNE_COUNT, count)
        if pruned < 0:
            raise MetricValueError(STALE_PRUNE_COUNT, count, reason="stale prune count must be >= 0")
        self.increment(STALE_PRUNE_COUNT, pruned)
        self.increment(f"{operation}_event_count", 1)
        self.observe("last_stale_prune_count", pruned)

    # ------------------------------------------------------------------ #
    # Derived metrics and exports
    # ------------------------------------------------------------------ #
    def rejection_rate(self, operation: Optional[str] = None) -> float:
        with self.lock:
            if operation is not None:
                op = validate_metric_name(operation)
                events = self._operation_results.get(op, deque())
                return self._rate_from_events(events)

            merged: Deque[bool] = deque(maxlen=self.config.rejection_rate_window)
            for events in self._operation_results.values():
                merged.extend(events)
            return self._rate_from_events(merged)

    def snapshot(self, *, include_samples: bool = False) -> Dict[str, Any]:
        try:
            with self.lock:
                observations = {
                    key: stats.to_dict(include_samples=include_samples)
                    for key, stats in self.observations.items()
                }
                rejection_rates = {
                    op: self._rate_from_events(events)
                    for op, events in self._operation_results.items()
                }
                return {
                    "component": {
                        "name": self.component_name,
                        "enabled": self.enabled,
                        "created_at": self._created_at,
                        "generated_at": utcnow_iso(),
                    },
                    "config": {
                        "log_interval": self.config.log_interval,
                        "max_observations_per_metric": self.config.max_observations_per_metric,
                        "keep_observation_history": self.config.keep_observation_history,
                        "slow_operation_threshold_seconds": self.config.slow_operation_threshold_seconds,
                        "lock_contention_threshold_seconds": self.config.lock_contention_threshold_seconds,
                        "rejection_rate_window": self.config.rejection_rate_window,
                    },
                    "counters": dict(self.counters),
                    "observations": observations,
                    "rejection_reasons": dict(self.rejection_reasons),
                    "derived": {
                        "rejection_rate": self.rejection_rate(),
                        "rejection_rates": rejection_rates,
                    },
                    "signals": self.signal_snapshot_locked(observations=observations),
                }
        except Exception as exc:
            raise MetricSnapshotError(str(exc)) from exc

    def signal_snapshot_locked(self, *, observations: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        obs = observations if observations is not None else {
            key: stats.to_dict() for key, stats in self.observations.items()
        }
        return {
            "push_latency": obs.get(PUSH_LATENCY, {}),
            "sample_latency": obs.get(SAMPLE_LATENCY, {}),
            "lock_wait": obs.get(LOCK_WAIT, {}),
            "lock_contention_count": float(self.counters.get(LOCK_CONTENTION, 0.0)),
            "rejection_count": float(self.counters.get(REJECTION_COUNT, 0.0)),
            "rejection_rate": self.rejection_rate(),
            "stale_prune_count": float(self.counters.get(STALE_PRUNE_COUNT, 0.0)),
        }

    def export_numpy(self) -> Dict[str, np.ndarray]:
        snap = self.snapshot()
        counters = snap["counters"]
        observations = snap["observations"]

        return {
            "counter_keys": np.array(list(counters.keys()), dtype=object),
            "counter_values": np.array(list(counters.values()), dtype=np.float32),
            "observation_keys": np.array(list(observations.keys()), dtype=object),
            "observation_means": np.array(
                [entry.get("mean", 0.0) for entry in observations.values()],
                dtype=np.float32,
            ),
            "observation_p95": np.array(
                [entry.get("p95", 0.0) for entry in observations.values()],
                dtype=np.float32,
            ),
        }

    def export_prometheus(self, *, prefix: str = "buffer") -> str:
        metric_prefix = validate_metric_name(prefix).replace("-", "_")
        snap = self.snapshot()
        lines = []
        for name, value in snap["counters"].items():
            safe_name = str(name).replace("-", "_").replace(".", "_")
            lines.append(f"{metric_prefix}_{safe_name} {float(value)}")
        for name, stats in snap["observations"].items():
            safe_name = str(name).replace("-", "_").replace(".", "_")
            lines.append(f"{metric_prefix}_{safe_name}_count {float(stats.get('count', 0.0))}")
            lines.append(f"{metric_prefix}_{safe_name}_mean {float(stats.get('mean', 0.0))}")
            lines.append(f"{metric_prefix}_{safe_name}_p95 {float(stats.get('p95', 0.0))}")
        return "\n".join(lines)

    def reset(self) -> None:
        with self.lock:
            self.counters.clear()
            self.observations.clear()
            self.rejection_reasons.clear()
            self._operation_results.clear()
            self._created_at = utcnow_iso()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _remember_operation_result(self, operation: str, *, accepted: bool) -> None:
        with self.lock:
            if operation not in self._operation_results:
                self._operation_results[operation] = deque(maxlen=self.config.rejection_rate_window)
            self._operation_results[operation].append(bool(accepted))

    @staticmethod
    def _rate_from_events(events: Sequence[bool]) -> float:
        if not events:
            return 0.0
        total = len(events)
        rejected = sum(1 for accepted in events if not accepted)
        return float(rejected / total)

    def _record_slow_operation(self, metric_name: str, elapsed: float) -> None:
        if elapsed < self.config.slow_operation_threshold_seconds:
            return
        self.increment("slow_operation_count", 1)
        if self.config.emit_slow_operation_logs:
            logger.warning("[%s] slow metric %s=%.6fs", self.component_name, metric_name, elapsed)


__all__ = [
    "FairnessMetrics",
    "BufferTelemetry",
]


if __name__ == "__main__":
    print("\n=== Running  Buffer Telemetry ===\n")
    printer.status("TEST", " Buffer Telemetry initialized", "info")

    telemetry = BufferTelemetry(
        component_name="telemetry_test",
        user_config={"telemetry": {"enabled": True, "log_interval": 0, "slow_operation_threshold_seconds": 10}},
    )

    with telemetry.time_push():
        time.sleep(0.001)
    with telemetry.time_sample():
        time.sleep(0.001)

    telemetry.record_lock_contention("push", 0.002, acquired=True)
    telemetry.record_rejection("sample", "insufficient_samples")
    telemetry.record_acceptance("sample")
    telemetry.record_stale_prune(3)

    violation, message = FairnessMetrics.demographic_parity(
        ["agent_a", "agent_b"],
        {"agent_a": 0.50, "agent_b": 0.54},
        threshold=0.10,
    )
    assert not violation, message

    snapshot = telemetry.snapshot()
    assert snapshot["signals"]["stale_prune_count"] == 3.0
    assert snapshot["signals"]["rejection_count"] == 1.0
    assert 0.0 < snapshot["derived"]["rejection_rates"]["sample"] < 1.0
    assert "push_latency_seconds" in snapshot["observations"]
    assert telemetry.export_numpy()["counter_values"].size > 0

    printer.status("TEST", " Buffer Telemetry checks passed", "success")
    print("\n=== Test ran successfully ===\n")
