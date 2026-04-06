"""Production-ready data quality monitor for an autonomous financial agent.

This module tracks the health of market-data, news, sentiment, and execution
adjacent providers used by an autonomous financial agent. It is designed to be
used by orchestration code that needs fast, machine-actionable answers to
questions like:

- Is this provider still healthy enough to trust?
- Is the data fresh enough to trade on?
- Should we fallback to another source?
- Should we degrade a workflow or open a circuit breaker?
- Which symbols are repeatedly problematic for a given source?

Key features:
- Thread-safe source and symbol level monitoring
- Sliding-window event aggregation and health scoring
- Reliability drift tracking with configurable penalties/rewards
- Freshness and latency monitoring
- Fallback / retry / rate-limit tracking
- Circuit breaker support
- Structured incident logging and JSON snapshot export
- Optional integration with ``financial_errors.py``
- Backwards-compatible helper methods matching the original monitor shape
"""

from __future__ import annotations

import json
import math
import threading
import time

from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple, Union


from .config_loader import load_global_config, get_config_section
from .financial_errors import (DataStalenessError, ErrorContext, FinancialAgentError,
                               InvalidConfigurationError, MissingConfigurationError,
                               ProviderRateLimitError, ProviderTimeoutError,
                               TrendMonitoringError, classify_external_exception, log_error)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Data Quality Monitor")
printer = PrettyPrinter

def _import_sibling_module(module_name: str):
    """Attempt to load a sibling module when package imports are unavailable."""
    import importlib.util
    import sys

    current_file = Path(__file__).resolve()
    candidate = current_file.with_name(f"{module_name}.py")
    if not candidate.exists():
        raise ModuleNotFoundError(module_name)

    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(module_name)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


JSONDict = Dict[str, Any]
MaybeExc = Optional[BaseException]


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    FALLBACK = "fallback"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PARTIAL = "partial"
    STALE = "stale"
    VALIDATION_FAILURE = "validation_failure"
    CIRCUIT_OPEN = "circuit_open"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNSTABLE = "unstable"
    FAILING = "failing"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class MonitorThresholds:
    default_reliability: float = 0.90
    min_reliability: float = 0.10
    max_reliability: float = 1.00
    success_reward: float = 0.010
    failure_penalty: float = 0.050
    fallback_penalty: float = 0.035
    rate_limit_penalty: float = 0.040
    timeout_penalty: float = 0.050
    partial_penalty: float = 0.020
    stale_penalty: float = 0.045
    validation_penalty: float = 0.030
    ewma_alpha: float = 0.20
    lookback_seconds: int = 3600
    max_event_history: int = 1000
    max_symbol_event_history: int = 250
    healthy_staleness_seconds: int = 60
    degraded_staleness_seconds: int = 300
    circuit_breaker_failures: int = 5
    circuit_breaker_cooldown_seconds: int = 120
    degraded_reliability: float = 0.80
    unstable_reliability: float = 0.65
    failing_reliability: float = 0.50
    fallback_rate_warning: float = 0.15
    error_rate_unstable: float = 0.25
    error_rate_failing: float = 0.45
    p95_latency_warning_ms: float = 2500.0
    p95_latency_failing_ms: float = 5000.0
    stale_trade_data_max_age_seconds: float = 5.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "MonitorThresholds":
        if not data:
            return cls()
        valid = {k: data[k] for k in cls.__dataclass_fields__.keys() if k in data}
        instance = cls(**valid)
        instance.validate()
        return instance

    def validate(self) -> None:
        if not (0 <= self.min_reliability <= self.default_reliability <= self.max_reliability <= 1.0):
            raise InvalidConfigurationError(
                "Invalid reliability configuration for data quality monitor.",
                details={
                    "min_reliability": self.min_reliability,
                    "default_reliability": self.default_reliability,
                    "max_reliability": self.max_reliability,
                },
            )
        if self.lookback_seconds <= 0 or self.max_event_history <= 0 or self.max_symbol_event_history <= 0:
            raise InvalidConfigurationError(
                "Monitor history and lookback settings must be positive.",
                details={
                    "lookback_seconds": self.lookback_seconds,
                    "max_event_history": self.max_event_history,
                    "max_symbol_event_history": self.max_symbol_event_history,
                },
            )
        if self.circuit_breaker_failures <= 0 or self.circuit_breaker_cooldown_seconds <= 0:
            raise InvalidConfigurationError(
                "Circuit-breaker settings must be positive.",
                details={
                    "circuit_breaker_failures": self.circuit_breaker_failures,
                    "circuit_breaker_cooldown_seconds": self.circuit_breaker_cooldown_seconds,
                },
            )


@dataclass(slots=True)
class EventRecord:
    timestamp: float
    source: str
    event_type: EventType
    symbol: Optional[str] = None
    success: bool = False
    latency_ms: Optional[float] = None
    freshness_seconds: Optional[float] = None
    retry_count: int = 0
    status_code: Optional[int] = None
    error_name: Optional[str] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    fallback_to: Optional[str] = None
    record_count: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        payload = asdict(self)
        payload["event_type"] = self.event_type.value
        return payload


@dataclass(slots=True)
class SourceState:
    source: str
    reliability: float
    created_at: float
    updated_at: float
    last_success_ts: Optional[float] = None
    last_failure_ts: Optional[float] = None
    last_fallback_ts: Optional[float] = None
    last_error_name: Optional[str] = None
    last_error_message: Optional[str] = None
    last_status_code: Optional[int] = None
    consecutive_failures: int = 0
    total_events: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_fallbacks: int = 0
    total_rate_limits: int = 0
    total_timeouts: int = 0
    total_partials: int = 0
    total_stale_events: int = 0
    total_validation_failures: int = 0
    latency_ewma_ms: Optional[float] = None
    freshness_ewma_seconds: Optional[float] = None
    circuit_open_until: Optional[float] = None
    events: Deque[EventRecord] = field(default_factory=deque)


@dataclass(slots=True)
class SymbolState:
    symbol: str
    source: str
    updated_at: float
    last_success_ts: Optional[float] = None
    last_failure_ts: Optional[float] = None
    consecutive_failures: int = 0
    fallback_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    reliability: float = 0.90
    events: Deque[EventRecord] = field(default_factory=deque)


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------


class DataQualityMonitor:
    """Monitor source reliability, freshness, latency, fallback pressure, and incidents.

    The monitor is intentionally side-effect light: it tracks in-memory state and can
    optionally export/import snapshots to JSON. It does not assume any specific data
    provider SDK or trading engine.
    """

    DEFAULT_PRIMARY_SOURCES = ("alphavantage", "finnhub", "yahoofinance")

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        thresholds: Optional[MonitorThresholds] = None,
        primary_sources: Optional[Iterable[str]] = None,
        time_fn: Optional[callable] = None,
    ) -> None:
        self._time_fn = time_fn or time.time
        self._lock = threading.RLock()
        self.config = load_global_config()
        self.monitor_config = get_config_section('data_quality_monitor')

        self.thresholds = thresholds or MonitorThresholds.from_mapping(self.config.get("thresholds"))

        configured_sources = self.config.get("primary_sources") or self.config.get("tracked_sources")
        sources_iter = primary_sources or configured_sources or self.DEFAULT_PRIMARY_SOURCES
        self.primary_sources = tuple(self._normalize_source_name(s) for s in sources_iter if s)

        self._source_states: Dict[str, SourceState] = {}
        self._symbol_states: Dict[Tuple[str, str], SymbolState] = {}
        self._global_events: Deque[EventRecord] = deque(maxlen=self.thresholds.max_event_history)
        self._incidents: Deque[JSONDict] = deque(maxlen=self.config.get("max_incident_history", 500))
        self._counters = {
            "fallback_triggers": Counter(),
            "source_reliability": defaultdict(lambda: self.thresholds.default_reliability),
            "freshness_scores": {},
        }

        # Pre-initialize primary sources
        now = self._now()
        for source in self.primary_sources:
            self._source_states[source] = SourceState(
                source=source,
                reliability=self.thresholds.default_reliability,
                created_at=now,
                updated_at=now,
                events=deque(maxlen=self.thresholds.max_event_history),
            )

    # ------------------------------------------------------------------
    # Public recording API
    # ------------------------------------------------------------------

    def log_success(
        self,
        source: str,
        *,
        symbol: Optional[str] = None,
        latency_ms: Optional[float] = None,
        freshness_seconds: Optional[float] = None,
        record_count: Optional[int] = None,
        status_code: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record a successful provider interaction."""
        event = EventRecord(
            timestamp=self._now(),
            source=self._normalize_source_name(source),
            symbol=self._normalize_symbol(symbol),
            event_type=EventType.SUCCESS,
            success=True,
            latency_ms=self._coerce_non_negative(latency_ms, "latency_ms"),
            freshness_seconds=self._coerce_non_negative(freshness_seconds, "freshness_seconds"),
            record_count=record_count,
            status_code=status_code,
            details=dict(details or {}),
        )
        self._record_event(event)

    def log_failure(
        self,
        source: str,
        *,
        error: MaybeExc = None,
        symbol: Optional[str] = None,
        latency_ms: Optional[float] = None,
        freshness_seconds: Optional[float] = None,
        retry_count: int = 0,
        status_code: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record a failed provider interaction.

        ``error`` may be a domain error from ``financial_errors.py`` or any third-party
        exception; it will be normalized where possible.
        """
        normalized_error: Optional[BaseException] = None
        error_name: Optional[str] = None
        error_code: Optional[str] = None
        message: Optional[str] = None

        if error is not None:
            try:
                normalized_error = classify_external_exception(
                    error,
                    context=ErrorContext(component="data_quality_monitor", provider=source, symbol=symbol),
                    message=str(error),
                )
            except Exception:
                normalized_error = error
            error_name = type(normalized_error).__name__
            error_code = getattr(normalized_error, "code", None)
            message = str(normalized_error)

        event_type = EventType.FAILURE
        if isinstance(normalized_error, ProviderTimeoutError):
            event_type = EventType.TIMEOUT
        elif isinstance(normalized_error, ProviderRateLimitError):
            event_type = EventType.RATE_LIMIT
        elif isinstance(normalized_error, DataStalenessError):
            event_type = EventType.STALE
        elif isinstance(normalized_error, (ValueError, TypeError, MissingConfigurationError, InvalidConfigurationError)):
            event_type = EventType.VALIDATION_FAILURE

        event = EventRecord(
            timestamp=self._now(),
            source=self._normalize_source_name(source),
            symbol=self._normalize_symbol(symbol),
            event_type=event_type,
            success=False,
            latency_ms=self._coerce_non_negative(latency_ms, "latency_ms"),
            freshness_seconds=self._coerce_non_negative(freshness_seconds, "freshness_seconds"),
            retry_count=max(0, int(retry_count)),
            status_code=status_code,
            error_name=error_name,
            error_code=error_code,
            message=message,
            details=dict(details or {}),
        )
        self._record_event(event)

        if normalized_error is not None:
            try:
                log_error(normalized_error, logger_=logger, include_traceback=False)
            except Exception:
                logger.error("Source failure for %s: %s", source, normalized_error)

    def log_fallback(
        self,
        source: str,
        *,
        fallback_to: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record a fallback from one source to another."""
        event = EventRecord(
            timestamp=self._now(),
            source=self._normalize_source_name(source),
            symbol=self._normalize_symbol(symbol),
            event_type=EventType.FALLBACK,
            success=False,
            fallback_to=self._normalize_source_name(fallback_to) if fallback_to else None,
            details=dict(details or {}),
        )
        self._record_event(event)

    def log_partial_data(
        self,
        source: str,
        *,
        symbol: Optional[str] = None,
        record_count: Optional[int] = None,
        expected_count: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload = dict(details or {})
        if expected_count is not None:
            payload["expected_count"] = expected_count
        event = EventRecord(
            timestamp=self._now(),
            source=self._normalize_source_name(source),
            symbol=self._normalize_symbol(symbol),
            event_type=EventType.PARTIAL,
            success=False,
            record_count=record_count,
            details=payload,
        )
        self._record_event(event)

    def log_stale_data(
        self,
        source: str,
        *,
        symbol: Optional[str] = None,
        freshness_seconds: float,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        event = EventRecord(
            timestamp=self._now(),
            source=self._normalize_source_name(source),
            symbol=self._normalize_symbol(symbol),
            event_type=EventType.STALE,
            success=False,
            freshness_seconds=self._coerce_non_negative(freshness_seconds, "freshness_seconds"),
            details=dict(details or {}),
        )
        self._record_event(event)

    def log_validation_failure(
        self,
        source: str,
        *,
        symbol: Optional[str] = None,
        message: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        event = EventRecord(
            timestamp=self._now(),
            source=self._normalize_source_name(source),
            symbol=self._normalize_symbol(symbol),
            event_type=EventType.VALIDATION_FAILURE,
            success=False,
            message=message,
            details=dict(details or {}),
        )
        self._record_event(event)

    @contextmanager
    def monitor_operation(
        self,
        source: str,
        *,
        symbol: Optional[str] = None,
        expected_max_age_seconds: Optional[float] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[MutableMapping[str, Any]]:
        """Context manager that records success/failure automatically.

        Example:
            with monitor.monitor_operation("finnhub", symbol="AAPL") as state:
                data = fetch(...)
                state["latency_ms"] = ...
                state["freshness_seconds"] = ...
                state["record_count"] = len(data)
        """
        state: MutableMapping[str, Any] = {
            "latency_ms": None,
            "freshness_seconds": None,
            "record_count": None,
            "status_code": None,
            "details": dict(details or {}),
        }
        start = self._now()
        try:
            yield state
            latency_ms = state.get("latency_ms")
            if latency_ms is None:
                latency_ms = (self._now() - start) * 1000.0

            freshness_seconds = state.get("freshness_seconds")
            if (
                expected_max_age_seconds is not None
                and freshness_seconds is not None
                and freshness_seconds > expected_max_age_seconds
            ):
                self.log_stale_data(
                    source,
                    symbol=symbol,
                    freshness_seconds=freshness_seconds,
                    details={
                        **dict(state.get("details") or {}),
                        "expected_max_age_seconds": expected_max_age_seconds,
                    },
                )
            else:
                self.log_success(
                    source,
                    symbol=symbol,
                    latency_ms=latency_ms,
                    freshness_seconds=freshness_seconds,
                    record_count=state.get("record_count"),
                    status_code=state.get("status_code"),
                    details=state.get("details"),
                )
        except BaseException as exc:
            self.log_failure(
                source,
                error=exc,
                symbol=symbol,
                latency_ms=(self._now() - start) * 1000.0,
                freshness_seconds=state.get("freshness_seconds"),
                status_code=state.get("status_code"),
                details=state.get("details"),
            )
            raise

    # ------------------------------------------------------------------
    # Health / decision API
    # ------------------------------------------------------------------

    def get_reliability(self, source: str) -> float:
        with self._lock:
            state = self._ensure_source_state(source)
            return round(state.reliability, 6)

    def get_freshness(self, source: str) -> str:
        report = self.get_source_report(source)
        freshness_age = report["freshness"]["age_seconds"]
        if freshness_age is None:
            return "Unknown"
        if freshness_age < 60:
            return "Live (<1 min)"
        if freshness_age < 3600:
            return f"Fresh (~{int(freshness_age / 60)} min ago)"
        return "Stale (>1 hour ago)"

    def get_fallback_count(self, source: str) -> int:
        with self._lock:
            state = self._ensure_source_state(source)
            return int(state.total_fallbacks)

    def is_circuit_open(self, source: str) -> bool:
        with self._lock:
            state = self._ensure_source_state(source)
            return bool(state.circuit_open_until and state.circuit_open_until > self._now())

    def should_fallback(self, source: str) -> bool:
        """Return True if a provider looks too weak for primary use."""
        report = self.get_source_report(source)
        return report["health"]["status"] in {
            HealthStatus.FAILING.value,
            HealthStatus.CIRCUIT_OPEN.value,
        }

    def should_pause_source(self, source: str) -> bool:
        report = self.get_source_report(source)
        return report["health"]["status"] == HealthStatus.CIRCUIT_OPEN.value

    def get_preferred_sources(self) -> List[str]:
        """Return sources sorted by health and reliability."""
        reports = [self.get_source_report(s) for s in self._iter_known_sources()]
        sort_key = lambda item: (
            self._health_rank(item["health"]["status"]),
            item["health"]["reliability"],
            -1 * (item["freshness"]["age_seconds"] or math.inf),
        )
        reports.sort(key=sort_key, reverse=True)
        return [r["source"] for r in reports]

    def get_source_report(self, source: str) -> JSONDict:
        with self._lock:
            state = self._ensure_source_state(source)
            now = self._now()
            self._prune_events(state.events, now)
            metrics = self._compute_window_metrics(state.events, now)
            freshness_age = None if state.last_success_ts is None else max(0.0, now - state.last_success_ts)
            status, reasons = self._evaluate_health(state, metrics, freshness_age, now)

            return {
                "source": state.source,
                "health": {
                    "status": status.value,
                    "reasons": reasons,
                    "reliability": round(state.reliability, 6),
                    "circuit_open": bool(state.circuit_open_until and state.circuit_open_until > now),
                    "circuit_open_until": state.circuit_open_until,
                    "consecutive_failures": state.consecutive_failures,
                },
                "freshness": {
                    "age_seconds": freshness_age,
                    "last_success_ts": state.last_success_ts,
                    "freshness_ewma_seconds": state.freshness_ewma_seconds,
                },
                "latency": {
                    "ewma_ms": state.latency_ewma_ms,
                    "p50_ms": metrics["p50_latency_ms"],
                    "p95_ms": metrics["p95_latency_ms"],
                    "avg_ms": metrics["avg_latency_ms"],
                },
                "window_metrics": metrics,
                "totals": {
                    "events": state.total_events,
                    "successes": state.total_successes,
                    "failures": state.total_failures,
                    "fallbacks": state.total_fallbacks,
                    "rate_limits": state.total_rate_limits,
                    "timeouts": state.total_timeouts,
                    "partials": state.total_partials,
                    "stale_events": state.total_stale_events,
                    "validation_failures": state.total_validation_failures,
                },
                "last_error": {
                    "name": state.last_error_name,
                    "message": state.last_error_message,
                    "status_code": state.last_status_code,
                    "last_failure_ts": state.last_failure_ts,
                },
            }

    def get_symbol_report(self, symbol: str, source: Optional[str] = None) -> JSONDict:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            raise ValueError("symbol is required")

        with self._lock:
            rows: List[JSONDict] = []
            for key, state in self._symbol_states.items():
                if key[0] != normalized_symbol:
                    continue
                if source and key[1] != self._normalize_source_name(source):
                    continue
                now = self._now()
                self._prune_events(state.events, now)
                metrics = self._compute_window_metrics(state.events, now)
                rows.append(
                    {
                        "source": state.source,
                        "symbol": state.symbol,
                        "reliability": round(state.reliability, 6),
                        "last_success_ts": state.last_success_ts,
                        "last_failure_ts": state.last_failure_ts,
                        "consecutive_failures": state.consecutive_failures,
                        "fallback_count": state.fallback_count,
                        "success_count": state.success_count,
                        "failure_count": state.failure_count,
                        "window_metrics": metrics,
                    }
                )

        rows.sort(key=lambda x: x["reliability"], reverse=True)
        return {"symbol": normalized_symbol, "sources": rows}

    def get_quality_report(self, symbol: Optional[str] = None, include_incidents: bool = False) -> JSONDict:
        """Backwards-compatible comprehensive report for the whole monitor."""
        sources = {source: self.get_source_report(source) for source in self._iter_known_sources()}
        report = {
            "generated_at": self._now(),
            "summary": self.get_overall_health_report(),
            "primary_sources": {
                "reliability_score": {s: sources[s]["health"]["reliability"] for s in sources},
                "freshness_score": {s: sources[s]["freshness"]["age_seconds"] for s in sources},
                "fallback_events_count": {s: sources[s]["totals"]["fallbacks"] for s in sources},
                "status": {s: sources[s]["health"]["status"] for s in sources},
            },
            "sources": sources,
            "symbol_specific": self.get_symbol_report(symbol) if symbol else {},
        }
        if include_incidents:
            report["recent_incidents"] = list(self._incidents)
        return report

    def get_overall_health_report(self) -> JSONDict:
        reports = [self.get_source_report(source) for source in self._iter_known_sources()]
        if not reports:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "source_count": 0,
                "healthy_sources": 0,
                "degraded_sources": 0,
                "failing_sources": 0,
                "circuit_open_sources": 0,
                "fallback_pressure": 0.0,
                "recommended_source_order": [],
            }

        statuses = [r["health"]["status"] for r in reports]
        fallback_pressure = mean(r["window_metrics"]["fallback_rate"] for r in reports) if reports else 0.0
        overall_status = HealthStatus.HEALTHY.value
        if any(s == HealthStatus.CIRCUIT_OPEN.value for s in statuses):
            overall_status = HealthStatus.UNSTABLE.value
        if any(s == HealthStatus.FAILING.value for s in statuses):
            overall_status = HealthStatus.FAILING.value
        elif all(s == HealthStatus.HEALTHY.value for s in statuses):
            overall_status = HealthStatus.HEALTHY.value
        elif any(s in {HealthStatus.DEGRADED.value, HealthStatus.UNSTABLE.value} for s in statuses):
            overall_status = HealthStatus.DEGRADED.value

        return {
            "status": overall_status,
            "source_count": len(reports),
            "healthy_sources": sum(1 for s in statuses if s == HealthStatus.HEALTHY.value),
            "degraded_sources": sum(1 for s in statuses if s == HealthStatus.DEGRADED.value),
            "unstable_sources": sum(1 for s in statuses if s == HealthStatus.UNSTABLE.value),
            "failing_sources": sum(1 for s in statuses if s == HealthStatus.FAILING.value),
            "circuit_open_sources": sum(1 for s in statuses if s == HealthStatus.CIRCUIT_OPEN.value),
            "fallback_pressure": round(fallback_pressure, 6),
            "recommended_source_order": self.get_preferred_sources(),
        }

    def assert_trade_data_healthy(self, source: str, *, max_age_seconds: Optional[float] = None) -> None:
        """Raise when data from a source is too stale or the source is failing.

        This is useful right before inference or order execution.
        """
        report = self.get_source_report(source)
        age_seconds = report["freshness"]["age_seconds"]
        max_age = max_age_seconds or self.thresholds.stale_trade_data_max_age_seconds
        if age_seconds is None or age_seconds > max_age:
            raise DataStalenessError(
                f"Source '{source}' data is stale for trading.",
                context=ErrorContext(component="data_quality_monitor", operation="assert_trade_data_healthy", provider=source),
                details={"age_seconds": age_seconds, "max_age_seconds": max_age, "report": report},
            )
        if report["health"]["status"] in {HealthStatus.FAILING.value, HealthStatus.CIRCUIT_OPEN.value}:
            raise TrendMonitoringError(
                f"Source '{source}' is not healthy enough for trade-critical use.",
                context=ErrorContext(component="data_quality_monitor", operation="assert_trade_data_healthy", provider=source),
                details={"report": report},
            )

    # ------------------------------------------------------------------
    # Snapshot / maintenance API
    # ------------------------------------------------------------------

    def export_snapshot(self, path: Union[str, Path]) -> Path:
        snapshot_path = Path(path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exported_at": self._now(),
            "thresholds": asdict(self.thresholds),
            "primary_sources": list(self.primary_sources),
            "summary": self.get_overall_health_report(),
            "sources": {source: self.get_source_report(source) for source in self._iter_known_sources()},
            "symbols": {
                f"{symbol}:{source}": self.get_symbol_report(symbol, source)
                for (symbol, source) in list(self._symbol_states.keys())
            },
            "recent_incidents": list(self._incidents),
        }
        snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return snapshot_path

    def reset(self, source: Optional[str] = None, symbol: Optional[str] = None) -> None:
        with self._lock:
            if source is None and symbol is None:
                self._source_states.clear()
                self._symbol_states.clear()
                self._global_events.clear()
                self._incidents.clear()
                self._counters["fallback_triggers"].clear()
                self._counters["source_reliability"].clear()
                self._counters["freshness_scores"].clear()
                now = self._now()
                for item in self.primary_sources:
                    self._source_states[item] = SourceState(
                        source=item,
                        reliability=self.thresholds.default_reliability,
                        created_at=now,
                        updated_at=now,
                        events=deque(maxlen=self.thresholds.max_event_history),
                    )
                return

            normalized_source = self._normalize_source_name(source) if source else None
            normalized_symbol = self._normalize_symbol(symbol)

            if normalized_source and normalized_symbol:
                self._symbol_states.pop((normalized_symbol, normalized_source), None)
                return
            if normalized_source:
                self._source_states.pop(normalized_source, None)
                to_delete = [key for key in self._symbol_states if key[1] == normalized_source]
                for key in to_delete:
                    self._symbol_states.pop(key, None)
                return
            if normalized_symbol:
                to_delete = [key for key in self._symbol_states if key[0] == normalized_symbol]
                for key in to_delete:
                    self._symbol_states.pop(key, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_event(self, event: EventRecord) -> None:
        with self._lock:
            source_state = self._ensure_source_state(event.source)
            symbol_state = self._ensure_symbol_state(event.symbol, event.source) if event.symbol else None
            now = event.timestamp

            self._global_events.append(event)
            source_state.events.append(event)
            self._prune_events(source_state.events, now)
            if symbol_state is not None:
                symbol_state.events.append(event)
                self._prune_events(symbol_state.events, now)

            source_state.total_events += 1
            source_state.updated_at = now

            if event.latency_ms is not None:
                source_state.latency_ewma_ms = self._update_ewma(source_state.latency_ewma_ms, event.latency_ms)
            if event.freshness_seconds is not None:
                source_state.freshness_ewma_seconds = self._update_ewma(
                    source_state.freshness_ewma_seconds,
                    event.freshness_seconds,
                )

            # Backwards-compatible counters
            self._counters["source_reliability"][event.source] = source_state.reliability
            if event.event_type == EventType.SUCCESS:
                self._counters["freshness_scores"][event.source] = now
            if event.event_type == EventType.FALLBACK:
                self._counters["fallback_triggers"][event.source] += 1

            penalty = 0.0
            if event.event_type == EventType.SUCCESS:
                source_state.total_successes += 1
                source_state.last_success_ts = now
                source_state.consecutive_failures = 0
                penalty = -self.thresholds.success_reward
            elif event.event_type == EventType.FAILURE:
                source_state.total_failures += 1
                source_state.last_failure_ts = now
                source_state.consecutive_failures += 1
                penalty = self.thresholds.failure_penalty
            elif event.event_type == EventType.FALLBACK:
                source_state.total_fallbacks += 1
                source_state.last_fallback_ts = now
                penalty = self.thresholds.fallback_penalty
            elif event.event_type == EventType.RATE_LIMIT:
                source_state.total_failures += 1
                source_state.total_rate_limits += 1
                source_state.last_failure_ts = now
                source_state.consecutive_failures += 1
                penalty = self.thresholds.rate_limit_penalty
            elif event.event_type == EventType.TIMEOUT:
                source_state.total_failures += 1
                source_state.total_timeouts += 1
                source_state.last_failure_ts = now
                source_state.consecutive_failures += 1
                penalty = self.thresholds.timeout_penalty
            elif event.event_type == EventType.PARTIAL:
                source_state.total_partials += 1
                penalty = self.thresholds.partial_penalty
            elif event.event_type == EventType.STALE:
                source_state.total_stale_events += 1
                source_state.last_failure_ts = now
                source_state.consecutive_failures += 1
                penalty = self.thresholds.stale_penalty
            elif event.event_type == EventType.VALIDATION_FAILURE:
                source_state.total_validation_failures += 1
                source_state.last_failure_ts = now
                source_state.consecutive_failures += 1
                penalty = self.thresholds.validation_penalty

            if event.error_name:
                source_state.last_error_name = event.error_name
                source_state.last_error_message = event.message
                source_state.last_status_code = event.status_code

            source_state.reliability = self._clamp_reliability(source_state.reliability - penalty)
            self._counters["source_reliability"][event.source] = source_state.reliability

            if symbol_state is not None:
                self._apply_event_to_symbol_state(symbol_state, event, penalty)

            if self._should_open_circuit(source_state, now):
                source_state.circuit_open_until = now + self.thresholds.circuit_breaker_cooldown_seconds
                incident = self._build_incident_payload(
                    source_state.source,
                    "circuit_opened",
                    event,
                    {
                        "consecutive_failures": source_state.consecutive_failures,
                        "cooldown_seconds": self.thresholds.circuit_breaker_cooldown_seconds,
                    },
                )
                self._incidents.append(incident)
                logger.warning("Opened circuit for source '%s' after repeated failures", source_state.source)
            elif source_state.circuit_open_until and source_state.circuit_open_until <= now:
                source_state.circuit_open_until = None
                source_state.consecutive_failures = 0

            self._maybe_record_incident(source_state, event)
            self._log_event(event, source_state)

    def _apply_event_to_symbol_state(self, state: SymbolState, event: EventRecord, penalty: float) -> None:
        state.updated_at = event.timestamp
        if event.event_type == EventType.SUCCESS:
            state.success_count += 1
            state.last_success_ts = event.timestamp
            state.consecutive_failures = 0
        elif event.event_type == EventType.FALLBACK:
            state.fallback_count += 1
        else:
            state.failure_count += 1
            state.last_failure_ts = event.timestamp
            state.consecutive_failures += 1
        state.reliability = self._clamp_reliability(state.reliability - penalty)

    def _maybe_record_incident(self, state: SourceState, event: EventRecord) -> None:
        if event.event_type in {EventType.FAILURE, EventType.TIMEOUT, EventType.RATE_LIMIT, EventType.STALE}:
            self._incidents.append(
                self._build_incident_payload(
                    state.source,
                    "source_issue",
                    event,
                    {
                        "reliability": state.reliability,
                        "consecutive_failures": state.consecutive_failures,
                    },
                )
            )

    def _build_incident_payload(
        self,
        source: str,
        incident_type: str,
        event: EventRecord,
        details: Optional[Mapping[str, Any]] = None,
    ) -> JSONDict:
        return {
            "timestamp": event.timestamp,
            "incident_type": incident_type,
            "source": source,
            "symbol": event.symbol,
            "event_type": event.event_type.value,
            "error_name": event.error_name,
            "error_code": event.error_code,
            "message": event.message,
            "details": {**dict(event.details or {}), **dict(details or {})},
        }

    def _ensure_source_state(self, source: str) -> SourceState:
        normalized = self._normalize_source_name(source)
        if not normalized:
            raise ValueError("source is required")
        state = self._source_states.get(normalized)
        if state is None:
            now = self._now()
            state = SourceState(
                source=normalized,
                reliability=self.thresholds.default_reliability,
                created_at=now,
                updated_at=now,
                events=deque(maxlen=self.thresholds.max_event_history),
            )
            self._source_states[normalized] = state
        return state

    def _ensure_symbol_state(self, symbol: Optional[str], source: str) -> Optional[SymbolState]:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            return None
        normalized_source = self._normalize_source_name(source)
        key = (normalized_symbol, normalized_source)
        state = self._symbol_states.get(key)
        if state is None:
            state = SymbolState(
                symbol=normalized_symbol,
                source=normalized_source,
                updated_at=self._now(),
                reliability=self.thresholds.default_reliability,
                events=deque(maxlen=self.thresholds.max_symbol_event_history),
            )
            self._symbol_states[key] = state
        return state

    def _compute_window_metrics(self, events: Deque[EventRecord], now: float) -> JSONDict:
        cutoff = now - self.thresholds.lookback_seconds
        window = [event for event in events if event.timestamp >= cutoff]
        total = len(window)
        successes = sum(1 for e in window if e.event_type == EventType.SUCCESS)
        failures = sum(1 for e in window if e.event_type in {EventType.FAILURE, EventType.TIMEOUT, EventType.RATE_LIMIT, EventType.STALE, EventType.VALIDATION_FAILURE})
        fallbacks = sum(1 for e in window if e.event_type == EventType.FALLBACK)
        partials = sum(1 for e in window if e.event_type == EventType.PARTIAL)
        timeouts = sum(1 for e in window if e.event_type == EventType.TIMEOUT)
        rate_limits = sum(1 for e in window if e.event_type == EventType.RATE_LIMIT)
        stale_count = sum(1 for e in window if e.event_type == EventType.STALE)
        latencies = sorted(e.latency_ms for e in window if e.latency_ms is not None)

        return {
            "lookback_seconds": self.thresholds.lookback_seconds,
            "events": total,
            "successes": successes,
            "failures": failures,
            "fallbacks": fallbacks,
            "partials": partials,
            "timeouts": timeouts,
            "rate_limits": rate_limits,
            "stale_events": stale_count,
            "success_rate": round(successes / total, 6) if total else 0.0,
            "error_rate": round(failures / total, 6) if total else 0.0,
            "fallback_rate": round(fallbacks / total, 6) if total else 0.0,
            "partial_rate": round(partials / total, 6) if total else 0.0,
            "avg_latency_ms": round(mean(latencies), 3) if latencies else None,
            "p50_latency_ms": self._percentile(latencies, 50),
            "p95_latency_ms": self._percentile(latencies, 95),
        }

    def _evaluate_health(
        self,
        state: SourceState,
        metrics: Mapping[str, Any],
        freshness_age: Optional[float],
        now: float,
    ) -> Tuple[HealthStatus, List[str]]:
        reasons: List[str] = []

        if state.circuit_open_until and state.circuit_open_until > now:
            reasons.append("circuit_open")
            return HealthStatus.CIRCUIT_OPEN, reasons

        if metrics["events"] == 0:
            if freshness_age is None:
                reasons.append("no_observations")
                return HealthStatus.UNKNOWN, reasons

        if freshness_age is not None:
            if freshness_age > self.thresholds.degraded_staleness_seconds:
                reasons.append("stale_data")
            elif freshness_age > self.thresholds.healthy_staleness_seconds:
                reasons.append("aging_data")

        if state.reliability <= self.thresholds.failing_reliability:
            reasons.append("low_reliability")
            return HealthStatus.FAILING, reasons
        if metrics["error_rate"] >= self.thresholds.error_rate_failing:
            reasons.append("high_error_rate")
            return HealthStatus.FAILING, reasons
        if metrics["p95_latency_ms"] is not None and metrics["p95_latency_ms"] >= self.thresholds.p95_latency_failing_ms:
            reasons.append("high_latency")
            return HealthStatus.FAILING, reasons

        if state.reliability <= self.thresholds.unstable_reliability:
            reasons.append("reliability_drift")
            return HealthStatus.UNSTABLE, reasons
        if metrics["error_rate"] >= self.thresholds.error_rate_unstable:
            reasons.append("elevated_error_rate")
            return HealthStatus.UNSTABLE, reasons
        if metrics["fallback_rate"] >= self.thresholds.fallback_rate_warning:
            reasons.append("fallback_pressure")
            return HealthStatus.UNSTABLE, reasons
        if freshness_age is not None and freshness_age > self.thresholds.degraded_staleness_seconds:
            return HealthStatus.UNSTABLE, reasons

        if state.reliability <= self.thresholds.degraded_reliability:
            reasons.append("slightly_degraded_reliability")
            return HealthStatus.DEGRADED, reasons
        if metrics["p95_latency_ms"] is not None and metrics["p95_latency_ms"] >= self.thresholds.p95_latency_warning_ms:
            reasons.append("latency_warning")
            return HealthStatus.DEGRADED, reasons
        if freshness_age is not None and freshness_age > self.thresholds.healthy_staleness_seconds:
            return HealthStatus.DEGRADED, reasons

        if not reasons:
            reasons.append("within_thresholds")
        return HealthStatus.HEALTHY, reasons

    def _should_open_circuit(self, state: SourceState, now: float) -> bool:
        if state.circuit_open_until and state.circuit_open_until > now:
            return False
        return state.consecutive_failures >= self.thresholds.circuit_breaker_failures

    def _prune_events(self, events: Deque[EventRecord], now: float) -> None:
        cutoff = now - self.thresholds.lookback_seconds
        while events and events[0].timestamp < cutoff and len(events) > 1:
            events.popleft()

    def _iter_known_sources(self) -> Tuple[str, ...]:
        with self._lock:
            sources = set(self.primary_sources) | set(self._source_states.keys())
        return tuple(sorted(sources))

    def _log_event(self, event: EventRecord, state: SourceState) -> None:
        msg = (
            "Data quality event source=%s type=%s symbol=%s reliability=%.3f "
            "consecutive_failures=%s latency_ms=%s freshness_seconds=%s"
        )
        args = (
            event.source,
            event.event_type.value,
            event.symbol,
            state.reliability,
            state.consecutive_failures,
            event.latency_ms,
            event.freshness_seconds,
        )
        if event.event_type == EventType.SUCCESS:
            logger.debug(msg, *args)
        elif event.event_type == EventType.FALLBACK:
            logger.warning(msg + " fallback_to=%s", *args, event.fallback_to)
        else:
            logger.warning(msg + " error=%s", *args, event.error_name or event.message)

    def _health_rank(self, status: str) -> int:
        order = {
            HealthStatus.HEALTHY.value: 5,
            HealthStatus.DEGRADED.value: 4,
            HealthStatus.UNKNOWN.value: 3,
            HealthStatus.UNSTABLE.value: 2,
            HealthStatus.FAILING.value: 1,
            HealthStatus.CIRCUIT_OPEN.value: 0,
        }
        return order.get(status, -1)

    def _update_ewma(self, current: Optional[float], value: float) -> float:
        if current is None:
            return float(value)
        alpha = self.thresholds.ewma_alpha
        return (alpha * float(value)) + ((1.0 - alpha) * current)

    def _clamp_reliability(self, value: float) -> float:
        return max(self.thresholds.min_reliability, min(self.thresholds.max_reliability, float(value)))

    def _normalize_source_name(self, source: Optional[str]) -> str:
        if source is None:
            return ""
        normalized = str(source).strip().lower()
        normalized = normalized.replace(" ", "")
        normalized = normalized.replace("_", "")
        normalized = normalized.replace("-", "")
        if normalized.endswith("api") and normalized not in {"polygonapi", "newsapi"}:
            normalized = normalized[:-3]
        return normalized

    def _normalize_symbol(self, symbol: Optional[str]) -> Optional[str]:
        if symbol is None:
            return None
        cleaned = str(symbol).strip().upper()
        return cleaned or None

    def _coerce_non_negative(self, value: Optional[float], field_name: str) -> Optional[float]:
        if value is None:
            return None
        numeric = float(value)
        if numeric < 0:
            raise ValueError(f"{field_name} must be non-negative")
        return numeric

    def _percentile(self, values: List[Optional[float]], percentile: float) -> Optional[float]:
        filtered = [float(v) for v in values if v is not None]
        if not filtered:
            return None
        if len(filtered) == 1:
            return round(filtered[0], 3)
        filtered.sort()
        k = (len(filtered) - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(filtered[int(k)], 3)
        d0 = filtered[f] * (c - k)
        d1 = filtered[c] * (k - f)
        return round(d0 + d1, 3)

    def _now(self) -> float:
        return float(self._time_fn())


__all__ = [
    "DataQualityMonitor",
    "EventRecord",
    "EventType",
    "HealthStatus",
    "MonitorThresholds",
    "SourceState",
    "SymbolState",
]