"""
Lightweight metric tracking for performance (timings) and memory usage changes.

This module provides a production-ready lightweight metric store for the base
agent subsystem. It is designed for low-overhead operational telemetry where
full observability stacks may be unnecessary or unavailable, while still
preserving enough structure for diagnostics, performance analysis, regression
checks, and runtime introspection.

The store focuses on timed operations, memory deltas, sampled numeric values,
counters, gauges, rolling histories, and compact summaries. It is intentionally
lightweight, but it is not trivial: it enforces consistent validation,
thread-safe state transitions, bounded retention, serialization safety,
snapshot import/export, and structured error handling aligned with the base
error taxonomy.

Key design goals:
- low-overhead tracking for hot paths and subsystem internals
- consistent metric naming and category isolation across components
- bounded retention and rolling summaries suitable for long-lived processes
- explicit start/stop semantics for timed sections and memory delta capture
- safe snapshotting/export for debugging, recovery, and offline inspection
- integration with base helpers and base error handling conventions
"""

from __future__ import annotations

import psutil

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any, Deque, Dict, Iterator, List, MutableMapping, Optional, Set, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.base_errors import *
from .utils.base_helpers import *
from .base_memory import BaseMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Light Metric Store")
printer = PrettyPrinter()


@dataclass
class MetricRecord:
    """A single metric observation captured by the store."""

    metric_name: str
    category: str
    value: float
    metric_type: str
    unit: Optional[str] = None
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "category": self.category,
            "value": self.value,
            "metric_type": self.metric_type,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": to_json_safe(self.metadata),
        }


@dataclass
class ActiveMetricSession:
    """Represents an in-flight timing/memory tracking session."""

    metric_name: str
    category: str
    started_at_iso: str
    started_monotonic: float
    start_memory_rss_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str]:
        return (self.category, self.metric_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "category": self.category,
            "started_at_iso": self.started_at_iso,
            "started_monotonic": self.started_monotonic,
            "start_memory_rss_bytes": self.start_memory_rss_bytes,
            "metadata": to_json_safe(self.metadata),
        }


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate summary for a metric series."""

    metric_name: str
    category: str
    metric_type: str
    count: int
    total: float
    minimum: float
    maximum: float
    average: float
    last_value: float
    last_timestamp: Optional[str]
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "category": self.category,
            "metric_type": self.metric_type,
            "count": self.count,
            "total": self.total,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "average": self.average,
            "last_value": self.last_value,
            "last_timestamp": self.last_timestamp,
            "unit": self.unit,
        }


class LightMetricStore:
    """Lightweight metric tracking for performance (timings) and memory usage changes."""

    _SUPPORTED_SUMMARY_FIELDS = {
        "count",
        "total",
        "minimum",
        "maximum",
        "average",
        "last_value",
    }
    _SUPPORTED_METRIC_TYPES = {
        "timing",
        "memory_delta",
        "memory_rss",
        "value",
        "counter",
        "gauge",
    }

    def __init__(self) -> None:
        self.config = load_global_config()
        self.lm_config = get_config_section("lm_store") or {}
        self._lock = RLock()
        self._process = psutil.Process()
        self._store_id = generate_request_id("lm_store")
        self._started_at = utc_now_iso()

        self.enable_memory_tracking = self._get_config_bool("enable_memory_tracking", True)
        self.default_category = self._get_config_str("default_category", "performance", normalize=True)
        self.max_history_size = self._get_config_int("max_history_size", 1000, minimum=10)
        self.max_records_per_metric = self._get_config_int("max_records_per_metric", 250, minimum=1)
        self.max_active_trackers = self._get_config_int("max_active_trackers", 2048, minimum=1)
        self.auto_prune_empty_categories = self._get_config_bool("auto_prune_empty_categories", True)
        self.enable_snapshots = self._get_config_bool("enable_snapshots", True)
        self.auto_snapshot = self._get_config_bool("auto_snapshot", False)
        self.snapshot_pretty = self._get_config_bool("snapshot_pretty", True)
        self.snapshot_path = self.lm_config.get("snapshot_path", "data/light_metric_store_snapshot.json")
        self.summary_sort_by = self._get_config_str("summary_sort_by", "average", normalize=False)
        ensure_one_of(
            self.summary_sort_by,
            list(self._SUPPORTED_SUMMARY_FIELDS),
            "summary_sort_by",
            error_cls=BaseConfigurationError,
            config=self.lm_config,
        )
        self._memory = BaseMemory()

        self.metrics: Dict[str, Dict[str, Dict[str, Deque[MetricRecord]]]] = {
            "timings": defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_records_per_metric))),
            "memory_deltas": defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_records_per_metric))),
            "memory_rss": defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_records_per_metric))),
            "values": defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_records_per_metric))),
            "counters": defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_records_per_metric))),
            "gauges": defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_records_per_metric))),
        }
        self._active_sessions: Dict[Tuple[str, str], ActiveMetricSession] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history_size)
        self._metric_units: Dict[Tuple[str, str, str], Optional[str]] = {}
        self._writes = 0
        self._reads = 0
        self._starts = 0
        self._stops = 0
        self._resets = 0
        self._snapshot_saves = 0
        self._snapshot_loads = 0

        logger.info("Light Metric Store successfully initialized")

    def _get_config_bool(self, key: str, default: bool) -> bool:
        return coerce_bool(self.lm_config.get(key, default), default=default)

    def _get_config_int(self, key: str, default: int, *, minimum: Optional[int] = None) -> int:
        value = coerce_int(self.lm_config.get(key, default), default=default, minimum=minimum)
        if minimum is not None and value < minimum:
            raise BaseConfigurationError(
                f"Configuration value '{key}' must be >= {minimum}.",
                self.lm_config,
                component="LightMetricStore",
                operation="configuration",
                context={"key": key, "value": value, "minimum": minimum},
            )
        return value

    def _get_config_str(self, key: str, default: str, *, normalize: bool = False) -> str:
        value = ensure_non_empty_string(
            self.lm_config.get(key, default),
            key,
            config=self.lm_config,
            error_cls=BaseConfigurationError,
        )
        return normalize_identifier(value, lowercase=True, separator="_", max_length=120) if normalize else value
    
    def _make_record_key(self, metric_type: str, category: str, metric_name: str) -> str:
        """Generate a unique key for a metric record."""
        timestamp = utc_now_iso().replace(":", "-")  # colon not safe for filesystem keys
        uid = uuid.uuid4().hex[:8]
        return f"{metric_type}/{category}/{metric_name}/{timestamp}_{uid}"

    def _normalize_category(self, category: Optional[str]) -> str:
        value = category or self.default_category
        return normalize_identifier(value, lowercase=True, separator="_", max_length=120)

    def _normalize_metric_name(self, metric_name: str) -> str:
        value = ensure_non_empty_string(
            metric_name,
            "metric_name",
            config=self.lm_config,
            error_cls=BaseValidationError,
        )
        normalized = normalize_identifier(value, lowercase=True, separator="_", max_length=160)
        if len(normalized) > 160:
            raise BaseValidationError(
                "metric_name must be 160 characters or fewer after normalization.",
                self.lm_config,
                component="LightMetricStore",
                operation="normalize_metric_name",
                context={"metric_name": metric_name},
            )
        return normalized

    def _normalize_metadata(self, metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if metadata is None:
            return {}
        ensure_mapping(metadata, "metadata", config=self.lm_config, error_cls=BaseValidationError)
        return drop_none_values(dict(metadata), recursive=True, drop_empty=False)

    def _history_append(self, event: str, **details: Any) -> None:
        self._history.append(
            {
                "timestamp": utc_now_iso(),
                "event": event,
                "details": to_json_safe(drop_none_values(details, recursive=True, drop_empty=False)),
            }
        )

    def _record(self, metric_type: str, record: MetricRecord) -> MetricRecord:
        # metric_type is one of "timings", "memory_deltas", "memory_rss", "values", "counters", "gauges"
        bucket_name = metric_type  # already maps
        normalized_category = self._normalize_category(record.category)
        normalized_metric = self._normalize_metric_name(record.metric_name)
    
        key = self._make_record_key(bucket_name, normalized_category, normalized_metric)
        tags = (bucket_name, normalized_category, normalized_metric)
    
        # Build metadata from record fields
        metadata = {
            "unit": record.unit,
            "original_timestamp": record.timestamp,
            **record.metadata,
        }
    
        # Store in BaseMemory with TTL (optional, can be configured via base_memory.default_ttl_seconds)
        entry = self._memory.put(
            key=key,
            value=record.value,
            namespace="lm_metrics",
            metadata=metadata,
            tags=tags,
            ttl_seconds=None,  # or allow per‑metric TTL from config
            source="light_metric_store",
            persistent=True,
        )
    
        # Enforce max_records_per_metric by pruning oldest entries for this metric
        self._prune_oldest_records(bucket_name, normalized_category, normalized_metric)
    
        self._writes += 1
        self._history_append("record_metric", metric_type=metric_type, metric_name=normalized_metric,
                             category=normalized_category, value=record.value, unit=record.unit)
        return record

    def _prune_oldest_records(self, metric_type: str, category: str, metric_name: str) -> None:
        """Delete oldest records when the count exceeds max_records_per_metric."""
        prefix = f"{metric_type}/{category}/{metric_name}/"
        # Retrieve all entries (or just keys) for this metric
        entries = self._memory.search(
            namespace="lm_metrics",
            key_prefix=prefix,
            return_entries=True,
            # include_expired=False because we only prune active records
            limit=self.max_records_per_metric + 10  # get a few extra to decide what to delete
        )
        if len(entries) <= self.max_records_per_metric:
            return

        # Entries are already sorted by key (timestamp) if we rely on BaseMemory's natural order?
        # BaseMemory.search does not guarantee order, so we sort by key which contains timestamp.
        entries.sort(key=lambda e: e.key)
        to_delete = entries[:-self.max_records_per_metric]
        for entry in to_delete:
            self._memory.delete(entry.key, namespace="lm_metrics")
    
    def _current_rss_bytes(self) -> int:
        return int(self._process.memory_info().rss)

    def _summarize_records(
        self,
        records: Iterable[MetricRecord],
        *,
        metric_name: str,
        category: str,
        metric_type: str,
        unit: Optional[str] = None,
    ) -> Optional[MetricSummary]:
        record_list = list(records)
        if not record_list:
            return None
        values = [float(item.value) for item in record_list]
        total = sum(values)
        count = len(values)
        return MetricSummary(
            metric_name=metric_name,
            category=category,
            metric_type=metric_type,
            count=count,
            total=total,
            minimum=min(values),
            maximum=max(values),
            average=(total / count) if count else 0.0,
            last_value=values[-1],
            last_timestamp=record_list[-1].timestamp,
            unit=unit,
        )

    def _maybe_snapshot(self) -> None:
        if self.enable_snapshots and self.auto_snapshot and self.snapshot_path:
            self.save_snapshot(self.snapshot_path)

    @property
    def active_tracking_count(self) -> int:
        return len(self._active_sessions)

    @property
    def total_record_count(self) -> int:
        total = 0
        for category_map in self.metrics.values():
            for metric_map in category_map.values():
                for records in metric_map.values():
                    total += len(records)
        return total

    def __len__(self) -> int:
        return self.total_record_count

    def __repr__(self) -> str:
        return (
            f"<LightMetricStore id={self._store_id} records={self.total_record_count} "
            f"active={self.active_tracking_count} default_category='{self.default_category}'>"
        )

    def start_tracking(
        self,
        metric_name: str,
        category: Optional[str] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        allow_restart: bool = False,
    ) -> ActiveMetricSession:
        normalized_metric = self._normalize_metric_name(metric_name)
        normalized_category = self._normalize_category(category)
        session_key = (normalized_category, normalized_metric)

        with self._lock:
            if session_key in self._active_sessions and not allow_restart:
                raise BaseStateError(
                    "Tracking already started for the specified metric.",
                    self.lm_config,
                    component="LightMetricStore",
                    operation="start_tracking",
                    context={"metric_name": normalized_metric, "category": normalized_category},
                )
            if len(self._active_sessions) >= self.max_active_trackers:
                raise BaseResourceError(
                    "Maximum active tracker count exceeded.",
                    self.lm_config,
                    component="LightMetricStore",
                    operation="start_tracking",
                    context={
                        "max_active_trackers": self.max_active_trackers,
                        "active_tracking_count": len(self._active_sessions),
                    },
                )

            session = ActiveMetricSession(
                metric_name=normalized_metric,
                category=normalized_category,
                started_at_iso=utc_now_iso(),
                started_monotonic=monotonic_seconds(),
                start_memory_rss_bytes=self._current_rss_bytes() if self.enable_memory_tracking else 0,
                metadata=self._normalize_metadata(metadata),
            )
            self._active_sessions[session_key] = session
            self._starts += 1
            self._history_append(
                "start_tracking",
                metric_name=normalized_metric,
                category=normalized_category,
                metadata=session.metadata,
            )
            return ActiveMetricSession(**session.to_dict())

    def stop_tracking(
        self,
        metric_name: str,
        category: Optional[str] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        record_memory_rss: bool = True,
    ) -> Dict[str, MetricRecord]:
        normalized_metric = self._normalize_metric_name(metric_name)
        normalized_category = self._normalize_category(category)
        session_key = (normalized_category, normalized_metric)

        with self._lock:
            session = self._active_sessions.pop(session_key, None)
            if session is None:
                raise BaseStateError(
                    "Tracking was stopped for a metric that was not started.",
                    self.lm_config,
                    component="LightMetricStore",
                    operation="stop_tracking",
                    context={"metric_name": normalized_metric, "category": normalized_category},
                )

            elapsed_seconds = max(0.0, monotonic_seconds() - session.started_monotonic)
            merged_metadata = deep_merge_dicts(session.metadata, self._normalize_metadata(metadata))
            timing_record = self._record(
                "timings",
                MetricRecord(
                    metric_name=normalized_metric,
                    category=normalized_category,
                    value=elapsed_seconds,
                    metric_type="timing",
                    unit="seconds",
                    metadata=merged_metadata,
                ),
            )
            result: Dict[str, MetricRecord] = {"timing": timing_record}

            if self.enable_memory_tracking:
                current_rss_bytes = self._current_rss_bytes()
                memory_delta_mb = (current_rss_bytes - session.start_memory_rss_bytes) / float(1024 ** 2)
                result["memory_delta"] = self._record(
                    "memory_deltas",
                    MetricRecord(
                        metric_name=normalized_metric,
                        category=normalized_category,
                        value=memory_delta_mb,
                        metric_type="memory_delta",
                        unit="MB",
                        metadata=merged_metadata,
                    ),
                )
                if record_memory_rss:
                    result["memory_rss"] = self._record(
                        "memory_rss",
                        MetricRecord(
                            metric_name=normalized_metric,
                            category=normalized_category,
                            value=current_rss_bytes / float(1024 ** 2),
                            metric_type="memory_rss",
                            unit="MB",
                            metadata=merged_metadata,
                        ),
                    )

            self._stops += 1
            self._history_append(
                "stop_tracking",
                metric_name=normalized_metric,
                category=normalized_category,
                elapsed_seconds=elapsed_seconds,
                recorded=list(result.keys()),
            )
            self._maybe_snapshot()
            return {key: value for key, value in result.items()}

    @contextmanager
    def track(
        self,
        metric_name: str,
        category: Optional[str] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[ActiveMetricSession]:
        session = self.start_tracking(metric_name, category=category, metadata=metadata)
        try:
            yield session
        finally:
            self.stop_tracking(metric_name, category=category)

    def time_callable(
        self,
        metric_name: str,
        func: Any,
        *args: Any,
        category: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        ensure_callable(func, "func", config=self.lm_config, error_cls=BaseValidationError)
        with self.track(metric_name, category=category, metadata=metadata):
            return func(*args, **kwargs)

    def record_value(
        self,
        metric_name: str,
        value: float,
        *,
        category: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MetricRecord:
        numeric_value = ensure_numeric_range(
            value,
            "value",
            config=self.lm_config,
            error_cls=BaseValidationError,
        )
        normalized_metric = self._normalize_metric_name(metric_name)
        normalized_category = self._normalize_category(category)
        with self._lock:
            record = self._record(
                "values",
                MetricRecord(
                    metric_name=normalized_metric,
                    category=normalized_category,
                    value=float(numeric_value),
                    metric_type="value",
                    unit=unit,
                    metadata=self._normalize_metadata(metadata),
                ),
            )
            self._maybe_snapshot()
            return record

    def increment_counter(
        self,
        metric_name: str,
        amount: float = 1.0,
        *,
        category: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MetricRecord:
        delta = ensure_numeric_range(amount, "amount", config=self.lm_config, error_cls=BaseValidationError)
        normalized_metric = self._normalize_metric_name(metric_name)
        normalized_category = self._normalize_category(category)
        with self._lock:
            existing = self.metrics["counters"][normalized_category][normalized_metric]
            previous = existing[-1].value if existing else 0.0
            record = self._record(
                "counters",
                MetricRecord(
                    metric_name=normalized_metric,
                    category=normalized_category,
                    value=float(previous + delta),
                    metric_type="counter",
                    unit=unit,
                    metadata=self._normalize_metadata(metadata),
                ),
            )
            self._maybe_snapshot()
            return record

    def set_gauge(
        self,
        metric_name: str,
        value: float,
        *,
        category: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MetricRecord:
        numeric_value = ensure_numeric_range(value, "value", config=self.lm_config, error_cls=BaseValidationError)
        normalized_metric = self._normalize_metric_name(metric_name)
        normalized_category = self._normalize_category(category)
        with self._lock:
            record = self._record(
                "gauges",
                MetricRecord(
                    metric_name=normalized_metric,
                    category=normalized_category,
                    value=float(numeric_value),
                    metric_type="gauge",
                    unit=unit,
                    metadata=self._normalize_metadata(metadata),
                ),
            )
            self._maybe_snapshot()
            return record

    def get_metric_records(self, metric_name: str, *, category: Optional[str] = None,
                           metric_type: Optional[str] = None) -> List[Dict[str, Any]]:
        normalized_metric = self._normalize_metric_name(metric_name)
        normalized_category = self._normalize_category(category)
    
        with self._lock:
            self._reads += 1
            if metric_type:
                ensure_one_of(metric_type, self._SUPPORTED_METRIC_TYPES, "metric_type",
                              config=self.lm_config)
                bucket_name = self._bucket_name_for_metric_type(metric_type)
                prefix = f"{bucket_name}/{normalized_category}/{normalized_metric}/"
                entries = self._memory.search(
                    namespace="lm_metrics",
                    key_prefix=prefix,
                    return_entries=True,
                    include_expired=False
                )
            else:
                # Search across all metric types
                entries = []
                for mt in self._SUPPORTED_METRIC_TYPES:
                    bucket = self._bucket_name_for_metric_type(mt)
                    prefix = f"{bucket}/{normalized_category}/{normalized_metric}/"
                    entries.extend(self._memory.search(
                        namespace="lm_metrics", key_prefix=prefix, return_entries=True
                    ))
            # Convert MemoryEntry back to record dict
            records = []
            for entry in entries:
                record_dict = {
                    "metric_name": entry.tags[2],  # metric_name is third tag
                    "category": entry.tags[1],
                    "value": entry.value,
                    "metric_type": entry.tags[0],
                    "unit": entry.metadata.get("unit"),
                    "timestamp": entry.metadata.get("original_timestamp", entry.created_at),
                    "metadata": entry.metadata,
                }
                records.append(record_dict)
            return records

    def _bucket_name_for_metric_type(self, metric_type: str) -> str:
        mapping = {
            "timing": "timings",
            "memory_delta": "memory_deltas",
            "memory_rss": "memory_rss",
            "value": "values",
            "counter": "counters",
            "gauge": "gauges",
        }
        return mapping[metric_type]

    def get_metric_summary(self, metric_name: str, *, category: Optional[str] = None,
                           metric_type: str = "timing") -> Optional[MetricSummary]:
        normalized_metric = self._normalize_metric_name(metric_name)
        normalized_category = self._normalize_category(category)
        bucket_name = self._bucket_name_for_metric_type(metric_type)
    
        prefix = f"{bucket_name}/{normalized_category}/{normalized_metric}/"
        entries = self._memory.search(
            namespace="lm_metrics",
            key_prefix=prefix,
            return_entries=True,
            include_expired=False
        )
        if not entries:
            return None
    
        # Sort by timestamp (from key) to get chronological order
        entries.sort(key=lambda e: e.key)
        values = [e.value for e in entries]
        unit = entries[0].metadata.get("unit") if entries else None
    
        return MetricSummary(
            metric_name=normalized_metric,
            category=normalized_category,
            metric_type=metric_type,
            count=len(values),
            total=sum(values),
            minimum=min(values),
            maximum=max(values),
            average=sum(values)/len(values),
            last_value=values[-1],
            last_timestamp=entries[-1].metadata.get("original_timestamp", entries[-1].created_at),
            unit=unit,
        )
    
    def get_metrics_summary(self, category: Optional[str] = None) -> Dict[str, Any]:
        normalized_category = self._normalize_category(category)
        summary = {bucket: {} for bucket in ["timings", "memory_deltas", "memory_rss",
                                             "values", "counters", "gauges"]}
        summary["category"] = normalized_category
    
        with self._lock:
            self._reads += 1
            # Discover all metric names for this category
            for bucket in summary.keys():
                prefix = f"{bucket}/{normalized_category}/"
                # Use search with key_prefix to get unique metric names
                entries = self._memory.search(
                    namespace="lm_metrics", key_prefix=prefix, return_entries=True,
                    limit=10000  # high enough to get all
                )
                # Group by metric_name (third tag component)
                by_metric = {}
                for e in entries:
                    metric_name = e.tags[2]
                    by_metric.setdefault(metric_name, []).append(e)
                for metric_name, metric_entries in by_metric.items():
                    metric_entries.sort(key=lambda e: e.key)
                    values = [e.value for e in metric_entries]
                    unit = metric_entries[0].metadata.get("unit")
                    metric_summary = MetricSummary(
                        metric_name=metric_name,
                        category=normalized_category,
                        metric_type=bucket.rstrip('s'),  # e.g. "timings" -> "timing"
                        count=len(values),
                        total=sum(values),
                        minimum=min(values),
                        maximum=max(values),
                        average=sum(values)/len(values),
                        last_value=values[-1],
                        last_timestamp=metric_entries[-1].metadata.get("original_timestamp",
                                                                       metric_entries[-1].created_at),
                        unit=unit,
                    )
                    summary[bucket][metric_name] = metric_summary.to_dict()
        return summary

    def get_all_summaries(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        with self._lock:
            categories = set()
            for bucket in self.metrics.values():
                categories.update(bucket.keys())
        return {category: self.get_metrics_summary(category) for category in sorted(categories)}

    def recent_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        bounded = coerce_int(limit, 50, minimum=1)
        with self._lock:
            return list(self._history)[-bounded:]

    def list_categories(self) -> List[str]:
        with self._lock:
            categories: Set[str] = set()
            for bucket in self.metrics.values():
                categories.update(bucket.keys())
            categories.update(session.category for session in self._active_sessions.values())
            return sorted(categories)

    def list_metrics(self, category: Optional[str] = None) -> List[str]:
        normalized_category = self._normalize_category(category)
        names: Set[str] = set()
        with self._lock:
            for bucket in self.metrics.values():
                names.update(bucket.get(normalized_category, {}).keys())
            for current_category, metric_name in self._active_sessions.keys():
                if current_category == normalized_category:
                    names.add(metric_name)
        return sorted(names)

    def active_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [session.to_dict() for session in self._active_sessions.values()]

    def reset_metric(self, metric_name: str, *, category: Optional[str] = None,
                     metric_type: Optional[str] = None) -> int:
        normalized_metric = self._normalize_metric_name(metric_name)
        normalized_category = self._normalize_category(category)
        removed = 0
        with self._lock:
            types_to_clear = [self._bucket_name_for_metric_type(metric_type)] if metric_type else list(self._SUPPORTED_METRIC_TYPES)
            for mt in types_to_clear:
                prefix = f"{mt}/{normalized_category}/{normalized_metric}/"
                entries = self._memory.search(namespace="lm_metrics", key_prefix=prefix,
                                              return_entries=True, include_expired=True)
                for entry in entries:
                    if self._memory.delete(entry.key, namespace="lm_metrics"):
                        removed += 1
            # Also remove from active sessions if present
            active_key = (normalized_category, normalized_metric)
            self._active_sessions.pop(active_key, None)
            self._resets += 1
            self._history_append("reset_metric", metric_name=normalized_metric,
                                 category=normalized_category, removed=removed)
            return removed
    
    def clear(self, *, category: Optional[str] = None) -> Dict[str, int]:
        with self._lock:
            removed = {"records": 0, "active_sessions": 0}
            if category is None:
                # Delete entire namespace in BaseMemory
                removed["records"] = len(self._memory.search(namespace="lm_metrics", limit=1000000))
                self._memory.clear(namespace="lm_metrics")
                self._active_sessions.clear()
            else:
                normalized_category = self._normalize_category(category)
                # Delete all records with any metric_type under this category
                for mt in self._SUPPORTED_METRIC_TYPES:
                    bucket = self._bucket_name_for_metric_type(mt)
                    prefix = f"{bucket}/{normalized_category}/"
                    entries = self._memory.search(namespace="lm_metrics", key_prefix=prefix,
                                                  return_entries=True)
                    for entry in entries:
                        if self._memory.delete(entry.key, namespace="lm_metrics"):
                            removed["records"] += 1
                # Remove active sessions for this category
                keys_to_pop = [k for k in self._active_sessions if k[0] == normalized_category]
                removed["active_sessions"] = len(keys_to_pop)
                for k in keys_to_pop:
                    self._active_sessions.pop(k)
            self._resets += 1
            self._history_append("clear", category=category, removed=removed)
            return removed

    def compact(self) -> Dict[str, int]:
        with self._lock:
            expired = self._memory.cleanup_expired(force=True)
            # Remove any empty namespaces from index (BaseMemory does not auto‑prune empty namespaces)
            self._memory.compact()
            self._history_append("compact", removed_expired=expired)
            return {"removed_categories": 0, "removed_expired": expired}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            mem_stats = self._memory.stats()
            category_count = len(self.list_categories())  # still useful
            return {
                "store_id": self._store_id,
                "started_at": self._started_at,
                "uptime_seconds": max(0.0, (utc_now() - datetime.fromisoformat(self._started_at.replace("Z", "+00:00"))).total_seconds()),
                "categories": category_count,
                "total_record_count": mem_stats.total_entries,
                "active_tracking_count": len(self._active_sessions),
                "writes": self._writes,
                "reads": self._reads,
                "starts": self._starts,
                "stops": self._stops,
                "resets": self._resets,
                "snapshot_saves": self._snapshot_saves,
                "snapshot_loads": self._snapshot_loads,
                "max_history_size": self.max_history_size,
                "max_records_per_metric": self.max_records_per_metric,
                "base_memory_stats": mem_stats.to_dict(),
            }

    def to_serializable(self) -> Dict[str, Any]:
        # Simply export the entire BaseMemory snapshot plus our in‑memory state
        mem_dict = self._memory.to_dict(include_values=True, redact=False)
        return {
            "meta": {
                "store_id": self._store_id,
                "started_at": self._started_at,
                "exported_at": utc_now_iso(),
                "default_category": self.default_category,
                "config": to_json_safe(self.lm_config),
                "stats": self.stats(),
            },
            "base_memory_snapshot": mem_dict,  # full memory state
            "active_sessions": [s.to_dict() for s in self._active_sessions.values()],
            "history": list(self._history),
        }
    
    def get_all_metrics_json(self, pretty: bool = True) -> str:
        try:
            return self._memory.to_json(include_values=True, pretty=pretty, redact=False)
        except Exception as exc:
            raise BaseSerializationError.wrap(
                exc,
                message="Failed to export LightMetricStore metrics to JSON.",
                config=self.lm_config,
                component="LightMetricStore",
                operation="get_all_metrics_json",
            ) from exc

    def save_snapshot(self, path: Optional[str] = None) -> str:
        if not self.enable_snapshots:
            raise BaseStateError(
                "Snapshot support is disabled for this LightMetricStore instance.",
                self.lm_config,
                component="LightMetricStore",
                operation="save_snapshot",
            )
        # Delegate entirely to BaseMemory
        snapshot_path = self._memory.save_snapshot(path or self.snapshot_path,
                                                   include_values=True,
                                                   redact=None)
        self._snapshot_saves += 1
        self._history_append("save_snapshot", path=snapshot_path)
        return snapshot_path
    
    def load_snapshot(self, path: Optional[str] = None, *, merge: bool = False) -> int:
        # Delegate to BaseMemory; note: merge=False will clear existing metrics in BaseMemory
        loaded = self._memory.load_snapshot(path or self.snapshot_path, merge=merge, must_exist=True)
        # Also clear any in‑memory caches that became stale (e.g. _metric_units)
        self._metric_units.clear()
        # Optionally rebuild any transient indices if needed
        self._snapshot_loads += 1
        self._history_append("load_snapshot", path=path or self.snapshot_path, merge=merge,
                             loaded_records=loaded)
        return loaded


if __name__ == "__main__":
    print("\n=== Running Light Metric Store ===\n")
    printer.status("TEST", "Light Metric Store initialized", "info")

    store = LightMetricStore()
    printer.pretty("CONFIG", store.lm_config, "info")

    store.start_tracking("planner_cycle", category="runtime", metadata={"phase": "boot"})
    monotonic_seconds()
    stop_result = store.stop_tracking("planner_cycle", category="runtime", metadata={"status": "ok"})
    printer.pretty("STOP_RESULT", {key: value.to_dict() for key, value in stop_result.items()}, "success")

    with store.track("embedding_lookup", category="runtime", metadata={"kind": "context"}):
        total = sum(range(5000))
        store.record_value("lookup_items", 5000, category="runtime", unit="count")
        store.increment_counter("embedding_calls", 1, category="runtime", unit="count")
        store.set_gauge("lookup_total", float(total), category="runtime", unit="units")

    timing_summary = store.get_metric_summary("planner_cycle", category="runtime", metric_type="timing")
    category_summary = store.get_metrics_summary("runtime")
    stats = store.stats()
    history = store.recent_history(limit=10)

    printer.pretty("TIMING_SUMMARY", timing_summary.to_dict() if timing_summary else None, "success")
    printer.pretty("CATEGORY_SUMMARY", category_summary, "success")
    printer.pretty("STATS", stats, "success")
    printer.pretty("HISTORY", history, "info")

    snapshot_path = store.save_snapshot("light_metric_store_test_snapshot.json")
    printer.status("SAVE", f"Snapshot saved to {snapshot_path}", "success")

    restored = LightMetricStore()
    restored.load_snapshot("light_metric_store_test_snapshot.json", merge=False)
    printer.pretty("RESTORED_STATS", restored.stats(), "success")

    if Path("light_metric_store_test_snapshot.json").exists():
        Path("light_metric_store_test_snapshot.json").unlink()

    print("\n=== Test ran successfully ===\n")
