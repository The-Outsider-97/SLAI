"""
Communication KPIs and per-channel telemetry emission.

This module owns the network subsystem's runtime telemetry surface. It provides
structured metric aggregation for channels, endpoints, routes, and global
network activity without taking over transport execution, routing strategy,
reliability policy, or delivery lifecycle state management.

Its responsibilities are deliberately focused on measurement and telemetry:

- recording transport and delivery outcomes,
- aggregating per-channel and per-endpoint KPIs,
- maintaining bounded event and latency histories,
- deriving health-oriented summaries from observed traffic,
- exposing JSON-safe snapshots for observability and debugging,
- and publishing telemetry snapshots into `NetworkMemory` for broader network
  coordination.

The metrics layer is intentionally integration-friendly. It reuses the shared
network helper and error abstractions so telemetry can be emitted safely from
other network modules without duplicating validation, serialization, or error
normalization behavior.
"""

from __future__ import annotations

import math

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils import *
from .network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Network Metrics")
printer = PrettyPrinter()


_TELEMETRY_ENDPOINT_METRICS_KEY = "network.telemetry.endpoint_metrics"
_TELEMETRY_ROUTE_METRICS_KEY = "network.telemetry.route_metrics"
_TELEMETRY_GLOBAL_METRICS_KEY = "network.telemetry.global_metrics"
_TELEMETRY_RECENT_EVENTS_KEY = "network.telemetry.recent_events"
_TELEMETRY_BROADCAST_MAP_KEY = "network.telemetry.broadcast_map"


@dataclass(slots=True)
class TelemetryEvent:
    """
    Normalized telemetry event captured by the network metrics layer.

    Events are stored in a JSON-safe representation so they can be reused for
    structured logs, in-memory telemetry snapshots, and observability exports.
    """

    event_id: str
    occurred_at: datetime
    channel: str
    protocol: str
    endpoint: Optional[str] = None
    route: Optional[str] = None
    operation: Optional[str] = None
    success: bool = True
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    retry_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    delivery_state: Optional[str] = None
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    fanout_total: Optional[int] = None
    fanout_success: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, sanitize_logs_enabled: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "event_id": self.event_id,
            "occurred_at": self.occurred_at.isoformat(),
            "channel": self.channel,
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "route": self.route,
            "operation": self.operation,
            "success": self.success,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "delivery_state": self.delivery_state,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "fanout_total": self.fanout_total,
            "fanout_success": self.fanout_success,
            "error": self.error,
            "metadata": self.metadata,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if sanitize_logs_enabled:
            return sanitize_for_logging(payload)
        return json_safe(payload)


@dataclass(slots=True)
class RollingLatencyWindow:
    """Bounded latency sample window with percentile calculations."""

    max_samples: int
    samples: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=max(1, int(self.max_samples)))

    def add(self, value_ms: float) -> None:
        if value_ms < 0:
            raise PayloadValidationError(
                "Latency values must be non-negative.",
                context={"operation": "metrics_add_latency"},
                details={"latency_ms": value_ms},
            )
        self.samples.append(float(value_ms))

    def clear(self) -> None:
        self.samples.clear()

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"count": 0}

        ordered = sorted(self.samples)
        count = len(ordered)
        total = sum(ordered)
        return {
            "count": count,
            "min": round(ordered[0], 4),
            "max": round(ordered[-1], 4),
            "avg": round(total / count, 4),
            "p50": round(self._percentile(ordered, 50.0), 4),
            "p90": round(self._percentile(ordered, 90.0), 4),
            "p95": round(self._percentile(ordered, 95.0), 4),
            "p99": round(self._percentile(ordered, 99.0), 4),
            "last": round(float(self.samples[-1]), 4),
        }

    @staticmethod
    def _percentile(ordered_values: Sequence[float], percentile: float) -> float:
        if not ordered_values:
            raise PayloadValidationError(
                "Cannot compute a percentile for an empty latency window.",
                context={"operation": "metrics_percentile"},
                details={"percentile": percentile},
            )
        if len(ordered_values) == 1:
            return float(ordered_values[0])

        rank = (len(ordered_values) - 1) * (percentile / 100.0)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return float(ordered_values[lower])

        lower_value = float(ordered_values[lower])
        upper_value = float(ordered_values[upper])
        return lower_value + (upper_value - lower_value) * (rank - lower)


@dataclass(slots=True)
class MetricAggregate:
    """Mutable KPI bucket used for channel, endpoint, route, and global scopes."""

    scope: str
    key: str
    max_latency_samples: int
    first_seen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    last_status_code: Optional[int] = None
    last_error: Optional[Dict[str, Any]] = None
    total_events: int = 0
    success_total: int = 0
    failure_total: int = 0
    timeout_total: int = 0
    events_with_retry_total: int = 0
    retry_attempts_total: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    bytes_sent_total: int = 0
    bytes_received_total: int = 0
    fanout_total: int = 0
    fanout_success_total: int = 0
    operation_counts: Counter = field(default_factory=Counter)
    status_code_counts: Counter = field(default_factory=Counter)
    delivery_state_counts: Counter = field(default_factory=Counter)
    error_code_counts: Counter = field(default_factory=Counter)
    latency: RollingLatencyWindow = field(init=False)

    def __post_init__(self) -> None:
        self.latency = RollingLatencyWindow(self.max_latency_samples)

    def record(self, event: TelemetryEvent) -> None:
        self.total_events += 1
        self.last_updated_at = event.occurred_at
        self.last_status_code = event.status_code

        if event.success:
            self.success_total += 1
            self.last_success_at = event.occurred_at
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.failure_total += 1
            self.last_failure_at = event.occurred_at
            self.consecutive_failures += 1
            self.consecutive_successes = 0

        if event.retry_count > 0:
            self.events_with_retry_total += 1
            self.retry_attempts_total += int(event.retry_count)

        if event.latency_ms is not None:
            self.latency.add(float(event.latency_ms))

        if event.operation:
            self.operation_counts[str(event.operation)] += 1
        if event.status_code is not None:
            self.status_code_counts[str(int(event.status_code))] += 1
        if event.delivery_state:
            self.delivery_state_counts[str(event.delivery_state)] += 1

        if event.error is not None:
            self.last_error = json_safe(event.error)
            error_code = (
                event.error.get("error_code")
                or event.error.get("code")
                or event.error.get("type")
                or "UNKNOWN"
            )
            self.error_code_counts[str(error_code)] += 1
            if self._is_timeout_error(event.error):
                self.timeout_total += 1
        elif event.status_code in {408, 504}:
            self.timeout_total += 1

        self.bytes_sent_total += int(event.bytes_sent)
        self.bytes_received_total += int(event.bytes_received)

        if event.fanout_total is not None:
            self.fanout_total += int(event.fanout_total)
        if event.fanout_success is not None:
            self.fanout_success_total += int(event.fanout_success)

    def to_snapshot(self) -> Dict[str, Any]:
        latency_snapshot = self.latency.summary()
        success_rate = (self.success_total / self.total_events) if self.total_events else None
        error_rate = (self.failure_total / self.total_events) if self.total_events else None
        retry_rate = (self.events_with_retry_total / self.total_events) if self.total_events else None
        fanout_success_rate = None
        if self.fanout_total > 0:
            fanout_success_rate = self.fanout_success_total / self.fanout_total

        snapshot: Dict[str, Any] = {
            "scope": self.scope,
            "key": self.key,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at is not None else None,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at is not None else None,
            "last_status_code": self.last_status_code,
            "last_error": json_safe(self.last_error),
            "total_events": self.total_events,
            "success_total": self.success_total,
            "failure_total": self.failure_total,
            "success_rate": round(success_rate, 6) if success_rate is not None else None,
            "error_rate": round(error_rate, 6) if error_rate is not None else None,
            "timeout_total": self.timeout_total,
            "events_with_retry_total": self.events_with_retry_total,
            "retry_attempts_total": self.retry_attempts_total,
            "retry_rate": round(retry_rate, 6) if retry_rate is not None else None,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "bytes_sent_total": self.bytes_sent_total,
            "bytes_received_total": self.bytes_received_total,
            "avg_bytes_sent": round(self.bytes_sent_total / self.total_events, 4) if self.total_events else None,
            "avg_bytes_received": round(self.bytes_received_total / self.total_events, 4) if self.total_events else None,
            "fanout_total": self.fanout_total,
            "fanout_success_total": self.fanout_success_total,
            "fanout_success_rate": round(fanout_success_rate, 6) if fanout_success_rate is not None else None,
            "latency_ms": latency_snapshot,
            "operation_counts": dict(self.operation_counts),
            "status_code_counts": dict(self.status_code_counts),
            "delivery_state_counts": dict(self.delivery_state_counts),
            "error_code_counts": dict(self.error_code_counts),
        }
        return {key: value for key, value in snapshot.items() if value not in (None, {}, [])}

    @staticmethod
    def _is_timeout_error(error_payload: Mapping[str, Any]) -> bool:
        error_code = str(error_payload.get("error_code") or error_payload.get("code") or "").upper()
        error_type = str(error_payload.get("error_type") or error_payload.get("type") or "").upper()
        return "TIMEOUT" in error_code or "TIMEOUT" in error_type


class NetworkMetrics:
    """
    Runtime metric aggregator for the network subsystem.

    The class tracks communication KPIs across several scopes:
    - global network activity,
    - channel/protocol families,
    - endpoints,
    - and route identifiers.

    It also publishes structured snapshots into `NetworkMemory` so routing,
    reliability, policy, and observability layers can consume consistent
    telemetry without having to recalculate it.
    """

    def __init__(self, memory: Optional[NetworkMemory] = None) -> None:
        self.config = load_global_config()
        self.metrics_config = get_config_section("network_metrics") or {}
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.emit_memory_snapshots = self._get_bool_config("emit_memory_snapshots", True)
        self.store_endpoint_metrics = self._get_bool_config("store_endpoint_metrics", True)
        self.store_route_metrics = self._get_bool_config("store_route_metrics", True)
        self.store_recent_events_in_memory = self._get_bool_config("store_recent_events_in_memory", True)

        self.default_snapshot_ttl_seconds = self._get_non_negative_int_config("default_snapshot_ttl_seconds", 600)
        self.channel_snapshot_ttl_seconds = self._get_non_negative_int_config("channel_snapshot_ttl_seconds", 600)
        self.endpoint_snapshot_ttl_seconds = self._get_non_negative_int_config("endpoint_snapshot_ttl_seconds", 600)
        self.route_snapshot_ttl_seconds = self._get_non_negative_int_config("route_snapshot_ttl_seconds", 600)
        self.global_snapshot_ttl_seconds = self._get_non_negative_int_config("global_snapshot_ttl_seconds", 600)
        self.recent_events_snapshot_ttl_seconds = self._get_non_negative_int_config("recent_events_snapshot_ttl_seconds", 300)

        self.max_event_history_size = max(1, self._get_non_negative_int_config("max_event_history_size", 2000))
        self.max_broadcast_history_size = max(1, self._get_non_negative_int_config("max_broadcast_history_size", 250))
        self.max_latency_samples_per_scope = max(1, self._get_non_negative_int_config("max_latency_samples_per_scope", 500))

        self.degraded_latency_ms_threshold = float(self.metrics_config.get("degraded_latency_ms_threshold", 1000.0) or 1000.0)
        self.degraded_success_rate_threshold = float(self.metrics_config.get("degraded_success_rate_threshold", 0.95) or 0.95)
        self.degraded_error_rate_threshold = float(self.metrics_config.get("degraded_error_rate_threshold", 0.10) or 0.10)
        self.unhealthy_success_rate_threshold = float(self.metrics_config.get("unhealthy_success_rate_threshold", 0.80) or 0.80)
        self.unhealthy_error_rate_threshold = float(self.metrics_config.get("unhealthy_error_rate_threshold", 0.25) or 0.25)
        self.unhealthy_consecutive_failures_threshold = max(
            1,
            int(self.metrics_config.get("unhealthy_consecutive_failures_threshold", 5) or 5),
        )

        self._started_at = utcnow()
        self._global = MetricAggregate("global", "network", self.max_latency_samples_per_scope)
        self._channels: Dict[str, MetricAggregate] = {}
        self._endpoints: Dict[str, MetricAggregate] = {}
        self._routes: Dict[str, MetricAggregate] = {}
        self._recent_events: Deque[Dict[str, Any]] = deque(maxlen=self.max_event_history_size)
        self._broadcast_map: Deque[Dict[str, Any]] = deque(maxlen=self.max_broadcast_history_size)
        self._stats: Dict[str, int] = {
            "events_recorded": 0,
            "memory_emits": 0,
            "broadcast_events": 0,
            "errors_normalized": 0,
        }

    # ------------------------------------------------------------------ #
    # Public recording API
    # ------------------------------------------------------------------ #
    def record_event(
        self,
        *,
        channel: str,
        endpoint: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        operation: Optional[str] = None,
        success: bool = True,
        status_code: Optional[int] = None,
        latency_ms: Optional[float] = None,
        retry_count: int = 0,
        bytes_sent: Optional[int] = None,
        bytes_received: Optional[int] = None,
        delivery_state: Optional[str] = None,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        fanout_total: Optional[int] = None,
        fanout_success: Optional[int] = None,
        error: Optional[BaseException | Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a normalized network telemetry event.

        This is the primary ingestion point for adapters, lifecycle handlers,
        reliability flows, and higher-level network orchestration.
        """
        try:
            if not self.enabled:
                return {"enabled": False, "recorded": False}

            event = self._build_event(
                channel=channel,
                endpoint=endpoint,
                protocol=protocol,
                route=route,
                operation=operation,
                success=success,
                status_code=status_code,
                latency_ms=latency_ms,
                retry_count=retry_count,
                bytes_sent=bytes_sent,
                bytes_received=bytes_received,
                delivery_state=delivery_state,
                message_id=message_id,
                correlation_id=correlation_id,
                fanout_total=fanout_total,
                fanout_success=fanout_success,
                error=error,
                metadata=metadata,
            )

            with self._lock:
                self._global.record(event)
                channel_bucket = self._get_or_create_bucket(self._channels, "channel", event.channel)
                channel_bucket.record(event)

                endpoint_bucket = None
                if event.endpoint:
                    endpoint_bucket = self._get_or_create_bucket(self._endpoints, "endpoint", event.endpoint)
                    endpoint_bucket.record(event)

                route_bucket = None
                if event.route:
                    route_bucket = self._get_or_create_bucket(self._routes, "route", event.route)
                    route_bucket.record(event)

                event_payload = event.to_dict(sanitize_logs_enabled=self.sanitize_logs)
                self._recent_events.append(event_payload)
                self._stats["events_recorded"] += 1

                result = {
                    "recorded": True,
                    "event": event_payload,
                    "global": self._global.to_snapshot(),
                    "channel": channel_bucket.to_snapshot(),
                    "endpoint": endpoint_bucket.to_snapshot() if endpoint_bucket is not None else None,
                    "route": route_bucket.to_snapshot() if route_bucket is not None else None,
                }

            if self.emit_memory_snapshots:
                self._publish_event_to_memory(
                    event,
                    channel_snapshot=result["channel"],
                    endpoint_snapshot=result.get("endpoint"),
                    route_snapshot=result.get("route"),
                    global_snapshot=result["global"],
                )
            return {k: v for k, v in result.items() if v is not None}
        except NetworkError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary normalization is intentional.
            raise normalize_network_exception(
                exc,
                operation="record_metric_event",
                channel=channel,
                endpoint=endpoint,
                protocol=protocol,
                route=route,
                correlation_id=correlation_id,
                metadata=metadata,
            ) from exc

    def record_success(
        self,
        *,
        channel: str,
        endpoint: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        operation: Optional[str] = None,
        status_code: Optional[int] = None,
        latency_ms: Optional[float] = None,
        retry_count: int = 0,
        bytes_sent: Optional[int] = None,
        bytes_received: Optional[int] = None,
        delivery_state: Optional[str] = None,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper for successful network outcomes."""
        return self.record_event(
            channel=channel,
            endpoint=endpoint,
            protocol=protocol,
            route=route,
            operation=operation,
            success=True,
            status_code=status_code,
            latency_ms=latency_ms,
            retry_count=retry_count,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            delivery_state=delivery_state,
            message_id=message_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

    def record_failure(
        self,
        *,
        channel: str,
        endpoint: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        operation: Optional[str] = None,
        error: BaseException | Mapping[str, Any],
        status_code: Optional[int] = None,
        latency_ms: Optional[float] = None,
        retry_count: int = 0,
        bytes_sent: Optional[int] = None,
        bytes_received: Optional[int] = None,
        delivery_state: Optional[str] = None,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper for failed network outcomes."""
        return self.record_event(
            channel=channel,
            endpoint=endpoint,
            protocol=protocol,
            route=route,
            operation=operation,
            success=False,
            status_code=status_code,
            latency_ms=latency_ms,
            retry_count=retry_count,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            delivery_state=delivery_state,
            message_id=message_id,
            correlation_id=correlation_id,
            error=error,
            metadata=metadata,
        )

    def record_broadcast_outcome(
        self,
        *,
        channel: str,
        route: Optional[str],
        targets: Sequence[str],
        successes: Sequence[str],
        protocol: Optional[str] = None,
        operation: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a fan-out / broadcast success map.

        A broadcast outcome also emits a regular metric event so aggregated KPIs
        stay aligned with per-event telemetry.
        """
        try:
            target_values = ensure_sequence(targets, field_name="targets", allow_none=False)
            success_values = ensure_sequence(successes, field_name="successes", allow_none=False)
            normalized_targets = [ensure_non_empty_string(str(item), field_name="target") for item in target_values]
            normalized_successes = [ensure_non_empty_string(str(item), field_name="success_target") for item in success_values]
            success_set = set(normalized_successes)
            failure_targets = [item for item in normalized_targets if item not in success_set]

            snapshot = {
                "broadcast_id": generate_message_id("broadcast"),
                "recorded_at": utc_timestamp(),
                "channel": normalize_channel_name(channel),
                "protocol": normalize_protocol_name(protocol or channel),
                "route": route,
                "operation": operation,
                "target_total": len(normalized_targets),
                "success_total": len(normalized_successes),
                "failure_total": len(failure_targets),
                "success_rate": round(len(normalized_successes) / len(normalized_targets), 6) if normalized_targets else None,
                "targets": normalized_targets,
                "successes": normalized_successes,
                "failures": failure_targets,
                "correlation_id": correlation_id,
                "metadata": normalize_metadata(metadata),
            }

            self.record_event(
                channel=channel,
                protocol=protocol,
                route=route,
                operation=operation or "broadcast",
                success=len(failure_targets) == 0,
                retry_count=0,
                fanout_total=len(normalized_targets),
                fanout_success=len(normalized_successes),
                correlation_id=correlation_id,
                metadata=metadata,
            )

            with self._lock:
                self._broadcast_map.append(
                    sanitize_for_logging(snapshot) if self.sanitize_logs else json_safe(snapshot)
                )
                self._stats["broadcast_events"] += 1

            if self.emit_memory_snapshots:
                self.memory.set(
                    _TELEMETRY_BROADCAST_MAP_KEY,
                    list(self._broadcast_map),
                    ttl_seconds=self.recent_events_snapshot_ttl_seconds,
                    source="network_metrics",
                    metadata={"scope": "broadcast_map"},
                )
                self._stats["memory_emits"] += 1
            return snapshot
        except NetworkError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary normalization is intentional.
            raise normalize_network_exception(
                exc,
                operation="record_broadcast_outcome",
                channel=channel,
                protocol=protocol,
                route=route,
                correlation_id=correlation_id,
                metadata=metadata,
            ) from exc

    # ------------------------------------------------------------------ #
    # Snapshot / query API
    # ------------------------------------------------------------------ #
    def get_channel_metrics(self, channel: Optional[str] = None) -> Dict[str, Any]:
        """Return one channel snapshot or all channel snapshots."""
        with self._lock:
            if channel is None:
                return {key: bucket.to_snapshot() for key, bucket in sorted(self._channels.items())}
            normalized_channel = normalize_channel_name(channel)
            bucket = self._channels.get(normalized_channel)
            return bucket.to_snapshot() if bucket is not None else {}

    def get_endpoint_metrics(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Return one endpoint snapshot or all endpoint snapshots."""
        with self._lock:
            if endpoint is None:
                return {key: bucket.to_snapshot() for key, bucket in sorted(self._endpoints.items())}
            normalized_endpoint = self._normalize_endpoint_reference(endpoint)
            bucket = self._endpoints.get(normalized_endpoint)
            return bucket.to_snapshot() if bucket is not None else {}

    def get_route_metrics(self, route: Optional[str] = None) -> Dict[str, Any]:
        """Return one route snapshot or all route snapshots."""
        with self._lock:
            if route is None:
                return {key: bucket.to_snapshot() for key, bucket in sorted(self._routes.items())}
            normalized_route = ensure_non_empty_string(route, field_name="route")
            bucket = self._routes.get(normalized_route)
            return bucket.to_snapshot() if bucket is not None else {}

    def get_global_metrics(self) -> Dict[str, Any]:
        """Return global aggregated telemetry for the network subsystem."""
        with self._lock:
            snapshot = self._global.to_snapshot()
            snapshot.update(
                {
                    "generated_at": utc_timestamp(),
                    "unique_channels": len(self._channels),
                    "unique_endpoints": len(self._endpoints),
                    "unique_routes": len(self._routes),
                    "stats": dict(self._stats),
                }
            )
            return snapshot

    def get_recent_events(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return bounded recent telemetry events."""
        with self._lock:
            events = list(self._recent_events)
            if limit is None:
                return events
            if isinstance(limit, bool) or int(limit) < 0:
                raise PayloadValidationError(
                    "Recent event limit must be a non-negative integer.",
                    context={"operation": "get_recent_metric_events"},
                    details={"limit": limit},
                )
            if limit == 0:
                return []
            return events[-int(limit):]

    def get_broadcast_map(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return recent broadcast / fan-out summaries."""
        with self._lock:
            items = list(self._broadcast_map)
            if limit is None:
                return items
            if isinstance(limit, bool) or int(limit) < 0:
                raise PayloadValidationError(
                    "Broadcast map limit must be a non-negative integer.",
                    context={"operation": "get_broadcast_map"},
                    details={"limit": limit},
                )
            if limit == 0:
                return []
            return items[-int(limit):]

    def export_snapshot(
        self,
        *,
        include_recent_events: bool = True,
        include_broadcast_map: bool = True,
    ) -> Dict[str, Any]:
        """Export a JSON-safe telemetry snapshot spanning all metric scopes."""
        with self._lock:
            payload: Dict[str, Any] = {
                "started_at": self._started_at.isoformat(),
                "generated_at": utc_timestamp(),
                "config": {
                    "enabled": self.enabled,
                    "sanitize_logs": self.sanitize_logs,
                    "emit_memory_snapshots": self.emit_memory_snapshots,
                    "max_event_history_size": self.max_event_history_size,
                    "max_latency_samples_per_scope": self.max_latency_samples_per_scope,
                },
                "global": self._global.to_snapshot(),
                "channels": {key: bucket.to_snapshot() for key, bucket in sorted(self._channels.items())},
                "endpoints": {key: bucket.to_snapshot() for key, bucket in sorted(self._endpoints.items())},
                "routes": {key: bucket.to_snapshot() for key, bucket in sorted(self._routes.items())},
                "stats": dict(self._stats),
            }
            if include_recent_events:
                payload["recent_events"] = list(self._recent_events)
            if include_broadcast_map:
                payload["broadcast_map"] = list(self._broadcast_map)
            return json_safe(payload)

    def flush_to_memory(self) -> Dict[str, Any]:
        """Publish the current in-process telemetry snapshot into `NetworkMemory`."""
        with self._lock:
            channel_snapshots = {key: bucket.to_snapshot() for key, bucket in self._channels.items()}
            endpoint_snapshots = {key: bucket.to_snapshot() for key, bucket in self._endpoints.items()}
            route_snapshots = {key: bucket.to_snapshot() for key, bucket in self._routes.items()}
            global_snapshot = self.get_global_metrics()
            recent_events = list(self._recent_events)
            broadcast_map = list(self._broadcast_map)

        if not self.emit_memory_snapshots:
            return {
                "emitted": False,
                "reason": "memory_snapshot_emission_disabled",
                "global": global_snapshot,
            }

        self.memory.set(
            _TELEMETRY_GLOBAL_METRICS_KEY,
            global_snapshot,
            ttl_seconds=self.global_snapshot_ttl_seconds,
            source="network_metrics",
            metadata={"scope": "global"},
        )
        self.memory.set(
            _TELEMETRY_ENDPOINT_METRICS_KEY,
            endpoint_snapshots,
            ttl_seconds=self.endpoint_snapshot_ttl_seconds,
            source="network_metrics",
            metadata={"scope": "endpoint"},
        )
        if self.store_route_metrics:
            self.memory.set(
                _TELEMETRY_ROUTE_METRICS_KEY,
                route_snapshots,
                ttl_seconds=self.route_snapshot_ttl_seconds,
                source="network_metrics",
                metadata={"scope": "route"},
            )
        if self.store_recent_events_in_memory:
            self.memory.set(
                _TELEMETRY_RECENT_EVENTS_KEY,
                recent_events,
                ttl_seconds=self.recent_events_snapshot_ttl_seconds,
                source="network_metrics",
                metadata={"scope": "recent_events"},
            )
        self.memory.set(
            _TELEMETRY_BROADCAST_MAP_KEY,
            broadcast_map,
            ttl_seconds=self.recent_events_snapshot_ttl_seconds,
            source="network_metrics",
            metadata={"scope": "broadcast_map"},
        )
        for channel_name, snapshot in channel_snapshots.items():
            self.memory.record_channel_metrics(
                channel_name,
                snapshot,
                ttl_seconds=self.channel_snapshot_ttl_seconds,
                merge_existing=False,
                metadata={"scope": "channel"},
            )
        self._stats["memory_emits"] += 1
        return {
            "emitted": True,
            "global": global_snapshot,
            "channel_count": len(channel_snapshots),
            "endpoint_count": len(endpoint_snapshots),
            "route_count": len(route_snapshots),
        }

    def reset(self) -> None:
        """Reset all in-process telemetry aggregates and histories."""
        with self._lock:
            self._global = MetricAggregate("global", "network", self.max_latency_samples_per_scope)
            self._channels.clear()
            self._endpoints.clear()
            self._routes.clear()
            self._recent_events.clear()
            self._broadcast_map.clear()
            self._stats = {
                "events_recorded": 0,
                "memory_emits": 0,
                "broadcast_events": 0,
                "errors_normalized": 0,
            }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_event(
        self,
        *,
        channel: str,
        endpoint: Optional[str],
        protocol: Optional[str],
        route: Optional[str],
        operation: Optional[str],
        success: bool,
        status_code: Optional[int],
        latency_ms: Optional[float],
        retry_count: int,
        bytes_sent: Optional[int],
        bytes_received: Optional[int],
        delivery_state: Optional[str],
        message_id: Optional[str],
        correlation_id: Optional[str],
        fanout_total: Optional[int],
        fanout_success: Optional[int],
        error: Optional[BaseException | Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]],
    ) -> TelemetryEvent:
        normalized_channel = normalize_channel_name(channel)
        normalized_protocol = normalize_protocol_name(protocol or normalized_channel)
        normalized_endpoint = self._normalize_endpoint_reference(endpoint) if endpoint is not None else None
        normalized_route = ensure_non_empty_string(route, field_name="route") if route is not None else None
        normalized_operation = ensure_non_empty_string(operation, field_name="operation") if operation is not None else None
        normalized_delivery_state = (
            ensure_non_empty_string(delivery_state, field_name="delivery_state").lower()
            if delivery_state is not None
            else None
        )

        resolved_latency = self._coerce_non_negative_number(latency_ms, field_name="latency_ms", as_float=True)
        resolved_retry_count = self._coerce_non_negative_int(retry_count, field_name="retry_count")
        resolved_bytes_sent = self._coerce_non_negative_int(bytes_sent or 0, field_name="bytes_sent")
        resolved_bytes_received = self._coerce_non_negative_int(bytes_received or 0, field_name="bytes_received")
        resolved_status_code = None if status_code is None else self._coerce_status_code(status_code)
        resolved_fanout_total = None if fanout_total is None else self._coerce_non_negative_int(fanout_total, field_name="fanout_total")
        resolved_fanout_success = None if fanout_success is None else self._coerce_non_negative_int(fanout_success, field_name="fanout_success")
        if resolved_fanout_total is not None and resolved_fanout_success is not None and resolved_fanout_success > resolved_fanout_total:
            raise PayloadValidationError(
                "Fan-out success count cannot exceed total fan-out count.",
                context={"operation": "build_metric_event", "channel": normalized_channel, "endpoint": normalized_endpoint},
                details={"fanout_total": resolved_fanout_total, "fanout_success": resolved_fanout_success},
            )

        normalized_error = None
        if error is not None:
            normalized_error = self._normalize_error(
                error,
                channel=normalized_channel,
                endpoint=normalized_endpoint,
                protocol=normalized_protocol,
                route=normalized_route,
                correlation_id=correlation_id,
                operation=normalized_operation or "metric_event",
                status_code=resolved_status_code,
                latency_ms=resolved_latency,
                retry_count=resolved_retry_count,
                metadata=metadata,
            )

        event = TelemetryEvent(
            event_id=generate_message_id("metric"),
            occurred_at=utcnow(),
            channel=normalized_channel,
            protocol=normalized_protocol,
            endpoint=normalized_endpoint,
            route=normalized_route,
            operation=normalized_operation,
            success=bool(success),
            status_code=resolved_status_code,
            latency_ms=resolved_latency,
            retry_count=resolved_retry_count,
            bytes_sent=resolved_bytes_sent,
            bytes_received=resolved_bytes_received,
            delivery_state=normalized_delivery_state,
            message_id=message_id,
            correlation_id=correlation_id,
            fanout_total=resolved_fanout_total,
            fanout_success=resolved_fanout_success,
            error=normalized_error,
            metadata=normalize_metadata(metadata),
        )
        return event

    def _publish_event_to_memory(
        self,
        event: TelemetryEvent,
        *,
        channel_snapshot: Mapping[str, Any],
        endpoint_snapshot: Optional[Mapping[str, Any]],
        route_snapshot: Optional[Mapping[str, Any]],
        global_snapshot: Mapping[str, Any],
    ) -> None:
        self.memory.record_channel_metrics(
            event.channel,
            channel_snapshot,
            ttl_seconds=self.channel_snapshot_ttl_seconds,
            merge_existing=False,
            metadata={"scope": "channel", "protocol": event.protocol},
        )

        self.memory.set(
            _TELEMETRY_GLOBAL_METRICS_KEY,
            global_snapshot,
            ttl_seconds=self.global_snapshot_ttl_seconds,
            source="network_metrics",
            metadata={"scope": "global"},
        )

        if self.store_recent_events_in_memory:
            self.memory.set(
                _TELEMETRY_RECENT_EVENTS_KEY,
                list(self._recent_events),
                ttl_seconds=self.recent_events_snapshot_ttl_seconds,
                source="network_metrics",
                metadata={"scope": "recent_events"},
            )

        if event.endpoint and endpoint_snapshot and self.store_endpoint_metrics:
            endpoint_metrics = self.get_endpoint_metrics()
            self.memory.set(
                _TELEMETRY_ENDPOINT_METRICS_KEY,
                endpoint_metrics,
                ttl_seconds=self.endpoint_snapshot_ttl_seconds,
                source="network_metrics",
                metadata={"scope": "endpoint"},
            )
            endpoint_status = self._derive_endpoint_status(endpoint_snapshot)
            endpoint_latency = self._select_health_latency(endpoint_snapshot)
            self.memory.update_endpoint_health(
                event.endpoint,
                status=endpoint_status,
                latency_ms=endpoint_latency,
                success_rate=endpoint_snapshot.get("success_rate"),
                error_rate=endpoint_snapshot.get("error_rate"),
                last_error=endpoint_snapshot.get("last_error"),
                metadata={"source": "network_metrics", "scope": "endpoint_health"},
                ttl_seconds=self.endpoint_snapshot_ttl_seconds,
            )

        if event.route and route_snapshot and self.store_route_metrics:
            self.memory.set(
                _TELEMETRY_ROUTE_METRICS_KEY,
                self.get_route_metrics(),
                ttl_seconds=self.route_snapshot_ttl_seconds,
                source="network_metrics",
                metadata={"scope": "route"},
            )

        self._stats["memory_emits"] += 1

    def _get_or_create_bucket(
        self,
        registry: Dict[str, MetricAggregate],
        scope: str,
        key: str,
    ) -> MetricAggregate:
        bucket = registry.get(key)
        if bucket is None:
            bucket = MetricAggregate(scope, key, self.max_latency_samples_per_scope)
            registry[key] = bucket
        return bucket

    def _derive_endpoint_status(self, snapshot: Mapping[str, Any]) -> str:
        error_rate = snapshot.get("error_rate")
        success_rate = snapshot.get("success_rate")
        consecutive_failures = int(snapshot.get("consecutive_failures", 0) or 0)
        latency = snapshot.get("latency_ms") if isinstance(snapshot.get("latency_ms"), Mapping) else {}
        p95_latency = latency.get("p95")

        if consecutive_failures >= self.unhealthy_consecutive_failures_threshold:
            return "unhealthy"
        if success_rate is not None and float(success_rate) < self.unhealthy_success_rate_threshold:
            return "unhealthy"
        if error_rate is not None and float(error_rate) >= self.unhealthy_error_rate_threshold:
            return "unhealthy"
        if success_rate is not None and float(success_rate) < self.degraded_success_rate_threshold:
            return "degraded"
        if error_rate is not None and float(error_rate) >= self.degraded_error_rate_threshold:
            return "degraded"
        if p95_latency is not None and float(p95_latency) >= self.degraded_latency_ms_threshold:
            return "degraded"
        return "healthy"

    def _select_health_latency(self, snapshot: Mapping[str, Any]) -> Optional[int]:
        latency = snapshot.get("latency_ms") if isinstance(snapshot.get("latency_ms"), Mapping) else None
        if not isinstance(latency, Mapping):
            return None
        candidate = latency.get("p95")
        if candidate is None:
            candidate = latency.get("avg")
        if candidate is None:
            candidate = latency.get("last")
        if candidate is None:
            return None
        return int(round(float(candidate)))

    def _normalize_error(
        self,
        error: BaseException | Mapping[str, Any],
        *,
        channel: str,
        endpoint: Optional[str],
        protocol: str,
        route: Optional[str],
        correlation_id: Optional[str],
        operation: str,
        status_code: Optional[int],
        latency_ms: Optional[float],
        retry_count: int,
        metadata: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        try:
            if isinstance(error, Mapping):
                normalized = json_safe(error)
            elif isinstance(error, NetworkError):
                normalized = error.to_memory_snapshot()
            else:
                normalized = build_error_snapshot(
                    error,
                    operation=operation,
                    channel=channel,
                    endpoint=endpoint,
                    protocol=protocol,
                    route=route,
                    correlation_id=correlation_id,
                    status_code=status_code,
                    attempt=retry_count if retry_count > 0 else None,
                    metadata=merge_mappings(metadata, {"latency_ms": latency_ms}),
                )
            self._stats["errors_normalized"] += 1
            return normalized
        except NetworkError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary normalization is intentional.
            raise normalize_network_exception(
                exc,
                operation="normalize_metric_error",
                channel=channel,
                endpoint=endpoint,
                protocol=protocol,
                route=route,
                correlation_id=correlation_id,
                metadata=metadata,
            ) from exc

    def _normalize_endpoint_reference(self, endpoint: str) -> str:
        text = ensure_non_empty_string(str(endpoint), field_name="endpoint")
        if "://" in text:
            try:
                return normalize_endpoint(text)
            except NetworkError:
                return text
        return text

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.metrics_config.get(name, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise NetworkConfigurationError(
            "Invalid boolean value in network metrics configuration.",
            context={"operation": "network_metrics_config"},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.metrics_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in network metrics configuration.",
                context={"operation": "network_metrics_config"},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Metrics configuration values must be non-negative.",
                context={"operation": "network_metrics_config"},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _coerce_non_negative_int(self, value: Any, *, field_name: str) -> int:
        try:
            if isinstance(value, bool):
                raise TypeError("boolean is not valid")
            coerced = int(value)
        except Exception as exc:  # noqa: BLE001 - normalization is intentional here.
            raise PayloadValidationError(
                f"{field_name} must be an integer.",
                context={"operation": "metrics_validate"},
                details={"field_name": field_name, "received_value": repr(value)},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise PayloadValidationError(
                f"{field_name} must be non-negative.",
                context={"operation": "metrics_validate"},
                details={"field_name": field_name, "received_value": coerced},
            )
        return coerced

    def _coerce_non_negative_number(self, value: Any, *, field_name: str, as_float: bool = False) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, bool):
                raise TypeError("boolean is not valid")
            coerced = float(value)
        except Exception as exc:  # noqa: BLE001 - normalization is intentional here.
            raise PayloadValidationError(
                f"{field_name} must be numeric.",
                context={"operation": "metrics_validate"},
                details={"field_name": field_name, "received_value": repr(value)},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise PayloadValidationError(
                f"{field_name} must be non-negative.",
                context={"operation": "metrics_validate"},
                details={"field_name": field_name, "received_value": coerced},
            )
        return float(coerced) if as_float else coerced

    def _coerce_status_code(self, value: Any) -> int:
        try:
            if isinstance(value, bool):
                raise TypeError("boolean is not valid")
            coerced = int(value)
        except Exception as exc:  # noqa: BLE001 - normalization is intentional here.
            raise PayloadValidationError(
                "HTTP status code must be an integer.",
                context={"operation": "metrics_validate"},
                details={"field_name": "status_code", "received_value": repr(value)},
                cause=exc,
            ) from exc
        if coerced < 100 or coerced > 599:
            raise PayloadValidationError(
                "HTTP status code must be between 100 and 599.",
                context={"operation": "metrics_validate"},
                details={"field_name": "status_code", "received_value": coerced},
            )
        return coerced


if __name__ == "__main__":
    print("\n=== Running Network Metrics ===\n")
    printer.status("TEST", "Network Metrics initialized", "info")

    memory = NetworkMemory()
    metrics = NetworkMetrics(memory=memory)

    metrics.record_success(
        channel="http",
        endpoint="https://api.example.com/v1/jobs",
        protocol="https",
        route="primary",
        operation="relay",
        status_code=200,
        latency_ms=84.2,
        bytes_sent=512,
        bytes_received=2048,
        delivery_state="acked",
        message_id="msg_metrics_001",
        correlation_id="corr_metrics_001",
        metadata={"region": "eu-west", "task_class": "relay"},
    )
    printer.status("TEST", "Recorded successful HTTP event", "info")

    metrics.record_failure(
        channel="http",
        endpoint="https://api.example.com/v1/jobs",
        protocol="https",
        route="primary",
        operation="relay",
        error=TimeoutError("upstream request timeout"),
        status_code=504,
        latency_ms=1220.8,
        retry_count=1,
        bytes_sent=256,
        bytes_received=0,
        delivery_state="failed",
        message_id="msg_metrics_002",
        correlation_id="corr_metrics_002",
        metadata={"region": "eu-west", "backoff_ms": 250},
    )
    printer.status("TEST", "Recorded failed HTTP event", "info")

    metrics.record_success(
        channel="queue",
        endpoint="jobs-primary",
        protocol="queue",
        route="secondary",
        operation="publish",
        status_code=202,
        latency_ms=34.5,
        retry_count=0,
        bytes_sent=1024,
        bytes_received=0,
        delivery_state="sent",
        message_id="msg_metrics_003",
        correlation_id="corr_metrics_003",
        metadata={"queue": "jobs-primary"},
    )
    printer.status("TEST", "Recorded queue event", "info")

    broadcast_snapshot = metrics.record_broadcast_outcome(
        channel="http",
        protocol="https",
        route="fanout_primary",
        operation="broadcast",
        targets=[
            "https://api-a.example.com/v1/notify",
            "https://api-b.example.com/v1/notify",
            "https://api-c.example.com/v1/notify",
        ],
        successes=[
            "https://api-a.example.com/v1/notify",
            "https://api-c.example.com/v1/notify",
        ],
        correlation_id="corr_metrics_broadcast",
        metadata={"campaign": "incident-fanout"},
    )
    printer.status("TEST", "Recorded broadcast outcome", "info")

    channel_metrics = metrics.get_channel_metrics()
    endpoint_metrics = metrics.get_endpoint_metrics()
    route_metrics = metrics.get_route_metrics()
    global_metrics = metrics.get_global_metrics()
    recent_events = metrics.get_recent_events(limit=3)
    flush_result = metrics.flush_to_memory()
    export_snapshot = metrics.export_snapshot(include_recent_events=True, include_broadcast_map=True)

    print("Global Metrics:", stable_json_dumps(global_metrics))
    print("Channel Metrics:", stable_json_dumps(channel_metrics))
    print("Endpoint Metrics:", stable_json_dumps(endpoint_metrics))
    print("Route Metrics:", stable_json_dumps(route_metrics))
    print("Recent Events:", stable_json_dumps(recent_events))
    print("Broadcast Snapshot:", stable_json_dumps(broadcast_snapshot))
    print("Flush Result:", stable_json_dumps(flush_result))
    print(
        "Memory Health:",
        stable_json_dumps(memory.get_network_health()),
    )
    print(
        "Export Summary:",
        stable_json_dumps(
            {
                "global_total_events": export_snapshot["global"]["total_events"],
                "channel_count": len(export_snapshot["channels"]),
                "endpoint_count": len(export_snapshot["endpoints"]),
                "route_count": len(export_snapshot["routes"]),
            }
        ),
    )

    assert global_metrics["total_events"] == 4
    assert channel_metrics["http"]["total_events"] == 3
    assert channel_metrics["queue"]["success_total"] == 1
    assert endpoint_metrics["https://api.example.com:443/v1/jobs"]["failure_total"] == 1
    assert route_metrics["primary"]["events_with_retry_total"] == 1
    assert memory.get("network.telemetry.global_metrics")["total_events"] == 4
    assert memory.get("network.telemetry.channel_metrics")["http"]["total_events"] == 3
    assert len(memory.get("network.telemetry.recent_events")) >= 1
    assert len(metrics.get_broadcast_map(limit=1)) == 1

    printer.status("TEST", "All Network Metrics checks passed", "info")
    print("\n=== Test ran successfully ===\n")
