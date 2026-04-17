"""
Circuit breaker engine for SLAI's Network Agent reliability subsystem.

This module provides the production-grade circuit-breaker layer that sits
beneath NetworkReliability and alongside RetryPolicy and FailoverManager. It
owns circuit-state transitions, failure-window accounting, open/half-open/closed
eligibility decisions, probe gating, and circuit-health snapshots so the
broader network stack can reason about endpoint and channel protection through
one consistent contract.

The circuit-breaker layer is intentionally scoped to circuit protection. It is
responsible for:
- canonical circuit-profile normalization and config-backed defaults,
- per-endpoint/channel/route circuit identity and state ownership,
- failure/success accounting with configurable thresholds and windows,
- open -> half-open -> closed transition management,
- request admission decisions and retry-after style reopen timing,
- structured synchronization into NetworkMemory for the wider network stack.

It does not own retry timing, failover target selection, route ranking, or
transport execution. Those concerns belong to RetryPolicy, FailoverManager,
routing, and the specialized adapters. This module owns circuit truth and
admission decisions those layers consult.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from time import monotonic, sleep
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils import *
from ..network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Circuit Breaker")
printer = PrettyPrinter()


_CIRCUIT_BREAKER_LAST_KEY = "network.reliability.circuit_breaker.last"
_CIRCUIT_BREAKER_SNAPSHOT_KEY = "network.reliability.circuit_breaker.snapshot"
_CIRCUIT_BREAKER_HISTORY_KEY = "network.reliability.circuit_breaker.history"
_CIRCUIT_BREAKER_ACTIVE_KEY = "network.reliability.circuit_breaker.active"
_CIRCUIT_BREAKER_PROFILE_KEY = "network.reliability.circuit_breaker.profiles"
_CIRCUIT_BREAKER_PREFIX = "network.reliability.circuit_breaker.circuit"

_VALID_CIRCUIT_STATES = {"closed", "open", "half_open"}
_VALID_SCOPE_TYPES = {"endpoint", "channel", "route", "custom"}
_DEFAULT_TERMINAL_STATES: Tuple[str, ...] = ("open", "half_open", "closed")


@dataclass(slots=True)
class CircuitProfile:
    """Config-backed circuit profile used to evaluate circuit transitions."""

    name: str
    failure_threshold: int
    consecutive_failure_threshold: int
    success_threshold: int
    half_open_max_requests: int
    open_timeout_ms: int
    failure_window_seconds: int
    minimum_request_volume: int = 1
    count_timeouts: bool = True
    count_connection_errors: bool = True
    count_transport_errors: bool = True
    count_server_errors: bool = True
    count_rate_limits: bool = False
    count_unknown_errors: bool = False
    track_only_retryable_errors: bool = False
    reopen_on_half_open_failure: bool = True
    close_on_consecutive_successes: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "failure_threshold": self.failure_threshold,
            "consecutive_failure_threshold": self.consecutive_failure_threshold,
            "success_threshold": self.success_threshold,
            "half_open_max_requests": self.half_open_max_requests,
            "open_timeout_ms": self.open_timeout_ms,
            "failure_window_seconds": self.failure_window_seconds,
            "minimum_request_volume": self.minimum_request_volume,
            "count_timeouts": self.count_timeouts,
            "count_connection_errors": self.count_connection_errors,
            "count_transport_errors": self.count_transport_errors,
            "count_server_errors": self.count_server_errors,
            "count_rate_limits": self.count_rate_limits,
            "count_unknown_errors": self.count_unknown_errors,
            "track_only_retryable_errors": self.track_only_retryable_errors,
            "reopen_on_half_open_failure": self.reopen_on_half_open_failure,
            "close_on_consecutive_successes": self.close_on_consecutive_successes,
            "metadata": json_safe(self.metadata),
        }


@dataclass(slots=True)
class CircuitTransitionRecord:
    """Single transition or observation event emitted by the circuit breaker."""

    circuit_key: str
    from_state: str
    to_state: str
    occurred_at: str
    reason: str
    endpoint: Optional[str] = None
    channel: Optional[str] = None
    protocol: Optional[str] = None
    route: Optional[str] = None
    scope_type: str = "endpoint"
    failure_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    request_count: int = 0
    rejection_count: int = 0
    error_snapshot: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "circuit_key": self.circuit_key,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "occurred_at": self.occurred_at,
            "reason": self.reason,
            "endpoint": self.endpoint,
            "channel": self.channel,
            "protocol": self.protocol,
            "route": self.route,
            "scope_type": self.scope_type,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "request_count": self.request_count,
            "rejection_count": self.rejection_count,
            "error_snapshot": json_safe(self.error_snapshot),
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class CircuitRecord:
    """Authoritative circuit record tracked by the circuit breaker."""

    circuit_key: str
    profile_name: str
    state: str
    scope_type: str
    endpoint: Optional[str] = None
    channel: Optional[str] = None
    protocol: Optional[str] = None
    route: Optional[str] = None
    created_at: str = field(default_factory=lambda: utc_timestamp())
    updated_at: str = field(default_factory=lambda: utc_timestamp())
    opened_at: Optional[str] = None
    half_opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_failure_at: Optional[str] = None
    open_until: Optional[str] = None
    request_count: int = 0
    failure_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    half_open_in_flight: int = 0
    rejection_count: int = 0
    trip_count: int = 0
    reset_count: int = 0
    last_error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    transition_history: List[Dict[str, Any]] = field(default_factory=list)
    failure_timestamps: List[str] = field(default_factory=list)

    def is_open(self) -> bool:
        return self.state == "open"

    def is_closed(self) -> bool:
        return self.state == "closed"

    def is_half_open(self) -> bool:
        return self.state == "half_open"

    def open_timeout_elapsed(self) -> bool:
        if not self.open_until:
            return True
        try:
            return _parse_timestamp(self.open_until) <= utcnow()
        except Exception:
            return False

    def to_dict(self, *, include_history: bool = True) -> Dict[str, Any]:
        payload = {
            "circuit_key": self.circuit_key,
            "profile_name": self.profile_name,
            "state": self.state,
            "scope_type": self.scope_type,
            "endpoint": self.endpoint,
            "channel": self.channel,
            "protocol": self.protocol,
            "route": self.route,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "opened_at": self.opened_at,
            "half_opened_at": self.half_opened_at,
            "closed_at": self.closed_at,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "open_until": self.open_until,
            "request_count": self.request_count,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "half_open_in_flight": self.half_open_in_flight,
            "rejection_count": self.rejection_count,
            "trip_count": self.trip_count,
            "reset_count": self.reset_count,
            "last_error": json_safe(self.last_error),
            "metadata": json_safe(self.metadata),
            "failure_timestamps": list(self.failure_timestamps),
        }
        if include_history:
            payload["transition_history"] = json_safe(self.transition_history)
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class CircuitDecision:
    """Serializable admission decision returned by the circuit breaker."""

    circuit: CircuitRecord
    profile: CircuitProfile
    allow_request: bool
    state: str
    reason: str
    retry_after_ms: int = 0
    probe_request: bool = False
    decision_at: str = field(default_factory=lambda: utc_timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_request": self.allow_request,
            "state": self.state,
            "reason": self.reason,
            "retry_after_ms": self.retry_after_ms,
            "probe_request": self.probe_request,
            "decision_at": self.decision_at,
            "metadata": json_safe(self.metadata),
            "profile": self.profile.to_dict(),
            "circuit": self.circuit.to_dict(include_history=False),
        }


class CircuitBreaker:
    """
    Canonical circuit-breaker owner for the network reliability domain.

    The breaker keeps an in-process authoritative view of circuit state and
    synchronizes the important lifecycle moments into NetworkMemory so the rest
    of the network subsystem can observe circuit posture without directly
    depending on this module's internal storage.
    """

    def __init__(self, memory: Optional[NetworkMemory] = None, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.circuit_config = merge_mappings(
            get_config_section("network_reliability") or {},
            get_config_section("network_circuit_breaker") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_memory_snapshots = self._get_bool_config("record_memory_snapshots", True)
        self.record_history = self._get_bool_config("record_history", True)
        self.auto_transition_half_open = self._get_bool_config("auto_transition_half_open", True)
        self.allow_forced_states = self._get_bool_config("allow_forced_states", True)
        self.use_retry_after = self._get_bool_config("use_retry_after", True)
        self.classify_http_statuses = self._get_bool_config("classify_http_statuses", True)
        self.classify_network_errors = self._get_bool_config("classify_network_errors", True)
        self.count_only_retryable_errors = self._get_bool_config("count_only_retryable_errors", False)
        self.auto_expire_idle_circuits = self._get_bool_config("auto_expire_idle_circuits", True)

        self.default_profile_name = self._get_optional_string_config("default_profile") or "default"
        self.default_scope_type = self._get_scope_type_config("default_scope", "endpoint")
        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 7200)
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))
        self.max_profiles = max(1, self._get_non_negative_int_config("max_profiles", 32))
        self.max_circuits = max(1, self._get_non_negative_int_config("max_circuits", 5000))
        self.idle_ttl_seconds = self._get_non_negative_int_config("idle_ttl_seconds", 86400)

        self.default_failure_threshold = max(1, self._get_non_negative_int_config("failure_threshold", 5))
        self.default_consecutive_failure_threshold = max(1, self._get_non_negative_int_config("consecutive_failure_threshold", 5))
        self.default_success_threshold = max(1, self._get_non_negative_int_config("success_threshold", 2))
        self.default_half_open_max_requests = max(1, self._get_non_negative_int_config("half_open_max_requests", 1))
        self.default_open_timeout_ms = max(1, self._get_non_negative_int_config("open_timeout_ms", 30000))
        self.default_failure_window_seconds = max(1, self._get_non_negative_int_config("failure_window_seconds", 60))
        self.default_minimum_request_volume = max(1, self._get_non_negative_int_config("minimum_request_volume", 1))

        self._profiles: Dict[str, CircuitProfile] = {}
        self._circuits: Dict[str, CircuitRecord] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history_size)
        self._stats: Dict[str, int] = {
            "profiles_loaded": 0,
            "circuit_creations": 0,
            "admission_checks": 0,
            "requests_allowed": 0,
            "requests_blocked": 0,
            "state_transitions": 0,
            "opened": 0,
            "half_opened": 0,
            "closed": 0,
            "successes": 0,
            "failures": 0,
            "rejections": 0,
            "purges": 0,
        }
        self._started_at = utc_timestamp()

        self._load_profiles()
        self._sync_snapshot_memory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_profiles(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._profiles[name].to_dict() for name in sorted(self._profiles.keys())]

    def get_profile(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        return self._resolve_profile(profile_name).to_dict()

    def register_profile(self, profile_name: str, config: Mapping[str, Any]) -> Dict[str, Any]:
        profile = self._profile_from_mapping(profile_name, config)
        with self._lock:
            if len(self._profiles) >= self.max_profiles and profile.name not in self._profiles:
                raise ReliabilityError(
                    "Circuit profile registry capacity has been reached.",
                    context={"operation": "register_circuit_profile"},
                    details={"max_profiles": self.max_profiles, "profile_name": profile.name},
                )
            self._profiles[profile.name] = profile
            self._stats["profiles_loaded"] += 1
            self._sync_snapshot_memory()
            return profile.to_dict()

    def allow_request(
        self,
        *,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        profile: Optional[str | Mapping[str, Any] | CircuitProfile] = None,
        scope_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "allow_request": True,
                "state": "disabled",
                "reason": "circuit breaker disabled by configuration",
                "retry_after_ms": 0,
                "probe_request": False,
                "metadata": json_safe(normalize_metadata(metadata)),
            }

        resolved_profile = self._resolve_profile(profile)
        normalized_scope = self._resolve_scope_type(scope_type, endpoint=endpoint, channel=channel, route=route)
        normalized_metadata = normalize_metadata(metadata)

        with self._lock:
            self._expire_idle_locked()
            record = self._get_or_create_record_locked(
                profile=resolved_profile,
                scope_type=normalized_scope,
                endpoint=endpoint,
                channel=channel,
                protocol=protocol,
                route=route,
                metadata=normalized_metadata,
            )
            self._stats["admission_checks"] += 1
            record.request_count += 1
            record.updated_at = utc_timestamp()

            allow_request = True
            retry_after_ms = 0
            probe_request = False
            reason = "circuit is closed"

            if record.is_open():
                if self.auto_transition_half_open and record.open_timeout_elapsed():
                    self._transition_locked(record, "half_open", reason="open timeout elapsed; entering half-open")
                    record.half_open_in_flight = 1
                    probe_request = True
                    reason = "circuit transitioned to half_open and allows a probe request"
                else:
                    allow_request = False
                    retry_after_ms = self._retry_after_ms(record)
                    record.rejection_count += 1
                    self._stats["requests_blocked"] += 1
                    self._stats["rejections"] += 1
                    reason = "circuit is open and the open timeout has not elapsed"
            elif record.is_half_open():
                if record.half_open_in_flight >= resolved_profile.half_open_max_requests:
                    allow_request = False
                    retry_after_ms = self._retry_after_ms(record)
                    record.rejection_count += 1
                    self._stats["requests_blocked"] += 1
                    self._stats["rejections"] += 1
                    reason = "half-open probe capacity has been reached"
                else:
                    record.half_open_in_flight += 1
                    probe_request = True
                    reason = "circuit is half_open and allows a probe request"
            else:
                self._stats["requests_allowed"] += 1

            if allow_request and not record.is_closed():
                self._stats["requests_allowed"] += 1

            decision = CircuitDecision(
                circuit=record,
                profile=resolved_profile,
                allow_request=allow_request,
                state=record.state,
                reason=reason,
                retry_after_ms=retry_after_ms,
                probe_request=probe_request,
                metadata=normalized_metadata,
            )
            self._append_history_locked("allow_request", decision.to_dict())
            self._sync_record_memory(record)
            self._sync_snapshot_memory()
            return decision.to_dict()

    def record_success(
        self,
        *,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        profile: Optional[str | Mapping[str, Any] | CircuitProfile] = None,
        scope_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_profile = self._resolve_profile(profile)
        normalized_scope = self._resolve_scope_type(scope_type, endpoint=endpoint, channel=channel, route=route)
        normalized_metadata = normalize_metadata(metadata)

        with self._lock:
            record = self._get_or_create_record_locked(
                profile=resolved_profile,
                scope_type=normalized_scope,
                endpoint=endpoint,
                channel=channel,
                protocol=protocol,
                route=route,
                metadata=normalized_metadata,
            )
            now = utc_timestamp()
            record.updated_at = now
            record.last_success_at = now
            record.success_count += 1
            record.consecutive_successes += 1
            record.consecutive_failures = 0
            if record.is_half_open() and record.half_open_in_flight > 0:
                record.half_open_in_flight -= 1

            if record.is_half_open() and resolved_profile.close_on_consecutive_successes and record.consecutive_successes >= resolved_profile.success_threshold:
                self._transition_locked(record, "closed", reason="half-open success threshold reached")
                record.failure_timestamps = []
                record.half_open_in_flight = 0
                record.reset_count += 1
            elif record.is_closed():
                # keep closed but clear stale failure-window pressure on healthy traffic.
                self._prune_failure_window_locked(record, resolved_profile)

            self._stats["successes"] += 1
            snapshot = record.to_dict()
            self._append_history_locked("record_success", snapshot)
            self._sync_record_memory(record)
            self._sync_snapshot_memory()
            return snapshot

    def record_failure(
        self,
        error: BaseException | Mapping[str, Any],
        *,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        profile: Optional[str | Mapping[str, Any] | CircuitProfile] = None,
        scope_type: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_after_ms: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_profile = self._resolve_profile(profile)
        normalized_scope = self._resolve_scope_type(scope_type, endpoint=endpoint, channel=channel, route=route)
        normalized_metadata = normalize_metadata(metadata)

        normalized_error, error_snapshot, resolved_status = self._normalize_error_payload(
            error,
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route,
            status_code=status_code,
            metadata=normalized_metadata,
        )
        counted, count_reason = self._should_count_failure(
            normalized_error,
            status_code=resolved_status,
            profile=resolved_profile,
        )

        with self._lock:
            record = self._get_or_create_record_locked(
                profile=resolved_profile,
                scope_type=normalized_scope,
                endpoint=endpoint,
                channel=channel,
                protocol=protocol,
                route=route,
                metadata=normalized_metadata,
            )
            now = utc_timestamp()
            record.updated_at = now
            record.last_failure_at = now
            record.last_error = error_snapshot
            record.consecutive_successes = 0
            if record.is_half_open() and record.half_open_in_flight > 0:
                record.half_open_in_flight -= 1

            if counted:
                record.failure_count += 1
                record.consecutive_failures += 1
                record.failure_timestamps.append(now)
                self._prune_failure_window_locked(record, resolved_profile)

            opened = False
            if record.is_half_open() and resolved_profile.reopen_on_half_open_failure and counted:
                self._transition_locked(
                    record,
                    "open",
                    reason="half-open probe failed",
                    error_snapshot=error_snapshot,
                    retry_after_ms=retry_after_ms,
                    profile=resolved_profile,
                )
                opened = True
            elif record.is_closed() and counted and self._should_open_locked(record, resolved_profile):
                self._transition_locked(
                    record,
                    "open",
                    reason="failure threshold reached",
                    error_snapshot=error_snapshot,
                    retry_after_ms=retry_after_ms,
                    profile=resolved_profile,
                )
                opened = True

            self._stats["failures"] += 1
            snapshot = merge_mappings(
                record.to_dict(),
                {
                    "failure_counted": counted,
                    "count_reason": count_reason,
                    "status_code": resolved_status,
                    "opened": opened,
                },
            )
            self._append_history_locked("record_failure", snapshot)
            self._sync_record_memory(record)
            self._sync_snapshot_memory()
            return snapshot

    def force_open(self, key_or_endpoint: str, *, reason: str = "manual_open", metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return self._force_state(key_or_endpoint, "open", reason=reason, metadata=metadata)

    def force_close(self, key_or_endpoint: str, *, reason: str = "manual_close", metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return self._force_state(key_or_endpoint, "closed", reason=reason, metadata=metadata)

    def force_half_open(self, key_or_endpoint: str, *, reason: str = "manual_half_open", metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return self._force_state(key_or_endpoint, "half_open", reason=reason, metadata=metadata)

    def get_circuit(self, key_or_endpoint: str, *, include_history: bool = True) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._find_record_locked(key_or_endpoint)
            if record is None:
                return None
            self._expire_idle_locked()
            return record.to_dict(include_history=include_history)

    def require_circuit(self, key_or_endpoint: str, *, include_history: bool = True) -> Dict[str, Any]:
        with self._lock:
            record = self._find_record_locked(key_or_endpoint)
            if record is None:
                raise CircuitBreakerOpenError(
                    "Requested circuit is not registered.",
                    context={"operation": "require_circuit", "endpoint": str(key_or_endpoint)},
                    details={"lookup": str(key_or_endpoint)},
                )
            return record.to_dict(include_history=include_history)

    def list_circuits(
        self,
        *,
        state: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        include_history: bool = False,
    ) -> List[Dict[str, Any]]:
        normalized_state = self._normalize_state(state) if state is not None else None
        normalized_channel = normalize_channel_name(channel) if channel is not None else None
        normalized_protocol = normalize_protocol_name(protocol) if protocol is not None else None
        with self._lock:
            self._expire_idle_locked()
            payload: List[Dict[str, Any]] = []
            for record in self._circuits.values():
                if normalized_state is not None and record.state != normalized_state:
                    continue
                if normalized_channel is not None and record.channel != normalized_channel:
                    continue
                if normalized_protocol is not None and record.protocol != normalized_protocol:
                    continue
                payload.append(record.to_dict(include_history=include_history))
            payload.sort(key=self._sort_record_dict)
            return payload

    def purge_idle_circuits(self, *, older_than_seconds: Optional[int] = None) -> int:
        threshold_seconds = self.idle_ttl_seconds if older_than_seconds is None else max(0, int(older_than_seconds))
        cutoff = utcnow() - timedelta(seconds=threshold_seconds)
        purged = 0
        with self._lock:
            for circuit_key, record in list(self._circuits.items()):
                updated_at = _parse_timestamp(record.updated_at)
                if updated_at > cutoff:
                    continue
                self._remove_record_locked(circuit_key)
                purged += 1
            if purged:
                self._stats["purges"] += purged
                self._sync_snapshot_memory()
            return purged

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            active = [record.to_dict(include_history=False) for record in self._circuits.values()]
            return {
                "generated_at": utc_timestamp(),
                "started_at": self._started_at,
                "stats": dict(self._stats),
                "circuit_count": len(self._circuits),
                "profile_count": len(self._profiles),
                "states": self._state_counts(),
                "profiles": [self._profiles[name].to_dict() for name in sorted(self._profiles.keys())],
                "circuits": sorted(active, key=self._sort_record_dict),
                "history_size": len(self._history),
                "default_profile": self.default_profile_name,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_profiles(self) -> None:
        configured_profiles = ensure_mapping(self.circuit_config.get("profiles"), field_name="profiles", allow_none=True)
        if not configured_profiles:
            configured_profiles = {
                "default": {},
                "aggressive": {
                    "failure_threshold": max(3, min(self.default_failure_threshold, 4)),
                    "consecutive_failure_threshold": max(2, min(self.default_consecutive_failure_threshold, 3)),
                    "success_threshold": 1,
                    "half_open_max_requests": 1,
                    "open_timeout_ms": max(5000, min(self.default_open_timeout_ms, 15000)),
                },
                "conservative": {
                    "failure_threshold": max(self.default_failure_threshold, 6),
                    "consecutive_failure_threshold": max(self.default_consecutive_failure_threshold, 6),
                    "success_threshold": max(self.default_success_threshold, 2),
                    "half_open_max_requests": 1,
                    "open_timeout_ms": max(self.default_open_timeout_ms, 45000),
                },
                "streaming": {
                    "failure_threshold": max(3, min(self.default_failure_threshold, 4)),
                    "consecutive_failure_threshold": max(2, min(self.default_consecutive_failure_threshold, 3)),
                    "success_threshold": 1,
                    "half_open_max_requests": 1,
                    "open_timeout_ms": max(3000, min(self.default_open_timeout_ms, 10000)),
                    "failure_window_seconds": max(15, min(self.default_failure_window_seconds, 30)),
                    "count_timeouts": True,
                    "count_transport_errors": True,
                },
            }

        for profile_name, profile_cfg in configured_profiles.items():
            profile = self._profile_from_mapping(profile_name, ensure_mapping(profile_cfg, field_name="profile", allow_none=True))
            self._profiles[profile.name] = profile
            self._stats["profiles_loaded"] += 1

        if self.default_profile_name not in self._profiles:
            self._profiles[self.default_profile_name] = self._profile_from_mapping(self.default_profile_name, {})
            self._stats["profiles_loaded"] += 1

    def _profile_from_mapping(self, profile_name: str, profile_config: Mapping[str, Any]) -> CircuitProfile:
        name = ensure_non_empty_string(str(profile_name), field_name="profile_name").strip().lower()
        cfg = ensure_mapping(profile_config, field_name="profile_config", allow_none=True)
        return CircuitProfile(
            name=name,
            failure_threshold=max(1, self._coerce_int(cfg.get("failure_threshold"), self.default_failure_threshold, non_negative=True)),
            consecutive_failure_threshold=max(1, self._coerce_int(cfg.get("consecutive_failure_threshold"), self.default_consecutive_failure_threshold, non_negative=True)),
            success_threshold=max(1, self._coerce_int(cfg.get("success_threshold"), self.default_success_threshold, non_negative=True)),
            half_open_max_requests=max(1, self._coerce_int(cfg.get("half_open_max_requests"), self.default_half_open_max_requests, non_negative=True)),
            open_timeout_ms=max(1, self._coerce_int(cfg.get("open_timeout_ms"), self.default_open_timeout_ms, non_negative=True)),
            failure_window_seconds=max(1, self._coerce_int(cfg.get("failure_window_seconds"), self.default_failure_window_seconds, non_negative=True)),
            minimum_request_volume=max(1, self._coerce_int(cfg.get("minimum_request_volume"), self.default_minimum_request_volume, non_negative=True)),
            count_timeouts=self._coerce_bool(cfg.get("count_timeouts"), True),
            count_connection_errors=self._coerce_bool(cfg.get("count_connection_errors"), True),
            count_transport_errors=self._coerce_bool(cfg.get("count_transport_errors"), True),
            count_server_errors=self._coerce_bool(cfg.get("count_server_errors"), True),
            count_rate_limits=self._coerce_bool(cfg.get("count_rate_limits"), False),
            count_unknown_errors=self._coerce_bool(cfg.get("count_unknown_errors"), False),
            track_only_retryable_errors=self._coerce_bool(cfg.get("track_only_retryable_errors"), self.count_only_retryable_errors),
            reopen_on_half_open_failure=self._coerce_bool(cfg.get("reopen_on_half_open_failure"), True),
            close_on_consecutive_successes=self._coerce_bool(cfg.get("close_on_consecutive_successes"), True),
            metadata=normalize_metadata(cfg.get("metadata")),
        )

    def _resolve_profile(self, profile: Optional[str | Mapping[str, Any] | CircuitProfile]) -> CircuitProfile:
        if profile is None:
            profile_name = self.default_profile_name
            if profile_name not in self._profiles:
                raise NetworkConfigurationError(
                    "Default circuit profile is not registered.",
                    context={"operation": "resolve_circuit_profile"},
                    details={"default_profile": profile_name},
                )
            return self._profiles[profile_name]

        if isinstance(profile, CircuitProfile):
            return profile

        if isinstance(profile, Mapping):
            return self._profile_from_mapping(str(profile.get("name") or "inline"), profile)

        profile_name = ensure_non_empty_string(str(profile), field_name="profile").strip().lower()
        if profile_name not in self._profiles:
            raise NetworkConfigurationError(
                "Requested circuit profile is not registered.",
                context={"operation": "resolve_circuit_profile"},
                details={"profile": profile_name},
            )
        return self._profiles[profile_name]

    def _resolve_scope_type(self, scope_type: Optional[str], *, endpoint: Optional[str], channel: Optional[str], route: Optional[str]) -> str:
        if scope_type is not None:
            return self._get_scope_type_config("scope_type", scope_type)
        if endpoint is not None:
            return "endpoint"
        if route is not None:
            return "route"
        if channel is not None:
            return "channel"
        return self.default_scope_type

    def _build_circuit_key(
        self,
        *,
        scope_type: str,
        endpoint: Optional[str],
        channel: Optional[str],
        protocol: Optional[str],
        route: Optional[str],
    ) -> str:
        normalized_scope = self._get_scope_type_config("scope_type", scope_type)
        normalized_endpoint = self._safe_endpoint(endpoint)
        normalized_channel = normalize_channel_name(channel) if channel is not None else None
        normalized_protocol = normalize_protocol_name(protocol) if protocol is not None else None
        normalized_route = str(route).strip() if route is not None and str(route).strip() else None

        if normalized_scope == "endpoint":
            if not normalized_endpoint:
                raise PayloadValidationError(
                    "Endpoint-scoped circuit keys require an endpoint.",
                    context={"operation": "build_circuit_key", "scope_type": normalized_scope},
                )
            return f"endpoint:{normalized_endpoint}"
        if normalized_scope == "route":
            if not normalized_route:
                raise PayloadValidationError(
                    "Route-scoped circuit keys require a route.",
                    context={"operation": "build_circuit_key", "scope_type": normalized_scope},
                )
            return f"route:{normalized_route}:{normalized_channel or 'none'}:{normalized_protocol or 'none'}"
        if normalized_scope == "channel":
            if not normalized_channel:
                raise PayloadValidationError(
                    "Channel-scoped circuit keys require a channel.",
                    context={"operation": "build_circuit_key", "scope_type": normalized_scope},
                )
            return f"channel:{normalized_channel}:{normalized_protocol or 'none'}"
        return generate_idempotency_key(
            {
                "endpoint": normalized_endpoint,
                "channel": normalized_channel,
                "protocol": normalized_protocol,
                "route": normalized_route,
                "scope_type": normalized_scope,
            },
            namespace="circuit_breaker",
        )

    def _get_or_create_record_locked(
        self,
        *,
        profile: CircuitProfile,
        scope_type: str,
        endpoint: Optional[str],
        channel: Optional[str],
        protocol: Optional[str],
        route: Optional[str],
        metadata: Mapping[str, Any],
    ) -> CircuitRecord:
        circuit_key = self._build_circuit_key(
            scope_type=scope_type,
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route,
        )
        existing = self._circuits.get(circuit_key)
        if existing is not None:
            if metadata:
                existing.metadata = merge_mappings(existing.metadata, metadata)
            return existing

        self._ensure_capacity_locked(incoming_key=circuit_key)
        now = utc_timestamp()
        record = CircuitRecord(
            circuit_key=circuit_key,
            profile_name=profile.name,
            state="closed",
            scope_type=scope_type,
            endpoint=self._safe_endpoint(endpoint),
            channel=normalize_channel_name(channel) if channel is not None else None,
            protocol=normalize_protocol_name(protocol) if protocol is not None else None,
            route=str(route).strip() if route is not None and str(route).strip() else None,
            created_at=now,
            updated_at=now,
            closed_at=now,
            metadata=dict(metadata),
        )
        self._circuits[circuit_key] = record
        self._stats["circuit_creations"] += 1
        return record

    def _should_open_locked(self, record: CircuitRecord, profile: CircuitProfile) -> bool:
        self._prune_failure_window_locked(record, profile)
        if record.request_count < profile.minimum_request_volume:
            return False
        if record.consecutive_failures >= profile.consecutive_failure_threshold:
            return True
        return len(record.failure_timestamps) >= profile.failure_threshold

    def _should_count_failure(
        self,
        error: Optional[NetworkError],
        *,
        status_code: Optional[int],
        profile: CircuitProfile,
    ) -> Tuple[bool, str]:
        if error is not None and profile.track_only_retryable_errors and not getattr(error, "retryable", False):
            return False, "error is not marked retryable and profile tracks retryable failures only"

        if error is not None:
            if isinstance(error, RateLimitExceededError):
                return profile.count_rate_limits, "rate-limit error classification"
            if isinstance(error, (ConnectionTimeoutError, DeliveryTimeoutError)):
                return profile.count_timeouts, "timeout error classification"
            if isinstance(error, NetworkConnectionError):
                return profile.count_connection_errors, "connection error classification"
            if isinstance(error, NetworkTransportError):
                return profile.count_transport_errors, "transport error classification"
            if isinstance(error, (AuthorizationFailedError, AuthenticationFailedError, PolicyViolationError, PayloadValidationError)):
                return False, "policy/auth/payload errors do not count toward circuit opening"
            if getattr(error, "retryable", False) and not profile.count_unknown_errors:
                return True, "retryable network error classification"

        if status_code is not None and self.classify_http_statuses:
            if status_code == 429:
                return profile.count_rate_limits, "HTTP 429 classification"
            if 500 <= status_code <= 599:
                return profile.count_server_errors, f"HTTP {status_code} server error classification"
            if status_code in {408, 425}:
                return profile.count_timeouts, f"HTTP {status_code} timeout classification"
            return False, f"HTTP {status_code} does not count toward circuit opening"

        if error is None:
            return False, "no error provided"

        if self.classify_network_errors and (is_retryable_exception(error) or is_transient_exception(error)):
            return True, "transient exception classification"

        return profile.count_unknown_errors, "unknown error classification"

    def _normalize_error_payload(
        self,
        error: BaseException | Mapping[str, Any],
        *,
        endpoint: Optional[str],
        channel: Optional[str],
        protocol: Optional[str],
        route: Optional[str],
        status_code: Optional[int],
        metadata: Mapping[str, Any],
    ) -> Tuple[Optional[NetworkError], Dict[str, Any], Optional[int]]:
        if isinstance(error, Mapping):
            snapshot = json_safe(error)
            inferred_status_code = snapshot.get("status_code") if isinstance(snapshot, Mapping) else None
            return None, snapshot if isinstance(snapshot, dict) else {"error": snapshot}, int(inferred_status_code) if inferred_status_code is not None else status_code

        normalized_error = normalize_network_exception(
            error,
            operation="circuit_breaker_record_failure",
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route,
            status_code=status_code,
            metadata=metadata,
        )
        snapshot = normalized_error.to_memory_snapshot()
        return normalized_error, snapshot, normalized_error.status_code or status_code

    def _transition_locked(
        self,
        record: CircuitRecord,
        to_state: str,
        *,
        reason: str,
        error_snapshot: Optional[Mapping[str, Any]] = None,
        retry_after_ms: Optional[int] = None,
        profile: Optional[CircuitProfile] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        normalized_to_state = self._normalize_state(to_state)
        from_state = record.state
        if from_state == normalized_to_state:
            return

        record.state = normalized_to_state
        record.updated_at = utc_timestamp()
        if normalized_to_state == "open":
            record.opened_at = record.updated_at
            open_timeout_ms = (profile.open_timeout_ms if profile is not None else self.default_open_timeout_ms)
            if retry_after_ms is not None and self.use_retry_after:
                open_timeout_ms = max(open_timeout_ms, int(retry_after_ms))
            record.open_until = (utcnow() + timedelta(milliseconds=max(1, int(open_timeout_ms)))).isoformat()
            record.trip_count += 1
            self._stats["opened"] += 1
        elif normalized_to_state == "half_open":
            record.half_opened_at = record.updated_at
            record.half_open_in_flight = 0
            self._stats["half_opened"] += 1
        elif normalized_to_state == "closed":
            record.closed_at = record.updated_at
            record.open_until = None
            record.half_open_in_flight = 0
            record.failure_count = 0
            record.consecutive_failures = 0
            record.reset_count += 1
            self._stats["closed"] += 1

        if error_snapshot is not None:
            record.last_error = json_safe(error_snapshot)
        transition = CircuitTransitionRecord(
            circuit_key=record.circuit_key,
            from_state=from_state,
            to_state=normalized_to_state,
            occurred_at=record.updated_at,
            reason=ensure_non_empty_string(reason, field_name="reason"),
            endpoint=record.endpoint,
            channel=record.channel,
            protocol=record.protocol,
            route=record.route,
            scope_type=record.scope_type,
            failure_count=record.failure_count,
            consecutive_failures=record.consecutive_failures,
            consecutive_successes=record.consecutive_successes,
            request_count=record.request_count,
            rejection_count=record.rejection_count,
            error_snapshot=json_safe(error_snapshot),
            metadata=normalize_metadata(metadata),
        ).to_dict()
        record.transition_history.append(transition)
        if len(record.transition_history) > self.max_history_size:
            record.transition_history = record.transition_history[-self.max_history_size :]
        self._stats["state_transitions"] += 1
        self._append_history_locked("transition", transition)

    def _force_state(self, key_or_endpoint: str, state: str, *, reason: str, metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not self.allow_forced_states:
            raise PolicyViolationError(
                "Forced circuit-state transitions are disabled by configuration.",
                context={"operation": "force_circuit_state"},
                details={"target_state": state},
            )
        normalized_state = self._normalize_state(state)
        with self._lock:
            record = self._find_record_locked(key_or_endpoint)
            if record is None:
                raise DeliveryStateError(
                    "Requested circuit is not registered.",
                    context={"operation": "force_circuit_state", "endpoint": str(key_or_endpoint)},
                    details={"lookup": str(key_or_endpoint)},
                )
            self._transition_locked(record, normalized_state, reason=reason, metadata=metadata)
            self._sync_record_memory(record)
            self._sync_snapshot_memory()
            return record.to_dict()

    def _find_record_locked(self, key_or_endpoint: str) -> Optional[CircuitRecord]:
        lookup = ensure_non_empty_string(str(key_or_endpoint), field_name="circuit_lookup").strip()
        if lookup in self._circuits:
            return self._circuits[lookup]
        safe_endpoint = self._safe_endpoint(lookup)
        for record in self._circuits.values():
            if record.endpoint == safe_endpoint:
                return record
        return None

    def _prune_failure_window_locked(self, record: CircuitRecord, profile: CircuitProfile) -> None:
        if not record.failure_timestamps:
            return
        cutoff = utcnow() - timedelta(seconds=max(1, int(profile.failure_window_seconds)))
        kept: List[str] = []
        for raw in record.failure_timestamps:
            try:
                if _parse_timestamp(raw) >= cutoff:
                    kept.append(raw)
            except Exception:
                continue
        record.failure_timestamps = kept

    def _ensure_capacity_locked(self, *, incoming_key: str) -> None:
        if incoming_key in self._circuits:
            return
        if len(self._circuits) < self.max_circuits:
            return
        oldest_key = min(self._circuits.items(), key=lambda item: _parse_timestamp(item[1].updated_at))[0]
        self._remove_record_locked(oldest_key)
        self._stats["purges"] += 1

    def _remove_record_locked(self, circuit_key: str) -> None:
        record = self._circuits.pop(circuit_key, None)
        if record is None:
            return
        try:
            if record.endpoint:
                self.memory.set_endpoint_circuit_state(
                    record.endpoint,
                    "closed",
                    ttl_seconds=self.snapshot_ttl_seconds,
                    metadata={"circuit_removed": True, "circuit_key": record.circuit_key},
                )
            self.memory.delete(self._circuit_memory_key(circuit_key))
        except Exception:
            return

    def _expire_idle_locked(self) -> None:
        if not self.auto_expire_idle_circuits or self.idle_ttl_seconds <= 0:
            return
        cutoff = utcnow() - timedelta(seconds=self.idle_ttl_seconds)
        for circuit_key, record in list(self._circuits.items()):
            try:
                updated_at = _parse_timestamp(record.updated_at)
            except Exception:
                continue
            if updated_at <= cutoff and record.is_closed():
                self._remove_record_locked(circuit_key)

    def _retry_after_ms(self, record: CircuitRecord) -> int:
        if not record.open_until:
            return 0
        try:
            delta = _parse_timestamp(record.open_until) - utcnow()
            return max(0, int(delta.total_seconds() * 1000))
        except Exception:
            return 0

    def _state_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self._circuits.values():
            counts[record.state] = counts.get(record.state, 0) + 1
        return counts

    def _sort_record_dict(self, item: Mapping[str, Any]) -> Tuple[str, str]:
        return (str(item.get("state", "")), str(item.get("circuit_key", "")))

    def _append_history_locked(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "occurred_at": utc_timestamp(),
            "payload": sanitize_for_logging(payload) if self.sanitize_logs else json_safe(payload),
        }
        self._history.append(event)
        if self.record_history:
            self.memory.append(
                _CIRCUIT_BREAKER_HISTORY_KEY,
                event,
                max_items=self.max_history_size,
                ttl_seconds=self.history_ttl_seconds,
                source="circuit_breaker",
            )

    def _sync_record_memory(self, record: CircuitRecord) -> None:
        if not self.record_memory_snapshots:
            return
        if record.endpoint:
            self.memory.set_endpoint_circuit_state(
                record.endpoint,
                record.state,
                ttl_seconds=self.snapshot_ttl_seconds,
                metadata={
                    "circuit_key": record.circuit_key,
                    "profile_name": record.profile_name,
                    "protocol": record.protocol,
                    "channel": record.channel,
                    "route": record.route,
                    "rejection_count": record.rejection_count,
                    "trip_count": record.trip_count,
                },
            )
        self.memory.set(
            _CIRCUIT_BREAKER_LAST_KEY,
            record.to_dict(include_history=False),
            ttl_seconds=self.snapshot_ttl_seconds,
            source="circuit_breaker",
        )
        self.memory.set(
            self._circuit_memory_key(record.circuit_key),
            record.to_dict(include_history=True),
            ttl_seconds=self.snapshot_ttl_seconds,
            source="circuit_breaker",
        )

    def _sync_snapshot_memory(self) -> None:
        if not self.record_memory_snapshots:
            return
        self.memory.set(
            _CIRCUIT_BREAKER_PROFILE_KEY,
            {"profiles": [profile.to_dict() for profile in self._profiles.values()], "generated_at": utc_timestamp()},
            ttl_seconds=self.snapshot_ttl_seconds,
            source="circuit_breaker",
        )
        self.memory.set(
            _CIRCUIT_BREAKER_ACTIVE_KEY,
            {
                "generated_at": utc_timestamp(),
                "circuits": [record.to_dict(include_history=False) for record in self._circuits.values()],
                "count": len(self._circuits),
            },
            ttl_seconds=self.snapshot_ttl_seconds,
            source="circuit_breaker",
        )
        self.memory.set(
            _CIRCUIT_BREAKER_SNAPSHOT_KEY,
            self.get_snapshot(),
            ttl_seconds=self.snapshot_ttl_seconds,
            source="circuit_breaker",
        )

    def _circuit_memory_key(self, circuit_key: str) -> str:
        return f"{_CIRCUIT_BREAKER_PREFIX}.{ensure_non_empty_string(circuit_key, field_name='circuit_key')}"

    def _normalize_state(self, value: str) -> str:
        normalized = ensure_non_empty_string(str(value), field_name="state").strip().lower()
        if normalized not in _VALID_CIRCUIT_STATES:
            raise NetworkConfigurationError(
                "Invalid circuit state.",
                context={"operation": "circuit_breaker_state"},
                details={"state": value, "allowed": sorted(_VALID_CIRCUIT_STATES)},
            )
        return normalized

    def _get_bool_config(self, name: str, default: bool) -> bool:
        return self._coerce_bool(self.circuit_config.get(name, default), default)

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        return self._coerce_int(self.circuit_config.get(name, default), default, non_negative=True)

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.circuit_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_scope_type_config(self, name: str, default: str) -> str:
        value = str(default if name == "scope_type" else self.circuit_config.get(name, default)).strip().lower() or default
        if value not in _VALID_SCOPE_TYPES:
            raise NetworkConfigurationError(
                "Invalid circuit scope type in configuration.",
                context={"operation": "circuit_breaker_config"},
                details={"scope_type": value, "allowed": sorted(_VALID_SCOPE_TYPES)},
            )
        return value

    def _coerce_bool(self, value: Any, default: bool) -> bool:
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
            "Invalid boolean value in circuit-breaker configuration.",
            context={"operation": "circuit_breaker_config"},
            details={"config_value": value},
        )

    def _coerce_int(self, value: Any, default: int, *, non_negative: bool = False) -> int:
        if value is None:
            value = default
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in circuit-breaker configuration.",
                context={"operation": "circuit_breaker_config"},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if non_negative and coerced < 0:
            raise NetworkConfigurationError(
                "Circuit-breaker integer configuration value must be non-negative.",
                context={"operation": "circuit_breaker_config"},
                details={"config_value": value},
            )
        return coerced
    
    def _safe_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        """Normalize an endpoint string for circuit keys and comparisons."""
        if endpoint is None:
            return None
        try:
            return normalize_endpoint(endpoint)
        except Exception:
            # Fallback: return the raw string if normalization fails
            return endpoint.strip()


# ----------------------------------------------------------------------
# Local utility helper
# ----------------------------------------------------------------------
def _parse_timestamp(value: str) -> datetime:
    text = ensure_non_empty_string(str(value), field_name="timestamp")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    print("\n=== Running Network Circuit Breaker ===\n")
    printer.status("TEST", "Network Circuit Breaker initialized", "info")

    memory = NetworkMemory()
    breaker = CircuitBreaker(
        memory=memory,
        config={
            "default_profile": "default",
            "profiles": {
                "default": {
                    "failure_threshold": 2,
                    "consecutive_failure_threshold": 2,
                    "success_threshold": 1,
                    "half_open_max_requests": 1,
                    "open_timeout_ms": 100,
                    "failure_window_seconds": 10,
                }
            },
        },
    )

    endpoint = "https://api.example.com/v1/jobs"

    first = breaker.allow_request(endpoint=endpoint, channel="http", protocol="http")
    printer.status("TEST", "Initial request admission evaluated", "info")

    fail_one = breaker.record_failure(
        TimeoutError("upstream timeout"),
        endpoint=endpoint,
        channel="http",
        protocol="http",
    )
    printer.status("TEST", "First failure recorded", "info")

    fail_two = breaker.record_failure(
        TimeoutError("upstream timeout again"),
        endpoint=endpoint,
        channel="http",
        protocol="http",
    )
    printer.status("TEST", "Second failure recorded and circuit opened", "info")

    blocked = breaker.allow_request(endpoint=endpoint, channel="http", protocol="http")
    printer.status("TEST", "Open-circuit rejection evaluated", "info")

    sleep(0.15)
    probe = breaker.allow_request(endpoint=endpoint, channel="http", protocol="http")
    printer.status("TEST", "Half-open probe evaluated", "info")

    closed = breaker.record_success(endpoint=endpoint, channel="http", protocol="http")
    printer.status("TEST", "Probe success recorded and circuit closed", "success")

    snapshot = breaker.get_snapshot()

    print("Initial Admission:", stable_json_dumps(first))
    print("Failure One:", stable_json_dumps(fail_one))
    print("Failure Two:", stable_json_dumps(fail_two))
    print("Blocked Decision:", stable_json_dumps(blocked))
    print("Probe Decision:", stable_json_dumps(probe))
    print("Closed Snapshot:", stable_json_dumps(closed))
    print("Snapshot:", stable_json_dumps(snapshot))

    assert first["allow_request"] is True
    assert fail_two["state"] == "open"
    assert blocked["allow_request"] is False
    assert probe["allow_request"] is True
    assert probe["state"] == "half_open"
    assert closed["state"] == "closed"
    assert memory.get("network.reliability.circuit_breaker.last") is not None
    assert memory.get("network.reliability.circuit_breaker.snapshot") is not None

    printer.status("TEST", "All Network Circuit Breaker checks passed", "success")
    print("\n=== Test ran successfully ===\n")
