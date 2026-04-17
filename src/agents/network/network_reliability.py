"""
Reliability & Recovery Controls
- Retry policies (exponential backoff, jitter, max attempts).
- Circuit breaker per endpoint/channel.
- Fallback routing and degraded transport modes.
- Delivery guarantees (best-effort vs at-least-once).

This module provides the high-level reliability coordinator for SLAI's Network
Agent. It sits above the lower-level reliability engines and is responsible for
turning their individual decisions into one coherent reliability outcome for
runtime transport orchestration.

The coordinator is intentionally scoped to orchestration, not ownership of the
lower-level algorithms. It is responsible for:
- preflight admission checks through the circuit breaker,
- failure evaluation through retry, circuit, and failover coordination,
- success/failure synchronization across reliability components,
- production-safe aggregation of recovery decisions and snapshots,
- memory-backed visibility for the broader network subsystem.

It does not own adapter execution, route discovery, transport IO, or delivery
state-machine semantics. Those concerns belong to adapters, routing, and
lifecycle modules. This module converts transport outcomes into reliability
truth the rest of the Network Agent can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .utils import *
from .reliability import *
from .network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Network Reliability")
printer = PrettyPrinter

__all__ = [
    "ReliabilityDecision",
    "NetworkReliability",
]


_NETWORK_RELIABILITY_LAST_KEY = "network.reliability.last"
_NETWORK_RELIABILITY_SNAPSHOT_KEY = "network.reliability.snapshot"
_NETWORK_RELIABILITY_HISTORY_KEY = "network.reliability.history"
_NETWORK_RELIABILITY_ADMISSION_KEY = "network.reliability.admission.last"
_NETWORK_RELIABILITY_RECOVERY_KEY = "network.reliability.recovery.last"
_NETWORK_RELIABILITY_SUCCESS_KEY = "network.reliability.success.last"

_VALID_COMPONENT_NAMES = {"retry", "circuit", "failover"}
_VALID_ACTIONS = {
    "allow",
    "block",
    "retry",
    "fail",
    "failover",
    "standby",
    "exhausted",
    "failover_exhausted",
    "success",
    "unknown",
}


@dataclass(slots=True)
class ReliabilityDecision:
    """Serializable aggregate decision emitted by the reliability coordinator."""

    stage: str
    action: str
    reason: str
    allow_request: bool
    should_retry: bool = False
    requires_failover: bool = False
    exhausted: bool = False
    current_route: Dict[str, Any] = field(default_factory=dict)
    circuit: Optional[Dict[str, Any]] = None
    retry: Optional[Dict[str, Any]] = None
    failover: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    message_id: Optional[str] = None
    generated_at: str = field(default_factory=utc_timestamp)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "stage": self.stage,
            "action": self.action,
            "reason": self.reason,
            "allow_request": self.allow_request,
            "should_retry": self.should_retry,
            "requires_failover": self.requires_failover,
            "exhausted": self.exhausted,
            "current_route": json_safe(self.current_route),
            "circuit": json_safe(self.circuit),
            "retry": json_safe(self.retry),
            "failover": json_safe(self.failover),
            "correlation_id": self.correlation_id,
            "message_id": self.message_id,
            "generated_at": self.generated_at,
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


class NetworkReliability:
    """
    High-level reliability coordinator for retries, failover, and circuit state.

    The coordinator deliberately depends on the lower-level reliability engines
    rather than re-implementing their logic. This avoids overlapping ownership
    and keeps the module aligned with SLAI's Network Agent boundaries.

    Circular-import safety:
    - This module imports only the lower-level reliability engines.
    - It does not import NetworkStream, NetworkAdapters, or NetworkAgent.
    - The lower-level reliability engines do not import this coordinator.

    Initialization safety:
    - Shared dependencies can be injected to prevent duplicate instances.
    - When injected components are used, they must share the same NetworkMemory
      instance to avoid split-brain reliability state.
    """

    def __init__(
        self,
        memory: Optional[NetworkMemory] = None,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        failover_manager: Optional[FailoverManager] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = load_global_config()
        self.reliability_config = merge_mappings(
            get_config_section("network_reliability") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self._lock = RLock()

        self.memory = self._resolve_memory(
            explicit_memory=memory,
            retry_policy=retry_policy,
            circuit_breaker=circuit_breaker,
            failover_manager=failover_manager,
        )

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_memory_snapshots = self._get_bool_config("record_memory_snapshots", True)
        self.record_history = self._get_bool_config("record_history", True)
        self.retry_enabled = self._get_bool_config("retry_enabled", True)
        self.circuit_breaker_enabled = self._get_bool_config("circuit_breaker_enabled", True)
        self.failover_enabled = self._get_bool_config("failover_enabled", True)
        self.prefer_component_reason = self._get_bool_config("prefer_component_reason", True)
        self.auto_record_success_to_circuit = self._get_bool_config("auto_record_success_to_circuit", True)
        self.auto_record_failure_to_circuit = self._get_bool_config("auto_record_failure_to_circuit", True)
        self.evaluate_failover_on_blocked_request = self._get_bool_config("evaluate_failover_on_blocked_request", True)
        self.failover_on_retry_exhausted = self._get_bool_config("failover_on_retry_exhausted", True)
        self.require_current_route_for_failover = self._get_bool_config("require_current_route_for_failover", True)

        self.default_retry_profile = self._get_optional_string_config("default_retry_profile") or "default"
        self.default_circuit_profile = self._get_optional_string_config("default_circuit_profile") or "default"
        self.default_failover_profile = self._get_optional_string_config("default_failover_profile") or "default"
        self.default_scope_type = self._get_optional_string_config("default_scope_type") or "endpoint"
        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 7200)
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))

        retry_component_config = ensure_mapping(self.reliability_config.get("retry_policy"), field_name="retry_policy", allow_none=True)
        circuit_component_config = ensure_mapping(self.reliability_config.get("circuit_breaker"), field_name="circuit_breaker", allow_none=True)
        failover_component_config = ensure_mapping(self.reliability_config.get("failover_manager"), field_name="failover_manager", allow_none=True)

        self.retry_policy = retry_policy or RetryPolicy(memory=self.memory, config=retry_component_config)
        self.circuit_breaker = circuit_breaker or CircuitBreaker(memory=self.memory, config=circuit_component_config)
        self.failover_manager = failover_manager or FailoverManager(memory=self.memory, config=failover_component_config)

        self._validate_component_memory("retry", self.retry_policy)
        self._validate_component_memory("circuit", self.circuit_breaker)
        self._validate_component_memory("failover", self.failover_manager)

        self._history: List[Dict[str, Any]] = []
        self._stats: Dict[str, int] = {
            "admission_checks": 0,
            "admission_allowed": 0,
            "admission_blocked": 0,
            "recovery_evaluations": 0,
            "success_records": 0,
            "failures_recorded": 0,
            "retry_decisions": 0,
            "failover_decisions": 0,
            "failover_selected": 0,
        }
        self._started_at = utc_timestamp()
        self._sync_snapshot_memory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_profiles(self) -> Dict[str, Any]:
        return {
            "retry": self.retry_policy.list_profiles(),
            "circuit": self.circuit_breaker.list_profiles(),
            "failover": self.failover_manager.list_profiles(),
        }

    def get_profile(self, component: str, profile_name: Optional[str] = None) -> Dict[str, Any]:
        normalized_component = self._normalize_component_name(component)
        if normalized_component == "retry":
            return self.retry_policy.get_profile(profile_name)
        if normalized_component == "circuit":
            return self.circuit_breaker.get_profile(profile_name)
        return self.failover_manager.get_profile(profile_name)

    def admit_request(
        self,
        *,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        current_route: Optional[Mapping[str, Any]] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        request_context: Optional[Mapping[str, Any]] = None,
        circuit_profile: Optional[str | Mapping[str, Any] | CircuitProfile] = None,
        failover_profile: Optional[str | Mapping[str, Any] | FailoverProfile] = None,
        attempt: int = 1,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError(
                "NetworkReliability is disabled by configuration.",
                context={"operation": "admit_request", "endpoint": endpoint, "channel": channel, "protocol": protocol},
            )

        normalized_request = ensure_mapping(request_context, field_name="request_context", allow_none=True)
        normalized_metadata = normalize_metadata(metadata)
        route_snapshot = self._build_current_route(
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route,
            current_route=current_route,
            request_context=normalized_request,
        )

        circuit_payload: Optional[Dict[str, Any]] = None
        failover_payload: Optional[Dict[str, Any]] = None
        reason = "request admitted"
        action = "allow"
        allow_request = True

        if self.circuit_breaker_enabled:
            circuit_payload = self.circuit_breaker.allow_request(
                endpoint=route_snapshot.get("endpoint"),
                channel=route_snapshot.get("channel"),
                protocol=route_snapshot.get("protocol"),
                route=route_snapshot.get("route"),
                profile=circuit_profile or self.default_circuit_profile,
                scope_type=self.default_scope_type,
                metadata=normalized_metadata,
            )
            allow_request = bool(circuit_payload.get("allow_request", True))
            reason = str(circuit_payload.get("reason", reason))
            action = "allow" if allow_request else "block"

        if not allow_request and self.failover_enabled and self.evaluate_failover_on_blocked_request:
            if not self.require_current_route_for_failover or route_snapshot:
                failover_error = CircuitBreakerOpenError(
                    "Circuit breaker denied request admission.",
                    context={
                        "operation": "admit_request",
                        "endpoint": route_snapshot.get("endpoint"),
                        "channel": route_snapshot.get("channel"),
                        "protocol": route_snapshot.get("protocol"),
                        "route": route_snapshot.get("route"),
                        "correlation_id": correlation_id,
                        "attempt": max(1, int(attempt)),
                    },
                )
                failover_payload = self.failover_manager.evaluate(
                    failover_error,
                    current_route=route_snapshot,
                    attempt=max(1, int(attempt)),
                    candidates=candidates,
                    profile=failover_profile or self.default_failover_profile,
                    request_context=normalized_request,
                    correlation_id=correlation_id,
                    message_id=message_id,
                    metadata=normalized_metadata,
                )
                if ensure_mapping(failover_payload.get("decision"), field_name="decision", allow_none=True).get("selected_route") is not None:
                    action = "failover"
                    reason = str(ensure_mapping(failover_payload.get("decision"), field_name="decision", allow_none=True).get("reason") or reason)

        decision = ReliabilityDecision(
            stage="admission",
            action=action,
            reason=reason,
            allow_request=allow_request,
            should_retry=False,
            requires_failover=bool(
                failover_payload
                and bool(ensure_mapping(failover_payload.get("decision"), field_name="decision", allow_none=True).get("requires_failover", False))
            ),
            exhausted=bool(
                failover_payload
                and bool(ensure_mapping(failover_payload.get("decision"), field_name="decision", allow_none=True).get("exhausted", False))
            ),
            current_route=route_snapshot,
            circuit=circuit_payload,
            failover=failover_payload,
            correlation_id=correlation_id,
            message_id=message_id,
            metadata=normalized_metadata,
        )

        with self._lock:
            self._stats["admission_checks"] += 1
            if allow_request:
                self._stats["admission_allowed"] += 1
            else:
                self._stats["admission_blocked"] += 1
            if failover_payload is not None:
                self._stats["failover_decisions"] += 1
                if ensure_mapping(failover_payload.get("decision"), field_name="decision", allow_none=True).get("selected_route") is not None:
                    self._stats["failover_selected"] += 1
            self._record_decision_locked(decision, _NETWORK_RELIABILITY_ADMISSION_KEY)

        return decision.to_dict()

    def allow_request(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.admit_request(*args, **kwargs)

    def record_failure(
        self,
        error: BaseException | Mapping[str, Any],
        *,
        attempt: int,
        max_attempts: Optional[int] = None,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        current_route: Optional[Mapping[str, Any]] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        request_context: Optional[Mapping[str, Any]] = None,
        retry_profile: Optional[str | Mapping[str, Any] | RetryProfile] = None,
        circuit_profile: Optional[str | Mapping[str, Any] | CircuitProfile] = None,
        failover_profile: Optional[str | Mapping[str, Any] | FailoverProfile] = None,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        idempotent: bool = True,
        status_code: Optional[int] = None,
        retry_after_ms: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError(
                "NetworkReliability is disabled by configuration.",
                context={"operation": "record_failure", "endpoint": endpoint, "channel": channel, "protocol": protocol},
            )

        normalized_attempt = max(1, int(attempt))
        normalized_request = ensure_mapping(request_context, field_name="request_context", allow_none=True)
        normalized_metadata = normalize_metadata(metadata)
        route_snapshot = self._build_current_route(
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route,
            current_route=current_route,
            request_context=normalized_request,
        )

        circuit_payload: Optional[Dict[str, Any]] = None
        retry_payload: Optional[Dict[str, Any]] = None
        failover_payload: Optional[Dict[str, Any]] = None

        normalized_error = error if isinstance(error, Mapping) else normalize_network_exception(
            error,
            operation="network_reliability_record_failure",
            endpoint=route_snapshot.get("endpoint"),
            channel=route_snapshot.get("channel"),
            protocol=route_snapshot.get("protocol"),
            route=route_snapshot.get("route"),
            correlation_id=correlation_id,
            status_code=status_code,
            retry_after_ms=retry_after_ms,
            metadata={"message_id": message_id, **normalized_metadata},
        )

        if self.circuit_breaker_enabled and self.auto_record_failure_to_circuit:
            circuit_payload = self.circuit_breaker.record_failure(
                normalized_error,
                endpoint=route_snapshot.get("endpoint"),
                channel=route_snapshot.get("channel"),
                protocol=route_snapshot.get("protocol"),
                route=route_snapshot.get("route"),
                profile=circuit_profile or self.default_circuit_profile,
                scope_type=self.default_scope_type,
                status_code=status_code,
                retry_after_ms=retry_after_ms,
                metadata=normalized_metadata,
            )

        if self.retry_enabled:
            retry_payload = self.retry_policy.evaluate(
                normalized_error,
                attempt=normalized_attempt,
                max_attempts=max_attempts,
                profile=retry_profile or self.default_retry_profile,
                retry_after_ms=retry_after_ms,
                endpoint=route_snapshot.get("endpoint"),
                channel=route_snapshot.get("channel"),
                protocol=route_snapshot.get("protocol"),
                route=route_snapshot.get("route"),
                correlation_id=correlation_id,
                message_id=message_id,
                idempotent=idempotent,
                status_code=status_code,
                metadata=normalized_metadata,
            )

        requires_failover = False
        exhausted = False
        action = "fail"
        reason = getattr(normalized_error, "message", None) or str(normalized_error)
        should_retry = False

        if retry_payload is not None:
            retry_decision = ensure_mapping(retry_payload.get("decision"), field_name="decision", allow_none=True)
            requires_failover = bool(retry_decision.get("requires_failover", False))
            should_retry = bool(retry_decision.get("should_retry", False))
            exhausted = bool(retry_decision.get("exhausted", False))
            action = str(retry_decision.get("recommended_action") or action)
            reason = str(retry_decision.get("reason") or reason)

        should_evaluate_failover = bool(
            self.failover_enabled
            and route_snapshot
            and (
                requires_failover
                or bool(normalized_request.get("require_failover", False))
                or (self.failover_on_retry_exhausted and exhausted)
            )
        )

        if should_evaluate_failover:
            failover_payload = self.failover_manager.evaluate(
                normalized_error,
                current_route=route_snapshot,
                attempt=normalized_attempt,
                max_attempts=max_attempts,
                candidates=candidates,
                profile=failover_profile or self.default_failover_profile,
                request_context=normalized_request,
                correlation_id=correlation_id,
                message_id=message_id,
                metadata=normalized_metadata,
            )
            failover_decision = ensure_mapping(failover_payload.get("decision"), field_name="decision", allow_none=True)
            if failover_decision.get("selected_route") is not None:
                action = "failover"
                reason = str(failover_decision.get("reason") or reason)
            elif failover_decision.get("recommended_action"):
                action = str(failover_decision.get("recommended_action"))
                if self.prefer_component_reason:
                    reason = str(failover_decision.get("reason") or reason)
            exhausted = exhausted or bool(failover_decision.get("exhausted", False))
            requires_failover = requires_failover or bool(failover_decision.get("requires_failover", False))

        decision = ReliabilityDecision(
            stage="recovery",
            action=action if action in _VALID_ACTIONS else "unknown",
            reason=reason,
            allow_request=False,
            should_retry=should_retry,
            requires_failover=requires_failover,
            exhausted=exhausted,
            current_route=route_snapshot,
            circuit=circuit_payload,
            retry=retry_payload,
            failover=failover_payload,
            correlation_id=correlation_id,
            message_id=message_id,
            metadata=normalized_metadata,
        )

        with self._lock:
            self._stats["recovery_evaluations"] += 1
            self._stats["failures_recorded"] += 1
            if retry_payload is not None:
                self._stats["retry_decisions"] += 1
            if failover_payload is not None:
                self._stats["failover_decisions"] += 1
                if ensure_mapping(failover_payload.get("decision"), field_name="decision", allow_none=True).get("selected_route") is not None:
                    self._stats["failover_selected"] += 1
            self._record_decision_locked(decision, _NETWORK_RELIABILITY_RECOVERY_KEY)

        return decision.to_dict()

    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.record_failure(*args, **kwargs)

    def plan_recovery(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.record_failure(*args, **kwargs)

    def record_success(
        self,
        *,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        current_route: Optional[Mapping[str, Any]] = None,
        previous_route: Optional[Mapping[str, Any]] = None,
        used_failover: Optional[bool] = None,
        circuit_profile: Optional[str | Mapping[str, Any] | CircuitProfile] = None,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError(
                "NetworkReliability is disabled by configuration.",
                context={"operation": "record_success", "endpoint": endpoint, "channel": channel, "protocol": protocol},
            )

        normalized_metadata = normalize_metadata(metadata)
        route_snapshot = self._build_current_route(
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route,
            current_route=current_route,
            request_context={},
        )
        circuit_payload: Optional[Dict[str, Any]] = None
        failover_event: Optional[Dict[str, Any]] = None

        if self.circuit_breaker_enabled and self.auto_record_success_to_circuit:
            circuit_payload = self.circuit_breaker.record_success(
                endpoint=route_snapshot.get("endpoint"),
                channel=route_snapshot.get("channel"),
                protocol=route_snapshot.get("protocol"),
                route=route_snapshot.get("route"),
                profile=circuit_profile or self.default_circuit_profile,
                scope_type=self.default_scope_type,
                metadata=normalized_metadata,
            )

        if self.failover_enabled and self._should_record_failover_success(route_snapshot, previous_route, used_failover):
            failover_event = self.failover_manager.record_failover_success(
                route_snapshot,
                current_route=previous_route,
                metadata=normalized_metadata,
            )

        decision = ReliabilityDecision(
            stage="success",
            action="success",
            reason="successful transport outcome recorded",
            allow_request=True,
            current_route=route_snapshot,
            circuit=circuit_payload,
            failover=failover_event,
            correlation_id=correlation_id,
            message_id=message_id,
            metadata=normalized_metadata,
        )

        with self._lock:
            self._stats["success_records"] += 1
            self._record_decision_locked(decision, _NETWORK_RELIABILITY_SUCCESS_KEY)

        return decision.to_dict()

    def record_failover_failure(
        self,
        selected_route: Mapping[str, Any],
        error: BaseException | Mapping[str, Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.failover_manager.record_failover_failure(selected_route, error, metadata=metadata)

    def get_network_health(self) -> Dict[str, Any]:
        memory_health = self.memory.get_network_health() if hasattr(self.memory, "get_network_health") else {}
        circuit_snapshot = self.circuit_breaker.get_snapshot() if self.circuit_breaker_enabled else {}
        retry_snapshot = self.retry_policy.get_snapshot() if self.retry_enabled else {}
        failover_snapshot = self.failover_manager.get_snapshot() if self.failover_enabled else {}
        return {
            "generated_at": utc_timestamp(),
            "memory": json_safe(memory_health),
            "circuits": json_safe(circuit_snapshot),
            "retry": json_safe(retry_snapshot),
            "failover": json_safe(failover_snapshot),
        }

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "started_at": self._started_at,
                "enabled": self.enabled,
                "stats": dict(self._stats),
                "history_size": len(self._history),
                "default_profiles": {
                    "retry": self.default_retry_profile,
                    "circuit": self.default_circuit_profile,
                    "failover": self.default_failover_profile,
                },
                "config": {
                    "retry_enabled": self.retry_enabled,
                    "circuit_breaker_enabled": self.circuit_breaker_enabled,
                    "failover_enabled": self.failover_enabled,
                    "auto_record_success_to_circuit": self.auto_record_success_to_circuit,
                    "auto_record_failure_to_circuit": self.auto_record_failure_to_circuit,
                    "evaluate_failover_on_blocked_request": self.evaluate_failover_on_blocked_request,
                    "failover_on_retry_exhausted": self.failover_on_retry_exhausted,
                    "require_current_route_for_failover": self.require_current_route_for_failover,
                },
                "components": {
                    "retry": self.retry_policy.get_snapshot() if self.retry_enabled else {"enabled": False},
                    "circuit": self.circuit_breaker.get_snapshot() if self.circuit_breaker_enabled else {"enabled": False},
                    "failover": self.failover_manager.get_snapshot() if self.failover_enabled else {"enabled": False},
                },
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_memory(
        self,
        *,
        explicit_memory: Optional[NetworkMemory],
        retry_policy: Optional[RetryPolicy],
        circuit_breaker: Optional[CircuitBreaker],
        failover_manager: Optional[FailoverManager],
    ) -> NetworkMemory:
        if explicit_memory is not None:
            return explicit_memory
        for component in (retry_policy, circuit_breaker, failover_manager):
            if component is not None and hasattr(component, "memory"):
                memory = getattr(component, "memory")
                if memory is not None:
                    return memory
        return NetworkMemory()

    def _validate_component_memory(self, component_name: str, component: Any) -> None:
        component_memory = getattr(component, "memory", None)
        if component_memory is None:
            raise NetworkConfigurationError(
                "Injected reliability component does not expose a memory surface.",
                context={"operation": "network_reliability_init"},
                details={"component": component_name},
            )
        if component_memory is not self.memory:
            raise NetworkConfigurationError(
                "Injected reliability components must share the same NetworkMemory instance.",
                context={"operation": "network_reliability_init"},
                details={"component": component_name},
            )

    def _normalize_component_name(self, component: str) -> str:
        normalized = ensure_non_empty_string(str(component), field_name="component").strip().lower()
        if normalized not in _VALID_COMPONENT_NAMES:
            raise NetworkConfigurationError(
                "Unknown reliability component name.",
                context={"operation": "network_reliability_component"},
                details={"component": component, "allowed": sorted(_VALID_COMPONENT_NAMES)},
            )
        return normalized

    def _record_decision_locked(self, decision: ReliabilityDecision, stage_key: str) -> None:
        payload = decision.to_dict()
        event = {
            "stage": decision.stage,
            "action": decision.action,
            "occurred_at": decision.generated_at,
            "payload": sanitize_for_logging(payload) if self.sanitize_logs else json_safe(payload),
        }
        self._history.append(event)
        if len(self._history) > self.max_history_size:
            self._history = self._history[-self.max_history_size:]
        if self.record_history:
            self._safe_memory_append(_NETWORK_RELIABILITY_HISTORY_KEY, event)
        self._safe_memory_set(stage_key, payload)
        self._safe_memory_set(_NETWORK_RELIABILITY_LAST_KEY, payload)
        self._sync_snapshot_memory()

    def _sync_snapshot_memory(self) -> None:
        if not self.record_memory_snapshots:
            return
        self._safe_memory_set(_NETWORK_RELIABILITY_SNAPSHOT_KEY, self.get_snapshot())

    def _safe_memory_set(self, key: str, value: Any) -> None:
        if not self.record_memory_snapshots:
            return
        try:
            if hasattr(self.memory, "set"):
                self.memory.set(
                    key,
                    value,
                    ttl_seconds=self.snapshot_ttl_seconds,
                    source="network_reliability",
                )
        except Exception as exc:  # noqa: BLE001 - snapshot sync should not break primary decision flow.
            logger.warning("NetworkReliability memory set failed for %s: %s", key, exc)

    def _safe_memory_append(self, key: str, value: Any) -> None:
        if not self.record_memory_snapshots:
            return
        try:
            if hasattr(self.memory, "append"):
                self.memory.append(
                    key,
                    value,
                    max_items=self.max_history_size,
                    ttl_seconds=self.history_ttl_seconds,
                    source="network_reliability",
                )
            elif hasattr(self.memory, "set"):
                existing = self.memory.get(key, default=[]) if hasattr(self.memory, "get") else []
                if not isinstance(existing, list):
                    existing = []
                payload = list(existing)
                payload.append(value)
                payload = payload[-self.max_history_size :]
                self.memory.set(
                    key,
                    payload,
                    ttl_seconds=self.history_ttl_seconds,
                    source="network_reliability",
                )
        except Exception as exc:  # noqa: BLE001 - history sync should not break primary decision flow.
            logger.warning("NetworkReliability memory append failed for %s: %s", key, exc)

    def _build_current_route(
        self,
        *,
        endpoint: Optional[str],
        channel: Optional[str],
        protocol: Optional[str],
        route: Optional[str],
        current_route: Optional[Mapping[str, Any]],
        request_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base = ensure_mapping(current_route, field_name="current_route", allow_none=True)
        resolved_protocol = normalize_protocol_name(base.get("protocol") or protocol or base.get("channel") or channel or "http")
        resolved_channel = normalize_channel_name(base.get("channel") or channel or resolved_protocol)
        raw_endpoint = base.get("endpoint") or endpoint
        resolved_endpoint = self._safe_endpoint(raw_endpoint)
        resolved_route = str(base.get("route") or route).strip() if (base.get("route") or route) is not None and str(base.get("route") or route).strip() else None

        secure = base.get("secure")
        if secure is None:
            if resolved_endpoint and isinstance(resolved_endpoint, str) and "://" in resolved_endpoint:
                try:
                    secure = bool(parse_endpoint(resolved_endpoint, default_scheme=resolved_protocol, protocol=resolved_protocol, require_host=False).secure)
                except Exception:
                    secure = is_secure_protocol(resolved_protocol)
            else:
                secure = is_secure_protocol(resolved_protocol)

        snapshot = merge_mappings(
            base,
            {
                "endpoint": resolved_endpoint,
                "channel": resolved_channel,
                "protocol": resolved_protocol,
                "route": resolved_route,
                "secure": bool(secure),
                "metadata": normalize_metadata(base.get("metadata")),
            },
        )

        if request_context.get("preferred_region") is not None and snapshot.get("region") is None:
            snapshot["region"] = request_context.get("preferred_region")
        if request_context.get("tls_required") is not None:
            snapshot["tls_required"] = bool(request_context.get("tls_required"))
        return snapshot

    def _safe_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        if endpoint is None:
            return None
        text = str(endpoint).strip()
        if not text:
            return None
        try:
            if "://" in text:
                return normalize_endpoint(text)
        except Exception:
            return text
        return text

    def _should_record_failover_success(
        self,
        current_route: Mapping[str, Any],
        previous_route: Optional[Mapping[str, Any]],
        used_failover: Optional[bool],
    ) -> bool:
        if used_failover is not None:
            return bool(used_failover)
        if previous_route is None:
            return False
        previous = self._build_current_route(
            endpoint=None,
            channel=None,
            protocol=None,
            route=None,
            current_route=previous_route,
            request_context={},
        )
        return self._route_identity(current_route) != self._route_identity(previous)

    def _route_identity(self, route: Mapping[str, Any]) -> str:
        route_map = ensure_mapping(route, field_name="route", allow_none=True)
        return stable_json_dumps(
            {
                "endpoint": route_map.get("endpoint"),
                "channel": route_map.get("channel"),
                "protocol": route_map.get("protocol"),
                "route": route_map.get("route"),
            }
        )

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.reliability_config.get(name, default)
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
            "Invalid boolean value in network reliability configuration.",
            context={"operation": "network_reliability_config"},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.reliability_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in network reliability configuration.",
                context={"operation": "network_reliability_config"},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Network reliability integer configuration value must be non-negative.",
                context={"operation": "network_reliability_config"},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.reliability_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None


if __name__ == "__main__":
    print("\n=== Running Network Reliability ===\n")
    printer.status("TEST", "Network Reliability initialized", "info")

    memory = NetworkMemory()
    reliability = NetworkReliability(
        memory=memory,
        config={
            "default_retry_profile": "default",
            "default_circuit_profile": "default",
            "default_failover_profile": "default",
            "retry_policy": {
                "profiles": {
                    "default": {
                        "max_attempts": 3,
                        "initial_backoff_ms": 50,
                        "max_backoff_ms": 200,
                    }
                }
            },
            "circuit_breaker": {
                "profiles": {
                    "default": {
                        "failure_threshold": 1,
                        "consecutive_failure_threshold": 1,
                        "success_threshold": 1,
                        "half_open_max_requests": 1,
                        "open_timeout_ms": 100,
                        "failure_window_seconds": 10,
                    }
                }
            },
            "failover_manager": {
                "profiles": {
                    "default": {
                        "max_attempts": 3,
                        "max_candidates": 8,
                    }
                }
            },
        },
    )

    current_route = {
        "endpoint": "https://primary.example.com/v1/jobs",
        "protocol": "http",
        "channel": "http",
        "route": "primary",
        "region": "us-east-1",
        "priority": 120,
        "cost": 1.0,
        "status": "degraded",
        "health_score": 0.30,
        "secure": True,
        "circuit_state": "open",
    }
    candidates = [
        {
            "endpoint": "https://secondary.example.com/v1/jobs",
            "protocol": "http",
            "channel": "http",
            "route": "secondary",
            "region": "us-east-1",
            "priority": 110,
            "cost": 1.1,
            "status": "healthy",
            "health_score": 0.95,
            "secure": True,
        },
        {
            "endpoint": "https://tertiary.example.com/v1/jobs",
            "protocol": "http",
            "channel": "http",
            "route": "tertiary",
            "region": "us-west-2",
            "priority": 100,
            "cost": 1.2,
            "status": "healthy",
            "health_score": 0.91,
            "secure": True,
        },
    ]

    admission = reliability.admit_request(
        endpoint=current_route["endpoint"],
        channel=current_route["channel"],
        protocol=current_route["protocol"],
        route=current_route["route"],
        current_route=current_route,
        candidates=candidates,
        attempt=1,
        correlation_id="corr_rel_001",
        message_id="msg_rel_001",
        metadata={"phase": "preflight"},
    )
    printer.status("TEST", "Admission decision generated", "info")

    retry_timeout = reliability.record_failure(
        TimeoutError("upstream timeout"),
        attempt=1,
        current_route=current_route,
        candidates=candidates,
        endpoint=current_route["endpoint"],
        channel=current_route["channel"],
        protocol=current_route["protocol"],
        route=current_route["route"],
        correlation_id="corr_rel_002",
        message_id="msg_rel_002",
        idempotent=True,
        metadata={"phase": "send"},
    )
    printer.status("TEST", "Retry-oriented recovery decision generated", "info")

    failover_required = reliability.record_failure(
        NegativeAcknowledgementError(
            "Transport requested reroute.",
            context={
                "operation": "send",
                "channel": "http",
                "protocol": "http",
                "endpoint": current_route["endpoint"],
                "route": current_route["route"],
            },
        ),
        attempt=1,
        current_route=current_route,
        candidates=candidates,
        endpoint=current_route["endpoint"],
        channel=current_route["channel"],
        protocol=current_route["protocol"],
        route=current_route["route"],
        correlation_id="corr_rel_003",
        message_id="msg_rel_003",
        idempotent=True,
        metadata={"phase": "reroute"},
    )
    printer.status("TEST", "Failover-oriented recovery decision generated", "info")

    success = reliability.record_success(
        current_route=candidates[0],
        previous_route=current_route,
        correlation_id="corr_rel_004",
        message_id="msg_rel_004",
        metadata={"phase": "complete", "used_failover": True},
    )
    printer.status("TEST", "Success path recorded", "success")

    snapshot = reliability.get_snapshot()
    network_health = reliability.get_network_health()

    print("Admission:", stable_json_dumps(admission))
    print("Retry Recovery:", stable_json_dumps(retry_timeout))
    print("Failover Recovery:", stable_json_dumps(failover_required))
    print("Success:", stable_json_dumps(success))
    print("Snapshot:", stable_json_dumps(snapshot))
    print("Network Health:", stable_json_dumps(network_health))

    assert admission["stage"] == "admission"
    assert retry_timeout["stage"] == "recovery"
    assert retry_timeout["retry"]["decision"]["should_retry"] is True
    assert failover_required["failover"]["decision"]["selected_route"] is not None
    assert success["stage"] == "success"
    assert snapshot["stats"]["recovery_evaluations"] >= 2

    if hasattr(memory, "get"):
        assert memory.get(_NETWORK_RELIABILITY_LAST_KEY, default=None) is not None
        assert memory.get(_NETWORK_RELIABILITY_SNAPSHOT_KEY, default=None) is not None

    printer.status("TEST", "All Network Reliability checks passed", "success")
    print("\n=== Test ran successfully ===\n")
