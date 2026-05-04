"""
Failover management for SLAI's Network Agent reliability subsystem.

This module provides the production-grade failover manager that sits beneath
NetworkReliability and alongside RetryPolicy and CircuitBreaker. It owns
alternative-route evaluation, failover-attempt planning, fallback-target
selection, failover exhaustion semantics, and memory-backed failover snapshots
so the broader network stack can reason about rerouting behavior through one
consistent contract.

The failover manager is intentionally scoped to fallback-route decisioning. It
is responsible for:
- canonical failover-profile normalization and config-backed defaults,
- candidate failover-route discovery from explicit route sets and endpoint
  registry inventory,
- alternative-path selection across protocol/channel/region/health/cost
  dimensions,
- policy-aware route ranking by consulting RoutePolicy when available,
- structured synchronization into NetworkMemory for the wider network stack.

It does not own route discovery policy, retry timing, circuit-state mutation,
or transport execution. Those concerns belong to routing, RetryPolicy,
CircuitBreaker, and the specialized adapters. This module owns failover-truth
and failover-target recommendations those layers consult.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils import *
from ..network_memory import NetworkMemory
from ..routing import EndpointRegistry, RoutePolicy
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Failover Manager")
printer = PrettyPrinter()


_FAILOVER_MANAGER_LAST_KEY = "network.reliability.failover_manager.last"
_FAILOVER_MANAGER_SNAPSHOT_KEY = "network.reliability.failover_manager.snapshot"
_FAILOVER_MANAGER_HISTORY_KEY = "network.reliability.failover_manager.history"
_FAILOVER_MANAGER_PROFILE_KEY = "network.reliability.failover_manager.profiles"
_FAILOVER_MANAGER_SELECTED_KEY = "network.reliability.failover_manager.selected"

_DEFAULT_HEALTHY_STATUSES = ("healthy", "available", "up", "connected", "idle")
_DEFAULT_DEGRADED_STATUSES = ("degraded", "warning", "limited", "slow")
_DEFAULT_UNHEALTHY_STATUSES = ("down", "failed", "unhealthy", "blocked", "closed")


@dataclass(slots=True)
class FailoverProfile:
    """Config-backed failover profile used to evaluate fallback decisions."""

    name: str
    max_attempts: int
    max_candidates: int
    require_different_endpoint: bool = True
    exclude_current_endpoint: bool = False
    exclude_current_route: bool = True
    allow_protocol_change: bool = True
    allow_channel_change: bool = True
    prefer_same_protocol: bool = True
    prefer_same_channel: bool = True
    prefer_same_region: bool = True
    prefer_secure_routes: bool = True
    allow_degraded_endpoints: bool = True
    allow_unhealthy_endpoints: bool = False
    disallow_open_circuit: bool = True
    respect_required_failover_disposition: bool = True
    minimum_health_score: float = 0.0
    max_cost_multiplier: float = 2.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_attempts": self.max_attempts,
            "max_candidates": self.max_candidates,
            "require_different_endpoint": self.require_different_endpoint,
            "exclude_current_endpoint": self.exclude_current_endpoint,
            "exclude_current_route": self.exclude_current_route,
            "allow_protocol_change": self.allow_protocol_change,
            "allow_channel_change": self.allow_channel_change,
            "prefer_same_protocol": self.prefer_same_protocol,
            "prefer_same_channel": self.prefer_same_channel,
            "prefer_same_region": self.prefer_same_region,
            "prefer_secure_routes": self.prefer_secure_routes,
            "allow_degraded_endpoints": self.allow_degraded_endpoints,
            "allow_unhealthy_endpoints": self.allow_unhealthy_endpoints,
            "disallow_open_circuit": self.disallow_open_circuit,
            "respect_required_failover_disposition": self.respect_required_failover_disposition,
            "minimum_health_score": round(float(self.minimum_health_score), 6),
            "max_cost_multiplier": round(float(self.max_cost_multiplier), 6),
            "metadata": json_safe(self.metadata),
        }


@dataclass(slots=True)
class FailoverCandidateScore:
    """Detailed scoring record for a single failover candidate."""

    candidate: Dict[str, Any]
    total_score: float = 0.0
    allowed: bool = True
    reasons: List[str] = field(default_factory=list)
    disqualification_reasons: List[str] = field(default_factory=list)
    route_policy_score: float = 0.0
    health_score: float = 0.0
    security_score: float = 0.0
    affinity_score: float = 0.0
    diversity_score: float = 0.0
    cost_score: float = 0.0
    penalty_score: float = 0.0

    def add_reason(self, reason: str) -> None:
        text = str(reason).strip()
        if text:
            self.reasons.append(text)

    def disqualify(self, reason: str) -> None:
        text = str(reason).strip()
        if text:
            self.allowed = False
            self.disqualification_reasons.append(text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": json_safe(self.candidate),
            "total_score": round(float(self.total_score), 6),
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "disqualification_reasons": list(self.disqualification_reasons),
            "route_policy_score": round(float(self.route_policy_score), 6),
            "health_score": round(float(self.health_score), 6),
            "security_score": round(float(self.security_score), 6),
            "affinity_score": round(float(self.affinity_score), 6),
            "diversity_score": round(float(self.diversity_score), 6),
            "cost_score": round(float(self.cost_score), 6),
            "penalty_score": round(float(self.penalty_score), 6),
        }


@dataclass(slots=True)
class FailoverAttemptRecord:
    """Single failover evaluation captured by the failover manager."""

    profile_name: str
    attempt: int
    max_attempts: int
    requires_failover: bool
    should_failover: bool
    exhausted: bool
    reason: str
    recommended_action: str
    current_route: Dict[str, Any]
    selected_route: Optional[Dict[str, Any]] = None
    candidate_count: int = 0
    viable_candidate_count: int = 0
    correlation_id: Optional[str] = None
    message_id: Optional[str] = None
    route: Optional[str] = None
    endpoint: Optional[str] = None
    channel: Optional[str] = None
    protocol: Optional[str] = None
    occurred_at: str = field(default_factory=utc_timestamp)
    error_snapshot: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "profile_name": self.profile_name,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "requires_failover": self.requires_failover,
            "should_failover": self.should_failover,
            "exhausted": self.exhausted,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "current_route": json_safe(self.current_route),
            "selected_route": json_safe(self.selected_route),
            "candidate_count": self.candidate_count,
            "viable_candidate_count": self.viable_candidate_count,
            "correlation_id": self.correlation_id,
            "message_id": self.message_id,
            "route": self.route,
            "endpoint": self.endpoint,
            "channel": self.channel,
            "protocol": self.protocol,
            "occurred_at": self.occurred_at,
            "error_snapshot": self.error_snapshot,
            "candidates": json_safe(self.candidates),
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class FailoverDecision:
    """Serializable failover decision returned by the failover manager."""

    profile: FailoverProfile
    attempt_record: FailoverAttemptRecord

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "decision": self.attempt_record.to_dict(),
        }


class FailoverManager:
    """
    Canonical failover manager for the network reliability domain.

    The manager converts current-route state, error context, and available
    alternatives into a bounded failover decision with a recommended next
    route and eligibility rationale.
    """

    def __init__(
        self,
        memory: Optional[NetworkMemory] = None,
        endpoint_registry: Optional[EndpointRegistry] = None,
        route_policy: Optional[RoutePolicy] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = load_global_config()
        self.failover_config = merge_mappings(
            get_config_section("network_reliability") or {},
            get_config_section("network_failover_manager") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self.endpoint_registry = endpoint_registry or EndpointRegistry(memory=self.memory)
        self.route_policy = route_policy or RoutePolicy(memory=self.memory, endpoint_registry=self.endpoint_registry)
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_memory_snapshots = self._get_bool_config("record_memory_snapshots", True)
        self.record_history = self._get_bool_config("record_history", True)
        self.use_route_policy = self._get_bool_config("use_route_policy", True)
        self.use_endpoint_registry = self._get_bool_config("use_endpoint_registry", True)
        self.fail_on_no_viable_failover = self._get_bool_config("fail_on_no_viable_failover", True)
        self.allow_explicit_candidates_only = self._get_bool_config("allow_explicit_candidates_only", False)
        self.allow_degraded_endpoints = self._get_bool_config("allow_degraded_endpoints", True)
        self.allow_unhealthy_endpoints = self._get_bool_config("allow_unhealthy_endpoints", False)
        self.disallow_open_circuit = self._get_bool_config("disallow_open_circuit", True)
        self.allow_same_endpoint_if_different_route = self._get_bool_config("allow_same_endpoint_if_different_route", False)
        self.prefer_same_region = self._get_bool_config("prefer_same_region", True)
        self.prefer_same_protocol = self._get_bool_config("prefer_same_protocol", True)
        self.prefer_same_channel = self._get_bool_config("prefer_same_channel", True)
        self.prefer_secure_routes = self._get_bool_config("prefer_secure_routes", True)
        self.prefer_healthy_endpoints = self._get_bool_config("prefer_healthy_endpoints", True)
        self.respect_required_failover_disposition = self._get_bool_config("respect_required_failover_disposition", True)

        self.default_profile_name = self._get_optional_string_config("default_profile") or "default"
        self.default_scope = self._get_optional_string_config("default_scope") or "endpoint"
        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 7200)
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))
        self.max_profiles = max(1, self._get_non_negative_int_config("max_profiles", 32))
        self.max_candidates = max(1, self._get_non_negative_int_config("max_candidates", 64))

        self.default_max_attempts = max(1, self._get_non_negative_int_config("max_attempts", 3))
        self.default_max_candidates = max(1, self._get_non_negative_int_config("max_candidate_routes", 16))
        self.default_minimum_health_score = self._coerce_float(
            self.failover_config.get("minimum_health_score"),
            0.0,
            minimum=0.0,
            maximum=1.0,
        )
        self.default_max_cost_multiplier = self._coerce_float(
            self.failover_config.get("max_cost_multiplier"),
            2.0,
            minimum=0.0,
        )

        self.healthy_statuses = self._get_status_sequence("healthy_statuses", _DEFAULT_HEALTHY_STATUSES)
        self.degraded_statuses = self._get_status_sequence("degraded_statuses", _DEFAULT_DEGRADED_STATUSES)
        self.unhealthy_statuses = self._get_status_sequence("unhealthy_statuses", _DEFAULT_UNHEALTHY_STATUSES)

        self._profiles: Dict[str, FailoverProfile] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history_size)
        self._stats: Dict[str, int] = {
            "profiles_loaded": 0,
            "evaluations": 0,
            "failovers_selected": 0,
            "failovers_denied": 0,
            "failover_exhausted": 0,
            "failover_required": 0,
            "registry_candidates_used": 0,
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
                    "Failover profile registry capacity has been reached.",
                    context={"operation": "register_failover_profile"},
                    details={"max_profiles": self.max_profiles, "profile_name": profile.name},
                )
            self._profiles[profile.name] = profile
            self._stats["profiles_loaded"] += 1
            self._sync_snapshot_memory()
            return profile.to_dict()

    def build_candidates(
        self,
        current_route: Mapping[str, Any],
        *,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        request_context: Optional[Mapping[str, Any]] = None,
        profile: Optional[str | Mapping[str, Any] | FailoverProfile] = None,
    ) -> List[Dict[str, Any]]:
        normalized_current = self._normalize_route(current_route)
        resolved_profile = self._resolve_profile(profile)
        normalized_request = ensure_mapping(request_context, field_name="request_context", allow_none=True)
        built = self._build_candidates_internal(normalized_current, candidates, normalized_request, resolved_profile)
        return [item for item in built]

    def evaluate(
        self,
        error: Optional[BaseException | Mapping[str, Any]] = None,
        *,
        current_route: Mapping[str, Any],
        attempt: int,
        max_attempts: Optional[int] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        profile: Optional[str | Mapping[str, Any] | FailoverProfile] = None,
        request_context: Optional[Mapping[str, Any]] = None,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError(
                "FailoverManager is disabled by configuration.",
                context={"operation": "evaluate_failover"},
            )

        normalized_current = self._normalize_route(current_route)
        normalized_attempt = max(1, int(attempt))
        resolved_profile = self._resolve_profile(profile)
        normalized_max_attempts = max(1, int(max_attempts or resolved_profile.max_attempts))
        normalized_request = ensure_mapping(request_context, field_name="request_context", allow_none=True)
        normalized_metadata = normalize_metadata(metadata)
        normalized_error, error_snapshot = self._normalize_error_payload(
            error,
            endpoint=normalized_current.get("endpoint"),
            channel=normalized_current.get("channel"),
            protocol=normalized_current.get("protocol"),
            route=normalized_current.get("route"),
            correlation_id=correlation_id,
            message_id=message_id,
            metadata=normalized_metadata,
        )

        disposition = self._resolve_retry_disposition(normalized_error)
        explicit_require_failover = bool(normalized_request.get("require_failover", False))
        requires_failover = bool(
            explicit_require_failover
            or (
                self.respect_required_failover_disposition
                and resolved_profile.respect_required_failover_disposition
                and disposition == RetryDisposition.REQUIRED_FAILOVER
            )
        )
        if requires_failover:
            self._stats["failover_required"] += 1

        built_candidates = self._build_candidates_internal(
            normalized_current,
            candidates,
            normalized_request,
            resolved_profile,
        )
        scored_candidates = self._score_candidates(
            normalized_current,
            built_candidates,
            resolved_profile,
            request_context=normalized_request,
        )
        viable_candidates = [item for item in scored_candidates if bool(item.get("allowed", False))]
        exhausted = normalized_attempt >= normalized_max_attempts
        selected = viable_candidates[0]["candidate"] if viable_candidates and not exhausted else None

        should_failover = bool(requires_failover and not exhausted and selected is not None)
        recommended_action = "continue"
        reason = "failover not required under current policy"

        if requires_failover and exhausted:
            recommended_action = "failover_exhausted"
            reason = "failover is required but failover attempts are exhausted"
        elif requires_failover and selected is None:
            recommended_action = "fail"
            reason = "failover is required but no viable alternative route exists"
        elif should_failover:
            recommended_action = "failover"
            reason = "viable failover candidate selected"
        elif selected is not None:
            recommended_action = "standby"
            reason = "alternative route exists but failover is not currently required"

        if requires_failover and selected is None and self.fail_on_no_viable_failover:
            raise FailoverExhaustedError(
                "Failover is required but no viable alternative route is available.",
                context={
                    "operation": "evaluate_failover",
                    "channel": normalized_current.get("channel"),
                    "protocol": normalized_current.get("protocol"),
                    "endpoint": normalized_current.get("endpoint"),
                    "route": normalized_current.get("route"),
                    "correlation_id": correlation_id,
                },
                details={
                    "attempt": normalized_attempt,
                    "max_attempts": normalized_max_attempts,
                    "current_route": sanitize_for_logging(normalized_current) if self.sanitize_logs else json_safe(normalized_current),
                    "candidate_count": len(scored_candidates),
                },
                cause=normalized_error,
            )

        attempt_record = FailoverAttemptRecord(
            profile_name=resolved_profile.name,
            attempt=normalized_attempt,
            max_attempts=normalized_max_attempts,
            requires_failover=requires_failover,
            should_failover=should_failover,
            exhausted=exhausted,
            reason=reason,
            recommended_action=recommended_action,
            current_route=normalized_current,
            selected_route=selected,
            candidate_count=len(scored_candidates),
            viable_candidate_count=len(viable_candidates),
            correlation_id=correlation_id,
            message_id=message_id,
            route=normalized_current.get("route"),
            endpoint=normalized_current.get("endpoint"),
            channel=normalized_current.get("channel"),
            protocol=normalized_current.get("protocol"),
            error_snapshot=error_snapshot,
            candidates=scored_candidates,
            metadata=normalized_metadata,
        )
        decision = FailoverDecision(profile=resolved_profile, attempt_record=attempt_record)

        with self._lock:
            self._stats["evaluations"] += 1
            if should_failover:
                self._stats["failovers_selected"] += 1
            else:
                self._stats["failovers_denied"] += 1
            if exhausted and requires_failover:
                self._stats["failover_exhausted"] += 1
            self._append_history_locked(decision)
            self._sync_decision_memory(decision)
            self._sync_snapshot_memory()

        return decision.to_dict()

    def plan_failover(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.evaluate(*args, **kwargs)

    def record_failover_success(
        self,
        selected_route: Mapping[str, Any],
        *,
        current_route: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_selected = self._normalize_route(selected_route)
        normalized_current = self._normalize_route(current_route) if current_route is not None else None
        event = {
            "event": "failover_success",
            "selected_route": normalized_selected,
            "current_route": normalized_current,
            "recorded_at": utc_timestamp(),
            "metadata": normalize_metadata(metadata),
        }
        with self._lock:
            self.memory.set_route_selection(
                normalized_selected,
                candidate_routes=[normalized_selected],
                route_id=str(normalized_selected.get("route") or normalized_selected.get("endpoint") or "failover"),
                reason="failover_success",
                metadata=normalize_metadata(metadata),
            )
            self.memory.append(
                _FAILOVER_MANAGER_HISTORY_KEY,
                event,
                max_items=self.max_history_size,
                ttl_seconds=self.history_ttl_seconds,
                source="failover_manager",
            )
        return event

    def record_failover_failure(
        self,
        selected_route: Mapping[str, Any],
        error: BaseException | Mapping[str, Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_selected = self._normalize_route(selected_route)
        error_snapshot = self._normalize_error_payload(
            error,
            endpoint=normalized_selected.get("endpoint"),
            channel=normalized_selected.get("channel"),
            protocol=normalized_selected.get("protocol"),
            route=normalized_selected.get("route"),
            correlation_id=None,
            message_id=None,
            metadata=normalize_metadata(metadata),
        )[1]
        event = {
            "event": "failover_failure",
            "selected_route": normalized_selected,
            "error": error_snapshot,
            "recorded_at": utc_timestamp(),
            "metadata": normalize_metadata(metadata),
        }
        with self._lock:
            self.memory.append(
                _FAILOVER_MANAGER_HISTORY_KEY,
                event,
                max_items=self.max_history_size,
                ttl_seconds=self.history_ttl_seconds,
                source="failover_manager",
            )
        return event

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "started_at": self._started_at,
                "stats": dict(self._stats),
                "profile_count": len(self._profiles),
                "profiles": [self._profiles[name].to_dict() for name in sorted(self._profiles.keys())],
                "history_size": len(self._history),
                "default_profile": self.default_profile_name,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_profiles(self) -> None:
        configured_profiles = ensure_mapping(self.failover_config.get("profiles"), field_name="profiles", allow_none=True)
        if not configured_profiles:
            configured_profiles = {
                "default": {},
                "aggressive": {
                    "max_attempts": max(self.default_max_attempts, 4),
                    "max_candidates": max(self.default_max_candidates, 20),
                    "allow_protocol_change": True,
                    "allow_channel_change": True,
                    "allow_degraded_endpoints": True,
                },
                "conservative": {
                    "max_attempts": min(self.default_max_attempts, 2),
                    "max_candidates": min(self.default_max_candidates, 8),
                    "allow_protocol_change": False,
                    "allow_channel_change": False,
                    "allow_degraded_endpoints": False,
                    "minimum_health_score": max(self.default_minimum_health_score, 0.5),
                },
                "streaming": {
                    "max_attempts": max(2, self.default_max_attempts),
                    "max_candidates": max(self.default_max_candidates, 12),
                    "prefer_same_protocol": True,
                    "prefer_same_channel": True,
                    "allow_protocol_change": False,
                    "allow_channel_change": False,
                },
            }

        for profile_name, profile_config in configured_profiles.items():
            self._profiles[str(profile_name).strip().lower()] = self._profile_from_mapping(
                profile_name,
                ensure_mapping(profile_config, field_name="profile", allow_none=True),
            )
            self._stats["profiles_loaded"] += 1

        if self.default_profile_name not in self._profiles:
            self._profiles[self.default_profile_name] = self._profile_from_mapping(self.default_profile_name, {})
            self._stats["profiles_loaded"] += 1

    def _profile_from_mapping(self, profile_name: str, profile_config: Mapping[str, Any]) -> FailoverProfile:
        name = ensure_non_empty_string(str(profile_name), field_name="profile_name").strip().lower()
        cfg = ensure_mapping(profile_config, field_name="profile_config", allow_none=True)
        return FailoverProfile(
            name=name,
            max_attempts=max(1, self._coerce_int(cfg.get("max_attempts"), self.default_max_attempts, non_negative=True)),
            max_candidates=max(1, self._coerce_int(cfg.get("max_candidates"), self.default_max_candidates, non_negative=True)),
            require_different_endpoint=self._coerce_bool(cfg.get("require_different_endpoint"), True),
            exclude_current_endpoint=self._coerce_bool(cfg.get("exclude_current_endpoint"), False),
            exclude_current_route=self._coerce_bool(cfg.get("exclude_current_route"), True),
            allow_protocol_change=self._coerce_bool(cfg.get("allow_protocol_change"), True),
            allow_channel_change=self._coerce_bool(cfg.get("allow_channel_change"), True),
            prefer_same_protocol=self._coerce_bool(cfg.get("prefer_same_protocol"), self.prefer_same_protocol),
            prefer_same_channel=self._coerce_bool(cfg.get("prefer_same_channel"), self.prefer_same_channel),
            prefer_same_region=self._coerce_bool(cfg.get("prefer_same_region"), self.prefer_same_region),
            prefer_secure_routes=self._coerce_bool(cfg.get("prefer_secure_routes"), self.prefer_secure_routes),
            allow_degraded_endpoints=self._coerce_bool(cfg.get("allow_degraded_endpoints"), self.allow_degraded_endpoints),
            allow_unhealthy_endpoints=self._coerce_bool(cfg.get("allow_unhealthy_endpoints"), self.allow_unhealthy_endpoints),
            disallow_open_circuit=self._coerce_bool(cfg.get("disallow_open_circuit"), self.disallow_open_circuit),
            respect_required_failover_disposition=self._coerce_bool(
                cfg.get("respect_required_failover_disposition"),
                self.respect_required_failover_disposition,
            ),
            minimum_health_score=self._coerce_float(
                cfg.get("minimum_health_score"),
                self.default_minimum_health_score,
                minimum=0.0,
                maximum=1.0,
            ),
            max_cost_multiplier=self._coerce_float(
                cfg.get("max_cost_multiplier"),
                self.default_max_cost_multiplier,
                minimum=0.0,
            ),
            metadata=normalize_metadata(cfg.get("metadata")),
        )

    def _resolve_profile(self, profile: Optional[str | Mapping[str, Any] | FailoverProfile]) -> FailoverProfile:
        if profile is None:
            profile_name = self.default_profile_name
            if profile_name not in self._profiles:
                raise NetworkConfigurationError(
                    "Default failover profile is not registered.",
                    context={"operation": "resolve_failover_profile"},
                    details={"default_profile": profile_name},
                )
            return self._profiles[profile_name]

        if isinstance(profile, FailoverProfile):
            return profile

        if isinstance(profile, Mapping):
            return self._profile_from_mapping(str(profile.get("name") or "inline"), profile)

        profile_name = ensure_non_empty_string(str(profile), field_name="profile").strip().lower()
        if profile_name not in self._profiles:
            raise NetworkConfigurationError(
                "Requested failover profile is not registered.",
                context={"operation": "resolve_failover_profile"},
                details={"profile": profile_name},
            )
        return self._profiles[profile_name]

    def _normalize_route(self, route: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        route_map = ensure_mapping(route, field_name="route", allow_none=True)
        endpoint = self._safe_endpoint(route_map.get("endpoint"))
        protocol = normalize_protocol_name(route_map.get("protocol") or route_map.get("channel") or "http")
        channel = normalize_channel_name(route_map.get("channel") or protocol)
        metadata = normalize_metadata(route_map.get("metadata"))
        health_score = self._coerce_optional_float(route_map.get("health_score"))
        if health_score is None:
            health_score = self._status_health_score(str(route_map.get("status", "unknown") or "unknown"))
        secure = route_map.get("secure")
        if secure is None:
            secure = is_secure_protocol(protocol)
            if endpoint:
                try:
                    secure = parse_endpoint(endpoint, default_scheme=protocol, protocol=protocol, require_host=False).secure
                except Exception:
                    secure = is_secure_protocol(protocol)

        normalized = {
            "endpoint": endpoint,
            "route": str(route_map.get("route")).strip() if route_map.get("route") is not None and str(route_map.get("route")).strip() else None,
            "channel": channel,
            "protocol": protocol,
            "region": str(route_map.get("region")).strip() if route_map.get("region") is not None and str(route_map.get("region")).strip() else None,
            "zone": str(route_map.get("zone")).strip() if route_map.get("zone") is not None and str(route_map.get("zone")).strip() else None,
            "priority": int(route_map.get("priority", 100) or 100),
            "cost": self._coerce_float(route_map.get("cost"), 1.0, minimum=0.0),
            "weight": self._coerce_float(route_map.get("weight"), 1.0, minimum=0.0),
            "status": str(route_map.get("status", "unknown") or "unknown").strip().lower(),
            "health_score": health_score,
            "secure": bool(secure),
            "circuit_state": str(route_map.get("circuit_state", "") or "").strip().lower() or None,
            "capabilities": ensure_mapping(route_map.get("capabilities"), field_name="capabilities", allow_none=True),
            "metadata": metadata,
        }
        return normalized

    def _build_candidates_internal(
        self,
        current_route: Mapping[str, Any],
        candidates: Optional[Sequence[Mapping[str, Any]]],
        request_context: Mapping[str, Any],
        profile: FailoverProfile,
    ) -> List[Dict[str, Any]]:
        built: List[Dict[str, Any]] = []
        seen: set[str] = set()

        explicit = ensure_sequence(candidates, field_name="candidates", allow_none=True, coerce_scalar=False)
        for raw in explicit:
            if not isinstance(raw, Mapping):
                raise PayloadValidationError(
                    "Each failover candidate must be a mapping-like object.",
                    context={"operation": "build_failover_candidates"},
                    details={"received_type": type(raw).__name__},
                )
            candidate = self._normalize_route(raw)
            identity = self._route_identity(candidate)
            if identity not in seen:
                built.append(candidate)
                seen.add(identity)

        if self.use_endpoint_registry and not self.allow_explicit_candidates_only:
            registry_candidates = self.endpoint_registry.get_candidates(
                protocol=current_route.get("protocol"),
                channel=current_route.get("channel"),
                region=current_route.get("region") if self.prefer_same_region else None,
                include_unavailable=False,
                include_degraded=profile.allow_degraded_endpoints,
                include_unhealthy=profile.allow_unhealthy_endpoints,
                capability_constraints=ensure_mapping(request_context.get("required_capabilities"), field_name="required_capabilities", allow_none=True),
            )
            if registry_candidates:
                self._stats["registry_candidates_used"] += 1
            for raw in registry_candidates:
                candidate = self._normalize_route(raw)
                identity = self._route_identity(candidate)
                if identity not in seen:
                    built.append(candidate)
                    seen.add(identity)

        return built[: min(self.max_candidates, profile.max_candidates)]

    def _score_candidates(
        self,
        current_route: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        profile: FailoverProfile,
        *,
        request_context: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        route_policy_scores = self._evaluate_with_route_policy(current_route, candidates, request_context, profile)
        scored: List[Dict[str, Any]] = []
        current_identity = self._route_identity(current_route)
        current_endpoint = current_route.get("endpoint")
        current_protocol = current_route.get("protocol")
        current_channel = current_route.get("channel")
        current_region = current_route.get("region")
        current_cost = max(0.0, float(current_route.get("cost") or 1.0))

        for candidate in candidates:
            score = FailoverCandidateScore(candidate=json_safe(candidate))
            candidate_identity = self._route_identity(candidate)
            candidate_endpoint = candidate.get("endpoint")
            candidate_protocol = candidate.get("protocol")
            candidate_channel = candidate.get("channel")
            candidate_region = candidate.get("region")
            candidate_status = str(candidate.get("status", "unknown")).lower()
            candidate_health = self._coerce_float(candidate.get("health_score"), 0.0, minimum=0.0, maximum=1.0)
            candidate_cost = self._coerce_float(candidate.get("cost"), 1.0, minimum=0.0)
            candidate_secure = bool(candidate.get("secure", False))
            circuit_state = str(candidate.get("circuit_state", "") or "").lower() or None

            if profile.exclude_current_route and candidate_identity == current_identity:
                score.disqualify("current route is excluded from failover candidates")
            if profile.require_different_endpoint and current_endpoint and candidate_endpoint == current_endpoint:
                score.disqualify("failover candidate must use a different endpoint")
            if profile.exclude_current_endpoint and current_endpoint and candidate_endpoint == current_endpoint:
                score.disqualify("current endpoint is excluded from failover candidates")
            if not profile.allow_protocol_change and current_protocol and candidate_protocol != current_protocol:
                score.disqualify("protocol changes are disabled for this failover profile")
            if not profile.allow_channel_change and current_channel and candidate_channel != current_channel:
                score.disqualify("channel changes are disabled for this failover profile")
            if profile.disallow_open_circuit and circuit_state == "open":
                score.disqualify("candidate circuit is open")
            if not profile.allow_degraded_endpoints and candidate_status in set(self.degraded_statuses):
                score.disqualify("degraded endpoints are disabled for this failover profile")
            if not profile.allow_unhealthy_endpoints and candidate_status in set(self.unhealthy_statuses):
                score.disqualify("unhealthy endpoints are disabled for this failover profile")
            if candidate_health < profile.minimum_health_score:
                score.disqualify("candidate health score is below the configured minimum")
            if current_cost > 0 and profile.max_cost_multiplier > 0 and candidate_cost > (current_cost * profile.max_cost_multiplier):
                score.disqualify("candidate cost exceeds the configured failover multiplier")

            policy_payload = route_policy_scores.get(candidate_identity)
            if policy_payload is not None:
                route_policy_score = self._coerce_optional_float(
                    ensure_mapping(policy_payload.get("score"), field_name="score", allow_none=True).get("total_score")
                )
                score.route_policy_score = route_policy_score or 0.0
                if not bool(policy_payload.get("allowed", True)):
                    score.disqualify("route policy rejected this candidate")
                else:
                    score.add_reason("route policy approved candidate")

            if self.prefer_healthy_endpoints:
                score.health_score += candidate_health * 30.0
                score.add_reason(f"health_score={candidate_health:.3f}")
            if profile.prefer_secure_routes and candidate_secure:
                score.security_score += 10.0
                score.add_reason("secure candidate preferred")
            if profile.prefer_same_region and current_region and candidate_region == current_region:
                score.affinity_score += 8.0
                score.add_reason("same-region affinity matched")
            if profile.prefer_same_protocol and current_protocol and candidate_protocol == current_protocol:
                score.affinity_score += 6.0
                score.add_reason("same-protocol affinity matched")
            if profile.prefer_same_channel and current_channel and candidate_channel == current_channel:
                score.affinity_score += 5.0
                score.add_reason("same-channel affinity matched")
            if candidate_endpoint and current_endpoint and candidate_endpoint != current_endpoint:
                score.diversity_score += 8.0
                score.add_reason("different endpoint diversity bonus")
            elif candidate_identity != current_identity and self.allow_same_endpoint_if_different_route:
                score.diversity_score += 3.0
                score.add_reason("alternate route on same endpoint allowed")

            score.cost_score += max(0.0, 5.0 / max(candidate_cost, 1.0))
            if candidate_status in set(self.degraded_statuses):
                score.penalty_score += 10.0
                score.add_reason("degraded candidate penalty applied")
            if candidate_status in set(self.unhealthy_statuses):
                score.penalty_score += 100.0
                score.add_reason("unhealthy candidate penalty applied")
            if circuit_state == "half_open":
                score.penalty_score += 5.0
                score.add_reason("half-open circuit penalty applied")

            score.total_score = round(
                score.route_policy_score
                + score.health_score
                + score.security_score
                + score.affinity_score
                + score.diversity_score
                + score.cost_score
                - score.penalty_score,
                6,
            )
            scored.append(score.to_dict())

        scored.sort(
            key=lambda item: (
                0 if bool(item.get("allowed", False)) else 1,
                -float(item.get("total_score", 0.0) or 0.0),
                str(ensure_mapping(item.get("candidate"), field_name="candidate", allow_none=True).get("endpoint") or ""),
            )
        )
        return scored

    def _evaluate_with_route_policy(
        self,
        current_route: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        request_context: Mapping[str, Any],
        profile: FailoverProfile,
    ) -> Dict[str, Dict[str, Any]]:
        if not self.use_route_policy or not candidates:
            return {}

        constraints: Dict[str, Any] = {}
        if not profile.allow_protocol_change and current_route.get("protocol"):
            constraints["required_protocol"] = current_route.get("protocol")
        if not profile.allow_channel_change and current_route.get("channel"):
            constraints["required_channel"] = current_route.get("channel")
        if profile.prefer_same_region and current_route.get("region"):
            constraints["preferred_region"] = current_route.get("region")
        if profile.prefer_secure_routes:
            constraints["require_secure"] = bool(request_context.get("tls_required", False))
        constraints["max_cost"] = max(0.0, float(current_route.get("cost") or 1.0)) * max(0.0, profile.max_cost_multiplier)

        try:
            evaluations = self.route_policy.evaluate_candidates(
                candidates,
                constraints=constraints,
                request_context=request_context,
            )
        except Exception as exc:
            logger.warning("RoutePolicy evaluation failed during failover scoring: %s", exc)
            return {}

        indexed: Dict[str, Dict[str, Any]] = {}
        for item in evaluations:
            candidate = ensure_mapping(item.get("candidate"), field_name="candidate", allow_none=True)
            indexed[self._route_identity(candidate)] = dict(item)
        return indexed

    def _normalize_error_payload(
        self,
        error: Optional[BaseException | Mapping[str, Any]],
        *,
        endpoint: Optional[str],
        channel: Optional[str],
        protocol: Optional[str],
        route: Optional[str],
        correlation_id: Optional[str],
        message_id: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Tuple[Optional[NetworkError], Optional[Dict[str, Any]]]:
        if error is None:
            return None, None
        if isinstance(error, Mapping):
            snapshot = json_safe(error)
            return None, snapshot if isinstance(snapshot, dict) else {"error": snapshot}
        normalized = normalize_network_exception(
            error,
            operation="failover_manager_evaluate",
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route,
            correlation_id=correlation_id,
            metadata={"message_id": message_id, **dict(metadata)},
        )
        return normalized, normalized.to_memory_snapshot()

    def _resolve_retry_disposition(self, error: Optional[NetworkError]) -> RetryDisposition:
        if error is None:
            return RetryDisposition.CONDITIONAL
        disposition = getattr(error, "retry_disposition", RetryDisposition.CONDITIONAL)
        if isinstance(disposition, RetryDisposition):
            return disposition
        try:
            return RetryDisposition(str(disposition))
        except Exception:
            return RetryDisposition.CONDITIONAL

    def _route_identity(self, route: Mapping[str, Any]) -> str:
        route_map = ensure_mapping(route, field_name="route")
        endpoint = str(route_map.get("endpoint") or "none").strip().lower()
        route_id = str(route_map.get("route") or "default").strip().lower()
        protocol = normalize_protocol_name(route_map.get("protocol") or route_map.get("channel") or "http")
        channel = normalize_channel_name(route_map.get("channel") or protocol)
        return f"{protocol}:{channel}:{endpoint}:{route_id}"

    def _status_health_score(self, status: str) -> float:
        normalized = str(status).strip().lower()
        if normalized in set(self.healthy_statuses):
            return 1.0
        if normalized in set(self.degraded_statuses):
            return 0.6
        if normalized in set(self.unhealthy_statuses):
            return 0.1
        return 0.5

    def _append_history_locked(self, decision: FailoverDecision) -> None:
        payload = decision.to_dict()
        self._history.append(payload)
        if self.record_history:
            self.memory.append(
                _FAILOVER_MANAGER_HISTORY_KEY,
                payload,
                max_items=self.max_history_size,
                ttl_seconds=self.history_ttl_seconds,
                source="failover_manager",
            )

    def _sync_decision_memory(self, decision: FailoverDecision) -> None:
        if not self.record_memory_snapshots:
            return
        payload = decision.to_dict()
        self.memory.set(
            _FAILOVER_MANAGER_LAST_KEY,
            payload,
            ttl_seconds=self.snapshot_ttl_seconds,
            source="failover_manager",
        )
        selected = ensure_mapping(payload.get("decision"), field_name="decision", allow_none=True).get("selected_route")
        if selected is not None:
            self.memory.set(
                _FAILOVER_MANAGER_SELECTED_KEY,
                selected,
                ttl_seconds=self.snapshot_ttl_seconds,
                source="failover_manager",
            )
            try:
                self.memory.set_route_selection(
                    ensure_mapping(selected, field_name="selected_route"),
                    candidate_routes=[
                        ensure_mapping(item.get("candidate"), field_name="candidate", allow_none=True)
                        for item in ensure_sequence(
                            ensure_mapping(payload.get("decision"), field_name="decision", allow_none=True).get("candidates"),
                            field_name="candidates",
                            allow_none=True,
                            coerce_scalar=False,
                        )
                    ],
                    route_id=str(ensure_mapping(selected, field_name="selected_route", allow_none=True).get("route") or ensure_mapping(selected, field_name="selected_route", allow_none=True).get("endpoint") or "failover"),
                    reason="failover_selection",
                    metadata={"profile": decision.profile.name},
                )
            except Exception:
                return

    def _sync_snapshot_memory(self) -> None:
        if not self.record_memory_snapshots:
            return
        self.memory.set(
            _FAILOVER_MANAGER_PROFILE_KEY,
            {"profiles": [profile.to_dict() for profile in self._profiles.values()], "generated_at": utc_timestamp()},
            ttl_seconds=self.snapshot_ttl_seconds,
            source="failover_manager",
        )
        self.memory.set(
            _FAILOVER_MANAGER_SNAPSHOT_KEY,
            self.get_snapshot(),
            ttl_seconds=self.snapshot_ttl_seconds,
            source="failover_manager",
        )

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

    def _get_bool_config(self, name: str, default: bool) -> bool:
        return self._coerce_bool(self.failover_config.get(name, default), default)

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        return self._coerce_int(self.failover_config.get(name, default), default, non_negative=True)

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.failover_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_status_sequence(self, name: str, default: Sequence[str]) -> Tuple[str, ...]:
        value = self.failover_config.get(name, default)
        values = ensure_sequence(value, field_name=name, allow_none=True, coerce_scalar=True)
        normalized: Dict[str, None] = {}
        for item in values:
            text = ensure_non_empty_string(str(item), field_name=name).strip().lower()
            normalized[text] = None
        return tuple(normalized.keys()) or tuple(str(item).strip().lower() for item in default)

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
            "Invalid boolean value in failover manager configuration.",
            context={"operation": "failover_manager_config"},
            details={"config_value": value},
        )

    def _coerce_int(self, value: Any, default: int, *, non_negative: bool = False) -> int:
        if value is None:
            value = default
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in failover manager configuration.",
                context={"operation": "failover_manager_config"},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if non_negative and coerced < 0:
            raise NetworkConfigurationError(
                "Failover manager integer configuration value must be non-negative.",
                context={"operation": "failover_manager_config"},
                details={"config_value": value},
            )
        return coerced

    def _coerce_float(self, value: Any, default: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
        if value is None:
            value = default
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid float value in failover manager configuration.",
                context={"operation": "failover_manager_config"},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if minimum is not None and coerced < minimum:
            raise NetworkConfigurationError(
                "Failover manager float configuration value is below the allowed minimum.",
                context={"operation": "failover_manager_config"},
                details={"config_value": value, "minimum": minimum},
            )
        if maximum is not None and coerced > maximum:
            raise NetworkConfigurationError(
                "Failover manager float configuration value exceeds the allowed maximum.",
                context={"operation": "failover_manager_config"},
                details={"config_value": value, "maximum": maximum},
            )
        return coerced

    def _coerce_optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


if __name__ == "__main__":
    print("\n=== Running Network Failover Manager ===\n")
    printer.status("TEST", "Network Failover Manager initialized", "info")

    memory = NetworkMemory()
    registry = EndpointRegistry(memory=memory)
    route_policy = RoutePolicy(memory=memory, endpoint_registry=registry)
    manager = FailoverManager(memory=memory, endpoint_registry=registry, route_policy=route_policy)

    registry.register_endpoint(
        "https://primary.example.com/v1/jobs",
        protocol="http",
        channel="http",
        region="us-east-1",
        priority=120,
        metadata={"role": "primary"},
    )
    registry.register_endpoint(
        "https://secondary.example.com/v1/jobs",
        protocol="http",
        channel="http",
        region="us-east-1",
        priority=110,
        metadata={"role": "secondary"},
    )
    registry.register_endpoint(
        "https://tertiary.example.com/v1/jobs",
        protocol="http",
        channel="http",
        region="us-west-2",
        priority=100,
        metadata={"role": "tertiary"},
    )

    registry.update_health("https://primary.example.com:443/v1/jobs", status="degraded", latency_ms=900, success_rate=0.55, error_rate=0.40, circuit_state="open")
    registry.update_health("https://secondary.example.com:443/v1/jobs", status="healthy", latency_ms=120, success_rate=0.98, error_rate=0.01)
    registry.update_health("https://tertiary.example.com:443/v1/jobs", status="healthy", latency_ms=180, success_rate=0.95, error_rate=0.03)

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

    decision = manager.evaluate(
        NegativeAcknowledgementError(
            "Transport requested reroute.",
            context={
                "operation": "send",
                "channel": "http",
                "protocol": "http",
                "endpoint": "https://primary.example.com/v1/jobs",
            },
        ),
        current_route=current_route,
        attempt=1,
        correlation_id="corr_failover_001",
        message_id="msg_failover_001",
        request_context={"require_failover": True},
        metadata={"phase": "send"},
    )
    printer.status("TEST", "Failover decision generated", "info")

    assert decision["decision"]["requires_failover"] is True
    assert decision["decision"]["should_failover"] is True
    assert decision["decision"]["selected_route"]["endpoint"] == "https://secondary.example.com:443/v1/jobs"
    assert memory.get("network.reliability.failover_manager.last") is not None
    assert memory.get("network.reliability.failover_manager.snapshot") is not None

    print("Failover Decision:", stable_json_dumps(decision))
    print("Snapshot:", stable_json_dumps(manager.get_snapshot()))

    printer.status("TEST", "All Network Failover Manager checks passed", "success")
    print("\n=== Test ran successfully ===\n")
