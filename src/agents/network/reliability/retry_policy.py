"""
Retry policy engine for SLAI's Network Agent reliability subsystem.

This module provides the production-grade retry-policy layer that sits beneath
NetworkReliability and alongside CircuitBreaker and FailoverManager. It owns
retry classification, retry-window calculation, backoff and jitter policy,
retry exhaustion semantics, and retry-plan snapshots so the broader network
stack can reason about re-attempt behavior through one consistent contract.

The retry-policy layer is intentionally scoped to retry decisioning. It is
responsible for:
- canonical retry-profile normalization and config-backed defaults,
- retryability / transience classification from SLAI-native network errors,
- bounded retry decisions with exponential, linear, fixed, and decorrelated backoff,
- jitter handling and optional Retry-After honoring,
- failover-aware retry disposition handling,
- structured synchronization into NetworkMemory for the wider network stack.

It does not own route selection, circuit-state persistence, failover route
selection, or transport execution. Those concerns belong to routing,
CircuitBreaker, FailoverManager, and the specialized adapters. This module owns
retry policy truth and retry timing recommendations those layers consult.
"""

from __future__ import annotations

import random

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils import *
from ..network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Retry Policy")
printer = PrettyPrinter()

_RETRY_POLICY_LAST_KEY = "network.reliability.retry_policy.last"
_RETRY_POLICY_SNAPSHOT_KEY = "network.reliability.retry_policy.snapshot"
_RETRY_POLICY_HISTORY_KEY = "network.reliability.retry_policy.history"
_RETRY_POLICY_PROFILE_KEY = "network.reliability.retry_policy.profiles"

_VALID_BACKOFF_STRATEGIES = {"exponential", "linear", "fixed", "decorrelated"}
_VALID_JITTER_MODES = {"none", "full", "equal", "decorrelated"}
_DEFAULT_RETRYABLE_STATUS_CODES = (408, 409, 425, 429, 500, 502, 503, 504)
_DEFAULT_NON_RETRYABLE_STATUS_CODES = (400, 401, 403, 404, 405, 410, 413, 422)


@dataclass(slots=True)
class RetryProfile:
    """Config-backed retry profile used to evaluate retry decisions."""

    name: str
    max_attempts: int
    initial_backoff_ms: int
    max_backoff_ms: int
    backoff_multiplier: float = 2.0
    backoff_strategy: str = "exponential"
    jitter_ratio: float = 0.2
    jitter_mode: str = "full"
    retry_on_timeouts: bool = True
    retry_on_connection_errors: bool = True
    retry_on_rate_limits: bool = True
    retry_on_server_errors: bool = True
    retry_on_unknown_errors: bool = False
    retry_on_duplicate_active: bool = False
    respect_retry_after: bool = True
    failover_on_required_disposition: bool = True
    allow_retry_when_non_idempotent: bool = False
    retryable_status_codes: Tuple[int, ...] = field(default_factory=lambda: tuple(_DEFAULT_RETRYABLE_STATUS_CODES))
    non_retryable_status_codes: Tuple[int, ...] = field(default_factory=lambda: tuple(_DEFAULT_NON_RETRYABLE_STATUS_CODES))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_attempts": self.max_attempts,
            "initial_backoff_ms": self.initial_backoff_ms,
            "max_backoff_ms": self.max_backoff_ms,
            "backoff_multiplier": round(float(self.backoff_multiplier), 6),
            "backoff_strategy": self.backoff_strategy,
            "jitter_ratio": round(float(self.jitter_ratio), 6),
            "jitter_mode": self.jitter_mode,
            "retry_on_timeouts": self.retry_on_timeouts,
            "retry_on_connection_errors": self.retry_on_connection_errors,
            "retry_on_rate_limits": self.retry_on_rate_limits,
            "retry_on_server_errors": self.retry_on_server_errors,
            "retry_on_unknown_errors": self.retry_on_unknown_errors,
            "retry_on_duplicate_active": self.retry_on_duplicate_active,
            "respect_retry_after": self.respect_retry_after,
            "failover_on_required_disposition": self.failover_on_required_disposition,
            "allow_retry_when_non_idempotent": self.allow_retry_when_non_idempotent,
            "retryable_status_codes": list(self.retryable_status_codes),
            "non_retryable_status_codes": list(self.non_retryable_status_codes),
            "metadata": json_safe(self.metadata),
        }


@dataclass(slots=True)
class RetryAttemptRecord:
    """Single retry attempt evaluation captured by the retry policy."""

    profile_name: str
    attempt: int
    next_attempt: Optional[int]
    should_retry: bool
    exhausted: bool
    requires_failover: bool
    wait_ms: int
    retry_disposition: str
    recommended_action: str
    reason: str
    status_code: Optional[int] = None
    correlation_id: Optional[str] = None
    message_id: Optional[str] = None
    endpoint: Optional[str] = None
    channel: Optional[str] = None
    protocol: Optional[str] = None
    route: Optional[str] = None
    occurred_at: str = field(default_factory=utc_timestamp)
    error_snapshot: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "profile_name": self.profile_name,
            "attempt": self.attempt,
            "next_attempt": self.next_attempt,
            "should_retry": self.should_retry,
            "exhausted": self.exhausted,
            "requires_failover": self.requires_failover,
            "wait_ms": self.wait_ms,
            "retry_disposition": self.retry_disposition,
            "recommended_action": self.recommended_action,
            "reason": self.reason,
            "status_code": self.status_code,
            "correlation_id": self.correlation_id,
            "message_id": self.message_id,
            "endpoint": self.endpoint,
            "channel": self.channel,
            "protocol": self.protocol,
            "route": self.route,
            "occurred_at": self.occurred_at,
            "error_snapshot": self.error_snapshot,
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class RetryDecision:
    """Serializable retry decision returned by the retry policy."""

    profile: RetryProfile
    attempt_record: RetryAttemptRecord
    normalized_error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "decision": self.attempt_record.to_dict(),
            "normalized_error": json_safe(self.normalized_error),
        }


class RetryPolicy:
    """
    Canonical retry-policy owner for the network reliability domain.

    The policy engine converts errors, attempts, and caller constraints into a
    bounded retry decision with a recommended next action and wait interval.
    """

    def __init__(self, memory: Optional[NetworkMemory] = None, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.retry_policy_config = merge_mappings(
            get_config_section("network_reliability") or {},
            get_config_section("network_retry_policy") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_memory_snapshots = self._get_bool_config("record_memory_snapshots", True)
        self.record_history = self._get_bool_config("record_history", True)
        self.use_retry_after = self._get_bool_config("use_retry_after", True)
        self.failover_enabled = self._get_bool_config("failover_enabled", True)
        self.require_idempotent_for_unsafe_retries = self._get_bool_config("require_idempotent_for_unsafe_retries", True)
        self.classify_http_statuses = self._get_bool_config("classify_http_statuses", True)
        self.classify_network_errors = self._get_bool_config("classify_network_errors", True)
        self.classify_unknown_errors_as_retryable = self._get_bool_config("classify_unknown_errors_as_retryable", False)
        self.prefer_error_disposition = self._get_bool_config("prefer_error_disposition", True)
        self.use_deterministic_jitter = self._get_bool_config("use_deterministic_jitter", True)

        self.default_profile_name = self._get_optional_string_config("default_profile") or "default"
        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 7200)
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))
        self.max_profiles = max(1, self._get_non_negative_int_config("max_profiles", 32))

        self.default_max_attempts = max(1, self._get_non_negative_int_config("max_attempts", 3))
        self.default_initial_backoff_ms = max(1, self._get_non_negative_int_config("initial_backoff_ms", 250))
        self.default_max_backoff_ms = max(self.default_initial_backoff_ms, self._get_non_negative_int_config("max_backoff_ms", 5000))
        self.default_backoff_multiplier = self._coerce_float(self.retry_policy_config.get("backoff_multiplier"), 2.0, minimum=1.0)
        self.default_jitter_ratio = self._coerce_float(self.retry_policy_config.get("jitter_ratio"), 0.2, minimum=0.0, maximum=1.0)
        self.default_backoff_strategy = self._get_backoff_strategy_config("backoff_strategy", "exponential")
        self.default_jitter_mode = self._get_jitter_mode_config("jitter_mode", "full")

        self._profiles: Dict[str, RetryProfile] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history_size)
        self._stats: Dict[str, int] = {
            "profiles_loaded": 0,
            "decisions": 0,
            "retries_allowed": 0,
            "retries_denied": 0,
            "retry_exhausted": 0,
            "failover_required": 0,
            "retry_after_honored": 0,
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
        profile = self._resolve_profile(profile_name)
        return profile.to_dict()

    def register_profile(self, profile_name: str, config: Mapping[str, Any]) -> Dict[str, Any]:
        profile = self._profile_from_mapping(profile_name, config)
        with self._lock:
            if len(self._profiles) >= self.max_profiles and profile.name not in self._profiles:
                raise ReliabilityError(
                    "Retry profile registry capacity has been reached.",
                    context={"operation": "register_retry_profile"},
                    details={"max_profiles": self.max_profiles, "profile_name": profile.name},
                )
            self._profiles[profile.name] = profile
            self._stats["profiles_loaded"] += 1
            self._sync_snapshot_memory()
            return profile.to_dict()

    def compute_delay_ms(
        self,
        attempt: int,
        *,
        profile: Optional[str | Mapping[str, Any] | RetryProfile] = None,
        retry_after_ms: Optional[int] = None,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> int:
        resolved_profile = self._resolve_profile(profile)
        normalized_attempt = max(1, int(attempt))
        base_delay = self._compute_base_delay_ms(normalized_attempt, resolved_profile)
        jittered = self._apply_jitter(
            base_delay,
            attempt=normalized_attempt,
            profile=resolved_profile,
            correlation_id=correlation_id,
            message_id=message_id,
        )
        if retry_after_ms is not None and self.use_retry_after and resolved_profile.respect_retry_after:
            self._stats["retry_after_honored"] += 1
            return max(int(retry_after_ms), jittered)
        return jittered

    def evaluate(
        self,
        error: Optional[BaseException | Mapping[str, Any]] = None,
        *,
        attempt: int,
        max_attempts: Optional[int] = None,
        profile: Optional[str | Mapping[str, Any]] = None,
        retry_after_ms: Optional[int] = None,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        route: Optional[str] = None,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        idempotent: bool = True,
        status_code: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError(
                "RetryPolicy is disabled by configuration.",
                context={"operation": "evaluate_retry_policy", "channel": channel, "protocol": protocol, "endpoint": endpoint},
            )

        resolved_profile = self._resolve_profile(profile)
        normalized_attempt = max(1, int(attempt))
        normalized_max_attempts = max(1, int(max_attempts or resolved_profile.max_attempts))
        normalized_metadata = normalize_metadata(metadata)
        normalized_error, error_snapshot, resolved_status_code = self._normalize_error_payload(
            error,
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route,
            correlation_id=correlation_id,
            message_id=message_id,
            status_code=status_code,
            metadata=normalized_metadata,
        )

        disposition = self._resolve_retry_disposition(normalized_error)
        classification = self._classify_retryability(
            normalized_error,
            status_code=resolved_status_code,
            profile=resolved_profile,
            idempotent=idempotent,
        )

        exhausted = normalized_attempt >= normalized_max_attempts
        should_retry = bool(classification["retryable"] and not exhausted)
        requires_failover = bool(
            self.failover_enabled
            and resolved_profile.failover_on_required_disposition
            and disposition == RetryDisposition.REQUIRED_FAILOVER
        )

        wait_ms = 0
        recommended_action = "fail"
        reason = classification["reason"]

        if requires_failover and exhausted:
            should_retry = False
            recommended_action = "failover_exhausted"
            reason = "retry disposition requires failover but retry attempts are exhausted"
        elif requires_failover and should_retry:
            recommended_action = "failover"
            reason = "retry disposition requires failover before next attempt"
        elif should_retry:
            recommended_action = "retry"
            wait_ms = self.compute_delay_ms(
                normalized_attempt,
                profile=resolved_profile,
                retry_after_ms=retry_after_ms,
                correlation_id=correlation_id,
                message_id=message_id,
            )
        elif exhausted:
            recommended_action = "exhausted"
        elif classification["retryable"]:
            recommended_action = "retry"
        
        if self.require_idempotent_for_unsafe_retries and not idempotent and should_retry and not resolved_profile.allow_retry_when_non_idempotent:
            should_retry = False
            wait_ms = 0
            recommended_action = "fail"
            reason = "retry denied because the operation is non-idempotent under current policy"

        attempt_record = RetryAttemptRecord(
            profile_name=resolved_profile.name,
            attempt=normalized_attempt,
            next_attempt=(normalized_attempt + 1) if should_retry else None,
            should_retry=should_retry,
            exhausted=exhausted,
            requires_failover=requires_failover,
            wait_ms=wait_ms,
            retry_disposition=disposition.value if isinstance(disposition, RetryDisposition) else str(disposition),
            recommended_action=recommended_action,
            reason=reason,
            status_code=resolved_status_code,
            correlation_id=correlation_id,
            message_id=message_id,
            endpoint=self._safe_endpoint(endpoint),
            channel=normalize_channel_name(channel) if channel is not None else None,
            protocol=normalize_protocol_name(protocol) if protocol is not None else None,
            route=str(route).strip() if route is not None and str(route).strip() else None,
            error_snapshot=error_snapshot,
            metadata=normalized_metadata,
        )
        decision = RetryDecision(profile=resolved_profile, attempt_record=attempt_record, normalized_error=error_snapshot)

        with self._lock:
            self._stats["decisions"] += 1
            if should_retry:
                self._stats["retries_allowed"] += 1
            else:
                self._stats["retries_denied"] += 1
            if exhausted:
                self._stats["retry_exhausted"] += 1
            if requires_failover:
                self._stats["failover_required"] += 1
            self._append_history_locked(decision)
            self._sync_decision_memory(decision)
            self._sync_snapshot_memory()

        if should_retry or exhausted or requires_failover:
            self._record_retry_event(
                decision,
                endpoint=endpoint,
                channel=channel,
                route=route,
                correlation_id=correlation_id,
                message_id=message_id,
            )

        return decision.to_dict()

    def should_retry(self, *args: Any, **kwargs: Any) -> bool:
        return bool(self.evaluate(*args, **kwargs)["decision"]["should_retry"])

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
        configured_profiles = ensure_mapping(self.retry_policy_config.get("profiles"), field_name="profiles", allow_none=True)
        if not configured_profiles:
            configured_profiles = {
                "default": {},
                "aggressive": {
                    "max_attempts": max(self.default_max_attempts, 5),
                    "initial_backoff_ms": max(50, min(self.default_initial_backoff_ms, 200)),
                    "max_backoff_ms": max(self.default_max_backoff_ms, 3000),
                    "jitter_ratio": max(self.default_jitter_ratio, 0.25),
                },
                "conservative": {
                    "max_attempts": min(self.default_max_attempts, 2),
                    "initial_backoff_ms": max(self.default_initial_backoff_ms, 500),
                    "max_backoff_ms": max(self.default_max_backoff_ms, 8000),
                    "jitter_ratio": min(1.0, max(self.default_jitter_ratio, 0.10)),
                    "retry_on_unknown_errors": False,
                },
                "streaming": {
                    "max_attempts": max(2, self.default_max_attempts),
                    "initial_backoff_ms": max(100, min(self.default_initial_backoff_ms, 250)),
                    "max_backoff_ms": max(2000, min(self.default_max_backoff_ms, 5000)),
                    "allow_retry_when_non_idempotent": False,
                },
            }

        for profile_name, profile_config in configured_profiles.items():
            self._profiles[str(profile_name).strip().lower()] = self._profile_from_mapping(profile_name, ensure_mapping(profile_config, field_name="profile", allow_none=True))
            self._stats["profiles_loaded"] += 1
            logger.debug(f"Profiles after loading: {list(self._profiles.keys())}")

        if self.default_profile_name not in self._profiles:
            self._profiles[self.default_profile_name] = self._profile_from_mapping(self.default_profile_name, {})
            self._stats["profiles_loaded"] += 1

    def _profile_from_mapping(self, profile_name: str, profile_config: Mapping[str, Any]) -> RetryProfile:
        name = ensure_non_empty_string(str(profile_name), field_name="profile_name").strip().lower()
        cfg = ensure_mapping(profile_config, field_name="profile_config", allow_none=True)
        retryable_codes = tuple(
            sorted({int(item) for item in ensure_sequence(cfg.get("retryable_status_codes", _DEFAULT_RETRYABLE_STATUS_CODES), field_name="retryable_status_codes", allow_none=True, coerce_scalar=True)})
        )
        non_retryable_codes = tuple(
            sorted({int(item) for item in ensure_sequence(cfg.get("non_retryable_status_codes", _DEFAULT_NON_RETRYABLE_STATUS_CODES), field_name="non_retryable_status_codes", allow_none=True, coerce_scalar=True)})
        )
        return RetryProfile(
            name=name,
            max_attempts=max(1, self._coerce_int(cfg.get("max_attempts"), self.default_max_attempts, non_negative=True)),
            initial_backoff_ms=max(1, self._coerce_int(cfg.get("initial_backoff_ms"), self.default_initial_backoff_ms, non_negative=True)),
            max_backoff_ms=max(1, self._coerce_int(cfg.get("max_backoff_ms"), self.default_max_backoff_ms, non_negative=True)),
            backoff_multiplier=self._coerce_float(cfg.get("backoff_multiplier"), self.default_backoff_multiplier, minimum=1.0),
            backoff_strategy=self._validate_backoff_strategy(cfg.get("backoff_strategy") or self.default_backoff_strategy),
            jitter_ratio=self._coerce_float(cfg.get("jitter_ratio"), self.default_jitter_ratio, minimum=0.0, maximum=1.0),
            jitter_mode=self._validate_jitter_mode(cfg.get("jitter_mode") or self.default_jitter_mode),
            retry_on_timeouts=self._coerce_bool(cfg.get("retry_on_timeouts"), True),
            retry_on_connection_errors=self._coerce_bool(cfg.get("retry_on_connection_errors"), True),
            retry_on_rate_limits=self._coerce_bool(cfg.get("retry_on_rate_limits"), True),
            retry_on_server_errors=self._coerce_bool(cfg.get("retry_on_server_errors"), True),
            retry_on_unknown_errors=self._coerce_bool(cfg.get("retry_on_unknown_errors"), self.classify_unknown_errors_as_retryable),
            retry_on_duplicate_active=self._coerce_bool(cfg.get("retry_on_duplicate_active"), False),
            respect_retry_after=self._coerce_bool(cfg.get("respect_retry_after"), True),
            failover_on_required_disposition=self._coerce_bool(cfg.get("failover_on_required_disposition"), True),
            allow_retry_when_non_idempotent=self._coerce_bool(cfg.get("allow_retry_when_non_idempotent"), False),
            retryable_status_codes=retryable_codes,
            non_retryable_status_codes=non_retryable_codes,
            metadata=normalize_metadata(cfg.get("metadata")),
        )

    def _resolve_profile(
        self,
        profile: Optional[str | Mapping[str, Any] | RetryProfile],
    ) -> RetryProfile:
        if profile is None:
            profile_name = self.default_profile_name
            if profile_name not in self._profiles:
                raise NetworkConfigurationError(
                    "Default retry profile is not registered.",
                    context={"operation": "resolve_retry_profile"},
                    details={"default_profile": profile_name},
                )
            return self._profiles[profile_name]
    
        if isinstance(profile, RetryProfile):
            return profile
    
        if isinstance(profile, Mapping):
            return self._profile_from_mapping(str(profile.get("name") or "inline"), profile)
    
        profile_name = ensure_non_empty_string(str(profile), field_name="profile").strip().lower()
        if profile_name not in self._profiles:
            raise NetworkConfigurationError(
                "Requested retry profile is not registered.",
                context={"operation": "resolve_retry_profile"},
                details={"profile": profile_name},
            )
        return self._profiles[profile_name]

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
        status_code: Optional[int],
        metadata: Mapping[str, Any],
    ) -> Tuple[Optional[NetworkError], Optional[Dict[str, Any]], Optional[int]]:
        if error is None:
            return None, None, status_code

        if isinstance(error, Mapping):
            snapshot = json_safe(error)
            inferred_status_code = snapshot.get("status_code") if isinstance(snapshot, Mapping) else None
            return None, snapshot if isinstance(snapshot, dict) else {"error": snapshot}, int(inferred_status_code) if inferred_status_code is not None else status_code

        normalized_error = normalize_network_exception(
            error,
            operation="retry_policy_evaluate",
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route,
            correlation_id=correlation_id,
            metadata={"message_id": message_id, **dict(metadata)},
            status_code=status_code,
        )
        snapshot = normalized_error.to_memory_snapshot()
        return normalized_error, snapshot, normalized_error.status_code or status_code

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

    def _classify_retryability(
        self,
        error: Optional[NetworkError],
        *,
        status_code: Optional[int],
        profile: RetryProfile,
        idempotent: bool,
    ) -> Dict[str, Any]:
        if error is not None:
            code = getattr(error, "code", None)
            if isinstance(error, RateLimitExceededError) and profile.retry_on_rate_limits:
                return {"retryable": True, "reason": "rate-limit error is retryable under profile"}
            if isinstance(error, (ConnectionTimeoutError, DeliveryTimeoutError)) and profile.retry_on_timeouts:
                return {"retryable": True, "reason": "timeout error is retryable under profile"}
            if isinstance(error, (NetworkConnectionError, NetworkTransportError, RoutingError)) and profile.retry_on_connection_errors:
                return {"retryable": True, "reason": "connection or transport error is retryable under profile"}
            if isinstance(error, RetryExhaustedError):
                return {"retryable": False, "reason": "retry exhaustion is terminal for current policy"}
            if isinstance(error, CircuitBreakerOpenError):
                return {"retryable": True, "reason": "circuit is open; retry may proceed only after policy delay or failover"}
            if isinstance(error, DuplicateMessageError):
                return {"retryable": profile.retry_on_duplicate_active, "reason": "duplicate-active retryability follows profile configuration"}
            if error.retryable:
                return {"retryable": True, "reason": f"network error code {code or type(error).__name__} is marked retryable"}

        if status_code is not None and self.classify_http_statuses:
            if status_code in set(profile.non_retryable_status_codes):
                return {"retryable": False, "reason": f"HTTP status {status_code} is explicitly non-retryable"}
            if status_code in set(profile.retryable_status_codes):
                return {"retryable": True, "reason": f"HTTP status {status_code} is explicitly retryable"}
            if 500 <= status_code <= 599 and profile.retry_on_server_errors:
                return {"retryable": True, "reason": f"server status {status_code} is retryable under profile"}
            if status_code == 429 and profile.retry_on_rate_limits:
                return {"retryable": True, "reason": "HTTP 429 is retryable under profile"}

        if error is None:
            return {"retryable": False, "reason": "no error or status code was provided to classify"}

        if self.classify_network_errors and (is_retryable_exception(error) or is_transient_exception(error)):
            return {"retryable": True, "reason": "error is transient or retryable by exception classification"}

        if profile.retry_on_unknown_errors:
            return {"retryable": True, "reason": "unknown error is retryable under profile"}

        return {"retryable": False, "reason": "error is not retryable under current policy"}

    def _compute_base_delay_ms(self, attempt: int, profile: RetryProfile) -> int:
        strategy = profile.backoff_strategy
        if strategy == "fixed":
            base = profile.initial_backoff_ms
        elif strategy == "linear":
            base = profile.initial_backoff_ms * attempt
        elif strategy == "decorrelated":
            # Upper-bounded variant suitable for stateless evaluation.
            prev = profile.initial_backoff_ms * (profile.backoff_multiplier ** max(0, attempt - 2))
            base = random.randint(profile.initial_backoff_ms, min(profile.max_backoff_ms, int(prev * profile.backoff_multiplier)))
        else:
            base = int(profile.initial_backoff_ms * (profile.backoff_multiplier ** max(0, attempt - 1)))
        return max(0, min(int(base), int(profile.max_backoff_ms)))

    def _apply_jitter(
        self,
        base_delay_ms: int,
        *,
        attempt: int,
        profile: RetryProfile,
        correlation_id: Optional[str],
        message_id: Optional[str],
    ) -> int:
        jitter_ratio = max(0.0, min(1.0, float(profile.jitter_ratio)))
        if profile.jitter_mode == "none" or jitter_ratio <= 0.0 or base_delay_ms <= 0:
            return int(base_delay_ms)

        rnd = self._random_for_jitter(
            attempt=attempt,
            profile_name=profile.name,
            correlation_id=correlation_id,
            message_id=message_id,
        )
        spread = int(base_delay_ms * jitter_ratio)

        if profile.jitter_mode == "equal":
            lower = max(0, base_delay_ms - spread)
            upper = min(profile.max_backoff_ms, base_delay_ms + spread)
            return int(rnd.uniform(lower, upper))
        if profile.jitter_mode == "decorrelated":
            lower = max(profile.initial_backoff_ms, base_delay_ms // 2)
            upper = min(profile.max_backoff_ms, max(lower, int(base_delay_ms * 1.5)))
            return int(rnd.uniform(lower, upper))
        # full jitter
        return int(rnd.uniform(0, min(profile.max_backoff_ms, max(1, base_delay_ms))))

    def _random_for_jitter(
        self,
        *,
        attempt: int,
        profile_name: str,
        correlation_id: Optional[str],
        message_id: Optional[str],
    ) -> random.Random:
        if not self.use_deterministic_jitter:
            return random.Random()
        seed = generate_idempotency_key(
            {
                "attempt": attempt,
                "profile": profile_name,
                "correlation_id": correlation_id,
                "message_id": message_id,
            },
            namespace="retry_policy_jitter",
        )
        return random.Random(seed)

    def _append_history_locked(self, decision: RetryDecision) -> None:
        payload = decision.to_dict()
        self._history.append(payload)
        if self.record_history:
            self.memory.append(
                _RETRY_POLICY_HISTORY_KEY,
                payload,
                max_items=self.max_history_size,
                ttl_seconds=self.history_ttl_seconds,
                source="retry_policy",
            )

    def _sync_decision_memory(self, decision: RetryDecision) -> None:
        if not self.record_memory_snapshots:
            return
        self.memory.set(
            _RETRY_POLICY_LAST_KEY,
            decision.to_dict(),
            ttl_seconds=self.snapshot_ttl_seconds,
            source="retry_policy",
        )

    def _sync_snapshot_memory(self) -> None:
        if not self.record_memory_snapshots:
            return
        self.memory.set(
            _RETRY_POLICY_PROFILE_KEY,
            {"profiles": [profile.to_dict() for profile in self._profiles.values()], "generated_at": utc_timestamp()},
            ttl_seconds=self.snapshot_ttl_seconds,
            source="retry_policy",
        )
        self.memory.set(
            _RETRY_POLICY_SNAPSHOT_KEY,
            self.get_snapshot(),
            ttl_seconds=self.snapshot_ttl_seconds,
            source="retry_policy",
        )

    def _record_retry_event(
        self,
        decision: RetryDecision,
        *,
        endpoint: Optional[str],
        channel: Optional[str],
        route: Optional[str],
        correlation_id: Optional[str],
        message_id: Optional[str],
    ) -> None:
        attempt_record = decision.attempt_record
        if attempt_record.error_snapshot is None:
            return
        try:
            self.memory.record_retry_event(
                attempt_record.error_snapshot,
                attempt=attempt_record.attempt,
                max_attempts=decision.profile.max_attempts,
                endpoint=self._safe_endpoint(endpoint),
                channel=normalize_channel_name(channel) if channel is not None else None,
                route=str(route).strip() if route is not None and str(route).strip() else None,
                correlation_id=correlation_id,
                message_id=message_id,
                metadata={
                    "profile": decision.profile.name,
                    "wait_ms": attempt_record.wait_ms,
                    "recommended_action": attempt_record.recommended_action,
                    "requires_failover": attempt_record.requires_failover,
                },
            )
        except Exception:
            # Retry telemetry should not break retry evaluation.
            return

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
        return self._coerce_bool(self.retry_policy_config.get(name, default), default)

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        return self._coerce_int(self.retry_policy_config.get(name, default), default, non_negative=True)

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.retry_policy_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_backoff_strategy_config(self, name: str, default: str) -> str:
        value = str(self.retry_policy_config.get(name, default)).strip().lower() or default
        return self._validate_backoff_strategy(value)

    def _get_jitter_mode_config(self, name: str, default: str) -> str:
        value = str(self.retry_policy_config.get(name, default)).strip().lower() or default
        return self._validate_jitter_mode(value)

    def _validate_backoff_strategy(self, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in _VALID_BACKOFF_STRATEGIES:
            raise NetworkConfigurationError(
                "Invalid retry backoff strategy in configuration.",
                context={"operation": "retry_policy_config"},
                details={"backoff_strategy": value, "allowed": sorted(_VALID_BACKOFF_STRATEGIES)},
            )
        return normalized

    def _validate_jitter_mode(self, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in _VALID_JITTER_MODES:
            raise NetworkConfigurationError(
                "Invalid retry jitter mode in configuration.",
                context={"operation": "retry_policy_config"},
                details={"jitter_mode": value, "allowed": sorted(_VALID_JITTER_MODES)},
            )
        return normalized

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
            "Invalid boolean value in retry policy configuration.",
            context={"operation": "retry_policy_config"},
            details={"config_value": value},
        )

    def _coerce_int(self, value: Any, default: int, *, non_negative: bool = False) -> int:
        if value is None:
            value = default
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in retry policy configuration.",
                context={"operation": "retry_policy_config"},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if non_negative and coerced < 0:
            raise NetworkConfigurationError(
                "Retry policy integer configuration value must be non-negative.",
                context={"operation": "retry_policy_config"},
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
                "Invalid float value in retry policy configuration.",
                context={"operation": "retry_policy_config"},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if minimum is not None and coerced < minimum:
            raise NetworkConfigurationError(
                "Retry policy float configuration value is below the allowed minimum.",
                context={"operation": "retry_policy_config"},
                details={"config_value": value, "minimum": minimum},
            )
        if maximum is not None and coerced > maximum:
            raise NetworkConfigurationError(
                "Retry policy float configuration value exceeds the allowed maximum.",
                context={"operation": "retry_policy_config"},
                details={"config_value": value, "maximum": maximum},
            )
        return coerced


if __name__ == "__main__":
    print("\n=== Running Retry Policy ===\n")
    printer.status("TEST", "Retry Policy initialized", "info")

    memory = NetworkMemory()
    policy = RetryPolicy(memory=memory)

    timeout_decision = policy.evaluate(
        TimeoutError("upstream timeout"),
        attempt=1,
        profile="default",
        endpoint="https://api.example.com/v1/jobs",
        channel="http",
        protocol="http",
        route="primary",
        correlation_id="corr_retry_001",
        message_id="msg_retry_001",
        idempotent=True,
        metadata={"phase": "send"},
    )
    printer.status("TEST", "Timeout retry decision generated", "info")

    rate_limited = policy.evaluate(
        network_error_from_http_status(
            429,
            endpoint="https://api.example.com/v1/jobs",
            operation="send",
            channel="http",
            protocol="http",
            retry_after_ms=1200,
        ),
        attempt=2,
        profile="default",
        retry_after_ms=1200,
        endpoint="https://api.example.com/v1/jobs",
        channel="http",
        protocol="http",
        route="primary",
        correlation_id="corr_retry_002",
        message_id="msg_retry_002",
        idempotent=True,
    )
    printer.status("TEST", "Rate-limit retry decision generated", "info")

    exhausted = policy.evaluate(
        TimeoutError("persistent timeout"),
        attempt=3,
        max_attempts=3,
        profile="conservative",
        endpoint="https://api.example.com/v1/jobs",
        channel="http",
        protocol="http",
        route="primary",
        correlation_id="corr_retry_003",
        message_id="msg_retry_003",
        idempotent=True,
    )
    printer.status("TEST", "Exhausted retry decision generated", "info")

    failover_required = policy.evaluate(
        NegativeAcknowledgementError(
            "Transport requested reroute.",
            context={"operation": "send", "channel": "queue", "protocol": "queue", "endpoint": "amqp://broker.internal:5672/vhost"},
        ),
        attempt=1,
        profile="default",
        endpoint="amqp://broker.internal:5672/vhost",
        channel="queue",
        protocol="queue",
        route="secondary",
        correlation_id="corr_retry_004",
        message_id="msg_retry_004",
        idempotent=True,
    )
    printer.status("TEST", "Failover-aware retry decision generated", "info")

    snapshot = policy.get_snapshot()

    print("Timeout Decision:", stable_json_dumps(timeout_decision))
    print("Rate-Limited Decision:", stable_json_dumps(rate_limited))
    print("Exhausted Decision:", stable_json_dumps(exhausted))
    print("Failover Decision:", stable_json_dumps(failover_required))
    print("Snapshot:", stable_json_dumps(snapshot))

    assert timeout_decision["decision"]["should_retry"] is True
    assert rate_limited["decision"]["wait_ms"] >= 1200
    assert exhausted["decision"]["exhausted"] is True
    assert failover_required["decision"]["requires_failover"] is True
    assert memory.get("network.reliability.retry_policy.last") is not None
    assert memory.get("network.reliability.retry_policy.snapshot") is not None

    printer.status("TEST", "All Retry Policy checks passed", "success")
    print("\n=== Test ran successfully ===\n")
