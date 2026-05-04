"""
Delivery state machine for SLAI's Network Agent lifecycle subsystem.

This module provides the production-grade delivery state machine that sits
beneath NetworkLifecycle and above transport adapters. It centralizes message
and delivery-state ownership so the broader network stack can reason about
queued, sent, received, acknowledged, failed, expired, retrying, and
dead-lettered deliveries through a single consistent contract.

The state machine is intentionally scoped to delivery-state ownership. It is
responsible for:
- canonical envelope normalization and registration,
- transition validation and terminal-state enforcement,
- retry bookkeeping and dead-letter escalation,
- idempotency-aware message registration,
- lifecycle timestamps, retry counters, and expiration windows,
- structured synchronization into NetworkMemory for the wider network stack.

It does not own transport execution, route selection, policy arbitration, or
retry timing strategy. Those concerns belong to adapters, routing, policy, and
reliability modules. This module owns the delivery truth model those layers
update and consume.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils import *
from ..network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Delivery State Machine")
printer = PrettyPrinter()


_DELIVERY_STATE_MACHINE_LAST_KEY = "network.lifecycle.delivery_state_machine.last"
_DELIVERY_STATE_MACHINE_SNAPSHOT_KEY = "network.lifecycle.delivery_state_machine.snapshot"
_DELIVERY_STATE_MACHINE_HISTORY_KEY = "network.lifecycle.delivery_state_machine.history"
_DELIVERY_STATE_MACHINE_ACTIVE_KEY = "network.lifecycle.delivery_state_machine.active"
_DELIVERY_STATE_MACHINE_DEAD_LETTER_KEY = "network.lifecycle.delivery_state_machine.dead_letter"

_DEFAULT_TERMINAL_STATES = ("acked", "expired", "dead_lettered", "cancelled")
_DEFAULT_ACTIVE_STATES = ("initialized", "queued", "selected", "sent", "received", "retrying", "failed", "nacked")
_DEFAULT_STATE_ORDER = (
    "initialized",
    "queued",
    "selected",
    "sent",
    "received",
    "acked",
    "nacked",
    "retrying",
    "failed",
    "expired",
    "dead_lettered",
    "cancelled",
)

_DEFAULT_ALLOWED_TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    "initialized": ("queued", "selected", "failed", "expired", "cancelled", "dead_lettered"),
    "queued": ("selected", "sent", "failed", "expired", "cancelled", "dead_lettered"),
    "selected": ("queued", "sent", "failed", "expired", "cancelled", "dead_lettered"),
    "sent": ("received", "acked", "nacked", "retrying", "failed", "expired", "dead_lettered"),
    "received": ("acked", "nacked", "retrying", "failed", "expired", "dead_lettered"),
    "nacked": ("retrying", "queued", "failed", "dead_lettered", "expired"),
    "retrying": ("queued", "selected", "sent", "failed", "dead_lettered", "expired"),
    "failed": ("retrying", "queued", "dead_lettered", "expired"),
    "acked": (),
    "expired": (),
    "dead_lettered": (),
    "cancelled": (),
}


@dataclass(slots=True)
class DeliveryTransitionRecord:
    """Single transition event emitted by the delivery state machine."""

    message_id: str
    from_state: str
    to_state: str
    occurred_at: str
    correlation_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    route: Optional[str] = None
    endpoint: Optional[str] = None
    channel: Optional[str] = None
    retry_count: int = 0
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "message_id": self.message_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "occurred_at": self.occurred_at,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "route": self.route,
            "endpoint": self.endpoint,
            "channel": self.channel,
            "retry_count": self.retry_count,
            "error": self.error,
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class DeliveryRecord:
    """Authoritative delivery record tracked by the state machine."""

    message_id: str
    correlation_id: str
    idempotency_key: str
    state: str
    envelope: Dict[str, Any]
    channel: str
    protocol: str
    endpoint: Optional[str] = None
    route: Optional[str] = None
    delivery_mode: str = "at_least_once"
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    queued_at: Optional[str] = None
    selected_at: Optional[str] = None
    sent_at: Optional[str] = None
    received_at: Optional[str] = None
    acked_at: Optional[str] = None
    nacked_at: Optional[str] = None
    failed_at: Optional[str] = None
    expired_at: Optional[str] = None
    dead_lettered_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    expires_at: Optional[str] = None
    retry_count: int = 0
    max_attempts: Optional[int] = None
    last_error: Optional[Dict[str, Any]] = None
    payload_fingerprint: Optional[str] = None
    dead_letter_queue: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    transition_history: List[Dict[str, Any]] = field(default_factory=list)

    def is_terminal(self, terminal_states: Sequence[str]) -> bool:
        return self.state in set(terminal_states)

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            expiry = _parse_timestamp(self.expires_at)
        except Exception:
            return False
        return expiry <= _utcnow()

    def to_dict(self, *, include_envelope: bool = True, include_history: bool = True) -> Dict[str, Any]:
        payload = {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "state": self.state,
            "channel": self.channel,
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "route": self.route,
            "delivery_mode": self.delivery_mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "queued_at": self.queued_at,
            "selected_at": self.selected_at,
            "sent_at": self.sent_at,
            "received_at": self.received_at,
            "acked_at": self.acked_at,
            "nacked_at": self.nacked_at,
            "failed_at": self.failed_at,
            "expired_at": self.expired_at,
            "dead_lettered_at": self.dead_lettered_at,
            "cancelled_at": self.cancelled_at,
            "expires_at": self.expires_at,
            "retry_count": self.retry_count,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
            "payload_fingerprint": self.payload_fingerprint,
            "dead_letter_queue": self.dead_letter_queue,
            "metadata": json_safe(self.metadata),
            "active": not self.is_terminal(_DEFAULT_TERMINAL_STATES),
        }
        if include_envelope:
            payload["envelope"] = json_safe(self.envelope)
        if include_history:
            payload["transition_history"] = json_safe(self.transition_history)
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


class DeliveryStateMachine:
    """
    Canonical delivery-state owner for the network lifecycle domain.

    The state machine keeps an in-process authoritative view of message
    deliveries while synchronizing the important lifecycle moments into
    NetworkMemory so the rest of the network subsystem can observe state
    without directly depending on this module's internal storage.
    """

    def __init__(self, memory: Optional[NetworkMemory] = None, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.state_machine_config = merge_mappings(
            get_config_section("network_delivery_state_machine") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_transition_history = self._get_bool_config("record_transition_history", True)
        self.record_memory_snapshots = self._get_bool_config("record_memory_snapshots", True)
        self.strict_transitions = self._get_bool_config("strict_transitions", True)
        self.enforce_idempotency = self._get_bool_config("enforce_idempotency", True)
        self.allow_terminal_reentry = self._get_bool_config("allow_terminal_reentry", False)
        self.auto_expire_on_read = self._get_bool_config("auto_expire_on_read", True)
        self.auto_register_on_transition = self._get_bool_config("auto_register_on_transition", False)
        self.auto_increment_retry_on_retrying = self._get_bool_config("auto_increment_retry_on_retrying", True)
        self.enable_dead_letter = self._get_bool_config("enable_dead_letter", True)
        self.require_correlation_id = self._get_bool_config("require_correlation_id", True)
        self.require_idempotency_key = self._get_bool_config("require_idempotency_key", True)

        self.default_protocol = normalize_protocol_name(self._get_optional_string_config("default_protocol") or "http")
        self.default_channel = normalize_channel_name(self._get_optional_string_config("default_channel") or self.default_protocol)
        self.default_initial_state = self._get_state_name("default_initial_state", "queued")
        self.default_delivery_mode = ensure_non_empty_string(
            str(self.state_machine_config.get("default_delivery_mode", "at_least_once")),
            field_name="default_delivery_mode",
        ).lower()

        self.default_ttl_seconds = self._get_non_negative_int_config("default_ttl_seconds", 7200)
        self.pending_expire_after_seconds = self._get_non_negative_int_config("pending_expire_after_seconds", 1800)
        self.sent_expire_after_seconds = self._get_non_negative_int_config("sent_expire_after_seconds", 7200)
        self.failed_expire_after_seconds = self._get_non_negative_int_config("failed_expire_after_seconds", 7200)
        self.dead_letter_ttl_seconds = self._get_non_negative_int_config("dead_letter_ttl_seconds", 86400)
        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 7200)
        self.max_deliveries = max(1, self._get_non_negative_int_config("max_deliveries", 5000))
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))
        self.max_transition_history_per_message = max(1, self._get_non_negative_int_config("max_transition_history_per_message", 50))
        self.default_max_attempts = max(1, self._get_non_negative_int_config("default_max_attempts", 3))

        self.terminal_states = self._get_state_sequence("terminal_states", _DEFAULT_TERMINAL_STATES)
        self.active_states = self._get_state_sequence("active_states", _DEFAULT_ACTIVE_STATES)
        self.state_order = self._get_state_sequence("state_order", _DEFAULT_STATE_ORDER)
        self.allowed_transitions = self._build_allowed_transitions()

        self._deliveries: Dict[str, DeliveryRecord] = {}
        self._by_correlation_id: Dict[str, str] = {}
        self._by_idempotency_key: Dict[str, str] = {}
        self._dead_letter: Dict[str, Dict[str, Any]] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history_size)
        self._stats: Dict[str, int] = {
            "registrations": 0,
            "transitions": 0,
            "acks": 0,
            "nacks": 0,
            "failures": 0,
            "expirations": 0,
            "dead_letters": 0,
            "retry_increments": 0,
            "purges": 0,
            "evictions": 0,
        }
        self._started_at = utc_timestamp()

        self._sync_snapshot_memory()

    # ------------------------------------------------------------------
    # Registration and lookup
    # ------------------------------------------------------------------
    def register_delivery(
        self,
        envelope: Optional[Mapping[str, Any]] = None,
        *,
        payload: Any = None,
        state: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        endpoint: Optional[str] = None,
        route: Optional[str] = None,
        delivery_mode: Optional[str] = None,
        timeout_ms: Optional[Any] = None,
        max_attempts: Optional[int] = None,
        expires_in_seconds: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError("DeliveryStateMachine is disabled by configuration.", context={"operation": "register_delivery"})

        built_envelope = build_message_envelope(
            ensure_mapping(envelope, field_name="envelope", allow_none=True),
            payload=payload,
            channel=channel,
            protocol=protocol,
            endpoint=endpoint,
            route=route,
            timeout_ms=timeout_ms,
            delivery_mode=delivery_mode or self.default_delivery_mode,
            metadata=metadata,
        )
        normalized_state = self._get_state_name("state", state or self.default_initial_state)
        delivery = self._record_from_envelope(
            built_envelope,
            state=normalized_state,
            max_attempts=max_attempts,
            expires_in_seconds=expires_in_seconds,
        )

        with self._lock:
            existing = self._deliveries.get(delivery.message_id)
            if existing is not None:
                raise DuplicateMessageError(
                    "A delivery with the same message_id is already registered.",
                    context={
                        "operation": "register_delivery",
                        "channel": delivery.channel,
                        "protocol": delivery.protocol,
                        "endpoint": delivery.endpoint,
                        "correlation_id": delivery.correlation_id,
                        "idempotency_key": delivery.idempotency_key,
                    },
                    details={"message_id": delivery.message_id},
                )

            if self.enforce_idempotency and delivery.idempotency_key in self._by_idempotency_key:
                existing_id = self._by_idempotency_key[delivery.idempotency_key]
                existing_record = self._deliveries.get(existing_id)
                if existing_record is not None and existing_record.payload_fingerprint != delivery.payload_fingerprint:
                    raise IdempotencyViolationError(
                        "An existing idempotency key was reused with a different payload fingerprint.",
                        context={
                            "operation": "register_delivery",
                            "channel": delivery.channel,
                            "protocol": delivery.protocol,
                            "endpoint": delivery.endpoint,
                            "correlation_id": delivery.correlation_id,
                            "idempotency_key": delivery.idempotency_key,
                        },
                        details={
                            "message_id": delivery.message_id,
                            "existing_message_id": existing_id,
                        },
                    )

            self._ensure_capacity_locked(incoming_key=delivery.message_id)
            self._deliveries[delivery.message_id] = delivery
            self._by_correlation_id[delivery.correlation_id] = delivery.message_id
            self._by_idempotency_key[delivery.idempotency_key] = delivery.message_id
            self._stats["registrations"] += 1
            self._append_transition_locked(delivery, from_state="untracked", to_state=delivery.state, metadata={"registration": True})
            self._sync_delivery_memory(delivery)
            self._sync_snapshot_memory()
            return delivery.to_dict()
        
    
    def _ensure_capacity_locked(self, *, incoming_key: str) -> None:
        """
        Enforce the configured maximum number of tracked deliveries.
    
        Capacity eviction is conservative:
        1. Never evict the incoming key if it already exists.
        2. Try to auto-expire stale active records first.
        3. Evict the oldest terminal record if capacity is still exhausted.
        4. Fail fast if only active non-terminal deliveries remain.
        """
        normalized_key = ensure_non_empty_string(str(incoming_key), field_name="incoming_key")
    
        if normalized_key in self._deliveries:
            return
    
        if len(self._deliveries) < self.max_deliveries:
            return
    
        # First pass: convert stale active records into terminal expired records.
        self._expire_stale_locked()
    
        if len(self._deliveries) < self.max_deliveries:
            return
    
        terminal_candidates = [
            record
            for record in self._deliveries.values()
            if record.is_terminal(self.terminal_states)
        ]
    
        if terminal_candidates:
            oldest_terminal = min(
                terminal_candidates,
                key=lambda item: _parse_timestamp(item.updated_at),
            )
            self._evict_record_locked(
                oldest_terminal.message_id,
                reason="capacity_eviction",
            )
            return
    
        raise DeliveryStateError(
            "Delivery state machine capacity has been reached and no terminal records are available for eviction.",
            context={"operation": "register_delivery"},
            details={
                "incoming_key": normalized_key,
                "max_deliveries": self.max_deliveries,
                "active_count": len(self._deliveries),
            },
        )
    
    
    def _evict_record_locked(self, message_id: str, *, reason: str) -> None:
        """
        Evict a tracked delivery record while preserving an auditable history trail.
        """
        normalized_message_id = ensure_non_empty_string(str(message_id), field_name="message_id")
        record = self._deliveries.get(normalized_message_id)
        if record is None:
            return
    
        eviction_event = {
            "message_id": record.message_id,
            "state": record.state,
            "reason": ensure_non_empty_string(reason, field_name="reason"),
            "occurred_at": utc_timestamp(),
            "correlation_id": record.correlation_id,
            "idempotency_key": record.idempotency_key,
            "channel": record.channel,
            "protocol": record.protocol,
            "endpoint": record.endpoint,
            "route": record.route,
        }
    
        self._history.append(json_safe(eviction_event))
        self.memory.append(
            _DELIVERY_STATE_MACHINE_HISTORY_KEY,
            eviction_event,
            max_items=self.max_history_size,
            ttl_seconds=self.history_ttl_seconds,
            source="delivery_state_machine",
        )
    
        self._remove_record_locked(normalized_message_id)
        self._stats["evictions"] += 1

    def get_delivery(self, message_or_key: str, *, include_history: bool = True) -> Optional[Dict[str, Any]]:
        record = self._find_record(message_or_key, auto_expire=self.auto_expire_on_read)
        if record is None:
            return None
        return record.to_dict(include_history=include_history)

    def require_delivery(self, message_or_key: str, *, include_history: bool = True) -> Dict[str, Any]:
        record = self._find_record(message_or_key, auto_expire=self.auto_expire_on_read)
        if record is None:
            raise DeliveryStateError(
                "Requested delivery record is not registered.",
                context={"operation": "require_delivery"},
                details={"lookup": str(message_or_key)},
            )
        return record.to_dict(include_history=include_history)

    def list_deliveries(
        self,
        *,
        state: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        include_terminal: bool = True,
        include_history: bool = False,
    ) -> List[Dict[str, Any]]:
        normalized_state = self._get_state_name("state", state) if state is not None else None
        normalized_channel = normalize_channel_name(channel) if channel is not None else None
        normalized_protocol = normalize_protocol_name(protocol) if protocol is not None else None
        with self._lock:
            self._expire_stale_locked()
            payload: List[Dict[str, Any]] = []
            for record in self._deliveries.values():
                if normalized_state is not None and record.state != normalized_state:
                    continue
                if normalized_channel is not None and record.channel != normalized_channel:
                    continue
                if normalized_protocol is not None and record.protocol != normalized_protocol:
                    continue
                if not include_terminal and record.is_terminal(self.terminal_states):
                    continue
                payload.append(record.to_dict(include_history=include_history))
            payload.sort(key=self._sort_delivery_dict)
            return payload

    # ------------------------------------------------------------------
    # Transition operations
    # ------------------------------------------------------------------
    def transition(
        self,
        message_or_key: str,
        to_state: str,
        *,
        error: Optional[BaseException | Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        update_fields: Optional[Mapping[str, Any]] = None,
        increment_retry: Optional[bool] = None,
        expires_in_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized_to_state = self._get_state_name("to_state", to_state)
        with self._lock:
            record = self._require_record_locked(message_or_key)
            self._expire_record_if_needed_locked(record)

            from_state = record.state
            if from_state == normalized_to_state:
                self._apply_record_updates_locked(record, update_fields, metadata, expires_in_seconds)
                self._sync_delivery_memory(record)
                return record.to_dict()

            self._validate_transition_locked(record, normalized_to_state)
            normalized_error = self._normalize_error(error, record=record) if error is not None else None
            retry_increment = bool(increment_retry) if increment_retry is not None else (
                self.auto_increment_retry_on_retrying and normalized_to_state == "retrying"
            )
            if retry_increment:
                record.retry_count += 1
                self._stats["retry_increments"] += 1
                self.memory.increment_retry_count(
                    message_id=record.message_id,
                    correlation_id=record.correlation_id,
                    amount=1,
                    ttl_seconds=self.default_ttl_seconds,
                )
                if normalized_error is not None:
                    self.memory.record_retry_event(
                        normalized_error,
                        attempt=record.retry_count,
                        max_attempts=record.max_attempts,
                        endpoint=record.endpoint,
                        channel=record.channel,
                        route=record.route,
                        correlation_id=record.correlation_id,
                        message_id=record.message_id,
                        metadata=normalize_metadata(metadata),
                    )

            record.state = normalized_to_state
            record.updated_at = utc_timestamp()
            record.last_error = normalized_error or record.last_error
            self._stamp_state_timestamp_locked(record, normalized_to_state)
            self._apply_record_updates_locked(record, update_fields, metadata, expires_in_seconds)

            if normalized_to_state == "acked":
                self._stats["acks"] += 1
            elif normalized_to_state == "nacked":
                self._stats["nacks"] += 1
            elif normalized_to_state == "failed":
                self._stats["failures"] += 1
            elif normalized_to_state == "expired":
                self._stats["expirations"] += 1
            elif normalized_to_state == "dead_lettered":
                self._stats["dead_letters"] += 1
                self._record_dead_letter_locked(record, metadata=metadata)

            self._stats["transitions"] += 1
            self._append_transition_locked(record, from_state=from_state, to_state=normalized_to_state, error=normalized_error, metadata=metadata)
            self._sync_delivery_memory(record)
            self._sync_snapshot_memory()
            return record.to_dict()

    def mark_queued(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "queued", **kwargs)

    def mark_selected(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "selected", **kwargs)

    def mark_sent(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "sent", **kwargs)

    def mark_received(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "received", **kwargs)

    def mark_acked(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "acked", **kwargs)

    def mark_nacked(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "nacked", **kwargs)

    def mark_retrying(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("increment_retry", True)
        return self.transition(message_or_key, "retrying", **kwargs)

    def mark_failed(self, message_or_key: str, error: Optional[BaseException | Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "failed", error=error, **kwargs)

    def mark_expired(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "expired", **kwargs)

    def mark_dead_lettered(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "dead_lettered", **kwargs)

    def cancel(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(message_or_key, "cancelled", **kwargs)

    def expire_stale_deliveries(self) -> List[Dict[str, Any]]:
        with self._lock:
            expired: List[Dict[str, Any]] = []
            for record in list(self._deliveries.values()):
                if record.is_terminal(self.terminal_states):
                    continue
                if record.is_expired():
                    from_state = record.state
                    record.state = "expired"
                    record.updated_at = utc_timestamp()
                    self._stamp_state_timestamp_locked(record, "expired")
                    self._stats["expirations"] += 1
                    self._stats["transitions"] += 1
                    self._append_transition_locked(record, from_state=from_state, to_state="expired", metadata={"auto_expired": True})
                    self._sync_delivery_memory(record)
                    expired.append(record.to_dict())
            if expired:
                self._sync_snapshot_memory()
            return expired

    def purge_terminal_deliveries(self, *, older_than_seconds: Optional[int] = None) -> int:
        threshold_seconds = self.default_ttl_seconds if older_than_seconds is None else max(0, int(older_than_seconds))
        cutoff = _utcnow() - timedelta(seconds=threshold_seconds)
        purged = 0
        with self._lock:
            for message_id, record in list(self._deliveries.items()):
                if not record.is_terminal(self.terminal_states):
                    continue
                updated_at = _parse_timestamp(record.updated_at)
                if updated_at > cutoff:
                    continue
                self._remove_record_locked(message_id)
                purged += 1
            if purged:
                self._stats["purges"] += purged
                self._sync_snapshot_memory()
            return purged

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._expire_stale_locked()
            active = [record.to_dict(include_history=False) for record in self._deliveries.values() if not record.is_terminal(self.terminal_states)]
            terminal = [record.to_dict(include_history=False) for record in self._deliveries.values() if record.is_terminal(self.terminal_states)]
            return {
                "generated_at": utc_timestamp(),
                "started_at": self._started_at,
                "stats": dict(self._stats),
                "delivery_count": len(self._deliveries),
                "active_count": len(active),
                "terminal_count": len(terminal),
                "dead_letter_count": len(self._dead_letter),
                "active": sorted(active, key=self._sort_delivery_dict),
                "terminal": sorted(terminal, key=self._sort_delivery_dict),
                "history_size": len(self._history),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_from_envelope(
        self,
        envelope: Mapping[str, Any],
        *,
        state: str,
        max_attempts: Optional[int],
        expires_in_seconds: Optional[int],
    ) -> DeliveryRecord:
        envelope_map = ensure_mapping(envelope, field_name="envelope")
        message_id = ensure_non_empty_string(str(envelope_map.get("message_id")), field_name="message_id")
        correlation_id = ensure_non_empty_string(str(envelope_map.get("correlation_id")), field_name="correlation_id")
        idempotency_key = ensure_non_empty_string(str(envelope_map.get("idempotency_key")), field_name="idempotency_key")

        if self.require_correlation_id and not correlation_id:
            raise PayloadValidationError("correlation_id is required for delivery registration.", context={"operation": "register_delivery"})
        if self.require_idempotency_key and not idempotency_key:
            raise PayloadValidationError("idempotency_key is required for delivery registration.", context={"operation": "register_delivery"})

        normalized_channel = normalize_channel_name(envelope_map.get("channel") or self.default_channel)
        normalized_protocol = normalize_protocol_name(envelope_map.get("protocol") or normalized_channel or self.default_protocol)
        delivery_mode = ensure_non_empty_string(str(envelope_map.get("delivery_mode") or self.default_delivery_mode), field_name="delivery_mode").lower()
        payload_fingerprint = self._payload_fingerprint(envelope_map.get("payload"))
        expires_at = self._resolve_expiry_timestamp(state, envelope_map, expires_in_seconds)
        
        dead_letter_queue: Optional[str] = None
        
        if envelope_map.get("dead_letter_queue") is not None:
            dead_letter_queue = str(envelope_map.get("dead_letter_queue")).strip() or None
        else:
            envelope_metadata = ensure_mapping(
                envelope_map.get("metadata"),
                field_name="envelope_metadata",
                allow_none=True,
            )
            if envelope_metadata.get("dead_letter_queue") is not None:
                dead_letter_queue = str(envelope_metadata.get("dead_letter_queue")).strip() or None
        
        record = DeliveryRecord(
            message_id=message_id,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            state=state,
            envelope=json_safe(envelope_map),
            channel=normalized_channel,
            protocol=normalized_protocol,
            endpoint=self._safe_endpoint(envelope_map.get("endpoint")),
            route=str(envelope_map.get("route")) if envelope_map.get("route") is not None else None,
            delivery_mode=delivery_mode,
            created_at=str(envelope_map.get("created_at") or utc_timestamp()),
            updated_at=utc_timestamp(),
            retry_count=int(envelope_map.get("retry_count") or 0),
            max_attempts=max(1, int(max_attempts or envelope_map.get("max_attempts") or self.default_max_attempts)),
            last_error=None,
            payload_fingerprint=payload_fingerprint,
            dead_letter_queue=dead_letter_queue,
            expires_at=expires_at,
            metadata=normalize_metadata(envelope_map.get("metadata")),
        )
        self._stamp_state_timestamp_locked(record, state)
        return record

    def _find_record(self, message_or_key: str, *, auto_expire: bool) -> Optional[DeliveryRecord]:
        lookup = ensure_non_empty_string(str(message_or_key), field_name="message_or_key")
        with self._lock:
            record = self._resolve_record_locked(lookup)
            if record is None:
                return None
            if auto_expire:
                self._expire_record_if_needed_locked(record)
            return record

    def _require_record_locked(self, message_or_key: str) -> DeliveryRecord:
        record = self._resolve_record_locked(str(message_or_key))
        if record is None:
            raise DeliveryStateError(
                "Requested delivery record is not registered.",
                context={"operation": "require_delivery"},
                details={"lookup": str(message_or_key)},
            )
        return record

    def _resolve_record_locked(self, lookup: str) -> Optional[DeliveryRecord]:
        if lookup in self._deliveries:
            return self._deliveries[lookup]
        message_id = self._by_correlation_id.get(lookup) or self._by_idempotency_key.get(lookup)
        if message_id is None:
            return None
        return self._deliveries.get(message_id)

    def _validate_transition_locked(self, record: DeliveryRecord, to_state: str) -> None:
        if record.is_terminal(self.terminal_states) and not self.allow_terminal_reentry:
            raise DeliveryStateError(
                "Cannot transition a terminal delivery record without terminal reentry enabled.",
                context={
                    "operation": "delivery_transition",
                    "channel": record.channel,
                    "protocol": record.protocol,
                    "endpoint": record.endpoint,
                    "correlation_id": record.correlation_id,
                    "idempotency_key": record.idempotency_key,
                },
                details={"message_id": record.message_id, "state": record.state, "target_state": to_state},
            )
        # Dead-lettering is always allowed from any non-terminal state if enabled
        if to_state == "dead_lettered" and self.enable_dead_letter and not record.is_terminal(self.terminal_states):
            return
        allowed = set(self.allowed_transitions.get(record.state, ()))
        if self.strict_transitions and to_state not in allowed:
            raise DeliveryStateError(
                "Requested delivery transition is not allowed by the state machine.",
                context={
                    "operation": "delivery_transition",
                    "channel": record.channel,
                    "protocol": record.protocol,
                    "endpoint": record.endpoint,
                    "correlation_id": record.correlation_id,
                    "idempotency_key": record.idempotency_key,
                },
                details={
                    "message_id": record.message_id,
                    "from_state": record.state,
                    "to_state": to_state,
                    "allowed_transitions": sorted(allowed),
                },
            )

    def _append_transition_locked(
        self,
        record: DeliveryRecord,
        *,
        from_state: str,
        to_state: str,
        error: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        transition = DeliveryTransitionRecord(
            message_id=record.message_id,
            from_state=from_state,
            to_state=to_state,
            occurred_at=utc_timestamp(),
            correlation_id=record.correlation_id,
            idempotency_key=record.idempotency_key,
            route=record.route,
            endpoint=record.endpoint,
            channel=record.channel,
            retry_count=record.retry_count,
            error=json_safe(error) if error is not None else None,
            metadata=normalize_metadata(metadata),
        ).to_dict()
        if self.record_transition_history:
            record.transition_history.append(transition)
            if len(record.transition_history) > self.max_transition_history_per_message:
                record.transition_history = record.transition_history[-self.max_transition_history_per_message :]
        self._history.append(transition)
        self.memory.append(
            _DELIVERY_STATE_MACHINE_HISTORY_KEY,
            transition,
            max_items=self.max_history_size,
            ttl_seconds=self.history_ttl_seconds,
            source="delivery_state_machine",
        )

    def _sync_delivery_memory(self, record: DeliveryRecord) -> None:
        if not self.record_memory_snapshots:
            return
        self.memory.record_delivery_state(
            record.state,
            message_id=record.message_id,
            correlation_id=record.correlation_id,
            endpoint=record.endpoint,
            channel=record.channel,
            route=record.route,
            retry_count=record.retry_count,
            error=record.last_error,
            ttl_seconds=self.default_ttl_seconds,
            metadata={
                "delivery_mode": record.delivery_mode,
                "idempotency_key": record.idempotency_key,
                "max_attempts": record.max_attempts,
            },
        )
        self.memory.set(
            _DELIVERY_STATE_MACHINE_LAST_KEY,
            record.to_dict(include_history=False),
            ttl_seconds=self.snapshot_ttl_seconds,
            source="delivery_state_machine",
        )
        self.memory.set(
            self._message_key(record.message_id),
            record.to_dict(include_history=True),
            ttl_seconds=self.default_ttl_seconds,
            source="delivery_state_machine",
        )
        if record.is_terminal(self.terminal_states):
            self.memory.delete(self._active_key(record.message_id))
        else:
            self.memory.set(
                self._active_key(record.message_id),
                record.to_dict(include_history=False),
                ttl_seconds=self.default_ttl_seconds,
                source="delivery_state_machine",
            )

    def _sync_snapshot_memory(self) -> None:
        if not self.record_memory_snapshots:
            return
        active = [record.to_dict(include_history=False) for record in self._deliveries.values() if not record.is_terminal(self.terminal_states)]
        snapshot = {
            "generated_at": utc_timestamp(),
            "started_at": self._started_at,
            "stats": dict(self._stats),
            "delivery_count": len(self._deliveries),
            "active_count": len(active),
            "dead_letter_count": len(self._dead_letter),
            "states": self._state_counts(),
        }
        self.memory.set(
            _DELIVERY_STATE_MACHINE_SNAPSHOT_KEY,
            snapshot,
            ttl_seconds=self.snapshot_ttl_seconds,
            source="delivery_state_machine",
        )
        self.memory.set(
            _DELIVERY_STATE_MACHINE_ACTIVE_KEY,
            {"active": active, "generated_at": utc_timestamp()},
            ttl_seconds=self.snapshot_ttl_seconds,
            source="delivery_state_machine",
        )
        if self._dead_letter:
            self.memory.set(
                _DELIVERY_STATE_MACHINE_DEAD_LETTER_KEY,
                {"messages": json_safe(self._dead_letter), "generated_at": utc_timestamp()},
                ttl_seconds=self.dead_letter_ttl_seconds,
                source="delivery_state_machine",
            )

    def _apply_record_updates_locked(
        self,
        record: DeliveryRecord,
        update_fields: Optional[Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]],
        expires_in_seconds: Optional[int],
    ) -> None:
        updates = ensure_mapping(update_fields, field_name="update_fields", allow_none=True)
        if metadata is not None:
            record.metadata = merge_mappings(record.metadata, normalize_metadata(metadata))
        if expires_in_seconds is not None:
            record.expires_at = (_utcnow() + timedelta(seconds=max(0, int(expires_in_seconds)))).isoformat()
        if not updates:
            return
        for key, value in updates.items():
            if key == "metadata":
                record.metadata = merge_mappings(record.metadata, normalize_metadata(value if isinstance(value, Mapping) else {"value": value}))
            elif key == "envelope" and isinstance(value, Mapping):
                record.envelope = merge_mappings(record.envelope, value)
            elif hasattr(record, key):
                setattr(record, key, json_safe(value) if key in {"last_error", "metadata", "envelope"} else value)
            else:
                record.metadata[str(key)] = json_safe(value)

    def _expire_stale_locked(self) -> None:
        for record in list(self._deliveries.values()):
            self._expire_record_if_needed_locked(record)

    def _expire_record_if_needed_locked(self, record: DeliveryRecord) -> None:
        if record.is_terminal(self.terminal_states):
            return
        if not record.is_expired():
            return
        from_state = record.state
        record.state = "expired"
        record.updated_at = utc_timestamp()
        self._stamp_state_timestamp_locked(record, "expired")
        self._stats["expirations"] += 1
        self._stats["transitions"] += 1
        self._append_transition_locked(record, from_state=from_state, to_state="expired", metadata={"auto_expired": True})
        self._sync_delivery_memory(record)

    def _stamp_state_timestamp_locked(self, record: DeliveryRecord, state: str) -> None:
        now = utc_timestamp()
        if state == "queued":
            record.queued_at = now
        elif state == "selected":
            record.selected_at = now
        elif state == "sent":
            record.sent_at = now
        elif state == "received":
            record.received_at = now
        elif state == "acked":
            record.acked_at = now
        elif state == "nacked":
            record.nacked_at = now
        elif state == "failed":
            record.failed_at = now
        elif state == "expired":
            record.expired_at = now
        elif state == "dead_lettered":
            record.dead_lettered_at = now
        elif state == "cancelled":
            record.cancelled_at = now

    def _record_dead_letter_locked(self, record: DeliveryRecord, *, metadata: Optional[Mapping[str, Any]]) -> None:
        if not self.enable_dead_letter:
            raise DeadLetterQueueError(
                "Dead-letter handling is disabled by configuration.",
                context={
                    "operation": "dead_letter",
                    "channel": record.channel,
                    "protocol": record.protocol,
                    "endpoint": record.endpoint,
                    "correlation_id": record.correlation_id,
                },
                details={"message_id": record.message_id},
            )
        self._dead_letter[record.message_id] = {
            "message_id": record.message_id,
            "correlation_id": record.correlation_id,
            "idempotency_key": record.idempotency_key,
            "dead_letter_queue": record.dead_letter_queue,
            "state": record.state,
            "recorded_at": utc_timestamp(),
            "last_error": record.last_error,
            "metadata": normalize_metadata(metadata),
            "envelope": json_safe(record.envelope),
        }

    def _remove_record_locked(self, message_id: str) -> None:
        record = self._deliveries.pop(message_id, None)
        if record is None:
            return
        self._by_correlation_id.pop(record.correlation_id, None)
        self._by_idempotency_key.pop(record.idempotency_key, None)
        self.memory.delete(self._message_key(message_id))
        self.memory.delete(self._active_key(message_id))

    def _build_allowed_transitions(self) -> Dict[str, Tuple[str, ...]]:
        configured = ensure_mapping(self.state_machine_config.get("allowed_transitions"), field_name="allowed_transitions", allow_none=True)
        transitions: Dict[str, Tuple[str, ...]] = {}
        for state in self.state_order:
            configured_targets = configured.get(state)
            if configured_targets is None:
                transitions[state] = tuple(_DEFAULT_ALLOWED_TRANSITIONS.get(state, ()))
                continue
            targets = tuple(self._get_state_sequence(f"allowed_transitions.{state}", configured_targets))
            transitions[state] = targets
        return transitions

    def _resolve_expiry_timestamp(
        self,
        state: str,
        envelope: Mapping[str, Any],
        expires_in_seconds: Optional[int],
    ) -> Optional[str]:
        if envelope.get("expires_at") is not None:
            return str(envelope.get("expires_at"))
        if expires_in_seconds is not None:
            return (_utcnow() + timedelta(seconds=max(0, int(expires_in_seconds)))).isoformat()
        if state in {"initialized", "queued", "selected", "retrying"} and self.pending_expire_after_seconds > 0:
            return (_utcnow() + timedelta(seconds=self.pending_expire_after_seconds)).isoformat()
        if state in {"sent", "received", "nacked"} and self.sent_expire_after_seconds > 0:
            return (_utcnow() + timedelta(seconds=self.sent_expire_after_seconds)).isoformat()
        if state == "failed" and self.failed_expire_after_seconds > 0:
            return (_utcnow() + timedelta(seconds=self.failed_expire_after_seconds)).isoformat()
        if self.default_ttl_seconds > 0:
            return (_utcnow() + timedelta(seconds=self.default_ttl_seconds)).isoformat()
        return None

    def _normalize_error(self, error: BaseException | Mapping[str, Any], *, record: DeliveryRecord) -> Dict[str, Any]:
        if isinstance(error, Mapping):
            return json_safe(error)
        if isinstance(error, NetworkError):
            return error.to_memory_snapshot()
        return build_error_snapshot(
            error,
            operation="delivery_state_machine",
            endpoint=record.endpoint,
            channel=record.channel,
            route=record.route,
            correlation_id=record.correlation_id,
            max_attempts=record.max_attempts,
            attempt=record.retry_count,
            metadata={"message_id": record.message_id, "idempotency_key": record.idempotency_key},
        )

    def _payload_fingerprint(self, payload: Any) -> str:
        return generate_idempotency_key(json_safe(payload), namespace="delivery_payload")

    def _resolve_lookup_id(self, value: Any) -> str:
        return ensure_non_empty_string(str(value), field_name="lookup")

    def _message_key(self, message_id: str) -> str:
        return f"network.lifecycle.delivery.message.{ensure_non_empty_string(message_id, field_name='message_id')}"

    def _active_key(self, message_id: str) -> str:
        return f"network.lifecycle.delivery.active.{ensure_non_empty_string(message_id, field_name='message_id')}"

    def _state_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self._deliveries.values():
            counts[record.state] = counts.get(record.state, 0) + 1
        return counts

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

    def _sort_delivery_dict(self, item: Mapping[str, Any]) -> Tuple[int, str, str]:
        state = str(item.get("state", "")).lower()
        try:
            state_rank = self.state_order.index(state)
        except ValueError:
            state_rank = len(self.state_order)
        return (state_rank, str(item.get("updated_at", "")), str(item.get("message_id", "")))

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.state_machine_config.get(name, default)
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
            "Invalid boolean value in delivery state machine configuration.",
            context={"operation": "delivery_state_machine_config"},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.state_machine_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in delivery state machine configuration.",
                context={"operation": "delivery_state_machine_config"},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Configuration value must be non-negative.",
                context={"operation": "delivery_state_machine_config"},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.state_machine_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_state_name(self, field_name: str, value: Optional[str]) -> str:
        if value is None:
            raise PayloadValidationError(
                f"{field_name} must not be None.",
                context={"operation": "delivery_state_machine_config"},
                details={"field_name": field_name},
            )
        text = ensure_non_empty_string(str(value), field_name=field_name).strip().lower()
        return text

    def _get_state_sequence(self, name: str, default: Sequence[str] | Any) -> Tuple[str, ...]:
        value = self.state_machine_config.get(name, default)
        values = ensure_sequence(value, field_name=name, allow_none=True, coerce_scalar=True)
        normalized: Dict[str, None] = {}
        for item in values:
            text = ensure_non_empty_string(str(item), field_name=name).strip().lower()
            normalized[text] = None
        return tuple(normalized.keys()) or tuple(str(item).strip().lower() for item in default)


def _parse_timestamp(value: str) -> datetime:
    text = ensure_non_empty_string(str(value), field_name="timestamp")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print("\n=== Running Delivery State Machine ===\n")
    printer.status("TEST", "Delivery State Machine initialized", "info")

    memory = NetworkMemory()
    machine = DeliveryStateMachine(memory=memory)

    registered = machine.register_delivery(
        payload={"task": "relay", "payload": {"hello": "world"}},
        channel="http",
        protocol="http",
        endpoint="https://api.example.com/v1/jobs",
        route="primary",
        delivery_mode="at_least_once",
        metadata={"tenant": "demo", "priority": "normal"},
    )
    printer.status("TEST", "Delivery registered", "info")

    selected = machine.mark_selected(registered["message_id"], metadata={"selector": "channel_selector"})
    printer.status("TEST", "Delivery selected", "info")

    sent = machine.mark_sent(registered["message_id"], metadata={"adapter": "http"})
    printer.status("TEST", "Delivery marked sent", "info")

    received = machine.mark_received(registered["message_id"], metadata={"recv": True})
    printer.status("TEST", "Delivery marked received", "info")

    acked = machine.mark_acked(registered["message_id"], metadata={"ack_source": "transport"})
    printer.status("TEST", "Delivery acknowledged", "info")

    retry_registered = machine.register_delivery(
        payload={"task": "retry", "payload": {"hello": "again"}},
        channel="queue",
        protocol="queue",
        endpoint="amqp://broker.internal:5672/vhost",
        route="secondary",
        metadata={"dead_letter_queue": "jobs.primary.dlq"},
    )
    printer.status("TEST", "Retry-path delivery registered", "info")

    machine.mark_sent(retry_registered["message_id"])
    failed = machine.mark_failed(
        retry_registered["message_id"],
        TimeoutError("upstream timeout"),
        metadata={"phase": "initial_send"},
    )
    printer.status("TEST", "Delivery marked failed", "info")

    retrying = machine.mark_retrying(
        retry_registered["message_id"],
        error=TimeoutError("retry scheduling"),
        metadata={"backoff_ms": 250},
    )
    printer.status("TEST", "Delivery moved to retrying", "info")

    requeued = machine.mark_queued(retry_registered["message_id"], metadata={"requeued": True})
    printer.status("TEST", "Retry delivery re-queued", "info")

    dead_lettered = machine.mark_dead_lettered(
        retry_registered["message_id"],
        metadata={"reason": "exhausted_retries"},
    )
    printer.status("TEST", "Delivery dead-lettered", "info")

    snapshot = machine.get_snapshot()
    deliveries = machine.list_deliveries(include_history=True)
    expired = machine.expire_stale_deliveries()

    print("Registered:", stable_json_dumps(registered))
    print("Selected:", stable_json_dumps(selected))
    print("Sent:", stable_json_dumps(sent))
    print("Received:", stable_json_dumps(received))
    print("Acked:", stable_json_dumps(acked))
    print("Failed:", stable_json_dumps(failed))
    print("Retrying:", stable_json_dumps(retrying))
    print("Requeued:", stable_json_dumps(requeued))
    print("Dead Lettered:", stable_json_dumps(dead_lettered))
    print("Snapshot:", stable_json_dumps(snapshot))
    print("Deliveries:", stable_json_dumps(deliveries))
    print("Expired:", stable_json_dumps(expired))

    assert registered["state"] == machine.default_initial_state
    assert acked["state"] == "acked"
    assert dead_lettered["state"] == "dead_lettered"
    assert memory.get("network.delivery.state") is not None
    assert memory.get("network.lifecycle.delivery_state_machine.snapshot") is not None
    assert memory.get("network.lifecycle.delivery_state_machine.last") is not None
    assert memory.get("network.lifecycle.delivery_state_machine.dead_letter") is not None

    printer.status("TEST", "All Delivery State Machine checks passed", "info")
    print("\n=== Test ran successfully ===\n")
