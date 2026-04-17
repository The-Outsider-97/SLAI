"""
Idempotency coordination for SLAI's Network Agent lifecycle subsystem.

This module provides the production-grade idempotency layer that completes the
lifecycle package alongside EnvelopeManager and DeliveryStateMachine. It owns
idempotency-key reservation, duplicate/replay detection, payload-fingerprint
validation, completion replay snapshots, and key-state synchronization so the
broader network stack can safely reason about at-least-once and replay-aware
operations through one consistent contract.

The idempotency layer is intentionally scoped to idempotency ownership. It is
responsible for:
- canonical idempotency-key reservation and lookup,
- payload-fingerprint validation and mismatch detection,
- duplicate active-request detection,
- completion replay semantics for safely repeatable requests,
- failure/retry re-acquisition semantics for the same idempotency key,
- expiration, release, and terminal-state bookkeeping,
- structured synchronization into NetworkMemory for the wider network stack.

It does not own delivery-state transitions, retry timing, route selection, or
transport execution. Those concerns belong to DeliveryStateMachine,
Reliability, routing, and the specialized adapters. This module owns the
idempotency truth model those layers consult and update.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils import *
from ..network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Idempotency")
printer = PrettyPrinter()


_IDEMPOTENCY_LAST_KEY = "network.lifecycle.idempotency.last"
_IDEMPOTENCY_SNAPSHOT_KEY = "network.lifecycle.idempotency.snapshot"
_IDEMPOTENCY_HISTORY_KEY = "network.lifecycle.idempotency.history"
_IDEMPOTENCY_ACTIVE_KEY = "network.lifecycle.idempotency.active"
_IDEMPOTENCY_COMPLETED_KEY = "network.lifecycle.idempotency.completed"

_DEFAULT_TERMINAL_STATUSES = ("completed", "released", "expired")
_DEFAULT_ACTIVE_STATUSES = ("reserved", "processing", "failed")
_DEFAULT_STATUS_ORDER = ("reserved", "processing", "completed", "failed", "released", "expired")
_DEFAULT_ALLOWED_TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    "reserved": ("processing", "completed", "failed", "released", "expired"),
    "processing": ("completed", "failed", "released", "expired"),
    "completed": ("completed", "released", "expired"),
    "failed": ("reserved", "processing", "released", "expired"),
    "released": (),
    "expired": (),
}


@dataclass(slots=True)
class IdempotencyTransitionRecord:
    """Single transition event emitted by the idempotency manager."""

    idempotency_key: str
    from_status: str
    to_status: str
    occurred_at: str
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    owner_token: Optional[str] = None
    replay_count: int = 0
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "idempotency_key": self.idempotency_key,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "occurred_at": self.occurred_at,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "owner_token": self.owner_token,
            "replay_count": self.replay_count,
            "error": self.error,
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class IdempotencyRecord:
    """Authoritative idempotency record tracked by the lifecycle subsystem."""

    idempotency_key: str
    payload_fingerprint: str
    status: str
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    channel: Optional[str] = None
    protocol: Optional[str] = None
    endpoint: Optional[str] = None
    route: Optional[str] = None
    operation: Optional[str] = None
    owner_token: Optional[str] = None
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    reserved_at: Optional[str] = None
    processing_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    released_at: Optional[str] = None
    expired_at: Optional[str] = None
    expires_at: Optional[str] = None
    replay_count: int = 0
    response_snapshot: Optional[Dict[str, Any]] = None
    error_snapshot: Optional[Dict[str, Any]] = None
    envelope_snapshot: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    transition_history: List[Dict[str, Any]] = field(default_factory=list)

    def is_terminal(self, terminal_statuses: Sequence[str]) -> bool:
        return self.status in set(terminal_statuses)

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            return _parse_timestamp(self.expires_at) <= _utcnow()
        except Exception:
            return False

    def to_dict(self, *, include_history: bool = True) -> Dict[str, Any]:
        payload = {
            "idempotency_key": self.idempotency_key,
            "payload_fingerprint": self.payload_fingerprint,
            "status": self.status,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "channel": self.channel,
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "route": self.route,
            "operation": self.operation,
            "owner_token": self.owner_token,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reserved_at": self.reserved_at,
            "processing_at": self.processing_at,
            "completed_at": self.completed_at,
            "failed_at": self.failed_at,
            "released_at": self.released_at,
            "expired_at": self.expired_at,
            "expires_at": self.expires_at,
            "replay_count": self.replay_count,
            "response_snapshot": self.response_snapshot,
            "error_snapshot": self.error_snapshot,
            "envelope_snapshot": self.envelope_snapshot,
            "metadata": json_safe(self.metadata),
            "active": not self.is_terminal(_DEFAULT_TERMINAL_STATUSES),
        }
        if include_history:
            payload["transition_history"] = json_safe(self.transition_history)
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


class IdempotencyManager:
    """
    Canonical idempotency owner for the network lifecycle domain.

    The manager keeps an in-process authoritative view of idempotency claims and
    synchronizes the important lifecycle moments into NetworkMemory so the rest
    of the network subsystem can observe key-state without directly depending on
    this module's internal storage.
    """

    def __init__(self, memory: Optional[NetworkMemory] = None, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.idempotency_config = merge_mappings(
            get_config_section("network_idempotency") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_memory_snapshots = self._get_bool_config("record_memory_snapshots", True)
        self.record_history = self._get_bool_config("record_history", True)
        self.enforce_fingerprint_match = self._get_bool_config("enforce_fingerprint_match", True)
        self.reject_duplicates_while_active = self._get_bool_config("reject_duplicates_while_active", True)
        self.strict_duplicate_error = self._get_bool_config("strict_duplicate_error", False)
        self.allow_replay_after_completion = self._get_bool_config("allow_replay_after_completion", True)
        self.allow_retry_after_failure = self._get_bool_config("allow_retry_after_failure", True)
        self.auto_generate_key_if_missing = self._get_bool_config("auto_generate_key_if_missing", True)
        self.auto_expire_on_read = self._get_bool_config("auto_expire_on_read", True)
        self.normalize_endpoints = self._get_bool_config("normalize_endpoints", True)
        self.require_channel = self._get_bool_config("require_channel", True)
        self.require_protocol = self._get_bool_config("require_protocol", True)

        self.default_protocol = normalize_protocol_name(self._get_optional_string_config("default_protocol") or "http")
        self.default_channel = normalize_channel_name(self._get_optional_string_config("default_channel") or self.default_protocol)
        self.default_status = self._get_status_name("default_status", "reserved")
        self.default_owner = self._get_optional_string_config("default_owner")

        self.default_timeout_ms = coerce_timeout_ms(
            self.idempotency_config.get("default_timeout_ms"),
            default=5000,
            minimum=1,
            maximum=300000,
        )
        self.default_expires_in_seconds = self._get_non_negative_int_config("default_expires_in_seconds", 0)
        self.reservation_ttl_seconds = self._get_non_negative_int_config("reservation_ttl_seconds", 1800)
        self.processing_ttl_seconds = self._get_non_negative_int_config("processing_ttl_seconds", 7200)
        self.completed_ttl_seconds = self._get_non_negative_int_config("completed_ttl_seconds", 86400)
        self.failed_ttl_seconds = self._get_non_negative_int_config("failed_ttl_seconds", 7200)
        self.released_ttl_seconds = self._get_non_negative_int_config("released_ttl_seconds", 3600)
        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 7200)
        self.max_records = max(1, self._get_non_negative_int_config("max_records", 5000))
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))
        self.max_transitions_per_key = max(1, self._get_non_negative_int_config("max_transitions_per_key", 50))

        self.terminal_statuses = self._get_status_sequence("terminal_statuses", _DEFAULT_TERMINAL_STATUSES)
        self.active_statuses = self._get_status_sequence("active_statuses", _DEFAULT_ACTIVE_STATUSES)
        self.status_order = self._get_status_sequence("status_order", _DEFAULT_STATUS_ORDER)
        self.allowed_transitions = self._build_allowed_transitions()

        self._records: Dict[str, IdempotencyRecord] = {}
        self._by_message_id: Dict[str, str] = {}
        self._by_correlation_id: Dict[str, str] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history_size)
        self._stats: Dict[str, int] = {
            "reservations": 0,
            "duplicates": 0,
            "replays": 0,
            "reacquisitions": 0,
            "transitions": 0,
            "completions": 0,
            "failures": 0,
            "releases": 0,
            "expirations": 0,
            "purges": 0,
        }
        self._started_at = utc_timestamp()

        self._sync_snapshot_memory()

    # ------------------------------------------------------------------
    # Reservation and lookup
    # ------------------------------------------------------------------
    def reserve(
        self,
        envelope: Optional[Mapping[str, Any]] = None,
        *,
        payload: Any = None,
        idempotency_key: Optional[str] = None,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        endpoint: Optional[str] = None,
        route: Optional[str] = None,
        operation: Optional[str] = None,
        owner_token: Optional[str] = None,
        timeout_ms: Optional[Any] = None,
        expires_in_seconds: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError("IdempotencyManager is disabled by configuration.", context={"operation": "reserve_idempotency"})

        envelope_map = self._build_envelope_snapshot(
            envelope=envelope,
            payload=payload,
            idempotency_key=idempotency_key,
            message_id=message_id,
            correlation_id=correlation_id,
            channel=channel,
            protocol=protocol,
            endpoint=endpoint,
            route=route,
            operation=operation,
            timeout_ms=timeout_ms,
            metadata=metadata,
        )
        record = self._record_from_envelope(
            envelope_map,
            owner_token=owner_token,
            expires_in_seconds=expires_in_seconds,
        )

        with self._lock:
            existing = self._records.get(record.idempotency_key)
            if existing is None:
                self._ensure_capacity_locked(incoming_key=record.idempotency_key)
                self._records[record.idempotency_key] = record
                self._index_record_locked(record)
                self._stats["reservations"] += 1
                self._append_transition_locked(record, from_status="untracked", to_status=record.status, metadata={"reservation": True})
                self._sync_record_memory(record)
                self._sync_snapshot_memory()
                return {
                    "outcome": "reserved",
                    "duplicate": False,
                    "replay": False,
                    "record": record.to_dict(),
                }

            self._expire_record_if_needed_locked(existing)
            self._validate_fingerprint_reuse(existing, record.payload_fingerprint)
            # existing.last_seen_at = utc_timestamp() if hasattr(existing, "last_seen_at") else None  # compatibility no-op
            self._stats["duplicates"] += 1

            if existing.status == "completed" and self.allow_replay_after_completion:
                existing.replay_count += 1
                existing.updated_at = utc_timestamp()
                self._stats["replays"] += 1
                self._append_transition_locked(existing, from_status="completed", to_status="completed", metadata={"replay": True})
                self._sync_record_memory(existing)
                self._sync_snapshot_memory()
                return {
                    "outcome": "replay",
                    "duplicate": True,
                    "replay": True,
                    "record": existing.to_dict(),
                    "response_snapshot": json_safe(existing.response_snapshot),
                }

            if existing.status == "failed" and self.allow_retry_after_failure:
                previous_status = existing.status
                existing.status = "reserved"
                existing.updated_at = utc_timestamp()
                existing.reserved_at = existing.updated_at
                existing.error_snapshot = None
                existing.message_id = record.message_id or existing.message_id
                existing.correlation_id = record.correlation_id or existing.correlation_id
                existing.endpoint = record.endpoint or existing.endpoint
                existing.route = record.route or existing.route
                existing.channel = record.channel or existing.channel
                existing.protocol = record.protocol or existing.protocol
                existing.owner_token = record.owner_token or existing.owner_token
                existing.envelope_snapshot = record.envelope_snapshot or existing.envelope_snapshot
                existing.metadata = merge_mappings(existing.metadata, record.metadata)
                self._index_record_locked(existing)
                self._stats["reacquisitions"] += 1
                self._stats["transitions"] += 1
                self._append_transition_locked(existing, from_status=previous_status, to_status="reserved", metadata={"reacquired": True})
                self._sync_record_memory(existing)
                self._sync_snapshot_memory()
                return {
                    "outcome": "reacquired",
                    "duplicate": True,
                    "replay": False,
                    "record": existing.to_dict(),
                }

            if self.reject_duplicates_while_active and existing.status in set(self.active_statuses):
                if self.strict_duplicate_error:
                    raise DuplicateMessageError(
                        "An active idempotency claim already exists for this key.",
                        context={
                            "operation": "reserve_idempotency",
                            "channel": existing.channel,
                            "protocol": existing.protocol,
                            "endpoint": existing.endpoint,
                            "correlation_id": existing.correlation_id,
                            "idempotency_key": existing.idempotency_key,
                        },
                        details={"existing_status": existing.status, "message_id": existing.message_id},
                    )
                self._append_transition_locked(existing, from_status=existing.status, to_status=existing.status, metadata={"duplicate_active": True})
                self._sync_record_memory(existing)
                self._sync_snapshot_memory()
                return {
                    "outcome": "duplicate_active",
                    "duplicate": True,
                    "replay": False,
                    "record": existing.to_dict(),
                }

            return {
                "outcome": "existing",
                "duplicate": True,
                "replay": False,
                "record": existing.to_dict(),
            }

    def get_record(self, idempotency_key_or_lookup: str, *, include_history: bool = True) -> Optional[Dict[str, Any]]:
        record = self._find_record(idempotency_key_or_lookup, auto_expire=self.auto_expire_on_read)
        if record is None:
            return None
        return record.to_dict(include_history=include_history)

    def require_record(self, idempotency_key_or_lookup: str, *, include_history: bool = True) -> Dict[str, Any]:
        record = self._find_record(idempotency_key_or_lookup, auto_expire=self.auto_expire_on_read)
        if record is None:
            raise DeliveryStateError(
                "Requested idempotency record is not registered.",
                context={"operation": "require_idempotency_record"},
                details={"lookup": str(idempotency_key_or_lookup)},
            )
        return record.to_dict(include_history=include_history)

    def list_records(self, *, status: Optional[str] = None, channel: Optional[str] = None,
                     protocol: Optional[str] = None, include_terminal: bool = True,
                     include_history: bool = False) -> List[Dict[str, Any]]:
        normalized_status = self._get_status_name("status", status) if status is not None else None
        normalized_channel = normalize_channel_name(channel) if channel is not None else None
        normalized_protocol = normalize_protocol_name(protocol) if protocol is not None else None
        with self._lock:
            self._expire_stale_locked()
            payload: List[Dict[str, Any]] = []
            for record in self._records.values():
                if normalized_status is not None and record.status != normalized_status:
                    continue
                if normalized_channel is not None and record.channel != normalized_channel:
                    continue
                if normalized_protocol is not None and record.protocol != normalized_protocol:
                    continue
                if not include_terminal and record.is_terminal(self.terminal_statuses):
                    continue
                payload.append(record.to_dict(include_history=include_history))
            payload.sort(key=self._sort_record_dict)
            return payload

    # ------------------------------------------------------------------
    # Transition operations
    # ------------------------------------------------------------------
    def transition(self, idempotency_key_or_lookup: str, to_status: str,  *,
        response_snapshot: Optional[Mapping[str, Any]] = None,
        error: Optional[BaseException | Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        update_fields: Optional[Mapping[str, Any]] = None,
        expires_in_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized_to_status = self._get_status_name("to_status", to_status)
        with self._lock:
            record = self._require_record_locked(idempotency_key_or_lookup)
            self._expire_record_if_needed_locked(record)

            from_status = record.status
            if from_status == normalized_to_status:
                self._apply_record_updates_locked(record, update_fields, metadata, expires_in_seconds)
                if response_snapshot is not None:
                    record.response_snapshot = json_safe(response_snapshot)
                if error is not None:
                    record.error_snapshot = self._normalize_error(error, record=record)
                self._sync_record_memory(record)
                return record.to_dict()

            self._validate_transition_locked(record, normalized_to_status)
            normalized_error = self._normalize_error(error, record=record) if error is not None else None

            record.status = normalized_to_status
            record.updated_at = utc_timestamp()
            if normalized_to_status == "completed":
                record.response_snapshot = json_safe(response_snapshot) if response_snapshot is not None else record.response_snapshot
                record.error_snapshot = None
                self._stats["completions"] += 1
            elif normalized_to_status == "failed":
                record.error_snapshot = normalized_error
                self._stats["failures"] += 1
            elif normalized_to_status == "released":
                self._stats["releases"] += 1
            elif normalized_to_status == "expired":
                self._stats["expirations"] += 1

            self._stamp_status_timestamp_locked(record, normalized_to_status)
            self._apply_record_updates_locked(record, update_fields, metadata, expires_in_seconds)
            self._stats["transitions"] += 1
            self._append_transition_locked(record, from_status=from_status, to_status=normalized_to_status, error=normalized_error, metadata=metadata)
            self._sync_record_memory(record)
            self._sync_snapshot_memory()
            return record.to_dict()

    def mark_processing(self, idempotency_key_or_lookup: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(idempotency_key_or_lookup, "processing", **kwargs)

    def mark_completed(
        self,
        idempotency_key_or_lookup: str,
        response_snapshot: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.transition(idempotency_key_or_lookup, "completed", response_snapshot=response_snapshot, **kwargs)

    def mark_failed(
        self,
        idempotency_key_or_lookup: str,
        error: Optional[BaseException | Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.transition(idempotency_key_or_lookup, "failed", error=error, **kwargs)

    def release(self, idempotency_key_or_lookup: str, **kwargs: Any) -> Dict[str, Any]:
        return self.transition(idempotency_key_or_lookup, "released", **kwargs)

    def expire_stale_records(self) -> List[Dict[str, Any]]:
        with self._lock:
            expired: List[Dict[str, Any]] = []
            for record in list(self._records.values()):
                if record.is_terminal(self.terminal_statuses):
                    continue
                if record.is_expired():
                    from_status = record.status
                    record.status = "expired"
                    record.updated_at = utc_timestamp()
                    self._stamp_status_timestamp_locked(record, "expired")
                    self._stats["expirations"] += 1
                    self._stats["transitions"] += 1
                    self._append_transition_locked(record, from_status=from_status, to_status="expired", metadata={"auto_expired": True})
                    self._sync_record_memory(record)
                    expired.append(record.to_dict())
            if expired:
                self._sync_snapshot_memory()
            return expired

    def purge_terminal_records(self, *, older_than_seconds: Optional[int] = None) -> int:
        threshold_seconds = self.completed_ttl_seconds if older_than_seconds is None else max(0, int(older_than_seconds))
        cutoff = _utcnow() - timedelta(seconds=threshold_seconds)
        purged = 0
        with self._lock:
            for key, record in list(self._records.items()):
                if not record.is_terminal(self.terminal_statuses):
                    continue
                updated_at = _parse_timestamp(record.updated_at)
                if updated_at > cutoff:
                    continue
                self._remove_record_locked(key)
                purged += 1
            if purged:
                self._stats["purges"] += purged
                self._sync_snapshot_memory()
            return purged

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._expire_stale_locked()
            active = [record.to_dict(include_history=False) for record in self._records.values() if not record.is_terminal(self.terminal_statuses)]
            terminal = [record.to_dict(include_history=False) for record in self._records.values() if record.is_terminal(self.terminal_statuses)]
            return {
                "generated_at": utc_timestamp(),
                "started_at": self._started_at,
                "stats": dict(self._stats),
                "record_count": len(self._records),
                "active_count": len(active),
                "terminal_count": len(terminal),
                "history_size": len(self._history),
                "active": sorted(active, key=self._sort_record_dict),
                "terminal": sorted(terminal, key=self._sort_record_dict),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_envelope_snapshot(
        self,
        *,
        envelope: Optional[Mapping[str, Any]],
        payload: Any,
        idempotency_key: Optional[str],
        message_id: Optional[str],
        correlation_id: Optional[str],
        channel: Optional[str],
        protocol: Optional[str],
        endpoint: Optional[str],
        route: Optional[str],
        operation: Optional[str],
        timeout_ms: Optional[Any],
        metadata: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        base = ensure_mapping(envelope, field_name="envelope", allow_none=True)
        payload_value = payload if payload is not None else base.get("payload")
        channel_value = channel or base.get("channel") or self.default_channel
        protocol_value = protocol or base.get("protocol") or self.default_protocol
        metadata_map = normalize_metadata(merge_mappings(base.get("metadata"), metadata))
        timeout_value = timeout_ms if timeout_ms is not None else base.get("timeout_ms") or self.default_timeout_ms

        effective_idempotency_key = idempotency_key or base.get("idempotency_key")
        if effective_idempotency_key is None and self.auto_generate_key_if_missing:
            effective_idempotency_key = generate_idempotency_key(
                {
                    "payload": json_safe(payload_value),
                    "route": route or base.get("route"),
                    "operation": operation or base.get("operation"),
                    "channel": channel_value,
                    "protocol": protocol_value,
                },
                namespace="network_idempotency",
                route=route or base.get("route"),
                operation=operation or base.get("operation"),
            )

        built = build_message_envelope(
            base,
            payload=payload_value,
            channel=channel_value,
            protocol=protocol_value,
            endpoint=self._safe_endpoint(endpoint or base.get("endpoint")),
            route=route or base.get("route"),
            message_id=message_id or base.get("message_id"),
            correlation_id=correlation_id or base.get("correlation_id"),
            idempotency_key=effective_idempotency_key,
            timeout_ms=timeout_value,
            metadata=metadata_map,
            content_type=base.get("content_type"),
        )
        built["operation"] = operation or base.get("operation")
        return built

    def _record_from_envelope(
        self,
        envelope: Mapping[str, Any],
        *,
        owner_token: Optional[str],
        expires_in_seconds: Optional[int],
    ) -> IdempotencyRecord:
        envelope_map = ensure_mapping(envelope, field_name="envelope")
        idempotency_key = envelope_map.get("idempotency_key")
        if idempotency_key is None:
            raise PayloadValidationError(
                "idempotency_key is required to reserve an idempotency record.",
                context={"operation": "reserve_idempotency"},
            )
        message_id = str(envelope_map.get("message_id")) if envelope_map.get("message_id") is not None else None
        correlation_id = str(envelope_map.get("correlation_id")) if envelope_map.get("correlation_id") is not None else None
        channel = normalize_channel_name(envelope_map.get("channel") or self.default_channel)
        protocol = normalize_protocol_name(envelope_map.get("protocol") or channel or self.default_protocol)
        if self.require_channel and not channel:
            raise PayloadValidationError("channel is required for idempotency reservation.", context={"operation": "reserve_idempotency"})
        if self.require_protocol and not protocol:
            raise PayloadValidationError("protocol is required for idempotency reservation.", context={"operation": "reserve_idempotency"})

        payload_fingerprint = self._payload_fingerprint(envelope_map.get("payload"))
        expires_at = self._resolve_expiry_timestamp(self.default_status, envelope_map, expires_in_seconds)
        owner_value = owner_token or self.default_owner or str(envelope_map.get("session_id") or envelope_map.get("correlation_id") or generate_session_id(prefix="idem"))
        now = utc_timestamp()

        record = IdempotencyRecord(
            idempotency_key=ensure_non_empty_string(str(idempotency_key), field_name="idempotency_key"),
            payload_fingerprint=payload_fingerprint,
            status=self.default_status,
            message_id=message_id,
            correlation_id=correlation_id,
            channel=channel,
            protocol=protocol,
            endpoint=self._safe_endpoint(envelope_map.get("endpoint")),
            route=str(envelope_map.get("route")) if envelope_map.get("route") is not None else None,
            operation=str(envelope_map.get("operation")) if envelope_map.get("operation") is not None else None,
            owner_token=owner_value,
            created_at=now,
            updated_at=now,
            reserved_at=now,
            expires_at=expires_at,
            envelope_snapshot=json_safe(envelope_map),
            metadata=normalize_metadata(envelope_map.get("metadata")),
        )
        return record

    def _find_record(self, lookup: str, *, auto_expire: bool) -> Optional[IdempotencyRecord]:
        token = ensure_non_empty_string(str(lookup), field_name="lookup")
        with self._lock:
            record = self._resolve_record_locked(token)
            if record is None:
                return None
            if auto_expire:
                self._expire_record_if_needed_locked(record)
            return record

    def _require_record_locked(self, lookup: str) -> IdempotencyRecord:
        record = self._resolve_record_locked(str(lookup))
        if record is None:
            raise DeliveryStateError(
                "Requested idempotency record is not registered.",
                context={"operation": "require_idempotency_record"},
                details={"lookup": str(lookup)},
            )
        return record

    def _resolve_record_locked(self, lookup: str) -> Optional[IdempotencyRecord]:
        if lookup in self._records:
            return self._records[lookup]
        key = self._by_message_id.get(lookup) or self._by_correlation_id.get(lookup)
        if key is None:
            return None
        return self._records.get(key)

    def _validate_fingerprint_reuse(self, existing: IdempotencyRecord, new_fingerprint: str) -> None:
        if not self.enforce_fingerprint_match:
            return
        if existing.payload_fingerprint == new_fingerprint:
            return
        raise IdempotencyViolationError(
            "An existing idempotency key was reused with a different payload fingerprint.",
            context={
                "operation": "reserve_idempotency",
                "channel": existing.channel,
                "protocol": existing.protocol,
                "endpoint": existing.endpoint,
                "correlation_id": existing.correlation_id,
                "idempotency_key": existing.idempotency_key,
            },
            details={"message_id": existing.message_id, "status": existing.status},
        )

    def _validate_transition_locked(self, record: IdempotencyRecord, to_status: str) -> None:
        if record.is_terminal(self.terminal_statuses) and to_status != record.status:
            raise DeliveryStateError(
                "Cannot transition a terminal idempotency record.",
                context={
                    "operation": "idempotency_transition",
                    "channel": record.channel,
                    "protocol": record.protocol,
                    "endpoint": record.endpoint,
                    "correlation_id": record.correlation_id,
                    "idempotency_key": record.idempotency_key,
                },
                details={"from_status": record.status, "to_status": to_status},
            )
        allowed = set(self.allowed_transitions.get(record.status, ()))
        if to_status not in allowed:
            raise DeliveryStateError(
                "Requested idempotency transition is not allowed.",
                context={
                    "operation": "idempotency_transition",
                    "channel": record.channel,
                    "protocol": record.protocol,
                    "endpoint": record.endpoint,
                    "correlation_id": record.correlation_id,
                    "idempotency_key": record.idempotency_key,
                },
                details={"from_status": record.status, "to_status": to_status, "allowed": sorted(allowed)},
            )

    def _append_transition_locked(
        self,
        record: IdempotencyRecord,
        *,
        from_status: str,
        to_status: str,
        error: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        transition = IdempotencyTransitionRecord(
            idempotency_key=record.idempotency_key,
            from_status=from_status,
            to_status=to_status,
            occurred_at=utc_timestamp(),
            message_id=record.message_id,
            correlation_id=record.correlation_id,
            owner_token=record.owner_token,
            replay_count=record.replay_count,
            error=json_safe(error) if error is not None else None,
            metadata=normalize_metadata(metadata),
        ).to_dict()
        if self.record_history:
            record.transition_history.append(transition)
            if len(record.transition_history) > self.max_transitions_per_key:
                record.transition_history = record.transition_history[-self.max_transitions_per_key :]
        self._history.append(transition)
        self.memory.append(
            _IDEMPOTENCY_HISTORY_KEY,
            transition,
            max_items=self.max_history_size,
            ttl_seconds=self.history_ttl_seconds,
            source="idempotency",
        )

    def _sync_record_memory(self, record: IdempotencyRecord) -> None:
        if not self.record_memory_snapshots:
            return
        self.memory.set(
            _IDEMPOTENCY_LAST_KEY,
            record.to_dict(include_history=False),
            ttl_seconds=self.snapshot_ttl_seconds,
            source="idempotency",
        )
        self.memory.set(
            self._record_key(record.idempotency_key),
            record.to_dict(include_history=True),
            ttl_seconds=self._ttl_for_status(record.status),
            source="idempotency",
        )
        if record.status == "completed":
            self.memory.set(
                self._completed_key(record.idempotency_key),
                record.to_dict(include_history=False),
                ttl_seconds=self.completed_ttl_seconds,
                source="idempotency",
            )
        if record.is_terminal(self.terminal_statuses):
            self.memory.delete(self._active_key(record.idempotency_key))
        else:
            self.memory.set(
                self._active_key(record.idempotency_key),
                record.to_dict(include_history=False),
                ttl_seconds=self._ttl_for_status(record.status),
                source="idempotency",
            )

    def _sync_snapshot_memory(self) -> None:
        if not self.record_memory_snapshots:
            return
        active = [record.to_dict(include_history=False) for record in self._records.values() if not record.is_terminal(self.terminal_statuses)]
        completed = [record.to_dict(include_history=False) for record in self._records.values() if record.status == "completed"]
        snapshot = {
            "generated_at": utc_timestamp(),
            "started_at": self._started_at,
            "stats": dict(self._stats),
            "record_count": len(self._records),
            "active_count": len(active),
            "completed_count": len(completed),
            "states": self._status_counts(),
            "history_size": len(self._history),
        }
        self.memory.set(
            _IDEMPOTENCY_SNAPSHOT_KEY,
            snapshot,
            ttl_seconds=self.snapshot_ttl_seconds,
            source="idempotency",
        )
        self.memory.set(
            _IDEMPOTENCY_ACTIVE_KEY,
            {"active": active, "generated_at": utc_timestamp()},
            ttl_seconds=self.snapshot_ttl_seconds,
            source="idempotency",
        )
        self.memory.set(
            _IDEMPOTENCY_COMPLETED_KEY,
            {"completed": completed, "generated_at": utc_timestamp()},
            ttl_seconds=self.completed_ttl_seconds,
            source="idempotency",
        )

    def _apply_record_updates_locked(
        self,
        record: IdempotencyRecord,
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
            elif key == "response_snapshot":
                record.response_snapshot = json_safe(value)
            elif key == "error_snapshot":
                record.error_snapshot = json_safe(value)
            elif key == "envelope_snapshot":
                record.envelope_snapshot = json_safe(value)
            elif hasattr(record, key):
                setattr(record, key, json_safe(value) if key in {"metadata", "response_snapshot", "error_snapshot", "envelope_snapshot"} else value)
            else:
                record.metadata[str(key)] = json_safe(value)

    def _expire_stale_locked(self) -> None:
        for record in list(self._records.values()):
            self._expire_record_if_needed_locked(record)

    def _expire_record_if_needed_locked(self, record: IdempotencyRecord) -> None:
        if record.is_terminal(self.terminal_statuses):
            return
        if not record.is_expired():
            return
        from_status = record.status
        record.status = "expired"
        record.updated_at = utc_timestamp()
        self._stamp_status_timestamp_locked(record, "expired")
        self._stats["expirations"] += 1
        self._stats["transitions"] += 1
        self._append_transition_locked(record, from_status=from_status, to_status="expired", metadata={"auto_expired": True})
        self._sync_record_memory(record)

    def _stamp_status_timestamp_locked(self, record: IdempotencyRecord, status: str) -> None:
        now = utc_timestamp()
        if status == "reserved":
            record.reserved_at = now
        elif status == "processing":
            record.processing_at = now
        elif status == "completed":
            record.completed_at = now
        elif status == "failed":
            record.failed_at = now
        elif status == "released":
            record.released_at = now
        elif status == "expired":
            record.expired_at = now

    def _index_record_locked(self, record: IdempotencyRecord) -> None:
        if record.message_id:
            self._by_message_id[record.message_id] = record.idempotency_key
        if record.correlation_id:
            self._by_correlation_id[record.correlation_id] = record.idempotency_key

    def _remove_record_locked(self, idempotency_key: str) -> None:
        record = self._records.pop(idempotency_key, None)
        if record is None:
            return
        if record.message_id:
            self._by_message_id.pop(record.message_id, None)
        if record.correlation_id:
            self._by_correlation_id.pop(record.correlation_id, None)
        self.memory.delete(self._record_key(idempotency_key))
        self.memory.delete(self._active_key(idempotency_key))
        self.memory.delete(self._completed_key(idempotency_key))

    def _ensure_capacity_locked(self, *, incoming_key: str) -> None:
        if incoming_key in self._records:
            return
        if len(self._records) < self.max_records:
            return
        oldest_key = min(self._records.items(), key=lambda item: _parse_timestamp(item[1].updated_at))[0]
        self._remove_record_locked(oldest_key)
        self._stats["purges"] += 1

    def _build_allowed_transitions(self) -> Dict[str, Tuple[str, ...]]:
        configured = ensure_mapping(self.idempotency_config.get("allowed_transitions"), field_name="allowed_transitions", allow_none=True)
        transitions: Dict[str, Tuple[str, ...]] = {}
        for status in self.status_order:
            configured_targets = configured.get(status)
            if configured_targets is None:
                transitions[status] = tuple(_DEFAULT_ALLOWED_TRANSITIONS.get(status, ()))
                continue
            targets = tuple(self._get_status_sequence(f"allowed_transitions.{status}", configured_targets))
            transitions[status] = targets
        return transitions

    def _resolve_expiry_timestamp(
        self,
        status: str,
        envelope: Mapping[str, Any],
        expires_in_seconds: Optional[int],
    ) -> Optional[str]:
        if envelope.get("expires_at") is not None:
            return str(envelope.get("expires_at"))
        if expires_in_seconds is not None:
            return (_utcnow() + timedelta(seconds=max(0, int(expires_in_seconds)))).isoformat()
        if self.default_expires_in_seconds > 0:
            return (_utcnow() + timedelta(seconds=self.default_expires_in_seconds)).isoformat()
        ttl = self._ttl_for_status(status)
        if ttl > 0:
            return (_utcnow() + timedelta(seconds=ttl)).isoformat()
        return None

    def _ttl_for_status(self, status: str) -> int:
        if status == "reserved":
            return self.reservation_ttl_seconds
        if status == "processing":
            return self.processing_ttl_seconds
        if status == "completed":
            return self.completed_ttl_seconds
        if status == "failed":
            return self.failed_ttl_seconds
        if status == "released":
            return self.released_ttl_seconds
        if status == "expired":
            return self.released_ttl_seconds
        return self.snapshot_ttl_seconds

    def _normalize_error(self, error: BaseException | Mapping[str, Any], *, record: IdempotencyRecord) -> Dict[str, Any]:
        if isinstance(error, Mapping):
            return json_safe(error)
        if isinstance(error, NetworkError):
            return error.to_memory_snapshot()
        return build_error_snapshot(
            error,
            operation="idempotency",
            endpoint=record.endpoint,
            channel=record.channel,
            route=record.route,
            correlation_id=record.correlation_id,
            metadata={"message_id": record.message_id, "idempotency_key": record.idempotency_key},
        )

    def _payload_fingerprint(self, payload: Any) -> str:
        return generate_idempotency_key(json_safe(payload), namespace="idempotency_payload")

    def _record_key(self, idempotency_key: str) -> str:
        digest = generate_idempotency_key(idempotency_key, namespace="idempotency_record")[:24]
        return f"network.lifecycle.idempotency.key.{digest}"

    def _active_key(self, idempotency_key: str) -> str:
        digest = generate_idempotency_key(idempotency_key, namespace="idempotency_active")[:24]
        return f"network.lifecycle.idempotency.active.{digest}"

    def _completed_key(self, idempotency_key: str) -> str:
        digest = generate_idempotency_key(idempotency_key, namespace="idempotency_completed")[:24]
        return f"network.lifecycle.idempotency.completed.{digest}"

    def _status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self._records.values():
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def _safe_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        if endpoint is None:
            return None
        text = str(endpoint).strip()
        if not text:
            return None
        if not self.normalize_endpoints:
            return text
        try:
            if "://" in text:
                return normalize_endpoint(text)
        except Exception:
            return text
        return text

    def _sort_record_dict(self, item: Mapping[str, Any]) -> Tuple[int, str, str]:
        status = str(item.get("status", "")).lower()
        try:
            rank = self.status_order.index(status)
        except ValueError:
            rank = len(self.status_order)
        return (rank, str(item.get("updated_at", "")), str(item.get("idempotency_key", "")))

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.idempotency_config.get(name, default)
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
            "Invalid boolean value in idempotency configuration.",
            context={"operation": "idempotency_config"},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.idempotency_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in idempotency configuration.",
                context={"operation": "idempotency_config"},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Configuration value must be non-negative.",
                context={"operation": "idempotency_config"},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.idempotency_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_status_name(self, field_name: str, value: Optional[str]) -> str:
        if value is None:
            raise PayloadValidationError(
                f"{field_name} must not be None.",
                context={"operation": "idempotency_config"},
                details={"field_name": field_name},
            )
        return ensure_non_empty_string(str(value), field_name=field_name).strip().lower()

    def _get_status_sequence(self, name: str, default: Sequence[str] | Any) -> Tuple[str, ...]:
        value = self.idempotency_config.get(name, default)
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
    print("\n=== Running Idempotency ===\n")
    printer.status("TEST", "Idempotency initialized", "info")

    memory = NetworkMemory()
    manager = IdempotencyManager(memory=memory)

    reserved = manager.reserve(
        payload={"task": "relay", "payload": {"hello": "world"}},
        channel="http",
        protocol="http",
        endpoint="https://api.example.com/v1/relay",
        route="primary",
        operation="send",
        metadata={"tenant": "demo"},
    )
    printer.status("TEST", "Idempotency reserved", "info")

    processing = manager.mark_processing(reserved["record"]["idempotency_key"], metadata={"stage": "dispatch"})
    printer.status("TEST", "Idempotency moved to processing", "info")

    completed = manager.mark_completed(
        reserved["record"]["idempotency_key"],
        response_snapshot={"status": "ok", "delivery_id": "dlv_001"},
        metadata={"adapter": "http"},
    )
    printer.status("TEST", "Idempotency completed", "info")

    replay = manager.reserve(
        payload={"task": "relay", "payload": {"hello": "world"}},
        idempotency_key=reserved["record"]["idempotency_key"],
        channel="http",
        protocol="http",
        endpoint="https://api.example.com/v1/relay",
        operation="send",
    )
    printer.status("TEST", "Replay handled", "info")

    failed_reserved = manager.reserve(
        payload={"task": "retry-me", "payload": {"n": 1}},
        idempotency_key="idem_retry_demo",
        channel="queue",
        protocol="queue",
        endpoint="amqp://broker.internal:5672/vhost",
        route="secondary",
        operation="publish",
    )
    printer.status("TEST", "Retry idempotency reserved", "info")

    failed = manager.mark_failed(
        "idem_retry_demo",
        TimeoutError("transient transport timeout"),
        metadata={"attempt": 1},
    )
    printer.status("TEST", "Idempotency marked failed", "info")

    reacquired = manager.reserve(
        payload={"task": "retry-me", "payload": {"n": 1}},
        idempotency_key="idem_retry_demo",
        channel="queue",
        protocol="queue",
        endpoint="amqp://broker.internal:5672/vhost",
        operation="publish",
    )
    printer.status("TEST", "Failed idempotency reacquired", "info")

    released = manager.release("idem_retry_demo", metadata={"cleanup": True})
    printer.status("TEST", "Idempotency released", "info")

    snapshot = manager.get_snapshot()
    records = manager.list_records(include_history=True)
    expired = manager.expire_stale_records()

    print("Reserved:", stable_json_dumps(reserved))
    print("Processing:", stable_json_dumps(processing))
    print("Completed:", stable_json_dumps(completed))
    print("Replay:", stable_json_dumps(replay))
    print("Failed:", stable_json_dumps(failed))
    print("Reacquired:", stable_json_dumps(reacquired))
    print("Released:", stable_json_dumps(released))
    print("Snapshot:", stable_json_dumps(snapshot))
    print("Records:", stable_json_dumps(records))
    print("Expired:", stable_json_dumps(expired))

    assert reserved["outcome"] == "reserved"
    assert processing["status"] == "processing"
    assert completed["status"] == "completed"
    assert replay["outcome"] == "replay"
    assert failed["status"] == "failed"
    assert reacquired["outcome"] == "reacquired"
    assert released["status"] == "released"
    assert memory.get("network.lifecycle.idempotency.snapshot") is not None
    assert memory.get("network.lifecycle.idempotency.last") is not None
    assert memory.get("network.lifecycle.idempotency.history") is not None

    printer.status("TEST", "All Idempotency checks passed", "info")
    print("\n=== Test ran successfully ===\n")
