"""
State snapshots for route decisions, endpoint health, retry history, and session context.

This module owns the shared in-process memory model for the network subsystem.
It centralizes structured state capture for transport orchestration without
stepping into routing policy, adapter execution, lifecycle state machines, or
business logic. The goal is to provide a reliable, thread-safe memory surface
that other network modules can depend on for:

- route decisions and route candidate snapshots,
- delivery state progression and retry/error context,
- endpoint health and circuit posture,
- session lifecycle snapshots,
- policy decision records,
- telemetry and per-channel metric snapshots.

The memory layer is intentionally JSON-safe and observability-friendly so the
same structures can be reused for agent coordination, structured logging,
telemetry emission, and debugging.
"""

from __future__ import annotations

import re

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils import *
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Network Memory")
printer = PrettyPrinter()

_ROUTE_SELECTED_KEY = "network.route.selected"
_ROUTE_CANDIDATES_KEY = "network.route.candidates"
_DELIVERY_STATE_KEY = "network.delivery.state"
_DELIVERY_RETRY_COUNT_KEY = "network.delivery.retry_count"
_DELIVERY_LAST_ERROR_KEY = "network.delivery.last_error"
_ENDPOINT_HEALTH_KEY = "network.endpoint.health"
_ENDPOINT_CIRCUIT_STATE_KEY = "network.endpoint.circuit_state"
_SESSION_SNAPSHOT_KEY = "network.session.snapshot"
_POLICY_DECISION_KEY = "network.policy.decision"
_TELEMETRY_CHANNEL_METRICS_KEY = "network.telemetry.channel_metrics"

_KEY_TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")
_MAX_KEY_TOKEN_LENGTH = 96


@dataclass(slots=True)
class MemoryEntry:
    """
    Single memory entry with metadata.

    Values are stored in a JSON-safe form so they can be exported, persisted,
    or passed across agent boundaries without additional coercion.
    """

    key: str
    value: Any
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    version: int = 1
    source: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= _utcnow()

    @property
    def ttl_seconds(self) -> Optional[int]:
        if self.expires_at is None:
            return None
        remaining = int((self.expires_at - _utcnow()).total_seconds())
        return max(0, remaining)

    def to_dict(self, *, include_value: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at is not None else None,
            "expired": self.expired,
            "ttl_seconds": self.ttl_seconds,
            "version": self.version,
            "source": self.source,
            "tags": list(self.tags),
            "metadata": json_safe(self.metadata),
        }
        if include_value:
            payload["value"] = json_safe(self.value)
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


class NetworkMemory:
    """
    Thread-safe network-domain memory for shared snapshots and histories.

    The class offers both a generic key/value API and a set of domain-specific
    helpers aligned to the Network Agent contract. This keeps higher-level
    modules concise while still allowing explicit control when they need to
    store custom network state.
    """

    CONTRACT_KEYS = {
        "route_selected": _ROUTE_SELECTED_KEY,
        "route_candidates": _ROUTE_CANDIDATES_KEY,
        "delivery_state": _DELIVERY_STATE_KEY,
        "delivery_retry_count": _DELIVERY_RETRY_COUNT_KEY,
        "delivery_last_error": _DELIVERY_LAST_ERROR_KEY,
        "endpoint_health": _ENDPOINT_HEALTH_KEY,
        "endpoint_circuit_state": _ENDPOINT_CIRCUIT_STATE_KEY,
        "session_snapshot": _SESSION_SNAPSHOT_KEY,
        "policy_decision": _POLICY_DECISION_KEY,
        "channel_metrics": _TELEMETRY_CHANNEL_METRICS_KEY,
    }

    def __init__(self) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("network_memory") or {}
        self._lock = RLock()

        self.key_prefix = self._get_str_config("key_prefix", "network")
        self.default_ttl_seconds = self._get_non_negative_int_config("default_ttl_seconds", 0)
        self.max_entries = max(1, self._get_non_negative_int_config("max_entries", 5000))
        self.auto_prune_on_write = self._get_bool_config("auto_prune_on_write", True)
        self.prune_interval_seconds = max(1, self._get_non_negative_int_config("prune_interval_seconds", 60))
        self.evict_oldest_on_capacity = self._get_bool_config("evict_oldest_on_capacity", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)

        self.max_event_log_size = max(1, self._get_non_negative_int_config("max_event_log_size", 1000))
        self.max_route_history_size = max(1, self._get_non_negative_int_config("max_route_history_size", 250))
        self.max_delivery_history_size = max(1, self._get_non_negative_int_config("max_delivery_history_size", 500))
        self.max_retry_history_size = max(1, self._get_non_negative_int_config("max_retry_history_size", 500))
        self.max_policy_history_size = max(1, self._get_non_negative_int_config("max_policy_history_size", 250))

        self.route_snapshot_ttl_seconds = self._get_non_negative_int_config("route_snapshot_ttl_seconds", 900)
        self.delivery_state_ttl_seconds = self._get_non_negative_int_config("delivery_state_ttl_seconds", 7200)
        self.endpoint_health_ttl_seconds = self._get_non_negative_int_config("endpoint_health_ttl_seconds", 600)
        self.session_snapshot_ttl_seconds = self._get_non_negative_int_config("session_snapshot_ttl_seconds", 1800)
        self.policy_decision_ttl_seconds = self._get_non_negative_int_config("policy_decision_ttl_seconds", 1800)
        self.channel_metrics_ttl_seconds = self._get_non_negative_int_config("channel_metrics_ttl_seconds", 600)
        self.error_snapshot_ttl_seconds = self._get_non_negative_int_config("error_snapshot_ttl_seconds", 7200)

        self._entries: Dict[str, MemoryEntry] = {}
        self._event_log: Deque[Dict[str, Any]] = deque(maxlen=self.max_event_log_size)
        self._route_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_route_history_size)
        self._delivery_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_delivery_history_size)
        self._retry_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_retry_history_size)
        self._policy_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_policy_history_size)

        self._started_at = _utcnow()
        self._last_pruned_at = self._started_at
        self._stats: Dict[str, int] = {
            "writes": 0,
            "reads": 0,
            "deletes": 0,
            "prunes": 0,
            "evictions": 0,
            "touches": 0,
            "increments": 0,
            "appends": 0,
        }

    # --------------------------------------------------------------------- #
    # Generic memory API
    # --------------------------------------------------------------------- #
    def set(self, key: str, value: Any, *,
        ttl_seconds: Optional[int] = None,
        source: Optional[str] = None,
        tags: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store or replace a memory entry."""
        normalized_key = self._validate_key(key)
        normalized_source = str(source).strip() if source is not None and str(source).strip() else None
        normalized_tags = normalize_tags(tags)
        normalized_metadata = normalize_metadata(metadata)
        normalized_value = json_safe(value)

        with self._lock:
            self._maybe_prune_locked()
            self._ensure_capacity_locked(incoming_key=normalized_key)

            now = _utcnow()
            existing = self._entries.get(normalized_key)
            expires_at = self._resolve_expiry_locked(ttl_seconds)
            entry = MemoryEntry(
                key=normalized_key,
                value=normalized_value,
                created_at=existing.created_at if existing is not None else now,
                updated_at=now,
                expires_at=expires_at,
                version=(existing.version + 1) if existing is not None else 1,
                source=normalized_source or (existing.source if existing is not None else None),
                tags=normalized_tags or (existing.tags if existing is not None else ()),
                metadata=merge_mappings(existing.metadata if existing is not None else {}, normalized_metadata),
            )
            self._entries[normalized_key] = entry
            self._stats["writes"] += 1
            self._append_event_locked(
                "set",
                key=normalized_key,
                value={"version": entry.version, "ttl_seconds": entry.ttl_seconds},
                metadata={"source": entry.source, "tags": list(entry.tags)},
            )
            return entry.to_dict(include_value=True)

    def set_many(self, values: Mapping[str, Any], *,
        ttl_seconds: Optional[int] = None,
        source: Optional[str] = None,
        tags: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Store several entries atomically under a single lock."""
        items = ensure_mapping(values, field_name="values")
        results: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            self._maybe_prune_locked()
            for key in items.keys():
                self._ensure_capacity_locked(incoming_key=self._validate_key(key))
            for key, value in items.items():
                results[str(key)] = self.set(
                    str(key),
                    value,
                    ttl_seconds=ttl_seconds,
                    source=source,
                    tags=tags,
                    metadata=metadata,
                )
        return results

    def merge(self, key: str, value: Mapping[str, Any], *,
        ttl_seconds: Optional[int] = None,
        source: Optional[str] = None,
        tags: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        deep: bool = True,
    ) -> Dict[str, Any]:
        """Merge a mapping value into an existing mapping entry."""
        normalized_key = self._validate_key(key)
        incoming = ensure_mapping(value, field_name="value")
        with self._lock:
            current = self.get(normalized_key, default={}, include_metadata=False)
            if current is None:
                current = {}
            if not isinstance(current, Mapping):
                raise DeliveryStateError(
                    "Cannot merge mapping data into a non-mapping memory entry.",
                    context={"operation": "memory_merge"},
                    details={"key": normalized_key, "existing_type": type(current).__name__},
                )
            merged = merge_mappings(current, incoming, deep=deep)
            return self.set(
                normalized_key,
                merged,
                ttl_seconds=ttl_seconds,
                source=source,
                tags=tags,
                metadata=metadata,
            )

    def get(self, key: str, *, default: Any = None,
            include_metadata: bool = False) -> Any:
        """Retrieve a value or entry payload by key."""
        normalized_key = self._validate_key(key)
        with self._lock:
            self._prune_key_if_expired_locked(normalized_key)
            entry = self._entries.get(normalized_key)
            self._stats["reads"] += 1
            if entry is None:
                return default
            return entry.to_dict(include_value=True) if include_metadata else json_safe(entry.value)

    def require(self, key: str, *, include_metadata: bool = False) -> Any:
        """Retrieve a required key or raise a network-native error."""
        value = self.get(key, default=None, include_metadata=include_metadata)
        if value is None:
            raise DeliveryStateError(
                "Required memory key is missing.",
                context={"operation": "memory_require"},
                details={"key": self._validate_key(key)},
            )
        return value

    def get_entry(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the full memory entry metadata for a key."""
        return self.get(key, default=None, include_metadata=True)

    def exists(self, key: str) -> bool:
        """Return whether a non-expired key exists."""
        normalized_key = self._validate_key(key)
        with self._lock:
            self._prune_key_if_expired_locked(normalized_key)
            return normalized_key in self._entries

    def touch(self, key: str, *, ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Refresh the expiration timestamp of an existing entry."""
        normalized_key = self._validate_key(key)
        with self._lock:
            self._prune_key_if_expired_locked(normalized_key)
            entry = self._entries.get(normalized_key)
            if entry is None:
                raise DeliveryStateError(
                    "Cannot touch a missing memory entry.",
                    context={"operation": "memory_touch"},
                    details={"key": normalized_key},
                )
            entry.expires_at = self._resolve_expiry_locked(ttl_seconds)
            entry.updated_at = _utcnow()
            entry.version += 1
            self._stats["touches"] += 1
            self._append_event_locked(
                "touch",
                key=normalized_key,
                value={"version": entry.version, "ttl_seconds": entry.ttl_seconds},
            )
            return entry.to_dict(include_value=True)

    def increment(self, key: str, *,
        amount: int = 1,
        initial_value: int = 0,
        ttl_seconds: Optional[int] = None,
        source: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Increment an integer memory entry and return the new value."""
        normalized_key = self._validate_key(key)
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise PayloadValidationError(
                "Increment amount must be an integer.",
                context={"operation": "memory_increment"},
                details={"key": normalized_key, "received_type": type(amount).__name__},
            )

        with self._lock:
            current = self.get(normalized_key, default=initial_value, include_metadata=False)
            if not isinstance(current, int):
                raise DeliveryStateError(
                    "Cannot increment a non-integer memory entry.",
                    context={"operation": "memory_increment"},
                    details={"key": normalized_key, "existing_type": type(current).__name__},
                )
            updated_value = current + amount
            self.set(
                normalized_key,
                updated_value,
                ttl_seconds=ttl_seconds,
                source=source,
                metadata=metadata,
            )
            self._stats["increments"] += 1
            return updated_value

    def append(self, key: str, item: Any, *,
        max_items: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
        source: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> List[Any]:
        """Append an item to a list-valued entry and return the resulting list."""
        normalized_key = self._validate_key(key)
        resolved_max_items = None if max_items is None else max(1, int(max_items))
        with self._lock:
            current = self.get(normalized_key, default=[], include_metadata=False)
            if current is None:
                current = []
            if not isinstance(current, list):
                raise DeliveryStateError(
                    "Cannot append to a non-list memory entry.",
                    context={"operation": "memory_append"},
                    details={"key": normalized_key, "existing_type": type(current).__name__},
                )
            current.append(json_safe(item))
            if resolved_max_items is not None and len(current) > resolved_max_items:
                current = current[-resolved_max_items:]
            self.set(
                normalized_key,
                current,
                ttl_seconds=ttl_seconds,
                source=source,
                metadata=metadata,
            )
            self._stats["appends"] += 1
            return current

    def pop(self, key: str, *, default: Any = None, include_metadata: bool = False) -> Any:
        """Remove and return a key's value or entry payload."""
        normalized_key = self._validate_key(key)
        with self._lock:
            self._prune_key_if_expired_locked(normalized_key)
            entry = self._entries.pop(normalized_key, None)
            if entry is None:
                return default
            self._stats["deletes"] += 1
            self._append_event_locked("pop", key=normalized_key)
            return entry.to_dict(include_value=True) if include_metadata else json_safe(entry.value)

    def delete(self, key: str) -> bool:
        """Delete a key if present."""
        normalized_key = self._validate_key(key)
        with self._lock:
            removed = self._entries.pop(normalized_key, None)
            if removed is None:
                return False
            self._stats["deletes"] += 1
            self._append_event_locked("delete", key=normalized_key)
            return True

    def clear(self, *, prefix: Optional[str] = None) -> int:
        """Clear all entries or all entries under a prefix."""
        with self._lock:
            if prefix is None:
                removed = len(self._entries)
                self._entries.clear()
                if removed:
                    self._stats["deletes"] += removed
                    self._append_event_locked("clear", metadata={"removed": removed})
                return removed

            normalized_prefix = self._validate_key(prefix)
            keys_to_remove = [key for key in self._entries if key.startswith(normalized_prefix)]
            for key in keys_to_remove:
                self._entries.pop(key, None)
            if keys_to_remove:
                self._stats["deletes"] += len(keys_to_remove)
                self._append_event_locked(
                    "clear_prefix",
                    key=normalized_prefix,
                    metadata={"removed": len(keys_to_remove)},
                )
            return len(keys_to_remove)

    def keys(self, *, prefix: Optional[str] = None) -> List[str]:
        """Return stored keys, optionally filtered by prefix."""
        with self._lock:
            self._prune_expired_locked()
            if prefix is None:
                return sorted(self._entries.keys())
            normalized_prefix = self._validate_key(prefix)
            return sorted(key for key in self._entries.keys() if key.startswith(normalized_prefix))

    def list_entries(self, *,
        prefix: Optional[str] = None,
        include_value: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return serialized memory entries, optionally filtered by prefix."""
        with self._lock:
            self._prune_expired_locked()
            entries: Iterable[MemoryEntry]
            if prefix is None:
                entries = self._entries.values()
            else:
                normalized_prefix = self._validate_key(prefix)
                entries = (entry for key, entry in self._entries.items() if key.startswith(normalized_prefix))
            return [entry.to_dict(include_value=include_value) for entry in sorted(entries, key=lambda item: item.key)]

    def prune_expired(self) -> int:
        """Remove expired entries and return the number pruned."""
        with self._lock:
            return self._prune_expired_locked()

    def export_snapshot(self, *,
        prefix: Optional[str] = None,
        include_values: bool = True,
        include_histories: bool = True,
    ) -> Dict[str, Any]:
        """Export the current memory state into a JSON-safe snapshot."""
        with self._lock:
            self._prune_expired_locked()
            entries = self.list_entries(prefix=prefix, include_value=include_values)
            payload: Dict[str, Any] = {
                "started_at": self._started_at.isoformat(),
                "last_pruned_at": self._last_pruned_at.isoformat(),
                "entry_count": len(entries),
                "stats": dict(self._stats),
                "entries": entries,
            }
            if include_histories:
                payload["histories"] = {
                    "events": list(self._event_log),
                    "routes": list(self._route_history),
                    "deliveries": list(self._delivery_history),
                    "retries": list(self._retry_history),
                    "policies": list(self._policy_history),
                }
            return json_safe(payload)

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics for the memory module."""
        with self._lock:
            self._prune_expired_locked()
            return {
                "started_at": self._started_at.isoformat(),
                "last_pruned_at": self._last_pruned_at.isoformat(),
                "entry_count": len(self._entries),
                "stats": dict(self._stats),
                "history_sizes": {
                    "events": len(self._event_log),
                    "routes": len(self._route_history),
                    "deliveries": len(self._delivery_history),
                    "retries": len(self._retry_history),
                    "policies": len(self._policy_history),
                },
            }

    # --------------------------------------------------------------------- #
    # Domain-specific network helpers
    # --------------------------------------------------------------------- #
    def set_route_selection( self, selected_route: Mapping[str, Any],
        candidate_routes: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        route_id: Optional[str] = None,
        reason: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record the latest route decision and candidate set.

        The canonical contract keys are updated, and a bounded route history is
        also maintained for diagnostics and observability.
        """
        selected = ensure_mapping(selected_route, field_name="selected_route")
        candidates = [ensure_mapping(item, field_name="candidate_route") for item in (candidate_routes or [])]

        snapshot = {
            "selected": json_safe(selected),
            "candidates": json_safe(candidates),
            "route_id": route_id,
            "reason": reason,
            "recorded_at": utc_timestamp(),
            "metadata": normalize_metadata(metadata),
        }
        resolved_ttl = self._coalesce_ttl(ttl_seconds, self.route_snapshot_ttl_seconds)

        with self._lock:
            self.set(_ROUTE_SELECTED_KEY, snapshot["selected"], ttl_seconds=resolved_ttl, source="route_selection")
            self.set(_ROUTE_CANDIDATES_KEY, snapshot["candidates"], ttl_seconds=resolved_ttl, source="route_selection")
            self._route_history.append(snapshot)
            self._append_event_locked("route_selection", key=_ROUTE_SELECTED_KEY, value=snapshot)
        return snapshot

    def get_route_selection(self) -> Dict[str, Any]:
        """Return the current top-level route selection contract snapshot."""
        return {
            "selected": self.get(_ROUTE_SELECTED_KEY, default=None),
            "candidates": self.get(_ROUTE_CANDIDATES_KEY, default=[]),
            "recent": list(self._route_history),
        }

    def record_delivery_state(
        self,
        state: str,
        *,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        route: Optional[str] = None,
        retry_count: Optional[int] = None,
        error: Optional[BaseException | Mapping[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a delivery state transition and related state keys."""
        normalized_state = ensure_non_empty_string(state, field_name="state").lower()
        delivery_snapshot = {
            "state": normalized_state,
            "message_id": message_id,
            "correlation_id": correlation_id,
            "endpoint": self._normalize_endpoint_reference(endpoint),
            "channel": normalize_channel_name(channel) if channel is not None else None,
            "route": route,
            "retry_count": retry_count if retry_count is not None else 0,
            "recorded_at": utc_timestamp(),
            "metadata": normalize_metadata(metadata),
        }
        if error is not None:
            delivery_snapshot["last_error"] = self._normalize_error_value(
                error,
                operation="delivery_state",
                endpoint=endpoint,
                channel=channel,
                route=route,
                correlation_id=correlation_id,
            )

        resolved_ttl = self._coalesce_ttl(ttl_seconds, self.delivery_state_ttl_seconds)
        entity_key = self._entity_key("network.delivery.by_message", message_id or correlation_id or "latest")

        with self._lock:
            self.set(_DELIVERY_STATE_KEY, delivery_snapshot, ttl_seconds=resolved_ttl, source="delivery")
            self.set(entity_key, delivery_snapshot, ttl_seconds=resolved_ttl, source="delivery")
            if retry_count is not None:
                self.set(
                    _DELIVERY_RETRY_COUNT_KEY,
                    {"retry_count": int(retry_count), "message_id": message_id, "correlation_id": correlation_id},
                    ttl_seconds=resolved_ttl,
                    source="delivery",
                )
            if error is not None:
                self.set(
                    _DELIVERY_LAST_ERROR_KEY,
                    delivery_snapshot["last_error"],
                    ttl_seconds=self._coalesce_ttl(ttl_seconds, self.error_snapshot_ttl_seconds),
                    source="delivery",
                )
            self._delivery_history.append(delivery_snapshot)
            self._append_event_locked("delivery_state", key=_DELIVERY_STATE_KEY, value=delivery_snapshot)
        return delivery_snapshot

    def increment_retry_count(self, *,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        amount: int = 1,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Increment the shared retry count contract and return its snapshot."""
        identifier = message_id or correlation_id or "global"
        counter_key = self._entity_key("network.delivery.retry_count.by_message", identifier)
        resolved_ttl = self._coalesce_ttl(ttl_seconds, self.delivery_state_ttl_seconds)

        with self._lock:
            entity_value = self.increment(counter_key, amount=amount, initial_value=0, ttl_seconds=resolved_ttl, source="retry")
            aggregate = {
                "retry_count": entity_value,
                "message_id": message_id,
                "correlation_id": correlation_id,
                "updated_at": utc_timestamp(),
            }
            self.set(_DELIVERY_RETRY_COUNT_KEY, aggregate, ttl_seconds=resolved_ttl, source="retry")
            return aggregate

    def record_retry_event(self,
        error: BaseException | Mapping[str, Any],
        *,
        attempt: int,
        max_attempts: Optional[int] = None,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        route: Optional[str] = None,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a retry event into bounded retry history and last-error state."""
        if attempt < 1:
            raise PayloadValidationError(
                "Retry attempt must be >= 1.",
                context={"operation": "record_retry_event"},
                details={"attempt": attempt},
            )
        normalized_error = self._normalize_error_value(
            error,
            operation="retry",
            endpoint=endpoint,
            channel=channel,
            route=route,
            correlation_id=correlation_id,
            attempt=attempt,
            max_attempts=max_attempts,
            metadata=metadata,
        )
        snapshot = {
            "attempt": attempt,
            "max_attempts": max_attempts,
            "message_id": message_id,
            "correlation_id": correlation_id,
            "endpoint": self._normalize_endpoint_reference(endpoint),
            "channel": normalize_channel_name(channel) if channel is not None else None,
            "route": route,
            "error": normalized_error,
            "recorded_at": utc_timestamp(),
            "metadata": normalize_metadata(metadata),
        }

        with self._lock:
            self._retry_history.append(snapshot)
            self.set(
                _DELIVERY_LAST_ERROR_KEY,
                normalized_error,
                ttl_seconds=self.error_snapshot_ttl_seconds,
                source="retry",
            )
            self._append_event_locked("retry_event", key=_DELIVERY_LAST_ERROR_KEY, value=snapshot)
        return snapshot

    def update_endpoint_health(self, endpoint: str, *,
        status: str = "healthy",
        latency_ms: Optional[int] = None,
        success_rate: Optional[float] = None,
        error_rate: Optional[float] = None,
        circuit_state: Optional[str] = None,
        last_error: Optional[BaseException | Mapping[str, Any]] = None,
        capabilities: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update endpoint health and endpoint circuit snapshots."""
        endpoint_ref = self._normalize_endpoint_reference(endpoint, required=True)
        # required=True ensures endpoint_ref is not None (raises otherwise)
        assert endpoint_ref is not None
        snapshot = {
            "endpoint": endpoint_ref,
            "status": ensure_non_empty_string(status, field_name="status").lower(),
            "latency_ms": int(latency_ms) if latency_ms is not None else None,
            "success_rate": float(success_rate) if success_rate is not None else None,
            "error_rate": float(error_rate) if error_rate is not None else None,
            "circuit_state": circuit_state.strip().lower() if isinstance(circuit_state, str) and circuit_state.strip() else None,
            "capabilities": json_safe(capabilities or {}),
            "last_checked_at": utc_timestamp(),
            "metadata": normalize_metadata(metadata),
        }
        if last_error is not None:
            snapshot["last_error"] = self._normalize_error_value(
                last_error,
                operation="endpoint_health",
                endpoint=endpoint_ref,
                metadata=metadata,
            )

        resolved_ttl = self._coalesce_ttl(ttl_seconds, self.endpoint_health_ttl_seconds)
        endpoint_key = self._entity_key(_ENDPOINT_HEALTH_KEY, endpoint_ref)

        with self._lock:
            current_health = self.get(_ENDPOINT_HEALTH_KEY, default={}, include_metadata=False)
            if current_health is None:
                current_health = {}
            if not isinstance(current_health, Mapping):
                current_health = {}
            merged_health = dict(current_health)
            merged_health[endpoint_ref] = snapshot
            self.set(_ENDPOINT_HEALTH_KEY, merged_health, ttl_seconds=resolved_ttl, source="endpoint_health")
            self.set(endpoint_key, snapshot, ttl_seconds=resolved_ttl, source="endpoint_health")
            if snapshot["circuit_state"] is not None:
                self.set_endpoint_circuit_state(
                    endpoint_ref,
                    snapshot["circuit_state"],
                    ttl_seconds=resolved_ttl,
                    metadata=metadata,
                )
            self._append_event_locked("endpoint_health", key=endpoint_key, value=snapshot)
        return snapshot

    def set_endpoint_circuit_state(
        self,
        endpoint: str,
        circuit_state: str,
        *,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Set the current circuit breaker state for an endpoint."""
        endpoint_ref = self._normalize_endpoint_reference(endpoint, required=True)
        assert endpoint_ref is not None
        normalized_circuit_state = ensure_non_empty_string(circuit_state, field_name="circuit_state").lower()
        snapshot = {
            "endpoint": endpoint_ref,
            "circuit_state": normalized_circuit_state,
            "updated_at": utc_timestamp(),
            "metadata": normalize_metadata(metadata),
        }
        resolved_ttl = self._coalesce_ttl(ttl_seconds, self.endpoint_health_ttl_seconds)
        endpoint_key = self._entity_key(_ENDPOINT_CIRCUIT_STATE_KEY, endpoint_ref)

        with self._lock:
            current_states = self.get(_ENDPOINT_CIRCUIT_STATE_KEY, default={}, include_metadata=False)
            if current_states is None or not isinstance(current_states, Mapping):
                current_states = {}
            merged_states = dict(current_states)
            merged_states[endpoint_ref] = snapshot
            self.set(_ENDPOINT_CIRCUIT_STATE_KEY, merged_states, ttl_seconds=resolved_ttl, source="circuit")
            self.set(endpoint_key, snapshot, ttl_seconds=resolved_ttl, source="circuit")
            self._append_event_locked("circuit_state", key=endpoint_key, value=snapshot)
        return snapshot

    def update_session_snapshot(
        self,
        session_id: str,
        snapshot: Mapping[str, Any],
        *,
        ttl_seconds: Optional[int] = None,
        merge_existing: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store session lifecycle state for a connection or transport session."""
        normalized_session_id = ensure_non_empty_string(session_id, field_name="session_id")
        incoming_snapshot = ensure_mapping(snapshot, field_name="snapshot")
        resolved_ttl = self._coalesce_ttl(ttl_seconds, self.session_snapshot_ttl_seconds)
        session_key = self._entity_key(_SESSION_SNAPSHOT_KEY, normalized_session_id)

        with self._lock:
            current_sessions = self.get(_SESSION_SNAPSHOT_KEY, default={}, include_metadata=False)
            if current_sessions is None or not isinstance(current_sessions, Mapping):
                current_sessions = {}
            current_entity = current_sessions.get(normalized_session_id, {}) if isinstance(current_sessions, Mapping) else {}
            if merge_existing and isinstance(current_entity, Mapping):
                merged_snapshot = merge_mappings(current_entity, incoming_snapshot, {"updated_at": utc_timestamp(), "metadata": normalize_metadata(metadata)})
            else:
                merged_snapshot = merge_mappings(incoming_snapshot, {"updated_at": utc_timestamp(), "metadata": normalize_metadata(metadata)})
            merged_snapshot["session_id"] = normalized_session_id
            merged_sessions = dict(current_sessions)
            merged_sessions[normalized_session_id] = merged_snapshot
            self.set(_SESSION_SNAPSHOT_KEY, merged_sessions, ttl_seconds=resolved_ttl, source="session")
            self.set(session_key, merged_snapshot, ttl_seconds=resolved_ttl, source="session")
            self._append_event_locked("session_snapshot", key=session_key, value=merged_snapshot)
        return merged_snapshot

    def record_policy_decision(
        self,
        policy_name: str,
        decision: str | Mapping[str, Any],
        *,
        endpoint: Optional[str] = None,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        reason: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a policy decision relevant to network transport behavior."""
        normalized_policy_name = ensure_non_empty_string(policy_name, field_name="policy_name")
        normalized_protocol = normalize_protocol_name(protocol) if protocol is not None else None
        normalized_channel = normalize_channel_name(channel) if channel is not None else None
        if isinstance(decision, Mapping):
            decision_value = json_safe(decision)
        else:
            decision_value = ensure_non_empty_string(str(decision), field_name="decision").lower()

        snapshot = {
            "policy_name": normalized_policy_name,
            "decision": decision_value,
            "endpoint": self._normalize_endpoint_reference(endpoint),
            "protocol": normalized_protocol,
            "channel": normalized_channel,
            "reason": reason,
            "recorded_at": utc_timestamp(),
            "metadata": normalize_metadata(metadata),
        }
        resolved_ttl = self._coalesce_ttl(ttl_seconds, self.policy_decision_ttl_seconds)
        policy_key = self._entity_key(_POLICY_DECISION_KEY, normalized_policy_name)

        with self._lock:
            current_policies = self.get(_POLICY_DECISION_KEY, default={}, include_metadata=False)
            if current_policies is None or not isinstance(current_policies, Mapping):
                current_policies = {}
            merged_policies = dict(current_policies)
            merged_policies[normalized_policy_name] = snapshot
            self.set(_POLICY_DECISION_KEY, merged_policies, ttl_seconds=resolved_ttl, source="policy")
            self.set(policy_key, snapshot, ttl_seconds=resolved_ttl, source="policy")
            self._policy_history.append(snapshot)
            self._append_event_locked("policy_decision", key=policy_key, value=snapshot)
        return snapshot

    def record_channel_metrics(
        self,
        channel: str,
        metrics: Mapping[str, Any],
        *,
        ttl_seconds: Optional[int] = None,
        merge_existing: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a telemetry snapshot for a specific channel or protocol family."""
        normalized_channel = normalize_channel_name(channel)
        metrics_map = ensure_mapping(metrics, field_name="metrics")
        resolved_ttl = self._coalesce_ttl(ttl_seconds, self.channel_metrics_ttl_seconds)
        channel_key = self._entity_key(_TELEMETRY_CHANNEL_METRICS_KEY, normalized_channel)
    
        with self._lock:
            current_metrics = self.get(_TELEMETRY_CHANNEL_METRICS_KEY, default={}, include_metadata=False)
            if current_metrics is None or not isinstance(current_metrics, Mapping):
                current_metrics = {}
    
            existing_metrics = current_metrics.get(normalized_channel, {}) if isinstance(current_metrics, Mapping) else {}
    
            if merge_existing and isinstance(existing_metrics, Mapping):
                merged_channel_metrics = merge_mappings(
                    existing_metrics,
                    metrics_map,
                    {"updated_at": utc_timestamp(), "metadata": normalize_metadata(metadata)},
                )
            else:
                # Replace completely, but keep required envelope fields
                merged_channel_metrics = merge_mappings(
                    metrics_map,
                    {"channel": normalized_channel, "updated_at": utc_timestamp(), "metadata": normalize_metadata(metadata)},
                )
    
            merged_metrics = dict(current_metrics)
            merged_metrics[normalized_channel] = merged_channel_metrics
    
            self.set(_TELEMETRY_CHANNEL_METRICS_KEY, merged_metrics, ttl_seconds=resolved_ttl, source="metrics")
            self.set(channel_key, merged_channel_metrics, ttl_seconds=resolved_ttl, source="metrics")
            self._append_event_locked("channel_metrics", key=channel_key, value=merged_channel_metrics)
    
        return merged_channel_metrics

    
    def get_network_health(self) -> Dict[str, Any]:
        """
        Return a coarse health snapshot spanning endpoints, circuits, sessions,
        and channel metrics.
        """
        with self._lock:
            self._prune_expired_locked()
            endpoint_health = self.get(_ENDPOINT_HEALTH_KEY, default={}, include_metadata=False)
            circuit_state = self.get(_ENDPOINT_CIRCUIT_STATE_KEY, default={}, include_metadata=False)
            session_snapshot = self.get(_SESSION_SNAPSHOT_KEY, default={}, include_metadata=False)
            channel_metrics = self.get(_TELEMETRY_CHANNEL_METRICS_KEY, default={}, include_metadata=False)
            delivery_state = self.get(_DELIVERY_STATE_KEY, default=None, include_metadata=False)
            last_error = self.get(_DELIVERY_LAST_ERROR_KEY, default=None, include_metadata=False)
    
            degraded_endpoints: List[str] = []
            open_circuits: List[str] = []
    
            if isinstance(endpoint_health, Mapping):
                for endpoint, snapshot in endpoint_health.items():
                    if isinstance(snapshot, Mapping):
                        status = str(snapshot.get("status", "")).lower()
                        if status in {"degraded", "unhealthy", "down"}:
                            degraded_endpoints.append(str(endpoint))
    
            if isinstance(circuit_state, Mapping):
                for endpoint, snapshot in circuit_state.items():
                    if isinstance(snapshot, Mapping):
                        state = str(snapshot.get("circuit_state", "")).lower()
                        if state in {"open", "half_open"}:
                            open_circuits.append(str(endpoint))
    
            return {
                "generated_at": utc_timestamp(),
                "entry_count": len(self._entries),
                "degraded_endpoints": sorted(degraded_endpoints),
                "open_circuits": sorted(open_circuits),
                "endpoint_health_count": len(endpoint_health) if isinstance(endpoint_health, Mapping) else 0,
                "session_count": len(session_snapshot) if isinstance(session_snapshot, Mapping) else 0,
                "channel_metric_count": len(channel_metrics) if isinstance(channel_metrics, Mapping) else 0,
                "latest_delivery_state": delivery_state,
                "latest_delivery_error": last_error,
                "stats": dict(self._stats),
            }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _get_str_config(self, name: str, default: str) -> str:
        value = self.memory_config.get(name, default)
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return text

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.memory_config.get(name, default)
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
            "Invalid boolean value in network memory configuration.",
            context={"operation": "network_memory_config"},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.memory_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in network memory configuration.",
                context={"operation": "network_memory_config"},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Configuration value must be non-negative.",
                context={"operation": "network_memory_config"},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _validate_key(self, key: str) -> str:
        normalized_key = ensure_non_empty_string(key, field_name="memory_key")
        return normalized_key

    def _entity_key(self, base_key: str, identifier: str) -> str:
        token = self._tokenize_identifier(identifier)
        return f"{self._validate_key(base_key)}.{token}"

    def _tokenize_identifier(self, value: Any) -> str:
        text = ensure_non_empty_string(str(value), field_name="identifier").lower()
        text = _KEY_TOKEN_PATTERN.sub("_", text).strip("._-")
        if not text:
            return "value"
        if len(text) > _MAX_KEY_TOKEN_LENGTH:
            digest = generate_idempotency_key(text, namespace="memory_key")
            text = f"{text[:48]}_{digest[:16]}"
        return text

    def _coalesce_ttl(self, ttl_seconds: Optional[int], fallback: int) -> Optional[int]:
        if ttl_seconds is None:
            ttl_seconds = fallback
        return ttl_seconds if ttl_seconds not in (0, None) else None

    def _resolve_expiry_locked(self, ttl_seconds: Optional[int]) -> Optional[datetime]:
        resolved_ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if resolved_ttl in (None, 0):
            return None
        if isinstance(resolved_ttl, bool):
            raise PayloadValidationError(
                "TTL seconds must be an integer value.",
                context={"operation": "memory_set"},
                details={"ttl_seconds": ttl_seconds},
            )
        if not isinstance(resolved_ttl, int):
            try:
                resolved_ttl = int(resolved_ttl)
            except (TypeError, ValueError) as exc:
                raise PayloadValidationError(
                    "TTL seconds must be an integer value.",
                    context={"operation": "memory_set"},
                    details={"ttl_seconds": ttl_seconds},
                    cause=exc,
                ) from exc
        if resolved_ttl < 0:
            raise PayloadValidationError(
                "TTL seconds must be non-negative.",
                context={"operation": "memory_set"},
                details={"ttl_seconds": ttl_seconds},
            )
        return _utcnow() + timedelta(seconds=int(resolved_ttl))

    def _normalize_endpoint_reference(self, endpoint: Optional[str], *, required: bool = False) -> Optional[str]:
        if endpoint is None:
            if required:
                raise PayloadValidationError(
                    "Endpoint is required.",
                    context={"operation": "memory_endpoint_normalization"},
                )
            return None
        text = ensure_non_empty_string(str(endpoint), field_name="endpoint")
        if "://" in text:
            try:
                return normalize_endpoint(text)
            except NetworkError:
                return text
        return text

    def _normalize_error_value(
        self,
        error: BaseException | Mapping[str, Any],
        *,
        operation: str,
        endpoint: Optional[str] = None,
        channel: Optional[str] = None,
        route: Optional[str] = None,
        correlation_id: Optional[str] = None,
        attempt: Optional[int] = None,
        max_attempts: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if isinstance(error, Mapping):
            return json_safe(error)
        if isinstance(error, NetworkError):
            return error.to_memory_snapshot()
        return build_error_snapshot(
            error,
            operation=operation,
            endpoint=self._normalize_endpoint_reference(endpoint),
            channel=normalize_channel_name(channel) if channel is not None else None,
            route=route,
            correlation_id=correlation_id,
            attempt=attempt,
            max_attempts=max_attempts,
            metadata=metadata,
        )

    def _maybe_prune_locked(self) -> None:
        if not self.auto_prune_on_write:
            return
        now = _utcnow()
        if (now - self._last_pruned_at).total_seconds() >= self.prune_interval_seconds:
            self._prune_expired_locked(now=now)

    def _prune_key_if_expired_locked(self, key: str) -> None:
        entry = self._entries.get(key)
        if entry is not None and entry.expired:
            self._entries.pop(key, None)
            self._stats["prunes"] += 1

    def _prune_expired_locked(self, *, now: Optional[datetime] = None) -> int:
        reference = now or _utcnow()
        expired_keys = [key for key, entry in self._entries.items() if entry.expires_at is not None and entry.expires_at <= reference]
        for key in expired_keys:
            self._entries.pop(key, None)
        if expired_keys:
            self._stats["prunes"] += len(expired_keys)
            self._append_event_locked(
                "prune_expired",
                metadata={"removed_keys": expired_keys, "removed_count": len(expired_keys)},
            )
        self._last_pruned_at = reference
        return len(expired_keys)

    def _ensure_capacity_locked(self, *, incoming_key: str) -> None:
        if incoming_key in self._entries:
            return
        if len(self._entries) < self.max_entries:
            return

        self._prune_expired_locked()
        if len(self._entries) < self.max_entries:
            return

        if not self.evict_oldest_on_capacity:
            raise ReliabilityError(
                "Network memory capacity exceeded and eviction is disabled.",
                context={"operation": "memory_capacity"},
                details={"max_entries": self.max_entries, "incoming_key": incoming_key},
            )

        oldest_key = min(self._entries.items(), key=lambda item: item[1].updated_at)[0]
        self._entries.pop(oldest_key, None)
        self._stats["evictions"] += 1
        self._append_event_locked(
            "evict_oldest",
            key=oldest_key,
            metadata={"incoming_key": incoming_key, "max_entries": self.max_entries},
        )

    def _append_event_locked(
        self,
        event_type: str,
        *,
        key: Optional[str] = None,
        value: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        snapshot = {
            "event_type": event_type,
            "key": key,
            "occurred_at": utc_timestamp(),
            "value": sanitize_for_logging(value) if self.sanitize_logs else json_safe(value),
            "metadata": sanitize_for_logging(metadata) if self.sanitize_logs else json_safe(metadata),
        }
        self._event_log.append(json_safe(snapshot))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print("\n=== Running Network Memory ===\n")
    printer.status("TEST", "Network Memory initialized", "info")

    memory = NetworkMemory()

    route_snapshot = memory.set_route_selection(
        selected_route={
            "channel": "http",
            "endpoint": "https://api.example.com/v1/jobs",
            "latency_ms": 43,
            "health_score": 0.98,
        },
        candidate_routes=[
            {"channel": "http", "endpoint": "https://api.example.com/v1/jobs", "score": 0.98},
            {"channel": "queue", "endpoint": "jobs-primary", "score": 0.91},
        ],
        route_id="route_primary_http",
        reason="lowest_latency_with_policy_pass",
        metadata={"region": "eu-west", "task_class": "relay"},
    )
    printer.status("TEST", "Route selection stored", "info")

    delivery_snapshot = memory.record_delivery_state(
        "sent",
        message_id="msg_demo_001",
        correlation_id="corr_demo_001",
        endpoint="https://api.example.com/v1/jobs",
        channel="http",
        route="primary",
        retry_count=1,
        metadata={"delivery_mode": "at_least_once"},
    )
    printer.status("TEST", "Delivery state recorded", "info")

    retry_snapshot = memory.record_retry_event(
        TimeoutError("upstream acknowledgment timeout"),
        attempt=2,
        max_attempts=5,
        endpoint="https://api.example.com/v1/jobs",
        channel="http",
        route="primary",
        correlation_id="corr_demo_001",
        message_id="msg_demo_001",
        metadata={"backoff_ms": 250},
    )
    printer.status("TEST", "Retry event recorded", "info")

    endpoint_health = memory.update_endpoint_health(
        "https://api.example.com/v1/jobs",
        status="degraded",
        latency_ms=151,
        success_rate=0.94,
        error_rate=0.06,
        circuit_state="half_open",
        last_error=TimeoutError("recent elevated timeout rate"),
        capabilities={"streaming": False, "payload_caps": {"max_bytes": 1048576}},
        metadata={"region": "eu-west"},
    )
    printer.status("TEST", "Endpoint health updated", "info")

    session_snapshot = memory.update_session_snapshot(
        "sess_demo_001",
        {
            "channel": "websocket",
            "endpoint": "wss://stream.example.com/events",
            "state": "connected",
            "last_seen_at": utc_timestamp(),
        },
        metadata={"auth_mode": "token"},
    )
    printer.status("TEST", "Session snapshot updated", "info")

    policy_snapshot = memory.record_policy_decision(
        "destination_allowlist",
        "allowed",
        endpoint="https://api.example.com/v1/jobs",
        protocol="https",
        channel="http",
        reason="matched configured allowlist entry",
        metadata={"policy_version": "2026.04"},
    )
    printer.status("TEST", "Policy decision recorded", "info")

    metrics_snapshot = memory.record_channel_metrics(
        "http",
        {
            "success_rate": 0.987,
            "retry_rate": 0.041,
            "p95_latency_ms": 122,
            "requests": 481,
        },
        metadata={"window": "5m"},
    )
    printer.status("TEST", "Channel metrics recorded", "info")

    health_snapshot = memory.get_network_health()
    exported = memory.export_snapshot(include_values=False, include_histories=True)

    print("Route Snapshot:", stable_json_dumps(route_snapshot))
    print("Delivery Snapshot:", stable_json_dumps(delivery_snapshot))
    print("Retry Snapshot:", stable_json_dumps(retry_snapshot))
    print("Endpoint Health:", stable_json_dumps(endpoint_health))
    print("Session Snapshot:", stable_json_dumps(session_snapshot))
    print("Policy Snapshot:", stable_json_dumps(policy_snapshot))
    print("Metrics Snapshot:", stable_json_dumps(metrics_snapshot))
    print("Health Snapshot:", stable_json_dumps(health_snapshot))
    print("Export Summary:", stable_json_dumps({"entry_count": exported["entry_count"], "stats": exported["stats"]}))

    assert memory.exists("network.route.selected")
    assert memory.get("network.delivery.state")["state"] == "sent"
    assert memory.get("network.delivery.last_error")["error_code"] in {"DELIVERY_TIMEOUT", "CONNECTION_TIMEOUT", "UNEXPECTED_NETWORK_ERROR"}
    assert "https://api.example.com:443/v1/jobs" in memory.get("network.endpoint.health")
    assert memory.get("network.policy.decision")["destination_allowlist"]["decision"] == "allowed"
    assert memory.get("network.telemetry.channel_metrics")["http"]["requests"] == 481

    printer.status("TEST", "All Network Memory checks passed", "info")
    print("\n=== Test ran successfully ===\n")
