"""Shared runtime identity, lifecycle, and degradation contracts for SLAI agents.

The contract deliberately separates three concepts that must not be conflated:

``definition_id``
    The registered implementation identity (``name@version``).
``instance_id``
    The unique identity of one constructed runtime object.
``scope_id``
    A non-secret fingerprint of the constructor context used for cache reuse.

The same status model is used by ``BaseAgent`` and ``AgentFactory`` so health
payloads remain consistent and optional persistence/telemetry failures are
observable without making the primary agent operation fail.
"""

from __future__ import annotations

import hashlib
import json
import time

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Set, Tuple


class RuntimeContractViolation(RuntimeError):
    """Raised when a caller attempts an invalid runtime contract operation."""


class RuntimeLifecycle(str, Enum):
    """Canonical lifecycle states for one runtime instance."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class RuntimeHealth(str, Enum):
    """Canonical health states derived from lifecycle and degradation data."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


_ALLOWED_TRANSITIONS: Dict[RuntimeLifecycle, Set[RuntimeLifecycle]] = {
    RuntimeLifecycle.INITIALIZING: {
        RuntimeLifecycle.ACTIVE,
        RuntimeLifecycle.DEGRADED,
        RuntimeLifecycle.STOPPING,
        RuntimeLifecycle.FAILED,
    },
    RuntimeLifecycle.ACTIVE: {
        RuntimeLifecycle.DEGRADED,
        RuntimeLifecycle.STOPPING,
        RuntimeLifecycle.FAILED,
    },
    RuntimeLifecycle.DEGRADED: {
        RuntimeLifecycle.ACTIVE,
        RuntimeLifecycle.STOPPING,
        RuntimeLifecycle.FAILED,
    },
    RuntimeLifecycle.STOPPING: {
        RuntimeLifecycle.STOPPED,
        RuntimeLifecycle.FAILED,
    },
    RuntimeLifecycle.STOPPED: set(),
    RuntimeLifecycle.FAILED: {
        RuntimeLifecycle.STOPPING,
        RuntimeLifecycle.STOPPED,
    },
}


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _scope_value(value: Any) -> Any:
    """Return a deterministic, non-secret shape for runtime scope hashing."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return {
            str(key): _scope_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_scope_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_scope_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return {"type": _qualified_type(value), "object_id": id(value)}


def build_runtime_scope_id(
    *,
    shared_memory: Any,
    config: Optional[Mapping[str, Any]] = None,
    constructor_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build a short cache-scope fingerprint without exposing constructor data."""
    memory_scope: Any
    if shared_memory is None:
        memory_scope = "factory-managed"
    else:
        memory_scope = {"type": _qualified_type(shared_memory), "object_id": id(shared_memory)}

    payload = {
        "shared_memory": memory_scope,
        "config": _scope_value(config or {}),
        "constructor_kwargs": _scope_value(constructor_kwargs or {}),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class AgentRuntimeIdentity:
    """Stable definition identity plus unique runtime-instance identity."""

    agent_type: str
    version: str
    instance_id: str
    scope_id: str

    @property
    def definition_id(self) -> str:
        return f"{self.agent_type}@{self.version}"

    @property
    def cache_key(self) -> str:
        return f"instance:{self.definition_id}:{self.scope_id}"

    def to_dict(self) -> Dict[str, str]:
        return {
            "agent_type": self.agent_type,
            "version": self.version,
            "definition_id": self.definition_id,
            "instance_id": self.instance_id,
            "scope_id": self.scope_id,
            "cache_key": self.cache_key,
        }


@dataclass(slots=True)
class RuntimeDegradation:
    """Aggregated failure state for one runtime channel and operation."""

    channel: str
    operation: str
    message: str
    error_type: str
    first_seen_at: float
    last_seen_at: float
    occurrences: int = 1
    retryable: bool = True

    def update(self, error: BaseException | str, *, retryable: bool) -> None:
        self.message = str(error)
        self.error_type = type(error).__name__ if isinstance(error, BaseException) else "RuntimeDegradation"
        self.last_seen_at = time.time()
        self.occurrences += 1
        self.retryable = retryable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "operation": self.operation,
            "message": self.message,
            "error_type": self.error_type,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "occurrences": self.occurrences,
            "retryable": self.retryable,
        }


class RuntimeStatus:
    """Thread-safe lifecycle and degraded-channel state for one runtime owner."""

    def __init__(self, lifecycle: RuntimeLifecycle | str = RuntimeLifecycle.INITIALIZING) -> None:
        self._lock = RLock()
        self._lifecycle = RuntimeLifecycle(lifecycle)
        self._degradations: Dict[Tuple[str, str], RuntimeDegradation] = {}
        self._updated_at = time.time()

    @property
    def lifecycle(self) -> RuntimeLifecycle:
        with self._lock:
            return self._lifecycle

    @property
    def health(self) -> RuntimeHealth:
        with self._lock:
            if self._lifecycle in {RuntimeLifecycle.FAILED, RuntimeLifecycle.STOPPED}:
                return RuntimeHealth.UNAVAILABLE
            if self._degradations or self._lifecycle == RuntimeLifecycle.DEGRADED:
                return RuntimeHealth.DEGRADED
            return RuntimeHealth.HEALTHY

    def transition(self, target: RuntimeLifecycle | str) -> RuntimeLifecycle:
        requested = RuntimeLifecycle(target)
        with self._lock:
            current = self._lifecycle
            if requested == RuntimeLifecycle.ACTIVE and self._degradations:
                requested = RuntimeLifecycle.DEGRADED
            if requested == current:
                return current
            if requested not in _ALLOWED_TRANSITIONS[current]:
                raise RuntimeContractViolation(
                    f"Invalid runtime lifecycle transition: {current.value} -> {requested.value}"
                )
            self._lifecycle = requested
            self._updated_at = time.time()
            return requested

    def mark_degraded(
        self,
        channel: str,
        operation: str,
        error: BaseException | str,
        *,
        retryable: bool = True,
    ) -> RuntimeDegradation:
        normalized_channel = str(channel).strip().lower()
        normalized_operation = str(operation).strip().lower()
        if not normalized_channel or not normalized_operation:
            raise RuntimeContractViolation("Degradation channel and operation must be non-empty")

        key = (normalized_channel, normalized_operation)
        with self._lock:
            record = self._degradations.get(key)
            if record is None:
                now = time.time()
                record = RuntimeDegradation(
                    channel=normalized_channel,
                    operation=normalized_operation,
                    message=str(error),
                    error_type=type(error).__name__ if isinstance(error, BaseException) else "RuntimeDegradation",
                    first_seen_at=now,
                    last_seen_at=now,
                    retryable=retryable,
                )
                self._degradations[key] = record
            else:
                record.update(error, retryable=retryable)
            if self._lifecycle == RuntimeLifecycle.ACTIVE:
                self._lifecycle = RuntimeLifecycle.DEGRADED
            self._updated_at = time.time()
            return record

    def mark_recovered(self, channel: str, operation: str) -> bool:
        key = (str(channel).strip().lower(), str(operation).strip().lower())
        with self._lock:
            removed = self._degradations.pop(key, None) is not None
            if removed:
                if not self._degradations and self._lifecycle == RuntimeLifecycle.DEGRADED:
                    self._lifecycle = RuntimeLifecycle.ACTIVE
                self._updated_at = time.time()
            return removed

    def snapshot(self, identity: Optional[AgentRuntimeIdentity] = None) -> Dict[str, Any]:
        with self._lock:
            health = self.health
            degradations = [
                record.to_dict()
                for _, record in sorted(self._degradations.items(), key=lambda item: item[0])
            ]
            payload: Dict[str, Any] = {
                "status": health.value,
                "health": health.value,
                "lifecycle": self._lifecycle.value,
                "degraded_channels": sorted({record.channel for record in self._degradations.values()}),
                "degradations": degradations,
                "updated_at": self._updated_at,
            }
            if identity is not None:
                payload["identity"] = identity.to_dict()
            return payload
