"""Runtime ownership primitives for the SLAI AgentFactory.

This module separates definition registration from runtime ownership. It owns
no agents and imports no AgentFactory implementation, which keeps the runtime
contracts reusable without creating a circular dependency.
"""

from __future__ import annotations

import time
import uuid

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, RLock
from typing import Any, Callable, Dict, Iterator, Optional

from ..runtime_contracts import AgentRuntimeIdentity


class CircuitState(str, Enum):
    """Construction-circuit states for one definition and runtime scope."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class FailureCircuit:
    """Bounded retry state for one exact construction key.

    Retryable failures open the circuit only after ``failure_threshold``
    consecutive failures. A single probe is admitted after the cooldown.
    Non-retryable failures remain open until an explicit reset.
    """

    key: str
    failure_threshold: int = 3
    base_cooldown_seconds: float = 5.0
    max_cooldown_seconds: float = 300.0
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    total_failures: int = 0
    retryable_failures: int = 0
    opened_count: int = 0
    backoff_level: int = 0
    permanent: bool = False
    first_failure_at: Optional[float] = None
    last_failure_at: Optional[float] = None
    last_success_at: Optional[float] = None
    retry_at: Optional[float] = None
    last_error: Optional[str] = None
    last_error_type: Optional[str] = None
    _probe_active: bool = field(default=False, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if self.base_cooldown_seconds < 0 or self.max_cooldown_seconds < 0:
            raise ValueError("circuit cooldown values must be >= 0")
        if self.max_cooldown_seconds < self.base_cooldown_seconds:
            raise ValueError("max_cooldown_seconds must be >= base_cooldown_seconds")

    def allow_attempt(self, now: Optional[float] = None) -> bool:
        """Return whether construction may proceed, reserving half-open probes."""

        current = time.time() if now is None else float(now)
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.permanent:
                return False
            if self.state == CircuitState.OPEN:
                if self.retry_at is None or current < self.retry_at:
                    return False
                self.state = CircuitState.HALF_OPEN
                self._probe_active = False
            if self.state == CircuitState.HALF_OPEN:
                if self._probe_active:
                    return False
                self._probe_active = True
                return True
            return False

    def record_failure(self, error: BaseException | str, *, retryable: bool) -> None:
        now = time.time()
        with self._lock:
            previous_state = self.state
            self.total_failures += 1
            self.consecutive_failures += 1
            self.retryable_failures += int(bool(retryable))
            self.first_failure_at = self.first_failure_at or now
            self.last_failure_at = now
            self.last_error = str(error)
            self.last_error_type = type(error).__name__ if isinstance(error, BaseException) else "RuntimeFailure"
            self._probe_active = False

            should_open = (not retryable) or previous_state == CircuitState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold
            if not should_open:
                self.state = CircuitState.CLOSED
                self.retry_at = None
                return

            self.state = CircuitState.OPEN
            self.opened_count += 1
            self.permanent = not retryable
            if self.permanent:
                self.retry_at = None
                return
            exponent = self.backoff_level
            self.backoff_level += 1
            cooldown = min(
                self.max_cooldown_seconds,
                self.base_cooldown_seconds * (2 ** exponent),
            )
            self.retry_at = now + cooldown

    def record_success(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.permanent = False
            self.backoff_level = 0
            self.retry_at = None
            self.last_success_at = time.time()

            @property
            def runtime_key(self):
                raise NotImplementedError

            self._probe_active = False

    def cancel_probe(self) -> None:
        """Release a reserved half-open probe that did not reach construction.

        A factory may need to consult more than one circuit before attempting
        construction (for example, a definition circuit and a scope circuit).
        If a later circuit rejects the attempt, any earlier half-open
        reservation must be returned or the circuit would remain artificially
        busy until another failure/reset event.
        """

        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self._probe_active = False

    def reset(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.permanent = False
            self.backoff_level = 0
            self.retry_at = None
            self.last_error = None
            self.last_error_type = None
            self._probe_active = False

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            return {
                "key": self.key,
                "state": self.state.value,
                "permanent": self.permanent,
                "failure_threshold": self.failure_threshold,
                "consecutive_failures": self.consecutive_failures,
                "total_failures": self.total_failures,
                "retryable_failures": self.retryable_failures,
                "opened_count": self.opened_count,
                "backoff_level": self.backoff_level,
                "first_failure_at": self.first_failure_at,
                "last_failure_at": self.last_failure_at,
                "last_success_at": self.last_success_at,
                "retry_at": self.retry_at,
                "retry_after_seconds": None if self.retry_at is None else max(0.0, self.retry_at - now),
                "last_error": self.last_error,
                "last_error_type": self.last_error_type,
            }


@dataclass(frozen=True, slots=True)
class AgentHandle:
    """Stable, non-owning reference to one exact managed runtime instance."""

    identity: AgentRuntimeIdentity
    implementation: str = "in_process"
    issued_at: float = field(default_factory=time.time, compare=False)

    @property
    def runtime_key(self) -> str:
        """Compatibility alias for the exact runtime cache key."""

        return self.identity.cache_key

    @property
    def cache_key(self) -> str:
        """Compatibility alias for the exact runtime key."""

        return self.runtime_key

    @property
    def scope_key(self) -> str:
        return self.identity.cache_key

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.identity.to_dict(),
            "implementation": self.implementation,
            "issued_at": self.issued_at,
        }


class AgentLease:
    """Explicit ownership claim that prevents release while it is active."""

    __slots__ = (
        "_handle",
        "_instance",
        "_lease_id",
        "_release_callback",
        "_released",
        "_lock",
        "_acquired_at",
    )

    def __init__(
        self,
        handle: AgentHandle,
        instance: Any,
        release_callback: Callable[[str], bool],
        *,
        lease_id: Optional[str] = None,
    ) -> None:
        self._handle = handle
        self._instance = instance
        self._lease_id = str(lease_id or uuid.uuid4().hex)
        self._release_callback = release_callback
        self._released = False
        self._lock = RLock()
        self._acquired_at = time.time()

    @property
    def handle(self) -> AgentHandle:
        return self._handle

    @property
    def instance(self) -> Any:
        with self._lock:
            if self._released:
                raise RuntimeError("Agent lease has already been released")
            return self._instance

    @property
    def lease_id(self) -> str:
        return self._lease_id

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def release(self) -> bool:
        with self._lock:
            if self._released:
                return False
            released = bool(self._release_callback(self._lease_id))
            self._released = True
            self._instance = None
            return released

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "lease_id": self._lease_id,
                "released": self._released,
                "acquired_at": self._acquired_at,
                "handle": self._handle.to_dict(),
            }

    def __enter__(self) -> Any:
        return self.instance

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


@dataclass(slots=True)
class _SingleFlightEntry:
    lock: Lock = field(default_factory=Lock)
    users: int = 0


class KeyedSingleFlight:
    """Per-key construction serialization with bounded lock retention."""

    def __init__(self) -> None:
        self._guard = RLock()
        self._entries: Dict[str, _SingleFlightEntry] = {}

    @contextmanager
    def acquire(self, key: str) -> Iterator[None]:
        normalized = str(key)
        with self._guard:
            entry = self._entries.get(normalized)
            if entry is None:
                entry = _SingleFlightEntry()
                self._entries[normalized] = entry
            entry.users += 1
        entry.lock.acquire()
        try:
            yield
        finally:
            entry.lock.release()
            with self._guard:
                entry.users -= 1
                if entry.users == 0 and not entry.lock.locked():
                    self._entries.pop(normalized, None)

    def size(self) -> int:
        with self._guard:
            return len(self._entries)


__all__ = [
    "AgentHandle",
    "AgentLease",
    "CircuitState",
    "FailureCircuit",
    "KeyedSingleFlight",
]
