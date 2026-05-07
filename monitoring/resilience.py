"""
monitoring/resilience.py
────────────────────────
Production-grade resilience primitives used across the monitoring subsystem.

  • RetryPolicy        – exponential back-off with full jitter
  • CircuitBreaker     – per-key state machine (CLOSED → OPEN → HALF-OPEN)
  • TokenBucketLimiter – thread-safe token-bucket rate limiter
  • StructuredLogger   – thin wrapper that emits plain or JSON log records
"""

from __future__ import annotations

import json
import math
import random
import threading
import time

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Resilience Primitives")
printer = PrettyPrinter()

F = TypeVar("F", bound=Callable[..., Any])

# ──────────────────────────────────────────────
# Retry policy
# ──────────────────────────────────────────────
@dataclass
class RetryPolicy:
    """
    Exponential back-off with full jitter.

    Delay formula (full jitter):
        sleep = random.uniform(0, min(max_delay, base_delay * backoff_factor ** attempt))
    """
    max_retries: int = 3
    base_delay: float = 0.5        # seconds
    max_delay: float = 30.0        # seconds cap
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (OSError, TimeoutError, ConnectionError)
    )

    def _sleep_duration(self, attempt: int) -> float:
        cap = min(self.max_delay, self.base_delay * (self.backoff_factor ** attempt))
        return random.uniform(0, cap) if self.jitter else cap

    def execute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Call *fn* with *args*/*kwargs*, retrying on retryable exceptions.
        Raises the last exception if all attempts fail.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except self.retryable_exceptions as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self._sleep_duration(attempt)
                    time.sleep(delay)
            except Exception:
                raise  # Non-retryable – propagate immediately
        raise last_exc  # type: ignore[misc]

    def decorator(self, fn: F) -> F:
        """Use as @policy.decorator to wrap a function."""
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.execute(fn, *args, **kwargs)
        return wrapper  # type: ignore[return-value]


# ──────────────────────────────────────────────
# Circuit breaker
# ──────────────────────────────────────────────
class CBState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing fast
    HALF_OPEN = "half_open"  # Probing recovery


class CircuitBreakerOpen(Exception):
    """Raised when a call is blocked by an open circuit."""


@dataclass
class _CBBucket:
    state: CBState = CBState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    opened_at: float = 0.0
    half_open_calls: int = 0


class CircuitBreakerRegistry:
    """
    Thread-safe registry of per-key circuit breakers.

    Usage::

        registry = CircuitBreakerRegistry(failure_threshold=5, recovery_timeout=60)
        result = registry.call("smtp_transport", smtp_send, *args, **kwargs)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._buckets: dict[str, _CBBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, key: str) -> _CBBucket:
        if key not in self._buckets:
            self._buckets[key] = _CBBucket()
        return self._buckets[key]

    def state_of(self, key: str) -> CBState:
        with self._lock:
            return self._bucket(key).state

    def call(self, key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            bucket = self._bucket(key)
            now = time.monotonic()

            if bucket.state == CBState.OPEN:
                if now - bucket.opened_at >= self.recovery_timeout:
                    bucket.state = CBState.HALF_OPEN
                    bucket.half_open_calls = 0
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit '{key}' is OPEN. "
                        f"Retry after {self.recovery_timeout - (now - bucket.opened_at):.1f}s."
                    )

            if bucket.state == CBState.HALF_OPEN:
                if bucket.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen(
                        f"Circuit '{key}' is HALF-OPEN and probe quota exhausted."
                    )
                bucket.half_open_calls += 1

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            with self._lock:
                bucket = self._bucket(key)
                bucket.failure_count += 1
                if bucket.state == CBState.HALF_OPEN:
                    bucket.state = CBState.OPEN
                    bucket.opened_at = time.monotonic()
                elif bucket.failure_count >= self.failure_threshold:
                    bucket.state = CBState.OPEN
                    bucket.opened_at = time.monotonic()
            raise exc
        else:
            with self._lock:
                bucket = self._bucket(key)
                if bucket.state == CBState.HALF_OPEN:
                    bucket.state = CBState.CLOSED
                    bucket.failure_count = 0
                    bucket.success_count = 0
                else:
                    bucket.failure_count = max(0, bucket.failure_count - 1)
            return result

    def reset(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)

    def status(self) -> dict[str, str]:
        with self._lock:
            return {k: v.state.value for k, v in self._buckets.items()}


# ──────────────────────────────────────────────
# Token-bucket rate limiter
# ──────────────────────────────────────────────
class RateLimitExceeded(Exception):
    """Raised when the token bucket is empty."""


class TokenBucketLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Parameters
    ----------
    capacity:
        Maximum tokens (burst size).
    refill_per_second:
        Tokens added per second (continuous refill).
    """

    def __init__(self, capacity: int = 10, refill_per_second: float = 1.0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be > 0")
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._refill_rate = refill_per_second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def acquire(self, tokens: int = 1, block: bool = False, timeout: float = 5.0) -> bool:
        """
        Consume *tokens* from the bucket.

        Parameters
        ----------
        block:
            If True, wait up to *timeout* seconds for tokens to become available.
        Returns True on success, raises RateLimitExceeded if blocking times out
        or if non-blocking and insufficient tokens.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            if not block:
                raise RateLimitExceeded(
                    f"Rate limit exceeded: need {tokens} token(s), bucket empty."
                )
            wait = tokens / self._refill_rate
            if time.monotonic() + wait > deadline:
                raise RateLimitExceeded(
                    f"Rate limit exceeded: could not acquire {tokens} token(s) within {timeout}s."
                )
            time.sleep(min(wait, 0.05))

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def capacity(self) -> float:
        return self._capacity