"""Rate limiter with token bucket algorithm and optional Redis backend.

This module provides a thread‑safe rate limiter that can be used to control
request rates per user, IP, or any custom key. For multi‑instance deployments,
a Redis backend is recommended.
"""

from __future__ import annotations

import math
import threading
import time

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .utils.config_loader import get_config_section
from .utils.functions_error import (RateLimitConfigurationError, RateLimitError, RateLimitExceeded)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Rate Limiter")
printer = PrettyPrinter

@dataclass(frozen=True)
class RateLimitDecision:
    """Result of evaluating a rate-limited request for a single key."""

    key: str
    allowed: bool
    requested: int
    limit: int
    remaining: float
    retry_after: float = 0.0

# In‑memory backend (single instance)
class _TokenBucket:
    """Thread-safe token bucket for a single key."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = int(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.last_refill)
        if elapsed <= 0:
            return
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, requested: int = 1) -> RateLimitDecision:
        with self._lock:
            now = time.monotonic()
            self._refill(now)

            if self.tokens >= requested:
                self.tokens -= requested
                return RateLimitDecision(
                    key="",
                    allowed=True,
                    requested=requested,
                    limit=self.capacity,
                    remaining=max(0.0, self.tokens),
                    retry_after=0.0,
                )

            if self.refill_rate <= 0:
                retry_after = math.inf
            else:
                retry_after = max(0.0, (requested - self.tokens) / self.refill_rate)

            return RateLimitDecision(
                key="",
                allowed=False,
                requested=requested,
                limit=self.capacity,
                remaining=max(0.0, self.tokens),
                retry_after=retry_after,
            )


class InMemoryStore:
    """In-memory bucket store suitable for single-process deployments."""

    def __init__(self):
        self._buckets: Dict[str, _TokenBucket] = {}
        self._lock = threading.RLock()

    def get_or_create(self, key: str, capacity: int, refill_rate: float) -> _TokenBucket:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(capacity=capacity, refill_rate=refill_rate)
                self._buckets[key] = bucket
            return bucket

    def delete(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


class RedisStore:
    """Redis-backed token bucket store for distributed deployments."""

    _LUA_CONSUME = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local requested = tonumber(ARGV[3])
    local now = tonumber(ARGV[4])

    local data = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(data[1])
    local last_refill = tonumber(data[2])

    if tokens == nil then
        tokens = capacity
        last_refill = now
    else
        local elapsed = math.max(0, now - last_refill)
        tokens = math.min(capacity, tokens + elapsed * refill_rate)
        last_refill = now
    end

    local allowed = 0
    local retry_after = 0
    if tokens >= requested then
        tokens = tokens - requested
        allowed = 1
    else
        if refill_rate > 0 then
            retry_after = math.max(0, (requested - tokens) / refill_rate)
        else
            retry_after = -1
        end
    end

    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)

    local safe_refill_rate = refill_rate
    if safe_refill_rate <= 0 then
        safe_refill_rate = 0.000001
    end
    local ttl_seconds = math.max(1, math.ceil((capacity / safe_refill_rate) * 2))
    redis.call('EXPIRE', key, ttl_seconds)

    return {allowed, tostring(tokens), tostring(retry_after)}
    """

    def __init__(self, redis_client: Any, key_prefix: str = "ratelimit:"):
        if redis_client is None:
            raise RateLimitConfigurationError("redis_client must be provided for Redis backend")
        self.redis = redis_client
        self.key_prefix = str(key_prefix or "ratelimit:")
        self._script = self.redis.register_script(self._LUA_CONSUME)

    def get_or_create(self, key: str, capacity: int, refill_rate: float) -> "_RedisTokenBucket":
        return _RedisTokenBucket(
            redis_client=self.redis,
            script=self._script,
            key=f"{self.key_prefix}{key}",
            capacity=capacity,
            refill_rate=refill_rate,
        )

    def delete(self, key: str) -> None:
        try:
            self.redis.delete(f"{self.key_prefix}{key}")
        except Exception as exc:  # pragma: no cover - depends on redis client implementation
            raise RateLimitError(f"Failed to delete rate-limit bucket for key '{key}': {exc}") from exc


class _RedisTokenBucket:
    """Redis-backed token bucket using a Lua script for atomic updates."""

    def __init__(self, redis_client: Any, script: Any, key: str, capacity: int, refill_rate: float):
        self.redis = redis_client
        self._script = script
        self.key = key
        self.capacity = int(capacity)
        self.refill_rate = float(refill_rate)

    def consume(self, requested: int = 1) -> RateLimitDecision:
        now = time.time()
        try:
            result = self._script(
                keys=[self.key],
                args=[self.capacity, self.refill_rate, requested, now],
            )
        except Exception as exc:  # pragma: no cover - depends on redis client implementation
            raise RateLimitError(f"Redis rate-limit evaluation failed for key '{self.key}': {exc}") from exc

        allowed_raw, remaining_raw, retry_after_raw = result
        retry_after = float(retry_after_raw)
        if retry_after < 0:
            retry_after = math.inf

        return RateLimitDecision(
            key=self.key,
            allowed=bool(int(allowed_raw)),
            requested=requested,
            limit=self.capacity,
            remaining=max(0.0, float(remaining_raw)),
            retry_after=retry_after,
        )


class RateLimiter:
    """Token-bucket rate limiter using either memory or Redis as the backing store."""

    def __init__(self, capacity: int, refill_rate: float, store: Any):
        self.capacity = self._validate_capacity(capacity)
        self.refill_rate = self._validate_refill_rate(refill_rate)
        self._store = self._validate_store(store)

    @classmethod
    def from_config(cls, backend: str = "memory", redis_client: Any = None) -> "RateLimiter":
        """Create a ``RateLimiter`` from the ``rate_limiter`` config section."""
        try:
            raw_config = get_config_section("rate_limiter") or {}
        except Exception as exc:
            logger.warning(f"Unable to load 'rate_limiter' config section, using defaults: {exc}")
            raw_config = {}

        if not isinstance(raw_config, dict):
            raise RateLimitConfigurationError("Config section 'rate_limiter' must be a mapping")

        capacity = raw_config.get("capacity", 100)
        refill_rate = raw_config.get("refill_rate", 10.0)
        key_prefix = raw_config.get("redis_key_prefix", "ratelimit:")
        normalized_backend = str(backend).strip().lower()

        if normalized_backend == "memory":
            store = InMemoryStore()
        elif normalized_backend == "redis":
            store = RedisStore(redis_client=redis_client, key_prefix=key_prefix)
        else:
            raise RateLimitConfigurationError(f"Unknown rate limiter backend: {backend}")

        return cls(capacity=capacity, refill_rate=refill_rate, store=store)

    def evaluate(self, key: str, tokens: int = 1) -> RateLimitDecision:
        """Evaluate whether ``tokens`` can be consumed for ``key``."""
        normalized_key = self._validate_key(key)
        requested = self._validate_requested_tokens(tokens, capacity=self.capacity)
        bucket = self._store.get_or_create(normalized_key, self.capacity, self.refill_rate)
        decision = bucket.consume(requested)
        return RateLimitDecision(
            key=normalized_key,
            allowed=decision.allowed,
            requested=requested,
            limit=self.capacity,
            remaining=decision.remaining,
            retry_after=decision.retry_after,
        )

    def allow(self, key: str, tokens: int = 1) -> bool:
        """Return ``True`` when the request is allowed, otherwise ``False``."""
        return self.evaluate(key, tokens=tokens).allowed

    def check(self, key: str, tokens: int = 1) -> None:
        """Raise ``RateLimitExceeded`` when the request exceeds the configured limit."""
        decision = self.evaluate(key, tokens=tokens)
        if not decision.allowed:
            raise RateLimitExceeded(decision.key, retry_after=decision.retry_after)

    def reset(self, key: str) -> None:
        """Reset the bucket for ``key`` when the backend supports deletion."""
        normalized_key = self._validate_key(key)
        delete = getattr(self._store, "delete", None)
        if delete is None:
            raise RateLimitError("Configured rate-limit store does not support reset")
        delete(normalized_key)

    @staticmethod
    def _validate_capacity(capacity: int) -> int:
        if isinstance(capacity, bool) or int(capacity) <= 0:
            raise RateLimitConfigurationError("capacity must be a positive integer")
        return int(capacity)

    @staticmethod
    def _validate_refill_rate(refill_rate: float) -> float:
        value = float(refill_rate)
        if value <= 0:
            raise RateLimitConfigurationError("refill_rate must be greater than 0")
        return value

    @staticmethod
    def _validate_store(store: Any) -> Any:
        if store is None or not hasattr(store, "get_or_create"):
            raise RateLimitConfigurationError(
                "store must provide get_or_create(key, capacity, refill_rate)"
            )
        return store

    @staticmethod
    def _validate_key(key: str) -> str:
        normalized = str(key).strip()
        if not normalized:
            raise ValueError("key must be a non-empty string")
        return normalized

    @staticmethod
    def _validate_requested_tokens(tokens: int, capacity: int) -> int:
        if isinstance(tokens, bool):
            raise ValueError("tokens must be a positive integer")
        requested = int(tokens)
        if requested <= 0:
            raise ValueError("tokens must be a positive integer")
        if requested > capacity:
            raise ValueError("tokens requested cannot exceed bucket capacity")
        return requested

    def __repr__(self) -> str:
        return (
            f"RateLimiter(capacity={self.capacity}, "
            f"refill_rate={self.refill_rate}, store={self._store.__class__.__name__})"
        )


__all__ = [
    "RateLimiter",
    "RateLimitDecision",
    "InMemoryStore",
    "RedisStore",
]

if __name__ == "__main__":
    print("\n=== Running RateLimiter smoke test ===\n")
    printer.status("TEST", "Starting RateLimiter tests", "info")

    limiter = RateLimiter(capacity=3, refill_rate=2.0, store=InMemoryStore())
    key = "demo-user"

    first = limiter.evaluate(key)
    second = limiter.evaluate(key)
    third = limiter.evaluate(key)
    fourth = limiter.evaluate(key)

    print("Limiter:", limiter)
    print("Attempt 1:", first)
    print("Attempt 2:", second)
    print("Attempt 3:", third)
    print("Attempt 4:", fourth)

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is True
    assert fourth.allowed is False
    assert fourth.retry_after > 0

    time.sleep(0.6)
    recovered = limiter.evaluate(key)
    print("After refill:", recovered)
    assert recovered.allowed is True

    limiter.reset(key)
    reset_check = limiter.evaluate(key, tokens=3)
    print("After reset:", reset_check)
    assert reset_check.allowed is True

    try:
        limiter.check(key, tokens=4)
    except ValueError as exc:
        print("Expected validation error:", exc)

    print("\n=== RateLimiter smoke test passed ===\n")