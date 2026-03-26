"""Rate limiter with token bucket algorithm and optional Redis backend.

This module provides a thread‑safe rate limiter that can be used to control
request rates per user, IP, or any custom key. For multi‑instance deployments,
a Redis backend is recommended.
"""

from __future__ import annotations

import time
import threading

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

from .utils.config_loader import get_config_section
from .utils.functions_error import RateLimitError
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Rate Limiter")
printer = PrettyPrinter

# In‑memory backend (single instance)
class _TokenBucket:
    """Token bucket for a single key."""
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate    # tokens per second
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class InMemoryStore:
    """In‑memory token bucket store (for single‑process deployments)."""
    def __init__(self):
        self._buckets: Dict[str, _TokenBucket] = {}
        self._lock = threading.RLock()

    def get_or_create(self, key: str, capacity: int, refill_rate: float) -> _TokenBucket:
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = _TokenBucket(capacity, refill_rate)
            return self._buckets[key]

# Optional Redis backend (if redis is installed)
class RedisStore:
    """Redis token bucket store for distributed deployments."""
    def __init__(self, redis_client, key_prefix: str = "ratelimit:"):
        self.redis = redis_client
        self.key_prefix = key_prefix

    def get_or_create(self, key: str, capacity: int, refill_rate: float) -> "_RedisTokenBucket":
        return _RedisTokenBucket(self.redis, self.key_prefix + key, capacity, refill_rate)

class _RedisTokenBucket:
    """Redis‑backed token bucket using Lua script for atomicity."""
    def __init__(self, redis_client, key: str, capacity: int, refill_rate: float):
        self.redis = redis_client
        self.key = key
        self.capacity = capacity
        self.refill_rate = refill_rate   # tokens per second

        # Lua script to consume tokens atomically
        self._lua_consume = """
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
            local elapsed = now - last_refill
            tokens = math.min(capacity, tokens + elapsed * refill_rate)
            last_refill = now
        end

        local allowed = tokens >= requested
        if allowed then
            tokens = tokens - requested
        end

        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
        redis.call('EXPIRE', key, 3600)   -- keep bucket for 1 hour
        return allowed and 1 or 0
        """

        self._script = self.redis.register_script(self._lua_consume)

    def consume(self, tokens: int = 1) -> bool:
        now = time.time()
        result = self._script(keys=[self.key], args=[self.capacity, self.refill_rate, tokens, now])
        return bool(result)


class RateLimiter:
    """
    Rate limiter that can be used with in‑memory or Redis backend.

    Usage:
        limiter = RateLimiter.from_config(backend="memory")
        if limiter.allow("user:123", tokens=1):
            # process request
        else:
            raise RateLimitError("Rate limit exceeded")
    """
    def __init__(self, capacity: int, refill_rate: float, store):
        """
        Args:
            capacity: Maximum number of tokens the bucket can hold.
            refill_rate: Tokens added per second.
            store: A store that provides `get_or_create(key, capacity, refill_rate)`.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._store = store

    @classmethod
    def from_config(cls, backend: str = "memory", redis_client=None):
        """
        Create a RateLimiter from configuration (section 'rate_limiter').
        """
        config = get_config_section('rate_limiter')
        capacity = config.get('capacity', 100)
        refill_rate = config.get('refill_rate', 10)   # tokens per second

        if backend == "memory":
            store = InMemoryStore()
        elif backend == "redis":
            if redis_client is None:
                raise ValueError("redis_client must be provided for Redis backend")
            store = RedisStore(redis_client, key_prefix=config.get('redis_key_prefix', 'ratelimit:'))
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return cls(capacity, refill_rate, store)

    def allow(self, key: str, tokens: int = 1) -> bool:
        """Return True if the request is allowed, False otherwise."""
        bucket = self._store.get_or_create(key, self.capacity, self.refill_rate)
        return bucket.consume(tokens)

    def check(self, key: str, tokens: int = 1) -> None:
        """Raise RateLimitError if limit is exceeded."""
        if not self.allow(key, tokens):
            raise RateLimitError(f"Rate limit exceeded for key: {key}")