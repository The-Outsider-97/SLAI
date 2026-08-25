"""
Thread-safe bounded LRU cache for reasoning types.

This module is designed to sit alongside ``ReasoningMemory``:
- ``ReasoningMemory`` keeps durable/replay-oriented reasoning experiences.
- ``ReasoningCache`` keeps short-lived, fast lookup state for reasoning type
  factories, combined reasoning plans, expensive intermediate results, and
  strategy-specific runtime artifacts.

Design goals:
- Preserve the reasoning subsystem config flow:
  ``load_global_config()`` + ``get_config_section("reasoning_cache")``.
- Reuse shared reasoning errors and helpers instead of duplicating validation,
  timestamps, bounded integer normalization, or JSON-safe state conversion.
- Provide lifecycle metadata per cache entry.
- Maintain runtime counters for cache behavior.
- Remain thread-safe with a bounded OrderedDict-backed LRU implementation.
- Support optional TTL without forcing TTL on every entry.
- Integrate optionally with ``ReasoningMemory`` without constructing memory by
  default or creating a hard runtime dependency cycle.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import random
import time

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Cache")
printer = PrettyPrinter()

CacheKey = Tuple[str, str]
CacheFactory = Callable[[], Any]
_MISSING = object()


@dataclass
class CacheEntry:
    """A single cached value with lifecycle metadata and access counters.

    ``cache_key`` is an internal stable key ``(namespace, digest)``.  ``key_repr``
    stores a bounded readable representation of the original key for diagnostics.
    The original key is intentionally not required to be stored because callers may
    pass unhashable, very large, or unserializable objects.
    """

    cache_key: CacheKey
    key_repr: str
    value: Any
    namespace: str
    created_at_ms: int
    last_accessed_at_ms: int
    last_updated_at_ms: int
    expires_at_ms: Optional[int] = None
    ttl_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    hit_count: int = 0
    write_count: int = 1
    refresh_count: int = 0
    invalidated: bool = False
    invalidated_at_ms: Optional[int] = None
    invalidation_reason: Optional[str] = None

    def is_expired(self, now_ms: Optional[int] = None) -> bool:
        """Return True when the entry has passed its expiration timestamp."""
        if self.expires_at_ms is None:
            return False
        current = monotonic_timestamp_ms() if now_ms is None else int(now_ms)
        return current >= int(self.expires_at_ms)

    def is_active(self, now_ms: Optional[int] = None) -> bool:
        """Return True when the entry is neither invalidated nor expired."""
        return not self.invalidated and not self.is_expired(now_ms)

    def touch(self, now_ms: Optional[int] = None, *, hit: bool = True) -> None:
        """Record an access on the entry."""
        current = monotonic_timestamp_ms() if now_ms is None else int(now_ms)
        self.last_accessed_at_ms = current
        self.access_count += 1
        if hit:
            self.hit_count += 1

    def mark_updated(self, now_ms: Optional[int] = None) -> None:
        """Record a value refresh/update."""
        current = monotonic_timestamp_ms() if now_ms is None else int(now_ms)
        self.last_updated_at_ms = current
        self.write_count += 1
        self.refresh_count += 1
        self.invalidated = False
        self.invalidated_at_ms = None
        self.invalidation_reason = None

    def invalidate(self, *, reason: str = "manual", now_ms: Optional[int] = None) -> None:
        """Mark the entry as invalidated without deleting it."""
        current = monotonic_timestamp_ms() if now_ms is None else int(now_ms)
        self.invalidated = True
        self.invalidated_at_ms = current
        self.invalidation_reason = reason

    def age_seconds(self, now_ms: Optional[int] = None) -> float:
        """Return entry age in seconds."""
        current = monotonic_timestamp_ms() if now_ms is None else int(now_ms)
        return max(0.0, (current - self.created_at_ms) / 1000.0)

    def idle_seconds(self, now_ms: Optional[int] = None) -> float:
        """Return seconds since last access."""
        current = monotonic_timestamp_ms() if now_ms is None else int(now_ms)
        return max(0.0, (current - self.last_accessed_at_ms) / 1000.0)

    def remaining_ttl_seconds(self, now_ms: Optional[int] = None) -> Optional[float]:
        """Return remaining TTL in seconds, or None for non-expiring entries."""
        if self.expires_at_ms is None:
            return None
        current = monotonic_timestamp_ms() if now_ms is None else int(now_ms)
        return max(0.0, (self.expires_at_ms - current) / 1000.0)

    def to_metadata(self, *, now_ms: Optional[int] = None) -> Dict[str, Any]:
        """Return JSON-safe metadata without exposing the cached value."""
        current = monotonic_timestamp_ms() if now_ms is None else int(now_ms)
        return json_safe_reasoning_state(
            {
                "namespace": self.namespace,
                "cache_key": list(self.cache_key),
                "key_repr": self.key_repr,
                "created_at_ms": self.created_at_ms,
                "last_accessed_at_ms": self.last_accessed_at_ms,
                "last_updated_at_ms": self.last_updated_at_ms,
                "expires_at_ms": self.expires_at_ms,
                "ttl_seconds": self.ttl_seconds,
                "age_seconds": self.age_seconds(current),
                "idle_seconds": self.idle_seconds(current),
                "remaining_ttl_seconds": self.remaining_ttl_seconds(current),
                "access_count": self.access_count,
                "hit_count": self.hit_count,
                "write_count": self.write_count,
                "refresh_count": self.refresh_count,
                "invalidated": self.invalidated,
                "invalidated_at_ms": self.invalidated_at_ms,
                "invalidation_reason": self.invalidation_reason,
                "expired": self.is_expired(current),
                "active": self.is_active(current),
                "metadata": dict(self.metadata),
            }
        )


@dataclass
class CacheCounters:
    """Runtime counters describing cache behavior."""

    gets: int = 0
    sets: int = 0
    refreshes: int = 0
    hits: int = 0
    misses: int = 0
    expired: int = 0
    invalidated_hits: int = 0
    evictions: int = 0
    deletes: int = 0
    invalidations: int = 0
    namespace_invalidations: int = 0
    clears: int = 0
    prunes: int = 0
    contains_checks: int = 0
    get_or_set_calls: int = 0
    factory_calls: int = 0
    factory_errors: int = 0
    memory_events: int = 0
    high_watermark: int = 0
    started_at_ms: int = field(default_factory=monotonic_timestamp_ms)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def miss_rate(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total else 0.0

    def to_dict(self, *, current_size: int, capacity: int) -> Dict[str, Any]:
        now = monotonic_timestamp_ms()
        return {
            "gets": self.gets,
            "sets": self.sets,
            "refreshes": self.refreshes,
            "hits": self.hits,
            "misses": self.misses,
            "expired": self.expired,
            "invalidated_hits": self.invalidated_hits,
            "evictions": self.evictions,
            "deletes": self.deletes,
            "invalidations": self.invalidations,
            "namespace_invalidations": self.namespace_invalidations,
            "clears": self.clears,
            "prunes": self.prunes,
            "contains_checks": self.contains_checks,
            "get_or_set_calls": self.get_or_set_calls,
            "factory_calls": self.factory_calls,
            "factory_errors": self.factory_errors,
            "memory_events": self.memory_events,
            "hit_rate": self.hit_rate(),
            "miss_rate": self.miss_rate(),
            "current_size": current_size,
            "capacity": capacity,
            "saturation": current_size / max(1, capacity),
            "high_watermark": self.high_watermark,
            "started_at_ms": self.started_at_ms,
            "uptime_seconds": max(0.0, (now - self.started_at_ms) / 1000.0),
        }


class ReasoningCache:
    """Thread-safe bounded LRU cache with optional TTL and rich metadata.

    The cache is intentionally generic so it can be used by ``ReasoningTypes``
    for strategy instances, combined-reasoning plans, discovery results, or by
    individual reasoning types for expensive intermediate products.

    Existing subsystem conventions are preserved:
    - configuration is loaded from ``reasoning_config.yaml``;
    - errors are emitted through reasoning subsystem exception classes;
    - helper utilities perform bounded iteration and timestamp normalization.
    """

    MODULE_VERSION = "2.1.0"

    def __init__(self, *,
        namespace: Optional[str] = None,
        max_size: Optional[int] = None,
        default_ttl_seconds: Optional[float] = None,
        memory: Optional[Any] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.cache_config: Dict[str, Any] = get_config_section("reasoning_cache")
        self._refresh_runtime_config(
            namespace=namespace,
            max_size=max_size,
            default_ttl_seconds=default_ttl_seconds,
        )

        self.lock = RLock()
        self._entries: "OrderedDict[CacheKey, CacheEntry]" = OrderedDict()
        self._counters = CacheCounters()
        self._memory = memory
        self._last_cleanup_at_ms = monotonic_timestamp_ms()
        self._rng = random.Random(self.seed) if self.seed is not None else random.Random()

        logger.info(
            "ReasoningCache initialized | namespace=%s | max_size=%s | ttl=%s | ttl_enabled=%s",
            self.namespace,
            self.max_size,
            self.default_ttl_seconds,
            self.ttl_enabled,
        )
        printer.status("INIT", f"Reasoning Cache initialized with capacity={self.max_size}", "success")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _refresh_runtime_config(self, *,
        namespace: Optional[str],
        max_size: Optional[int],
        default_ttl_seconds: Optional[float]) -> None:
        """Validate and cache frequently used config values."""
        self.namespace = self._normalize_namespace(
            namespace or self.cache_config.get("namespace", "reasoning_types")
        )
        self.max_size = bounded_iterations(
            max_size if max_size is not None else self.cache_config.get("max_size", 1024),
            minimum=1,
            maximum=10_000_000,
        )
        self.ttl_enabled = bool(self.cache_config.get("ttl_enabled", True))
        self.default_ttl_seconds = self._optional_float(
            default_ttl_seconds if default_ttl_seconds is not None else self.cache_config.get("default_ttl_seconds", 300.0),
            key="default_ttl_seconds",
            allow_none=True,
            minimum=0.0,
        )
        self.cleanup_interval_seconds = self._optional_float(
            self.cache_config.get("cleanup_interval_seconds", 30.0),
            key="cleanup_interval_seconds",
            allow_none=False,
            minimum=0.0,
        ) or 0.0
        self.ttl_jitter_seconds = self._optional_float(
            self.cache_config.get("ttl_jitter_seconds", 0.0),
            key="ttl_jitter_seconds",
            allow_none=False,
            minimum=0.0,
        ) or 0.0
        self.evict_expired_before_lru = bool(self.cache_config.get("evict_expired_before_lru", True))
        self.prune_expired_on_write = bool(self.cache_config.get("prune_expired_on_write", True))
        self.delete_expired_on_read = bool(self.cache_config.get("delete_expired_on_read", True))
        self.cache_none_values = bool(self.cache_config.get("cache_none_values", True))
        self.copy_on_read = bool(self.cache_config.get("copy_on_read", False))
        self.copy_on_write = bool(self.cache_config.get("copy_on_write", False))
        self.track_key_reprs = bool(self.cache_config.get("track_key_reprs", True))
        self.record_memory_events = bool(self.cache_config.get("record_memory_events", False))
        self.memory_event_tag = str(self.cache_config.get("memory_event_tag", "reasoning_cache")).strip() or "reasoning_cache"
        self.memory_event_priority = self._optional_float(
            self.cache_config.get("memory_event_priority", 0.3),
            key="memory_event_priority",
            allow_none=False,
            minimum=0.0,
        ) or 0.0
        self.max_key_repr_length = bounded_iterations(
            self.cache_config.get("max_key_repr_length", 256),
            minimum=16,
            maximum=100_000,
        )
        self.max_export_entries = bounded_iterations(
            self.cache_config.get("max_export_entries", 256),
            minimum=1,
            maximum=1_000_000,
        )
        self.strict_keys = bool(self.cache_config.get("strict_keys", False))
        self.seed = self._optional_int(self.cache_config.get("seed"))
        self.hash_algorithm = str(self.cache_config.get("hash_algorithm", "sha256")).strip().lower()
        if self.hash_algorithm not in hashlib.algorithms_available:
            raise ReasoningConfigurationError(
                "reasoning_cache.hash_algorithm is not available",
                context={"hash_algorithm": self.hash_algorithm},
            )
        if self.ttl_enabled and self.default_ttl_seconds is not None and self.default_ttl_seconds == 0:
            logger.warning("reasoning_cache.default_ttl_seconds=0 means entries expire immediately")

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                "reasoning_cache.seed must be an integer or None",
                cause=exc,
                context={"seed": value},
            ) from exc

    @staticmethod
    def _optional_float(value: Any, *, key: str, allow_none: bool,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None) -> Optional[float]:
        if value is None:
            if allow_none:
                return None
            raise ReasoningConfigurationError(f"reasoning_cache.{key} cannot be None")
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            if allow_none:
                return None
            raise ReasoningConfigurationError(f"reasoning_cache.{key} cannot be None")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                f"reasoning_cache.{key} must be numeric",
                cause=exc,
                context={"key": key, "value": value},
            ) from exc
        if not math.isfinite(parsed):
            raise ReasoningConfigurationError(
                f"reasoning_cache.{key} must be finite",
                context={"key": key, "value": value},
            )
        if minimum is not None and parsed < minimum:
            raise ReasoningConfigurationError(
                f"reasoning_cache.{key} must be >= {minimum}",
                context={"key": key, "value": parsed, "minimum": minimum},
            )
        if maximum is not None and parsed > maximum:
            raise ReasoningConfigurationError(
                f"reasoning_cache.{key} must be <= {maximum}",
                context={"key": key, "value": parsed, "maximum": maximum},
            )
        return parsed

    @staticmethod
    def _normalize_namespace(namespace: Any) -> str:
        text = str(namespace).strip()
        if not text:
            raise ReasoningValidationError("Cache namespace cannot be empty")
        return text

    # ------------------------------------------------------------------
    # Key normalization and value copying
    # ------------------------------------------------------------------
    def _make_cache_key(self, key: Any, namespace: Optional[str] = None) -> CacheKey:
        ns = self._normalize_namespace(namespace or self.namespace)
        key_payload = self._stable_key_payload(key)
        digest = hashlib.new(self.hash_algorithm, key_payload.encode("utf-8")).hexdigest()
        return ns, digest

    def _stable_key_payload(self, key: Any) -> str:
        if self.strict_keys:
            if not isinstance(key, (str, int, float, bool, tuple, frozenset)):
                raise ReasoningValidationError(
                    "Strict cache keys must be primitive/hash-stable values",
                    context={"key_type": type(key).__name__},
                )
        try:
            return json.dumps(key, sort_keys=True, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return repr(key)

    def _key_repr(self, key: Any) -> str:
        if not self.track_key_reprs:
            return "<hidden>"
        payload = self._stable_key_payload(key)
        if len(payload) <= self.max_key_repr_length:
            return payload
        return f"{payload[: self.max_key_repr_length - 16]}...<truncated>"

    @staticmethod
    def _safe_copy(value: Any, *, enabled: bool) -> Any:
        if not enabled:
            return value
        try:
            return copy.deepcopy(value)
        except Exception:
            logger.debug("Cache value could not be deep-copied; returning original reference")
            return value

    def _resolve_ttl_seconds(self, ttl_seconds: Optional[float]) -> Optional[float]:
        if not self.ttl_enabled:
            return None
        ttl = self.default_ttl_seconds if ttl_seconds is None else self._optional_float(
            ttl_seconds,
            key="ttl_seconds",
            allow_none=True,
            minimum=0.0,
        )
        if ttl is None:
            return None
        jitter = self._rng.uniform(0.0, self.ttl_jitter_seconds) if self.ttl_jitter_seconds > 0 else 0.0
        return max(0.0, float(ttl) + jitter)

    @staticmethod
    def _expires_at_ms(now_ms: int, ttl_seconds: Optional[float]) -> Optional[int]:
        if ttl_seconds is None:
            return None
        return int(now_ms + ttl_seconds * 1000.0)

    # ------------------------------------------------------------------
    # Core cache API
    # ------------------------------------------------------------------
    def attach_memory(self, memory: Optional[Any]) -> None:
        """Attach or detach a memory object used for optional event recording."""
        with self.lock:
            self._memory = memory

    def set(self, key: Any, value: Any, *, ttl_seconds: Optional[float] = None,
            namespace: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> CacheEntry:
        """Insert or refresh a cache entry and return its metadata object."""
        if value is None and not self.cache_none_values:
            raise ReasoningValidationError("None values are disabled for this cache")
        with self.lock:
            self._maybe_prune_expired_locked()
            now_ms = monotonic_timestamp_ms()
            cache_key = self._make_cache_key(key, namespace)
            resolved_ttl = self._resolve_ttl_seconds(ttl_seconds)
            expires_at = self._expires_at_ms(now_ms, resolved_ttl)
            stored_value = self._safe_copy(value, enabled=self.copy_on_write)
            entry_metadata = dict(metadata or {})

            if cache_key in self._entries:
                entry = self._entries[cache_key]
                entry.value = stored_value
                entry.ttl_seconds = resolved_ttl
                entry.expires_at_ms = expires_at
                entry.metadata = entry_metadata
                entry.mark_updated(now_ms)
                self._entries.move_to_end(cache_key)
                self._counters.refreshes += 1
                action = "refresh"
            else:
                entry = CacheEntry(
                    cache_key=cache_key,
                    key_repr=self._key_repr(key),
                    value=stored_value,
                    namespace=cache_key[0],
                    created_at_ms=now_ms,
                    last_accessed_at_ms=now_ms,
                    last_updated_at_ms=now_ms,
                    expires_at_ms=expires_at,
                    ttl_seconds=resolved_ttl,
                    metadata=entry_metadata,
                )
                self._entries[cache_key] = entry
                action = "set"

            self._counters.sets += 1
            self._ensure_capacity_locked()
            self._counters.high_watermark = max(self._counters.high_watermark, len(self._entries))
            self._record_event(action, entry)
            return entry

    put = set

    def get(self, key: Any, default: Any = None, *, namespace: Optional[str] = None,
            include_entry: bool = False, touch: bool = True) -> Any:
        """Return a cached value, an entry when requested, or ``default`` on miss."""
        with self.lock:
            self._counters.gets += 1
            cache_key = self._make_cache_key(key, namespace)
            entry = self._entries.get(cache_key)
            if entry is None:
                self._counters.misses += 1
                return default

            now_ms = monotonic_timestamp_ms()
            if entry.invalidated:
                self._counters.invalidated_hits += 1
                self._counters.misses += 1
                return default

            if entry.is_expired(now_ms):
                self._counters.expired += 1
                self._counters.misses += 1
                if self.delete_expired_on_read:
                    self._delete_key_locked(cache_key, reason="expired_on_read")
                return default

            if touch:
                entry.touch(now_ms)
                self._entries.move_to_end(cache_key)
            self._counters.hits += 1
            self._record_event("hit", entry)
            if include_entry:
                return entry
            return self._safe_copy(entry.value, enabled=self.copy_on_read)

    def get_entry(self, key: Any, *, namespace: Optional[str] = None, touch: bool = True) -> Optional[CacheEntry]:
        """Return the active cache entry for a key, or None."""
        value = self.get(key, default=_MISSING, namespace=namespace, include_entry=True, touch=touch)
        return None if value is _MISSING else value

    def get_or_set(self, key: Any, factory: CacheFactory, *, ttl_seconds: Optional[float] = None,
                   namespace: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> Any:
        """Return a cached value or compute/store it through ``factory``."""
        if not callable(factory):
            raise ReasoningValidationError("factory must be callable", context={"factory_type": type(factory).__name__})
        self._counters.get_or_set_calls += 1
        cached = self.get(key, default=_MISSING, namespace=namespace)
        if cached is not _MISSING:
            return cached
        try:
            self._counters.factory_calls += 1
            value = factory()
        except ReasoningError:
            self._counters.factory_errors += 1
            raise
        except Exception as exc:
            self._counters.factory_errors += 1
            raise MemoryOperationError(
                "Cache factory execution failed",
                cause=exc,
                context={"key": self._key_repr(key), "namespace": namespace or self.namespace},
            ) from exc
        self.set(key, value, ttl_seconds=ttl_seconds, namespace=namespace, metadata=metadata)
        return self._safe_copy(value, enabled=self.copy_on_read)

    def contains(self, key: Any, *, namespace: Optional[str] = None) -> bool:
        """Return True if an active entry exists for key."""
        with self.lock:
            self._counters.contains_checks += 1
            cache_key = self._make_cache_key(key, namespace)
            entry = self._entries.get(cache_key)
            if entry is None or entry.invalidated:
                return False
            if entry.is_expired():
                if self.delete_expired_on_read:
                    self._delete_key_locked(cache_key, reason="expired_contains")
                return False
            return True

    def delete(self, key: Any, *, namespace: Optional[str] = None) -> bool:
        """Delete a cache entry by key."""
        with self.lock:
            cache_key = self._make_cache_key(key, namespace)
            return self._delete_key_locked(cache_key, reason="delete")

    def invalidate(self, key: Any, *, namespace: Optional[str] = None, reason: str = "manual") -> bool:
        """Mark a cache entry invalid while keeping metadata available."""
        with self.lock:
            cache_key = self._make_cache_key(key, namespace)
            entry = self._entries.get(cache_key)
            if entry is None:
                return False
            entry.invalidate(reason=reason)
            self._entries.move_to_end(cache_key)
            self._counters.invalidations += 1
            self._record_event("invalidate", entry)
            return True

    def invalidate_namespace(self, namespace: str, *, reason: str = "namespace") -> int:
        """Invalidate all entries in a namespace and return affected count."""
        ns = self._normalize_namespace(namespace)
        with self.lock:
            affected = 0
            now_ms = monotonic_timestamp_ms()
            for cache_key, entry in list(self._entries.items()):
                if cache_key[0] == ns and not entry.invalidated:
                    entry.invalidate(reason=reason, now_ms=now_ms)
                    affected += 1
            if affected:
                self._counters.namespace_invalidations += 1
                self._counters.invalidations += affected
            return affected

    def clear(self, *, namespace: Optional[str] = None) -> int:
        """Clear all entries, or only entries within a namespace."""
        with self.lock:
            if namespace is None:
                removed = len(self._entries)
                self._entries.clear()
            else:
                ns = self._normalize_namespace(namespace)
                keys = [key for key in self._entries if key[0] == ns]
                for key in keys:
                    self._entries.pop(key, None)
                removed = len(keys)
            self._counters.clears += 1
            self._counters.deletes += removed
            return removed

    # ------------------------------------------------------------------
    # LRU / TTL maintenance
    # ------------------------------------------------------------------
    def prune_expired(self) -> int:
        """Remove expired entries and return number pruned."""
        with self.lock:
            return self._prune_expired_locked()

    def _maybe_prune_expired_locked(self) -> None:
        if not self.prune_expired_on_write:
            return
        if self.cleanup_interval_seconds <= 0:
            self._prune_expired_locked()
            return
        now_ms = monotonic_timestamp_ms()
        if (now_ms - self._last_cleanup_at_ms) / 1000.0 >= self.cleanup_interval_seconds:
            self._prune_expired_locked(now_ms=now_ms)
            self._last_cleanup_at_ms = now_ms

    def _prune_expired_locked(self, now_ms: Optional[int] = None) -> int:
        current = monotonic_timestamp_ms() if now_ms is None else int(now_ms)
        expired_keys = [key for key, entry in self._entries.items() if entry.is_expired(current)]
        for key in expired_keys:
            self._entries.pop(key, None)
        if expired_keys:
            self._counters.prunes += 1
            self._counters.expired += len(expired_keys)
            self._counters.deletes += len(expired_keys)
        return len(expired_keys)

    def _ensure_capacity_locked(self) -> None:
        if len(self._entries) <= self.max_size:
            return
        if self.evict_expired_before_lru:
            self._prune_expired_locked()
        while len(self._entries) > self.max_size:
            _, evicted = self._entries.popitem(last=False)
            evicted.invalidate(reason="lru_evicted")
            self._counters.evictions += 1
            self._record_event("evict", evicted)

    def _delete_key_locked(self, cache_key: CacheKey, *, reason: str) -> bool:
        entry = self._entries.pop(cache_key, None)
        if entry is None:
            return False
        entry.invalidate(reason=reason)
        self._counters.deletes += 1
        self._record_event(reason, entry)
        return True

    # ------------------------------------------------------------------
    # Introspection and export
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.size()

    def size(self, *, include_inactive: bool = True) -> int:
        with self.lock:
            if include_inactive:
                return len(self._entries)
            now_ms = monotonic_timestamp_ms()
            return sum(1 for entry in self._entries.values() if entry.is_active(now_ms))

    def keys(self, *, namespace: Optional[str] = None, active_only: bool = True) -> List[str]:
        """Return diagnostic key representations in LRU order."""
        with self.lock:
            now_ms = monotonic_timestamp_ms()
            ns = self._normalize_namespace(namespace) if namespace is not None else None
            result: List[str] = []
            for cache_key, entry in self._entries.items():
                if ns is not None and cache_key[0] != ns:
                    continue
                if active_only and not entry.is_active(now_ms):
                    continue
                result.append(entry.key_repr)
            return result

    def entries(self, *, namespace: Optional[str] = None, active_only: bool = True) -> List[CacheEntry]:
        """Return cache entries in LRU order."""
        with self.lock:
            now_ms = monotonic_timestamp_ms()
            ns = self._normalize_namespace(namespace) if namespace is not None else None
            result: List[CacheEntry] = []
            for cache_key, entry in self._entries.items():
                if ns is not None and cache_key[0] != ns:
                    continue
                if active_only and not entry.is_active(now_ms):
                    continue
                result.append(entry)
            return result

    def values(self, *, namespace: Optional[str] = None, active_only: bool = True) -> List[Any]:
        """Return active cached values in LRU order."""
        return [self._safe_copy(entry.value, enabled=self.copy_on_read) for entry in self.entries(namespace=namespace, active_only=active_only)]

    def items(self, *, namespace: Optional[str] = None, active_only: bool = True) -> List[Tuple[str, Any]]:
        """Return ``(key_repr, value)`` pairs in LRU order."""
        return [
            (entry.key_repr, self._safe_copy(entry.value, enabled=self.copy_on_read))
            for entry in self.entries(namespace=namespace, active_only=active_only)
        ]

    def metrics(self) -> Dict[str, Any]:
        """Return runtime cache counters and health metrics."""
        with self.lock:
            namespace_counts: Dict[str, int] = {}
            active_count = 0
            expired_count = 0
            invalidated_count = 0
            now_ms = monotonic_timestamp_ms()
            for entry in self._entries.values():
                namespace_counts[entry.namespace] = namespace_counts.get(entry.namespace, 0) + 1
                if entry.invalidated:
                    invalidated_count += 1
                elif entry.is_expired(now_ms):
                    expired_count += 1
                else:
                    active_count += 1
            payload = self._counters.to_dict(current_size=len(self._entries), capacity=self.max_size)
            payload.update(
                {
                    "active_entries": active_count,
                    "expired_entries": expired_count,
                    "invalidated_entries": invalidated_count,
                    "namespaces": namespace_counts,
                    "namespace": self.namespace,
                    "ttl_enabled": self.ttl_enabled,
                    "default_ttl_seconds": self.default_ttl_seconds,
                    "cleanup_interval_seconds": self.cleanup_interval_seconds,
                    "module_version": self.MODULE_VERSION,
                }
            )
            return json_safe_reasoning_state(payload)

    def diagnostics(self) -> Dict[str, Any]:
        """Return compact operational diagnostics for health checks."""
        with self.lock:
            now_ms = monotonic_timestamp_ms()
            oldest = next(iter(self._entries.values()), None)
            newest = next(reversed(self._entries.values()), None) if self._entries else None
            return json_safe_reasoning_state(
                {
                    "metrics": self.metrics(),
                    "oldest_entry": oldest.to_metadata(now_ms=now_ms) if oldest else None,
                    "newest_entry": newest.to_metadata(now_ms=now_ms) if newest else None,
                    "config": {
                        "max_size": self.max_size,
                        "ttl_enabled": self.ttl_enabled,
                        "default_ttl_seconds": self.default_ttl_seconds,
                        "delete_expired_on_read": self.delete_expired_on_read,
                        "evict_expired_before_lru": self.evict_expired_before_lru,
                        "copy_on_read": self.copy_on_read,
                        "copy_on_write": self.copy_on_write,
                        "record_memory_events": self.record_memory_events,
                    },
                }
            )

    def export_state(self, *, include_values: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
        """Export JSON-safe cache state for logging or debugging."""
        with self.lock:
            max_items = self.max_export_entries if limit is None else bounded_iterations(
                limit, minimum=1, maximum=max(1, len(self._entries))
            )
            now_ms = monotonic_timestamp_ms()
            exported: List[Dict[str, Any]] = []
            for entry in list(self._entries.values())[:max_items]:
                item = entry.to_metadata(now_ms=now_ms)
                if include_values:
                    try:
                        json.dumps(entry.value, default=str)
                        item["value"] = entry.value
                    except TypeError:
                        item["value_repr"] = repr(entry.value)
                exported.append(item)
            return json_safe_reasoning_state(
                {
                    "module_version": self.MODULE_VERSION,
                    "metrics": self.metrics(),
                    "entries": exported,
                    "truncated": len(self._entries) > len(exported),
                }
            )

    def reset_counters(self) -> None:
        """Reset runtime counters without clearing cache entries."""
        with self.lock:
            high_watermark = len(self._entries)
            self._counters = CacheCounters(high_watermark=high_watermark)

    # ------------------------------------------------------------------
    # Decorator support
    # ------------------------------------------------------------------
    def memoize(self, *, namespace: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        key_builder: Optional[Callable[..., Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that caches function results.

        ``key_builder`` receives the same ``*args`` and ``**kwargs`` as the
        wrapped function and should return a stable key.  Without it, the cache
        key is based on function module/name plus arguments.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if key_builder is not None:
                    key = key_builder(*args, **kwargs)
                else:
                    key = {
                        "module": getattr(func, "__module__", ""),
                        "qualname": getattr(func, "__qualname__", getattr(func, "__name__", "callable")),
                        "args": args,
                        "kwargs": kwargs,
                    }
                return self.get_or_set(
                    key,
                    lambda: func(*args, **kwargs),
                    ttl_seconds=ttl_seconds,
                    namespace=namespace,
                    metadata=metadata,
                )

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Memory integration
    # ------------------------------------------------------------------
    def _record_event(self, action: str, entry: CacheEntry) -> None:
        if not self.record_memory_events or self._memory is None:
            return
        payload = {
            "type": "reasoning_cache_event",
            "action": action,
            "namespace": entry.namespace,
            "key_repr": entry.key_repr,
            "created_at_ms": entry.created_at_ms,
            "last_accessed_at_ms": entry.last_accessed_at_ms,
            "expired": entry.is_expired(),
            "invalidated": entry.invalidated,
            "size": len(self._entries),
        }
        try:
            self._memory.add(payload, priority=self.memory_event_priority, tag=self.memory_event_tag)
            self._counters.memory_events += 1
        except Exception as exc:
            logger.warning("Failed to record reasoning cache event in memory: %s", exc)

    # ------------------------------------------------------------------
    # Iteration protocol
    # ------------------------------------------------------------------
    def __contains__(self, key: Any) -> bool:
        return self.contains(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __repr__(self) -> str:
        metrics = self.metrics()
        return (
            f"ReasoningCache(namespace={self.namespace!r}, size={metrics['current_size']}, "
            f"capacity={self.max_size}, hit_rate={metrics['hit_rate']:.3f})"
        )


if __name__ == "__main__":
    print("\n=== Running Reasoning Cache ===\n")
    printer.status("TEST", "Reasoning Cache initialized", "info")

    cache = ReasoningCache(namespace="reasoning_cache_test", max_size=3, default_ttl_seconds=0.2)
    cache.set("abduction", {"strategy": "abduction", "ready": True}, metadata={"component": "reasoning_types"})
    cache.set("deduction", {"strategy": "deduction", "ready": True})
    cache.set("induction", {"strategy": "induction", "ready": True})

    assert cache.contains("abduction") is True
    assert cache.get("abduction")["strategy"] == "abduction"

    cache.set("analogical", {"strategy": "analogical"})
    assert cache.contains("abduction") is True
    assert cache.contains("analogical") is True
    assert len(cache) == 3

    computed = cache.get_or_set("cause_effect", lambda: {"strategy": "cause_effect"}, ttl_seconds=None)
    assert computed["strategy"] == "cause_effect"
    assert len(cache) == 3

    cache.set("short_lived", "value", ttl_seconds=0.01)
    time.sleep(0.01)
    assert cache.get("short_lived", default="expired") == "expired"

    @cache.memoize(ttl_seconds=None)
    def combine(a: int, b: int) -> int:
        return a + b

    assert combine(2, 3) == 5
    assert combine(2, 3) == 5

    metrics = cache.metrics()
    assert metrics["hits"] >= 2
    assert metrics["misses"] >= 1
    assert metrics["capacity"] == 3
    assert cache.prune_expired() >= 0

    exported = cache.export_state()
    assert exported["module_version"] == ReasoningCache.MODULE_VERSION
    assert "entries" in exported

    print("\n=== Test ran successfully ===\n")
