"""Factory cache utilities.

Provides a production-ready in-memory cache for factory internals such as
registry lookups, dependency resolution results, adaptation artifacts, worker
payload normalisation, and orchestration-level memoisation.

The cache is intentionally lightweight and dependency-free while still offering
production-friendly behavior:

- bounded LRU storage;
- optional per-entry or default TTL;
- thread-safe mutation and reads;
- explicit cache statistics;
- structured error handling through ``factory_errors``;
- validation delegated to ``factory_helpers``;
- configuration loaded through ``factory_config.yaml`` via the existing
  factory config loader.

Design notes
------------
This module does not duplicate helper behavior. Shared validation, timing,
payload normalisation, and safe serialisation are imported from
``factory_helpers``. Error classes are imported from ``factory_errors``. Local
imports are not wrapped in ``try``/``except`` so integration failures surface
immediately during startup and tests.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Generic, Hashable, Iterable, Iterator, Mapping, Optional, Tuple, TypeVar, cast

from .utils.config_loader import get_config_section, load_global_config
from .utils.factory_errors import *
from .utils.factory_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Factory Cache")
printer = PrettyPrinter()

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
_MISSING = object()


@dataclass(slots=True)
class CacheEntry(Generic[V]):
    """Single cache entry with lifecycle metadata."""

    value: V
    created_at: float
    updated_at: float
    expires_at: Optional[float] = None
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Return ``True`` when the entry has exceeded its TTL."""
        if self.expires_at is None:
            return False
        return (now_epoch_seconds() if now is None else now) >= self.expires_at

    def ttl_remaining(self, now: Optional[float] = None) -> Optional[float]:
        """Return remaining TTL seconds, or ``None`` for non-expiring entries."""
        if self.expires_at is None:
            return None
        current = now_epoch_seconds() if now is None else now
        return max(0.0, self.expires_at - current)

    def touch(self) -> None:
        """Update hit and access metadata for successful reads."""
        self.hits += 1
        self.updated_at = now_epoch_seconds()

    def to_metadata(self, *, include_value: bool = False, redact: bool = True) -> Dict[str, Any]:
        """Return a safe metadata representation of the cache entry."""
        payload: Dict[str, Any] = {
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "ttl_remaining": self.ttl_remaining(),
            "hits": self.hits,
            "metadata": safe_serialize(self.metadata, redact=redact),
        }
        if include_value:
            payload["value"] = safe_serialize(self.value, redact=redact)
        else:
            payload["value_type"] = type(self.value).__name__
        return payload


@dataclass(slots=True)
class CacheStats:
    """Runtime counters for cache behavior."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    updates: int = 0
    deletes: int = 0
    pops: int = 0
    evictions: int = 0
    expirations: int = 0
    clears: int = 0
    cleanups: int = 0

    @property
    def requests(self) -> int:
        """Total lookup attempts tracked by the cache."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Return hit rate in the range ``0.0`` to ``1.0``."""
        return (self.hits / self.requests) if self.requests else 0.0

    def to_dict(self, *, size: int, max_size: int) -> Dict[str, Any]:
        """Return a stable, serialisable stats payload."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "updates": self.updates,
            "deletes": self.deletes,
            "pops": self.pops,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "clears": self.clears,
            "cleanups": self.cleanups,
            "requests": self.requests,
            "hit_rate": self.hit_rate,
            "size": size,
            "max_size": max_size,
        }


class FactoryCache(Generic[K, V]):
    """Thread-safe bounded LRU cache with optional TTL.

    Parameters
    ----------
    max_size:
        Maximum number of entries retained. When omitted, the value is read from
        the ``factory_cache.max_size`` config key.
    default_ttl_seconds:
        Optional TTL applied to entries when no per-entry TTL is provided. When
        omitted, the value is read from ``factory_cache.default_ttl_seconds``.
        Pass ``None`` explicitly to keep entries from expiring by time.
    name:
        Human-readable cache name used in logs and snapshots.
    config_section:
        Config section to read. Defaults to ``factory_cache`` and should not be
        changed outside tests or specialised cache instances.
    cleanup_interval_seconds:
        Optional opportunistic cleanup interval. Cleanup is only performed from
        foreground cache operations; no background thread is started.
    record_stats:
        Whether counters should be updated. Keeping it configurable allows tests
        and diagnostics to disable stats noise without changing behavior.
    """

    def __init__(self, max_size: Optional[int] = None, default_ttl_seconds: Any = _MISSING, *, name: Optional[str] = None,
                 config_section: str = "factory_cache", cleanup_interval_seconds: Any = _MISSING,
                 record_stats: Any = _MISSING) -> None:
        self.config = load_global_config()
        self.cache_config = get_config_section(config_section)
        self.config_section = config_section

        configured_name = self.cache_config.get("name", "factory_cache")
        configured_max_size = self.cache_config.get("max_size", 256)
        configured_default_ttl = self.cache_config.get("default_ttl_seconds", 300)
        configured_cleanup_interval = self.cache_config.get("cleanup_interval_seconds", 0)
        configured_record_stats = self.cache_config.get("record_stats", True)

        resolved_max_size = configured_max_size if max_size is None else max_size
        resolved_default_ttl = configured_default_ttl if default_ttl_seconds is _MISSING else default_ttl_seconds
        resolved_cleanup_interval = configured_cleanup_interval if cleanup_interval_seconds is _MISSING else cleanup_interval_seconds
        resolved_record_stats = configured_record_stats if record_stats is _MISSING else record_stats

        try:
            self.max_size, self.default_ttl_seconds = validate_cache_config(resolved_max_size, resolved_default_ttl)
            self.cleanup_interval_seconds = self._validate_cleanup_interval(resolved_cleanup_interval)
            self.record_stats = bool(resolved_record_stats)
        except FactoryCacheError:
            raise
        except Exception as exc:
            raise CacheConfigurationError(
                "Factory cache configuration is invalid",
                context={
                    "config_section": config_section,
                    "max_size": resolved_max_size,
                    "default_ttl_seconds": resolved_default_ttl,
                    "cleanup_interval_seconds": resolved_cleanup_interval,
                    "record_stats": resolved_record_stats,
                },
                cause=exc,
            ) from exc

        self.name = str(name or configured_name or "factory_cache")
        self._store: "OrderedDict[K, CacheEntry[V]]" = OrderedDict()
        self._stats = CacheStats()
        self._lock = RLock()
        self._last_cleanup_at = now_epoch_seconds()

        logger.info(
            "Factory Cache initialized | name=%s max_size=%s default_ttl_seconds=%s cleanup_interval_seconds=%s",
            self.name,
            self.max_size,
            self.default_ttl_seconds,
            self.cleanup_interval_seconds,
        )

    @staticmethod
    def _validate_cleanup_interval(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        interval = coerce_number(value, field_name="cleanup_interval_seconds")
        if interval < 0:
            raise CacheConfigurationError(
                "cleanup_interval_seconds must be >= 0",
                context={"cleanup_interval_seconds": value},
            )
        return interval

    def _record_hit(self) -> None:
        if self.record_stats:
            self._stats.hits += 1

    def _record_miss(self) -> None:
        if self.record_stats:
            self._stats.misses += 1

    def _compute_expiry(self, ttl_seconds: Optional[float]) -> Optional[float]:
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        try:
            validated_ttl = validate_cache_ttl(ttl, field_name="ttl_seconds")
        except FactoryCacheError:
            raise
        except Exception as exc:
            raise CacheTTLError(
                "ttl_seconds must be > 0 when provided",
                context={"ttl_seconds": ttl_seconds, "default_ttl_seconds": self.default_ttl_seconds},
                cause=exc,
            ) from exc
        return None if validated_ttl is None else now_epoch_seconds() + validated_ttl

    def _remove_expired_unlocked(self, key: K, entry: CacheEntry[V], *, count_miss: bool = False) -> bool:
        if not entry.is_expired():
            return False
        self._store.pop(key, None)
        if self.record_stats:
            self._stats.expirations += 1
            if count_miss:
                self._stats.misses += 1
        return True

    def _evict_if_needed_unlocked(self) -> None:
        try:
            validate_cache_capacity(len(self._store), self.max_size)
        except CacheCapacityError:
            # Being above capacity is expected immediately after insertion; the
            # validation call still guards against corrupted max_size values.
            pass

        while len(self._store) > self.max_size:
            try:
                self._store.popitem(last=False)
            except Exception as exc:
                raise CacheEvictionError(
                    "Unable to evict least-recently-used cache entry",
                    context={"cache": self.name, "size": len(self._store), "max_size": self.max_size},
                    cause=exc,
                ) from exc
            if self.record_stats:
                self._stats.evictions += 1

    def _maybe_cleanup_unlocked(self) -> None:
        if not self.cleanup_interval_seconds:
            return
        now = now_epoch_seconds()
        if now - self._last_cleanup_at >= self.cleanup_interval_seconds:
            self._cleanup_expired_unlocked()
            self._last_cleanup_at = now

    def _cleanup_expired_unlocked(self) -> int:
        expired_keys = [key for key, entry in self._store.items() if entry.is_expired()]
        for key in expired_keys:
            self._store.pop(key, None)
        removed = len(expired_keys)
        if self.record_stats:
            self._stats.expirations += removed
            self._stats.cleanups += 1
        return removed

    def set(self, key: K, value: V, ttl_seconds: Optional[float] = None, *,
            metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Insert or update a cache value."""
        validated_key = validate_cache_key(key)
        expires_at = self._compute_expiry(ttl_seconds)
        entry_metadata = normalize_payload(metadata)
        now = now_epoch_seconds()

        with self._lock:
            self._maybe_cleanup_unlocked()
            is_update = validated_key in self._store
            if is_update:
                old_entry = self._store.pop(validated_key)
                created_at = old_entry.created_at
                hits = old_entry.hits
            else:
                created_at = now
                hits = 0

            self._store[validated_key] = CacheEntry(
                value=value,
                created_at=created_at,
                updated_at=now,
                expires_at=expires_at,
                hits=hits,
                metadata=entry_metadata,
            )
            self._store.move_to_end(validated_key, last=True)
            if self.record_stats:
                self._stats.sets += 1
                if is_update:
                    self._stats.updates += 1
            self._evict_if_needed_unlocked()

    def get(self, key: K, default: Optional[V] = None, *, touch: bool = True) -> Optional[V]:
        """Return a cached value or ``default`` when missing/expired."""
        validated_key = validate_cache_key(key)
        with self._lock:
            self._maybe_cleanup_unlocked()
            entry = self._store.get(validated_key)
            if entry is None:
                self._record_miss()
                return default
            if self._remove_expired_unlocked(validated_key, entry, count_miss=True):
                return default
            if touch:
                entry.touch()
                self._store.move_to_end(validated_key, last=True)
            self._record_hit()
            return entry.value

    def get_or_set(self, key: K, factory: Any, ttl_seconds: Optional[float] = None, *,
                   metadata: Optional[Mapping[str, Any]] = None) -> V:
        """Return cached value, or compute/store one through ``factory``.

        ``factory`` may be a zero-argument callable or a direct value. This keeps
        cache call sites simple while avoiding repeated dependency-resolution or
        adaptation work.
        """
        current = self.get(key, default=_MISSING)  # type: ignore[arg-type]
        if current is not _MISSING:
            return cast(V, current)   # replace the ignore with cast
        value = factory() if callable(factory) else factory
        self.set(key, cast(V, value), ttl_seconds=ttl_seconds, metadata=metadata)
        return cast(V, value)

    def peek(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Return a value without moving it to the LRU tail or incrementing hits."""
        return self.get(key, default=default, touch=False)

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Remove and return a value, or ``default`` when missing/expired."""
        validated_key = validate_cache_key(key)
        with self._lock:
            entry = self._store.pop(validated_key, None)
            if entry is None:
                return default
            if entry.is_expired():
                if self.record_stats:
                    self._stats.expirations += 1
                return default
            if self.record_stats:
                self._stats.pops += 1
            return entry.value

    def delete(self, key: K) -> bool:
        """Delete a cache entry and return whether it existed."""
        validated_key = validate_cache_key(key)
        with self._lock:
            existed = self._store.pop(validated_key, None) is not None
            if existed and self.record_stats:
                self._stats.deletes += 1
            return existed

    def clear(self) -> None:
        """Remove all cache entries while preserving counters."""
        with self._lock:
            self._store.clear()
            if self.record_stats:
                self._stats.clears += 1

    def reset(self) -> None:
        """Clear entries and reset runtime counters."""
        with self._lock:
            self._store.clear()
            self._stats = CacheStats()
            self._last_cleanup_at = now_epoch_seconds()

    def cleanup_expired(self) -> int:
        """Remove expired entries and return the number removed."""
        with self._lock:
            removed = self._cleanup_expired_unlocked()
            self._last_cleanup_at = now_epoch_seconds()
            return removed

    def has(self, key: K) -> bool:
        """Return whether a non-expired entry exists for ``key``."""
        return self.get(key, default=_MISSING, touch=False) is not _MISSING  # type: ignore[arg-type]

    def touch(self, key: K) -> bool:
        """Mark an entry as recently used without returning the value."""
        validated_key = validate_cache_key(key)
        with self._lock:
            entry = self._store.get(validated_key)
            if entry is None:
                self._record_miss()
                return False
            if self._remove_expired_unlocked(validated_key, entry, count_miss=True):
                return False
            entry.touch()
            self._store.move_to_end(validated_key, last=True)
            self._record_hit()
            return True

    def ttl_remaining(self, key: K) -> Optional[float]:
        """Return remaining TTL seconds for a key, or ``None`` when missing/non-expiring."""
        validated_key = validate_cache_key(key)
        with self._lock:
            entry = self._store.get(validated_key)
            if entry is None:
                return None
            if self._remove_expired_unlocked(validated_key, entry):
                raise CacheEntryExpiredError(
                    "Cache entry expired while reading TTL",
                    context={"cache": self.name, "key": safe_serialize(validated_key)},
                )
            return entry.ttl_remaining()

    def set_max_size(self, max_size: int) -> None:
        """Update max size and evict LRU entries if needed."""
        validated_size = ensure_positive_int(max_size, "max_size")
        with self._lock:
            self.max_size = validated_size
            self._evict_if_needed_unlocked()

    def set_default_ttl(self, default_ttl_seconds: Optional[float]) -> None:
        """Update the default TTL used by future writes."""
        self.default_ttl_seconds = validate_cache_ttl(default_ttl_seconds, field_name="default_ttl_seconds")

    def size(self) -> int:
        """Return the number of currently stored entries."""
        with self._lock:
            self._maybe_cleanup_unlocked()
            return len(self._store)

    def is_empty(self) -> bool:
        """Return whether the cache currently contains no live entries."""
        return self.size() == 0

    def keys(self) -> Tuple[K, ...]:
        """Return non-expired keys in LRU order."""
        with self._lock:
            self._maybe_cleanup_unlocked()
            return tuple(self._store.keys())

    def values(self) -> Tuple[V, ...]:
        """Return non-expired values in LRU order."""
        with self._lock:
            self._maybe_cleanup_unlocked()
            return tuple(entry.value for entry in self._store.values())

    def items(self) -> Iterator[Tuple[K, V]]:
        """Yield non-expired ``(key, value)`` pairs in LRU order."""
        with self._lock:
            self._maybe_cleanup_unlocked()
            snapshot = tuple((key, entry.value) for key, entry in self._store.items())
        yield from snapshot

    def entry_metadata(self, key: K, *, include_value: bool = False, redact: bool = True) -> Optional[Dict[str, Any]]:
        """Return metadata for a single live entry."""
        validated_key = validate_cache_key(key)
        with self._lock:
            entry = self._store.get(validated_key)
            if entry is None:
                return None
            if self._remove_expired_unlocked(validated_key, entry):
                return None
            return entry.to_metadata(include_value=include_value, redact=redact)

    def stats(self) -> Dict[str, Any]:
        """Return validated runtime cache statistics."""
        with self._lock:
            self._maybe_cleanup_unlocked()
            payload = self._stats.to_dict(size=len(self._store), max_size=self.max_size)
        try:
            validate_cache_stats(payload)
        except CacheStatsError:
            raise
        except Exception as exc:
            raise CacheStatsError(
                "Factory cache statistics failed validation",
                context={"cache": self.name, "stats": payload},
                cause=exc,
            ) from exc
        return payload

    def snapshot(self, *, include_values: bool = False, redact: bool = True) -> Dict[str, Any]:
        """Return a diagnostic snapshot for observability and tests."""
        with self._lock:
            self._maybe_cleanup_unlocked()
            entries = {
                safe_serialize(key, redact=redact): entry.to_metadata(include_value=include_values, redact=redact)
                for key, entry in self._store.items()
            }
            return {
                "name": self.name,
                "config_section": self.config_section,
                "max_size": self.max_size,
                "default_ttl_seconds": self.default_ttl_seconds,
                "cleanup_interval_seconds": self.cleanup_interval_seconds,
                "record_stats": self.record_stats,
                "size": len(self._store),
                "entries": entries,
                "stats": self._stats.to_dict(size=len(self._store), max_size=self.max_size),
            }

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, key: object) -> bool:
        try:
            return self.has(key)  # type: ignore[arg-type]
        except CacheKeyError:
            return False

    def __iter__(self) -> Iterator[K]:
        return iter(self.keys())

    def __repr__(self) -> str:
        return (
            f"FactoryCache(name={self.name!r}, size={self.size()}, max_size={self.max_size}, "
            f"default_ttl_seconds={self.default_ttl_seconds!r})"
        )


if __name__ == "__main__":
    print("\n=== Running Factory Cache ===\n")
    printer.status("TEST", "Factory Cache initialized", "info")

    cache: FactoryCache[str, Dict[str, Any]] = FactoryCache(max_size=3, default_ttl_seconds=1.0, name="factory_cache_test")

    cache.set("registry:adaptive", {"agent": "adaptive", "version": "1.0.0"})
    cache.set("registry:alignment", {"agent": "alignment", "version": "1.0.0"})
    cache.set("dependency:adaptive", {"deps": ["base"]})

    assert cache.size() == 3
    assert cache.get("registry:adaptive") == {"agent": "adaptive", "version": "1.0.0"}
    assert cache.has("registry:alignment") is True

    cache.set("worker:payload", {"status": "normalised"})
    assert cache.size() == 3
    assert cache.has("registry:alignment") is False
    assert cache.get("worker:payload") == {"status": "normalised"}

    computed = cache.get_or_set("adaptation:latest", lambda: {"risk_threshold": 0.7}, ttl_seconds=1.0)
    assert computed == {"risk_threshold": 0.7}
    assert cache.get("adaptation:latest") == {"risk_threshold": 0.7}

    popped = cache.pop("worker:payload")
    assert popped == {"status": "normalised"}
    assert cache.delete("registry:adaptive") is True

    stats = cache.stats()
    assert stats["sets"] >= 5
    assert stats["hits"] >= 4
    assert stats["size"] <= stats["max_size"]

    snapshot = cache.snapshot(include_values=True)
    assert snapshot["name"] == "factory_cache_test"
    assert snapshot["size"] <= 3

    cache.clear()
    assert cache.is_empty() is True

    print("\n=== Test ran successfully ===\n")
