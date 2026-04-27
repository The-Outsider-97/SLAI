"""
Provides a thread-safe, in-memory shared storage mechanism with features
tailored for AI model coordination and data sharing within a single process.

This module is the collaborative runtime's local memory fabric. It supports
versioned values, TTL expiration, access tracking, priority retrieval,
compare-and-swap updates, bounded memory eviction, tags, snapshots,
persistence, pub/sub callbacks, intervention logging, integrity checks, and an
optional ``multiprocessing.BaseManager`` proxy for multi-process access.

Production notes
----------------
- The default ``SharedMemory`` object remains a singleton to preserve existing
  subsystem expectations.
- Public method names from the previous implementation are retained.
- Local project imports are intentionally direct; config loading remains owned by
  ``.utils.config_loader``.
- Serialization helpers and collaboration errors are used at runtime boundaries
  where diagnostics or persistence can fail.
- Values are process-local Python objects unless accessed through the manager
  proxy. For durable or distributed storage beyond a single host, use an
  external backing service such as Redis, SQLite, Postgres, or an object store.
"""

from __future__ import annotations

import fnmatch
import heapq
import os
import pickle
import random
import sys
import tempfile
import threading
import time
import multiprocessing

from collections import OrderedDict, defaultdict, deque, namedtuple
from collections.abc import Callable, Hashable, Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from multiprocessing.managers import BaseManager, NamespaceProxy  # type: ignore

from .utils.config_loader import load_global_config, get_config_section
from .utils.collaboration_error import *
from .utils.collaborative_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Shared Memory")
printer=PrettyPrinter()

# Using deque for versions allows efficient append and limiting version history size.
VersionedItem = namedtuple("VersionedItem", ["timestamp", "value"])

_BYTES_PER_MB = 1024 ** 2
_DEFAULT_MANAGER_ADDRESS = ("127.0.0.1", 8000)
_DEFAULT_MANAGER_AUTHKEY = b"secret"


def _make_proxy_method(method_name: str) -> Callable[..., Any]:
    """Create a proxy method without using exec in the class body."""

    def proxy_method(self: NamespaceProxy, *args: Any, **kwargs: Any) -> Any:
        return self._callmethod(method_name, args, kwargs)

    proxy_method.__name__ = method_name
    proxy_method.__qualname__ = f"SharedMemoryProxy.{method_name}"
    return proxy_method


class SharedMemoryProxy(NamespaceProxy):
    """Custom proxy exposing ``SharedMemory`` methods through BaseManager."""

    _exposed_ = ("__contains__", "__len__", "put", "set", "get", "append",
                 "get_all_versions", "get_access_time", "get_next_prioritized_item",
                 "delete", "clear_all", "register_callback", "publish", "subscribe",
                 "unsubscribe", "notify", "get_usage_stats", "metrics", "save_to_file",
                 "load_from_file", "increment", "get_latest_snapshot", "log_intervention",
                 "configure", "compare_and_swap", "get_all_keys", "get_by_tag",
                 "cleanup_expired", "validate_integrity", "compact", "health_check",
                 "snapshot_state", "restore_snapshot", "close")

    def __contains__(self, key: Hashable) -> bool:
        return self._callmethod("__contains__", (key,))

    def __len__(self) -> int:
        return self._callmethod("__len__")


for _proxy_method_name in SharedMemoryProxy._exposed_:
    if _proxy_method_name not in {"__contains__", "__len__"}:
        setattr(SharedMemoryProxy, _proxy_method_name, _make_proxy_method(_proxy_method_name))

class SharedMemory:
    """
    A thread-safe shared memory implementation for multy- and single-process use.

    Provides versioned storage, expiration, access tracking, priority queuing,
    and basic locking for conflict resolution.

    Note: This implementation uses threading.Lock() and is suitable for use
    with multiple threads within the *same* process. It is *not* suitable
    for inter-process communication (IPC) across different processes.
    For IPC, consider using `multiprocessing.Manager` or external solutions
    like Redis or Memcached.
    """
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance
    
    def __init__(self):
        # Using deque allows efficient append and limiting version count if max_versions is set
        # Force conversion to integer or None
        if getattr(self, "_SharedMemory__initialized", False):
            return
        if self.__initialized:
            return
        self.__initialized = True
        self.config = load_global_config()
        self.memory_config = get_config_section("shared_memory")

        self.max_memory = self._coerce_memory_bytes(self.memory_config.get("max_memory_mb", 100))
        self._max_versions = self._validate_max_versions(self.memory_config.get("max_versions", 10)) or 10
        self.ttl_check_interval = coerce_float(
            self.memory_config.get("ttl_check_interval", 30),
            default=30.0,
            minimum=0.05,
            maximum=24 * 60 * 60,
        )
        self.network_latency = self.memory_config.get("network_latency", 0.0)
        self._default_ttl = self._normalize_ttl(self.memory_config.get("default_ttl", 3600), allow_immediate=False)
        self._enable_background_cleaner = coerce_bool(self.memory_config.get("enable_background_cleaner", True), default=True)
        self._async_callbacks = coerce_bool(self.memory_config.get("async_callbacks", True), default=True)
        self._callback_threads_daemon = coerce_bool(self.memory_config.get("callback_threads_daemon", True), default=True)
        self._allow_oversized_items = coerce_bool(self.memory_config.get("allow_oversized_items", True), default=True)
        self._max_key_repr_length = coerce_int(self.memory_config.get("max_key_repr_length", 256), default=256, minimum=32)
        self._max_intervention_log_entries = coerce_int(
            self.memory_config.get("max_intervention_log_entries", 1000),
            default=1000,
            minimum=1,
        )
        self._snapshot_prefix = str(self.memory_config.get("snapshot_prefix", "snapshots:") or "snapshots:")
        self._intervention_log_key = str(self.memory_config.get("intervention_log_key", "intervention_logs") or "intervention_logs")

        self._lock = multiprocessing.RLock()
        self._data: dict[Hashable, deque[VersionedItem]] = {}
        self._expiration: dict[Hashable, float] = {}
        self.subscribers: dict[str, list[Callable[[Any], Any]]] = defaultdict(list)
        self.callbacks: dict[str, list[Callable[[Any], Any]]] = defaultdict(list)
        self._tags: dict[Hashable, list[str]] = defaultdict(list)
        self._size_by_key: dict[Hashable, int] = {}

        self.current_memory = 0
        self._access_log: OrderedDict[Hashable, float] = OrderedDict()
        self._priority_queue: list[tuple[float, float, Hashable]] = []
        self._last_cleanup_time: Optional[float] = None
        self._last_cleanup_count = 0
        self._last_eviction_time: Optional[float] = None
        self._last_evicted_key: Optional[str] = None
        self._operation_counts: dict[str, int] = defaultdict(int)
        self._operation_failures: dict[str, int] = defaultdict(int)
        self._created_at = time.time()
        self._closed = False
        self.base_state: dict[str, Any] = {}

        self._stop_cleaner_event = threading.Event()
        self._cleaner_thread: Optional[threading.Thread] = None
        if self._enable_background_cleaner:
            self._start_expiration_cleaner()

        logger.info(
            "Shared Memory successfully initialized with max_memory_mb=%s, max_versions=%s, default_ttl=%s",
            round(self.max_memory / _BYTES_PER_MB, 3),
            self._max_versions,
            self._default_ttl,
        )

    # ------------------------------------------------------------------
    # Configuration and error helpers
    # ------------------------------------------------------------------
    def _load_memory_config(self) -> dict[str, Any]:
        """Load shared-memory config from the collaborative config section.

        The project config loader already points this subsystem at the
        collaborative runtime configuration. Keep the source of truth explicit:
        ``shared_memory`` in ``collaborative_config.yaml``.
        """

        shared_memory_config = get_config_section("shared_memory") or {}
        if isinstance(shared_memory_config, Mapping):
            return dict(shared_memory_config)
        return {}

    def _record_operation(self, operation: str) -> None:
        self._operation_counts[operation] += 1

    def _record_failure(self, operation: str) -> None:
        self._operation_failures[operation] += 1

    def _raise_memory_error(
        self,
        operation: str,
        reason: str,
        *,
        key: Optional[Hashable] = None,
        cause: Optional[BaseException] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._record_failure(operation)
        key_repr = None if key is None else self._safe_key_repr(key)
        raise SharedMemoryFailureError(
            operation=operation,
            reason=reason,
            key=key_repr,
            context={
                "memory_stats": self._safe_stats_for_error(),
                **dict(context or {}),
            },
            cause=cause,
        ) from cause # type: ignore

    def _safe_stats_for_error(self) -> dict[str, Any]:
        return {
            "item_count": len(getattr(self, "_data", {}) or {}),
            "current_memory_mb": round(float(getattr(self, "current_memory", 0)) / _BYTES_PER_MB, 3),
            "max_memory_mb": round(float(getattr(self, "max_memory", 0)) / _BYTES_PER_MB, 3),
            "priority_queue_size": len(getattr(self, "_priority_queue", []) or []),
        }

    def _safe_key_repr(self, key: Any) -> str:
        text = repr(key)
        if len(text) > self._max_key_repr_length:
            return text[: self._max_key_repr_length - 3] + "..."
        return text

    def _validate_key(self, key: Hashable, *, operation: str) -> Hashable:
        if key is None:
            self._raise_memory_error(operation, "key must not be None", key=key)
        try:
            hash(key)
        except Exception as exc:
            self._raise_memory_error(operation, "key must be hashable", key=key, cause=exc)
        return key

    def _coerce_memory_bytes(self, max_memory_mb: Any) -> int:
        mb = coerce_float(max_memory_mb, default=100.0, minimum=0.001)
        return max(1, int(mb * _BYTES_PER_MB))

    # ------------------------------------------------------------------
    # Metrics and health
    # ------------------------------------------------------------------
    def get_usage_stats(self) -> dict:
        """Returns detailed statistics about memory usage and system performance."""

        self._record_operation("get_usage_stats")
        with self._lock:
            self._cleanup_expired_locked(time.time())
            memory_usage_pct = (self.current_memory / self.max_memory) * 100 if self.max_memory > 0 else 0.0
            available_memory_mb = max(0, self.max_memory - self.current_memory) / _BYTES_PER_MB
            current_time = time.time()
            pending_expiration_count = sum(1 for expiry in self._expiration.values() if expiry <= current_time)
            total_items = len(self._data)
            avg_item_size = self.current_memory / total_items if total_items > 0 else 0

            return {
                "current_memory_mb": round(self.current_memory / _BYTES_PER_MB, 6),
                "max_memory_mb": round(self.max_memory / _BYTES_PER_MB, 6),
                "available_memory_mb": round(available_memory_mb, 6),
                "memory_usage_percentage": round(memory_usage_pct, 2),
                "item_count": total_items,
                "avg_item_size_kb": round(avg_item_size / 1024, 3),
                "max_versions_per_item": self._max_versions,
                "expiration_count": len(self._expiration),
                "pending_expiration_cleanup": pending_expiration_count,
                "default_ttl_seconds": self._default_ttl,
                "access_log_size": len(self._access_log),
                "priority_queue_size": len(self._priority_queue),
                "tagged_key_count": len(self._tags),
                "subscription_count": sum(len(callbacks) for callbacks in self.subscribers.values()),
                "callback_count": sum(len(callbacks) for callbacks in self.callbacks.values()),
                "ttl_check_interval": self.ttl_check_interval,
                "background_cleaner_enabled": self._enable_background_cleaner,
                "background_cleaner_alive": bool(self._cleaner_thread and self._cleaner_thread.is_alive()),
                "last_cleanup": self._last_cleanup_time,
                "last_cleanup_count": self._last_cleanup_count,
                "last_eviction_time": self._last_eviction_time,
                "last_evicted_key": self._last_evicted_key,
                "closed": self._closed,
                "uptime_seconds": round(current_time - self._created_at, 3),
            }

    def metrics(self) -> dict:
        """Returns operational metrics and usage patterns of the shared memory."""

        self._record_operation("metrics")
        with self._lock:
            current_time = time.time()
            self._cleanup_expired_locked(current_time)
            access_times = list(self._access_log.values())
            time_since_last_access = current_time - max(access_times) if access_times else 0
            time_to_expiry = [expiry - current_time for expiry in self._expiration.values() if expiry > current_time]
            priorities = [-priority for priority, _, key in self._priority_queue if key in self._data]

            return {
                "access_count": len(self._access_log),
                "time_since_last_access_seconds": round(time_since_last_access, 3),
                "access_pattern": {
                    "min_access_time": min(access_times) if access_times else 0,
                    "max_access_time": max(access_times) if access_times else 0,
                    "avg_access_time": sum(access_times) / len(access_times) if access_times else 0,
                },
                "expiration_metrics": {
                    "earliest_expiry_seconds": min(time_to_expiry) if time_to_expiry else 0,
                    "latest_expiry_seconds": max(time_to_expiry) if time_to_expiry else 0,
                    "avg_expiry_seconds": round(sum(time_to_expiry) / len(time_to_expiry), 3) if time_to_expiry else 0,
                    "expired_items_pending": sum(1 for expiry in self._expiration.values() if expiry <= current_time),
                },
                "priority_queue_metrics": {
                    "highest_priority": max(priorities) if priorities else 0,
                    "lowest_priority": min(priorities) if priorities else 0,
                    "avg_priority": round(sum(priorities) / len(priorities), 3) if priorities else 0,
                    "stale_entries": sum(1 for _, _, key in self._priority_queue if key not in self._data),
                },
                "operation_counts": dict(self._operation_counts),
                "operation_failures": dict(self._operation_failures),
                "cleanup_metrics": {
                    "last_cleanup_run": self._last_cleanup_time,
                    "items_cleaned_last_run": self._last_cleanup_count,
                },
            }

    def health_check(self) -> dict:
        """Return a compact operational health report."""

        self._record_operation("health_check")
        integrity = self.validate_integrity()
        stats = self.get_usage_stats()
        healthy = bool(integrity.get("ok")) and not self._closed
        return {
            "ok": healthy,
            "status": "healthy" if healthy else "degraded",
            "timestamp": utc_timestamp(),
            "integrity": integrity,
            "usage": stats,
        }

    # ==============================
    #  2. Core Public API
    # ==============================
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            from pympler import asizeof
            return int(asizeof.asizeof(obj))
        except ImportError:
            # fallback to shallow size
            try:
                return sys.getsizeof(obj)
            except Exception:
                return 1024
        except Exception as exc:
            self._raise_memory_error("calculate_size", "failed to calculate object size", cause=exc)
            raise

    def put(self, key, value, ttl=None, priority=None, tags=None, **kwargs):
        """Store a versioned value, optionally with TTL, priority, and tags."""

        self._record_operation("put")
        self._ensure_open("put")
        key = self._validate_key(key, operation="put")
        self._simulate_network()
        current_time = time.time()
        notify = coerce_bool(kwargs.pop("notify", True), default=True)
        metadata = normalize_metadata(kwargs.pop("metadata", None), drop_none=True) if kwargs else {}
        if kwargs:
            metadata.update(normalize_metadata(kwargs, drop_none=True))

        with self._lock:
            try:
                effective_ttl = self._normalize_ttl(ttl, allow_immediate=True)
                if tags is not None:
                    self._tags[key] = list(normalize_tags(tags))

                if effective_ttl is not None and effective_ttl <= 0:
                    self._remove_key(key)
                    self._last_cleanup_time = current_time
                    self._last_cleanup_count += 1
                    if notify:
                        self._notify_change_locked(key, None, event_type="expired", metadata=metadata)
                    return current_time

                self._store_version_locked(key, value, current_time=current_time)
                self._set_expiration_locked(key, effective_ttl, current_time=current_time)

                if priority is not None:
                    priority_value = coerce_float(priority, default=0.0)
                    heapq.heappush(self._priority_queue, (-priority_value, current_time, key))

                self._evict_until_within_limit_locked(protected_key=key)

                if notify:
                    self._notify_change_locked(key, value, event_type="put", metadata=metadata)
                return current_time
            except CollaborationError: # type: ignore
                raise
            except Exception as exc:
                self._raise_memory_error(
                    "put",
                    "unexpected put failure",
                    key=key,
                    cause=exc,
                    context={"ttl": ttl, "priority": priority, "tags": list(normalize_tags(tags)) if tags else []},
                )

    def set(self, key, value, *, ttl=None, **kwargs):
        """Set a value in shared memory with TTL and versioning."""

        self._record_operation("set")
        return self.put(key, value, ttl=ttl, **kwargs)

    def get(self, key, version_timestamp=None, update_access=True, default=None):
        """Retrieve the latest value, or the latest version at/before a timestamp."""

        self._record_operation("get")
        self._ensure_open("get")
        key = self._validate_key(key, operation="get")
        self._simulate_network()

        with self._lock:
            current_time = time.time()
            if key not in self._data or self._is_expired(key, current_time):
                if key in self._data:
                    self._remove_key(key)
                return default

            if update_access:
                self._touch_access_locked(key, current_time)

            versions = self._data.get(key)
            if not versions:
                return default

            if version_timestamp is None:
                return versions[-1].value

            try:
                version_ts = float(version_timestamp)
            except (TypeError, ValueError):
                return default

            for version in reversed(versions):
                if version.timestamp <= version_ts:
                    return version.value
            return default

    def get_by_tag(self, tag, limit=None):
        """Retrieve items by tag with optional limit."""

        self._record_operation("get_by_tag")
        normalized_tag = str(tag or "").strip()
        if not normalized_tag:
            return []
        max_items = coerce_int(limit, default=0, minimum=0) if limit is not None else 0

        with self._lock:
            current_time = time.time()
            results = []
            for key in list(self._data.keys()):
                if self._is_expired(key, current_time):
                    self._remove_key(key)
                    continue
                if normalized_tag not in self._tags.get(key, []):
                    continue
                versions = self._data.get(key)
                if not versions:
                    continue
                results.append({"key": key, "value": versions[-1].value, "tags": list(self._tags.get(key, []))})
                if max_items and len(results) >= max_items:
                    break
            return results

    def save_to_file(self, filename):
        """Persist memory state atomically using pickle.

        This method is intended for trusted local state only. Pickle files should
        never be loaded from untrusted sources.
        """

        self._record_operation("save_to_file")
        self._ensure_open("save_to_file")
        path = Path(filename)
        try:
            path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path("") else None
            with self._lock:
                self._cleanup_expired_locked(time.time())
                payload = {
                    "schema_version": 2,
                    "created_at": utc_timestamp(),
                    "data": self._data,
                    "expiration": self._expiration,
                    "access_log": self._access_log,
                    "priority_queue": self._priority_queue,
                    "tags": dict(self._tags),
                    "size_by_key": self._size_by_key,
                    "base_state": self.base_state,
                    "operation_counts": dict(self._operation_counts),
                }

            temp_dir = str(path.parent if str(path.parent) else Path.cwd())
            fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=temp_dir)
            try:
                with os.fdopen(fd, "wb") as file_obj:
                    pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
                os.replace(temp_name, path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            logger.info("Shared memory state saved to %s", path)
            return str(path)
        except Exception as exc:
            self._raise_memory_error("save_to_file", "failed to persist shared memory state", cause=exc, context={"filename": str(filename)})

    def load_from_file(self, filename):
        """Load memory state from a trusted pickle file."""

        self._record_operation("load_from_file")
        self._ensure_open("load_from_file")
        path = Path(filename)
        try:
            with path.open("rb") as file_obj:
                payload = pickle.load(file_obj)

            with self._lock:
                if isinstance(payload, tuple) and len(payload) == 3:
                    # Backwards compatibility with the previous implementation.
                    self._data, self._expiration, self._access_log = payload
                    self._priority_queue = []
                    self._tags = defaultdict(list)
                    self.base_state = {}
                elif isinstance(payload, Mapping):
                    self._data = dict(payload.get("data", {}))
                    self._expiration = dict(payload.get("expiration", {}))
                    self._access_log = OrderedDict(payload.get("access_log", {}))
                    self._priority_queue = list(payload.get("priority_queue", []))
                    self._tags = defaultdict(list, {k: list(v) for k, v in dict(payload.get("tags", {})).items()})
                    self.base_state = dict(payload.get("base_state", {}) or {})
                    self._operation_counts.update(dict(payload.get("operation_counts", {}) or {}))
                else:
                    self._raise_memory_error("load_from_file", "unsupported shared memory persistence payload", context={"filename": str(filename)})

                self._normalize_loaded_state_locked()
                self._cleanup_expired_locked(time.time())
                self._clean_priority_queue_locked()
            logger.info("Shared memory state loaded from %s", path)
            return True
        except CollaborationError: # type: ignore
            raise
        except Exception as exc:
            self._raise_memory_error("load_from_file", "failed to load shared memory state", cause=exc, context={"filename": str(filename)})

    def delete(self, key):
        """Deletes a key and all its associated data, versions, and metadata."""

        self._record_operation("delete")
        self._ensure_open("delete")
        key = self._validate_key(key, operation="delete")
        self._simulate_network()
        with self._lock:
            existed = key in self._data
            if existed:
                self._remove_key(key)
                self._notify_change_locked(key, None, event_type="delete")
            return existed

    def clear_all(self):
        """Clear all stored data while keeping registered callbacks/subscribers."""

        self._record_operation("clear_all")
        self._ensure_open("clear_all")
        with self._lock:
            self._data.clear()
            self._expiration.clear()
            self._access_log.clear()
            self._priority_queue.clear()
            self._tags.clear()
            self._size_by_key.clear()
            self.current_memory = 0
            self._last_cleanup_time = time.time()
            self._last_cleanup_count = 0
        return True

    # ==============================
    #  3. Versioning and Append
    # ==============================
    def append(self, key: str, value: Any, ttl: Optional[int] = None, priority: Optional[int] = None):
        """Append a new version for a key and notify callbacks/subscribers."""

        self._record_operation("append")
        return self.put(key, value, ttl=ttl, priority=priority, notify=True, metadata={"operation": "append"})

    def _safe_callback_call(self, cb, value):
        """Helper to call callbacks safely."""

        try:
            cb(value)
        except Exception as exc:
            logger.error("Callback error: %s", exc, exc_info=True)
            self._record_failure("callback")

    def get_all_versions(self, key, update_access=True):
        """Retrieves all available versions of a value associated with a key."""

        self._record_operation("get_all_versions")
        self._ensure_open("get_all_versions")
        key = self._validate_key(key, operation="get_all_versions")
        with self._lock:
            current_time = time.time()
            if key not in self._data or self._is_expired(key, current_time):
                if key in self._data:
                    self._remove_key(key)
                return []

            if update_access:
                self._touch_access_locked(key, current_time)
            return list(self._data.get(key, ()))

    def get_access_time(self, key):
        """Gets the last access timestamp for a key."""

        self._record_operation("get_access_time")
        self._ensure_open("get_access_time")
        key = self._validate_key(key, operation="get_access_time")
        with self._lock:
            current_time = time.time()
            if self._is_expired(key, current_time):
                if key in self._data:
                    self._remove_key(key)
                return None
            return self._access_log.get(key)

    # ==============================
    #  4. Priority Queue Support
    # ==============================
    def get_next_prioritized_item(self, remove=True):
        """
        Retrieves the highest priority key from the queue.

        Returns:
            tuple(priority, key) or None. Priority is returned as the original
            positive value.
        """

        self._record_operation("get_next_prioritized_item")
        self._ensure_open("get_next_prioritized_item")
        with self._lock:
            current_time = time.time()
            while self._priority_queue:
                neg_priority, timestamp, key = self._priority_queue[0]
                if key not in self._data or self._is_expired(key, current_time):
                    heapq.heappop(self._priority_queue)
                    if key in self._data:
                        self._remove_key(key)
                    continue

                if remove:
                    heapq.heappop(self._priority_queue)
                return (-neg_priority, key)
            return None

    # ===================================
    #  5. TTL and Expiration Management
    # ===================================
    def cleanup_expired(self):
        """Removes all items that have passed their expiration time."""

        self._record_operation("cleanup_expired")
        self._ensure_open("cleanup_expired")
        with self._lock:
            return self._cleanup_expired_locked(time.time())

    def _start_expiration_cleaner(self):
        """Start a daemon cleaner thread once per instance."""

        if self._cleaner_thread and self._cleaner_thread.is_alive():
            return

        def cleaner():
            while not self._stop_cleaner_event.wait(self.ttl_check_interval):
                try:
                    with self._lock:
                        self._cleanup_expired_locked(time.time())
                except Exception as exc:  # noqa: BLE001 - background safety boundary.
                    logger.error("Shared memory expiration cleaner failed: %s", exc, exc_info=True)
                    self._record_failure("expiration_cleaner")

        self._cleaner_thread = threading.Thread(target=cleaner, name="SharedMemoryExpirationCleaner", daemon=True)
        self._cleaner_thread.start()

    def _clean_priority_queue(self):
        """Remove entries for non-existent or expired keys."""

        self._record_operation("clean_priority_queue")
        with self._lock:
            self._clean_priority_queue_locked()

    def _is_expired(self, key, current_time):
        """Checks if a key is expired without locking."""

        return key in self._expiration and self._expiration[key] <= current_time

    # ===================================
    #  6. Callbacks and Subscriptions
    # ===================================
    def register_callback(self, channel: str, callback: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Register a callback for specific key updates."""

        self._record_operation("register_callback")
        self._ensure_open("register_callback")
        self._validate_callback(callback, operation="register_callback")
        normalized_channel = self._normalize_channel(channel)
        with self._lock:
            self.callbacks[normalized_channel].append(callback)
        return callback

    def publish(self, channel, message):
        """Publish a message to subscribers of a channel."""

        self._record_operation("publish")
        self._ensure_open("publish")
        normalized_channel = self._normalize_channel(channel)
        payload = json_safe(message)
        with self._lock:
            callbacks = self._matching_subscribers_locked(normalized_channel)
        for callback in callbacks:
            self._dispatch_callback(callback, payload)
        return len(callbacks)

    def subscribe(self, channel, callback, *, once: bool = True):
        """Subscribe to a channel or fnmatch pattern.

        ``once=True`` preserves the previous one-shot subscription behavior.
        Pass ``once=False`` for persistent pub/sub subscriptions.
        """

        self._record_operation("subscribe")
        self._ensure_open("subscribe")
        self._validate_callback(callback, operation="subscribe")
        normalized_channel = self._normalize_channel(channel)

        if once:
            def wrapper(value):
                try:
                    callback(value)
                finally:
                    self.unsubscribe(normalized_channel, wrapper)

            wrapper.__name__ = getattr(callback, "__name__", "subscriber_wrapper")
            subscribed_callback = wrapper
        else:
            subscribed_callback = callback

        with self._lock:
            self.subscribers[normalized_channel].append(subscribed_callback)
        return subscribed_callback

    def unsubscribe(self, channel, callback):
        """Unsubscribe a callback."""

        self._record_operation("unsubscribe")
        normalized_channel = self._normalize_channel(channel)
        with self._lock:
            callbacks = self.subscribers.get(normalized_channel)
            if not callbacks:
                return False
            if callback in callbacks:
                callbacks.remove(callback)
                if not callbacks:
                    self.subscribers.pop(normalized_channel, None)
                return True
        return False

    def notify(self, channel, value):
        """Notify direct callbacks and pub/sub subscribers for a channel."""

        self._record_operation("notify")
        self._ensure_open("notify")
        normalized_channel = self._normalize_channel(channel)
        with self._lock:
            callbacks = list(self.callbacks.get(normalized_channel, []))
            callbacks.extend(self._matching_subscribers_locked(normalized_channel))
        for callback in callbacks:
            self._dispatch_callback(callback, value)
        return len(callbacks)

    # ==============================
    #  7. Validation Helpers
    # ==============================
    def _validate_ttl(self, value):
        """Validate TTL input (supports int/float/timedelta)."""

        return self._normalize_ttl(value, allow_immediate=False)

    def _validate_max_versions(self, value):
        """Ensure max_versions is a positive integer or None."""

        if value is None:
            return None
        parsed = coerce_int(value, default=0)
        if parsed > 0:
            return parsed
        logger.warning("Invalid max_versions: %s. Defaulting to None.", value)
        return None

    def _validate_network_latency(self, value):
        """Ensure network latency is a non-negative float."""

        return coerce_float(value, default=0.0, minimum=0.0, maximum=5.0)

    def compare_and_swap(self, key, expected_value, new_value):
        """Atomically replace a value when the current latest value matches expected_value."""

        self._record_operation("compare_and_swap")
        self._ensure_open("compare_and_swap")
        key = self._validate_key(key, operation="compare_and_swap")
        with self._lock:
            current_time = time.time()
            if key not in self._data or self._is_expired(key, current_time):
                if key in self._data:
                    self._remove_key(key)
                current = None
            else:
                versions = self._data.get(key)
                current = versions[-1].value if versions else None

            if current == expected_value:
                self._store_version_locked(key, new_value, current_time=current_time)
                self._set_expiration_locked(key, self._default_ttl, current_time=current_time)
                self._evict_until_within_limit_locked(protected_key=key)
                self._notify_change_locked(key, new_value, event_type="compare_and_swap")
                return True
            return False

    # ==============================
    #  8. Network Simulation
    # ==============================
    def _simulate_network(self):
        """Simulates network latency for distributed coordination scenarios."""

        if self.network_latency > 0:
            jitter = random.uniform(-0.1, 0.1) * self.network_latency
            effective_delay = max(0.0, self.network_latency + jitter)
            time.sleep(effective_delay)

    @property
    def network_latency(self):
        """Get current network latency with validation."""

        return self._network_latency

    @network_latency.setter
    def network_latency(self, value):
        """Set network latency with validation and logging."""

        self._network_latency = self._validate_network_latency(value)
        logger.info("Updated network latency to %.3fs", self._network_latency)

    # ===================================
    #  9. Eviction and LRU Management
    # ===================================
    def _evict_lru(self):
        """Evict the least-recently-used key and return it."""

        with self._lock:
            return self._evict_lru_locked()

    # ==============================
    #  10. Private Key Handling
    # ==============================
    def _remove_key(self, key):
        """Removes a key and associated metadata without locking."""

        if key in self._data:
            self.current_memory -= self._size_by_key.get(key, 0)
            self.current_memory = max(0, self.current_memory)
            del self._data[key]
        self._expiration.pop(key, None)
        self._access_log.pop(key, None)
        self._tags.pop(key, None)
        self._size_by_key.pop(key, None)

    # ===================================
    #  11. Miscellaneous Magic Methods
    # ===================================
    def __len__(self):
        """Returns the exact count of non-expired keys, with live expiration checks."""

        with self._lock:
            self._cleanup_expired_locked(time.time())
            return len(self._data)

    def __contains__(self, key):
        """Checks if a non-expired key exists in memory using ``key in shared_memory``."""

        if key is None:
            return False
        try:
            hash(key)
        except Exception:
            return False
        with self._lock:
            current_time = time.time()
            if key in self._data and not self._is_expired(key, current_time):
                return True
            if key in self._data and self._is_expired(key, current_time):
                self._remove_key(key)
            return False

    def get_all_keys(self):
        """Return all non-expired keys."""

        self._record_operation("get_all_keys")
        with self._lock:
            self._cleanup_expired_locked(time.time())
            return list(self._data.keys())

    # ========== Others ============
    def configure(self, default_ttl=None, max_versions=10):
        """Update runtime TTL/version configuration."""

        self._record_operation("configure")
        with self._lock:
            if default_ttl is not None:
                self._default_ttl = self._validate_ttl(default_ttl)
            if max_versions is not None:
                validated = self._validate_max_versions(max_versions)
                if validated is not None and validated != self._max_versions:
                    self._max_versions = validated
                    self._resize_version_deques_locked()
            return {"default_ttl": self._default_ttl, "max_versions": self._max_versions}

    def increment(self, key, delta=1):
        """Atomically increment a numeric value."""

        self._record_operation("increment")
        self._ensure_open("increment")
        key = self._validate_key(key, operation="increment")
        try:
            numeric_delta = coerce_float(delta, default=0.0)
            with self._lock:
                current = self.get(key, update_access=False, default=0)
                if current is None:
                    current = 0
                if not isinstance(current, (int, float)) or isinstance(current, bool):
                    self._raise_memory_error(
                        "increment",
                        "current value is not numeric",
                        key=key,
                        context={"current_type": type(current).__name__},
                    )
                new_value = current + numeric_delta
                if isinstance(current, int) and isinstance(delta, int):
                    new_value = int(new_value)
                self.put(key, new_value)
                return new_value
        except CollaborationError: # type: ignore
            raise
        except Exception as exc:
            self._raise_memory_error("increment", "failed to increment value", key=key, cause=exc)

    def get_latest_snapshot(self):
        """Retrieve the most recent system snapshot with metadata."""

        self._record_operation("get_latest_snapshot")
        with self._lock:
            snapshot_keys = sorted(
                [key for key in self.get_all_keys() if str(key).startswith(self._snapshot_prefix)],
                key=lambda item: self.get_access_time(item) or 0,
                reverse=True,
            )
            if not snapshot_keys:
                logger.warning("No snapshots available in shared memory")
                return None

            latest_key = snapshot_keys[0]
            snapshot = self.get(latest_key)
            if snapshot is None:
                return None

            snapshot_meta = {
                "snapshot_id": latest_key,
                "timestamp": self.get_access_time(latest_key),
                "size_bytes": self._calculate_size(snapshot),
                "source": self.memory_config.get("snapshot_source", "collaborative_agent"),
                "fingerprint": stable_hash(snapshot, length=16),
            }
            return {"metadata": snapshot_meta, "data": snapshot}

    def log_intervention(self, report=None, human_input=None, timestamp=None):
        """Log intervention events with comprehensive metadata."""

        self._record_operation("log_intervention")
        if timestamp is None:
            timestamp_dt = datetime.now(timezone.utc)
        elif isinstance(timestamp, datetime):
            timestamp_dt = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp_dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)

        intervention_record = {
            "event_id": generate_correlation_id("intervention"),
            "timestamp": timestamp_dt.isoformat(),
            "report": sanitize_for_logging(report or {}),
            "human_input": sanitize_for_logging(human_input or {}),
            "system_state": {
                "memory_usage": self.get_usage_stats(),
                "active_threads": threading.active_count(),
                "priority_queue_size": len(self._priority_queue),
            },
            "diagnostics": {
                "memory_integrity": self.validate_integrity(),
                "expired_items": len([key for key in self._expiration if self._expiration[key] <= time.time()]),
            },
        }

        with self._lock:
            current_log = self.get(self._intervention_log_key, default=[])
            if not isinstance(current_log, list):
                current_log = []
            current_log.append(intervention_record)
            if len(current_log) > self._max_intervention_log_entries:
                current_log = current_log[-self._max_intervention_log_entries :]
            self.put(self._intervention_log_key, current_log, ttl=timedelta(days=365), notify=False)

        self.publish(
            channel="system_events",
            message={
                "event_type": "human_intervention",
                "timestamp": timestamp_dt.isoformat(),
                "severity": "critical",
                "event_id": intervention_record["event_id"],
            },
        )
        logger.info("Logged intervention event at %s", timestamp_dt.isoformat())
        return intervention_record

    def validate_integrity(self) -> dict:
        """Validate memory structure integrity."""

        self._record_operation("validate_integrity")
        with self._lock:
            data_keys = set(self._data.keys())
            access_keys = set(self._access_log.keys())
            expiration_keys = set(self._expiration.keys())
            size_keys = set(self._size_by_key.keys())
            tag_keys = set(self._tags.keys())
            queue_keys = [key for _, _, key in self._priority_queue]
            expected_size = sum(self._calculate_key_size_locked(key) for key in self._data)
            errors = []

            if not access_keys.issubset(data_keys):
                errors.append("access_log_contains_unknown_keys")
            if not expiration_keys.issubset(data_keys):
                errors.append("expiration_contains_unknown_keys")
            if not size_keys.issubset(data_keys):
                errors.append("size_index_contains_unknown_keys")
            if not tag_keys.issubset(data_keys):
                errors.append("tag_index_contains_unknown_keys")
            if any(key not in data_keys for key in queue_keys):
                errors.append("priority_queue_contains_unknown_keys")
            if abs(expected_size - self.current_memory) > max(256, int(expected_size * 0.01)):
                errors.append("memory_counter_mismatch")

            result = {
                "ok": not errors,
                "errors": errors,
                "data_consistency": access_keys.issubset(data_keys),
                "expiration_consistency": expiration_keys.issubset(data_keys),
                "priority_queue_valid": all(key in data_keys for key in queue_keys),
                "size_index_consistency": size_keys.issubset(data_keys),
                "tag_index_consistency": tag_keys.issubset(data_keys),
                "memory_counter_consistency": abs(expected_size - self.current_memory) <= max(256, int(expected_size * 0.01)),
                "total_items": len(self._data),
                "total_size_mb": self.current_memory / _BYTES_PER_MB,
                "expected_size_mb": expected_size / _BYTES_PER_MB,
            }
            return result

    def compact(self) -> dict:
        """Remove expired/stale metadata and recalculate memory counters."""

        self._record_operation("compact")
        with self._lock:
            expired_count = self._cleanup_expired_locked(time.time())
            self._clean_priority_queue_locked()
            self._access_log = OrderedDict((key, ts) for key, ts in self._access_log.items() if key in self._data)
            self._expiration = {key: expiry for key, expiry in self._expiration.items() if key in self._data}
            self._tags = defaultdict(list, {key: tags for key, tags in self._tags.items() if key in self._data})
            self._recalculate_memory_locked()
            return {"expired_removed": expired_count, "integrity": self.validate_integrity()}

    def snapshot_state(self, *, include_values: bool = True, redact: bool = True) -> dict:
        """Create a JSON-safe snapshot of current memory state."""

        self._record_operation("snapshot_state")
        with self._lock:
            self._cleanup_expired_locked(time.time())
            items = {}
            for key, versions in self._data.items():
                key_text = self._safe_key_repr(key)
                latest = versions[-1].value if versions else None
                items[key_text] = {
                    "version_count": len(versions),
                    "latest_timestamp": versions[-1].timestamp if versions else None,
                    "expires_at": self._expiration.get(key),
                    "last_access": self._access_log.get(key),
                    "tags": list(self._tags.get(key, [])),
                    "size_bytes": self._size_by_key.get(key, 0),
                }
                if include_values:
                    value = json_safe(latest)
                    items[key_text]["latest_value"] = sanitize_for_logging(value) if redact else value
            return {
                "snapshot_id": generate_correlation_id("shared-memory-snapshot"),
                "timestamp": utc_timestamp(),
                "stats": self.get_usage_stats(),
                "items": items,
            }

    def restore_snapshot(self, snapshot: Mapping[str, Any], *, clear_existing: bool = False) -> bool:
        """Restore values from a snapshot created with include_values=True.

        This intentionally restores only latest values from JSON-safe snapshots;
        full version history restoration is handled by ``load_from_file``.
        """

        self._record_operation("restore_snapshot")
        source = dict(snapshot or {})
        items = source.get("items", {})
        if not isinstance(items, Mapping):
            self._raise_memory_error("restore_snapshot", "snapshot items must be a mapping")
        with self._lock:
            if clear_existing:
                self.clear_all()
            for key_text, item in items.items():
                if not isinstance(item, Mapping) or "latest_value" not in item:
                    continue
                self.put(key_text, item.get("latest_value"), tags=item.get("tags"), notify=False)
        return True

    def close(self):
        """Stop background worker resources for this process-local instance."""

        self._record_operation("close")
        self._closed = True
        self._stop_cleaner_event.set()
        thread = self._cleaner_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        return True

    # ------------------------------------------------------------------
    # Internal mutation helpers. Caller must hold self._lock unless noted.
    # ------------------------------------------------------------------
    def _ensure_open(self, operation: str) -> None:
        if self._closed:
            self._raise_memory_error(operation, "shared memory instance is closed")

    def _normalize_ttl(self, value: Any, *, allow_immediate: bool) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, timedelta):
            seconds = value.total_seconds()
        else:
            try:
                seconds = float(value)
            except Exception as exc:
                self._raise_memory_error("validate_ttl", "TTL must be numeric seconds, timedelta, or None", cause=exc)
                return None   # unreachable but satisfies type checker
        if seconds <= 0:
            if allow_immediate:
                return 0.0
            self._raise_memory_error("validate_ttl", "TTL must be positive or None")
        return seconds

    def _store_version_locked(self, key: Hashable, value: Any, *, current_time: float) -> None:
        if key not in self._data:
            self._data[key] = deque(maxlen=self._max_versions)
        versions = self._data[key]
        versions.append(VersionedItem(timestamp=current_time, value=value))
        self._refresh_key_size_locked(key)
        self._touch_access_locked(key, current_time)

    def _set_expiration_locked(self, key: Hashable, ttl: Optional[float], *, current_time: float) -> None:
        effective_ttl = self._default_ttl if ttl is None else ttl
        if effective_ttl is None:
            self._expiration.pop(key, None)
        elif effective_ttl <= 0:
            self._remove_key(key)
        else:
            self._expiration[key] = current_time + effective_ttl

    def _touch_access_locked(self, key: Hashable, timestamp: Optional[float] = None) -> None:
        timestamp = time.time() if timestamp is None else timestamp
        if key in self._access_log:
            self._access_log.move_to_end(key)
        self._access_log[key] = timestamp

    def _refresh_key_size_locked(self, key: Hashable) -> None:
        old_size = self._size_by_key.get(key, 0)
        new_size = self._calculate_key_size_locked(key)
        self._size_by_key[key] = new_size
        self.current_memory += new_size - old_size
        self.current_memory = max(0, self.current_memory)

    def _calculate_key_size_locked(self, key: Hashable) -> int:
        versions = self._data.get(key)
        if not versions:
            return 0
        return sum(self._calculate_size(version.value) for version in versions)

    def _recalculate_memory_locked(self) -> None:
        self._size_by_key = {key: self._calculate_key_size_locked(key) for key in self._data}
        self.current_memory = sum(self._size_by_key.values())

    def _evict_until_within_limit_locked(self, *, protected_key: Optional[Hashable] = None) -> None:
        if self.current_memory <= self.max_memory:
            return
        if protected_key is not None and self._size_by_key.get(protected_key, 0) > self.max_memory and not self._allow_oversized_items:
            self._remove_key(protected_key)
            self._raise_memory_error(
                "evict",
                "item exceeds configured memory limit",
                key=protected_key,
                context={"max_memory_bytes": self.max_memory},
            )

        while self.current_memory > self.max_memory and self._access_log:
            candidate = next(iter(self._access_log))
            if candidate == protected_key and len(self._access_log) == 1:
                logger.warning(
                    "Shared memory item %s exceeds max memory but oversized items are allowed.",
                    self._safe_key_repr(candidate),
                )
                break
            self._evict_lru_locked(protected_key=protected_key)

    def _evict_lru_locked(self, protected_key: Optional[Hashable] = None) -> Optional[Hashable]:
        for key in list(self._access_log.keys()):
            if protected_key is not None and key == protected_key and len(self._access_log) > 1:
                continue
            if key in self._data:
                self._remove_key(key)
                self._last_eviction_time = time.time()
                self._last_evicted_key = self._safe_key_repr(key)
                self._clean_priority_queue_locked()
                logger.info("Evicted LRU shared-memory key: %s", self._last_evicted_key)
                return key
            self._access_log.pop(key, None)
        return None

    def _cleanup_expired_locked(self, current_time: float) -> int:
        expired_keys = [key for key, expiry in list(self._expiration.items()) if expiry <= current_time]
        for key in expired_keys:
            if key in self._data:
                latest = self._data[key][-1].value if self._data[key] else None
                self._remove_key(key)
                self._notify_change_locked(key, latest, event_type="expired")
            else:
                self._expiration.pop(key, None)
        self._last_cleanup_time = current_time
        self._last_cleanup_count = len(expired_keys)
        if expired_keys:
            self._clean_priority_queue_locked()
        return len(expired_keys)

    def _clean_priority_queue_locked(self) -> None:
        current_time = time.time()
        self._priority_queue = [
            item for item in self._priority_queue
            if item[2] in self._data and not self._is_expired(item[2], current_time)
        ]
        heapq.heapify(self._priority_queue)

    def _resize_version_deques_locked(self) -> None:
        for key, versions in list(self._data.items()):
            self._data[key] = deque(list(versions)[-self._max_versions :], maxlen=self._max_versions)
            self._refresh_key_size_locked(key)

    def _normalize_loaded_state_locked(self) -> None:
        normalized_data = {}
        for key, versions in dict(self._data).items():
            if isinstance(versions, deque):
                normalized_versions = deque(list(versions)[-self._max_versions :], maxlen=self._max_versions)
            elif isinstance(versions, list):
                normalized_versions = deque(versions[-self._max_versions :], maxlen=self._max_versions)
            else:
                normalized_versions = deque([VersionedItem(timestamp=time.time(), value=versions)], maxlen=self._max_versions)
            normalized_data[key] = normalized_versions
        self._data = normalized_data
        self._access_log = OrderedDict((key, ts) for key, ts in self._access_log.items() if key in self._data)
        for key in self._data:
            self._access_log.setdefault(key, time.time())
        self._recalculate_memory_locked()

    def _normalize_channel(self, channel: Any) -> str:
        text = str(channel or "").strip()
        if not text:
            self._raise_memory_error("channel", "channel must be a non-empty string")
        return text

    def _validate_callback(self, callback: Any, *, operation: str) -> None:
        if not callable(callback):
            self._raise_memory_error(operation, "callback must be callable", context={"callback_type": type(callback).__name__})

    def _matching_subscribers_locked(self, channel: str) -> list[Callable[[Any], Any]]:
        callbacks: list[Callable[[Any], Any]] = []
        direct = self.subscribers.get(channel, [])
        callbacks.extend(list(direct))
        for subscribed_channel, subscribed_callbacks in list(self.subscribers.items()):
            if subscribed_channel == channel:
                continue
            if fnmatch.fnmatch(channel, subscribed_channel):
                callbacks.extend(list(subscribed_callbacks))
        return callbacks

    def _dispatch_callback(self, callback: Callable[[Any], Any], value: Any) -> None:
        if self._async_callbacks:
            thread = threading.Thread(
                target=self._safe_callback_call,
                args=(callback, value),
                daemon=self._callback_threads_daemon,
            )
            thread.start()
        else:
            self._safe_callback_call(callback, value)

    def _notify_change_locked(
        self,
        key: Hashable,
        value: Any,
        *,
        event_type: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        event = {
            "event_type": event_type,
            "key": self._safe_key_repr(key),
            "timestamp": utc_timestamp(),
            "metadata": normalize_metadata(metadata or {}, drop_none=True),
            "value_fingerprint": stable_hash(value, length=16) if value is not None else None,
        }
        callbacks = list(self.callbacks.get(str(key), []))
        callbacks.extend(self._matching_subscribers_locked(str(key)))
        callbacks.extend(self._matching_subscribers_locked("memory_events"))
        for callback in callbacks:
            self._dispatch_callback(callback, event)


class _SharedMemoryManager(BaseManager):
    pass


_SharedMemoryManager.register(
    "SharedMemory",
    callable=SharedMemory,
    proxytype=SharedMemoryProxy,
)


class SharedMemoryManager:
    """Manager for shared memory instances."""

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._manager = None
            cls._instance._shared_memory = None
        return cls._instance

    def start(self, address=_DEFAULT_MANAGER_ADDRESS, authkey=_DEFAULT_MANAGER_AUTHKEY):
        if self._manager:
            return
        self._manager = _SharedMemoryManager(address=address, authkey=authkey)
        self._manager.start()
        self._shared_memory = self._manager.SharedMemory() # type: ignore
        return self._shared_memory

    def connect(self, address=_DEFAULT_MANAGER_ADDRESS, authkey=_DEFAULT_MANAGER_AUTHKEY):
        class _RemoteManager(BaseManager):
            pass

        _RemoteManager.register("SharedMemory")
        manager = _RemoteManager(address=address, authkey=authkey)
        manager.connect()
        self._manager = manager
        self._shared_memory = manager.SharedMemory() # type: ignore
        return self._shared_memory

    def get_shared_memory(self):
        if not self._manager:
            self.start()
        return self._shared_memory

    def shutdown(self):
        if self._manager:
            self._manager.shutdown()
            self._manager = None
            self._shared_memory = None
        return True


# Global access point
def get_shared_memory():
    return SharedMemoryManager().get_shared_memory()


__all__ = [
    "VersionedItem",
    "SharedMemoryProxy",
    "SharedMemory",
    "SharedMemoryManager",
    "get_shared_memory",
]


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("\n=== Running Shared Memory ===\n")
    printer.status("TEST", "Shared Memory initialized", "info")

    sm = SharedMemory()
    sm.clear_all()
    sm.configure(default_ttl=60, max_versions=3)

    # Basic set/get with versioning.
    ts1 = sm.set("k1", {"v": 1}, ttl=10, tags=["unit", "versioned"])
    time.sleep(0.001)
    sm.set("k1", {"v": 2}, ttl=10, tags=["unit", "versioned"])
    assert sm.get("k1") == {"v": 2}
    assert sm.get("k1", version_timestamp=ts1) == {"v": 1}
    assert len(sm.get_all_versions("k1")) == 2
    assert sm.get_by_tag("unit", limit=5)

    # TTL expiration.
    sm.set("temp", "x", ttl=0.01)
    time.sleep(0.03)
    assert sm.get("temp") is None
    assert sm.cleanup_expired() >= 0

    # Compare-and-swap and increment.
    sm.set("counter", 1)
    assert sm.compare_and_swap("counter", 1, 2) is True
    assert sm.compare_and_swap("counter", 1, 3) is False
    assert sm.increment("counter", 3) == 5

    # Priority queue retrieval.
    sm.put("low", "L", priority=1)
    sm.put("high", "H", priority=5)
    top = sm.get_next_prioritized_item()
    assert top == (5.0, "high")

    # Callback and pub/sub behavior.
    callback_events = []
    subscriber_events = []
    sm.register_callback("callback_key", lambda event: callback_events.append(event))
    sm.subscribe("memory_events", lambda event: subscriber_events.append(event), once=False)
    sm.set("callback_key", {"ok": True})
    time.sleep(0.05)
    assert callback_events
    assert subscriber_events

    # Snapshot and intervention logging.
    sm.set("snapshots:test", {"state": "ok"}, ttl=60)
    latest_snapshot = sm.get_latest_snapshot()
    assert latest_snapshot and latest_snapshot["data"] == {"state": "ok"}
    intervention = sm.log_intervention(report={"reason": "test"}, human_input={"approved": True})
    assert intervention["event_id"]

    # Persistence.
    test_path = Path("report/shared_memory_test.pkl")
    saved_path = sm.save_to_file(test_path)
    assert Path(saved_path).exists() # type: ignore
    sm.clear_all()
    assert sm.get("k1") is None
    assert sm.load_from_file(test_path) is True
    assert sm.get("counter") == 5

    # Integrity, metrics, and compacting.
    integrity = sm.validate_integrity()
    assert integrity["ok"], integrity
    stats = sm.get_usage_stats()
    assert "current_memory_mb" in stats and "item_count" in stats
    metrics = sm.metrics()
    assert "operation_counts" in metrics
    health = sm.health_check()
    assert health["ok"]
    compacted = sm.compact()
    assert compacted["integrity"]["ok"]

    # Snapshot export and restore latest values.
    exported = sm.snapshot_state(include_values=True)
    assert exported["items"]
    sm.restore_snapshot(exported, clear_existing=True)
    assert len(sm) > 0

    printer.status("TEST", "Shared Memory tests completed", "success")
    print("\n=== Test ran successfully ===\n")
