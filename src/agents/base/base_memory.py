"""
Base memory subsystem for the Base Agent stack.

This module provides the production-grade in-process memory layer used by the
base agent subsystem. It is responsible for maintaining structured runtime
state, short- and medium-lived contextual facts, transient working memory,
namespaced key-value storage, bounded retention, TTL-based expiry, metadata and
tag-aware retrieval, snapshot persistence, and operational telemetry.

The implementation is intentionally generic so it can support agent
coordination, task context, error memory, prompt-state handoff, cached
intermediate results, lightweight session persistence, and other subsystem
concerns without forcing those higher-level modules to reimplement storage,
validation, serialisation, cleanup, and consistency policies.

Key design goals:
- deterministic and auditable memory records with explicit metadata
- strong validation and structured error handling using the base error stack
- bounded growth through capacity controls and cleanup policies
- safe persistence and export semantics for diagnostics and recovery
- namespaced isolation so multiple subsystems can share one memory backend
- practical ergonomics for scalar, mapping, and sequence-oriented memory flows
"""

from __future__ import annotations

import os
import copy

from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.base_errors import *
from .utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # type: ignore

logger = get_logger("Base Memory")
printer = PrettyPrinter


@dataclass
class MemoryEntry:
    """Structured record stored inside ``BaseMemory``."""

    key: str
    value: Any
    namespace: str
    created_at: str
    updated_at: str
    version: int = 1
    expires_at: Optional[str] = None
    last_accessed_at: Optional[str] = None
    access_count: int = 0
    tags: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Optional[str] = None
    value_type: str = "unknown"
    size_estimate_bytes: int = 0
    source: Optional[str] = None
    persistent: bool = True

    def is_expired(self, reference_time: Optional[datetime] = None) -> bool:
        if not self.expires_at:
            return False
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        try:
            expiry_dt = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return False
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        return reference_time >= expiry_dt

    def touch(self, touched_at: Optional[str] = None) -> None:
        current = touched_at or utc_now_iso()
        self.last_accessed_at = current
        self.access_count += 1

    def to_dict(self, *, include_value: bool = True, redact: bool = False) -> Dict[str, Any]:
        payload = {
            "key": self.key,
            "namespace": self.namespace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "expires_at": self.expires_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "tags": list(self.tags),
            "metadata": redact_mapping(self.metadata) if redact else to_json_safe(self.metadata),
            "fingerprint": self.fingerprint,
            "value_type": self.value_type,
            "size_estimate_bytes": self.size_estimate_bytes,
            "source": self.source,
            "persistent": self.persistent,
        }
        if include_value:
            value = to_json_safe(self.value)
            payload["value"] = redact_mapping(value) if redact and isinstance(value, ABCMapping) else value
        return payload


@dataclass(frozen=True)
class MemoryStats:
    total_entries: int
    total_namespaces: int
    expired_entries: int
    history_length: int
    max_entries: int
    max_entries_per_namespace: int
    cleanup_runs: int
    evictions: int
    writes: int
    reads: int
    deletes: int
    hits: int
    misses: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "total_namespaces": self.total_namespaces,
            "expired_entries": self.expired_entries,
            "history_length": self.history_length,
            "max_entries": self.max_entries,
            "max_entries_per_namespace": self.max_entries_per_namespace,
            "cleanup_runs": self.cleanup_runs,
            "evictions": self.evictions,
            "writes": self.writes,
            "reads": self.reads,
            "deletes": self.deletes,
            "hits": self.hits,
            "misses": self.misses,
        }


class BaseMemory:
    """Thread-safe namespaced memory backend for the base agent subsystem."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("base_memory") or {}
        self._lock = RLock()
        self._started_at = utc_now_iso()
        self._memory_id = generate_request_id("base_memory")

        self.default_namespace = self._get_config_str("default_namespace", "default", normalize=True)
        self.max_entries = self._get_config_int("max_entries", 10000, minimum=1)
        self.max_entries_per_namespace = self._get_config_int("max_entries_per_namespace", 2500, minimum=1)
        self.default_ttl_seconds = self._get_optional_positive_float("default_ttl_seconds", None)
        self.cleanup_interval_seconds = self._get_config_float("cleanup_interval_seconds", 60.0, minimum=0.0)
        self.enable_history = self._get_config_bool("enable_history", True)
        self.history_limit = self._get_config_int("history_limit", 500, minimum=1)
        self.enable_persistence = self._get_config_bool("enable_persistence", False)
        self.auto_persist = self._get_config_bool("auto_persist", False)
        self.auto_load_persistence = self._get_config_bool("auto_load_persistence", False)
        self.persistence_path = self.memory_config.get("persistence_path", "data/base_memory_snapshot.json")
        self.enable_fingerprints = self._get_config_bool("enable_fingerprints", True)
        self.enforce_capacity_on_write = self._get_config_bool("enforce_capacity_on_write", True)
        self.export_redact_secrets = self._get_config_bool("export_redact_secrets", False)
        self.snapshot_pretty = self._get_config_bool("snapshot_pretty", True)
        self.max_query_results = self._get_config_int("max_query_results", 100, minimum=1)
        self.eviction_policy = self.memory_config.get("eviction_policy", "least_recently_used")
        ensure_one_of(
            self.eviction_policy,
            ["least_recently_used", "least_frequently_used", "oldest"],
            "eviction_policy",
            error_cls=BaseConfigurationError,
            config=self.memory_config,
        )

        self._store: Dict[str, OrderedDict[str, MemoryEntry]] = {}
        self._tag_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._type_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)
        self._last_cleanup_monotonic = monotonic_seconds()
        self._cleanup_runs = 0
        self._evictions = 0
        self._writes = 0
        self._reads = 0
        self._deletes = 0
        self._hits = 0
        self._misses = 0

        if self.enable_persistence and self.auto_load_persistence:
            try:
                self.load_snapshot(self.persistence_path, merge=True, must_exist=False)
            except Exception as exc:
                raise BaseInitializationError.wrap(
                    exc,
                    message="BaseMemory failed to auto-load its persistence snapshot.",
                    config=self.memory_config,
                    component="BaseMemory",
                    operation="__init__",
                    context={"persistence_path": self.persistence_path},
                ) from exc

        logger.info("Base Memory successfully initialized")

    def _get_config_bool(self, key: str, default: bool) -> bool:
        return coerce_bool(self.memory_config.get(key, default), default=default)

    def _get_config_int(self, key: str, default: int, *, minimum: Optional[int] = None) -> int:
        value = coerce_int(self.memory_config.get(key, default), default=default, minimum=minimum)
        if minimum is not None and value < minimum:
            raise BaseConfigurationError(
                f"Configuration value '{key}' must be >= {minimum}.",
                self.memory_config,
                component="BaseMemory",
                operation="configuration",
                context={"key": key, "value": value, "minimum": minimum},
            )
        return value

    def _get_config_float(self, key: str, default: float, *, minimum: Optional[float] = None) -> float:
        value = coerce_float(self.memory_config.get(key, default), default=default, minimum=minimum)
        if minimum is not None and value < minimum:
            raise BaseConfigurationError(
                f"Configuration value '{key}' must be >= {minimum}.",
                self.memory_config,
                component="BaseMemory",
                operation="configuration",
                context={"key": key, "value": value, "minimum": minimum},
            )
        return value

    def _get_optional_positive_float(self, key: str, default: Optional[float]) -> Optional[float]:
        raw = self.memory_config.get(key, default)
        if raw in (None, "", "none", "None"):
            return None
        value = coerce_float(raw, default=default or 0.0)
        if value < 0:
            raise BaseConfigurationError(
                f"Configuration value '{key}' must be non-negative or null.",
                self.memory_config,
                component="BaseMemory",
                operation="configuration",
                context={"key": key, "value": raw},
            )
        return value

    def _get_config_str(self, key: str, default: str, *, normalize: bool = False) -> str:
        value = ensure_non_empty_string(self.memory_config.get(key, default), key, config=self.memory_config, error_cls=BaseConfigurationError)
        return normalize_identifier(value, lowercase=True) if normalize else value

    def _normalize_namespace(self, namespace: Optional[str]) -> str:
        value = namespace or self.default_namespace
        return normalize_identifier(value, lowercase=True, separator="_", max_length=120)

    def _normalize_key(self, key: str) -> str:
        normalized = ensure_non_empty_string(key, "key", config=self.memory_config, error_cls=BaseValidationError)
        if len(normalized) > 256:
            raise BaseValidationError(
                "Memory keys must be 256 characters or fewer.",
                self.memory_config,
                component="BaseMemory",
                operation="normalize_key",
                context={"key_length": len(normalized), "key": normalized[:128]},
            )
        return normalized

    def _normalize_tags(self, tags: Optional[Iterable[str]]) -> Tuple[str, ...]:
        normalized: List[str] = []
        for tag in ensure_list(tags):
            text = normalize_identifier(tag, lowercase=True, separator="_", max_length=64)
            if text not in normalized:
                normalized.append(text)
        return tuple(normalized)

    def _normalize_metadata(self, metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if metadata is None:
            return {}
        ensure_mapping(metadata, "metadata", config=self.memory_config, error_cls=BaseValidationError)
        return drop_none_values(dict(metadata), recursive=True, drop_empty=False)

    def _namespace_store(self, namespace: str, *, create: bool = True) -> OrderedDict[str, MemoryEntry]:
        if namespace not in self._store:
            if not create:
                raise BaseStateError(
                    f"Namespace '{namespace}' does not exist.",
                    self.memory_config,
                    component="BaseMemory",
                    operation="namespace_lookup",
                    context={"namespace": namespace},
                )
            self._store[namespace] = OrderedDict()
        return self._store[namespace]

    def _estimate_size_bytes(self, value: Any) -> int:
        try:
            return len(json_dumps(value, pretty=False, sort_keys=True).encode("utf-8"))
        except Exception:
            return len(safe_repr(value).encode("utf-8"))

    def _compute_fingerprint(self, value: Any) -> Optional[str]:
        if not self.enable_fingerprints:
            return None
        try:
            return stable_fingerprint(value, algorithm="sha256", length=32)
        except Exception:
            return None

    def _compute_expiry(self, ttl_seconds: Optional[float]) -> Optional[str]:
        effective_ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if effective_ttl in (None, "", "none", "None"):
            return None
        ttl_value = ensure_numeric_range(
            effective_ttl,
            "ttl_seconds",
            minimum=0.0,
            inclusive=True,
            config=self.memory_config,
            error_cls=BaseValidationError,
        )
        expiry = utc_now() + timedelta(seconds=ttl_value)
        return expiry.isoformat()

    def _compose_ref(self, namespace: str, key: str) -> Tuple[str, str]:
        return namespace, key

    def _index_entry(self, entry: MemoryEntry) -> None:
        ref = self._compose_ref(entry.namespace, entry.key)
        for tag in entry.tags:
            self._tag_index[tag].add(ref)
        self._type_index[entry.value_type].add(ref)

    def _deindex_entry(self, entry: MemoryEntry) -> None:
        ref = self._compose_ref(entry.namespace, entry.key)
        for tag in entry.tags:
            bucket = self._tag_index.get(tag)
            if bucket is not None:
                bucket.discard(ref)
                if not bucket:
                    self._tag_index.pop(tag, None)
        type_bucket = self._type_index.get(entry.value_type)
        if type_bucket is not None:
            type_bucket.discard(ref)
            if not type_bucket:
                self._type_index.pop(entry.value_type, None)

    def _record_history(self, action: str, *, namespace: str, key: Optional[str] = None, details: Optional[Mapping[str, Any]] = None) -> None:
        if not self.enable_history:
            return
        payload = {
            "timestamp": utc_now_iso(),
            "action": action,
            "namespace": namespace,
            "key": key,
            "details": to_json_safe(details or {}),
        }
        self._history.append(payload)

    def _maybe_cleanup_locked(self, force: bool = False) -> int:
        now_monotonic = monotonic_seconds()
        if not force and self.cleanup_interval_seconds > 0 and (now_monotonic - self._last_cleanup_monotonic) < self.cleanup_interval_seconds:
            return 0
        removed = self._cleanup_expired_locked()
        self._last_cleanup_monotonic = now_monotonic
        return removed

    def _cleanup_expired_locked(self) -> int:
        removed = 0
        now = utc_now()
        for namespace, bucket in list(self._store.items()):
            for key, entry in list(bucket.items()):
                if entry.is_expired(now):
                    self._deindex_entry(entry)
                    del bucket[key]
                    removed += 1
            if not bucket:
                self._store.pop(namespace, None)
        self._cleanup_runs += 1
        if removed:
            self._record_history("cleanup_expired", namespace="*", details={"removed": removed})
        return removed

    def _select_eviction_candidate(self, namespace: Optional[str] = None) -> Optional[Tuple[str, str, MemoryEntry]]:
        candidates: List[Tuple[str, str, MemoryEntry]] = []
        namespaces = [namespace] if namespace else list(self._store.keys())
        for current_namespace in namespaces:
            bucket = self._store.get(current_namespace, OrderedDict())
            for key, entry in bucket.items():
                candidates.append((current_namespace, key, entry))
        if not candidates:
            return None

        def sort_key(item: Tuple[str, str, MemoryEntry]) -> Tuple[Any, ...]:
            _, _, entry = item
            if self.eviction_policy == "least_frequently_used":
                return (entry.access_count, entry.last_accessed_at or entry.updated_at, entry.created_at)
            if self.eviction_policy == "oldest":
                return (entry.created_at, entry.last_accessed_at or entry.updated_at, entry.access_count)
            return (entry.last_accessed_at or entry.updated_at, entry.access_count, entry.created_at)

        candidates.sort(key=sort_key)
        return candidates[0]

    def _evict_one_locked(self, namespace: Optional[str] = None) -> bool:
        candidate = self._select_eviction_candidate(namespace=namespace)
        if candidate is None:
            return False
        ns, key, entry = candidate
        bucket = self._store.get(ns)
        if bucket is None or key not in bucket:
            return False
        self._deindex_entry(entry)
        del bucket[key]
        if not bucket:
            self._store.pop(ns, None)
        self._evictions += 1
        self._record_history("evict", namespace=ns, key=key, details={"policy": self.eviction_policy})
        return True

    def _enforce_capacity_locked(self, namespace: Optional[str] = None) -> None:
        if not self.enforce_capacity_on_write:
            return
        while self.total_entries > self.max_entries:
            if not self._evict_one_locked(namespace=None):
                raise BaseResourceError(
                    "Global memory capacity exceeded, but no eviction candidate was available.",
                    self.memory_config,
                    component="BaseMemory",
                    operation="capacity_enforcement",
                    context={"total_entries": self.total_entries, "max_entries": self.max_entries},
                )
        if namespace:
            bucket = self._store.get(namespace, OrderedDict())
            while len(bucket) > self.max_entries_per_namespace:
                if not self._evict_one_locked(namespace=namespace):
                    raise BaseResourceError(
                        "Namespace memory capacity exceeded, but no eviction candidate was available.",
                        self.memory_config,
                        component="BaseMemory",
                        operation="capacity_enforcement",
                        context={"namespace": namespace, "namespace_entries": len(bucket), "max_entries_per_namespace": self.max_entries_per_namespace},
                    )
                bucket = self._store.get(namespace, OrderedDict())

    def _build_entry(
        self,
        *,
        key: str,
        namespace: str,
        value: Any,
        metadata: Optional[Mapping[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        ttl_seconds: Optional[float] = None,
        source: Optional[str] = None,
        persistent: bool = True,
        previous: Optional[MemoryEntry] = None,
    ) -> MemoryEntry:
        now = utc_now_iso()
        normalized_metadata = self._normalize_metadata(metadata)
        normalized_tags = self._normalize_tags(tags)
        return MemoryEntry(
            key=key,
            value=copy.deepcopy(value),
            namespace=namespace,
            created_at=previous.created_at if previous else now,
            updated_at=now,
            version=(previous.version + 1) if previous else 1,
            expires_at=self._compute_expiry(ttl_seconds),
            last_accessed_at=previous.last_accessed_at if previous else now,
            access_count=previous.access_count if previous else 0,
            tags=normalized_tags,
            metadata=normalized_metadata,
            fingerprint=self._compute_fingerprint(value),
            value_type=type(value).__name__,
            size_estimate_bytes=self._estimate_size_bytes(value),
            source=source,
            persistent=bool(persistent),
        )

    @property
    def total_entries(self) -> int:
        return sum(len(bucket) for bucket in self._store.values())

    @property
    def total_namespaces(self) -> int:
        return len(self._store)

    def __len__(self) -> int:
        return self.total_entries

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self.has(key)

    def __repr__(self) -> str:
        return f"<BaseMemory id={self._memory_id} namespaces={self.total_namespaces} entries={self.total_entries}>"

    def put(
        self,
        key: str,
        value: Any,
        *,
        namespace: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        ttl_seconds: Optional[float] = None,
        source: Optional[str] = None,
        persistent: bool = True,
    ) -> MemoryEntry:
        normalized_key = self._normalize_key(key)
        normalized_namespace = self._normalize_namespace(namespace)
        with self._lock:
            self._maybe_cleanup_locked()
            bucket = self._namespace_store(normalized_namespace, create=True)
            previous = bucket.get(normalized_key)
            entry = self._build_entry(
                key=normalized_key,
                namespace=normalized_namespace,
                value=value,
                metadata=metadata,
                tags=tags,
                ttl_seconds=ttl_seconds,
                source=source,
                persistent=persistent,
                previous=previous,
            )
            if previous is not None:
                self._deindex_entry(previous)
            bucket[normalized_key] = entry
            bucket.move_to_end(normalized_key)
            self._index_entry(entry)
            self._writes += 1
            self._record_history("put", namespace=normalized_namespace, key=normalized_key, details={"version": entry.version, "ttl_seconds": ttl_seconds, "tags": list(entry.tags)})
            self._enforce_capacity_locked(namespace=normalized_namespace)
            if self.enable_persistence and self.auto_persist and entry.persistent:
                self.save_snapshot(self.persistence_path)
            return copy.deepcopy(entry)

    set = put

    def set_many(
        self,
        items: Mapping[str, Any],
        *,
        namespace: Optional[str] = None,
        metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
        tags: Optional[Mapping[str, Iterable[str]]] = None,
        ttl_seconds: Optional[float] = None,
        source: Optional[str] = None,
        persistent: bool = True,
    ) -> Dict[str, MemoryEntry]:
        ensure_mapping(items, "items", config=self.memory_config, error_cls=BaseValidationError)
        created: Dict[str, MemoryEntry] = {}
        for item_key, item_value in items.items():
            created[str(item_key)] = self.put(
                str(item_key),
                item_value,
                namespace=namespace,
                metadata=(metadata or {}).get(str(item_key)) if metadata else None,
                tags=(tags or {}).get(str(item_key)) if tags else None,
                ttl_seconds=ttl_seconds,
                source=source,
                persistent=persistent,
            )
        return created

    def _lookup_entry_locked(self, namespace: str, key: str, *, delete_if_expired: bool = True) -> Optional[MemoryEntry]:
        bucket = self._store.get(namespace)
        if bucket is None:
            return None
        entry = bucket.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            if delete_if_expired:
                self._deindex_entry(entry)
                del bucket[key]
                if not bucket:
                    self._store.pop(namespace, None)
            return None
        return entry

    def get(self, key: str, default: Any = None, *, namespace: Optional[str] = None, touch: bool = True, return_entry: bool = False) -> Any:
        normalized_key = self._normalize_key(key)
        normalized_namespace = self._normalize_namespace(namespace)
        with self._lock:
            self._reads += 1
            self._maybe_cleanup_locked()
            entry = self._lookup_entry_locked(normalized_namespace, normalized_key)
            if entry is None:
                self._misses += 1
                return default
            self._hits += 1
            if touch:
                entry.touch()
                entry.updated_at = entry.updated_at
                bucket = self._namespace_store(normalized_namespace, create=False)
                bucket.move_to_end(normalized_key)
            return copy.deepcopy(entry if return_entry else entry.value)

    def require(self, key: str, *, namespace: Optional[str] = None, touch: bool = True, return_entry: bool = False) -> Any:
        value = self.get(key, namespace=namespace, touch=touch, return_entry=return_entry, default=None)
        if value is None and not self.has(key, namespace=namespace):
            raise BaseStateError(
                f"Required memory key '{key}' was not found.",
                self.memory_config,
                component="BaseMemory",
                operation="require",
                context={"key": key, "namespace": self._normalize_namespace(namespace)},
            )
        return value

    def get_entry(self, key: str, *, namespace: Optional[str] = None, touch: bool = True) -> Optional[MemoryEntry]:
        entry = self.get(key, namespace=namespace, touch=touch, return_entry=True, default=None)
        return entry

    def has(self, key: str, *, namespace: Optional[str] = None) -> bool:
        normalized_key = self._normalize_key(key)
        normalized_namespace = self._normalize_namespace(namespace)
        with self._lock:
            return self._lookup_entry_locked(normalized_namespace, normalized_key) is not None

    def delete(self, key: str, *, namespace: Optional[str] = None) -> bool:
        normalized_key = self._normalize_key(key)
        normalized_namespace = self._normalize_namespace(namespace)
        with self._lock:
            bucket = self._store.get(normalized_namespace)
            if bucket is None or normalized_key not in bucket:
                return False
            entry = bucket[normalized_key]
            self._deindex_entry(entry)
            del bucket[normalized_key]
            if not bucket:
                self._store.pop(normalized_namespace, None)
            self._deletes += 1
            self._record_history("delete", namespace=normalized_namespace, key=normalized_key)
            if self.enable_persistence and self.auto_persist and entry.persistent:
                self.save_snapshot(self.persistence_path)
            return True

    def pop(self, key: str, default: Any = None, *, namespace: Optional[str] = None, return_entry: bool = False) -> Any:
        normalized_key = self._normalize_key(key)
        normalized_namespace = self._normalize_namespace(namespace)
        with self._lock:
            bucket = self._store.get(normalized_namespace)
            if bucket is None:
                return default
            entry = bucket.get(normalized_key)
            if entry is None or entry.is_expired():
                if entry is not None:
                    self._deindex_entry(entry)
                    del bucket[normalized_key]
                if bucket is not None and not bucket:
                    self._store.pop(normalized_namespace, None)
                return default
            self._deindex_entry(entry)
            del bucket[normalized_key]
            if not bucket:
                self._store.pop(normalized_namespace, None)
            self._deletes += 1
            self._record_history("pop", namespace=normalized_namespace, key=normalized_key)
            return copy.deepcopy(entry if return_entry else entry.value)

    def clear(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace is None:
                removed = self.total_entries
                self._store.clear()
                self._tag_index.clear()
                self._type_index.clear()
                self._record_history("clear_all", namespace="*")
                return removed
            normalized_namespace = self._normalize_namespace(namespace)
            bucket = self._store.pop(normalized_namespace, OrderedDict())
            for entry in bucket.values():
                self._deindex_entry(entry)
            removed = len(bucket)
            if removed:
                self._record_history("clear_namespace", namespace=normalized_namespace, details={"removed": removed})
            return removed

    def touch(self, key: str, *, namespace: Optional[str] = None, ttl_seconds: Optional[float] = None) -> MemoryEntry:
        normalized_key = self._normalize_key(key)
        normalized_namespace = self._normalize_namespace(namespace)
        with self._lock:
            entry = self._lookup_entry_locked(normalized_namespace, normalized_key)
            if entry is None:
                raise BaseStateError(
                    f"Cannot touch missing memory key '{normalized_key}'.",
                    self.memory_config,
                    component="BaseMemory",
                    operation="touch",
                    context={"key": normalized_key, "namespace": normalized_namespace},
                )
            entry.touch()
            if ttl_seconds is not None:
                entry.expires_at = self._compute_expiry(ttl_seconds)
            self._record_history("touch", namespace=normalized_namespace, key=normalized_key, details={"ttl_seconds": ttl_seconds})
            return copy.deepcopy(entry)

    def extend_ttl(self, key: str, additional_seconds: float, *, namespace: Optional[str] = None) -> MemoryEntry:
        additional = ensure_numeric_range(additional_seconds, "additional_seconds", minimum=0.0, config=self.memory_config, error_cls=BaseValidationError)
        normalized_key = self._normalize_key(key)
        normalized_namespace = self._normalize_namespace(namespace)
        with self._lock:
            entry = self._lookup_entry_locked(normalized_namespace, normalized_key)
            if entry is None:
                raise BaseStateError(
                    f"Cannot extend TTL for missing memory key '{normalized_key}'.",
                    self.memory_config,
                    component="BaseMemory",
                    operation="extend_ttl",
                    context={"key": normalized_key, "namespace": normalized_namespace},
                )
            current_expiry = datetime.fromisoformat(entry.expires_at) if entry.expires_at else utc_now()
            if current_expiry.tzinfo is None:
                current_expiry = current_expiry.replace(tzinfo=timezone.utc)
            entry.expires_at = (current_expiry + timedelta(seconds=additional)).isoformat()
            self._record_history("extend_ttl", namespace=normalized_namespace, key=normalized_key, details={"additional_seconds": additional})
            return copy.deepcopy(entry)

    def increment(self, key: str, amount: float = 1.0, *, namespace: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None, tags: Optional[Iterable[str]] = None) -> MemoryEntry:
        delta = ensure_numeric_range(amount, "amount", config=self.memory_config, error_cls=BaseValidationError)
        current = self.get(key, default=0, namespace=namespace)
        if not isinstance(current, (int, float)):
            raise BaseStateError(
                f"Cannot increment non-numeric memory key '{key}'.",
                self.memory_config,
                component="BaseMemory",
                operation="increment",
                context={"key": key, "current_type": type(current).__name__},
            )
        return self.put(key, current + delta, namespace=namespace, metadata=metadata, tags=tags)

    def decrement(self, key: str, amount: float = 1.0, *, namespace: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None, tags: Optional[Iterable[str]] = None) -> MemoryEntry:
        return self.increment(key, -float(amount), namespace=namespace, metadata=metadata, tags=tags)

    def append(self, key: str, item: Any, *, namespace: Optional[str] = None, unique: bool = False, metadata: Optional[Mapping[str, Any]] = None, tags: Optional[Iterable[str]] = None) -> MemoryEntry:
        current = self.get(key, default=None, namespace=namespace)
        if current is None:
            sequence: List[Any] = []
        elif isinstance(current, list):
            sequence = list(current)
        elif isinstance(current, tuple):
            sequence = list(current)
        else:
            raise BaseStateError(
                f"Cannot append to non-sequence memory key '{key}'.",
                self.memory_config,
                component="BaseMemory",
                operation="append",
                context={"key": key, "current_type": type(current).__name__},
            )
        if unique and item in sequence:
            return self.get_entry(key, namespace=namespace) or self.put(key, sequence, namespace=namespace, metadata=metadata, tags=tags)
        sequence.append(copy.deepcopy(item))
        return self.put(key, sequence, namespace=namespace, metadata=metadata, tags=tags)

    def merge(self, key: str, value: Mapping[str, Any], *, namespace: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None, tags: Optional[Iterable[str]] = None, deep: bool = True) -> MemoryEntry:
        ensure_mapping(value, "value", config=self.memory_config, error_cls=BaseValidationError)
        current = self.get(key, default=None, namespace=namespace)
        if current is None:
            merged: Dict[str, Any] = dict(value)
        elif isinstance(current, ABCMapping):
            merged = deep_merge_dicts(current, value) if deep else {**dict(current), **dict(value)}
        else:
            raise BaseStateError(
                f"Cannot merge mapping into non-mapping memory key '{key}'.",
                self.memory_config,
                component="BaseMemory",
                operation="merge",
                context={"key": key, "current_type": type(current).__name__},
            )
        return self.put(key, merged, namespace=namespace, metadata=metadata, tags=tags)

    def keys(self, *, namespace: Optional[str] = None, include_expired: bool = False) -> List[str]:
        with self._lock:
            if namespace is None:
                result: List[str] = []
                for ns in list(self._store.keys()):
                    result.extend(self.keys(namespace=ns, include_expired=include_expired))
                return result
            normalized_namespace = self._normalize_namespace(namespace)
            bucket = self._store.get(normalized_namespace, OrderedDict())
            if include_expired:
                return list(bucket.keys())
            return [key for key, entry in bucket.items() if not entry.is_expired()]

    def namespaces(self) -> List[str]:
        with self._lock:
            return sorted(self._store.keys())

    def items(self, *, namespace: Optional[str] = None, include_expired: bool = False, return_entries: bool = False) -> List[Tuple[str, Any]]:
        with self._lock:
            if namespace is None:
                result: List[Tuple[str, Any]] = []
                for ns in self.namespaces():
                    result.extend(self.items(namespace=ns, include_expired=include_expired, return_entries=return_entries))
                return result
            normalized_namespace = self._normalize_namespace(namespace)
            bucket = self._store.get(normalized_namespace, OrderedDict())
            output: List[Tuple[str, Any]] = []
            for key, entry in bucket.items():
                if not include_expired and entry.is_expired():
                    continue
                output.append((key, copy.deepcopy(entry if return_entries else entry.value)))
            return output

    def search(
        self,
        *,
        namespace: Optional[str] = None,
        key_prefix: Optional[str] = None,
        text: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata_filters: Optional[Mapping[str, Any]] = None,
        value_type: Optional[str] = None,
        include_expired: bool = False,
        limit: Optional[int] = None,
        return_entries: bool = True,
    ) -> List[Any]:
        normalized_namespace = self._normalize_namespace(namespace) if namespace else None
        normalized_tags = set(self._normalize_tags(tags)) if tags else set()
        normalized_text = normalize_text(text, lowercase=True) if text else None
        effective_limit = limit or self.max_query_results
        ensure_numeric_range(effective_limit, "limit", minimum=1, inclusive=True, config=self.memory_config, error_cls=BaseValidationError)
        metadata_filters = dict(metadata_filters or {})

        with self._lock:
            results: List[Any] = []
            namespaces = [normalized_namespace] if normalized_namespace else list(self._store.keys())
            for current_namespace in namespaces:
                bucket = self._store.get(current_namespace, OrderedDict())
                for key, entry in bucket.items():
                    if not include_expired and entry.is_expired():
                        continue
                    if key_prefix and not key.startswith(key_prefix):
                        continue
                    if normalized_tags and not normalized_tags.issubset(set(entry.tags)):
                        continue
                    if value_type and entry.value_type != value_type:
                        continue
                    if metadata_filters:
                        matched = True
                        for meta_key, meta_value in metadata_filters.items():
                            if entry.metadata.get(meta_key) != meta_value:
                                matched = False
                                break
                        if not matched:
                            continue
                    if normalized_text:
                        haystacks = [key, json_dumps(entry.metadata, pretty=False), safe_repr(entry.value)]
                        joined = normalize_text(" ".join(haystacks), lowercase=True)
                        if normalized_text not in joined:
                            continue
                    results.append(copy.deepcopy(entry if return_entries else entry.value))
                    if len(results) >= effective_limit:
                        return results
            return results

    def get_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if limit is None or limit >= len(self._history):
                return [copy.deepcopy(item) for item in self._history]
            return [copy.deepcopy(item) for item in list(self._history)[-limit:]]

    def cleanup_expired(self, *, force: bool = True) -> int:
        with self._lock:
            return self._maybe_cleanup_locked(force=force)

    def compact(self) -> Dict[str, int]:
        with self._lock:
            removed = self._maybe_cleanup_locked(force=True)
            self._enforce_capacity_locked()
            stats = {"expired_removed": removed, "entries_remaining": self.total_entries, "namespaces_remaining": self.total_namespaces}
            self._record_history("compact", namespace="*", details=stats)
            return stats

    def get_namespace_stats(self, namespace: str) -> Dict[str, Any]:
        normalized_namespace = self._normalize_namespace(namespace)
        with self._lock:
            bucket = self._store.get(normalized_namespace, OrderedDict())
            expired = sum(1 for entry in bucket.values() if entry.is_expired())
            active = len(bucket) - expired
            total_bytes = sum(entry.size_estimate_bytes for entry in bucket.values() if not entry.is_expired())
            tags: Set[str] = set()
            value_types: Set[str] = set()
            for entry in bucket.values():
                tags.update(entry.tags)
                value_types.add(entry.value_type)
            return {
                "namespace": normalized_namespace,
                "entries": len(bucket),
                "active_entries": active,
                "expired_entries": expired,
                "estimated_size_bytes": total_bytes,
                "tags": sorted(tags),
                "value_types": sorted(value_types),
            }

    def stats(self) -> MemoryStats:
        with self._lock:
            expired_entries = 0
            for bucket in self._store.values():
                expired_entries += sum(1 for entry in bucket.values() if entry.is_expired())
            return MemoryStats(
                total_entries=self.total_entries,
                total_namespaces=self.total_namespaces,
                expired_entries=expired_entries,
                history_length=len(self._history),
                max_entries=self.max_entries,
                max_entries_per_namespace=self.max_entries_per_namespace,
                cleanup_runs=self._cleanup_runs,
                evictions=self._evictions,
                writes=self._writes,
                reads=self._reads,
                deletes=self._deletes,
                hits=self._hits,
                misses=self._misses,
            )

    def to_dict(self, *, include_values: bool = True, namespace: Optional[str] = None, redact: Optional[bool] = None) -> Dict[str, Any]:
        redact_exports = self.export_redact_secrets if redact is None else bool(redact)
        with self._lock:
            namespaces = [self._normalize_namespace(namespace)] if namespace else list(self._store.keys())
            memory_payload: Dict[str, Any] = {}
            for current_namespace in namespaces:
                bucket = self._store.get(current_namespace, OrderedDict())
                if not bucket:
                    continue
                memory_payload[current_namespace] = {
                    key: entry.to_dict(include_value=include_values, redact=redact_exports)
                    for key, entry in bucket.items()
                    if not entry.is_expired()
                }
            return {
                "memory_id": self._memory_id,
                "started_at": self._started_at,
                "exported_at": utc_now_iso(),
                "config_path": self.config.get("__config_path__"),
                "stats": self.stats().to_dict(),
                "memory": memory_payload,
            }

    def to_json(self, *, include_values: bool = True, namespace: Optional[str] = None, pretty: Optional[bool] = None, redact: Optional[bool] = None) -> str:
        return json_dumps(self.to_dict(include_values=include_values, namespace=namespace, redact=redact), pretty=self.snapshot_pretty if pretty is None else bool(pretty))

    def save_snapshot(self, path: Optional[str] = None, *, include_values: bool = True, redact: Optional[bool] = None) -> str:
        snapshot_path = path or self.persistence_path
        try:
            resolved_path = Path(snapshot_path)
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Invalid snapshot path for BaseMemory.",
                config=self.memory_config,
                component="BaseMemory",
                operation="save_snapshot",
                context={"path": snapshot_path},
            ) from exc

        try:
            if resolved_path.parent and not resolved_path.parent.exists():
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(self.to_json(include_values=include_values, pretty=self.snapshot_pretty, redact=redact), encoding="utf-8")
            self._record_history("save_snapshot", namespace="*", details={"path": str(resolved_path)})
            return str(resolved_path)
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Failed to save BaseMemory snapshot.",
                config=self.memory_config,
                component="BaseMemory",
                operation="save_snapshot",
                context={"path": str(resolved_path)},
            ) from exc

    snapshot = save_snapshot

    def load_snapshot(self, path: Optional[str] = None, *, merge: bool = True, must_exist: bool = True) -> int:
        snapshot_path = path or self.persistence_path
        resolved_path = Path(snapshot_path)
        if not resolved_path.exists():
            if must_exist:
                raise BaseIOError(
                    "BaseMemory snapshot file was not found.",
                    self.memory_config,
                    component="BaseMemory",
                    operation="load_snapshot",
                    context={"path": str(resolved_path)},
                )
            return 0
        try:
            payload = json_loads(resolved_path.read_text(encoding="utf-8"), default=None)
            if payload is None:
                raise BaseSerializationError(
                    "Snapshot payload could not be parsed as JSON.",
                    self.memory_config,
                    component="BaseMemory",
                    operation="load_snapshot",
                    context={"path": str(resolved_path)},
                )
            loaded = self.load_from_dict(payload, merge=merge)
            self._record_history("load_snapshot", namespace="*", details={"path": str(resolved_path), "loaded_entries": loaded})
            return loaded
        except BaseError:
            raise
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Failed to load BaseMemory snapshot.",
                config=self.memory_config,
                component="BaseMemory",
                operation="load_snapshot",
                context={"path": str(resolved_path)},
            ) from exc

    def load_from_dict(self, payload: Mapping[str, Any], *, merge: bool = True) -> int:
        ensure_mapping(payload, "payload", config=self.memory_config, error_cls=BaseValidationError)
        memory_section = payload.get("memory")
        ensure_mapping(memory_section, "payload.memory", config=self.memory_config, error_cls=BaseValidationError)

        loaded_entries = 0
        with self._lock:
            if not merge:
                self.clear()
            for namespace, items in memory_section.items():
                ensure_mapping(items, f"memory[{namespace}]", config=self.memory_config, error_cls=BaseValidationError)
                for key, entry_payload in items.items():
                    ensure_mapping(entry_payload, f"memory[{namespace}][{key}]", config=self.memory_config, error_cls=BaseValidationError)
                    entry_payload = dict(entry_payload)
                    self.put(
                        str(key),
                        entry_payload.get("value"),
                        namespace=str(namespace),
                        metadata=entry_payload.get("metadata"),
                        tags=entry_payload.get("tags"),
                        ttl_seconds=None,
                        source=entry_payload.get("source"),
                        persistent=bool(entry_payload.get("persistent", True)),
                    )
                    if entry_payload.get("expires_at"):
                        bucket = self._namespace_store(self._normalize_namespace(str(namespace)), create=False)
                        bucket[str(key)].expires_at = entry_payload.get("expires_at")
                    loaded_entries += 1
        return loaded_entries

    def clone_namespace(self, source_namespace: str, target_namespace: str, *, overwrite: bool = False) -> int:
        source = self._normalize_namespace(source_namespace)
        target = self._normalize_namespace(target_namespace)
        if source == target:
            raise BaseValidationError(
                "source_namespace and target_namespace must differ.",
                self.memory_config,
                component="BaseMemory",
                operation="clone_namespace",
                context={"namespace": source},
            )
        with self._lock:
            source_bucket = self._store.get(source, OrderedDict())
            if not source_bucket:
                return 0
            if not overwrite and target in self._store and self._store[target]:
                raise BaseStateError(
                    "Target namespace already exists and overwrite is disabled.",
                    self.memory_config,
                    component="BaseMemory",
                    operation="clone_namespace",
                    context={"source": source, "target": target},
                )
            if overwrite:
                self.clear(target)
            count = 0
            for key, entry in source_bucket.items():
                if entry.is_expired():
                    continue
                self.put(
                    key,
                    entry.value,
                    namespace=target,
                    metadata=entry.metadata,
                    tags=entry.tags,
                    ttl_seconds=None,
                    source=entry.source,
                    persistent=entry.persistent,
                )
                target_bucket = self._namespace_store(target, create=False)
                target_bucket[key].expires_at = entry.expires_at
                count += 1
            self._record_history("clone_namespace", namespace=target, details={"source": source, "count": count})
            return count

    def remove_namespace(self, namespace: str) -> int:
        return self.clear(namespace)

    def export_namespace(self, namespace: str, *, redact: Optional[bool] = None) -> Dict[str, Any]:
        return self.to_dict(namespace=namespace, redact=redact)


__all__ = ["MemoryEntry", "MemoryStats", "BaseMemory"]


if __name__ == "__main__":
    print("\n=== Running Base Memory ===\n")
    printer.status("TEST", "Base Memory initialized", "info")

    memory = BaseMemory()
    printer.pretty("CONFIG", memory.memory_config, "info")

    memory.put(
        "session_id",
        "session-001",
        namespace="runtime",
        metadata={"owner": "base_agent", "scope": "test"},
        tags=["session", "runtime"],
        source="test_block",
    )
    memory.put(
        "working_context",
        {"topic": "memory_refactor", "step": 1, "status": "active"},
        namespace="runtime",
        metadata={"owner": "planner"},
        tags=["context", "planning"],
        source="test_block",
    )
    memory.increment("counter", 2, namespace="metrics", metadata={"unit": "count"}, tags=["metric"])
    memory.increment("counter", 3, namespace="metrics", metadata={"unit": "count"}, tags=["metric"])
    memory.append("events", {"kind": "start", "ok": True}, namespace="events", tags=["event"])
    memory.append("events", {"kind": "step", "ok": True}, namespace="events", tags=["event"])
    memory.merge("working_context", {"step": 2, "owner": "base_agent"}, namespace="runtime", tags=["context", "planning"])

    runtime_context = memory.get("working_context", namespace="runtime")
    counter_value = memory.get("counter", namespace="metrics")
    search_results = memory.search(namespace="runtime", text="memory_refactor")
    namespace_stats = memory.get_namespace_stats("runtime")
    stats_before_snapshot = memory.stats().to_dict()

    printer.pretty("RUNTIME_CONTEXT", runtime_context, "success")
    printer.pretty("COUNTER", counter_value, "success")
    printer.pretty("SEARCH_RESULTS", [entry.to_dict() for entry in search_results], "success")
    printer.pretty("RUNTIME_STATS", namespace_stats, "success")
    printer.pretty("MEMORY_STATS", stats_before_snapshot, "success")

    snapshot_path = memory.save_snapshot("base_memory_test_snapshot.json")
    printer.status("SAVE", f"Snapshot saved to {snapshot_path}", "success")

    restored = BaseMemory()
    restored.load_snapshot("base_memory_test_snapshot.json", merge=False)
    restored_runtime = restored.get("working_context", namespace="runtime")
    printer.pretty("RESTORED_RUNTIME", restored_runtime, "success")

    try:
        restored.merge("counter", {"bad": "type"}, namespace="metrics")
    except BaseError as exc:
        printer.pretty("EXPECTED_ERROR", exc.to_dict(include_traceback=False), "warning")

    compacted = restored.compact()
    printer.pretty("COMPACT", compacted, "info")
    printer.pretty("HISTORY", restored.get_history(limit=10), "info")

    if os.path.exists("base_memory_test_snapshot.json"):
        os.remove("base_memory_test_snapshot.json")

    print("\n=== Test ran successfully ===\n")
