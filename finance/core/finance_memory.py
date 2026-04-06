
"""Production-ready financial memory for checkpointing, caching, and tagging.

This module provides a durable, thread-safe in-process memory layer for an
autonomous financial agent. It is optimized for:

- checkpointing state safely to disk
- TTL-aware caching of expensive intermediate results
- tag-based retrieval and indexing
- bounded storage with configurable eviction
- structured error handling and recovery semantics

The configuration loading path intentionally preserves the original project
behavior. The project import is attempted first, with fallbacks only to make the
module runnable in isolation during local testing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
import uuid

from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from finance.core.utils.config_loader import load_global_config, get_config_section
from finance.core.utils.financial_errors import (CheckpointWriteError, ConfigurationError, ErrorContext,
                                                 FinancialAgentError, InvalidConfigurationError, log_error,
                                                 PersistenceError, ResourceExhaustionError, ValidationError,
                                                 StateStoreUnavailableError, classify_external_exception)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Finance Memory")
printer = PrettyPrinter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_FORMAT_VERSION = 2
DEFAULT_MAX_SIZE = 5000
DEFAULT_CHECKPOINT_DIR = "finance/checkpoints"
DEFAULT_CHECKPOINT_FREQ = 500
DEFAULT_TAG_RETENTION_DAYS = 30
DEFAULT_EVICTION_POLICY = "LRU"
DEFAULT_TTL_SECONDS = 300
DEFAULT_MAX_VERSIONS = 5
DEFAULT_MAX_MEMORY_MB = 200
VALID_EVICTION_POLICIES = {"LRU", "FIFO", "PRIORITY"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EntryMetadata:
    entry_id: str
    data_type: str
    created_at: float
    updated_at: float
    last_accessed_at: float
    access_count: int
    priority: str
    priority_value: int
    tags: List[str] = field(default_factory=list)
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    expires_at: Optional[float] = None
    namespace: Optional[str] = None
    cache_key: Optional[str] = None
    version: Optional[int] = None
    checkpointable: bool = True
    pinned: bool = False
    immutable: bool = False
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EntryMetadata":
        payload = dict(payload)
        return cls(
            entry_id=str(payload["entry_id"]),
            data_type=str(payload["data_type"]),
            created_at=float(payload["created_at"]),
            updated_at=float(payload.get("updated_at", payload["created_at"])),
            last_accessed_at=float(payload.get("last_accessed_at", payload["created_at"])),
            access_count=int(payload.get("access_count", 0)),
            priority=str(payload.get("priority", "medium")),
            priority_value=int(payload.get("priority_value", 1)),
            tags=list(payload.get("tags", [])),
            size_bytes=int(payload.get("size_bytes", 0)),
            ttl_seconds=payload.get("ttl_seconds"),
            expires_at=payload.get("expires_at"),
            namespace=payload.get("namespace"),
            cache_key=payload.get("cache_key"),
            version=payload.get("version"),
            checkpointable=bool(payload.get("checkpointable", True)),
            pinned=bool(payload.get("pinned", False)),
            immutable=bool(payload.get("immutable", False)),
            source=payload.get("source"),
            extra=dict(payload.get("extra", {})),
        )


@dataclass(slots=True)
class MemoryEntry:
    data: Any
    metadata: EntryMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {"data": self.data, "metadata": self.metadata.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryEntry":
        return cls(
            data=payload.get("data"),
            metadata=EntryMetadata.from_dict(payload["metadata"]),
        )


# ---------------------------------------------------------------------------
# FinanceMemory
# ---------------------------------------------------------------------------

class FinanceMemory:
    """Durable memory system for financial agents.

    Responsibilities:
    - checkpointing: atomic, integrity-checked state snapshots
    - caching: TTL-aware cache entries with version retention
    - tagging: fast tag-based indexing and retrieval
    - bounded memory: configurable eviction and cleanup
    - observability: statistics, snapshots, and structured failures
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("finance_memory") or {}
        self.shared_memory_config = get_config_section("shared_memory") or {}

        self.max_size = int(self.memory_config.get("max_size", DEFAULT_MAX_SIZE))
        self.eviction_policy = str(
            self.memory_config.get("eviction_policy", DEFAULT_EVICTION_POLICY)
        ).upper()
        self.checkpoint_dir = str(
            self.memory_config.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)
        )
        self.checkpoint_freq = int(
            self.memory_config.get("checkpoint_freq", DEFAULT_CHECKPOINT_FREQ)
        )
        self.auto_save = bool(self.memory_config.get("auto_save", True))
        self.tag_retention_days = int(
            self.memory_config.get("tag_retention", DEFAULT_TAG_RETENTION_DAYS)
        )
        self.default_ttl_seconds = int(
            self.shared_memory_config.get("default_ttl", DEFAULT_TTL_SECONDS)
        )
        self.max_versions = int(
            self.shared_memory_config.get("max_versions", DEFAULT_MAX_VERSIONS)
        )
        self.max_memory_bytes = int(
            self.shared_memory_config.get("max_memory_mb", DEFAULT_MAX_MEMORY_MB)
        ) * 1024 * 1024

        self.priority_map = self._build_priority_map(
            self.memory_config.get("priority_levels"),
            self.memory_config.get("priority_map"),
        )
        self.default_priority = "medium"

        self.store: "OrderedDict[str, MemoryEntry]" = OrderedDict()
        self.tag_index: MutableMapping[str, set[str]] = defaultdict(set)
        self.cache_index: Dict[Tuple[str, str], str] = {}
        self.version_index: MutableMapping[Tuple[str, str], List[str]] = defaultdict(list)
        self.tag_last_seen: Dict[str, float] = {}
        self.access_counter = 0
        self.last_checkpoint: Optional[str] = None
        self._mutation_count = 0
        self._stats = {
            "writes": 0,
            "reads": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "deletes": 0,
            "expired_purges": 0,
            "checkpoints_created": 0,
            "checkpoints_loaded": 0,
            "cache_sets": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.lock = RLock()

        self._validate_configuration()
        self._ensure_checkpoint_dir()

        if printer is not None:  # pragma: no cover - presentation only
            printer.status("INIT", "Finance Memory successfully initialized", "success")
        logger.info(
            "Finance Memory initialized | max_size=%s eviction=%s checkpoint_dir=%s",
            self.max_size,
            self.eviction_policy,
            self.checkpoint_dir,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _context(self, operation: str, **metadata: Any) -> ErrorContext:
        return ErrorContext(
            component="finance_memory",
            operation=operation,
            metadata=metadata or {},
        )

    def _build_priority_map(
        self,
        raw_priority_levels: Any,
        raw_priority_map: Any,
    ) -> Dict[str, int]:
        default_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        priority_map: Dict[str, int] = {}
        if isinstance(raw_priority_levels, Mapping):
            for key, value in raw_priority_levels.items():
                try:
                    priority_map[str(key).lower()] = int(value)
                except Exception:
                    continue
        elif isinstance(raw_priority_levels, (list, tuple, set)):
            for idx, name in enumerate(raw_priority_levels):
                priority_map[str(name).lower()] = idx

        if isinstance(raw_priority_map, Mapping):
            for key, value in raw_priority_map.items():
                try:
                    priority_map[str(key).lower()] = int(value)
                except Exception:
                    continue

        if not priority_map:
            priority_map = dict(default_map)

        for key, value in default_map.items():
            priority_map.setdefault(key, value)

        return priority_map

    def _validate_configuration(self) -> None:
        try:
            if self.max_size <= 0:
                raise InvalidConfigurationError(
                    "finance_memory.max_size must be a positive integer.",
                    context=self._context("validate_config", max_size=self.max_size),
                )
            if self.checkpoint_freq <= 0:
                raise InvalidConfigurationError(
                    "finance_memory.checkpoint_freq must be a positive integer.",
                    context=self._context(
                        "validate_config", checkpoint_freq=self.checkpoint_freq
                    ),
                )
            if self.eviction_policy not in VALID_EVICTION_POLICIES:
                raise InvalidConfigurationError(
                    f"finance_memory.eviction_policy must be one of {sorted(VALID_EVICTION_POLICIES)}.",
                    context=self._context(
                        "validate_config", eviction_policy=self.eviction_policy
                    ),
                )
            if self.max_versions <= 0:
                raise InvalidConfigurationError(
                    "shared_memory.max_versions must be a positive integer.",
                    context=self._context("validate_config", max_versions=self.max_versions),
                )
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("validate_config"),
                message="Failed to validate finance memory configuration.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def _ensure_checkpoint_dir(self) -> None:
        try:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            handled = StateStoreUnavailableError(
                "Failed to create or access checkpoint directory.",
                context=self._context("ensure_checkpoint_dir", checkpoint_dir=self.checkpoint_dir),
                details={"checkpoint_dir": self.checkpoint_dir},
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def _now(self) -> float:
        return time.time()

    def _normalize_priority(self, priority: str) -> Tuple[str, int]:
        label = (priority or self.default_priority).strip().lower()
        if label not in self.priority_map:
            logger.warning("Unknown priority '%s'. Falling back to '%s'.", label, self.default_priority)
            label = self.default_priority
        return label, self.priority_map[label]

    def _normalize_tags(self, tags: Optional[Iterable[str]]) -> List[str]:
        if not tags:
            return []
        normalized: List[str] = []
        seen = set()
        for tag in tags:
            cleaned = str(tag).strip().lower()
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        return normalized

    def _estimate_size_bytes(self, payload: Any) -> int:
        try:
            return len(json.dumps(self._make_json_safe(payload), sort_keys=True).encode("utf-8"))
        except Exception:
            return 0

    def _generate_entry_id(self, data_type: str) -> str:
        safe_type = "".join(ch for ch in str(data_type).lower() if ch.isalnum() or ch in ("_", "-")) or "entry"
        return f"fin_{safe_type}_{int(self._now() * 1000)}_{self.access_counter}_{uuid.uuid4().hex[:8]}"

    def _is_expired(self, entry: MemoryEntry, *, now: Optional[float] = None) -> bool:
        if entry.metadata.expires_at is None:
            return False
        reference_time = self._now() if now is None else now
        return reference_time >= entry.metadata.expires_at

    def _touch_locked(self, entry_id: str) -> None:
        entry = self.store.get(entry_id)
        if entry is None:
            return
        now = self._now()
        entry.metadata.access_count += 1
        entry.metadata.last_accessed_at = now
        entry.metadata.updated_at = max(entry.metadata.updated_at, now)
        self.store.move_to_end(entry_id)
        self.access_counter += 1

    def _index_entry_locked(self, entry_id: str, entry: MemoryEntry) -> None:
        for tag in entry.metadata.tags:
            self.tag_index[tag].add(entry_id)
            self.tag_last_seen[tag] = self._now()

        if entry.metadata.namespace and entry.metadata.cache_key:
            key = (entry.metadata.namespace, entry.metadata.cache_key)
            self.cache_index[key] = entry_id
            versions = self.version_index[key]
            if entry_id not in versions:
                versions.append(entry_id)

    def _deindex_entry_locked(self, entry_id: str, entry: MemoryEntry) -> None:
        for tag in list(entry.metadata.tags):
            ids = self.tag_index.get(tag)
            if ids:
                ids.discard(entry_id)
                if not ids:
                    self.tag_index.pop(tag, None)

        if entry.metadata.namespace and entry.metadata.cache_key:
            key = (entry.metadata.namespace, entry.metadata.cache_key)
            if self.cache_index.get(key) == entry_id:
                self.cache_index.pop(key, None)

            versions = self.version_index.get(key)
            if versions:
                self.version_index[key] = [candidate for candidate in versions if candidate != entry_id]
                if self.version_index[key]:
                    self.cache_index[key] = self.version_index[key][-1]
                else:
                    self.version_index.pop(key, None)

    def _delete_locked(self, entry_id: str) -> bool:
        entry = self.store.get(entry_id)
        if entry is None:
            return False
        self._deindex_entry_locked(entry_id, entry)
        del self.store[entry_id]
        self._stats["deletes"] += 1
        return True

    def _maybe_trim_versions_locked(self, namespace: str, cache_key: str) -> None:
        key = (namespace, cache_key)
        versions = self.version_index.get(key, [])
        while len(versions) > self.max_versions:
            oldest_id = versions.pop(0)
            if oldest_id in self.store:
                self._delete_locked(oldest_id)

    def _eviction_candidates_locked(self) -> List[str]:
        if self.eviction_policy == "FIFO":
            return list(self.store.keys())

        if self.eviction_policy == "LRU":
            return list(self.store.keys())

        # PRIORITY: lower priority first, then oldest last_accessed, then oldest created
        sortable = []
        for entry_id, entry in self.store.items():
            meta = entry.metadata
            sortable.append(
                (
                    meta.priority_value,
                    meta.last_accessed_at,
                    meta.created_at,
                    entry_id,
                )
            )
        sortable.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[-1] for item in sortable]

    def _evict_one_locked(self) -> bool:
        for entry_id in self._eviction_candidates_locked():
            entry = self.store.get(entry_id)
            if entry is None:
                continue
            if entry.metadata.pinned:
                continue
            if entry.metadata.immutable:
                continue
            self._delete_locked(entry_id)
            self._stats["evictions"] += 1
            logger.warning("Evicted memory entry '%s' due to capacity pressure.", entry_id)
            return True
        return False

    def _enforce_capacity_locked(self) -> None:
        self._purge_expired_locked()

        if self.max_size <= 0:
            return

        while len(self.store) > self.max_size:
            if not self._evict_one_locked():
                raise ResourceExhaustionError(
                    "Memory capacity exceeded but no evictable entries remain.",
                    context=self._context(
                        "enforce_capacity",
                        max_size=self.max_size,
                        current_entries=len(self.store),
                    ),
                    details={"max_size": self.max_size, "current_entries": len(self.store)},
                )

        while self._current_size_bytes_locked() > self.max_memory_bytes:
            if not self._evict_one_locked():
                raise ResourceExhaustionError(
                    "Memory byte limit exceeded but no evictable entries remain.",
                    context=self._context(
                        "enforce_capacity",
                        max_memory_bytes=self.max_memory_bytes,
                        current_size_bytes=self._current_size_bytes_locked(),
                    ),
                    details={
                        "max_memory_bytes": self.max_memory_bytes,
                        "current_size_bytes": self._current_size_bytes_locked(),
                    },
                )

    def _current_size_bytes_locked(self) -> int:
        return sum(entry.metadata.size_bytes for entry in self.store.values())

    def _purge_expired_locked(self) -> int:
        now = self._now()
        expired_ids = [
            entry_id
            for entry_id, entry in self.store.items()
            if self._is_expired(entry, now=now)
        ]
        for entry_id in expired_ids:
            self._delete_locked(entry_id)
        if expired_ids:
            self._stats["expired_purges"] += len(expired_ids)
            logger.info("Purged %s expired memory entries.", len(expired_ids))
        return len(expired_ids)

    def _record_mutation_locked(self) -> None:
        self._mutation_count += 1
        self.access_counter += 1
        if self.auto_save and self._mutation_count % self.checkpoint_freq == 0:
            try:
                self.create_checkpoint()
            except FinancialAgentError:
                raise
            except Exception as exc:
                handled = classify_external_exception(
                    exc,
                    context=self._context(
                        "auto_checkpoint", mutation_count=self._mutation_count
                    ),
                    message="Automatic finance memory checkpoint failed.",
                )
                log_error(handled, logger_=logger)
                raise handled from exc

    def _safe_get_nested(self, data: Any, dotted_key: str) -> Any:
        current = data
        for part in dotted_key.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _matches_filters(
        self,
        entry: MemoryEntry,
        data_filters: Optional[Mapping[str, Any]],
        metadata_filters: Optional[Mapping[str, Any]],
    ) -> bool:
        if data_filters:
            for key, expected in data_filters.items():
                actual = self._safe_get_nested(entry.data, key)
                if actual != expected:
                    return False

        if metadata_filters:
            meta_dict = entry.metadata.to_dict()
            for key, expected in metadata_filters.items():
                actual = self._safe_get_nested(meta_dict, key)
                if actual != expected:
                    return False

        return True

    def _make_json_safe(self, value: Any, *, _depth: int = 0, _max_depth: int = 20) -> Any:
        if _depth > _max_depth:
            return repr(value)

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Decimal):
            return float(value)

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, bytes):
            return {"__type__": "bytes", "base64": base64.b64encode(value).decode("ascii")}

        if is_dataclass(value):
            return self._make_json_safe(asdict(value), _depth=_depth + 1)

        if isinstance(value, Mapping):
            return {
                str(k): self._make_json_safe(v, _depth=_depth + 1)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(v, _depth=_depth + 1) for v in value]

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self._make_json_safe(value.to_dict(), _depth=_depth + 1)
            except Exception:
                return repr(value)

        if hasattr(value, "__dict__"):
            try:
                return self._make_json_safe(vars(value), _depth=_depth + 1)
            except Exception:
                return repr(value)

        return repr(value)

    def _checkpoint_payload_locked(self, *, include_config: bool) -> Dict[str, Any]:
        serializable_store = OrderedDict(
            (
                entry_id,
                {
                    "data": self._make_json_safe(entry.data),
                    "metadata": entry.metadata.to_dict(),
                },
            )
            for entry_id, entry in self.store.items()
            if entry.metadata.checkpointable
        )

        payload = {
            "meta": {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "entry_count": len(serializable_store),
                "access_counter": self.access_counter,
                "mutation_count": self._mutation_count,
                "stats": dict(self._stats),
                "config_snapshot": self._make_json_safe(self.config) if include_config else None,
            },
            "store": serializable_store,
            "tag_index": {tag: sorted(ids) for tag, ids in self.tag_index.items()},
            "cache_index": {
                f"{namespace}::{cache_key}": entry_id
                for (namespace, cache_key), entry_id in self.cache_index.items()
            },
            "version_index": {
                f"{namespace}::{cache_key}": list(entry_ids)
                for (namespace, cache_key), entry_ids in self.version_index.items()
            },
        }
        return payload

    def _checkpoint_wrapper_locked(self, *, include_config: bool) -> Dict[str, Any]:
        payload = self._checkpoint_payload_locked(include_config=include_config)
        checksum = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {"payload": payload, "checksum": checksum}

    def _atomic_write_json(self, path: str, payload: Mapping[str, Any]) -> None:
        temp_dir = os.path.dirname(path) or "."
        fd, temp_path = tempfile.mkstemp(prefix=".finance_memory_", suffix=".tmp", dir=temp_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _validate_checkpoint_payload(self, wrapper: Mapping[str, Any]) -> Dict[str, Any]:
        if "payload" not in wrapper or "checksum" not in wrapper:
            raise ValidationError(
                "Checkpoint file is missing required fields.",
                context=self._context("validate_checkpoint"),
                details={"required_fields": ["payload", "checksum"]},
            )

        payload = dict(wrapper["payload"])
        expected_checksum = str(wrapper["checksum"])
        actual_checksum = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual_checksum != expected_checksum:
            raise CheckpointWriteError(
                "Checkpoint checksum verification failed.",
                context=self._context("validate_checkpoint"),
                details={
                    "expected_checksum": expected_checksum,
                    "actual_checksum": actual_checksum,
                },
            )

        meta = payload.get("meta", {})
        format_version = int(meta.get("format_version", -1))
        if format_version > CHECKPOINT_FORMAT_VERSION:
            raise ValidationError(
                "Checkpoint format version is newer than this runtime supports.",
                context=self._context(
                    "validate_checkpoint",
                    checkpoint_format_version=format_version,
                    runtime_format_version=CHECKPOINT_FORMAT_VERSION,
                ),
                details={
                    "checkpoint_format_version": format_version,
                    "runtime_format_version": CHECKPOINT_FORMAT_VERSION,
                },
            )

        if "store" not in payload:
            raise ValidationError(
                "Checkpoint payload missing 'store'.",
                context=self._context("validate_checkpoint"),
            )

        return payload

    def _rebuild_indexes_locked(self) -> None:
        self.tag_index = defaultdict(set)
        self.cache_index = {}
        self.version_index = defaultdict(list)
        self.tag_last_seen = {}
        for entry_id, entry in self.store.items():
            self._index_entry_locked(entry_id, entry)

    # ------------------------------------------------------------------
    # Public API - writes
    # ------------------------------------------------------------------

    def add_batch_data(
        self,
        batch_id_from_manager: str,
        single_batch_data_content: Dict[str, float],
    ) -> str:
        financial_memory_entry_data = {
            "source_batch_id": batch_id_from_manager,
            "timestamp": self._now(),
            "data": single_batch_data_content,
        }
        return self.add_financial_data(
            data=financial_memory_entry_data,
            data_type="batch",
            tags=["batch_data", f"src_batch_{batch_id_from_manager}"],
            priority="high",
            metadata={"batch_id": batch_id_from_manager},
        )

    def add_financial_data(
        self,
        data: Any,
        data_type: str,
        tags: Optional[Sequence[str]] = None,
        priority: str = "medium",
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        ttl_seconds: Optional[int] = None,
        namespace: Optional[str] = None,
        key: Optional[str] = None,
        version: Optional[int] = None,
        checkpointable: bool = True,
        pinned: bool = False,
        immutable: bool = False,
        source: Optional[str] = None,
    ) -> str:
        if not data_type or not str(data_type).strip():
            raise ValidationError(
                "data_type is required when adding financial data.",
                context=self._context("add_financial_data"),
            )

        try:
            normalized_priority, priority_value = self._normalize_priority(priority)
            normalized_tags = self._normalize_tags(tags)
            now = self._now()
            entry_id = self._generate_entry_id(data_type)
            metadata_dict = dict(metadata or {})

            effective_ttl = ttl_seconds
            if effective_ttl is not None and int(effective_ttl) <= 0:
                raise ValidationError(
                    "ttl_seconds must be positive when provided.",
                    context=self._context(
                        "add_financial_data", data_type=data_type, ttl_seconds=ttl_seconds
                    ),
                )

            entry_metadata = EntryMetadata(
                entry_id=entry_id,
                data_type=str(data_type),
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
                access_count=0,
                priority=normalized_priority,
                priority_value=priority_value,
                tags=normalized_tags,
                size_bytes=self._estimate_size_bytes(data),
                ttl_seconds=int(effective_ttl) if effective_ttl is not None else None,
                expires_at=(now + int(effective_ttl)) if effective_ttl is not None else None,
                namespace=namespace,
                cache_key=key,
                version=version,
                checkpointable=bool(checkpointable),
                pinned=bool(pinned),
                immutable=bool(immutable),
                source=source,
                extra=self._make_json_safe(metadata_dict),
            )

            entry = MemoryEntry(data=data, metadata=entry_metadata)

            with self.lock:
                self.store[entry_id] = entry
                self._index_entry_locked(entry_id, entry)
                if namespace and key:
                    self._maybe_trim_versions_locked(namespace, key)
                self._enforce_capacity_locked()
                self._stats["writes"] += 1
                self._record_mutation_locked()

            return entry_id

        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context(
                    "add_financial_data", data_type=data_type, namespace=namespace, key=key
                ),
                message="Failed to add financial data to memory.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def set_cache(
        self,
        cache_key: str,
        value: Any,
        *,
        namespace: str = "default",
        ttl_seconds: Optional[int] = None,
        tags: Optional[Sequence[str]] = None,
        priority: str = "medium",
        metadata: Optional[Mapping[str, Any]] = None,
        checkpointable: bool = True,
        pinned: bool = False,
        immutable: bool = False,
    ) -> str:
        if not cache_key or not str(cache_key).strip():
            raise ValidationError(
                "cache_key is required.",
                context=self._context("set_cache"),
            )

        try:
            with self.lock:
                key = (namespace, str(cache_key))
                existing_versions = self.version_index.get(key, [])
                next_version = 1
                if existing_versions:
                    current_entry = self.store.get(existing_versions[-1])
                    if current_entry and current_entry.metadata.version is not None:
                        next_version = int(current_entry.metadata.version) + 1

            merged_tags = list(tags or [])
            merged_tags.extend([f"cache:{namespace}", f"cache_key:{cache_key}"])
            entry_id = self.add_financial_data(
                data=value,
                data_type="cache",
                tags=merged_tags,
                priority=priority,
                metadata=metadata,
                ttl_seconds=self.default_ttl_seconds if ttl_seconds is None else ttl_seconds,
                namespace=namespace,
                key=str(cache_key),
                version=next_version,
                checkpointable=checkpointable,
                pinned=pinned,
                immutable=immutable,
            )
            self._stats["cache_sets"] += 1
            return entry_id
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("set_cache", namespace=namespace, cache_key=cache_key),
                message="Failed to set cache entry.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def add_tags(self, entry_id: str, tags: Sequence[str]) -> bool:
        try:
            normalized_tags = self._normalize_tags(tags)
            if not normalized_tags:
                return False

            with self.lock:
                entry = self.store.get(entry_id)
                if entry is None:
                    return False
                for tag in normalized_tags:
                    if tag not in entry.metadata.tags:
                        entry.metadata.tags.append(tag)
                        self.tag_index[tag].add(entry_id)
                        self.tag_last_seen[tag] = self._now()
                entry.metadata.updated_at = self._now()
                self._record_mutation_locked()
                return True
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("add_tags", entry_id=entry_id),
                message="Failed to add tags to memory entry.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def remove_tags(self, entry_id: str, tags: Sequence[str]) -> bool:
        try:
            normalized_tags = self._normalize_tags(tags)
            if not normalized_tags:
                return False

            with self.lock:
                entry = self.store.get(entry_id)
                if entry is None:
                    return False
                original_tags = set(entry.metadata.tags)
                entry.metadata.tags = [tag for tag in entry.metadata.tags if tag not in normalized_tags]
                removed = original_tags.difference(entry.metadata.tags)
                for tag in removed:
                    ids = self.tag_index.get(tag)
                    if ids:
                        ids.discard(entry_id)
                        if not ids:
                            self.tag_index.pop(tag, None)
                entry.metadata.updated_at = self._now()
                self._record_mutation_locked()
                return bool(removed)
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("remove_tags", entry_id=entry_id),
                message="Failed to remove tags from memory entry.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def delete(self, entry_id: str) -> bool:
        try:
            with self.lock:
                if entry_id not in self.store:
                    return False
                deleted = self._delete_locked(entry_id)
                if deleted:
                    self._record_mutation_locked()
                return deleted
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("delete", entry_id=entry_id),
                message="Failed to delete memory entry.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def invalidate_cache(
        self,
        *,
        cache_key: Optional[str] = None,
        namespace: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> int:
        try:
            removed = 0
            with self.lock:
                candidate_ids: set[str] = set()

                if cache_key is not None:
                    namespaces = [namespace] if namespace is not None else [
                        ns for (ns, ck) in self.cache_index.keys() if ck == cache_key
                    ]
                    for ns in namespaces:
                        version_key = (ns, cache_key)
                        candidate_ids.update(self.version_index.get(version_key, []))

                if tag is not None:
                    candidate_ids.update(self.tag_index.get(tag.lower(), set()))

                if cache_key is None and tag is None:
                    for entry_id, entry in self.store.items():
                        if entry.metadata.data_type == "cache":
                            if namespace is None or entry.metadata.namespace == namespace:
                                candidate_ids.add(entry_id)

                for entry_id in list(candidate_ids):
                    entry = self.store.get(entry_id)
                    if entry and entry.metadata.data_type == "cache":
                        if namespace is None or entry.metadata.namespace == namespace:
                            if self._delete_locked(entry_id):
                                removed += 1

                if removed:
                    self._record_mutation_locked()

            return removed
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context(
                    "invalidate_cache", cache_key=cache_key, namespace=namespace, tag=tag
                ),
                message="Failed to invalidate cache entries.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def clear(self, *, include_pinned: bool = False) -> int:
        try:
            removed = 0
            with self.lock:
                for entry_id in list(self.store.keys()):
                    entry = self.store[entry_id]
                    if entry.metadata.pinned and not include_pinned:
                        continue
                    if self._delete_locked(entry_id):
                        removed += 1
                if removed:
                    self._record_mutation_locked()
            return removed
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("clear", include_pinned=include_pinned),
                message="Failed to clear finance memory.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    # ------------------------------------------------------------------
    # Public API - reads
    # ------------------------------------------------------------------

    def get(
        self,
        entry_id: Optional[str] = None,
        tag: Optional[str] = None,
        *,
        include_expired: bool = False,
        touch: bool = True,
        default: Any = None,
    ) -> Any:
        try:
            with self.lock:
                self._stats["reads"] += 1

                if entry_id is not None:
                    entry = self.store.get(entry_id)
                    if entry is None:
                        self._stats["misses"] += 1
                        return default

                    if not include_expired and self._is_expired(entry):
                        self._delete_locked(entry_id)
                        self._stats["misses"] += 1
                        return default

                    if touch:
                        self._touch_locked(entry_id)
                    self._stats["hits"] += 1
                    return entry.to_dict()

                if tag is not None:
                    normalized_tag = str(tag).strip().lower()
                    entries = []
                    for candidate_id in list(self.tag_index.get(normalized_tag, set())):
                        entry = self.store.get(candidate_id)
                        if entry is None:
                            continue
                        if not include_expired and self._is_expired(entry):
                            self._delete_locked(candidate_id)
                            continue
                        if touch:
                            self._touch_locked(candidate_id)
                        entries.append(entry.to_dict())
                    if entries:
                        self._stats["hits"] += 1
                    else:
                        self._stats["misses"] += 1
                    return entries

                results = []
                for candidate_id, entry in list(self.store.items()):
                    if not include_expired and self._is_expired(entry):
                        self._delete_locked(candidate_id)
                        continue
                    results.append(entry.to_dict())
                return results
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("get", entry_id=entry_id, tag=tag),
                message="Failed to retrieve memory entry.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def get_cache(
        self,
        cache_key: str,
        *,
        namespace: str = "default",
        version: Optional[int] = None,
        include_expired: bool = False,
        touch: bool = True,
        default: Any = None,
    ) -> Any:
        if not cache_key or not str(cache_key).strip():
            raise ValidationError(
                "cache_key is required.",
                context=self._context("get_cache"),
            )

        try:
            with self.lock:
                self._stats["reads"] += 1
                version_key = (namespace, str(cache_key))

                target_entry_id: Optional[str] = None
                if version is None:
                    target_entry_id = self.cache_index.get(version_key)
                else:
                    for entry_id in reversed(self.version_index.get(version_key, [])):
                        entry = self.store.get(entry_id)
                        if entry and entry.metadata.version == version:
                            target_entry_id = entry_id
                            break

                if target_entry_id is None:
                    self._stats["misses"] += 1
                    self._stats["cache_misses"] += 1
                    return default

                entry = self.store.get(target_entry_id)
                if entry is None:
                    self._stats["misses"] += 1
                    self._stats["cache_misses"] += 1
                    return default

                if not include_expired and self._is_expired(entry):
                    self._delete_locked(target_entry_id)
                    self._stats["misses"] += 1
                    self._stats["cache_misses"] += 1
                    return default

                if touch:
                    self._touch_locked(target_entry_id)

                self._stats["hits"] += 1
                self._stats["cache_hits"] += 1
                return entry.data
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("get_cache", namespace=namespace, cache_key=cache_key),
                message="Failed to retrieve cache entry.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def query(
        self,
        data_type: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        filters: Optional[Mapping[str, Any]] = None,
        metadata_filters: Optional[Mapping[str, Any]] = None,
        *,
        limit: int = 100,
        match_all_tags: bool = False,
        include_expired: bool = False,
        sort_by: str = "metadata.updated_at",
        descending: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            normalized_tags = self._normalize_tags(tags)
            with self.lock:
                self._stats["reads"] += 1
                self._purge_expired_locked()

                if limit <= 0:
                    return []

                candidate_ids: Optional[set[str]] = None

                if normalized_tags:
                    tag_sets = [set(self.tag_index.get(tag, set())) for tag in normalized_tags]
                    if tag_sets:
                        candidate_ids = (
                            set.intersection(*tag_sets) if match_all_tags else set.union(*tag_sets)
                        )

                if data_type:
                    type_ids = {
                        entry_id
                        for entry_id, entry in self.store.items()
                        if entry.metadata.data_type == data_type
                    }
                    candidate_ids = type_ids if candidate_ids is None else candidate_ids.intersection(type_ids)

                if candidate_ids is None:
                    candidate_ids = set(self.store.keys())

                entries: List[Tuple[str, MemoryEntry]] = []
                for entry_id in candidate_ids:
                    entry = self.store.get(entry_id)
                    if entry is None:
                        continue
                    if not include_expired and self._is_expired(entry):
                        self._delete_locked(entry_id)
                        continue
                    if not self._matches_filters(entry, filters, metadata_filters):
                        continue
                    entries.append((entry_id, entry))

                def sort_value(item: Tuple[str, MemoryEntry]) -> Any:
                    _, entry = item
                    if sort_by.startswith("metadata."):
                        return self._safe_get_nested(entry.metadata.to_dict(), sort_by.split("metadata.", 1)[1])
                    return self._safe_get_nested(entry.data, sort_by)

                entries.sort(key=sort_value, reverse=descending)
                results = [entry.to_dict() for _, entry in entries[:limit]]

                if results:
                    self._stats["hits"] += len(results)
                else:
                    self._stats["misses"] += 1
                return results
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("query", data_type=data_type, tags=normalized_tags if 'normalized_tags' in locals() else tags),
                message="Failed to query finance memory.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def get_by_backtest(self, backtest_id: str) -> List[Dict[str, Any]]:
        return self.query(
            data_type="metric",
            tags=[f"backtest_{backtest_id}"],
            filters={"backtest_id": backtest_id},
            limit=100,
        )

    def get_model_state(self, model_version: str) -> Optional[Dict[str, Any]]:
        results = self.query(
            data_type="model",
            tags=[f"model_{model_version}"],
            metadata_filters={"extra.model_version": model_version},
            limit=1,
        )
        if not results:
            results = self.query(
                data_type="model",
                filters={"model_version": model_version},
                limit=1,
            )
        if not results:
            results = self.query(
                data_type="model",
                filters={"version": model_version},
                limit=1,
            )
        return results[0] if results else None

    def exists(self, entry_id: str) -> bool:
        with self.lock:
            entry = self.store.get(entry_id)
            if entry is None:
                return False
            if self._is_expired(entry):
                self._delete_locked(entry_id)
                return False
            return True

    # ------------------------------------------------------------------
    # Cleanup and maintenance
    # ------------------------------------------------------------------

    def clean_expired(
        self,
        *,
        max_age_days: Optional[int] = None,
        include_high_priority: bool = False,
        include_pinned: bool = False,
    ) -> int:
        try:
            with self.lock:
                removed = self._purge_expired_locked()
                if max_age_days is not None:
                    cutoff = self._now() - (int(max_age_days) * 86400)
                    stale_ids = []
                    for entry_id, entry in self.store.items():
                        if entry.metadata.created_at >= cutoff:
                            continue
                        if entry.metadata.pinned and not include_pinned:
                            continue
                        if entry.metadata.priority in {"high", "critical"} and not include_high_priority:
                            continue
                        stale_ids.append(entry_id)

                    for entry_id in stale_ids:
                        self._delete_locked(entry_id)
                    removed += len(stale_ids)

                self._cleanup_old_tags_locked()
                if removed:
                    self._record_mutation_locked()
                return removed
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context(
                    "clean_expired",
                    max_age_days=max_age_days,
                    include_high_priority=include_high_priority,
                    include_pinned=include_pinned,
                ),
                message="Failed to clean expired finance memory entries.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def _cleanup_old_tags_locked(self) -> int:
        if self.tag_retention_days <= 0:
            return 0
        cutoff = self._now() - (self.tag_retention_days * 86400)
        removed = 0
        for tag in list(self.tag_index.keys()):
            if self.tag_index[tag]:
                continue
            last_seen = self.tag_last_seen.get(tag, 0.0)
            if last_seen < cutoff:
                self.tag_index.pop(tag, None)
                self.tag_last_seen.pop(tag, None)
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def create_checkpoint(
        self,
        name: Optional[str] = None,
        *,
        include_config: bool = True,
    ) -> str:
        try:
            checkpoint_name = name or f"finance_memory_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

            with self.lock:
                wrapper = self._checkpoint_wrapper_locked(include_config=include_config)

            self._atomic_write_json(checkpoint_path, wrapper)

            with self.lock:
                self.last_checkpoint = checkpoint_path
                self._stats["checkpoints_created"] += 1

            logger.info("Created finance memory checkpoint: %s", checkpoint_path)
            return checkpoint_path

        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = CheckpointWriteError(
                "Failed to create finance memory checkpoint.",
                context=self._context(
                    "create_checkpoint", checkpoint_dir=self.checkpoint_dir, name=name
                ),
                details={"checkpoint_dir": self.checkpoint_dir, "name": name},
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def load_checkpoint(
        self,
        path: str,
        *,
        merge: bool = False,
        restore_config: bool = False,
    ) -> bool:
        try:
            if not path or not str(path).strip():
                raise ValidationError(
                    "Checkpoint path is required.",
                    context=self._context("load_checkpoint"),
                )
            if not os.path.exists(path):
                raise StateStoreUnavailableError(
                    "Checkpoint file does not exist.",
                    context=self._context("load_checkpoint", path=path),
                    details={"path": path},
                )

            with open(path, "r", encoding="utf-8") as handle:
                wrapper = json.load(handle)

            payload = self._validate_checkpoint_payload(wrapper)
            store_payload = payload.get("store", {})
            restored_entries = OrderedDict(
                (entry_id, MemoryEntry.from_dict(entry_payload))
                for entry_id, entry_payload in store_payload.items()
            )

            with self.lock:
                if merge:
                    for entry_id, entry in restored_entries.items():
                        self.store[entry_id] = entry
                else:
                    self.store = restored_entries

                self.access_counter = int(payload.get("meta", {}).get("access_counter", self.access_counter))
                self._mutation_count = int(payload.get("meta", {}).get("mutation_count", self._mutation_count))
                self._stats.update(payload.get("meta", {}).get("stats", {}))
                self.last_checkpoint = path
                self._rebuild_indexes_locked()
                self._enforce_capacity_locked()
                self._stats["checkpoints_loaded"] += 1

                if restore_config:
                    config_snapshot = payload.get("meta", {}).get("config_snapshot")
                    if isinstance(config_snapshot, Mapping):
                        self.config.update(config_snapshot)

            logger.info("Loaded finance memory checkpoint: %s", path)
            return True

        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context(
                    "load_checkpoint",
                    path=path,
                    merge=merge,
                    restore_config=restore_config,
                ),
                message="Failed to load finance memory checkpoint.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def list_checkpoints(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            checkpoint_dir = Path(self.checkpoint_dir)
            if not checkpoint_dir.exists():
                return []

            files = sorted(
                [path for path in checkpoint_dir.glob("*.json") if path.is_file()],
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            results = []
            for file_path in files[: max(limit, 0)]:
                stat = file_path.stat()
                results.append(
                    {
                        "path": str(file_path),
                        "name": file_path.name,
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
                    }
                )
            return results
        except Exception as exc:
            handled = classify_external_exception(
                exc,
                context=self._context("list_checkpoints", checkpoint_dir=self.checkpoint_dir),
                message="Failed to list finance memory checkpoints.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def restore_latest_checkpoint(self, *, merge: bool = False, restore_config: bool = False) -> Optional[str]:
        checkpoints = self.list_checkpoints(limit=1)
        if not checkpoints:
            return None
        latest = checkpoints[0]["path"]
        self.load_checkpoint(latest, merge=merge, restore_config=restore_config)
        return latest

    # ------------------------------------------------------------------
    # Reporting and utilities
    # ------------------------------------------------------------------

    @contextmanager
    def checkpoint_on_success(self, name: Optional[str] = None) -> Iterator[MutableMapping[str, Any]]:
        state: MutableMapping[str, Any] = {"ok": True, "checkpoint_path": None}
        try:
            yield state
            checkpoint_path = self.create_checkpoint(name=name)
            state["checkpoint_path"] = checkpoint_path
        except BaseException as exc:
            state["ok"] = False
            handled = exc if isinstance(exc, FinancialAgentError) else classify_external_exception(
                exc,
                context=self._context("checkpoint_on_success"),
                message="Operation wrapped by checkpoint_on_success failed.",
            )
            log_error(handled, logger_=logger)
            raise handled from exc

    def export_snapshot(self, *, include_entries: bool = False, include_config: bool = False) -> Dict[str, Any]:
        with self.lock:
            snapshot = {
                "summary": self.get_statistics(),
                "last_checkpoint": self.last_checkpoint,
                "checkpoint_dir": self.checkpoint_dir,
            }
            if include_config:
                snapshot["config"] = self._make_json_safe(self.config)
            if include_entries:
                snapshot["entries"] = [entry.to_dict() for entry in self.store.values()]
            return snapshot

    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            type_distribution: Dict[str, int] = defaultdict(int)
            priority_distribution: Dict[str, int] = defaultdict(int)
            namespaces: Dict[str, int] = defaultdict(int)

            for entry in self.store.values():
                meta = entry.metadata
                type_distribution[meta.data_type] += 1
                priority_distribution[meta.priority] += 1
                if meta.namespace:
                    namespaces[meta.namespace] += 1

            total_entries = len(self.store)
            total_size_bytes = self._current_size_bytes_locked()
            expired_entries = sum(1 for entry in self.store.values() if self._is_expired(entry))
            cache_entries = sum(1 for entry in self.store.values() if entry.metadata.data_type == "cache")

            hit_denominator = self._stats["hits"] + self._stats["misses"]
            cache_hit_denominator = self._stats["cache_hits"] + self._stats["cache_misses"]

            return {
                "total_entries": total_entries,
                "total_size_bytes": total_size_bytes,
                "total_size_mb": round(total_size_bytes / (1024 * 1024), 4),
                "max_entries": self.max_size,
                "max_memory_bytes": self.max_memory_bytes,
                "utilization_ratio": round(total_entries / self.max_size, 4) if self.max_size else 0.0,
                "type_distribution": dict(type_distribution),
                "priority_distribution": dict(priority_distribution),
                "namespace_distribution": dict(namespaces),
                "tag_count": len(self.tag_index),
                "cache_entries": cache_entries,
                "expired_entries": expired_entries,
                "checkpoint_info": {
                    "last_checkpoint": self.last_checkpoint,
                    "checkpoint_dir": self.checkpoint_dir,
                    "checkpoint_freq": self.checkpoint_freq,
                    "auto_save": self.auto_save,
                },
                "stats": {
                    **self._stats,
                    "hit_rate": round(self._stats["hits"] / hit_denominator, 4) if hit_denominator else None,
                    "cache_hit_rate": (
                        round(self._stats["cache_hits"] / cache_hit_denominator, 4)
                        if cache_hit_denominator
                        else None
                    ),
                },
            }


__all__ = ["FinanceMemory", "EntryMetadata", "MemoryEntry"]


if __name__ == "__main__":  # pragma: no cover - demonstration only
    memory = FinanceMemory()

    market_data_id = memory.add_financial_data(
        data={
            "symbol": "AAPL",
            "open": 182.3,
            "high": 184.2,
            "low": 181.7,
            "close": 183.5,
            "volume": 12543210,
        },
        data_type="market",
        tags=["tech", "nasdaq"],
        priority="medium",
        metadata={"symbol": "AAPL"},
    )

    cache_id = memory.set_cache(
        "feature_matrix:AAPL",
        {"rows": 120, "cols": 48, "generated_at": datetime.utcnow()},
        namespace="features",
        ttl_seconds=120,
        tags=["aapl", "features"],
        priority="high",
    )

    model_id = memory.add_financial_data(
        data={
            "model_type": "lstm",
            "model_version": "v3.1",
            "accuracy": 0.87,
        },
        data_type="model",
        tags=["production", "model_v3.1"],
        priority="critical",
        metadata={"model_version": "v3.1"},
        pinned=True,
    )

    print("Market entry:", memory.get(market_data_id))
    print("Cached features:", memory.get_cache("feature_matrix:AAPL", namespace="features"))
    print("Model state:", memory.get_model_state("v3.1"))

    checkpoint_path = memory.create_checkpoint("finance_memory_demo.json")
    print("Checkpoint created:", checkpoint_path)
    print("Statistics:", json.dumps(memory.get_statistics(), indent=2))
