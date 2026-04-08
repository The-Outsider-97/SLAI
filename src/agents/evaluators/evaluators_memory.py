
from __future__ import annotations

import json
import os
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from collections import defaultdict, OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

from .utils.config_loader import load_global_config, get_config_section
from .utils.evaluation_errors import (ConfigLoadError, EvaluationError, OperationalError,
                                      MemoryAccessError, ValidationFailureError)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Evaluators Memory")
printer = PrettyPrinter


@dataclass(slots=True)
class MemoryEntryMetadata:
    """Metadata describing a stored evaluation-memory entry."""

    entry_id: str
    created_at: str
    updated_at: str
    last_accessed_at: str
    access_count: int
    priority: str
    priority_value: int
    tags: List[str]
    size_bytes: int
    checksum: str
    source: Optional[str] = None
    category: str = "general"
    expires_at: Optional[str] = None
    version: int = 1
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed_at = _utcnow().isoformat()

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return _parse_timestamp(self.expires_at) <= _utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryEntry:
    """Serialized record stored inside the evaluation memory."""

    data: Any
    metadata: MemoryEntryMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "metadata": self.metadata.to_dict(),
        }


class EvaluatorsMemory:
    """
    Production-grade memory system for evaluation workflows.

    Responsibilities
    ----------------
    - Cache evaluation artifacts with rich metadata and tags
    - Support LRU/FIFO eviction with priority-aware retention
    - Persist and restore checkpoints safely
    - Provide flexible querying and full-text search over stored entries
    - Integrate with the shared evaluation error model for diagnostics
    """

    _DEFAULT_PRIORITY_LABELS = ["low", "medium", "high", "critical"]

    def __init__(self) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))
        self.memory_config = get_config_section("evaluators_memory")

        self.max_size = self._require_positive_integer(
            self.memory_config.get("max_size", 5000),
            "evaluators_memory.max_size",
        )
        self.eviction_policy = self._normalize_eviction_policy(
            self.memory_config.get("eviction_policy", "LRU")
        )
        self.auto_save = bool(self.memory_config.get("auto_save", True))
        self.tag_retention_days = self._require_non_negative_integer(
            self.memory_config.get("tag_retention", 7),
            "evaluators_memory.tag_retention",
        )
        self.priority_levels = self._require_positive_integer(
            self.memory_config.get("priority_levels", 3),
            "evaluators_memory.priority_levels",
        )
        self.checkpoint_freq = self._require_positive_integer(
            self.memory_config.get("checkpoint_freq", 500),
            "evaluators_memory.checkpoint_freq",
        )
        self.access_counter = self._require_non_negative_integer(
            self.memory_config.get("access_count", 0),
            "evaluators_memory.access_count",
        )
        self.operation_counter = 0
        self.priority_map = self._build_priority_map(self.priority_levels)
        self.default_priority = "medium" if "medium" in self.priority_map else next(iter(self.priority_map))

        checkpoint_dir = self.memory_config.get("checkpoint_dir", "src/agents/evaluators/checkpoints")
        self.checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_dir)

        self.store: "OrderedDict[str, MemoryEntry]" = OrderedDict()
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.priority_index: Dict[str, Set[str]] = defaultdict(set)
        self.lock = RLock()
        self.last_checkpoint: Optional[str] = None
        self.stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "updates": 0,
            "deletes": 0,
            "evictions": 0,
            "queries": 0,
            "searches": 0,
            "checkpoints_created": 0,
            "checkpoints_loaded": 0,
        }

        self._init_checkpoint_dir()
        logger.info(
            "EvaluatorsMemory initialized: capacity=%d policy=%s checkpoint_dir=%s",
            self.max_size,
            self.eviction_policy,
            self.checkpoint_dir,
        )

    # ------------------------------------------------------------------
    # Public storage API
    # ------------------------------------------------------------------

    def add(
        self,
        entry: Any,
        tags: Optional[Sequence[str]] = None,
        priority: str | int = "medium",
        *,
        entry_id: Optional[str] = None,
        source: Optional[str] = None,
        category: str = "general",
        metadata: Optional[Mapping[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        replace: bool = False,
    ) -> str:
        """
        Add an evaluation artifact to memory.

        Parameters
        ----------
        entry:
            Arbitrary payload to cache.
        tags:
            Zero or more tags used for retrieval and grouping.
        priority:
            String or integer priority indicator.
        entry_id:
            Optional explicit key. When omitted, an ID is generated.
        source:
            Optional source system or evaluator name.
        category:
            Logical grouping, e.g. ``performance`` or ``security``.
        metadata:
            Additional free-form metadata stored alongside the entry.
        ttl_seconds:
            Optional per-entry expiry.
        replace:
            Allow replacing an existing entry when ``entry_id`` is supplied.
        """
        normalized_tags = self._normalize_tags(tags)
        priority_label, priority_value = self._normalize_priority(priority)
        normalized_source = self._normalize_optional_string(source, "source")
        normalized_category = self._normalize_non_empty_string(category, "category")
        custom_metadata = self._normalize_metadata_mapping(metadata)

        with self.lock:
            target_id = entry_id or self._generate_entry_id()
            if target_id in self.store and not replace:
                raise MemoryAccessError("add", target_id, "entry already exists")

            if target_id in self.store and replace:
                self._remove_entry(target_id, reason="replace")

            now = _utcnow()
            serialized_entry = self._safe_serialize(entry)
            size_bytes = len(serialized_entry.encode("utf-8"))
            checksum = hashlib.sha256(serialized_entry.encode("utf-8")).hexdigest()
            expires_at = (
                (now + timedelta(seconds=self._require_positive_integer(ttl_seconds, "ttl_seconds")))
                .isoformat()
                if ttl_seconds is not None
                else None
            )

            entry_metadata = MemoryEntryMetadata(
                entry_id=target_id,
                created_at=now.isoformat(),
                updated_at=now.isoformat(),
                last_accessed_at=now.isoformat(),
                access_count=0,
                priority=priority_label,
                priority_value=priority_value,
                tags=normalized_tags,
                size_bytes=size_bytes,
                checksum=checksum,
                source=normalized_source,
                category=normalized_category,
                expires_at=expires_at,
                custom_metadata=custom_metadata,
            )

            self.store[target_id] = MemoryEntry(data=entry, metadata=entry_metadata)
            if self.eviction_policy == "LRU":
                self.store.move_to_end(target_id)

            self._index_entry(target_id, self.store[target_id])
            self._evict_until_within_capacity()
            self._bump_operation_counters("writes")

        return target_id

    def add_evaluation_result(
        self,
        evaluator_name: str,
        result: Mapping[str, Any],
        *,
        tags: Optional[Sequence[str]] = None,
        priority: str | int = "medium",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Store a structured evaluator result with normalized metadata."""
        name = self._normalize_non_empty_string(evaluator_name, "evaluator_name")
        if not isinstance(result, Mapping):
            raise ValidationFailureError(
                "evaluation_result_mapping",
                type(result).__name__,
                "mapping",
            )

        combined_tags = set(self._normalize_tags(tags))
        combined_tags.update({"evaluation", name.casefold()})
        payload = {
            "evaluator": name,
            "result": dict(result),
            "stored_at": _utcnow().isoformat(),
        }
        return self.add(
            payload,
            tags=sorted(combined_tags),
            priority=priority,
            source=name,
            category="evaluation",
            metadata=metadata,
        )

    def add_error(
        self,
        error: EvaluationError,
        *,
        tags: Optional[Sequence[str]] = None,
        priority: str | int = "high",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Store a structured evaluation error object for later inspection."""
        if not isinstance(error, EvaluationError):
            raise ValidationFailureError("evaluation_error_instance", type(error).__name__, "EvaluationError")

        combined_tags = set(self._normalize_tags(tags))
        combined_tags.update({"error", error.error_type.name.lower()})
        payload = error.to_audit_dict()
        return self.add(
            payload,
            tags=sorted(combined_tags),
            priority=priority,
            source="evaluation_error",
            category="errors",
            metadata=metadata,
        )

    def get(
        self,
        entry_id: Optional[str] = None,
        tag: Optional[str] = None,
        *,
        include_expired: bool = False,
    ) -> Any:
        """
        Retrieve entries by ID or by tag.

        Returns a single serialized entry when ``entry_id`` is provided, a list
        of serialized entries when ``tag`` is provided, or all entries
        otherwise.
        """
        with self.lock:
            if entry_id is not None:
                entry = self.store.get(entry_id)
                if entry is None:
                    self.stats["misses"] += 1
                    return None

                if entry.metadata.is_expired() and not include_expired:
                    self._remove_entry(entry_id, reason="expired")
                    self.stats["misses"] += 1
                    return None

                self._touch_entry(entry_id, entry)
                self.stats["hits"] += 1
                return entry.to_dict()

            if tag is not None:
                normalized_tag = self._normalize_non_empty_string(tag, "tag")
                records = []
                for candidate_id in list(self.tag_index.get(normalized_tag, set())):
                    record = self.store.get(candidate_id)
                    if record is None:
                        self.tag_index[normalized_tag].discard(candidate_id)
                        continue
                    if record.metadata.is_expired() and not include_expired:
                        self._remove_entry(candidate_id, reason="expired")
                        continue
                    records.append(record.to_dict())
                return records

            results = []
            for candidate_id, record in list(self.store.items()):
                if record.metadata.is_expired() and not include_expired:
                    self._remove_entry(candidate_id, reason="expired")
                    continue
                results.append(record.to_dict())
            return results

    def update(
        self,
        entry_id: str,
        *,
        entry: Any = None,
        tags: Optional[Sequence[str]] = None,
        priority: Optional[str | int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update an existing entry in place and rebuild indexes as needed."""
        with self.lock:
            record = self.store.get(entry_id)
            if record is None:
                raise MemoryAccessError("update", entry_id, "entry not found")

            self._deindex_entry(entry_id, record)

            if entry is not None:
                record.data = entry
                serialized_entry = self._safe_serialize(entry)
                record.metadata.size_bytes = len(serialized_entry.encode("utf-8"))
                record.metadata.checksum = hashlib.sha256(serialized_entry.encode("utf-8")).hexdigest()

            if tags is not None:
                record.metadata.tags = self._normalize_tags(tags)

            if priority is not None:
                priority_label, priority_value = self._normalize_priority(priority)
                record.metadata.priority = priority_label
                record.metadata.priority_value = priority_value

            if metadata is not None:
                merged = dict(record.metadata.custom_metadata)
                merged.update(self._normalize_metadata_mapping(metadata))
                record.metadata.custom_metadata = merged

            if source is not None:
                record.metadata.source = self._normalize_optional_string(source, "source")

            if category is not None:
                record.metadata.category = self._normalize_non_empty_string(category, "category")

            if ttl_seconds is not None:
                if ttl_seconds <= 0:
                    record.metadata.expires_at = None
                else:
                    record.metadata.expires_at = (
                        _utcnow() + timedelta(seconds=self._require_positive_integer(ttl_seconds, "ttl_seconds"))
                    ).isoformat()

            record.metadata.updated_at = _utcnow().isoformat()
            self._touch_entry(entry_id, record, count_as_access=False)
            self._index_entry(entry_id, record)
            self._bump_operation_counters("updates")
            return record.to_dict()

    def remove(self, entry_id: str) -> bool:
        """Remove a stored entry and return whether it existed."""
        with self.lock:
            if entry_id not in self.store:
                return False
            self._remove_entry(entry_id, reason="remove")
            return True

    def clear(self) -> None:
        """Remove all in-memory entries and reset indexes."""
        with self.lock:
            self.store.clear()
            self.tag_index.clear()
            self.category_index.clear()
            self.priority_index.clear()

    # ------------------------------------------------------------------
    # Query and search
    # ------------------------------------------------------------------

    def query(
        self,
        tags: Optional[Sequence[str]] = None,
        filters: Optional[Sequence[str]] = None,
        *,
        limit: int = 10,
        match_all_tags: bool = False,
        include_expired: bool = False,
        sort_by: str = "recent",
    ) -> List[Dict[str, Any]]:
        """Query entries by tags and lightweight filter expressions."""
        limit_value = self._require_positive_integer(limit, "limit")
        normalized_tags = self._normalize_tags(tags) if tags else []
        parsed_filters = [self._parse_filter_expression(item) for item in (filters or [])]

        with self.lock:
            self.stats["queries"] += 1
            candidate_ids = self._select_candidate_ids(normalized_tags, match_all_tags)
            records = []

            for entry_id in candidate_ids:
                record = self.store.get(entry_id)
                if record is None:
                    continue
                if record.metadata.is_expired() and not include_expired:
                    self._remove_entry(entry_id, reason="expired")
                    continue
                if parsed_filters and not self._matches_filters(record, parsed_filters):
                    continue
                records.append(record)

            sorted_records = self._sort_records(records, sort_by=sort_by)
            return [record.to_dict() for record in sorted_records[:limit_value]]

    def search_entries(
        self,
        search_term: str,
        fields: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        case_sensitive: bool = False,
        *,
        limit: int = 50,
        include_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform text search across entry data and optional metadata fields.
        """
        normalized_term = self._normalize_non_empty_string(search_term, "search_term")
        limit_value = self._require_positive_integer(limit, "limit")
        normalized_tags = self._normalize_tags(tags) if tags else []

        matcher = (
            (lambda content: normalized_term in content)
            if case_sensitive
            else (lambda content: normalized_term.casefold() in content.casefold())
        )

        with self.lock:
            self.stats["searches"] += 1
            candidate_ids = self._select_candidate_ids(normalized_tags, match_all_tags=False)
            results: List[Dict[str, Any]] = []

            for entry_id in candidate_ids:
                record = self.store.get(entry_id)
                if record is None:
                    continue
                if record.metadata.is_expired():
                    self._remove_entry(entry_id, reason="expired")
                    continue

                search_spaces = []
                if fields:
                    for field_name in fields:
                        value = self._extract_field_value(record, field_name)
                        if value is not None:
                            search_spaces.append(str(value))
                else:
                    search_spaces.append(self._safe_serialize(record.data))
                    if include_metadata:
                        search_spaces.append(self._safe_serialize(record.metadata.to_dict()))

                combined = "\n".join(search_spaces)
                if matcher(combined):
                    results.append(record.to_dict())
                    if len(results) >= limit_value:
                        break

            return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def create_checkpoint(self, name: Optional[str] = None, *, include_config: bool = False) -> str:
        """Persist the full memory state to a checkpoint file."""
        checkpoint_name = self._normalize_checkpoint_name(name)
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        payload = {
            "format_version": 1,
            "created_at": _utcnow().isoformat(),
            "store": [record.to_dict() for record in self.store.values()],
            "stats": dict(self.stats),
            "counters": {
                "access_counter": self.access_counter,
                "operation_counter": self.operation_counter,
            },
        }
        if include_config:
            payload["config"] = {"evaluators_memory": dict(self.memory_config)}

        try:
            with open(checkpoint_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
        except (OSError, TypeError, ValueError) as exc:
            raise MemoryAccessError("checkpoint_write", str(checkpoint_path), str(exc)) from exc

        self.last_checkpoint = str(checkpoint_path)
        self.stats["checkpoints_created"] += 1
        logger.info("Checkpoint created: %s", checkpoint_path)
        return str(checkpoint_path)

    def load_checkpoint(self, path: str, *, merge: bool = False) -> Dict[str, Any]:
        """Load a previously created checkpoint and rebuild all indexes."""
        checkpoint_path = Path(self._normalize_non_empty_string(path, "path"))
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise MemoryAccessError("checkpoint_read", str(checkpoint_path), str(exc)) from exc

        if not isinstance(payload, Mapping):
            raise ValidationFailureError("checkpoint_payload", type(payload).__name__, "mapping")
        if "store" not in payload:
            raise ValidationFailureError("checkpoint_store_field", list(payload.keys()), "store")

        loaded_entries = self._deserialize_checkpoint_entries(payload["store"])

        with self.lock:
            if not merge:
                self.clear()

            for record in loaded_entries:
                self.store[record.metadata.entry_id] = record

            self._rebuild_indexes()
            counters = payload.get("counters", {})
            if isinstance(counters, Mapping):
                self.access_counter = int(counters.get("access_counter", self.access_counter))
                self.operation_counter = int(counters.get("operation_counter", self.operation_counter))

            self.last_checkpoint = str(checkpoint_path)
            self.stats["checkpoints_loaded"] += 1

        logger.info(
            "Checkpoint loaded: %s entries=%d merge=%s",
            checkpoint_path,
            len(loaded_entries),
            merge,
        )
        return {
            "path": str(checkpoint_path),
            "loaded_entries": len(loaded_entries),
            "merge": merge,
        }

    def list_checkpoints(self) -> List[str]:
        """List available checkpoint files ordered by most recent first."""
        if not self.checkpoint_dir.exists():
            return []
        candidates = sorted(
            (path for path in self.checkpoint_dir.iterdir() if path.is_file() and path.suffix == ".json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        return [str(path) for path in candidates]

    # ------------------------------------------------------------------
    # Maintenance and statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return memory-system utilization, indexing, and checkpoint statistics."""
        with self.lock:
            expired_count = sum(1 for record in self.store.values() if record.metadata.is_expired())
            total_bytes = sum(record.metadata.size_bytes for record in self.store.values())

            return {
                "total_entries": len(self.store),
                "capacity": {
                    "max_size": self.max_size,
                    "utilization_ratio": (len(self.store) / self.max_size) if self.max_size else 0.0,
                    "eviction_policy": self.eviction_policy,
                },
                "tag_distribution": {tag: len(entry_ids) for tag, entry_ids in sorted(self.tag_index.items())},
                "category_distribution": {
                    category: len(entry_ids) for category, entry_ids in sorted(self.category_index.items())
                },
                "priority_distribution": {
                    priority: len(entry_ids) for priority, entry_ids in sorted(self.priority_index.items())
                },
                "memory_usage_bytes": total_bytes,
                "expired_entries": expired_count,
                "counters": {
                    "access_counter": self.access_counter,
                    "operation_counter": self.operation_counter,
                },
                "checkpoint_info": {
                    "last_checkpoint": self.last_checkpoint,
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "available_checkpoints": self.list_checkpoints(),
                },
                "stats": dict(self.stats),
            }

    def clean_expired_tags(self) -> int:
        """
        Remove tags older than the configured retention horizon without deleting
        the underlying entry.
        """
        cutoff = _utcnow() - timedelta(days=self.tag_retention_days)
        removed_tags = 0

        with self.lock:
            for record in self.store.values():
                created_at = _parse_timestamp(record.metadata.created_at)
                if created_at > cutoff:
                    continue
                if record.metadata.tags:
                    removed_tags += len(record.metadata.tags)
                    record.metadata.tags = []
                    record.metadata.updated_at = _utcnow().isoformat()

            self._rebuild_indexes()
        return removed_tags

    def prune_expired_entries(self) -> int:
        """Remove all entries whose explicit TTL has expired."""
        removed = 0
        with self.lock:
            for entry_id, record in list(self.store.items()):
                if record.metadata.is_expired():
                    self._remove_entry(entry_id, reason="expired")
                    removed += 1
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_checkpoint_dir(self) -> None:
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        except OSError as exc:
            raise OperationalError(
                f"Failed to initialize checkpoint directory: {self.checkpoint_dir}",
                context={"error": str(exc), "checkpoint_dir": str(self.checkpoint_dir)},
            ) from exc

    def _resolve_checkpoint_dir(self, configured_path: Any) -> Path:
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ConfigLoadError(self.config_path, "evaluators_memory.checkpoint_dir", "path must be a non-empty string")

        candidate = Path(configured_path.strip())
        if candidate.is_absolute():
            return candidate

        config_file = self.config.get("__config_path__")
        if not config_file:
            return candidate

        config_path = Path(config_file).resolve()

        if candidate.parts and candidate.parts[0] == "src" and "src" in config_path.parts:
            src_index = config_path.parts.index("src")
            root = Path(*config_path.parts[:src_index]) if src_index > 0 else Path("/")
            return root / candidate

        return candidate

    def _build_priority_map(self, level_count: int) -> Dict[str, int]:
        if level_count <= 0:
            raise ConfigLoadError(self.config_path, "evaluators_memory.priority_levels", "must be positive")

        if level_count <= len(self._DEFAULT_PRIORITY_LABELS):
            labels = self._DEFAULT_PRIORITY_LABELS[:level_count]
        else:
            labels = self._DEFAULT_PRIORITY_LABELS + [f"p{index}" for index in range(len(self._DEFAULT_PRIORITY_LABELS), level_count)]
        return {label: index for index, label in enumerate(labels)}

    def _normalize_eviction_policy(self, value: Any) -> str:
        policy = str(value or "LRU").strip().upper()
        if policy not in {"LRU", "FIFO"}:
            raise ConfigLoadError(self.config_path, "evaluators_memory.eviction_policy", f"unsupported policy '{value}'")
        return policy

    def _normalize_tags(self, tags: Optional[Sequence[str]]) -> List[str]:
        if tags is None:
            return []
        if isinstance(tags, str):
            tags = [tags]

        normalized = []
        seen = set()
        for tag in tags:
            clean = self._normalize_non_empty_string(tag, "tag")
            if clean not in seen:
                normalized.append(clean)
                seen.add(clean)
        return normalized

    def _normalize_priority(self, priority: str | int) -> tuple[str, int]:
        if isinstance(priority, int):
            if priority < 0 or priority >= self.priority_levels:
                raise ValidationFailureError("priority_range", priority, f"0..{self.priority_levels - 1}")
            label = [name for name, value in self.priority_map.items() if value == priority][0]
            return label, priority

        label = str(priority or self.default_priority).strip().lower()
        if label not in self.priority_map:
            raise ValidationFailureError("priority_label", label, list(self.priority_map.keys()))
        return label, self.priority_map[label]

    def _normalize_metadata_mapping(self, metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if metadata is None:
            return {}
        if not isinstance(metadata, Mapping):
            raise ValidationFailureError("metadata_mapping", type(metadata).__name__, "mapping")
        return dict(metadata)

    def _normalize_optional_string(self, value: Optional[str], field_name: str) -> Optional[str]:
        if value is None:
            return None
        normalized = self._normalize_non_empty_string(value, field_name)
        return normalized

    def _normalize_non_empty_string(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationFailureError(field_name, value, "non-empty string")
        return value.strip()

    def _generate_entry_id(self) -> str:
        now = _utcnow()
        digest = hashlib.sha256(f"{now.isoformat()}:{self.access_counter}:{self.operation_counter}".encode("utf-8")).hexdigest()[:12]
        return f"eval_{now.strftime('%Y%m%dT%H%M%S%f')}_{digest}"

    def _touch_entry(self, entry_id: str, record: MemoryEntry, *, count_as_access: bool = True) -> None:
        if count_as_access:
            record.metadata.touch()
            self.access_counter += 1
        else:
            record.metadata.last_accessed_at = _utcnow().isoformat()
        if self.eviction_policy == "LRU":
            self.store.move_to_end(entry_id)

    def _index_entry(self, entry_id: str, record: MemoryEntry) -> None:
        for tag in record.metadata.tags:
            self.tag_index[tag].add(entry_id)
        self.category_index[record.metadata.category].add(entry_id)
        self.priority_index[record.metadata.priority].add(entry_id)

    def _deindex_entry(self, entry_id: str, record: MemoryEntry) -> None:
        for tag in record.metadata.tags:
            entry_ids = self.tag_index.get(tag)
            if entry_ids:
                entry_ids.discard(entry_id)
                if not entry_ids:
                    self.tag_index.pop(tag, None)

        category_entries = self.category_index.get(record.metadata.category)
        if category_entries:
            category_entries.discard(entry_id)
            if not category_entries:
                self.category_index.pop(record.metadata.category, None)

        priority_entries = self.priority_index.get(record.metadata.priority)
        if priority_entries:
            priority_entries.discard(entry_id)
            if not priority_entries:
                self.priority_index.pop(record.metadata.priority, None)

    def _remove_entry(self, entry_id: str, *, reason: str) -> None:
        record = self.store.get(entry_id)
        if record is None:
            return

        self._deindex_entry(entry_id, record)
        del self.store[entry_id]
        if reason == "eviction":
            self.stats["evictions"] += 1
        else:
            self.stats["deletes"] += 1

    def _evict_until_within_capacity(self) -> None:
        while len(self.store) > self.max_size:
            candidate = self._select_eviction_candidate()
            if candidate is None:
                break
            self._remove_entry(candidate, reason="eviction")

    def _select_eviction_candidate(self) -> Optional[str]:
        if not self.store:
            return None

        expired_ids = [entry_id for entry_id, record in self.store.items() if record.metadata.is_expired()]
        if expired_ids:
            return expired_ids[0]

        if self.eviction_policy == "FIFO":
            return next(iter(self.store))

        lowest_priority = min(record.metadata.priority_value for record in self.store.values())
        for entry_id, record in self.store.items():
            if record.metadata.priority_value == lowest_priority:
                return entry_id
        return next(iter(self.store))

    def _bump_operation_counters(self, stat_name: str) -> None:
        self.operation_counter += 1
        self.stats[stat_name] += 1
        if self.auto_save and self.operation_counter % self.checkpoint_freq == 0:
            self.create_checkpoint()

    def _safe_serialize(self, value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError("entry_serialization", type(value).__name__, f"JSON-serializable payload: {exc}") from exc

    def _select_candidate_ids(self, tags: Sequence[str], match_all_tags: bool) -> List[str]:
        if not tags:
            return list(self.store.keys())

        tag_sets = [self.tag_index.get(tag, set()) for tag in tags]
        if not tag_sets:
            return []

        if match_all_tags:
            candidate_ids = set.intersection(*tag_sets) if all(tag_sets) else set()
        else:
            candidate_ids = set.union(*tag_sets)

        ordered = [entry_id for entry_id in self.store.keys() if entry_id in candidate_ids]
        return ordered

    def _parse_filter_expression(self, expression: str) -> Dict[str, str]:
        if not isinstance(expression, str) or ":" not in expression:
            raise ValidationFailureError("filter_expression", expression, "field:value")
        field_name, expected = expression.split(":", 1)
        return {
            "field": self._normalize_non_empty_string(field_name, "filter_field"),
            "expected": expected.strip(),
        }

    def _matches_filters(self, record: MemoryEntry, filters: Sequence[Mapping[str, str]]) -> bool:
        for definition in filters:
            actual = self._extract_field_value(record, definition["field"])
            if actual is None:
                return False
            if str(actual) != definition["expected"]:
                return False
        return True

    def _extract_field_value(self, record: MemoryEntry, field_name: str) -> Any:
        normalized = field_name.strip()
        if normalized.startswith("metadata."):
            return self._extract_nested_value(record.metadata.to_dict(), normalized.split(".")[1:])
        if normalized.startswith("data."):
            base = record.data if isinstance(record.data, Mapping) else {"value": record.data}
            return self._extract_nested_value(base, normalized.split(".")[1:])

        if isinstance(record.data, Mapping):
            value = self._extract_nested_value(record.data, normalized.split("."))
            if value is not None:
                return value

        metadata_value = self._extract_nested_value(record.metadata.to_dict(), normalized.split("."))
        return metadata_value

    def _extract_nested_value(self, payload: Any, path: Sequence[str]) -> Any:
        current = payload
        for token in path:
            if isinstance(current, Mapping):
                if token not in current:
                    return None
                current = current[token]
            elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
                try:
                    current = current[int(token)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return current

    def _sort_records(self, records: Sequence[MemoryEntry], *, sort_by: str) -> List[MemoryEntry]:
        mode = str(sort_by or "recent").strip().lower()
        if mode == "oldest":
            return sorted(records, key=lambda item: item.metadata.created_at)
        if mode == "access_count":
            return sorted(records, key=lambda item: item.metadata.access_count, reverse=True)
        if mode == "priority":
            return sorted(
                records,
                key=lambda item: (item.metadata.priority_value, item.metadata.last_accessed_at),
                reverse=True,
            )
        if mode == "size":
            return sorted(records, key=lambda item: item.metadata.size_bytes, reverse=True)
        return sorted(records, key=lambda item: item.metadata.last_accessed_at, reverse=True)

    def _deserialize_checkpoint_entries(self, payload: Any) -> List[MemoryEntry]:
        if not isinstance(payload, list):
            raise ValidationFailureError("checkpoint_entries", type(payload).__name__, "list")

        records: List[MemoryEntry] = []
        for item in payload:
            if not isinstance(item, Mapping):
                raise ValidationFailureError("checkpoint_entry_item", type(item).__name__, "mapping")
            metadata_payload = item.get("metadata")
            if not isinstance(metadata_payload, Mapping):
                raise ValidationFailureError("checkpoint_entry_metadata", type(metadata_payload).__name__, "mapping")

            metadata = MemoryEntryMetadata(
                entry_id=self._normalize_non_empty_string(metadata_payload.get("entry_id"), "entry_id"),
                created_at=_parse_timestamp(str(metadata_payload.get("created_at"))).isoformat(),
                updated_at=_parse_timestamp(str(metadata_payload.get("updated_at"))).isoformat(),
                last_accessed_at=_parse_timestamp(str(metadata_payload.get("last_accessed_at"))).isoformat(),
                access_count=self._require_non_negative_integer(metadata_payload.get("access_count", 0), "access_count"),
                priority=self._normalize_non_empty_string(metadata_payload.get("priority"), "priority"),
                priority_value=self._require_non_negative_integer(metadata_payload.get("priority_value", 0), "priority_value"),
                tags=self._normalize_tags(metadata_payload.get("tags", [])),
                size_bytes=self._require_non_negative_integer(metadata_payload.get("size_bytes", 0), "size_bytes"),
                checksum=self._normalize_non_empty_string(metadata_payload.get("checksum"), "checksum"),
                source=metadata_payload.get("source"),
                category=self._normalize_non_empty_string(metadata_payload.get("category", "general"), "category"),
                expires_at=str(metadata_payload.get("expires_at")) if metadata_payload.get("expires_at") else None,
                version=self._require_positive_integer(metadata_payload.get("version", 1), "version"),
                custom_metadata=self._normalize_metadata_mapping(metadata_payload.get("custom_metadata", {})),
            )
            records.append(MemoryEntry(data=item.get("data"), metadata=metadata))
        return records

    def _rebuild_indexes(self) -> None:
        self.tag_index = defaultdict(set)
        self.category_index = defaultdict(set)
        self.priority_index = defaultdict(set)
        for entry_id, record in self.store.items():
            self._index_entry(entry_id, record)

    def _normalize_checkpoint_name(self, name: Optional[str]) -> str:
        if name is None:
            return f"eval_memory_{_utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        normalized = self._normalize_non_empty_string(name, "checkpoint_name")
        if not normalized.endswith(".json"):
            normalized = f"{normalized}.json"
        return normalized

    def _require_positive_integer(self, value: Any, field_name: str) -> int:
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigLoadError(self.config_path, field_name, "must be a positive integer") from exc
        if integer <= 0:
            raise ConfigLoadError(self.config_path, field_name, "must be a positive integer")
        return integer

    def _require_non_negative_integer(self, value: Any, field_name: str) -> int:
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigLoadError(self.config_path, field_name, "must be a non-negative integer") from exc
        if integer < 0:
            raise ConfigLoadError(self.config_path, field_name, "must be a non-negative integer")
        return integer


# ----------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailureError("timestamp", value, "ISO-8601 string")

    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationFailureError("timestamp", value, "valid ISO-8601 string") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


if __name__ == "__main__":
    print("\n=== Running Evaluators Memory ===\n")
    memory = EvaluatorsMemory()

    record_id = memory.add(
        entry={"metric": "accuracy", "value": 0.92, "model": "v3"},
        tags=["final_evaluation", "model_v3"],
        priority="high",
        source="performance_evaluator",
        category="evaluation",
        metadata={"split": "test"},
    )
    print("Stored entry:", record_id)
    print("Statistics:", memory.get_statistics())

    checkpoint = memory.create_checkpoint("important_eval")
    print("Checkpoint:", checkpoint)

    search = memory.search_entries(search_term="accuracy", include_metadata=True)
    print("Search results:", len(search))

    print("\n=== Successfully Ran Evaluators Memory ===\n")
