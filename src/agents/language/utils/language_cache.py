"""
Production-grade cache for the language subsystem.

Core Function:
Stores and retrieves expensive language artifacts such as embeddings, dialogue
summaries, parsed linguistic payloads, NLU/NLG intermediate results, and small
language-resource lookups with TTLs, metadata, eviction, persistence, and
structured diagnostics.

Responsibilities:
- Cache embeddings and language summaries with stable text keys.
- Provide generic namespaced cache entries for future language modules.
- Track cache hits, misses, evictions, expirations, saves, loads, and failures.
- Support LRU, LFU, and FIFO eviction strategies without duplicating cache logic.
- Persist cache state atomically using the configured language_config.yaml values.
- Reuse language_helpers.py for normalization, hashing, JSON safety, IDs, paths,
  text validation, coercion, serialization helpers, and logging-safe payloads.
- Reuse language_error.py for structured cache/config/resource diagnostics.

Why it matters:
Language modules repeatedly compute embeddings, summaries, parsed frames,
normalization results, and lexical lookups. A stable cache keeps the language
agent fast and consistent across turns while preventing cache concerns from
leaking into orthography, NLP, NLU, dialogue context, and NLG modules.
"""

from __future__ import annotations

import gzip
import os
import pickle
import tempfile
import time as time_module
import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union

from .config_loader import load_global_config, get_config_section
from .language_error import *
from .language_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Language Cache")
printer = PrettyPrinter()

CacheKey = str
CacheKind = str
CacheNamespace = str
MetadataFilter = Callable[[Dict[str, Any]], bool]
ValueFactory = Callable[[], Any]


@dataclass(frozen=True)
class CacheLookup:
    """Result metadata for a cache lookup."""

    key: CacheKey
    kind: CacheKind
    namespace: CacheNamespace
    hit: bool
    expired: bool = False
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self), drop_empty=True)


@dataclass(frozen=True)
class SimilarityMatch:
    """Embedding similarity result."""

    key: CacheKey
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> Tuple[str, float]:
        return self.key, self.score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "score": self.score,
            "metadata": json_safe(self.metadata),
        }


@dataclass
class CacheEntry:
    """Serializable cache entry with TTL, metadata, and access counters."""

    key: CacheKey
    kind: CacheKind
    namespace: CacheNamespace
    value: Any
    created_at: float
    updated_at: float
    accessed_at: float
    ttl_seconds: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    hits: int = 0
    size_bytes: int = 0

    @property
    def age_seconds(self) -> float:
        return max(0.0, time_module.time() - float(self.created_at))

    @property
    def idle_seconds(self) -> float:
        return max(0.0, time_module.time() - float(self.accessed_at))

    @property
    def expires_at(self) -> Optional[float]:
        if self.ttl_seconds is None or self.ttl_seconds <= 0:
            return None
        return float(self.updated_at) + float(self.ttl_seconds)

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_seconds is None or self.ttl_seconds <= 0:
            return False
        current = time_module.time() if now is None else float(now)
        return current - float(self.updated_at) > float(self.ttl_seconds)

    def touch(self, *, timestamp: Optional[float] = None) -> None:
        self.accessed_at = time_module.time() if timestamp is None else float(timestamp)
        self.hits += 1

    def refresh(self, *, timestamp: Optional[float] = None) -> None:
        now = time_module.time() if timestamp is None else float(timestamp)
        self.updated_at = now
        self.accessed_at = now

    def to_metadata(self, *, include_key: bool = True, include_value_preview: bool = False) -> Dict[str, Any]:
        payload = {
            "kind": self.kind,
            "namespace": self.namespace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
            "age_seconds": round(self.age_seconds, 6),
            "idle_seconds": round(self.idle_seconds, 6),
            "hits": self.hits,
            "size_bytes": self.size_bytes,
            "metadata": json_safe(self.metadata),
        }
        if include_key:
            payload["key"] = self.key
        if include_value_preview:
            payload["value_preview"] = compact_text(self.value, max_length=256)
        return prune_none(payload, drop_empty=True)


@dataclass
class LanguageCacheStats:
    """Operational counters for cache observability."""

    started_at: float = field(default_factory=time_module.time)
    writes: int = 0
    hits: int = 0
    misses: int = 0
    expirations: int = 0
    evictions: int = 0
    deletes: int = 0
    saves: int = 0
    loads: int = 0
    load_failures: int = 0
    save_failures: int = 0
    similarity_queries: int = 0
    bytes_estimated: int = 0

    @property
    def uptime_seconds(self) -> float:
        return max(0.0, time_module.time() - self.started_at)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["uptime_seconds"] = round(self.uptime_seconds, 6)
        payload["hit_rate"] = round(self.hit_rate, 6)
        return payload


@dataclass(frozen=True)
class LanguageCacheConfig:
    """Runtime configuration loaded from language_config.yaml."""

    version: str = "2.0"
    max_size: int = 1000
    summary_max_size: int = 500
    object_max_size: int = 500
    expiry_seconds: Optional[float] = 3600.0
    embedding_ttl_seconds: Optional[float] = None
    summary_ttl_seconds: Optional[float] = None
    object_ttl_seconds: Optional[float] = None
    cache_path: Optional[Path] = None
    metadata_export_path: Optional[Path] = None
    strategy_name: str = "LRU"
    hash_algorithm: str = "sha256"
    hash_length: int = 64
    key_namespace: str = "language"
    normalize_text_keys: bool = True
    store_text_preview: bool = True
    text_preview_length: int = 160
    autosave: bool = True
    load_on_init: bool = True
    save_interval_seconds: float = 60.0
    prune_expired_on_save: bool = True
    enable_compression: bool = True
    enable_encryption: bool = False
    serialization_protocol: int = pickle.HIGHEST_PROTOCOL
    persist_embeddings: bool = True
    persist_summaries: bool = True
    persist_objects: bool = True
    clone_on_read: bool = True
    detach_tensors: bool = True
    tensor_storage_device: str = "cpu"
    similarity_min_score: float = 0.6
    similarity_top_k: int = 1
    strict_persistence: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "LanguageCacheConfig":
        section = dict(config or {})
        expiry = _optional_seconds(section.get("expiry_seconds"), default=3600.0)
        cache_path_value = first_non_none(section.get("cache_path"), section.get("path"))
        metadata_export_value = section.get("metadata_export_path")
        strategy_name = first_non_none(section.get("strategy_name"), section.get("strategy"), default="LRU")
        return cls(
            version=str(section.get("version", "2.0")),
            max_size=coerce_int(section.get("max_size"), default=1000, minimum=1),
            summary_max_size=coerce_int(section.get("summary_max_size"), default=500, minimum=1),
            object_max_size=coerce_int(section.get("object_max_size"), default=500, minimum=1),
            expiry_seconds=expiry,
            embedding_ttl_seconds=_optional_seconds(section.get("embedding_ttl_seconds"), default=None),
            summary_ttl_seconds=_optional_seconds(section.get("summary_ttl_seconds"), default=None),
            object_ttl_seconds=_optional_seconds(section.get("object_ttl_seconds"), default=None),
            cache_path=Path(cache_path_value) if cache_path_value else None,
            metadata_export_path=Path(metadata_export_value) if metadata_export_value else None,
            strategy_name=normalize_identifier_component(strategy_name, default="LRU", lowercase=False).upper(),
            hash_algorithm=str(section.get("hash_algorithm", "sha256") or "sha256").lower(),
            hash_length=coerce_int(section.get("hash_length"), default=64, minimum=8, maximum=128),
            key_namespace=normalize_identifier_component(section.get("key_namespace", "language"), default="language"),
            normalize_text_keys=coerce_bool(section.get("normalize_text_keys"), default=True),
            store_text_preview=coerce_bool(section.get("store_text_preview"), default=True),
            text_preview_length=coerce_int(section.get("text_preview_length"), default=160, minimum=16, maximum=2000),
            autosave=coerce_bool(section.get("autosave"), default=True),
            load_on_init=coerce_bool(section.get("load_on_init"), default=True),
            save_interval_seconds=coerce_float(section.get("save_interval_seconds"), default=60.0, minimum=0.0),
            prune_expired_on_save=coerce_bool(section.get("prune_expired_on_save"), default=True),
            enable_compression=coerce_bool(section.get("enable_compression"), default=True),
            enable_encryption=coerce_bool(section.get("enable_encryption"), default=False),
            serialization_protocol=coerce_int(section.get("serialization_protocol"), default=pickle.HIGHEST_PROTOCOL, minimum=0, maximum=pickle.HIGHEST_PROTOCOL),
            persist_embeddings=coerce_bool(section.get("persist_embeddings"), default=True),
            persist_summaries=coerce_bool(section.get("persist_summaries"), default=True),
            persist_objects=coerce_bool(section.get("persist_objects"), default=True),
            clone_on_read=coerce_bool(section.get("clone_on_read"), default=True),
            detach_tensors=coerce_bool(section.get("detach_tensors"), default=True),
            tensor_storage_device=str(section.get("tensor_storage_device", "cpu") or "cpu"),
            similarity_min_score=coerce_probability(section.get("similarity_min_score"), default=0.6),
            similarity_top_k=coerce_int(section.get("similarity_top_k"), default=1, minimum=1),
            strict_persistence=coerce_bool(section.get("strict_persistence"), default=False),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["cache_path"] = str(self.cache_path) if self.cache_path else None
        payload["metadata_export_path"] = str(self.metadata_export_path) if self.metadata_export_path else None
        return prune_none(payload, drop_empty=True)


class BaseCacheStrategy(ABC):
    """Abstract eviction strategy used by all cache stores."""

    name = "BASE"

    def __init__(self, max_size: int):
        self.max_size = max(1, int(max_size))

    @abstractmethod
    def record_write(self, key: CacheKey, entry: CacheEntry) -> None:
        """Record that a key was written."""

    @abstractmethod
    def record_access(self, key: CacheKey, entry: CacheEntry) -> None:
        """Record that a key was accessed."""

    @abstractmethod
    def discard(self, key: CacheKey) -> None:
        """Remove strategy state for a key."""

    @abstractmethod
    def choose_eviction_key(self, entries: Mapping[CacheKey, CacheEntry]) -> Optional[CacheKey]:
        """Choose the next eviction key from current entries."""

    def add_item(self, key: CacheKey, value: Any) -> None:
        """Backward-compatible alias for previous strategy API."""
        if isinstance(value, CacheEntry):
            self.record_write(key, value)

    def get_next_eviction_key(self) -> str:
        """Backward-compatible API; real eviction uses choose_eviction_key."""
        return ""


class LRUCacheStrategy(BaseCacheStrategy):
    """Least Recently Used eviction."""

    name = "LRU"

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.access_order: "OrderedDict[str, None]" = OrderedDict()

    def record_write(self, key: CacheKey, entry: CacheEntry) -> None:
        self.record_access(key, entry)

    def record_access(self, key: CacheKey, entry: CacheEntry) -> None:
        self.access_order.pop(key, None)
        self.access_order[key] = None

    def discard(self, key: CacheKey) -> None:
        self.access_order.pop(key, None)

    def choose_eviction_key(self, entries: Mapping[CacheKey, CacheEntry]) -> Optional[CacheKey]:
        for key in self.access_order.keys():
            if key in entries:
                return key
        return next(iter(entries.keys()), None)

    def get_next_eviction_key(self) -> str:
        return next(iter(self.access_order.keys()), "")


class LFUCacheStrategy(BaseCacheStrategy):
    """Least Frequently Used eviction with age tie-breaker."""

    name = "LFU"

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.access_count: Dict[str, int] = defaultdict(int)

    def record_write(self, key: CacheKey, entry: CacheEntry) -> None:
        self.access_count.setdefault(key, 0)

    def record_access(self, key: CacheKey, entry: CacheEntry) -> None:
        self.access_count[key] += 1

    def discard(self, key: CacheKey) -> None:
        self.access_count.pop(key, None)

    def choose_eviction_key(self, entries: Mapping[CacheKey, CacheEntry]) -> Optional[CacheKey]:
        if not entries:
            return None
        return min(entries, key=lambda key: (self.access_count.get(key, 0), entries[key].accessed_at, entries[key].created_at))

    def get_next_eviction_key(self) -> str:
        if not self.access_count:
            return ""
        return min(self.access_count, key=lambda key: self.access_count[key])


class FIFOCacheStrategy(BaseCacheStrategy):
    """First In, First Out eviction."""

    name = "FIFO"

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.insert_order: deque[str] = deque()

    def record_write(self, key: CacheKey, entry: CacheEntry) -> None:
        if key in self.insert_order:
            self.insert_order.remove(key)
        self.insert_order.append(key)

    def record_access(self, key: CacheKey, entry: CacheEntry) -> None:
        return None

    def discard(self, key: CacheKey) -> None:
        if key in self.insert_order:
            self.insert_order.remove(key)

    def choose_eviction_key(self, entries: Mapping[CacheKey, CacheEntry]) -> Optional[CacheKey]:
        while self.insert_order:
            key = self.insert_order[0]
            if key in entries:
                return key
            self.insert_order.popleft()
        return next(iter(entries.keys()), None)

    def get_next_eviction_key(self) -> str:
        return self.insert_order[0] if self.insert_order else ""


class LanguageCache:
    """Shared cache for language embeddings, summaries, and future artifacts."""

    VERSION = "2.0"
    SERIALIZATION_PROTOCOL = pickle.HIGHEST_PROTOCOL
    EMBEDDING_KIND = "embedding"
    SUMMARY_KIND = "summary"
    OBJECT_KIND = "object"

    def __init__(self):
        self.config = load_global_config()
        self.cache_config = get_config_section("language_cache")
        self.settings = LanguageCacheConfig.from_mapping(self.cache_config)
        self.VERSION = self.settings.version
        self.SERIALIZATION_PROTOCOL = self.settings.serialization_protocol

        self.max_size = self.settings.max_size
        self.expiry_seconds = self.settings.expiry_seconds
        self.cache_path = self.settings.cache_path
        self.strategy_name = self.settings.strategy_name
        self.enable_compression = self.settings.enable_compression
        self.enable_encryption = self.settings.enable_encryption

        self.embedding_cache: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self.summary_cache: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self.object_cache: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self.stats = LanguageCacheStats()
        self.diagnostics = LanguageDiagnostics()
        self._last_save = 0.0

        self._validate_settings()
        self._init_caches()
        self._prepare_storage()

        if self.settings.load_on_init and self.cache_path is not None and self.cache_path.exists():
            self._load_from_disk()

        logger.info("LanguageCache initialized with strategy=%s, max_size=%s, cache_path=%s", self.strategy_name, self.max_size, self.cache_path)

    def _validate_settings(self) -> None:
        if self.enable_encryption:
            raise ConfigurationLanguageError(
                ConfigurationIssue(
                    code=LanguageErrorCode.CONFIG_VALUE_INVALID,
                    message="language_cache.enable_encryption is enabled, but encryption is not configured in this cache implementation.",
                    module="LanguageCache",
                    details={"setting": "language_cache.enable_encryption", "value": True},
                ),
                recoverable=False,
            )
        if self.settings.tensor_storage_device != "cpu" and not torch.cuda.is_available():
            raise ConfigurationLanguageError(
                ConfigurationIssue(
                    code=LanguageErrorCode.CONFIG_VALUE_INVALID,
                    message="Configured tensor_storage_device requires CUDA, but CUDA is unavailable.",
                    module="LanguageCache",
                    details={"tensor_storage_device": self.settings.tensor_storage_device},
                ),
                recoverable=False,
            )

    def _init_caches(self) -> None:
        self.embedding_strategy = self._make_strategy(self.settings.max_size)
        self.summary_strategy = self._make_strategy(self.settings.summary_max_size)
        self.object_strategy = self._make_strategy(self.settings.object_max_size)

    def _make_strategy(self, max_size: int) -> BaseCacheStrategy:
        strategy_map = {
            "LRU": LRUCacheStrategy,
            "LFU": LFUCacheStrategy,
            "FIFO": FIFOCacheStrategy,
        }
        strategy_class = strategy_map.get(self.strategy_name.upper())
        if strategy_class is None:
            raise ConfigurationLanguageError(
                ConfigurationIssue(
                    code=LanguageErrorCode.CONFIG_VALUE_INVALID,
                    message="Unsupported language cache strategy.",
                    module="LanguageCache",
                    details={"strategy_name": self.strategy_name, "supported": sorted(strategy_map)},
                ),
                recoverable=False,
            )
        return strategy_class(max_size=max_size)

    def _prepare_storage(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.metadata_export_path is not None:
            self.settings.metadata_export_path.parent.mkdir(parents=True, exist_ok=True)

    def _normalize_text_key(self, text: str) -> str:
        raw = require_non_empty_string(text, "text")
        if self.settings.normalize_text_keys:
            return normalize_for_comparison(raw)
        return raw

    def _hash_key(self, text: str, algorithm: Optional[str] = None) -> str:
        """Generate a stable cache key for text while preserving public compatibility."""
        normalized = self._normalize_text_key(text)
        active_algorithm = (algorithm or self.settings.hash_algorithm).lower()
        digest = stable_hash(normalized, algorithm=active_algorithm, length=self.settings.hash_length)
        return f"{active_algorithm}:{digest}"

    def make_key(self, value: Any, *, kind: CacheKind = OBJECT_KIND, namespace: Optional[str] = None) -> CacheKey:
        active_namespace = normalize_identifier_component(namespace or self.settings.key_namespace, default="language")
        active_kind = normalize_identifier_component(kind, default=self.OBJECT_KIND)
        if isinstance(value, str):
            digest = self._hash_key(value).split(":", 1)[1]
            algorithm = self.settings.hash_algorithm
        else:
            algorithm = self.settings.hash_algorithm
            digest = stable_hash(value, algorithm=algorithm, length=self.settings.hash_length)
        return f"{active_namespace}:{active_kind}:{algorithm}:{digest}"

    def _embedding_key(self, text: str) -> CacheKey:
        return self.make_key(text, kind=self.EMBEDDING_KIND, namespace="embedding")

    def _summary_key(self, session_id: str) -> CacheKey:
        return self.make_key(require_non_empty_string(session_id, "session_id"), kind=self.SUMMARY_KIND, namespace="summary")

    def _object_key(self, namespace: str, key: Any) -> CacheKey:
        return self.make_key(key, kind=self.OBJECT_KIND, namespace=namespace)

    def _effective_ttl(self, kind: CacheKind, custom_ttl: Optional[Union[int, float]]) -> Optional[float]:
        if custom_ttl is not None:
            return _optional_seconds(custom_ttl, default=None)
        if kind == self.EMBEDDING_KIND:
            return first_non_none(self.settings.embedding_ttl_seconds, self.settings.expiry_seconds)
        if kind == self.SUMMARY_KIND:
            return first_non_none(self.settings.summary_ttl_seconds, self.settings.expiry_seconds)
        return first_non_none(self.settings.object_ttl_seconds, self.settings.expiry_seconds)

    def _metadata_for_text(self, text: str, metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        payload = dict(json_safe(metadata or {})) if isinstance(metadata, Mapping) else {}
        if self.settings.store_text_preview:
            payload.setdefault("text_preview", truncate_text(text, self.settings.text_preview_length))
            payload.setdefault("text_fingerprint", fingerprint_text(text))
        return payload

    def _prepare_tensor(self, embedding: torch.Tensor) -> torch.Tensor:
        if not isinstance(embedding, torch.Tensor):
            raise CacheLanguageError(
                CacheIssue(
                    code=LanguageErrorCode.CACHE_TYPE_MISMATCH,
                    message="Embedding cache values must be torch.Tensor instances.",
                    severity=Severity.ERROR,
                    module="LanguageCache",
                    details={"received_type": type(embedding).__name__},
                ),
                recoverable=True,
            )
        tensor = embedding
        if self.settings.detach_tensors:
            tensor = tensor.detach()
        tensor = tensor.to(self.settings.tensor_storage_device)
        return tensor.contiguous()

    def _clone_tensor_for_read(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.settings.clone_on_read:
            return tensor.clone()
        return tensor

    def _estimate_entry_size(self, value: Any, metadata: Optional[Mapping[str, Any]] = None) -> int:
        if isinstance(value, torch.Tensor):
            base = int(value.numel() * value.element_size())
        elif isinstance(value, str):
            base = len(value.encode("utf-8", errors="replace"))
        else:
            base = len(stable_json_dumps(json_safe(value)).encode("utf-8", errors="replace"))
        meta_size = len(stable_json_dumps(json_safe(metadata or {})).encode("utf-8", errors="replace"))
        return base + meta_size

    def _store_entry(
        self,
        store: MutableMapping[CacheKey, CacheEntry],
        strategy: BaseCacheStrategy,
        max_size: int,
        *,
        key: CacheKey,
        kind: CacheKind,
        namespace: CacheNamespace,
        value: Any,
        ttl_seconds: Optional[float],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CacheEntry:
        now = time_module.time()
        safe_metadata = dict(json_safe(metadata or {})) if isinstance(metadata, Mapping) else {}
        entry = CacheEntry(
            key=key,
            kind=kind,
            namespace=namespace,
            value=value,
            created_at=now,
            updated_at=now,
            accessed_at=now,
            ttl_seconds=ttl_seconds,
            metadata=safe_metadata,
            size_bytes=self._estimate_entry_size(value, safe_metadata),
        )
        store[key] = entry
        if isinstance(store, OrderedDict):
            store.move_to_end(key)
        strategy.record_write(key, entry)
        self.stats.writes += 1
        self._evict_if_needed(store, strategy, max_size)
        self._autosave_if_needed()
        return entry

    def _get_entry(
        self,
        store: MutableMapping[CacheKey, CacheEntry],
        strategy: BaseCacheStrategy,
        *,
        key: CacheKey,
        update_access: bool = True,
    ) -> Optional[CacheEntry]:
        entry = store.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        if entry.is_expired():
            self._remove_entry(store, strategy, key=key)
            self.stats.expirations += 1
            self.stats.misses += 1
            return None
        if update_access:
            entry.touch()
            if isinstance(store, OrderedDict):
                store.move_to_end(key)
            strategy.record_access(key, entry)
        self.stats.hits += 1
        return entry

    def _remove_entry(self, store: MutableMapping[CacheKey, CacheEntry], strategy: BaseCacheStrategy, *, key: CacheKey) -> bool:
        existed = key in store
        if existed:
            store.pop(key, None)
            strategy.discard(key)
            self.stats.deletes += 1
        return existed

    def _evict_if_needed(self, store: MutableMapping[CacheKey, CacheEntry], strategy: BaseCacheStrategy, max_size: int) -> None:
        while len(store) > max_size:
            evict_key = strategy.choose_eviction_key(store)
            if evict_key is None:
                return
            store.pop(evict_key, None)
            strategy.discard(evict_key)
            self.stats.evictions += 1

    def _is_expired(self, timestamp: float) -> bool:
        """Backward-compatible timestamp expiry helper."""
        if self.expiry_seconds is None or self.expiry_seconds <= 0:
            return False
        return time_module.time() - float(timestamp) > float(self.expiry_seconds)

    def add_embedding(
        self,
        text: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        custom_ttl: Optional[int] = None,
    ) -> CacheKey:
        """Add or replace an embedding and return its stable cache key."""
        text_value = require_non_empty_string(text, "text")
        tensor = self._prepare_tensor(embedding)
        key = self._embedding_key(text_value)
        ttl = self._effective_ttl(self.EMBEDDING_KIND, custom_ttl)
        entry_metadata = self._metadata_for_text(text_value, metadata)
        self._store_entry(
            self.embedding_cache,
            self.embedding_strategy,
            self.settings.max_size,
            key=key,
            kind=self.EMBEDDING_KIND,
            namespace="embedding",
            value=tensor,
            ttl_seconds=ttl,
            metadata=entry_metadata,
        )
        return key

    def get_embedding(self, text: str, update_access: bool = True) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Retrieve an embedding and metadata by source text."""
        key = self._embedding_key(text)
        entry = self._get_entry(self.embedding_cache, self.embedding_strategy, key=key, update_access=update_access)
        if entry is None:
            return None
        return self._clone_tensor_for_read(entry.value), dict(entry.metadata)

    def get_embedding_by_key(self, key: CacheKey, update_access: bool = True) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Retrieve an embedding directly by cache key."""
        entry = self._get_entry(self.embedding_cache, self.embedding_strategy, key=key, update_access=update_access)
        if entry is None:
            return None
        return self._clone_tensor_for_read(entry.value), dict(entry.metadata)

    def has_embedding(self, text: str) -> bool:
        return self.get_embedding(text, update_access=False) is not None

    def most_similar(
        self,
        embedding: torch.Tensor,
        top_k: int = 1,
        min_similarity: float = 0.6,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[str, float]]:
        """Find nearest cached embeddings and return backward-compatible (key, score) tuples."""
        matches = self.find_similar_embeddings(
            embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            filter_func=filter_func,
        )
        return [match.to_tuple() for match in matches]

    def find_similar_embeddings(
        self,
        embedding: torch.Tensor,
        *,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        filter_func: Optional[MetadataFilter] = None,
        update_access: bool = True,
    ) -> List[SimilarityMatch]:
        """Find nearest cached embeddings with metadata-aware filtering."""
        if not isinstance(embedding, torch.Tensor):
            raise CacheLanguageError(
                CacheIssue(
                    code=LanguageErrorCode.CACHE_TYPE_MISMATCH,
                    message="Similarity query embedding must be a torch.Tensor.",
                    severity=Severity.ERROR,
                    module="LanguageCache",
                    details={"received_type": type(embedding).__name__},
                ),
                recoverable=True,
            )

        self.stats.similarity_queries += 1
        now = time_module.time()
        query = self._prepare_tensor(embedding).float().flatten()
        candidates: List[Tuple[str, CacheEntry]] = []
        expired_keys: List[str] = []

        for key, entry in list(self.embedding_cache.items()):
            if entry.is_expired(now):
                expired_keys.append(key)
                continue
            if filter_func is not None and not filter_func(dict(entry.metadata)):
                continue
            tensor = entry.value.float().flatten()
            if tensor.numel() != query.numel():
                continue
            candidates.append((key, entry))

        for key in expired_keys:
            self._remove_entry(self.embedding_cache, self.embedding_strategy, key=key)
            self.stats.expirations += 1

        if not candidates:
            return []

        matrix = torch.stack([entry.value.float().flatten() for _, entry in candidates])
        scores = F.cosine_similarity(query.unsqueeze(0), matrix, dim=1)
        threshold = self.settings.similarity_min_score if min_similarity is None else coerce_probability(min_similarity, default=self.settings.similarity_min_score)
        limit = self.settings.similarity_top_k if top_k is None else coerce_int(top_k, default=self.settings.similarity_top_k, minimum=1)

        matches: List[SimilarityMatch] = []
        for index, score_value in enumerate(scores.tolist()):
            score = float(score_value)
            if score < threshold:
                continue
            key, entry = candidates[index]
            if update_access:
                entry.touch()
                self.embedding_strategy.record_access(key, entry)
                self.embedding_cache.move_to_end(key)
            matches.append(SimilarityMatch(key=key, score=score, metadata=dict(entry.metadata)))

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def set_summary(
        self,
        session_id: str,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
        custom_ttl: Optional[int] = None,
    ) -> CacheKey:
        """Store a dialogue/session summary and return its stable key."""
        session = require_non_empty_string(session_id, "session_id")
        summary_text = validate_response_text(summary, min_length=1)
        key = self._summary_key(session)
        ttl = self._effective_ttl(self.SUMMARY_KIND, custom_ttl)
        entry_metadata = dict(json_safe(metadata or {}))
        entry_metadata.setdefault("session_id", session)
        entry_metadata.setdefault("summary_fingerprint", fingerprint_text(summary_text))
        self._store_entry(
            self.summary_cache,
            self.summary_strategy,
            self.settings.summary_max_size,
            key=key,
            kind=self.SUMMARY_KIND,
            namespace="summary",
            value=summary_text,
            ttl_seconds=ttl,
            metadata=entry_metadata,
        )
        return key

    def get_summary(self, session_id: str, update_access: bool = True) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Retrieve a dialogue/session summary and metadata."""
        key = self._summary_key(session_id)
        entry = self._get_entry(self.summary_cache, self.summary_strategy, key=key, update_access=update_access)
        if entry is None:
            return None
        return str(entry.value), dict(entry.metadata)

    def get_summary_by_key(self, key: CacheKey, update_access: bool = True) -> Optional[Tuple[str, Dict[str, Any]]]:
        entry = self._get_entry(self.summary_cache, self.summary_strategy, key=key, update_access=update_access)
        if entry is None:
            return None
        return str(entry.value), dict(entry.metadata)

    def set_value(
        self,
        namespace: str,
        key: Any,
        value: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        custom_ttl: Optional[Union[int, float]] = None,
    ) -> CacheKey:
        """Store a generic language artifact in a namespaced object cache."""
        active_namespace = normalize_identifier_component(namespace, default="object")
        cache_key = self._object_key(active_namespace, key)
        ttl = self._effective_ttl(self.OBJECT_KIND, custom_ttl)
        self._store_entry(
            self.object_cache,
            self.object_strategy,
            self.settings.object_max_size,
            key=cache_key,
            kind=self.OBJECT_KIND,
            namespace=active_namespace,
            value=json_safe(value),
            ttl_seconds=ttl,
            metadata=metadata,
        )
        return cache_key

    def get_value(self, namespace: str, key: Any, *, default: Any = None, update_access: bool = True) -> Any:
        active_namespace = normalize_identifier_component(namespace, default="object")
        cache_key = self._object_key(active_namespace, key)
        entry = self._get_entry(self.object_cache, self.object_strategy, key=cache_key, update_access=update_access)
        return default if entry is None else entry.value

    def get_or_set(
        self,
        namespace: str,
        key: Any,
        factory: ValueFactory,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        custom_ttl: Optional[Union[int, float]] = None,
    ) -> Any:
        value = self.get_value(namespace, key, default=None)
        if value is not None:
            return value
        computed = factory()
        self.set_value(namespace, key, computed, metadata=metadata, custom_ttl=custom_ttl)
        return computed

    def delete_value(self, namespace: str, key: Any) -> bool:
        active_namespace = normalize_identifier_component(namespace, default="object")
        cache_key = self._object_key(active_namespace, key)
        return self._remove_entry(self.object_cache, self.object_strategy, key=cache_key)

    def lookup(self, kind: CacheKind, key: CacheKey) -> CacheLookup:
        store, strategy = self._store_for_kind(kind)
        entry = store.get(key)
        if entry is None:
            return CacheLookup(key=key, kind=kind, namespace="", hit=False, reason="missing")
        if entry.is_expired():
            self._remove_entry(store, strategy, key=key)
            self.stats.expirations += 1
            return CacheLookup(key=key, kind=kind, namespace=entry.namespace, hit=False, expired=True, reason="expired", metadata=dict(entry.metadata))
        return CacheLookup(key=key, kind=kind, namespace=entry.namespace, hit=True, metadata=dict(entry.metadata))

    def _store_for_kind(self, kind: CacheKind) -> Tuple["OrderedDict[str, CacheEntry]", BaseCacheStrategy]:
        normalized = normalize_identifier_component(kind, default=self.OBJECT_KIND)
        if normalized == self.EMBEDDING_KIND:
            return self.embedding_cache, self.embedding_strategy
        if normalized == self.SUMMARY_KIND:
            return self.summary_cache, self.summary_strategy
        return self.object_cache, self.object_strategy

    def clear(self, clear_disk: bool = False) -> None:
        """Clear all in-memory cache stores and optionally remove persisted state."""
        self.embedding_cache.clear()
        self.summary_cache.clear()
        self.object_cache.clear()
        self._init_caches()
        if clear_disk and self.cache_path is not None and self.cache_path.exists():
            self.cache_path.unlink()
            logger.info("Removed disk cache at %s", self.cache_path)

    def clean_expired(self) -> Tuple[int, int]:
        """Remove expired embeddings and summaries. Returns legacy two-count tuple."""
        counts = self.clean_expired_all()
        return counts.get(self.EMBEDDING_KIND, 0), counts.get(self.SUMMARY_KIND, 0)

    def clean_expired_all(self) -> Dict[str, int]:
        now = time_module.time()
        counts = {
            self.EMBEDDING_KIND: self._clean_expired_store(self.embedding_cache, self.embedding_strategy, now=now),
            self.SUMMARY_KIND: self._clean_expired_store(self.summary_cache, self.summary_strategy, now=now),
            self.OBJECT_KIND: self._clean_expired_store(self.object_cache, self.object_strategy, now=now),
        }
        total = sum(counts.values())
        self.stats.expirations += total
        logger.info("Cleaned expired cache entries: %s", counts)
        return counts

    def _clean_expired_store(self, store: MutableMapping[str, CacheEntry], strategy: BaseCacheStrategy, *, now: float) -> int:
        expired_keys = [key for key, entry in store.items() if entry.is_expired(now)]
        for key in expired_keys:
            store.pop(key, None)
            strategy.discard(key)
        return len(expired_keys)

    def compact(self) -> Dict[str, Any]:
        """Prune expired entries and return current cache statistics."""
        removed = self.clean_expired_all()
        return {"removed": removed, "stats": self.stats_snapshot()}

    def stats_snapshot(self) -> Dict[str, Any]:
        embedding_bytes = sum(entry.size_bytes for entry in self.embedding_cache.values())
        summary_bytes = sum(entry.size_bytes for entry in self.summary_cache.values())
        object_bytes = sum(entry.size_bytes for entry in self.object_cache.values())
        self.stats.bytes_estimated = embedding_bytes + summary_bytes + object_bytes
        return {
            "version": self.VERSION,
            "strategy": self.strategy_name,
            "counts": {
                "embeddings": len(self.embedding_cache),
                "summaries": len(self.summary_cache),
                "objects": len(self.object_cache),
                "total": len(self.embedding_cache) + len(self.summary_cache) + len(self.object_cache),
            },
            "capacity": {
                "embeddings": self.settings.max_size,
                "summaries": self.settings.summary_max_size,
                "objects": self.settings.object_max_size,
            },
            "bytes_estimated": {
                "embeddings": embedding_bytes,
                "summaries": summary_bytes,
                "objects": object_bytes,
                "total": self.stats.bytes_estimated,
            },
            "stats": self.stats.to_dict(),
            "config": self.settings.to_dict(),
        }

    def export_metadata(self, path: Union[str, Path]) -> bool:
        """Export cache metadata for inspection without dumping cached values."""
        target = resolve_path(path, field_name="metadata_export_path")
        payload = {
            "exported_at": utc_timestamp(),
            "cache": self.stats_snapshot(),
            "embeddings": [entry.to_metadata() for entry in self.embedding_cache.values()],
            "summaries": [entry.to_metadata() for entry in self.summary_cache.values()],
            "objects": [entry.to_metadata(include_value_preview=False) for entry in self.object_cache.values()],
        }
        save_json_file(target, payload, pretty=True)
        return True

    def _autosave_if_needed(self) -> None:
        if not self.settings.autosave:
            return
        if self.cache_path is None:
            return
        if self.settings.save_interval_seconds <= 0:
            self.save_to_disk(force=True)
            return
        if time_module.time() - self._last_save >= self.settings.save_interval_seconds:
            self.save_to_disk(force=False)

    def _payload_for_disk(self) -> Dict[str, Any]:
        if self.settings.prune_expired_on_save:
            self.clean_expired_all()
        return {
            "version": self.VERSION,
            "created_at": utc_timestamp(),
            "timestamp": time_module.time(),
            "config": self.settings.to_dict(),
            "strategy": self.strategy_name,
            "stats": self.stats.to_dict(),
            "embedding_cache": self.embedding_cache if self.settings.persist_embeddings else OrderedDict(),
            "summary_cache": self.summary_cache if self.settings.persist_summaries else OrderedDict(),
            "object_cache": self.object_cache if self.settings.persist_objects else OrderedDict(),
        }

    def save_to_disk(self, force: bool = False) -> None:
        """Persist cache state atomically according to language_config.yaml."""
        if self.cache_path is None:
            return
        if not force and self.settings.save_interval_seconds > 0 and time_module.time() - self._last_save < self.settings.save_interval_seconds:
            return

        payload = self._payload_for_disk()
        target = self.cache_path
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile("wb", dir=str(target.parent), delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                if self.enable_compression:
                    with gzip.GzipFile(fileobj=temp_file, mode="wb") as gzip_file:
                        pickle.dump(payload, gzip_file, protocol=self.SERIALIZATION_PROTOCOL)
                else:
                    pickle.dump(payload, temp_file, protocol=self.SERIALIZATION_PROTOCOL)
            os.replace(temp_path, target)
            self._last_save = time_module.time()
            self.stats.saves += 1
            logger.debug("Language cache saved to %s", target)
        except Exception as exc:
            self.stats.save_failures += 1
            issue = CacheIssue(
                code=LanguageErrorCode.CACHE_SERIALIZATION_FAILED,
                message="Failed to save language cache to disk.",
                severity=Severity.ERROR,
                module="LanguageCache",
                details={"cache_path": str(target), "exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
            self.diagnostics.add(issue)
            logger.error("Failed to save language cache to %s: %s", target, exc)
            if self.settings.strict_persistence:
                raise CacheLanguageError(issue, recoverable=False, cause=exc) from exc

    def _load_from_disk(self) -> None:
        """Load persisted cache state and migrate older cache layouts when possible."""
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            payload = self._read_disk_payload(self.cache_path)
            if not isinstance(payload, Mapping):
                raise ValueError("Persisted cache payload is not a mapping.")

            persisted_version = str(payload.get("version", ""))
            if persisted_version and persisted_version.split(".")[0] != self.VERSION.split(".")[0]:
                logger.warning("Language cache major version changed from %s to %s; attempting migration.", persisted_version, self.VERSION)

            self.embedding_cache = self._coerce_store(payload.get("embedding_cache", OrderedDict()), kind=self.EMBEDDING_KIND, namespace="embedding")
            self.summary_cache = self._coerce_store(payload.get("summary_cache", OrderedDict()), kind=self.SUMMARY_KIND, namespace="summary")
            self.object_cache = self._coerce_store(payload.get("object_cache", OrderedDict()), kind=self.OBJECT_KIND, namespace="object")
            self._init_caches()
            self._rebuild_strategies()
            self.clean_expired_all()
            self.stats.loads += 1
            logger.info("Loaded language cache from %s: %s", self.cache_path, self.stats_snapshot()["counts"])
        except Exception as exc:
            self.stats.load_failures += 1
            issue = CacheIssue(
                code="CACHE.PERSISTENCE.LOAD_FAILED",
                message="Failed to load language cache from disk.",
                severity=Severity.ERROR,
                module="LanguageCache",
                details={"cache_path": str(self.cache_path), "exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
            self.diagnostics.add(issue)
            logger.error("Failed to load language cache from %s: %s", self.cache_path, exc)
            self.embedding_cache = OrderedDict()
            self.summary_cache = OrderedDict()
            self.object_cache = OrderedDict()
            self._init_caches()
            if self.settings.strict_persistence:
                raise CacheLanguageError(issue, recoverable=False, cause=exc) from exc

    def _read_disk_payload(self, path: Path) -> Any:
        with open(path, "rb") as raw_file:
            prefix = raw_file.read(2)
            raw_file.seek(0)
            if prefix == b"\x1f\x8b" or self.enable_compression:
                with gzip.GzipFile(fileobj=raw_file, mode="rb") as gzip_file:
                    return pickle.load(gzip_file)
            return pickle.load(raw_file)

    def _coerce_store(self, value: Any, *, kind: CacheKind, namespace: CacheNamespace) -> "OrderedDict[str, CacheEntry]":
        store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        if not isinstance(value, Mapping):
            return store
        for key, raw_entry in value.items():
            entry = self._coerce_entry(str(key), raw_entry, kind=kind, namespace=namespace)
            if entry is not None:
                store[entry.key] = entry
        return store

    def _coerce_entry(self, key: str, raw_entry: Any, *, kind: CacheKind, namespace: CacheNamespace) -> Optional[CacheEntry]:
        if isinstance(raw_entry, CacheEntry):
            return raw_entry
        now = time_module.time()
        if kind == self.EMBEDDING_KIND and isinstance(raw_entry, tuple) and len(raw_entry) >= 4:
            tensor, timestamp, metadata, ttl = raw_entry[:4]
            if not isinstance(tensor, torch.Tensor):
                return None
            return CacheEntry(
                key=key,
                kind=kind,
                namespace=namespace,
                value=self._prepare_tensor(tensor),
                created_at=float(timestamp),
                updated_at=float(timestamp),
                accessed_at=float(timestamp),
                ttl_seconds=_optional_seconds(ttl, default=self._effective_ttl(kind, None)),
                metadata=dict(json_safe(metadata or {})),
                size_bytes=self._estimate_entry_size(tensor, metadata or {}),
            )
        if kind == self.SUMMARY_KIND and isinstance(raw_entry, Mapping):
            summary = raw_entry.get("summary")
            timestamp = float(raw_entry.get("timestamp", now))
            metadata = dict(json_safe(raw_entry.get("metadata", {})))
            ttl = _optional_seconds(raw_entry.get("ttl"), default=self._effective_ttl(kind, None))
            return CacheEntry(
                key=key,
                kind=kind,
                namespace=namespace,
                value=str(summary or ""),
                created_at=timestamp,
                updated_at=timestamp,
                accessed_at=timestamp,
                ttl_seconds=ttl,
                metadata=metadata,
                size_bytes=self._estimate_entry_size(str(summary or ""), metadata),
            )
        return None

    def _rebuild_strategies(self) -> None:
        for key, entry in self.embedding_cache.items():
            self.embedding_strategy.record_write(key, entry)
        for key, entry in self.summary_cache.items():
            self.summary_strategy.record_write(key, entry)
        for key, entry in self.object_cache.items():
            self.object_strategy.record_write(key, entry)

    def diagnostics_result(self) -> LanguageResult[Dict[str, Any]]:
        return LanguageResult(data=self.stats_snapshot(), issues=list(self.diagnostics.issues), metadata={"module": "LanguageCache"})

    def __len__(self) -> int:
        return len(self.embedding_cache) + len(self.summary_cache) + len(self.object_cache)

    def __contains__(self, key: str) -> bool:
        return key in self.embedding_cache or key in self.summary_cache or key in self.object_cache

    def __repr__(self) -> str:
        counts = self.stats_snapshot()["counts"]
        return f"LanguageCache(version={self.VERSION!r}, strategy={self.strategy_name!r}, counts={counts})"


def _optional_seconds(value: Any, *, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    number = coerce_float(value, default=-1.0)
    if number <= 0:
        return None
    return number


if __name__ == "__main__":
    print("\n=== Running Language Cache ===\n")
    printer.status("TEST", "Language Cache initialized", "info")

    cache = LanguageCache()
    cache.clear(clear_disk=False)

    text = "They call that love from where I come from."
    embedding = torch.ones(8)
    metadata = {
        "topic": "emotion classification",
        "author": "user",
        "tags": ["nlp", "torch", "session"],
        "language": "en",
        "summary_type": "final",
    }

    embedding_key = cache.add_embedding(text=text, embedding=embedding, metadata=metadata)
    retrieved_embedding = cache.get_embedding(text=text)
    similar = cache.most_similar(embedding=embedding, top_k=3, min_similarity=0.6)

    session_id = generate_language_id("session")
    summary_text = "Explored syntactic transformations and their impact on language model outputs."
    summary_key = cache.set_summary(session_id=session_id, summary=summary_text, metadata=metadata)
    retrieved_summary = cache.get_summary(session_id=session_id)

    object_key = cache.set_value(
        "nlu_parse",
        {"utterance": text, "version": 1},
        {"intent": "describe_emotion", "confidence": 0.91},
        metadata={"source": "test_block"},
    )
    retrieved_object = cache.get_value("nlu_parse", {"utterance": text, "version": 1})

    expired_counts = cache.clean_expired()
    stats = cache.stats_snapshot()

    if retrieved_embedding is None:
        raise AssertionError("Embedding lookup failed.")
    if not torch.equal(retrieved_embedding[0], embedding):
        raise AssertionError("Retrieved embedding does not match cached embedding.")
    if not similar:
        raise AssertionError("Similarity lookup failed.")
    if retrieved_summary is None or retrieved_summary[0] != summary_text:
        raise AssertionError("Summary lookup failed.")
    if retrieved_object.get("intent") != "describe_emotion":
        raise AssertionError("Generic object cache lookup failed.")

    printer.pretty("Embedding key", embedding_key, "success")
    printer.pretty("Summary key", summary_key, "success")
    printer.pretty("Object key", object_key, "success")
    printer.pretty("Retrieved embedding metadata", retrieved_embedding[1], "success")
    printer.pretty("Similarity", similar, "success")
    printer.pretty("Retrieved summary", retrieved_summary, "success")
    printer.pretty("Retrieved object", retrieved_object, "success")
    printer.pretty("Expired counts", expired_counts, "success")
    printer.pretty("Stats", stats, "success")

    if cache.cache_path is not None:
        cache.save_to_disk(force=True)
        printer.status("TEST", f"Cache persisted to {cache.cache_path}", "success")

    print("\n=== Test ran successfully ===\n")
