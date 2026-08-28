"""
Key features of Perception Memory:
1. Efficient Caching System   - in-memory LRU cache with tag/modality indexing,
                                  nested-structure support (tensors, dicts of tensors)
2. Memory Optimization        - item-count AND memory-budget eviction, optional
                                  disk spillover for evicted items, TTL expiry
3. Robust Tagging System       - free-form tags plus first-class modality tags
                                  backed by ``perception_helpers.normalize_modality_name``
4. Performance Features        - gradient checkpointing passthrough, LRU promotion,
                                  O(1) tag lookups
5. Diagnostic Capabilities     - memory_stats, key/tag introspection, checkpoint
                                  listing, human-readable __repr__
6. Safety Mechanisms           - thread-safe operations, finite-value validation,
                                  config validation, structured perception errors
"""

import os
import time
import glob
import json
import torch
import hashlib
import threading
 
from contextlib import contextmanager
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from collections import OrderedDict, defaultdict
from typing import Optional, Tuple, Dict, Any, Union, List, Callable, Mapping, Sequence

from .utils.config_loader import load_global_config, get_config_section
from .utils.perception_errors import *
from .utils.perception_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Perception Memory")
printer = PrettyPrinter()

# A single cached item may be a plain tensor or an arbitrary nested
# structure of tensors (e.g. a per-modality feature dict).
CacheableValue = Union[torch.Tensor, Mapping[str, Any], Sequence[Any]]

class PerceptionMemory(nn.Module):
    """
    Perception Memory prioritizes memory efficiency while maintaining flexibility for different use cases.
    The combination of in-memory caching, gradient checkpointing, optional disk
    spillover, and disk checkpointing provides a comprehensive memory management
    solution suitable for large-scale, multi-modal perception systems.
    """
 
    _COMPONENT = "perception_memory"
 
    def __init__(self, enable_checkpointing: Optional[bool] = None, enable_cache: Optional[bool] = None):
        super().__init__()
        self.config = load_global_config()
        self.memory_config = get_config_section('perception_memory')
 
        # Allow overriding config via arguments
        self.enable_checkpointing = enable_checkpointing if enable_checkpointing is not None else self.memory_config.get('enable_checkpointing', True)
        self.enable_cache = enable_cache if enable_cache is not None else self.memory_config.get('enable_cache', True)
 
        self.checkpoint_dir = self.memory_config.get('checkpoint_dir')
        self.cache_dir = self.memory_config.get('cache_dir')
        self.max_cache_size = self.memory_config.get('max_cache_size', 100)
 
        # --- Memory-optimization configuration (see perception_config.yaml) ---
        self.max_memory_mb = self.memory_config.get('max_memory_mb', 0)
        self.cache_ttl_seconds = self.memory_config.get('cache_ttl_seconds', 0)
        self.eviction_policy = self.memory_config.get('eviction_policy', 'lru')
        self.enable_disk_persistence = self.memory_config.get('enable_disk_persistence', False)
        self.persist_index_filename = self.memory_config.get('persist_index_filename', 'memory_index.json')
        self.checkpoint_retention = self.memory_config.get('checkpoint_retention', 20)
        self.validate_finite = self.memory_config.get('validate_finite', True)
        self.key_algorithm = self.memory_config.get('key_algorithm', 'sha256')
        self.lock_timeout_seconds = self.memory_config.get('lock_timeout_seconds', 5.0)
 
        self._validate_configuration()
 
        if self.enable_checkpointing and self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.enable_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
 
        # Re-entrant lock guards every mutation of cache/tag_index/disk_index so
        # PerceptionMemory is safe to share across trainer/inference threads.
        self._lock = threading.RLock()
 
        # In-memory cache storage
        self.cache = OrderedDict()
        self.tag_index = defaultdict(list)
 
        # Disk-spillover index: key -> {"path", "tags", "metadata", "created_at"}
        self.disk_index: Dict[str, Dict[str, Any]] = {}
        if self.enable_disk_persistence and self.cache_dir:
            self._load_disk_index()
 
        # Memory metrics
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.spill_count = 0
        self.expired_count = 0
        self.total_stored = 0
 
    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------
    def _validate_configuration(self) -> None:
        """Fail fast on an inconsistent memory configuration."""
        ensure_instance(self.enable_cache, bool, "enable_cache", component=self._COMPONENT)
        ensure_instance(self.enable_checkpointing, bool, "enable_checkpointing", component=self._COMPONENT)
 
        if self.enable_cache:
            ensure_instance(self.max_cache_size, int, "max_cache_size", component=self._COMPONENT)
            ensure_in_range(self.max_cache_size, "max_cache_size", minimum=1, component=self._COMPONENT)
 
        ensure_in_range(self.max_memory_mb, "max_memory_mb", minimum=0, component=self._COMPONENT)
        ensure_in_range(self.cache_ttl_seconds, "cache_ttl_seconds", minimum=0, component=self._COMPONENT)
        ensure_one_of(self.eviction_policy, ("lru", "size"), "eviction_policy", component=self._COMPONENT)
        ensure_instance(self.checkpoint_retention, int, "checkpoint_retention", component=self._COMPONENT)
        ensure_in_range(self.checkpoint_retention, "checkpoint_retention", minimum=0, component=self._COMPONENT)
        ensure_in_range(self.lock_timeout_seconds, "lock_timeout_seconds", minimum=0.001, component=self._COMPONENT)
 
        if not hasattr(hashlib, self.key_algorithm):
            raise UnsupportedPerceptionOptionError(
                f"Unsupported key_algorithm '{self.key_algorithm}'.",
                component=self._COMPONENT,
                details={"key_algorithm": self.key_algorithm},
            )
 
        if self.enable_disk_persistence and not self.cache_dir:
            raise MissingPerceptionConfigurationError(
                "enable_disk_persistence requires 'cache_dir' to be configured.",
                component=self._COMPONENT,
            )
        if self.enable_checkpointing and not self.checkpoint_dir:
            raise MissingPerceptionConfigurationError(
                "enable_checkpointing requires 'checkpoint_dir' to be configured.",
                component=self._COMPONENT,
            )
 
    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------
    @contextmanager
    def _guarded_lock(self):
        """Acquire ``self._lock`` with a bounded wait so a stuck caller surfaces
        a diagnosable error instead of hanging the whole perception pipeline."""
        acquired = self._lock.acquire(timeout=self.lock_timeout_seconds)
        if not acquired:
            raise PerceptionStateError(
                "Timed out waiting for the PerceptionMemory lock.",
                component=self._COMPONENT, retryable=True,
                details={"timeout_seconds": self.lock_timeout_seconds},
            )
        try:
            yield
        finally:
            self._lock.release()
 
    # ------------------------------------------------------------------
    # Key / size helpers
    # ------------------------------------------------------------------
    def _generate_key(self, *args) -> str:
        """Generate a deterministic key from input arguments using the configured algorithm."""
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.new(self.key_algorithm, key_str.encode()).hexdigest()
 
    def _structure_key(self, value: CacheableValue) -> str:
        """Generate a deterministic content‑based key for any cacheable value."""
        content_hash = self._hash_value(value)
        return self._generate_key(content_hash)
    
    def _hash_value(self, value: Any) -> str:
        """Recursively hash a value and all tensors it contains."""
        if isinstance(value, torch.Tensor):
            # Ensure data is on CPU and detached, then hash its raw bytes
            data = value.detach().cpu().numpy().tobytes()
            return hashlib.new(self.key_algorithm, data).hexdigest()
        if isinstance(value, Mapping):
            # Sort keys to ensure deterministic order
            items = [f"{k}:{self._hash_value(v)}" for k, v in sorted(value.items(), key=lambda x: str(x[0]))]
            combined = "|".join(items)
            return hashlib.new(self.key_algorithm, combined.encode()).hexdigest()
        if isinstance(value, (list, tuple)):
            combined = "|".join(self._hash_value(item) for item in value)
            return hashlib.new(self.key_algorithm, combined.encode()).hexdigest()
        # Fallback for non‑tensor primitives
        return hashlib.new(self.key_algorithm, str(value).encode()).hexdigest()
 
    def _estimate_size_mb(self, value: Any) -> float:
        """Recursively estimate the resident size (MB) of a cached value."""
        if isinstance(value, torch.Tensor):
            return value.element_size() * value.numel() / (1024 ** 2)
        if isinstance(value, Mapping):
            return sum(self._estimate_size_mb(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._estimate_size_mb(item) for item in value)
        return 0.0
 
    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def cache_item(self,
                 tensor: CacheableValue,
                 key: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None,
                 modality: Optional[str] = None) -> str:
        """Cache a tensor (or nested structure of tensors) with tags and metadata.
 
        Args:
            tensor: A ``torch.Tensor`` or an arbitrary nested dict/list/tuple
                containing tensors (e.g. a multi-modal feature bundle).
            key: Optional explicit cache key; generated deterministically otherwise.
            tags: Free-form tags used for bulk retrieval/clearing.
            metadata: Arbitrary JSON-serializable bookkeeping data.
            modality: If given, validated against the perception modality
                vocabulary (``text``/``vision``/``audio``) and recorded both as
                a ``modality:<name>`` tag and in ``metadata['modality']``.
        """
        if not self.enable_cache:
            return ""
 
        with self._guarded_lock():
            self.purge_expired()
 
            resolved_tags = list(tags or [])
            resolved_metadata = dict(metadata or {})
            if modality is not None:
                normalized_modality = normalize_modality_name(modality)
                resolved_tags.append(f"modality:{normalized_modality}")
                resolved_metadata["modality"] = normalized_modality
 
            if self.validate_finite:
                self._validate_finite_structure(tensor)
 
            if key is None:
                key = self._structure_key(tensor)
 
            # Item-count based LRU eviction (existing contract).
            if len(self.cache) >= self.max_cache_size:
                evicted_key, _ = self.cache.popitem(last=False)
                self._evict_key(evicted_key, spill=self.enable_disk_persistence, reason="capacity")
 
            # Detach (and move to CPU) the whole structure to save device memory.
            cached_value = detach_tree(tensor, cpu=True)
 
            self.cache[key] = {
                'tensor': cached_value,
                'tags': resolved_tags,
                'metadata': resolved_metadata,
                'access_count': 0,
                'created_at': time.time(),
                'last_accessed': time.time(),
            }
 
            for tag in resolved_tags:
                self.tag_index[tag].append(key)
 
            self.total_stored += 1
            self._enforce_memory_budget()
 
            logger.debug(
                f"Cached item {key} tags={resolved_tags} "
                f"size_mb={self._estimate_size_mb(cached_value):.4f}"
            )
            return key
 
    def _validate_finite_structure(self, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            ensure_finite_tensor(value, "tensor", component=self._COMPONENT)
        elif isinstance(value, Mapping):
            for item in value.values():
                self._validate_finite_structure(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._validate_finite_structure(item)
 
    def _remove_from_tag_index(self, key: str):
        """Remove key from tag index."""
        empty_tags = []
        for tag, keys in self.tag_index.items():
            if key in keys:
                keys.remove(key)
            if not keys:
                empty_tags.append(tag)
        for tag in empty_tags:
            del self.tag_index[tag]
 
    # ------------------------------------------------------------------
    # Eviction / expiry / disk spillover
    # ------------------------------------------------------------------
    def _evict_key(self, key: str, *, spill: bool, reason: str) -> None:
        """Remove ``key`` from the in-memory cache, optionally spilling to disk."""
        entry = self.cache.pop(key, None)
        if entry is None:
            return
        self._remove_from_tag_index(key)
        self.eviction_count += 1
 
        if spill and self.cache_dir:
            self._spill_to_disk(key, entry)
        logger.debug(f"Evicted cache item {key} (reason={reason}, spilled={spill})")
 
    def _spill_to_disk(self, key: str, entry: Dict[str, Any]) -> None:
        """Persist an evicted entry to ``cache_dir`` so it can be reloaded later."""
        try:
            path = os.path.join(self.cache_dir, f"{key}.pt")
            torch.save({'tensor': entry['tensor'], 'metadata': entry['metadata']}, path)
            self.disk_index[key] = {
                'path': path,
                'tags': entry['tags'],
                'metadata': entry['metadata'],
                'created_at': entry.get('created_at', time.time()),
            }
            for tag in entry['tags']:
                if key not in self.tag_index[tag]:
                    self.tag_index[tag].append(key)
            self.spill_count += 1
            self._save_disk_index()
        except OSError as exc:
            raise wrap_exception(
                exc, PerceptionStateError, f"Failed to spill cache item '{key}' to disk.",
                component=self._COMPONENT, details={"key": key, "cache_dir": self.cache_dir},
            )
 
    def _promote_from_disk(self, key: str, device: Optional[torch.device]) -> Optional[Any]:
        """Reload a disk-spilled item into the in-memory cache (LRU promotion)."""
        record = self.disk_index.get(key)
        if record is None:
            return None
        try:
            payload = torch.load(record['path'], map_location='cpu')
        except (OSError, RuntimeError) as exc:
            raise wrap_exception(
                exc, PerceptionStateError, f"Failed to load spilled cache item '{key}'.",
                component=self._COMPONENT, details={"key": key, "path": record['path']},
            )
 
        self.cache[key] = {
            'tensor': payload['tensor'],
            'tags': record['tags'],
            'metadata': record['metadata'],
            'access_count': 0,
            'created_at': record.get('created_at', time.time()),
            'last_accessed': time.time(),
        }
        try:
            os.remove(record['path'])
        except OSError:
            logger.debug(f"Could not remove spilled file for promoted key {key}")
        del self.disk_index[key]
        self._save_disk_index()
        value = self.cache[key]['tensor']
        return value.to(device) if device and isinstance(value, torch.Tensor) else value
 
    def _enforce_memory_budget(self) -> None:
        """Evict items (optionally spilling to disk) until under ``max_memory_mb``."""
        if not self.max_memory_mb:
            return
        while self.cache and self._current_memory_mb() > self.max_memory_mb:
            key = self._select_eviction_candidate()
            self._evict_key(key, spill=self.enable_disk_persistence, reason="memory_budget")
 
    def _select_eviction_candidate(self) -> str:
        if self.eviction_policy == "size":
            return max(self.cache.keys(), key=lambda k: self._estimate_size_mb(self.cache[k]['tensor']))
        return next(iter(self.cache))  # oldest / least-recently-used
 
    def _current_memory_mb(self) -> float:
        return sum(self._estimate_size_mb(item['tensor']) for item in self.cache.values())
 
    def purge_expired(self) -> int:
        """Remove cache entries older than ``cache_ttl_seconds``. Returns count removed."""
        if not self.cache_ttl_seconds:
            return 0
        now = time.time()
        expired = [
            key for key, entry in self.cache.items()
            if (now - entry['created_at']) > self.cache_ttl_seconds
        ]
        for key in expired:
            self._evict_key(key, spill=False, reason="ttl_expired")
            self.expired_count += 1
        if expired:
            logger.debug(f"Purged {len(expired)} expired cache item(s)")
        return len(expired)
 
    # ------------------------------------------------------------------
    # Disk index persistence (manifest only; tensors live in .pt files)
    # ------------------------------------------------------------------
    def _index_path(self) -> str:
        return os.path.join(self.cache_dir, self.persist_index_filename)
 
    def _save_disk_index(self) -> None:
        try:
            manifest = {
                key: {'path': rec['path'], 'tags': rec['tags'], 'created_at': rec['created_at']}
                for key, rec in self.disk_index.items()
            }
            with open(self._index_path(), 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle)
        except OSError as exc:
            logger.error(f"Failed to persist disk cache index: {exc}")
 
    def _load_disk_index(self) -> None:
        path = self._index_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                manifest = json.load(handle)
            for key, rec in manifest.items():
                if os.path.exists(rec['path']):
                    self.disk_index[key] = {
                        'path': rec['path'], 'tags': rec.get('tags', []),
                        'metadata': {}, 'created_at': rec.get('created_at', time.time()),
                    }
                    for tag in rec.get('tags', []):
                        self.tag_index[tag].append(key)
            logger.info(f"Restored {len(self.disk_index)} spilled cache entries from {path}")
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"Failed to load disk cache index at {path}: {exc}")
 
    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self,
               key: Optional[str] = None,
               tag: Optional[str] = None,
               device: Optional[torch.device] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Retrieve a cached value by key or tag, promoting disk-spilled items."""
        if not self.enable_cache:
            raise PerceptionStateError(
                "Cache is disabled.", component=self._COMPONENT, remediation="Enable caching before retrieving.",
            )
 
        resolved_device = resolve_torch_device(device) if device is not None else None
 
        with self._guarded_lock():
            self.purge_expired()
 
            if key is not None:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    self.cache[key]['access_count'] += 1
                    self.cache[key]['last_accessed'] = time.time()
                    self.hit_count += 1
                    value = self.cache[key]['tensor']
                    return value.to(resolved_device) if resolved_device and isinstance(value, torch.Tensor) else value
 
                if self.enable_disk_persistence and key in self.disk_index:
                    promoted = self._promote_from_disk(key, resolved_device)
                    if promoted is not None:
                        self.hit_count += 1
                        return promoted
 
                self.miss_count += 1
                raise KeyError(f"Key not found in cache: {key}")
 
            if tag is not None:
                keys = list(self.tag_index.get(tag, []))
                if keys:
                    results = []
                    for item_key in keys:
                        results.append(self.retrieve(key=item_key, device=device))
                    return results
                self.miss_count += 1
                raise KeyError(f"Tag not found in cache: {tag}")
 
            raise InvalidPerceptionValueError(
                "Either 'key' or 'tag' must be provided.", component=self._COMPONENT,
            )
 
    def retrieve_by_modality(self, modality: str, device: Optional[torch.device] = None) -> List[torch.Tensor]:
        """Convenience accessor for items tagged with a given perception modality."""
        normalized = normalize_modality_name(modality)
        return self.retrieve(tag=f"modality:{normalized}", device=device)
 
    def contains(self, key: str) -> bool:
        """Check whether ``key`` is resident in memory or on disk."""
        with self._guarded_lock():
            return key in self.cache or key in self.disk_index
 
    def keys(self, tag: Optional[str] = None) -> List[str]:
        """List cache keys, optionally filtered by tag."""
        with self._guarded_lock():
            if tag is None:
                return list(dict.fromkeys([*self.cache.keys(), *self.disk_index.keys()]))
            return list(self.tag_index.get(tag, []))
 
    def clear_cache(self, key: Optional[str] = None, tag: Optional[str] = None):
        """Clear cache by key, tag, or entirely (in memory and on disk)."""
        with self._guarded_lock():
            if key:
                self.cache.pop(key, None)
                self._remove_disk_entry(key)
                self._remove_from_tag_index(key)
                logger.debug(f"Cleared cache item: {key}")
                return
 
            if tag:
                keys = list(self.tag_index.pop(tag, []))
                for item_key in keys:
                    self.cache.pop(item_key, None)
                    self._remove_disk_entry(item_key)
                    # An item can carry several tags; scrub it from all of
                    # them, not just the one it was cleared by.
                    self._remove_from_tag_index(item_key)
                logger.debug(f"Cleared {len(keys)} items with tag: {tag}")
                return
 
            # Clear entire cache (memory + disk)
            for item_key in list(self.disk_index.keys()):
                self._remove_disk_entry(item_key)
            self.cache.clear()
            self.tag_index.clear()
            logger.info("PerceptionMemory cache cleared")
 
    def _remove_disk_entry(self, key: str) -> None:
        record = self.disk_index.pop(key, None)
        if record is None:
            return
        try:
            if os.path.exists(record['path']):
                os.remove(record['path'])
        except OSError as exc:
            logger.error(f"Failed to remove spilled file for key {key}: {exc}")
        self._save_disk_index()
 
    # ------------------------------------------------------------------
    # Gradient checkpointing passthrough
    # ------------------------------------------------------------------
    def run_checkpointed(self,
                       fn: Callable,
                       *args,
                       preserve_rng_state: bool = False,
                       **kwargs) -> Any:
        """Run function with gradient checkpointing."""
        if not self.enable_checkpointing:
            return fn(*args, **kwargs)
 
        return torch_checkpoint(
            fn,
            *args,
            use_reentrant=False,
            preserve_rng_state=preserve_rng_state,
            **kwargs
        )
 
    # ------------------------------------------------------------------
    # Disk checkpoints (full tensor snapshots, distinct from cache spillover)
    # ------------------------------------------------------------------
    def checkpoint(self,
                 tensor: torch.Tensor,
                 file_prefix: str = "memory_checkpoint",
                 metadata: Optional[Dict] = None) -> str:
        """Save tensor to disk checkpoint, pruning old files beyond retention limit."""
        if not self.enable_checkpointing:
            raise PerceptionStateError(
                "Checkpointing is disabled.", component=self._COMPONENT,
                remediation="Enable checkpointing before calling checkpoint().",
            )
        require_tensor(tensor, "tensor", component=self._COMPONENT)
        if self.validate_finite:
            ensure_finite_tensor(tensor, "tensor", component=self._COMPONENT)
 
        metadata = dict(metadata or {})
        metadata.update({
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
        })
 
        file_hash = hashlib.new(self.key_algorithm, tensor.detach().cpu().numpy().tobytes()).hexdigest()
        filename = f"{self.checkpoint_dir}/{file_prefix}_{file_hash}.pt"
 
        try:
            torch.save({'tensor': tensor.detach().cpu(), 'metadata': metadata}, filename)
        except OSError as exc:
            raise wrap_exception(
                exc, PerceptionStateError, f"Failed to write checkpoint to {filename}.",
                component=self._COMPONENT, details={"filename": filename},
            )
 
        logger.debug(f"Checkpoint saved to {filename}")
        self._prune_checkpoints(file_prefix)
        return filename
 
    def _prune_checkpoints(self, file_prefix: str) -> int:
        """Delete the oldest checkpoint files beyond ``checkpoint_retention``."""
        if not self.checkpoint_retention:
            return 0
        pattern = os.path.join(self.checkpoint_dir, f"{file_prefix}_*.pt")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        stale = files[self.checkpoint_retention:]
        for path in stale:
            try:
                os.remove(path)
            except OSError as exc:
                logger.error(f"Failed to prune checkpoint {path}: {exc}")
        if stale:
            logger.debug(f"Pruned {len(stale)} stale checkpoint(s) for prefix '{file_prefix}'")
        return len(stale)
 
    def list_checkpoints(self, file_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """List checkpoint files with basic diagnostics, newest first."""
        pattern = os.path.join(self.checkpoint_dir, f"{file_prefix or '*'}_*.pt")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        return [
            {'path': path, 'size_mb': os.path.getsize(path) / (1024 ** 2), 'mtime': os.path.getmtime(path)}
            for path in files
        ]
 
    def load_checkpoint(self, filename: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Load tensor from disk checkpoint."""
        resolved_device = resolve_torch_device(device) if device is not None else None
        try:
            payload = torch.load(filename, map_location=resolved_device)
        except (OSError, RuntimeError) as exc:
            raise wrap_exception(
                exc, PerceptionStateError, f"Failed to load checkpoint from {filename}.",
                component=self._COMPONENT, details={"filename": filename},
            )
        tensor = payload['tensor']
        logger.debug(f"Loaded checkpoint from {filename}")
        return tensor.to(resolved_device) if resolved_device else tensor
 
    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._guarded_lock():
            cache_size_mb = self._current_memory_mb()
            disk_size_mb = sum(
                os.path.getsize(rec['path']) / (1024 ** 2)
                for rec in self.disk_index.values() if os.path.exists(rec['path'])
            )
            return {
                'cache_size': len(self.cache),
                'disk_spilled_items': len(self.disk_index),
                'memory_usage_mb': cache_size_mb,
                'disk_usage_mb': disk_size_mb,
                'hit_rate': self.hit_count / (self.hit_count + self.miss_count)
                            if (self.hit_count + self.miss_count) > 0 else 0,
                'total_stored': self.total_stored,
                'eviction_count': self.eviction_count,
                'spill_count': self.spill_count,
                'expired_count': self.expired_count,
                'enable_cache': self.enable_cache,
                'enable_checkpointing': self.enable_checkpointing,
                'enable_disk_persistence': self.enable_disk_persistence,
            }
 
    def toggle_cache(self, enable: bool):
        """Enable or disable caching."""
        self.enable_cache = enable
        logger.info(f"Caching {'enabled' if enable else 'disabled'}")
 
    def toggle_checkpointing(self, enable: bool):
        """Enable or disable gradient checkpointing."""
        self.enable_checkpointing = enable
        logger.info(f"Gradient checkpointing {'enabled' if enable else 'disabled'}")
 
    # ------------------------------------------------------------------
    # Unified forward interface
    # ------------------------------------------------------------------
    def forward(self,
                input: Optional[Union[torch.Tensor, str, List[str]]] = None,
                operation: str = 'auto',
                tags: Optional[List[str]] = None,
                metadata: Optional[Dict] = None,
                device: Optional[torch.device] = None,
                file_prefix: str = "memory_forward",
                preserve_rng_state: bool = False,
                checkpoint_fn: Optional[Callable] = None,
                modality: Optional[str] = None) -> Union[torch.Tensor, str, List[torch.Tensor], None]:
        """
        Unified forward method for memory operations with intelligent auto-detection.
        """
        if operation == 'auto':
            if isinstance(input, torch.Tensor):
                operation = 'cache'
            elif isinstance(input, str):
                operation = 'retrieve'
            elif isinstance(input, list) and all(isinstance(i, str) for i in input):
                operation = 'retrieve_by_tag'
            else:
                raise InvalidPerceptionValueError(
                    "Could not auto-detect operation from input type.", component=self._COMPONENT,
                    details={"input_type": type(input).__name__},
                )
 
        if operation == 'cache':
            require_tensor(input, "input", component=self._COMPONENT)
            return self.cache_item(input, tags=tags, metadata=metadata, modality=modality)
 
        elif operation == 'retrieve':
            ensure_instance(input, str, "input", component=self._COMPONENT)
            return self.retrieve(key=input, device=device)
 
        elif operation == 'retrieve_by_tag':
            resolved_tags = input if isinstance(input, list) else [input]
            for item in resolved_tags:
                ensure_instance(item, str, "tag", component=self._COMPONENT)
 
            results = {}
            for tag in resolved_tags:
                items = self.retrieve(tag=tag, device=device)
                for i, item in enumerate(items):
                    item_key = f"{tag}_{i}"
                    if item_key not in results:
                        results[item_key] = {'tensor': item, 'count': 1}
                    else:
                        results[item_key]['count'] += 1
 
            matched_items = [v['tensor'] for v in results.values() if v['count'] == len(resolved_tags)]
            return matched_items
 
        elif operation == 'checkpoint':
            require_tensor(input, "input", component=self._COMPONENT)
            return self.checkpoint(input, file_prefix=file_prefix, metadata=metadata)
 
        elif operation == 'run_checkpointed':
            ensure_not_none(checkpoint_fn, "checkpoint_fn", component=self._COMPONENT)
            return self.run_checkpointed(
                checkpoint_fn,
                input,
                preserve_rng_state=preserve_rng_state,
                **(metadata or {})
            )
 
        elif operation == 'update':
            ensure_instance(input, tuple, "input", component=self._COMPONENT)
            ensure(len(input) == 2, "Input must be a tuple (key, tensor) for update.",
                   exc_type=InvalidPerceptionValueError, component=self._COMPONENT)
            key, tensor = input
            self.clear_cache(key=key)
            return self.cache_item(tensor, key=key, tags=tags, metadata=metadata, modality=modality)
 
        elif operation == 'purge_expired':
            return self.purge_expired()
 
        elif operation == 'stats':
            return self.memory_stats()
 
        else:
            raise UnsupportedPerceptionOptionError(
                f"Unsupported operation: {operation}", component=self._COMPONENT,
                details={"operation": operation},
            )
 
    def __repr__(self):
        stats = self.memory_stats()
        return (f"PerceptionMemory(cache_size={stats['cache_size']}, "
                f"disk_spilled={stats['disk_spilled_items']}, "
                f"memory={stats['memory_usage_mb']:.2f}MB, "
                f"hit_rate={stats['hit_rate']:.2f})")
 
__all__ = ["PerceptionMemory"]

if __name__ == "__main__":
    print("\n=== Running Perception Memory ===\n")
    printer.status("TEST", "Perception Memory initialized", "info")
 
    memory = PerceptionMemory(enable_checkpointing=True, enable_cache=True)
    print(memory)
 
    # --- Basic tensor cache/retrieve ---
    tensor = torch.randn(1, 128, 512)
    key = memory.cache_item(tensor, tags=["feature", "test"], metadata={"source": "unit_test"})
    assert torch.allclose(memory.retrieve(key=key), tensor)
    printer.status("TEST", f"cache/retrieve OK (key={key[:8]}...)", "success")
 
    # --- Modality-tagged multimodal dict caching ---
    bundle = {"vision": torch.randn(4, 16), "audio": torch.randn(4, 8)}
    bkey = memory.cache_item(bundle, modality="vision", tags=["bundle"])
    fetched = memory.retrieve(key=bkey)
    assert torch.equal(fetched["vision"], bundle["vision"])
    assert memory.retrieve_by_modality("vision")
    printer.status("TEST", "modality tagging + nested structure OK", "success")
 
    # --- Tag-based bulk retrieval + clearing ---
    assert len(memory.retrieve(tag="bundle")) == 1
    memory.clear_cache(tag="bundle")
    assert not memory.contains(bkey)
    printer.status("TEST", "tag retrieval/clear OK", "success")
 
    # --- LRU + memory-budget eviction ---
    small_mem = PerceptionMemory(enable_checkpointing=False, enable_cache=True)
    small_mem.max_cache_size = 2
    k1 = small_mem.cache_item(torch.randn(2, 2))
    k2 = small_mem.cache_item(torch.randn(2, 2))
    k3 = small_mem.cache_item(torch.randn(2, 2))
    assert not small_mem.contains(k1) and small_mem.contains(k3)
    printer.status("TEST", "LRU eviction OK", "success")
 
    # --- Disk checkpoint round-trip + retention ---
    ckpt_path = memory.checkpoint(tensor, file_prefix="unit_test")
    loaded = memory.load_checkpoint(ckpt_path)
    assert torch.allclose(loaded, tensor)
    printer.status("TEST", "checkpoint save/load OK", "success")
 
    # --- Error handling ---
    try:
        memory.retrieve(key="does-not-exist")
        raise AssertionError("expected KeyError")
    except KeyError:
        printer.status("TEST", "missing-key error handling OK", "success")
 
    try:
        memory.cache_item(torch.tensor([float("nan")]))
        raise AssertionError("expected NonFiniteLossError")
    except NonFiniteLossError:
        printer.status("TEST", "non-finite validation OK", "success")
 
    printer.pretty("STATS", memory.memory_stats(), "info")
    print("\n=== Test ran successfully ===\n")
