import gzip
import hashlib
import json
import os
import pickle
import shelve
import threading
import time
import lz4.frame

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

from .utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Memory")
printer = PrettyPrinter

class ExecutionMemory:
    """
    Production-oriented execution memory manager with:
    - Two-tier cache (in-memory + shelve-backed disk cache)
    - TTL-aware cache retrieval and cleanup
    - Compression-aware persistence
    - Versioned checkpoints with tag indexing and repair helpers
    - Cookie persistence with atomic writes and convenience helpers
    - Explicit lifecycle management (flush/close/context manager)
    - Thread-safe operations and operational statistics
    """

    _TAG_INDEX_KEY = "__tag_index"
    _CHECKPOINT_META_KEY = "__checkpoint_meta"
    _EXPORT_FORMAT_VERSION = 1

    def __init__(self):
        self.lock = threading.RLock()
        self._closed = False

        self.config = load_global_config()
        self.memory_config = get_config_section("execution_memory") or {}
        self._load_and_normalize_config()

        self.cache_dir = self._ensure_dir(self.cache_dir)
        self.checkpoint_dir = self._ensure_dir(self.checkpoint_dir)
        self.cookie_jar_path = os.path.abspath(self.cookie_jar_path)

        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.disk_cache = self._init_shelve(os.path.join(self.cache_dir, "agent_cache.db"))
        self.checkpoint_store = self._init_shelve(
            os.path.join(self.checkpoint_dir, "checkpoints.db")
        )
        
        # Statistics and monitoring must exist before any helper uses them
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        self.cache_evictions = 0
        self.checkpoints_created = 0
        self.checkpoints_restored = 0
        self.cookies_saved = 0
        self.cookies_loaded = 0
        self.last_validation = time.time()
        self.last_cleanup = time.time()
        
        self._validate_checkpoint_store(rebuild=True)
        self.cookies: Dict[str, Any] = self._load_cookies()

        logger.info(
            "Execution Memory initialized "
            f"(compression={self.compression_algorithm}, cache_ttl={self.cache_ttl}s)"
        )

    # ------------------------------------------------------------------
    # Configuration and lifecycle
    # ------------------------------------------------------------------
    def _load_and_normalize_config(self) -> None:
        """Load, validate, and normalize configuration values."""
        memory_config = self.memory_config

        self.cache_dir = memory_config.get("cache_dir")
        self.checkpoint_dir = memory_config.get("checkpoint_dir")
        self.cookie_jar_path = memory_config.get("cookie_jar")

        self.cache_ttl = max(1, int(memory_config.get("cache_ttl", 3600)))
        self.max_memory_cache = max(1, int(memory_config.get("max_memory_cache", 500)))
        self.compression_threshold = max(0, int(memory_config.get("compression_threshold", 1024)))
        self.cleanup_interval = max(30, int(memory_config.get("cleanup_interval", 3600)))
        self.compression_algorithm = str(memory_config.get("compression", "lz4")).lower()
        if self.compression_algorithm not in {"none", "lz4", "gzip"}:
            logger.warning(
                "Unsupported compression '%s'; falling back to 'none'",
                self.compression_algorithm,
            )
            self.compression_algorithm = "none"

        self.export_compression = str(
            memory_config.get("export_compression", self.compression_algorithm)
        ).lower()
        if self.export_compression not in {"none", "lz4", "gzip"}:
            self.export_compression = "lz4"

        self.default_cookie_ttl = max(1, int(memory_config.get("default_cookie_ttl", 30 * 86400)))
        self.max_checkpoints = max(1, int(memory_config.get("max_checkpoints", 1000)))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ExecutionMemory is already closed")

    def close(self) -> None:
        """Flush state and close all persistent stores."""
        with self.lock:
            if self._closed:
                return
            try:
                self.flush()
            finally:
                try:
                    self.disk_cache.close()
                finally:
                    try:
                        self.checkpoint_store.close()
                    finally:
                        self._closed = True
                        logger.info("Execution Memory closed")

    def flush(self) -> None:
        """Persist pending cookie and shelve changes."""
        with self.lock:
            self._ensure_open()
            self.save_cookies()
            self.disk_cache.sync()
            self.checkpoint_store.sync()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Avoid destructor-time exceptions during interpreter shutdown.
            pass

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------
    def _ensure_dir(self, path: str) -> str:
        full_path = os.path.abspath(path)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def _init_shelve(self, path: str) -> shelve.DbfilenameShelf:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return shelve.open(path, flag="c", protocol=pickle.HIGHEST_PROTOCOL, writeback=False)

    def _atomic_write_bytes(self, path: str, payload: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)

    def _serialize(self, data: Any) -> bytes:
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def _compress_bytes(self, payload: bytes, algorithm: Optional[str] = None) -> Tuple[bytes, str, bool]:
        algorithm = (algorithm or self.compression_algorithm).lower()
        if len(payload) < self.compression_threshold or algorithm == "none":
            return payload, "none", False
        if algorithm == "lz4":
            return lz4.frame.compress(payload), "lz4", True
        if algorithm == "gzip":
            return gzip.compress(payload), "gzip", True
        return payload, "none", False

    def _decompress_bytes(self, payload: bytes, algorithm: Optional[str] = None) -> bytes:
        algo = (algorithm or "auto").lower()
        if algo == "lz4" or (algo == "auto" and payload.startswith(b"\x04\x22\x4d\x18")):
            return lz4.frame.decompress(payload)
        if algo == "gzip" or (algo == "auto" and payload.startswith(b"\x1f\x8b")):
            return gzip.decompress(payload)
        return payload

    def _pack_data(self, data: Any, algorithm: Optional[str] = None) -> Dict[str, Any]:
        serialized = self._serialize(data)
        blob, used_algorithm, compressed = self._compress_bytes(serialized, algorithm)
        return {
            "blob": blob,
            "compression": used_algorithm,
            "compressed": compressed,
            "raw_size": len(serialized),
            "stored_size": len(blob),
        }

    def _unpack_data(self, payload: Dict[str, Any]) -> Any:
        blob = payload["blob"]
        compression = payload.get("compression", "auto")
        decompressed = self._decompress_bytes(blob, compression)
        return pickle.loads(decompressed)

    def _run_periodic_maintenance(self) -> None:
        now = time.time()
        if now - self.last_cleanup >= self.cleanup_interval:
            self.clean_cache()
            self.last_cleanup = now
        if now - self.last_validation >= self.cleanup_interval:
            self.validate_cache(repair=True)
            self._validate_checkpoint_store(rebuild=False)
            self.last_validation = now

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _normalize_cache_params(self, params: Optional[Dict[str, Any]]) -> str:
        if not params:
            return ""
        try:
            return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
        except TypeError:
            # Fallback for nested non-JSON-serializable values.
            return urlencode(sorted((str(k), repr(v)) for k, v in params.items()))

    def _cache_key(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
    ) -> str:
        base = f"{namespace}::{url}::{self._normalize_cache_params(params)}"
        digest = hashlib.blake2b(base.encode("utf-8"), digest_size=16).hexdigest()
        return f"cache::{namespace}::{digest}"

    def _is_entry_expired(self, entry: Dict[str, Any], now: Optional[float] = None) -> bool:
        now = now or time.time()
        expire_at = entry.get("expire")
        if expire_at is None:
            timestamp = entry.get("timestamp", now)
            ttl = entry.get("ttl", self.cache_ttl)
            expire_at = timestamp + ttl
        return expire_at <= now

    def _prune_memory_cache_if_needed(self) -> None:
        while len(self.memory_cache) > self.max_memory_cache:
            oldest_key = min(
                self.memory_cache,
                key=lambda k: self.memory_cache[k].get("timestamp", 0.0),
            )
            del self.memory_cache[oldest_key]
            self.cache_evictions += 1

    def set_cache(
        self,
        url: str,
        data: Any,
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        namespace: str = "default",
        memory_only: bool = False,
    ) -> str:
        """Store a cache entry in memory and optionally on disk."""
        with self.lock:
            self._ensure_open()
            ttl = max(1, int(ttl or self.cache_ttl))
            now = time.time()
            key = self._cache_key(url, params=params, namespace=namespace)
            entry = {
                "data": data,
                "timestamp": now,
                "expire": now + ttl,
                "ttl": ttl,
                "namespace": namespace,
                "url": url,
                "params": params or {},
            }
            self.memory_cache[key] = entry
            self._prune_memory_cache_if_needed()

            if not memory_only:
                packed = self._pack_data(data)
                self.disk_cache[key] = {
                    "payload": packed,
                    "timestamp": now,
                    "expire": entry["expire"],
                    "ttl": ttl,
                    "namespace": namespace,
                    "url": url,
                    "params": params or {},
                }

            self.cache_sets += 1
            self._run_periodic_maintenance()
            return key

    def get_cache(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        default: Any = None,
    ) -> Any:
        """Retrieve a cache value, promoting valid disk entries into memory."""
        with self.lock:
            self._ensure_open()
            key = self._cache_key(url, params=params, namespace=namespace)
            now = time.time()

            entry = self.memory_cache.get(key)
            if entry is not None:
                if not self._is_entry_expired(entry, now):
                    self.cache_hits += 1
                    self._run_periodic_maintenance()
                    return entry["data"]
                del self.memory_cache[key]

            if key in self.disk_cache:
                try:
                    disk_entry = self.disk_cache[key]
                    if self._is_entry_expired(disk_entry, now):
                        del self.disk_cache[key]
                    else:
                        payload = disk_entry.get("payload")
                        data = self._unpack_data(payload) if payload else disk_entry.get("data")
                        promoted = {
                            "data": data,
                            "timestamp": disk_entry.get("timestamp", now),
                            "expire": disk_entry.get("expire", now + self.cache_ttl),
                            "ttl": disk_entry.get("ttl", self.cache_ttl),
                            "namespace": disk_entry.get("namespace", namespace),
                            "url": disk_entry.get("url", url),
                            "params": disk_entry.get("params", params or {}),
                        }
                        self.memory_cache[key] = promoted
                        self._prune_memory_cache_if_needed()
                        self.cache_hits += 1
                        self._run_periodic_maintenance()
                        return data
                except Exception as exc:
                    logger.error("Corrupted cache entry %s: %s. Removing.", key, exc)
                    try:
                        del self.disk_cache[key]
                    except Exception:
                        pass

            self.cache_misses += 1
            self._run_periodic_maintenance()
            return default

    def has_cache(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
    ) -> bool:
        return self.get_cache(url, params=params, namespace=namespace, default=None) is not None

    def delete_cache(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
    ) -> bool:
        with self.lock:
            self._ensure_open()
            key = self._cache_key(url, params=params, namespace=namespace)
            removed = False
            if key in self.memory_cache:
                del self.memory_cache[key]
                removed = True
            if key in self.disk_cache:
                del self.disk_cache[key]
                removed = True
            return removed

    def clear_cache(self, namespace: Optional[str] = None) -> Dict[str, int]:
        """Clear cache entries, optionally scoped to a namespace."""
        with self.lock:
            self._ensure_open()
            memory_removed = 0
            disk_removed = 0

            if namespace is None:
                memory_removed = len(self.memory_cache)
                disk_removed = len(self.disk_cache)
                self.memory_cache.clear()
                self.disk_cache.clear()
                return {"memory_removed": memory_removed, "disk_removed": disk_removed}

            memory_keys = [
                k for k, v in self.memory_cache.items() if v.get("namespace", "default") == namespace
            ]
            for key in memory_keys:
                del self.memory_cache[key]
            memory_removed = len(memory_keys)

            disk_keys = [
                k
                for k in list(self.disk_cache.keys())
                if isinstance(k, str)
                and k.startswith("cache::")
                and self.disk_cache.get(k, {}).get("namespace", "default") == namespace
            ]
            for key in disk_keys:
                del self.disk_cache[key]
            disk_removed = len(disk_keys)

            return {"memory_removed": memory_removed, "disk_removed": disk_removed}

    def validate_cache(self, repair: bool = True) -> Dict[str, int]:
        """Validate disk cache entries and optionally repair by removing bad ones."""
        with self.lock:
            self._ensure_open()
            scanned = 0
            corrupted = 0
            expired = 0
            removed = 0
            now = time.time()

            for key in list(self.disk_cache.keys()):
                if not isinstance(key, str) or not key.startswith("cache::"):
                    continue
                scanned += 1
                try:
                    entry = self.disk_cache[key]
                    if self._is_entry_expired(entry, now):
                        expired += 1
                        if repair:
                            del self.disk_cache[key]
                            removed += 1
                        continue
                    payload = entry.get("payload")
                    if payload:
                        self._unpack_data(payload)
                except Exception:
                    corrupted += 1
                    if repair:
                        try:
                            del self.disk_cache[key]
                            removed += 1
                        except Exception:
                            pass

            return {
                "scanned": scanned,
                "expired": expired,
                "corrupted": corrupted,
                "removed": removed,
            }

    def clean_cache(self) -> Dict[str, int]:
        """Remove expired entries from memory and disk cache."""
        with self.lock:
            self._ensure_open()
            now = time.time()

            memory_expired = [
                key for key, value in self.memory_cache.items() if self._is_entry_expired(value, now)
            ]
            for key in memory_expired:
                del self.memory_cache[key]

            disk_expired = []
            for key in list(self.disk_cache.keys()):
                if not isinstance(key, str) or not key.startswith("cache::"):
                    continue
                try:
                    if self._is_entry_expired(self.disk_cache[key], now):
                        disk_expired.append(key)
                except Exception:
                    disk_expired.append(key)
            for key in disk_expired:
                try:
                    del self.disk_cache[key]
                except Exception:
                    pass

            logger.info(
                "Cache cleanup complete (memory_removed=%s, disk_removed=%s)",
                len(memory_expired),
                len(disk_expired),
            )
            return {
                "memory_removed": len(memory_expired),
                "disk_removed": len(disk_expired),
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        with self.lock:
            self._ensure_open()
            total_requests = self.cache_hits + self.cache_misses
            return {
                "memory_entries": len(self.memory_cache),
                "disk_entries": len([k for k in self.disk_cache.keys() if str(k).startswith("cache::")]),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_sets": self.cache_sets,
                "cache_evictions": self.cache_evictions,
                "hit_ratio": (self.cache_hits / total_requests) if total_requests else 0.0,
                "last_cleanup": datetime.fromtimestamp(self.last_cleanup).isoformat(),
                "last_validation": datetime.fromtimestamp(self.last_validation).isoformat(),
                "compression_algorithm": self.compression_algorithm,
            }

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def _validate_checkpoint_store(self, rebuild: bool = False) -> None:
        with self.lock:
            self._ensure_open() if hasattr(self, "checkpoint_store") else None

            if self._TAG_INDEX_KEY not in self.checkpoint_store:
                self.checkpoint_store[self._TAG_INDEX_KEY] = {}
            if self._CHECKPOINT_META_KEY not in self.checkpoint_store:
                self.checkpoint_store[self._CHECKPOINT_META_KEY] = {
                    "created_at": time.time(),
                    "version": 1,
                }

            if rebuild:
                self._rebuild_checkpoint_tag_index()

    def _rebuild_checkpoint_tag_index(self) -> Dict[str, List[str]]:
        tag_index: Dict[str, List[str]] = {}
        for key in list(self.checkpoint_store.keys()):
            if not isinstance(key, str) or not key.startswith("chk_"):
                continue
            try:
                entry = self.checkpoint_store[key]
                for tag in entry.get("tags", []) or []:
                    tag_index.setdefault(tag, []).append(key)
            except Exception as exc:
                logger.warning("Skipping corrupted checkpoint metadata for %s: %s", key, exc)
        self.checkpoint_store[self._TAG_INDEX_KEY] = tag_index
        return tag_index

    def _checkpoint_key(self, checkpoint_id: Optional[str] = None) -> str:
        base = checkpoint_id or f"chk_{int(time.time() * 1000)}_{os.getpid()}"
        return base

    def _next_checkpoint_version(self, checkpoint_id: str) -> int:
        version = 1
        while f"{checkpoint_id}_v{version}" in self.checkpoint_store:
            version += 1
        return version

    def create_checkpoint(
        self,
        state: Any,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Create a versioned checkpoint with tags, metadata, and optional expiry."""
        with self.lock:
            self._ensure_open()
            checkpoint_id = self._checkpoint_key(checkpoint_id)
            version = self._next_checkpoint_version(checkpoint_id)
            full_id = f"{checkpoint_id}_v{version}"
            created = time.time()
            expire = created + ttl if ttl else None
            tags = sorted(set(tags or []))
            metadata = dict(metadata or {})

            packed = self._pack_data(state)
            entry = {
                "payload": packed,
                "created": created,
                "expire": expire,
                "tags": tags,
                "metadata": metadata,
                "size": packed["stored_size"],
                "raw_size": packed["raw_size"],
                "compression": packed["compression"],
                "version": version,
            }
            self.checkpoint_store[full_id] = entry

            tag_index = self.checkpoint_store.get(self._TAG_INDEX_KEY, {})
            for tag in tags:
                ids = tag_index.setdefault(tag, [])
                if full_id not in ids:
                    ids.append(full_id)
            self.checkpoint_store[self._TAG_INDEX_KEY] = tag_index

            self.checkpoints_created += 1
            self.prune_checkpoints(max_entries=self.max_checkpoints)
            return full_id

    def restore_checkpoint(self, checkpoint_id: str, default: Any = None) -> Any:
        with self.lock:
            self._ensure_open()
            if checkpoint_id not in self.checkpoint_store:
                return default
    
            entry = self.checkpoint_store[checkpoint_id]
            expire = entry.get("expire")
            if expire and expire <= time.time():
                logger.warning("Checkpoint %s expired; deleting before restore", checkpoint_id)
                self.delete_checkpoint(checkpoint_id)
                return default
    
            try:
                if "payload" in entry:
                    data = self._unpack_data(entry["payload"])
                elif "state" in entry:
                    # Legacy checkpoint format support
                    legacy_blob = entry["state"]
                    compression = entry.get("compression", "auto")
                    if compression == "lz4":
                        raw = lz4.frame.decompress(legacy_blob)
                    elif compression == "gzip":
                        raw = gzip.decompress(legacy_blob)
                    else:
                        raw = legacy_blob
                    data = pickle.loads(raw)
                else:
                    logger.error("Checkpoint %s has no payload/state field", checkpoint_id)
                    return default
            except Exception as exc:
                logger.error("Failed to restore checkpoint %s: %s", checkpoint_id, exc)
                return default
    
            self._migrate_legacy_checkpoint(checkpoint_id, entry, data)
            self.checkpoints_restored += 1
            return data

    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            self._ensure_open()
            if checkpoint_id not in self.checkpoint_store:
                return None
    
            entry = self.checkpoint_store[checkpoint_id]
    
            size = entry.get("size", 0)
            raw_size = entry.get("raw_size", size)
            compression = entry.get("compression", "none")
    
            return {
                "id": checkpoint_id,
                "created": datetime.fromtimestamp(entry["created"]).isoformat(),
                "expire": datetime.fromtimestamp(entry["expire"]).isoformat()
                if entry.get("expire")
                else None,
                "tags": entry.get("tags", []),
                "size": size,
                "raw_size": raw_size,
                "compression": compression,
                "version": entry.get("version", 1),
                "metadata": entry.get("metadata", {}),
                "schema": "new" if "payload" in entry else "legacy" if "state" in entry else "unknown",
            }
        
    def _migrate_legacy_checkpoint(self, checkpoint_id: str, entry: Dict[str, Any], state: Any) -> None:
        packed = self._pack_data(state)
        entry["payload"] = packed
        entry["size"] = packed["stored_size"]
        entry["raw_size"] = packed["raw_size"]
        entry["compression"] = packed["compression"]
        entry["version"] = entry.get("version", 1)
        entry.pop("state", None)
        self.checkpoint_store[checkpoint_id] = entry

    def find_checkpoints(
        self,
        tag: Optional[str] = None,
        min_size: int = 0,
        max_age: Optional[int] = None,
        include_expired: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self.lock:
            self._ensure_open()
            now = time.time()
            results: List[Dict[str, Any]] = []

            if tag:
                tag_index = self.checkpoint_store.get(self._TAG_INDEX_KEY, {})
                checkpoint_ids = list(tag_index.get(tag, []))
            else:
                checkpoint_ids = [
                    k for k in self.checkpoint_store.keys() if isinstance(k, str) and k.startswith("chk_")
                ]

            for checkpoint_id in checkpoint_ids:
                if checkpoint_id not in self.checkpoint_store:
                    continue
                entry = self.checkpoint_store[checkpoint_id]
                created = entry.get("created", now)
                expire = entry.get("expire")
                if not include_expired and expire and expire <= now:
                    continue
                if entry.get("size", 0) < min_size:
                    continue
                if max_age is not None and (now - created) > max_age:
                    continue
                results.append(
                    {
                        "id": checkpoint_id,
                        "created": datetime.fromtimestamp(created).isoformat(),
                        "expire": datetime.fromtimestamp(expire).isoformat() if expire else None,
                        "tags": entry.get("tags", []),
                        "size": entry.get("size", 0),
                        "raw_size": entry.get("raw_size", 0),
                        "compression": entry.get("compression", "none"),
                        "metadata": entry.get("metadata", {}),
                    }
                )

            results.sort(key=lambda item: item["created"], reverse=True)
            return results[:limit] if limit else results

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        with self.lock:
            self._ensure_open()
            if checkpoint_id not in self.checkpoint_store:
                return False

            tags = list(self.checkpoint_store[checkpoint_id].get("tags", []))
            del self.checkpoint_store[checkpoint_id]

            tag_index = self.checkpoint_store.get(self._TAG_INDEX_KEY, {})
            for tag in tags:
                ids = tag_index.get(tag, [])
                if checkpoint_id in ids:
                    ids.remove(checkpoint_id)
                if not ids and tag in tag_index:
                    del tag_index[tag]
            self.checkpoint_store[self._TAG_INDEX_KEY] = tag_index
            return True

    def prune_checkpoints(
        self,
        max_entries: Optional[int] = None,
        max_age: Optional[int] = None,
    ) -> Dict[str, int]:
        """Prune checkpoints by age and/or total count."""
        with self.lock:
            self._ensure_open()
            max_entries = max_entries or self.max_checkpoints
            deleted = 0
            now = time.time()

            checkpoint_ids = [
                k for k in self.checkpoint_store.keys() if isinstance(k, str) and k.startswith("chk_")
            ]
            checkpoint_ids.sort(
                key=lambda cid: self.checkpoint_store[cid].get("created", 0), reverse=True
            )

            if max_age is not None:
                for cid in list(checkpoint_ids):
                    if (now - self.checkpoint_store[cid].get("created", now)) > max_age:
                        if self.delete_checkpoint(cid):
                            deleted += 1
                        checkpoint_ids.remove(cid)

            if len(checkpoint_ids) > max_entries:
                for cid in checkpoint_ids[max_entries:]:
                    if self.delete_checkpoint(cid):
                        deleted += 1

            return {"deleted": deleted, "remaining": len(self.find_checkpoints())}

    # ------------------------------------------------------------------
    # Cookies
    # ------------------------------------------------------------------
    def _load_cookies(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.cookie_jar_path):
                with open(self.cookie_jar_path, "rb") as fh:
                    cookies = pickle.load(fh)
                if isinstance(cookies, dict):
                    self.cookies_loaded = getattr(self, "cookies_loaded", 0) + 1
                    return cookies
                logger.warning("Cookie jar did not contain a dict; resetting")
        except Exception as exc:
            logger.error("Cookie load failed: %s", exc)
        return {}

    def save_cookies(self) -> None:
        with self.lock:
            payload = self._serialize(self.cookies)
            self._atomic_write_bytes(self.cookie_jar_path, payload)
            self.cookies_saved += 1

    def _cookie_key(self, domain: str, name: str, path: str = "/") -> str:
        return f"{domain}|{path}|{name}"

    def set_cookie(
        self,
        name: str,
        value: Any,
        domain: str,
        path: str = "/",
        expires_at: Optional[float] = None,
        secure: bool = False,
        http_only: bool = False,
        same_site: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        with self.lock:
            self._ensure_open()
            cookie_key = self._cookie_key(domain, name, path)
            self.cookies[cookie_key] = {
                "name": name,
                "value": value,
                "domain": domain,
                "path": path,
                "created": time.time(),
                "expires_at": expires_at,
                "secure": secure,
                "http_only": http_only,
                "same_site": same_site,
                "metadata": metadata or {},
            }
            return cookie_key

    def get_cookie(
        self,
        name: str,
        domain: str,
        path: str = "/",
        default: Any = None,
    ) -> Any:
        with self.lock:
            self._ensure_open()
            cookie_key = self._cookie_key(domain, name, path)
            cookie = self.cookies.get(cookie_key)
            if not cookie:
                return default
            expires_at = cookie.get("expires_at")
            if expires_at is not None and expires_at <= time.time():
                del self.cookies[cookie_key]
                return default
            return cookie["value"]

    def delete_cookie(self, name: str, domain: str, path: str = "/") -> bool:
        with self.lock:
            self._ensure_open()
            cookie_key = self._cookie_key(domain, name, path)
            if cookie_key in self.cookies:
                del self.cookies[cookie_key]
                return True
            return False

    def list_cookies(self, domain: Optional[str] = None, include_expired: bool = False) -> List[Dict[str, Any]]:
        with self.lock:
            self._ensure_open()
            now = time.time()
            results = []
            for key, cookie in list(self.cookies.items()):
                expires_at = cookie.get("expires_at")
                expired = expires_at is not None and expires_at <= now
                if expired and not include_expired:
                    continue
                if domain and cookie.get("domain") != domain:
                    continue
                results.append({"key": key, **cookie})
            results.sort(key=lambda item: (item.get("domain", ""), item.get("name", "")))
            return results

    def purge_expired_cookies(self) -> int:
        with self.lock:
            self._ensure_open()
            now = time.time()
            expired_keys = [
                key
                for key, cookie in self.cookies.items()
                if cookie.get("expires_at") is not None and cookie["expires_at"] <= now
            ]
            for key in expired_keys:
                del self.cookies[key]
            return len(expired_keys)

    # ------------------------------------------------------------------
    # Import / export
    # ------------------------------------------------------------------
    def export_memory(self, path: str, include_disk_cache: bool = True) -> str:
        """Export current memory state to a compressed portable snapshot."""
        with self.lock:
            self._ensure_open()
            export_state = {
                "format_version": self._EXPORT_FORMAT_VERSION,
                "timestamp": time.time(),
                "memory_cache": self.memory_cache,
                "cookies": self.cookies,
                "checkpoints": {
                    key: value
                    for key, value in self.checkpoint_store.items()
                    if isinstance(key, str)
                },
                "disk_cache": dict(self.disk_cache) if include_disk_cache else {},
                "stats": self.get_cache_stats(),
            }
            packed = self._pack_data(export_state, algorithm=self.export_compression)
            payload = self._serialize(packed)
            self._atomic_write_bytes(os.path.abspath(path), payload)
            logger.info("Exported execution memory to %s", os.path.abspath(path))
            return os.path.abspath(path)

    def import_memory(self, path: str, merge: bool = True) -> Dict[str, int]:
        """Import a previously exported memory snapshot."""
        with self.lock:
            self._ensure_open()
            with open(path, "rb") as fh:
                packed = pickle.load(fh)
            state = self._unpack_data(packed)

            if not merge:
                self.memory_cache.clear()
                self.cookies.clear()
                for key in list(self.disk_cache.keys()):
                    del self.disk_cache[key]
                for key in list(self.checkpoint_store.keys()):
                    del self.checkpoint_store[key]
                self._validate_checkpoint_store(rebuild=False)

            self.memory_cache.update(state.get("memory_cache", {}))
            self.cookies.update(state.get("cookies", {}))
            for key, value in state.get("disk_cache", {}).items():
                self.disk_cache[key] = value
            for key, value in state.get("checkpoints", {}).items():
                self.checkpoint_store[key] = value

            self._validate_checkpoint_store(rebuild=True)
            logger.info("Imported execution memory from %s", os.path.abspath(path))
            return {
                "memory_cache_entries": len(state.get("memory_cache", {})),
                "cookies": len(state.get("cookies", {})),
                "disk_cache_entries": len(state.get("disk_cache", {})),
                "checkpoints": len(state.get("checkpoints", {})),
            }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        with self.lock:
            self._ensure_open()
            return {
                "cache": self.get_cache_stats(),
                "checkpoints": {
                    "count": len(
                        [
                            key
                            for key in self.checkpoint_store.keys()
                            if isinstance(key, str) and key.startswith("chk_")
                        ]
                    ),
                    "created": self.checkpoints_created,
                    "restored": self.checkpoints_restored,
                },
                "cookies": {
                    "count": len(self.cookies),
                    "saved": self.cookies_saved,
                    "loaded": self.cookies_loaded,
                },
                "closed": self._closed,
            }


if __name__ == "__main__":
    print("\n=== Running Execution Memory Test ===\n")
    printer.status("Init", "Execution Memory initialized", "success")

    with ExecutionMemory() as memory:
        memory.set_cache(
            "https://api.example.com/data",
            {"results": [1, 2, 3]},
            params={"page": 1},
            namespace="api",
        )
        cached_data = memory.get_cache(
            "https://api.example.com/data", params={"page": 1}, namespace="api"
        )
        printer.pretty("Cache Test", bool(cached_data), "success")

        checkpoint_id = memory.create_checkpoint(
            {"position": (10.5, 20.3), "inventory": ["item1", "item2"]},
            tags=["mission_critical", "backup"],
            metadata={"mission": "reconnaissance"},
        )
        printer.pretty("Checkpoint Created", checkpoint_id, "info")

        restored = memory.restore_checkpoint(checkpoint_id)
        printer.pretty("Checkpoint Restore", bool(restored), "success")

        memory.set_cookie("session", "abc123", domain="api.example.com", secure=True)
        printer.pretty(
            "Cookie Exists",
            bool(memory.get_cookie("session", domain="api.example.com")),
            "success",
        )

        stats = memory.summary()
        printer.pretty("Summary", stats, "info")

    print("\n=== Simulation Complete ===")
