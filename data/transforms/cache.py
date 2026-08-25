"""
Caching for transforms: memoize transform outputs to memory and/or disk.
"""
from __future__ import annotations

import hashlib
import json
import threading

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_transform import Transform
from .registry import list_transforms, _TRANSFORM_REGISTRY, _REGISTRY_LOCK
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Cache Transform")
printer = PrettyPrinter()

try:
    import joblib  # type: ignore
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# In-memory FIFO cache (thread-safe)
# ---------------------------------------------------------------------------
class _MemoryCache:
    """Thread-safe fixed-capacity dict with FIFO eviction.

    Using ``collections.OrderedDict`` ensures O(1) eviction of the oldest
    entry while keeping insertion order.  Accessing an item (``get``) does
    **not** change its position – this is pure FIFO, not LRU.
    """

    def __init__(self, maxsize: int = 128) -> None:
        if maxsize < 1:
            raise DataConfigError(
                "_MemoryCache maxsize must be >= 1",
                context={"maxsize": maxsize},
            )
        from collections import OrderedDict
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                # Refresh position (move to end) – still FIFO on eviction
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)  # FIFO eviction
                self._cache[key] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# CachedTransform
# ---------------------------------------------------------------------------
class CachedTransform(Transform):
    """Wrap any ``Transform`` with transparent memory and/or disk caching.

    Cache key is a SHA-256 digest of the serialised values of *hash_keys*
    (plus the modality) so that semantically identical records hit the cache
    regardless of insertion order or unrelated fields.

    Config keys (``transforms.cache``):

    * ``cache_type`` (str) — ``"memory"``, ``"disk"``, or ``"both"``.
    * ``memory_maxsize`` (int, default ``128``) — in-memory cache capacity.
    * ``hash_keys`` (list[str]) — record fields included in the cache key.

    Parameters
    ----------
    inner:
        The wrapped transform to cache.
    cache_dir:
        Required when ``cache_type`` is ``"disk"`` or ``"both"``.
        Overrides the config value.
    memory_maxsize:
        In-memory cache capacity (overrides config).
    hash_keys:
        Fields used to derive the cache key (overrides config).
    """

    def __init__(
        self,
        inner: Transform,
        cache_dir: Optional[str] = None,
        memory_maxsize: Optional[int] = None,
        hash_keys: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(name=f"Cached({inner.name})")
        self.cache_cfg: Dict[str, Any] = get_config_section("transforms").get("cache", {})

        self.cache_type: str = str(self.cache_cfg.get("cache_type", "memory"))
        if self.cache_type not in ("memory", "disk", "both"):
            raise DataConfigError(
                f"CachedTransform: unsupported cache_type '{self.cache_type}'",
                context={"valid": ["memory", "disk", "both"]},
            )

        _maxsize: int = memory_maxsize or int(self.cache_cfg.get("memory_maxsize", 128))
        _hash_keys: Tuple[str, ...] = hash_keys or tuple(
            self.cache_cfg.get("hash_keys", ["text", "image", "audio"])
        )

        self.inner: Transform = inner
        self.hash_keys: Tuple[str, ...] = _hash_keys
        self.memory_cache: Optional[_MemoryCache] = (
            _MemoryCache(maxsize=_maxsize) if self.cache_type in ("memory", "both") else None
        )

        _cfg_dir: Optional[str] = self.cache_cfg.get("cache_dir")
        self.cache_dir: Optional[Path] = (
            Path(cache_dir or _cfg_dir).expanduser().resolve()
            if (cache_dir or _cfg_dir)
            else None
        )

        if self.cache_type in ("disk", "both"):
            if not JOBLIB_AVAILABLE:
                raise DataConfigError(
                    "CachedTransform disk caching requires joblib: pip install joblib",
                    context={"cache_type": self.cache_type},
                )
            if self.cache_dir is None:
                raise DataConfigError(
                    "CachedTransform: cache_dir is required when cache_type is 'disk' or 'both'",
                    context={"cache_type": self.cache_type},
                )
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache-key helpers
    # ------------------------------------------------------------------
    def _make_key(self, record: Dict[str, Any], modality: str) -> str:
        """Return a stable SHA-256 hex key for *record* at *modality*."""
        subset: Dict[str, Any] = {k: record.get(k) for k in self.hash_keys}
        subset["_modality"] = modality
        try:
            serialised = json.dumps(subset, sort_keys=True, default=str)
        except Exception as exc:
            raise DataTransformError(
                "CachedTransform: failed to serialise record for cache key",
                context={"hash_keys": list(self.hash_keys)},
                cause=exc,
            ) from exc
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    def _disk_path(self, key: str) -> Path:
        """Two-level shard path: ``<cache_dir>/<key[:2]>/<key>.joblib``."""
        if self.cache_dir is None:
            raise DataConfigError(
                "CachedTransform._disk_path called without cache_dir",
                context={},
            )
        return self.cache_dir / key[:2] / f"{key}.joblib"

    # ------------------------------------------------------------------
    # Transform interface
    # ------------------------------------------------------------------
    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        key = self._make_key(record, modality)

        # 1. Memory hit
        if self.memory_cache:
            cached = self.memory_cache.get(key)
            if cached is not None:
                logger.debug({"event": "cache_hit", "layer": "memory", "key": key[:8]})
                return cached

        # 2. Disk hit
        if self.cache_type in ("disk", "both"):
            path = self._disk_path(key)
            if path.exists():
                try:
                    cached = joblib.load(path)
                    if self.memory_cache:
                        self.memory_cache.set(key, cached)
                    logger.debug({"event": "cache_hit", "layer": "disk", "key": key[:8]})
                    return cached
                except Exception as exc:
                    logger.warning({
                        "event": "cache_disk_load_failed",
                        "path": str(path),
                        "error": str(exc),
                    })

        # 3. Compute
        result = self.inner(record, modality)

        # 4. Store
        if self.memory_cache:
            self.memory_cache.set(key, result)

        if self.cache_type in ("disk", "both"):
            path = self._disk_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                joblib.dump(result, path)
                logger.debug({"event": "cache_written", "layer": "disk", "key": key[:8]})
            except Exception as exc:
                logger.warning({
                    "event": "cache_disk_write_failed",
                    "path": str(path),
                    "error": str(exc),
                })

        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_config(self) -> Dict[str, Any]:
        return {
            "type": "cached_transform",
            "params": {
                "inner": self.inner.to_config(),
                "cache_type": self.cache_type,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "memory_maxsize": self.memory_cache.maxsize if self.memory_cache else 128,
                "hash_keys": list(self.hash_keys),
            },
        }

    def _get_params(self) -> Dict[str, Any]:
        return {
            "inner": self.inner.name,
            "cache_type": self.cache_type,
            "hash_keys": list(self.hash_keys),
        }


# ---------------------------------------------------------------------------
# Idempotent manual registration (avoids double‑execution errors)
# ---------------------------------------------------------------------------
def _register_cached_transform() -> None:
    registered = list_transforms()
    with _REGISTRY_LOCK:
        if "cached_transform" not in registered:
            _TRANSFORM_REGISTRY["cached_transform"] = CachedTransform
            logger.debug({"event": "transform_registered", "name": "cached_transform", "class": "CachedTransform"})

_register_cached_transform()


if __name__ == "__main__":
    print("\n=== Running cache ===\n")
    printer.status("TEST", "cache initialized", "info")

    # _MemoryCache — basic get/set/eviction (FIFO)
    mc = _MemoryCache(maxsize=3)
    mc.set("a", 1)
    mc.set("b", 2)
    mc.set("c", 3)
    # Order: a, b, c
    assert mc.get("a") == 1   # does NOT change order
    # Now set "d" – should evict the oldest (a)
    mc.set("d", 4)
    assert mc.get("a") is None   # a evicted
    assert mc.get("b") == 2      # b still there
    assert mc.get("d") == 4
    assert len(mc) == 3
    printer.status("PASS", "_MemoryCache FIFO eviction", "success")

    mc.clear()
    assert len(mc) == 0
    printer.status("PASS", "_MemoryCache.clear()", "success")

    # _MemoryCache — invalid maxsize
    try:
        _MemoryCache(maxsize=0)
        assert False
    except DataConfigError:
        printer.status("PASS", "_MemoryCache rejects maxsize=0", "success")

    # CachedTransform — memory caching, call count tracking
    from unittest.mock import patch

    call_count = [0]

    class _CountingTransform(Transform):
        def __call__(self, record, modality):
            call_count[0] += 1
            record["computed"] = True
            return record
        def _get_params(self): return {}

    # Mock config to guarantee memory cache
    with patch('data.transforms.cache.get_config_section') as mock_get_config:
        def fake_get_config(section):
            if section == 'transforms':
                return {'cache': {'cache_type': 'memory', 'memory_maxsize': 128}}
            return {}
        mock_get_config.side_effect = fake_get_config

        ct = CachedTransform(_CountingTransform(), memory_maxsize=10)
        rec = {"text": "hello"}
        r1 = ct(rec.copy(), "text")
        r2 = ct(rec.copy(), "text")
        assert call_count[0] == 1, f"inner called {call_count[0]} times, expected 1"
        assert r1["computed"] is True
        printer.status("PASS", "CachedTransform memory cache hit on second call", "success")

        # Different record triggers a new computation
        r3 = ct({"text": "world"}, "text")
        assert call_count[0] == 2
        printer.status("PASS", "CachedTransform cache miss on different record", "success")

        # to_config round-trip
        cfg = ct.to_config()
        assert cfg["type"] == "cached_transform"
        assert cfg["params"]["cache_type"] == "memory"
        assert cfg["params"]["inner"]["type"] == "_CountingTransform"
        printer.status("PASS", "CachedTransform.to_config serialises correctly", "success")

    print("\n=== Test ran successfully ===\n")