"""STEM-specific deterministic computation cache and reproducibility metadata.

This is deliberately not a replacement for SLAI SharedMemory or Knowledge
memory. It memorizes scientific computations using precision-sensitive keys.
"""
from __future__ import annotations

__version__ = "2.3.0"

import copy
import hashlib
import json
import time

from collections import OrderedDict
from threading import RLock
from typing import Any, Dict, Mapping, Optional

from .utils.config_loader import get_config_section, load_global_config
from .utils.stem_errors import STEMValidationError
from .utils.stem_helpers import json_safe
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Memory")
printer = PrettyPrinter()


class STEMMemory:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.mem_config = dict(get_config_section("stem_memory", config=self.config) or {})
        if config:
            self.mem_config.update(dict(config))
        self.max_entries = int(self.mem_config.get("max_entries", 512))
        self.ttl_seconds = float(self.mem_config.get("ttl_seconds", 3600.0))
        if self.max_entries < 1 or self.ttl_seconds < 0.0:
            raise STEMValidationError("Invalid STEM memory limits")
        self._cache: OrderedDict[str, tuple[float, Any, Mapping[str, Any]]] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def build_key(*, operation: str, inputs: Any, method: Optional[str] = None, precision: Any = None, tolerances: Any = None, algorithm_version: str = "2.3.0", constants_version: Optional[str] = None, boundary_conditions: Any = None, seed: Optional[int] = None, extra: Any = None) -> str:
        payload = {
            "operation": operation, "inputs": json_safe(inputs), "method": method,
            "precision": json_safe(precision), "tolerances": json_safe(tolerances),
            "algorithm_version": algorithm_version, "constants_version": constants_version,
            "boundary_conditions": json_safe(boundary_conditions), "seed": seed, "extra": json_safe(extra),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> Any:
        now = time.monotonic()
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                self.misses += 1
                return None
            created, value, _ = item
            if self.ttl_seconds > 0.0 and now - created > self.ttl_seconds:
                del self._cache[key]; self.misses += 1; return None
            self._cache.move_to_end(key); self.hits += 1
            return copy.deepcopy(value)

    def put(self, key: str, value: Any, *, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if not isinstance(key, str) or not key:
            raise STEMValidationError("cache key must be non-empty string")
        with self._lock:
            self._cache[key] = (time.monotonic(), copy.deepcopy(value), dict(metadata or {}))
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)

    def metadata(self, key: str) -> Optional[Mapping[str, Any]]:
        with self._lock:
            item = self._cache.get(key)
            return None if item is None else copy.deepcopy(item[2])

    def clear(self) -> None:
        with self._lock:
            self._cache.clear(); self.hits = self.misses = 0

    def stats(self) -> Mapping[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {"entries": len(self._cache), "hits": self.hits, "misses": self.misses, "hit_rate": self.hits / total if total else 0.0, "max_entries": self.max_entries, "ttl_seconds": self.ttl_seconds}


__all__ = ["STEMMemory"]
