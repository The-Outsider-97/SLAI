"""Thread-safe deterministic computation memory for the SLAI STEM subsystem.

``STEMMemory`` is deliberately local to STEM.  It is not a second SharedMemory,
Knowledge store, or checkpoint backend.  It memoizes exact scientific
computations using canonical, process-stable keys and exposes JSON-compatible
state for the central SLAI checkpoint framework.

Scientific cache identity follows the reproducibility principles used
throughout the STEM subsystem: result-affecting inputs, units, precision,
tolerances, method/version, boundary/initial conditions, constant-set identity,
seed, relevant configuration, and template identity all participate when
supplied.
"""
from __future__ import annotations

__version__ = "2.3.0"

import copy
import hashlib
import inspect
import json
import math
import time

from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .stem_types import (
    ConvergenceStatus,
    Dimension,
    Distribution,
    NumericResult,
    PhysicalConstant,
    PrecisionPolicy,
    Quantity,
    SolverResult,
    Unit,
    Uncertainty as UncertaintyValue,
    UncertaintyType,
)
from .utils.config_loader import get_config_section, load_global_config
from .utils.stem_errors import STEMConfigurationError, STEMValidationError
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Memory")
printer = PrettyPrinter()

_STATE_SCHEMA = "slai.stem.memory.v1"
_KEY_SCHEMA = "slai.stem.cache-key.v1"
_TYPE_MARKER = "__stem_type__"


@dataclass
class _CacheEntry:
    key: str
    created_at: float
    expires_at: Optional[float]
    value: Any
    metadata: Dict[str, Any]

    def expired(self, now: Optional[float] = None) -> bool:
        return self.expires_at is not None and (time.time() if now is None else now) >= self.expires_at


class STEMMemory:
    """Bounded LRU cache for deterministic STEM calculations.

    Runtime values are retained as real Python/STEM objects.  Checkpoint export
    uses a tagged JSON representation so supported scientific value objects can
    be reconstructed safely after restore.  Entries that cannot be restored
    faithfully (for example a callable interpolation closure) remain valid
    runtime cache entries but are omitted from durable state.
    """

    STATE_SCHEMA = _STATE_SCHEMA
    KEY_SCHEMA = _KEY_SCHEMA

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.mem_config = dict(get_config_section("stem_memory", config=self.config) or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise STEMConfigurationError("STEMMemory config override must be a mapping")
            self.mem_config.update(dict(config))

        self.max_entries = self._positive_int(self.mem_config.get("max_entries", 512), "max_entries")
        self.ttl_seconds = self._non_negative_float(self.mem_config.get("ttl_seconds", 3600.0), "ttl_seconds")
        self.algorithm_version = str(self.mem_config.get("algorithm_version", __version__)).strip() or __version__

        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._evictions = 0
        self._expirations = 0
        self._invalidations = 0
        self._checkpoint_skips = 0

    # ------------------------------------------------------------------
    # Configuration and stable canonicalization
    # ------------------------------------------------------------------
    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise STEMConfigurationError(f"{name} must be a positive integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise STEMConfigurationError(f"{name} must be a positive integer", cause=exc) from exc
        if parsed < 1:
            raise STEMConfigurationError(f"{name} must be >= 1", context={"value": value})
        return parsed

    @staticmethod
    def _non_negative_float(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise STEMConfigurationError(f"{name} must be a non-negative number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise STEMConfigurationError(f"{name} must be a non-negative number", cause=exc) from exc
        if not math.isfinite(parsed) or parsed < 0.0:
            raise STEMConfigurationError(f"{name} must be finite and >= 0", context={"value": value})
        return parsed

    @classmethod
    def _callable_identity(cls, value: Any) -> Mapping[str, Any]:
        target = value.__func__ if inspect.ismethod(value) else value
        payload: Dict[str, Any] = {
            "kind": "callable",
            "module": getattr(target, "__module__", None),
            "qualname": getattr(target, "__qualname__", getattr(target, "__name__", type(target).__name__)),
        }
        code = getattr(target, "__code__", None)
        if code is not None:
            digest = hashlib.sha256()
            digest.update(code.co_code)
            digest.update(repr(code.co_consts).encode("utf-8", errors="replace"))
            digest.update(repr(code.co_names).encode("utf-8", errors="replace"))
            digest.update(repr(code.co_varnames).encode("utf-8", errors="replace"))
            payload["code_sha256"] = digest.hexdigest()
        defaults = getattr(target, "__defaults__", None)
        if defaults is not None:
            payload["defaults"] = cls._canonicalize(defaults)
        kwdefaults = getattr(target, "__kwdefaults__", None)
        if kwdefaults is not None:
            payload["kwdefaults"] = cls._canonicalize(kwdefaults)
        closure = getattr(target, "__closure__", None)
        if closure:
            payload["closure"] = [cls._canonicalize(cell.cell_contents) for cell in closure]
        return payload

    @classmethod
    def _canonicalize(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if math.isnan(value):
                return {"float": "nan"}
            if math.isinf(value):
                return {"float": "+inf" if value > 0 else "-inf"}
            if value == 0.0:
                return 0.0
            return value
        if isinstance(value, complex):
            return {"complex": [cls._canonicalize(value.real), cls._canonicalize(value.imag)]}
        if isinstance(value, Fraction):
            return {"fraction": [value.numerator, value.denominator]}
        if isinstance(value, bytes):
            return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
        if isinstance(value, Enum):
            return {"enum": f"{type(value).__module__}.{type(value).__qualname__}", "value": cls._canonicalize(value.value)}
        if callable(value):
            return cls._callable_identity(value)
        if isinstance(value, Mapping):
            items = [
                [cls._canonicalize(key), cls._canonicalize(item)]
                for key, item in value.items()
            ]
            items.sort(
                key=lambda pair: json.dumps(
                    pair[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
            )
            return {"mapping": items}
        if isinstance(value, (list, tuple)):
            return [cls._canonicalize(item) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [cls._canonicalize(item) for item in value]
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": cls._canonicalize(value.to_dict()),
            }
        if is_dataclass(value):
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": {field.name: cls._canonicalize(getattr(value, field.name)) for field in fields(value)},
            }
        if hasattr(value, "tolist") and callable(value.tolist):
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": cls._canonicalize(value.tolist()),
            }
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }

    @classmethod
    def stable_digest(cls, value: Any) -> str:
        encoded = json.dumps(
            cls._canonicalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def build_cache_identity(
        cls,
        *,
        domain: str,
        operation: str,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
        method: Optional[str] = None,
        precision: Any = None,
        tolerances: Any = None,
        algorithm_version: str = __version__,
        constants_version: Optional[str] = None,
        initial_conditions: Any = None,
        boundary_conditions: Any = None,
        seed: Optional[int] = None,
        config_digest: Optional[str] = None,
        template_digest: Optional[str] = None,
        extra: Any = None,
    ) -> Mapping[str, Any]:
        if not isinstance(domain, str) or not domain.strip():
            raise STEMValidationError("cache domain must be a non-empty string")
        if not isinstance(operation, str) or not operation.strip():
            raise STEMValidationError("cache operation must be a non-empty string")
        if kwargs is not None and not isinstance(kwargs, Mapping):
            raise STEMValidationError("cache kwargs must be a mapping")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise STEMValidationError("cache seed must be an integer or None")

        return {
            "schema": _KEY_SCHEMA,
            "domain": domain.strip().lower(),
            "operation": operation.strip(),
            "args": cls._canonicalize(tuple(args)),
            "kwargs": cls._canonicalize(dict(kwargs or {})),
            "method": method,
            "precision": cls._canonicalize(precision),
            "tolerances": cls._canonicalize(tolerances),
            "algorithm_version": str(algorithm_version),
            "constants_version": constants_version,
            "initial_conditions": cls._canonicalize(initial_conditions),
            "boundary_conditions": cls._canonicalize(boundary_conditions),
            "seed": seed,
            "config_digest": config_digest,
            "template_digest": template_digest,
            "extra": cls._canonicalize(extra),
        }

    @classmethod
    def build_cache_key(cls, **kwargs: Any) -> str:
        identity = cls.build_cache_identity(**kwargs)
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def build_key(
        *,
        operation: str,
        inputs: Any,
        method: Optional[str] = None,
        precision: Any = None,
        tolerances: Any = None,
        algorithm_version: str = __version__,
        constants_version: Optional[str] = None,
        boundary_conditions: Any = None,
        seed: Optional[int] = None,
        extra: Any = None,
    ) -> str:
        """Compatibility wrapper for the v2.3 pre-hardening cache-key API."""
        return STEMMemory.build_cache_key(
            domain="legacy",
            operation=operation,
            args=(inputs,),
            method=method,
            precision=precision,
            tolerances=tolerances,
            algorithm_version=algorithm_version,
            constants_version=constants_version,
            boundary_conditions=boundary_conditions,
            seed=seed,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Runtime cache
    # ------------------------------------------------------------------
    def _expiry_for(self, created_at: float, ttl_seconds: Optional[float]) -> Optional[float]:
        ttl = self.ttl_seconds if ttl_seconds is None else self._non_negative_float(ttl_seconds, "ttl_seconds")
        return None if ttl == 0.0 else created_at + ttl

    def _drop_if_expired_locked(self, key: str, now: Optional[float] = None) -> bool:
        entry = self._cache.get(key)
        if entry is None or not entry.expired(now):
            return False
        del self._cache[key]
        self._expirations += 1
        return True

    def lookup(self, key: str) -> Tuple[bool, Any]:
        if not isinstance(key, str) or not key:
            raise STEMValidationError("cache key must be a non-empty string")
        now = time.time()
        with self._lock:
            if self._drop_if_expired_locked(key, now):
                self._misses += 1
                return False, None
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return False, None
            self._cache.move_to_end(key)
            self._hits += 1
            return True, copy.deepcopy(entry.value)

    def get(self, key: str, default: Any = None) -> Any:
        found, value = self.lookup(key)
        return value if found else default

    def contains(self, key: str) -> bool:
        if not isinstance(key, str) or not key:
            return False
        with self._lock:
            self._drop_if_expired_locked(key)
            return key in self._cache

    def put(
        self,
        key: str,
        value: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        if not isinstance(key, str) or not key:
            raise STEMValidationError("cache key must be a non-empty string")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise STEMValidationError("cache metadata must be a mapping")
        created_at = time.time()
        entry = _CacheEntry(
            key=key,
            created_at=created_at,
            expires_at=self._expiry_for(created_at, ttl_seconds),
            value=copy.deepcopy(value),
            metadata=copy.deepcopy(dict(metadata or {})),
        )
        with self._lock:
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._writes += 1
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
                self._evictions += 1

    def metadata(self, key: str) -> Optional[Mapping[str, Any]]:
        if not isinstance(key, str) or not key:
            return None
        with self._lock:
            self._drop_if_expired_locked(key)
            entry = self._cache.get(key)
            return None if entry is None else copy.deepcopy(entry.metadata)

    def invalidate(self, key: str) -> bool:
        if not isinstance(key, str) or not key:
            raise STEMValidationError("cache key must be a non-empty string")
        with self._lock:
            removed = self._cache.pop(key, None)
            if removed is not None:
                self._invalidations += 1
                return True
            return False

    def prune(self) -> int:
        now = time.time()
        with self._lock:
            expired = [key for key, entry in self._cache.items() if entry.expired(now)]
            for key in expired:
                del self._cache[key]
            self._expirations += len(expired)
            return len(expired)

    def clear(self, *, reset_stats: bool = True) -> int:
        with self._lock:
            removed = len(self._cache)
            self._cache.clear()
            if reset_stats:
                self._hits = self._misses = self._writes = 0
                self._evictions = self._expirations = self._invalidations = 0
                self._checkpoint_skips = 0
            return removed

    def __len__(self) -> int:
        self.prune()
        with self._lock:
            return len(self._cache)

    def stats(self) -> Mapping[str, Any]:
        self.prune()
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total else 0.0,
                "writes": self._writes,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "invalidations": self._invalidations,
                "checkpoint_skips": self._checkpoint_skips,
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "algorithm_version": self.algorithm_version,
            }

    # ------------------------------------------------------------------
    # JSON-safe typed checkpoint state
    # ------------------------------------------------------------------
    @classmethod
    def _encode_float(cls, value: float) -> Any:
        if math.isfinite(value):
            return value
        return {_TYPE_MARKER: "float", "value": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}

    @classmethod
    def _decode_float(cls, payload: Mapping[str, Any]) -> float:
        marker = payload.get("value")
        if marker == "nan":
            return math.nan
        if marker == "+inf":
            return math.inf
        if marker == "-inf":
            return -math.inf
        raise STEMValidationError("Invalid serialized floating-point marker")

    @classmethod
    def _encode_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return cls._encode_float(value)
        if isinstance(value, complex):
            return {_TYPE_MARKER: "complex", "real": cls._encode_float(value.real), "imag": cls._encode_float(value.imag)}
        if isinstance(value, Fraction):
            return {_TYPE_MARKER: "fraction", "numerator": value.numerator, "denominator": value.denominator}
        if isinstance(value, Dimension):
            return {_TYPE_MARKER: "Dimension", "data": value.to_dict()}
        if isinstance(value, Unit):
            return {_TYPE_MARKER: "Unit", "data": value.to_dict()}
        if isinstance(value, PrecisionPolicy):
            return {
                _TYPE_MARKER: "PrecisionPolicy",
                "data": {
                    "significant_digits": value.significant_digits,
                    "decimal_places": value.decimal_places,
                    "rounding_mode": value.rounding_mode,
                    "absolute_tolerance": value.absolute_tolerance,
                    "relative_tolerance": value.relative_tolerance,
                    "nan_policy": value.nan_policy,
                },
            }
        if isinstance(value, UncertaintyValue):
            return {
                _TYPE_MARKER: "Uncertainty",
                "value": cls._encode_value(value.value),
                "coverage_factor": cls._encode_value(value.coverage_factor),
                "distribution": value.distribution.value,
                "uncertainty_type": value.uncertainty_type.value,
                "degrees_freedom": cls._encode_value(value.degrees_freedom),
                "unit": cls._encode_value(value.unit),
            }
        if isinstance(value, Quantity):
            return {
                _TYPE_MARKER: "Quantity",
                "magnitude": cls._encode_value(value.magnitude),
                "unit": cls._encode_value(value.unit),
                "uncertainty": cls._encode_value(value.uncertainty),
                "precision": cls._encode_value(value.precision),
            }
        if isinstance(value, NumericResult):
            return {
                _TYPE_MARKER: "NumericResult",
                "value": cls._encode_value(value.value),
                "unit": cls._encode_value(value.unit),
                "uncertainty": cls._encode_value(value.uncertainty),
                "method": value.method,
                "precision": cls._encode_value(value.precision),
                "absolute_error": cls._encode_value(value.absolute_error),
                "relative_error": cls._encode_value(value.relative_error),
                "residual": cls._encode_value(value.residual),
                "condition_estimate": cls._encode_value(value.condition_estimate),
                "warnings": cls._encode_value(tuple(value.warnings)),
                "metadata": cls._encode_value(dict(value.metadata)),
            }
        if isinstance(value, SolverResult):
            return {
                _TYPE_MARKER: "SolverResult",
                "solution": cls._encode_value(value.solution),
                "residual": cls._encode_value(value.residual),
                "iterations": value.iterations,
                "converged": value.converged,
                "status": value.status.value,
                "message": value.message,
                "diagnostics": cls._encode_value(dict(value.diagnostics)),
                "runtime": cls._encode_value(value.runtime),
            }
        if isinstance(value, PhysicalConstant):
            return {
                _TYPE_MARKER: "PhysicalConstant",
                "name": value.name,
                "symbol": value.symbol,
                "value": cls._encode_value(value.value),
                "unit": cls._encode_value(value.unit),
                "uncertainty": cls._encode_value(value.uncertainty),
                "exact": value.exact,
                "source": value.source,
            }
        if isinstance(value, Enum):
            return {_TYPE_MARKER: "enum-value", "value": cls._encode_value(value.value)}
        if isinstance(value, Mapping):
            return {
                _TYPE_MARKER: "mapping",
                "items": [
                    [cls._encode_value(key), cls._encode_value(item)]
                    for key, item in value.items()
                ],
            }
        if isinstance(value, tuple):
            return {_TYPE_MARKER: "tuple", "items": [cls._encode_value(item) for item in value]}
        if isinstance(value, list):
            return [cls._encode_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items = [cls._encode_value(item) for item in value]
            items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            return {_TYPE_MARKER: "set", "items": items}
        if callable(value):
            raise TypeError("callables are runtime-only cache values")
        if hasattr(value, "tolist") and callable(value.tolist):
            return {_TYPE_MARKER: "array", "data": cls._encode_value(value.tolist())}
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return {_TYPE_MARKER: "generic-to-dict", "data": cls._encode_value(value.to_dict())}
        raise TypeError(f"Unsupported durable STEM cache value: {type(value).__module__}.{type(value).__qualname__}")

    @classmethod
    def _decode_value(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._decode_value(item) for item in value]
        if not isinstance(value, Mapping):
            return value
        marker = value.get(_TYPE_MARKER)
        if marker is None:
            return {str(key): cls._decode_value(item) for key, item in value.items()}
        if marker == "float":
            return cls._decode_float(value)
        if marker == "complex":
            return complex(float(cls._decode_value(value["real"])), float(cls._decode_value(value["imag"])))
        if marker == "fraction":
            return Fraction(int(value["numerator"]), int(value["denominator"]))
        if marker == "Dimension":
            return Dimension.from_dict(value["data"])
        if marker == "Unit":
            return Unit.from_dict(value["data"])
        if marker == "PrecisionPolicy":
            return PrecisionPolicy(**dict(value["data"]))
        if marker == "Uncertainty":
            unit = cls._decode_value(value.get("unit"))
            if unit is not None and not isinstance(unit, Unit):
                raise STEMValidationError("Serialized Uncertainty unit is invalid")
            return UncertaintyValue(
                value=float(cls._decode_value(value.get("value"))),
                coverage_factor=float(cls._decode_value(value.get("coverage_factor", 1.0))),
                distribution=Distribution(value.get("distribution", Distribution.NORMAL.value)),
                uncertainty_type=UncertaintyType(value.get("uncertainty_type", UncertaintyType.TYPE_B.value)),
                degrees_freedom=cls._decode_value(value.get("degrees_freedom")),
                unit=unit,
            )
        if marker == "Quantity":
            unit = cls._decode_value(value.get("unit"))
            uncertainty = cls._decode_value(value.get("uncertainty"))
            precision = cls._decode_value(value.get("precision"))
            if not isinstance(unit, Unit):
                raise STEMValidationError("Serialized Quantity unit is invalid")
            return Quantity(
                magnitude=float(cls._decode_value(value.get("magnitude"))),
                unit=unit,
                uncertainty=uncertainty,
                precision=precision,
            )
        if marker == "NumericResult":
            unit = cls._decode_value(value.get("unit"))
            uncertainty = cls._decode_value(value.get("uncertainty"))
            precision = cls._decode_value(value.get("precision"))
            warnings = cls._decode_value(value.get("warnings", {_TYPE_MARKER: "tuple", "items": []}))
            metadata = cls._decode_value(value.get("metadata", {}))
            return NumericResult(
                value=cls._decode_value(value.get("value")),
                unit=unit,
                uncertainty=uncertainty,
                method=value.get("method"),
                precision=precision,
                absolute_error=cls._decode_value(value.get("absolute_error")),
                relative_error=cls._decode_value(value.get("relative_error")),
                residual=cls._decode_value(value.get("residual")),
                condition_estimate=cls._decode_value(value.get("condition_estimate")),
                warnings=tuple(warnings),
                metadata=metadata,
            )
        if marker == "SolverResult":
            return SolverResult(
                solution=cls._decode_value(value.get("solution")),
                residual=cls._decode_value(value.get("residual")),
                iterations=int(value.get("iterations", 0)),
                converged=bool(value.get("converged", False)),
                status=ConvergenceStatus(value.get("status", ConvergenceStatus.FAILED.value)),
                message=str(value.get("message", "")),
                diagnostics=cls._decode_value(value.get("diagnostics", {})),
                runtime=cls._decode_value(value.get("runtime")),
            )
        if marker == "PhysicalConstant":
            unit = cls._decode_value(value.get("unit"))
            uncertainty = cls._decode_value(value.get("uncertainty"))
            if not isinstance(unit, Unit):
                raise STEMValidationError("Serialized PhysicalConstant unit is invalid")
            return PhysicalConstant(
                name=str(value["name"]),
                symbol=str(value["symbol"]),
                value=float(cls._decode_value(value.get("value"))),
                unit=unit,
                uncertainty=uncertainty,
                exact=bool(value.get("exact", False)),
                source=value.get("source"),
            )
        if marker == "mapping":
            result: Dict[Any, Any] = {}
            for pair in value.get("items", []):
                if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray)) or len(pair) != 2:
                    raise STEMValidationError("Serialized mapping entry is invalid")
                key = cls._decode_value(pair[0])
                try:
                    result[key] = cls._decode_value(pair[1])
                except TypeError as exc:
                    raise STEMValidationError(
                        "Serialized mapping key is not hashable", context={"key_type": type(key).__name__}, cause=exc
                    ) from exc
            return result
        if marker == "tuple":
            return tuple(cls._decode_value(item) for item in value.get("items", []))
        if marker == "set":
            return set(cls._decode_value(item) for item in value.get("items", []))
        if marker in {"array", "generic-to-dict"}:
            return cls._decode_value(value.get("data"))
        if marker == "enum-value":
            return cls._decode_value(value.get("value"))
        raise STEMValidationError("Unknown serialized STEM cache type", context={"type": marker})

    def state_dict(self) -> Mapping[str, Any]:
        """Return checkpoint-safe state owned solely by this local memory."""
        self.prune()
        durable_entries = []
        skipped = 0
        with self._lock:
            for entry in self._cache.values():
                try:
                    encoded_value = self._encode_value(entry.value)
                    encoded_metadata = self._encode_value(entry.metadata)
                except (TypeError, STEMValidationError):
                    skipped += 1
                    continue
                durable_entries.append(
                    {
                        "key": entry.key,
                        "created_at": entry.created_at,
                        "expires_at": entry.expires_at,
                        "value": encoded_value,
                        "metadata": encoded_metadata,
                    }
                )
            self._checkpoint_skips += skipped
            return {
                "schema": _STATE_SCHEMA,
                "version": __version__,
                "algorithm_version": self.algorithm_version,
                "created_at": time.time(),
                "entries": durable_entries,
                "stats": {
                    "hits": self._hits,
                    "misses": self._misses,
                    "writes": self._writes,
                    "evictions": self._evictions,
                    "expirations": self._expirations,
                    "invalidations": self._invalidations,
                    "checkpoint_skips": self._checkpoint_skips,
                },
            }

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True, merge: bool = False) -> int:
        """Restore validated local cache state without changing current limits.

        Saved capacity/TTL are intentionally not authoritative.  The active
        process keeps its current configuration and applies the stricter current
        TTL to restored entries, preventing stale cache reuse after deployment
        configuration changes.
        """
        if not isinstance(state, Mapping):
            raise STEMValidationError("STEM memory state must be a mapping")
        if state.get("schema") != _STATE_SCHEMA:
            raise STEMValidationError(
                "Incompatible STEM memory state schema",
                context={"actual": state.get("schema"), "expected": _STATE_SCHEMA},
            )
        saved_version = str(state.get("version", ""))
        if strict and saved_version != __version__:
            raise STEMValidationError(
                "STEM memory state version is incompatible",
                context={"actual": saved_version, "expected": __version__},
            )
        saved_algorithm = str(state.get("algorithm_version", ""))
        if strict and saved_algorithm != self.algorithm_version:
            raise STEMValidationError(
                "STEM memory algorithm version is incompatible",
                context={"actual": saved_algorithm, "expected": self.algorithm_version},
            )
        entries = state.get("entries", [])
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
            raise STEMValidationError("STEM memory entries must be a sequence")

        now = time.time()
        restored: OrderedDict[str, _CacheEntry] = OrderedDict()
        for raw in entries:
            if not isinstance(raw, Mapping):
                if strict:
                    raise STEMValidationError("Invalid STEM memory entry payload")
                continue
            key = raw.get("key")
            if not isinstance(key, str) or not key:
                if strict:
                    raise STEMValidationError("Restored STEM memory entry has invalid key")
                continue
            try:
                created_at = float(raw.get("created_at", now))
                saved_expiry = raw.get("expires_at")
                expires_at = None if saved_expiry is None else float(saved_expiry)
                if not math.isfinite(created_at) or (expires_at is not None and not math.isfinite(expires_at)):
                    raise ValueError("non-finite timestamp")
                if self.ttl_seconds > 0.0:
                    current_policy_expiry = created_at + self.ttl_seconds
                    expires_at = current_policy_expiry if expires_at is None else min(expires_at, current_policy_expiry)
                if expires_at is not None and expires_at <= now:
                    continue
                value = self._decode_value(raw.get("value"))
                metadata = self._decode_value(raw.get("metadata", {}))
                if not isinstance(metadata, Mapping):
                    raise STEMValidationError("Restored STEM cache metadata must be a mapping")
                entry_algorithm = metadata.get("algorithm_version")
                if entry_algorithm is not None and str(entry_algorithm) != self.algorithm_version:
                    continue
                restored[key] = _CacheEntry(key, created_at, expires_at, value, dict(metadata))
            except (TypeError, ValueError, KeyError, STEMValidationError) as exc:
                if strict:
                    raise STEMValidationError("Failed to restore STEM memory entry", context={"key": key}, cause=exc) from exc

        with self._lock:
            if not merge:
                self._cache.clear()
            for key, entry in restored.items():
                self._cache[key] = entry
                self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
                self._evictions += 1

            stats = state.get("stats", {})
            if isinstance(stats, Mapping):
                for attr, key_name in (
                    ("_hits", "hits"), ("_misses", "misses"), ("_writes", "writes"),
                    ("_evictions", "evictions"), ("_expirations", "expirations"),
                    ("_invalidations", "invalidations"), ("_checkpoint_skips", "checkpoint_skips"),
                ):
                    raw_value = stats.get(key_name, getattr(self, attr))
                    try:
                        parsed = max(0, int(raw_value))
                    except (TypeError, ValueError):
                        parsed = getattr(self, attr)
                    setattr(self, attr, parsed)
            return len(restored)

    def serialize_state(self) -> Mapping[str, Any]:
        return self.state_dict()

    def restore_state(self, state: Mapping[str, Any], *, strict: bool = True, merge: bool = False) -> int:
        return self.load_state_dict(state, strict=strict, merge=merge)

    def snapshot(self) -> Mapping[str, Any]:
        return self.state_dict()

    def restore(self, state: Mapping[str, Any], *, strict: bool = True, merge: bool = False) -> int:
        return self.load_state_dict(state, strict=strict, merge=merge)


__all__ = ["STEMMemory"]


if __name__ == "__main__":
    configure_logging()
    printer.status("TEST", "STEMMemory deterministic cache validation", "info")

    memory = STEMMemory({"max_entries": 8, "ttl_seconds": 0.05})
    base = dict(domain="numerical_methods", operation="bisection", args=("x^2-4",), method="bisection")
    key_a = memory.build_cache_key(**base, tolerances={"tol": 1e-9}, seed=7)
    key_b = memory.build_cache_key(**base, tolerances={"tol": 1e-12}, seed=7)
    key_c = memory.build_cache_key(**base, tolerances={"tol": 1e-9}, seed=8)
    key_d = memory.build_cache_key(**base, tolerances={"tol": 1e-9}, seed=7, extra={"unit": "m"})
    assert len({key_a, key_b, key_c, key_d}) == 4

    metre = Unit("m", "metre", Dimension({"L": 1.0}))
    result = NumericResult(value=2.0, unit=metre, method="bisection", residual=0.0)
    memory.put(key_a, result, metadata={"algorithm_version": memory.algorithm_version})
    found, cached = memory.lookup(key_a)
    assert found and isinstance(cached, NumericResult) and cached.value == 2.0

    state = memory.state_dict()
    restored = STEMMemory({"max_entries": 8, "ttl_seconds": 0.05})
    assert restored.load_state_dict(state) == 1
    assert isinstance(restored.get(key_a), NumericResult)
    assert restored.invalidate(key_a)

    ttl_key = memory.build_cache_key(domain="units", operation="convert", args=(1.0, "m", "cm"))
    memory.put(ttl_key, 100.0, ttl_seconds=0.01)
    time.sleep(0.02)
    assert memory.get(ttl_key) is None
    assert memory.stats()["expirations"] >= 1

    printer.status("SUCCESS", "STEMMemory cache, identity, TTL, and restore checks passed", "success")
