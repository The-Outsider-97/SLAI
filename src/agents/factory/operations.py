"""Validated execution operations for SLAI specialized subagents.

This module owns the contract between the orchestrator-generated
specialist subagents and their executable behaviour. It provides:

- structural validation of executable specialist specifications;
- bounded, cache-assisted execution of the supported operations;
- structured error projection into the factory error taxonomy;
- configuration-driven resource limits.

Design notes
------------
- Validation is deterministic and pure, so validated specifications are
  memoised in a bounded :class:`FactoryCache`. This avoids redundant
  structural validation across repeated delegations to the same
  generated specialist without requiring callers to cache anything.
- Execution always validates first; the cache only short-circuits the
  *validation* step, never the payload inspection.
- Every failure is a :class:`FactoryError` subclass so the orchestrator
  and higher-level callers receive machine-actionable results rather
  than bare ``TypeError`` / ``ValueError``.
- All numeric bounds (input size, field count, field length) and cache
  parameters (size, TTL) come from ``factory_config.yaml`` under the
  ``specialist_operations`` section. Module-level defaults exist only as
  fallbacks for missing or malformed configuration.
"""

from __future__ import annotations

import math

from collections.abc import Mapping
from threading import RLock
from typing import Any, Optional

from .utils.config_loader import get_config_section
from .utils.factory_errors import *
from .utils.factory_helpers import *
from .factory_cache import FactoryCache
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Factory Operations")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_OPERATIONS = frozenset({
    "numeric_summary",
    "project_records",
})

#: Top-level configuration section name in ``factory_config.yaml``.
_CONFIG_SECTION = "specialist_operations"

#: Component label used in structured errors.
_COMPONENT = "factory_operations"

#: Fallback limits. These are used only when the configuration omits or
#: invalidates a key; the authoritative values live in the config file.
_DEFAULT_LIMITS: dict[str, Any] = {
    "max_items": 10_000,
    "max_fields": 64,
    "max_field_name_length": 128,
    "cache_max_size": 128,
    "cache_ttl_seconds": 300.0,
}


# ---------------------------------------------------------------------------
# Specification cache
#
# Validation is pure and deterministic, so caching the *validated* form of
# a specification is safe as long as the key covers the entire input. The
# cache is process-local, thread-safe, bounded, and TTL-controlled through
# the same FactoryCache used elsewhere in the factory package.
# ---------------------------------------------------------------------------
_spec_cache: Optional[FactoryCache] = None
_spec_cache_lock = RLock()


def _get_spec_cache() -> FactoryCache:
    """Return the process-local specialist-specification cache.

    The cache is created lazily so that importing this module has no
    side effects (no filesystem probes, no logging) until an operation
    is actually executed.
    """
    global _spec_cache
    with _spec_cache_lock:
        if _spec_cache is None:
            limits = _load_limits()
            _spec_cache = FactoryCache(
                max_size=int(limits["cache_max_size"]),
                default_ttl_seconds=float(limits["cache_ttl_seconds"]),
                name="specialist_spec_cache",
            )
        return _spec_cache


def clear_spec_cache() -> None:
    """Clear the specialist-specification cache.

    Exposed for tests and diagnostics. Not required on the normal
    execution path; the cache is bounded and self-expiring.
    """
    with _spec_cache_lock:
        if _spec_cache is not None:
            _spec_cache.clear()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _load_limits() -> dict[str, Any]:
    """Load operation limits from ``factory_config.yaml``.

    Invalid individual values fall back to module defaults rather than
    raising: a malformed *optional* limit must not disable the entire
    operations subsystem. A malformed section (non-mapping) degrades to
    defaults as well, and the reason is logged once per call.
    """
    try:
        section = get_config_section(_CONFIG_SECTION) or {}
    except Exception as exc:
        logger.warning(
            "Unable to load '%s' config section; using defaults: %s",
            _CONFIG_SECTION,
            exc,
        )
        section = {}

    if not isinstance(section, Mapping):
        logger.warning(
            "Config section '%s' is not a mapping; using defaults",
            _CONFIG_SECTION,
        )
        section = {}

    limits: dict[str, Any] = dict(_DEFAULT_LIMITS)
    for key, default in _DEFAULT_LIMITS.items():
        raw = section.get(key, default)
        try:
            parsed = float(raw) if isinstance(default, float) else int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid '%s.%s' value %r; using default %r",
                _CONFIG_SECTION,
                key,
                raw,
                default,
            )
            parsed = default
        if parsed <= 0:
            logger.warning(
                "'%s.%s' must be > 0; using default %r",
                _CONFIG_SECTION,
                key,
                default,
            )
            parsed = default
        limits[key] = parsed
    return limits


# ---------------------------------------------------------------------------
# Error constructors
#
# Thin, explicit constructors keep call sites readable and guarantee
# consistent ``component`` / ``operation`` annotation. Using the raw
# exception classes inline would allow the fields to drift between call
# sites.
# ---------------------------------------------------------------------------
def _spec_type_error(message: str, **context: Any) -> FactoryTypeError:
    return FactoryTypeError(
        message,
        component=_COMPONENT,
        operation="validate_specialist_spec",
        context=context,
    )


def _spec_value_error(message: str, **context: Any) -> FactoryValidationError:
    return FactoryValidationError(
        message,
        component=_COMPONENT,
        operation="validate_specialist_spec",
        context=context,
    )


def _exec_type_error(message: str, **context: Any) -> FactoryTypeError:
    return FactoryTypeError(
        message,
        component=_COMPONENT,
        operation="execute_operation",
        context=context,
    )


def _exec_value_error(message: str, **context: Any) -> FactoryValidationError:
    return FactoryValidationError(
        message,
        component=_COMPONENT,
        operation="execute_operation",
        context=context,
    )


# ---------------------------------------------------------------------------
# Cache-key derivation and defensive copies
# ---------------------------------------------------------------------------
def _spec_cache_key(spec: Mapping[str, Any]) -> str:
    """Return a stable, bounded cache key for a *raw* specification.

    The key must not raise on malformed input because the cache lookup
    happens *before* validation. Malformed specs therefore receive a
    deterministic key that cannot collide with any valid spec key
    (valid keys always begin with a known operation name).

    Both the operation name and every field string are length-bounded
    so that a hostile payload cannot force an unbounded key.
    """
    try:
        operation = str(spec.get("operation", ""))[:64]
        raw_fields = spec.get("fields", ())
        if isinstance(raw_fields, (list, tuple)):
            field_text = ",".join(str(f)[:128] for f in raw_fields[:256])
        else:
            field_text = ""
        return f"spec|{operation}|{field_text}"
    except Exception:
        # Defensive: key derivation must never propagate.
        return "spec|invalid|"


def _copy_validated(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive, shallow-immutable copy of a validated spec.

    The cached value must not be exposed by reference, because callers
    could mutate the ``fields`` list and corrupt subsequent reads.
    """
    result: dict[str, Any] = {"operation": spec["operation"]}
    if "fields" in spec:
        result["fields"] = list(spec["fields"])
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_specialist_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalise an executable specialist specification.

    The specification is a mapping with exactly two accepted keys:

    * ``operation`` (required) — one of :data:`SUPPORTED_OPERATIONS`;
    * ``fields`` (optional, only for ``project_records``) — a
      non-empty sequence of unique field names.

    Validated specifications are memoised in a bounded TTL cache. The
    returned dictionary is always a fresh copy so callers cannot
    mutate cached state.

    Raises:
        FactoryTypeError: when the specification is not a mapping.
        FactoryValidationError: for any other structural violation.
    """
    if not isinstance(spec, Mapping):
        raise _spec_type_error(
            "specialist_spec must be a mapping",
            actual_type=type(spec).__name__,
        )

    cache = _get_spec_cache()
    cache_key = _spec_cache_key(spec)

    cached = cache.get(cache_key)
    if cached is not None:
        return _copy_validated(cached)

    validated = _validate_specialist_spec_uncached(spec)
    cache.set(cache_key, validated)
    return _copy_validated(validated)


def _validate_specialist_spec_uncached(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a specification without consulting the cache."""
    limits = _load_limits()

    operation = spec.get("operation")
    if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS:
        raise _spec_value_error(
            f"Unsupported specialist operation: {operation!r}",
            operation_repr=repr(operation)[:128],
            supported=sorted(SUPPORTED_OPERATIONS),
        )

    unknown = set(spec) - {"operation", "fields"}
    if unknown:
        raise _spec_value_error(
            "Unknown specialist specification fields",
            unknown=sorted(str(k) for k in unknown),
            allowed=["operation", "fields"],
        )

    if operation == "numeric_summary":
        if "fields" in spec:
            raise _spec_value_error(
                "numeric_summary does not accept fields",
                operation=operation,
            )
        return {"operation": operation}

    # ---- project_records --------------------------------------------
    fields = spec.get("fields")
    if isinstance(fields, (str, bytes)) or not isinstance(fields, (list, tuple)):
        raise _spec_value_error(
            "project_records requires a fields sequence",
            actual_type=type(fields).__name__,
        )

    max_fields = int(limits["max_fields"])
    if not 1 <= len(fields) <= max_fields:
        raise _spec_value_error(
            f"project_records requires 1 to {max_fields} fields",
            field_count=len(fields),
            max_fields=max_fields,
        )

    max_name_length = int(limits["max_field_name_length"])
    for index, field in enumerate(fields):
        if (
            not isinstance(field, str)
            or not field
            or len(field) > max_name_length
        ):
            raise _spec_value_error(
                "Invalid projection field name",
                index=index,
                field_repr=repr(field)[:128],
                max_length=max_name_length,
            )

    if len(set(fields)) != len(fields):
        raise _spec_value_error(
            "Duplicate projection fields",
            fields=list(fields),
        )

    return {"operation": operation, "fields": list(fields)}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def execute_operation(spec: Mapping[str, Any], payload: Any) -> dict[str, Any]:
    """Execute a validated specialist operation on bounded input.

    The specification is validated (cache-assisted) before any payload
    inspection. This preserves the module's existing public contract
    while guaranteeing that execution never runs against an unvalidated
    specification.
    """
    validated = validate_specialist_spec(spec)
    operation = validated["operation"]
    limits = _load_limits()

    if isinstance(payload, Mapping):
        key = "values" if operation == "numeric_summary" else "records"
        if key not in payload:
            raise _exec_value_error(
                f"Missing required payload key: {key}",
                operation=operation,
                required_key=key,
                available_keys=sorted(str(k) for k in payload.keys()),
            )
        data = payload[key]
    else:
        data = payload

    if not isinstance(data, (list, tuple)):
        raise _exec_type_error(
            "Specialist input must be a list or tuple",
            actual_type=type(data).__name__,
        )

    max_items = int(limits["max_items"])
    if len(data) > max_items:
        raise _exec_value_error(
            "Specialist input exceeds configured safety limit",
            item_count=len(data),
            max_items=max_items,
        )

    if operation == "numeric_summary":
        result = _execute_numeric_summary(data)
    elif operation == "project_records":
        result = _execute_project_records(data, validated["fields"])
    else:
        # Unreachable while SUPPORTED_OPERATIONS stays in sync with the
        # dispatch table; retained as a defensive invariant.
        raise _exec_value_error(
            f"Unsupported operation: {operation}",
            operation=operation,
        )

    return {
        "status": "success",
        "operation": operation,
        "result": result,
    }


def _execute_numeric_summary(data) -> dict[str, Any]:
    """Compute a numeric summary over a bounded sequence of finite numbers."""
    values: list[float] = []
    for index, item in enumerate(data):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _exec_type_error(
                "Only numeric values are supported",
                index=index,
                actual_type=type(item).__name__,
            )
        value = float(item)
        if not math.isfinite(value):
            raise _exec_value_error(
                "Non-finite numeric value",
                index=index,
                value_repr=repr(value),
            )
        values.append(value)

    if not values:
        raise _exec_value_error("At least one number is required")

    total = math.fsum(values)
    return {
        "count": len(values),
        "sum": total,
        "mean": total / len(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _execute_project_records(data, fields: list[str]) -> list[dict[str, Any]]:
    """Project a bounded sequence of mappings onto the declared fields."""
    result: list[dict[str, Any]] = []
    for index, record in enumerate(data):
        if not isinstance(record, Mapping):
            raise _exec_type_error(
                "Expected a mapping for each record",
                index=index,
                actual_type=type(record).__name__,
            )
        missing = [field for field in fields if field not in record]
        if missing:
            raise _exec_value_error(
                "Record is missing required fields",
                index=index,
                missing_fields=missing,
            )
        result.append({field: record[field] for field in fields})
    return result


__all__ = [
    "SUPPORTED_OPERATIONS",
    "validate_specialist_spec",
    "execute_operation",
    "clear_spec_cache",
]


if __name__ == "__main__":
    print("\n=== Running Factory Operations ===\n")
    printer.status("TEST", "Factory Operations initialized", "info")

    # ---- numeric_summary ------------------------------------------
    spec = {"operation": "numeric_summary"}
    out = execute_operation(spec, {"values": [1, 2, 3, 4]})
    assert out["status"] == "success", out
    assert out["result"]["count"] == 4
    assert out["result"]["sum"] == 10.0
    assert out["result"]["mean"] == 2.5

    # ---- project_records ------------------------------------------
    spec = {"operation": "project_records", "fields": ["a", "b"]}
    out = execute_operation(spec, {"records": [{"a": 1, "b": 2, "c": 3}]})
    assert out["status"] == "success", out
    assert out["result"] == [{"a": 1, "b": 2}], out

    # ---- cache hit path -------------------------------------------
    out_again = validate_specialist_spec(spec)
    assert out_again == spec, out_again
    # The returned dict must not alias the cached list.
    out_again["fields"].append("c")
    assert validate_specialist_spec(spec)["fields"] == ["a", "b"]

    # ---- structured error path ------------------------------------
    try:
        validate_specialist_spec({"operation": "unknown_op"})
    except FactoryValidationError as exc:
        assert exc.component == "factory_operations"
        assert exc.operation == "validate_specialist_spec"
    else:
        raise AssertionError("Expected FactoryValidationError")

    try:
        execute_operation({"operation": "numeric_summary"}, {"values": [1, "x"]})
    except FactoryTypeError as exc:
        assert exc.operation == "execute_operation"
    else:
        raise AssertionError("Expected FactoryTypeError")

    # ---- limits are config-driven ---------------------------------
    limits = _load_limits()
    assert "max_items" in limits and limits["max_items"] > 0

    clear_spec_cache()
    printer.status("TEST", "Factory Operations checks passed", "success")
    print("\n=== Test ran successfully ===\n")