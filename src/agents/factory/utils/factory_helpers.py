"""
Shared helper utilities for the factory and agent-orchestration subsystem.

This module is intentionally standalone from ``factory_errors.py`` in the
opposite dependency direction: helpers may import the factory error contract,
but the error module must never import this helper module. There are no lazy
imports here. All error classes used by helper functions are imported at module
load time so dependency issues are visible immediately during tests and startup.

Responsibilities covered here:

- timing helpers used by cache and observability modules;
- safe serialisation and redaction for diagnostics;
- validation helpers for metadata, registry state, dependency graphs, metrics,
  adaptation output, remote-worker payloads, cache configuration, and
  observability events;
- wrapping helpers that convert low-level exceptions into factory-domain
  exceptions;
- convenience constructors for common orchestration failure paths.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import time as time_module
import traceback

from datetime import date, datetime, time as datetime_time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

from .config_loader import get_config_section, load_global_config
from .factory_errors import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Factory Helpers")
printer = PrettyPrinter()


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

SENSITIVE_KEY_PATTERNS = ("authorization", "access_token", "refresh_token", "id_token", "api_key",
                          "apikey", "token", "secret", "password", "passwd", "credential", "csrf",
                          "cookie", "session", "xsrf", "bearer", "private_key", "client_secret",
                          "stdin", "stdout", "stderr", "payload",)

DEFAULT_MAX_STRING_LENGTH = 2_000
DEFAULT_MAX_SEQUENCE_LENGTH = 50
DEFAULT_MAX_MAPPING_LENGTH = 80
DEFAULT_MAX_SERIALIZATION_DEPTH = 6
REDACTION_PLACEHOLDER = "[REDACTED]"

_AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_MODULE_SEGMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CLASS_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9_.-]+)?$")
_OBSERVABILITY_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")


def now_epoch_seconds() -> float:
    """Return wall-clock epoch timestamp in seconds."""
    return time_module.time()


def monotonic_ms() -> float:
    """Return monotonic clock time in milliseconds."""
    return time_module.perf_counter() * 1000.0


def _truncate(text: Any, max_length: int = DEFAULT_MAX_STRING_LENGTH) -> str:
    value = str(text)
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}…[truncated {len(value) - max_length} chars]"


def _safe_repr(value: Any, max_length: int = DEFAULT_MAX_STRING_LENGTH) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    return _truncate(text, max_length)


def _is_sensitive_key(key: Any) -> bool:
    key_text = str(key).lower()
    return any(pattern in key_text for pattern in SENSITIVE_KEY_PATTERNS)


def safe_serialize(
    value: Any,
    *,
    redact: bool = True,
    depth: int = 0,
    max_depth: int = DEFAULT_MAX_SERIALIZATION_DEPTH,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    max_mapping_length: int = DEFAULT_MAX_MAPPING_LENGTH,
) -> Any:
    """Convert arbitrary values into JSON-safe primitives for diagnostics."""
    try:
        if depth >= max_depth:
            return _safe_repr(value, max_string_length)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return _truncate(value, max_string_length)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (datetime, date, datetime_time)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return f"<bytes length={len(value)}>"
        if isinstance(value, BaseException):
            return {"type": value.__class__.__name__, "message": _truncate(str(value), max_string_length)}
        if hasattr(value, "item") and callable(value.item):
            try:
                return safe_serialize(value.item(), redact=redact, depth=depth + 1)
            except Exception:
                pass
        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return safe_serialize(value.tolist(), redact=redact, depth=depth + 1)
            except Exception:
                pass
        if hasattr(value, "shape") or hasattr(value, "dtype"):
            snapshot: Dict[str, Any] = {"type": type(value).__name__}
            if hasattr(value, "shape"):
                try:
                    snapshot["shape"] = tuple(getattr(value, "shape"))
                except Exception:
                    snapshot["shape"] = _safe_repr(getattr(value, "shape", None))
            if hasattr(value, "dtype"):
                snapshot["dtype"] = str(getattr(value, "dtype", None))
            return snapshot
        if isinstance(value, Mapping):
            output: Dict[str, Any] = {}
            for index, (key, item_value) in enumerate(value.items()):
                if index >= max_mapping_length:
                    remaining = len(value) - max_mapping_length if hasattr(value, "__len__") else "unknown"
                    output["__truncated__"] = f"{remaining} additional keys omitted"
                    break
                key_text = str(key)
                if redact and _is_sensitive_key(key_text):
                    output[key_text] = REDACTION_PLACEHOLDER
                else:
                    output[key_text] = safe_serialize(
                        item_value,
                        redact=redact,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_string_length=max_string_length,
                        max_sequence_length=max_sequence_length,
                        max_mapping_length=max_mapping_length,
                    )
            return output
        if isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
            payload = [
                safe_serialize(
                    item,
                    redact=redact,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_string_length=max_string_length,
                    max_sequence_length=max_sequence_length,
                    max_mapping_length=max_mapping_length,
                )
                for item in items[:max_sequence_length]
            ]
            if len(items) > max_sequence_length:
                payload.append(f"…[{len(items) - max_sequence_length} additional items omitted]")
            return payload
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return safe_serialize(value.to_dict(), redact=redact, depth=depth + 1)
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return safe_serialize(vars(value), redact=redact, depth=depth + 1)
            except Exception:
                pass
        return _safe_repr(value, max_string_length)
    except Exception as exc:
        return f"<unserialisable {type(value).__name__}: {type(exc).__name__}>"


def sanitize_context(context: Optional[Mapping[str, Any]], *, redact: bool = True) -> Dict[str, Any]:
    """Return a JSON-safe context dictionary."""
    if not context:
        return {}
    serialised = safe_serialize(dict(context), redact=redact)
    return serialised if isinstance(serialised, dict) else {"value": serialised}


def build_fingerprint(code: str, error_type: str, message: str, category: Union[str, Dict[str, Any]]) -> str:
    """Build a stable short fingerprint for error grouping."""
    source = json.dumps(
        {"code": code, "type": error_type, "message": message, "category": category},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def exception_cause_payload(
    cause: Optional[BaseException],
    *,
    include_traceback: bool = False,
    redact: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return a JSON-safe representation of an exception cause."""
    if cause is None:
        return None
    payload: Dict[str, Any] = {
        "type": cause.__class__.__name__,
        "message": safe_serialize(str(cause), redact=redact),
    }
    if include_traceback:
        payload["traceback"] = _truncate(
            "".join(traceback.format_exception(type(cause), cause, cause.__traceback__)),
            10_000,
        )
    return payload


def normalize_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a shallow-copied payload dict, defaulting to an empty dict."""
    if not payload:
        return {}
    if not isinstance(payload, Mapping):
        raise FactoryTypeError(
            "payload must be a mapping",
            context={"actual_type": type(payload).__name__},
        )
    return dict(payload)


def ensure_positive_int(value: int, field_name: str) -> int:
    """Validate that an integer is strictly positive."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise FactoryTypeError(
            f"{field_name} must be an integer",
            context={"field": field_name, "actual_type": type(value).__name__},
        )
    if value <= 0:
        raise InvalidFieldValueError(
            f"{field_name} must be > 0",
            context={"field": field_name, "value": value},
        )
    return value


def ensure_positive_optional_float(value: Optional[float], field_name: str) -> Optional[float]:
    """Validate that an optional float value is positive when provided."""
    if value is None:
        return None
    number = coerce_number(value, field_name=field_name)
    if number <= 0:
        raise InvalidFieldValueError(
            f"{field_name} must be > 0 when provided",
            context={"field": field_name, "value": value},
        )
    return number


def ensure_non_negative_number(value: float, field_name: str) -> float:
    """Validate that a number is non-negative."""
    number = coerce_number(value, field_name=field_name)
    if number < 0:
        raise InvalidFieldValueError(
            f"{field_name} must be >= 0",
            context={"field": field_name, "value": value},
        )
    return number


def coerce_number(value: Any, *, field_name: str) -> float:
    """Coerce a numeric/scalar-like value into float."""
    try:
        if hasattr(value, "item") and callable(value.item):
            # The item() method may return a scalar (int, float, np.generic, etc.)
            item_val = cast(Any, value.item())
            return float(item_val)
        return float(value)
    except Exception as exc:
        raise FactoryTypeError(
            f"{field_name} must be numeric or scalar-convertible",
            context={"field": field_name, "actual_type": type(value).__name__},
            cause=exc,
        ) from exc


def require_mapping(value: Any, field_name: str, *, allow_empty: bool = True) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactoryTypeError(
            f"{field_name} must be a mapping",
            context={"field": field_name, "actual_type": type(value).__name__},
        )
    if not allow_empty and not value:
        raise EmptyCollectionError(f"{field_name} must not be empty", context={"field": field_name})
    return value


def require_mutable_mapping(value: Any, field_name: str, *, allow_empty: bool = True) -> MutableMapping[str, Any]:
    if not isinstance(value, MutableMapping):
        raise FactoryTypeError(
            f"{field_name} must be a mutable mapping",
            context={"field": field_name, "actual_type": type(value).__name__},
        )
    if not allow_empty and not value:
        raise EmptyCollectionError(f"{field_name} must not be empty", context={"field": field_name})
    return value


def require_sequence(value: Any, field_name: str, *, allow_empty: bool = True) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise FactoryTypeError(
            f"{field_name} must be a sequence",
            context={"field": field_name, "actual_type": type(value).__name__},
        )
    if not allow_empty and len(value) == 0:
        raise EmptyCollectionError(f"{field_name} must not be empty", context={"field": field_name})
    return value


def require_non_empty_string(value: Any, field_name: str, *, max_length: Optional[int] = None) -> str:
    if not isinstance(value, str):
        raise FactoryTypeError(
            f"{field_name} must be a string",
            context={"field": field_name, "actual_type": type(value).__name__},
        )
    text = value.strip()
    if not text:
        raise MissingRequiredFieldError(f"{field_name} must not be empty", context={"field": field_name})
    if max_length is not None and len(text) > max_length:
        raise InvalidFieldValueError(
            f"{field_name} exceeds maximum length of {max_length}",
            context={"field": field_name, "max_length": max_length, "actual_length": len(text)},
        )
    return text


def require_required_keys(payload: Mapping[str, Any], required_keys: Iterable[str], *, payload_name: str = "payload") -> None:
    missing = [key for key in required_keys if key not in payload or payload.get(key) in (None, "")]
    if missing:
        raise MissingRequiredFieldError(
            f"{payload_name} is missing required keys: {', '.join(missing)}",
            context={"payload": payload_name, "missing_keys": missing, "available_keys": sorted(str(k) for k in payload.keys())},
        )


def validate_agent_name(name: Any, *, max_length: int = 64, allow_dotted: bool = True) -> str:
    text = require_non_empty_string(name, "name", max_length=max_length)
    if not allow_dotted and "." in text:
        raise InvalidAgentNameError("Agent name must not contain dots", context={"name": text})
    if not _AGENT_NAME_PATTERN.match(text):
        raise InvalidAgentNameError(
            "Agent name must start with a letter and contain only letters, digits, underscores, hyphens, or dots",
            context={"name": text},
        )
    return text


def validate_class_name(class_name: Any) -> str:
    text = require_non_empty_string(class_name, "class_name", max_length=256)
    if not _CLASS_NAME_PATTERN.match(text):
        raise InvalidClassNameError("class_name must be a valid Python class identifier", context={"class_name": text})
    return text


def validate_module_path(module_path: Any, *, allowed_prefixes: Optional[Iterable[str]] = None) -> str:
    text = require_non_empty_string(module_path, "module_path", max_length=1_000)
    parts = text.split(".")
    if any(not part or not _MODULE_SEGMENT_PATTERN.match(part) for part in parts):
        raise InvalidModulePathError("module_path must be a dotted Python import path", context={"module_path": text})
    prefixes = tuple(prefix for prefix in (allowed_prefixes or ()) if prefix)
    if prefixes and not any(text.startswith(prefix) for prefix in prefixes):
        raise ModulePolicyViolationError(
            "module_path is outside the configured factory allowlist",
            context={"module_path": text, "allowed_prefixes": prefixes},
        )
    return text


def validate_version(version: Any, *, strict_semver: bool = False) -> str:
    text = require_non_empty_string(version, "version", max_length=128)
    if strict_semver and not _SEMVER_PATTERN.match(text):
        raise InvalidVersionError(
            "version must follow semantic version format: MAJOR.MINOR.PATCH",
            context={"version": text},
        )
    return text


def validate_required_params(required_params: Any) -> Tuple[str, ...]:
    if required_params in (None, ""):
        return tuple()
    if isinstance(required_params, tuple) and all(isinstance(item, str) and item.strip() for item in required_params):
        return tuple(item.strip() for item in required_params)
    sequence = require_sequence(required_params, "required_params", allow_empty=True)
    cleaned = []
    for index, item in enumerate(sequence):
        try:
            cleaned.append(require_non_empty_string(item, f"required_params[{index}]"))
        except FactoryError as exc:
            raise InvalidFieldValueError(
                "required_params must contain non-empty parameter names",
                context={"index": index, "value": item},
                cause=exc,
            ) from exc
    return tuple(cleaned)


def validate_dependency_names(dependencies: Any, *, agent_name: Optional[str] = None) -> Tuple[str, ...]:
    if dependencies in (None, ""):
        return tuple()
    sequence = require_sequence(dependencies, "dependencies", allow_empty=True)
    cleaned = []
    for index, dependency in enumerate(sequence):
        dep_name = require_non_empty_string(dependency, f"dependencies[{index}]", max_length=128)
        if agent_name and dep_name == agent_name:
            raise InvalidDependencySpecError(
                "Agent cannot depend on itself",
                context={"agent_name": agent_name, "dependency": dep_name},
            )
        cleaned.append(dep_name)
    return tuple(cleaned)


def validate_agent_metadata_dict(
    data: Any,
    *,
    required_fields: Iterable[str] = ("name", "class_name", "module_path"),
    allowed_module_prefixes: Optional[Iterable[str]] = None,
    max_name_length: int = 64,
    strict_version: bool = False,
) -> Dict[str, Any]:
    payload = dict(require_mapping(data, "agent_metadata", allow_empty=False))
    require_required_keys(payload, required_fields, payload_name="agent_metadata")
    payload["name"] = validate_agent_name(payload.get("name"), max_length=max_name_length)
    payload["class_name"] = validate_class_name(payload.get("class_name"))
    payload["module_path"] = validate_module_path(payload.get("module_path"), allowed_prefixes=allowed_module_prefixes)
    if payload.get("version") not in (None, ""):
        payload["version"] = validate_version(payload["version"], strict_semver=strict_version)
    if payload.get("required_params") is not None:
        payload["required_params"] = validate_required_params(payload.get("required_params"))
    if payload.get("dependencies") is not None:
        payload["dependencies"] = list(validate_dependency_names(payload.get("dependencies"), agent_name=payload["name"]))
    return payload


def validate_registry_object(registry: Any) -> Any:
    if registry is None:
        raise FactoryStateError("Registry is not initialised")
    if not hasattr(registry, "agents"):
        raise FactoryStateError("Registry must expose an agents mapping", context={"registry_type": type(registry).__name__})
    if not isinstance(getattr(registry, "agents"), Mapping):
        raise FactoryStateError("registry.agents must be a mapping", context={"agents_type": type(getattr(registry, "agents")).__name__})
    return registry


def ensure_agent_registered(registry: Any, agent_name: str) -> None:
    registry = validate_registry_object(registry)
    name = validate_agent_name(agent_name)
    if name not in registry.agents:
        raise AgentNotRegisteredError(
            f"Agent '{name}' is not registered",
            context={"agent_name": name, "registered_agents": sorted(str(k) for k in registry.agents.keys())},
        )


def ensure_agent_not_registered(registry: Any, agent_name: str) -> None:
    registry = validate_registry_object(registry)
    name = validate_agent_name(agent_name)
    if name in registry.agents:
        raise DuplicateAgentRegistrationError(f"Agent '{name}' is already registered", context={"agent_name": name})


def validate_dependency_graph(graph: Any, *, known_agents: Optional[Iterable[str]] = None) -> Mapping[str, Iterable[str]]:
    dependency_graph = require_mapping(graph, "dependency_graph", allow_empty=True)
    known = set(str(name) for name in known_agents or dependency_graph.keys())
    visiting = set()
    visited = set()
    path: list[str] = []

    def _visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle_start = path.index(node) if node in path else 0
            cycle = path[cycle_start:] + [node]
            raise CircularDependencyError("Circular dependency detected", context={"cycle": cycle})
        visiting.add(node)
        path.append(node)
        for dependency in dependency_graph.get(node, ()) or ():
            dep_name = require_non_empty_string(dependency, "dependency")
            if known and dep_name not in known:
                raise MissingDependencyError(
                    "Dependency is not registered or known",
                    context={"agent_name": node, "dependency": dep_name, "known_agents": sorted(known)},
                )
            if dep_name in dependency_graph:
                _visit(dep_name)
        visiting.remove(node)
        visited.add(node)
        path.pop()

    for candidate in dependency_graph.keys():
        _visit(str(candidate))
    return dependency_graph


def validate_metric_payload(metrics: Any) -> Mapping[str, Any]:
    payload = require_mapping(metrics, "metrics", allow_empty=False)
    for metric_name in ("fairness", "performance", "bias"):
        if metric_name in payload and payload[metric_name] is not None and not isinstance(payload[metric_name], Mapping):
            raise MetricsValidationError(
                f"Metric '{metric_name}' must be a mapping when provided",
                context={"metric": metric_name, "actual_type": type(payload[metric_name]).__name__},
            )
    return payload


def validate_agent_types(agent_types: Any) -> Tuple[str, ...]:
    sequence = require_sequence(agent_types, "agent_types", allow_empty=False)
    return tuple(require_non_empty_string(agent_type, f"agent_types[{index}]", max_length=128) for index, agent_type in enumerate(sequence))


def validate_adjustment_map(adjustments: Any) -> Mapping[str, Any]:
    payload = require_mapping(adjustments, "adjustments", allow_empty=True)
    for key, value in payload.items():
        if not str(key).endswith("_adjustment"):
            raise MetricsValidationError("Adjustment keys should use the *_adjustment naming convention", context={"key": key})
        coerce_number(value, field_name=str(key))
    return payload


def validate_safety_bounds(bounds: Any) -> Mapping[str, float]:
    payload = require_mapping(bounds, "safety_bounds", allow_empty=False)
    cleaned: Dict[str, float] = {}
    for key, value in payload.items():
        bound = coerce_number(value, field_name=f"safety_bounds[{key}]")
        if bound < 0:
            raise SafetyBoundError("Safety bounds must be non-negative", context={"agent_type": key, "bound": bound})
        cleaned[str(key)] = bound
    return cleaned


def validate_factory_object(factory: Any) -> Any:
    if factory is None:
        raise FactoryStateError("Factory is not initialised")
    if not hasattr(factory, "registry"):
        raise FactoryStateError("Factory must expose a registry", context={"factory_type": type(factory).__name__})
    validate_registry_object(factory.registry)
    return factory


def validate_remote_worker_payload(payload: Any) -> Dict[str, Any]:
    data = dict(require_mapping(payload, "remote_worker_payload", allow_empty=False))
    require_required_keys(data, ("module_path", "class_name", "method"), payload_name="remote_worker_payload")
    data["module_path"] = validate_module_path(data["module_path"])
    data["class_name"] = validate_class_name(data["class_name"])
    data["method"] = require_non_empty_string(data["method"], "method", max_length=256)
    if data.get("args") is not None and not isinstance(data.get("args"), (list, tuple)):
        raise RemoteWorkerPayloadError("remote_worker_payload.args must be a list or tuple", context={"actual_type": type(data.get("args")).__name__})
    if data.get("kwargs") is not None and not isinstance(data.get("kwargs"), Mapping):
        raise RemoteWorkerPayloadError("remote_worker_payload.kwargs must be a mapping", context={"actual_type": type(data.get("kwargs")).__name__})
    return data


def validate_worker_response(response: Any) -> Mapping[str, Any]:
    payload = require_mapping(response, "remote_worker_response", allow_empty=False)
    status = payload.get("status")
    if status not in {"ok", "error", "degraded"}:
        raise RemoteWorkerResultError("Remote worker response status is invalid", context={"status": status, "available_keys": sorted(str(key) for key in payload.keys())})
    if status == "ok" and "result" not in payload:
        raise RemoteWorkerResultError("Remote worker ok response is missing result", context={"response": payload})
    if status in {"error", "degraded"} and not any(key in payload for key in ("error", "message")):
        raise RemoteWorkerResultError("Remote worker failure response is missing error/message", context={"response": payload})
    return payload


def validate_cache_config(max_size: Any, default_ttl_seconds: Optional[Any] = None) -> Tuple[int, Optional[float]]:
    size = ensure_positive_int(max_size, "max_size")
    ttl = ensure_positive_optional_float(default_ttl_seconds, "default_ttl_seconds")
    return size, ttl


def validate_cache_key(key: Any, *, field_name: str = "cache_key") -> Any:
    try:
        hash(key)
    except Exception as exc:
        raise CacheKeyError(
            f"{field_name} must be hashable",
            context={"field": field_name, "actual_type": type(key).__name__},
            cause=exc,
        ) from exc
    return key


def validate_cache_ttl(ttl_seconds: Optional[Any], *, field_name: str = "ttl_seconds") -> Optional[float]:
    return ensure_positive_optional_float(ttl_seconds, field_name)


def validate_cache_stats(stats: Any) -> Mapping[str, Any]:
    payload = require_mapping(stats, "cache_stats", allow_empty=False)
    required = ("hits", "misses", "sets", "evictions", "expirations", "size", "max_size")
    require_required_keys(payload, required, payload_name="cache_stats")
    for field in required:
        ensure_non_negative_number(payload[field], f"cache_stats[{field}]")
    if coerce_number(payload["size"], field_name="cache_stats[size]") > coerce_number(payload["max_size"], field_name="cache_stats[max_size]"):
        raise CacheStatsError("cache_stats.size must not exceed cache_stats.max_size", context={"stats": payload})
    return payload


def validate_cache_capacity(current_size: Any, max_size: Any) -> Tuple[int, int]:
    current = int(ensure_non_negative_number(current_size, "current_size"))
    maximum = ensure_positive_int(max_size, "max_size")
    if current > maximum:
        raise CacheCapacityError("current_size must not exceed max_size", context={"current_size": current, "max_size": maximum})
    return current, maximum


def validate_observability_name(name: Any, *, field_name: str = "name") -> str:
    text = require_non_empty_string(name, field_name, max_length=256)
    if not _OBSERVABILITY_NAME_PATTERN.match(text):
        raise FactoryObservabilityError(
            f"{field_name} must start with a letter and contain only letters, digits, underscores, dots, colons, or hyphens",
            context={"field": field_name, "value": text},
        )
    return text


def validate_event_buffer_size(event_buffer_size: Any) -> int:
    try:
        return ensure_positive_int(event_buffer_size, "event_buffer_size")
    except FactoryError as exc:
        raise ObservabilityConfigurationError(
            "event_buffer_size must be a positive integer",
            context={"event_buffer_size": event_buffer_size},
            cause=exc,
        ) from exc


def validate_counter_increment(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CounterUpdateError("Counter increment value must be an integer", context={"actual_type": type(value).__name__})
    if value < 0:
        raise CounterUpdateError("Counter increment value must be >= 0", context={"value": value})
    return value


def validate_gauge_value(value: Any) -> float:
    try:
        return coerce_number(value, field_name="gauge_value")
    except FactoryError as exc:
        raise GaugeUpdateError("Gauge value must be numeric", context={"value": value}, cause=exc) from exc


def validate_timing_duration(duration_ms: Any) -> float:
    try:
        duration = coerce_number(duration_ms, field_name="duration_ms")
    except FactoryError as exc:
        raise TimingObservationError("duration_ms must be numeric", context={"duration_ms": duration_ms}, cause=exc) from exc
    if duration < 0:
        raise TimingObservationError("duration_ms must be >= 0", context={"duration_ms": duration})
    return duration


def validate_event_payload(event_type: Any, payload: Optional[Any] = None) -> Tuple[str, Dict[str, Any]]:
    name = validate_observability_name(event_type, field_name="event_type")
    if payload is None:
        return name, {}
    if not isinstance(payload, Mapping):
        raise EventRecordingError("event payload must be a mapping", context={"actual_type": type(payload).__name__})
    return name, dict(payload)


def validate_recent_event_limit(limit: Any) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise FactoryTypeError("recent event limit must be an integer", context={"actual_type": type(limit).__name__})
    if limit < 0:
        raise InvalidFieldValueError("recent event limit must be >= 0", context={"limit": limit})
    return limit


def map_exception_to_factory_error(exc: BaseException, *, operation: Optional[str] = None) -> Optional[Type[FactoryError]]:
    """Map common low-level exceptions to factory-specific error classes."""
    if isinstance(exc, FactoryError):
        return type(exc)
    if isinstance(exc, json.JSONDecodeError):
        return RemoteWorkerSerializationError
    if isinstance(exc, ModuleNotFoundError):
        return AgentModuleImportError
    if isinstance(exc, ImportError):
        return AgentModuleImportError
    if isinstance(exc, AttributeError):
        if operation and any(token in operation.lower() for token in ("invoke", "call", "method")):
            return AgentInvocationError
        return AgentClassResolutionError
    if isinstance(exc, KeyError):
        return AgentNotRegisteredError
    if isinstance(exc, TimeoutError):
        return FactoryTimeoutError
    if isinstance(exc, OSError):
        return FactoryIOError
    if isinstance(exc, TypeError):
        return FactoryTypeError
    if isinstance(exc, ValueError):
        return FactoryValidationError
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        if "torch" in message:
            return TorchUnavailableError
        return FactoryRuntimeError
    return None


def wrap_factory_exception(
    exc: BaseException,
    *,
    message: Optional[str] = None,
    operation: Optional[str] = None,
    component: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    details: Optional[Mapping[str, Any]] = None,
    default_error_cls: Optional[Type[FactoryError]] = None,
    retryable: Optional[bool] = None,
    severity: Optional[Union[str, FactoryErrorSeverity]] = None,
) -> FactoryError:
    """Convert any exception into a factory-domain exception."""
    if isinstance(exc, FactoryError):
        if context:
            exc.context.update(sanitize_context(context, redact=False))
        if details:
            exc.details.update(sanitize_context(details, redact=False))
        if operation and not exc.operation:
            exc.operation = operation
        if component and not exc.component:
            exc.component = component
        return exc

    error_cls = map_exception_to_factory_error(exc, operation=operation) or default_error_cls or FactoryRuntimeError
    return error_cls(
        message or str(exc) or getattr(error_cls, "default_message", "Factory operation failed"),
        operation=operation,
        component=component,
        context=context,
        details=details,
        cause=exc,
        retryable=retryable,
        severity=severity,
    )


def guard_factory_operation(
    *,
    error_cls: Type[FactoryError] = FactoryRuntimeError,
    operation: Optional[str] = None,
    component: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator that wraps arbitrary failures in a semantic factory error."""

    def _decorator(func: F) -> F:
        @functools.wraps(func)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except FactoryError:
                raise
            except Exception as exc:
                raise error_cls.from_exception(
                    exc,
                    operation=operation or getattr(func, "__name__", None),
                    component=component,
                    context=context,
                ) from exc

        return _wrapped  # type: ignore[return-value]

    return _decorator


def is_retryable_error(error: Union[FactoryError, BaseException, Mapping[str, Any]]) -> bool:
    """Return whether a factory error/result should be considered retryable."""
    if isinstance(error, FactoryError):
        return error.retryable
    if isinstance(error, Mapping):
        if "retryable" in error:
            return bool(error["retryable"])
        nested = error.get("error")
        if isinstance(nested, Mapping) and "retryable" in nested:
            return bool(nested["retryable"])
        return False
    return wrap_factory_exception(error).retryable


def missing_agent(agent_name: str, *, registry: Optional[Any] = None) -> AgentNotRegisteredError:
    context: Dict[str, Any] = {"agent_name": agent_name}
    if registry is not None and hasattr(registry, "agents"):
        context["registered_agents"] = sorted(str(key) for key in getattr(registry, "agents", {}).keys())
    return AgentNotRegisteredError(f"Agent '{agent_name}' is not registered", context=context)


def version_unavailable(agent_name: str, version: str, *, available_versions: Optional[Iterable[str]] = None) -> AgentVersionUnavailableError:
    return AgentVersionUnavailableError(
        f"Version {version} is not available for agent '{agent_name}'",
        context={"agent_name": agent_name, "requested_version": version, "available_versions": tuple(available_versions or ())},
    )


def dependency_cycle(cycle: Sequence[str]) -> CircularDependencyError:
    return CircularDependencyError("Agent dependency graph contains a cycle", context={"cycle": list(cycle)})


def module_policy_violation(module_path: str, *, allowed_prefixes: Iterable[str]) -> ModulePolicyViolationError:
    return ModulePolicyViolationError(
        "Agent module path violates configured allowlist",
        context={"module_path": module_path, "allowed_prefixes": tuple(allowed_prefixes)},
    )


def worker_degraded(agent_type: str, *, method: Optional[str] = None, error: Optional[Any] = None) -> DegradedAgentError:
    return DegradedAgentError(
        f"Agent '{agent_type}' is running in degraded mode",
        context={"agent_type": agent_type, "method": method, "worker_error": error},
        operation=method,
    )


def metrics_dependency_unavailable(cause: Optional[BaseException] = None) -> TorchUnavailableError:
    return TorchUnavailableError(cause=cause, context={"dependency": "torch"})


def cache_entry_expired(key: Any, *, expires_at: Optional[float] = None) -> CacheEntryExpiredError:
    return CacheEntryExpiredError("Factory cache entry expired", context={"key": safe_serialize(key), "expires_at": expires_at})


def cache_eviction_failed(key: Any, *, cause: Optional[BaseException] = None) -> CacheEvictionError:
    return CacheEvictionError("Factory cache eviction failed", context={"key": safe_serialize(key)}, cause=cause)


def observability_snapshot_failed(*, cause: Optional[BaseException] = None) -> ObservabilitySnapshotError:
    return ObservabilitySnapshotError("Factory observability snapshot failed", cause=cause)


def observability_reset_failed(*, cause: Optional[BaseException] = None) -> ObservabilityResetError:
    return ObservabilityResetError("Factory observability reset failed", cause=cause)
