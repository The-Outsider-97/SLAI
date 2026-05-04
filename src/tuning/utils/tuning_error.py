"""Production-ready error taxonomy and handling helpers for the tuning module.

This module provides a structured, serialization-safe error model for
hyperparameter search workflows (grid and Bayesian), including configuration
loading, search-space validation, model evaluation, persistence, and
reporting.

Design goals:
- keep the public API small and familiar
- preserve rich causal context for observability
- never fail while trying to serialize or log an error
- keep secrets and oversized payloads out of logs by default
- support progressive enrichment as errors move up the stack
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from traceback import format_exception
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, TypeVar, Union
from uuid import uuid4

JSONDict = Dict[str, Any]
T = TypeVar("T")

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_ITEMS = 50
DEFAULT_MAX_STRING_LENGTH = 2048
DEFAULT_TRACEBACK_LINE_LIMIT = 40
SENSITIVE_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "private_key",
        "client_secret",
        "connection_string",
        "cookie",
        "session",
        "signature",
    }
)


class TuningErrorCode(str, Enum):
    """Stable error codes for tuning pipeline failures."""

    CONFIG_ERROR = "TUNING_CONFIG_ERROR"
    VALIDATION_ERROR = "TUNING_VALIDATION_ERROR"
    SEARCH_SPACE_ERROR = "TUNING_SEARCH_SPACE_ERROR"
    STRATEGY_ERROR = "TUNING_STRATEGY_ERROR"
    EVALUATION_ERROR = "TUNING_EVALUATION_ERROR"
    OPTIMIZATION_ERROR = "TUNING_OPTIMIZATION_ERROR"
    PERSISTENCE_ERROR = "TUNING_PERSISTENCE_ERROR"
    REPORTING_ERROR = "TUNING_REPORTING_ERROR"
    DEPENDENCY_ERROR = "TUNING_DEPENDENCY_ERROR"
    INTERNAL_ERROR = "TUNING_INTERNAL_ERROR"


class TuningSeverity(str, Enum):
    """Severity hints for logging, alerting, and orchestration."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True, frozen=True)
class TuningErrorContext:
    """Execution context attached to tuning failures.

    The object is intentionally immutable to make it safe to pass across layers
    without accidental mutation. Unknown fields from mappings are folded into
    ``metadata`` instead of being dropped.
    """

    component: Optional[str] = None
    operation: Optional[str] = None
    strategy: Optional[str] = None
    model_type: Optional[str] = None
    fold_index: Optional[int] = None
    iteration: Optional[int] = None
    n_calls: Optional[int] = None
    random_state: Optional[int] = None
    config_path: Optional[str] = None
    output_path: Optional[str] = None
    dataset_shape: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TuningErrorContext":
        if not isinstance(data, Mapping):
            return cls(metadata={"invalid_context": safe_serialize(data)})

        valid_fields = {field_name for field_name in cls.__dataclass_fields__.keys()}
        payload: Dict[str, Any] = {}
        raw_metadata = data.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {"metadata_value": safe_serialize(raw_metadata)}

        for key, value in data.items():
            if key == "metadata":
                continue
            if key in valid_fields:
                payload[key] = value
            else:
                metadata[key] = value

        payload["metadata"] = metadata
        return cls(**payload)

    def merge(self, other: Optional[Union["TuningErrorContext", Mapping[str, Any]]]) -> "TuningErrorContext":
        if other is None:
            return self

        incoming = ensure_context(other)
        base = asdict(self)
        updates = asdict(incoming)
        merged: Dict[str, Any] = {}

        for key, value in base.items():
            if key == "metadata":
                merged[key] = {**(value or {}), **(updates.get(key) or {})}
            elif key == "parameters":
                base_params = value or {}
                update_params = updates.get(key) or {}
                if base_params or update_params:
                    merged[key] = {**base_params, **update_params}
                else:
                    merged[key] = None
            else:
                incoming_value = updates.get(key)
                merged[key] = incoming_value if incoming_value is not None else value

        return TuningErrorContext.from_mapping(merged)

    def to_dict(self, *, redact_sensitive: bool = True) -> JSONDict:
        return safe_serialize(asdict(self), redact_sensitive=redact_sensitive)


class TuningError(Exception):
    """Base exception for all tuning-related failures.

    Instances are designed to be safe for logging, tracing, and transport. They
    can be progressively enriched while preserving identifiers and causal links.
    """

    default_code = TuningErrorCode.INTERNAL_ERROR
    default_severity = TuningSeverity.ERROR
    default_retryable = False

    def __init__(self, message: str, *,
        code: Optional[TuningErrorCode] = None,
        severity: Optional[TuningSeverity] = None,
        retryable: Optional[bool] = None,
        context: Optional[Union[TuningErrorContext, Mapping[str, Any]]] = None,
        details: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        tags: Optional[Iterable[str]] = None,
        error_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.severity = severity or self.default_severity
        self.retryable = self.default_retryable if retryable is None else retryable
        self.context = ensure_context(context)
        self.details: JSONDict = dict(details or {})
        self.cause = cause
        self.tags = _normalize_tags(tags)
        self.error_id = error_id or uuid4().hex
        self.timestamp = timestamp or _utc_now_iso()
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self.message!r}, code={self.code.value!r}, "
            f"severity={self.severity.value!r}, retryable={self.retryable!r}, error_id={self.error_id!r})"
        )

    def clone(self, *,
        message: Optional[str] = None,
        code: Optional[TuningErrorCode] = None,
        severity: Optional[TuningSeverity] = None,
        retryable: Optional[bool] = None,
        context: Optional[Union[TuningErrorContext, Mapping[str, Any]]] = None,
        details: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> "TuningError":
        merged_context = self.context.merge(context)
        merged_details: JSONDict = {**self.details, **dict(details or {})}
        merged_tags = tuple(dict.fromkeys((*self.tags, *tuple(tags or ()))))

        return self.__class__(
            message or self.message,
            code=code or self.code,
            severity=severity or self.severity,
            retryable=self.retryable if retryable is None else retryable,
            context=merged_context,
            details=merged_details,
            cause=self.cause if cause is None else cause,
            tags=merged_tags,
            error_id=self.error_id,
            timestamp=self.timestamp,
        )

    def with_context(self, context: Optional[Union[TuningErrorContext, Mapping[str, Any]]]) -> "TuningError":
        return self.clone(context=context)

    def with_details(self, details: Optional[Mapping[str, Any]]) -> "TuningError":
        return self.clone(details=details)

    def with_tags(self, tags: Optional[Iterable[str]]) -> "TuningError":
        return self.clone(tags=tags)

    def iter_cause_chain(self, *, limit: int = 5) -> list[JSONDict]:
        chain: list[JSONDict] = []
        current: Optional[BaseException] = self.cause
        steps = 0

        while current is not None and steps < limit:
            chain.append(
                {
                    "name": current.__class__.__name__,
                    "message": _truncate_text(str(current), DEFAULT_MAX_STRING_LENGTH),
                }
            )
            steps += 1
            if isinstance(current, TuningError):
                current = current.cause
            else:
                current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

        return chain

    def to_dict(self, *,
        include_cause: bool = True,
        include_cause_chain: bool = True,
        include_traceback: bool = False,
        redact_sensitive: bool = True,
    ) -> JSONDict:
        payload: JSONDict = {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "error_name": self.__class__.__name__,
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "retryable": self.retryable,
            "context": self.context.to_dict(redact_sensitive=redact_sensitive),
            "details": safe_serialize(self.details, redact_sensitive=redact_sensitive),
            "tags": list(self.tags),
        }

        if include_cause and self.cause is not None:
            payload["cause"] = safe_serialize(
                {
                    "name": self.cause.__class__.__name__,
                    "message": str(self.cause),
                },
                redact_sensitive=redact_sensitive,
            )

        if include_cause_chain and self.cause is not None:
            payload["cause_chain"] = safe_serialize(
                self.iter_cause_chain(),
                redact_sensitive=redact_sensitive,
            )

        if include_traceback and self.cause is not None:
            payload["traceback"] = _format_traceback(self.cause)

        return payload

    def to_log_record(self, *, redact_sensitive: bool = True, include_traceback: bool = False) -> JSONDict:
        """Return a logging-friendly dictionary payload."""
        return self.to_dict(
            redact_sensitive=redact_sensitive,
            include_traceback=include_traceback,
        )


class TuningConfigError(TuningError):
    default_code = TuningErrorCode.CONFIG_ERROR


class TuningValidationError(TuningError):
    default_code = TuningErrorCode.VALIDATION_ERROR


class TuningSearchSpaceError(TuningError):
    default_code = TuningErrorCode.SEARCH_SPACE_ERROR


class TuningStrategyError(TuningError):
    default_code = TuningErrorCode.STRATEGY_ERROR


class TuningEvaluationError(TuningError):
    default_code = TuningErrorCode.EVALUATION_ERROR
    default_retryable = True


class TuningOptimizationError(TuningError):
    default_code = TuningErrorCode.OPTIMIZATION_ERROR


class TuningPersistenceError(TuningError):
    default_code = TuningErrorCode.PERSISTENCE_ERROR


class TuningReportingError(TuningError):
    default_code = TuningErrorCode.REPORTING_ERROR


class TuningDependencyError(TuningError):
    default_code = TuningErrorCode.DEPENDENCY_ERROR


class TuningInternalError(TuningError):
    default_code = TuningErrorCode.INTERNAL_ERROR
    default_severity = TuningSeverity.CRITICAL


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalize_tags(tags: Optional[Iterable[str]]) -> tuple[str, ...]:
    if tags is None:
        return ()

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def _truncate_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in SENSITIVE_FIELD_NAMES or any(token in lowered for token in SENSITIVE_FIELD_NAMES)


def _format_traceback(exc: BaseException, *, limit: int = DEFAULT_TRACEBACK_LINE_LIMIT) -> list[str]:
    formatted = [line.rstrip("\n") for line in format_exception(type(exc), exc, exc.__traceback__)]
    if len(formatted) <= limit:
        return formatted
    return [*formatted[: limit - 1], "... traceback truncated ..."]


def safe_serialize(value: Any, *, _depth: int = 0,
    redact_sensitive: bool = True,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
    _field_name: Optional[str] = None,
) -> Any:
    """Serialize arbitrary values into JSON-safe structures.

    This function is intentionally defensive. It avoids raising serialization
    errors while still preserving enough structure for debugging.
    """
    if redact_sensitive and _field_name and _is_sensitive_key(_field_name):
        return "***REDACTED***"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_text(value, max_string_length)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, BaseException):
        return {
            "name": value.__class__.__name__,
            "message": _truncate_text(str(value), max_string_length),
        }

    if _depth >= max_depth:
        return f"<{type(value).__name__} depth_limit_reached>"

    if is_dataclass(value) and not isinstance(value, type):
        try:
            return safe_serialize(
                asdict(value),
                redact_sensitive=redact_sensitive,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
                _field_name=_field_name,
            )
        except Exception:
            return repr(value)

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return safe_serialize(
                value.to_dict(),
                redact_sensitive=redact_sensitive,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
                _field_name=_field_name,
            )
        except Exception:
            return repr(value)

    if isinstance(value, Mapping):
        serialized: JSONDict = {}
        items = list(value.items())
        truncated = len(items) > max_items
        for key, item in items[:max_items]:
            key_str = _truncate_text(str(key), max_string_length)
            serialized[key_str] = safe_serialize(
                item,
                redact_sensitive=redact_sensitive,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
                _field_name=key_str,
            )
        if truncated:
            serialized["__truncated__"] = f"{len(items) - max_items} additional items omitted"
        return serialized

    if isinstance(value, (list, tuple, set, frozenset, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        serialized_items = [
            safe_serialize(
                item,
                redact_sensitive=redact_sensitive,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
                _field_name=_field_name,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            serialized_items.append(f"... {len(items) - max_items} additional items omitted")
        return serialized_items

    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def ensure_context(
    context: Optional[Union[TuningErrorContext, Mapping[str, Any], Any]],
) -> TuningErrorContext:
    """Normalize input context into ``TuningErrorContext``.

    Invalid inputs are folded into metadata instead of raising, because the
    error infrastructure itself should stay operational during failure paths.
    """
    if context is None:
        return TuningErrorContext()
    if isinstance(context, TuningErrorContext):
        return context
    if isinstance(context, Mapping):
        return TuningErrorContext.from_mapping(context)
    return TuningErrorContext(metadata={"invalid_context": safe_serialize(context)})


def enrich_context(
    base: Optional[Union[TuningErrorContext, Mapping[str, Any]]],
    updates: Optional[Mapping[str, Any]] = None,
) -> TuningErrorContext:
    """Merge a base context with updates and return a new context."""
    base_context = ensure_context(base)
    return base_context.merge(updates or {})


def raise_for_condition(condition: bool, message: str, *,
    error_cls: type[TuningError] = TuningValidationError,
    context: Optional[Union[TuningErrorContext, Mapping[str, Any]]] = None,
    details: Optional[Mapping[str, Any]] = None,
    severity: Optional[TuningSeverity] = None,
    retryable: Optional[bool] = None,
    tags: Optional[Iterable[str]] = None,
) -> None:
    """Raise a typed tuning error when a condition is true."""
    if condition:
        raise error_cls(
            message,
            context=context,
            details=details,
            severity=severity,
            retryable=retryable,
            tags=tags,
        )


def require_dependency(condition: bool, dependency_name: str, *,
    message: Optional[str] = None, details: Optional[Mapping[str, Any]] = None,
    context: Optional[Union[TuningErrorContext, Mapping[str, Any]]] = None,
) -> None:
    """Raise ``TuningDependencyError`` when a required dependency is unavailable."""
    raise_for_condition(
        not condition,
        message or f"Required dependency is unavailable: {dependency_name}.",
        error_cls=TuningDependencyError,
        context=context,
        details={"dependency": dependency_name, **dict(details or {})},
    )


def wrap_exception(exc: BaseException, *, message: str,
    error_cls: type[TuningError] = TuningInternalError,
    context: Optional[Union[TuningErrorContext, Mapping[str, Any]]] = None,
    details: Optional[Mapping[str, Any]] = None,
    severity: Optional[TuningSeverity] = None,
    retryable: Optional[bool] = None,
    tags: Optional[Iterable[str]] = None,
) -> TuningError:
    """Wrap unknown exceptions in a typed tuning error.

    Existing ``TuningError`` instances are enriched rather than double-wrapped.
    Unknown exceptions are promoted to the requested error class, with the
    original exception attached as the cause and summarized into details.
    """
    if not isinstance(exc, BaseException):
        exc = TypeError(f"wrap_exception expected BaseException, received {type(exc).__name__}")

    if not issubclass(error_cls, TuningError):
        raise TypeError("error_cls must be a subclass of TuningError")

    incoming_details: JSONDict = dict(details or {})

    if isinstance(exc, TuningError):
        if not incoming_details and context is None and severity is None and retryable is None and not tags:
            return exc
        return exc.clone(
            context=context,
            details=incoming_details,
            severity=severity,
            retryable=retryable,
            tags=tags,
        )

    if "original_exception" not in incoming_details:
        incoming_details["original_exception"] = {
            "name": exc.__class__.__name__,
            "message": str(exc),
        }

    return error_cls(
        message,
        context=context,
        details=incoming_details,
        cause=exc,
        severity=severity,
        retryable=retryable,
        tags=tags,
    )


def error_boundary(*, error_cls: type[TuningError] = TuningInternalError,
    context: Optional[Union[TuningErrorContext, Mapping[str, Any]]] = None,
    message: str = "Tuning operation failed.",
    detail_builder: Optional[Callable[[BaseException, tuple[Any, ...], dict[str, Any]], Mapping[str, Any]]] = None,
    context_builder: Optional[Callable[[BaseException, tuple[Any, ...], dict[str, Any]], Optional[Union[TuningErrorContext, Mapping[str, Any]]]]] = None,
    severity: Optional[TuningSeverity] = None,
    retryable: Optional[bool] = None,
    tags: Optional[Iterable[str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that normalizes arbitrary exceptions to ``TuningError`` types.

    ``context_builder`` and ``detail_builder`` make it possible to attach
    runtime state without polluting the wrapped function body.
    """

    if not issubclass(error_cls, TuningError):
        raise TypeError("error_cls must be a subclass of TuningError")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                static_context = ensure_context(context)
                dynamic_context: Optional[Union[TuningErrorContext, Mapping[str, Any]]] = None
                built_details: JSONDict = {
                    "function": getattr(func, "__qualname__", getattr(func, "__name__", "unknown")),
                    "module": getattr(func, "__module__", "unknown"),
                }

                if context_builder is not None:
                    try:
                        dynamic_context = context_builder(exc, args, kwargs)
                    except Exception as builder_exc:  # noqa: BLE001
                        built_details["context_builder_error"] = {
                            "name": builder_exc.__class__.__name__,
                            "message": str(builder_exc),
                        }

                if detail_builder is not None:
                    try:
                        built_details.update(dict(detail_builder(exc, args, kwargs) or {}))
                    except Exception as builder_exc:  # noqa: BLE001
                        built_details["detail_builder_error"] = {
                            "name": builder_exc.__class__.__name__,
                            "message": str(builder_exc),
                        }

                merged_context = static_context.merge(dynamic_context)
                raise wrap_exception(
                    exc,
                    message=message,
                    error_cls=error_cls,
                    context=merged_context,
                    details=built_details,
                    severity=severity,
                    retryable=retryable,
                    tags=tags,
                ) from exc

        return wrapped

    return decorator


__all__ = [
    "TuningErrorCode",
    "TuningSeverity",
    "TuningErrorContext",
    "TuningError",
    "TuningConfigError",
    "TuningValidationError",
    "TuningSearchSpaceError",
    "TuningStrategyError",
    "TuningEvaluationError",
    "TuningOptimizationError",
    "TuningPersistenceError",
    "TuningReportingError",
    "TuningDependencyError",
    "TuningInternalError",
    "ensure_context",
    "enrich_context",
    "raise_for_condition",
    "require_dependency",
    "safe_serialize",
    "wrap_exception",
    "error_boundary",
]
