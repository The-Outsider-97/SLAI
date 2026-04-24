"""
Structured exception hierarchy and validation helpers for the Base Agent stack.

This module provides a production-ready, domain-specific exception taxonomy for
BaseAgent and adjacent submodules. It focuses on deterministic error codes,
safe context capture, robust serialisation, explicit severity/retry semantics,
and reusable validation helpers.

Design goals:
- Stable, deterministic error codes suitable for logging and alerting.
- Rich, serialisable context for audit trails and human oversight.
- Reusable validation helpers for inputs, configuration, state, and schemas.
- Safe behaviour inside error paths so exception construction does not itself
  introduce secondary failures.
- Backward compatibility with the previous ``BaseError(message, config, ...)``
  calling style.
"""

from __future__ import annotations

import json
import traceback

from enum import Enum
from datetime import datetime, timezone
from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Type, TypeVar, Union

from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter  # type: ignore

logger = get_logger("Base Error")
printer = PrettyPrinter

T = TypeVar("T")

_VALID_SEVERITIES = {"low", "medium", "high", "critical"}
_DEFAULT_MAX_SERIALIZATION_DEPTH = 5
_DEFAULT_MAX_COLLECTION_ITEMS = 25
_DEFAULT_MAX_STRING_LENGTH = 500


class BaseErrorType(str, Enum):
    """Canonical error domains across the base agent stack."""

    UNKNOWN = "unknown"
    CONFIGURATION = "configuration"
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    STATE = "state"
    RUNTIME = "runtime"
    SERIALIZATION = "serialization"
    DEPENDENCY = "dependency"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    EXTERNAL_SERVICE = "external_service"
    IO = "io"


# Mapping from error type to unique, deterministic error code.
# Format: BAS-1xxx (Base Agent)
_ERROR_CODE_MAP: Dict[BaseErrorType, str] = {
    BaseErrorType.UNKNOWN: "BAS-1000",
    BaseErrorType.CONFIGURATION: "BAS-1100",
    BaseErrorType.INITIALIZATION: "BAS-1101",
    BaseErrorType.VALIDATION: "BAS-1200",
    BaseErrorType.STATE: "BAS-1201",
    BaseErrorType.SERIALIZATION: "BAS-1300",
    BaseErrorType.DEPENDENCY: "BAS-1400",
    BaseErrorType.TIMEOUT: "BAS-1500",
    BaseErrorType.RESOURCE: "BAS-1501",
    BaseErrorType.EXTERNAL_SERVICE: "BAS-1600",
    BaseErrorType.IO: "BAS-1700",
    BaseErrorType.RUNTIME: "BAS-1900",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_repr(value: Any, max_length: int = _DEFAULT_MAX_STRING_LENGTH) -> str:
    """Return a bounded repr that never raises."""
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _truncate_text(value: str, max_length: int = _DEFAULT_MAX_STRING_LENGTH) -> str:
    return value if len(value) <= max_length else value[: max_length - 3] + "..."


def _coerce_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, ABCMapping):
        try:
            return dict(value)
        except Exception:
            return {"raw": _safe_repr(value)}
    return {"raw": _safe_repr(value)}


def _coerce_severity(value: Any, default: str = "medium") -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _VALID_SEVERITIES:
            return lowered
    return default


def _json_safe(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = _DEFAULT_MAX_SERIALIZATION_DEPTH,
    max_items: int = _DEFAULT_MAX_COLLECTION_ITEMS,
) -> Any:
    """
    Convert arbitrary Python objects into JSON-safe primitives.

    The implementation is intentionally defensive because it is used from error
    paths where secondary failures are unacceptable.
    """
    if depth >= max_depth:
        return _safe_repr(value)

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_text(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, bytes):
        try:
            return _truncate_text(value.decode("utf-8", errors="replace"))
        except Exception:
            return _safe_repr(value)

    if isinstance(value, BaseException):
        return {
            "type": value.__class__.__name__,
            "message": _truncate_text(str(value)),
        }

    if isinstance(value, ABCMapping):
        result: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                result["__truncated__"] = True
                result["__remaining_items__"] = max(0, len(value) - max_items) if hasattr(value, "__len__") else True
                break
            result[str(key)] = _json_safe(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = list(value)
        payload = [
            _json_safe(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for item in sequence[:max_items]
        ]
        if len(sequence) > max_items:
            payload.append({"__truncated__": True, "__remaining_items__": len(sequence) - max_items})
        return payload

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict(), depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value), depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception:
            pass

    return _safe_repr(value)


def _load_error_config(override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Load and merge the base error config section without raising."""
    base: Dict[str, Any] = {}
    try:
        loaded = load_global_config()
        if isinstance(loaded, dict):
            base["_global"] = loaded
    except Exception:
        base["_global"] = {}

    try:
        section = get_config_section("base_error")
        if isinstance(section, dict):
            base.update(section)
    except Exception:
        pass

    base.update(_coerce_mapping(override))
    return base


def _severity_to_level(severity: str) -> int:
    return {
        "low": logger.INFO,
        "medium": logger.WARNING,
        "high": logger.ERROR,
        "critical": logger.CRITICAL,
    }.get(severity, logger.ERROR)


class BaseError(Exception):
    """
    Base exception for the base agent stack.

    Instances are designed to be both human-readable and machine-actionable.
    They support deterministic codes, structured context, serialisation,
    logging, and exception wrapping for lower-level failures.
    """

    error_type: BaseErrorType = BaseErrorType.UNKNOWN
    default_code = _ERROR_CODE_MAP[BaseErrorType.UNKNOWN]
    default_severity = "medium"
    default_retryable = False
    default_category: Union[str, Dict[str, Any]] = "base"

    def __init__(
        self,
        message: str,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message)

        self.message = str(message)
        self.config = self._safe_load_global_config()
        self.global_config = self.config
        self.error_config = _load_error_config(config)
        self.config_override = _coerce_mapping(config)

        self.code = str(kwargs.get("code") or _ERROR_CODE_MAP.get(self.error_type, self.default_code))
        self.severity = _coerce_severity(
            kwargs.get("severity", self.error_config.get("default_severity", self.default_severity)),
            self.default_severity,
        )
        self.retryable = bool(kwargs.get("retryable", self.default_retryable))
        self.category = kwargs.get("category", self.default_category)
        self.context = _coerce_mapping(kwargs.get("context"))
        self.details = _coerce_mapping(kwargs.get("details"))
        self.metadata = _coerce_mapping(kwargs.get("metadata"))
        self.tags = tuple(str(tag) for tag in kwargs.get("tags", ()) or ())
        self.component = kwargs.get("component")
        self.operation = kwargs.get("operation")
        self.resolution_hint = kwargs.get("resolution_hint")
        self.timestamp = str(kwargs.get("timestamp") or _utc_now_iso())

        cause = kwargs.get("cause")
        if cause is not None and not isinstance(cause, BaseException):
            cause = TypeError(f"Invalid cause type for BaseError: {type(cause).__name__}")
        self.cause = cause
        if self.cause is not None:
            self.__cause__ = self.cause

        self.traceback_text = None
        include_traceback = bool(kwargs.get("include_traceback", self.error_config.get("include_traceback", False)))
        if include_traceback and self.cause is not None:
            self.traceback_text = "".join(
                traceback.format_exception(type(self.cause), self.cause, self.cause.__traceback__)
            )

        if kwargs.get("auto_log", False):
            self.log()

    @staticmethod
    def _safe_load_global_config() -> Dict[str, Any]:
        try:
            loaded = load_global_config()
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    def __str__(self) -> str:
        parts = [f"[{self.code}]", self.message]
        parts.append(f"severity={self.severity}")
        if self.component:
            parts.append(f"component={self.component}")
        if self.operation:
            parts.append(f"operation={self.operation}")
        if self.retryable:
            parts.append("retryable=True")
        return " ".join(parts)

    def summary(self) -> str:
        return f"{self.code}::{self.error_type.value}::{self.message}"

    def add_context(self, **context: Any) -> "BaseError":
        self.context.update(_coerce_mapping(context))
        return self

    def add_details(self, **details: Any) -> "BaseError":
        self.details.update(_coerce_mapping(details))
        return self

    def to_dict(self, include_traceback: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": self.__class__.__name__,
            "error_type": self.error_type.value,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "retryable": self.retryable,
            "category": _json_safe(self.category),
            "component": self.component,
            "operation": self.operation,
            "resolution_hint": self.resolution_hint,
            "timestamp": self.timestamp,
            "tags": list(self.tags),
            "context": _json_safe(self.context),
            "details": _json_safe(self.details),
            "metadata": _json_safe(self.metadata),
        }
        if self.cause is not None:
            payload["cause"] = _json_safe(self.cause)
        if include_traceback and self.traceback_text:
            payload["traceback"] = self.traceback_text
        return payload

    def to_json(self, *, indent: int = 2, include_traceback: bool = False, sort_keys: bool = True) -> str:
        try:
            return json.dumps(
                self.to_dict(include_traceback=include_traceback),
                indent=indent,
                sort_keys=sort_keys,
            )
        except Exception:
            fallback = {
                "type": self.__class__.__name__,
                "code": self.code,
                "message": self.message,
                "serialization_error": True,
            }
            return json.dumps(fallback, indent=indent, sort_keys=sort_keys)

    def log(self, level: Optional[int] = None, include_traceback: bool = False, extra: Optional[Mapping[str, Any]] = None) -> None:
        """Emit the error payload through the configured logger."""
        log_level = level if level is not None else _severity_to_level(self.severity)
        payload = self.to_dict(include_traceback=include_traceback)
        if extra:
            payload["extra"] = _json_safe(dict(extra))
        try:
            logger.log(log_level, self.summary(), extra={"base_error": payload})
        except Exception:
            logger.log(log_level, f"{self.summary()} payload={_safe_repr(payload)}")

    @classmethod
    def wrap(
        cls: Type["BaseError"],
        exc: BaseException,
        message: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "BaseError":
        """
        Wrap an arbitrary exception into the current BaseError subclass.

        Existing BaseError instances are preserved and enriched instead of being
        double-wrapped.
        """
        if isinstance(exc, BaseError):
            if message and exc.message != message:
                exc.details.setdefault("wrapped_message", exc.message)
                exc.message = message
            if kwargs.get("context"):
                exc.context.update(_coerce_mapping(kwargs["context"]))
            if kwargs.get("details"):
                exc.details.update(_coerce_mapping(kwargs["details"]))
            return exc

        final_message = message or str(exc) or exc.__class__.__name__
        details = _coerce_mapping(kwargs.pop("details", None))
        details.setdefault("wrapped_exception_type", exc.__class__.__name__)
        details.setdefault("wrapped_exception_message", str(exc))
        return cls(final_message, config=config, cause=exc, details=details, **kwargs)

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "BaseError":
        return cls.wrap(exc, config=config, **kwargs)


class ConfigurationError(BaseError):
    error_type = BaseErrorType.CONFIGURATION
    default_code = _ERROR_CODE_MAP[BaseErrorType.CONFIGURATION]
    default_severity = "high"
    default_category = "configuration"


class InitializationError(BaseError):
    error_type = BaseErrorType.INITIALIZATION
    default_code = _ERROR_CODE_MAP[BaseErrorType.INITIALIZATION]
    default_severity = "high"
    default_category = "initialization"


class ValidationError(BaseError):
    error_type = BaseErrorType.VALIDATION
    default_code = _ERROR_CODE_MAP[BaseErrorType.VALIDATION]
    default_severity = "medium"
    default_category = "validation"


class StateError(BaseError):
    error_type = BaseErrorType.STATE
    default_code = _ERROR_CODE_MAP[BaseErrorType.STATE]
    default_severity = "high"
    default_category = "state"


class RuntimeBaseError(BaseError):
    error_type = BaseErrorType.RUNTIME
    default_code = _ERROR_CODE_MAP[BaseErrorType.RUNTIME]
    default_severity = "high"
    default_category = "runtime"


class SerializationError(BaseError):
    error_type = BaseErrorType.SERIALIZATION
    default_code = _ERROR_CODE_MAP[BaseErrorType.SERIALIZATION]
    default_severity = "medium"
    default_category = "serialization"


class DependencyError(BaseError):
    error_type = BaseErrorType.DEPENDENCY
    default_code = _ERROR_CODE_MAP[BaseErrorType.DEPENDENCY]
    default_severity = "high"
    default_category = "dependency"


class TimeoutBaseError(BaseError):
    error_type = BaseErrorType.TIMEOUT
    default_code = _ERROR_CODE_MAP[BaseErrorType.TIMEOUT]
    default_severity = "high"
    default_retryable = True
    default_category = "timeout"


class ResourceError(BaseError):
    error_type = BaseErrorType.RESOURCE
    default_code = _ERROR_CODE_MAP[BaseErrorType.RESOURCE]
    default_severity = "high"
    default_retryable = True
    default_category = "resource"


class ExternalServiceError(BaseError):
    error_type = BaseErrorType.EXTERNAL_SERVICE
    default_code = _ERROR_CODE_MAP[BaseErrorType.EXTERNAL_SERVICE]
    default_severity = "high"
    default_retryable = True
    default_category = "external_service"


class IOErrorBase(BaseError):
    error_type = BaseErrorType.IO
    default_code = _ERROR_CODE_MAP[BaseErrorType.IO]
    default_severity = "high"
    default_category = "io"


# Backward-compatible aliases with explicit "Base" prefixes for projects that
# prefer more descriptive exception names.
BaseConfigurationError = ConfigurationError
BaseInitializationError = InitializationError
BaseValidationError = ValidationError
BaseStateError = StateError
BaseRuntimeError = RuntimeBaseError
BaseSerializationError = SerializationError
BaseDependencyError = DependencyError
BaseTimeoutError = TimeoutBaseError
BaseResourceError = ResourceError
BaseExternalServiceError = ExternalServiceError
BaseIOError = IOErrorBase


def raise_if(
    condition: bool,
    message: str,
    *,
    error_cls: Type[BaseError] = ValidationError,
    config: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> None:
    if condition:
        raise error_cls(message, config=config, **kwargs)


def ensure_condition(
    condition: bool,
    message: str,
    *,
    error_cls: Type[BaseError] = ValidationError,
    config: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> None:
    raise_if(not condition, message, error_cls=error_cls, config=config, **kwargs)


def ensure_not_none(
    value: Optional[T],
    name: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    **kwargs: Any,
) -> T:
    if value is None:
        context = _coerce_mapping(kwargs.pop("context", None))
        context.setdefault("field", name)
        raise error_cls(f"'{name}' must not be None.", config=config, context=context, **kwargs)
    return value


def ensure_type(
    value: Any,
    expected_type: Union[Type[Any], Tuple[Type[Any], ...]],
    name: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    allow_none: bool = False,
    **kwargs: Any,
) -> Any:
    if value is None and allow_none:
        return value
    if not isinstance(value, expected_type):
        context = _coerce_mapping(kwargs.pop("context", None))
        expected = (
            [t.__name__ for t in expected_type]
            if isinstance(expected_type, tuple)
            else [expected_type.__name__]
        )
        context.update({
            "field": name,
            "expected_type": expected,
            "received_type": type(value).__name__,
        })
        raise error_cls(
            f"'{name}' must be of type {', '.join(expected)}; received {type(value).__name__}.",
            config=config,
            context=context,
            **kwargs,
        )
    return value


def ensure_subclass(
    value: Type[Any],
    expected_parent: Type[Any],
    name: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    **kwargs: Any,
) -> Type[Any]:
    if not isinstance(value, type) or not issubclass(value, expected_parent):
        context = _coerce_mapping(kwargs.pop("context", None))
        context.update({
            "field": name,
            "expected_parent": expected_parent.__name__,
            "received": _safe_repr(value),
        })
        raise error_cls(
            f"'{name}' must be a subclass of {expected_parent.__name__}.",
            config=config,
            context=context,
            **kwargs,
        )
    return value


def ensure_callable(
    value: Any,
    name: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    **kwargs: Any,
) -> Any:
    if not callable(value):
        context = _coerce_mapping(kwargs.pop("context", None))
        context.update({"field": name, "received_type": type(value).__name__})
        raise error_cls(f"'{name}' must be callable.", config=config, context=context, **kwargs)
    return value


def ensure_mapping(
    value: Any,
    name: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    allow_none: bool = False,
    **kwargs: Any,
) -> Mapping[str, Any]:
    if value is None and allow_none:
        return {}
    if not isinstance(value, ABCMapping):
        context = _coerce_mapping(kwargs.pop("context", None))
        context.update({"field": name, "received_type": type(value).__name__})
        raise error_cls(f"'{name}' must be a mapping type.", config=config, context=context, **kwargs)
    return value


def ensure_sequence(
    value: Any,
    name: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    allow_str: bool = False,
    **kwargs: Any,
) -> ABCSequence[Any]:
    is_sequence = isinstance(value, ABCSequence)
    is_str = isinstance(value, (str, bytes, bytearray))
    if not is_sequence or (is_str and not allow_str):
        context = _coerce_mapping(kwargs.pop("context", None))
        context.update({"field": name, "received_type": type(value).__name__})
        raise error_cls(f"'{name}' must be a non-string sequence.", config=config, context=context, **kwargs)
    return value


def ensure_non_empty_string(
    value: Any,
    name: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    strip: bool = True,
    **kwargs: Any,
) -> str:
    ensure_type(value, str, name, config=config, error_cls=error_cls, **kwargs)
    normalized = value.strip() if strip else value
    if not normalized:
        context = _coerce_mapping(kwargs.pop("context", None))
        context.update({"field": name})
        raise error_cls(f"'{name}' must be a non-empty string.", config=config, context=context, **kwargs)
    return normalized


def ensure_one_of(
    value: T,
    options: Iterable[T],
    name: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    casefold: bool = False,
    **kwargs: Any,
) -> T:
    materialized = list(options)
    if casefold and isinstance(value, str):
        normalized_value = value.casefold()
        normalized_options = [item.casefold() if isinstance(item, str) else item for item in materialized]
        valid = normalized_value in normalized_options
    else:
        valid = value in materialized
    if not valid:
        context = _coerce_mapping(kwargs.pop("context", None))
        context.update({"field": name, "allowed_values": materialized, "received": value})
        raise error_cls(
            f"'{name}' must be one of {materialized!r}; received {value!r}.",
            config=config,
            context=context,
            **kwargs,
        )
    return value


def ensure_keys(
    mapping: Mapping[str, Any],
    required_keys: Iterable[str],
    name: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    **kwargs: Any,
) -> Mapping[str, Any]:
    ensure_mapping(mapping, name, config=config, error_cls=error_cls)
    required = list(required_keys)
    missing = [key for key in required if key not in mapping]
    if missing:
        context = _coerce_mapping(kwargs.pop("context", None))
        context.update({"field": name, "missing_keys": missing, "required_keys": required})
        raise error_cls(
            f"'{name}' is missing required keys: {missing!r}.",
            config=config,
            context=context,
            **kwargs,
        )
    return mapping


def ensure_numeric_range(
    value: Union[int, float],
    name: str,
    *,
    minimum: Optional[Union[int, float]] = None,
    maximum: Optional[Union[int, float]] = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    **kwargs: Any,
) -> Union[int, float]:
    ensure_type(value, (int, float), name, config=config, error_cls=error_cls, **kwargs)

    too_low = False
    too_high = False
    if minimum is not None:
        too_low = value < minimum if inclusive_min else value <= minimum
    if maximum is not None:
        too_high = value > maximum if inclusive_max else value >= maximum

    if too_low or too_high:
        context = _coerce_mapping(kwargs.pop("context", None))
        context.update(
            {
                "field": name,
                "minimum": minimum,
                "maximum": maximum,
                "inclusive_min": inclusive_min,
                "inclusive_max": inclusive_max,
                "received": value,
            }
        )
        lower_symbol = ">=" if inclusive_min else ">"
        upper_symbol = "<=" if inclusive_max else "<"
        if minimum is not None and maximum is not None:
            expectation = f"{lower_symbol} {minimum} and {upper_symbol} {maximum}"
        elif minimum is not None:
            expectation = f"{lower_symbol} {minimum}"
        else:
            expectation = f"{upper_symbol} {maximum}"
        raise error_cls(
            f"'{name}' must be {expectation}; received {value}.",
            config=config,
            context=context,
            **kwargs,
        )
    return value


def ensure_positive(value: Union[int, float], name: str, *, allow_zero: bool = False,
                    config: Optional[Mapping[str, Any]] = None, error_cls: Type[BaseError] = ValidationError,
                    **kwargs: Any) -> Union[int, float]:
    minimum = 0 if allow_zero else 0
    return ensure_numeric_range(
        value,
        name,
        minimum=minimum,
        inclusive_min=allow_zero,
        config=config,
        error_cls=error_cls,
        **kwargs,
    )


def ensure_state(condition: bool, message: str, *, config: Optional[Mapping[str, Any]] = None,
                 error_cls: Type[BaseError] = StateError, **kwargs: Any) -> None:
    raise_if(not condition, message, error_cls=error_cls, config=config, **kwargs)


def ensure_schema(
    mapping: Mapping[str, Any],
    schema: Mapping[str, Union[Type[Any], Tuple[Type[Any], ...]]],
    name: str = "mapping",
    *,
    config: Optional[Mapping[str, Any]] = None,
    error_cls: Type[BaseError] = ValidationError,
    allow_extra: bool = True,
    **kwargs: Any,
) -> Mapping[str, Any]:
    """
    Validate a shallow mapping schema.

    Example:
        ensure_schema(payload, {"name": str, "age": int}, name="payload")
    """
    ensure_mapping(mapping, name, config=config, error_cls=error_cls)
    ensure_mapping(schema, f"{name}_schema", config=config, error_cls=error_cls)

    missing = [key for key in schema if key not in mapping]
    if missing:
        context = _coerce_mapping(kwargs.pop("context", None))
        context.update({"field": name, "missing_keys": missing})
        raise error_cls(
            f"'{name}' is missing schema-defined keys: {missing!r}.",
            config=config,
            context=context,
            **kwargs,
        )

    if not allow_extra:
        extra_keys = [key for key in mapping if key not in schema]
        if extra_keys:
            context = _coerce_mapping(kwargs.pop("context", None))
            context.update({"field": name, "unexpected_keys": extra_keys})
            raise error_cls(
                f"'{name}' contains unexpected keys: {extra_keys!r}.",
                config=config,
                context=context,
                **kwargs,
            )

    for key, expected_type in schema.items():
        ensure_type(
            mapping[key],
            expected_type,
            f"{name}.{key}",
            config=config,
            error_cls=error_cls,
            **kwargs,
        )
    return mapping