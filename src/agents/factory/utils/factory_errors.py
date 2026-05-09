"""
Production-grade structured exception hierarchy for the factory and
agent-orchestration subsystem.

This module intentionally contains the factory error contract only:

- canonical factory error domains and deterministic error codes;
- serialisable payload objects for logs, metrics, API responses, and tests;
- a common ``FactoryError`` base class with structured context and retry
  semantics;
- concrete exception classes for metadata, registry, dependency resolution,
  construction, orchestration, metrics adaptation, out-of-process workers,
  cache operations, observability, security, resources, timeout, IO, and
  runtime failures.

It deliberately does not import ``factory_helpers`` and does not expose
validation, wrapping, or convenience helper functions. Helper utilities belong
in ``factory_helpers.py``. Keeping the dependency direction one-way prevents
circular imports while preserving a stable error contract that other factory
modules can depend on safely during early initialisation and failure paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import traceback

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Type, TypeVar, Union

from .config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Factory Error")
printer = PrettyPrinter()

T = TypeVar("T", bound="FactoryError")


class FactoryErrorType(str, Enum):
    """Canonical error domains across the factory/orchestration subsystem."""

    UNKNOWN = "unknown"
    CONFIGURATION = "configuration"
    METADATA = "metadata"
    VALIDATION = "validation"
    REGISTRY = "registry"
    DEPENDENCY = "dependency"
    VERSION = "version"
    RESOLUTION = "resolution"
    IMPORT = "import"
    CONSTRUCTION = "construction"
    ORCHESTRATION = "orchestration"
    LIFECYCLE = "lifecycle"
    STATE = "state"
    METRICS = "metrics"
    ADAPTATION = "adaptation"
    SAFETY_BOUNDS = "safety_bounds"
    REMOTE_WORKER = "remote_worker"
    SUBPROCESS = "subprocess"
    SERIALIZATION = "serialization"
    SECURITY = "security"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    IO = "io"
    CACHE = "cache"
    OBSERVABILITY = "observability"
    RUNTIME = "runtime"


class FactoryErrorSeverity(str, Enum):
    """Operational severity levels used by factory logs and result payloads."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


VALID_SEVERITIES = {severity.value for severity in FactoryErrorSeverity}

_ERROR_CODE_MAP: Dict[FactoryErrorType, str] = {
    FactoryErrorType.UNKNOWN: "FAC-1000",
    FactoryErrorType.CONFIGURATION: "FAC-1100",
    FactoryErrorType.METADATA: "FAC-1200",
    FactoryErrorType.VALIDATION: "FAC-1201",
    FactoryErrorType.REGISTRY: "FAC-1300",
    FactoryErrorType.DEPENDENCY: "FAC-1301",
    FactoryErrorType.VERSION: "FAC-1302",
    FactoryErrorType.RESOLUTION: "FAC-1400",
    FactoryErrorType.IMPORT: "FAC-1401",
    FactoryErrorType.CONSTRUCTION: "FAC-1402",
    FactoryErrorType.ORCHESTRATION: "FAC-1500",
    FactoryErrorType.LIFECYCLE: "FAC-1501",
    FactoryErrorType.STATE: "FAC-1502",
    FactoryErrorType.METRICS: "FAC-1600",
    FactoryErrorType.ADAPTATION: "FAC-1601",
    FactoryErrorType.SAFETY_BOUNDS: "FAC-1602",
    FactoryErrorType.REMOTE_WORKER: "FAC-1700",
    FactoryErrorType.SUBPROCESS: "FAC-1701",
    FactoryErrorType.SERIALIZATION: "FAC-1702",
    FactoryErrorType.SECURITY: "FAC-1800",
    FactoryErrorType.RESOURCE: "FAC-1900",
    FactoryErrorType.TIMEOUT: "FAC-1901",
    FactoryErrorType.IO: "FAC-1902",
    FactoryErrorType.CACHE: "FAC-2000",
    FactoryErrorType.OBSERVABILITY: "FAC-2100",
    FactoryErrorType.RUNTIME: "FAC-1999",
}

SENSITIVE_KEY_PATTERNS = (
    "authorization",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "cookie",
    "session",
    "csrf",
    "xsrf",
    "bearer",
    "private_key",
    "client_secret",
    "stdin",
    "stdout",
    "stderr",
    "payload",
)

DEFAULT_MAX_STRING_LENGTH = 2_000
DEFAULT_MAX_SEQUENCE_LENGTH = 50
DEFAULT_MAX_MAPPING_LENGTH = 80
DEFAULT_MAX_SERIALIZATION_DEPTH = 6
REDACTION_PLACEHOLDER = "[REDACTED]"


@dataclass(frozen=True)
class FactoryErrorPayload:
    """Portable representation of a factory error for logs and API results."""

    code: str
    type: str
    message: str
    severity: str
    retryable: bool
    category: Union[str, Dict[str, Any]]
    component: Optional[str] = None
    operation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = field(default_factory=tuple)
    remediation: Optional[str] = None
    cause: Optional[Dict[str, Any]] = None
    retry_after_seconds: Optional[float] = None
    fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "type": self.type,
            "message": self.message,
            "severity": self.severity,
            "retryable": self.retryable,
            "category": self.category,
            "timestamp": self.timestamp,
            "context": self.context,
            "details": self.details,
            "metadata": self.metadata,
            "tags": list(self.tags),
        }
        if self.component is not None:
            payload["component"] = self.component
        if self.operation is not None:
            payload["operation"] = self.operation
        if self.remediation is not None:
            payload["remediation"] = self.remediation
        if self.cause is not None:
            payload["cause"] = self.cause
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        if self.fingerprint is not None:
            payload["fingerprint"] = self.fingerprint
        return payload


class FactoryError(Exception):
    """Base exception for the factory/orchestration subsystem.

    The class is dependency-light, import-safe, and machine-actionable. It does
    not rely on ``factory_helpers``. Any public validation or wrapping utilities
    should be implemented in ``factory_helpers.py`` and may import this class.
    """

    error_type: FactoryErrorType = FactoryErrorType.UNKNOWN
    default_code = "FAC-1000"
    default_message = "Factory error"
    default_severity = FactoryErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category: Union[str, Dict[str, Any]] = "factory"
    default_retry_after_seconds: Optional[float] = None

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        error_type: Optional[Union[str, FactoryErrorType]] = None,
        severity: Optional[Union[str, FactoryErrorSeverity]] = None,
        code: Optional[str] = None,
        retryable: Optional[bool] = None,
        category: Optional[Union[str, Dict[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
        details: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        remediation: Optional[str] = None,
        tags: Optional[Iterable[Any]] = None,
        cause: Optional[BaseException] = None,
        retry_after_seconds: Optional[float] = None,
        include_traceback: bool = False,
        auto_log: bool = False,
    ) -> None:
        resolved_type = self._coerce_error_type(error_type or self.error_type)
        resolved_code = str(code or self.default_code or _ERROR_CODE_MAP.get(resolved_type, "FAC-1000"))
        resolved_message = str(message or self.default_message or f"{resolved_type.value.replace('_', ' ').title()} error")
        resolved_severity = self._coerce_severity(severity, default=self.default_severity)

        super().__init__(f"[{resolved_code}] {resolved_message}")

        self.error_type = resolved_type
        self.code = resolved_code
        self.message = resolved_message
        self.severity = resolved_severity
        self.retryable = bool(self.default_retryable if retryable is None else retryable)
        self.category = category if category is not None else self.default_category
        self.context = self._sanitize_mapping(context, redact=False)
        self.details = self._sanitize_mapping(details, redact=False)
        self.metadata = self._sanitize_mapping(metadata, redact=False)
        self.component = component
        self.operation = operation
        self.tags = self._deduplicate_tags(tags)
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.remediation = remediation or self._default_remediation(self.error_type)
        self.cause = cause if isinstance(cause, BaseException) or cause is None else TypeError(
            f"Invalid cause type for FactoryError: {type(cause).__name__}"
        )
        self.retry_after_seconds = self.default_retry_after_seconds if retry_after_seconds is None else retry_after_seconds
        self.global_config = self._safe_load_config()
        self.error_config = self._safe_get_error_config()

        self.traceback_text: Optional[str] = None
        if include_traceback and self.cause is not None:
            self.traceback_text = "".join(traceback.format_exception(type(self.cause), self.cause, self.cause.__traceback__))
        if self.cause is not None:
            self.__cause__ = self.cause
        if auto_log:
            self.log(include_traceback=include_traceback)

    def __str__(self) -> str:
        parts = [f"[{self.code}]", f"[{self.severity.upper()}]", self.message]
        if self.component:
            parts.append(f"component={self.component}")
        if self.operation:
            parts.append(f"operation={self.operation}")
        if self.retryable:
            parts.append("retryable=True")
        if self.remediation:
            parts.append(f"remediation={self.remediation}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(code={self.code!r}, error_type={self.error_type.value!r}, "
            f"severity={self.severity!r}, retryable={self.retryable!r}, message={self.message!r})"
        )

    def summary(self) -> str:
        return f"{self.code}::{self.error_type.value}::{self.message}"

    def with_context(self: T, **context: Any) -> T:
        self.context.update(self._sanitize_mapping(context, redact=False))
        return self

    def with_details(self: T, **details: Any) -> T:
        self.details.update(self._sanitize_mapping(details, redact=False))
        return self

    def with_metadata(self: T, **metadata: Any) -> T:
        self.metadata.update(self._sanitize_mapping(metadata, redact=False))
        return self

    def to_payload(
        self,
        *,
        redact: bool = True,
        include_cause: bool = True,
        include_traceback: bool = False,
    ) -> FactoryErrorPayload:
        category = self._serialise(self.category, redact=redact)
        if not isinstance(category, (str, dict)):
            category = {"value": category}
        cause_payload = self._cause_payload(self.cause, include_traceback=include_traceback, redact=redact) if include_cause else None
        if include_traceback and self.traceback_text and cause_payload is not None:
            cause_payload["traceback"] = self._truncate(self.traceback_text, 10_000)
        return FactoryErrorPayload(
            code=self.code,
            type=self.error_type.value,
            message=self.message,
            severity=self.severity,
            retryable=self.retryable,
            category=category,
            component=self.component,
            operation=self.operation,
            timestamp=self.timestamp,
            context=self._sanitize_mapping(self.context, redact=redact),
            details=self._sanitize_mapping(self.details, redact=redact),
            metadata=self._sanitize_mapping(self.metadata, redact=redact),
            tags=self.tags,
            remediation=self.remediation,
            cause=cause_payload,
            retry_after_seconds=self.retry_after_seconds,
            fingerprint=self._fingerprint(self.code, self.error_type.value, self.message, category),
        )

    def to_dict(
        self,
        *,
        redact: bool = True,
        include_cause: bool = True,
        include_traceback: bool = False,
    ) -> Dict[str, Any]:
        return self.to_payload(redact=redact, include_cause=include_cause, include_traceback=include_traceback).to_dict()

    def to_json(
        self,
        *,
        indent: Optional[int] = None,
        redact: bool = True,
        include_cause: bool = True,
        include_traceback: bool = False,
        sort_keys: bool = True,
    ) -> str:
        try:
            return json.dumps(
                self.to_dict(redact=redact, include_cause=include_cause, include_traceback=include_traceback),
                indent=indent,
                sort_keys=sort_keys,
                default=str,
            )
        except Exception:
            return json.dumps(
                {
                    "type": self.__class__.__name__,
                    "code": self.code,
                    "message": self.message,
                    "serialization_error": True,
                },
                indent=indent,
                sort_keys=sort_keys,
            )

    def to_result(
        self,
        *,
        status: str = "error",
        action: Optional[str] = None,
        agent: Optional[str] = None,
        redact: bool = True,
        include_cause: bool = True,
        include_traceback: bool = False,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": status,
            "message": self.message,
            "code": self.code,
            "error_type": self.error_type.value,
            "severity": self.severity,
            "retryable": self.retryable,
            "error": self.to_dict(redact=redact, include_cause=include_cause, include_traceback=include_traceback),
        }
        if action:
            result["action"] = action
        if agent:
            result["agent"] = agent
        if self.retry_after_seconds is not None:
            result["retry_after_seconds"] = self.retry_after_seconds
        if extra:
            result.update(self._sanitize_mapping(extra, redact=redact))
        return result

    def to_degraded_result(
        self,
        *,
        agent: Optional[str] = None,
        action: Optional[str] = None,
        redact: bool = True,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.to_result(status="degraded", agent=agent, action=action, redact=redact, extra=extra)

    def to_log_record(self, *, include_traceback: bool = False) -> Dict[str, Any]:
        record = self.to_dict(redact=True, include_cause=True, include_traceback=include_traceback)
        record["event"] = "factory_error"
        return record

    def log(self, *, level: Optional[int] = None, include_traceback: bool = False) -> Dict[str, Any]:
        record = self.to_log_record(include_traceback=include_traceback)
        log_level = level if level is not None else self._severity_to_log_level(self.severity)
        try:
            logger.log(log_level, self.summary(), extra={"factory_error": record})
        except Exception:
            try:
                logger.log(log_level, "%s payload=%s", self.summary(), self._safe_repr(record))
            except Exception:
                pass
        return record

    @classmethod
    def from_exception(
        cls: Type[T],
        exc: BaseException,
        *,
        message: Optional[str] = None,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        details: Optional[Mapping[str, Any]] = None,
        default_error_cls: Optional[Type["FactoryError"]] = None,
        retryable: Optional[bool] = None,
        severity: Optional[Union[str, FactoryErrorSeverity]] = None,
    ) -> "FactoryError":
        if isinstance(exc, FactoryError):
            if context:
                exc.context.update(cls._sanitize_mapping(context, redact=False))
            if details:
                exc.details.update(cls._sanitize_mapping(details, redact=False))
            if operation and not exc.operation:
                exc.operation = operation
            if component and not exc.component:
                exc.component = component
            return exc

        error_cls = cls._map_exception(exc, operation=operation) or default_error_cls or cls
        merged_context: Dict[str, Any] = {}
        if context:
            merged_context.update(dict(context))
        return error_cls(
            message or str(exc) or getattr(error_cls, "default_message", "Factory operation failed"),
            context=merged_context,
            details=details,
            operation=operation,
            component=component,
            cause=exc,
            retryable=retryable,
            severity=severity,
        )

    @classmethod
    def wrap(cls: Type[T], exc: BaseException, message: Optional[str] = None, **kwargs: Any) -> "FactoryError":
        return cls.from_exception(exc, message=message, **kwargs)

    @staticmethod
    def _truncate(text: Any, max_length: int = DEFAULT_MAX_STRING_LENGTH) -> str:
        value = str(text)
        if len(value) <= max_length:
            return value
        return f"{value[:max_length]}…[truncated {len(value) - max_length} chars]"

    @classmethod
    def _safe_repr(cls, value: Any, max_length: int = DEFAULT_MAX_STRING_LENGTH) -> str:
        try:
            text = repr(value)
        except Exception:
            text = f"<unrepresentable {type(value).__name__}>"
        return cls._truncate(text, max_length)

    @staticmethod
    def _is_sensitive_key(key: Any) -> bool:
        key_text = str(key).lower()
        return any(pattern in key_text for pattern in SENSITIVE_KEY_PATTERNS)

    @classmethod
    def _serialise(
        cls,
        value: Any,
        *,
        redact: bool = True,
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_SERIALIZATION_DEPTH,
        max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        max_mapping_length: int = DEFAULT_MAX_MAPPING_LENGTH,
    ) -> Any:
        try:
            if depth >= max_depth:
                return cls._safe_repr(value, max_string_length)
            if value is None or isinstance(value, (bool, int, float)):
                return value
            if isinstance(value, str):
                return cls._truncate(value, max_string_length)
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, (datetime, date, time)):
                return value.isoformat()
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, bytes):
                return f"<bytes length={len(value)}>"
            if isinstance(value, BaseException):
                return {"type": value.__class__.__name__, "message": cls._truncate(str(value), max_string_length)}
            if hasattr(value, "item") and callable(value.item):
                try:
                    return cls._serialise(value.item(), redact=redact, depth=depth + 1)
                except Exception:
                    pass
            if hasattr(value, "tolist") and callable(value.tolist):
                try:
                    return cls._serialise(value.tolist(), redact=redact, depth=depth + 1)
                except Exception:
                    pass
            if hasattr(value, "shape") or hasattr(value, "dtype"):
                snapshot: Dict[str, Any] = {"type": type(value).__name__}
                if hasattr(value, "shape"):
                    try:
                        snapshot["shape"] = tuple(getattr(value, "shape"))
                    except Exception:
                        snapshot["shape"] = cls._safe_repr(getattr(value, "shape", None))
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
                    if redact and cls._is_sensitive_key(key_text):
                        output[key_text] = REDACTION_PLACEHOLDER
                    else:
                        output[key_text] = cls._serialise(
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
                    cls._serialise(
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
                    return cls._serialise(value.to_dict(), redact=redact, depth=depth + 1)
                except Exception:
                    pass
            if hasattr(value, "__dict__"):
                try:
                    return cls._serialise(vars(value), redact=redact, depth=depth + 1)
                except Exception:
                    pass
            return cls._safe_repr(value, max_string_length)
        except Exception as exc:
            return f"<unserialisable {type(value).__name__}: {type(exc).__name__}>"

    @classmethod
    def _sanitize_mapping(cls, context: Optional[Mapping[str, Any]], *, redact: bool = True) -> Dict[str, Any]:
        if not context:
            return {}
        serialised = cls._serialise(dict(context), redact=redact)
        return serialised if isinstance(serialised, dict) else {"value": serialised}

    @classmethod
    def _cause_payload(
        cls,
        cause: Optional[BaseException],
        *,
        include_traceback: bool = False,
        redact: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if cause is None:
            return None
        payload: Dict[str, Any] = {
            "type": cause.__class__.__name__,
            "message": cls._serialise(str(cause), redact=redact),
        }
        if include_traceback:
            payload["traceback"] = cls._truncate(
                "".join(traceback.format_exception(type(cause), cause, cause.__traceback__)),
                10_000,
            )
        return payload

    @staticmethod
    def _fingerprint(code: str, error_type: str, message: str, category: Union[str, Dict[str, Any]]) -> str:
        source = json.dumps(
            {"code": code, "type": error_type, "message": message, "category": category},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _coerce_error_type(value: Optional[Union[str, FactoryErrorType]]) -> FactoryErrorType:
        if value is None:
            return FactoryErrorType.UNKNOWN
        if isinstance(value, FactoryErrorType):
            return value
        try:
            return FactoryErrorType(str(value).strip().lower())
        except ValueError:
            return FactoryErrorType.UNKNOWN

    @staticmethod
    def _coerce_severity(value: Optional[Union[str, FactoryErrorSeverity]], default: str = "medium") -> str:
        if isinstance(value, FactoryErrorSeverity):
            return value.value
        candidate = (str(value).lower().strip() if value is not None else default).strip()
        if candidate not in VALID_SEVERITIES:
            raise ValueError(f"Invalid factory error severity: {candidate}. Must be one of {sorted(VALID_SEVERITIES)}.")
        return candidate

    @staticmethod
    def _deduplicate_tags(tags: Optional[Iterable[Any]]) -> Tuple[str, ...]:
        if not tags:
            return tuple()
        seen = []
        for tag in tags:
            text = str(tag).strip()
            if text and text not in seen:
                seen.append(text)
        return tuple(seen)

    @staticmethod
    def _safe_load_config() -> Dict[str, Any]:
        try:
            loaded = load_global_config()
            return loaded if isinstance(loaded, dict) else {}
        except Exception as exc:
            try:
                logger.debug("Unable to load factory config while building error: %s", exc)
            except Exception:
                pass
            return {}

    @staticmethod
    def _safe_get_error_config() -> Dict[str, Any]:
        for section_name in ("factory_errors", "factory_error", "errors"):
            try:
                section = get_config_section(section_name)
                if isinstance(section, dict) and section:
                    return section
            except Exception as exc:
                try:
                    logger.debug("Unable to load %s config section: %s", section_name, exc)
                except Exception:
                    pass
        return {}

    @staticmethod
    def _severity_to_log_level(severity: str) -> int:
        return {
            FactoryErrorSeverity.LOW.value: logging.INFO,
            FactoryErrorSeverity.MEDIUM.value: logging.WARNING,
            FactoryErrorSeverity.HIGH.value: logging.ERROR,
            FactoryErrorSeverity.CRITICAL.value: logging.CRITICAL,
        }.get(severity, logging.ERROR)

    @classmethod
    def _default_remediation(cls, error_type: FactoryErrorType) -> Optional[str]:
        remediations = cls._safe_get_error_config().get("remediation", {})
        if isinstance(remediations, Mapping):
            value = remediations.get(error_type.value) or remediations.get(error_type.name)
            if value:
                return str(value)

        fallback_map = {
            FactoryErrorType.CONFIGURATION: "Validate factory_config.yaml, required sections, and deployment-specific config paths.",
            FactoryErrorType.METADATA: "Validate agent name, class_name, module_path, version, required_params, and dependency declarations before registration.",
            FactoryErrorType.REGISTRY: "Check registry state, duplicate names, version index, and dependency graph consistency.",
            FactoryErrorType.DEPENDENCY: "Resolve dependencies before loading the agent and inspect the dependency graph for missing or cyclic nodes.",
            FactoryErrorType.RESOLUTION: "Verify the configured module path and class name can be imported without side effects.",
            FactoryErrorType.CONSTRUCTION: "Verify constructor parameters, dependency injection, and optional shared services before creating the agent.",
            FactoryErrorType.ORCHESTRATION: "Inspect routing policy, selected agent metadata, fallback eligibility, and lifecycle state.",
            FactoryErrorType.METRICS: "Validate metric payload shape and ensure required fairness, performance, and bias values are available before adaptation.",
            FactoryErrorType.ADAPTATION: "Inspect PID parameters, metric history, and adaptation outputs before updating factory-managed runtime parameters.",
            FactoryErrorType.SAFETY_BOUNDS: "Check configured per-agent safety bounds and clamp unsafe adjustments before applying them.",
            FactoryErrorType.REMOTE_WORKER: "Validate worker payload, subprocess status, stdout/stderr encoding, and remote result shape.",
            FactoryErrorType.SUBPROCESS: "Check Python executable, module path, environment, timeout, and worker stderr.",
            FactoryErrorType.SERIALIZATION: "Ensure payloads and results are JSON-serialisable or provide a safe serializer.",
            FactoryErrorType.SECURITY: "Verify module-path allowlists and class-resolution policy before dynamic imports.",
            FactoryErrorType.CACHE: "Check cache size, TTL, key hashability, eviction policy, expiration handling, and stats consistency.",
            FactoryErrorType.OBSERVABILITY: "Validate metric/event names, numeric values, event payloads, timing durations, and bounded event-buffer configuration.",
            FactoryErrorType.TIMEOUT: "Retry only if the operation is idempotent and increase timeouts only with supporting telemetry.",
            FactoryErrorType.IO: "Verify filesystem permissions, config paths, and deployment packaging.",
        }
        return fallback_map.get(error_type)

    @staticmethod
    def _map_exception(exc: BaseException, *, operation: Optional[str] = None) -> Optional[Type["FactoryError"]]:
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


# ---------------------------------------------------------------------------
# General, configuration, metadata, and validation errors
# ---------------------------------------------------------------------------
class UnknownFactoryError(FactoryError):
    error_type = FactoryErrorType.UNKNOWN
    default_code = "FAC-1000"
    default_message = "Unknown factory error"
    default_severity = FactoryErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "factory.unknown"


class FactoryConfigurationError(FactoryError):
    error_type = FactoryErrorType.CONFIGURATION
    default_code = "FAC-1100"
    default_message = "Factory configuration error"
    default_severity = FactoryErrorSeverity.HIGH.value
    default_retryable = False
    default_category = "factory.configuration"


class MissingFactoryConfigurationError(FactoryConfigurationError):
    default_code = "FAC-1101"
    default_message = "Required factory configuration is missing"


class InvalidFactoryConfigurationError(FactoryConfigurationError):
    default_code = "FAC-1102"
    default_message = "Factory configuration value is invalid"


class FactoryConfigurationLoadError(FactoryConfigurationError):
    default_code = "FAC-1103"
    default_message = "Factory configuration failed to load"
    default_retryable = True


class MissingFactoryConfigurationSectionError(MissingFactoryConfigurationError):
    default_code = "FAC-1104"
    default_message = "Required factory configuration section is missing"


class FactoryValidationError(FactoryError, ValueError):
    error_type = FactoryErrorType.VALIDATION
    default_code = "FAC-1201"
    default_message = "Factory validation failed"
    default_severity = FactoryErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "factory.validation"


class FactoryTypeError(FactoryValidationError, TypeError):
    default_code = "FAC-1202"
    default_message = "Factory input type is invalid"


class MissingRequiredFieldError(FactoryValidationError):
    default_code = "FAC-1203"
    default_message = "Required factory field is missing"


class InvalidFieldValueError(FactoryValidationError):
    default_code = "FAC-1204"
    default_message = "Factory field value is invalid"


class EmptyCollectionError(FactoryValidationError):
    default_code = "FAC-1205"
    default_message = "Factory collection must not be empty"


class FactoryMetadataError(FactoryError):
    error_type = FactoryErrorType.METADATA
    default_code = "FAC-1210"
    default_message = "Agent metadata error"
    default_severity = FactoryErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "factory.metadata"


class MissingMetadataFieldError(FactoryMetadataError):
    default_code = "FAC-1211"
    default_message = "Required agent metadata field is missing"


class InvalidMetadataFieldError(FactoryMetadataError, ValueError):
    default_code = "FAC-1212"
    default_message = "Agent metadata field is invalid"


class InvalidAgentNameError(InvalidMetadataFieldError):
    default_code = "FAC-1213"
    default_message = "Agent metadata name is invalid"


class InvalidClassNameError(InvalidMetadataFieldError):
    default_code = "FAC-1214"
    default_message = "Agent class name is invalid"


class InvalidModulePathError(InvalidMetadataFieldError):
    error_type = FactoryErrorType.SECURITY
    default_code = "FAC-1215"
    default_message = "Agent module path is invalid"
    default_category = "factory.metadata.module_path"


class InvalidVersionError(InvalidMetadataFieldError):
    error_type = FactoryErrorType.VERSION
    default_code = "FAC-1216"
    default_message = "Agent version is invalid"
    default_category = "factory.metadata.version"


class InvalidDependencySpecError(InvalidMetadataFieldError):
    error_type = FactoryErrorType.DEPENDENCY
    default_code = "FAC-1217"
    default_message = "Agent dependency declaration is invalid"
    default_category = "factory.metadata.dependencies"


# ---------------------------------------------------------------------------
# Registry, dependency graph, resolution, and construction errors
# ---------------------------------------------------------------------------
class FactoryRegistryError(FactoryError):
    error_type = FactoryErrorType.REGISTRY
    default_code = "FAC-1300"
    default_message = "Agent registry error"
    default_severity = FactoryErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "factory.registry"


class DuplicateAgentRegistrationError(FactoryRegistryError):
    default_code = "FAC-1301"
    default_message = "Agent is already registered"


class AgentNotRegisteredError(FactoryRegistryError, KeyError):
    default_code = "FAC-1302"
    default_message = "Agent is not registered"


class AgentVersionUnavailableError(FactoryRegistryError, LookupError):
    error_type = FactoryErrorType.VERSION
    default_code = "FAC-1303"
    default_message = "Requested agent version is unavailable"
    default_category = "factory.registry.version"


class RegistryStateError(FactoryRegistryError):
    error_type = FactoryErrorType.STATE
    default_code = "FAC-1304"
    default_message = "Agent registry state is invalid"
    default_category = "factory.registry.state"


class DependencyGraphError(FactoryRegistryError):
    error_type = FactoryErrorType.DEPENDENCY
    default_code = "FAC-1310"
    default_message = "Agent dependency graph error"
    default_category = "factory.registry.dependencies"


class MissingDependencyError(DependencyGraphError):
    default_code = "FAC-1311"
    default_message = "Required agent dependency is missing"


class CircularDependencyError(DependencyGraphError):
    default_code = "FAC-1312"
    default_message = "Agent dependency graph contains a cycle"


class DependencyResolutionError(DependencyGraphError):
    default_code = "FAC-1313"
    default_message = "Agent dependency resolution failed"
    default_retryable = True


class FactoryResolutionError(FactoryError):
    error_type = FactoryErrorType.RESOLUTION
    default_code = "FAC-1400"
    default_message = "Agent implementation resolution failed"
    default_severity = FactoryErrorSeverity.HIGH.value
    default_retryable = False
    default_category = "factory.resolution"


class AgentModuleImportError(FactoryResolutionError, ImportError):
    error_type = FactoryErrorType.IMPORT
    default_code = "FAC-1401"
    default_message = "Agent module import failed"
    default_category = "factory.resolution.import"


class AgentClassResolutionError(FactoryResolutionError, AttributeError):
    default_code = "FAC-1402"
    default_message = "Agent class could not be resolved from module"


class AgentConstructionError(FactoryResolutionError):
    error_type = FactoryErrorType.CONSTRUCTION
    default_code = "FAC-1403"
    default_message = "Agent construction failed"
    default_retryable = True
    default_category = "factory.construction"


class AgentDependencyInjectionError(AgentConstructionError):
    default_code = "FAC-1404"
    default_message = "Agent dependency injection failed"


# ---------------------------------------------------------------------------
# Orchestration, lifecycle, fallback, and runtime errors
# ---------------------------------------------------------------------------
class FactoryOrchestrationError(FactoryError):
    error_type = FactoryErrorType.ORCHESTRATION
    default_code = "FAC-1500"
    default_message = "Factory orchestration error"
    default_severity = FactoryErrorSeverity.HIGH.value
    default_retryable = True
    default_category = "factory.orchestration"


class AgentSelectionError(FactoryOrchestrationError):
    default_code = "FAC-1501"
    default_message = "Agent selection failed"


class AgentInitializationError(FactoryOrchestrationError):
    error_type = FactoryErrorType.LIFECYCLE
    default_code = "FAC-1502"
    default_message = "Agent initialization failed"
    default_retryable = True
    default_category = "factory.lifecycle.initialization"


class AgentInvocationError(FactoryOrchestrationError):
    default_code = "FAC-1503"
    default_message = "Agent invocation failed"
    default_retryable = True


class AgentLifecycleError(FactoryOrchestrationError):
    error_type = FactoryErrorType.LIFECYCLE
    default_code = "FAC-1504"
    default_message = "Agent lifecycle transition failed"
    default_category = "factory.lifecycle"


class FactoryStateError(FactoryOrchestrationError):
    error_type = FactoryErrorType.STATE
    default_code = "FAC-1505"
    default_message = "Factory state is invalid for this operation"
    default_retryable = False
    default_category = "factory.state"


class FallbackPolicyError(FactoryOrchestrationError):
    default_code = "FAC-1506"
    default_message = "Factory fallback policy failed"


class DegradedAgentError(FactoryOrchestrationError):
    default_code = "FAC-1507"
    default_message = "Agent is running in degraded mode"
    default_severity = FactoryErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "factory.orchestration.degraded"


class FactoryRuntimeError(FactoryError, RuntimeError):
    error_type = FactoryErrorType.RUNTIME
    default_code = "FAC-1999"
    default_message = "Factory runtime error"
    default_severity = FactoryErrorSeverity.HIGH.value
    default_retryable = True
    default_category = "factory.runtime"


# ---------------------------------------------------------------------------
# Metrics adaptation and safety-bound errors
# ---------------------------------------------------------------------------
class MetricsAdapterError(FactoryError):
    error_type = FactoryErrorType.METRICS
    default_code = "FAC-1600"
    default_message = "Metrics adapter error"
    default_severity = FactoryErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "factory.metrics"


class MetricsValidationError(MetricsAdapterError):
    default_code = "FAC-1601"
    default_message = "Metric payload validation failed"
    default_retryable = False


class PIDControlError(MetricsAdapterError):
    error_type = FactoryErrorType.ADAPTATION
    default_code = "FAC-1602"
    default_message = "PID control adjustment failed"
    default_category = "factory.metrics.pid"


class AdaptationRateError(MetricsAdapterError):
    error_type = FactoryErrorType.ADAPTATION
    default_code = "FAC-1603"
    default_message = "Adaptation rate is invalid or unsafe"
    default_category = "factory.metrics.adaptation_rate"


class SafetyBoundError(MetricsAdapterError):
    error_type = FactoryErrorType.SAFETY_BOUNDS
    default_code = "FAC-1604"
    default_message = "Safety-bound enforcement failed"
    default_category = "factory.metrics.safety_bounds"


class FactoryConfigUpdateError(MetricsAdapterError):
    error_type = FactoryErrorType.ADAPTATION
    default_code = "FAC-1605"
    default_message = "Factory-managed runtime config update failed"
    default_category = "factory.metrics.config_update"


class TorchUnavailableError(MetricsAdapterError):
    default_code = "FAC-1606"
    default_message = "Torch is unavailable for metrics adapter operations"
    default_severity = FactoryErrorSeverity.HIGH.value
    default_retryable = True
    default_category = "factory.metrics.dependency"


# ---------------------------------------------------------------------------
# Out-of-process worker and serialization errors
# ---------------------------------------------------------------------------
class RemoteWorkerError(FactoryError):
    error_type = FactoryErrorType.REMOTE_WORKER
    default_code = "FAC-1700"
    default_message = "Remote worker error"
    default_severity = FactoryErrorSeverity.HIGH.value
    default_retryable = True
    default_category = "factory.remote_worker"


class RemoteWorkerPayloadError(RemoteWorkerError):
    default_code = "FAC-1701"
    default_message = "Remote worker payload is invalid"
    default_retryable = False


class RemoteWorkerInvocationError(RemoteWorkerError):
    default_code = "FAC-1702"
    default_message = "Remote worker agent invocation failed"


class RemoteWorkerSerializationError(RemoteWorkerError):
    error_type = FactoryErrorType.SERIALIZATION
    default_code = "FAC-1703"
    default_message = "Remote worker serialization failed"
    default_retryable = False
    default_category = "factory.remote_worker.serialization"


class SubprocessExecutionError(RemoteWorkerError):
    error_type = FactoryErrorType.SUBPROCESS
    default_code = "FAC-1704"
    default_message = "Factory subprocess execution failed"
    default_retryable = True
    default_category = "factory.remote_worker.subprocess"


class RemoteWorkerResultError(RemoteWorkerError):
    default_code = "FAC-1705"
    default_message = "Remote worker returned an invalid result"
    default_retryable = True


# ---------------------------------------------------------------------------
# Cache errors
# ---------------------------------------------------------------------------
class FactoryCacheError(FactoryError):
    error_type = FactoryErrorType.CACHE
    default_code = "FAC-2000"
    default_message = "Factory cache error"
    default_severity = FactoryErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "factory.cache"


class CacheConfigurationError(FactoryCacheError):
    default_code = "FAC-2001"
    default_message = "Factory cache configuration is invalid"
    default_retryable = False


class CacheKeyError(FactoryCacheError, KeyError):
    default_code = "FAC-2002"
    default_message = "Factory cache key is invalid or unavailable"
    default_retryable = False


class CacheTTLError(FactoryCacheError):
    default_code = "FAC-2003"
    default_message = "Factory cache TTL is invalid"
    default_retryable = False


class CacheCapacityError(FactoryCacheError):
    default_code = "FAC-2004"
    default_message = "Factory cache capacity is invalid or exhausted"
    default_retryable = False


class CacheEntryExpiredError(FactoryCacheError):
    default_code = "FAC-2005"
    default_message = "Factory cache entry expired"
    default_severity = FactoryErrorSeverity.LOW.value
    default_retryable = True


class CacheEvictionError(FactoryCacheError):
    default_code = "FAC-2006"
    default_message = "Factory cache eviction failed"
    default_retryable = True


class CacheStatsError(FactoryCacheError):
    default_code = "FAC-2007"
    default_message = "Factory cache statistics are invalid"
    default_retryable = False


# ---------------------------------------------------------------------------
# Observability errors
# ---------------------------------------------------------------------------
class FactoryObservabilityError(FactoryError):
    error_type = FactoryErrorType.OBSERVABILITY
    default_code = "FAC-2100"
    default_message = "Factory observability error"
    default_severity = FactoryErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "factory.observability"


class ObservabilityConfigurationError(FactoryObservabilityError):
    default_code = "FAC-2101"
    default_message = "Factory observability configuration is invalid"
    default_retryable = False


class CounterUpdateError(FactoryObservabilityError):
    default_code = "FAC-2102"
    default_message = "Factory observability counter update failed"
    default_retryable = False


class GaugeUpdateError(FactoryObservabilityError):
    default_code = "FAC-2103"
    default_message = "Factory observability gauge update failed"
    default_retryable = False


class TimingObservationError(FactoryObservabilityError):
    default_code = "FAC-2104"
    default_message = "Factory timing observation failed"
    default_retryable = False


class EventRecordingError(FactoryObservabilityError):
    default_code = "FAC-2105"
    default_message = "Factory observability event recording failed"
    default_retryable = True


class ObservabilitySnapshotError(FactoryObservabilityError):
    default_code = "FAC-2106"
    default_message = "Factory observability snapshot failed"
    default_retryable = True


class ObservabilityResetError(FactoryObservabilityError):
    default_code = "FAC-2107"
    default_message = "Factory observability reset failed"
    default_retryable = True


# ---------------------------------------------------------------------------
# Security, resources, timeout, and IO errors
# ---------------------------------------------------------------------------
class FactorySecurityError(FactoryError):
    error_type = FactoryErrorType.SECURITY
    default_code = "FAC-1800"
    default_message = "Factory security policy error"
    default_severity = FactoryErrorSeverity.CRITICAL.value
    default_retryable = False
    default_category = "factory.security"


class ModulePolicyViolationError(FactorySecurityError):
    default_code = "FAC-1801"
    default_message = "Agent module path violates factory policy"


class UnsafeClassPathError(FactorySecurityError):
    default_code = "FAC-1802"
    default_message = "Agent class path is unsafe"


class FactoryResourceError(FactoryError):
    error_type = FactoryErrorType.RESOURCE
    default_code = "FAC-1900"
    default_message = "Factory resource error"
    default_severity = FactoryErrorSeverity.HIGH.value
    default_retryable = True
    default_category = "factory.resource"


class FactoryTimeoutError(FactoryResourceError, TimeoutError):
    error_type = FactoryErrorType.TIMEOUT
    default_code = "FAC-1901"
    default_message = "Factory operation timed out"
    default_retry_after_seconds = 1.0


class FactoryIOError(FactoryResourceError, OSError):
    error_type = FactoryErrorType.IO
    default_code = "FAC-1902"
    default_message = "Factory IO operation failed"
