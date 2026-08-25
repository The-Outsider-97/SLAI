"""Structured exception taxonomy for the SLAI tuning subsystem.

Failures are explicit, serializable, redacted, and progressively enrichable.
This module deliberately separates exception mechanics from lifecycle records:
runtime code raises ``TuningError`` subclasses, while completed trial artifacts
store the transport-safe ``ErrorRecord`` defined in ``tuning_types.py``.
"""

from __future__ import annotations

import inspect

from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import wraps
from traceback import format_exception
from typing import Any, Callable, Iterable, Mapping, TypeVar, cast
from uuid import uuid4

from .tuning_helpers import *


JSONDict = dict[str, Any]
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_CAUSE_CHAIN_LIMIT = 8
DEFAULT_TRACEBACK_LINE_LIMIT = 80


class TuningErrorCode(str, Enum):
    """Stable machine-readable error codes used across tuning components."""

    CONFIG_ERROR = "TUNING_CONFIG_ERROR"
    CONTRACT_ERROR = "TUNING_CONTRACT_ERROR"
    VALIDATION_ERROR = "TUNING_VALIDATION_ERROR"
    SEARCH_SPACE_ERROR = "TUNING_SEARCH_SPACE_ERROR"
    STRATEGY_ERROR = "TUNING_STRATEGY_ERROR"
    EVALUATION_ERROR = "TUNING_EVALUATION_ERROR"
    TRIAL_ERROR = "TUNING_TRIAL_ERROR"
    LIFECYCLE_ERROR = "TUNING_LIFECYCLE_ERROR"
    CHECKPOINT_ERROR = "TUNING_CHECKPOINT_ERROR"
    OPTIMIZATION_ERROR = "TUNING_OPTIMIZATION_ERROR"
    PROMOTION_ERROR = "TUNING_PROMOTION_ERROR"
    PERSISTENCE_ERROR = "TUNING_PERSISTENCE_ERROR"
    REPORTING_ERROR = "TUNING_REPORTING_ERROR"
    DEPENDENCY_ERROR = "TUNING_DEPENDENCY_ERROR"
    CANCELLED = "TUNING_CANCELLED"
    INTERNAL_ERROR = "TUNING_INTERNAL_ERROR"


class TuningSeverity(str, Enum):
    """Severity hint for logs, tracing, and supervisory agents."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class TuningErrorContext:
    """Immutable causal context attached to a tuning failure."""

    run_id: str | None = None
    trial_id: str | None = None
    component: str | None = None
    operation: str | None = None
    strategy: str | None = None
    model_type: str | None = None
    scenario_id: str | None = None
    seed: int | None = None
    fold_index: int | None = None
    iteration: int | None = None
    config_path: str | None = None
    output_path: str | None = None
    checkpoint_id: str | None = None
    transaction_id: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TuningErrorContext":
        if not isinstance(data, Mapping):
            return cls(metadata={"invalid_context": to_json_safe(data)})
        known = set(cls.__dataclass_fields__)
        payload: dict[str, Any] = {}
        raw_metadata = data.get("metadata", {})
        metadata = (
            dict(raw_metadata)
            if isinstance(raw_metadata, Mapping)
            else {"metadata_value": to_json_safe(raw_metadata)}
        )
        for key, value in data.items():
            if key == "metadata":
                continue
            if key in known:
                payload[key] = value
            else:
                metadata[str(key)] = value
        payload["metadata"] = metadata
        return cls(**payload)

    def merge(
        self, other: "TuningErrorContext | Mapping[str, Any] | None"
    ) -> "TuningErrorContext":
        if other is None:
            return self
        incoming = ensure_context(other)
        base = asdict(self)
        updates = asdict(incoming)
        merged: dict[str, Any] = {}
        for key, base_value in base.items():
            update_value = updates[key]
            if key in {"metadata", "parameters"}:
                merged[key] = {**dict(base_value or {}), **dict(update_value or {})}
            else:
                merged[key] = update_value if update_value is not None else base_value
        return TuningErrorContext(**merged)

    def to_dict(self, *, redact_sensitive: bool = True) -> JSONDict:
        return cast(
            JSONDict,
            to_json_safe(asdict(self), redact_sensitive=redact_sensitive),
        )


class TuningError(Exception):
    """Base exception with stable identity, cause, and safe serialization."""

    default_code = TuningErrorCode.INTERNAL_ERROR
    default_severity = TuningSeverity.ERROR
    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        code: TuningErrorCode | None = None,
        severity: TuningSeverity | None = None,
        retryable: bool | None = None,
        context: TuningErrorContext | Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
        tags: Iterable[str] | None = None,
        error_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        normalized_message = str(message).strip()
        if not normalized_message:
            raise ValueError("TuningError message must be non-empty")
        super().__init__(normalized_message)
        self.message = normalized_message
        self.code = code or self.default_code
        self.severity = severity or self.default_severity
        self.retryable = self.default_retryable if retryable is None else bool(retryable)
        self.context = ensure_context(context)
        self.details: JSONDict = dict(details or {})
        self.cause = cause
        self.tags = _normalize_tags(tags)
        self.error_id = error_id or uuid4().hex
        self.timestamp = timestamp or utc_iso()
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self.message!r}, "
            f"code={self.code.value!r}, severity={self.severity.value!r}, "
            f"retryable={self.retryable!r}, error_id={self.error_id!r})"
        )

    def clone(
        self,
        *,
        message: str | None = None,
        code: TuningErrorCode | None = None,
        severity: TuningSeverity | None = None,
        retryable: bool | None = None,
        context: TuningErrorContext | Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
        tags: Iterable[str] | None = None,
    ) -> "TuningError":
        return self.__class__(
            message or self.message,
            code=code or self.code,
            severity=severity or self.severity,
            retryable=self.retryable if retryable is None else retryable,
            context=self.context.merge(context),
            details={**self.details, **dict(details or {})},
            cause=self.cause if cause is None else cause,
            tags=(*self.tags, *tuple(tags or ())),
            error_id=self.error_id,
            timestamp=self.timestamp,
        )

    def with_context(self, context: TuningErrorContext | Mapping[str, Any] | None) -> "TuningError":
        return self.clone(context=context)

    def with_details(self, details: Mapping[str, Any] | None) -> "TuningError":
        return self.clone(details=details)

    def with_tags(self, tags: Iterable[str] | None) -> "TuningError":
        return self.clone(tags=tags)

    def iter_cause_chain(self, *, limit: int = DEFAULT_CAUSE_CHAIN_LIMIT) -> list[JSONDict]:
        if limit < 1:
            raise ValueError("Cause-chain limit must be positive")
        chain: list[JSONDict] = []
        seen: set[int] = set()
        current = self.cause
        while current is not None and len(chain) < limit and id(current) not in seen:
            seen.add(id(current))
            chain.append(
                {
                    "name": current.__class__.__name__,
                    "message": redact_text(str(current)),
                }
            )
            if isinstance(current, TuningError):
                current = current.cause
            else:
                current = getattr(current, "__cause__", None) or getattr(
                    current, "__context__", None
                )
        return chain

    def to_dict(
        self,
        *,
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
            "details": to_json_safe(
                self.details, redact_sensitive=redact_sensitive
            ),
            "tags": list(self.tags),
        }
        if include_cause and self.cause is not None:
            payload["cause"] = to_json_safe(
                {
                    "name": self.cause.__class__.__name__,
                    "message": str(self.cause),
                },
                redact_sensitive=redact_sensitive,
            )
        if include_cause_chain and self.cause is not None:
            payload["cause_chain"] = to_json_safe(
                self.iter_cause_chain(), redact_sensitive=redact_sensitive
            )
        if include_traceback and self.cause is not None:
            payload["traceback"] = _format_traceback(self.cause)
        return cast(JSONDict, to_json_safe(payload, redact_sensitive=redact_sensitive))

    def to_log_record(
        self, *, redact_sensitive: bool = True, include_traceback: bool = False
    ) -> JSONDict:
        return self.to_dict(
            redact_sensitive=redact_sensitive,
            include_traceback=include_traceback,
        )


class TuningConfigError(TuningError):
    default_code = TuningErrorCode.CONFIG_ERROR


class TuningContractError(TuningError):
    default_code = TuningErrorCode.CONTRACT_ERROR


class TuningValidationError(TuningError):
    default_code = TuningErrorCode.VALIDATION_ERROR


class TuningSearchSpaceError(TuningError):
    default_code = TuningErrorCode.SEARCH_SPACE_ERROR


class TuningStrategyError(TuningError):
    default_code = TuningErrorCode.STRATEGY_ERROR


class TuningEvaluationError(TuningError):
    default_code = TuningErrorCode.EVALUATION_ERROR
    default_retryable = True


class TuningTrialError(TuningError):
    default_code = TuningErrorCode.TRIAL_ERROR


class TuningLifecycleError(TuningError):
    default_code = TuningErrorCode.LIFECYCLE_ERROR


class TuningCheckpointError(TuningLifecycleError):
    default_code = TuningErrorCode.CHECKPOINT_ERROR


class TuningOptimizationError(TuningError):
    default_code = TuningErrorCode.OPTIMIZATION_ERROR


class TuningPromotionError(TuningLifecycleError):
    default_code = TuningErrorCode.PROMOTION_ERROR


class TuningPersistenceError(TuningError):
    default_code = TuningErrorCode.PERSISTENCE_ERROR


class TuningReportingError(TuningError):
    default_code = TuningErrorCode.REPORTING_ERROR


class TuningDependencyError(TuningError):
    default_code = TuningErrorCode.DEPENDENCY_ERROR


class TuningCancelledError(TuningError):
    default_code = TuningErrorCode.CANCELLED
    default_severity = TuningSeverity.INFO


class TuningInternalError(TuningError):
    default_code = TuningErrorCode.INTERNAL_ERROR
    default_severity = TuningSeverity.CRITICAL


def _normalize_tags(tags: Iterable[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tags or ():
        value = str(raw).strip()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _format_traceback(error: BaseException, *, limit: int = DEFAULT_TRACEBACK_LINE_LIMIT) -> list[str]:
    lines = [
        redact_text(line.rstrip("\n"))
        for line in format_exception(type(error), error, error.__traceback__)
    ]
    if len(lines) <= limit:
        return lines
    return [*lines[: limit - 1], "... traceback truncated ..."]


def ensure_context(context: TuningErrorContext | Mapping[str, Any] | Any | None) -> TuningErrorContext:
    """Normalize context without permitting error reporting itself to fail."""

    if context is None:
        return TuningErrorContext()
    if isinstance(context, TuningErrorContext):
        return context
    if isinstance(context, Mapping):
        try:
            return TuningErrorContext.from_mapping(context)
        except Exception as exc:
            return TuningErrorContext(
                metadata={
                    "invalid_context": to_json_safe(context),
                    "normalization_error": str(exc),
                }
            )
    return TuningErrorContext(metadata={"invalid_context": to_json_safe(context)})


def enrich_context(
    base: TuningErrorContext | Mapping[str, Any] | None,
    updates: Mapping[str, Any] | None = None,
) -> TuningErrorContext:
    return ensure_context(base).merge(updates)


def raise_for_condition(
    condition: bool,
    message: str,
    *,
    error_cls: type[TuningError] = TuningValidationError,
    context: TuningErrorContext | Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    severity: TuningSeverity | None = None,
    retryable: bool | None = None,
    tags: Iterable[str] | None = None,
) -> None:
    """Raise a typed tuning error when ``condition`` is true."""

    if not issubclass(error_cls, TuningError):
        raise TypeError("error_cls must derive from TuningError")
    if condition:
        raise error_cls(
            message,
            context=context,
            details=details,
            severity=severity,
            retryable=retryable,
            tags=tags,
        )


def require_dependency(
    value: T | None,
    dependency_name: str,
    *,
    context: TuningErrorContext | Mapping[str, Any] | None = None,
    install_hint: str | None = None,
) -> T:
    """Return an optional dependency or raise a stable dependency error."""

    if value is not None:
        return value
    details: dict[str, Any] = {"dependency": dependency_name}
    if install_hint:
        details["install_hint"] = install_hint
    raise TuningDependencyError(
        f"Required tuning dependency {dependency_name!r} is unavailable.",
        context=context,
        details=details,
    )


def wrap_exception(
    error: BaseException,
    *,
    message: str | None = None,
    error_cls: type[TuningError] = TuningInternalError,
    context: TuningErrorContext | Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    severity: TuningSeverity | None = None,
    retryable: bool | None = None,
    tags: Iterable[str] | None = None,
) -> TuningError:
    """Enrich an existing tuning error or wrap a foreign exception once."""

    if not issubclass(error_cls, TuningError):
        raise TypeError("error_cls must derive from TuningError")
    if isinstance(error, TuningError):
        return error.clone(
            message=message,
            context=context,
            details=details,
            severity=severity,
            retryable=retryable,
            tags=tags,
        )
    return error_cls(
        message or str(error) or error.__class__.__name__,
        context=context,
        details={"original_error_type": error.__class__.__name__, **dict(details or {})},
        severity=severity,
        retryable=retryable,
        cause=error,
        tags=tags,
    )


def error_boundary(
    *,
    error_cls: type[TuningError] = TuningInternalError,
    message: str | None = None,
    context: TuningErrorContext | Mapping[str, Any] | Callable[..., Any] | None = None,
    details: Mapping[str, Any] | Callable[..., Any] | None = None,
    tags: Iterable[str] | None = None,
) -> Callable[[F], F]:
    """Decorate a sync or async function with consistent exception wrapping.

    ``context`` and ``details`` may be callables receiving the wrapped
    function's arguments.  Failures in those diagnostic callables are folded
    into metadata instead of replacing the original exception.
    """

    if not issubclass(error_cls, TuningError):
        raise TypeError("error_cls must derive from TuningError")

    def decorator(function: F) -> F:
        boundary_message = message or f"{function.__qualname__} failed"

        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await function(*args, **kwargs)
                except TuningCancelledError:
                    raise
                except Exception as exc:
                    resolved_context, resolved_details = _resolve_diagnostics(
                        context, details, args, kwargs
                    )
                    raise wrap_exception(
                        exc,
                        message=boundary_message,
                        error_cls=error_cls,
                        context=resolved_context,
                        details=resolved_details,
                        tags=tags,
                    ) from exc

            return cast(F, async_wrapper)

        @wraps(function)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return function(*args, **kwargs)
            except TuningCancelledError:
                raise
            except Exception as exc:
                resolved_context, resolved_details = _resolve_diagnostics(
                    context, details, args, kwargs
                )
                raise wrap_exception(
                    exc,
                    message=boundary_message,
                    error_cls=error_cls,
                    context=resolved_context,
                    details=resolved_details,
                    tags=tags,
                ) from exc

        return cast(F, sync_wrapper)

    return decorator


def _resolve_diagnostics(
    context: TuningErrorContext | Mapping[str, Any] | Callable[..., Any] | None,
    details: Mapping[str, Any] | Callable[..., Any] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[TuningErrorContext, Mapping[str, Any]]:
    diagnostic_errors: dict[str, str] = {}
    raw_context: Any = context
    raw_details: Any = details
    if callable(context):
        try:
            raw_context = context(*args, **kwargs)
        except Exception as exc:
            raw_context = None
            diagnostic_errors["context_factory_error"] = str(exc)
    if callable(details):
        try:
            raw_details = details(*args, **kwargs)
        except Exception as exc:
            raw_details = None
            diagnostic_errors["details_factory_error"] = str(exc)
    resolved_context = ensure_context(raw_context)
    if diagnostic_errors:
        resolved_context = resolved_context.merge({"metadata": diagnostic_errors})
    if raw_details is None:
        resolved_details: Mapping[str, Any] = {}
    elif isinstance(raw_details, Mapping):
        resolved_details = raw_details
    else:
        resolved_details = {"invalid_details": to_json_safe(raw_details)}
    return resolved_context, resolved_details


__all__ = [
    "TuningCancelledError",
    "TuningCheckpointError",
    "TuningConfigError",
    "TuningContractError",
    "TuningDependencyError",
    "TuningError",
    "TuningErrorCode",
    "TuningErrorContext",
    "TuningEvaluationError",
    "TuningInternalError",
    "TuningLifecycleError",
    "TuningOptimizationError",
    "TuningPersistenceError",
    "TuningPromotionError",
    "TuningReportingError",
    "TuningSearchSpaceError",
    "TuningSeverity",
    "TuningStrategyError",
    "TuningTrialError",
    "TuningValidationError",
    "enrich_context",
    "ensure_context",
    "error_boundary",
    "raise_for_condition",
    "require_dependency",
    "wrap_exception",
]
