"""
Buffer Error Handling Module

Production-ready exception hierarchy for the complete buffer subsystem.

This module centralises all buffer-facing errors and warnings for:
- replay buffers and distributed replay
- reservoir replay
- sequence replay and segment tree primitives
- n-step transition processing
- transition validation/coercion
- network buffering, backpressure, TTL, ack/nack flows, and fairness keys
- eviction policy selection
- configuration loading/reloading
- telemetry, metric summarisation, and fairness checks
- persistence and bulk operations

Design principles:
1. Backward compatible: keeps the existing public exception names.
2. Structured: every BufferError can expose deterministic metadata via to_dict().
3. Lightweight: no logger imports and no side effects at import time.
4. Recoverable-aware: errors can tell callers whether retry/recovery is realistic.
5. Module-specific: each subsystem has concrete errors instead of generic ValueError.
"""

from __future__ import annotations

import math
import numpy as np # type: ignore

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union


# -----------------------------------------------------------------------------
# Metadata primitives
# -----------------------------------------------------------------------------

class BufferCategory(str, Enum):
    """Coarse error domains used for reporting, filtering, and telemetry."""

    BASE = "base"
    CONFIG = "config"
    VALIDATION = "validation"
    CAPACITY = "capacity"
    STATE = "state"
    SAMPLING = "sampling"
    REPLAY = "replay"
    RESERVOIR = "reservoir"
    NETWORK = "network"
    NSTEP = "nstep"
    SEGMENT_TREE = "segment_tree"
    EVICTION = "eviction"
    SEQUENCE = "sequence"
    TELEMETRY = "telemetry"
    FAIRNESS = "fairness"
    PERSISTENCE = "persistence"
    OPERATION = "operation"


class BufferSeverity(str, Enum):
    """Operational severity for exception consumers."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BufferErrorCode(str, Enum):
    """Stable symbolic error codes for buffer subsystem failures."""

    BUFFER_ERROR = "BUFFER_ERROR"
    BUFFER_WARNING = "BUFFER_WARNING"

    CONFIG_ERROR = "BUFFER_CONFIG_ERROR"
    CONFIG_FILE_NOT_FOUND = "BUFFER_CONFIG_FILE_NOT_FOUND"
    CONFIG_PATH_ERROR = "BUFFER_CONFIG_PATH_ERROR"
    CONFIG_PARSE_ERROR = "BUFFER_CONFIG_PARSE_ERROR"
    CONFIG_SECTION_INVALID = "BUFFER_CONFIG_SECTION_INVALID"
    CONFIG_VALUE_MISSING = "BUFFER_CONFIG_VALUE_MISSING"
    CONFIG_VALUE_INVALID = "BUFFER_CONFIG_VALUE_INVALID"
    CONFIG_RELOAD_FAILED = "BUFFER_CONFIG_RELOAD_FAILED"

    TRANSITION_INVALID = "BUFFER_TRANSITION_INVALID"
    TRANSITION_TYPE_INVALID = "BUFFER_TRANSITION_TYPE_INVALID"
    TRANSITION_LENGTH_INVALID = "BUFFER_TRANSITION_LENGTH_INVALID"
    TRANSITION_REWARD_INVALID = "BUFFER_TRANSITION_REWARD_INVALID"
    TRANSITION_DONE_INVALID = "BUFFER_TRANSITION_DONE_INVALID"
    TRANSITION_STATE_NONE = "BUFFER_TRANSITION_STATE_NONE"
    TRANSITION_SCHEMA_INVALID = "BUFFER_TRANSITION_SCHEMA_INVALID"
    TRANSITION_COERCION_FAILED = "BUFFER_TRANSITION_COERCION_FAILED"
    TRANSITION_BATCH_INVALID = "BUFFER_TRANSITION_BATCH_INVALID"

    CAPACITY_ERROR = "BUFFER_CAPACITY_ERROR"
    BUFFER_FULL = "BUFFER_FULL"
    BUFFER_EMPTY = "BUFFER_EMPTY"
    INSUFFICIENT_SAMPLES = "BUFFER_INSUFFICIENT_SAMPLES"
    INVALID_BATCH_SIZE = "BUFFER_INVALID_BATCH_SIZE"
    INDEX_OUT_OF_BOUNDS = "BUFFER_INDEX_OUT_OF_BOUNDS"
    STATE_ERROR = "BUFFER_STATE_ERROR"
    BUFFER_CLOSED = "BUFFER_CLOSED"
    LOCK_TIMEOUT = "BUFFER_LOCK_TIMEOUT"
    MUTATION_ERROR = "BUFFER_MUTATION_ERROR"

    REPLAY_ERROR = "BUFFER_REPLAY_ERROR"
    DISTRIBUTED_REPLAY_ERROR = "BUFFER_DISTRIBUTED_REPLAY_ERROR"
    RESERVOIR_ERROR = "BUFFER_RESERVOIR_ERROR"
    SAMPLING_ERROR = "BUFFER_SAMPLING_ERROR"
    SAMPLING_STRATEGY_INVALID = "BUFFER_SAMPLING_STRATEGY_INVALID"
    PRIORITY_SAMPLING_FAILED = "BUFFER_PRIORITY_SAMPLING_FAILED"
    PRIORITY_UPDATE_FAILED = "BUFFER_PRIORITY_UPDATE_FAILED"
    PRIORITY_MASS_INVALID = "BUFFER_PRIORITY_MASS_INVALID"
    AGENT_DISTRIBUTION_INVALID = "BUFFER_AGENT_DISTRIBUTION_INVALID"
    STALE_EXPERIENCE = "BUFFER_STALE_EXPERIENCE"

    PERSISTENCE_ERROR = "BUFFER_PERSISTENCE_ERROR"
    SAVE_FAILED = "BUFFER_SAVE_FAILED"
    LOAD_FAILED = "BUFFER_LOAD_FAILED"
    SERIALIZATION_FAILED = "BUFFER_SERIALIZATION_FAILED"

    NETWORK_ERROR = "BUFFER_NETWORK_ERROR"
    NETWORK_MESSAGE_INVALID = "BUFFER_NETWORK_MESSAGE_INVALID"
    MESSAGE_EXPIRED = "BUFFER_MESSAGE_EXPIRED"
    MESSAGE_NOT_FOUND = "BUFFER_MESSAGE_NOT_FOUND"
    MESSAGE_DUPLICATE = "BUFFER_MESSAGE_DUPLICATE"
    FAIRNESS_KEY_INVALID = "BUFFER_FAIRNESS_KEY_INVALID"
    FAIRNESS_SCHEDULING_FAILED = "BUFFER_FAIRNESS_SCHEDULING_FAILED"
    DROP_STRATEGY_INVALID = "BUFFER_DROP_STRATEGY_INVALID"
    BACKPRESSURE = "BUFFER_BACKPRESSURE"
    BACKPRESSURE_REJECTED = "BUFFER_BACKPRESSURE_REJECTED"
    INFLIGHT_LIMIT_REACHED = "BUFFER_INFLIGHT_LIMIT_REACHED"
    TTL_INVALID = "BUFFER_TTL_INVALID"
    ACK_FAILED = "BUFFER_ACK_FAILED"
    NACK_FAILED = "BUFFER_NACK_FAILED"

    NSTEP_ERROR = "BUFFER_NSTEP_ERROR"
    NSTEP_CONFIG_INVALID = "BUFFER_NSTEP_CONFIG_INVALID"
    NSTEP_WINDOW_INVALID = "BUFFER_NSTEP_WINDOW_INVALID"
    NSTEP_COMPUTATION_FAILED = "BUFFER_NSTEP_COMPUTATION_FAILED"
    NSTEP_TERMINAL_STATE_INVALID = "BUFFER_NSTEP_TERMINAL_STATE_INVALID"

    SEGMENT_TREE_ERROR = "BUFFER_SEGMENT_TREE_ERROR"
    SEGMENT_TREE_CAPACITY_INVALID = "BUFFER_SEGMENT_TREE_CAPACITY_INVALID"
    SEGMENT_TREE_INDEX_INVALID = "BUFFER_SEGMENT_TREE_INDEX_INVALID"
    SEGMENT_TREE_RANGE_INVALID = "BUFFER_SEGMENT_TREE_RANGE_INVALID"
    SEGMENT_TREE_PREFIXSUM_INVALID = "BUFFER_SEGMENT_TREE_PREFIXSUM_INVALID"
    SEGMENT_TREE_MASS_INVALID = "BUFFER_SEGMENT_TREE_MASS_INVALID"
    SEGMENT_TREE_OPERATION_FAILED = "BUFFER_SEGMENT_TREE_OPERATION_FAILED"

    EVICTION_ERROR = "BUFFER_EVICTION_ERROR"
    EVICTION_CONTEXT_INVALID = "BUFFER_EVICTION_CONTEXT_INVALID"
    EVICTION_POLICY_INVALID = "BUFFER_EVICTION_POLICY_INVALID"
    EVICTION_POLICY_UNSUPPORTED = "BUFFER_EVICTION_POLICY_UNSUPPORTED"
    EVICTION_SELECTION_FAILED = "BUFFER_EVICTION_SELECTION_FAILED"

    SEQUENCE_REPLAY_ERROR = "BUFFER_SEQUENCE_REPLAY_ERROR"
    SEQUENCE_LENGTH_INVALID = "BUFFER_SEQUENCE_LENGTH_INVALID"
    SEQUENCE_ASSEMBLY_FAILED = "BUFFER_SEQUENCE_ASSEMBLY_FAILED"
    SEQUENCE_PADDING_FAILED = "BUFFER_SEQUENCE_PADDING_FAILED"

    TELEMETRY_ERROR = "BUFFER_TELEMETRY_ERROR"
    TELEMETRY_METRIC_ERROR = "BUFFER_TELEMETRY_METRIC_ERROR"
    METRIC_VALUE_INVALID = "BUFFER_METRIC_VALUE_INVALID"
    METRIC_SNAPSHOT_FAILED = "BUFFER_METRIC_SNAPSHOT_FAILED"
    FAIRNESS_METRIC_ERROR = "BUFFER_FAIRNESS_METRIC_ERROR"
    FAIRNESS_VIOLATION = "BUFFER_FAIRNESS_VIOLATION"

    OPERATION_ERROR = "BUFFER_OPERATION_ERROR"
    PARTIAL_FAILURE = "BUFFER_PARTIAL_FAILURE"


@dataclass(frozen=True)
class BufferRecoveryHint:
    """Human/test friendly recovery hint attached to an error."""

    action: str
    retryable: bool = False
    fallback: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "retryable": self.retryable,
            "fallback": self.fallback,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class BufferErrorContext:
    """Structured context carried by all buffer exceptions."""

    component: Optional[str] = None
    operation: Optional[str] = None
    category: BufferCategory = BufferCategory.BASE
    item_id: Optional[str] = None
    index: Optional[int] = None
    field_name: Optional[str] = None
    config_section: Optional[str] = None
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "operation": self.operation,
            "category": self.category.value,
            "item_id": self.item_id,
            "index": self.index,
            "field_name": self.field_name,
            "config_section": self.config_section,
            "correlation_id": self.correlation_id,
            "details": dict(self.details),
            "created_at": self.created_at,
        }


def build_error_context(
    *,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    category: Union[BufferCategory, str] = BufferCategory.BASE,
    item_id: Optional[str] = None,
    index: Optional[int] = None,
    field_name: Optional[str] = None,
    config_section: Optional[str] = None,
    correlation_id: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> BufferErrorContext:
    """Build a normalized BufferErrorContext from loose caller metadata."""
    resolved_category = category if isinstance(category, BufferCategory) else BufferCategory(str(category))
    return BufferErrorContext(
        component=component,
        operation=operation,
        category=resolved_category,
        item_id=item_id,
        index=index,
        field_name=field_name,
        config_section=config_section,
        correlation_id=correlation_id,
        details=dict(details or {}),
    )


def _safe_type_name(value: Any) -> str:
    return type(value).__name__


def _safe_value(value: Any, *, max_len: int = 160) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{_safe_type_name(value)} repr unavailable>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _normalize_code(code: Union[BufferErrorCode, str]) -> BufferErrorCode:
    if isinstance(code, BufferErrorCode):
        return code
    try:
        return BufferErrorCode(str(code))
    except ValueError:
        return BufferErrorCode.BUFFER_ERROR


def _normalize_severity(severity: Union[BufferSeverity, str]) -> BufferSeverity:
    if isinstance(severity, BufferSeverity):
        return severity
    try:
        return BufferSeverity(str(severity).lower())
    except ValueError:
        return BufferSeverity.ERROR


# -----------------------------------------------------------------------------
# Base classes
# -----------------------------------------------------------------------------

class BufferError(Exception):
    """Base class for all buffer-related exceptions.

    Args:
        message: Human-readable error message.
        code: Stable symbolic code for programmatic handling.
        context: Optional structured subsystem context.
        severity: Operational severity.
        recoverable: Whether retry/fallback may be appropriate.
        cause: Original exception when this error wraps another failure.
        recovery_hint: Optional suggested remediation.
        details: Extra structured metadata merged into context.details.
    """

    default_code = BufferErrorCode.BUFFER_ERROR
    default_category = BufferCategory.BASE
    default_severity = BufferSeverity.ERROR
    default_recoverable = False

    def __init__(
        self,
        message: str,
        *args: Any,
        code: Union[BufferErrorCode, str, None] = None,
        context: Optional[BufferErrorContext] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        severity: Union[BufferSeverity, str, None] = None,
        recoverable: Optional[bool] = None,
        cause: Optional[BaseException] = None,
        recovery_hint: Optional[BufferRecoveryHint] = None,
        details: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ) -> None:
        self.message = str(message)
        self.code = _normalize_code(code or self.default_code)
        self.severity = _normalize_severity(severity or self.default_severity)
        self.recoverable = self.default_recoverable if recoverable is None else bool(recoverable)
        self.cause = cause
        self.recovery_hint = recovery_hint

        merged_details = dict(context.details if context else {})
        merged_details.update(dict(details or {}))
        self.context = context or BufferErrorContext(
            component=component,
            operation=operation,
            category=self.default_category,
            details=merged_details,
        )
        if context and (component or operation or details):
            self.context = BufferErrorContext(
                component=component or context.component,
                operation=operation or context.operation,
                category=context.category,
                item_id=context.item_id,
                index=context.index,
                field_name=context.field_name,
                config_section=context.config_section,
                correlation_id=context.correlation_id,
                details=merged_details,
                created_at=context.created_at,
            )

        super().__init__(self.message, *args)

    @property
    def category(self) -> BufferCategory:
        return self.context.category

    def to_dict(self, *, include_cause: bool = True) -> Dict[str, Any]:
        payload = {
            "error_type": type(self).__name__,
            "message": self.message,
            "code": self.code.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "context": self.context.to_dict(),
            "recovery_hint": self.recovery_hint.to_dict() if self.recovery_hint else None,
        }
        if include_cause and self.cause is not None:
            payload["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }
        return payload

    def brief(self) -> str:
        comp = f"[{self.context.component}] " if self.context.component else ""
        op = f" during {self.context.operation}" if self.context.operation else ""
        return f"{comp}{self.code.value}{op}: {self.message}"

    def with_detail(self, key: str, value: Any) -> "BufferError":
        updated = dict(self.context.details)
        updated[str(key)] = value
        self.context = BufferErrorContext(
            component=self.context.component,
            operation=self.context.operation,
            category=self.context.category,
            item_id=self.context.item_id,
            index=self.context.index,
            field_name=self.context.field_name,
            config_section=self.context.config_section,
            correlation_id=self.context.correlation_id,
            details=updated,
            created_at=self.context.created_at,
        )
        return self


class BufferWarning(Warning):
    """Base warning for non-fatal buffer subsystem issues."""

    default_code = BufferErrorCode.BUFFER_WARNING
    default_category = BufferCategory.BASE
    default_severity = BufferSeverity.WARNING

    def __init__(
        self,
        message: str,
        *,
        code: Union[BufferErrorCode, str, None] = None,
        context: Optional[BufferErrorContext] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.message = str(message)
        self.code = _normalize_code(code or self.default_code)
        merged_details = dict(context.details if context else {})
        merged_details.update(dict(details or {}))
        self.context = context or BufferErrorContext(category=self.default_category, details=merged_details)
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warning_type": type(self).__name__,
            "message": self.message,
            "code": self.code.value,
            "category": self.context.category.value,
            "severity": self.default_severity.value,
            "context": self.context.to_dict(),
        }


# -----------------------------------------------------------------------------
# Configuration errors
# -----------------------------------------------------------------------------

class BufferConfigError(BufferError):
    """Raised when buffer configuration is invalid, missing, or inaccessible."""

    default_code = BufferErrorCode.CONFIG_ERROR
    default_category = BufferCategory.CONFIG


class ConfigFileNotFoundError(BufferConfigError, FileNotFoundError):
    """Raised when the expected buffer config file does not exist."""

    default_code = BufferErrorCode.CONFIG_FILE_NOT_FOUND
    default_recoverable = True

    def __init__(self, path: Union[str, Any], message: Optional[str] = None, **kwargs: Any) -> None:
        self.path = str(path)
        super().__init__(
            message or f"Buffer config file not found: {self.path}",
            recovery_hint=BufferRecoveryHint(
                action="Verify the config path or provide an explicit config_path.",
                retryable=True,
            ),
            details={"path": self.path},
            **kwargs,
        )


class ConfigPathError(BufferConfigError):
    """Raised when a config path cannot be resolved or is invalid."""

    default_code = BufferErrorCode.CONFIG_PATH_ERROR

    def __init__(self, path: Any, reason: str, **kwargs: Any) -> None:
        self.path = path
        self.reason = reason
        super().__init__(
            f"Invalid config path {_safe_value(path)}: {reason}",
            details={"path": _safe_value(path), "reason": reason},
            **kwargs,
        )


class ConfigParseError(BufferConfigError, ValueError):
    """Raised when the buffer YAML config cannot be parsed as a mapping."""

    default_code = BufferErrorCode.CONFIG_PARSE_ERROR

    def __init__(self, path: Union[str, Any], reason: str, cause: Optional[BaseException] = None, **kwargs: Any) -> None:
        self.path = str(path)
        self.reason = reason
        super().__init__(
            f"Failed to parse buffer config {self.path}: {reason}",
            cause=cause,
            details={"path": self.path, "reason": reason},
            **kwargs,
        )


class InvalidConfigSectionError(BufferConfigError, KeyError):
    """Raised when a required config section is malformed or absent."""

    default_code = BufferErrorCode.CONFIG_SECTION_INVALID

    def __init__(
        self,
        section: str,
        expected: str = "mapping",
        actual: Any = None,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.section = str(section)
        self.expected = expected
        self.actual = actual
        actual_text = "missing" if actual is None else _safe_type_name(actual)
        super().__init__(
            message or f"Invalid config section '{self.section}': expected {expected}, got {actual_text}",
            context=build_error_context(category=BufferCategory.CONFIG, config_section=self.section),
            details={"section": self.section, "expected": expected, "actual_type": actual_text},
            **kwargs,
        )


class MissingConfigValueError(BufferConfigError, KeyError):
    """Raised when a required config key is missing."""

    default_code = BufferErrorCode.CONFIG_VALUE_MISSING

    def __init__(self, section: str, key: str, **kwargs: Any) -> None:
        self.section = str(section)
        self.key = str(key)
        super().__init__(
            f"Missing required config value '{self.section}.{self.key}'",
            context=build_error_context(category=BufferCategory.CONFIG, config_section=self.section, field_name=self.key),
            details={"section": self.section, "key": self.key},
            **kwargs,
        )


class ConfigValueError(BufferConfigError, ValueError):
    """Raised when a configuration value is outside the acceptable type/range."""

    default_code = BufferErrorCode.CONFIG_VALUE_INVALID

    def __init__(
        self,
        param_name: str,
        value: Any,
        expected: str,
        message: Optional[str] = None,
        *,
        section: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.param_name = str(param_name)
        self.value = value
        self.expected = expected
        msg = message or f"Invalid config value for '{self.param_name}': {_safe_value(value)} (expected {expected})"
        super().__init__(
            msg,
            context=build_error_context(
                category=BufferCategory.CONFIG,
                config_section=section,
                field_name=self.param_name,
            ),
            details={
                "param_name": self.param_name,
                "value": _safe_value(value),
                "actual_type": _safe_type_name(value),
                "expected": expected,
                "section": section,
            },
            **kwargs,
        )


class ConfigReloadError(BufferConfigError):
    """Raised when reloading configuration fails."""

    default_code = BufferErrorCode.CONFIG_RELOAD_FAILED
    default_recoverable = True

    def __init__(self, path: Optional[Union[str, Any]] = None, reason: str = "reload failed", **kwargs: Any) -> None:
        self.path = None if path is None else str(path)
        self.reason = reason
        super().__init__(
            f"Buffer config reload failed{f' for {self.path}' if self.path else ''}: {reason}",
            recovery_hint=BufferRecoveryHint(action="Retry config load or fall back to cached config.", retryable=True),
            details={"path": self.path, "reason": reason},
            **kwargs,
        )


# -----------------------------------------------------------------------------
# Transition / data validation errors
# -----------------------------------------------------------------------------

class TransitionValidationError(BufferError, ValueError):
    """Raised when a transition payload does not conform to the expected schema."""

    default_code = BufferErrorCode.TRANSITION_INVALID
    default_category = BufferCategory.VALIDATION

    def __init__(
        self,
        message: str = "Invalid transition payload",
        *,
        field_name: Optional[str] = None,
        value: Any = None,
        expected: Optional[str] = None,
        transition_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.field_name = field_name
        self.value = value
        self.expected = expected
        self.transition_index = transition_index
        extra_details = dict(kwargs.pop("details", {}) or {})
        extra_details.update({
            "field_name": field_name,
            "value": _safe_value(value) if value is not None else None,
            "actual_type": _safe_type_name(value) if value is not None else None,
            "expected": expected,
            "transition_index": transition_index,
        })
        super().__init__(
            message,
            context=build_error_context(
                category=BufferCategory.VALIDATION,
                field_name=field_name,
                index=transition_index,
            ),
            details=extra_details,
            **kwargs,
        )


class TransitionTypeError(TransitionValidationError, TypeError):
    """Raised when a transition is not a tuple/list-compatible payload."""

    default_code = BufferErrorCode.TRANSITION_TYPE_INVALID

    def __init__(self, transition: Any, expected: str = "tuple or list", **kwargs: Any) -> None:
        self.transition = transition
        super().__init__(
            f"Transition must be {expected}, got {_safe_type_name(transition)}.",
            value=transition,
            expected=expected,
            **kwargs,
        )


class TransitionLengthError(TransitionValidationError):
    """Raised when a transition tuple/list has the wrong number of elements."""

    default_code = BufferErrorCode.TRANSITION_LENGTH_INVALID

    def __init__(self, expected: int, actual: int, **kwargs: Any) -> None:
        self.expected = int(expected)
        self.actual = int(actual)
        super().__init__(
            f"Transition length must be {self.expected}, got {self.actual}.",
            field_name="transition",
            value=self.actual,
            expected=str(self.expected),
            **kwargs,
        )


class TransitionRewardError(TransitionValidationError):
    """Raised when reward is non-numeric or exceeds the allowed absolute range."""

    default_code = BufferErrorCode.TRANSITION_REWARD_INVALID

    def __init__(self, reward: Any, max_abs: Optional[float] = None, reason: Optional[str] = None, **kwargs: Any) -> None:
        self.reward = reward
        self.max_abs = max_abs
        if reason is None:
            reason = f"Reward must be numeric, got {_safe_type_name(reward)}"
            if max_abs is not None:
                reason += f" and abs(reward) must be <= {max_abs}"
        super().__init__(
            reason,
            field_name="reward",
            value=reward,
            expected="numeric float-compatible value",
            details={"max_abs": max_abs},
            **kwargs,
        )


class TransitionDoneError(TransitionValidationError):
    """Raised when the done flag is not boolean."""

    default_code = BufferErrorCode.TRANSITION_DONE_INVALID

    def __init__(self, done: Any, **kwargs: Any) -> None:
        self.done = done
        super().__init__(
            f"Done flag must be boolean, got {_safe_type_name(done)}.",
            field_name="done",
            value=done,
            expected="bool",
            **kwargs,
        )


class TransitionNoneStateError(TransitionValidationError):
    """Raised when state or next_state is None but schema disallows it."""

    default_code = BufferErrorCode.TRANSITION_STATE_NONE

    def __init__(self, field_name: str, **kwargs: Any) -> None:
        self.field_name = str(field_name)
        super().__init__(
            f"{self.field_name} cannot be None.",
            field_name=self.field_name,
            value=None,
            expected="non-None state payload",
            **kwargs,
        )


class TransitionSchemaError(TransitionValidationError):
    """Raised when the transition schema itself is invalid."""

    default_code = BufferErrorCode.TRANSITION_SCHEMA_INVALID

    def __init__(self, reason: str, schema: Any = None, **kwargs: Any) -> None:
        self.reason = reason
        self.schema = schema
        super().__init__(
            f"Invalid transition schema: {reason}",
            value=schema,
            expected="valid TransitionSchema",
            **kwargs,
        )


class TransitionCoercionError(TransitionValidationError):
    """Raised when coercion into the canonical transition shape fails."""

    default_code = BufferErrorCode.TRANSITION_COERCION_FAILED

    def __init__(self, field_name: str, value: Any, target_type: str, reason: str, **kwargs: Any) -> None:
        self.target_type = target_type
        self.reason = reason
        super().__init__(
            f"Could not coerce transition field '{field_name}' to {target_type}: {reason}",
            field_name=field_name,
            value=value,
            expected=target_type,
            **kwargs,
        )


class TransitionBatchValidationError(TransitionValidationError):
    """Raised when bulk validation has invalid transition entries."""

    default_code = BufferErrorCode.TRANSITION_BATCH_INVALID

    def __init__(
        self,
        invalid_count: int,
        total_count: int,
        errors: Optional[Sequence[Union[str, BaseException]]] = None,
        **kwargs: Any,
    ) -> None:
        self.invalid_count = int(invalid_count)
        self.total_count = int(total_count)
        self.errors = list(errors or [])
        super().__init__(
            f"Transition batch validation failed: {self.invalid_count}/{self.total_count} invalid.",
            details={"invalid_count": self.invalid_count, "total_count": self.total_count, "errors": summarize_errors(self.errors)},
            **kwargs,
        )


# -----------------------------------------------------------------------------
# Capacity, storage, and lifecycle errors
# -----------------------------------------------------------------------------

class BufferCapacityError(BufferError, ValueError):
    """Raised when a buffer operation fails due to capacity constraints."""

    default_code = BufferErrorCode.CAPACITY_ERROR
    default_category = BufferCategory.CAPACITY


class BufferFullError(BufferCapacityError):
    """Raised when an operation requires free space but the buffer is full."""

    default_code = BufferErrorCode.BUFFER_FULL
    default_recoverable = True

    def __init__(self, capacity: int, operation: str = "push", current_size: Optional[int] = None, **kwargs: Any) -> None:
        self.capacity = int(capacity)
        self.operation = str(operation)
        self.current_size = current_size
        super().__init__(
            f"Cannot {self.operation} - buffer at capacity {self.capacity}.",
            operation=self.operation,
            recovery_hint=BufferRecoveryHint(
                action="Evict, increase capacity, or use a drop/reject strategy.",
                retryable=True,
                fallback="eviction_policy",
            ),
            details={"capacity": self.capacity, "current_size": current_size},
            **kwargs,
        )


class BufferEmptyError(BufferCapacityError):
    """Raised when sampling, dequeue, ack, or pop is attempted on an empty buffer."""

    default_code = BufferErrorCode.BUFFER_EMPTY
    default_recoverable = True

    def __init__(self, operation: str = "sample", **kwargs: Any) -> None:
        self.operation = str(operation)
        super().__init__(
            f"Cannot {self.operation} - buffer is empty.",
            operation=self.operation,
            recovery_hint=BufferRecoveryHint(action="Wait for producers to add data before retrying.", retryable=True),
            **kwargs,
        )


class InsufficientSamplesError(BufferCapacityError):
    """Raised when requested batch size exceeds available samples."""

    default_code = BufferErrorCode.INSUFFICIENT_SAMPLES
    default_recoverable = True

    def __init__(self, requested: int, available: int, replace: bool = False, **kwargs: Any) -> None:
        self.requested = int(requested)
        self.available = int(available)
        self.replace = bool(replace)
        msg = f"Requested {self.requested} samples but only {self.available} available"
        if not self.replace:
            msg += " (replace=False)"
        super().__init__(
            msg,
            operation="sample",
            recovery_hint=BufferRecoveryHint(
                action="Reduce batch_size, enable replacement, or wait for more samples.",
                retryable=True,
            ),
            details={"requested": self.requested, "available": self.available, "replace": self.replace},
            **kwargs,
        )


class InvalidBatchSizeError(BufferCapacityError):
    """Raised when batch_size is zero, negative, non-integer, or otherwise invalid."""

    default_code = BufferErrorCode.INVALID_BATCH_SIZE

    def __init__(self, batch_size: Any, reason: str = "batch_size must be a positive integer", **kwargs: Any) -> None:
        self.batch_size = batch_size
        self.reason = reason
        super().__init__(
            f"Invalid batch_size {_safe_value(batch_size)}: {reason}.",
            operation="sample",
            details={"batch_size": _safe_value(batch_size), "reason": reason},
            **kwargs,
        )


class IndexOutOfBoundsError(BufferError, IndexError):
    """Raised when an index is outside the valid range."""

    default_code = BufferErrorCode.INDEX_OUT_OF_BOUNDS
    default_category = BufferCategory.STATE

    def __init__(self, index: int, size: int, operation: str = "access", **kwargs: Any) -> None:
        self.index = int(index)
        self.size = int(size)
        self.operation = str(operation)
        super().__init__(
            f"Index {self.index} out of bounds [0, {self.size}) for operation '{self.operation}'.",
            operation=self.operation,
            context=build_error_context(category=BufferCategory.STATE, operation=self.operation, index=self.index),
            details={"index": self.index, "size": self.size},
            **kwargs,
        )


class BufferStateError(BufferError, RuntimeError):
    """Raised when buffer internal state is inconsistent for the requested operation."""

    default_code = BufferErrorCode.STATE_ERROR
    default_category = BufferCategory.STATE


class BufferClosedError(BufferStateError):
    """Raised when an operation is attempted after the buffer was closed/shut down."""

    default_code = BufferErrorCode.BUFFER_CLOSED

    def __init__(self, operation: str = "operate", **kwargs: Any) -> None:
        self.operation = str(operation)
        super().__init__(f"Cannot {self.operation}: buffer is closed.", operation=self.operation, **kwargs)


class BufferLockTimeoutError(BufferStateError, TimeoutError):
    """Raised when a buffer lock cannot be acquired within the configured timeout."""

    default_code = BufferErrorCode.LOCK_TIMEOUT
    default_recoverable = True

    def __init__(self, operation: str, timeout_seconds: float, **kwargs: Any) -> None:
        self.operation = str(operation)
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(
            f"Timed out after {self.timeout_seconds:.3f}s while acquiring buffer lock for '{self.operation}'.",
            operation=self.operation,
            recovery_hint=BufferRecoveryHint(action="Retry the operation or inspect lock contention.", retryable=True),
            details={"timeout_seconds": self.timeout_seconds},
            **kwargs,
        )


class BufferMutationError(BufferStateError):
    """Raised when mutation leaves buffer indexes, metadata, or statistics inconsistent."""

    default_code = BufferErrorCode.MUTATION_ERROR

    def __init__(self, operation: str, reason: str, **kwargs: Any) -> None:
        self.operation = str(operation)
        self.reason = reason
        super().__init__(
            f"Buffer mutation failed during '{self.operation}': {reason}",
            operation=self.operation,
            details={"reason": reason},
            **kwargs,
        )


# -----------------------------------------------------------------------------
# Replay, sampling, and priority errors
# -----------------------------------------------------------------------------

class ReplayBufferError(BufferError):
    """Base error for replay buffer implementations."""

    default_code = BufferErrorCode.REPLAY_ERROR
    default_category = BufferCategory.REPLAY


class DistributedReplayBufferError(ReplayBufferError):
    """Base error for distributed replay buffer failures."""

    default_code = BufferErrorCode.DISTRIBUTED_REPLAY_ERROR


class ReservoirBufferError(ReplayBufferError):
    """Base error for reservoir replay failures."""

    default_code = BufferErrorCode.RESERVOIR_ERROR
    default_category = BufferCategory.RESERVOIR


class SamplingError(ReplayBufferError, ValueError):
    """Base for errors related to sampling strategies."""

    default_code = BufferErrorCode.SAMPLING_ERROR
    default_category = BufferCategory.SAMPLING


class InvalidSamplingStrategyError(SamplingError):
    """Raised when an unknown sampling strategy is requested."""

    default_code = BufferErrorCode.SAMPLING_STRATEGY_INVALID

    def __init__(self, strategy: str, valid_strategies: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        self.strategy = str(strategy)
        self.valid_strategies = list(valid_strategies or [])
        msg = f"Unknown sampling strategy: '{self.strategy}'"
        if self.valid_strategies:
            msg += f". Valid: {self.valid_strategies}"
        super().__init__(msg, details={"strategy": self.strategy, "valid_strategies": self.valid_strategies}, **kwargs)


class PrioritySamplingError(SamplingError):
    """Raised when prioritized sampling fails."""

    default_code = BufferErrorCode.PRIORITY_SAMPLING_FAILED

    def __init__(self, reason: str = "prioritized sampling failed", **kwargs: Any) -> None:
        self.reason = reason
        super().__init__(f"Priority sampling failed: {reason}", details={"reason": reason}, **kwargs)


class PriorityUpdateError(SamplingError):
    """Raised when priority updates do not match valid indices or values."""

    default_code = BufferErrorCode.PRIORITY_UPDATE_FAILED

    def __init__(self, reason: str, indices: Optional[Sequence[int]] = None, **kwargs: Any) -> None:
        self.reason = reason
        self.indices = list(indices or [])
        super().__init__(
            f"Priority update failed: {reason}",
            operation="update_priorities",
            details={"reason": reason, "indices": self.indices},
            **kwargs,
        )


class PriorityMassError(SamplingError):
    """Raised when sampling priority mass is zero, negative, NaN, or non-finite."""

    default_code = BufferErrorCode.PRIORITY_MASS_INVALID

    def __init__(self, total_mass: Any, reason: str = "invalid priority mass", **kwargs: Any) -> None:
        self.total_mass = total_mass
        self.reason = reason
        super().__init__(
            f"Invalid priority mass {_safe_value(total_mass)}: {reason}",
            details={"total_mass": _safe_value(total_mass), "reason": reason},
            **kwargs,
        )


class AgentDistributionError(SamplingError):
    """Raised when agent distribution for balanced sampling is invalid."""

    default_code = BufferErrorCode.AGENT_DISTRIBUTION_INVALID

    def __init__(self, distribution: Any, reason: str, **kwargs: Any) -> None:
        self.distribution = distribution
        self.reason = reason
        super().__init__(
            f"Invalid agent distribution: {reason}",
            details={"distribution": _safe_value(distribution), "reason": reason},
            **kwargs,
        )


class StaleExperienceError(ReplayBufferError):
    """Raised when an operation targets an experience that has expired or been pruned."""

    default_code = BufferErrorCode.STALE_EXPERIENCE
    default_recoverable = True

    def __init__(self, index: Optional[int] = None, age_seconds: Optional[float] = None, threshold_seconds: Optional[float] = None, **kwargs: Any) -> None:
        self.index = index
        self.age_seconds = age_seconds
        self.threshold_seconds = threshold_seconds
        super().__init__(
            "Experience is stale or no longer available.",
            recovery_hint=BufferRecoveryHint(action="Refresh sample indices and retry.", retryable=True),
            details={"index": index, "age_seconds": age_seconds, "threshold_seconds": threshold_seconds},
            **kwargs,
        )


# -----------------------------------------------------------------------------
# Persistence errors
# -----------------------------------------------------------------------------

class BufferPersistenceError(BufferError, IOError):
    """Base for save/load/serialization errors."""

    default_code = BufferErrorCode.PERSISTENCE_ERROR
    default_category = BufferCategory.PERSISTENCE


class BufferSaveError(BufferPersistenceError):
    """Raised when saving buffer state fails."""

    default_code = BufferErrorCode.SAVE_FAILED
    default_recoverable = True

    def __init__(self, filepath: str, reason: str, **kwargs: Any) -> None:
        self.filepath = str(filepath)
        self.reason = reason
        super().__init__(
            f"Failed to save buffer to {self.filepath}: {reason}",
            operation="save",
            recovery_hint=BufferRecoveryHint(action="Check path, permissions, and serialization compatibility.", retryable=True),
            details={"filepath": self.filepath, "reason": reason},
            **kwargs,
        )


class BufferLoadError(BufferPersistenceError):
    """Raised when loading buffer state fails."""

    default_code = BufferErrorCode.LOAD_FAILED
    default_recoverable = True

    def __init__(self, filepath: str, reason: str, **kwargs: Any) -> None:
        self.filepath = str(filepath)
        self.reason = reason
        super().__init__(
            f"Failed to load buffer from {self.filepath}: {reason}",
            operation="load",
            recovery_hint=BufferRecoveryHint(action="Check file existence, schema version, and corruption.", retryable=True),
            details={"filepath": self.filepath, "reason": reason},
            **kwargs,
        )


class BufferSerializationError(BufferPersistenceError):
    """Raised when state cannot be serialized/deserialized safely."""

    default_code = BufferErrorCode.SERIALIZATION_FAILED

    def __init__(self, operation: str, reason: str, **kwargs: Any) -> None:
        self.operation = str(operation)
        self.reason = reason
        super().__init__(
            f"Buffer serialization failed during '{self.operation}': {reason}",
            operation=self.operation,
            details={"reason": reason},
            **kwargs,
        )


# -----------------------------------------------------------------------------
# Network buffer specific errors
# -----------------------------------------------------------------------------

class NetworkBufferError(BufferError):
    """Base for network-related buffer exceptions."""

    default_code = BufferErrorCode.NETWORK_ERROR
    default_category = BufferCategory.NETWORK


class NetworkMessageError(NetworkBufferError, ValueError):
    """Raised when a network message payload/envelope is invalid."""

    default_code = BufferErrorCode.NETWORK_MESSAGE_INVALID

    def __init__(self, reason: str, message_id: Optional[str] = None, **kwargs: Any) -> None:
        self.reason = reason
        self.message_id = message_id
        super().__init__(
            f"Invalid network message{f' {message_id}' if message_id else ''}: {reason}",
            context=build_error_context(category=BufferCategory.NETWORK, item_id=message_id),
            details={"reason": reason, "message_id": message_id},
            **kwargs,
        )


class MessageExpiredError(NetworkBufferError):
    """Raised when trying to operate on an expired message."""

    default_code = BufferErrorCode.MESSAGE_EXPIRED
    default_recoverable = True

    def __init__(self, message_id: str, expired_at: Any, **kwargs: Any) -> None:
        self.message_id = str(message_id)
        self.expired_at = expired_at
        super().__init__(
            f"Message {self.message_id} expired at {expired_at}.",
            context=build_error_context(category=BufferCategory.NETWORK, item_id=self.message_id),
            recovery_hint=BufferRecoveryHint(action="Drop the message or request a fresh producer payload.", retryable=False),
            details={"message_id": self.message_id, "expired_at": str(expired_at)},
            **kwargs,
        )


class MessageNotFoundError(NetworkBufferError, LookupError):
    """Raised when a message ID does not exist in the buffer."""

    default_code = BufferErrorCode.MESSAGE_NOT_FOUND
    default_recoverable = True

    def __init__(self, message_id: str, operation: str = "lookup", **kwargs: Any) -> None:
        self.message_id = str(message_id)
        self.operation = str(operation)
        super().__init__(
            f"Message {self.message_id} not found in buffer for operation '{self.operation}'.",
            operation=self.operation,
            context=build_error_context(category=BufferCategory.NETWORK, item_id=self.message_id, operation=self.operation),
            recovery_hint=BufferRecoveryHint(action="Ignore idempotent ack/nack or refresh message state.", retryable=True),
            **kwargs,
        )


class DuplicateMessageError(NetworkBufferError):
    """Raised when a message ID already exists and duplicates are disallowed."""

    default_code = BufferErrorCode.MESSAGE_DUPLICATE

    def __init__(self, message_id: str, **kwargs: Any) -> None:
        self.message_id = str(message_id)
        super().__init__(
            f"Duplicate message id: {self.message_id}",
            context=build_error_context(category=BufferCategory.NETWORK, item_id=self.message_id),
            details={"message_id": self.message_id},
            **kwargs,
        )


class FairnessKeyError(NetworkBufferError, ValueError):
    """Raised when a fairness key is invalid or misconfigured."""

    default_code = BufferErrorCode.FAIRNESS_KEY_INVALID
    default_category = BufferCategory.FAIRNESS

    def __init__(self, fairness_key: Any = None, reason: str = "invalid fairness key", **kwargs: Any) -> None:
        self.fairness_key = fairness_key
        self.reason = reason
        super().__init__(
            f"Invalid fairness key {_safe_value(fairness_key)}: {reason}",
            details={"fairness_key": _safe_value(fairness_key), "reason": reason},
            **kwargs,
        )


class FairnessSchedulingError(NetworkBufferError):
    """Raised when fairness scheduling cannot select a valid key/message."""

    default_code = BufferErrorCode.FAIRNESS_SCHEDULING_FAILED
    default_category = BufferCategory.FAIRNESS
    default_recoverable = True

    def __init__(self, reason: str, active_keys: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        self.reason = reason
        self.active_keys = list(active_keys or [])
        super().__init__(
            f"Fairness scheduling failed: {reason}",
            operation="dequeue",
            recovery_hint=BufferRecoveryHint(action="Prune empty keys and retry scheduling.", retryable=True),
            details={"reason": reason, "active_keys": self.active_keys},
            **kwargs,
        )


class DropStrategyError(NetworkBufferError, ValueError):
    """Raised when an unsupported drop strategy is configured."""

    default_code = BufferErrorCode.DROP_STRATEGY_INVALID

    def __init__(self, strategy: str, valid_strategies: Sequence[str], **kwargs: Any) -> None:
        self.strategy = str(strategy)
        self.valid_strategies = list(valid_strategies)
        super().__init__(
            f"Invalid drop strategy '{self.strategy}'. Valid: {self.valid_strategies}",
            details={"strategy": self.strategy, "valid_strategies": self.valid_strategies},
            **kwargs,
        )


class BackpressureError(NetworkBufferError):
    """Base class for backpressure decisions and failures."""

    default_code = BufferErrorCode.BACKPRESSURE
    default_recoverable = True

    def __init__(self, reason: str, capacity: Optional[int] = None, size: Optional[int] = None, **kwargs: Any) -> None:
        self.reason = reason
        self.capacity = capacity
        self.size = size
        extra_details = dict(kwargs.pop("details", {}) or {})
        extra_details.update({"reason": reason, "capacity": capacity, "size": size})
        super().__init__(
            f"Backpressure triggered: {reason}",
            operation="enqueue",
            recovery_hint=BufferRecoveryHint(action="Retry later, reduce producer rate, or adjust drop policy.", retryable=True),
            details=extra_details,
            **kwargs,
        )


class BackpressureRejectedError(BackpressureError):
    """Raised when an incoming message is rejected due to configured backpressure."""

    default_code = BufferErrorCode.BACKPRESSURE_REJECTED

    def __init__(self, reason: str = "incoming message rejected", **kwargs: Any) -> None:
        super().__init__(reason=reason, **kwargs)


class InflightLimitError(BackpressureError):
    """Raised when a fairness key exceeds the per-key inflight limit."""

    default_code = BufferErrorCode.INFLIGHT_LIMIT_REACHED

    def __init__(self, fairness_key: str, limit: int, current: int, **kwargs: Any) -> None:
        self.fairness_key = str(fairness_key)
        self.limit = int(limit)
        self.current = int(current)
        super().__init__(
            reason=f"max_per_key_inflight reached for '{self.fairness_key}'",
            details={"fairness_key": self.fairness_key, "limit": self.limit, "current": self.current},
            **kwargs,
        )


class TTLValidationError(NetworkBufferError, ValueError):
    """Raised when ttl_seconds is invalid."""

    default_code = BufferErrorCode.TTL_INVALID

    def __init__(self, ttl_seconds: Any, reason: str = "ttl_seconds must be >= 0 or None", **kwargs: Any) -> None:
        self.ttl_seconds = ttl_seconds
        self.reason = reason
        super().__init__(
            f"Invalid ttl_seconds {_safe_value(ttl_seconds)}: {reason}",
            details={"ttl_seconds": _safe_value(ttl_seconds), "reason": reason},
            **kwargs,
        )


class AckError(NetworkBufferError):
    """Raised when ack fails for a non-idempotent caller path."""

    default_code = BufferErrorCode.ACK_FAILED
    default_recoverable = True

    def __init__(self, message_id: str, reason: str, **kwargs: Any) -> None:
        self.message_id = str(message_id)
        self.reason = reason
        super().__init__(
            f"Ack failed for message {self.message_id}: {reason}",
            operation="ack",
            context=build_error_context(category=BufferCategory.NETWORK, operation="ack", item_id=self.message_id),
            recovery_hint=BufferRecoveryHint(action="Treat repeated ack as idempotent or refresh message state.", retryable=True),
            details={"reason": reason},
            **kwargs,
        )


class NackError(NetworkBufferError):
    """Raised when nack/requeue fails for a non-idempotent caller path."""

    default_code = BufferErrorCode.NACK_FAILED
    default_recoverable = True

    def __init__(self, message_id: str, reason: str, **kwargs: Any) -> None:
        self.message_id = str(message_id)
        self.reason = reason
        super().__init__(
            f"Nack failed for message {self.message_id}: {reason}",
            operation="nack",
            context=build_error_context(category=BufferCategory.NETWORK, operation="nack", item_id=self.message_id),
            recovery_hint=BufferRecoveryHint(action="Drop or requeue after refreshing message state.", retryable=True),
            details={"reason": reason},
            **kwargs,
        )


# -----------------------------------------------------------------------------
# N-step buffer errors
# -----------------------------------------------------------------------------

class NStepBufferError(BufferError):
    """Base for n-step transformation errors."""

    default_code = BufferErrorCode.NSTEP_ERROR
    default_category = BufferCategory.NSTEP


class NStepConfigError(NStepBufferError, ValueError):
    """Raised when n-step configuration is invalid."""

    default_code = BufferErrorCode.NSTEP_CONFIG_INVALID

    def __init__(self, param_name: str = "nstep", value: Any = None, expected: str = "valid n-step config", **kwargs: Any) -> None:
        self.param_name = str(param_name)
        self.value = value
        self.expected = expected
        super().__init__(
            f"Invalid n-step config '{self.param_name}': {_safe_value(value)} (expected {expected})",
            details={"param_name": self.param_name, "value": _safe_value(value), "expected": expected},
            **kwargs,
        )


class NStepWindowError(NStepBufferError):
    """Raised when the n-step queue/window cannot produce a valid output."""

    default_code = BufferErrorCode.NSTEP_WINDOW_INVALID

    def __init__(self, window_size: int, n_step: int, reason: str, **kwargs: Any) -> None:
        self.window_size = int(window_size)
        self.n_step = int(n_step)
        self.reason = reason
        super().__init__(
            f"Invalid n-step window size {self.window_size} for n_step={self.n_step}: {reason}",
            details={"window_size": self.window_size, "n_step": self.n_step, "reason": reason},
            **kwargs,
        )


class NStepComputationError(NStepBufferError):
    """Raised when discounted return computation fails."""

    default_code = BufferErrorCode.NSTEP_COMPUTATION_FAILED

    def __init__(self, reason: str, **kwargs: Any) -> None:
        self.reason = reason
        super().__init__(f"N-step computation failed: {reason}", details={"reason": reason}, **kwargs)


class NStepTerminalStateError(NStepBufferError):
    """Raised when terminal handling conflicts with queue/window state."""

    default_code = BufferErrorCode.NSTEP_TERMINAL_STATE_INVALID

    def __init__(self, reason: str, **kwargs: Any) -> None:
        self.reason = reason
        super().__init__(f"Invalid n-step terminal state: {reason}", details={"reason": reason}, **kwargs)


# -----------------------------------------------------------------------------
# Segment tree errors
# -----------------------------------------------------------------------------

class SegmentTreeError(BufferError):
    """Base for segment tree errors."""

    default_code = BufferErrorCode.SEGMENT_TREE_ERROR
    default_category = BufferCategory.SEGMENT_TREE


class SegmentTreeCapacityError(SegmentTreeError, ValueError):
    """Raised when segment tree capacity is invalid."""

    default_code = BufferErrorCode.SEGMENT_TREE_CAPACITY_INVALID

    def __init__(self, capacity: Any, reason: str = "capacity must be > 0", **kwargs: Any) -> None:
        self.capacity = capacity
        self.reason = reason
        super().__init__(
            f"Invalid segment tree capacity {_safe_value(capacity)}: {reason}",
            details={"capacity": _safe_value(capacity), "reason": reason},
            **kwargs,
        )


class SegmentTreeIndexError(SegmentTreeError, IndexError):
    """Raised when an index is outside tree capacity."""

    default_code = BufferErrorCode.SEGMENT_TREE_INDEX_INVALID

    def __init__(self, index: int, capacity: int, operation: str = "access", **kwargs: Any) -> None:
        self.index = int(index)
        self.capacity = int(capacity)
        self.operation = str(operation)
        super().__init__(
            f"Index {self.index} out of bounds for segment tree capacity {self.capacity} during '{self.operation}'.",
            operation=self.operation,
            context=build_error_context(category=BufferCategory.SEGMENT_TREE, operation=self.operation, index=self.index),
            details={"index": self.index, "capacity": self.capacity},
            **kwargs,
        )


class SegmentTreeRangeError(SegmentTreeError, ValueError):
    """Raised when reduce/start/end range is invalid."""

    default_code = BufferErrorCode.SEGMENT_TREE_RANGE_INVALID

    def __init__(self, start: int, end: int, capacity: int, **kwargs: Any) -> None:
        self.start = int(start)
        self.end = int(end)
        self.capacity = int(capacity)
        super().__init__(
            f"Invalid segment tree range [{self.start}, {self.end}) for capacity {self.capacity}.",
            operation="reduce",
            details={"start": self.start, "end": self.end, "capacity": self.capacity},
            **kwargs,
        )


class SegmentTreePrefixSumError(SegmentTreeError, ValueError):
    """Raised when prefix sum is outside valid priority mass range."""

    default_code = BufferErrorCode.SEGMENT_TREE_PREFIXSUM_INVALID

    def __init__(self, prefixsum: float, total: float, reason: Optional[str] = None, **kwargs: Any) -> None:
        self.prefixsum = float(prefixsum)
        self.total = float(total)
        self.reason = reason or "prefix sum must be within [0, total_mass]"
        super().__init__(
            f"Prefix sum {self.prefixsum} invalid for total mass {self.total}: {self.reason}",
            operation="find_prefixsum_idx",
            details={"prefixsum": self.prefixsum, "total": self.total, "reason": self.reason},
            **kwargs,
        )


class SegmentTreeMassError(SegmentTreeError):
    """Raised when tree aggregate mass is invalid for sampling."""

    default_code = BufferErrorCode.SEGMENT_TREE_MASS_INVALID

    def __init__(self, total_mass: Any, reason: str = "total mass must be finite and non-negative", **kwargs: Any) -> None:
        self.total_mass = total_mass
        self.reason = reason
        super().__init__(
            f"Invalid segment tree mass {_safe_value(total_mass)}: {reason}",
            details={"total_mass": _safe_value(total_mass), "reason": reason},
            **kwargs,
        )


class SegmentTreeOperationError(SegmentTreeError):
    """Raised when an aggregate operation fails."""

    default_code = BufferErrorCode.SEGMENT_TREE_OPERATION_FAILED

    def __init__(self, operation: str, reason: str, **kwargs: Any) -> None:
        self.operation = str(operation)
        self.reason = reason
        super().__init__(
            f"Segment tree operation '{self.operation}' failed: {reason}",
            operation=self.operation,
            details={"reason": reason},
            **kwargs,
        )


# -----------------------------------------------------------------------------
# Eviction policy errors
# -----------------------------------------------------------------------------

class EvictionError(BufferError):
    """Base for eviction policy errors."""

    default_code = BufferErrorCode.EVICTION_ERROR
    default_category = BufferCategory.EVICTION


class EvictionContextError(EvictionError, ValueError):
    """Raised when eviction context is invalid."""

    default_code = BufferErrorCode.EVICTION_CONTEXT_INVALID

    def __init__(self, context: Any, reason: str = "empty sequence", **kwargs: Any) -> None:
        self.context_payload = context
        self.reason = reason
        super().__init__(
            f"Cannot evict: {reason}",
            operation="evict",
            details={"context": _safe_value(context), "reason": reason},
            **kwargs,
        )


class EvictionPolicyError(EvictionError):
    """Raised when eviction policy configuration or execution fails."""

    default_code = BufferErrorCode.EVICTION_POLICY_INVALID

    def __init__(self, policy: str, reason: str, **kwargs: Any) -> None:
        self.policy = str(policy)
        self.reason = reason
        extra_details = dict(kwargs.pop("details", {}) or {})
        extra_details.update({"policy": self.policy, "reason": reason})
        super().__init__(
            f"Eviction policy '{self.policy}' invalid: {reason}",
            details=extra_details,
            **kwargs,
        )


class UnsupportedEvictionPolicyError(EvictionPolicyError):
    """Raised when build_eviction_policy receives an unknown policy name."""

    default_code = BufferErrorCode.EVICTION_POLICY_UNSUPPORTED

    def __init__(self, policy: str, valid_policies: Sequence[str], **kwargs: Any) -> None:
        self.valid_policies = list(valid_policies)
        super().__init__(
            policy=policy,
            reason=f"unsupported policy; valid policies: {self.valid_policies}",
            details={"valid_policies": self.valid_policies},
            **kwargs,
        )


class EvictionSelectionError(EvictionError):
    """Raised when a policy cannot select a valid eviction index."""

    default_code = BufferErrorCode.EVICTION_SELECTION_FAILED

    def __init__(self, policy: str, reason: str, item_count: Optional[int] = None, **kwargs: Any) -> None:
        self.policy = str(policy)
        self.reason = reason
        self.item_count = item_count
        super().__init__(
            f"Eviction selection failed for policy '{self.policy}': {reason}",
            operation="select_index",
            details={"policy": self.policy, "reason": reason, "item_count": item_count},
            **kwargs,
        )


# -----------------------------------------------------------------------------
# Sequence replay errors
# -----------------------------------------------------------------------------

class SequenceReplayError(BufferError):
    """Base for sequence replay buffer errors."""

    default_code = BufferErrorCode.SEQUENCE_REPLAY_ERROR
    default_category = BufferCategory.SEQUENCE


class InvalidSequenceLengthError(SequenceReplayError, ValueError):
    """Raised when a sequence does not meet length constraints."""

    default_code = BufferErrorCode.SEQUENCE_LENGTH_INVALID

    def __init__(self, actual: int, min_length: int, max_length: Optional[int] = None, **kwargs: Any) -> None:
        self.actual = int(actual)
        self.min_length = int(min_length)
        self.max_length = None if max_length is None else int(max_length)
        if self.max_length is None:
            msg = f"Sequence length {self.actual} below minimum {self.min_length}."
        else:
            msg = f"Sequence length {self.actual} outside [{self.min_length}, {self.max_length}]."
        super().__init__(
            msg,
            details={"actual": self.actual, "min_length": self.min_length, "max_length": self.max_length},
            **kwargs,
        )


class SequenceReplayAssemblyError(SequenceReplayError):
    """Raised when stored transitions cannot form a valid replay sequence."""

    default_code = BufferErrorCode.SEQUENCE_ASSEMBLY_FAILED

    def __init__(self, reason: str, sequence_id: Optional[str] = None, **kwargs: Any) -> None:
        self.reason = reason
        self.sequence_id = sequence_id
        super().__init__(
            f"Sequence replay assembly failed{f' for {sequence_id}' if sequence_id else ''}: {reason}",
            details={"sequence_id": sequence_id, "reason": reason},
            **kwargs,
        )


class SequencePaddingError(SequenceReplayError):
    """Raised when sequence padding/truncation fails."""

    default_code = BufferErrorCode.SEQUENCE_PADDING_FAILED

    def __init__(self, target_length: int, reason: str, **kwargs: Any) -> None:
        self.target_length = int(target_length)
        self.reason = reason
        super().__init__(
            f"Sequence padding failed for target_length={self.target_length}: {reason}",
            details={"target_length": self.target_length, "reason": reason},
            **kwargs,
        )


# -----------------------------------------------------------------------------
# Telemetry and fairness errors/warnings
# -----------------------------------------------------------------------------

class TelemetryError(BufferError):
    """Raised when telemetry collection fails."""

    default_code = BufferErrorCode.TELEMETRY_ERROR
    default_category = BufferCategory.TELEMETRY
    default_recoverable = True


class TelemetryMetricError(TelemetryError, ValueError):
    """Raised when a metric name or value is invalid."""

    default_code = BufferErrorCode.TELEMETRY_METRIC_ERROR

    def __init__(self, metric_name: Any, reason: str, **kwargs: Any) -> None:
        self.metric_name = metric_name
        self.reason = reason
        extra_details = dict(kwargs.pop("details", {}) or {})
        extra_details.update({"metric_name": _safe_value(metric_name), "reason": reason})
        super().__init__(
            f"Invalid telemetry metric {_safe_value(metric_name)}: {reason}",
            details=extra_details,
            **kwargs,
        )


class MetricValueError(TelemetryMetricError):
    """Raised when metric value cannot be converted to a finite float."""

    default_code = BufferErrorCode.METRIC_VALUE_INVALID

    def __init__(self, metric_name: str, value: Any, reason: str = "value must be numeric and finite", **kwargs: Any) -> None:
        self.value = value
        super().__init__(
            metric_name=metric_name,
            reason=reason,
            details={"value": _safe_value(value), "actual_type": _safe_type_name(value)},
            **kwargs,
        )


class MetricSnapshotError(TelemetryError):
    """Raised when telemetry snapshot/export fails."""

    default_code = BufferErrorCode.METRIC_SNAPSHOT_FAILED

    def __init__(self, reason: str, **kwargs: Any) -> None:
        self.reason = reason
        super().__init__(f"Telemetry snapshot failed: {reason}", operation="snapshot", details={"reason": reason}, **kwargs)


class FairnessMetricError(BufferError, ValueError):
    """Raised when fairness checks receive insufficient or malformed data."""

    default_code = BufferErrorCode.FAIRNESS_METRIC_ERROR
    default_category = BufferCategory.FAIRNESS

    def __init__(self, required: str, actual: Any, metric_name: Optional[str] = None, **kwargs: Any) -> None:
        self.required = required
        self.actual = actual
        self.metric_name = metric_name
        super().__init__(
            f"Fairness check requires {required}, got {_safe_type_name(actual)}.",
            details={"required": required, "actual_type": _safe_type_name(actual), "metric_name": metric_name},
            **kwargs,
        )


class FairnessViolationError(BufferError):
    """Raised when configured fairness constraints are violated in strict mode."""

    default_code = BufferErrorCode.FAIRNESS_VIOLATION
    default_category = BufferCategory.FAIRNESS

    def __init__(self, metric_name: str, message: str, threshold: Optional[float] = None, observed: Optional[float] = None, **kwargs: Any) -> None:
        self.metric_name = metric_name
        self.threshold = threshold
        self.observed = observed
        super().__init__(
            f"Fairness violation for {metric_name}: {message}",
            details={"metric_name": metric_name, "threshold": threshold, "observed": observed},
            **kwargs,
        )


class TelemetryDisabledWarning(BufferWarning):
    """Warning emitted when telemetry is disabled but callers expect metrics."""

    default_category = BufferCategory.TELEMETRY


class FairnessViolationWarning(BufferWarning):
    """Non-fatal warning counterpart for fairness guardrail violations."""

    default_category = BufferCategory.FAIRNESS


# -----------------------------------------------------------------------------
# Bulk / aggregated operation errors
# -----------------------------------------------------------------------------

class BufferOperationError(BufferError):
    """Wraps multiple errors during bulk operations."""

    default_code = BufferErrorCode.OPERATION_ERROR
    default_category = BufferCategory.OPERATION

    def __init__(
        self,
        message: str,
        errors: Optional[Sequence[Union[str, BaseException]]] = None,
        *,
        operation: Optional[str] = None,
        partial_results: Any = None,
        **kwargs: Any,
    ) -> None:
        self.errors = list(errors or [])
        self.partial_results = partial_results
        super().__init__(
            f"{message} ({len(self.errors)} sub-errors)",
            operation=operation,
            details={"error_count": len(self.errors), "errors": summarize_errors(self.errors)},
            **kwargs,
        )

    def add_error(self, error: Union[str, BaseException]) -> None:
        self.errors.append(error)
        self.with_detail("error_count", len(self.errors))
        self.with_detail("errors", summarize_errors(self.errors))


class BufferPartialFailureError(BufferOperationError):
    """Raised when part of a bulk operation succeeds and part fails."""

    default_code = BufferErrorCode.PARTIAL_FAILURE
    default_recoverable = True

    def __init__(self, message: str, errors: Sequence[Union[str, BaseException]], succeeded: int = 0, failed: Optional[int] = None, **kwargs: Any) -> None:
        self.succeeded = int(succeeded)
        self.failed = len(errors) if failed is None else int(failed)
        super().__init__(
            message,
            errors,
            partial_results={"succeeded": self.succeeded, "failed": self.failed},
            recovery_hint=BufferRecoveryHint(action="Retry failed items only if operation is idempotent.", retryable=True),
            **kwargs,
        )
        self.with_detail("succeeded", self.succeeded)
        self.with_detail("failed", self.failed)


# -----------------------------------------------------------------------------
# Helper utilities for call sites
# -----------------------------------------------------------------------------

E = TypeVar("E", bound=BufferError)


def summarize_errors(errors: Sequence[Union[str, BaseException]], *, limit: int = 20) -> List[Dict[str, Any]]:
    """Return compact, structured summaries for nested/bulk errors."""
    summaries: List[Dict[str, Any]] = []
    for error in list(errors)[:limit]:
        if isinstance(error, BufferError):
            summaries.append(error.to_dict(include_cause=False))
        elif isinstance(error, BaseException):
            summaries.append({"error_type": type(error).__name__, "message": str(error)})
        else:
            summaries.append({"error_type": "str", "message": str(error)})
    if len(errors) > limit:
        summaries.append({"error_type": "truncated", "message": f"{len(errors) - limit} additional errors omitted"})
    return summaries


def coerce_buffer_error(
    exc: BaseException,
    *,
    error_cls: Type[E] = BufferError,  # type: ignore[assignment]
    message: Optional[str] = None,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    code: Optional[BufferErrorCode] = None,
    recoverable: Optional[bool] = None,
) -> BufferError:
    """Convert an arbitrary exception into a BufferError while preserving the cause.

    Existing BufferError instances are returned unchanged unless extra component or
    operation metadata is supplied, in which case the metadata is merged in-place.
    """
    if isinstance(exc, BufferError):
        if component or operation:
            exc.context = BufferErrorContext(
                component=component or exc.context.component,
                operation=operation or exc.context.operation,
                category=exc.context.category,
                item_id=exc.context.item_id,
                index=exc.context.index,
                field_name=exc.context.field_name,
                config_section=exc.context.config_section,
                correlation_id=exc.context.correlation_id,
                details=dict(exc.context.details),
                created_at=exc.context.created_at,
            )
        return exc

    resolved_message = message or str(exc) or type(exc).__name__
    return error_cls(
        resolved_message,
        component=component,
        operation=operation,
        code=code,
        recoverable=recoverable,
        cause=exc,
        details={"wrapped_error_type": type(exc).__name__},
    )


def ensure(condition: bool, error_factory: Union[BufferError, Callable[[], BufferError]]) -> None:
    """Raise a BufferError from a factory when condition is false."""
    if condition:
        return
    if isinstance(error_factory, BufferError):
        raise error_factory
    raise error_factory()


def ensure_positive_int(value: Any, name: str = "value", *, component: Optional[str] = None, operation: Optional[str] = None) -> int:
    """Validate and return a positive integer."""
    try:
        resolved = int(value)
    except Exception as exc:
        raise ConfigValueError(
            name,
            value,
            "positive integer",
            component=component,
            operation=operation,
            cause=exc,
        ) from exc
    if resolved <= 0:
        raise ConfigValueError(name, value, "positive integer", component=component, operation=operation)
    return resolved


def ensure_non_empty(items: Sequence[Any], operation: str = "operate", *, error_cls: Type[BufferError] = BufferEmptyError) -> None:
    """Raise a standard buffer error when a sequence is empty."""
    if len(items) == 0:
        if error_cls is BufferEmptyError:
            raise BufferEmptyError(operation=operation)
        raise error_cls(f"Cannot {operation}: sequence is empty.", operation=operation)


def ensure_index_in_bounds(index: int, size: int, operation: str = "access") -> int:
    """Validate a [0, size) index and return it."""
    idx = int(index)
    if idx < 0 or idx >= int(size):
        raise IndexOutOfBoundsError(index=idx, size=int(size), operation=operation)
    return idx



# -----------------------------------------------------------------------------
# Small helpers: validation/telemetry compatibility without duplicating modules
# -----------------------------------------------------------------------------

def positive_int(value: Any, field_name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValueError(field_name, value, "positive integer", section="reservoir") from exc
    if resolved <= 0:
        raise ConfigValueError(field_name, value, "positive integer > 0", section="reservoir")
    return resolved


def non_negative_float(value: Any, field_name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValueError(field_name, value, "non-negative float", section="reservoir") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ConfigValueError(field_name, value, "finite non-negative float", section="reservoir")
    return resolved


def validate_batch_size(batch_size: Any) -> int:
    if isinstance(batch_size, bool):
        raise InvalidBatchSizeError(batch_size, reason="batch_size must be an integer, not bool")
    try:
        resolved = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise InvalidBatchSizeError(batch_size, reason="batch_size must be a positive integer") from exc
    if resolved <= 0:
        raise InvalidBatchSizeError(batch_size, reason="batch_size must be > 0")
    return resolved


def telemetry_call(telemetry: Any, method: str, *args: Any, **kwargs: Any) -> None:
    fn = getattr(telemetry, method, None)
    if callable(fn):
        fn(*args, **kwargs)


def telemetry_increment(telemetry: Any, name: str, amount: float = 1.0) -> None:
    fn = getattr(telemetry, "increment", None)
    if callable(fn):
        fn(name, amount)


def telemetry_observe(telemetry: Any, name: str, value: float) -> None:
    fn = getattr(telemetry, "observe", None)
    if callable(fn):
        fn(name, value)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def validate_metric_name(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise TelemetryMetricError(name, "metric name cannot be empty")
    return normalized


def validate_metric_value(metric_name: str, value: Any) -> float:
    if not _is_finite_number(value):
        raise MetricValueError(metric_name, value)
    return float(value)

