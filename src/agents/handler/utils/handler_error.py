from __future__ import annotations

__version__ = "2.0.0"

import hashlib
import json
import re
import time
import traceback

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, Mapping, Optional, Sequence, Type, TypeVar, Union

from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Handler Error")
printer = PrettyPrinter()


class FailureSeverity(str, Enum):
    """Canonical severity values used by the Handler subsystem."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def normalize(cls, value: Union[str, "FailureSeverity", None]) -> "FailureSeverity":
        if isinstance(value, cls):
            return value
        normalized = str(value or cls.MEDIUM.value).strip().lower()
        for severity in cls:
            if severity.value == normalized:
                return severity
        return cls.MEDIUM

    @property
    def rank(self) -> int:
        return {
            FailureSeverity.LOW: 10,
            FailureSeverity.MEDIUM: 20,
            FailureSeverity.HIGH: 30,
            FailureSeverity.CRITICAL: 40,
        }[self]

    def at_least(self, other: Union[str, "FailureSeverity"]) -> bool:
        return self.rank >= FailureSeverity.normalize(other).rank


class HandlerErrorType(str, Enum):
    """Structured error domains for Handler-level policy and telemetry."""

    GENERIC = "HandlerError"
    NORMALIZATION = "NormalizationError"
    RECOVERY = "RecoveryError"
    SLA = "SLAError"
    ESCALATION = "EscalationError"
    TELEMETRY = "TelemetryError"
    INTELLIGENCE = "IntelligenceError"
    POLICY = "PolicyError"
    CONFIGURATION = "ConfigurationError"
    VALIDATION = "ValidationError"
    SECURITY = "SecurityError"
    SERIALIZATION = "SerializationError"
    CIRCUIT_BREAKER = "CircuitBreakerError"
    DEPENDENCY = "DependencyError"
    RESOURCE = "ResourceError"
    TIMEOUT = "TimeoutError"

    @classmethod
    def normalize(cls, value: Union[str, "HandlerErrorType", None]) -> str:
        if isinstance(value, cls):
            return value.value
        raw = str(value or cls.GENERIC.value).strip()
        if not raw:
            return cls.GENERIC.value
        for error_type in cls:
            if raw == error_type.value or raw.lower() == error_type.name.lower():
                return error_type.value
        return raw


class HandlerErrorVisibility(str, Enum):
    """Controls how much information is exposed when serializing an error."""

    PUBLIC = "public"
    TELEMETRY = "telemetry"
    INTERNAL = "internal"

    @classmethod
    def normalize(cls, value: Union[str, "HandlerErrorVisibility", None]) -> "HandlerErrorVisibility":
        if isinstance(value, cls):
            return value
        normalized = str(value or cls.TELEMETRY.value).strip().lower()
        for visibility in cls:
            if visibility.value == normalized:
                return visibility
        return cls.TELEMETRY


class HandlerRecoveryAction(str, Enum):
    """Policy-level action hints consumed by HandlerAgent and surrounding orchestration."""

    NONE = "none"
    RETRY = "retry"
    DEGRADE = "degrade"
    ESCALATE = "escalate"
    FAIL_FAST = "fail_fast"
    QUARANTINE = "quarantine"

    @classmethod
    def normalize(cls, value: Union[str, "HandlerRecoveryAction", None]) -> "HandlerRecoveryAction":
        if isinstance(value, cls):
            return value
        normalized = str(value or cls.NONE.value).strip().lower()
        for action in cls:
            if action.value == normalized:
                return action
        return cls.NONE


@dataclass(frozen=True)
class HandlerErrorPolicy:
    """
    Serialization and handling policy for HandlerError.

    This object intentionally has no dependency on HandlerPolicy, config loaders, or helper modules.
    Callers may construct it from YAML/config data and pass it into errors when stricter behavior
    is needed. The error layer remains focused on error representation, secure serialization,
    and policy decisions only.
    """

    name: str = "handler_error.default"
    expose_internal_messages: bool = False
    include_context_in_public: bool = False
    include_context_in_telemetry: bool = True
    include_traceback_in_internal: bool = False
    allow_critical_retry: bool = False
    max_message_chars: int = 500
    max_context_depth: int = 4
    max_context_items: int = 50
    max_sequence_items: int = 20
    max_string_chars: int = 500
    redaction_text: str = "[REDACTED]"
    public_messages: Mapping[str, str] = field(
        default_factory=lambda: {
            FailureSeverity.LOW.value: "A recoverable handler issue occurred.",
            FailureSeverity.MEDIUM.value: "The handler encountered an issue while processing the request.",
            FailureSeverity.HIGH.value: "The handler encountered a high-severity issue and applied safeguards.",
            FailureSeverity.CRITICAL.value: "A critical handler safeguard was triggered.",
        }
    )
    retryable_by_severity: Mapping[str, bool] = field(
        default_factory=lambda: {
            FailureSeverity.LOW.value: True,
            FailureSeverity.MEDIUM.value: True,
            FailureSeverity.HIGH.value: False,
            FailureSeverity.CRITICAL.value: False,
        }
    )
    action_by_severity: Mapping[str, str] = field(
        default_factory=lambda: {
            FailureSeverity.LOW.value: HandlerRecoveryAction.RETRY.value,
            FailureSeverity.MEDIUM.value: HandlerRecoveryAction.DEGRADE.value,
            FailureSeverity.HIGH.value: HandlerRecoveryAction.ESCALATE.value,
            FailureSeverity.CRITICAL.value: HandlerRecoveryAction.FAIL_FAST.value,
        }
    )
    action_by_type: Mapping[str, str] = field(
        default_factory=lambda: {
            HandlerErrorType.SECURITY.value: HandlerRecoveryAction.QUARANTINE.value,
            HandlerErrorType.POLICY.value: HandlerRecoveryAction.FAIL_FAST.value,
            HandlerErrorType.CONFIGURATION.value: HandlerRecoveryAction.ESCALATE.value,
            HandlerErrorType.SLA.value: HandlerRecoveryAction.DEGRADE.value,
            HandlerErrorType.TELEMETRY.value: HandlerRecoveryAction.DEGRADE.value,
            HandlerErrorType.ESCALATION.value: HandlerRecoveryAction.FAIL_FAST.value,
            HandlerErrorType.CIRCUIT_BREAKER.value: HandlerRecoveryAction.ESCALATE.value,
            HandlerErrorType.TIMEOUT.value: HandlerRecoveryAction.RETRY.value,
            HandlerErrorType.RESOURCE.value: HandlerRecoveryAction.DEGRADE.value,
        }
    )
    retryable_by_type: Mapping[str, bool] = field(
        default_factory=lambda: {
            HandlerErrorType.SECURITY.value: False,
            HandlerErrorType.POLICY.value: False,
            HandlerErrorType.CONFIGURATION.value: False,
            HandlerErrorType.VALIDATION.value: False,
            HandlerErrorType.SLA.value: False,
            HandlerErrorType.ESCALATION.value: False,
            HandlerErrorType.TELEMETRY.value: True,
            HandlerErrorType.INTELLIGENCE.value: True,
            HandlerErrorType.TIMEOUT.value: True,
            HandlerErrorType.RESOURCE.value: True,
        }
    )
    sensitive_key_fragments: Sequence[str] = field(
        default_factory=lambda: (
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "api_key",
            "apikey",
            "access_key",
            "private_key",
            "credential",
            "authorization",
            "auth",
            "cookie",
            "session",
            "jwt",
            "bearer",
            "email",
            "phone",
            "ssn",
            "credit_card",
            "card_number",
        )
    )
    sensitive_message_patterns: Sequence[str] = field(
        default_factory=lambda: (
            r"(?i)(bearer\s+)[a-z0-9._\-~+/]+=*",
            r"(?i)((?:password|passwd|pwd|secret|token|api[_-]?key|authorization)\s*[:=]\s*)[^\s,;]+",
            r"(?i)(://[^:/\s]+:)[^@\s]+(@)",
            r"(?i)\b(?:[a-z0-9_.+\-]+)@(?:(?:[a-z0-9\-]+\.)+[a-z]{2,})\b",
        )
    )

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]]) -> "HandlerErrorPolicy":
        """Build an error policy from YAML/config data without binding this module to a loader."""
        if not isinstance(config, Mapping):
            return cls()

        return cls(
            name=str(config.get("name", cls.name)),
            expose_internal_messages=bool(config.get("expose_internal_messages", False)),
            include_context_in_public=bool(config.get("include_context_in_public", False)),
            include_context_in_telemetry=bool(config.get("include_context_in_telemetry", True)),
            include_traceback_in_internal=bool(config.get("include_traceback_in_internal", False)),
            allow_critical_retry=bool(config.get("allow_critical_retry", False)),
            max_message_chars=max(80, int(config.get("max_message_chars", 500))),
            max_context_depth=max(1, int(config.get("max_context_depth", 4))),
            max_context_items=max(1, int(config.get("max_context_items", 50))),
            max_sequence_items=max(1, int(config.get("max_sequence_items", 20))),
            max_string_chars=max(32, int(config.get("max_string_chars", 500))),
            redaction_text=str(config.get("redaction_text", "[REDACTED]")),
            public_messages=dict(config.get("public_messages", {})) or cls().public_messages,
            retryable_by_severity=dict(config.get("retryable_by_severity", {})) or cls().retryable_by_severity,
            action_by_severity=dict(config.get("action_by_severity", {})) or cls().action_by_severity,
            action_by_type=dict(config.get("action_by_type", {})) or cls().action_by_type,
            retryable_by_type=dict(config.get("retryable_by_type", {})) or cls().retryable_by_type,
            sensitive_key_fragments=tuple(config.get("sensitive_key_fragments", cls().sensitive_key_fragments)),
            sensitive_message_patterns=tuple(config.get("sensitive_message_patterns", cls().sensitive_message_patterns)),
        )

    def resolve_retryable(
        self,
        *,
        error_type: str,
        severity: FailureSeverity,
        explicit_retryable: Optional[bool] = None,
    ) -> bool:
        if severity == FailureSeverity.CRITICAL and not self.allow_critical_retry:
            return False
        if explicit_retryable is not None:
            return bool(explicit_retryable)
        if error_type in self.retryable_by_type:
            return bool(self.retryable_by_type[error_type])
        return bool(self.retryable_by_severity.get(severity.value, False))

    def resolve_action(
        self,
        *,
        error_type: str,
        severity: FailureSeverity,
        retryable: bool,
        explicit_action: Optional[Union[str, HandlerRecoveryAction]] = None,
    ) -> HandlerRecoveryAction:
        if explicit_action is not None:
            return HandlerRecoveryAction.normalize(explicit_action)
        if error_type in self.action_by_type:
            return HandlerRecoveryAction.normalize(self.action_by_type[error_type])
        if retryable:
            return HandlerRecoveryAction.RETRY
        return HandlerRecoveryAction.normalize(self.action_by_severity.get(severity.value))

    def public_message_for(self, *, severity: FailureSeverity, message: str) -> str:
        if self.expose_internal_messages:
            return self.sanitize_message(message)
        return self.public_messages.get(severity.value, self.public_messages[FailureSeverity.MEDIUM.value])

    def sanitize_message(self, value: Any) -> str:
        text = str(value or "")
        if len(text) > self.max_message_chars:
            text = f"{text[: self.max_message_chars]}..."
        for pattern in self.sensitive_message_patterns:
            text = re.sub(pattern, self._redacted_replacement, text)
        return text

    def sanitize_context(self, value: Any) -> Any:
        return self._sanitize_value(value=value, depth=0, seen=set())

    def fingerprint_payload(self, payload: Mapping[str, Any]) -> str:
        serialized = json.dumps(self.sanitize_context(payload), sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _redacted_replacement(self, match: re.Match[str]) -> str:
        groups = match.groups()
        if len(groups) >= 2:
            return f"{groups[0]}{self.redaction_text}{groups[-1] if groups[-1] == '@' else ''}"
        if len(groups) == 1:
            return f"{groups[0]}{self.redaction_text}"
        return self.redaction_text

    def _is_sensitive_key(self, key: Any) -> bool:
        normalized_key = str(key).strip().lower().replace("-", "_")
        return any(fragment in normalized_key for fragment in self.sensitive_key_fragments)

    def _sanitize_value(self, *, value: Any, depth: int, seen: set[int]) -> Any:
        if depth > self.max_context_depth:
            return "[MAX_DEPTH]"

        if value is None or isinstance(value, (bool, int, float)):
            return value

        if isinstance(value, str):
            text = self.sanitize_message(value)
            if len(text) > self.max_string_chars:
                return f"{text[: self.max_string_chars]}..."
            return text

        if isinstance(value, bytes):
            return f"[bytes:{len(value)}]"

        object_id = id(value)
        if object_id in seen:
            return "[CIRCULAR]"
        seen.add(object_id)

        if isinstance(value, Mapping):
            sanitized: Dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= self.max_context_items:
                    sanitized["[TRUNCATED]"] = f"{len(value) - self.max_context_items} additional item(s) omitted"
                    break
                safe_key = str(key)
                if self._is_sensitive_key(safe_key):
                    sanitized[safe_key] = self.redaction_text
                else:
                    sanitized[safe_key] = self._sanitize_value(value=item, depth=depth + 1, seen=seen)
            seen.discard(object_id)
            return sanitized

        if isinstance(value, (list, tuple, set, frozenset)):
            sequence = list(value)
            sanitized_items = [
                self._sanitize_value(value=item, depth=depth + 1, seen=seen)
                for item in sequence[: self.max_sequence_items]
            ]
            if len(sequence) > self.max_sequence_items:
                sanitized_items.append(f"[TRUNCATED:{len(sequence) - self.max_sequence_items}]")
            seen.discard(object_id)
            return sanitized_items

        if isinstance(value, BaseException):
            seen.discard(object_id)
            return {
                "exception_type": type(value).__name__,
                "message": self.sanitize_message(str(value)),
            }

        safe_repr = self.sanitize_message(repr(value))
        seen.discard(object_id)
        if len(safe_repr) > self.max_string_chars:
            return f"{safe_repr[: self.max_string_chars]}..."
        return safe_repr


THandlerError = TypeVar("THandlerError", bound="HandlerError")


@dataclass
class HandlerError(Exception):
    """
    Production-grade Handler exception.

    The class provides:
    - canonical severity/type/action metadata
    - policy-based retry/action decisions
    - safe public, telemetry, and internal serialization
    - bounded, redacted context transport
    - exception conversion without leaking raw secrets
    """

    message: str
    error_type: Union[str, HandlerErrorType] = HandlerErrorType.GENERIC.value
    severity: Union[str, FailureSeverity] = FailureSeverity.MEDIUM
    retryable: Optional[bool] = None
    context: Mapping[str, Any] = field(default_factory=dict)
    cause: Optional[Union[str, BaseException]] = None
    code: Optional[str] = None
    action: Optional[Union[str, HandlerRecoveryAction]] = None
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    details: Optional[Mapping[str, Any]] = None
    policy: HandlerErrorPolicy = field(default_factory=HandlerErrorPolicy, repr=False, compare=False)
    timestamp: float = field(default_factory=time.time)

    schema_version: ClassVar[str] = "handler.error.v2"

    def __post_init__(self) -> None:
        normalized_type = HandlerErrorType.normalize(self.error_type)
        normalized_severity = FailureSeverity.normalize(self.severity)
        resolved_retryable = self.policy.resolve_retryable(
            error_type=normalized_type,
            severity=normalized_severity,
            explicit_retryable=self.retryable,
        )
        resolved_action = self.policy.resolve_action(
            error_type=normalized_type,
            severity=normalized_severity,
            retryable=resolved_retryable,
            explicit_action=self.action,
        )

        object.__setattr__(self, "message", str(self.message or "Handler error occurred"))
        object.__setattr__(self, "error_type", normalized_type)
        object.__setattr__(self, "severity", normalized_severity)
        object.__setattr__(self, "retryable", resolved_retryable)
        object.__setattr__(self, "action", resolved_action)
        object.__setattr__(self, "context", dict(self.context or {}))
        object.__setattr__(self, "details", dict(self.details or {}))

        if isinstance(self.cause, BaseException):
            object.__setattr__(self, "__cause__", self.cause)

        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return self.policy.sanitize_message(self.message)

    @property
    def severity_value(self) -> str:
        return FailureSeverity.normalize(self.severity).value

    @property
    def action_value(self) -> str:
        return HandlerRecoveryAction.normalize(self.action).value

    @property
    def cause_type(self) -> Optional[str]:
        if isinstance(self.cause, BaseException):
            return type(self.cause).__name__
        if self.cause:
            return str(self.cause)
        return None

    @property
    def cause_message(self) -> Optional[str]:
        if isinstance(self.cause, BaseException):
            return self.policy.sanitize_message(str(self.cause))
        return None

    @property
    def is_retryable(self) -> bool:
        return bool(self.retryable)

    @property
    def should_escalate(self) -> bool:
        return HandlerRecoveryAction.normalize(self.action) in {
            HandlerRecoveryAction.ESCALATE,
            HandlerRecoveryAction.FAIL_FAST,
            HandlerRecoveryAction.QUARANTINE,
        }

    @property
    def should_fail_fast(self) -> bool:
        return HandlerRecoveryAction.normalize(self.action) == HandlerRecoveryAction.FAIL_FAST

    @property
    def should_quarantine(self) -> bool:
        return HandlerRecoveryAction.normalize(self.action) == HandlerRecoveryAction.QUARANTINE

    @property
    def fingerprint(self) -> str:
        return self.policy.fingerprint_payload(
            {
                "error_type": self.error_type,
                "message": self.message,
                "severity": self.severity_value,
                "code": self.code,
                "source": self.source,
                "context": self.context,
            }
        )

    def policy_decision(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.name,
            "action": self.action_value,
            "retryable": self.is_retryable,
            "escalate": self.should_escalate,
            "fail_fast": self.should_fail_fast,
            "quarantine": self.should_quarantine,
        }

    def to_dict(
        self,
        *,
        visibility: Union[str, HandlerErrorVisibility] = HandlerErrorVisibility.TELEMETRY,
        include_context: Optional[bool] = None,
        include_traceback: Optional[bool] = None,
    ) -> Dict[str, Any]:
        visibility = HandlerErrorVisibility.normalize(visibility)
        sanitized_message = self.policy.sanitize_message(self.message)
        public_message = self.policy.public_message_for(
            severity=FailureSeverity.normalize(self.severity),
            message=self.message,
        )

        if include_context is None:
            include_context = {
                HandlerErrorVisibility.PUBLIC: self.policy.include_context_in_public,
                HandlerErrorVisibility.TELEMETRY: self.policy.include_context_in_telemetry,
                HandlerErrorVisibility.INTERNAL: True,
            }[visibility]

        if include_traceback is None:
            include_traceback = visibility == HandlerErrorVisibility.INTERNAL and self.policy.include_traceback_in_internal

        payload: Dict[str, Any] = {
            "schema": self.schema_version,
            "error_type": self.error_type,
            "message": public_message if visibility == HandlerErrorVisibility.PUBLIC else sanitized_message,
            "severity": self.severity_value,
            "retryable": self.is_retryable,
            "action": self.action_value,
            "code": self.code,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
            "policy": self.policy_decision(),
        }

        if visibility != HandlerErrorVisibility.PUBLIC:
            payload["cause"] = {
                "type": self.cause_type,
                "message": self.cause_message,
            }
            payload["details"] = self.policy.sanitize_context(self.details or {})

        if include_context:
            payload["context"] = self.policy.sanitize_context(self.context)

        if include_traceback and isinstance(self.cause, BaseException):
            payload["traceback"] = self.policy.sanitize_message(
                "".join(traceback.format_exception(type(self.cause), self.cause, self.cause.__traceback__))
            )

        return payload

    def to_public_dict(self) -> Dict[str, Any]:
        return self.to_dict(visibility=HandlerErrorVisibility.PUBLIC)

    def to_telemetry_dict(self) -> Dict[str, Any]:
        return self.to_dict(visibility=HandlerErrorVisibility.TELEMETRY)

    def to_internal_dict(self) -> Dict[str, Any]:
        return self.to_dict(visibility=HandlerErrorVisibility.INTERNAL)

    def to_failure_payload(self) -> Dict[str, Any]:
        """Return the compact normalized shape used by HandlerAgent recovery/telemetry."""
        return {
            "type": self.error_type,
            "message": self.policy.sanitize_message(self.message),
            "severity": self.severity_value,
            "retryable": self.is_retryable,
            "context_hash": self.fingerprint,
            "timestamp": self.timestamp,
            "policy_action": self.action_value,
            "code": self.code,
            "source": self.source,
            "correlation_id": self.correlation_id,
        }

    def with_context(self: THandlerError, **context: Any) -> "HandlerError": # type: ignore
        merged_context = dict(self.context or {})
        merged_context.update(context)
        return HandlerError(
            message=self.message,
            error_type=self.error_type,
            severity=self.severity,
            retryable=self.retryable,
            context=merged_context,
            cause=self.cause,
            code=self.code,
            action=self.action,
            source=self.source,
            correlation_id=self.correlation_id,
            details=self.details,
            policy=self.policy,
            timestamp=self.timestamp,
        )

    @classmethod
    def from_exception(
        cls: Type[THandlerError],
        exc: BaseException,
        *,
        error_type: Union[str, HandlerErrorType] = HandlerErrorType.GENERIC,
        severity: Union[str, FailureSeverity] = FailureSeverity.MEDIUM,
        retryable: Optional[bool] = None,
        context: Optional[Mapping[str, Any]] = None,
        code: Optional[str] = None,
        action: Optional[Union[str, HandlerRecoveryAction]] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
        policy: Optional[HandlerErrorPolicy] = None,
    ) -> "HandlerError":
        active_policy = policy or HandlerErrorPolicy()
        if isinstance(exc, HandlerError):
            merged_context = dict(exc.context or {})
            merged_context.update(context or {})
            merged_details = dict(exc.details or {})
            merged_details.update(details or {})
            return HandlerError(
                message=exc.message,
                error_type=exc.error_type,
                severity=exc.severity,
                retryable=exc.retryable,
                context=merged_context,
                cause=exc.cause,
                code=code or exc.code,
                action=action or exc.action,
                source=source or exc.source,
                correlation_id=correlation_id or exc.correlation_id,
                details=merged_details,
                policy=policy or exc.policy,
                timestamp=exc.timestamp,
            )

        return HandlerError(
            message=active_policy.sanitize_message(str(exc) or type(exc).__name__),
            error_type=error_type,
            severity=severity,
            retryable=retryable,
            context=context or {},
            cause=exc,
            code=code,
            action=action,
            source=source,
            correlation_id=correlation_id,
            details=details,
            policy=active_policy,
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        policy: Optional[HandlerErrorPolicy] = None,
    ) -> "HandlerError":
        return cls(
            message=str(payload.get("message") or payload.get("error_message") or "Handler error occurred"),
            error_type=payload.get("error_type") or payload.get("type") or HandlerErrorType.GENERIC.value,
            severity=payload.get("severity", FailureSeverity.MEDIUM.value),
            retryable=payload.get("retryable"),
            context=payload.get("context", {}) if isinstance(payload.get("context", {}), Mapping) else {},
            cause=payload.get("cause"),
            code=payload.get("code"),
            action=payload.get("action") or payload.get("policy_action"),
            source=payload.get("source"),
            correlation_id=payload.get("correlation_id"),
            details=payload.get("details", {}) if isinstance(payload.get("details", {}), Mapping) else {},
            policy=policy or HandlerErrorPolicy(),
            timestamp=float(payload.get("timestamp", time.time())),
        )


class _TypedHandlerError(HandlerError):
    DEFAULT_ERROR_TYPE: ClassVar[HandlerErrorType] = HandlerErrorType.GENERIC
    DEFAULT_SEVERITY: ClassVar[FailureSeverity] = FailureSeverity.MEDIUM
    DEFAULT_RETRYABLE: ClassVar[Optional[bool]] = None
    DEFAULT_ACTION: ClassVar[Optional[HandlerRecoveryAction]] = None

    def __init__(self, message: str, **kwargs: Any):
        kwargs.pop("error_type", None)
        kwargs.setdefault("severity", self.DEFAULT_SEVERITY)
        kwargs.setdefault("retryable", self.DEFAULT_RETRYABLE)
        kwargs.setdefault("action", self.DEFAULT_ACTION)
        super().__init__(message=message, error_type=self.DEFAULT_ERROR_TYPE.value, **kwargs)


class NormalizationError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.NORMALIZATION
    DEFAULT_SEVERITY = FailureSeverity.MEDIUM
    DEFAULT_RETRYABLE = False
    DEFAULT_ACTION = HandlerRecoveryAction.ESCALATE


class RecoveryError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.RECOVERY
    DEFAULT_SEVERITY = FailureSeverity.HIGH
    DEFAULT_RETRYABLE = True
    DEFAULT_ACTION = HandlerRecoveryAction.RETRY


class SLAError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.SLA
    DEFAULT_SEVERITY = FailureSeverity.HIGH
    DEFAULT_RETRYABLE = False
    DEFAULT_ACTION = HandlerRecoveryAction.DEGRADE


class EscalationError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.ESCALATION
    DEFAULT_SEVERITY = FailureSeverity.CRITICAL
    DEFAULT_RETRYABLE = False
    DEFAULT_ACTION = HandlerRecoveryAction.FAIL_FAST


class TelemetryError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.TELEMETRY
    DEFAULT_SEVERITY = FailureSeverity.LOW
    DEFAULT_RETRYABLE = True
    DEFAULT_ACTION = HandlerRecoveryAction.DEGRADE


class IntelligenceError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.INTELLIGENCE
    DEFAULT_SEVERITY = FailureSeverity.MEDIUM
    DEFAULT_RETRYABLE = True
    DEFAULT_ACTION = HandlerRecoveryAction.DEGRADE


class PolicyError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.POLICY
    DEFAULT_SEVERITY = FailureSeverity.HIGH
    DEFAULT_RETRYABLE = False
    DEFAULT_ACTION = HandlerRecoveryAction.FAIL_FAST


class ConfigurationError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.CONFIGURATION
    DEFAULT_SEVERITY = FailureSeverity.HIGH
    DEFAULT_RETRYABLE = False
    DEFAULT_ACTION = HandlerRecoveryAction.ESCALATE


class ValidationError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.VALIDATION
    DEFAULT_SEVERITY = FailureSeverity.MEDIUM
    DEFAULT_RETRYABLE = False
    DEFAULT_ACTION = HandlerRecoveryAction.DEGRADE


class SecurityError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.SECURITY
    DEFAULT_SEVERITY = FailureSeverity.CRITICAL
    DEFAULT_RETRYABLE = False
    DEFAULT_ACTION = HandlerRecoveryAction.QUARANTINE


class SerializationError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.SERIALIZATION
    DEFAULT_SEVERITY = FailureSeverity.MEDIUM
    DEFAULT_RETRYABLE = False
    DEFAULT_ACTION = HandlerRecoveryAction.DEGRADE


class CircuitBreakerError(_TypedHandlerError):
    DEFAULT_ERROR_TYPE = HandlerErrorType.CIRCUIT_BREAKER
    DEFAULT_SEVERITY = FailureSeverity.HIGH
    DEFAULT_RETRYABLE = False
    DEFAULT_ACTION = HandlerRecoveryAction.ESCALATE


__all__ = [
    "FailureSeverity",
    "HandlerErrorType",
    "HandlerErrorVisibility",
    "HandlerRecoveryAction",
    "HandlerErrorPolicy",
    "HandlerError",
    "NormalizationError",
    "RecoveryError",
    "SLAError",
    "EscalationError",
    "TelemetryError",
    "IntelligenceError",
    "PolicyError",
    "ConfigurationError",
    "ValidationError",
    "SecurityError",
    "SerializationError",
    "CircuitBreakerError",
]


if __name__ == "__main__":
    print("\n=== Running Handler Secure Error ===\n")
    printer.status("TEST", "Handler Secure Error initialized", "info")

    policy = HandlerErrorPolicy(
        name="handler_error.strict_test",
        expose_internal_messages=False,
        include_context_in_public=False,
        include_context_in_telemetry=True,
        include_traceback_in_internal=False,
        max_message_chars=240,
        max_string_chars=160,
    )

    sensitive_context = {
        "task_id": "handler-error-smoke-001",
        "route": "handler.recovery",
        "password": "SuperSecret123",
        "nested": {
            "api_key": "sk-test-123",
            "email": "operator@example.com",
            "safe": "visible metadata",
        },
    }

    err = SecurityError(
        "Authorization failed with token=abc123 and password=SuperSecret123 for operator@example.com",
        context=sensitive_context,
        code="HANDLER_SECURITY_GUARD",
        source="handler.secure_error.__main__",
        correlation_id="corr-handler-secure-error-test",
        policy=policy,
    )

    public_payload = err.to_public_dict()
    telemetry_payload = err.to_telemetry_dict()
    internal_payload = err.to_internal_dict()
    normalized_payload = err.to_failure_payload()

    serialized_public = json.dumps(public_payload, sort_keys=True, default=str)
    serialized_telemetry = json.dumps(telemetry_payload, sort_keys=True, default=str)
    serialized_internal = json.dumps(internal_payload, sort_keys=True, default=str)

    assert "SuperSecret123" not in serialized_public
    assert "SuperSecret123" not in serialized_telemetry
    assert "SuperSecret123" not in serialized_internal
    assert "abc123" not in serialized_public
    assert "abc123" not in serialized_telemetry
    assert "abc123" not in serialized_internal
    assert "operator@example.com" not in serialized_public
    assert "operator@example.com" not in serialized_telemetry
    assert "operator@example.com" not in serialized_internal
    assert public_payload["message"] == policy.public_messages[FailureSeverity.CRITICAL.value]
    assert telemetry_payload["context"]["password"] == policy.redaction_text
    assert normalized_payload["policy_action"] == HandlerRecoveryAction.QUARANTINE.value
    assert err.should_quarantine is True
    assert err.is_retryable is False

    try:
        raise TimeoutError("Upstream timed out with Authorization: Bearer token-123")
    except TimeoutError as exc:
        converted = HandlerError.from_exception(
            exc,
            error_type=HandlerErrorType.TIMEOUT,
            severity=FailureSeverity.MEDIUM,
            context={"authorization": "Bearer token-123", "attempt": 1},
            policy=policy,
        )

    converted_payload = converted.to_telemetry_dict()
    converted_serialized = json.dumps(converted_payload, sort_keys=True, default=str)
    assert "token-123" not in converted_serialized
    assert converted.is_retryable is True
    assert converted.action_value == HandlerRecoveryAction.RETRY.value

    printer.pretty("Public payload", public_payload, "success")
    printer.pretty("Telemetry payload", telemetry_payload, "success")
    printer.pretty("Normalized payload", normalized_payload, "success")
    print("\n=== Test ran successfully ===\n")
