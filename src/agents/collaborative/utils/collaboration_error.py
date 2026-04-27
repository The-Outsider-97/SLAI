from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import traceback

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Union

from .config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Collaborative Error")
printer = PrettyPrinter

JsonMapping = Dict[str, Any]
ErrorTypeInput = Union["CollaborationErrorType", str]
SeverityInput = Union["CollaborationSeverity", str]


class CollaborationErrorType(str, Enum):
    """Stable taxonomy for failures in the collaborative agent runtime."""

    OVERLOAD = "Collaboration System Overload"
    NO_CAPABLE_AGENT = "No Capable Agent Found"
    ROUTING_FAILURE = "Task Routing Failure"
    DELEGATION_FAILURE = "Task Delegation Failure"
    REGISTRATION_FAILURE = "Agent Registration Failure"
    SHARED_MEMORY_FAILURE = "Shared Memory Access Failure"


class CollaborationSeverity(str, Enum):
    """Normalized severity values used in audit records and reports."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class CollaborationErrorConfig:
    """Runtime options for collaboration error serialization and reporting.

    Values are intentionally conservative and are safe when the YAML config only
    defines a subset of the keys. The current project config contains
    `collaboration_error.log_errors`, while the rest of these options can be
    added incrementally without changing this module.
    """

    forensic_hash_algorithm: str = "sha256"
    error_id_hash_algorithm: str = "sha256"
    forensic_hash_salt: str = "collaboration"
    error_id_length: int = 12
    report_format: str = "markdown"
    include_forensic_hash: bool = True
    include_context: bool = True
    include_collaborative_agent_state: bool = True
    include_remediation_guidance: bool = True
    include_traceback: bool = False
    traceback_limit: int = 8
    log_errors: bool = True
    redact_sensitive_fields: bool = True
    sensitive_field_names: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "api_key",
                "apikey",
                "auth",
                "authorization",
                "bearer",
                "client_secret",
                "credential",
                "credentials",
                "jwt",
                "password",
                "private_key",
                "refresh_token",
                "secret",
                "session",
                "token",
            }
        )
    )

    @classmethod
    def from_runtime(cls) -> "CollaborationErrorConfig":
        global_config = _safe_load_global_config()
        error_config = _safe_get_config_section("collaboration_error")

        def pick(key: str, default: Any) -> Any:
            if key in error_config:
                return error_config[key]
            return global_config.get(key, default)

        configured_sensitive_fields = pick("sensitive_field_names", None)
        if configured_sensitive_fields is None:
            sensitive_fields = cls().sensitive_field_names
        else:
            sensitive_fields = frozenset(str(item).strip().lower() for item in configured_sensitive_fields if str(item).strip())

        return cls(
            forensic_hash_algorithm=str(pick("forensic_hash_algorithm", cls.forensic_hash_algorithm)).lower(),
            error_id_hash_algorithm=str(pick("error_id_hash_algorithm", cls.error_id_hash_algorithm)).lower(),
            forensic_hash_salt=str(pick("forensic_hash_salt", cls.forensic_hash_salt)),
            error_id_length=_bounded_int(pick("error_id_length", cls.error_id_length), minimum=8, maximum=64, default=cls.error_id_length),
            report_format=str(pick("report_format", cls.report_format)).strip().lower() or cls.report_format,
            include_forensic_hash=_as_bool(pick("include_forensic_hash", cls.include_forensic_hash)),
            include_context=_as_bool(pick("include_context", cls.include_context)),
            include_collaborative_agent_state=_as_bool(
                pick("include_collaborative_agent_state", cls.include_collaborative_agent_state)
            ),
            include_remediation_guidance=_as_bool(
                pick("include_remediation_guidance", cls.include_remediation_guidance)
            ),
            include_traceback=_as_bool(pick("include_traceback", cls.include_traceback)),
            traceback_limit=_bounded_int(pick("traceback_limit", cls.traceback_limit), minimum=1, maximum=50, default=cls.traceback_limit),
            log_errors=_as_bool(pick("log_errors", cls.log_errors)),
            redact_sensitive_fields=_as_bool(pick("redact_sensitive_fields", cls.redact_sensitive_fields)),
            sensitive_field_names=sensitive_fields,
        )


class CollaborationError(Exception):
    """Base exception for collaborative agent failures.

    The class is designed to be useful both at runtime and after the fact:
    every instance receives a compact error id, a tamper-evident forensic hash,
    normalized severity, redacted audit payloads, optional causal exception
    details, and markdown/JSON incident reporting.
    """

    def __init__(
        self,
        error_type: ErrorTypeInput,
        message: str,
        severity: SeverityInput = CollaborationSeverity.MEDIUM,
        context: Optional[Mapping[str, Any]] = None,
        collaborative_agent_state: Optional[Mapping[str, Any]] = None,
        remediation_guidance: Optional[str] = None,
        *,
        cause: Optional[BaseException] = None,
        retryable: Optional[bool] = None,
        component: str = "collaboration",
        config: Optional[CollaborationErrorConfig] = None,
    ):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("CollaborationError message must be a non-empty string.")

        super().__init__(message)
        self.raw_message = message.strip()
        self.error_type = _coerce_error_type(error_type)
        self.severity = _coerce_severity(severity).value
        self.context = _coerce_mapping(context, field_name="context")
        self.collaborative_agent_state = _coerce_mapping(
            collaborative_agent_state,
            field_name="collaborative_agent_state",
        )
        self.remediation_guidance = remediation_guidance
        self.component = component or "collaboration"
        self.cause = cause
        self.__cause__ = cause
        self.retryable = _default_retryable(self.error_type) if retryable is None else bool(retryable)
        self.config = config or CollaborationErrorConfig.from_runtime()
        self.timestamp = time.time()
        self.timestamp_utc = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        self.error_id = self._generate_error_id()
        self.forensic_hash = self._generate_forensic_hash()

        if self.config.log_errors:
            self._log_creation()

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        error_type: ErrorTypeInput = CollaborationErrorType.ROUTING_FAILURE,
        message: Optional[str] = None,
        severity: SeverityInput = CollaborationSeverity.HIGH,
        context: Optional[Mapping[str, Any]] = None,
        collaborative_agent_state: Optional[Mapping[str, Any]] = None,
        remediation_guidance: Optional[str] = None,
        retryable: Optional[bool] = None,
        component: str = "collaboration",
    ) -> "CollaborationError":
        """Wrap an arbitrary exception while preserving cause metadata."""

        return cls(
            error_type=error_type,
            message=message or f"{type(exc).__name__}: {exc}",
            severity=severity,
            context=context,
            collaborative_agent_state=collaborative_agent_state,
            remediation_guidance=remediation_guidance,
            cause=exc,
            retryable=retryable,
            component=component,
        )

    def _generate_error_id(self) -> str:
        hasher = _new_hash(self.config.error_id_hash_algorithm)
        entropy = ":".join(
            [
                str(time.time_ns()),
                secrets.token_hex(16),
                str(os.getpid()),
                self.error_type.name,
                self.severity,
            ]
        )
        hasher.update(entropy.encode("utf-8"))
        return hasher.hexdigest()[: self.config.error_id_length]

    def _generate_forensic_hash(self) -> str:
        hasher = _new_hash(self.config.forensic_hash_algorithm)
        audit_source = {
            "component": self.component,
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "error_type": self.error_type.value,
            "error_type_code": self.error_type.name,
            "severity": self.severity,
            "message": self.raw_message,
            "retryable": self.retryable,
            "context": self._redacted(self.context),
            "collaborative_agent_state_snapshot": self._redacted(self.collaborative_agent_state),
            "remediation_guidance": self.remediation_guidance,
            "cause": self._cause_summary(),
        }
        hasher.update(self.config.forensic_hash_salt.encode("utf-8"))
        hasher.update(_json_dumps(audit_source).encode("utf-8"))
        return hasher.hexdigest()

    def _cause_summary(self) -> Optional[JsonMapping]:
        if self.cause is None:
            return None
        return {
            "type": type(self.cause).__name__,
            "module": type(self.cause).__module__,
            "message": str(self.cause),
        }

    def _cause_traceback(self) -> Optional[list[str]]:
        if self.cause is None or self.cause.__traceback__ is None:
            return None
        return traceback.format_exception(
            type(self.cause),
            self.cause,
            self.cause.__traceback__,
            limit=self.config.traceback_limit,
        )

    def _redacted(self, value: Any) -> Any:
        if not self.config.redact_sensitive_fields:
            return _json_safe(value)
        return _redact(value, self.config.sensitive_field_names)

    def _log_creation(self) -> None:
        payload = {
            "error_id": self.error_id,
            "error_type": self.error_type.name,
            "severity": self.severity,
            "retryable": self.retryable,
        }
        log_message = "Collaboration error created: %s"
        if self.severity == CollaborationSeverity.CRITICAL.value:
            logger.critical(log_message, payload)
        elif self.severity == CollaborationSeverity.HIGH.value:
            logger.error(log_message, payload)
        else:
            logger.warning(log_message, payload)

    def to_audit_format(
        self,
        *,
        include_traceback: Optional[bool] = None,
        redacted: bool = True,
    ) -> JsonMapping:
        """Return a JSON-serializable audit payload.

        Args:
            include_traceback: Overrides the configured traceback inclusion flag.
            redacted: When True, sensitive context/state fields are replaced with
                `[REDACTED]`. Keep True for logs, reports, telemetry, and user-visible
                output.
        """

        include_tb = self.config.include_traceback if include_traceback is None else bool(include_traceback)
        context = self._redacted(self.context) if redacted else _json_safe(self.context)
        state = self._redacted(self.collaborative_agent_state) if redacted else _json_safe(self.collaborative_agent_state)

        payload: JsonMapping = {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "component": self.component,
            "error_type": self.error_type.value,
            "error_type_code": self.error_type.name,
            "severity": self.severity,
            "message": str(self),
            "raw_message": self.raw_message,
            "retryable": self.retryable,
            "forensic_hash": self.forensic_hash,
            "context": context,
            "collaborative_agent_state_snapshot": state,
            "remediation_guidance": self.remediation_guidance,
            "cause": self._cause_summary(),
        }
        if include_tb:
            payload["traceback"] = self._cause_traceback()
        return payload

    def to_json(self, *, indent: Optional[int] = 2, include_traceback: Optional[bool] = None) -> str:
        return json.dumps(
            self.to_audit_format(include_traceback=include_traceback),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )

    def generate_report(self, report_format: Optional[str] = None) -> str:
        """Render the incident as JSON or markdown according to config/argument."""

        output_format = (report_format or self.config.report_format or "markdown").strip().lower()
        if output_format == "json":
            return self.to_json(indent=2)
        if output_format not in {"markdown", "md"}:
            logger.warning("Unsupported collaboration error report format '%s'; falling back to markdown.", output_format)

        audit = self.to_audit_format()
        lines = [
            "# Collaboration Incident Report",
            f"**Generated**: {audit['timestamp_utc']}",
            f"**Component**: `{audit['component']}`",
            f"**Error ID**: `{audit['error_id']}`",
            f"**Error Type**: {audit['error_type']} (`{audit['error_type_code']}`)",
            f"**Severity**: {str(audit['severity']).upper()}",
            f"**Retryable**: {audit['retryable']}",
            "---",
            f"**Message**: {audit['message']}",
        ]

        if self.config.include_forensic_hash:
            lines.append(f"**Forensic Hash**: `{audit['forensic_hash']}`")

        if audit.get("cause"):
            lines.extend(["", "## Cause", "```json", _json_dumps(audit["cause"], indent=2), "```"])

        if self.config.include_context:
            lines.extend(["", "## Context Details", "```json", _json_dumps(audit["context"], indent=2), "```"])

        if self.config.include_collaborative_agent_state:
            lines.extend(
                [
                    "",
                    "## Collaborative Agent State",
                    "```json",
                    _json_dumps(audit["collaborative_agent_state_snapshot"], indent=2),
                    "```",
                ]
            )

        if self.config.include_remediation_guidance:
            guidance = audit.get("remediation_guidance") or "No specific guidance provided."
            lines.extend(["", "## Remediation Guidance", str(guidance)])

        if audit.get("traceback"):
            lines.extend(["", "## Traceback", "```text", "".join(audit["traceback"]), "```"])

        return "\n".join(lines)

    def with_context(self, **context: Any) -> "CollaborationError":
        """Mutate-in-place helper for adding lightweight diagnostic context.

        The forensic hash is regenerated so the audit payload remains internally
        consistent after the context update.
        """

        self.context.update(context)
        self.forensic_hash = self._generate_forensic_hash()
        return self

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(error_id={self.error_id!r}, "
            f"error_type={self.error_type.name!r}, severity={self.severity!r}, "
            f"message={self.raw_message!r})"
        )

    def __str__(self) -> str:
        return f"[{self.error_type.name} - {self.severity.upper()}] {self.raw_message}"


class OverloadError(CollaborationError):
    def __init__(
        self,
        message: str = "Collaboration system load exceeded.",
        context: Optional[Mapping[str, Any]] = None,
        *,
        current_load: Optional[int] = None,
        max_load: Optional[int] = None,
        collaborative_agent_state: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        merged_context = _merge_context(
            context,
            current_load=current_load,
            max_load=max_load,
        )
        super().__init__(
            CollaborationErrorType.OVERLOAD,
            message,
            severity=CollaborationSeverity.HIGH,
            context=merged_context,
            collaborative_agent_state=collaborative_agent_state,
            remediation_guidance="Scale out workers, reduce queue depth, or throttle incoming tasks.",
            cause=cause,
            retryable=True,
        )


class NoCapableAgentError(CollaborationError):
    def __init__(self, task_type: str, required_capabilities: Optional[Sequence[str]] = None, *,
                 available_agents: Optional[Mapping[str, Any]] = None,
                 context: Optional[Mapping[str, Any]] = None,
                 collaborative_agent_state: Optional[Mapping[str, Any]] = None):
        required = list(required_capabilities or [])
        merged_context = _merge_context(
            context,
            task_type=task_type,
            required_capabilities=required,
            available_agents=available_agents,
        )
        super().__init__(
            CollaborationErrorType.NO_CAPABLE_AGENT,
            f"No capable agent found for task type '{task_type}'.",
            severity=CollaborationSeverity.MEDIUM,
            context=merged_context,
            collaborative_agent_state=collaborative_agent_state,
            remediation_guidance="Register a compatible agent, repair agent health, or relax task capability requirements.",
            retryable=False,
        )


class RoutingFailureError(CollaborationError):
    def __init__(self, task_type: str, reason: str, *,
        attempted_agents: Optional[Sequence[str]] = None,
        context: Optional[Mapping[str, Any]] = None,
        collaborative_agent_state: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        merged_context = _merge_context(
            context,
            task_type=task_type,
            reason=reason,
            attempted_agents=list(attempted_agents or []),
        )
        super().__init__(
            CollaborationErrorType.ROUTING_FAILURE,
            f"Routing failed for task type '{task_type}': {reason}",
            severity=CollaborationSeverity.HIGH,
            context=merged_context,
            collaborative_agent_state=collaborative_agent_state,
            remediation_guidance="Review routing policy, fallback order, agent runtime health, and reliability circuit state.",
            cause=cause,
            retryable=True,
        )


class DelegationFailureError(CollaborationError):
    def __init__(self, task_type: str, agent_name: str, reason: str, *, context: Optional[Mapping[str, Any]] = None,
                 collaborative_agent_state: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None):
        merged_context = _merge_context(
            context,
            task_type=task_type,
            agent_name=agent_name,
            reason=reason,
        )
        super().__init__(
            CollaborationErrorType.DELEGATION_FAILURE,
            f"Delegation to agent '{agent_name}' failed for task type '{task_type}': {reason}",
            severity=CollaborationSeverity.HIGH,
            context=merged_context,
            collaborative_agent_state=collaborative_agent_state,
            remediation_guidance="Inspect the selected agent, task payload contract, retry policy, and downstream dependency health.",
            cause=cause,
            retryable=True,
        )


class RegistrationFailureError(CollaborationError):
    def __init__(self, agent_name: str, reason: str, *, capabilities: Optional[Sequence[str]] = None,
                 context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None):
        merged_context = _merge_context(
            context,
            agent_name=agent_name,
            reason=reason,
            capabilities=list(capabilities or []),
        )
        super().__init__(
            CollaborationErrorType.REGISTRATION_FAILURE,
            f"Agent registration failed for '{agent_name}': {reason}",
            severity=CollaborationSeverity.MEDIUM,
            context=merged_context,
            remediation_guidance="Validate agent metadata, declared capabilities, version compatibility, and constructor dependencies.",
            cause=cause,
            retryable=False,
        )


class SharedMemoryFailureError(CollaborationError):
    def __init__(self, operation: str, reason: str, *, key: Optional[str] = None,
                 context: Optional[Mapping[str, Any]] = None,
                 cause: Optional[BaseException] = None):
        merged_context = _merge_context(
            context,
            operation=operation,
            key=key,
            reason=reason,
        )
        target = f" for key '{key}'" if key else ""
        super().__init__(
            CollaborationErrorType.SHARED_MEMORY_FAILURE,
            f"Shared memory operation '{operation}' failed{target}: {reason}",
            severity=CollaborationSeverity.HIGH,
            context=merged_context,
            remediation_guidance="Check shared-memory availability, locks, TTL/eviction pressure, serialization compatibility, and storage limits.",
            cause=cause,
            retryable=True,
        )


# Backwards-compatible aliases for callers that use shorter names.
AgentRegistrationError = RegistrationFailureError
SharedMemoryAccessError = SharedMemoryFailureError


def as_collaboration_error(exc: BaseException, *,
    error_type: ErrorTypeInput = CollaborationErrorType.ROUTING_FAILURE,
    message: Optional[str] = None,
    severity: SeverityInput = CollaborationSeverity.HIGH,
    context: Optional[Mapping[str, Any]] = None,
    collaborative_agent_state: Optional[Mapping[str, Any]] = None,
    remediation_guidance: Optional[str] = None,
) -> CollaborationError:
    """Normalize arbitrary exceptions into CollaborationError instances."""

    if isinstance(exc, CollaborationError):
        if context:
            exc.with_context(**dict(context))
        return exc
    return CollaborationError.from_exception(
        exc,
        error_type=error_type,
        message=message,
        severity=severity,
        context=context,
        collaborative_agent_state=collaborative_agent_state,
        remediation_guidance=remediation_guidance,
    )


def _safe_load_global_config() -> JsonMapping:
    if load_global_config is None:
        return {}
    try:
        loaded = load_global_config()
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        logger.debug("Unable to load global collaboration config while creating error: %s", exc)
        return {}


def _safe_get_config_section(section_name: str) -> JsonMapping:
    if get_config_section is None:
        return {}
    try:
        section = get_config_section(section_name)
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Unable to load config section '%s' while creating error: %s", section_name, exc)
        return {}


def _new_hash(algorithm: str):
    try:
        return hashlib.new((algorithm or "sha256").lower())
    except (TypeError, ValueError):
        logger.warning("Unsupported hash algorithm '%s'; falling back to sha256.", algorithm)
        return hashlib.sha256()


def _coerce_error_type(value: ErrorTypeInput) -> CollaborationErrorType:
    if isinstance(value, CollaborationErrorType):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        for member in CollaborationErrorType:
            if normalized == member.value or normalized.upper() == member.name:
                return member
    raise ValueError(f"Unsupported collaboration error type: {value!r}")


def _coerce_severity(value: SeverityInput) -> CollaborationSeverity:
    if isinstance(value, CollaborationSeverity):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for member in CollaborationSeverity:
            if normalized == member.value or normalized.upper() == member.name:
                return member
    raise ValueError(f"Unsupported collaboration error severity: {value!r}")


def _coerce_mapping(value: Optional[Mapping[str, Any]], *, field_name: str) -> JsonMapping:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping/dict when provided.")
    return dict(value)


def _merge_context(base: Optional[Mapping[str, Any]], **values: Any) -> JsonMapping:
    merged = dict(base or {})
    for key, value in values.items():
        if value is not None:
            merged[key] = value
    return merged


def _default_retryable(error_type: CollaborationErrorType) -> bool:
    return error_type in {
        CollaborationErrorType.OVERLOAD,
        CollaborationErrorType.ROUTING_FAILURE,
        CollaborationErrorType.DELEGATION_FAILURE,
        CollaborationErrorType.SHARED_MEMORY_FAILURE,
    }


def _bounded_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _redact(value: Any, sensitive_fields: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        redacted: JsonMapping = {}
        for key, item in value.items():
            key_str = str(key)
            normalized_key = key_str.strip().lower()
            if _is_sensitive_key(normalized_key, sensitive_fields):
                redacted[key_str] = "[REDACTED]"
            else:
                redacted[key_str] = _redact(item, sensitive_fields)
        return redacted

    if isinstance(value, (str, bytes, bytearray)) or value is None:
        return _json_safe(value)

    if isinstance(value, Sequence):
        return [_redact(item, sensitive_fields) for item in value]

    if isinstance(value, set):
        return sorted((_redact(item, sensitive_fields) for item in value), key=repr)

    return _json_safe(value)


def _is_sensitive_key(key: str, sensitive_fields: frozenset[str]) -> bool:
    if key in sensitive_fields:
        return True
    return any(marker in key for marker in sensitive_fields if len(marker) >= 5)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=_json_default)
        return value
    except (TypeError, ValueError):
        return _json_default(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {key: _json_safe(item) for key, item in vars(value).items() if not key.startswith("_")}
        except Exception:
            pass
    return repr(value)


def _json_dumps(value: Any, *, indent: Optional[int] = None) -> str:
    return json.dumps(
        value,
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    )


if __name__ == "__main__":
    overload = OverloadError(
        "System load exceeded (120/100)",
        context={"current_load": 120, "max_load": 100, "token": "secret-value"},
    )
    assert overload.error_type == CollaborationErrorType.OVERLOAD
    assert overload.error_id
    assert overload.forensic_hash
    assert overload.retryable is True
    assert "System load exceeded" in str(overload)
    assert overload.to_audit_format()["context"]["token"] == "[REDACTED]"

    routing = RoutingFailureError("translate", "all workers failed", attempted_agents=["A", "B"])
    routing_report = routing.generate_report()
    assert "Routing failed" in routing_report
    assert "Collaboration Incident Report" in routing_report

    no_agent = NoCapableAgentError("summarize", ["nlp", "summarization"])
    audit = no_agent.to_audit_format()
    assert audit["error_type"] == CollaborationErrorType.NO_CAPABLE_AGENT.value
    assert audit["context"]["required_capabilities"] == ["nlp", "summarization"]
    assert no_agent.retryable is False

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        wrapped = as_collaboration_error(exc, context={"task_type": "translate"})
        assert wrapped.cause is exc
        assert wrapped.to_audit_format()["cause"]["type"] == "RuntimeError"

    print("All collaboration_error.py tests passed.")
