"""
Structured error taxonomy and normalization helpers for SLAI's Network Agent.

This module is intentionally richer than a minimal exception file because the
Network Agent sits at the boundary of transport orchestration, routing,
reliability control, lifecycle management, and policy enforcement. The error
layer therefore has to serve multiple roles at once:

1. Preserve enough context for reliable debugging and observability.
2. Expose stable error codes for agent-to-agent and memory integration.
3. Classify retryability/transience for recovery coordinators.
4. Normalize third-party / stdlib failures into a SLAI-native taxonomy.
5. Produce JSON-safe payloads for telemetry, logs, and shared memory.
"""

from __future__ import annotations

import json
import socket
import ssl
import traceback

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, TypeVar

class NetworkErrorSeverity(str, Enum):
    """Operational severity used for logging, telemetry, and escalation."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NetworkErrorCategory(str, Enum):
    """Coarse-grained domain classification for network failures."""

    CONFIGURATION = "configuration"
    ADAPTER = "adapter"
    CONNECTION = "connection"
    TRANSPORT = "transport"
    ROUTING = "routing"
    DELIVERY = "delivery"
    LIFECYCLE = "lifecycle"
    RELIABILITY = "reliability"
    POLICY = "policy"
    SECURITY = "security"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    THROTTLING = "throttling"
    PAYLOAD = "payload"
    SERIALIZATION = "serialization"
    UNKNOWN = "unknown"


class RetryDisposition(str, Enum):
    """
    Retry policy hint consumed by reliability orchestration.

    NEVER:
        Do not retry under any normal circumstances.
    SAFE:
        Retry is generally safe and expected.
    CONDITIONAL:
        Retry may be possible, but only when the caller confirms contextual
        safety (for example, idempotent operations or a different route).
    REQUIRED_FAILOVER:
        Retry should happen only through rerouting / failover.
    """

    NEVER = "never"
    SAFE = "safe"
    CONDITIONAL = "conditional"
    REQUIRED_FAILOVER = "required_failover"


PrimitiveJSON = Optional[str | int | float | bool]
JSONLike = PrimitiveJSON | Dict[str, Any] | Sequence[Any]


@dataclass(frozen=True, slots=True)
class NetworkErrorContext:
    """
    Structured operational context carried by `NetworkError`.

    This keeps the top-level exception constructor manageable while still giving
    routing, delivery, reliability, and policy modules enough surface area to
    attach precise metadata.
    """

    operation: Optional[str] = None
    channel: Optional[str] = None
    endpoint: Optional[str] = None
    protocol: Optional[str] = None
    route: Optional[str] = None
    correlation_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    session_id: Optional[str] = None
    policy_name: Optional[str] = None
    circuit_state: Optional[str] = None
    status_code: Optional[int] = None
    timeout_ms: Optional[int] = None
    payload_size: Optional[int] = None
    attempt: Optional[int] = None
    max_attempts: Optional[int] = None
    retry_after_ms: Optional[int] = None
    tls_required: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "operation": self.operation,
            "channel": self.channel,
            "endpoint": self.endpoint,
            "protocol": self.protocol,
            "route": self.route,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "session_id": self.session_id,
            "policy_name": self.policy_name,
            "circuit_state": self.circuit_state,
            "status_code": self.status_code,
            "timeout_ms": self.timeout_ms,
            "payload_size": self.payload_size,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "retry_after_ms": self.retry_after_ms,
            "tls_required": self.tls_required,
            "metadata": _json_safe(self.metadata),
        }
        return {key: value for key, value in data.items() if value is not None and value != {}}

    @classmethod
    def from_value(cls, value: Optional["NetworkErrorContext | Mapping[str, Any]"]) -> "NetworkErrorContext":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            raw = dict(value)
            known_fields = {
                "operation",
                "channel",
                "endpoint",
                "protocol",
                "route",
                "correlation_id",
                "idempotency_key",
                "session_id",
                "policy_name",
                "circuit_state",
                "status_code",
                "timeout_ms",
                "payload_size",
                "attempt",
                "max_attempts",
                "retry_after_ms",
                "tls_required",
                "metadata",
            }
            kwargs = {k: raw[k] for k in known_fields if k in raw}
            metadata = dict(kwargs.get("metadata") or {})
            for key, value_ in raw.items():
                if key not in known_fields and value_ is not None:
                    metadata[key] = value_
            kwargs["metadata"] = metadata
            return cls(**kwargs)
        raise TypeError(f"Unsupported context type: {type(value)!r}")

    def merged(self, overrides: Optional[Mapping[str, Any]] = None) -> "NetworkErrorContext":
        if not overrides:
            return self
        base = self.to_dict()
        metadata = dict(base.pop("metadata", {}))
        for key, value in dict(overrides).items():
            if key == "metadata" and isinstance(value, Mapping):
                metadata.update(dict(value))
            elif key in base or key in self.__dataclass_fields__:
                base[key] = value
            else:
                metadata[key] = value
        if metadata:
            base["metadata"] = metadata
        return NetworkErrorContext.from_value(base)


class NetworkError(Exception):
    """
    Base SLAI-native network exception.

    The class exposes stable machine-readable properties alongside a rich context
    object so that the same exception can support:
    - debug logs,
    - observability events,
    - shared memory snapshots,
    - retry/failover decisions,
    - higher-level policy escalation.
    """

    default_code = "NETWORK_ERROR"
    default_category = NetworkErrorCategory.UNKNOWN
    default_severity = NetworkErrorSeverity.ERROR
    default_retryable = False
    default_transient = False
    default_retry_disposition = RetryDisposition.NEVER
    default_alertable = False

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        category: Optional[NetworkErrorCategory] = None,
        severity: Optional[NetworkErrorSeverity] = None,
        retryable: Optional[bool] = None,
        transient: Optional[bool] = None,
        retry_disposition: Optional[RetryDisposition] = None,
        alertable: Optional[bool] = None,
        context: Optional[NetworkErrorContext | Mapping[str, Any]] = None,
        details: Optional[Mapping[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
        occurred_at: Optional[datetime] = None,
        **context_overrides: Any,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.code = code or self.default_code
        self.category = category or self.default_category
        self.severity = severity or self.default_severity
        self.retryable = self.default_retryable if retryable is None else bool(retryable)
        self.transient = self.default_transient if transient is None else bool(transient)
        self.retry_disposition = retry_disposition or self.default_retry_disposition
        self.alertable = self.default_alertable if alertable is None else bool(alertable)
        self.context = NetworkErrorContext.from_value(context).merged(context_overrides)
        self.details: Dict[str, Any] = dict(details or {})
        self.tags = tuple(dict.fromkeys(tags or ()))
        self.cause = cause
        self.occurred_at = occurred_at or _utcnow()

    @property
    def root_cause(self) -> BaseException | None:
        cause = self.cause
        while isinstance(cause, NetworkError) and cause.cause is not None:
            cause = cause.cause
        return cause

    @property
    def root_cause_type(self) -> Optional[str]:
        cause = self.root_cause
        return type(cause).__name__ if cause is not None else None

    @property
    def status_code(self) -> Optional[int]:
        return self.context.status_code

    @property
    def correlation_id(self) -> Optional[str]:
        return self.context.correlation_id

    @property
    def operation(self) -> Optional[str]:
        return self.context.operation

    def with_context(self, **overrides: Any) -> "NetworkError":
        return self.__class__(
            self.message,
            code=self.code,
            category=self.category,
            severity=self.severity,
            retryable=self.retryable,
            transient=self.transient,
            retry_disposition=self.retry_disposition,
            alertable=self.alertable,
            context=self.context.merged(overrides),
            details=self.details,
            tags=self.tags,
            cause=self.cause,
            occurred_at=self.occurred_at,
        )

    def to_dict(self, *, include_cause: bool = True, include_traceback: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "category": self.category.value,
            "severity": self.severity.value,
            "retryable": self.retryable,
            "transient": self.transient,
            "retry_disposition": self.retry_disposition.value,
            "alertable": self.alertable,
            "occurred_at": self.occurred_at.isoformat(),
            "context": self.context.to_dict(),
            "details": _json_safe(self.details),
            "tags": list(self.tags),
        }
        if include_cause and self.cause is not None:
            payload["cause"] = _serialize_exception(self.cause, include_traceback=include_traceback)
        return payload

    def to_memory_snapshot(self) -> Dict[str, Any]:
        """Compact snapshot suitable for shared-memory persistence."""
        payload = self.to_dict(include_cause=True, include_traceback=False)
        context = payload.get("context", {})
        snapshot = {
            "error_type": payload["type"],
            "error_code": payload["code"],
            "category": payload["category"],
            "severity": payload["severity"],
            "message": payload["message"],
            "retryable": payload["retryable"],
            "transient": payload["transient"],
            "retry_disposition": payload["retry_disposition"],
            "occurred_at": payload["occurred_at"],
            "operation": context.get("operation"),
            "channel": context.get("channel"),
            "endpoint": context.get("endpoint"),
            "protocol": context.get("protocol"),
            "route": context.get("route"),
            "correlation_id": context.get("correlation_id"),
            "attempt": context.get("attempt"),
            "max_attempts": context.get("max_attempts"),
            "status_code": context.get("status_code"),
            "policy_name": context.get("policy_name"),
            "circuit_state": context.get("circuit_state"),
            "cause_type": payload.get("cause", {}).get("type") if isinstance(payload.get("cause"), Mapping) else None,
        }
        return {k: v for k, v in snapshot.items() if v is not None}

    def __str__(self) -> str:
        suffix_parts = []
        if self.code:
            suffix_parts.append(f"code={self.code}")
        if self.operation:
            suffix_parts.append(f"operation={self.operation}")
        if self.context.channel:
            suffix_parts.append(f"channel={self.context.channel}")
        if self.context.endpoint:
            suffix_parts.append(f"endpoint={self.context.endpoint}")
        if not suffix_parts:
            return self.message
        return f"{self.message} ({', '.join(suffix_parts)})"


class NetworkConfigurationError(NetworkError):
    default_code = "NETWORK_CONFIGURATION_ERROR"
    default_category = NetworkErrorCategory.CONFIGURATION
    default_severity = NetworkErrorSeverity.ERROR


class AdapterError(NetworkError):
    default_code = "ADAPTER_ERROR"
    default_category = NetworkErrorCategory.ADAPTER
    default_severity = NetworkErrorSeverity.ERROR


class AdapterNotFoundError(AdapterError):
    default_code = "ADAPTER_NOT_FOUND"
    default_retry_disposition = RetryDisposition.CONDITIONAL


class AdapterInitializationError(AdapterError):
    default_code = "ADAPTER_INITIALIZATION_FAILED"
    default_retry_disposition = RetryDisposition.CONDITIONAL


class AdapterCapabilityError(AdapterError):
    default_code = "ADAPTER_CAPABILITY_MISMATCH"
    default_retry_disposition = RetryDisposition.CONDITIONAL


class NetworkConnectionError(NetworkError):
    default_code = "NETWORK_CONNECTION_ERROR"
    default_category = NetworkErrorCategory.CONNECTION
    default_retryable = True
    default_transient = True
    default_retry_disposition = RetryDisposition.SAFE


class DNSResolutionError(NetworkConnectionError):
    default_code = "DNS_RESOLUTION_FAILED"
    default_retry_disposition = RetryDisposition.CONDITIONAL


class ConnectionTimeoutError(NetworkConnectionError):
    default_code = "CONNECTION_TIMEOUT"


class ConnectionRejectedError(NetworkConnectionError):
    default_code = "CONNECTION_REJECTED"
    default_retry_disposition = RetryDisposition.REQUIRED_FAILOVER


class TLSHandshakeError(NetworkConnectionError):
    default_code = "TLS_HANDSHAKE_FAILED"
    default_category = NetworkErrorCategory.SECURITY
    default_retry_disposition = RetryDisposition.CONDITIONAL


class SessionUnavailableError(NetworkConnectionError):
    default_code = "SESSION_UNAVAILABLE"
    default_retry_disposition = RetryDisposition.CONDITIONAL


class SessionClosedError(NetworkConnectionError):
    default_code = "SESSION_CLOSED"


class NetworkTransportError(NetworkError):
    default_code = "NETWORK_TRANSPORT_ERROR"
    default_category = NetworkErrorCategory.TRANSPORT
    default_retryable = True
    default_transient = True
    default_retry_disposition = RetryDisposition.SAFE


class SendFailureError(NetworkTransportError):
    default_code = "SEND_FAILED"


class ReceiveFailureError(NetworkTransportError):
    default_code = "RECEIVE_FAILED"


class AcknowledgementError(NetworkTransportError):
    default_code = "ACKNOWLEDGEMENT_FAILED"
    default_retry_disposition = RetryDisposition.CONDITIONAL


class NegativeAcknowledgementError(AcknowledgementError):
    default_code = "NEGATIVE_ACKNOWLEDGEMENT"
    default_retry_disposition = RetryDisposition.REQUIRED_FAILOVER


class PayloadError(NetworkError):
    default_code = "PAYLOAD_ERROR"
    default_category = NetworkErrorCategory.PAYLOAD
    default_severity = NetworkErrorSeverity.ERROR


class PayloadValidationError(PayloadError):
    default_code = "PAYLOAD_VALIDATION_FAILED"
    default_retryable = False
    default_retry_disposition = RetryDisposition.NEVER


class PayloadSerializationError(PayloadError):
    default_code = "PAYLOAD_SERIALIZATION_FAILED"
    default_category = NetworkErrorCategory.SERIALIZATION


class PayloadDeserializationError(PayloadError):
    default_code = "PAYLOAD_DESERIALIZATION_FAILED"
    default_category = NetworkErrorCategory.SERIALIZATION


class PayloadTooLargeError(PayloadError):
    default_code = "PAYLOAD_TOO_LARGE"
    default_retry_disposition = RetryDisposition.CONDITIONAL


class RoutingError(NetworkError):
    default_code = "ROUTING_ERROR"
    default_category = NetworkErrorCategory.ROUTING
    default_retryable = True
    default_transient = True
    default_retry_disposition = RetryDisposition.REQUIRED_FAILOVER


class NoRouteAvailableError(RoutingError):
    default_code = "NO_ROUTE_AVAILABLE"
    default_retryable = False
    default_transient = False
    default_retry_disposition = RetryDisposition.NEVER


class EndpointUnavailableError(RoutingError):
    default_code = "ENDPOINT_UNAVAILABLE"


class EndpointDegradedError(RoutingError):
    default_code = "ENDPOINT_DEGRADED"


class ProtocolNegotiationError(RoutingError):
    default_code = "PROTOCOL_NEGOTIATION_FAILED"
    default_retry_disposition = RetryDisposition.CONDITIONAL


class ReliabilityError(NetworkError):
    default_code = "RELIABILITY_ERROR"
    default_category = NetworkErrorCategory.RELIABILITY
    default_severity = NetworkErrorSeverity.ERROR


class CircuitBreakerOpenError(ReliabilityError):
    default_code = "CIRCUIT_BREAKER_OPEN"
    default_retryable = False
    default_retry_disposition = RetryDisposition.REQUIRED_FAILOVER
    default_severity = NetworkErrorSeverity.WARNING


class RetryExhaustedError(ReliabilityError):
    default_code = "RETRY_EXHAUSTED"
    default_retryable = False
    default_retry_disposition = RetryDisposition.REQUIRED_FAILOVER


class FailoverExhaustedError(ReliabilityError):
    default_code = "FAILOVER_EXHAUSTED"
    default_retryable = False
    default_retry_disposition = RetryDisposition.NEVER
    default_severity = NetworkErrorSeverity.CRITICAL
    default_alertable = True


class DeliveryError(NetworkError):
    default_code = "DELIVERY_ERROR"
    default_category = NetworkErrorCategory.DELIVERY
    default_severity = NetworkErrorSeverity.ERROR


class DeliveryTimeoutError(DeliveryError):
    default_code = "DELIVERY_TIMEOUT"
    default_retryable = True
    default_transient = True
    default_retry_disposition = RetryDisposition.SAFE


class DeliveryExpiredError(DeliveryError):
    default_code = "DELIVERY_EXPIRED"
    default_retryable = False
    default_retry_disposition = RetryDisposition.NEVER


class DeliveryStateError(DeliveryError):
    default_code = "DELIVERY_STATE_INVALID"
    default_category = NetworkErrorCategory.LIFECYCLE
    default_retry_disposition = RetryDisposition.CONDITIONAL


class DeadLetterQueueError(DeliveryError):
    default_code = "DEAD_LETTER_ROUTING_FAILED"
    default_severity = NetworkErrorSeverity.CRITICAL
    default_alertable = True


class IdempotencyViolationError(DeliveryError):
    default_code = "IDEMPOTENCY_VIOLATION"
    default_retryable = False
    default_retry_disposition = RetryDisposition.NEVER


class DuplicateMessageError(DeliveryError):
    default_code = "DUPLICATE_MESSAGE"
    default_retryable = False
    default_retry_disposition = RetryDisposition.NEVER
    default_severity = NetworkErrorSeverity.WARNING


class PolicyViolationError(NetworkError):
    default_code = "POLICY_VIOLATION"
    default_category = NetworkErrorCategory.POLICY
    default_severity = NetworkErrorSeverity.WARNING
    default_retryable = False
    default_retry_disposition = RetryDisposition.NEVER


class DestinationDeniedError(PolicyViolationError):
    default_code = "DESTINATION_DENIED"


class ProtocolDeniedError(PolicyViolationError):
    default_code = "PROTOCOL_DENIED"


class PortDeniedError(PolicyViolationError):
    default_code = "PORT_DENIED"


class TLSRequiredError(PolicyViolationError):
    default_code = "TLS_REQUIRED"
    default_category = NetworkErrorCategory.SECURITY


class CertificateValidationError(PolicyViolationError):
    default_code = "CERTIFICATE_VALIDATION_FAILED"
    default_category = NetworkErrorCategory.SECURITY


class AuthenticationFailedError(NetworkError):
    default_code = "AUTHENTICATION_FAILED"
    default_category = NetworkErrorCategory.AUTHENTICATION
    default_severity = NetworkErrorSeverity.ERROR
    default_retryable = False
    default_retry_disposition = RetryDisposition.CONDITIONAL


class AuthorizationFailedError(NetworkError):
    default_code = "AUTHORIZATION_FAILED"
    default_category = NetworkErrorCategory.AUTHORIZATION
    default_severity = NetworkErrorSeverity.ERROR
    default_retryable = False
    default_retry_disposition = RetryDisposition.NEVER


class RateLimitExceededError(NetworkError):
    default_code = "RATE_LIMIT_EXCEEDED"
    default_category = NetworkErrorCategory.THROTTLING
    default_severity = NetworkErrorSeverity.WARNING
    default_retryable = True
    default_transient = True
    default_retry_disposition = RetryDisposition.SAFE


def is_network_error(exc: BaseException) -> bool:
    return isinstance(exc, NetworkError)


TRANSIENT_ERROR_TYPES: Tuple[type[BaseException], ...] = (
    TimeoutError,
    socket.timeout,
    socket.gaierror,
    ConnectionError,
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
)


RETRYABLE_NETWORK_TYPES: Tuple[type[NetworkError], ...] = (
    NetworkConnectionError,
    NetworkTransportError,
    RoutingError,
    DeliveryTimeoutError,
    RateLimitExceededError,
)


_HTTP_STATUS_TO_ERROR: Dict[int, type[NetworkError]] = {
    400: PayloadValidationError,
    401: AuthenticationFailedError,
    403: AuthorizationFailedError,
    404: EndpointUnavailableError,
    408: DeliveryTimeoutError,
    409: IdempotencyViolationError,
    413: PayloadTooLargeError,
    421: NoRouteAvailableError,
    425: RateLimitExceededError,
    426: ProtocolNegotiationError,
    429: RateLimitExceededError,
    495: CertificateValidationError,
    496: TLSRequiredError,
    497: ProtocolDeniedError,
    500: NetworkTransportError,
    502: EndpointUnavailableError,
    503: EndpointDegradedError,
    504: DeliveryTimeoutError,
}


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, NetworkError):
        return exc.retryable
    if isinstance(exc, (TimeoutError, socket.timeout, socket.gaierror)):
        return True
    if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
        return True
    if isinstance(exc, ssl.SSLEOFError):
        return True
    return False


def is_transient_exception(exc: BaseException) -> bool:
    if isinstance(exc, NetworkError):
        return exc.transient
    return isinstance(exc, TRANSIENT_ERROR_TYPES)


def network_error_from_http_status(
    status_code: int,
    *,
    message: Optional[str] = None,
    endpoint: Optional[str] = None,
    operation: Optional[str] = None,
    channel: Optional[str] = None,
    protocol: Optional[str] = None,
    retry_after_ms: Optional[int] = None,
    details: Optional[Mapping[str, Any]] = None,
    cause: Optional[BaseException] = None,
) -> NetworkError:
    """
    Translate HTTP status outcomes into SLAI-native network exceptions.

    This is intentionally conservative: codes that are commonly retryable are
    marked as such, while authentication/authorization/payload validation
    failures are treated as terminal until caller action changes the input.
    """

    error_cls = _HTTP_STATUS_TO_ERROR.get(status_code, NetworkTransportError if status_code >= 500 else NetworkError)
    reason_message = message or _default_http_message(status_code)
    return error_cls(
        reason_message,
        context={
            "status_code": status_code,
            "endpoint": endpoint,
            "operation": operation,
            "channel": channel,
            "protocol": protocol,
            "retry_after_ms": retry_after_ms,
        },
        details=details,
        cause=cause,
    )


def normalize_network_exception(
    exc: BaseException,
    *,
    message: Optional[str] = None,
    operation: Optional[str] = None,
    channel: Optional[str] = None,
    endpoint: Optional[str] = None,
    protocol: Optional[str] = None,
    route: Optional[str] = None,
    correlation_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    session_id: Optional[str] = None,
    status_code: Optional[int] = None,
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    timeout_ms: Optional[int] = None,
    retry_after_ms: Optional[int] = None,
    policy_name: Optional[str] = None,
    circuit_state: Optional[str] = None,
    payload_size: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> NetworkError:
    """
    Normalize arbitrary exceptions into the SLAI network taxonomy.

    Normalization rules prefer determinism over overfitting. The same raw
    exception type will always map to the same SLAI-native exception family
    unless the caller supplies a stronger signal such as an HTTP status code.
    """

    context = {
        "operation": operation,
        "channel": channel,
        "endpoint": endpoint,
        "protocol": protocol,
        "route": route,
        "correlation_id": correlation_id,
        "idempotency_key": idempotency_key,
        "session_id": session_id,
        "status_code": status_code,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "timeout_ms": timeout_ms,
        "retry_after_ms": retry_after_ms,
        "policy_name": policy_name,
        "circuit_state": circuit_state,
        "payload_size": payload_size,
        "metadata": dict(metadata or {}),
    }

    if isinstance(exc, NetworkError):
        return exc.with_context(**context)

    if status_code is not None:
        return network_error_from_http_status(
            status_code,
            message=message or str(exc),
            endpoint=endpoint,
            operation=operation,
            channel=channel,
            protocol=protocol,
            retry_after_ms=retry_after_ms,
            details={
                "normalized_from": type(exc).__name__,
                "raw_message": str(exc),
                "route": route,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "payload_size": payload_size,
                **dict(metadata or {}),
            },
            cause=exc,
        )

    details = {
        "normalized_from": type(exc).__name__,
        "raw_message": str(exc),
        "traceback": traceback.format_exception_only(type(exc), exc),
    }
    if metadata:
        details.update(dict(metadata))

    normalized_message = message or str(exc) or type(exc).__name__

    if isinstance(exc, json.JSONDecodeError):
        return PayloadDeserializationError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, (TypeError, UnicodeEncodeError)):
        return PayloadSerializationError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, UnicodeDecodeError):
        return PayloadDeserializationError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, ValueError) and operation in {"serialize", "deserialize", "encode", "decode"}:
        return PayloadValidationError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, socket.gaierror):
        return DNSResolutionError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, (TimeoutError, socket.timeout)):
        if operation in {"deliver", "await_ack", "ack", "receive"}:
            return DeliveryTimeoutError(normalized_message, context=context, details=details, cause=exc)
        return ConnectionTimeoutError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, ConnectionRefusedError):
        return ConnectionRejectedError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, ssl.CertificateError):
        return CertificateValidationError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, ssl.SSLError):
        return TLSHandshakeError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, BrokenPipeError):
        return SendFailureError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
        return NetworkTransportError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, PermissionError):
        return AuthorizationFailedError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, FileNotFoundError) and operation == "adapter_lookup":
        return AdapterNotFoundError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, NotImplementedError):
        return AdapterCapabilityError(normalized_message, context=context, details=details, cause=exc)
    if isinstance(exc, OSError):
        return NetworkTransportError(normalized_message, context=context, details=details, cause=exc)

    return NetworkError(
        normalized_message,
        code="UNEXPECTED_NETWORK_ERROR",
        category=NetworkErrorCategory.UNKNOWN,
        severity=NetworkErrorSeverity.ERROR,
        retryable=is_retryable_exception(exc),
        transient=is_transient_exception(exc),
        retry_disposition=RetryDisposition.CONDITIONAL if is_retryable_exception(exc) else RetryDisposition.NEVER,
        context=context,
        details=details,
        cause=exc,
    )


def build_error_snapshot(exc: BaseException, **context: Any) -> Dict[str, Any]:
    """Normalize then serialize an exception into a JSON-safe snapshot."""
    return normalize_network_exception(exc, **context).to_memory_snapshot()


def raise_normalized_network_error(exc: BaseException, **context: Any) -> None:
    """Raise a normalized SLAI-native network exception from any exception."""
    raise normalize_network_exception(exc, **context) from exc


F = TypeVar("F", bound=Callable[..., Any])


def network_error_boundary(
    *,
    operation: Optional[str] = None,
    channel: Optional[str] = None,
    endpoint: Optional[str] = None,
    protocol: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator that normalizes arbitrary exceptions at module boundaries.

    Example:
        @network_error_boundary(operation="send", protocol="http")
        def send(...):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except NetworkError:
                raise
            except Exception as exc:  # noqa: BLE001 - boundary normalization is intentional.
                raise normalize_network_exception(
                    exc,
                    operation=operation or func.__name__,
                    channel=channel,
                    endpoint=endpoint,
                    protocol=protocol,
                ) from exc

        return wrapped  # type: ignore[return-value]

    return decorator


def _default_http_message(status_code: int) -> str:
    if status_code == 400:
        return "HTTP request payload failed validation."
    if status_code == 401:
        return "Authentication failed for outbound network request."
    if status_code == 403:
        return "Authorization failed for outbound network request."
    if status_code == 404:
        return "Target endpoint could not be found."
    if status_code == 408:
        return "Network delivery timed out while awaiting response."
    if status_code == 409:
        return "Idempotency or request state conflict detected."
    if status_code == 413:
        return "Payload exceeds remote or local transport limits."
    if status_code == 421:
        return "No viable route or misdirected request detected."
    if status_code in {425, 429}:
        return "Remote endpoint rate-limited the request."
    if status_code == 426:
        return "Remote endpoint requires a different protocol or upgrade."
    if status_code in {495, 496, 497}:
        return "TLS or certificate policy validation failed."
    if status_code == 500:
        return "Remote endpoint encountered an internal transport error."
    if status_code == 502:
        return "Upstream endpoint is unavailable or returned an invalid gateway response."
    if status_code == 503:
        return "Target endpoint is degraded or temporarily unavailable."
    if status_code == 504:
        return "Upstream request timed out."
    if 500 <= status_code <= 599:
        return f"HTTP server-side failure ({status_code})."
    if 400 <= status_code <= 499:
        return f"HTTP client-side failure ({status_code})."
    return f"Unexpected HTTP status code encountered: {status_code}."


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_exception(exc: BaseException, *, include_traceback: bool = False) -> Dict[str, Any]:
    if isinstance(exc, NetworkError):
        return exc.to_dict(include_cause=True, include_traceback=include_traceback)
    payload: Dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if include_traceback:
        payload["traceback"] = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return payload


def _json_safe(value: Any, *, max_depth: int = 5, _depth: int = 0) -> Any:
    if _depth >= max_depth:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v, max_depth=max_depth, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, max_depth=max_depth, _depth=_depth + 1) for v in value]
    if isinstance(value, BaseException):
        return _serialize_exception(value, include_traceback=False)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
