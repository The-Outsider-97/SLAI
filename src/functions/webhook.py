"""
Secure webhook delivery for SLAI's reusable application function layer.

This module provides a framework-agnostic outbound webhook service with:

- deterministic JSON event envelopes;
- HMAC-SHA256 request signing;
- signature verification helpers for compatible receivers;
- stable event identifiers for receiver-side idempotency;
- bounded payload and response sizes;
- retry/backoff for transient network and HTTP failures;
- HTTP Retry-After support;
- HTTPS-by-default endpoint validation;
- redirect suppression;
- SSRF-aware destination checks;
- injectable backend abstractions for deterministic testing.

Security boundary
-----------------
Webhook endpoints should be application-configured destinations rather than
arbitrary per-request URLs supplied by untrusted callers.

The destination validation implemented here is defense-in-depth. Deployment-
level egress controls and explicit destination allowlists remain appropriate
for security-sensitive or multi-tenant deployments.

The module intentionally does not duplicate:

- ratelimiter.py:
    caller/API admission control;

- transport.py:
    LoRa, serial, mesh, LTE, and SATCOM channel transport;

- email.py:
    SMTP delivery;

- phone_verification.py:
    SMS OTP delivery and verification;

- auth.py:
    identity, sessions, credentials, and authorization.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import random
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .utils.config_loader import get_config_section
from .utils.functions_error import *
from .utils.functions_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Webhook Service")
printer = PrettyPrinter()

# ============================================================================
# Constants
# ============================================================================

_DEFAULT_USER_AGENT = "SLAI-Webhook/2.3"
_DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024
_DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024
_DEFAULT_SIGNATURE_TOLERANCE_SECONDS = 300
_DEFAULT_RETRY_STATUS_CODES: Tuple[int, ...] = (
    408,  # Request Timeout
    425,  # Too Early
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
)

_EVENT_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_SIGNATURE_PATTERN = re.compile(r"^v1=([0-9a-fA-F]{64})$")
_RESERVED_HEADERS = frozenset({
    "content-length",
    "content-type",
    "host",
    "idempotency-key",
    "user-agent",
    "x-slai-webhook-event",
    "x-slai-webhook-id",
    "x-slai-webhook-signature",
    "x-slai-webhook-timestamp",
})


# ============================================================================
# Configuration normalization
# ============================================================================


def _normalize_secret(secret: Any) -> bytes:
    """
    Normalize webhook secret material.

    At least 32 bytes are required to provide an appropriate HMAC key size.
    """
    if isinstance(secret, str):
        normalized = secret.encode("utf-8")

    elif isinstance(
            secret,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        normalized = bytes(secret)

    else:
        raise WebhookConfigurationError("webhook secret must be str or bytes-like")

    if len(normalized) < 32:
        raise WebhookConfigurationError("webhook secret must contain at least 32 bytes of key material")

    return normalized


def _coerce_bool(
    value: Any,
    field_name: str,
) -> bool:
    """Safely normalize configuration booleans."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
                "1",
                "true",
                "yes",
                "on",
        }:
            return True

        if normalized in {
                "0",
                "false",
                "no",
                "off",
        }:
            return False

    if (isinstance(value, int) and not isinstance(value, bool)
            and value in (0, 1)):
        return bool(value)

    raise WebhookConfigurationError(f"{field_name} must be a boolean")


def _require_non_negative_int(
    value: Any,
    field_name: str,
) -> int:
    """Validate an integer >= 0 without silently truncating floats."""
    if isinstance(value, bool):
        raise WebhookConfigurationError(f"{field_name} must be an integer")

    if isinstance(value, int):
        result = value

    elif isinstance(value, str):
        candidate = value.strip()

        if not candidate.isdigit():
            raise WebhookConfigurationError(f"{field_name} must be an integer")

        result = int(candidate)

    else:
        raise WebhookConfigurationError(f"{field_name} must be an integer")

    if result < 0:
        raise WebhookConfigurationError(f"{field_name} must be >= 0")

    return result


def _require_positive_int(value: Any, field_name: str) -> int:
    """Validate an integer > 0."""
    result = _require_non_negative_int(value, field_name)

    if result <= 0:
        raise WebhookConfigurationError(f"{field_name} must be > 0")

    return result


def _require_non_negative_float(value: Any, field_name: str) -> float:
    """Validate a finite numeric value >= 0."""
    if isinstance(value, bool):
        raise WebhookConfigurationError(f"{field_name} must be numeric")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WebhookConfigurationError(
            f"{field_name} must be numeric") from exc

    if result < 0:
        raise WebhookConfigurationError(f"{field_name} must be >= 0")

    if result == float("inf") or result != result:
        raise WebhookConfigurationError(f"{field_name} must be finite")

    return result


def _require_positive_float(value: Any, field_name: str) -> float:
    """Validate a finite numeric value > 0."""
    result = _require_non_negative_float(value, field_name)

    if result <= 0:
        raise WebhookConfigurationError(f"{field_name} must be > 0")

    return result


# ============================================================================
# Header validation
# ============================================================================


def _validate_header_pair(name: str, value: str) -> Tuple[str, str]:
    """Validate a caller-provided HTTP header."""
    if (not isinstance(name, str)
            or _HEADER_NAME_PATTERN.fullmatch(name) is None):
        raise WebhookConfigurationError(f"invalid HTTP header name: {name!r}")

    if not isinstance(value, str):
        raise WebhookConfigurationError(
            f"HTTP header {name!r} value must be a string")

    if "\r" in value or "\n" in value:
        raise WebhookConfigurationError(
            f"HTTP header {name!r} contains a prohibited newline")

    if name.lower() in _RESERVED_HEADERS:
        raise WebhookConfigurationError(
            f"HTTP header {name!r} is managed by WebhookService")

    return name, value


def _validate_headers(headers: Optional[Mapping[str, str]], ) -> Dict[str, str]:
    """Validate a header mapping."""
    if headers is None:
        return {}

    if not isinstance(headers, Mapping):
        raise WebhookConfigurationError("headers must be a mapping")

    validated: Dict[str, str] = {}

    for name, value in headers.items():
        header_name, header_value = _validate_header_pair(
            name,
            value,
        )

        validated[header_name] = header_value

    return validated


def _merge_headers(base: Mapping[str, str], override: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """
    Merge headers case-insensitively.

    Per-send values replace default values without producing duplicate
    differently-cased header names.
    """
    result = dict(base)

    additions = _validate_headers(override)

    for name, value in additions.items():
        existing = next(
            (current for current in result if current.lower() == name.lower()),
            None,
        )

        if existing is not None:
            result.pop(existing, None)

        result[name] = value

    return result


# ============================================================================
# URL and destination validation
# ============================================================================


def _redact_url(url: str) -> str:
    """
    Return a log-safe endpoint representation.

    Query parameters, fragments, and embedded credentials are never logged.
    """
    try:
        parsed = urlsplit(url)
    except Exception:
        return "<invalid-webhook-url>"

    hostname = parsed.hostname or ""

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    try:
        port = parsed.port
    except ValueError:
        port = None

    netloc = (f"{hostname}:{port}" if port is not None else hostname)

    return urlunsplit((
        parsed.scheme,
        netloc,
        parsed.path or "/",
        "",
        "",
    ))


def _validate_endpoint_syntax(url: str, *, allow_http: bool) -> SplitResult:
    """Validate endpoint structure without performing network access."""
    if not isinstance(url, str) or not url.strip():
        raise WebhookConfigurationError(
            "webhook endpoint must be a non-empty string")

    candidate = url.strip()
    parsed = urlsplit(candidate)

    allowed_schemes = {"https"}

    if allow_http:
        allowed_schemes.add("http")

    if parsed.scheme.lower() not in allowed_schemes:
        required = ("HTTPS" if not allow_http else "HTTP or HTTPS")

        raise WebhookSecurityError(
            f"webhook endpoint must use {required}",
            endpoint=_redact_url(candidate),
        )

    if not parsed.hostname:
        raise WebhookConfigurationError(
            "webhook endpoint must include a hostname")

    if (parsed.username is not None or parsed.password is not None):
        raise WebhookSecurityError(
            "credentials embedded in webhook URLs are not permitted",
            endpoint=_redact_url(candidate),
        )

    if parsed.fragment:
        raise WebhookConfigurationError(
            "webhook endpoint must not contain a URL fragment")

    try:
        port = parsed.port
    except ValueError as exc:
        raise WebhookConfigurationError(
            "webhook endpoint contains an invalid port") from exc

    if (port is not None and not 1 <= port <= 65535):
        raise WebhookConfigurationError(
            "webhook endpoint port must be between 1 and 65535")

    return parsed


def _is_public_ip(address: str) -> bool:
    """Return whether an address is globally routable."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False

    return parsed.is_global


def _assert_safe_destination(
    parsed: SplitResult,
    *,
    allow_private_hosts: bool,
) -> None:
    """
    Resolve and reject non-public destinations unless explicitly enabled.

    This prevents direct use of loopback, link-local, private, reserved,
    multicast, and other non-global addresses in the default configuration.
    """
    if allow_private_hosts:
        return

    hostname = parsed.hostname

    if hostname is None:
        raise WebhookSecurityError("webhook endpoint contains no hostname")

    normalized_host = hostname.lower()

    if normalized_host in {
            "localhost",
            "localhost.localdomain",
    }:
        raise WebhookSecurityError(
            "local webhook destinations are blocked",
            endpoint=_redact_url(parsed.geturl()),
        )

    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        if not literal_ip.is_global:
            raise WebhookSecurityError(
                "non-public webhook destination is blocked",
                endpoint=_redact_url(parsed.geturl()),
            )

        return

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)

    try:
        resolved = socket.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )

    except socket.gaierror as exc:
        raise WebhookDeliveryError(
            endpoint=_redact_url(parsed.geturl()),
            reason=f"DNS resolution failed: {exc}",
            retryable=True,
        ) from exc

    addresses = {
        address
        for entry in resolved
        if entry and entry[4] and isinstance((address := entry[4][0]), str)
    }

    if not addresses:
        raise WebhookDeliveryError(
            endpoint=_redact_url(parsed.geturl()),
            reason="DNS resolution returned no usable addresses",
            retryable=True,
        )

    blocked = sorted(address for address in addresses
                     if not _is_public_ip(address))

    if blocked:
        raise WebhookSecurityError(
            "webhook destination resolved to a non-public address",
            endpoint=_redact_url(parsed.geturl()),
            details={
                "blocked_addresses": blocked,
            },
        )


# ============================================================================
# JSON serialization
# ============================================================================


def _canonical_json_bytes(value: Mapping[str, Any], ) -> bytes:
    """
    Serialize JSON deterministically.

    Stable serialization matters because the exact bytes are authenticated
    by the HMAC signature.
    """
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        return serialized.encode("utf-8")

    except (
            TypeError,
            ValueError,
            OverflowError,
            UnicodeEncodeError,
            RecursionError,
    ) as exc:
        raise WebhookSerializationError(str(exc)) from exc


# ============================================================================
# HTTP Retry-After
# ============================================================================


def _header_value(
    headers: Mapping[str, str],
    name: str,
) -> Optional[str]:
    """Case-insensitive response-header lookup."""
    target = name.lower()

    for header_name, value in headers.items():
        if header_name.lower() == target:
            return value

    return None


def _parse_retry_after(
    headers: Mapping[str, str],
    *,
    now: Optional[datetime] = None,
) -> Optional[float]:
    """
    Parse Retry-After as delay-seconds or HTTP-date.

    Invalid values are ignored rather than treated as fatal delivery errors.
    """
    raw = _header_value(
        headers,
        "Retry-After",
    )

    if raw is None:
        return None

    candidate = raw.strip()

    if not candidate:
        return None

    if candidate.isdigit():
        return float(candidate)

    try:
        parsed = parsedate_to_datetime(candidate)

    except (
            TypeError,
            ValueError,
            OverflowError,
    ):
        return None

    if parsed is None:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    reference = now or _utc_now()

    return max(
        (parsed.astimezone(timezone.utc) - reference).total_seconds(),
        0.0,
    )


# ============================================================================
# Signature helpers
# ============================================================================


def sign_webhook_payload(
    secret: Any,
    *,
    body: bytes,
    timestamp: int,
    event_id: str,
) -> str:
    """
    Produce an SLAI webhook v1 HMAC-SHA256 signature.

    Signed byte sequence:

        <timestamp>.<event_id>.<raw_json_body>
    """
    key = _normalize_secret(secret)

    if not isinstance(
            body,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        raise TypeError("body must be bytes-like")

    if (isinstance(timestamp, bool) or not isinstance(timestamp, int)
            or timestamp < 0):
        raise ValueError("timestamp must be a non-negative integer")

    if (not isinstance(event_id, str) or not event_id.strip()):
        raise ValueError("event_id must be a non-empty string")

    signed_payload = (f"{timestamp}.{event_id}.".encode("utf-8") + bytes(body))

    digest = hmac.new(
        key,
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    return f"v1={digest}"


def verify_webhook_signature(
    secret: Any,
    *,
    body: bytes,
    timestamp: int,
    event_id: str,
    signature: str,
    tolerance_seconds: int = _DEFAULT_SIGNATURE_TOLERANCE_SECONDS,
    now_epoch_seconds: Optional[int] = None,
) -> bool:
    """
    Verify an SLAI webhook v1 signature.

    The timestamp check restricts the replay window. A webhook receiver that
    requires strict once-only event handling must additionally persist and
    reject already-processed event IDs.
    """
    if (isinstance(tolerance_seconds, bool)
            or not isinstance(tolerance_seconds, int)
            or tolerance_seconds < 0):
        raise ValueError("tolerance_seconds must be an integer >= 0")

    if not isinstance(signature, str):
        raise WebhookSignatureError("signature must be a string")

    normalized_signature = signature.strip()

    if (_SIGNATURE_PATTERN.fullmatch(normalized_signature) is None):
        raise WebhookSignatureError("malformed webhook signature")

    if (isinstance(timestamp, bool) or not isinstance(timestamp, int)
            or timestamp < 0):
        raise WebhookSignatureError("invalid webhook timestamp")

    current = (int(time.time())
               if now_epoch_seconds is None else int(now_epoch_seconds))

    if (abs(current - timestamp) > tolerance_seconds):
        raise WebhookSignatureError(
            "webhook signature timestamp is outside tolerance")

    expected = sign_webhook_payload(
        secret,
        body=body,
        timestamp=timestamp,
        event_id=event_id,
    )

    if not hmac.compare_digest(
            expected,
            normalized_signature,
    ):
        raise WebhookSignatureError("webhook signature mismatch")

    return True


# ============================================================================
# Event model
# ============================================================================


@dataclass(frozen=True)
class WebhookEvent:
    """
    Application event sent through a webhook.

    event_id
        Stable identifier used for idempotency.

    event_type
        Machine-readable event category, e.g. ``user.phone_verified``.

    payload
        JSON-compatible event data.

    created_at
        Time at which the event was created.
    """

    event_id: str
    event_type: str
    payload: Mapping[str, Any]
    created_at: datetime

    def __post_init__(self) -> None:
        if (not isinstance(self.event_id, str) or not self.event_id.strip()):
            raise WebhookConfigurationError(
                "event_id must be a non-empty string")

        if (not isinstance(self.event_type, str)
                or _EVENT_TYPE_PATTERN.fullmatch(self.event_type) is None):
            raise WebhookConfigurationError(
                "event_type must match "
                "[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")

        if not isinstance(
                self.payload,
                Mapping,
        ):
            raise WebhookConfigurationError("payload must be a mapping")

        _isoformat_utc(self.created_at)

    @classmethod
    def create(
        cls,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        event_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> "WebhookEvent":
        """Create a new webhook event."""
        return cls(
            event_id=(event_id or str(uuid.uuid4())),
            event_type=event_type,
            payload=dict(payload),
            created_at=(created_at or _utc_now()),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the canonical event envelope."""
        return {
            "id": self.event_id,
            "type": self.event_type,
            "created_at": _isoformat_utc(self.created_at),
            "data": dict(self.payload),
        }


# ============================================================================
# HTTP response model
# ============================================================================


@dataclass(frozen=True)
class WebhookHTTPResponse:
    """Bounded response returned by a WebhookBackend."""

    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    body_truncated: bool = False

    def __post_init__(self) -> None:
        if (isinstance(self.status_code, bool)
                or not isinstance(self.status_code, int)
                or not 100 <= self.status_code <= 599):
            raise ValueError("status_code must be a valid HTTP status code")

        if not isinstance(
                self.headers,
                Mapping,
        ):
            raise TypeError("headers must be a mapping")

        if not isinstance(
                self.body,
                bytes,
        ):
            raise TypeError("body must be bytes")


# ============================================================================
# Delivery result
# ============================================================================


@dataclass(frozen=True)
class WebhookDeliveryResult:
    """Metadata describing a successful webhook delivery."""

    event_id: str
    event_type: str
    endpoint: str
    status_code: int
    attempts: int
    elapsed_seconds: float
    delivered_at: datetime
    response_headers: Mapping[str, str] = field(default_factory=dict)
    response_body: bytes = b""
    response_body_truncated: bool = False


# ============================================================================
# Retry policy
# ============================================================================


@dataclass(frozen=True)
class WebhookRetryPolicy:
    """
    Bounded retry policy for transient webhook failures.

    ``max_retries`` excludes the initial request. Therefore max_retries=3
    permits at most four total attempts.
    """

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_status_codes: Tuple[int, ...] = (_DEFAULT_RETRY_STATUS_CODES)

    def __post_init__(self) -> None:
        max_retries = _require_non_negative_int(self.max_retries, "max_retries")
        base_delay = _require_non_negative_float(self.base_delay, "base_delay")
        max_delay = _require_non_negative_float(self.max_delay, "max_delay")

        if max_delay < base_delay:
            raise WebhookConfigurationError("max_delay must be >= base_delay")

        backoff_factor = _require_positive_float(self.backoff_factor, "backoff_factor")
        if backoff_factor < 1.0:
            raise WebhookConfigurationError("backoff_factor must be >= 1")

        if not isinstance(self.jitter, bool):
            raise WebhookConfigurationError("jitter must be a boolean")

        normalized_statuses = []

        for status in self.retry_status_codes:
            if (isinstance(status, bool) or not isinstance(status, int)
                    or not 100 <= status <= 599):
                raise WebhookConfigurationError(
                    "retry_status_codes must contain valid HTTP status codes")

            normalized_statuses.append(status)

        object.__setattr__(self, "max_retries", max_retries)
        object.__setattr__(self, "base_delay", base_delay)
        object.__setattr__(self, "max_delay", max_delay,)
        object.__setattr__(self, "backoff_factor", backoff_factor)
        object.__setattr__(self, "retry_status_codes", tuple(dict.fromkeys(normalized_statuses)))

    def delay_seconds(
        self,
        retry_index: int,
        *,
        retry_after_seconds: Optional[float] = None,
    ) -> float:
        """
        Compute the delay preceding one retry.

        retry_index=0 corresponds to the first retry after the initial request.
        """
        if (isinstance(retry_index, bool) or not isinstance(retry_index, int)
                or retry_index < 0):
            raise ValueError("retry_index must be an integer >= 0")

        backoff = min(
            self.max_delay,
            self.base_delay * (self.backoff_factor**retry_index),
        )

        if retry_after_seconds is not None:
            requested = max(
                float(retry_after_seconds),
                0.0,
            )

            # Keep caller-controlled upper bounds deterministic.
            return min(
                self.max_delay,
                max(
                    backoff,
                    requested,
                ),
            )

        if (self.jitter and backoff > 0):
            return random.uniform(
                0.0,
                backoff,
            )

        return backoff


# ============================================================================
# Backend abstraction
# ============================================================================


class WebhookBackend(ABC):
    """HTTP backend contract for webhook delivery."""

    @abstractmethod
    def send(
        self,
        *,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> WebhookHTTPResponse:
        """Perform one HTTP POST attempt."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
        raise NotImplementedError


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """
    Disable HTTP redirects.

    A validated destination must not be allowed to redirect the sender toward
    a different, potentially internal, address.
    """

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        return None


class UrllibWebhookBackend(WebhookBackend):
    """
    Standard-library webhook backend.

    HTTP redirects are disabled. HTTP error responses are returned as
    WebhookHTTPResponse objects so WebhookService can apply retry policy.
    """

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    @staticmethod
    def _read_bounded(
        stream: Any,
        max_bytes: int,
    ) -> Tuple[bytes, bool]:
        """Read at most max_bytes while detecting truncation."""
        data = stream.read(max_bytes + 1)

        if len(data) > max_bytes:
            return (
                data[:max_bytes],
                True,
            )

        return (
            data,
            False,
        )

    def send(
        self,
        *,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> WebhookHTTPResponse:
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=dict(headers),
            method="POST",
        )

        try:
            with self._opener.open(
                    request,
                    timeout=timeout_seconds,
            ) as response:
                response_body, truncated = self._read_bounded(
                    response,
                    max_response_bytes,
                )

                return WebhookHTTPResponse(
                    status_code=int(response.status),
                    headers=dict(response.headers.items()),
                    body=response_body,
                    body_truncated=truncated,
                )

        except urllib.error.HTTPError as exc:
            response_body, truncated = self._read_bounded(
                exc,
                max_response_bytes,
            )

            return WebhookHTTPResponse(
                status_code=int(exc.code),
                headers=(dict(exc.headers.items()) if exc.headers else {}),
                body=response_body,
                body_truncated=truncated,
            )

        except urllib.error.URLError as exc:
            reason = getattr(
                exc,
                "reason",
                exc,
            )

            retryable = not isinstance(
                reason,
                ssl.SSLCertVerificationError,
            )

            raise WebhookDeliveryError(
                endpoint=_redact_url(url),
                reason=str(reason),
                retryable=retryable,
            ) from exc

        except ssl.SSLCertVerificationError as exc:
            raise WebhookDeliveryError(
                endpoint=_redact_url(url),
                reason=str(exc),
                retryable=False,
            ) from exc

        except (
                OSError,
                TimeoutError,
        ) as exc:
            raise WebhookDeliveryError(
                endpoint=_redact_url(url),
                reason=str(exc),
                retryable=True,
            ) from exc

    def close(self) -> None:
        """urllib backend is stateless."""
        pass


# ============================================================================
# Webhook service
# ============================================================================


class WebhookService:
    """
    Secure outbound JSON webhook service bound to one trusted endpoint.

    Binding each service instance to one endpoint is intentional. It keeps
    webhook delivery distinct from a generic arbitrary-URL HTTP client and
    substantially narrows the SSRF attack surface.

    Create separate service instances when an application requires multiple
    webhook destinations.
    """

    def __init__(
        self,
        endpoint: str,
        secret: Any,
        *,
        backend: Optional[WebhookBackend] = None,
        timeout_seconds: float = 10.0,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        allow_http: bool = False,
        allow_private_hosts: bool = False,
        default_headers: Optional[Mapping[str, str]] = None,
        retry_policy: Optional[WebhookRetryPolicy] = None,
        user_agent: str = _DEFAULT_USER_AGENT,
    ) -> None:
        if not isinstance(
                allow_http,
                bool,
        ):
            raise WebhookConfigurationError("allow_http must be a boolean")

        if not isinstance(
                allow_private_hosts,
                bool,
        ):
            raise WebhookConfigurationError(
                "allow_private_hosts must be a boolean")

        parsed_endpoint = _validate_endpoint_syntax(
            endpoint,
            allow_http=allow_http,
        )

        if not isinstance(
                user_agent,
                str,
        ) or not user_agent.strip():
            raise WebhookConfigurationError(
                "user_agent must be a non-empty string")

        if ("\r" in user_agent or "\n" in user_agent):
            raise WebhookConfigurationError(
                "user_agent contains a prohibited newline")

        self.endpoint = endpoint.strip()

        self._parsed_endpoint = (parsed_endpoint)

        self._secret = _normalize_secret(secret)

        self.backend = (backend or UrllibWebhookBackend())

        if not isinstance(
                self.backend,
                WebhookBackend,
        ):
            raise WebhookConfigurationError(
                "backend must implement WebhookBackend")

        self.timeout_seconds = _require_positive_float(
            timeout_seconds,
            "timeout_seconds",
        )

        self.max_payload_bytes = _require_positive_int(
            max_payload_bytes,
            "max_payload_bytes",
        )

        self.max_response_bytes = _require_non_negative_int(
            max_response_bytes,
            "max_response_bytes",
        )

        self.allow_http = allow_http

        self.allow_private_hosts = (allow_private_hosts)

        self.default_headers = _validate_headers(default_headers)

        self.retry_policy = (retry_policy or WebhookRetryPolicy())

        if not isinstance(
                self.retry_policy,
                WebhookRetryPolicy,
        ):
            raise WebhookConfigurationError(
                "retry_policy must be WebhookRetryPolicy")

        self.user_agent = (user_agent.strip())

        self._closed = False

        self._state_lock = (threading.RLock())

        logger.info(
            "Webhook service initialized for %s",
            self.endpoint_summary,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def endpoint_summary(self) -> str:
        """Return the endpoint without query parameters or credentials."""
        return _redact_url(self.endpoint)

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise WebhookConfigurationError("WebhookService is closed")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _serialize_event(
        self,
        event: WebhookEvent,
    ) -> bytes:
        body = _canonical_json_bytes(event.as_dict())

        if (len(body) > self.max_payload_bytes):
            raise WebhookPayloadTooLargeError(
                actual_bytes=len(body),
                max_bytes=self.max_payload_bytes,
            )

        return body

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------

    def _build_headers(
        self,
        event: WebhookEvent,
        body: bytes,
        *,
        timestamp: int,
        extra_headers: Optional[Mapping[str, str]],
    ) -> Dict[str, str]:
        headers = _merge_headers(
            self.default_headers,
            extra_headers,
        )

        headers.update({
            "Content-Type":
            "application/json; charset=utf-8",
            "User-Agent":
            self.user_agent,

            # Receiver-side idempotency.
            "Idempotency-Key":
            event.event_id,

            # SLAI webhook metadata.
            "X-SLAI-Webhook-Id":
            event.event_id,
            "X-SLAI-Webhook-Event":
            event.event_type,
            "X-SLAI-Webhook-Timestamp":
            str(timestamp),
            "X-SLAI-Webhook-Signature":
            sign_webhook_payload(
                self._secret,
                body=body,
                timestamp=timestamp,
                event_id=event.event_id,
            ),
        })

        return headers

    # ------------------------------------------------------------------
    # Public delivery API
    # ------------------------------------------------------------------

    def deliver(
        self,
        event: WebhookEvent,
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> WebhookDeliveryResult:
        """
        Deliver an existing WebhookEvent.

        Successful delivery is defined as any HTTP 2xx response.

        Retryable categories:
        - transient network errors;
        - configured retryable HTTP statuses.

        Permanent categories:
        - invalid destination/security policy;
        - TLS certificate verification failure;
        - non-retryable HTTP status;
        - malformed payload/configuration.
        """
        self._ensure_open()

        if not isinstance(
                event,
                WebhookEvent,
        ):
            raise TypeError("event must be WebhookEvent")

        body = self._serialize_event(event)

        started = time.monotonic()

        total_attempts = (self.retry_policy.max_retries + 1)

        for attempt_index in range(total_attempts):
            attempt_number = (attempt_index + 1)

            try:
                # Re-resolve immediately before each request so a destination
                # that changes into a non-public address is rejected.
                _assert_safe_destination(
                    self._parsed_endpoint,
                    allow_private_hosts=self.allow_private_hosts,
                )

                timestamp = int(time.time())

                request_headers = self._build_headers(
                    event,
                    body,
                    timestamp=timestamp,
                    extra_headers=headers,
                )

                response = self.backend.send(
                    url=self.endpoint,
                    body=body,
                    headers=request_headers,
                    timeout_seconds=self.timeout_seconds,
                    max_response_bytes=self.max_response_bytes,
                )

            except WebhookSecurityError:
                # Security policy violations must never be retried.
                raise

            except WebhookDeliveryError as exc:
                if not exc.retryable:
                    raise

                if (attempt_index >= self.retry_policy.max_retries):
                    raise WebhookRetryExhausted(
                        endpoint=self.endpoint_summary,
                        event_id=event.event_id,
                        attempts=attempt_number,
                        last_error=str(exc),
                        status_code=exc.status_code,
                    ) from exc

                delay = self.retry_policy.delay_seconds(attempt_index)

                logger.warning(
                    "Webhook delivery failed for event %s to %s "
                    "(attempt %s/%s); retrying in %.2fs",
                    event.event_id,
                    self.endpoint_summary,
                    attempt_number,
                    total_attempts,
                    delay,
                )

                if delay > 0:
                    time.sleep(delay)

                continue

            # ----------------------------------------------------------
            # Successful response
            # ----------------------------------------------------------
            if (200 <= response.status_code < 300):
                elapsed = (time.monotonic() - started)

                logger.info(
                    "Webhook event %s delivered to %s "
                    "with HTTP %s in %s attempt(s)",
                    event.event_id,
                    self.endpoint_summary,
                    response.status_code,
                    attempt_number,
                )

                return WebhookDeliveryResult(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    endpoint=self.endpoint_summary,
                    status_code=response.status_code,
                    attempts=attempt_number,
                    elapsed_seconds=elapsed,
                    delivered_at=_utc_now(),
                    response_headers=dict(response.headers),
                    response_body=response.body,
                    response_body_truncated=(response.body_truncated),
                )

            # ----------------------------------------------------------
            # Failed HTTP response
            # ----------------------------------------------------------
            retryable = (response.status_code
                         in self.retry_policy.retry_status_codes)

            delivery_error = WebhookDeliveryError(
                endpoint=self.endpoint_summary,
                reason=(f"HTTP {response.status_code}"),
                status_code=response.status_code,
                retryable=retryable,
                details={
                    "event_id": event.event_id,
                    "response_bytes": len(response.body),
                    "response_body_truncated": (response.body_truncated),
                },
            )

            if not retryable:
                raise delivery_error

            if (attempt_index >= self.retry_policy.max_retries):
                raise WebhookRetryExhausted(
                    endpoint=self.endpoint_summary,
                    event_id=event.event_id,
                    attempts=attempt_number,
                    last_error=str(delivery_error),
                    status_code=response.status_code,
                ) from delivery_error

            retry_after = _parse_retry_after(response.headers)

            delay = self.retry_policy.delay_seconds(
                attempt_index,
                retry_after_seconds=retry_after,
            )

            logger.warning(
                "Webhook event %s received retryable HTTP %s from %s "
                "(attempt %s/%s); retrying in %.2fs",
                event.event_id,
                response.status_code,
                self.endpoint_summary,
                attempt_number,
                total_attempts,
                delay,
            )

            if delay > 0:
                time.sleep(delay)

        # Every loop path returns or raises.
        raise RuntimeError("unreachable webhook delivery state")

    def send(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        event_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> WebhookDeliveryResult:
        """
        Construct and deliver an application event.

        Example:
            service.send(
                "user.phone_verified",
                {"user_id": "123"},
            )
        """
        event = WebhookEvent.create(
            event_type,
            payload,
            event_id=event_id,
            created_at=created_at,
        )

        return self.deliver(
            event,
            headers=headers,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(
        self,
        *,
        resolve_destination: bool = True,
    ) -> Dict[str, Any]:
        """
        Return non-secret configuration and destination-health information.

        This does not transmit a webhook.
        """
        self._ensure_open()

        destination_safe: Optional[bool] = None
        destination_error: Optional[str] = None

        if resolve_destination:
            try:
                _assert_safe_destination(
                    self._parsed_endpoint,
                    allow_private_hosts=self.allow_private_hosts,
                )

                destination_safe = True

            except (
                    WebhookSecurityError,
                    WebhookDeliveryError,
            ) as exc:
                destination_safe = False
                destination_error = str(exc)

        return {
            "endpoint": self.endpoint_summary,
            "scheme": (self._parsed_endpoint.scheme.lower()),
            "destination_safe": destination_safe,
            "destination_error": destination_error,
            "max_payload_bytes": self.max_payload_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_retries": self.retry_policy.max_retries,
            "allow_http": self.allow_http,
            "allow_private_hosts": self.allow_private_hosts,
            "closed": self.closed,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Idempotently release backend resources."""
        with self._state_lock:
            if self._closed:
                return

            self._closed = True

        self.backend.close()

        logger.info(
            "Webhook service closed for %s",
            self.endpoint_summary,
        )

    def __enter__(self, ) -> "WebhookService":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        endpoint: Optional[str] = None,
        secret: Optional[Any] = None,
        backend: Optional[WebhookBackend] = None,
    ) -> "WebhookService":
        """
        Build a service from ``functions_config.yaml``.

        Secrets are resolved from an environment variable by default rather
        than stored directly in YAML.
        """
        config = get_config_section("webhook")

        resolved_endpoint = (endpoint or config.get("endpoint"))

        if not resolved_endpoint:
            raise WebhookConfigurationError(
                "webhook.endpoint must be configured or passed explicitly")

        resolved_secret = secret

        if resolved_secret is None:
            secret_env = str(config.get(
                "secret_env",
                "SLAI_WEBHOOK_SECRET",
            )).strip()

            if not secret_env:
                raise WebhookConfigurationError(
                    "webhook.secret_env must be a non-empty "
                    "environment variable name")

            resolved_secret = os.getenv(secret_env)

        if resolved_secret is None:
            raise WebhookConfigurationError(
                "webhook secret is not configured; "
                "set the configured secret_env environment variable "
                "or pass secret explicitly")

        raw_headers = config.get(
            "headers",
            {},
        )

        if raw_headers is None:
            raw_headers = {}

        raw_statuses = config.get(
            "retry_status_codes",
            list(_DEFAULT_RETRY_STATUS_CODES),
        )

        if (not isinstance(
                raw_statuses,
                Sequence,
        ) or isinstance(
                raw_statuses,
            (
                str,
                bytes,
                bytearray,
            ),
        )):
            raise WebhookConfigurationError(
                "webhook.retry_status_codes must be a sequence "
                "of HTTP status codes")

        retry_policy = WebhookRetryPolicy(
            max_retries=_require_non_negative_int(
                config.get(
                    "retry_max",
                    3,
                ),
                "retry_max",
            ),
            base_delay=_require_non_negative_float(
                config.get(
                    "retry_base_delay",
                    0.5,
                ),
                "retry_base_delay",
            ),
            max_delay=_require_non_negative_float(
                config.get(
                    "retry_max_delay",
                    30.0,
                ),
                "retry_max_delay",
            ),
            backoff_factor=_require_positive_float(
                config.get(
                    "retry_backoff_factor",
                    2.0,
                ),
                "retry_backoff_factor",
            ),
            jitter=_coerce_bool(
                config.get(
                    "retry_jitter",
                    True,
                ),
                "retry_jitter",
            ),
            retry_status_codes=tuple(int(status) for status in raw_statuses),
        )

        return cls(
            str(resolved_endpoint),
            resolved_secret,
            backend=backend,
            timeout_seconds=_require_positive_float(
                config.get(
                    "timeout_seconds",
                    10.0,
                ),
                "timeout_seconds",
            ),
            max_payload_bytes=_require_positive_int(
                config.get(
                    "max_payload_bytes",
                    _DEFAULT_MAX_PAYLOAD_BYTES,
                ),
                "max_payload_bytes",
            ),
            max_response_bytes=_require_non_negative_int(
                config.get(
                    "max_response_bytes",
                    _DEFAULT_MAX_RESPONSE_BYTES,
                ),
                "max_response_bytes",
            ),
            allow_http=_coerce_bool(
                config.get(
                    "allow_http",
                    False,
                ),
                "allow_http",
            ),
            allow_private_hosts=_coerce_bool(
                config.get(
                    "allow_private_hosts",
                    False,
                ),
                "allow_private_hosts",
            ),
            default_headers=raw_headers,
            retry_policy=retry_policy,
            user_agent=str(config.get(
                "user_agent",
                _DEFAULT_USER_AGENT,
            )),
        )


# ============================================================================
# Public exports
# ============================================================================

__all__ = [
    "WebhookEvent",
    "WebhookHTTPResponse",
    "WebhookDeliveryResult",
    "WebhookRetryPolicy",
    "WebhookBackend",
    "UrllibWebhookBackend",
    "WebhookService",
    "sign_webhook_payload",
    "verify_webhook_signature",
]

# ============================================================================
# Standalone self-test
# ============================================================================

if __name__ == "__main__":
    print("\n=== Running Webhook Service ===\n")
    printer.status("TEST", "Webhook Service initialized", "info")

    class _CaptureWebhookBackend(WebhookBackend):
        """Deterministic backend used only by the standalone test."""

        def __init__(self) -> None:
            self.requests = []
            self.calls = 0
            self.closed = False

        def send(
            self,
            *,
            url: str,
            body: bytes,
            headers: Mapping[str, str],
            timeout_seconds: float,
            max_response_bytes: int,
        ) -> WebhookHTTPResponse:
            self.calls += 1

            self.requests.append({
                "url": url,
                "body": body,
                "headers": dict(headers),
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
            })

            # Exercise retry logic.
            if self.calls == 1:
                return WebhookHTTPResponse(
                    status_code=503,
                    headers={
                        "Retry-After": "0",
                    },
                    body=b"temporarily unavailable",
                )

            return WebhookHTTPResponse(
                status_code=204,
                headers={
                    "X-Request-ID": "test-request",
                },
                body=b"",
            )

        def close(self) -> None:
            self.closed = True

    try:
        backend = _CaptureWebhookBackend()

        secret = b"s" * 32

        service = WebhookService(
            "https://127.0.0.1/webhooks/slai",
            secret,
            backend=backend,

            # Localhost is deliberately used by this isolated test.
            allow_private_hosts=True,
            retry_policy=WebhookRetryPolicy(
                max_retries=2,
                base_delay=0.0,
                max_delay=0.0,
                jitter=False,
            ),
        )

        # --------------------------------------------------------------
        # Delivery and retry
        # --------------------------------------------------------------
        result = service.send(
            "user.phone_verified",
            {
                "user_id": "user-123",
                "phone_verified": True,
            },
            event_id="evt-test-001",
        )

        assert result.status_code == 204
        assert result.attempts == 2
        assert backend.calls == 2

        printer.status("DELIVERY", "Retry and delivery passed", "success")

        # --------------------------------------------------------------
        # Stable idempotency metadata
        # --------------------------------------------------------------
        last_request = (backend.requests[-1])

        request_headers = (last_request["headers"])

        assert (request_headers["Idempotency-Key"] == "evt-test-001")
        assert (request_headers["X-SLAI-Webhook-Id"] == "evt-test-001")
        assert (request_headers["X-SLAI-Webhook-Event"] == "user.phone_verified")

        printer.status("IDEMPOTENCY", "Stable event identity passed", "success")

        # --------------------------------------------------------------
        # Signature verification
        # --------------------------------------------------------------
        assert verify_webhook_signature(
            secret,
            body=last_request["body"],
            timestamp=int(request_headers["X-SLAI-Webhook-Timestamp"]),
            event_id="evt-test-001",
            signature=request_headers["X-SLAI-Webhook-Signature"],
        )

        printer.status("SIGNATURE", "HMAC signature verification passed","success")

        # --------------------------------------------------------------
        # Event envelope
        # --------------------------------------------------------------
        decoded = json.loads(last_request["body"].decode("utf-8"))

        assert (decoded["id"] == "evt-test-001")

        assert (decoded["type"] == "user.phone_verified")

        assert (decoded["data"]["phone_verified"] is True)

        printer.status("ENVELOPE", "Canonical webhook envelope passed", "success")

        # --------------------------------------------------------------
        # Oversized payload rejection
        # --------------------------------------------------------------
        small_service = WebhookService(
            "https://127.0.0.1/webhook",
            secret,
            backend=_CaptureWebhookBackend(),
            allow_private_hosts=True,
            max_payload_bytes=128,
            retry_policy=WebhookRetryPolicy(max_retries=0, ),
        )

        try:
            small_service.send(
                "payload.large",
                {
                    "value": "x" * 512,
                },
            )

            raise AssertionError("expected WebhookPayloadTooLargeError")

        except WebhookPayloadTooLargeError:
            printer.status("PAYLOAD", "Oversized payload rejected", "success")

        finally:
            small_service.close()

        # --------------------------------------------------------------
        # Malformed event type
        # --------------------------------------------------------------
        try:
            service.send(
                "invalid event type!",
                {
                    "ok": False,
                },
            )

            raise AssertionError("expected WebhookConfigurationError")

        except WebhookConfigurationError:
            printer.status("EVENT", "Invalid event type rejected", "success")

        # --------------------------------------------------------------
        # Health
        # --------------------------------------------------------------
        health = service.health_check(resolve_destination=False)

        assert (health["scheme"] == "https")
        assert (health["closed"] is False)

        printer.status("HEALTH", f"Health check passed: {health}", "success")

        # --------------------------------------------------------------
        # Shutdown
        # --------------------------------------------------------------
        service.close()

        assert backend.closed is True
        assert service.closed is True

        printer.status("SHUTDOWN", "Webhook service closed cleanly", "success")

    except Exception as exc:
        printer.status("FAIL", f"Webhook Service test failed: {exc}", "error")
        logger.exception("Webhook Service standalone test failed")

        raise

    print("\n=== Test ran successfully ===\n")
