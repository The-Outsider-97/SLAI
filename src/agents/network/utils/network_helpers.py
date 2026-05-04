"""
Shared helpers for the network subsystem.

This module centralizes the network helper logic. It exists to keep transport,
routing, lifecycle, policy, and telemetry code focused on their core concerns
instead of repeatedly re-implementing parsing, normalization, serialization,
redaction, endpoint handling, identifier generation, and config-backed utility
behaviors.

The helpers here are intentionally scoped to reusable network-domain utilities.
They do not own routing strategy, reliability policy, adapter execution, or
delivery state transitions. Instead, they provide the stable primitives those
modules depend on.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import re
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode, urlparse, urlunparse

from .config_loader import load_global_config, get_config_section
from .network_errors import *


GLOBAL_CONFIG = load_global_config()
NETWORK_HELPERS_CONFIG = get_config_section("network_helpers") or {}

_DEFAULT_PROTOCOL = str(NETWORK_HELPERS_CONFIG.get("default_protocol", "http")).strip().lower() or "http"
_DEFAULT_TIMEOUT_MS = int(NETWORK_HELPERS_CONFIG.get("default_timeout_ms", 5000) or 5000)
_MIN_TIMEOUT_MS = int(NETWORK_HELPERS_CONFIG.get("min_timeout_ms", 1) or 1)
_MAX_TIMEOUT_MS = int(NETWORK_HELPERS_CONFIG.get("max_timeout_ms", 300000) or 300000)
_DEFAULT_ENCODING = str(NETWORK_HELPERS_CONFIG.get("default_encoding", "utf-8"))
_MAX_PAYLOAD_BYTES = int(NETWORK_HELPERS_CONFIG.get("max_payload_bytes", 10 * 1024 * 1024) or (10 * 1024 * 1024))
_DEFAULT_CONTENT_TYPE = str(
    NETWORK_HELPERS_CONFIG.get("default_content_type", "application/json")
).strip() or "application/json"

_PROTOCOL_ALIASES = {
    "https": "https",
    "http": "http",
    "http1": "http",
    "http2": "http",
    "ws": "websocket",
    "wss": "websocket",
    "websocket": "websocket",
    "grpc": "grpc",
    "grpcs": "grpc",
    "mq": "queue",
    "queue": "queue",
    "amqp": "queue",
    "kafka": "queue",
    "sqs": "queue",
    "pubsub": "queue",
}

_SECURE_PROTOCOLS = {
    "https",
    "wss",
    "tls",
    "ssl",
    "grpcs",
}

_DEFAULT_PORTS = {
    "http": 80,
    "https": 443,
    "websocket": 80,
    "ws": 80,
    "wss": 443,
    "grpc": 50051,
    "grpcs": 443,
    "queue": 5672,
    "amqp": 5672,
    "amqps": 5671,
    "mqtt": 1883,
    "mqtts": 8883,
}

_SENSITIVE_KEYS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "api_key",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "passwd",
    "client_secret",
    "private_key",
    "certificate",
    "cookie",
    "set-cookie",
    "session",
    "session_id",
}
_SENSITIVE_KEYS.update(
    {str(v).strip().lower() for v in NETWORK_HELPERS_CONFIG.get("sensitive_keys", []) if str(v).strip()}
)

_CONTENT_TYPE_ALIASES = {
    "json": "application/json",
    "application/json": "application/json",
    "text": "text/plain",
    "text/plain": "text/plain",
    "binary": "application/octet-stream",
    "bytes": "application/octet-stream",
    "application/octet-stream": "application/octet-stream",
}


@dataclass(frozen=True, slots=True)
class ParsedEndpoint:
    """
    Normalized endpoint view used by adapters, policy checks, and routing logic.

    The object keeps both the original textual source and the parsed components
    so callers can preserve user-specified input while still working with a
    stable normalized representation for downstream decisions.
    """

    raw: str
    scheme: str
    protocol: str
    host: str
    port: int
    path: str = ""
    query: str = ""
    fragment: str = ""
    username: Optional[str] = None
    password: Optional[str] = None

    @property
    def secure(self) -> bool:
        return is_secure_protocol(self.scheme) or is_secure_protocol(self.protocol)

    @property
    def authority(self) -> str:
        credentials = ""
        if self.username:
            credentials = self.username
            if self.password:
                credentials += ":***"
            credentials += "@"
        return f"{credentials}{self.host}:{self.port}"

    @property
    def netloc(self) -> str:
        credentials = ""
        if self.username:
            credentials = self.username
            if self.password:
                credentials += f":{self.password}"
            credentials += "@"
        return f"{credentials}{self.host}:{self.port}"

    @property
    def normalized(self) -> str:
        return urlunparse(
            (
                self.scheme,
                self.netloc,
                self.path or "",
                "",
                self.query or "",
                self.fragment or "",
            )
        )

    @property
    def socket_address(self) -> Tuple[str, int]:
        return (self.host, self.port)

    def to_dict(self, *, redact_credentials: bool = True) -> Dict[str, Any]:
        netloc = self.authority if redact_credentials else self.netloc
        normalized = urlunparse(
            (
                self.scheme,
                netloc,
                self.path or "",
                "",
                self.query or "",
                self.fragment or "",
            )
        )
        return {
            "raw": self.raw,
            "scheme": self.scheme,
            "protocol": self.protocol,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "query": self.query,
            "fragment": self.fragment,
            "username": self.username,
            "secure": self.secure,
            "normalized": normalized,
        }


def utcnow() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return utcnow().isoformat()


def json_safe(value: Any, *, max_depth: int = 5, _depth: int = 0) -> Any:
    """
    Convert arbitrary values into a JSON-safe representation.

    This is deliberately conservative because the result is intended for
    telemetry, logs, shared-memory persistence, and error contexts.
    """
    if _depth >= max_depth:
        return repr(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, bytes):
        try:
            return value.decode(_DEFAULT_ENCODING)
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "length": len(value),
                "data": base64.b64encode(value).decode("ascii"),
            }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item, max_depth=max_depth, _depth=_depth + 1)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item, max_depth=max_depth, _depth=_depth + 1) for item in value]

    if isinstance(value, Exception):
        if isinstance(value, NetworkError):
            return value.to_dict(include_cause=True, include_traceback=False)
        return {"type": type(value).__name__, "message": str(value)}

    return repr(value)


def stable_json_dumps(value: Any, *, sort_keys: bool = True, ensure_ascii: bool = False,
                      separators: Tuple[str, str] = (",", ":")) -> str:
    """Serialize a value deterministically for hashing, caching, and telemetry."""
    try:
        return json.dumps(
            json_safe(value),
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            separators=separators,
        )
    except Exception as exc:  # noqa: BLE001 - normalization is intentional here.
        raise PayloadSerializationError(
            "Failed to serialize value to stable JSON.",
            context={"operation": "serialize"},
            details={"value_type": type(value).__name__},
            cause=exc,
        ) from exc


def merge_mappings(
    *mappings: Optional[Mapping[str, Any]],
    deep: bool = True,
    drop_none: bool = False,
) -> Dict[str, Any]:
    """
    Merge multiple mapping-like values into a new dictionary.

    Later mappings win. Nested mappings are merged recursively when `deep=True`.
    """

    merged: Dict[str, Any] = {}
    for candidate in mappings:
        if candidate is None:
            continue
        current = ensure_mapping(candidate, field_name="mapping")
        for key, value in current.items():
            if drop_none and value is None:
                continue
            if (
                deep
                and key in merged
                and isinstance(merged[key], Mapping)
                and isinstance(value, Mapping)
            ):
                merged[key] = merge_mappings(merged[key], value, deep=True, drop_none=drop_none)
            else:
                merged[key] = value
    return merged


def ensure_mapping(value: Any, *, field_name: str = "value", allow_none: bool = False) -> Dict[str, Any]:
    """Validate and normalize a mapping-like input into a concrete dictionary."""
    if value is None:
        if allow_none:
            return {}
        raise PayloadValidationError(
            f"{field_name} must be a mapping, received None.",
            context={"operation": "validate_mapping"},
            details={"field_name": field_name},
        )
    if isinstance(value, Mapping):
        return dict(value)
    raise PayloadValidationError(
        f"{field_name} must be a mapping-like object.",
        context={"operation": "validate_mapping"},
        details={"field_name": field_name, "received_type": type(value).__name__},
    )


def ensure_sequence(
    value: Any,
    *,
    field_name: str = "value",
    allow_none: bool = False,
    coerce_scalar: bool = False,
) -> Tuple[Any, ...]:
    """Validate sequence-like input while avoiding accidental string iteration."""
    if value is None:
        if allow_none:
            return ()
        raise PayloadValidationError(
            f"{field_name} must be a sequence, received None.",
            context={"operation": "validate_sequence"},
            details={"field_name": field_name},
        )
    if isinstance(value, (str, bytes, bytearray)):
        if coerce_scalar:
            return (value,)
        raise PayloadValidationError(
            f"{field_name} must be a sequence, not a scalar string/bytes value.",
            context={"operation": "validate_sequence"},
            details={"field_name": field_name, "received_type": type(value).__name__},
        )
    if isinstance(value, Sequence):
        return tuple(value)
    if coerce_scalar:
        return (value,)
    raise PayloadValidationError(
        f"{field_name} must be a sequence-like object.",
        context={"operation": "validate_sequence"},
        details={"field_name": field_name, "received_type": type(value).__name__},
    )


def ensure_non_empty_string(
    value: Any,
    *,
    field_name: str,
    strip: bool = True,
) -> str:
    """Require a non-empty string value."""
    if not isinstance(value, str):
        raise PayloadValidationError(
            f"{field_name} must be a string.",
            context={"operation": "validate_string"},
            details={"field_name": field_name, "received_type": type(value).__name__},
        )
    normalized = value.strip() if strip else value
    if not normalized:
        raise PayloadValidationError(
            f"{field_name} must not be empty.",
            context={"operation": "validate_string"},
            details={"field_name": field_name},
        )
    return normalized


def normalize_protocol_name(protocol: Optional[str]) -> str:
    """Normalize protocol names and common aliases into a stable internal form."""
    if protocol is None:
        return _DEFAULT_PROTOCOL
    normalized = ensure_non_empty_string(protocol, field_name="protocol").lower()
    return _PROTOCOL_ALIASES.get(normalized, normalized)


def normalize_channel_name(channel: Optional[str]) -> str:
    """
    Normalize a channel identifier.

    In the current network domain model, channel names and protocol names are
    closely aligned, but this helper keeps the intent explicit at call sites.
    """
    if channel is None:
        return _DEFAULT_PROTOCOL
    return normalize_protocol_name(channel)


def normalize_metadata(
    metadata: Optional[Mapping[str, Any]],
    *,
    drop_none: bool = True,
) -> Dict[str, Any]:
    """Normalize metadata for envelopes, telemetry, and error contexts."""
    if metadata is None:
        return {}
    source = ensure_mapping(metadata, field_name="metadata")
    normalized: Dict[str, Any] = {}
    for key, value in source.items():
        if drop_none and value is None:
            continue
        normalized[str(key)] = json_safe(value)
    return normalized


def normalize_tags(tags: Optional[Sequence[Any]]) -> Tuple[str, ...]:
    """Normalize tags while preserving order and removing duplicates."""
    if tags is None:
        return ()
    values = ensure_sequence(tags, field_name="tags", allow_none=True, coerce_scalar=True)
    deduplicated: Dict[str, None] = {}
    for item in values:
        text = str(item).strip()
        if text:
            deduplicated[text] = None
    return tuple(deduplicated.keys())


def normalize_headers(
    headers: Optional[Mapping[str, Any]],
    *,
    lowercase: bool = True,
    drop_none: bool = True,
) -> Dict[str, str]:
    """Normalize request/response headers into a clean string dictionary."""
    if headers is None:
        return {}
    source = ensure_mapping(headers, field_name="headers")
    normalized: Dict[str, str] = {}
    for key, value in source.items():
        if value is None and drop_none:
            continue
        key_text = ensure_non_empty_string(str(key), field_name="header_name")
        if lowercase:
            key_text = key_text.lower()
        if isinstance(value, bytes):
            
            try:
                value_text = value.decode(_DEFAULT_ENCODING)
            except UnicodeDecodeError as exc:
                raise PayloadSerializationError(
                    "Header value could not be decoded using the configured encoding.",
                    context={"operation": "normalize_headers"},
                    details={"header_name": key_text},
                    cause=exc,
                ) from exc
        elif isinstance(value, (list, tuple, set)):
            value_text = ", ".join(str(item) for item in value if item is not None)
        else:
            value_text = str(value)
        normalized[key_text] = value_text
    return normalized



def coerce_timeout_ms(
    timeout_ms: Optional[Any],
    *,
    default: Optional[int] = None,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """
    Validate and normalize a timeout value in milliseconds.

    Accepts integers, floats, and digit-like strings. The result is clamped to
    configured bounds unless explicit bounds are provided.
    """

    if timeout_ms is None:
        value = _DEFAULT_TIMEOUT_MS if default is None else int(default)
    else:
        try:
            if isinstance(timeout_ms, bool):
                raise TypeError("boolean is not a valid timeout value")
            value = int(float(timeout_ms))
        except Exception as exc:  # noqa: BLE001 - normalization is intentional here.
            raise PayloadValidationError(
                "Timeout value must be numeric milliseconds.",
                context={"operation": "coerce_timeout"},
                details={"received_value": repr(timeout_ms), "received_type": type(timeout_ms).__name__},
                cause=exc,
            ) from exc

    lower_bound = _MIN_TIMEOUT_MS if minimum is None else int(minimum)
    upper_bound = _MAX_TIMEOUT_MS if maximum is None else int(maximum)

    if value < lower_bound:
        return lower_bound
    if value > upper_bound:
        return upper_bound
    return value


def clamp_timeout_ms(
    timeout_ms: int,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Clamp an already numeric timeout to configured or supplied bounds."""
    lower_bound = _MIN_TIMEOUT_MS if minimum is None else int(minimum)
    upper_bound = _MAX_TIMEOUT_MS if maximum is None else int(maximum)
    return max(lower_bound, min(int(timeout_ms), upper_bound))


def generate_message_id(prefix: str = "msg") -> str:
    """Generate a collision-resistant message identifier."""
    return f"{prefix}_{uuid.uuid4().hex}"


def generate_correlation_id(prefix: str = "corr") -> str:
    """Generate a correlation identifier for distributed tracing."""
    return f"{prefix}_{uuid.uuid4().hex}"


def generate_session_id(prefix: str = "sess") -> str:
    """Generate a session identifier for connection/session snapshots."""
    return f"{prefix}_{uuid.uuid4().hex}"


def generate_idempotency_key(
    payload: Any,
    *,
    namespace: str = "network",
    route: Optional[str] = None,
    operation: Optional[str] = None,
) -> str:
    """
    Generate a deterministic idempotency key from stable request properties.

    The helper uses canonical JSON where possible and falls back to a safe
    textual representation if serialization is not straightforward.
    """

    basis = {
        "namespace": namespace,
        "route": route,
        "operation": operation,
        "payload": json_safe(payload),
    }
    digest = hashlib.sha256(stable_json_dumps(basis).encode(_DEFAULT_ENCODING)).hexdigest()
    return digest


def default_port_for_protocol(protocol: Optional[str]) -> Optional[int]:
    if protocol is None:
        return _DEFAULT_PORTS.get(_DEFAULT_PROTOCOL)
    normalized = normalize_protocol_name(protocol)
    original = str(protocol).strip().lower()
    return _DEFAULT_PORTS.get(normalized) or _DEFAULT_PORTS.get(original)


def is_secure_protocol(protocol: Optional[str]) -> bool:
    if protocol is None:
        return False
    normalized = str(protocol).strip().lower()
    return normalized in _SECURE_PROTOCOLS or normalized.endswith("+tls") or normalized.endswith("+ssl")


def is_ip_address(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def is_private_host(host: Optional[str]) -> bool:
    if not host:
        return False
    if is_ip_address(host):
        return ipaddress.ip_address(host).is_private
    host_lower = host.lower()
    if host_lower in {"localhost", "localhost.localdomain"}:
        return True
    return host_lower.endswith(".local") or host_lower.endswith(".internal")


def parse_endpoint(
    endpoint: str,
    *,
    default_scheme: Optional[str] = None,
    protocol: Optional[str] = None,
    require_host: bool = True,
    default_port: Optional[int] = None,
) -> ParsedEndpoint:
    """
    Parse and normalize an endpoint string into a structured representation.

    Supported forms include:
        - example.com
        - example.com:8443
        - https://example.com/api
        - ws://localhost:9000/stream
    """

    try:
        endpoint_text = ensure_non_empty_string(endpoint, field_name="endpoint")
        requested_protocol = normalize_protocol_name(protocol or default_scheme or _DEFAULT_PROTOCOL)

        endpoint_to_parse = endpoint_text
        if "://" not in endpoint_text:
            implied_scheme = default_scheme or protocol or _DEFAULT_PROTOCOL
            endpoint_to_parse = f"{implied_scheme}://{endpoint_text}"

        parsed = urlparse(endpoint_to_parse)

        scheme = (parsed.scheme or default_scheme or protocol or _DEFAULT_PROTOCOL).strip().lower()
        host = parsed.hostname or ""
        if require_host and not host:
            raise PayloadValidationError(
                "Endpoint must include a host.",
                context={"operation": "parse_endpoint", "endpoint": endpoint_text, "protocol": requested_protocol},
            )

        normalized_protocol = normalize_protocol_name(protocol or scheme)
        port = parsed.port or default_port or default_port_for_protocol(scheme) or default_port_for_protocol(normalized_protocol)
        if port is None:
            raise NetworkConfigurationError(
                "No port could be determined for endpoint.",
                context={"operation": "parse_endpoint", "endpoint": endpoint_text, "protocol": normalized_protocol},
                details={"scheme": scheme},
            )

        if port < 1 or port > 65535:
            raise PayloadValidationError(
                "Endpoint port must be between 1 and 65535.",
                context={"operation": "parse_endpoint", "endpoint": endpoint_text, "protocol": normalized_protocol},
                details={"port": port},
            )

        return ParsedEndpoint(
            raw=endpoint_text,
            scheme=scheme,
            protocol=normalized_protocol,
            host=host,
            port=int(port),
            path=parsed.path or "",
            query=parsed.query or "",
            fragment=parsed.fragment or "",
            username=parsed.username,
            password=parsed.password,
        )
    except NetworkError:
        raise
    except Exception as exc:  # noqa: BLE001 - boundary normalization is intentional.
        raise normalize_network_exception(
            exc,
            operation="parse_endpoint",
            endpoint=str(endpoint),
            protocol=protocol or default_scheme or _DEFAULT_PROTOCOL,
        ) from exc


def normalize_endpoint(
    endpoint: str | ParsedEndpoint,
    *,
    default_scheme: Optional[str] = None,
    protocol: Optional[str] = None,
    require_host: bool = True,
    default_port: Optional[int] = None,
) -> str:
    """Return a normalized endpoint string with explicit scheme and port."""
    if isinstance(endpoint, ParsedEndpoint):
        return endpoint.normalized
    parsed = parse_endpoint(
        endpoint,
        default_scheme=default_scheme,
        protocol=protocol,
        require_host=require_host,
        default_port=default_port,
    )
    return parsed.normalized


def build_endpoint(
    *,
    scheme: str,
    host: str,
    port: Optional[int] = None,
    path: str = "",
    query: Optional[Mapping[str, Any] | str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    fragment: str = "",
) -> str:
    """Construct a normalized endpoint string from explicit components."""
    normalized_scheme = ensure_non_empty_string(scheme, field_name="scheme").lower()
    normalized_host = ensure_non_empty_string(host, field_name="host")
    resolved_port = int(port) if port is not None else default_port_for_protocol(normalized_scheme)
    if resolved_port is None:
        raise NetworkConfigurationError(
            "A port must be provided when no protocol default exists.",
            context={"operation": "build_endpoint", "protocol": normalized_scheme, "endpoint": normalized_host},
        )
    if resolved_port < 1 or resolved_port > 65535:
        raise PayloadValidationError(
            "Port must be between 1 and 65535.",
            context={"operation": "build_endpoint", "protocol": normalized_scheme, "endpoint": normalized_host},
            details={"port": resolved_port},
        )

    normalized_path = path or ""
    if normalized_path and not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"

    if isinstance(query, Mapping):
        query_items = []
        for key, value in query.items():
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    query_items.append((str(key), "" if item is None else str(item)))
            else:
                query_items.append((str(key), "" if value is None else str(value)))
        query_string = urlencode(query_items, doseq=True)
    elif query is None:
        query_string = ""
    else:
        query_string = str(query)

    credentials = ""
    if username:
        credentials = username
        if password:
            credentials += f":{password}"
        credentials += "@"

    return urlunparse(
        (
            normalized_scheme,
            f"{credentials}{normalized_host}:{resolved_port}",
            normalized_path,
            "",
            query_string,
            fragment or "",
        )
    )


def estimate_payload_size(payload: Any, *, encoding: str = _DEFAULT_ENCODING) -> int:
    """
    Estimate serialized payload size in bytes.

    The function is intentionally exact for bytes/strings and stable for common
    structured payloads by serializing through the same normalization path that
    transport code will typically use.
    """
    if payload is None:
        return 0
    if isinstance(payload, bytes):
        return len(payload)
    if isinstance(payload, bytearray):
        return len(payload)
    if isinstance(payload, memoryview):
        return payload.nbytes
    if isinstance(payload, str):
        return len(payload.encode(encoding))
    try:
        return len(stable_json_dumps(payload).encode(encoding))
    except NetworkError:
        raise
    except Exception as exc:  # noqa: BLE001 - boundary normalization is intentional.
        raise normalize_network_exception(
            exc,
            operation="estimate_payload_size",
            metadata={"payload_type": type(payload).__name__},
        ) from exc


def infer_content_type(payload: Any, explicit_content_type: Optional[str] = None) -> str:
    """Infer a reasonable content type for a payload."""
    if explicit_content_type:
        normalized = str(explicit_content_type).strip().lower()
        return _CONTENT_TYPE_ALIASES.get(normalized, normalized)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return "application/octet-stream"
    if isinstance(payload, str):
        return "text/plain"
    return _DEFAULT_CONTENT_TYPE


def serialize_payload(
    payload: Any,
    *,
    content_type: Optional[str] = None,
    encoding: str = _DEFAULT_ENCODING,
    max_payload_bytes: Optional[int] = None,
) -> bytes:
    """
    Serialize a payload for transport.

    Supported content families:
        - application/json
        - text/plain
        - application/octet-stream
    """
    resolved_content_type = infer_content_type(payload, explicit_content_type=content_type)
    resolved_max_payload = _MAX_PAYLOAD_BYTES if max_payload_bytes is None else int(max_payload_bytes)

    try:
        if payload is None:
            body = b""
        elif resolved_content_type == "application/json":
            body = stable_json_dumps(payload).encode(encoding)
        elif resolved_content_type == "text/plain":
            if isinstance(payload, bytes):
                body = payload
            else:
                body = str(payload).encode(encoding)
        elif resolved_content_type == "application/octet-stream":
            if isinstance(payload, bytes):
                body = payload
            elif isinstance(payload, bytearray):
                body = bytes(payload)
            elif isinstance(payload, memoryview):
                body = payload.tobytes()
            else:
                raise PayloadSerializationError(
                    "Binary payloads must be bytes-like objects.",
                    context={"operation": "serialize"},
                    details={"payload_type": type(payload).__name__},
                )
        else:
            raise ProtocolNegotiationError(
                "Unsupported content type for payload serialization.",
                context={"operation": "serialize", "protocol": resolved_content_type},
                details={"content_type": resolved_content_type},
            )
    except NetworkError:
        raise
    except Exception as exc:  # noqa: BLE001 - boundary normalization is intentional.
        raise normalize_network_exception(
            exc,
            operation="serialize",
            protocol=resolved_content_type,
            metadata={"payload_type": type(payload).__name__},
        ) from exc

    if len(body) > resolved_max_payload:
        raise PayloadTooLargeError(
            "Serialized payload exceeds configured size limit.",
            context={
                "operation": "serialize",
                "protocol": resolved_content_type,
                "payload_size": len(body),
            },
            details={"max_payload_bytes": resolved_max_payload},
        )
    return body


def deserialize_payload(
    payload: bytes | bytearray | memoryview | str,
    *,
    content_type: Optional[str] = None,
    encoding: str = _DEFAULT_ENCODING,
) -> Any:
    """Deserialize transport payloads into application-level values."""
    resolved_content_type = infer_content_type(payload, explicit_content_type=content_type)
    raw = b""
    try:
        if isinstance(payload, memoryview):
            raw = payload.tobytes()
        elif isinstance(payload, bytearray):
            raw = bytes(payload)
        elif isinstance(payload, bytes):
            raw = payload
        else:
            raw = str(payload).encode(encoding)

        if resolved_content_type == "application/octet-stream":
            return raw

        decoded = raw.decode(encoding)

        if resolved_content_type == "text/plain":
            return decoded
        if resolved_content_type == "application/json":
            if not decoded.strip():
                return None
            return json.loads(decoded)

        raise ProtocolNegotiationError(
            "Unsupported content type for payload deserialization.",
            context={"operation": "deserialize", "protocol": resolved_content_type},
            details={"content_type": resolved_content_type},
        )
    except NetworkError:
        raise
    except json.JSONDecodeError as exc:
        raise PayloadDeserializationError(
            "Failed to deserialize JSON payload.",
            context={"operation": "deserialize", "protocol": resolved_content_type, "payload_size": len(raw)},
            details={"encoding": encoding},
            cause=exc,
        ) from exc
    except UnicodeDecodeError as exc:
        raise PayloadDeserializationError(
            "Failed to decode payload bytes using configured encoding.",
            context={"operation": "deserialize", "protocol": resolved_content_type, "payload_size": len(raw)},
            details={"encoding": encoding},
            cause=exc,
        ) from exc
    except Exception as exc:  # noqa: BLE001 - boundary normalization is intentional.
        raise normalize_network_exception(
            exc,
            operation="deserialize",
            protocol=resolved_content_type,
            payload_size=len(raw),
            metadata={"encoding": encoding},
        ) from exc


def coerce_payload_bytes(
    payload: Any,
    *,
    content_type: Optional[str] = None,
    encoding: str = _DEFAULT_ENCODING,
    max_payload_bytes: Optional[int] = None,
) -> bytes:
    """Return a bytes payload regardless of the original in-memory representation."""
    if isinstance(payload, bytes):
        if max_payload_bytes is not None and len(payload) > int(max_payload_bytes):
            raise PayloadTooLargeError(
                "Payload exceeds configured size limit.",
                context={"operation": "coerce_payload_bytes", "payload_size": len(payload)},
                details={"max_payload_bytes": int(max_payload_bytes)},
            )
        return payload
    return serialize_payload(
        payload,
        content_type=content_type,
        encoding=encoding,
        max_payload_bytes=max_payload_bytes,
    )


def build_message_envelope(
    envelope: Optional[Mapping[str, Any]] = None,
    *,
    payload: Any = None,
    channel: Optional[str] = None,
    protocol: Optional[str] = None,
    endpoint: Optional[str] = None,
    route: Optional[str] = None,
    message_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    headers: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    timeout_ms: Optional[Any] = None,
    delivery_mode: str = "at_least_once",
    content_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build and normalize a delivery envelope for the network lifecycle.

    The helper keeps lifecycle-centric normalization in one place without
    embedding actual state-machine logic. Modules can add richer delivery fields
    later while reusing the same canonical shape.
    """
    base = ensure_mapping(envelope, field_name="envelope", allow_none=True)
    merged_headers = normalize_headers(merge_mappings(base.get("headers"), headers))
    merged_metadata = normalize_metadata(merge_mappings(base.get("metadata"), metadata))

    resolved_payload = payload if payload is not None else base.get("payload")
    resolved_channel = normalize_channel_name(channel or base.get("channel"))
    resolved_protocol = normalize_protocol_name(protocol or base.get("protocol") or resolved_channel)
    resolved_endpoint = endpoint or base.get("endpoint")
    resolved_content_type = infer_content_type(
        resolved_payload,
        content_type or base.get("content_type"),
    )

    resolved_message_id = message_id or base.get("message_id") or generate_message_id()
    resolved_correlation_id = correlation_id or base.get("correlation_id") or generate_correlation_id()
    resolved_idempotency_key = (
        idempotency_key
        or base.get("idempotency_key")
        or generate_idempotency_key(
            resolved_payload,
            namespace="network",
            route=route or base.get("route"),
            operation=base.get("operation"),
        )
    )

    normalized: Dict[str, Any] = merge_mappings(
        base,
        {
            "message_id": resolved_message_id,
            "correlation_id": resolved_correlation_id,
            "idempotency_key": resolved_idempotency_key,
            "channel": resolved_channel,
            "protocol": resolved_protocol,
            "endpoint": resolved_endpoint,
            "route": route or base.get("route"),
            "payload": resolved_payload,
            "content_type": resolved_content_type,
            "headers": merged_headers,
            "metadata": merged_metadata,
            "timeout_ms": coerce_timeout_ms(timeout_ms if timeout_ms is not None else base.get("timeout_ms")),
            "delivery_mode": ensure_non_empty_string(
                delivery_mode or base.get("delivery_mode") or "at_least_once",
                field_name="delivery_mode",
            ),
            "created_at": base.get("created_at") or utc_timestamp(),
        },
    )

    normalized["payload_size"] = estimate_payload_size(normalized.get("payload"))
    return normalized


def extract_retry_after_ms(headers: Optional[Mapping[str, Any]]) -> Optional[int]:
    """
    Extract a retry-after duration from headers and return milliseconds.

    Supports both numeric seconds and RFC-style datetime values in a practical
    best-effort manner without taking a dependency on heavier HTTP libraries.
    """
    if not headers:
        return None

    normalized = normalize_headers(headers)
    value = normalized.get("retry-after")
    if not value:
        return None

    seconds_match = re.fullmatch(r"\s*(\d+)\s*", value)
    if seconds_match:
        return int(seconds_match.group(1)) * 1000

    try:
        retry_dt = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
        delta_ms = int((retry_dt - utcnow()).total_seconds() * 1000)
        return max(0, delta_ms)
    except ValueError:
        return None


def redact_sensitive_value(
    value: Any,
    *,
    replacement: str = "***REDACTED***",
    preserve_length: bool = False,
) -> Any:
    """Redact sensitive scalar values for logs and telemetry."""
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return sanitize_for_logging(value, replacement=replacement, preserve_length=preserve_length)
    if isinstance(value, bytes):
        if preserve_length:
            return f"{replacement}[bytes:{len(value)}]"
        return replacement
    if preserve_length:
        return f"{replacement}[len:{len(str(value))}]"
    return replacement


def redact_sensitive_mapping(
    mapping: Optional[Mapping[str, Any]],
    *,
    replacement: str = "***REDACTED***",
    preserve_length: bool = False,
) -> Dict[str, Any]:
    """Redact known secret-bearing keys in a mapping."""
    if mapping is None:
        return {}
    source = ensure_mapping(mapping, field_name="mapping")
    redacted: Dict[str, Any] = {}
    for key, value in source.items():
        key_text = str(key)
        if key_text.strip().lower() in _SENSITIVE_KEYS:
            redacted[key_text] = redact_sensitive_value(value, replacement=replacement, preserve_length=preserve_length)
        elif isinstance(value, Mapping):
            redacted[key_text] = redact_sensitive_mapping(
                value,
                replacement=replacement,
                preserve_length=preserve_length,
            )
        elif isinstance(value, (list, tuple)):
            redacted[key_text] = [
                redact_sensitive_mapping(item, replacement=replacement, preserve_length=preserve_length)
                if isinstance(item, Mapping)
                else item
                for item in value
            ]
        else:
            redacted[key_text] = value
    return redacted


def sanitize_for_logging(
    value: Any,
    *,
    replacement: str = "***REDACTED***",
    preserve_length: bool = False,
    max_depth: int = 5,
    _depth: int = 0,
) -> Any:
    """
    Produce a log-safe representation of arbitrary values.

    This combines JSON-safe normalization with recursive secret redaction.
    """
    if _depth >= max_depth:
        return repr(value)

    if isinstance(value, Mapping):
        return redact_sensitive_mapping(
            {
                str(key): sanitize_for_logging(
                    item,
                    replacement=replacement,
                    preserve_length=preserve_length,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
                for key, item in value.items()
            },
            replacement=replacement,
            preserve_length=preserve_length,
        )

    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            sanitize_for_logging(
                item,
                replacement=replacement,
                preserve_length=preserve_length,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if isinstance(value, bytes):
        return redact_sensitive_value(value, replacement=replacement, preserve_length=preserve_length)

    if isinstance(value, NetworkError):
        return sanitize_for_logging(
            value.to_dict(include_cause=True, include_traceback=False),
            replacement=replacement,
            preserve_length=preserve_length,
            max_depth=max_depth,
            _depth=_depth + 1,
        )

    return json_safe(value, max_depth=max_depth, _depth=_depth)