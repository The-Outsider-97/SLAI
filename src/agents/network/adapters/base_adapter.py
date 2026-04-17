"""
Base adapter contract and lifecycle scaffolding for SLAI's Network Agent.

This module defines the production-grade base class that all specialized
transport adapters inherit from. It exists to keep HTTP, WebSocket, gRPC,
queue, and future adapters aligned on the same operational contract without
forcing each implementation to repeatedly rebuild connection state handling,
envelope normalization, delivery bookkeeping, health snapshots, capability
exposure, structured error conversion, and observability-friendly metadata.

The base adapter owns the generic adapter lifecycle:
- configuration loading and adapter-level defaults,
- connection/session state handling,
- payload normalization and size enforcement,
- standardized send/receive/ack/nack/close orchestration,
- structured memory updates for sessions, deliveries, retries, and endpoint
  health,
- stable health/capability snapshots for routing and adapter selection.

It intentionally does not own protocol-specific transport logic. Specialized
adapters implement the concrete transport hooks while inheriting the shared
safety, state, memory, and error-handling behaviors defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from time import monotonic
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..utils import *
from ..network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Base Adapter")
printer = PrettyPrinter()

__all__ = [
    "AdapterCapabilities",
    "AdapterHealthSnapshot",
    "AdapterSessionState",
    "BaseAdapter",
]


@dataclass(slots=True)
class AdapterCapabilities:
    """
    Capability surface advertised by an adapter implementation.

    Routing and compatibility logic can use this snapshot without having to know
    adapter internals.
    """

    adapter_name: str
    protocol: str
    channel: str
    supports_streaming: bool = False
    supports_bidirectional_streaming: bool = False
    supports_ack: bool = True
    supports_nack: bool = True
    supports_batch_send: bool = False
    supports_headers: bool = True
    supports_tls: bool = True
    supports_reconnect: bool = True
    supports_receive: bool = True
    supports_request_reply: bool = True
    max_payload_bytes: Optional[int] = None
    default_port: Optional[int] = None
    auth_modes: Tuple[str, ...] = field(default_factory=tuple)
    content_types: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "protocol": self.protocol,
            "channel": self.channel,
            "supports_streaming": self.supports_streaming,
            "supports_bidirectional_streaming": self.supports_bidirectional_streaming,
            "supports_ack": self.supports_ack,
            "supports_nack": self.supports_nack,
            "supports_batch_send": self.supports_batch_send,
            "supports_headers": self.supports_headers,
            "supports_tls": self.supports_tls,
            "supports_reconnect": self.supports_reconnect,
            "supports_receive": self.supports_receive,
            "supports_request_reply": self.supports_request_reply,
            "max_payload_bytes": self.max_payload_bytes,
            "default_port": self.default_port,
            "auth_modes": list(self.auth_modes),
            "content_types": list(self.content_types),
            "metadata": json_safe(self.metadata),
        }


@dataclass(slots=True)
class AdapterHealthSnapshot:
    """Current health and reliability posture for an adapter instance."""

    adapter_name: str
    protocol: str
    channel: str
    endpoint: Optional[str] = None
    status: str = "unknown"
    connected: bool = False
    available: bool = True
    consecutive_failures: int = 0
    total_connects: int = 0
    total_sends: int = 0
    total_receives: int = 0
    total_acks: int = 0
    total_nacks: int = 0
    total_failures: int = 0
    last_latency_ms: Optional[float] = None
    last_error: Optional[Dict[str, Any]] = None
    last_connected_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    last_failure_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "adapter_name": self.adapter_name,
            "protocol": self.protocol,
            "channel": self.channel,
            "endpoint": self.endpoint,
            "status": self.status,
            "connected": self.connected,
            "available": self.available,
            "consecutive_failures": self.consecutive_failures,
            "total_connects": self.total_connects,
            "total_sends": self.total_sends,
            "total_receives": self.total_receives,
            "total_acks": self.total_acks,
            "total_nacks": self.total_nacks,
            "total_failures": self.total_failures,
            "last_latency_ms": self.last_latency_ms,
            "last_error": self.last_error,
            "last_connected_at": self.last_connected_at,
            "last_activity_at": self.last_activity_at,
            "last_failure_at": self.last_failure_at,
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class AdapterSessionState:
    """In-memory connection/session state for an adapter instance."""

    session_id: str
    adapter_name: str
    protocol: str
    channel: str
    endpoint: Optional[str] = None
    state: str = "idle"
    connected: bool = False
    last_connected_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    closed_at: Optional[str] = None
    close_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "session_id": self.session_id,
            "adapter_name": self.adapter_name,
            "protocol": self.protocol,
            "channel": self.channel,
            "endpoint": self.endpoint,
            "state": self.state,
            "connected": self.connected,
            "last_connected_at": self.last_connected_at,
            "last_activity_at": self.last_activity_at,
            "closed_at": self.closed_at,
            "close_reason": self.close_reason,
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


class BaseAdapter(ABC):
    """
    Abstract base class for all network transport adapters.

    Subclasses implement the transport-specific hooks while inheriting a stable
    adapter contract and lifecycle behavior.
    """

    DEFAULT_CONTENT_TYPES: Tuple[str, ...] = ("application/json", "text/plain", "application/octet-stream")
    DEFAULT_AUTH_MODES: Tuple[str, ...] = ("none",)

    def __init__(
        self,
        *,
        adapter_name: Optional[str] = None,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        memory: Optional[NetworkMemory] = None,
        config: Optional[Mapping[str, Any]] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        self.config = load_global_config()
        self.base_adapter_config = get_config_section("network_base_adapter") or {}
        self.adapters_config = get_config_section("network_adapters") or {}
        self.adapter_config = merge_mappings(
            self.base_adapter_config,
            self.adapters_config,
            ensure_mapping(config, field_name="config", allow_none=True),
        )

        self.adapter_name = ensure_non_empty_string(
            adapter_name or self.__class__.__name__.replace("Adapter", "").strip() or self.__class__.__name__,
            field_name="adapter_name",
        )
        self.protocol = normalize_protocol_name(protocol or self.adapter_config.get("protocol") or "http")
        self.channel = normalize_channel_name(channel or self.adapter_config.get("channel") or self.protocol)
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.default_timeout_ms = coerce_timeout_ms(
            self.adapter_config.get("default_timeout_ms"),
            default=5000,
            minimum=1,
            maximum=300000,
        )
        self.connect_timeout_ms = coerce_timeout_ms(
            self.adapter_config.get("connect_timeout_ms"),
            default=self.default_timeout_ms,
            minimum=1,
            maximum=300000,
        )
        self.receive_timeout_ms = coerce_timeout_ms(
            self.adapter_config.get("receive_timeout_ms"),
            default=self.default_timeout_ms,
            minimum=1,
            maximum=300000,
        )
        self.idle_ttl_seconds = self._get_non_negative_int_config("session_idle_ttl_seconds", 1800)
        self.delivery_ttl_seconds = self._get_non_negative_int_config("delivery_ttl_seconds", 7200)
        self.max_payload_bytes = self._get_non_negative_int_config("max_payload_bytes", 10 * 1024 * 1024)
        self.max_receive_payload_bytes = self._get_non_negative_int_config(
            "max_receive_payload_bytes",
            self.max_payload_bytes,
        )
        self.fail_health_status = self._get_status_config("failure_health_status", "degraded")
        self.initial_health_status = self._get_status_config("initial_health_status", "unknown")
        self.closed_health_status = self._get_status_config("closed_health_status", "idle")
        self.require_endpoint_on_connect = self._get_bool_config("require_endpoint_on_connect", True)
        self.record_delivery_state = self._get_bool_config("record_delivery_state", True)
        self.record_endpoint_health = self._get_bool_config("record_endpoint_health", True)
        self.record_session_snapshots = self._get_bool_config("record_session_snapshots", True)
        self.strict_ack_support = self._get_bool_config("strict_ack_support", False)
        self.strict_receive_support = self._get_bool_config("strict_receive_support", False)
        self.auto_normalize_endpoint = self._get_bool_config("auto_normalize_endpoint", True)
        self.auto_generate_correlation_id = self._get_bool_config("auto_generate_correlation_id", True)
        self.emit_log_safe_payloads = self._get_bool_config("emit_log_safe_payloads", True)

        resolved_endpoint = self._normalize_endpoint_reference(endpoint)
        self.session = AdapterSessionState(
            session_id=generate_session_id(prefix=f"sess_{self.adapter_name.lower()}"),
            adapter_name=self.adapter_name,
            protocol=self.protocol,
            channel=self.channel,
            endpoint=resolved_endpoint,
            state="idle",
            connected=False,
        )
        self.health = AdapterHealthSnapshot(
            adapter_name=self.adapter_name,
            protocol=self.protocol,
            channel=self.channel,
            endpoint=resolved_endpoint,
            status=self.initial_health_status,
            connected=False,
        )
        self._stats: Dict[str, int] = {
            "connects": 0,
            "connect_failures": 0,
            "sends": 0,
            "send_failures": 0,
            "receives": 0,
            "receive_failures": 0,
            "acks": 0,
            "ack_failures": 0,
            "nacks": 0,
            "nack_failures": 0,
            "closes": 0,
        }
        self._last_result: Optional[Dict[str, Any]] = None

        if self.record_session_snapshots:
            self._sync_session_memory("initialized")
        if self.record_endpoint_health and resolved_endpoint:
            self._sync_health_memory()

    # ------------------------------------------------------------------
    # Required transport hooks for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _connect_impl(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Perform protocol-specific connection establishment."""

    @abstractmethod
    def _send_impl(
        self,
        *,
        payload: bytes,
        envelope: Mapping[str, Any],
        timeout_ms: int,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Perform protocol-specific payload transmission."""

    @abstractmethod
    def _close_impl(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Perform protocol-specific resource cleanup."""

    def _receive_impl(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        raise AdapterCapabilityError(
            f"{self.adapter_name} does not implement receive().",
            context={"operation": "receive", "channel": self.channel, "protocol": self.protocol},
        )

    def _ack_impl(
        self,
        *,
        message_id: str,
        correlation_id: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        _ = (message_id, correlation_id, metadata)  # explicitly unused
        raise AdapterCapabilityError(
            f"{self.adapter_name} does not implement ack().",
            context={"operation": "ack", "channel": self.channel, "protocol": self.protocol},
        )
    
    def _nack_impl(
        self,
        *,
        message_id: str,
        correlation_id: Optional[str],
        reason: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        _ = (message_id, correlation_id, reason, metadata)  # explicitly unused
        raise AdapterCapabilityError(
            f"{self.adapter_name} does not implement nack().",
            context={"operation": "nack", "channel": self.channel, "protocol": self.protocol},
        )

    # ------------------------------------------------------------------
    # Public adapter contract
    # ------------------------------------------------------------------
    def connect(
        self,
        endpoint: Optional[str] = None,
        *,
        timeout_ms: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Establish a transport session against an endpoint."""
        normalized_metadata = normalize_metadata(metadata)
        resolved_timeout = coerce_timeout_ms(timeout_ms, default=self.connect_timeout_ms)
        with self._lock:
            target_endpoint = self._resolve_connection_endpoint(endpoint)
            start = monotonic()
            try:
                result = self._connect_impl(
                    endpoint=target_endpoint,
                    timeout_ms=resolved_timeout,
                    metadata=normalized_metadata,
                ) or {}
                latency_ms = round((monotonic() - start) * 1000.0, 3)
                self._stats["connects"] += 1
                self.health.total_connects += 1
                self.health.last_latency_ms = latency_ms
                self.health.connected = True
                self.health.status = "healthy"
                self.health.endpoint = target_endpoint
                self.health.last_connected_at = utc_timestamp()
                self.health.last_activity_at = self.health.last_connected_at
                self.health.consecutive_failures = 0
                self.health.last_error = None

                self.session.endpoint = target_endpoint
                self.session.connected = True
                self.session.state = "connected"
                self.session.last_connected_at = self.health.last_connected_at
                self.session.last_activity_at = self.health.last_connected_at
                self.session.closed_at = None
                self.session.close_reason = None

                snapshot = {
                    "adapter_name": self.adapter_name,
                    "protocol": self.protocol,
                    "channel": self.channel,
                    "endpoint": target_endpoint,
                    "connected": True,
                    "timeout_ms": resolved_timeout,
                    "latency_ms": latency_ms,
                    "session_id": self.session.session_id,
                    "result": json_safe(result),
                    "connected_at": self.health.last_connected_at,
                    "metadata": normalized_metadata,
                }
                self._last_result = snapshot
                self._sync_session_memory("connected")
                self._sync_health_memory(metadata={"latency_ms": latency_ms})
                return snapshot
            except NetworkError as exc:
                self._handle_operation_failure(
                    exc,
                    operation="connect",
                    endpoint=target_endpoint,
                    timeout_ms=resolved_timeout,
                    metadata=normalized_metadata,
                )
                raise
            except Exception as exc:
                normalized = normalize_network_exception(
                    exc,
                    operation="connect",
                    endpoint=target_endpoint,
                    channel=self.channel,
                    protocol=self.protocol,
                    timeout_ms=resolved_timeout,
                    session_id=self.session.session_id,
                    metadata=normalized_metadata,
                )
                self._handle_operation_failure(
                    normalized,
                    operation="connect",
                    endpoint=target_endpoint,
                    timeout_ms=resolved_timeout,
                    metadata=normalized_metadata,
                )
                raise normalized from exc

    def send(
        self,
        payload: Any,
        *,
        envelope: Optional[Mapping[str, Any]] = None,
        timeout_ms: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize and transmit a payload over the active transport."""
        with self._lock:
            self._ensure_connected(operation="send")
            normalized_metadata = normalize_metadata(metadata)
            resolved_timeout = coerce_timeout_ms(timeout_ms, default=self.default_timeout_ms)
            normalized_envelope = self._build_delivery_envelope(
                payload=payload,
                envelope=envelope,
                timeout_ms=resolved_timeout,
                metadata=normalized_metadata,
                content_type=content_type,
            )
            payload_bytes = coerce_payload_bytes(
                normalized_envelope["payload"],
                content_type=normalized_envelope.get("content_type"),
                max_payload_bytes=self.max_payload_bytes,
            )
            start = monotonic()
            try:
                result = self._send_impl(
                    payload=payload_bytes,
                    envelope=normalized_envelope,
                    timeout_ms=resolved_timeout,
                    metadata=normalized_metadata,
                ) or {}
                latency_ms = round((monotonic() - start) * 1000.0, 3)
                self._stats["sends"] += 1
                self.health.total_sends += 1
                self._mark_activity(latency_ms=latency_ms)
                self._record_delivery_state(
                    "sent",
                    envelope=normalized_envelope,
                    metadata={
                        **normalized_metadata,
                        "latency_ms": latency_ms,
                        "send_result": sanitize_for_logging(result) if self.emit_log_safe_payloads else json_safe(result),
                    },
                )
                snapshot = {
                    "adapter_name": self.adapter_name,
                    "protocol": self.protocol,
                    "channel": self.channel,
                    "endpoint": self.session.endpoint,
                    "message_id": normalized_envelope["message_id"],
                    "correlation_id": normalized_envelope["correlation_id"],
                    "idempotency_key": normalized_envelope["idempotency_key"],
                    "payload_size": len(payload_bytes),
                    "content_type": normalized_envelope.get("content_type"),
                    "timeout_ms": resolved_timeout,
                    "latency_ms": latency_ms,
                    "result": json_safe(result),
                    "metadata": normalized_metadata,
                }
                self._last_result = snapshot
                self._sync_session_memory("active")
                self._sync_health_memory(metadata={"latency_ms": latency_ms})
                return snapshot
            except NetworkError as exc:
                self._handle_send_failure(exc, normalized_envelope, resolved_timeout, normalized_metadata)
                raise
            except Exception as exc:
                normalized = normalize_network_exception(
                    exc,
                    operation="send",
                    endpoint=self.session.endpoint,
                    channel=self.channel,
                    protocol=self.protocol,
                    correlation_id=normalized_envelope.get("correlation_id"),
                    idempotency_key=normalized_envelope.get("idempotency_key"),
                    session_id=self.session.session_id,
                    timeout_ms=resolved_timeout,
                    payload_size=len(payload_bytes),
                    metadata=normalized_metadata,
                )
                self._handle_send_failure(normalized, normalized_envelope, resolved_timeout, normalized_metadata)
                raise normalized from exc

    def recv(
        self,
        *,
        timeout_ms: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Receive and normalize an inbound payload from the transport."""
        with self._lock:
            self._ensure_connected(operation="receive")
            if not self.capabilities.supports_receive and self.strict_receive_support:
                raise AdapterCapabilityError(
                    f"{self.adapter_name} is configured without receive support.",
                    context={"operation": "receive", "channel": self.channel, "protocol": self.protocol},
                )

            normalized_metadata = normalize_metadata(metadata)
            resolved_timeout = coerce_timeout_ms(timeout_ms, default=self.receive_timeout_ms)
            start = monotonic()
            try:
                raw = self._receive_impl(timeout_ms=resolved_timeout, metadata=normalized_metadata) or {}
                normalized_message = self._normalize_received_message(raw, resolved_timeout, normalized_metadata)
                latency_ms = round((monotonic() - start) * 1000.0, 3)
                self._stats["receives"] += 1
                self.health.total_receives += 1
                self._mark_activity(latency_ms=latency_ms)
                self._record_delivery_state(
                    "received",
                    envelope=normalized_message,
                    metadata={**normalized_metadata, "latency_ms": latency_ms},
                )
                snapshot = merge_mappings(
                    normalized_message,
                    {
                        "adapter_name": self.adapter_name,
                        "protocol": self.protocol,
                        "channel": self.channel,
                        "endpoint": self.session.endpoint,
                        "timeout_ms": resolved_timeout,
                        "latency_ms": latency_ms,
                        "metadata": normalized_metadata,
                    },
                )
                self._last_result = snapshot
                self._sync_session_memory("active")
                self._sync_health_memory(metadata={"latency_ms": latency_ms})
                return snapshot
            except NetworkError as exc:
                self._handle_operation_failure(
                    exc,
                    operation="receive",
                    endpoint=self.session.endpoint,
                    timeout_ms=resolved_timeout,
                    metadata=normalized_metadata,
                )
                self._stats["receive_failures"] += 1
                raise
            except Exception as exc:
                normalized = normalize_network_exception(
                    exc,
                    operation="receive",
                    endpoint=self.session.endpoint,
                    channel=self.channel,
                    protocol=self.protocol,
                    session_id=self.session.session_id,
                    timeout_ms=resolved_timeout,
                    metadata=normalized_metadata,
                )
                self._handle_operation_failure(
                    normalized,
                    operation="receive",
                    endpoint=self.session.endpoint,
                    timeout_ms=resolved_timeout,
                    metadata=normalized_metadata,
                )
                self._stats["receive_failures"] += 1
                raise normalized from exc

    def ack(
        self,
        message: str | Mapping[str, Any],
        *,
        correlation_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Acknowledge a previously received or delivered message."""
        with self._lock:
            self._ensure_connected(operation="ack")
            if not self.capabilities.supports_ack and self.strict_ack_support:
                raise AdapterCapabilityError(
                    f"{self.adapter_name} is configured without ack support.",
                    context={"operation": "ack", "channel": self.channel, "protocol": self.protocol},
                )
            message_id = self._extract_message_identifier(message)
            normalized_metadata = normalize_metadata(metadata)
            try:
                result = self._ack_impl(
                    message_id=message_id,
                    correlation_id=correlation_id,
                    metadata=normalized_metadata,
                ) or {}
                self._stats["acks"] += 1
                self.health.total_acks += 1
                self._mark_activity()
                snapshot = {
                    "adapter_name": self.adapter_name,
                    "protocol": self.protocol,
                    "channel": self.channel,
                    "endpoint": self.session.endpoint,
                    "message_id": message_id,
                    "correlation_id": correlation_id,
                    "acknowledged": True,
                    "result": json_safe(result),
                    "metadata": normalized_metadata,
                }
                self._record_delivery_state(
                    "acked",
                    envelope={"message_id": message_id, "correlation_id": correlation_id},
                    metadata=normalized_metadata,
                )
                self._last_result = snapshot
                self._sync_session_memory("active")
                self._sync_health_memory()
                return snapshot
            except NetworkError as exc:
                self._stats["ack_failures"] += 1
                self._handle_operation_failure(
                    exc,
                    operation="ack",
                    endpoint=self.session.endpoint,
                    metadata=normalized_metadata,
                    correlation_id=correlation_id,
                )
                raise
            except Exception as exc:
                self._stats["ack_failures"] += 1
                normalized = normalize_network_exception(
                    exc,
                    operation="ack",
                    endpoint=self.session.endpoint,
                    channel=self.channel,
                    protocol=self.protocol,
                    correlation_id=correlation_id,
                    session_id=self.session.session_id,
                    metadata={**normalized_metadata, "message_id": message_id},
                )
                self._handle_operation_failure(
                    normalized,
                    operation="ack",
                    endpoint=self.session.endpoint,
                    metadata=normalized_metadata,
                    correlation_id=correlation_id,
                )
                raise normalized from exc

    def nack(
        self,
        message: str | Mapping[str, Any],
        *,
        reason: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Negatively acknowledge a message so callers can reroute or retry."""
        with self._lock:
            self._ensure_connected(operation="nack")
            message_id = self._extract_message_identifier(message)
            normalized_metadata = normalize_metadata(metadata)
            try:
                result = self._nack_impl(
                    message_id=message_id,
                    correlation_id=correlation_id,
                    reason=reason,
                    metadata=normalized_metadata,
                ) or {}
                self._stats["nacks"] += 1
                self.health.total_nacks += 1
                self._mark_activity()
                snapshot = {
                    "adapter_name": self.adapter_name,
                    "protocol": self.protocol,
                    "channel": self.channel,
                    "endpoint": self.session.endpoint,
                    "message_id": message_id,
                    "correlation_id": correlation_id,
                    "reason": reason,
                    "nacked": True,
                    "result": json_safe(result),
                    "metadata": normalized_metadata,
                }
                self._record_delivery_state(
                    "nacked",
                    envelope={"message_id": message_id, "correlation_id": correlation_id},
                    metadata={**normalized_metadata, "reason": reason},
                )
                self._last_result = snapshot
                self._sync_session_memory("active")
                self._sync_health_memory()
                return snapshot
            except NetworkError as exc:
                self._stats["nack_failures"] += 1
                self._handle_operation_failure(
                    exc,
                    operation="nack",
                    endpoint=self.session.endpoint,
                    metadata={**normalized_metadata, "reason": reason},
                    correlation_id=correlation_id,
                )
                raise
            except Exception as exc:
                self._stats["nack_failures"] += 1
                normalized = normalize_network_exception(
                    exc,
                    operation="nack",
                    endpoint=self.session.endpoint,
                    channel=self.channel,
                    protocol=self.protocol,
                    correlation_id=correlation_id,
                    session_id=self.session.session_id,
                    metadata={**normalized_metadata, "message_id": message_id, "reason": reason},
                )
                self._handle_operation_failure(
                    normalized,
                    operation="nack",
                    endpoint=self.session.endpoint,
                    metadata={**normalized_metadata, "reason": reason},
                    correlation_id=correlation_id,
                )
                raise normalized from exc

    def close(
        self,
        *,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Close the active transport session and release resources."""
        normalized_metadata = normalize_metadata(metadata)
        with self._lock:
            try:
                result = self._close_impl(reason=reason, metadata=normalized_metadata) or {}
            except NetworkError:
                raise
            except Exception as exc:
                raise normalize_network_exception(
                    exc,
                    operation="close",
                    endpoint=self.session.endpoint,
                    channel=self.channel,
                    protocol=self.protocol,
                    session_id=self.session.session_id,
                    metadata={**normalized_metadata, "reason": reason},
                ) from exc

            self._stats["closes"] += 1
            closed_at = utc_timestamp()
            self.session.connected = False
            self.session.state = "closed"
            self.session.closed_at = closed_at
            self.session.close_reason = reason
            self.session.last_activity_at = closed_at

            self.health.connected = False
            self.health.status = self.closed_health_status
            self.health.last_activity_at = closed_at
            self.health.metadata["last_close_reason"] = reason

            self._sync_session_memory("closed", metadata={"reason": reason})
            self._sync_health_memory(metadata={"reason": reason})
            snapshot = {
                "adapter_name": self.adapter_name,
                "protocol": self.protocol,
                "channel": self.channel,
                "endpoint": self.session.endpoint,
                "closed": True,
                "closed_at": closed_at,
                "reason": reason,
                "result": json_safe(result),
                "metadata": normalized_metadata,
            }
            self._last_result = snapshot
            return snapshot

    # ------------------------------------------------------------------
    # Adapter state and metadata helpers
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name=self.adapter_name,
            protocol=self.protocol,
            channel=self.channel,
            supports_streaming=self._get_bool_config("supports_streaming", False),
            supports_bidirectional_streaming=self._get_bool_config("supports_bidirectional_streaming", False),
            supports_ack=self._get_bool_config("supports_ack", True),
            supports_nack=self._get_bool_config("supports_nack", True),
            supports_batch_send=self._get_bool_config("supports_batch_send", False),
            supports_headers=self._get_bool_config("supports_headers", True),
            supports_tls=self._get_bool_config("supports_tls", True),
            supports_reconnect=self._get_bool_config("supports_reconnect", True),
            supports_receive=self._get_bool_config("supports_receive", True),
            supports_request_reply=self._get_bool_config("supports_request_reply", True),
            max_payload_bytes=self.max_payload_bytes,
            default_port=default_port_for_protocol(self.protocol),
            auth_modes=tuple(self._get_sequence_config("auth_modes", self.DEFAULT_AUTH_MODES)),
            content_types=tuple(self._get_sequence_config("content_types", self.DEFAULT_CONTENT_TYPES)),
            metadata=normalize_metadata(self.adapter_config.get("capabilities_metadata")),
        )

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "protocol": self.protocol,
            "channel": self.channel,
            "session": self.session.to_dict(),
            "health": self.health.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "stats": dict(self._stats),
            "last_result": sanitize_for_logging(self._last_result) if self.emit_log_safe_payloads else json_safe(self._last_result),
        }

    def get_health_snapshot(self) -> Dict[str, Any]:
        return self.health.to_dict()

    def is_connected(self) -> bool:
        return bool(self.session.connected)

    def can_send(self, payload: Any, *, content_type: Optional[str] = None) -> bool:
        try:
            coerce_payload_bytes(
                payload,
                content_type=content_type,
                max_payload_bytes=self.max_payload_bytes,
            )
            return True
        except NetworkError:
            return False

    # ------------------------------------------------------------------
    # Internal shared logic
    # ------------------------------------------------------------------
    def _build_delivery_envelope(
        self,
        *,
        payload: Any,
        envelope: Optional[Mapping[str, Any]],
        timeout_ms: int,
        metadata: Mapping[str, Any],
        content_type: Optional[str],
    ) -> Dict[str, Any]:
        base_envelope = build_message_envelope(
            ensure_mapping(envelope, field_name="envelope", allow_none=True),
            payload=payload,
            channel=self.channel,
            protocol=self.protocol,
            endpoint=self.session.endpoint,
            timeout_ms=timeout_ms,
            metadata=metadata,
            content_type=content_type,
        )
        if self.auto_generate_correlation_id and not base_envelope.get("correlation_id"):
            base_envelope["correlation_id"] = generate_correlation_id()
        base_envelope["session_id"] = self.session.session_id
        return base_envelope

    def _normalize_received_message(
        self,
        raw: Mapping[str, Any],
        timeout_ms: int,
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        incoming = ensure_mapping(raw, field_name="received_message")
        payload = incoming.get("payload")
        content_type = infer_content_type(payload, explicit_content_type=incoming.get("content_type"))

        if isinstance(payload, (bytes, bytearray, memoryview, str)):
            deserialized_payload = deserialize_payload(
                payload,
                content_type=content_type,
            )
            payload_size = estimate_payload_size(payload)
        else:
            deserialized_payload = payload
            payload_size = estimate_payload_size(payload)

        if payload_size > self.max_receive_payload_bytes:
            raise PayloadTooLargeError(
                "Received payload exceeds adapter receive limit.",
                context={
                    "operation": "receive",
                    "channel": self.channel,
                    "protocol": self.protocol,
                    "endpoint": self.session.endpoint,
                    "payload_size": payload_size,
                },
                details={"max_receive_payload_bytes": self.max_receive_payload_bytes},
            )

        message_id = incoming.get("message_id") or generate_message_id(prefix=f"recv_{self.adapter_name.lower()}")
        correlation_id = incoming.get("correlation_id") or generate_correlation_id()

        return merge_mappings(
            incoming,
            {
                "message_id": message_id,
                "correlation_id": correlation_id,
                "payload": deserialized_payload,
                "payload_size": payload_size,
                "content_type": content_type,
                "timeout_ms": timeout_ms,
                "received_at": utc_timestamp(),
                "metadata": normalize_metadata(merge_mappings(incoming.get("metadata"), metadata)),
            },
        )

    def _extract_message_identifier(self, message: str | Mapping[str, Any]) -> str:
        if isinstance(message, Mapping):
            candidate = message.get("message_id") or message.get("id") or message.get("correlation_id")
            return ensure_non_empty_string(str(candidate), field_name="message_id")
        return ensure_non_empty_string(str(message), field_name="message_id")

    def _ensure_connected(self, *, operation: str) -> None:
        if not self.session.connected:
            raise SessionUnavailableError(
                f"{self.adapter_name} is not connected.",
                context={
                    "operation": operation,
                    "channel": self.channel,
                    "protocol": self.protocol,
                    "endpoint": self.session.endpoint,
                    "session_id": self.session.session_id,
                },
            )

    def _resolve_connection_endpoint(self, endpoint: Optional[str]) -> str:
        candidate = endpoint or self.session.endpoint or self.adapter_config.get("endpoint")
        if candidate is None and self.require_endpoint_on_connect:
            raise AdapterInitializationError(
                "Adapter connect() requires an endpoint but none was provided.",
                context={"operation": "connect", "channel": self.channel, "protocol": self.protocol},
                details={"adapter_name": self.adapter_name},
            )
        if candidate is None:
            return ""
        normalized = self._normalize_endpoint_reference(str(candidate), required=True)
        self.session.endpoint = normalized
        self.health.endpoint = normalized
        return normalized

    def _normalize_endpoint_reference(self, endpoint: Optional[str], *, required: bool = False) -> Optional[str]:
        if endpoint is None:
            if required:
                raise PayloadValidationError(
                    "Endpoint is required.",
                    context={"operation": "adapter_endpoint_normalization", "channel": self.channel, "protocol": self.protocol},
                )
            return None

        endpoint_text = ensure_non_empty_string(str(endpoint), field_name="endpoint")
        if not self.auto_normalize_endpoint:
            return endpoint_text

        if "://" in endpoint_text:
            return normalize_endpoint(endpoint_text, protocol=self.protocol)

        parsed_fallback = parse_endpoint(
            endpoint_text,
            default_scheme=self.protocol,
            protocol=self.protocol,
            require_host=True,
        )
        return parsed_fallback.normalized

    def _mark_activity(self, *, latency_ms: Optional[float] = None) -> None:
        now = utc_timestamp()
        self.session.last_activity_at = now
        self.health.last_activity_at = now
        self.health.last_latency_ms = latency_ms if latency_ms is not None else self.health.last_latency_ms
        self.health.status = "healthy" if self.session.connected else self.health.status
        self.health.consecutive_failures = 0
        self.health.last_error = None

    def _handle_send_failure(
        self,
        error: NetworkError,
        envelope: Mapping[str, Any],
        timeout_ms: int,
        metadata: Mapping[str, Any],
    ) -> None:
        self._stats["send_failures"] += 1
        self._handle_operation_failure(
            error,
            operation="send",
            endpoint=self.session.endpoint,
            timeout_ms=timeout_ms,
            metadata={**metadata, "message_id": envelope.get("message_id")},
            correlation_id=envelope.get("correlation_id"),
        )
        self._record_delivery_state(
            "failed",
            envelope=envelope,
            error=error,
            metadata={**metadata, "timeout_ms": timeout_ms},
        )

    def _handle_operation_failure(
        self,
        error: NetworkError,
        *,
        operation: str,
        endpoint: Optional[str],
        timeout_ms: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        now = utc_timestamp()
        self.health.connected = self.session.connected
        self.health.available = True
        self.health.status = self.fail_health_status
        self.health.total_failures += 1
        self.health.consecutive_failures += 1
        self.health.last_failure_at = now
        self.health.last_activity_at = now
        self.health.last_error = error.to_memory_snapshot()
        self.session.last_activity_at = now
        self.health.endpoint = endpoint or self.health.endpoint

        if operation == "connect":
            self._stats["connect_failures"] += 1
            self.session.connected = False
            self.session.state = "error"
            self.health.connected = False

        self._sync_session_memory(
            "error",
            metadata={"operation": operation, "timeout_ms": timeout_ms, **normalize_metadata(metadata)},
        )
        self._sync_health_memory(metadata={"operation": operation, **normalize_metadata(metadata)})

        if correlation_id is not None:
            self.memory.record_retry_event(
                error,
                attempt=1,
                max_attempts=1,
                endpoint=endpoint,
                channel=self.channel,
                route=None,
                correlation_id=correlation_id,
                message_id=None,
                metadata={"adapter_name": self.adapter_name, "operation": operation},
            )

    def _record_delivery_state(
        self,
        state: str,
        *,
        envelope: Mapping[str, Any],
        error: Optional[BaseException] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.record_delivery_state:
            return
        self.memory.record_delivery_state(
            state,
            message_id=str(envelope.get("message_id")) if envelope.get("message_id") is not None else None,
            correlation_id=str(envelope.get("correlation_id")) if envelope.get("correlation_id") is not None else None,
            endpoint=self.session.endpoint,
            channel=self.channel,
            route=str(envelope.get("route")) if envelope.get("route") is not None else None,
            retry_count=None,
            error=error,
            ttl_seconds=self.delivery_ttl_seconds,
            metadata={"adapter_name": self.adapter_name, **normalize_metadata(metadata)},
        )

    def _sync_session_memory(self, state: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if not self.record_session_snapshots:
            return
        self.session.state = state
        self.memory.update_session_snapshot(
            self.session.session_id,
            self.session.to_dict(),
            ttl_seconds=self.idle_ttl_seconds,
            merge_existing=True,
            metadata={"adapter_name": self.adapter_name, **normalize_metadata(metadata)},
        )

    def _sync_health_memory(self, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if not self.record_endpoint_health or not self.session.endpoint:
            return
        self.memory.update_endpoint_health(
            self.session.endpoint,
            status=self.health.status,
            latency_ms=int(self.health.last_latency_ms) if self.health.last_latency_ms is not None else None,
            success_rate=self._derive_success_rate(),
            error_rate=self._derive_error_rate(),
            circuit_state=None,
            last_error=self.health.last_error,
            capabilities=self.capabilities.to_dict(),
            metadata={"adapter_name": self.adapter_name, **normalize_metadata(metadata)},
            ttl_seconds=self.idle_ttl_seconds,
        )

    def _derive_success_rate(self) -> Optional[float]:
        total_attempts = self.health.total_sends + self._stats["send_failures"]
        if total_attempts <= 0:
            return None
        return round(self.health.total_sends / total_attempts, 6)

    def _derive_error_rate(self) -> Optional[float]:
        total_attempts = self.health.total_sends + self._stats["send_failures"]
        if total_attempts <= 0:
            return None
        return round(self._stats["send_failures"] / total_attempts, 6)

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.adapter_config.get(name, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise NetworkConfigurationError(
            "Invalid boolean value in base adapter configuration.",
            context={"operation": "base_adapter_config", "channel": self.channel, "protocol": self.protocol},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.adapter_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in base adapter configuration.",
                context={"operation": "base_adapter_config", "channel": self.channel, "protocol": self.protocol},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Configuration value must be non-negative.",
                context={"operation": "base_adapter_config", "channel": self.channel, "protocol": self.protocol},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _get_status_config(self, name: str, default: str) -> str:
        value = self.adapter_config.get(name, default)
        return ensure_non_empty_string(str(value), field_name=name).lower()

    def _get_sequence_config(self, name: str, default: Sequence[str]) -> Tuple[str, ...]:
        value = self.adapter_config.get(name, default)
        if isinstance(value, str):
            # Split on commas and strip each part
            parts = [p.strip() for p in value.split(",") if p.strip()]
            value = parts if parts else default
        values = ensure_sequence(value, field_name=name, allow_none=True, coerce_scalar=True)
        normalized: Dict[str, None] = {}
        for item in values:
            text = str(item).strip()
            if text:
                normalized[text] = None
        return tuple(normalized.keys()) or tuple(default)


class _DemoAdapter(BaseAdapter):
    """In-file demo adapter for contract validation and smoke testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(adapter_name="Demo", protocol="http", channel="http", **kwargs)
        self._queue: list[dict[str, Any]] = []

    def _connect_impl(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return {"connected": True, "endpoint": endpoint, "timeout_ms": timeout_ms, "metadata": dict(metadata)}

    def _send_impl(
        self,
        *,
        payload: bytes,
        envelope: Mapping[str, Any],
        timeout_ms: int,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        self._queue.append(
            {
                "message_id": envelope["message_id"],
                "correlation_id": envelope["correlation_id"],
                "payload": payload,
                "content_type": envelope.get("content_type"),
                "metadata": dict(metadata),
            }
        )
        return {"queued": True, "queue_depth": len(self._queue), "timeout_ms": timeout_ms}

    def _receive_impl(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if not self._queue:
            raise ReceiveFailureError(
                "No queued message available for receive.",
                context={"operation": "receive", "channel": self.channel, "protocol": self.protocol},
            )
        message = self._queue.pop(0)
        return {
            "message_id": message["message_id"],
            "correlation_id": message["correlation_id"],
            "payload": message["payload"],
            "content_type": message["content_type"],
            "metadata": merge_mappings(message.get("metadata"), metadata),
            "timeout_ms": timeout_ms,
        }

    def _ack_impl(
        self,
        *,
        message_id: str,
        correlation_id: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        return {"acknowledged": True, "message_id": message_id, "correlation_id": correlation_id, "metadata": dict(metadata)}

    def _nack_impl(
        self,
        *,
        message_id: str,
        correlation_id: Optional[str],
        reason: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        return {
            "nacked": True,
            "message_id": message_id,
            "correlation_id": correlation_id,
            "reason": reason,
            "metadata": dict(metadata),
        }

    def _close_impl(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self._queue.clear()
        return {"closed": True, "reason": reason, "metadata": dict(metadata)}


def _printer_status(label: str, message: str, level: str = "info") -> None:
    try:
        printer.status(label, message, level)
    except Exception:
        print(f"[{label}] {message}")


if __name__ == "__main__":
    print("\n=== Running Base Adapater ===\n")
    _printer_status("TEST", "Base Adapater initialized", "info")

    adapter = _DemoAdapter(endpoint="https://api.example.com/v1/relay")

    capabilities = adapter.capabilities.to_dict()
    _printer_status("TEST", "Capabilities resolved", "info")

    connected = adapter.connect(metadata={"region": "eu-west", "env": "test"})
    _printer_status("TEST", "Connection established", "info")

    sent = adapter.send(
        {"task": "relay", "payload": {"hello": "world"}},
        metadata={"priority": "normal"},
    )
    _printer_status("TEST", "Payload sent", "info")

    received = adapter.recv(metadata={"consumer": "demo"})
    _printer_status("TEST", "Payload received", "info")

    acked = adapter.ack(received["message_id"], correlation_id=received["correlation_id"], metadata={"result": "ok"})
    _printer_status("TEST", "Payload acknowledged", "info")

    nacked = adapter.nack(
        "msg_demo_secondary",
        correlation_id="corr_demo_secondary",
        reason="synthetic retry path",
        metadata={"retryable": True},
    )
    _printer_status("TEST", "Synthetic nack recorded", "info")

    state_snapshot = adapter.get_state_snapshot()
    health_snapshot = adapter.get_health_snapshot()
    memory_health = adapter.memory.get_network_health()

    closed = adapter.close(reason="demo complete", metadata={"cleanup": True})
    _printer_status("TEST", "Adapter closed", "info")

    print("Capabilities:", stable_json_dumps(capabilities))
    print("Connected:", stable_json_dumps(connected))
    print("Sent:", stable_json_dumps(sent))
    print("Received:", stable_json_dumps(received))
    print("Acked:", stable_json_dumps(acked))
    print("Nacked:", stable_json_dumps(nacked))
    print("State Snapshot:", stable_json_dumps(state_snapshot))
    print("Health Snapshot:", stable_json_dumps(health_snapshot))
    print("Memory Health:", stable_json_dumps(memory_health))
    print("Closed:", stable_json_dumps(closed))

    assert capabilities["supports_ack"] is True
    assert connected["connected"] is True
    assert sent["payload_size"] > 0
    assert received["payload"]["task"] == "relay"
    assert acked["acknowledged"] is True
    assert nacked["nacked"] is True
    assert state_snapshot["session"]["connected"] is True
    assert adapter.memory.get("network.session.snapshot")
    assert adapter.memory.get("network.endpoint.health")

    _printer_status("TEST", "All Base Adapater checks passed", "info")
    print("\n=== Test ran successfully ===\n")
