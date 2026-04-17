"""
Transport-bridge gRPC adapter implementation for SLAI's Network Agent.

This module provides the production-grade gRPC-oriented adapter that inherits
from the shared BaseAdapter contract. It keeps gRPC-specific semantics in one
place while relying on the base adapter for lifecycle orchestration,
structured memory updates, delivery bookkeeping, health snapshots, and
network-native error semantics.

The adapter is intentionally scoped to gRPC transport concerns:
- endpoint/session establishment for grpc:// and grpcs:// targets,
- method and call-type normalization,
- unary and streaming call orchestration through a transport bridge,
- request/response buffering for recv()-style consumption,
- application-level synthetic or transport-backed ack/nack semantics,
- adapter-local RPC state useful to routing and observability.

It does not own routing strategy, retry policy, circuit breaking, or policy
arbitration. Those belong to the higher-level network modules and agents.

Note:
    This adapter is implemented against a transport bridge interface rather
    than a direct grpcio dependency. That keeps the module importable and
    testable in environments where the gRPC runtime isn't present, while still
    giving specialized SLAI adapters a production-ready contract to integrate
    with real gRPC clients later.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..utils import *
from .base_adapter import BaseAdapter, AdapterCapabilities
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("gRPC Adapter")
printer = PrettyPrinter()

__all__ = ["GRPCTransportProtocol", "GRPCAdapter"]


_VALID_GRPC_CALL_TYPES = {"unary_unary", "unary_stream", "stream_unary", "stream_stream"}
_VALID_GRPC_ACK_MODES = {"synthetic", "transport", "disabled"}


@runtime_checkable
class GRPCTransportProtocol(Protocol):
    def connect(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None: ...
    def unary_unary(self, *, method: str, request: Any, timeout_ms: int, metadata: Mapping[str, Any]) -> Any: ...
    def unary_stream(self, *, method: str, request: Any, timeout_ms: int, metadata: Mapping[str, Any]) -> Iterable[Any]: ...
    def stream_unary(self, *, method: str, request: Sequence[Any], timeout_ms: int, metadata: Mapping[str, Any]) -> Any: ...
    def stream_stream(self, *, method: str, request: Sequence[Any], timeout_ms: int, metadata: Mapping[str, Any]) -> Iterable[Any]: ...
    def recv(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> Any: ...
    def ack(self, *, message_id: str, correlation_id: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None: ...
    def nack(self, *, message_id: str, correlation_id: Optional[str], reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None: ...
    def close(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None: ...

class GRPCAdapter(BaseAdapter):
    """
    Production-grade, transport-bridge gRPC adapter.

    The adapter understands gRPC method naming and call modes while delegating
    the actual transport execution to an injected bridge object.
    """

    DEFAULT_GRPC_CONTENT_TYPES: Tuple[str, ...] = (
        "application/json",
        "application/octet-stream",
        "text/plain",
    )
    DEFAULT_GRPC_AUTH_MODES: Tuple[str, ...] = ("none", "bearer", "mtls", "metadata")

    def __init__(
        self,
        *,
        memory=None,
        config: Optional[Mapping[str, Any]] = None,
        endpoint: Optional[str] = None,
        adapter_name: str = "GRPC",
        protocol: Optional[str] = None,
        transport: Optional[GRPCTransportProtocol] = None,
    ) -> None:
        provided_config = ensure_mapping(config, field_name="config", allow_none=True)
        section_config = get_config_section("network_grpc_adapter") or {}
        merged_grpc_config = merge_mappings(section_config, provided_config)
        self.grpc_adapter_config = merged_grpc_config
        self.service_name = str(merged_grpc_config.get("service_name")).strip() if merged_grpc_config.get("service_name") is not None else None
        self.default_call_type = str(merged_grpc_config.get("default_call_type", "unary_unary")).strip().lower() or "unary_unary"
        self.ack_mode = str(merged_grpc_config.get("ack_mode", "synthetic")).strip().lower() or "synthetic"
        self.nack_mode = str(merged_grpc_config.get("nack_mode", self.ack_mode)).strip().lower() or self.ack_mode

        inferred_protocol = protocol or merged_grpc_config.get("protocol") or self._infer_protocol_from_endpoint(endpoint) or "grpc"

        super().__init__(
            adapter_name=adapter_name,
            protocol=inferred_protocol,
            channel="grpc",
            memory=memory,
            config=merged_grpc_config,
            endpoint=endpoint or merged_grpc_config.get("endpoint"),
        )

        self.grpc_adapter_config = merge_mappings(section_config, self.adapter_config)
        self.transport = transport or self.grpc_adapter_config.get("transport")

        self.default_method = self._get_optional_string_config("default_method")
        self.service_name = self._get_optional_string_config("service_name")
        self.healthcheck_method = self._get_optional_string_config("healthcheck_method")
        self.default_call_type = self._get_call_type_config("default_call_type", "unary_unary")
        self.ack_mode = self._get_ack_mode_config("ack_mode", "synthetic")
        self.nack_mode = self._get_ack_mode_config("nack_mode", self.ack_mode)

        self.healthcheck_on_connect = self._get_bool_config("healthcheck_on_connect", False)
        self.require_method_on_send = self._get_bool_config("require_method_on_send", True)
        self.receive_from_buffer = self._get_bool_config("receive_from_buffer", True)
        self.consume_buffered_messages_on_recv = self._get_bool_config("consume_buffered_messages_on_recv", True)
        self.strict_method_format = self._get_bool_config("strict_method_format", False)
        self.use_tls = self._get_bool_config("use_tls", True)

        self.max_response_history_size = max(1, self._get_non_negative_int_config("max_response_history_size", 200))
        self.max_stream_buffer_size = max(1, self._get_non_negative_int_config("max_stream_buffer_size", 250))
        self.default_deadline_ms = coerce_timeout_ms(
            self.grpc_adapter_config.get("default_deadline_ms"),
            default=self.default_timeout_ms,
            minimum=1,
            maximum=300000,
        )

        self.authority = self._get_optional_string_config("authority")
        self.default_metadata = normalize_headers(
            ensure_mapping(self.grpc_adapter_config.get("default_metadata"), field_name="default_metadata", allow_none=True),
            lowercase=True,
        )

        self._parsed_endpoint: Optional[ParsedEndpoint] = None
        self._response_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_response_history_size)
        self._receive_buffer: Deque[Dict[str, Any]] = deque(maxlen=self.max_stream_buffer_size)
        self._last_method: Optional[str] = None
        self._last_call_type: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseAdapter protocol hooks
    # ------------------------------------------------------------------
    def _connect_impl(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self._ensure_transport_present(operation="connect")
        parsed = parse_endpoint(endpoint, default_scheme="grpc", protocol=self.protocol, require_host=True)
        self._parsed_endpoint = parsed
        connect_result = self._transport_call("connect", endpoint=parsed.normalized, timeout_ms=timeout_ms, metadata=metadata) or {}

        if self.healthcheck_on_connect and self.healthcheck_method:
            _ = self._invoke_rpc(
                method=self._normalize_method_name(self.healthcheck_method),
                call_type="unary_unary",
                request={"service": self.service_name or ""},
                timeout_ms=timeout_ms,
                metadata=metadata,
            )

        return {
            "endpoint": parsed.normalized,
            "host": parsed.host,
            "port": parsed.port,
            "scheme": parsed.scheme,
            "secure": parsed.secure or self.use_tls,
            "session_id": self.session.session_id,
            "healthcheck_on_connect": self.healthcheck_on_connect,
            "result": json_safe(connect_result),
            "metadata": normalize_metadata(metadata),
        }

    def _send_impl(
        self,
        *,
        payload: bytes,
        envelope: Mapping[str, Any],
        timeout_ms: int,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        self._ensure_transport_present(operation="send")
        method = self._resolve_rpc_method(envelope)
        call_type = self._resolve_call_type(envelope)
        rpc_metadata = self._build_rpc_metadata(metadata, envelope)
        request_payload = self._resolve_request_payload(payload, envelope)

        result = self._invoke_rpc(
            method=method,
            call_type=call_type,
            request=request_payload,
            timeout_ms=timeout_ms,
            metadata=rpc_metadata,
        )
        buffered_count = self._buffer_rpc_result(result, envelope=envelope, method=method, call_type=call_type)
        self._last_method = method
        self._last_call_type = call_type
        return {
            "invoked": True,
            "method": method,
            "call_type": call_type,
            "buffered_messages": buffered_count,
            "metadata": normalize_metadata(rpc_metadata),
        }

    def _receive_impl(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self.receive_from_buffer and self._receive_buffer:
            if self.consume_buffered_messages_on_recv:
                return self._receive_buffer.popleft()
            return self._receive_buffer[-1]

        self._ensure_transport_present(operation="receive")
        assert self.transport is not None
        if not hasattr(self.transport, "recv"):
            raise ReceiveFailureError(
                "No buffered gRPC message is available and the transport doesn't expose recv().",
                context={"operation": "receive", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
            )

        raw = self.transport.recv(timeout_ms=timeout_ms, metadata=metadata)
        return self._normalize_transport_message(raw, metadata=metadata, method=self._last_method, call_type=self._last_call_type)

    def _ack_impl(
        self,
        *,
        message_id: str,
        correlation_id: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        if self.ack_mode == "disabled":
            raise AdapterCapabilityError(
                "gRPC ack support is disabled by configuration.",
                context={"operation": "ack", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
            )
        if self.ack_mode == "synthetic":
            return {
                "acknowledged": True,
                "ack_mode": "synthetic",
                "message_id": message_id,
                "correlation_id": correlation_id,
                "metadata": normalize_metadata(metadata),
            }
        self._ensure_transport_present(operation="ack")
        assert self.transport is not None
        if not hasattr(self.transport, "ack"):
            raise AdapterCapabilityError(
                "Configured gRPC transport doesn't expose ack().",
                context={"operation": "ack", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
            )
        return self.transport.ack(message_id=message_id, correlation_id=correlation_id, metadata=metadata) or {
            "acknowledged": True,
            "ack_mode": "transport",
            "message_id": message_id,
            "correlation_id": correlation_id,
        }

    def _nack_impl(
        self,
        *,
        message_id: str,
        correlation_id: Optional[str],
        reason: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        if self.nack_mode == "disabled":
            raise AdapterCapabilityError(
                "gRPC nack support is disabled by configuration.",
                context={"operation": "nack", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
            )
        if self.nack_mode == "synthetic":
            return {
                "nacked": True,
                "nack_mode": "synthetic",
                "message_id": message_id,
                "correlation_id": correlation_id,
                "reason": reason,
                "metadata": normalize_metadata(metadata),
            }
        self._ensure_transport_present(operation="nack")
        assert self.transport is not None
        if not hasattr(self.transport, "nack"):
            raise AdapterCapabilityError(
                "Configured gRPC transport doesn't expose nack().",
                context={"operation": "nack", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
            )
        return self.transport.nack(message_id=message_id, correlation_id=correlation_id, reason=reason, metadata=metadata) or {
            "nacked": True,
            "nack_mode": "transport",
            "message_id": message_id,
            "correlation_id": correlation_id,
            "reason": reason,
        }

    def _close_impl(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        result: Mapping[str, Any] | None = None
        if self.transport is not None and hasattr(self.transport, "close"):
            result = self.transport.close(reason=reason, metadata=metadata)
        self._parsed_endpoint = None
        self._receive_buffer.clear()
        return {
            "closed": True,
            "endpoint": self.session.endpoint,
            "reason": reason,
            "result": json_safe(result),
            "metadata": normalize_metadata(metadata),
        }

    # ------------------------------------------------------------------
    # Adapter state and metadata helpers
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> AdapterCapabilities:
        capabilities = super().capabilities
        capabilities.supports_streaming = self._get_bool_config("supports_streaming", True)
        capabilities.supports_bidirectional_streaming = self._get_bool_config("supports_bidirectional_streaming", True)
        capabilities.supports_ack = self.ack_mode != "disabled"
        capabilities.supports_nack = self.nack_mode != "disabled"
        capabilities.supports_receive = True
        capabilities.supports_request_reply = True
        capabilities.supports_tls = True
        capabilities.auth_modes = tuple(self._get_sequence_config("auth_modes", self.DEFAULT_GRPC_AUTH_MODES))
        capabilities.content_types = tuple(self._get_sequence_config("content_types", self.DEFAULT_GRPC_CONTENT_TYPES))
        capabilities.metadata = merge_mappings(
            capabilities.metadata,
            normalize_metadata(self.grpc_adapter_config.get("capabilities_metadata")),
            {
                "transport_family": "grpc",
                "default_call_type": self.default_call_type,
                "service_name": self.service_name,
                "ack_mode": self.ack_mode,
                "nack_mode": self.nack_mode,
            },
        )
        return capabilities

    def set_transport(self, transport: GRPCTransportProtocol) -> None:
        self.transport = transport

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_transport_present(self, *, operation: str) -> None:
        if self.transport is None:
            raise AdapterInitializationError(
                "GRPCAdapter requires a transport bridge instance.",
                context={"operation": operation, "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
                details={"adapter_name": self.adapter_name},
            )

    def _transport_call(self, method_name: str, **kwargs: Any) -> Any:
        if self.transport is None or not hasattr(self.transport, method_name):
            raise AdapterCapabilityError(
                f"Configured gRPC transport doesn't expose {method_name}().",
                context={"operation": method_name, "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
            )
        transport_method = getattr(self.transport, method_name)
        try:
            return transport_method(**kwargs)
        except NetworkError:
            raise
        except Exception as exc:
            raise normalize_network_exception(
                exc,
                operation=method_name,
                endpoint=self.session.endpoint,
                channel=self.channel,
                protocol=self.protocol,
                session_id=self.session.session_id,
                metadata={"transport_method": method_name},
            ) from exc

    def _invoke_rpc(
        self,
        *,
        method: str,
        call_type: str,
        request: Any,
        timeout_ms: int,
        metadata: Mapping[str, Any],
    ) -> Any:
        if call_type == "unary_unary":
            return self._transport_call("unary_unary", method=method, request=request, timeout_ms=timeout_ms, metadata=metadata)
        if call_type == "unary_stream":
            return list(self._transport_call("unary_stream", method=method, request=request, timeout_ms=timeout_ms, metadata=metadata))
        if call_type == "stream_unary":
            sequence = ensure_sequence(request, field_name="request", allow_none=False, coerce_scalar=False)
            return self._transport_call("stream_unary", method=method, request=sequence, timeout_ms=timeout_ms, metadata=metadata)
        if call_type == "stream_stream":
            sequence = ensure_sequence(request, field_name="request", allow_none=False, coerce_scalar=False)
            return list(self._transport_call("stream_stream", method=method, request=sequence, timeout_ms=timeout_ms, metadata=metadata))
        raise AdapterCapabilityError(
            "Unsupported gRPC call type requested.",
            context={"operation": "grpc_invoke", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
            details={"call_type": call_type},
        )

    def _resolve_rpc_method(self, envelope: Mapping[str, Any]) -> str:
        metadata = ensure_mapping(envelope.get("metadata"), field_name="metadata", allow_none=True)
        method = metadata.get("grpc_method") or envelope.get("method") or self.default_method
        if method is None and self.require_method_on_send:
            raise AdapterInitializationError(
                "GRPCAdapter send() requires a gRPC method when require_method_on_send is enabled.",
                context={"operation": "send", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
            )
        if method is None:
            method = "UnknownService/UnknownMethod"
        return self._normalize_method_name(str(method))

    def _resolve_call_type(self, envelope: Mapping[str, Any]) -> str:
        metadata = ensure_mapping(envelope.get("metadata"), field_name="metadata", allow_none=True)
        candidate = metadata.get("grpc_call_type") or envelope.get("call_type") or self.default_call_type
        return self._get_call_type_config("grpc_call_type", str(candidate))

    def _build_rpc_metadata(self, metadata: Mapping[str, Any], envelope: Mapping[str, Any]) -> Dict[str, str]:
        envelope_metadata = ensure_mapping(envelope.get("metadata"), field_name="metadata", allow_none=True)
        grpc_metadata = ensure_mapping(envelope_metadata.get("grpc_metadata"), field_name="grpc_metadata", allow_none=True)
        merged = merge_mappings(self.default_metadata, grpc_metadata, metadata)
        return normalize_headers(merged, lowercase=True)

    def _resolve_request_payload(self, payload: bytes, envelope: Mapping[str, Any]) -> Any:
        original_payload = envelope.get("payload")
        if original_payload is None:
            return payload
        return original_payload

    def _buffer_rpc_result(
        self,
        result: Any,
        *,
        envelope: Mapping[str, Any],
        method: str,
        call_type: str,
    ) -> int:
        if result is None:
            return 0
        if call_type in {"unary_stream", "stream_stream"}:
            responses = list(result)
        else:
            responses = [result]

        for item in responses:
            self._receive_buffer.append(
                self._normalize_transport_message(
                    item,
                    metadata={"buffered": True},
                    method=method,
                    call_type=call_type,
                    correlation_id=str(envelope.get("correlation_id") or "") or None,
                )
            )
        return len(responses)

    def _normalize_transport_message(
        self,
        value: Any,
        *,
        metadata: Mapping[str, Any],
        method: Optional[str],
        call_type: Optional[str],
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            message_id = value.get("message_id") or generate_message_id(prefix="recv_grpc")
            resolved_correlation_id = value.get("correlation_id") or correlation_id or generate_correlation_id(prefix="corr_grpc")
            payload = value.get("payload", value)
            content_type = str(value.get("content_type") or infer_content_type(payload)).strip().lower()
            payload_size = estimate_payload_size(payload)
            snapshot = merge_mappings(
                value,
                {
                    "message_id": str(message_id),
                    "correlation_id": str(resolved_correlation_id),
                    "payload": payload,
                    "content_type": content_type,
                    "payload_size": payload_size,
                    "received_at": value.get("received_at") or utc_timestamp(),
                    "method": method,
                    "call_type": call_type,
                    "metadata": normalize_metadata(merge_mappings(value.get("metadata"), metadata)),
                },
            )
        else:
            payload = value
            content_type = infer_content_type(payload)
            payload_size = estimate_payload_size(payload)
            snapshot = {
                "message_id": generate_message_id(prefix="recv_grpc"),
                "correlation_id": correlation_id or generate_correlation_id(prefix="corr_grpc"),
                "payload": payload,
                "content_type": content_type,
                "payload_size": payload_size,
                "received_at": utc_timestamp(),
                "method": method,
                "call_type": call_type,
                "metadata": normalize_metadata(metadata),
            }

        self._response_history.append(json_safe(sanitize_for_logging(snapshot)))
        return snapshot

    def _normalize_method_name(self, method: str) -> str:
        candidate = ensure_non_empty_string(method, field_name="grpc_method")
        trimmed = candidate.strip().lstrip("/")
        if "/" not in trimmed:
            if self.service_name:
                trimmed = f"{self.service_name}/{trimmed}"
            elif self.strict_method_format:
                raise PayloadValidationError(
                    "gRPC method must be in 'Service/Method' format when strict_method_format is enabled.",
                    context={"operation": "grpc_method_normalization", "channel": self.channel, "protocol": self.protocol},
                    details={"method": candidate},
                )
        return f"/{trimmed}"

    def _get_call_type_config(self, name: str, default: str) -> str:
        value = str(self.grpc_adapter_config.get(name, default)).strip().lower() or default
        if value not in _VALID_GRPC_CALL_TYPES:
            raise NetworkConfigurationError(
                "Invalid gRPC call type configuration.",
                context={"operation": "grpc_adapter_config", "channel": self.channel, "protocol": self.protocol},
                details={"config_key": name, "config_value": value},
            )
        return value

    def _get_ack_mode_config(self, name: str, default: str) -> str:
        value = str(self.grpc_adapter_config.get(name, default)).strip().lower() or default
        if value not in _VALID_GRPC_ACK_MODES:
            raise NetworkConfigurationError(
                "Invalid gRPC ack/nack mode configuration.",
                context={"operation": "grpc_adapter_config", "channel": self.channel, "protocol": self.protocol},
                details={"config_key": name, "config_value": value},
            )
        return value

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.grpc_adapter_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _infer_protocol_from_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        if not endpoint:
            return None
        text = str(endpoint).strip().lower()
        if text.startswith("grpcs://"):
            return "grpc"
        if text.startswith("grpc://"):
            return "grpc"
        return None


class _DemoGRPCTransport:
    def __init__(self) -> None:
        self.connected = False
        self.endpoint: Optional[str] = None
        self.recv_queue: Deque[Any] = deque()

    def connect(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.connected = True
        self.endpoint = endpoint
        return {"connected": True, "endpoint": endpoint, "timeout_ms": timeout_ms, "metadata": dict(metadata)}

    def unary_unary(self, *, method: str, request: Any, timeout_ms: int, metadata: Mapping[str, Any]) -> Any:
        return {
            "method": method,
            "payload": {"echo": request, "status": "OK"},
            "content_type": "application/json",
            "metadata": dict(metadata),
            "timeout_ms": timeout_ms,
        }

    def unary_stream(self, *, method: str, request: Any, timeout_ms: int, metadata: Mapping[str, Any]) -> Iterable[Any]:
        return [
            {"method": method, "payload": {"chunk": 1, "request": request}, "content_type": "application/json", "metadata": dict(metadata)},
            {"method": method, "payload": {"chunk": 2, "request": request}, "content_type": "application/json", "metadata": dict(metadata)},
            {"method": method, "payload": {"chunk": 3, "request": request}, "content_type": "application/json", "metadata": dict(metadata)},
        ]

    def stream_unary(self, *, method: str, request: Sequence[Any], timeout_ms: int, metadata: Mapping[str, Any]) -> Any:
        return {
            "method": method,
            "payload": {"received": list(request), "status": "OK"},
            "content_type": "application/json",
            "metadata": dict(metadata),
            "timeout_ms": timeout_ms,
        }

    def stream_stream(self, *, method: str, request: Sequence[Any], timeout_ms: int, metadata: Mapping[str, Any]) -> Iterable[Any]:
        return [
            {"method": method, "payload": {"stream_item": item}, "content_type": "application/json", "metadata": dict(metadata)}
            for item in request
        ]

    def recv(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> Any:
        if not self.recv_queue:
            raise ReceiveFailureError(
                "Demo gRPC transport has no queued receive payload.",
                context={"operation": "receive", "channel": "grpc", "protocol": "grpc", "timeout_ms": timeout_ms},
            )
        return self.recv_queue.popleft()

    def ack(self, *, message_id: str, correlation_id: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return {"acknowledged": True, "message_id": message_id, "correlation_id": correlation_id, "metadata": dict(metadata)}

    def nack(self, *, message_id: str, correlation_id: Optional[str], reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return {"nacked": True, "message_id": message_id, "correlation_id": correlation_id, "reason": reason, "metadata": dict(metadata)}

    def close(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.connected = False
        return {"closed": True, "reason": reason, "metadata": dict(metadata)}


def _printer_status(label: str, message: str, level: str = "info") -> None:
    try:
        printer.status(label, message, level)
    except Exception:
        print(f"[{label}] {message}")


if __name__ == "__main__":
    print("\n=== Running gRPC Adapter ===\n")
    _printer_status("TEST", "gRPC Adapter initialized", "info")

    transport = _DemoGRPCTransport()
    adapter = GRPCAdapter(
        endpoint="grpc://127.0.0.1:50051",
        transport=transport,
        config={
            "service_name": "slai.network.NetworkService",
            "default_method": "Relay",
            "ack_mode": "synthetic",
            "nack_mode": "synthetic",
            "default_call_type": "unary_unary",
        },
    )

    capabilities = adapter.capabilities.to_dict()
    connected = adapter.connect(metadata={"env": "test", "region": "local"})
    unary_sent = adapter.send(
        {"task": "relay", "payload": {"hello": "grpc"}},
        envelope={"metadata": {"grpc_method": "slai.network.NetworkService/Relay", "grpc_call_type": "unary_unary"}},
        metadata={"tenant": "demo"},
    )
    unary_received = adapter.recv(metadata={"consumer": "unary"})
    stream_sent = adapter.send(
        {"task": "stream"},
        envelope={"metadata": {"grpc_method": "slai.network.NetworkService/StreamRelay", "grpc_call_type": "unary_stream"}},
        metadata={"tenant": "demo"},
    )
    stream_first = adapter.recv(metadata={"consumer": "stream"})
    stream_second = adapter.recv(metadata={"consumer": "stream"})
    acked = adapter.ack(unary_received["message_id"], correlation_id=unary_received["correlation_id"], metadata={"result": "ok"})
    nacked = adapter.nack("msg_grpc_demo", correlation_id="corr_grpc_demo", reason="synthetic retry path", metadata={"retryable": True})
    state_snapshot = adapter.get_state_snapshot()
    health_snapshot = adapter.get_health_snapshot()
    memory_health = adapter.memory.get_network_health()
    closed = adapter.close(reason="demo complete", metadata={"cleanup": True})

    print("Capabilities:", stable_json_dumps(capabilities))
    print("Connected:", stable_json_dumps(connected))
    print("Unary Sent:", stable_json_dumps(unary_sent))
    print("Unary Received:", stable_json_dumps(unary_received))
    print("Stream Sent:", stable_json_dumps(stream_sent))
    print("Stream First:", stable_json_dumps(stream_first))
    print("Stream Second:", stable_json_dumps(stream_second))
    print("Acked:", stable_json_dumps(acked))
    print("Nacked:", stable_json_dumps(nacked))
    print("State Snapshot:", stable_json_dumps(state_snapshot))
    print("Health Snapshot:", stable_json_dumps(health_snapshot))
    print("Memory Health:", stable_json_dumps(memory_health))
    print("Closed:", stable_json_dumps(closed))

    assert capabilities["supports_streaming"] is True
    assert connected["connected"] is True
    assert unary_sent["result"]["buffered_messages"] == 1
    assert unary_received["payload"]["echo"]["task"] == "relay"
    assert stream_sent["result"]["buffered_messages"] == 3
    assert stream_first["payload"]["chunk"] == 1
    assert stream_second["payload"]["chunk"] == 2
    assert acked["acknowledged"] is True
    assert nacked["nacked"] is True
    assert state_snapshot["session"]["connected"] is True
    assert adapter.memory.get("network.session.snapshot")
    assert adapter.memory.get("network.endpoint.health")

    _printer_status("TEST", "All gRPC Adapter checks passed", "info")
    print("\n=== Test ran successfully ===\n")
