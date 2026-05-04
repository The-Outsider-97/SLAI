"""
WebSocket adapter implementation for SLAI's Network Agent.

This module provides the production-grade WebSocket transport adapter that
inherits from the shared BaseAdapter contract. It keeps WebSocket-specific
behavior in one place while relying on the base adapter for lifecycle
orchestration, structured memory updates, delivery bookkeeping, health
snapshots, and network-native error semantics.

The adapter is intentionally scoped to WebSocket transport concerns:
- session establishment for ws:// and wss:// endpoints,
- bidirectional message transmission,
- frame-mode and content-type aware send/receive handling,
- configurable TLS posture for secure sockets,
- application-level synthetic or on-wire ack/nack semantics,
- ping support and receive buffering,
- adapter-local message state useful to routing and observability.

It does not own routing strategy, retry policy, circuit breaking, or policy
arbitration. Those belong to the higher-level network modules and agents.
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import websocket
import websockets

from collections import deque
from threading import Event, Thread
from time import monotonic, sleep
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

from ..utils import *
from .base_adapter import BaseAdapter, AdapterCapabilities
from ....utils.buffer.network_buffer import NetworkBuffer
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("WebSocket Adapter")
printer = PrettyPrinter()


_VALID_WS_SCHEMES = {"ws", "wss", "websocket"}
_VALID_ACK_MODES = {"synthetic", "frame", "disabled"}


class WebSocketAdapter(BaseAdapter):
    """
    Production-grade WebSocket transport adapter.

    The adapter supports long-lived bidirectional sessions, text/binary frame
    handling, JSON-aware receive normalization, configurable TLS posture, and
    optional application-level ack/nack signaling.
    """

    DEFAULT_WEBSOCKET_CONTENT_TYPES: Tuple[str, ...] = (
        "application/json",
        "text/plain",
        "application/octet-stream",
    )
    DEFAULT_WEBSOCKET_AUTH_MODES: Tuple[str, ...] = ("none", "bearer", "custom_header", "cookie")

    def __init__(self, *, memory=None, config: Optional[Mapping[str, Any]] = None,
                 endpoint: Optional[str] = None, adapter_name: str = "WebSocket",
                 protocol: Optional[str] = None) -> None:
        provided_config = ensure_mapping(config, field_name="config", allow_none=True)
        section_config = get_config_section("network_websocket_adapter") or {}
        merged_ws_config = merge_mappings(section_config, provided_config)
        self.websocket_adapter_config = merged_ws_config
        self.ack_mode = str(merged_ws_config.get("ack_mode", "synthetic")).strip().lower() or "synthetic"
        self.nack_mode = str(merged_ws_config.get("nack_mode", self.ack_mode)).strip().lower() or self.ack_mode
        self.subprotocols = tuple(
            str(item).strip()
            for item in ensure_sequence(merged_ws_config.get("subprotocols"), field_name="subprotocols", allow_none=True, coerce_scalar=True)
            if str(item).strip()
        )

        inferred_protocol = protocol or merged_ws_config.get("protocol") or self._infer_protocol_from_endpoint(endpoint) or "websocket"

        super().__init__(
            adapter_name=adapter_name,
            protocol=inferred_protocol,
            channel="websocket",
            memory=memory,
            config=merged_ws_config,
            endpoint=endpoint or merged_ws_config.get("endpoint"),
        )

        self.websocket_adapter_config = merge_mappings(section_config, self.adapter_config)

        self.verify_tls = self._get_bool_config("verify_tls", True)
        self.allow_insecure_tls = self._get_bool_config("allow_insecure_tls", False)
        self.ping_on_connect = self._get_bool_config("ping_on_connect", False)
        self.auto_detect_json_messages = self._get_bool_config("auto_detect_json_messages", True)
        self.reconnect_on_send_if_closed = self._get_bool_config("reconnect_on_send_if_closed", True)
        self.reconnect_on_receive_if_closed = self._get_bool_config("reconnect_on_receive_if_closed", False)
        self.capture_sent_messages = self._get_bool_config("capture_sent_messages", True)
        self.capture_received_messages = self._get_bool_config("capture_received_messages", True)
        self.strict_subprotocol_match = self._get_bool_config("strict_subprotocol_match", False)
        self.include_origin_header = self._get_bool_config("include_origin_header", False)

        self.default_send_mode = self._get_send_mode_config("default_send_mode", "auto")
        self.ack_mode = self._get_ack_mode_config("ack_mode", "synthetic")
        self.nack_mode = self._get_ack_mode_config("nack_mode", self.ack_mode)

        self.ping_timeout_ms = coerce_timeout_ms(
            self.websocket_adapter_config.get("ping_timeout_ms"),
            default=self.default_timeout_ms,
            minimum=1,
            maximum=300000,
        )
        self.close_timeout_seconds = max(1, self._get_non_negative_int_config("close_timeout_seconds", 3))
        self.close_status_code = max(1000, self._get_non_negative_int_config("close_status_code", 1000))
        self.max_message_history_size = max(1, self._get_non_negative_int_config("max_message_history_size", 200))
        self.max_queued_inbound_messages = max(1, self._get_non_negative_int_config("max_queued_inbound_messages", 200))

        self.origin = self._get_optional_string_config("origin")
        self.host_header = self._get_optional_string_config("host_header")
        self.ping_payload = self._get_optional_string_config("ping_payload") or "ping"

        self.subprotocols = self._get_sequence_config("subprotocols", ())
        self.extra_headers = normalize_headers(
            ensure_mapping(self.websocket_adapter_config.get("extra_headers"), field_name="extra_headers", allow_none=True),
            lowercase=False,
        )
        self.ssl_options = ensure_mapping(
            self.websocket_adapter_config.get("ssl_options"),
            field_name="ssl_options",
            allow_none=True,
        )

        self._ws: Optional[websocket.WebSocket] = None
        self._parsed_endpoint: Optional[ParsedEndpoint] = None
        self._connected_url: Optional[str] = None
        self._buffer = NetworkBuffer()
        self._last_subprotocol: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseAdapter protocol hooks
    # ------------------------------------------------------------------
    def _connect_impl(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        parsed = parse_endpoint(endpoint, default_scheme="ws", protocol=self.protocol, require_host=True)
        ws_url = self._render_websocket_url(parsed)
        header_lines = self._build_connection_headers(metadata)
        ssl_options = self._build_ssl_options(parsed)

        self._reset_socket()
        self._ws = websocket.create_connection(
            ws_url,
            timeout=timeout_ms / 1000.0,
            header=header_lines or None,
            subprotocols=list(self.subprotocols) or None,
            host=self.host_header,
            origin=self.origin if self.include_origin_header else None,
            sslopt=ssl_options if ws_url.startswith("wss://") else None,
            enable_multithread=True,
        )
        self._ws.settimeout(timeout_ms / 1000.0)
        self._parsed_endpoint = parsed
        self._connected_url = ws_url
        self._last_subprotocol = self._resolve_subprotocol()

        if self.strict_subprotocol_match and self.subprotocols and self._last_subprotocol not in set(self.subprotocols):
            raise ProtocolNegotiationError(
                "WebSocket connection established without an allowed subprotocol.",
                context={"operation": "connect", "endpoint": ws_url, "protocol": self.protocol, "channel": self.channel},
                details={"requested_subprotocols": list(self.subprotocols), "resolved_subprotocol": self._last_subprotocol},
            )

        if self.ping_on_connect:
            self._ws.ping(self.ping_payload.encode("utf-8"))

        return {
            "endpoint": ws_url,
            "session_id": self.session.session_id,
            "subprotocol": self._last_subprotocol,
            "secure": ws_url.startswith("wss://"),
            "verify_tls": self.verify_tls,
            "subprotocols": list(self.subprotocols),
            "metadata": normalize_metadata(metadata),
        }

    def _send_impl(self, *, payload: bytes, envelope: Mapping[str, Any],
                   timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self._ws is None or not self._socket_is_connected():
            if self.reconnect_on_send_if_closed:
                self._reconnect(timeout_ms=timeout_ms, metadata=metadata)
            else:
                raise SessionClosedError(
                    "WebSocket session is closed and reconnect_on_send_if_closed is disabled.",
                    context={"operation": "send", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
                )

        outbound_payload, opcode, frame_mode = self._prepare_outbound_frame(payload, envelope)
        bytes_sent = self._ws.send(outbound_payload, opcode=opcode)
        sent_at = utc_timestamp()

        return {
            "queued": True,
            "bytes_sent": bytes_sent,
            "frame_mode": frame_mode,
            "content_type": envelope.get("content_type"),
            "sent_at": sent_at,
            "history_depth": self._buffer.snapshot()["size"]
        }

    def _receive_impl(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        # 1. Try to dequeue from buffer – return as‑is, no re‑normalization
        if self._buffer.snapshot()["size"] > 0:
            messages = self._buffer.dequeue(max_items=1)
            if messages:
                msg = messages[0]
                # Restore content_type from metadata if missing
                if "content_type" not in msg and "metadata" in msg:
                    msg["content_type"] = msg["metadata"].get("content_type", "application/octet-stream")
                # Merge call metadata
                msg["metadata"] = normalize_metadata(
                    merge_mappings(msg.get("metadata"), metadata)
                )
                return msg
    
        # 2. Buffer empty – read from WebSocket
        if self._ws is None or not self._socket_is_connected():
            if self.reconnect_on_receive_if_closed:
                self._reconnect(timeout_ms=timeout_ms, metadata=metadata)
            else:
                raise SessionClosedError(
                    "WebSocket session is closed and reconnect_on_receive_if_closed is disabled.",
                    context={"operation": "receive", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
                )

        self._ws.settimeout(timeout_ms / 1000.0)
        try:
            frame = self._ws.recv()
        except websocket.WebSocketTimeoutException as exc:
            raise DeliveryTimeoutError(
                "Timed out while waiting for a WebSocket message.",
                context={
                    "operation": "receive",
                    "endpoint": self.session.endpoint,
                    "channel": self.channel,
                    "protocol": self.protocol,
                    "timeout_ms": timeout_ms,
                },
                cause=exc,
            ) from exc
        except websocket.WebSocketConnectionClosedException as exc:
            raise SessionClosedError(
                "WebSocket connection closed during receive().",
                context={"operation": "receive", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
                cause=exc,
            ) from exc

        # 3. Normalize the frame ONCE
        normalized = self._frame_to_message(frame, metadata)
    
        # 4. Enqueue into buffer
        enqueue_result = self._buffer.enqueue(
            payload=normalized["payload"],
            channel=self.channel,
            protocol=self.protocol,
            fairness_key=metadata.get("fairness_key", self.session.session_id),
            priority=metadata.get("priority", 0),
            ttl_seconds=metadata.get("ttl_seconds"),
            metadata={
                **normalized,   # includes content_type, message_id, correlation_id, received_at
                "raw_type": type(frame).__name__
            },
            message_id=normalized["message_id"]
        )
    
        if not enqueue_result["admitted"]:
            self._buffer.telemetry.increment("websocket_buffer_rejected")
            return None
    
        # 5. Dequeue the message we just enqueued (should be the same)
        messages = self._buffer.dequeue(max_items=1)
        if not messages:
            return None
        msg = messages[0]

        # Restore content_type from metadata if missing
        if "content_type" not in msg and "metadata" in msg:
            msg["content_type"] = msg["metadata"].get("content_type", "application/octet-stream")

        # Merge call metadata (though already present, ensures consistency)
        msg["metadata"] = normalize_metadata(
            merge_mappings(msg.get("metadata"), metadata)
        )
        return msg

    def _ack_impl(
        self,
        *,
        message_id: str,
        correlation_id: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        if self.ack_mode == "disabled":
            raise AdapterCapabilityError(
                "WebSocket ack support is disabled by configuration.",
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
        payload = {
            "type": "ack",
            "message_id": message_id,
            "correlation_id": correlation_id,
            "metadata": normalize_metadata(metadata),
            "sent_at": utc_timestamp(),
        }
        self._send_control_payload(payload)
        return {"acknowledged": True, "ack_mode": "frame", "message_id": message_id, "correlation_id": correlation_id}

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
                "WebSocket nack support is disabled by configuration.",
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
        payload = {
            "type": "nack",
            "message_id": message_id,
            "correlation_id": correlation_id,
            "reason": reason,
            "metadata": normalize_metadata(metadata),
            "sent_at": utc_timestamp(),
        }
        self._send_control_payload(payload)
        return {
            "nacked": True,
            "nack_mode": "frame",
            "message_id": message_id,
            "correlation_id": correlation_id,
            "reason": reason,
        }

    def _close_impl(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        closed_endpoint = self._connected_url or self.session.endpoint
        if self._ws is not None:
            try:
                self._ws.close(
                    status=self.close_status_code,
                    reason=(reason or "").encode("utf-8"),
                    timeout=self.close_timeout_seconds,
                )
            finally:
                self._ws = None
        self._parsed_endpoint = None
        self._connected_url = None
        self._last_subprotocol = None
        self._buffer.clear()
        return {
            "closed": True,
            "endpoint": closed_endpoint,
            "reason": reason,
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
        capabilities.supports_headers = True
        capabilities.supports_tls = True
        capabilities.supports_receive = True
        capabilities.supports_request_reply = False
        capabilities.auth_modes = tuple(self._get_sequence_config("auth_modes", self.DEFAULT_WEBSOCKET_AUTH_MODES))
        capabilities.content_types = tuple(self._get_sequence_config("content_types", self.DEFAULT_WEBSOCKET_CONTENT_TYPES))
        capabilities.metadata = merge_mappings(
            capabilities.metadata,
            normalize_metadata(self.websocket_adapter_config.get("capabilities_metadata")),
            {
                "transport_family": "websocket",
                "subprotocols": list(self.subprotocols),
                "ack_mode": self.ack_mode,
                "nack_mode": self.nack_mode,
            },
        )
        return capabilities

    def ping(self, *, payload: Optional[str] = None, timeout_ms: Optional[Any] = None) -> Dict[str, Any]:
        with self._lock:
            self._ensure_connected(operation="ping")
            resolved_timeout = coerce_timeout_ms(timeout_ms, default=self.ping_timeout_ms)
            try:
                self._ws.settimeout(resolved_timeout / 1000.0)
                ping_payload = (payload or self.ping_payload).encode("utf-8")
                started_at = monotonic()
                self._ws.ping(ping_payload)
                latency_ms = round((monotonic() - started_at) * 1000.0, 3)
                self._mark_activity(latency_ms=latency_ms)
                self._sync_health_memory(metadata={"ping": True, "latency_ms": latency_ms})
                return {
                    "adapter_name": self.adapter_name,
                    "protocol": self.protocol,
                    "channel": self.channel,
                    "endpoint": self.session.endpoint,
                    "pinged": True,
                    "latency_ms": latency_ms,
                    "payload": payload or self.ping_payload,
                }
            except Exception as exc:
                normalized = normalize_network_exception(
                    exc,
                    operation="ping",
                    endpoint=self.session.endpoint,
                    channel=self.channel,
                    protocol=self.protocol,
                    session_id=self.session.session_id,
                )
                self._handle_operation_failure(normalized, operation="ping", endpoint=self.session.endpoint)
                raise normalized from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _frame_to_message(self, frame: Any, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        received_at = utc_timestamp()
        if isinstance(frame, bytes):
            payload = frame
            content_type = "application/octet-stream"
            payload_size = len(frame)
        elif isinstance(frame, str):
            payload = frame
            payload_size = len(frame.encode("utf-8"))
            content_type = self._infer_inbound_content_type(frame)
        else:
            payload = json_safe(frame)
            payload_size = estimate_payload_size(payload)
            content_type = "application/json"

        return {
            "message_id": generate_message_id(prefix="recv_websocket"),
            "correlation_id": generate_correlation_id(prefix="corr_websocket"),
            "payload": payload,
            "content_type": content_type,
            "payload_size": payload_size,
            "received_at": received_at,
            "metadata": normalize_metadata(metadata),
        }

    def _prepare_outbound_frame(self, payload: bytes, envelope: Mapping[str, Any]) -> Tuple[bytes | str, int, str]:
        content_type = str(envelope.get("content_type") or "application/octet-stream").strip().lower()
        send_mode = self.default_send_mode
        if send_mode == "auto":
            if content_type == "application/octet-stream":
                send_mode = "binary"
            else:
                send_mode = "text"

        if send_mode == "binary":
            return payload, websocket.ABNF.OPCODE_BINARY, "binary"

        try:
            text_payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            return payload, websocket.ABNF.OPCODE_BINARY, "binary"
        return text_payload, websocket.ABNF.OPCODE_TEXT, "text"

    def _send_control_payload(self, payload: Mapping[str, Any]) -> None:
        if self._ws is None or not self._socket_is_connected():
            raise SessionClosedError(
                "Cannot send control payload over a closed WebSocket session.",
                context={"operation": "control_send", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
            )
        encoded = stable_json_dumps(payload)
        self._ws.send(encoded, opcode=websocket.ABNF.OPCODE_TEXT)

    def _reset_socket(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close(timeout=self.close_timeout_seconds)
            except Exception:
                pass
            finally:
                self._ws = None

    def _reconnect(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> None:
        if not self.session.endpoint:
            raise SessionUnavailableError(
                "Cannot reconnect WebSocket adapter without a known endpoint.",
                context={"operation": "reconnect", "channel": self.channel, "protocol": self.protocol},
            )
        self._connect_impl(endpoint=self.session.endpoint, timeout_ms=timeout_ms, metadata=metadata)
        self.session.connected = True
        self.session.state = "connected"
        self.health.connected = True
        self.health.status = "healthy"

    def _socket_is_connected(self) -> bool:
        return bool(self._ws is not None and getattr(self._ws, "connected", False))

    def _render_websocket_url(self, parsed: ParsedEndpoint) -> str:
        scheme = parsed.scheme.lower()
        if scheme == "websocket":
            scheme = "wss" if self.verify_tls else "ws"
        if scheme not in _VALID_WS_SCHEMES:
            raise ProtocolNegotiationError(
                "WebSocketAdapter only supports ws://, wss://, or websocket:// endpoints.",
                context={"operation": "connect", "endpoint": parsed.raw, "protocol": self.protocol},
                details={"scheme": parsed.scheme},
            )
        normalized_scheme = "wss" if scheme == "wss" else "ws"
        return urlunparse((normalized_scheme, parsed.netloc, parsed.path or "/", "", parsed.query or "", parsed.fragment or ""))

    def _build_connection_headers(self, metadata: Mapping[str, Any]) -> Sequence[str]:
        metadata_headers = ensure_mapping(metadata.get("headers"), field_name="headers", allow_none=True)
        headers = normalize_headers(merge_mappings(self.extra_headers, metadata_headers), lowercase=False)
        return [f"{key}: {value}" for key, value in headers.items()]

    def _build_ssl_options(self, parsed: ParsedEndpoint) -> Dict[str, Any]:
        if parsed.scheme.lower() not in {"wss", "websocket"}:
            return {}
        sslopt = dict(self.ssl_options)
        if not self.verify_tls or self.allow_insecure_tls:
            sslopt.setdefault("cert_reqs", ssl.CERT_NONE)
            sslopt.setdefault("check_hostname", False)
        else:
            sslopt.setdefault("cert_reqs", ssl.CERT_REQUIRED)
            sslopt.setdefault("check_hostname", True)
        return sslopt

    def _resolve_subprotocol(self) -> Optional[str]:
        if self._ws is None:
            return None
        try:
            return self._ws.getsubprotocol() or getattr(self._ws, "subprotocol", None)
        except Exception:
            return getattr(self._ws, "subprotocol", None)

    def _infer_inbound_content_type(self, payload: str) -> str:
        if not self.auto_detect_json_messages:
            return "text/plain"
        text = payload.strip()
        if not text:
            return "text/plain"
        if text.startswith("{") or text.startswith("["):
            try:
                json.loads(text)
                return "application/json"
            except Exception:
                return "text/plain"
        return "text/plain"

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.websocket_adapter_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_send_mode_config(self, name: str, default: str) -> str:
        value = str(self.websocket_adapter_config.get(name, default)).strip().lower() or default
        if value not in {"auto", "text", "binary"}:
            raise NetworkConfigurationError(
                "Invalid WebSocket send mode configuration.",
                context={"operation": "websocket_adapter_config", "channel": self.channel, "protocol": self.protocol},
                details={"config_key": name, "config_value": value},
            )
        return value

    def _get_ack_mode_config(self, name: str, default: str) -> str:
        value = str(self.websocket_adapter_config.get(name, default)).strip().lower() or default
        if value not in _VALID_ACK_MODES:
            raise NetworkConfigurationError(
                "Invalid WebSocket ack/nack mode configuration.",
                context={"operation": "websocket_adapter_config", "channel": self.channel, "protocol": self.protocol},
                details={"config_key": name, "config_value": value},
            )
        return value

    def _infer_protocol_from_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        if not endpoint:
            return None
        text = str(endpoint).strip().lower()
        if text.startswith("wss://"):
            return "websocket"
        if text.startswith("ws://"):
            return "websocket"
        return None


class _EchoServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._ready = Event()
        self._stop = Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._thread = Thread(target=self._run, daemon=True)

    async def _handler(self, conn) -> None:
        async for message in conn:
            await conn.send(message)

    async def _serve(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port, ping_interval=None):
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.05)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("WebSocket echo server did not start in time.")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    print("\n=== Running WebSocket Adapter ===\n")
    printer.status("TEST", "WebSocket Adapter initialized", "info")

    port = _find_free_port()
    server = _EchoServer("127.0.0.1", port)
    server.start()
    sleep(0.2)

    adapter = WebSocketAdapter(
        endpoint=f"ws://127.0.0.1:{port}/echo",
        config={
            "ping_on_connect": False,
            "ack_mode": "synthetic",
            "nack_mode": "synthetic",
        },
    )

    try:
        capabilities = adapter.capabilities.to_dict()
        connected = adapter.connect(metadata={"env": "test", "region": "local"})
        sent = adapter.send({"kind": "echo", "value": "hello websocket"}, metadata={"trace": "ws-demo"})
        received = adapter.recv(metadata={"consumer": "demo"})
        acked = adapter.ack(received["message_id"], correlation_id=received["correlation_id"], metadata={"result": "ok"})
        nacked = adapter.nack("msg_ws_demo", correlation_id="corr_ws_demo", reason="synthetic test", metadata={"retryable": True})
        pinged = adapter.ping(payload="health-check")
        state_snapshot = adapter.get_state_snapshot()
        health_snapshot = adapter.get_health_snapshot()
        memory_health = adapter.memory.get_network_health()
        closed = adapter.close(reason="demo complete", metadata={"cleanup": True})

        print("Capabilities:", stable_json_dumps(capabilities))
        print("Connected:", stable_json_dumps(connected))
        print("Sent:", stable_json_dumps(sent))
        print("Received:", stable_json_dumps(received))
        print("Acked:", stable_json_dumps(acked))
        print("Nacked:", stable_json_dumps(nacked))
        print("Pinged:", stable_json_dumps(pinged))
        print("State Snapshot:", stable_json_dumps(state_snapshot))
        print("Health Snapshot:", stable_json_dumps(health_snapshot))
        print("Memory Health:", stable_json_dumps(memory_health))
        print("Closed:", stable_json_dumps(closed))

        assert capabilities["supports_streaming"] is True
        assert connected["connected"] is True
        assert sent["payload_size"] > 0
        assert received["payload"]["kind"] == "echo"
        assert acked["acknowledged"] is True
        assert nacked["nacked"] is True
        assert pinged["pinged"] is True
        assert state_snapshot["session"]["connected"] is True
        assert adapter.memory.get("network.session.snapshot")
        assert adapter.memory.get("network.endpoint.health")

        printer.status("TEST", "All WebSocket Adapter checks passed", "info")
        print("\n=== Test ran successfully ===\n")
    finally:
        try:
            if adapter.is_connected():
                adapter.close(reason="cleanup")
        except Exception:
            pass
        server.stop()
