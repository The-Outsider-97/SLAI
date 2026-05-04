"""
Queue adapter implementation for SLAI's Network Agent.

This module provides the production-grade queue transport adapter that
inherits from the shared BaseAdapter contract. It centralizes queue- and
broker-oriented behavior while relying on the base adapter for lifecycle
orchestration, structured memory updates, delivery bookkeeping, health
snapshots, and network-native error semantics.

The adapter is intentionally scoped to queue transport concerns:
- broker/session establishment for queue-backed transports,
- publish/consume flows with queue, exchange, and routing-key awareness,
- ack/nack and dead-letter coordination,
- prefetch, visibility timeout, and receive buffering,
- adapter-local message state useful to routing and observability,
- transport-bridge integration that avoids hard-coding a single queue client.

It does not own routing strategy, retry policy, circuit breaking, or policy
arbitration. Those belong to the higher-level network modules and agents.

Note:
    This adapter is implemented against a transport bridge interface rather
    than a direct broker library dependency. That keeps the module importable
    and testable in environments where AMQP/SQS/Kafka clients are not
    installed, while still giving SLAI a production-ready adapter contract to
    integrate with a real queue transport later.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..utils import *
from .base_adapter import BaseAdapter
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Queue Adapter")
printer = PrettyPrinter()

__all__ = ["QueueTransportProtocol", "QueueEnvelope", "QueueAdapter"]


_VALID_QUEUE_ACK_MODES = {"synthetic", "transport", "disabled"}
_QUEUE_SCHEMES = {"amqp", "amqps", "queue", "mq", "sqs", "pubsub", "kafka"}


@runtime_checkable
class QueueTransportProtocol(Protocol):
    def connect(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None: ...
    def declare(
        self,
        *,
        queue_name: str,
        durable: bool,
        exclusive: bool,
        auto_delete: bool,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None: ...
    def publish(
        self,
        *,
        queue_name: str,
        payload: bytes,
        timeout_ms: int,
        metadata: Mapping[str, Any],
        headers: Mapping[str, Any],
        message_id: str,
        correlation_id: Optional[str],
        exchange: Optional[str],
        routing_key: Optional[str],
        delivery_mode: str,
    ) -> Mapping[str, Any] | None: ...
    def consume(self, *,
        queue_name: str,
        timeout_ms: int,
        metadata: Mapping[str, Any],
        auto_ack: bool,
        visibility_timeout_ms: Optional[int],
        prefetch_count: int,
    ) -> Mapping[str, Any] | None: ...
    def ack(self, *, receipt: Mapping[str, Any], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None: ...
    def nack(self, *,
        receipt: Mapping[str, Any],
        requeue: bool,
        reason: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None: ...
    def close(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None: ...


@dataclass(slots=True)
class QueueEnvelope:
    """Normalized queue publication target and transport attributes."""

    queue_name: str
    exchange: Optional[str]
    routing_key: Optional[str]
    delivery_mode: str
    content_type: str
    headers: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "queue_name": self.queue_name,
            "exchange": self.exchange,
            "routing_key": self.routing_key,
            "delivery_mode": self.delivery_mode,
            "content_type": self.content_type,
            "headers": json_safe(self.headers),
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


class QueueAdapter(BaseAdapter):
    """
    Production-grade queue transport adapter.

    The adapter understands publication targets, delivery acknowledgements,
    receive buffering, and queue-centric metadata while delegating the actual
    broker operations to an injected transport bridge.
    """

    DEFAULT_QUEUE_CONTENT_TYPES: Tuple[str, ...] = (
        "application/json",
        "text/plain",
        "application/octet-stream",
    )
    DEFAULT_QUEUE_AUTH_MODES: Tuple[str, ...] = ("none", "basic", "token", "mtls")

    def __init__(self, *, memory=None,
        config: Optional[Mapping[str, Any]] = None,
        endpoint: Optional[str] = None,
        adapter_name: str = "Queue",
        protocol: Optional[str] = None,
        transport: Optional[QueueTransportProtocol] = None,
    ) -> None:
        provided_config = ensure_mapping(config, field_name="config", allow_none=True)
        section_config = get_config_section("network_queue_adapter") or {}
        merged_queue_config = merge_mappings(section_config, provided_config)
        self.queue_adapter_config = merged_queue_config
        self.ack_mode = str(merged_queue_config.get("ack_mode", "transport")).strip().lower() or "transport"
        self.nack_mode = str(merged_queue_config.get("nack_mode", self.ack_mode)).strip().lower() or self.ack_mode

        inferred_protocol = protocol or merged_queue_config.get("protocol") or self._infer_protocol_from_endpoint(endpoint) or "queue"

        super().__init__(
            adapter_name=adapter_name,
            protocol=inferred_protocol,
            channel="queue",
            memory=memory,
            config=merged_queue_config,
            endpoint=endpoint or merged_queue_config.get("endpoint"),
        )

        self.queue_adapter_config = merge_mappings(section_config, self.adapter_config)
        self.transport = transport or self.queue_adapter_config.get("transport")

        self.default_queue = self._get_optional_string_config("default_queue")
        self.dead_letter_queue = self._get_optional_string_config("dead_letter_queue")
        self.default_exchange = self._get_optional_string_config("default_exchange")
        self.default_routing_key = self._get_optional_string_config("default_routing_key")
        self.ack_mode = self._get_ack_mode_config("ack_mode", "transport")
        self.nack_mode = self._get_ack_mode_config("nack_mode", self.ack_mode)

        self.auto_declare = self._get_bool_config("auto_declare", True)
        self.durable = self._get_bool_config("durable", True)
        self.exclusive = self._get_bool_config("exclusive", False)
        self.auto_delete = self._get_bool_config("auto_delete", False)
        self.auto_ack_on_receive = self._get_bool_config("auto_ack_on_receive", False)
        self.receive_from_buffer = self._get_bool_config("receive_from_buffer", True)
        self.consume_buffered_messages_on_recv = self._get_bool_config("consume_buffered_messages_on_recv", True)
        self.capture_sent_messages = self._get_bool_config("capture_sent_messages", True)
        self.capture_received_messages = self._get_bool_config("capture_received_messages", True)
        self.requeue_on_nack = self._get_bool_config("requeue_on_nack", True)
        self.purge_on_close = self._get_bool_config("purge_on_close", False)

        self.prefetch_count = max(1, self._get_non_negative_int_config("prefetch_count", 10))
        self.visibility_timeout_ms = self._get_optional_timeout_ms_config("visibility_timeout_ms")
        self.poll_timeout_ms = coerce_timeout_ms(
            self.queue_adapter_config.get("poll_timeout_ms"),
            default=self.receive_timeout_ms,
            minimum=1,
            maximum=300000,
        )
        self.max_message_history_size = max(1, self._get_non_negative_int_config("max_message_history_size", 250))
        self.max_queued_inbound_messages = max(1, self._get_non_negative_int_config("max_queued_inbound_messages", 250))

        self.default_headers = normalize_headers(
            ensure_mapping(self.queue_adapter_config.get("default_headers"), field_name="default_headers", allow_none=True),
            lowercase=False,
            drop_none=True,
        )
        self.default_metadata = normalize_metadata(
            ensure_mapping(self.queue_adapter_config.get("default_metadata"), field_name="default_metadata", allow_none=True)
        )

        self._parsed_endpoint: Optional[ParsedEndpoint] = None
        self._declared_queues: set[str] = set()
        self._message_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_message_history_size)
        self._receive_buffer: Deque[Dict[str, Any]] = deque(maxlen=self.max_queued_inbound_messages)
        self._inflight_receipts: Dict[str, Dict[str, Any]] = {}
        self._last_queue_name: Optional[str] = None
        self._last_exchange: Optional[str] = None
        self._last_routing_key: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseAdapter protocol hooks
    # ------------------------------------------------------------------
    def _connect_impl(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self._ensure_transport_present(operation="connect")
        assert self.transport is not None
        parsed = parse_endpoint(
            endpoint,
            default_scheme=self._infer_endpoint_scheme(endpoint) or "amqp",
            protocol=self.protocol,
            require_host=True,
        )
        self._parsed_endpoint = parsed
        connect_result = self.transport.connect(
            endpoint=parsed.normalized,
            timeout_ms=timeout_ms,
            metadata=normalize_metadata(merge_mappings(self.default_metadata, metadata)),
        ) or {}

        if self.auto_declare and self.default_queue:
            self._ensure_declared(self.default_queue, metadata)
        if self.auto_declare and self.dead_letter_queue:
            self._ensure_declared(self.dead_letter_queue, metadata)

        return {
            "endpoint": parsed.normalized,
            "host": parsed.host,
            "port": parsed.port,
            "scheme": parsed.scheme,
            "queue": self.default_queue,
            "dead_letter_queue": self.dead_letter_queue,
            "prefetch_count": self.prefetch_count,
            "visibility_timeout_ms": self.visibility_timeout_ms,
            "session_id": self.session.session_id,
            "result": json_safe(connect_result),
            "metadata": normalize_metadata(metadata),
        }

    def _send_impl(self, *, payload: bytes,
        envelope: Mapping[str, Any],
        timeout_ms: int,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        self._ensure_transport_present(operation="send")
        assert self.transport is not None
        target = self._resolve_publication_target(envelope, metadata)
        if self.auto_declare:
            self._ensure_declared(target.queue_name, metadata)

        publish_result = self.transport.publish(
            queue_name=target.queue_name,
            payload=payload,
            timeout_ms=timeout_ms,
            metadata=normalize_metadata(merge_mappings(self.default_metadata, target.metadata, metadata)),
            headers=target.headers,
            message_id=str(envelope["message_id"]),
            correlation_id=str(envelope.get("correlation_id")) if envelope.get("correlation_id") is not None else None,
            exchange=target.exchange,
            routing_key=target.routing_key,
            delivery_mode=target.delivery_mode,
        ) or {}

        published_at = utc_timestamp()
        self._last_queue_name = target.queue_name
        self._last_exchange = target.exchange
        self._last_routing_key = target.routing_key

        sent_snapshot = {
            "direction": "outbound",
            "queue_name": target.queue_name,
            "exchange": target.exchange,
            "routing_key": target.routing_key,
            "message_id": envelope.get("message_id"),
            "correlation_id": envelope.get("correlation_id"),
            "content_type": target.content_type,
            "delivery_mode": target.delivery_mode,
            "published_at": published_at,
            "payload_size": len(payload),
            "headers": sanitize_for_logging(target.headers) if self.emit_log_safe_payloads else json_safe(target.headers),
            "metadata": sanitize_for_logging(metadata) if self.emit_log_safe_payloads else json_safe(metadata),
            "result": sanitize_for_logging(publish_result) if self.emit_log_safe_payloads else json_safe(publish_result),
        }
        if self.capture_sent_messages:
            self._message_history.append(json_safe(sent_snapshot))

        return {
            "published": True,
            "queue_name": target.queue_name,
            "exchange": target.exchange,
            "routing_key": target.routing_key,
            "delivery_mode": target.delivery_mode,
            "message_id": envelope.get("message_id"),
            "correlation_id": envelope.get("correlation_id"),
            "published_at": published_at,
            "result": json_safe(publish_result),
        }

    def _receive_impl(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self.receive_from_buffer and self._receive_buffer:
            if self.consume_buffered_messages_on_recv:
                return self._receive_buffer.popleft()
            return self._receive_buffer[-1]

        self._ensure_transport_present(operation="receive")
        assert self.transport is not None
        queue_name = self._resolve_consume_queue(metadata)
        if self.auto_declare:
            self._ensure_declared(queue_name, metadata)

        raw = self.transport.consume(
            queue_name=queue_name,
            timeout_ms=timeout_ms,
            metadata=normalize_metadata(merge_mappings(self.default_metadata, metadata)),
            auto_ack=self.auto_ack_on_receive,
            visibility_timeout_ms=self.visibility_timeout_ms,
            prefetch_count=self.prefetch_count,
        )
        if raw is None:
            raise ReceiveFailureError(
                "No queue message available for receive.",
                context={"operation": "receive", "endpoint": self.session.endpoint, "channel": self.channel, "protocol": self.protocol},
                details={"queue_name": queue_name, "timeout_ms": timeout_ms},
            )

        normalized_message = self._normalize_transport_message(raw, queue_name=queue_name, metadata=metadata)
        if self.capture_received_messages:
            self._message_history.append(
                json_safe(
                    {
                        "direction": "inbound",
                        "queue_name": normalized_message.get("queue_name"),
                        "message_id": normalized_message.get("message_id"),
                        "correlation_id": normalized_message.get("correlation_id"),
                        "delivery_tag": normalized_message.get("delivery_tag"),
                        "received_at": normalized_message.get("received_at"),
                        "redelivered": normalized_message.get("redelivered"),
                        "payload_size": estimate_payload_size(normalized_message.get("payload")),
                    }
                )
            )

        if not self.auto_ack_on_receive:
            self._inflight_receipts[str(normalized_message["message_id"])] = json_safe(normalized_message)

        return normalized_message

    def _ack_impl(self, *,
        message_id: str,
        correlation_id: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        if self.ack_mode == "disabled":
            raise AdapterCapabilityError(
                f"{self.adapter_name} ack() is disabled by configuration.",
                context={"operation": "ack", "channel": self.channel, "protocol": self.protocol},
            )

        receipt = self._resolve_inflight_receipt(message_id=message_id, correlation_id=correlation_id)
        if self.ack_mode == "synthetic":
            receipt["acked_at"] = utc_timestamp()
            receipt["acknowledged"] = True
            self._inflight_receipts.pop(message_id, None)
            return {
                "acknowledged": True,
                "mode": "synthetic",
                "message_id": message_id,
                "correlation_id": correlation_id,
                "queue_name": receipt.get("queue_name"),
                "delivery_tag": receipt.get("delivery_tag"),
            }

        self._ensure_transport_present(operation="ack")
        assert self.transport is not None
        result = self.transport.ack(receipt=receipt, metadata=normalize_metadata(merge_mappings(self.default_metadata, metadata))) or {}
        self._inflight_receipts.pop(message_id, None)
        return {
            "acknowledged": True,
            "mode": "transport",
            "message_id": message_id,
            "correlation_id": correlation_id,
            "queue_name": receipt.get("queue_name"),
            "delivery_tag": receipt.get("delivery_tag"),
            "result": json_safe(result),
        }

    def _nack_impl(self, *,
        message_id: str,
        correlation_id: Optional[str],
        reason: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        if self.nack_mode == "disabled":
            raise AdapterCapabilityError(
                f"{self.adapter_name} nack() is disabled by configuration.",
                context={"operation": "nack", "channel": self.channel, "protocol": self.protocol},
            )

        receipt = self._resolve_inflight_receipt(message_id=message_id, correlation_id=correlation_id)
        queue_name = str(receipt.get("queue_name") or self.default_queue or "")
        if self.nack_mode == "synthetic":
            self._inflight_receipts.pop(message_id, None)
            synthetic_result = {
                "nacked": True,
                "mode": "synthetic",
                "message_id": message_id,
                "correlation_id": correlation_id,
                "queue_name": queue_name,
                "requeue": self.requeue_on_nack,
                "reason": reason,
            }
            if self.requeue_on_nack:
                self._receive_buffer.append(receipt)
            elif self.dead_letter_queue:
                synthetic_result["dead_letter_queue"] = self.dead_letter_queue
            return synthetic_result

        self._ensure_transport_present(operation="nack")
        assert self.transport is not None
        result = self.transport.nack(
            receipt=receipt,
            requeue=self.requeue_on_nack,
            reason=reason,
            metadata=normalize_metadata(
                merge_mappings(self.default_metadata, metadata, {"dead_letter_queue": self.dead_letter_queue})
            ),
        ) or {}
        self._inflight_receipts.pop(message_id, None)
        return {
            "nacked": True,
            "mode": "transport",
            "message_id": message_id,
            "correlation_id": correlation_id,
            "queue_name": queue_name,
            "requeue": self.requeue_on_nack,
            "reason": reason,
            "dead_letter_queue": self.dead_letter_queue,
            "result": json_safe(result),
        }

    def _close_impl(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        result: Dict[str, Any] = {}
        if self.transport is not None:
            result = json_safe(self.transport.close(reason=reason, metadata=normalize_metadata(merge_mappings(self.default_metadata, metadata))) or {})

        if self.purge_on_close:
            self._receive_buffer.clear()
            self._inflight_receipts.clear()
        self._declared_queues.clear()

        return {
            "closed": True,
            "reason": reason,
            "purged": self.purge_on_close,
            "result": result,
        }

    # ------------------------------------------------------------------
    # Adapter state helpers
    # ------------------------------------------------------------------
    def get_state_snapshot(self) -> Dict[str, Any]:
        base_snapshot = super().get_state_snapshot()
        base_snapshot["queue"] = {
            "default_queue": self.default_queue,
            "dead_letter_queue": self.dead_letter_queue,
            "default_exchange": self.default_exchange,
            "default_routing_key": self.default_routing_key,
            "prefetch_count": self.prefetch_count,
            "visibility_timeout_ms": self.visibility_timeout_ms,
            "ack_mode": self.ack_mode,
            "nack_mode": self.nack_mode,
            "declared_queues": sorted(self._declared_queues),
            "buffered_messages": len(self._receive_buffer),
            "inflight_messages": len(self._inflight_receipts),
            "last_queue_name": self._last_queue_name,
            "last_exchange": self._last_exchange,
            "last_routing_key": self._last_routing_key,
            "message_history_size": len(self._message_history),
        }
        return base_snapshot

    # ------------------------------------------------------------------
    # Internal queue-specific logic
    # ------------------------------------------------------------------
    def _resolve_publication_target(
        self,
        envelope: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> QueueEnvelope:
        queue_name = (
            envelope.get("queue")
            or envelope.get("queue_name")
            or envelope.get("destination")
            or metadata.get("queue")
            or self.default_queue
        )
        if queue_name is None:
            raise NoRouteAvailableError(
                "QueueAdapter send() requires a destination queue but none was resolved.",
                context={"operation": "send", "channel": self.channel, "protocol": self.protocol, "endpoint": self.session.endpoint},
                details={"adapter_name": self.adapter_name},
            )

        normalized_queue = ensure_non_empty_string(str(queue_name), field_name="queue_name")
        exchange = envelope.get("exchange") or metadata.get("exchange") or self.default_exchange
        routing_key = envelope.get("routing_key") or metadata.get("routing_key") or self.default_routing_key or normalized_queue
        content_type = infer_content_type(envelope.get("payload"), explicit_content_type=envelope.get("content_type"))
        headers = normalize_headers(
            merge_mappings(self.default_headers, ensure_mapping(envelope.get("headers"), field_name="headers", allow_none=True)),
            lowercase=False,
            drop_none=True,
        )
        delivery_mode = "persistent" if self.durable else "transient"

        return QueueEnvelope(
            queue_name=normalized_queue,
            exchange=str(exchange).strip() if exchange is not None and str(exchange).strip() else None,
            routing_key=str(routing_key).strip() if routing_key is not None and str(routing_key).strip() else None,
            delivery_mode=delivery_mode,
            content_type=content_type,
            headers=headers,
            metadata=normalize_metadata(metadata),
        )

    def _resolve_consume_queue(self, metadata: Mapping[str, Any]) -> str:
        queue_name = metadata.get("queue") or self.default_queue or self._last_queue_name
        if queue_name is None:
            raise NoRouteAvailableError(
                "QueueAdapter recv() requires a queue name but none was resolved.",
                context={"operation": "receive", "channel": self.channel, "protocol": self.protocol, "endpoint": self.session.endpoint},
                details={"adapter_name": self.adapter_name},
            )
        return ensure_non_empty_string(str(queue_name), field_name="queue_name")

    def _normalize_transport_message(
        self,
        raw: Mapping[str, Any],
        *,
        queue_name: str,
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        incoming = ensure_mapping(raw, field_name="transport_message")
        payload = incoming.get("payload")
        content_type = infer_content_type(payload, explicit_content_type=incoming.get("content_type"))

        if payload is None:
            payload = b""
        elif not isinstance(payload, (bytes, bytearray, memoryview, str)):
            payload = serialize_payload(payload, content_type=content_type, max_payload_bytes=self.max_receive_payload_bytes)

        message_id = incoming.get("message_id") or generate_message_id(prefix=f"queue_{self.adapter_name.lower()}")
        correlation_id = incoming.get("correlation_id") or generate_correlation_id(prefix=f"queue_{self.adapter_name.lower()}")
        normalized = {
            "message_id": str(message_id),
            "correlation_id": str(correlation_id),
            "payload": payload,
            "content_type": content_type,
            "queue_name": str(incoming.get("queue_name") or queue_name),
            "exchange": incoming.get("exchange"),
            "routing_key": incoming.get("routing_key"),
            "headers": normalize_headers(
                ensure_mapping(incoming.get("headers"), field_name="headers", allow_none=True),
                lowercase=False,
                drop_none=True,
            ),
            "received_at": utc_timestamp(),
            "delivery_tag": incoming.get("delivery_tag"),
            "receipt_handle": incoming.get("receipt_handle"),
            "redelivered": bool(incoming.get("redelivered", False)),
            "metadata": normalize_metadata(merge_mappings(incoming.get("metadata"), metadata)),
        }
        return normalized

    def _ensure_declared(self, queue_name: str, metadata: Mapping[str, Any]) -> None:
        normalized_queue = ensure_non_empty_string(queue_name, field_name="queue_name")
        if normalized_queue in self._declared_queues:
            return
        self._ensure_transport_present(operation="declare")
        assert self.transport is not None
        if not hasattr(self.transport, "declare"):
            self._declared_queues.add(normalized_queue)
            return
        self.transport.declare(
            queue_name=normalized_queue,
            durable=self.durable,
            exclusive=self.exclusive,
            auto_delete=self.auto_delete,
            metadata=normalize_metadata(merge_mappings(self.default_metadata, metadata)),
        )
        self._declared_queues.add(normalized_queue)

    def _resolve_inflight_receipt(self, *, message_id: str, correlation_id: Optional[str]) -> Dict[str, Any]:
        receipt = self._inflight_receipts.get(message_id)
        if receipt is not None:
            return receipt
        if correlation_id:
            for candidate in self._inflight_receipts.values():
                if str(candidate.get("correlation_id")) == str(correlation_id):
                    return candidate
        raise AcknowledgementError(
            "No in-flight queue receipt found for message acknowledgement.",
            context={"operation": "acknowledgement_lookup", "channel": self.channel, "protocol": self.protocol, "endpoint": self.session.endpoint},
            details={"message_id": message_id, "correlation_id": correlation_id},
        )

    def _ensure_transport_present(self, *, operation: str) -> None:
        if self.transport is None:
            raise AdapterInitializationError(
                "Queue transport bridge is not configured.",
                context={"operation": operation, "channel": self.channel, "protocol": self.protocol, "endpoint": self.session.endpoint},
                details={"adapter_name": self.adapter_name},
            )

    def _infer_protocol_from_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        if not endpoint:
            return None
        text = str(endpoint).strip().lower()
        if "://" not in text:
            return None
        scheme = text.split("://", 1)[0]
        return "queue" if scheme in _QUEUE_SCHEMES else None

    def _infer_endpoint_scheme(self, endpoint: Optional[str]) -> Optional[str]:
        if not endpoint:
            return None
        text = str(endpoint).strip().lower()
        if "://" not in text:
            return None
        scheme = text.split("://", 1)[0]
        return scheme if scheme in _QUEUE_SCHEMES else None

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.queue_adapter_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_ack_mode_config(self, name: str, default: str) -> str:
        value = str(self.queue_adapter_config.get(name, default)).strip().lower() or default
        if value not in _VALID_QUEUE_ACK_MODES:
            raise NetworkConfigurationError(
                "Invalid queue adapter acknowledgement mode configuration.",
                context={"operation": "queue_adapter_config", "channel": self.channel, "protocol": self.protocol},
                details={"config_key": name, "config_value": value, "allowed_values": sorted(_VALID_QUEUE_ACK_MODES)},
            )
        return value

    def _get_optional_timeout_ms_config(self, name: str) -> Optional[int]:
        value = self.queue_adapter_config.get(name)
        if value in (None, "", 0, "0"):
            return None
        return coerce_timeout_ms(value, default=self.default_timeout_ms, minimum=1, maximum=300000)


class _InMemoryQueueTransport:
    """In-file demo transport used for contract validation and smoke testing."""

    def __init__(self) -> None:
        self.connected = False
        self.endpoint: Optional[str] = None
        self.queues: Dict[str, Deque[Dict[str, Any]]] = {}
        self.inflight: Dict[str, Dict[str, Any]] = {}
        self.delivery_counter = 0

    def connect(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.connected = True
        self.endpoint = endpoint
        return {"connected": True, "endpoint": endpoint, "timeout_ms": timeout_ms, "metadata": dict(metadata)}

    def declare(
        self,
        *,
        queue_name: str,
        durable: bool,
        exclusive: bool,
        auto_delete: bool,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        self.queues.setdefault(queue_name, deque())
        return {
            "declared": True,
            "queue_name": queue_name,
            "durable": durable,
            "exclusive": exclusive,
            "auto_delete": auto_delete,
            "metadata": dict(metadata),
        }

    def publish(
        self,
        *,
        queue_name: str,
        payload: bytes,
        timeout_ms: int,
        metadata: Mapping[str, Any],
        headers: Mapping[str, Any],
        message_id: str,
        correlation_id: Optional[str],
        exchange: Optional[str],
        routing_key: Optional[str],
        delivery_mode: str,
    ) -> Mapping[str, Any] | None:
        self.queues.setdefault(queue_name, deque())
        self.delivery_counter += 1
        delivery_tag = f"delivery_{self.delivery_counter}"
        self.queues[queue_name].append(
            {
                "message_id": message_id,
                "correlation_id": correlation_id,
                "payload": payload,
                "queue_name": queue_name,
                "exchange": exchange,
                "routing_key": routing_key,
                "delivery_tag": delivery_tag,
                "headers": dict(headers),
                "metadata": dict(metadata),
                "content_type": headers.get("content-type") or headers.get("Content-Type") or "application/json",
                "redelivered": False,
                "delivery_mode": delivery_mode,
                "timeout_ms": timeout_ms,
            }
        )
        return {"published": True, "queue_depth": len(self.queues[queue_name]), "delivery_tag": delivery_tag}

    def consume(
        self,
        *,
        queue_name: str,
        timeout_ms: int,
        metadata: Mapping[str, Any],
        auto_ack: bool,
        visibility_timeout_ms: Optional[int],
        prefetch_count: int,
    ) -> Mapping[str, Any] | None:
        queue = self.queues.setdefault(queue_name, deque())
        if not queue:
            return None
        message = dict(queue.popleft())
        message["timeout_ms"] = timeout_ms
        message["prefetch_count"] = prefetch_count
        message["visibility_timeout_ms"] = visibility_timeout_ms
        message["metadata"] = merge_mappings(message.get("metadata"), metadata)
        if not auto_ack:
            self.inflight[str(message["message_id"])] = dict(message)
        return message

    def ack(self, *, receipt: Mapping[str, Any], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        message_id = str(receipt.get("message_id"))
        self.inflight.pop(message_id, None)
        return {"acknowledged": True, "message_id": message_id, "metadata": dict(metadata)}

    def nack(
        self,
        *,
        receipt: Mapping[str, Any],
        requeue: bool,
        reason: Optional[str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        message_id = str(receipt.get("message_id"))
        message = self.inflight.pop(message_id, dict(receipt))
        if requeue:
            queue_name = str(message.get("queue_name"))
            message["redelivered"] = True
            self.queues.setdefault(queue_name, deque()).appendleft(message)
        return {"nacked": True, "message_id": message_id, "requeue": requeue, "reason": reason, "metadata": dict(metadata)}

    def close(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.connected = False
        return {"closed": True, "reason": reason, "metadata": dict(metadata)}


if __name__ == "__main__":
    print("\n=== Running Queue Adapter ===\n")
    printer.status("TEST", "Queue Adapter initialized", "info")

    transport = _InMemoryQueueTransport()
    adapter = QueueAdapter(
        endpoint="amqp://broker.internal:5672/vhost",
        transport=transport,
        config={
            "default_queue": "jobs.primary",
            "dead_letter_queue": "jobs.primary.dlq",
            "default_exchange": "jobs.exchange",
            "default_routing_key": "jobs.primary",
            "ack_mode": "transport",
            "nack_mode": "transport",
            "auto_declare": True,
            "durable": True,
        },
    )

    capabilities = adapter.capabilities.to_dict()
    printer.status("TEST", "Capabilities resolved", "info")

    connected = adapter.connect(metadata={"region": "eu-west", "env": "test"})
    printer.status("TEST", "Broker connection established", "info")

    sent = adapter.send(
        {"task": "relay", "payload": {"hello": "queue"}},
        envelope={
            "queue": "jobs.primary",
            "headers": {"content-type": "application/json", "x-priority": "normal"},
            "route": "primary",
        },
        metadata={"publisher": "demo"},
    )
    printer.status("TEST", "Payload published", "info")

    received = adapter.recv(metadata={"consumer": "demo"})
    printer.status("TEST", "Payload consumed", "info")

    acked = adapter.ack(received["message_id"], correlation_id=received["correlation_id"], metadata={"result": "ok"})
    printer.status("TEST", "Payload acknowledged", "info")

    sent_retry = adapter.send(
        {"task": "retry-path", "payload": {"hello": "again"}},
        envelope={"queue": "jobs.primary", "headers": {"content-type": "application/json"}},
        metadata={"publisher": "demo"},
    )
    printer.status("TEST", "Second payload published", "info")

    received_retry = adapter.recv(metadata={"consumer": "demo"})
    nacked = adapter.nack(
        received_retry["message_id"],
        correlation_id=received_retry["correlation_id"],
        reason="synthetic retry path",
        metadata={"retryable": True},
    )
    printer.status("TEST", "Payload negatively acknowledged", "info")

    if nacked.get("requeue", False):
        requeued = adapter.recv(metadata={"consumer": "demo-retry"})
        printer.status("TEST", "Requeued payload consumed", "info")
        requeued_ack = adapter.ack(requeued["message_id"], correlation_id=requeued["correlation_id"], metadata={"result": "retried"})
        printer.status("TEST", "Requeued payload acknowledged", "info")
    else:
        requeued = None
        requeued_ack = None

    state_snapshot = adapter.get_state_snapshot()
    health_snapshot = adapter.get_health_snapshot()
    memory_health = adapter.memory.get_network_health()

    closed = adapter.close(reason="demo complete", metadata={"cleanup": True})
    printer.status("TEST", "Adapter closed", "info")

    print("Capabilities:", stable_json_dumps(capabilities))
    print("Connected:", stable_json_dumps(connected))
    print("Sent:", stable_json_dumps(sent))
    print("Received:", stable_json_dumps(received))
    print("Acked:", stable_json_dumps(acked))
    print("Sent Retry:", stable_json_dumps(sent_retry))
    print("Nacked:", stable_json_dumps(nacked))
    print("Requeued:", stable_json_dumps(requeued))
    print("Requeued Ack:", stable_json_dumps(requeued_ack))
    print("State Snapshot:", stable_json_dumps(state_snapshot))
    print("Health Snapshot:", stable_json_dumps(health_snapshot))
    print("Memory Health:", stable_json_dumps(memory_health))
    print("Closed:", stable_json_dumps(closed))

    assert capabilities["supports_ack"] is True
    assert connected["session_id"] == adapter.session.session_id
    assert sent["result"]["published"] is True
    assert received["payload"]["task"] == "relay"
    assert acked["acknowledged"] is True
    assert nacked["nacked"] is True
    assert state_snapshot["queue"]["default_queue"] == "jobs.primary"
    assert adapter.memory.get("network.session.snapshot")
    assert adapter.memory.get("network.endpoint.health")
    assert adapter.memory.get("network.delivery.state")
    if requeued is not None:
        assert requeued["payload"]["task"] == "retry-path"
        assert requeued_ack["acknowledged"] is True

    printer.status("TEST", "All Queue Adapter checks passed", "info")
    print("\n=== Test ran successfully ===\n")
