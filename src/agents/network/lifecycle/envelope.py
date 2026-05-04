"""
Canonical envelope management for SLAI's Network Agent lifecycle subsystem.

This module provides the production-grade envelope layer that sits beneath
NetworkLifecycle and alongside DeliveryStateMachine. It owns canonical message
envelope normalization so the broader network stack can construct, validate,
clone, serialize, and inspect envelopes through one consistent contract before
those envelopes are handed to routing, policy, reliability, and transport
execution.

The envelope layer is intentionally scoped to envelope ownership. It is
responsible for:
- canonical message envelope normalization and identity generation,
- correlation-id and idempotency-key enforcement,
- payload metadata, content-type, and transport serialization views,
- envelope validation and expiry/TTL normalization,
- retry, response, and dead-letter envelope derivation,
- JSON-safe snapshots synchronized into NetworkMemory.

It does not own delivery-state transitions, routing decisions, retry timing,
or transport execution. Those concerns belong to DeliveryStateMachine, routing,
reliability, and the specialized adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..utils import *
from ..network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Envelope")
printer = PrettyPrinter()


_ENVELOPE_LAST_KEY = "network.lifecycle.envelope.last"
_ENVELOPE_SNAPSHOT_KEY = "network.lifecycle.envelope.snapshot"
_ENVELOPE_HISTORY_KEY = "network.lifecycle.envelope.history"
_ENVELOPE_CACHE_KEY = "network.lifecycle.envelope.cache"

_DEFAULT_DELIVERY_MODES = ("best_effort", "at_least_once", "exactly_once")


@dataclass(slots=True)
class EnvelopeTransportView:
    """Transport-ready serialization view derived from a canonical envelope."""

    message_id: str
    correlation_id: str
    idempotency_key: str
    content_type: str
    payload_bytes: bytes
    payload_size: int
    headers: Dict[str, str]
    endpoint: Optional[str] = None
    route: Optional[str] = None
    timeout_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_payload_preview: bool = False, preview_limit: int = 128) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "content_type": self.content_type,
            "payload_size": self.payload_size,
            "headers": json_safe(self.headers),
            "endpoint": self.endpoint,
            "route": self.route,
            "timeout_ms": self.timeout_ms,
            "metadata": json_safe(self.metadata),
        }
        if include_payload_preview:
            payload["payload_preview"] = json_safe(self.payload_bytes[: max(0, int(preview_limit))])
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class EnvelopeRecord:
    """Canonical in-memory envelope representation."""

    message_id: str
    correlation_id: str
    idempotency_key: str
    channel: str
    protocol: str
    payload: Any
    content_type: str
    payload_size: int
    payload_fingerprint: str
    created_at: str
    delivery_mode: str = "at_least_once"
    endpoint: Optional[str] = None
    route: Optional[str] = None
    operation: Optional[str] = None
    timeout_ms: Optional[int] = None
    expires_at: Optional[str] = None
    priority: Optional[int] = None
    reply_to: Optional[str] = None
    dead_letter_queue: Optional[str] = None
    parent_message_id: Optional[str] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            return _parse_timestamp(self.expires_at) <= _utcnow()
        except Exception:
            return False

    def to_dict(self, *, include_payload: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "channel": self.channel,
            "protocol": self.protocol,
            "content_type": self.content_type,
            "payload_size": self.payload_size,
            "payload_fingerprint": self.payload_fingerprint,
            "created_at": self.created_at,
            "delivery_mode": self.delivery_mode,
            "endpoint": self.endpoint,
            "route": self.route,
            "operation": self.operation,
            "timeout_ms": self.timeout_ms,
            "expires_at": self.expires_at,
            "expired": self.is_expired(),
            "priority": self.priority,
            "reply_to": self.reply_to,
            "dead_letter_queue": self.dead_letter_queue,
            "parent_message_id": self.parent_message_id,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "headers": json_safe(self.headers),
            "metadata": json_safe(self.metadata),
        }
        if include_payload:
            payload["payload"] = json_safe(self.payload)
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


class EnvelopeManager:
    """
    Canonical envelope manager for the network lifecycle domain.

    The manager wraps the lower-level helper primitives with stronger lifecycle
    validation, richer envelope fields, envelope derivation flows, and shared
    memory synchronization. It is intended to be the authoritative place for
    envelope shaping before handoff to routing, delivery tracking, and
    transport adapters.
    """

    def __init__(self, memory: Optional[NetworkMemory] = None, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.envelope_config = merge_mappings(
            get_config_section("network_envelope") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_memory_snapshots = self._get_bool_config("record_memory_snapshots", True)
        self.record_history = self._get_bool_config("record_history", True)
        self.normalize_endpoints = self._get_bool_config("normalize_endpoints", True)
        self.require_correlation_id = self._get_bool_config("require_correlation_id", True)
        self.require_idempotency_key = self._get_bool_config("require_idempotency_key", True)
        self.require_channel = self._get_bool_config("require_channel", True)
        self.require_protocol = self._get_bool_config("require_protocol", True)
        self.require_content_type = self._get_bool_config("require_content_type", True)
        self.include_payload_fingerprint = self._get_bool_config("include_payload_fingerprint", True)
        self.preserve_payload_in_memory = self._get_bool_config("preserve_payload_in_memory", True)
        self.enforce_known_delivery_modes = self._get_bool_config("enforce_known_delivery_modes", True)

        self.default_protocol = normalize_protocol_name(self._get_optional_string_config("default_protocol") or "http")
        self.default_channel = normalize_channel_name(self._get_optional_string_config("default_channel") or self.default_protocol)
        self.default_delivery_mode = ensure_non_empty_string(
            str(self.envelope_config.get("default_delivery_mode", "at_least_once")),
            field_name="default_delivery_mode",
        ).lower()
        self.default_operation = self._get_optional_string_config("default_operation")
        self.default_reply_channel = self._get_optional_string_config("default_reply_channel")
        self.default_dead_letter_queue = self._get_optional_string_config("default_dead_letter_queue")

        self.default_timeout_ms = coerce_timeout_ms(
            self.envelope_config.get("default_timeout_ms"),
            default=5000,
            minimum=1,
            maximum=300000,
        )
        self.default_expires_in_seconds = self._get_non_negative_int_config("default_expires_in_seconds", 0)
        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 7200)
        self.max_envelopes = max(1, self._get_non_negative_int_config("max_envelopes", 5000))
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))
        self.max_payload_bytes = self._get_non_negative_int_config("max_payload_bytes", 10 * 1024 * 1024)
        self.response_timeout_ms = coerce_timeout_ms(
            self.envelope_config.get("response_timeout_ms"),
            default=self.default_timeout_ms,
            minimum=1,
            maximum=300000,
        )

        self._records: Dict[str, EnvelopeRecord] = {}
        self._history: List[Dict[str, Any]] = []
        self._stats: Dict[str, int] = {
            "builds": 0,
            "validations": 0,
            "transport_views": 0,
            "response_derivations": 0,
            "retry_derivations": 0,
            "dead_letter_derivations": 0,
            "deletes": 0,
        }
        self._started_at = utc_timestamp()

        self._sync_snapshot_memory()

    def build(
        self,
        envelope: Optional[Mapping[str, Any]] = None,
        *,
        payload: Any = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        endpoint: Optional[str] = None,
        route: Optional[str] = None,
        operation: Optional[str] = None,
        timeout_ms: Optional[Any] = None,
        delivery_mode: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
        priority: Optional[int] = None,
        reply_to: Optional[str] = None,
        dead_letter_queue: Optional[str] = None,
        headers: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError("EnvelopeManager is disabled by configuration.", context={"operation": "build_envelope"})

        base = ensure_mapping(envelope, field_name="envelope", allow_none=True)
        built = build_message_envelope(
            base,
            payload=payload,
            channel=channel or self.default_channel,
            protocol=protocol or self.default_protocol,
            endpoint=self._normalize_endpoint(endpoint or base.get("endpoint")),
            route=route,
            message_id=message_id,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            headers=headers,
            metadata=metadata,
            timeout_ms=timeout_ms or self.default_timeout_ms,
            delivery_mode=delivery_mode or self.default_delivery_mode,
            content_type=content_type,
        )
        built["operation"] = operation or base.get("operation") or self.default_operation
        built["priority"] = int(priority) if priority is not None else (int(base["priority"]) if base.get("priority") is not None else None)
        built["reply_to"] = reply_to or base.get("reply_to") or self.default_reply_channel
        built["dead_letter_queue"] = dead_letter_queue or base.get("dead_letter_queue") or self.default_dead_letter_queue
        built["parent_message_id"] = parent_message_id or base.get("parent_message_id")
        built["trace_id"] = trace_id or base.get("trace_id")
        built["session_id"] = session_id or base.get("session_id")
        built["created_at"] = str(base.get("created_at") or utc_timestamp())

        if expires_in_seconds is not None:
            built["expires_at"] = (_utcnow() + timedelta(seconds=max(0, int(expires_in_seconds)))).isoformat()
        elif self.default_expires_in_seconds > 0 and built.get("expires_at") is None:
            built["expires_at"] = (_utcnow() + timedelta(seconds=self.default_expires_in_seconds)).isoformat()

        record = self._record_from_mapping(built)
        validated = self.validate(record.to_dict())
        transport_preview = self.prepare_transport_view(validated).to_dict()

        with self._lock:
            self._ensure_capacity_locked(incoming_key=record.message_id)
            self._records[record.message_id] = self._record_from_mapping(validated)
            self._stats["builds"] += 1
            self._append_history_locked("build", validated)
            self._sync_envelope_memory(validated)
            self._sync_snapshot_memory()

        return merge_mappings(validated, {"transport": transport_preview})

    def normalize(self, envelope: Mapping[str, Any]) -> Dict[str, Any]:
        return self.build(envelope=envelope)

    def validate(self, envelope: Mapping[str, Any], *, strict: bool = True) -> Dict[str, Any]:
        self._stats["validations"] += 1
        record = self._record_from_mapping(envelope)
        if self.require_channel and not record.channel:
            raise PayloadValidationError("Envelope channel is required.", context={"operation": "validate_envelope", "message_id": record.message_id})
        if self.require_protocol and not record.protocol:
            raise PayloadValidationError("Envelope protocol is required.", context={"operation": "validate_envelope", "message_id": record.message_id})
        if self.require_correlation_id and not record.correlation_id:
            raise PayloadValidationError("Envelope correlation_id is required.", context={"operation": "validate_envelope", "message_id": record.message_id})
        if self.require_idempotency_key and not record.idempotency_key:
            raise PayloadValidationError("Envelope idempotency_key is required.", context={"operation": "validate_envelope", "message_id": record.message_id})
        if self.require_content_type and not record.content_type:
            raise PayloadValidationError("Envelope content_type is required.", context={"operation": "validate_envelope", "message_id": record.message_id})
        if self.enforce_known_delivery_modes and record.delivery_mode not in set(_DEFAULT_DELIVERY_MODES):
            raise PayloadValidationError(
                "Envelope delivery_mode is not recognized.",
                context={"operation": "validate_envelope", "message_id": record.message_id},
                details={"delivery_mode": record.delivery_mode, "allowed": list(_DEFAULT_DELIVERY_MODES)},
            )
        if strict and record.is_expired():
            raise DeliveryExpiredError(
                "Envelope has already expired.",
                context={
                    "operation": "validate_envelope",
                    "message_id": record.message_id,
                    "correlation_id": record.correlation_id,
                    "endpoint": record.endpoint,
                    "channel": record.channel,
                    "protocol": record.protocol,
                },
            )
        return record.to_dict()

    def prepare_transport_view(self, envelope: Mapping[str, Any]) -> EnvelopeTransportView:
        envelope_map = self.validate(envelope, strict=False)
        payload_bytes = coerce_payload_bytes(
            envelope_map.get("payload"),
            content_type=envelope_map.get("content_type"),
            max_payload_bytes=self.max_payload_bytes,
        )
        self._stats["transport_views"] += 1
        return EnvelopeTransportView(
            message_id=str(envelope_map["message_id"]),
            correlation_id=str(envelope_map["correlation_id"]),
            idempotency_key=str(envelope_map["idempotency_key"]),
            content_type=str(envelope_map["content_type"]),
            payload_bytes=payload_bytes,
            payload_size=len(payload_bytes),
            headers=normalize_headers(envelope_map.get("headers")),
            endpoint=self._normalize_endpoint(envelope_map.get("endpoint")),
            route=str(envelope_map.get("route")) if envelope_map.get("route") is not None else None,
            timeout_ms=int(envelope_map["timeout_ms"]) if envelope_map.get("timeout_ms") is not None else None,
            metadata=normalize_metadata(envelope_map.get("metadata")),
        )

    def derive_response(
        self,
        source_envelope: Mapping[str, Any],
        *,
        payload: Any = None,
        endpoint: Optional[str] = None,
        route: Optional[str] = None,
        content_type: Optional[str] = None,
        headers: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        timeout_ms: Optional[Any] = None,
        preserve_correlation_id: bool = True,
    ) -> Dict[str, Any]:
        source = self.validate(source_envelope, strict=False)
        result = self.build(
            payload=payload,
            channel=source.get("reply_to") or source.get("channel") or self.default_reply_channel or self.default_channel,
            protocol=source.get("protocol"),
            endpoint=endpoint or source.get("reply_to") or source.get("endpoint"),
            route=route or source.get("route"),
            operation="response",
            timeout_ms=timeout_ms or self.response_timeout_ms,
            delivery_mode=source.get("delivery_mode"),
            headers=merge_mappings(source.get("headers"), headers),
            metadata=merge_mappings(
                source.get("metadata"),
                normalize_metadata(metadata),
                {
                    "response_to_message_id": source["message_id"],
                    "response_to_correlation_id": source["correlation_id"],
                },
            ),
            correlation_id=source["correlation_id"] if preserve_correlation_id else None,
            parent_message_id=source["message_id"],
            trace_id=source.get("trace_id"),
            session_id=source.get("session_id"),
            content_type=content_type,
        )
        with self._lock:
            self._stats["response_derivations"] += 1
        return result

    def clone_for_retry(
        self,
        source_envelope: Mapping[str, Any],
        *,
        preserve_message_id: bool = True,
        preserve_correlation_id: bool = True,
        timeout_ms: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        route: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        source = self.validate(source_envelope, strict=False)
        source_metadata = ensure_mapping(source.get("metadata"), field_name="metadata", allow_none=True)
        retry_attempt = int(source_metadata.get("retry_attempt", 0) or 0) + 1
        result = self.build(
            envelope=source,
            payload=source.get("payload"),
            channel=source.get("channel"),
            protocol=source.get("protocol"),
            endpoint=endpoint or source.get("endpoint"),
            route=route or source.get("route"),
            operation=source.get("operation"),
            timeout_ms=timeout_ms or source.get("timeout_ms") or self.default_timeout_ms,
            delivery_mode=source.get("delivery_mode"),
            headers=source.get("headers"),
            metadata=merge_mappings(
                source_metadata,
                normalize_metadata(metadata),
                {
                    "retry_attempt": retry_attempt,
                    "retry_of_message_id": source["message_id"],
                },
            ),
            message_id=source["message_id"] if preserve_message_id else None,
            correlation_id=source["correlation_id"] if preserve_correlation_id else None,
            idempotency_key=source["idempotency_key"],
            parent_message_id=source.get("parent_message_id") or source["message_id"],
            trace_id=source.get("trace_id"),
            session_id=source.get("session_id"),
            content_type=source.get("content_type"),
        )
        with self._lock:
            self._stats["retry_derivations"] += 1
        return result

    def derive_dead_letter(
        self,
        source_envelope: Mapping[str, Any],
        error: BaseException | Mapping[str, Any],
        *,
        dead_letter_queue: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        source = self.validate(source_envelope, strict=False)
        error_snapshot = self._normalize_error(
            error,
            message_id=str(source["message_id"]),
            correlation_id=str(source["correlation_id"]),
            endpoint=source.get("endpoint"),
            channel=source.get("channel"),
            protocol=source.get("protocol"),
        )
        result = self.build(
            payload=source.get("payload"),
            channel=source.get("channel"),
            protocol=source.get("protocol"),
            endpoint=source.get("endpoint"),
            route=source.get("route"),
            operation="dead_letter",
            timeout_ms=source.get("timeout_ms"),
            delivery_mode=source.get("delivery_mode"),
            headers=source.get("headers"),
            metadata=merge_mappings(
                source.get("metadata"),
                normalize_metadata(metadata),
                {
                    "dead_letter_of_message_id": source["message_id"],
                    "dead_letter_of_correlation_id": source["correlation_id"],
                    "last_error": error_snapshot,
                },
            ),
            correlation_id=source["correlation_id"],
            parent_message_id=source["message_id"],
            trace_id=source.get("trace_id"),
            session_id=source.get("session_id"),
            dead_letter_queue=dead_letter_queue or source.get("dead_letter_queue") or self.default_dead_letter_queue,
            content_type=source.get("content_type"),
        )
        with self._lock:
            self._stats["dead_letter_derivations"] += 1
        return result

    def get(self, message_id: str, *, include_payload: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        normalized_message_id = ensure_non_empty_string(str(message_id), field_name="message_id")
        with self._lock:
            record = self._records.get(normalized_message_id)
            if record is None:
                return None
            preserve = self.preserve_payload_in_memory if include_payload is None else bool(include_payload)
            return record.to_dict(include_payload=preserve)

    def delete(self, message_id: str) -> bool:
        normalized_message_id = ensure_non_empty_string(str(message_id), field_name="message_id")
        with self._lock:
            removed = self._records.pop(normalized_message_id, None)
            if removed is None:
                return False
            self._stats["deletes"] += 1
            if self.record_memory_snapshots:
                self.memory.delete(self._message_key(normalized_message_id))
            self._sync_snapshot_memory()
            return True

    def list_envelopes(self, *, include_payload: Optional[bool] = None) -> List[Dict[str, Any]]:
        with self._lock:
            preserve = self.preserve_payload_in_memory if include_payload is None else bool(include_payload)
            items = [record.to_dict(include_payload=preserve) for record in self._records.values()]
            items.sort(key=lambda item: (str(item.get("created_at", "")), str(item.get("message_id", ""))))
            return items

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "started_at": self._started_at,
                "stats": dict(self._stats),
                "envelope_count": len(self._records),
                "history_size": len(self._history),
                "messages": sorted(list(self._records.keys())),
            }

    def _record_from_mapping(self, envelope: Mapping[str, Any]) -> EnvelopeRecord:
        envelope_map = ensure_mapping(envelope, field_name="envelope")
        payload = envelope_map.get("payload")
        content_type = str(envelope_map.get("content_type") or infer_content_type(payload, explicit_content_type=envelope_map.get("content_type")))
        payload_size = estimate_payload_size(payload)
        if payload_size > self.max_payload_bytes:
            raise PayloadTooLargeError(
                "Envelope payload exceeds configured max_payload_bytes.",
                context={"operation": "envelope_record_from_mapping", "payload_size": payload_size, "message_id": envelope_map.get("message_id")},
                details={"max_payload_bytes": self.max_payload_bytes},
            )
        payload_fingerprint = generate_idempotency_key(json_safe(payload), namespace="envelope_payload") if self.include_payload_fingerprint else ""
        return EnvelopeRecord(
            message_id=ensure_non_empty_string(str(envelope_map.get("message_id")), field_name="message_id"),
            correlation_id=ensure_non_empty_string(str(envelope_map.get("correlation_id")), field_name="correlation_id"),
            idempotency_key=ensure_non_empty_string(str(envelope_map.get("idempotency_key")), field_name="idempotency_key"),
            channel=normalize_channel_name(envelope_map.get("channel") or self.default_channel),
            protocol=normalize_protocol_name(envelope_map.get("protocol") or envelope_map.get("channel") or self.default_protocol),
            payload=payload,
            content_type=content_type,
            payload_size=payload_size,
            payload_fingerprint=payload_fingerprint,
            created_at=str(envelope_map.get("created_at") or utc_timestamp()),
            delivery_mode=ensure_non_empty_string(str(envelope_map.get("delivery_mode") or self.default_delivery_mode), field_name="delivery_mode").lower(),
            endpoint=self._normalize_endpoint(envelope_map.get("endpoint")),
            route=str(envelope_map.get("route")) if envelope_map.get("route") is not None else None,
            operation=str(envelope_map.get("operation")) if envelope_map.get("operation") is not None else self.default_operation,
            timeout_ms=None if envelope_map.get("timeout_ms") is None else coerce_timeout_ms(envelope_map.get("timeout_ms"), default=self.default_timeout_ms),
            expires_at=str(envelope_map.get("expires_at")) if envelope_map.get("expires_at") is not None else None,
            priority=int(envelope_map["priority"]) if envelope_map.get("priority") is not None else None,
            reply_to=str(envelope_map.get("reply_to")) if envelope_map.get("reply_to") is not None else None,
            dead_letter_queue=str(envelope_map.get("dead_letter_queue")) if envelope_map.get("dead_letter_queue") is not None else None,
            parent_message_id=str(envelope_map.get("parent_message_id")) if envelope_map.get("parent_message_id") is not None else None,
            trace_id=str(envelope_map.get("trace_id")) if envelope_map.get("trace_id") is not None else None,
            session_id=str(envelope_map.get("session_id")) if envelope_map.get("session_id") is not None else None,
            headers=normalize_headers(envelope_map.get("headers")),
            metadata=normalize_metadata(envelope_map.get("metadata")),
        )

    def _normalize_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        if endpoint is None:
            return None
        text = str(endpoint).strip()
        if not text:
            return None
        if not self.normalize_endpoints:
            return text
        try:
            if "://" in text:
                return normalize_endpoint(text)
        except Exception:
            return text
        return text

    def _normalize_error(
        self,
        error: BaseException | Mapping[str, Any],
        *,
        message_id: str,
        correlation_id: str,
        endpoint: Optional[str],
        channel: Optional[str],
        protocol: Optional[str],
    ) -> Dict[str, Any]:
        if isinstance(error, Mapping):
            return json_safe(error)
        if isinstance(error, NetworkError):
            return error.to_memory_snapshot()
        return build_error_snapshot(
            error,
            operation="envelope_dead_letter",
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            correlation_id=correlation_id,
            metadata={"message_id": message_id},
        )

    def _sync_envelope_memory(self, envelope: Mapping[str, Any]) -> None:
        if not self.record_memory_snapshots:
            return
        envelope_map = ensure_mapping(envelope, field_name="envelope")
        stored = dict(envelope_map)
        if not self.preserve_payload_in_memory:
            stored.pop("payload", None)
        self.memory.set(_ENVELOPE_LAST_KEY, stored, ttl_seconds=self.snapshot_ttl_seconds, source="envelope")
        self.memory.set(self._message_key(str(envelope_map["message_id"])), stored, ttl_seconds=self.snapshot_ttl_seconds, source="envelope")

    def _sync_snapshot_memory(self) -> None:
        if not self.record_memory_snapshots:
            return
        self.memory.set(_ENVELOPE_SNAPSHOT_KEY, self.get_snapshot(), ttl_seconds=self.snapshot_ttl_seconds, source="envelope")
        self.memory.set(
            _ENVELOPE_CACHE_KEY,
            {"generated_at": utc_timestamp(), "envelopes": [record.to_dict(include_payload=self.preserve_payload_in_memory) for record in self._records.values()]},
            ttl_seconds=self.snapshot_ttl_seconds,
            source="envelope",
        )

    def _append_history_locked(self, event_type: str, envelope: Mapping[str, Any]) -> None:
        if not self.record_history:
            return
        event = {
            "event_type": event_type,
            "occurred_at": utc_timestamp(),
            "envelope": sanitize_for_logging(envelope) if self.sanitize_logs else json_safe(envelope),
        }
        self._history.append(event)
        if len(self._history) > self.max_history_size:
            self._history = self._history[-self.max_history_size :]
        self.memory.append(
            _ENVELOPE_HISTORY_KEY,
            event,
            max_items=self.max_history_size,
            ttl_seconds=self.history_ttl_seconds,
            source="envelope",
        )

    def _ensure_capacity_locked(self, *, incoming_key: str) -> None:
        if incoming_key in self._records:
            return
        if len(self._records) < self.max_envelopes:
            return
        oldest_key = min(self._records.items(), key=lambda item: item[1].created_at)[0]
        self._records.pop(oldest_key, None)

    def _message_key(self, message_id: str) -> str:
        return f"network.lifecycle.envelope.message.{ensure_non_empty_string(message_id, field_name='message_id')}"

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.envelope_config.get(name, default)
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
            "Invalid boolean value in envelope configuration.",
            context={"operation": "envelope_config"},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.envelope_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in envelope configuration.",
                context={"operation": "envelope_config"},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Envelope configuration integer value must be non-negative.",
                context={"operation": "envelope_config"},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.envelope_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None


def _parse_timestamp(value: str) -> datetime:
    text = ensure_non_empty_string(str(value), field_name="timestamp")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print("\n=== Running Envelope ===\n")
    printer.status("TEST", "Envelope initialized", "info")

    memory = NetworkMemory()
    manager = EnvelopeManager(memory=memory)

    built = manager.build(
        payload={"task": "relay", "payload": {"hello": "world"}},
        channel="http",
        protocol="http",
        endpoint="https://api.example.com/v1/jobs",
        route="primary",
        operation="send",
        metadata={"tenant": "demo", "priority": "normal"},
        headers={"X-Test": "yes"},
        reply_to="https://api.example.com/v1/replies",
        dead_letter_queue="jobs.primary.dlq",
    )
    printer.status("TEST", "Envelope built", "info")

    validated = manager.validate(built)
    printer.status("TEST", "Envelope validated", "info")

    transport = manager.prepare_transport_view(validated)
    printer.status("TEST", "Transport view prepared", "info")

    response = manager.derive_response(
        validated,
        payload={"ok": True, "message": "received"},
        metadata={"response_source": "demo"},
    )
    printer.status("TEST", "Response envelope derived", "info")

    retry = manager.clone_for_retry(
        validated,
        metadata={"retry_reason": "temporary timeout"},
    )
    printer.status("TEST", "Retry envelope derived", "info")

    dead_letter = manager.derive_dead_letter(
        validated,
        TimeoutError("upstream timeout"),
        metadata={"phase": "send"},
    )
    printer.status("TEST", "Dead-letter envelope derived", "info")

    snapshot = manager.get_snapshot()
    cached = manager.get(validated["message_id"])
    all_envelopes = manager.list_envelopes()

    print("Built:", stable_json_dumps(built))
    print("Validated:", stable_json_dumps(validated))
    print("Transport:", stable_json_dumps(transport.to_dict(include_payload_preview=True)))
    print("Response:", stable_json_dumps(response))
    print("Retry:", stable_json_dumps(retry))
    print("Dead Letter:", stable_json_dumps(dead_letter))
    print("Snapshot:", stable_json_dumps(snapshot))
    print("Cached:", stable_json_dumps(cached))
    print("All Envelopes:", stable_json_dumps(all_envelopes))

    assert built["message_id"]
    assert built["correlation_id"]
    assert built["idempotency_key"]
    assert transport.payload_size > 0
    assert response["parent_message_id"] == built["message_id"]
    assert retry["idempotency_key"] == built["idempotency_key"]
    assert dead_letter["metadata"]["dead_letter_of_message_id"] == built["message_id"]
    assert memory.get("network.lifecycle.envelope.last") is not None
    assert memory.get("network.lifecycle.envelope.snapshot") is not None

    printer.status("TEST", "All Envelope checks passed", "info")
    print("\n=== Test ran successfully ===\n")
