"""
Send/Receive lifecycle orchestration for SLAI's Network Agent lifecycle subsystem.

This module provides the production-grade lifecycle facade that sits above the
lower-level lifecycle primitives:
- EnvelopeManager for canonical envelope ownership,
- DeliveryStateMachine for delivery truth and transitions,
- IdempotencyManager for replay-safe idempotency coordination.

NetworkLifecycle exists to keep the rest of the network stack from having to
coordinate these primitives manually for the common lifecycle flows involved in
networked operations:
- begin a delivery from payload/envelope input,
- reserve idempotency and detect replay/duplicate outcomes,
- register and advance canonical delivery state,
- derive retry, response, and dead-letter envelopes,
- complete, fail, cancel, expire, and purge lifecycle records,
- publish observability-friendly snapshots into shared NetworkMemory.

It does not own routing policy, retry timing algorithms, or transport
execution. Those concerns belong to routing, reliability, and adapter layers.
This module owns lifecycle coordination across the primitives that already hold
individual domain truth.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

from .utils import *
from .lifecycle import *
from .network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Network Lifecycle")
printer = PrettyPrinter()

_LIFECYCLE_LAST_KEY = "network.lifecycle.last"
_LIFECYCLE_SNAPSHOT_KEY = "network.lifecycle.snapshot"
_LIFECYCLE_HISTORY_KEY = "network.lifecycle.history"
_LIFECYCLE_ACTIVE_KEY = "network.lifecycle.active"


class NetworkLifecycle:
    """
    Coordinated lifecycle facade for send/receive/delivery state ownership.

    The class orchestrates envelope normalization, idempotency reservation, and
    delivery-state transitions so callers can work with one stable lifecycle
    surface instead of stitching together the lower-level lifecycle modules
    themselves.
    """

    def __init__(self, memory: Optional[NetworkMemory] = None, *,
                 config: Optional[Mapping[str, Any]] = None,
                 envelope_manager: Optional[EnvelopeManager] = None,
                 delivery_state_machine: Optional[DeliveryStateMachine] = None,
                 idempotency_manager: Optional[IdempotencyManager] = None) -> None:
        self.config = load_global_config()
        self.cycle_config = merge_mappings(
            get_config_section("network_lifecycle") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_memory_snapshots = self._get_bool_config("record_memory_snapshots", True)
        self.record_history = self._get_bool_config("record_history", True)
        self.auto_mark_processing_on_begin = self._get_bool_config("auto_mark_processing_on_begin", True)
        self.auto_complete_idempotency_on_ack = self._get_bool_config("auto_complete_idempotency_on_ack", True)
        self.auto_fail_idempotency_on_failure = self._get_bool_config("auto_fail_idempotency_on_failure", True)
        self.auto_release_idempotency_on_cancel = self._get_bool_config("auto_release_idempotency_on_cancel", True)
        self.auto_build_response_on_complete = self._get_bool_config("auto_build_response_on_complete", True)
        self.auto_build_retry_envelope = self._get_bool_config("auto_build_retry_envelope", True)
        self.auto_build_dead_letter_envelope = self._get_bool_config("auto_build_dead_letter_envelope", True)
        self.auto_expire_on_read = self._get_bool_config("auto_expire_on_read", True)
        # Default to True – an active idempotency claim must block new deliveries
        self.fail_on_duplicate_active = self._get_bool_config("fail_on_duplicate_active", True)
        self.enable_dead_letter = self._get_bool_config("enable_dead_letter", True)
        self.require_correlation_id = self._get_bool_config("require_correlation_id", True)
        self.require_idempotency_key = self._get_bool_config("require_idempotency_key", True)

        self.default_protocol = normalize_protocol_name(self._get_optional_string_config("default_protocol") or "http")
        self.default_channel = normalize_channel_name(self._get_optional_string_config("default_channel") or self.default_protocol)
        self.default_delivery_mode = ensure_non_empty_string(
            str(self.cycle_config.get("default_delivery_mode", "at_least_once")),
            field_name="default_delivery_mode",
        ).lower()
        self.default_receive_operation = self._get_optional_string_config("default_receive_operation") or "receive"
        self.default_request_timeout_ms = coerce_timeout_ms(
            self.cycle_config.get("default_request_timeout_ms"),
            default=5000,
            minimum=1,
            maximum=300000,
        )
        self.default_receive_timeout_ms = coerce_timeout_ms(
            self.cycle_config.get("default_receive_timeout_ms"),
            default=self.default_request_timeout_ms,
            minimum=1,
            maximum=300000,
        )
        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 7200)
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))

        self.envelopes = envelope_manager or EnvelopeManager(memory=self.memory)
        self.deliveries = delivery_state_machine or DeliveryStateMachine(memory=self.memory)
        self.idempotency = idempotency_manager or IdempotencyManager(memory=self.memory)

        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history_size)
        self._stats: Dict[str, int] = {
            "begins": 0,
            "receives": 0,
            "acks": 0,
            "nacks": 0,
            "failures": 0,
            "completions": 0,
            "retries": 0,
            "dead_letters": 0,
            "cancellations": 0,
            "purges": 0,
            "expires": 0,
        }
        self._started_at = utc_timestamp()

        self._sync_snapshot_memory()

    # ------------------------------------------------------------------
    # Lifecycle entry points
    # ------------------------------------------------------------------
    def begin_delivery(self, envelope: Optional[Mapping[str, Any]] = None, *, payload: Any = None,
                       channel: Optional[str] = None, protocol: Optional[str] = None,
                       endpoint: Optional[str] = None, route: Optional[str] = None, operation: Optional[str] = None,
                       timeout_ms: Optional[Any] = None,
                       delivery_mode: Optional[str] = None,
                       expires_in_seconds: Optional[int] = None,
                       max_attempts: Optional[int] = None,
                       metadata: Optional[Mapping[str, Any]] = None,
                       owner_token: Optional[str] = None,
                       initial_state: Optional[str] = None,
                       ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError("NetworkLifecycle is disabled by configuration.", context={"operation": "begin_delivery"})

        with self._lock:
            built_envelope = self.envelopes.build(
                envelope=envelope,
                payload=payload,
                channel=channel or self.default_channel,
                protocol=protocol or self.default_protocol,
                endpoint=endpoint,
                route=route,
                operation=operation,
                timeout_ms=timeout_ms or self.default_request_timeout_ms,
                delivery_mode=delivery_mode or self.default_delivery_mode,
                expires_in_seconds=expires_in_seconds,
                metadata=metadata,
            )
            self._validate_required_identity(built_envelope)

            reservation = self.idempotency.reserve(
                envelope=built_envelope,
                owner_token=owner_token,
                timeout_ms=timeout_ms or self.default_request_timeout_ms,
                metadata=metadata,
            )
            outcome = str(reservation.get("outcome", "reserved"))

            # Active duplicate – cannot proceed without breaking idempotency
            if outcome == "duplicate_active":
                if self.fail_on_duplicate_active:
                    active_record = ensure_mapping(reservation.get("record"), field_name="reservation.record")
                    raise DuplicateMessageError(
                        "An active idempotent lifecycle already exists for this key.",
                        context={
                            "operation": "begin_delivery",
                            "channel": active_record.get("channel"),
                            "protocol": active_record.get("protocol"),
                            "endpoint": active_record.get("endpoint"),
                            "correlation_id": active_record.get("correlation_id"),
                            "idempotency_key": active_record.get("idempotency_key"),
                        },
                        details={"status": active_record.get("status"), "message_id": active_record.get("message_id")},
                    )
                # If misconfigured to allow active duplicates, we still cannot register a new delivery.
                # Return a result indicating the conflict without creating a delivery record.
                result = {
                    "outcome": outcome,
                    "duplicate": True,
                    "replay": False,
                    "envelope": built_envelope,
                    "delivery": None,
                    "idempotency": reservation.get("record"),
                }
                self._append_history_locked("begin_delivery", result)
                self._sync_runtime_memory(result)
                return result

            # Replay – return cached response without creating a new delivery
            if outcome == "replay":
                record = ensure_mapping(reservation.get("record"), field_name="reservation.record")
                message_id = record.get("message_id")
                delivery = self.deliveries.get_delivery(str(message_id), include_history=True) if message_id else None
                response_snapshot = reservation.get("response_snapshot")
                result = {
                    "outcome": outcome,
                    "duplicate": True,
                    "replay": True,
                    "envelope": self.envelopes.get(str(message_id)) if message_id else built_envelope,
                    "delivery": delivery,
                    "idempotency": record,
                    "response_snapshot": json_safe(response_snapshot),
                }
                self._append_history_locked("begin_delivery", result)
                self._sync_runtime_memory(result)
                return result

            # Normal flow: reserve succeeded (new or reacquired)
            delivery = self._register_or_reuse_delivery(
                built_envelope,
                max_attempts=max_attempts,
                expires_in_seconds=expires_in_seconds,
                initial_state=initial_state,
            )

            if self.auto_mark_processing_on_begin:
                self.idempotency.mark_processing(
                    str(built_envelope["idempotency_key"]),
                    metadata={"message_id": built_envelope["message_id"]},
                )

            result = {
                "outcome": outcome,
                "duplicate": bool(reservation.get("duplicate", False)),
                "replay": bool(reservation.get("replay", False)),
                "envelope": built_envelope,
                "delivery": delivery,
                "idempotency": reservation.get("record"),
            }
            self._stats["begins"] += 1
            self._append_history_locked("begin_delivery", result)
            self._sync_runtime_memory(result)
            return result

    def receive_inbound(
        self,
        envelope: Optional[Mapping[str, Any]] = None,
        *,
        payload: Any = None,
        channel: Optional[str] = None,
        protocol: Optional[str] = None,
        endpoint: Optional[str] = None,
        route: Optional[str] = None,
        timeout_ms: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        owner_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            begun = self.begin_delivery(
                envelope=envelope,
                payload=payload,
                channel=channel,
                protocol=protocol,
                endpoint=endpoint,
                route=route,
                operation=self.default_receive_operation,
                timeout_ms=timeout_ms or self.default_receive_timeout_ms,
                metadata=metadata,
                owner_token=owner_token,
                initial_state="received",
            )
            # If begin_delivery returned a conflict (e.g. duplicate_active with fail_on_duplicate_active=False)
            if begun.get("delivery") is None:
                return begun
            delivery = ensure_mapping(begun["delivery"], field_name="delivery")
            received = self.deliveries.mark_received(
                str(delivery["message_id"]),
                metadata={"inbound": True, **normalize_metadata(metadata)},
            )
            begun["delivery"] = received
            self._stats["receives"] += 1
            self._append_history_locked("receive_inbound", begun)
            self._sync_runtime_memory(begun)
            return begun

    # ------------------------------------------------------------------
    # Coordinated state transitions
    # ------------------------------------------------------------------
    def mark_selected(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self._transition_delivery(message_or_key, "selected", **kwargs)

    def mark_sent(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self._transition_delivery(message_or_key, "sent", **kwargs)

    def mark_received(self, message_or_key: str, **kwargs: Any) -> Dict[str, Any]:
        return self._transition_delivery(message_or_key, "received", **kwargs)

    def acknowledge(
        self,
        message_or_key: str,
        *,
        response_snapshot: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            record = self._require_delivery_record(message_or_key)
            acked = self.deliveries.mark_acked(record["message_id"], metadata=metadata)
            idempotency_result = None
            if self.auto_complete_idempotency_on_ack:
                idempotency_result = self.idempotency.mark_completed(
                    acked["idempotency_key"],
                    response_snapshot=response_snapshot or {"message_id": acked["message_id"], "state": "acked"},
                    metadata={"acked": True, **normalize_metadata(metadata)},
                )
            result = {
                "delivery": acked,
                "idempotency": idempotency_result,
                "response_snapshot": json_safe(response_snapshot),
            }
            self._stats["acks"] += 1
            self._append_history_locked("acknowledge", result)
            self._sync_runtime_memory(result)
            return result

    def nack(
        self,
        message_or_key: str,
        *,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            record = self._require_delivery_record(message_or_key)
            result = self.deliveries.mark_nacked(
                record["message_id"],
                metadata={"reason": reason, **normalize_metadata(metadata)},
            )
            payload = {"delivery": result, "reason": reason}
            self._stats["nacks"] += 1
            self._append_history_locked("nack", payload)
            self._sync_runtime_memory(payload)
            return payload

    def fail(
        self,
        message_or_key: str,
        error: BaseException | Mapping[str, Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            record = self._require_delivery_record(message_or_key)
            failed = self.deliveries.mark_failed(record["message_id"], error=error, metadata=metadata)
            idempotency_result = None
            if self.auto_fail_idempotency_on_failure:
                idempotency_result = self.idempotency.mark_failed(
                    failed["idempotency_key"],
                    error=error,
                    metadata=normalize_metadata(metadata),
                )
            payload = {"delivery": failed, "idempotency": idempotency_result}
            self._stats["failures"] += 1
            self._append_history_locked("fail", payload)
            self._sync_runtime_memory(payload)
            return payload

    def complete(
        self,
        message_or_key: str,
        *,
        response_payload: Any = None,
        response_snapshot: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            record = self._require_delivery_record(message_or_key)
            ack_payload = response_snapshot
            response_envelope = None
            if self.auto_build_response_on_complete:
                response_envelope = self.envelopes.derive_response(
                    record["envelope"],
                    payload=response_payload,
                    metadata=metadata,
                )
                if ack_payload is None:
                    ack_payload = {"response_envelope": response_envelope}
            completed = self.acknowledge(record["message_id"], response_snapshot=ack_payload, metadata=metadata)
            completed["response_envelope"] = response_envelope
            self._stats["completions"] += 1
            self._append_history_locked("complete", completed)
            self._sync_runtime_memory(completed)
            return completed

    def plan_retry(
        self,
        message_or_key: str,
        error: Optional[BaseException | Mapping[str, Any]] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        preserve_message_id: bool = True,
        preserve_correlation_id: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            record = self._require_delivery_record(message_or_key)
            retrying = self.deliveries.mark_retrying(record["message_id"], error=error, metadata=metadata)
            retry_envelope = None
            if self.auto_build_retry_envelope:
                retry_envelope = self.envelopes.clone_for_retry(
                    record["envelope"],
                    preserve_message_id=preserve_message_id,
                    preserve_correlation_id=preserve_correlation_id,
                    metadata=metadata,
                )
            payload = {"delivery": retrying, "retry_envelope": retry_envelope}
            self._stats["retries"] += 1
            self._append_history_locked("plan_retry", payload)
            self._sync_runtime_memory(payload)
            return payload

    def dead_letter(
        self,
        message_or_key: str,
        error: BaseException | Mapping[str, Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enable_dead_letter:
            raise DeadLetterQueueError("Dead-letter handling is disabled by configuration.", context={"operation": "dead_letter"})
        with self._lock:
            record = self._require_delivery_record(message_or_key)
            dead_letter_envelope = None
            if self.auto_build_dead_letter_envelope:
                dead_letter_envelope = self.envelopes.derive_dead_letter(
                    record["envelope"],
                    error,
                    metadata=metadata,
                )
            dead_lettered = self.deliveries.mark_dead_lettered(record["message_id"], metadata=metadata)
            idempotency_result = None
            if self.auto_fail_idempotency_on_failure:
                idempotency_result = self.idempotency.mark_failed(
                    dead_lettered["idempotency_key"],
                    error=error,
                    metadata={"dead_lettered": True, **normalize_metadata(metadata)},
                )
            payload = {
                "delivery": dead_lettered,
                "dead_letter_envelope": dead_letter_envelope,
                "idempotency": idempotency_result,
            }
            self._stats["dead_letters"] += 1
            self._append_history_locked("dead_letter", payload)
            self._sync_runtime_memory(payload)
            return payload

    def cancel(self, message_or_key: str, *, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            record = self._require_delivery_record(message_or_key)
            cancelled = self.deliveries.cancel(record["message_id"], metadata=metadata)
            idempotency_result = None
            if self.auto_release_idempotency_on_cancel:
                idempotency_result = self.idempotency.release(
                    cancelled["idempotency_key"],
                    metadata={"cancelled": True, **normalize_metadata(metadata)},
                )
            payload = {"delivery": cancelled, "idempotency": idempotency_result}
            self._stats["cancellations"] += 1
            self._append_history_locked("cancel", payload)
            self._sync_runtime_memory(payload)
            return payload

    # ------------------------------------------------------------------
    # Read / maintenance APIs
    # ------------------------------------------------------------------
    def get_lifecycle_record(self, message_or_key: str) -> Dict[str, Any]:
        delivery = self._require_delivery_record(message_or_key)
        envelope = self.envelopes.get(delivery["message_id"], include_payload=True)
        idempotency = self.idempotency.get_record(delivery["idempotency_key"], include_history=True)
        payload = {
            "delivery": delivery,
            "envelope": envelope,
            "idempotency": idempotency,
        }
        self._sync_runtime_memory(payload)
        return payload

    def expire_stale(self) -> Dict[str, Any]:
        with self._lock:
            expired_deliveries = self.deliveries.expire_stale_deliveries()
            expired_idempotency = self.idempotency.expire_stale_records()
            self._stats["expires"] += len(expired_deliveries) + len(expired_idempotency)
            payload = {
                "expired_deliveries": expired_deliveries,
                "expired_idempotency": expired_idempotency,
            }
            self._append_history_locked("expire_stale", payload)
            self._sync_snapshot_memory()
            return payload

    def purge_terminal(self, *, older_than_seconds: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            purged_deliveries = self.deliveries.purge_terminal_deliveries(older_than_seconds=older_than_seconds)
            purged_idempotency = self.idempotency.purge_terminal_records(older_than_seconds=older_than_seconds)
            self._stats["purges"] += purged_deliveries + purged_idempotency
            payload = {
                "purged_deliveries": purged_deliveries,
                "purged_idempotency": purged_idempotency,
            }
            self._append_history_locked("purge_terminal", payload)
            self._sync_snapshot_memory()
            return payload

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if self.auto_expire_on_read:
                self.deliveries.expire_stale_deliveries()
                self.idempotency.expire_stale_records()
            snapshot = {
                "generated_at": utc_timestamp(),
                "started_at": self._started_at,
                "stats": dict(self._stats),
                "envelope": self.envelopes.get_snapshot(),
                "delivery": self.deliveries.get_snapshot(),
                "idempotency": self.idempotency.get_snapshot(),
                "history_size": len(self._history),
            }
            return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_or_reuse_delivery(
        self,
        envelope: Mapping[str, Any],
        *,
        max_attempts: Optional[int],
        expires_in_seconds: Optional[int],
        initial_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        message_id = ensure_non_empty_string(str(envelope["message_id"]), field_name="message_id")
        existing = self.deliveries.get_delivery(message_id, include_history=True)
        if existing is not None:
            return existing
        return self.deliveries.register_delivery(
            envelope=envelope,
            max_attempts=max_attempts,
            expires_in_seconds=expires_in_seconds,
            state=initial_state,
        )

    def _transition_delivery(self, message_or_key: str, to_state: str, **kwargs: Any) -> Dict[str, Any]:
        with self._lock:
            record = self._require_delivery_record(message_or_key)
            result = self.deliveries.transition(record["message_id"], to_state, **kwargs)
            payload = {"delivery": result}
            self._append_history_locked(f"transition:{to_state}", payload)
            self._sync_runtime_memory(payload)
            return payload

    def _require_delivery_record(self, message_or_key: str) -> Dict[str, Any]:
        record = self.deliveries.require_delivery(message_or_key, include_history=True)
        return ensure_mapping(record, field_name="delivery_record")

    def _validate_required_identity(self, envelope: Mapping[str, Any]) -> None:
        if self.require_correlation_id and not envelope.get("correlation_id"):
            raise PayloadValidationError(
                "Lifecycle begin_delivery requires correlation_id.",
                context={"operation": "begin_delivery"},
            )
        if self.require_idempotency_key and not envelope.get("idempotency_key"):
            raise PayloadValidationError(
                "Lifecycle begin_delivery requires idempotency_key.",
                context={"operation": "begin_delivery"},
            )

    def _sync_runtime_memory(self, payload: Mapping[str, Any]) -> None:
        if not self.record_memory_snapshots:
            return
        self.memory.set(
            _LIFECYCLE_LAST_KEY,
            json_safe(payload),
            ttl_seconds=self.snapshot_ttl_seconds,
            source="network_lifecycle",
        )
        delivery = ensure_mapping(payload.get("delivery"), field_name="payload.delivery", allow_none=True)
        if delivery:
            message_id = delivery.get("message_id")
            if message_id:
                self.memory.set(
                    f"{_LIFECYCLE_ACTIVE_KEY}.{message_id}",
                    {
                        "message_id": message_id,
                        "state": delivery.get("state"),
                        "correlation_id": delivery.get("correlation_id"),
                        "idempotency_key": delivery.get("idempotency_key"),
                        "endpoint": delivery.get("endpoint"),
                        "channel": delivery.get("channel"),
                        "protocol": delivery.get("protocol"),
                        "updated_at": delivery.get("updated_at"),
                    },
                    ttl_seconds=self.snapshot_ttl_seconds,
                    source="network_lifecycle",
                )
        self._sync_snapshot_memory()

    def _sync_snapshot_memory(self) -> None:
        if not self.record_memory_snapshots:
            return
        snapshot = self.get_snapshot()
        self.memory.set(
            _LIFECYCLE_SNAPSHOT_KEY,
            snapshot,
            ttl_seconds=self.snapshot_ttl_seconds,
            source="network_lifecycle",
        )

    def _append_history_locked(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.record_history:
            return
        event = {
            "event_type": ensure_non_empty_string(event_type, field_name="event_type"),
            "occurred_at": utc_timestamp(),
            "payload": sanitize_for_logging(payload) if self.sanitize_logs else json_safe(payload),
        }
        self._history.append(event)
        self.memory.append(
            _LIFECYCLE_HISTORY_KEY,
            event,
            max_items=self.max_history_size,
            ttl_seconds=self.history_ttl_seconds,
            source="network_lifecycle",
        )

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.cycle_config.get(name, default)
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
            "Invalid boolean value in network lifecycle configuration.",
            context={"operation": "network_lifecycle_config"},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.cycle_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in network lifecycle configuration.",
                context={"operation": "network_lifecycle_config"},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Configuration value must be non-negative.",
                context={"operation": "network_lifecycle_config"},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.cycle_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None


if __name__ == "__main__":
    print("\n=== Running Network Lifecycle ===\n")
    printer.status("TEST", "Network Lifecycle initialized", "info")

    memory = NetworkMemory()
    lifecycle = NetworkLifecycle(memory=memory)

    begun = lifecycle.begin_delivery(
        payload={"task": "relay", "payload": {"hello": "world"}},
        channel="http",
        protocol="http",
        endpoint="https://api.example.com/v1/jobs",
        route="primary",
        metadata={"tenant": "demo", "priority": "normal"},
    )
    printer.status("TEST", "Lifecycle begin_delivery completed", "info")

    selected = lifecycle.mark_selected(begun["delivery"]["message_id"], metadata={"selector": "channel_selector"})
    sent = lifecycle.mark_sent(begun["delivery"]["message_id"], metadata={"adapter": "http"})
    received = lifecycle.mark_received(begun["delivery"]["message_id"], metadata={"recv": True})
    completed = lifecycle.complete(
        begun["delivery"]["message_id"],
        response_payload={"ok": True, "result": "accepted"},
        metadata={"transport": "http"},
    )
    printer.status("TEST", "Lifecycle happy path completed", "info")

    retry_begin = lifecycle.begin_delivery(
        payload={"task": "retry", "payload": {"hello": "again"}},
        channel="queue",
        protocol="queue",
        endpoint="amqp://broker.internal:5672/jobs",
        route="secondary",
        metadata={"tenant": "demo", "priority": "high"},
    )
    lifecycle.mark_sent(retry_begin["delivery"]["message_id"])
    failed = lifecycle.fail(retry_begin["delivery"]["message_id"], TimeoutError("upstream timeout"), metadata={"phase": "send"})
    retry_plan = lifecycle.plan_retry(retry_begin["delivery"]["message_id"], TimeoutError("retry scheduled"), metadata={"backoff_ms": 250})
    dead_lettered = lifecycle.dead_letter(retry_begin["delivery"]["message_id"], TimeoutError("exhausted retries"), metadata={"reason": "exhausted_retries"})
    # cancelled = lifecycle.cancel(retry_begin["delivery"]["message_id"], metadata={"cleanup": True})
    printer.status("TEST", "Lifecycle retry/failure path completed", "info")

    snapshot = lifecycle.get_snapshot()
    record = lifecycle.get_lifecycle_record(begun["delivery"]["message_id"])
    expired = lifecycle.expire_stale()
    purged = lifecycle.purge_terminal(older_than_seconds=0)

    print("Begun:", stable_json_dumps(begun))
    print("Selected:", stable_json_dumps(selected))
    print("Sent:", stable_json_dumps(sent))
    print("Received:", stable_json_dumps(received))
    print("Completed:", stable_json_dumps(completed))
    print("Failed:", stable_json_dumps(failed))
    print("Retry Plan:", stable_json_dumps(retry_plan))
    print("Dead Lettered:", stable_json_dumps(dead_lettered))
    # print("Cancelled:", stable_json_dumps(cancelled))
    print("Record:", stable_json_dumps(record))
    print("Snapshot:", stable_json_dumps(snapshot))
    print("Expired:", stable_json_dumps(expired))
    print("Purged:", stable_json_dumps(purged))

    assert begun["delivery"]["state"] == lifecycle.deliveries.default_initial_state
    assert completed["delivery"]["state"] == "acked"
    assert completed["idempotency"]["status"] == "completed"
    assert retry_plan["delivery"]["state"] == "retrying"
    assert dead_lettered["delivery"]["state"] == "dead_lettered"
    assert memory.get("network.lifecycle.last") is not None
    assert memory.get("network.lifecycle.snapshot") is not None

    printer.status("TEST", "All Network Lifecycle checks passed", "info")
    print("\n=== Test ran successfully ===\n")
