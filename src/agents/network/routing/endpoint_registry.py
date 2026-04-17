"""
Endpoint registry for SLAI's Network Agent routing subsystem.

This module provides the production-grade endpoint registry used by the routing
stack beneath Network Stream. It owns structured endpoint discovery state,
capability metadata, health posture snapshots, and candidate enumeration for
runtime channel and route selection.

The registry is intentionally scoped to endpoint inventory and endpoint-level
operational state. It is responsible for:
- registering and updating endpoint definitions across channel/protocol families,
- normalizing endpoint identities and preserving logical queue/broker targets,
- tracking health, availability, circuit posture, and recent success/failure
  signals,
- exposing filtered candidate sets for channel selection and route policy,
- syncing endpoint-health state into shared NetworkMemory for the broader
  network subsystem.

It does not own route ranking, policy arbitration, retry logic, or transport
execution. Those concerns belong to Channel Selector, Route Policy,
Reliability, and the specialized adapters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils import *
from ..network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Endpoint Registry")
printer = PrettyPrinter()


_ENDPOINT_REGISTRY_SNAPSHOT_KEY = "network.routing.endpoint_registry.snapshot"
_ENDPOINT_REGISTRY_LAST_KEY = "network.routing.endpoint_registry.last"
_ENDPOINT_REGISTRY_HISTORY_KEY = "network.routing.endpoint_registry.history"
_ENDPOINT_REGISTRY_HEALTH_SUMMARY_KEY = "network.routing.endpoint_registry.health_summary"

_DEFAULT_HEALTHY_STATUSES = ("healthy", "available", "up", "idle", "connected")
_DEFAULT_DEGRADED_STATUSES = ("degraded", "warning", "limited", "slow")
_DEFAULT_UNHEALTHY_STATUSES = ("down", "failed", "unhealthy", "blocked", "closed")


@dataclass(slots=True)
class EndpointRecord:
    """Structured endpoint definition and its current operational posture."""

    endpoint_id: str
    endpoint: str
    protocol: str
    channel: str
    endpoint_type: str = "network"
    region: Optional[str] = None
    zone: Optional[str] = None
    priority: int = 100
    weight: float = 1.0
    secure: Optional[bool] = None
    available: bool = True
    enabled: bool = True
    status: str = "unknown"
    latency_ms: Optional[float] = None
    success_rate: Optional[float] = None
    error_rate: Optional[float] = None
    consecutive_failures: int = 0
    circuit_state: Optional[str] = None
    last_error: Optional[Dict[str, Any]] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    last_success_at: Optional[str] = None
    last_failure_at: Optional[str] = None

    def inferred_secure(self) -> bool:
        if self.secure is not None:
            return bool(self.secure)
        if self.endpoint_type == "network" and self.endpoint:
            try:
                return bool(parse_endpoint(self.endpoint, default_scheme=self.protocol, protocol=self.protocol, require_host=False).secure)
            except Exception:
                return is_secure_protocol(self.protocol)
        return is_secure_protocol(self.protocol)

    def health_score(self) -> float:
        base = 0.5
        status = self.status.lower().strip()
        if status in _DEFAULT_HEALTHY_STATUSES:
            base = 1.0
        elif status in _DEFAULT_DEGRADED_STATUSES:
            base = 0.65
        elif status in _DEFAULT_UNHEALTHY_STATUSES:
            base = 0.15

        if self.success_rate is not None:
            base *= max(0.0, min(1.0, float(self.success_rate)))
        if self.error_rate is not None:
            base *= max(0.0, min(1.0, 1.0 - float(self.error_rate)))
        if self.circuit_state and str(self.circuit_state).lower() == "open":
            base *= 0.05
        if not self.available or not self.enabled:
            base = 0.0
        if self.consecutive_failures > 0:
            base *= max(0.1, 1.0 - min(0.8, self.consecutive_failures * 0.1))
        return round(base, 6)

    def to_candidate(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "endpoint": self.endpoint,
            "protocol": self.protocol,
            "channel": self.channel,
            "region": self.region,
            "zone": self.zone,
            "priority": self.priority,
            "weight": self.weight,
            "secure": self.inferred_secure(),
            "available": self.available,
            "enabled": self.enabled,
            "status": self.status,
            "health_score": self.health_score(),
            "circuit_state": self.circuit_state,
            "capabilities": json_safe(self.capabilities),
            "tags": list(self.tags),
            "metadata": json_safe(self.metadata),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "endpoint_id": self.endpoint_id,
            "endpoint": self.endpoint,
            "protocol": self.protocol,
            "channel": self.channel,
            "endpoint_type": self.endpoint_type,
            "region": self.region,
            "zone": self.zone,
            "priority": int(self.priority),
            "weight": float(self.weight),
            "secure": self.inferred_secure(),
            "available": self.available,
            "enabled": self.enabled,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "consecutive_failures": self.consecutive_failures,
            "circuit_state": self.circuit_state,
            "last_error": self.last_error,
            "capabilities": json_safe(self.capabilities),
            "tags": list(self.tags),
            "metadata": json_safe(self.metadata),
            "health_score": self.health_score(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


class EndpointRegistry:
    """
    Runtime endpoint registry for the routing subsystem.

    The registry provides stable endpoint inventory and endpoint-health signals
    to Channel Selector, Route Policy, and Network Stream orchestration.
    """

    def __init__(
        self,
        memory: Optional[NetworkMemory] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = load_global_config()
        self.registry_config = merge_mappings(
            get_config_section("network_endpoint_registry") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_registry_snapshots = self._get_bool_config("record_registry_snapshots", True)
        self.sync_endpoint_health_to_memory = self._get_bool_config("sync_endpoint_health_to_memory", True)
        self.evict_oldest_on_capacity = self._get_bool_config("evict_oldest_on_capacity", True)
        self.allow_logical_targets = self._get_bool_config("allow_logical_targets", True)
        self.default_enabled = self._get_bool_config("default_enabled", True)
        self.default_available = self._get_bool_config("default_available", True)
        self.default_secure = self._get_bool_config("default_secure", False)

        self.default_protocol = normalize_protocol_name(self._get_optional_string_config("default_protocol") or "http")
        self.default_channel = normalize_channel_name(self._get_optional_string_config("default_channel") or self.default_protocol)
        self.default_status = self._get_status_config("default_status", "unknown")
        self.default_endpoint_type = self._get_optional_string_config("default_endpoint_type") or "network"
        self.default_region = self._get_optional_string_config("default_region")
        self.default_zone = self._get_optional_string_config("default_zone")

        self.snapshot_ttl_seconds = self._get_non_negative_int_config("snapshot_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 3600)
        self.max_registry_entries = max(1, self._get_non_negative_int_config("max_registry_entries", 5000))
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 1000))
        self.default_priority = self._coerce_int(self.registry_config.get("default_priority"), 100)
        self.default_weight = self._coerce_float(self.registry_config.get("default_weight"), 1.0, minimum=0.0)

        self.healthy_statuses = self._get_sequence_config("healthy_statuses", _DEFAULT_HEALTHY_STATUSES)
        self.degraded_statuses = self._get_sequence_config("degraded_statuses", _DEFAULT_DEGRADED_STATUSES)
        self.unhealthy_statuses = self._get_sequence_config("unhealthy_statuses", _DEFAULT_UNHEALTHY_STATUSES)

        self._records: Dict[str, EndpointRecord] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history_size)
        self._stats: Dict[str, int] = {
            "registrations": 0,
            "updates": 0,
            "health_updates": 0,
            "successes": 0,
            "failures": 0,
            "deletes": 0,
            "candidate_queries": 0,
            "evictions": 0,
        }
        self._started_at = utc_timestamp()

        self._sync_registry_memory()

    # ------------------------------------------------------------------
    # Public registry API
    # ------------------------------------------------------------------
    def register_endpoint(
        self,
        endpoint: str,
        *,
        endpoint_id: Optional[str] = None,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint_type: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        priority: Optional[int] = None,
        weight: Optional[float] = None,
        secure: Optional[bool] = None,
        enabled: Optional[bool] = None,
        available: Optional[bool] = None,
        status: Optional[str] = None,
        capabilities: Optional[Mapping[str, Any]] = None,
        tags: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ReliabilityError(
                "Endpoint registry is disabled by configuration.",
                context={"operation": "register_endpoint"},
            )

        normalized_endpoint, normalized_type = self._normalize_endpoint_value(
            endpoint,
            endpoint_type=endpoint_type,
            protocol=protocol,
            channel=channel,
        )
        normalized_protocol = normalize_protocol_name(protocol or self.default_protocol)
        normalized_channel = normalize_channel_name(channel or normalized_protocol or self.default_channel)
        normalized_endpoint_id = self._resolve_endpoint_id(endpoint_id, normalized_endpoint, normalized_protocol, normalized_channel)

        with self._lock:
            existing = self._records.get(normalized_endpoint_id)
            if existing is not None and not overwrite:
                raise DeliveryStateError(
                    "Endpoint already exists and overwrite is disabled.",
                    context={"operation": "register_endpoint", "endpoint": normalized_endpoint, "protocol": normalized_protocol, "channel": normalized_channel},
                    details={"endpoint_id": normalized_endpoint_id},
                )
            self._ensure_capacity_locked(incoming_key=normalized_endpoint_id)

            now = utc_timestamp()
            record = EndpointRecord(
                endpoint_id=normalized_endpoint_id,
                endpoint=normalized_endpoint,
                protocol=normalized_protocol,
                channel=normalized_channel,
                endpoint_type=normalized_type,
                region=(str(region).strip() if region is not None and str(region).strip() else self.default_region),
                zone=(str(zone).strip() if zone is not None and str(zone).strip() else self.default_zone),
                priority=self._coerce_int(priority, self.default_priority),
                weight=self._coerce_float(weight, self.default_weight, minimum=0.0),
                secure=self.default_secure if secure is None else bool(secure),
                enabled=self.default_enabled if enabled is None else bool(enabled),
                available=self.default_available if available is None else bool(available),
                status=self._normalize_status(status or self.default_status),
                latency_ms=existing.latency_ms if existing is not None else None,
                success_rate=existing.success_rate if existing is not None else None,
                error_rate=existing.error_rate if existing is not None else None,
                consecutive_failures=existing.consecutive_failures if existing is not None else 0,
                circuit_state=existing.circuit_state if existing is not None else None,
                last_error=existing.last_error if existing is not None else None,
                capabilities=normalize_metadata(capabilities),
                tags=normalize_tags(tags) if tags is not None else (existing.tags if existing is not None else ()),
                metadata=merge_mappings(existing.metadata if existing is not None else {}, normalize_metadata(metadata)),
                created_at=existing.created_at if existing is not None else now,
                updated_at=now,
                last_success_at=existing.last_success_at if existing is not None else None,
                last_failure_at=existing.last_failure_at if existing is not None else None,
            )
            self._records[normalized_endpoint_id] = record
            self._stats["registrations"] += 1 if existing is None else 0
            self._stats["updates"] += 1 if existing is not None else 0
            self._append_event_locked(
                "register_endpoint" if existing is None else "update_endpoint",
                record.to_dict(),
            )
            self._sync_record_memory(record)
            self._sync_registry_memory()
            return record.to_dict()

    def upsert_endpoint(self, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        return self.register_endpoint(endpoint, overwrite=True, **kwargs)

    def unregister_endpoint(self, endpoint_or_id: str) -> bool:
        endpoint_id = self._resolve_existing_id(endpoint_or_id)
        with self._lock:
            record = self._records.pop(endpoint_id, None)
            if record is None:
                return False
            self._stats["deletes"] += 1
            self._append_event_locked("unregister_endpoint", {"endpoint_id": endpoint_id, "endpoint": record.endpoint})
            self._sync_registry_memory()
            return True

    def get_endpoint(self, endpoint_or_id: str) -> Optional[Dict[str, Any]]:
        endpoint_id = self._resolve_existing_id(endpoint_or_id)
        with self._lock:
            record = self._records.get(endpoint_id)
            return None if record is None else record.to_dict()

    def require_endpoint(self, endpoint_or_id: str) -> Dict[str, Any]:
        payload = self.get_endpoint(endpoint_or_id)
        if payload is None:
            raise NoRouteAvailableError(
                "Requested endpoint is not registered.",
                context={"operation": "require_endpoint", "endpoint": str(endpoint_or_id)},
            )
        return payload

    def list_endpoints(
        self,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_type: Optional[str] = None,
        include_disabled: bool = False,
        include_unavailable: bool = True,
        tags: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        normalized_tags = {str(item).strip() for item in ensure_sequence(tags, field_name="tags", allow_none=True, coerce_scalar=True) if str(item).strip()}
        with self._lock:
            records: Iterable[EndpointRecord] = self._records.values()
            result: List[Dict[str, Any]] = []
            for record in records:
                if protocol is not None and record.protocol != normalize_protocol_name(protocol):
                    continue
                if channel is not None and record.channel != normalize_channel_name(channel):
                    continue
                if region is not None and (record.region or "") != str(region).strip():
                    continue
                if endpoint_type is not None and record.endpoint_type != str(endpoint_type).strip().lower():
                    continue
                if not include_disabled and not record.enabled:
                    continue
                if not include_unavailable and not record.available:
                    continue
                if normalized_tags and not normalized_tags.issubset(set(record.tags)):
                    continue
                result.append(record.to_dict())
            result.sort(key=self._sort_record_key)
            return result

    def get_candidates(
        self,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_type: Optional[str] = None,
        include_disabled: bool = False,
        include_unavailable: bool = False,
        include_degraded: bool = True,
        include_unhealthy: bool = False,
        tags: Optional[Sequence[Any]] = None,
        capability_constraints: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        normalized_constraints = ensure_mapping(capability_constraints, field_name="capability_constraints", allow_none=True)
        with self._lock:
            self._stats["candidate_queries"] += 1
            result: List[Dict[str, Any]] = []
            for record_dict in self.list_endpoints(
                protocol=protocol,
                channel=channel,
                region=region,
                endpoint_type=endpoint_type,
                include_disabled=include_disabled,
                include_unavailable=include_unavailable,
                tags=tags,
            ):
                status = str(record_dict.get("status", "")).lower()
                if not include_degraded and status in set(self.degraded_statuses):
                    continue
                if not include_unhealthy and status in set(self.unhealthy_statuses):
                    continue
                capabilities = ensure_mapping(record_dict.get("capabilities"), field_name="capabilities", allow_none=True)
                if not self._supports_capabilities(capabilities, normalized_constraints):
                    continue
                result.append({
                    **record_dict,
                    "candidate_source": "endpoint_registry",
                })
            result.sort(key=self._sort_candidate_dict)
            return result

    def update_health(
        self,
        endpoint_or_id: str,
        *,
        status: Optional[str] = None,
        latency_ms: Optional[float] = None,
        success_rate: Optional[float] = None,
        error_rate: Optional[float] = None,
        available: Optional[bool] = None,
        enabled: Optional[bool] = None,
        circuit_state: Optional[str] = None,
        last_error: Optional[BaseException | Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        endpoint_id = self._resolve_existing_id(endpoint_or_id)
        with self._lock:
            record = self._require_record_locked(endpoint_id)
            now = utc_timestamp()
            if status is not None:
                record.status = self._normalize_status(status)
            if latency_ms is not None:
                record.latency_ms = float(latency_ms)
            if success_rate is not None:
                record.success_rate = self._coerce_float(success_rate, record.success_rate or 0.0, minimum=0.0, maximum=1.0)
            if error_rate is not None:
                record.error_rate = self._coerce_float(error_rate, record.error_rate or 0.0, minimum=0.0, maximum=1.0)
            if available is not None:
                record.available = bool(available)
            if enabled is not None:
                record.enabled = bool(enabled)
            if circuit_state is not None:
                record.circuit_state = ensure_non_empty_string(str(circuit_state), field_name="circuit_state").lower()
            if last_error is not None:
                record.last_error = self._normalize_error(last_error, endpoint=record.endpoint, protocol=record.protocol, channel=record.channel)
                record.last_failure_at = now
            record.metadata = merge_mappings(record.metadata, normalize_metadata(metadata))
            record.updated_at = now
            self._stats["health_updates"] += 1
            self._append_event_locked("update_health", record.to_dict())
            self._sync_record_memory(record)
            self._sync_registry_memory()
            return record.to_dict()

    def set_circuit_state(
        self,
        endpoint_or_id: str,
        circuit_state: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        endpoint_id = self._resolve_existing_id(endpoint_or_id)
        with self._lock:
            record = self._require_record_locked(endpoint_id)
            record.circuit_state = ensure_non_empty_string(circuit_state, field_name="circuit_state").lower()
            record.updated_at = utc_timestamp()
            record.metadata = merge_mappings(record.metadata, normalize_metadata(metadata))
            self.memory.set_endpoint_circuit_state(record.endpoint, record.circuit_state, metadata={"endpoint_id": record.endpoint_id, **normalize_metadata(metadata)})
            self._append_event_locked("set_circuit_state", record.to_dict())
            self._sync_record_memory(record)
            self._sync_registry_memory()
            return record.to_dict()

    def mark_success(
        self,
        endpoint_or_id: str,
        *,
        latency_ms: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        endpoint_id = self._resolve_existing_id(endpoint_or_id)
        with self._lock:
            record = self._require_record_locked(endpoint_id)
            now = utc_timestamp()
            record.status = "healthy"
            record.available = True
            if latency_ms is not None:
                record.latency_ms = float(latency_ms)
            record.consecutive_failures = 0
            record.last_success_at = now
            if record.success_rate is None:
                record.success_rate = 1.0
            else:
                record.success_rate = round(min(1.0, (record.success_rate * 0.8) + 0.2), 6)
            if record.error_rate is None:
                record.error_rate = 0.0
            else:
                record.error_rate = round(max(0.0, record.error_rate * 0.7), 6)
            record.updated_at = now
            record.metadata = merge_mappings(record.metadata, normalize_metadata(metadata))
            self._stats["successes"] += 1
            self._append_event_locked("mark_success", record.to_dict())
            self._sync_record_memory(record)
            self._sync_registry_memory()
            return record.to_dict()

    def mark_failure(
        self,
        endpoint_or_id: str,
        error: BaseException | Mapping[str, Any],
        *,
        status: str = "degraded",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        endpoint_id = self._resolve_existing_id(endpoint_or_id)
        with self._lock:
            record = self._require_record_locked(endpoint_id)
            now = utc_timestamp()
            record.status = self._normalize_status(status)
            record.available = record.status not in set(self.unhealthy_statuses)
            record.consecutive_failures += 1
            record.last_failure_at = now
            record.last_error = self._normalize_error(error, endpoint=record.endpoint, protocol=record.protocol, channel=record.channel)
            if record.success_rate is None:
                record.success_rate = 0.0
            else:
                record.success_rate = round(max(0.0, record.success_rate * 0.75), 6)
            if record.error_rate is None:
                record.error_rate = 1.0
            else:
                record.error_rate = round(min(1.0, (record.error_rate * 0.8) + 0.2), 6)
            record.updated_at = now
            record.metadata = merge_mappings(record.metadata, normalize_metadata(metadata))
            self._stats["failures"] += 1
            self._append_event_locked("mark_failure", record.to_dict())
            self._sync_record_memory(record)
            self._sync_registry_memory()
            return record.to_dict()

    def get_health_summary(self) -> Dict[str, Any]:
        with self._lock:
            healthy = degraded = unhealthy = disabled = unavailable = 0
            protocols: Dict[str, int] = {}
            channels: Dict[str, int] = {}
            regions: Dict[str, int] = {}
            for record in self._records.values():
                status = record.status.lower()
                if not record.enabled:
                    disabled += 1
                if not record.available:
                    unavailable += 1
                if status in set(self.healthy_statuses):
                    healthy += 1
                elif status in set(self.degraded_statuses):
                    degraded += 1
                else:
                    unhealthy += 1
                protocols[record.protocol] = protocols.get(record.protocol, 0) + 1
                channels[record.channel] = channels.get(record.channel, 0) + 1
                if record.region:
                    regions[record.region] = regions.get(record.region, 0) + 1
            payload = {
                "generated_at": utc_timestamp(),
                "entry_count": len(self._records),
                "healthy_count": healthy,
                "degraded_count": degraded,
                "unhealthy_count": unhealthy,
                "disabled_count": disabled,
                "unavailable_count": unavailable,
                "protocol_counts": protocols,
                "channel_counts": channels,
                "region_counts": regions,
                "stats": dict(self._stats),
            }
            if self.record_registry_snapshots:
                self.memory.set(
                    _ENDPOINT_REGISTRY_HEALTH_SUMMARY_KEY,
                    payload,
                    ttl_seconds=self.snapshot_ttl_seconds,
                    source="endpoint_registry",
                )
            return payload

    def export_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            payload = {
                "generated_at": utc_timestamp(),
                "started_at": self._started_at,
                "entry_count": len(self._records),
                "stats": dict(self._stats),
                "endpoints": [record.to_dict() for record in sorted(self._records.values(), key=self._sort_record_key)],
                "history": list(self._history),
            }
            return json_safe(payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_endpoint_value(
        self,
        endpoint: str,
        *,
        endpoint_type: Optional[str],
        protocol: Optional[str],
        channel: Optional[str],
    ) -> Tuple[str, str]:
        endpoint_text = ensure_non_empty_string(str(endpoint), field_name="endpoint")
        resolved_type = str(endpoint_type or self.default_endpoint_type).strip().lower() or "network"
        if "://" in endpoint_text:
            return normalize_endpoint(endpoint_text, protocol=protocol or channel or self.default_protocol), "network"
        if self.allow_logical_targets:
            return endpoint_text, resolved_type if resolved_type != "network" else "logical"
        raise PayloadValidationError(
            "Endpoint registry requires URI-form endpoints when logical targets are disabled.",
            context={"operation": "normalize_endpoint", "protocol": protocol, "channel": channel},
            details={"endpoint": endpoint_text},
        )

    def _resolve_endpoint_id(self, endpoint_id: Optional[str], endpoint: str, protocol: str, channel: str) -> str:
        if endpoint_id is not None and str(endpoint_id).strip():
            return self._sanitize_identifier(str(endpoint_id))
        basis = {"endpoint": endpoint, "protocol": protocol, "channel": channel}
        digest = generate_idempotency_key(basis, namespace="endpoint_registry")[:16]
        endpoint_token = endpoint.replace("://", "_").replace("/", "_").replace(":", "_")
        endpoint_token = "_".join(part for part in endpoint_token.split("_") if part)[:48] or "endpoint"
        return self._sanitize_identifier(f"{channel}_{endpoint_token}_{digest}")

    def _resolve_existing_id(self, endpoint_or_id: str) -> str:
        token = ensure_non_empty_string(str(endpoint_or_id), field_name="endpoint_or_id")
        sanitized = self._sanitize_identifier(token)
        with self._lock:
            if sanitized in self._records:
                return sanitized
            for endpoint_id, record in self._records.items():
                if token == record.endpoint:
                    return endpoint_id
                if token == record.endpoint_id:
                    return endpoint_id
                try:
                    if "://" in token and normalize_endpoint(token, protocol=record.protocol) == record.endpoint:
                        return endpoint_id
                except Exception:
                    continue
        return sanitized

    def _sanitize_identifier(self, value: str) -> str:
        text = ensure_non_empty_string(value, field_name="endpoint_id").strip().lower()
        sanitized = []
        for char in text:
            sanitized.append(char if char.isalnum() or char in {"_", ".", "-"} else "_")
        normalized = "".join(sanitized).strip("._-")
        return normalized or "endpoint"

    def _supports_capabilities(self, capabilities: Mapping[str, Any], constraints: Mapping[str, Any]) -> bool:
        for key, expected in dict(constraints).items():
            if bool(capabilities.get(key, False)) != bool(expected):
                return False
        return True

    def _require_record_locked(self, endpoint_id: str) -> EndpointRecord:
        record = self._records.get(endpoint_id)
        if record is None:
            raise NoRouteAvailableError(
                "Endpoint is not registered.",
                context={"operation": "require_registered_endpoint", "endpoint": endpoint_id},
            )
        return record

    def _ensure_capacity_locked(self, *, incoming_key: str) -> None:
        if incoming_key in self._records:
            return
        if len(self._records) < self.max_registry_entries:
            return
        if not self.evict_oldest_on_capacity:
            raise ReliabilityError(
                "Endpoint registry capacity exceeded and eviction is disabled.",
                context={"operation": "endpoint_registry_capacity"},
                details={"max_registry_entries": self.max_registry_entries, "incoming_key": incoming_key},
            )
        oldest_key = min(self._records.items(), key=lambda item: item[1].updated_at)[0]
        self._records.pop(oldest_key, None)
        self._stats["evictions"] += 1
        self._append_event_locked("evict_oldest", {"evicted": oldest_key, "incoming_key": incoming_key})

    def _normalize_status(self, status: str) -> str:
        return ensure_non_empty_string(str(status), field_name="status").strip().lower()

    def _normalize_error(self, error: BaseException | Mapping[str, Any], *, endpoint: str, protocol: str, channel: str) -> Dict[str, Any]:
        if isinstance(error, Mapping):
            return json_safe(error)
        if isinstance(error, NetworkError):
            return error.to_memory_snapshot()
        return build_error_snapshot(
            error,
            operation="endpoint_registry",
            endpoint=endpoint,
            protocol=protocol,
            channel=channel,
        )

    def _sync_record_memory(self, record: EndpointRecord) -> None:
        if not self.sync_endpoint_health_to_memory:
            return
        self.memory.update_endpoint_health(
            record.endpoint,
            status=record.status,
            latency_ms=int(record.latency_ms) if record.latency_ms is not None else None,
            success_rate=record.success_rate,
            error_rate=record.error_rate,
            circuit_state=record.circuit_state,
            last_error=record.last_error,
            capabilities=record.capabilities,
            metadata={
                "endpoint_id": record.endpoint_id,
                "endpoint_type": record.endpoint_type,
                "region": record.region,
                "zone": record.zone,
                "tags": list(record.tags),
                **record.metadata,
            },
        )

    def _sync_registry_memory(self) -> None:
        if not self.record_registry_snapshots:
            return
        snapshot = self.export_snapshot()
        self.memory.set(
            _ENDPOINT_REGISTRY_SNAPSHOT_KEY,
            snapshot,
            ttl_seconds=self.snapshot_ttl_seconds,
            source="endpoint_registry",
        )
        self.memory.set(
            _ENDPOINT_REGISTRY_LAST_KEY,
            {
                "generated_at": utc_timestamp(),
                "entry_count": len(self._records),
                "stats": dict(self._stats),
            },
            ttl_seconds=self.snapshot_ttl_seconds,
            source="endpoint_registry",
        )

    def _append_event_locked(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "occurred_at": utc_timestamp(),
            "payload": sanitize_for_logging(payload) if self.sanitize_logs else json_safe(payload),
        }
        self._history.append(json_safe(event))
        if self.record_registry_snapshots:
            try:
                self.memory.append(
                    _ENDPOINT_REGISTRY_HISTORY_KEY,
                    event,
                    max_items=self.max_history_size,
                    ttl_seconds=self.history_ttl_seconds,
                    source="endpoint_registry",
                )
            except Exception:
                pass

    def _sort_record_key(self, record: EndpointRecord | Mapping[str, Any]) -> Tuple[int, float, str]:
        if isinstance(record, Mapping):
            return (-int(record.get("priority", 0) or 0), -float(record.get("health_score", 0.0) or 0.0), str(record.get("endpoint_id", "")))
        return (-int(record.priority), -float(record.health_score()), record.endpoint_id)

    def _sort_candidate_dict(self, payload: Mapping[str, Any]) -> Tuple[int, float, str]:
        return (-int(payload.get("priority", 0)), -float(payload.get("health_score", 0.0)), str(payload.get("endpoint_id", "")))

    def _get_bool_config(self, name: str, default: bool) -> bool:
        return self._coerce_bool(self.registry_config.get(name, default), default)

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        return self._coerce_int(self.registry_config.get(name, default), default, non_negative=True)

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.registry_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_status_config(self, name: str, default: str) -> str:
        return self._normalize_status(self.registry_config.get(name, default))

    def _get_sequence_config(self, name: str, default: Sequence[str]) -> Tuple[str, ...]:
        values = ensure_sequence(self.registry_config.get(name, default), field_name=name, allow_none=True, coerce_scalar=True)
        normalized: Dict[str, None] = {}
        for item in values:
            text = str(item).strip().lower()
            if text:
                normalized[text] = None
        return tuple(normalized.keys()) or tuple(default)

    def _coerce_bool(self, value: Any, default: bool) -> bool:
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
            "Invalid boolean value in endpoint registry configuration.",
            context={"operation": "endpoint_registry_config"},
            details={"config_value": value},
        )

    def _coerce_int(self, value: Any, default: int, *, non_negative: bool = False) -> int:
        if value is None:
            value = default
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in endpoint registry configuration.",
                context={"operation": "endpoint_registry_config"},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if non_negative and coerced < 0:
            raise NetworkConfigurationError(
                "Configuration integer value must be non-negative.",
                context={"operation": "endpoint_registry_config"},
                details={"config_value": value},
            )
        return coerced

    def _coerce_float(self, value: Any, default: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
        if value is None:
            value = default
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid float value in endpoint registry configuration.",
                context={"operation": "endpoint_registry_config"},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if minimum is not None:
            coerced = max(float(minimum), coerced)
        if maximum is not None:
            coerced = min(float(maximum), coerced)
        return round(coerced, 6)


class _printer_proxy:
    @staticmethod
    def status(label: str, message: str, level: str = "info") -> None:
        try:
            printer.status(label, message, level)
        except Exception:
            print(f"[{label}] {message}")


if __name__ == "__main__":
    print("\n=== Running Endpoint Registry ===\n")
    _printer_proxy.status("TEST", "Endpoint Registry initialized", "info")

    memory = NetworkMemory()
    registry = EndpointRegistry(memory=memory)

    http_ep = registry.register_endpoint(
        "https://api.example.com/v1/jobs",
        protocol="https",
        channel="http",
        endpoint_id="api_primary",
        region="eu-west",
        priority=120,
        weight=1.0,
        secure=True,
        capabilities={"supports_request_reply": True, "supports_headers": True},
        tags=["primary", "public"],
        metadata={"role": "relay_api"},
    )
    _printer_proxy.status("TEST", "HTTP endpoint registered", "info")

    ws_ep = registry.register_endpoint(
        "wss://stream.example.com/events",
        protocol="websocket",
        channel="websocket",
        endpoint_id="stream_primary",
        region="eu-west",
        priority=110,
        secure=True,
        capabilities={"supports_streaming": True, "supports_bidirectional_streaming": True},
        tags=["streaming", "primary"],
    )
    _printer_proxy.status("TEST", "WebSocket endpoint registered", "info")

    queue_ep = registry.register_endpoint(
        "jobs.primary",
        protocol="queue",
        channel="queue",
        endpoint_type="logical",
        endpoint_id="queue_primary",
        region="eu-west",
        priority=90,
        capabilities={"supports_ack": True, "supports_nack": True, "supports_batch_send": True},
        tags=["durable", "broker"],
    )
    _printer_proxy.status("TEST", "Queue endpoint registered", "info")

    registry.update_health(
        "api_primary",
        status="healthy",
        latency_ms=42.0,
        success_rate=0.991,
        error_rate=0.009,
        metadata={"window": "5m"},
    )
    registry.mark_success("api_primary", latency_ms=38.0, metadata={"path": "/v1/jobs"})
    registry.mark_failure("stream_primary", TimeoutError("elevated ping timeout"), status="degraded", metadata={"window": "5m"})
    registry.set_circuit_state("stream_primary", "half_open", metadata={"reason": "timeout spike"})
    _printer_proxy.status("TEST", "Endpoint health updated", "info")

    http_candidates = registry.get_candidates(protocol="https", channel="http", include_unavailable=False)
    stream_candidates = registry.get_candidates(channel="websocket", include_degraded=True)
    summary = registry.get_health_summary()
    snapshot = registry.export_snapshot()

    print("HTTP Endpoint:", stable_json_dumps(http_ep))
    print("WebSocket Endpoint:", stable_json_dumps(ws_ep))
    print("Queue Endpoint:", stable_json_dumps(queue_ep))
    print("HTTP Candidates:", stable_json_dumps(http_candidates))
    print("Stream Candidates:", stable_json_dumps(stream_candidates))
    print("Health Summary:", stable_json_dumps(summary))
    print("Snapshot Summary:", stable_json_dumps({"entry_count": snapshot["entry_count"], "stats": snapshot["stats"]}))

    assert registry.get_endpoint("api_primary") is not None
    assert registry.get_endpoint("https://api.example.com:443/v1/jobs") is not None
    assert len(http_candidates) == 1
    assert stream_candidates[0]["status"] == "degraded"
    assert summary["entry_count"] == 3
    assert memory.get("network.endpoint.health") is not None
    assert memory.get(_ENDPOINT_REGISTRY_SNAPSHOT_KEY) is not None

    _printer_proxy.status("TEST", "All Endpoint Registry checks passed", "info")
    print("\n=== Test ran successfully ===\n")
