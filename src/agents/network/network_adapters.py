"""
Protocol adapters (e.g., HTTP(S), WebSocket, gRPC, queue-based channels).
Common adapter contract:
    - connect()
    - send(payload)
    - recv()
    - ack() / nack()
    - close()
Adapter health scoring and compatibility metadata.

This module is the registry and selection facade for specialized network
adapters. It is responsible for:
- registering adapter implementations and their compatibility metadata,
- resolving adapters by name, protocol, channel, and endpoint posture,
- instantiating and optionally reusing adapter instances,
- exposing active-adapter state and registry snapshots,
- delegating generic connect/send/recv/ack/nack/close flows to the chosen
  adapter while preserving shared memory and structured error semantics.

It intentionally does not own routing strategy, retry policy, circuit
breaking, or transport-specific logic. Those remain in the routing,
reliability, and specialized adapter modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock, Thread
from time import monotonic, sleep
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

from .utils import *
from .adapters import *
from .network_memory import NetworkMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Network Adapters")
printer = PrettyPrinter()


_ADAPTER_REGISTRY_KEY = "network.adapters.registry"
_ADAPTER_ACTIVE_KEY = "network.adapters.active"
_ADAPTER_LAST_SELECTION_KEY = "network.adapters.last_selection"
_ADAPTER_EVENT_HISTORY_KEY = "network.adapters.events"

_DEFAULT_PRIORITY_ORDER = ("http", "websocket", "grpc", "queue")
_VALID_SELECTION_STRATEGIES = {"priority", "first_match"}
_DEFAULT_CAPABILITY_KEYS = (
    "supports_streaming",
    "supports_bidirectional_streaming",
    "supports_ack",
    "supports_nack",
    "supports_batch_send",
    "supports_headers",
    "supports_tls",
    "supports_reconnect",
    "supports_receive",
    "supports_request_reply",
)


@dataclass(slots=True)
class AdapterRegistrySpec:
    """Registry metadata for a specialized adapter implementation."""

    name: str
    adapter_cls: Type[Any]
    class_name: str
    protocols: Tuple[str, ...]
    channels: Tuple[str, ...]
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    priority: int = 100
    enabled: bool = True
    config_section: Optional[str] = None
    reuse_instances: bool = True
    capabilities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_name(self, value: Optional[str]) -> bool:
        if value is None:
            return False
        token = str(value).strip().lower()
        return token in {self.name, self.class_name.lower(), *self.aliases}

    def matches_protocol(self, value: Optional[str]) -> bool:
        if value is None:
            return False
        token = str(value).strip().lower()
        normalized = normalize_protocol_name(token)
        return token in self.protocols or normalized in self.protocols

    def matches_channel(self, value: Optional[str]) -> bool:
        if value is None:
            return False
        token = str(value).strip().lower()
        normalized = normalize_channel_name(token)
        return token in self.channels or normalized in self.channels

    def supports_constraints(self, constraints: Optional[Mapping[str, Any]]) -> bool:
        if not constraints:
            return True
        for key, expected in dict(constraints).items():
            if key in _DEFAULT_CAPABILITY_KEYS:
                actual = bool(self.capabilities.get(key, False))
                if bool(expected) != actual:
                    return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "protocols": list(self.protocols),
            "channels": list(self.channels),
            "aliases": list(self.aliases),
            "priority": self.priority,
            "enabled": self.enabled,
            "config_section": self.config_section,
            "reuse_instances": self.reuse_instances,
            "capabilities": json_safe(self.capabilities),
            "metadata": json_safe(self.metadata),
        }


@dataclass(slots=True)
class ManagedAdapterRecord:
    """Bookkeeping for an instantiated adapter managed by the registry facade."""

    cache_key: str
    spec_name: str
    endpoint: Optional[str]
    adapter: Any
    created_at: datetime
    last_used_at: datetime
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_used_at = _utcnow()
        self.usage_count += 1

    def to_dict(self) -> Dict[str, Any]:
        adapter_snapshot = None
        if hasattr(self.adapter, "get_state_snapshot"):
            try:
                adapter_snapshot = self.adapter.get_state_snapshot()
            except Exception as exc:  # noqa: BLE001 - log-safe snapshot fallback.
                adapter_snapshot = {"snapshot_error": repr(exc)}
        return {
            "cache_key": self.cache_key,
            "spec_name": self.spec_name,
            "endpoint": self.endpoint,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
            "usage_count": self.usage_count,
            "adapter_state": sanitize_for_logging(adapter_snapshot),
            "metadata": sanitize_for_logging(self.metadata),
        }


class NetworkAdapters:
    """
    Adapter registry + selection facade bridging adapters/ implementations.

    The class provides a stable operational layer above the specialized
    adapter classes. It registers supported adapters, selects the best fit for
    a request, optionally reuses instances, and exposes common delegation
    helpers for connect/send/receive/ack/nack/close.
    """

    def __init__(self, memory: Optional[NetworkMemory] = None) -> None:
        self.config = load_global_config()
        self.adapters_config = get_config_section("network_adapters") or {}
        self.memory = memory or NetworkMemory()
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.auto_register_defaults = self._get_bool_config("auto_register_defaults", True)
        self.reuse_adapter_instances = self._get_bool_config("reuse_adapter_instances", True)
        self.close_replaced_instances = self._get_bool_config("close_replaced_instances", True)
        self.record_registry_snapshots = self._get_bool_config("record_registry_snapshots", True)
        self.require_registered_adapter = self._get_bool_config("require_registered_adapter", True)
        self.prefer_connected_instances = self._get_bool_config("prefer_connected_instances", True)
        self.prefer_secure_for_tls_endpoints = self._get_bool_config("prefer_secure_for_tls_endpoints", True)

        self.default_adapter = self._get_optional_string_config("default_adapter") or "http"
        self.default_protocol = normalize_protocol_name(self._get_optional_string_config("default_protocol") or "http")
        self.default_channel = normalize_channel_name(self._get_optional_string_config("default_channel") or self.default_protocol)
        self.selection_strategy = self._get_selection_strategy_config("selection_strategy", "priority")

        self.max_active_adapters = max(1, self._get_non_negative_int_config("max_active_adapters", 64))
        self.registry_snapshot_ttl_seconds = self._get_non_negative_int_config("registry_snapshot_ttl_seconds", 1800)
        self.active_snapshot_ttl_seconds = self._get_non_negative_int_config("active_snapshot_ttl_seconds", 600)
        self.event_history_max = max(1, self._get_non_negative_int_config("event_history_max", 250))

        configured_priority_order = self._get_sequence_config("priority_order", _DEFAULT_PRIORITY_ORDER)
        self.priority_order = tuple(configured_priority_order) or _DEFAULT_PRIORITY_ORDER

        self._registry: Dict[str, AdapterRegistrySpec] = {}
        self._active_adapters: Dict[str, ManagedAdapterRecord] = {}
        self._selection_counts: Dict[str, int] = {}
        self._stats: Dict[str, int] = {
            "registrations": 0,
            "selections": 0,
            "creations": 0,
            "reuses": 0,
            "closes": 0,
            "close_all": 0,
            "delegated_connects": 0,
            "delegated_sends": 0,
            "delegated_receives": 0,
            "delegated_acks": 0,
            "delegated_nacks": 0,
        }
        self._started_at = _utcnow()

        if self.auto_register_defaults:
            self._register_default_adapters()
        self._sync_registry_memory()

    # ------------------------------------------------------------------
    # Registration and discovery
    # ------------------------------------------------------------------
    def register_adapter(self, name: str, adapter_cls: Type[Any], *,
        protocols: Sequence[str],
        channels: Sequence[str],
        aliases: Optional[Sequence[str]] = None,
        priority: int = 100,
        enabled: bool = True,
        config_section: Optional[str] = None,
        reuse_instances: Optional[bool] = None,
        capabilities: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_name = ensure_non_empty_string(name, field_name="adapter_name").strip().lower()
        if not isinstance(adapter_cls, type):
            raise AdapterInitializationError(
                "Adapter registry entry requires a class type.",
                context={"operation": "register_adapter", "channel": self.default_channel, "protocol": self.default_protocol},
                details={"adapter_name": normalized_name, "received_type": type(adapter_cls).__name__},
            )

        normalized_protocols = self._normalize_registry_tokens(protocols, kind="protocol")
        normalized_channels = self._normalize_registry_tokens(channels, kind="channel")
        normalized_aliases = tuple(
            dict.fromkeys(
                str(item).strip().lower()
                for item in ensure_sequence(aliases, field_name="aliases", allow_none=True, coerce_scalar=True)
                if str(item).strip()
            )
        )
        capability_map = self._normalize_capabilities(capabilities, config_section=config_section)
        spec = AdapterRegistrySpec(
            name=normalized_name,
            adapter_cls=adapter_cls,
            class_name=adapter_cls.__name__,
            protocols=normalized_protocols,
            channels=normalized_channels,
            aliases=normalized_aliases,
            priority=int(priority),
            enabled=bool(enabled),
            config_section=config_section,
            reuse_instances=self.reuse_adapter_instances if reuse_instances is None else bool(reuse_instances),
            capabilities=capability_map,
            metadata=normalize_metadata(metadata),
        )

        with self._lock:
            self._registry[normalized_name] = spec
            self._stats["registrations"] += 1
            self._append_event(
                "register_adapter",
                {
                    "adapter": spec.to_dict(),
                },
            )
            self._sync_registry_memory()
            return spec.to_dict()

    def unregister_adapter(self, name: str, *, close_instances: bool = True) -> bool:
        normalized_name = ensure_non_empty_string(name, field_name="adapter_name").strip().lower()
        with self._lock:
            spec = self._registry.pop(normalized_name, None)
            if spec is None:
                return False
            if close_instances:
                keys_to_close = [key for key, record in self._active_adapters.items() if record.spec_name == normalized_name]
                for cache_key in keys_to_close:
                    self._close_record_locked(cache_key, reason="adapter_unregistered")
            self._append_event("unregister_adapter", {"adapter_name": normalized_name})
            self._sync_registry_memory()
            return True

    def list_registered_adapters(self, *, enabled_only: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            specs = sorted(self._registry.values(), key=self._sort_spec_key)
            if enabled_only:
                specs = [spec for spec in specs if spec.enabled]
            return [spec.to_dict() for spec in specs]

    def get_registry_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "enabled": self.enabled,
                "default_adapter": self.default_adapter,
                "default_protocol": self.default_protocol,
                "default_channel": self.default_channel,
                "selection_strategy": self.selection_strategy,
                "registry_size": len(self._registry),
                "active_size": len(self._active_adapters),
                "priority_order": list(self.priority_order),
                "selection_counts": dict(self._selection_counts),
                "stats": dict(self._stats),
                "adapters": [spec.to_dict() for spec in sorted(self._registry.values(), key=self._sort_spec_key)],
            }

    def list_active_adapters(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [record.to_dict() for record in sorted(self._active_adapters.values(), key=lambda item: item.cache_key)]

    # ------------------------------------------------------------------
    # Selection and instantiation
    # ------------------------------------------------------------------
    def select_adapter(
        self,
        *,
        name: Optional[str] = None,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        spec = self._select_spec(
            name=name,
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            constraints=constraints,
            exclude=exclude,
        )
        with self._lock:
            self._stats["selections"] += 1
            self._selection_counts[spec.name] = self._selection_counts.get(spec.name, 0) + 1
        selection_snapshot = {
            "selected_adapter": spec.name,
            "protocol": protocol,
            "channel": channel,
            "endpoint": self._safe_endpoint(endpoint),
            "constraints": sanitize_for_logging(constraints) if self.sanitize_logs else json_safe(constraints),
            "selected_at": utc_timestamp(),
            "adapter": spec.to_dict(),
        }
        self._record_selection(selection_snapshot)
        return selection_snapshot

    def create_adapter(
        self,
        *,
        name: Optional[str] = None,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        reuse_instance: Optional[bool] = None,
        exclude: Optional[Sequence[str]] = None,
        **adapter_kwargs: Any,
    ) -> Any:
        if not self.enabled:
            raise AdapterInitializationError(
                "NetworkAdapters is disabled by configuration.",
                context={"operation": "create_adapter", "channel": channel, "protocol": protocol, "endpoint": endpoint},
            )

        spec = self._select_spec(
            name=name,
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            constraints=constraints,
            exclude=exclude,
        )
        resolved_reuse = spec.reuse_instances if reuse_instance is None else bool(reuse_instance)
        resolved_config = merge_mappings(
            get_config_section(spec.config_section) if spec.config_section else {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        cache_key = self._build_cache_key(spec.name, endpoint=endpoint, protocol=protocol, channel=channel, config=resolved_config, adapter_kwargs=adapter_kwargs)

        with self._lock:
            if resolved_reuse and cache_key in self._active_adapters:
                record = self._active_adapters[cache_key]
                record.touch()
                self._stats["reuses"] += 1
                self._sync_active_memory()
                return record.adapter
            self._evict_if_needed_locked(incoming_key=cache_key)

        init_kwargs = dict(adapter_kwargs)
        if "memory" not in init_kwargs:
            init_kwargs["memory"] = self.memory
        if "config" not in init_kwargs:
            init_kwargs["config"] = resolved_config
        if endpoint is not None and "endpoint" not in init_kwargs:
            init_kwargs["endpoint"] = endpoint
        if protocol is not None and "protocol" not in init_kwargs:
            init_kwargs["protocol"] = protocol

        try:
            adapter = spec.adapter_cls(**init_kwargs)
        except NetworkError:
            raise
        except Exception as exc:
            raise normalize_network_exception(
                exc,
                operation="create_adapter",
                endpoint=endpoint,
                channel=channel or spec.channels[0],
                protocol=protocol or spec.protocols[0],
                metadata={"adapter_name": spec.name, "adapter_class": spec.class_name},
            ) from exc

        with self._lock:
            record = ManagedAdapterRecord(
                cache_key=cache_key,
                spec_name=spec.name,
                endpoint=self._safe_endpoint(endpoint),
                adapter=adapter,
                created_at=_utcnow(),
                last_used_at=_utcnow(),
                usage_count=1,
                metadata={"requested_protocol": protocol, "requested_channel": channel},
            )
            if resolved_reuse:
                self._active_adapters[cache_key] = record
            self._stats["creations"] += 1
            self._append_event(
                "create_adapter",
                {
                    "adapter_name": spec.name,
                    "cache_key": cache_key,
                    "endpoint": self._safe_endpoint(endpoint),
                    "reuse_instance": resolved_reuse,
                },
            )
            self._sync_active_memory()
            return adapter

    def get_adapter(self, *, cache_key: str) -> Optional[Any]:
        normalized_key = ensure_non_empty_string(cache_key, field_name="cache_key")
        with self._lock:
            record = self._active_adapters.get(normalized_key)
            if record is None:
                return None
            record.touch()
            self._sync_active_memory()
            return record.adapter

    # ------------------------------------------------------------------
    # Delegated facade operations
    # ------------------------------------------------------------------
    def connect(self, *, endpoint: Optional[str] = None, timeout_ms: Optional[Any] = None,
                metadata: Optional[Mapping[str, Any]] = None, **selection_kwargs: Any) -> Dict[str, Any]:
        adapter = self.create_adapter(endpoint=endpoint, **selection_kwargs)
        self._stats["delegated_connects"] += 1
        return adapter.connect(endpoint=endpoint, timeout_ms=timeout_ms, metadata=metadata)

    def send(self, payload: Any, *, envelope: Optional[Mapping[str, Any]] = None,
             timeout_ms: Optional[Any] = None, metadata: Optional[Mapping[str, Any]] = None,
             content_type: Optional[str] = None, endpoint: Optional[str] = None, auto_connect: bool = True,
             **selection_kwargs: Any) -> Dict[str, Any]:
        adapter = self.create_adapter(endpoint=endpoint, **selection_kwargs)
        self._stats["delegated_sends"] += 1
        if auto_connect and hasattr(adapter, "is_connected") and not adapter.is_connected():
            adapter.connect(endpoint=endpoint, timeout_ms=timeout_ms, metadata=metadata)
        return adapter.send(payload, envelope=envelope, timeout_ms=timeout_ms, metadata=metadata, content_type=content_type)

    def recv(self, *, timeout_ms: Optional[Any] = None, metadata: Optional[Mapping[str, Any]] = None,
             endpoint: Optional[str] = None, auto_connect: bool = False, **selection_kwargs: Any) -> Dict[str, Any]:
        adapter = self.create_adapter(endpoint=endpoint, **selection_kwargs)
        self._stats["delegated_receives"] += 1
        if auto_connect and hasattr(adapter, "is_connected") and not adapter.is_connected():
            adapter.connect(endpoint=endpoint, timeout_ms=timeout_ms, metadata=metadata)
        return adapter.recv(timeout_ms=timeout_ms, metadata=metadata)

    def ack(self, message: str | Mapping[str, Any], *, correlation_id: Optional[str] = None,
            metadata: Optional[Mapping[str, Any]] = None, endpoint: Optional[str] = None, **selection_kwargs: Any) -> Dict[str, Any]:
        adapter = self.create_adapter(endpoint=endpoint, **selection_kwargs)
        self._stats["delegated_acks"] += 1
        return adapter.ack(message, correlation_id=correlation_id, metadata=metadata)

    def nack(self, message: str | Mapping[str, Any], *, reason: Optional[str] = None,
             correlation_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None,
             endpoint: Optional[str] = None, **selection_kwargs: Any) -> Dict[str, Any]:
        adapter = self.create_adapter(endpoint=endpoint, **selection_kwargs)
        self._stats["delegated_nacks"] += 1
        return adapter.nack(message, reason=reason, correlation_id=correlation_id, metadata=metadata)

    def close(self, *, name: Optional[str] = None, protocol: Optional[str] = None, channel: Optional[str] = None,
              endpoint: Optional[str] = None, reason: Optional[str] = None) -> int:
        closed = 0
        with self._lock:
            targets = [
                key for key, record in self._active_adapters.items()
                if (name is None or record.spec_name == str(name).strip().lower())
                and (endpoint is None or record.endpoint == self._safe_endpoint(endpoint))
                and (protocol is None or self._registry[record.spec_name].matches_protocol(protocol))
                and (channel is None or self._registry[record.spec_name].matches_channel(channel))
            ]
            for cache_key in targets:
                if self._close_record_locked(cache_key, reason=reason):
                    closed += 1
            if closed:
                self._sync_active_memory()
        return closed

    def close_all(self, *, reason: Optional[str] = None) -> int:
        with self._lock:
            keys = list(self._active_adapters.keys())
            closed = 0
            for cache_key in keys:
                if self._close_record_locked(cache_key, reason=reason or "close_all"):
                    closed += 1
            self._stats["close_all"] += 1
            self._sync_active_memory()
            return closed

    def get_active_health(self) -> Dict[str, Any]:
        with self._lock:
            payload: Dict[str, Any] = {}
            for cache_key, record in self._active_adapters.items():
                adapter = record.adapter
                if hasattr(adapter, "get_health_snapshot"):
                    payload[cache_key] = adapter.get_health_snapshot()
                else:
                    payload[cache_key] = {"adapter": record.spec_name, "endpoint": record.endpoint}
            return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_default_adapters(self) -> None:
        self._register_default(
            name="http",
            cls_ref = HTTPAdapter,
            protocols=("http", "https"),
            channels=("http",),
            aliases=("rest",),
            priority=self._priority_for("http", 100),
            config_section="network_http_adapter",
        )
        self._register_default(
            name="websocket",
            cls_ref = WebSocketAdapter,
            protocols=("websocket", "ws", "wss"),
            channels=("websocket",),
            aliases=("ws", "wss", "socket"),
            priority=self._priority_for("websocket", 90),
            config_section="network_websocket_adapter",
        )
        self._register_default(
            name="grpc",
            cls_ref = GRPCAdapter,
            protocols=("grpc", "grpcs"),
            channels=("grpc",),
            aliases=("rpc",),
            priority=self._priority_for("grpc", 80),
            config_section="network_grpc_adapter",
        )
        self._register_default(
            name="queue",
            cls_ref = QueueAdapter,
            protocols=("queue", "amqp", "amqps", "sqs", "kafka", "pubsub", "mq"),
            channels=("queue",),
            aliases=("broker", "mq"),
            priority=self._priority_for("queue", 70),
            config_section="network_queue_adapter",
        )

    def _register_default(
        self,
        *,
        name: str,
        cls_ref: Any,
        protocols: Sequence[str],
        channels: Sequence[str],
        aliases: Sequence[str],
        priority: int,
        config_section: str,
    ) -> None:
        if cls_ref is None:
            return
        adapter_defaults = ensure_mapping(self.adapters_config.get("adapter_defaults"), field_name="adapter_defaults", allow_none=True)
        per_adapter = ensure_mapping(adapter_defaults.get(name), field_name=f"adapter_defaults.{name}", allow_none=True)
        enabled = self._coerce_bool(per_adapter.get("enabled"), True)
        reuse_instances = self._coerce_bool(per_adapter.get("reuse_instances"), self.reuse_adapter_instances)
        self.register_adapter(
            name=name,
            adapter_cls=cls_ref,
            protocols=per_adapter.get("protocols") or protocols,
            channels=per_adapter.get("channels") or channels,
            aliases=per_adapter.get("aliases") or aliases,
            priority=self._coerce_int(per_adapter.get("priority"), priority),
            enabled=enabled,
            config_section=str(per_adapter.get("config_section") or config_section),
            reuse_instances=reuse_instances,
            capabilities=per_adapter,
            metadata={"default_registration": True},
        )

    def _select_spec(
        self,
        *,
        name: Optional[str],
        protocol: Optional[str],
        channel: Optional[str],
        endpoint: Optional[str],
        constraints: Optional[Mapping[str, Any]],
        exclude: Optional[Sequence[str]],
    ) -> AdapterRegistrySpec:
        excluded = {str(item).strip().lower() for item in ensure_sequence(exclude, field_name="exclude", allow_none=True, coerce_scalar=True) if str(item).strip()}

        requested_name = str(name).strip().lower() if name is not None else None
        requested_protocol = str(protocol).strip().lower() if protocol is not None else None
        requested_channel = str(channel).strip().lower() if channel is not None else None

        secure_endpoint = False
        if endpoint:
            try:
                secure_endpoint = bool(parse_endpoint(endpoint, default_scheme=self.default_protocol, protocol=protocol, require_host=False).secure)
            except Exception:
                secure_endpoint = False

        with self._lock:
            candidates = [spec for spec in self._registry.values() if spec.enabled and spec.name not in excluded]

        if not candidates and self.require_registered_adapter:
            raise AdapterNotFoundError(
                "No registered adapters are available.",
                context={"operation": "select_adapter", "channel": channel, "protocol": protocol, "endpoint": endpoint},
            )

        scored: List[Tuple[Tuple[int, int, int, int, int], AdapterRegistrySpec]] = []
        for spec in candidates:
            if requested_name and not spec.matches_name(requested_name):
                continue
            if requested_protocol and not spec.matches_protocol(requested_protocol):
                continue
            if requested_channel and not spec.matches_channel(requested_channel):
                continue
            if not spec.supports_constraints(constraints):
                continue

            name_score = 200 if requested_name and spec.matches_name(requested_name) else 0
            protocol_score = 120 if requested_protocol and spec.matches_protocol(requested_protocol) else 0
            channel_score = 100 if requested_channel and spec.matches_channel(requested_channel) else 0
            secure_score = 25 if secure_endpoint and self.prefer_secure_for_tls_endpoints and self._spec_supports_secure_endpoint(spec) else 0
            connected_score = 50 if self.prefer_connected_instances and self._has_connected_instance(spec.name, endpoint) else 0
            priority_score = int(spec.priority)

            if self.selection_strategy == "first_match":
                priority_score = 0
            scored.append(((name_score, protocol_score, channel_score, secure_score, connected_score + priority_score), spec))

        if not scored:
            raise NoRouteAvailableError(
                "No compatible adapter was found for the requested transport constraints.",
                context={"operation": "select_adapter", "channel": channel, "protocol": protocol, "endpoint": endpoint},
                details={
                    "requested_name": requested_name,
                    "requested_protocol": requested_protocol,
                    "requested_channel": requested_channel,
                    "constraints": sanitize_for_logging(constraints) if self.sanitize_logs else json_safe(constraints),
                    "exclude": list(excluded),
                },
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _has_connected_instance(self, spec_name: str, endpoint: Optional[str]) -> bool:
        normalized_endpoint = self._safe_endpoint(endpoint)
        for record in self._active_adapters.values():
            if record.spec_name != spec_name:
                continue
            if normalized_endpoint is not None and record.endpoint != normalized_endpoint:
                continue
            adapter = record.adapter
            if hasattr(adapter, "is_connected") and adapter.is_connected():
                return True
        return False

    def _spec_supports_secure_endpoint(self, spec: AdapterRegistrySpec) -> bool:
        name = spec.name
        protocols = set(spec.protocols)
        if name == "http":
            return "https" in protocols or bool(spec.capabilities.get("supports_tls", False))
        if name == "websocket":
            return "wss" in protocols or bool(spec.capabilities.get("supports_tls", False))
        if name == "grpc":
            return "grpcs" in protocols or bool(spec.capabilities.get("supports_tls", False))
        return bool(spec.capabilities.get("supports_tls", False))

    def _build_cache_key(
        self,
        adapter_name: str,
        *,
        endpoint: Optional[str],
        protocol: Optional[str],
        channel: Optional[str],
        config: Mapping[str, Any],
        adapter_kwargs: Mapping[str, Any],
    ) -> str:
        key_payload = {
            "adapter_name": adapter_name,
            "endpoint": self._safe_endpoint(endpoint),
            "protocol": protocol,
            "channel": channel,
            "config": json_safe(config),
            "adapter_kwargs": json_safe(adapter_kwargs),
        }
        digest = generate_idempotency_key(key_payload, namespace="network_adapters")
        return f"{adapter_name}:{digest[:24]}"

    def _evict_if_needed_locked(self, *, incoming_key: str) -> None:
        if incoming_key in self._active_adapters:
            return
        if len(self._active_adapters) < self.max_active_adapters:
            return
        oldest_key = min(self._active_adapters.items(), key=lambda item: item[1].last_used_at)[0]
        self._close_record_locked(oldest_key, reason="adapter_cache_evict")

    def _close_record_locked(self, cache_key: str, *, reason: Optional[str]) -> bool:
        record = self._active_adapters.pop(cache_key, None)
        if record is None:
            return False
        adapter = record.adapter
        try:
            if hasattr(adapter, "close"):
                adapter.close(reason=reason)
        except Exception as exc:  # noqa: BLE001 - cleanup path should not block cache maintenance.
            self._append_event(
                "close_error",
                {"cache_key": cache_key, "reason": reason, "error": build_error_snapshot(exc, operation="close_adapter_record")},
            )
        self._stats["closes"] += 1
        self._append_event("close_adapter", {"cache_key": cache_key, "reason": reason, "spec_name": record.spec_name})
        return True

    def _record_selection(self, snapshot: Mapping[str, Any]) -> None:
        with self._lock:
            self._append_event("select_adapter", snapshot)
            if self.record_registry_snapshots:
                self.memory.set(
                    _ADAPTER_LAST_SELECTION_KEY,
                    dict(snapshot),
                    ttl_seconds=self.active_snapshot_ttl_seconds,
                    source="network_adapters",
                )
                self._sync_registry_memory()

    def _sync_registry_memory(self) -> None:
        if not self.record_registry_snapshots:
            return
        self.memory.set(
            _ADAPTER_REGISTRY_KEY,
            self.get_registry_snapshot(),
            ttl_seconds=self.registry_snapshot_ttl_seconds,
            source="network_adapters",
        )
        self._sync_active_memory()

    def _sync_active_memory(self) -> None:
        if not self.record_registry_snapshots:
            return
        self.memory.set(
            _ADAPTER_ACTIVE_KEY,
            {
                "generated_at": utc_timestamp(),
                "active_count": len(self._active_adapters),
                "adapters": [record.to_dict() for record in sorted(self._active_adapters.values(), key=lambda item: item.cache_key)],
            },
            ttl_seconds=self.active_snapshot_ttl_seconds,
            source="network_adapters",
        )

    def _append_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "occurred_at": utc_timestamp(),
            "payload": sanitize_for_logging(payload) if self.sanitize_logs else json_safe(payload),
        }
        try:
            self.memory.append(
                _ADAPTER_EVENT_HISTORY_KEY,
                event,
                max_items=self.event_history_max,
                ttl_seconds=self.registry_snapshot_ttl_seconds,
                source="network_adapters",
            )
        except Exception:
            # Memory append failure should not prevent primary adapter orchestration.
            pass

    def _normalize_registry_tokens(self, values: Sequence[str], *, kind: str) -> Tuple[str, ...]:
        normalized: Dict[str, None] = {}
        for value in ensure_sequence(values, field_name=kind, allow_none=False, coerce_scalar=True):
            token = ensure_non_empty_string(str(value), field_name=kind).strip().lower()
            if kind == "protocol":
                normalized[token] = None
                normalized[normalize_protocol_name(token)] = None
            else:
                normalized[token] = None
                normalized[normalize_channel_name(token)] = None
        return tuple(normalized.keys())

    def _normalize_capabilities(self, capabilities: Optional[Mapping[str, Any]], *, config_section: Optional[str]) -> Dict[str, Any]:
        base = ensure_mapping(get_config_section(config_section) if config_section else {}, field_name="config_section", allow_none=True)
        incoming = ensure_mapping(capabilities, field_name="capabilities", allow_none=True)
        payload = merge_mappings(base, incoming)
        return {key: payload.get(key) for key in _DEFAULT_CAPABILITY_KEYS if key in payload} | {
            "auth_modes": payload.get("auth_modes"),
            "content_types": payload.get("content_types"),
            "capabilities_metadata": payload.get("capabilities_metadata"),
        }

    def _safe_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        if endpoint is None:
            return None
        try:
            if "://" in str(endpoint):
                return normalize_endpoint(str(endpoint))
        except Exception:
            return str(endpoint)
        return str(endpoint)

    def _priority_for(self, adapter_name: str, default: int) -> int:
        try:
            index = list(self.priority_order).index(adapter_name)
            return max(1, default + (len(self.priority_order) - index) * 10)
        except ValueError:
            return default

    def _sort_spec_key(self, spec: AdapterRegistrySpec) -> Tuple[int, str]:
        return (-int(spec.priority), spec.name)

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.adapters_config.get(name, default)
        return self._coerce_bool(value, default)

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.adapters_config.get(name, default)
        return self._coerce_int(value, default, non_negative=True)

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.adapters_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_selection_strategy_config(self, name: str, default: str) -> str:
        value = str(self.adapters_config.get(name, default)).strip().lower() or default
        if value not in _VALID_SELECTION_STRATEGIES:
            raise NetworkConfigurationError(
                "Invalid selection strategy in network adapters configuration.",
                context={"operation": "network_adapters_config", "channel": self.default_channel, "protocol": self.default_protocol},
                details={"config_key": name, "config_value": value},
            )
        return value

    def _get_sequence_config(self, name: str, default: Sequence[str]) -> Tuple[str, ...]:
        value = self.adapters_config.get(name, default)
        values = ensure_sequence(value, field_name=name, allow_none=True, coerce_scalar=True)
        normalized: Dict[str, None] = {}
        for item in values:
            text = str(item).strip()
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
            "Invalid boolean value in network adapters configuration.",
            context={"operation": "network_adapters_config", "channel": self.default_channel, "protocol": self.default_protocol},
            details={"config_value": value},
        )

    def _coerce_int(self, value: Any, default: int, *, non_negative: bool = False) -> int:
        if value is None:
            value = default
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in network adapters configuration.",
                context={"operation": "network_adapters_config", "channel": self.default_channel, "protocol": self.default_protocol},
                details={"config_value": value},
                cause=exc,
            ) from exc
        if non_negative and coerced < 0:
            raise NetworkConfigurationError(
                "Configuration integer value must be non-negative.",
                context={"operation": "network_adapters_config", "channel": self.default_channel, "protocol": self.default_protocol},
                details={"config_value": value},
            )
        return coerced


class _DemoHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _write(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        body = stable_json_dumps({"ok": True, "path": self.path, "method": "GET"}).encode("utf-8")
        self._write(200, body)

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(content_length) if content_length > 0 else b""
        body = stable_json_dumps({"ok": True, "payload": raw.decode("utf-8"), "method": "POST"}).encode("utf-8")
        self._write(200, body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _DemoHTTPServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.server = ThreadingHTTPServer((host, port), _DemoHTTPHandler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def host(self) -> str:
        return str(self.server.server_address[0])

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def start(self) -> None:
        self.thread.start()
        sleep(0.15)

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class _DemoGRPCTransport:
    def __init__(self) -> None:
        self.connected = False
        self.messages: List[Dict[str, Any]] = []

    def connect(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.connected = True
        return {"connected": True, "endpoint": endpoint, "timeout_ms": timeout_ms, "metadata": dict(metadata)}

    def unary_unary(self, *, method: str, request: Any, timeout_ms: int, metadata: Mapping[str, Any]) -> Any:
        payload = {"method": method, "request": request, "timeout_ms": timeout_ms, "metadata": dict(metadata)}
        self.messages.append(payload)
        return {"kind": "unary", **payload}

    def unary_stream(self, *, method: str, request: Any, timeout_ms: int, metadata: Mapping[str, Any]) -> Iterable[Any]:
        yield {"kind": "stream", "index": 1, "method": method, "request": request}
        yield {"kind": "stream", "index": 2, "method": method, "request": request}

    def stream_unary(self, *, method: str, request: Sequence[Any], timeout_ms: int, metadata: Mapping[str, Any]) -> Any:
        return {"kind": "stream_unary", "method": method, "items": list(request)}

    def stream_stream(self, *, method: str, request: Sequence[Any], timeout_ms: int, metadata: Mapping[str, Any]) -> Iterable[Any]:
        for idx, item in enumerate(request, start=1):
            yield {"kind": "stream_stream", "index": idx, "method": method, "item": item}

    def recv(self, *, timeout_ms: int, metadata: Mapping[str, Any]) -> Any:
        return {"kind": "recv", "timeout_ms": timeout_ms, "metadata": dict(metadata)}

    def ack(self, *, message_id: str, correlation_id: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return {"acknowledged": True, "message_id": message_id, "correlation_id": correlation_id}

    def nack(self, *, message_id: str, correlation_id: Optional[str], reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return {"nacked": True, "message_id": message_id, "correlation_id": correlation_id, "reason": reason}

    def close(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.connected = False
        return {"closed": True, "reason": reason}


class _DemoQueueTransport:
    def __init__(self) -> None:
        self.connected = False
        self.queues: Dict[str, List[Dict[str, Any]]] = {}

    def connect(self, *, endpoint: str, timeout_ms: int, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.connected = True
        return {"connected": True, "endpoint": endpoint}

    def declare(self, *, queue_name: str, durable: bool, exclusive: bool, auto_delete: bool, metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.queues.setdefault(queue_name, [])
        return {"declared": True, "queue_name": queue_name}

    def publish(self, *, queue_name: str, payload: bytes, timeout_ms: int, metadata: Mapping[str, Any], headers: Mapping[str, Any], message_id: str, correlation_id: Optional[str], exchange: Optional[str], routing_key: Optional[str], delivery_mode: str) -> Mapping[str, Any] | None:
        self.queues.setdefault(queue_name, []).append({
            "message_id": message_id,
            "correlation_id": correlation_id,
            "payload": payload,
            "content_type": headers.get("content-type", "application/json"),
            "receipt": {"queue_name": queue_name, "message_id": message_id},
        })
        return {"published": True, "queue_name": queue_name}

    def consume(self, *, queue_name: str, timeout_ms: int, metadata: Mapping[str, Any], auto_ack: bool, visibility_timeout_ms: Optional[int], prefetch_count: int) -> Mapping[str, Any] | None:
        queue = self.queues.setdefault(queue_name, [])
        if not queue:
            raise ReceiveFailureError("Queue is empty.", context={"operation": "consume", "channel": "queue", "protocol": "queue"})
        return queue.pop(0)

    def ack(self, *, receipt: Mapping[str, Any], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return {"acknowledged": True, "receipt": dict(receipt)}

    def nack(self, *, receipt: Mapping[str, Any], requeue: bool, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return {"nacked": True, "receipt": dict(receipt), "requeue": requeue, "reason": reason}

    def close(self, *, reason: Optional[str], metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.connected = False
        return {"closed": True, "reason": reason}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _printer_status(label: str, message: str, level: str = "info") -> None:
    try:
        printer.status(label, message, level)
    except Exception:
        print(f"[{label}] {message}")


if __name__ == "__main__":
    print("\n=== Running Network Adapters ===\n")
    _printer_status("TEST", "Network Adapters initialized", "info")

    manager = NetworkAdapters()
    registry_snapshot = manager.get_registry_snapshot()
    _printer_status("TEST", "Default adapters registered", "info")

    http_server = _DemoHTTPServer()
    http_server.start()
    http_endpoint = f"http://{http_server.host}:{http_server.port}/relay"

    try:
        http_selection = manager.select_adapter(protocol="http", endpoint=http_endpoint)
        _printer_status("TEST", "HTTP adapter selected", "info")

        http_connected = manager.connect(name="http", endpoint=http_endpoint, metadata={"env": "test"})
        _printer_status("TEST", "HTTP adapter connected through facade", "info")

        http_sent = manager.send(
            {"task": "relay", "payload": {"hello": "network adapters"}},
            name="http",
            endpoint=http_endpoint,
            metadata={"trace": "http-demo"},
        )
        _printer_status("TEST", "HTTP payload sent through facade", "info")

        http_received = manager.recv(name="http", endpoint=http_endpoint, metadata={"consumer": "demo"})
        _printer_status("TEST", "HTTP cached response received through facade", "info")

        grpc_adapter = manager.create_adapter(
            name="grpc",
            endpoint="grpc://127.0.0.1:50051",
            transport=_DemoGRPCTransport(),
            config={
                "service_name": "slai.network.NetworkService",
                "default_method": "Relay",
                "default_call_type": "unary_unary",
                "ack_mode": "synthetic",
                "nack_mode": "synthetic",
            },
        )
        grpc_capabilities = grpc_adapter.capabilities.to_dict()
        _printer_status("TEST", "gRPC adapter created", "info")

        queue_adapter = manager.create_adapter(
            name="queue",
            endpoint="amqp://127.0.0.1:5672",
            transport=_DemoQueueTransport(),
            config={
                "default_queue": "jobs.primary",
                "ack_mode": "transport",
                "nack_mode": "transport",
            },
        )
        queue_capabilities = queue_adapter.capabilities.to_dict()
        _printer_status("TEST", "Queue adapter created", "info")

        websocket_adapter = manager.create_adapter(
            name="websocket",
            endpoint="ws://127.0.0.1:8765/echo",
            reuse_instance=False,
        )
        websocket_capabilities = websocket_adapter.capabilities.to_dict()
        _printer_status("TEST", "WebSocket adapter created", "info")

        active_snapshot = manager.list_active_adapters()
        health_snapshot = manager.get_active_health()
        close_count = manager.close_all(reason="demo complete")
        _printer_status("TEST", "Managed adapters closed", "info")

        print("Registry Snapshot:", stable_json_dumps(registry_snapshot))
        print("HTTP Selection:", stable_json_dumps(http_selection))
        print("HTTP Connected:", stable_json_dumps(http_connected))
        print("HTTP Sent:", stable_json_dumps(http_sent))
        print("HTTP Received:", stable_json_dumps(http_received))
        print("gRPC Capabilities:", stable_json_dumps(grpc_capabilities))
        print("Queue Capabilities:", stable_json_dumps(queue_capabilities))
        print("WebSocket Capabilities:", stable_json_dumps(websocket_capabilities))
        print("Active Snapshot:", stable_json_dumps(active_snapshot))
        print("Health Snapshot:", stable_json_dumps(health_snapshot))
        print("Close Count:", close_count)

        assert registry_snapshot["registry_size"] >= 4
        assert http_selection["selected_adapter"] == "http"
        assert http_connected["connected"] is True
        assert http_sent["payload_size"] > 0
        assert http_received["payload"]["ok"] is True
        assert grpc_capabilities["supports_request_reply"] is True
        assert queue_capabilities["supports_ack"] is True
        assert websocket_capabilities["supports_streaming"] is True
        assert close_count >= 3
        assert manager.memory.get("network.adapters.registry") is not None
        assert manager.memory.get("network.adapters.active") is not None

        _printer_status("TEST", "All Network Adapters checks passed", "info")
        print("\n=== Test ran successfully ===\n")
    finally:
        try:
            manager.close_all(reason="cleanup")
        except Exception:
            pass
        http_server.stop()
