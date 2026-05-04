"""
Channel Selection & Routing
- Choose best channel by latency, reliability, payload type, policy, and cost.
- Primary + secondary route selection.
- Region-aware routing and endpoint affinity.

This module is the runtime orchestration layer that sits above the routing
primitives and below the higher-level Network Agent. It coordinates endpoint
candidate discovery, channel selection, route-policy evaluation, adapter
instantiation, and request/reply or streaming transport execution.

The module is intentionally scoped to runtime channel/routing orchestration. It
owns:
- combining endpoint-registry candidates with registered adapter capabilities,
- invoking ChannelSelector for transport-family selection,
- invoking RoutePolicy for primary/secondary route evaluation,
- creating and reusing specialized adapters through NetworkAdapters,
- opening, tracking, and closing stream-oriented runtime sessions,
- emitting structured routing/session snapshots to NetworkMemory.

It does not own endpoint inventory semantics, route ranking internals, retry
policy, circuit breaking, or transport-specific logic. Those remain in the
Endpoint Registry, Channel Selector, Route Policy, reliability modules, and the
specialized adapters themselves.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock, Thread
from time import sleep
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils import *
from .network_memory import NetworkMemory
from .network_adapters import NetworkAdapters
from .adapters import *
from .routing import *
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Network Stream")
printer = PrettyPrinter()

_STREAM_LAST_KEY = "network.stream.last"
_STREAM_HISTORY_KEY = "network.stream.history"
_STREAM_SNAPSHOT_KEY = "network.stream.snapshot"
_STREAM_ACTIVE_KEY = "network.stream.active"

_DEFAULT_ALLOWED_OPERATIONS = ("send", "receive", "request_reply", "stream")


@dataclass(slots=True)
class StreamSessionRecord:
    """Runtime stream-session record managed by NetworkStream."""

    stream_id: str
    adapter_name: str
    protocol: str
    channel: str
    endpoint: Optional[str]
    route: Dict[str, Any] = field(default_factory=dict)
    state: str = "initialized"
    connected: bool = False
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    last_activity_at: Optional[str] = None
    request_count: int = 0
    receive_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self, *, state: Optional[str] = None) -> None:
        now = utc_timestamp()
        self.updated_at = now
        self.last_activity_at = now
        if state:
            self.state = str(state).strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "stream_id": self.stream_id,
            "adapter_name": self.adapter_name,
            "protocol": self.protocol,
            "channel": self.channel,
            "endpoint": self.endpoint,
            "route": json_safe(self.route),
            "state": self.state,
            "connected": self.connected,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_activity_at": self.last_activity_at,
            "request_count": self.request_count,
            "receive_count": self.receive_count,
            "metadata": json_safe(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


class _AdapterFacade:
    """Compatibility wrapper around NetworkAdapters for runtime orchestration."""

    def __init__(self, manager: Optional[Any], memory: NetworkMemory) -> None:
        self.manager = manager
        self.memory = memory
        self._lock = RLock()
        self._active: Dict[str, Any] = {}
        self._specs: Dict[str, Dict[str, Any]] = self._build_specs()
        self.priority_order: Tuple[str, ...] = tuple(sorted(self._specs.keys(), key=lambda key: int(self._specs[key].get("priority", 0)), reverse=True))

    def list_registered_adapters(self, *, enabled_only: bool = False) -> List[Dict[str, Any]]:
        if self.manager is not None and hasattr(self.manager, "list_registered_adapters"):
            return self.manager.list_registered_adapters(enabled_only=enabled_only)
        specs = list(self._specs.values())
        if enabled_only:
            specs = [spec for spec in specs if spec.get("enabled", True)]
        return [json_safe(spec) for spec in sorted(specs, key=lambda item: int(item.get("priority", 0)), reverse=True)]

    def create_adapter(self, *, name: Optional[str] = None, protocol: Optional[str] = None, channel: Optional[str] = None, endpoint: Optional[str] = None, constraints: Optional[Mapping[str, Any]] = None, config: Optional[Mapping[str, Any]] = None, **adapter_kwargs: Any) -> Any:
        if self.manager is not None and hasattr(self.manager, "create_adapter"):
            return self.manager.create_adapter(name=name, protocol=protocol, channel=channel, endpoint=endpoint, constraints=constraints, config=config, **adapter_kwargs)

        spec = self._resolve_spec(name=name, protocol=protocol, channel=channel)
        cache_key = generate_idempotency_key({"name": spec["name"], "endpoint": endpoint, "config": json_safe(config), "kwargs": json_safe(adapter_kwargs)}, namespace="network_stream_adapters")
        with self._lock:
            if cache_key in self._active:
                return self._active[cache_key]
            init_kwargs = dict(adapter_kwargs)
            init_kwargs.setdefault("memory", self.memory)
            init_kwargs.setdefault("config", ensure_mapping(config, field_name="config", allow_none=True))
            if endpoint is not None:
                init_kwargs.setdefault("endpoint", endpoint)
            if protocol is not None:
                init_kwargs.setdefault("protocol", protocol)
            adapter = spec["cls"](**init_kwargs)
            self._active[cache_key] = adapter
            return adapter

    def get_active_health(self) -> Dict[str, Any]:
        if self.manager is not None and hasattr(self.manager, "get_active_health"):
            return self.manager.get_active_health()
        payload: Dict[str, Any] = {}
        with self._lock:
            for key, adapter in self._active.items():
                if hasattr(adapter, "get_health_snapshot"):
                    payload[key] = adapter.get_health_snapshot()
                else:
                    payload[key] = {"adapter": type(adapter).__name__}
        return payload

    def close(self, *, name: Optional[str] = None, protocol: Optional[str] = None, channel: Optional[str] = None, endpoint: Optional[str] = None, reason: Optional[str] = None) -> int:
        if self.manager is not None and hasattr(self.manager, "close"):
            return int(self.manager.close(name=name, protocol=protocol, channel=channel, endpoint=endpoint, reason=reason))
        closed = 0
        with self._lock:
            keys = list(self._active.keys())
            for key in keys:
                adapter = self._active[key]
                adapter_name = type(adapter).__name__.replace("Adapter", "").lower()
                adapter_protocol = getattr(adapter, "protocol", None)
                adapter_channel = getattr(adapter, "channel", None)
                adapter_endpoint = getattr(getattr(adapter, "session", None), "endpoint", None)
                if name is not None and adapter_name != str(name).strip().lower():
                    continue
                if protocol is not None and normalize_protocol_name(protocol) != normalize_protocol_name(adapter_protocol or protocol):
                    continue
                if channel is not None and normalize_channel_name(channel) != normalize_channel_name(adapter_channel or channel):
                    continue
                if endpoint is not None and str(endpoint) != str(adapter_endpoint):
                    continue
                if hasattr(adapter, "close"):
                    adapter.close(reason=reason)
                self._active.pop(key, None)
                closed += 1
        return closed

    def close_all(self, *, reason: Optional[str] = None) -> int:
        if self.manager is not None and hasattr(self.manager, "close_all"):
            return int(self.manager.close_all(reason=reason))
        return self.close(reason=reason)

    def _resolve_spec(self, *, name: Optional[str], protocol: Optional[str], channel: Optional[str]) -> Dict[str, Any]:
        if name is not None:
            token = str(name).strip().lower()
            for spec in self._specs.values():
                aliases = set(spec.get("aliases", [])) | {spec["name"], spec.get("class_name", "").lower()}
                if token in aliases:
                    return spec
        if protocol is not None:
            normalized = normalize_protocol_name(protocol)
            for spec in self._specs.values():
                if normalized in [normalize_protocol_name(item) for item in spec.get("protocols", [])]:
                    return spec
        if channel is not None:
            normalized = normalize_channel_name(channel)
            for spec in self._specs.values():
                if normalized in [normalize_channel_name(item) for item in spec.get("channels", [])]:
                    return spec
        default_spec = self._specs.get("http") or next(iter(self._specs.values()))
        return default_spec

    def _build_specs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "http": {
                "name": "http", "class_name": "HTTPAdapter", "cls": HTTPAdapter, "enabled": True, "priority": 140,
                "protocols": ["http", "https"], "channels": ["http"], "aliases": ["rest"],
                "capabilities": {"supports_request_reply": True, "supports_receive": True, "supports_headers": True, "supports_tls": True},
            },
            "websocket": {
                "name": "websocket", "class_name": "WebSocketAdapter", "cls": WebSocketAdapter, "enabled": True, "priority": 130,
                "protocols": ["websocket", "ws", "wss"], "channels": ["websocket"], "aliases": ["ws", "wss", "socket"],
                "capabilities": {"supports_streaming": True, "supports_bidirectional_streaming": True, "supports_receive": True, "supports_tls": True},
            },
            "grpc": {
                "name": "grpc", "class_name": "GRPCAdapter", "cls": GRPCAdapter, "enabled": True, "priority": 120,
                "protocols": ["grpc", "grpcs"], "channels": ["grpc"], "aliases": ["rpc"],
                "capabilities": {"supports_streaming": True, "supports_bidirectional_streaming": True, "supports_request_reply": True, "supports_receive": True, "supports_tls": True},
            },
            "queue": {
                "name": "queue", "class_name": "QueueAdapter", "cls": QueueAdapter, "enabled": True, "priority": 110,
                "protocols": ["queue", "amqp", "amqps", "mq", "sqs", "kafka", "pubsub"], "channels": ["queue"], "aliases": ["broker", "mq"],
                "capabilities": {"supports_ack": True, "supports_nack": True, "supports_batch_send": True, "supports_receive": True, "supports_tls": True},
            },
        }


class NetworkStream:
    """
    Runtime channel/routing orchestration for streaming and request/reply flows.

    The class bridges endpoint inventory, channel selection, route-policy
    evaluation, and specialized transport adapters into a single execution
    surface suitable for NetworkAgent and adjacent orchestration layers.
    """

    def __init__(
        self,
        memory: Optional[NetworkMemory] = None,
        *,
        adapters: Optional[NetworkAdapters] = None,
        channel_selector: Optional[ChannelSelector] = None,
        endpoint_registry: Optional[EndpointRegistry] = None,
        route_policy: Optional[RoutePolicy] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = load_global_config()
        self.stream_config = merge_mappings(
            get_config_section("network_stream") or {},
            ensure_mapping(config, field_name="config", allow_none=True),
        )
        self.memory = memory or NetworkMemory()
        raw_adapters = adapters or NetworkAdapters(memory=self.memory)
        self.adapters = _AdapterFacade(raw_adapters, self.memory)
        self.endpoint_registry = endpoint_registry or EndpointRegistry(memory=self.memory)
        self.channel_selector = channel_selector or ChannelSelector(memory=self.memory, adapters=self.adapters)
        self.route_policy = route_policy or RoutePolicy(memory=self.memory, endpoint_registry=self.endpoint_registry)
        self._lock = RLock()

        self.enabled = self._get_bool_config("enabled", True)
        self.sanitize_logs = self._get_bool_config("sanitize_logs", True)
        self.record_route_snapshots = self._get_bool_config("record_route_snapshots", True)
        self.record_session_snapshots = self._get_bool_config("record_session_snapshots", True)
        self.use_endpoint_registry = self._get_bool_config("use_endpoint_registry", True)
        self.include_registered_adapters = self._get_bool_config("include_registered_adapters", True)
        self.allow_direct_endpoint_candidates = self._get_bool_config("allow_direct_endpoint_candidates", True)
        self.auto_connect_adapters = self._get_bool_config("auto_connect_adapters", True)
        self.auto_close_ephemeral_adapters = self._get_bool_config("auto_close_ephemeral_adapters", False)
        self.fallback_to_secondary_route = self._get_bool_config("fallback_to_secondary_route", True)
        self.require_route_selection = self._get_bool_config("require_route_selection", True)
        self.prefer_preconnected_channels = self._get_bool_config("prefer_preconnected_channels", True)
        self.require_registered_endpoints = self._get_bool_config("require_registered_endpoints", False)

        self.default_protocol = normalize_protocol_name(self._get_optional_string_config("default_protocol") or "http")
        self.default_channel = normalize_channel_name(self._get_optional_string_config("default_channel") or self.default_protocol)
        self.default_region = self._get_optional_string_config("default_region")
        self.default_operation = self._get_optional_string_config("default_operation") or "send"

        self.default_request_timeout_ms = coerce_timeout_ms(
            self.stream_config.get("default_request_timeout_ms"),
            default=5000,
            minimum=1,
            maximum=300000,
        )
        self.default_receive_timeout_ms = coerce_timeout_ms(
            self.stream_config.get("default_receive_timeout_ms"),
            default=self.default_request_timeout_ms,
            minimum=1,
            maximum=300000,
        )

        self.selection_ttl_seconds = self._get_non_negative_int_config("selection_ttl_seconds", 900)
        self.session_ttl_seconds = self._get_non_negative_int_config("session_ttl_seconds", 1800)
        self.history_ttl_seconds = self._get_non_negative_int_config("history_ttl_seconds", 3600)
        self.max_history_size = max(1, self._get_non_negative_int_config("max_history_size", 500))
        self.max_candidate_routes = max(1, self._get_non_negative_int_config("max_candidate_routes", 32))
        self.default_primary_count = max(1, self._get_non_negative_int_config("default_primary_count", 1))
        self.default_secondary_count = self._get_non_negative_int_config("default_secondary_count", 1)

        self._sessions: Dict[str, StreamSessionRecord] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history_size)
        self._stats: Dict[str, int] = {
            "route_resolutions": 0,
            "route_failures": 0,
            "stream_opens": 0,
            "stream_closes": 0,
            "relays": 0,
            "receives": 0,
            "request_replies": 0,
            "fallbacks": 0,
        }
        self._started_at = utc_timestamp()

        self._sync_stream_memory()

    # ------------------------------------------------------------------
    # Public orchestration API
    # ------------------------------------------------------------------
    def register_endpoint(self, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        """Proxy endpoint registration into the routing registry."""
        return self.endpoint_registry.register_endpoint(endpoint, **kwargs)

    def build_route_candidates(
        self,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        region: Optional[str] = None,
        operation: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        built = self._collect_candidates(
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            region=region,
            operation=operation,
            candidates=candidates,
            constraints=constraints,
        )
        return [json_safe(item) for item in built]

    def resolve_route(
        self,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        region: Optional[str] = None,
        operation: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RoutingError(
                "NetworkStream is disabled by configuration.",
                context={"operation": "resolve_route", "channel": channel, "protocol": protocol, "endpoint": endpoint},
            )

        normalized_operation = self._normalize_operation(operation)
        normalized_constraints = self._normalize_constraints(constraints, operation=normalized_operation)
        normalized_metadata = normalize_metadata(metadata)
        collected_candidates = self._collect_candidates(
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            region=region,
            operation=normalized_operation,
            candidates=candidates,
            constraints=normalized_constraints,
        )

        if not collected_candidates and self.require_route_selection:
            self._stats["route_failures"] += 1
            raise NoRouteAvailableError(
                "No route candidates were available for NetworkStream resolution.",
                context={"operation": "resolve_route", "channel": channel, "protocol": protocol, "endpoint": endpoint},
                details={"constraints": sanitize_for_logging(normalized_constraints) if self.sanitize_logs else json_safe(normalized_constraints)},
            )

        selector_decision = self.channel_selector.select_channel(
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            candidates=collected_candidates,
            constraints=normalized_constraints,
            include_registered=False,
            metadata={**normalized_metadata, "operation": normalized_operation},
        )
        ranked_candidates = [ensure_mapping(item.get("candidate"), field_name="candidate") for item in selector_decision.get("candidates", [])]

        policy_snapshot = self.route_policy.select_routes(
            ranked_candidates,
            constraints=normalized_constraints,
            request_context={
                "preferred_region": region or self.default_region,
                "operation": normalized_operation,
                **normalized_metadata,
            },
            primary_count=self.default_primary_count,
            secondary_count=self.default_secondary_count,
        )

        selected_route = ensure_mapping(policy_snapshot.get("selected"), field_name="selected", allow_none=True)
        if not selected_route:
            selected_route = ensure_mapping(selector_decision.get("selected"), field_name="selected", allow_none=True)

        if not selected_route and self.require_route_selection:
            self._stats["route_failures"] += 1
            raise NoRouteAvailableError(
                "Route resolution completed without a viable selected route.",
                context={"operation": "resolve_route", "channel": channel, "protocol": protocol, "endpoint": endpoint},
                details={
                    "selector_decision": sanitize_for_logging(selector_decision) if self.sanitize_logs else json_safe(selector_decision),
                    "policy_snapshot": sanitize_for_logging(policy_snapshot) if self.sanitize_logs else json_safe(policy_snapshot),
                },
            )

        resolved = {
            "selected": json_safe(selected_route),
            "primary": json_safe(policy_snapshot.get("primary", [])),
            "secondary": json_safe(policy_snapshot.get("secondary", [])),
            "selector": json_safe(selector_decision),
            "policy": json_safe(policy_snapshot),
            "operation": normalized_operation,
            "requested_protocol": protocol,
            "requested_channel": channel,
            "requested_endpoint": self._safe_endpoint(endpoint),
            "requested_region": region or self.default_region,
            "constraints": json_safe(normalized_constraints),
            "metadata": json_safe(normalized_metadata),
            "resolved_at": utc_timestamp(),
        }

        with self._lock:
            self._stats["route_resolutions"] += 1
            self._append_history("resolve_route", resolved)
            self._record_route_snapshot(resolved)
        return resolved

    def open_stream(
        self,
        *,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        region: Optional[str] = None,
        operation: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        timeout_ms: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        **adapter_kwargs: Any,
    ) -> Dict[str, Any]:
        route_resolution = self.resolve_route(
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            region=region,
            operation=operation,
            candidates=candidates,
            constraints=constraints,
            metadata=metadata,
        )
        selected = ensure_mapping(route_resolution.get("selected"), field_name="selected")
        normalized_timeout = coerce_timeout_ms(timeout_ms, default=self.default_request_timeout_ms)
        normalized_metadata = normalize_metadata(metadata)

        adapter = self.adapters.create_adapter(
            name=selected.get("adapter_name"),
            protocol=selected.get("protocol"),
            channel=selected.get("channel"),
            endpoint=selected.get("endpoint") or endpoint,
            constraints=constraints,
            config=config,
            **adapter_kwargs,
        )

        connect_result = None
        if self.auto_connect_adapters and hasattr(adapter, "is_connected") and not adapter.is_connected():
            connect_result = adapter.connect(
                endpoint=selected.get("endpoint") or endpoint,
                timeout_ms=normalized_timeout,
                metadata=normalized_metadata,
            )

        session = self._register_session(
            route=selected,
            connected=bool(getattr(adapter, "is_connected", lambda: True)()),
            metadata={**normalized_metadata, "operation": self._normalize_operation(operation)},
        )

        with self._lock:
            self._stats["stream_opens"] += 1
            self._sync_stream_memory()

        return {
            "stream_id": session.stream_id,
            "route": json_safe(route_resolution),
            "connect_result": json_safe(connect_result),
            "session": session.to_dict(),
            "opened_at": utc_timestamp(),
        }

    def relay(
        self,
        payload: Any,
        *,
        envelope: Optional[Mapping[str, Any]] = None,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        region: Optional[str] = None,
        operation: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        timeout_ms: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        content_type: Optional[str] = None,
        await_response: bool = False,
        receive_timeout_ms: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
        **adapter_kwargs: Any,
    ) -> Dict[str, Any]:
        normalized_operation = self._normalize_operation(operation or ("request_reply" if await_response else "send"))
        opened = self.open_stream(
            protocol=protocol,
            channel=channel,
            endpoint=endpoint,
            region=region,
            operation=normalized_operation,
            candidates=candidates,
            constraints=constraints,
            timeout_ms=timeout_ms,
            metadata=metadata,
            config=config,
            **adapter_kwargs,
        )
        session = self._require_session(opened["stream_id"])
        route = ensure_mapping(opened["route"].get("selected"), field_name="selected")
        normalized_metadata = normalize_metadata(metadata)
        normalized_timeout = coerce_timeout_ms(timeout_ms, default=self.default_request_timeout_ms)

        adapter = self.adapters.create_adapter(
            name=session.adapter_name,
            protocol=session.protocol,
            channel=session.channel,
            endpoint=session.endpoint,
            constraints=constraints,
            config=config,
            **adapter_kwargs,
        )

        try:
            send_result = adapter.send(
                payload,
                envelope=envelope,
                timeout_ms=normalized_timeout,
                metadata=normalized_metadata,
                content_type=content_type,
            )
        except NetworkError:
            if self.fallback_to_secondary_route and opened["route"].get("secondary"):
                self._stats["fallbacks"] += 1
                return self._relay_via_secondary(
                    payload,
                    envelope=envelope,
                    route_resolution=opened["route"],
                    constraints=constraints,
                    timeout_ms=timeout_ms,
                    metadata=metadata,
                    content_type=content_type,
                    await_response=await_response,
                    receive_timeout_ms=receive_timeout_ms,
                    config=config,
                    **adapter_kwargs,
                )
            raise

        response = None
        if await_response:
            response = adapter.recv(
                timeout_ms=coerce_timeout_ms(receive_timeout_ms, default=self.default_receive_timeout_ms),
                metadata=normalized_metadata,
            )

        with self._lock:
            session.request_count += 1
            session.connected = bool(getattr(adapter, "is_connected", lambda: True)())
            session.route = merge_mappings(session.route, route)
            session.touch(state="active")
            self._stats["relays"] += 1
            if await_response:
                self._stats["request_replies"] += 1
            self._append_history(
                "relay",
                {
                    "stream_id": session.stream_id,
                    "route": json_safe(route),
                    "await_response": await_response,
                    "send_result": sanitize_for_logging(send_result) if self.sanitize_logs else json_safe(send_result),
                    "response": sanitize_for_logging(response) if self.sanitize_logs else json_safe(response),
                },
            )
            self._sync_session_memory(session)
            self._sync_stream_memory()

        if self.auto_close_ephemeral_adapters:
            self.close_stream(stream_id=session.stream_id, reason="ephemeral_relay_complete")

        return {
            "stream_id": session.stream_id,
            "route": json_safe(opened["route"]),
            "session": session.to_dict(),
            "send_result": json_safe(send_result),
            "response": json_safe(response),
            "completed_at": utc_timestamp(),
        }

    def receive(
        self,
        *,
        stream_id: Optional[str] = None,
        protocol: Optional[str] = None,
        channel: Optional[str] = None,
        endpoint: Optional[str] = None,
        region: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        timeout_ms: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        **adapter_kwargs: Any,
    ) -> Dict[str, Any]:
        normalized_timeout = coerce_timeout_ms(timeout_ms, default=self.default_receive_timeout_ms)
        normalized_metadata = normalize_metadata(metadata)

        if stream_id is not None:
            session = self._require_session(stream_id)
            adapter = self.adapters.create_adapter(
                name=session.adapter_name,
                protocol=session.protocol,
                channel=session.channel,
                endpoint=session.endpoint,
                constraints=constraints,
                config=config,
                **adapter_kwargs,
            )
        else:
            opened = self.open_stream(
                protocol=protocol,
                channel=channel,
                endpoint=endpoint,
                region=region,
                operation="receive",
                candidates=candidates,
                constraints=constraints,
                timeout_ms=normalized_timeout,
                metadata=metadata,
                config=config,
                **adapter_kwargs,
            )
            session = self._require_session(opened["stream_id"])
            adapter = self.adapters.create_adapter(
                name=session.adapter_name,
                protocol=session.protocol,
                channel=session.channel,
                endpoint=session.endpoint,
                constraints=constraints,
                config=config,
                **adapter_kwargs,
            )

        result = adapter.recv(timeout_ms=normalized_timeout, metadata=normalized_metadata)

        with self._lock:
            session.receive_count += 1
            session.connected = bool(getattr(adapter, "is_connected", lambda: True)())
            session.touch(state="active")
            self._stats["receives"] += 1
            self._append_history(
                "receive",
                {
                    "stream_id": session.stream_id,
                    "endpoint": session.endpoint,
                    "channel": session.channel,
                    "result": sanitize_for_logging(result) if self.sanitize_logs else json_safe(result),
                },
            )
            self._sync_session_memory(session)
            self._sync_stream_memory()

        return {
            "stream_id": session.stream_id,
            "session": session.to_dict(),
            "result": json_safe(result),
            "received_at": utc_timestamp(),
        }

    def request_reply(self, payload: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convenience wrapper for request/reply flows."""
        kwargs["await_response"] = True
        return self.relay(payload, **kwargs)

    def close_stream(self, *, stream_id: Optional[str] = None, reason: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if stream_id is None:
                closed = self.adapters.close_all(reason=reason or "network_stream_close_all")
                count = len(self._sessions)
                self._sessions.clear()
                self._stats["stream_closes"] += count
                self._append_history("close_all", {"closed": closed, "reason": reason})
                self._sync_stream_memory()
                return {"closed": closed, "session_count": count, "reason": reason}

            session = self._sessions.pop(ensure_non_empty_string(stream_id, field_name="stream_id"), None)
            if session is None:
                return {"closed": 0, "stream_id": stream_id, "reason": reason}

            closed = self.adapters.close(
                name=session.adapter_name,
                protocol=session.protocol,
                channel=session.channel,
                endpoint=session.endpoint,
                reason=reason or "network_stream_close",
            )
            session.connected = False
            session.touch(state="closed")
            self._stats["stream_closes"] += 1
            self._append_history("close_stream", {"stream_id": stream_id, "closed": closed, "reason": reason})
            self._sync_stream_memory()
            return {"closed": closed, "stream_id": stream_id, "session": session.to_dict(), "reason": reason}

    def get_stream_health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "enabled": self.enabled,
                "default_protocol": self.default_protocol,
                "default_channel": self.default_channel,
                "stats": dict(self._stats),
                "active_sessions": {key: session.to_dict() for key, session in self._sessions.items()},
                "channel_selector": self.channel_selector.get_selector_snapshot(),
                "endpoint_registry": self.endpoint_registry.get_health_summary(),
                "route_policy": self.route_policy.get_policy_snapshot(),
                "adapters": self.adapters.get_active_health(),
                "memory": self.memory.get_network_health(),
            }

    def get_stream_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_timestamp(),
                "enabled": self.enabled,
                "started_at": self._started_at,
                "stats": dict(self._stats),
                "session_count": len(self._sessions),
                "sessions": {key: value.to_dict() for key, value in self._sessions.items()},
                "history_size": len(self._history),
                "last_route": self.memory.get(_STREAM_LAST_KEY, default=None),
            }

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------
    def _relay_via_secondary(
        self,
        payload: Any,
        *,
        envelope: Optional[Mapping[str, Any]],
        route_resolution: Mapping[str, Any],
        constraints: Optional[Mapping[str, Any]],
        timeout_ms: Optional[Any],
        metadata: Optional[Mapping[str, Any]],
        content_type: Optional[str],
        await_response: bool,
        receive_timeout_ms: Optional[Any],
        config: Optional[Mapping[str, Any]],
        **adapter_kwargs: Any,
    ) -> Dict[str, Any]:
        secondaries = ensure_sequence(route_resolution.get("secondary"), field_name="secondary", allow_none=True, coerce_scalar=False)
        if not secondaries:
            raise NoRouteAvailableError(
                "Secondary route fallback was requested but no secondary routes are available.",
                context={"operation": "relay_via_secondary"},
            )

        fallback = ensure_mapping(secondaries[0], field_name="secondary_route")
        adapter = self.adapters.create_adapter(
            name=fallback.get("adapter_name"),
            protocol=fallback.get("protocol"),
            channel=fallback.get("channel"),
            endpoint=fallback.get("endpoint"),
            constraints=constraints,
            config=config,
            **adapter_kwargs,
        )
        normalized_metadata = normalize_metadata(metadata)
        normalized_timeout = coerce_timeout_ms(timeout_ms, default=self.default_request_timeout_ms)
        if self.auto_connect_adapters and hasattr(adapter, "is_connected") and not adapter.is_connected():
            adapter.connect(endpoint=fallback.get("endpoint"), timeout_ms=normalized_timeout, metadata=normalized_metadata)

        send_result = adapter.send(
            payload,
            envelope=envelope,
            timeout_ms=normalized_timeout,
            metadata={**normalized_metadata, "fallback": True},
            content_type=content_type,
        )
        response = None
        if await_response:
            response = adapter.recv(
                timeout_ms=coerce_timeout_ms(receive_timeout_ms, default=self.default_receive_timeout_ms),
                metadata={**normalized_metadata, "fallback": True},
            )

        session = self._register_session(route=fallback, connected=bool(getattr(adapter, "is_connected", lambda: True)()), metadata={**normalized_metadata, "fallback": True})
        session.request_count += 1
        if await_response:
            session.receive_count += 1
        session.touch(state="active")
        self._sync_session_memory(session)
        self._sync_stream_memory()

        return {
            "stream_id": session.stream_id,
            "route": json_safe(route_resolution),
            "fallback_route": json_safe(fallback),
            "send_result": json_safe(send_result),
            "response": json_safe(response),
            "completed_at": utc_timestamp(),
        }

    def _collect_candidates(
        self,
        *,
        protocol: Optional[str],
        channel: Optional[str],
        endpoint: Optional[str],
        region: Optional[str],
        operation: Optional[str],
        candidates: Optional[Sequence[Mapping[str, Any]]],
        constraints: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized_protocol = normalize_protocol_name(protocol) if protocol is not None else None
        normalized_channel = normalize_channel_name(channel) if channel is not None else None
        normalized_constraints = self._normalize_constraints(constraints, operation=operation)
        collected: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def add_candidate(item: Mapping[str, Any]) -> None:
            payload = ensure_mapping(item, field_name="candidate")
            key = generate_idempotency_key(
                {
                    "adapter_name": payload.get("adapter_name"),
                    "protocol": payload.get("protocol"),
                    "channel": payload.get("channel"),
                    "endpoint": payload.get("endpoint"),
                    "route": payload.get("route"),
                },
                namespace="network_stream_candidate",
            )
            if key in seen:
                return
            seen.add(key)
            collected.append(payload)

        for raw in ensure_sequence(candidates, field_name="candidates", allow_none=True, coerce_scalar=False):
            if not isinstance(raw, Mapping):
                raise PayloadValidationError(
                    "Each route candidate must be a mapping-like object.",
                    context={"operation": "collect_stream_candidates"},
                    details={"received_type": type(raw).__name__},
                )
            payload = ensure_mapping(raw, field_name="candidate")
            if payload.get("adapter_name"):
                add_candidate(payload)
            else:
                for expanded in self._expand_candidate_with_registered_adapters(
                    payload,
                    requested_protocol=normalized_protocol,
                    requested_channel=normalized_channel,
                ):
                    add_candidate(expanded)

        if self.use_endpoint_registry:
            registry_candidates = self.endpoint_registry.get_candidates(
                protocol=normalized_protocol,
                channel=normalized_channel,
                region=region or self.default_region,
                include_disabled=False,
                include_unavailable=False,
                include_degraded=bool(normalized_constraints.get("allow_degraded_routes", True)),
                include_unhealthy=bool(normalized_constraints.get("allow_unhealthy_routes", False)),
                capability_constraints=ensure_mapping(normalized_constraints.get("required_capabilities"), field_name="required_capabilities", allow_none=True),
            )
            for registry_candidate in registry_candidates:
                for expanded in self._expand_candidate_with_registered_adapters(
                    registry_candidate,
                    requested_protocol=normalized_protocol,
                    requested_channel=normalized_channel,
                ):
                    add_candidate(expanded)

        if endpoint and self.allow_direct_endpoint_candidates:
            direct_candidate = self._build_direct_candidate(
                endpoint=endpoint,
                protocol=normalized_protocol,
                channel=normalized_channel,
                region=region,
                constraints=normalized_constraints,
            )
            for expanded in self._expand_candidate_with_registered_adapters(
                direct_candidate,
                requested_protocol=normalized_protocol,
                requested_channel=normalized_channel,
            ):
                add_candidate(expanded)

        if endpoint and self.require_registered_endpoints:
            if not any(item.get("endpoint") == self._safe_endpoint(endpoint) for item in collected):
                raise NoRouteAvailableError(
                    "Direct endpoint use is disallowed unless the endpoint is registered.",
                    context={"operation": "collect_stream_candidates", "endpoint": endpoint, "protocol": protocol, "channel": channel},
                )

        if operation == "stream":
            collected.sort(key=lambda item: self._stream_preference_sort_key(item), reverse=True)
        else:
            collected = collected[: self.max_candidate_routes]
        return collected[: self.max_candidate_routes]

    def _expand_candidate_with_registered_adapters(
        self,
        candidate: Mapping[str, Any],
        *,
        requested_protocol: Optional[str],
        requested_channel: Optional[str],
    ) -> List[Dict[str, Any]]:
        payload = ensure_mapping(candidate, field_name="candidate")
        protocol = normalize_protocol_name(payload.get("protocol") or requested_protocol or self.default_protocol)
        channel = normalize_channel_name(payload.get("channel") or requested_channel or protocol)
        endpoint = self._safe_endpoint(payload.get("endpoint"))
        matches: List[Dict[str, Any]] = []

        for spec in self.adapters.list_registered_adapters(enabled_only=True):
            spec_protocols = [normalize_protocol_name(item) for item in spec.get("protocols", [])]
            spec_channels = [normalize_channel_name(item) for item in spec.get("channels", [])]
            if requested_protocol and requested_protocol not in spec_protocols:
                continue
            if requested_channel and requested_channel not in spec_channels:
                continue
            if protocol not in spec_protocols and channel not in spec_channels:
                continue
            matches.append(
                merge_mappings(
                    payload,
                    {
                        "adapter_name": spec["name"],
                        "protocol": protocol if protocol in spec_protocols else (spec_protocols[0] if spec_protocols else protocol),
                        "channel": channel if channel in spec_channels else (spec_channels[0] if spec_channels else channel),
                        "endpoint": endpoint,
                        "priority": int(payload.get("priority", spec.get("priority", 100)) or 100),
                        "capabilities": merge_mappings(spec.get("capabilities"), payload.get("capabilities")),
                        "metadata": merge_mappings(spec.get("metadata"), payload.get("metadata"), {"candidate_source": payload.get("candidate_source", "stream")}),
                    },
                )
            )

        if matches:
            return matches

        if payload.get("adapter_name"):
            return [merge_mappings(payload, {"endpoint": endpoint, "protocol": protocol, "channel": channel})]
        return []

    def _build_direct_candidate(
        self,
        *,
        endpoint: str,
        protocol: Optional[str],
        channel: Optional[str],
        region: Optional[str],
        constraints: Mapping[str, Any],
    ) -> Dict[str, Any]:
        normalized_endpoint = str(endpoint).strip()
        if not normalized_endpoint:
            raise PayloadValidationError(
                "endpoint must not be empty.",
                context={"operation": "build_direct_candidate"},
            )

        inferred_protocol = protocol or self._infer_protocol_from_endpoint(normalized_endpoint) or self.default_protocol
        normalized_protocol = normalize_protocol_name(inferred_protocol)
        normalized_channel = normalize_channel_name(channel or normalized_protocol)
        secure = None
        normalized_rendered_endpoint = normalized_endpoint
        if "://" in normalized_endpoint:
            try:
                parsed = parse_endpoint(normalized_endpoint, default_scheme=normalized_protocol, protocol=normalized_protocol, require_host=False)
                normalized_rendered_endpoint = parsed.normalized
                secure = parsed.secure
            except Exception:
                normalized_rendered_endpoint = normalized_endpoint
        elif normalized_channel != "queue":
            secure = is_secure_protocol(normalized_protocol)

        return {
            "endpoint": normalized_rendered_endpoint,
            "protocol": normalized_protocol,
            "channel": normalized_channel,
            "region": region or self.default_region,
            "priority": int(constraints.get("direct_candidate_priority", 100) or 100),
            "cost": float(constraints.get("direct_candidate_cost", 1.0) or 1.0),
            "weight": float(constraints.get("direct_candidate_weight", 1.0) or 1.0),
            "secure": secure,
            "capabilities": {},
            "metadata": {"candidate_source": "direct_endpoint"},
        }

    # ------------------------------------------------------------------
    # Internal session/memory helpers
    # ------------------------------------------------------------------
    def _register_session(self, *, route: Mapping[str, Any], connected: bool, metadata: Optional[Mapping[str, Any]] = None) -> StreamSessionRecord:
        payload = ensure_mapping(route, field_name="route")
        stream_id = generate_message_id(prefix="stream")
        session = StreamSessionRecord(
            stream_id=stream_id,
            adapter_name=ensure_non_empty_string(str(payload.get("adapter_name") or payload.get("channel") or payload.get("protocol") or "adapter"), field_name="adapter_name").lower(),
            protocol=normalize_protocol_name(payload.get("protocol") or self.default_protocol),
            channel=normalize_channel_name(payload.get("channel") or payload.get("protocol") or self.default_channel),
            endpoint=self._safe_endpoint(payload.get("endpoint")),
            route=json_safe(payload),
            state="connected" if connected else "initialized",
            connected=bool(connected),
            metadata=normalize_metadata(metadata),
        )
        session.touch(state=session.state)
        with self._lock:
            self._sessions[session.stream_id] = session
            self._sync_session_memory(session)
            self._sync_stream_memory()
        return session

    def _require_session(self, stream_id: str) -> StreamSessionRecord:
        normalized = ensure_non_empty_string(stream_id, field_name="stream_id")
        with self._lock:
            session = self._sessions.get(normalized)
            if session is None:
                raise SessionUnavailableError(
                    "Requested stream session is not active.",
                    context={"operation": "stream_session", "session_id": normalized},
                )
            return session

    def _record_route_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        if not self.record_route_snapshots:
            return
        self.memory.set(_STREAM_LAST_KEY, dict(snapshot), ttl_seconds=self.selection_ttl_seconds, source="network_stream")
        self.memory.append(_STREAM_HISTORY_KEY, dict(snapshot), max_items=self.max_history_size, ttl_seconds=self.history_ttl_seconds, source="network_stream")
        self.memory.set(_STREAM_SNAPSHOT_KEY, self.get_stream_snapshot(), ttl_seconds=self.selection_ttl_seconds, source="network_stream")

    def _sync_session_memory(self, session: StreamSessionRecord) -> None:
        if not self.record_session_snapshots:
            return
        self.memory.update_session_snapshot(
            session.stream_id,
            session.to_dict(),
            ttl_seconds=self.session_ttl_seconds,
            merge_existing=True,
            metadata={"source": "network_stream", "adapter_name": session.adapter_name},
        )

    def _sync_stream_memory(self) -> None:
        self.memory.set(
            _STREAM_ACTIVE_KEY,
            {
                "generated_at": utc_timestamp(),
                "session_count": len(self._sessions),
                "sessions": {key: value.to_dict() for key, value in self._sessions.items()},
                "stats": dict(self._stats),
            },
            ttl_seconds=self.session_ttl_seconds,
            source="network_stream",
        )
        if self.record_route_snapshots:
            self.memory.set(_STREAM_SNAPSHOT_KEY, self.get_stream_snapshot(), ttl_seconds=self.selection_ttl_seconds, source="network_stream")

    def _append_history(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "occurred_at": utc_timestamp(),
            "payload": sanitize_for_logging(payload) if self.sanitize_logs else json_safe(payload),
        }
        self._history.append(event)
        self.memory.append(_STREAM_HISTORY_KEY, event, max_items=self.max_history_size, ttl_seconds=self.history_ttl_seconds, source="network_stream")

    # ------------------------------------------------------------------
    # Config and normalization helpers
    # ------------------------------------------------------------------
    def _normalize_constraints(self, constraints: Optional[Mapping[str, Any]], *, operation: Optional[str] = None) -> Dict[str, Any]:
        payload = ensure_mapping(constraints, field_name="constraints", allow_none=True)
        normalized = merge_mappings(payload)
        normalized_operation = self._normalize_operation(operation or payload.get("operation"))
        normalized["operation"] = normalized_operation

        if normalized_operation == "stream":
            normalized.setdefault("required_capabilities", {})
            normalized["required_capabilities"] = merge_mappings(
                ensure_mapping(normalized.get("required_capabilities"), field_name="required_capabilities", allow_none=True),
                {"supports_streaming": True},
            )
        elif normalized_operation == "request_reply":
            normalized.setdefault("required_capabilities", {})
            normalized["required_capabilities"] = merge_mappings(
                ensure_mapping(normalized.get("required_capabilities"), field_name="required_capabilities", allow_none=True),
                {"supports_request_reply": True, "supports_receive": True},
            )
        elif normalized_operation == "receive":
            normalized.setdefault("required_capabilities", {})
            normalized["required_capabilities"] = merge_mappings(
                ensure_mapping(normalized.get("required_capabilities"), field_name="required_capabilities", allow_none=True),
                {"supports_receive": True},
            )

        return normalized

    def _normalize_operation(self, operation: Optional[str]) -> str:
        if operation is None:
            return self.default_operation
        normalized = ensure_non_empty_string(str(operation), field_name="operation").strip().lower()
        if normalized not in _DEFAULT_ALLOWED_OPERATIONS:
            raise PayloadValidationError(
                "Unsupported network stream operation.",
                context={"operation": "normalize_stream_operation"},
                details={"value": normalized, "allowed": list(_DEFAULT_ALLOWED_OPERATIONS)},
            )
        return normalized

    def _safe_endpoint(self, endpoint: Any) -> Optional[str]:
        if endpoint is None:
            return None
        text = str(endpoint).strip()
        if not text:
            return None
        if "://" in text:
            try:
                return normalize_endpoint(text)
            except Exception:
                return text
        return text

    def _infer_protocol_from_endpoint(self, endpoint: str) -> Optional[str]:
        text = str(endpoint).strip()
        if not text:
            return None
        if "://" in text:
            try:
                return normalize_protocol_name(parse_endpoint(text, require_host=False).protocol)
            except Exception:
                return None
        if "." not in text and "/" not in text:
            return "queue"
        return None

    def _stream_preference_sort_key(self, payload: Mapping[str, Any]) -> Tuple[int, int, float]:
        channel = normalize_channel_name(payload.get("channel") or payload.get("protocol") or self.default_channel)
        streaming = bool(ensure_mapping(payload.get("capabilities"), field_name="capabilities", allow_none=True).get("supports_streaming", False))
        priority = int(payload.get("priority", 0) or 0)
        secure = 1 if bool(payload.get("secure", False)) else 0
        return (1 if streaming else 0, priority, float(secure))

    def _get_bool_config(self, name: str, default: bool) -> bool:
        value = self.stream_config.get(name, default)
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
            "Invalid boolean value in network stream configuration.",
            context={"operation": "network_stream_config"},
            details={"config_key": name, "config_value": value},
        )

    def _get_non_negative_int_config(self, name: str, default: int) -> int:
        value = self.stream_config.get(name, default)
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise NetworkConfigurationError(
                "Invalid integer value in network stream configuration.",
                context={"operation": "network_stream_config"},
                details={"config_key": name, "config_value": value},
                cause=exc,
            ) from exc
        if coerced < 0:
            raise NetworkConfigurationError(
                "Configuration integer value must be non-negative.",
                context={"operation": "network_stream_config"},
                details={"config_key": name, "config_value": value},
            )
        return coerced

    def _get_optional_string_config(self, name: str) -> Optional[str]:
        value = self.stream_config.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None


if __name__ == "__main__":
    print("\n=== Running Network Stream ===\n")
    printer.status("TEST", "Network Stream initialized", "info")
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

    demo_server = _DemoHTTPServer()
    demo_server.start()
    http_endpoint = f"http://{demo_server.host}:{demo_server.port}/relay"

    try:
        memory = NetworkMemory()
        stream = NetworkStream(memory=memory)

        stream.register_endpoint(
            http_endpoint,
            endpoint_id="http_primary",
            protocol="http",
            channel="http",
            region="local",
            priority=120,
            secure=False,
            capabilities={"supports_request_reply": True, "supports_receive": True, "supports_headers": True},
            tags=["primary", "demo"],
            metadata={"role": "relay_api"},
        )
        stream.register_endpoint(
            "jobs.primary",
            endpoint_id="queue_fallback",
            protocol="queue",
            channel="queue",
            endpoint_type="logical",
            region="local",
            priority=90,
            secure=False,
            capabilities={"supports_ack": True, "supports_nack": True, "supports_batch_send": True},
            tags=["fallback"],
            metadata={"role": "broker"},
        )
        printer.status("TEST", "Endpoints registered", "info")

        stream.endpoint_registry.update_health(
            "http_primary",
            status="healthy",
            latency_ms=35.0,
            success_rate=0.995,
            error_rate=0.005,
            metadata={"window": "5m"},
        )
        stream.endpoint_registry.update_health(
            "queue_fallback",
            status="degraded",
            latency_ms=120.0,
            success_rate=0.91,
            error_rate=0.09,
            metadata={"window": "5m"},
        )
        memory.record_channel_metrics(
            "http",
            {"success_rate": 0.994, "error_rate": 0.006, "p95_latency_ms": 45, "requests": 128},
            metadata={"window": "5m"},
        )
        memory.record_channel_metrics(
            "queue",
            {"success_rate": 0.90, "error_rate": 0.10, "p95_latency_ms": 140, "requests": 64},
            metadata={"window": "5m"},
        )
        printer.status("TEST", "Health and metrics primed", "info")

        candidates = stream.build_route_candidates(protocol="http", channel="http", endpoint=http_endpoint, region="local")
        printer.status("TEST", "Route candidates built", "info")

        route = stream.resolve_route(
            protocol="http",
            channel="http",
            endpoint=http_endpoint,
            region="local",
            constraints={"preferred_regions": ["local"], "required_capabilities": {"supports_request_reply": True}},
            metadata={"trace": "resolve-demo"},
        )
        printer.status("TEST", "Route resolved", "info")

        relay_result = stream.relay(
            {"task": "relay", "payload": {"hello": "network stream"}},
            protocol="http",
            channel="http",
            endpoint=http_endpoint,
            metadata={"trace": "relay-demo"},
        )
        printer.status("TEST", "Relay completed", "info")

        receive_result = stream.receive(
            protocol="http",
            channel="http",
            endpoint=http_endpoint,
            metadata={"consumer": "demo"},
        )
        printer.status("TEST", "Receive completed", "info")

        health_snapshot = stream.get_stream_health()
        stream_snapshot = stream.get_stream_snapshot()
        close_result = stream.close_stream(reason="demo complete")
        printer.status("TEST", "Streams closed", "info")

        print("Candidates:", stable_json_dumps(candidates))
        print("Route:", stable_json_dumps(route))
        print("Relay Result:", stable_json_dumps(relay_result))
        print("Receive Result:", stable_json_dumps(receive_result))
        print("Health Snapshot:", stable_json_dumps(health_snapshot))
        print("Stream Snapshot:", stable_json_dumps(stream_snapshot))
        print("Close Result:", stable_json_dumps(close_result))

        assert len(candidates) >= 1
        assert route["selected"]["adapter_name"] == "http"
        assert relay_result["send_result"]["payload_size"] > 0
        assert receive_result["result"]["payload"]["ok"] is True
        assert health_snapshot["stats"]["route_resolutions"] >= 1
        assert memory.get("network.stream.snapshot") is not None
        assert memory.get("network.stream.active") is not None

        printer.status("TEST", "All Network Stream checks passed", "info")
        print("\n=== Test ran successfully ===\n")
    finally:
        demo_server.stop()
