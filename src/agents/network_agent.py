"""
Network Agent is specialized enough to own communications relay responsibilities,
while broad enough to support the full lifecycle of networked operations across SLAI.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional

from .base_agent import BaseAgent
from .base.utils.main_config_loader import load_global_config, get_config_section
from .network import (
    NetworkAdapters,
    NetworkLifecycle,
    NetworkMemory,
    NetworkMetrics,
    NetworkPolicy,
    NetworkReliability,
    NetworkStream,
)
from .network.utils import (
    NetworkError,
    PayloadValidationError,
    ReliabilityError,
    ensure_mapping,
    json_safe,
    normalize_network_exception,
    normalize_protocol_name,
    sanitize_for_logging,
)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Network Agent")
printer = PrettyPrinter()


class NetworkAgent(BaseAgent):
    """
    Runtime owner for communication orchestration in SLAI.

    Boundary contract:
    - ExecutionAgent owns *what* to execute.
    - NetworkAgent owns *how/where* payloads are transported.
    - BrowserAgent owns browser workflows.
    - HandlerAgent owns generic fallback/escalation; NetworkAgent contributes
      communication-native recovery actions.
    - ObservabilityAgent consumes emitted telemetry; NetworkAgent produces
      path-level telemetry and delivery metadata.
    - Safety/Privacy agents own sensitive policy decisions; NetworkAgent applies
      network policy gates and surfaces violations.
    """

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.shared_memory = shared_memory or self.shared_memory
        self.agent_factory = agent_factory

        # Keep config handling pattern intact while allowing runtime overrides.
        self.config = load_global_config()
        self.network_config = get_config_section("network_agent") or {}
        if config:
            self.network_config.update(dict(config))

        self.name = "NetworkAgent"
        self.enabled = bool(self.network_config.get("enabled", True))
        self.max_delivery_attempts = max(1, int(self.network_config.get("max_delivery_attempts", 3)))
        self.retry_sleep_enabled = bool(self.network_config.get("retry_sleep_enabled", False))
        self.max_retry_sleep_ms = max(0, int(self.network_config.get("max_retry_sleep_ms", 1500)))
        self.default_receive_timeout_ms = int(self.network_config.get("default_receive_timeout_ms", 5000))
        self.health_shared_key = str(self.network_config.get("health_shared_key", "network_agent:health"))
        self.last_result_shared_key = str(self.network_config.get("last_result_shared_key", "network_agent:last_result"))
        self.history_shared_key = str(self.network_config.get("history_shared_key", "network_agent:history"))
        self.max_history = max(1, int(self.network_config.get("max_history", 200)))

        # Network subsystem
        self.adapters = NetworkAdapters()
        self.stream = NetworkStream()
        self.lifecycle = NetworkLifecycle()
        self.reliability = NetworkReliability()
        self.policy = NetworkPolicy()
        self.metrics = NetworkMetrics()

    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        """Dispatch supported network operations in a consistent agent-facing interface."""
        if isinstance(task_data, Mapping):
            payload = dict(task_data)
            operation = str(payload.get("operation", "relay")).strip().lower()
            if operation in {"relay", "send", "dispatch"}:
                envelope = ensure_mapping(payload.get("envelope"), field_name="envelope")
                constraints = ensure_mapping(payload.get("constraints"), field_name="constraints", allow_none=True)
                return self.relay(envelope=envelope, constraints=constraints)
            if operation in {"receive", "recv", "inbound"}:
                channel = str(payload.get("channel", payload.get("protocol", "http")))
                timeout_ms = int(payload.get("timeout_ms", self.default_receive_timeout_ms))
                endpoint = payload.get("endpoint")
                protocol = payload.get("protocol")
                return self.receive(
                    channel=channel,
                    timeout_ms=timeout_ms,
                    endpoint=str(endpoint) if endpoint is not None else None,
                    protocol=str(protocol) if protocol is not None else None,
                )
            if operation in {"health", "status", "snapshot"}:
                return self.get_network_health()
            raise PayloadValidationError(
                "Unsupported network agent operation.",
                context={"operation": "perform_task"},
                details={"supported": ["relay", "receive", "health"], "received": operation},
            )

        raise PayloadValidationError(
            "NetworkAgent expects mapping task payload.",
            context={"operation": "perform_task"},
            details={"received_type": type(task_data).__name__},
        )

    def relay(self, envelope: dict, constraints: dict | None = None) -> dict:
        """Select route, apply policy checks, send payload, and finalize lifecycle."""
        if not self.enabled:
            return {"status": "disabled", "agent": self.name, "reason": "network agent disabled"}

        started = time.monotonic()
        request_envelope = ensure_mapping(envelope, field_name="envelope")
        constraint_map = ensure_mapping(constraints, field_name="constraints", allow_none=True)

        begin = self.lifecycle.begin_delivery(
            envelope=request_envelope,
            payload=request_envelope.get("payload"),
            channel=request_envelope.get("channel"),
            protocol=request_envelope.get("protocol"),
            endpoint=request_envelope.get("endpoint"),
            route=request_envelope.get("route"),
            operation=request_envelope.get("operation"),
            timeout_ms=request_envelope.get("timeout_ms"),
            metadata=request_envelope.get("metadata"),
        )

        canonical_envelope = ensure_mapping(begin.get("envelope"), field_name="canonical_envelope")
        delivery = ensure_mapping(begin.get("delivery"), field_name="delivery", allow_none=True)
        message_id = delivery.get("message_id") or canonical_envelope.get("message_id")
        correlation_id = canonical_envelope.get("correlation_id")
        payload = canonical_envelope.get("payload")

        attempt = 1
        max_attempts = int(delivery.get("max_attempts") or self.max_delivery_attempts)
        last_error: Optional[BaseException] = None
        last_reliability: Optional[Dict[str, Any]] = None
        last_route_resolution: Optional[Dict[str, Any]] = None

        while attempt <= max_attempts:
            route_resolution: Optional[Dict[str, Any]] = None
            selected_route: Dict[str, Any] = {}
            try:
                policy_result = self.policy.assert_allowed(
                    envelope=canonical_envelope,
                    endpoint=canonical_envelope.get("endpoint"),
                    protocol=canonical_envelope.get("protocol"),
                    channel=canonical_envelope.get("channel"),
                    constraints=constraint_map,
                )

                route_resolution = self.stream.resolve_route(
                    protocol=canonical_envelope.get("protocol"),
                    channel=canonical_envelope.get("channel"),
                    endpoint=canonical_envelope.get("endpoint"),
                    region=canonical_envelope.get("region"),
                    operation=canonical_envelope.get("operation"),
                    constraints=constraint_map,
                )
                last_route_resolution = route_resolution
                selected_route = ensure_mapping(route_resolution.get("selected"), field_name="selected_route")

                if message_id:
                    self.lifecycle.mark_selected(
                        str(message_id),
                        metadata={
                            "route": selected_route,
                            "attempt": attempt,
                        },
                    )

                admission = self.reliability.admit_request(
                    endpoint=selected_route.get("endpoint") or canonical_envelope.get("endpoint"),
                    channel=selected_route.get("channel") or canonical_envelope.get("channel"),
                    protocol=selected_route.get("protocol") or canonical_envelope.get("protocol"),
                    route=selected_route.get("route") or canonical_envelope.get("route"),
                    current_route=selected_route,
                    candidates=route_resolution.get("candidates"),
                    request_context={"operation": canonical_envelope.get("operation")},
                    attempt=attempt,
                    correlation_id=correlation_id,
                    message_id=message_id,
                    metadata={"agent": self.name},
                )

                if not bool(admission.get("allow_request", True)) and admission.get("action") != "failover":
                    raise ReliabilityError(
                        "Request blocked by reliability admission controls.",
                        context={
                            "operation": "relay",
                            "endpoint": selected_route.get("endpoint"),
                            "channel": selected_route.get("channel"),
                            "protocol": selected_route.get("protocol"),
                            "route": selected_route.get("route"),
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "correlation_id": correlation_id,
                        },
                        details={"admission": admission},
                    )

                stream_result = self.stream.relay(
                    payload,
                    envelope=canonical_envelope,
                    protocol=selected_route.get("protocol") or canonical_envelope.get("protocol"),
                    channel=selected_route.get("channel") or canonical_envelope.get("channel"),
                    endpoint=selected_route.get("endpoint") or canonical_envelope.get("endpoint"),
                    operation=canonical_envelope.get("operation"),
                    constraints=constraint_map,
                    timeout_ms=canonical_envelope.get("timeout_ms"),
                    metadata=canonical_envelope.get("metadata"),
                    await_response=bool(canonical_envelope.get("await_response", False)),
                    receive_timeout_ms=canonical_envelope.get("receive_timeout_ms"),
                )

                if message_id:
                    self.lifecycle.mark_sent(str(message_id), metadata={"attempt": attempt})
                    self.lifecycle.complete(
                        str(message_id),
                        response_snapshot={"stream": stream_result, "attempt": attempt},
                        metadata={"status": "delivered"},
                    )

                success_route = ensure_mapping(stream_result.get("route"), field_name="stream_result.route", allow_none=True)
                self.reliability.record_success(
                    endpoint=selected_route.get("endpoint") or canonical_envelope.get("endpoint"),
                    channel=selected_route.get("channel") or canonical_envelope.get("channel"),
                    protocol=selected_route.get("protocol") or canonical_envelope.get("protocol"),
                    route=selected_route.get("route") or canonical_envelope.get("route"),
                    current_route=selected_route,
                    previous_route=success_route.get("secondary") if success_route else None,
                    used_failover=bool(success_route.get("used_secondary", False)) if success_route else None,
                    correlation_id=correlation_id,
                    message_id=message_id,
                    metadata={"attempt": attempt},
                )

                latency_ms = max(0.0, (time.monotonic() - started) * 1000.0)
                self.metrics.record_success(
                    channel=selected_route.get("channel") or canonical_envelope.get("channel") or "http",
                    endpoint=selected_route.get("endpoint") or canonical_envelope.get("endpoint"),
                    protocol=selected_route.get("protocol") or canonical_envelope.get("protocol"),
                    route=selected_route.get("route") or canonical_envelope.get("route"),
                    operation=canonical_envelope.get("operation"),
                    latency_ms=latency_ms,
                    retry_count=max(0, attempt - 1),
                    delivery_state="acked",
                    message_id=message_id,
                    correlation_id=correlation_id,
                    metadata={"policy": policy_result.get("decision")},
                )

                result = {
                    "status": "delivered",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "lifecycle": begin,
                    "policy": policy_result,
                    "route": route_resolution,
                    "stream": stream_result,
                    "correlation_id": correlation_id,
                    "message_id": message_id,
                    "elapsed_ms": round(latency_ms, 3),
                }
                self._persist_agent_result(result)
                return result

            except NetworkError as exc:
                last_error = exc
                latency_ms = max(0.0, (time.monotonic() - started) * 1000.0)

                if message_id:
                    self.lifecycle.fail(
                        str(message_id),
                        exc,
                        metadata={"attempt": attempt},
                    )

                failure_route = selected_route or {}
                last_reliability = self.reliability.record_failure(
                    exc,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    endpoint=failure_route.get("endpoint") or canonical_envelope.get("endpoint"),
                    channel=failure_route.get("channel") or canonical_envelope.get("channel"),
                    protocol=failure_route.get("protocol") or canonical_envelope.get("protocol"),
                    route=failure_route.get("route") or canonical_envelope.get("route"),
                    current_route=failure_route,
                    candidates=(route_resolution or {}).get("candidates") if route_resolution else None,
                    request_context={"operation": canonical_envelope.get("operation")},
                    correlation_id=correlation_id,
                    message_id=message_id,
                    status_code=getattr(exc, "status_code", None),
                    metadata={"agent": self.name},
                )

                self.metrics.record_failure(
                    channel=failure_route.get("channel") or canonical_envelope.get("channel") or "http",
                    endpoint=failure_route.get("endpoint") or canonical_envelope.get("endpoint"),
                    protocol=failure_route.get("protocol") or canonical_envelope.get("protocol"),
                    route=failure_route.get("route") or canonical_envelope.get("route"),
                    operation=canonical_envelope.get("operation"),
                    error=exc,
                    latency_ms=latency_ms,
                    retry_count=max(0, attempt - 1),
                    delivery_state="failed",
                    message_id=message_id,
                    correlation_id=correlation_id,
                    metadata={"attempt": attempt},
                )

                should_retry = self._should_retry(last_reliability)
                if attempt < max_attempts and should_retry:
                    if message_id:
                        self.lifecycle.plan_retry(
                            str(message_id),
                            exc,
                            metadata={"attempt": attempt},
                        )
                    self._sleep_for_retry(last_reliability)
                    attempt += 1
                    continue

                if message_id:
                    self.lifecycle.dead_letter(
                        str(message_id),
                        exc,
                        metadata={"attempt": attempt, "reason": "retry_exhausted_or_not_retryable"},
                    )
                break

            except Exception as exc:  # noqa: BLE001 - final boundary normalization.
                normalized = normalize_network_exception(
                    exc,
                    operation="network_agent_relay",
                    endpoint=canonical_envelope.get("endpoint"),
                    protocol=canonical_envelope.get("protocol"),
                    channel=canonical_envelope.get("channel"),
                    route=canonical_envelope.get("route"),
                    correlation_id=correlation_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    metadata={"agent": self.name},
                )
                last_error = normalized
                if message_id:
                    self.lifecycle.fail(str(message_id), normalized, metadata={"attempt": attempt})
                break

        failure_payload = {
            "status": "failed",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "message_id": message_id,
            "correlation_id": correlation_id,
            "route": last_route_resolution,
            "reliability": last_reliability,
            "error": json_safe(last_error.to_dict() if isinstance(last_error, NetworkError) else last_error),
            "elapsed_ms": round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
        }
        self._persist_agent_result(failure_payload)
        if isinstance(last_error, NetworkError):
            raise last_error
        raise normalize_network_exception(
            last_error or RuntimeError("relay failed without explicit error"),
            operation="network_agent_relay",
            endpoint=canonical_envelope.get("endpoint"),
            protocol=canonical_envelope.get("protocol"),
            channel=canonical_envelope.get("channel"),
            route=canonical_envelope.get("route"),
            correlation_id=correlation_id,
            message_id=message_id,
            metadata={"agent": self.name},
        )

    def receive(
        self,
        channel: str,
        timeout_ms: int = 5000,
        *,
        endpoint: Optional[str] = None,
        protocol: Optional[str] = None,
    ) -> dict:
        """Receive and normalize inbound message from a channel."""
        if not self.enabled:
            return {"status": "disabled", "agent": self.name, "reason": "network agent disabled"}

        normalized_channel = normalize_protocol_name(channel)
        resolved_protocol = normalize_protocol_name(protocol) if protocol is not None else normalized_channel
        started = time.monotonic()
        stream_result = self.stream.receive(
            channel=normalized_channel,
            protocol=resolved_protocol,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
        )

        result_payload = ensure_mapping(stream_result.get("result"), field_name="stream_result.result", allow_none=True)
        envelope = ensure_mapping(result_payload.get("envelope"), field_name="result.envelope", allow_none=True)
        inbound_payload = result_payload.get("payload")

        lifecycle_result = self.lifecycle.receive_inbound(
            envelope=envelope or None,
            payload=inbound_payload,
            channel=envelope.get("channel") or normalized_channel,
            protocol=envelope.get("protocol") or resolved_protocol,
            endpoint=envelope.get("endpoint") or endpoint,
            route=envelope.get("route"),
            timeout_ms=timeout_ms,
            metadata={"source": "network_agent.receive"},
        )

        delivery = ensure_mapping(lifecycle_result.get("delivery"), field_name="lifecycle.delivery", allow_none=True)
        message_id = delivery.get("message_id") or envelope.get("message_id")
        correlation_id = delivery.get("correlation_id") or envelope.get("correlation_id")

        self.metrics.record_success(
            channel=envelope.get("channel") or normalized_channel,
            endpoint=envelope.get("endpoint") or endpoint,
            protocol=envelope.get("protocol") or resolved_protocol,
            route=envelope.get("route"),
            operation="receive",
            latency_ms=max(0.0, (time.monotonic() - started) * 1000.0),
            delivery_state="received",
            message_id=message_id,
            correlation_id=correlation_id,
            metadata={"source": "network_agent.receive"},
        )

        response = {
            "status": "received",
            "stream": stream_result,
            "lifecycle": lifecycle_result,
            "message_id": message_id,
            "correlation_id": correlation_id,
            "channel": normalized_channel,
        }
        self._persist_agent_result(response)
        return response

    def get_network_health(self) -> dict:
        """Return channel/endpoint health, circuit states, and recent failures."""
        health = {
            "agent": self.name,
            "enabled": self.enabled,
            "network_memory": self.network_memory.get_network_health(),
            "stream": self.stream.get_stream_health(),
            "reliability": self.reliability.get_network_health(),
            "policy": self.policy.get_policy_state(),
            "lifecycle": self.lifecycle.get_snapshot(),
            "metrics": self.metrics.export_snapshot(),
        }
        self.shared_memory.set(self.health_shared_key, sanitize_for_logging(health))
        return health

    def _should_retry(self, reliability_payload: Optional[Mapping[str, Any]]) -> bool:
        payload = ensure_mapping(reliability_payload, field_name="reliability_payload", allow_none=True)
        if not payload:
            return False
        action = str(payload.get("action", "")).strip().lower()
        if action in {"retry", "failover"}:
            return True
        return bool(payload.get("should_retry", False))

    def _sleep_for_retry(self, reliability_payload: Optional[Mapping[str, Any]]) -> None:
        if not self.retry_sleep_enabled:
            return
        payload = ensure_mapping(reliability_payload, field_name="reliability_payload", allow_none=True)
        retry_payload = ensure_mapping(payload.get("retry"), field_name="retry_payload", allow_none=True)
        decision = ensure_mapping(retry_payload.get("decision"), field_name="retry_decision", allow_none=True)
        retry_after_ms = int(decision.get("retry_after_ms") or 0)
        retry_after_ms = min(max(0, retry_after_ms), self.max_retry_sleep_ms)
        if retry_after_ms > 0:
            time.sleep(retry_after_ms / 1000.0)

    def _persist_agent_result(self, result: Mapping[str, Any]) -> None:
        snapshot = sanitize_for_logging(json_safe(result))
        self.shared_memory.set(self.last_result_shared_key, snapshot)
        history = self.shared_memory.get(self.history_shared_key) or []
        if not isinstance(history, list):
            history = [history]
        history.append(snapshot)
        if len(history) > self.max_history:
            history = history[-self.max_history :]
        self.shared_memory.set(self.history_shared_key, history)


if __name__ == "__main__":
    print("\n=== Running Network Agent ===\n")
    printer.status("TEST", "Network Agent initialized", "info")

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from time import sleep

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    factory = AgentFactory()
    agent = NetworkAgent(shared_memory=shared_memory, agent_factory=factory, config=None)

    # Inject fake subsystem modules for deterministic local smoke coverage.
    agent.policy = NetworkPolicy()
    agent.stream = NetworkStream()
    agent.lifecycle = NetworkLifecycle()
    agent.reliability = NetworkReliability()
    agent.metrics = NetworkMetrics()
    agent.network_memory = NetworkMemory()

    class _SmokeHTTPHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _write_json(self, status: int, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            self._write_json(200, '{"ok": true, "method": "GET", "source": "network_agent.__main__"}')

        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            if content_length > 0:
                self.rfile.read(content_length)
            self._write_json(200, '{"ok": true, "method": "POST", "source": "network_agent.__main__"}')

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    class _SmokeHTTPServer:
        def __init__(self) -> None:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SmokeHTTPHandler)
            self.thread = Thread(target=self.server.serve_forever, daemon=True)

        @property
        def endpoint(self) -> str:
            host, port = self.server.server_address
            return f"http://{host}:{port}/relay"

        def start(self) -> None:
            self.thread.start()
            sleep(0.15)

        def stop(self) -> None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2.0)

    smoke_server = _SmokeHTTPServer()
    smoke_server.start()
    smoke_endpoint = smoke_server.endpoint

    try:
        relay_task = {
            "operation": "relay",
            "envelope": {
                "payload": {"hello": "network"},
                "channel": "http",
                "protocol": "http",
                "endpoint": smoke_endpoint,
                "route": "primary",
                "operation": "send",
                "timeout_ms": 1500,
                "metadata": {"source": "__main__"},
            },
            "constraints": {"tls_required": False, "allow_loopback_hosts": True},
        }
        relay_result = agent.perform_task(relay_task)
        printer.pretty("NETWORK RELAY", relay_result, "success")

        receive_result = agent.perform_task(
            {
                "operation": "receive",
                "channel": "http",
                "protocol": "http",
                "endpoint": smoke_endpoint,
                "timeout_ms": 1500,
            }
        )
        printer.pretty("NETWORK RECEIVE", receive_result, "success")

        health_result = agent.perform_task({"operation": "health"})
        printer.pretty("NETWORK HEALTH", health_result, "success")
    finally:
        smoke_server.stop()

    print("\n=== Successfully ran the Network Agent smoke block ===\n")
