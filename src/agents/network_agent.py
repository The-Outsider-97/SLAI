"""
Network Agent is specialized enough to own communications relay responsibilities,
while broad enough to support the full lifecycle of networked operations across SLAI.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .base_agent import BaseAgent
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import load_global_config, get_config_section
from .network import *
from .network.delivery_predictor import DeliveryContext, DeliveryDecision, DeliveryPredictor, classify_error
from .network.utils.network_errors import *
from .network.utils.network_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Network Agent")
printer = PrettyPrinter()

# Errors that originate locally (policy, validation, lifecycle state, local credentials). They say
# nothing about the remote endpoint, so they never feed endpoint health or its circuit breaker, and
# retrying them cannot help.
_LOCAL_ERROR_TYPES = (
    PolicyViolationError,
    PayloadError,
    NetworkConfigurationError,
    DuplicateMessageError,
    IdempotencyViolationError,
    DeliveryStateError,
    DeliveryExpiredError,
    AuthenticationFailedError,
    AuthorizationFailedError,
)


@dataclass
class _AttemptPlan:
    """Per-attempt transport parameters chosen before sending."""

    timeout_ms: Optional[int]
    decision: Optional[Dict[str, Any]] = None
    prediction: Optional[DeliveryDecision] = None
    context: Optional[DeliveryContext] = None
    sent_at: float = 0.0
    transport_ms: Optional[float] = None


class _RelayState:
    """Mutable state of one relay() call shared by its attempt helpers."""

    __slots__ = (
        "begin", "envelope", "constraints", "message_id", "correlation_id", "started", "max_attempts",
        "deadline_ms", "deadline_at", "idempotent", "last_route_resolution", "last_reliability", "last_plan",
        "warnings", "payload_bytes", "last_error_class", "last_retry_prediction",
    )

    def __init__(self, *, begin, envelope, constraints, message_id, correlation_id, started, max_attempts,
                 deadline_ms, idempotent) -> None:
        self.begin = begin
        self.envelope = envelope
        self.constraints = constraints
        self.message_id = message_id
        self.correlation_id = correlation_id
        self.started = started
        self.max_attempts = max_attempts
        self.deadline_ms = deadline_ms
        self.deadline_at = None if deadline_ms is None else started + deadline_ms / 1000.0
        self.idempotent = idempotent
        self.last_route_resolution: Optional[Dict[str, Any]] = None
        self.last_reliability: Optional[Dict[str, Any]] = None
        self.last_plan: Optional[_AttemptPlan] = None
        self.warnings: List[str] = []
        self.payload_bytes: Optional[int] = None
        self.last_error_class: str = "none"
        self.last_retry_prediction: Optional[Dict[str, Any]] = None

    def remaining_ms(self) -> Optional[float]:
        if self.deadline_at is None:
            return None
        return (self.deadline_at - time.monotonic()) * 1000.0


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
        assert_valid_config_contract(
            global_config=self.config,
            agent_key="network_agent",
            agent_config=self.network_config,
            logger=logger,
            require_global_keys=False,
            require_agent_section=False,
            warn_unknown_global_keys=False,
        )

        self.name = "NetworkAgent"
        self.enabled = bool(self.network_config.get("enabled", True))
        self.max_delivery_attempts = max(1, int(self.network_config.get("max_delivery_attempts", 3)))
        self.default_retry_profile = str(self.network_config.get("default_retry_profile", "default")).strip() or "default"
        self.queue_retry_profile = str(self.network_config.get("queue_retry_profile", "queue_backpressure")).strip() or self.default_retry_profile
        self.retry_profile_by_channel = ensure_mapping(self.network_config.get("retry_profile_by_channel"), field_name="retry_profile_by_channel", allow_none=True)
        self.retry_sleep_enabled = bool(self.network_config.get("retry_sleep_enabled", False))
        self.max_retry_sleep_ms = max(0, int(self.network_config.get("max_retry_sleep_ms", 1500)))
        self.default_receive_timeout_ms = max(1, int(self.network_config.get("default_receive_timeout_ms", 5000)))
        self.health_shared_key = self._resolve_shared_memory_key("health_shared_key", "health", "network_agent:health")
        self.last_result_shared_key = self._resolve_shared_memory_key("last_result_shared_key", "last_result", "network_agent:last_result")
        self.history_shared_key = self._resolve_shared_memory_key("history_shared_key", "history", "network_agent:history")
        self.max_history = max(1, int(self.network_config.get("max_history", 200)))
        self.coordination_state_key = str(self.network_config.get("coordination_state_key", "network_agent:coordination_state"))
        self.event_stream_key = str(self.network_config.get("event_stream_key", "network_agent:event_stream"))
        self.max_event_stream = max(10, int(self.network_config.get("max_event_stream", 300)))
        self.publish_topic = str(self.network_config.get("publish_topic", "agent.network.events"))
        self.task_events_topic = str(self.network_config.get("task_events_topic", "task_events"))
        self.agent_stats_key = str(self.network_config.get("agent_stats_key", "agent_stats"))
        self.agent_presence_key = str(self.network_config.get("agent_presence_key", f"agent:{self.name}"))
        self.active_agent_key = str(self.network_config.get("active_agent_key", "active_agent"))

        # Network subsystem: keep all components on one shared memory surface
        # so route, lifecycle, reliability, and telemetry state stays coherent.
        self.network_memory = NetworkMemory()
        self.adapters = NetworkAdapters(memory=self.network_memory)
        self.stream = NetworkStream(memory=self.network_memory, adapters=self.adapters)
        self.lifecycle = NetworkLifecycle(memory=self.network_memory)
        self.reliability = NetworkReliability(memory=self.network_memory)
        self.policy = NetworkPolicy(memory=self.network_memory)
        self.metrics = NetworkMetrics(memory=self.network_memory)

        self._network_agent_closed = False
        self._closing = False
        self._bookkeeping_errors = 0
        # NetworkAgent bookkeeping read-modify-writes shared memory; serialize it so concurrent
        # relay()/receive() calls cannot lose updates (active_tasks, history, coordination state).
        self._state_lock = threading.RLock()

        # Delivery bounds. The attempt ceiling stops envelope-supplied max_attempts from causing retry storms.
        self.max_attempts_ceiling = max(self.max_delivery_attempts, int(self.network_config.get("max_delivery_attempts_ceiling", 10)))
        self.default_deadline_ms = max(0, int(self.network_config.get("default_deadline_ms", 0) or 0))
        self.max_deadline_ms = max(1000, int(self.network_config.get("max_deadline_ms", 3_600_000)))
        self.min_attempt_budget_ms = max(1, int(self.network_config.get("min_attempt_budget_ms", 10)))
        self.enforce_policy_on_receive = bool(self.network_config.get("enforce_policy_on_receive", True))

        # Predictive delivery controller. Defaults to shadow mode; it can only shorten a timeout or veto a
        # retry that reliability already allowed, and degrades to the deterministic path on any problem.
        predictor_config = dict(get_config_section("delivery_predictor") or {})
        predictor_config.update(dict(ensure_mapping(self.network_config.get("delivery_predictor"), field_name="delivery_predictor", allow_none=True)))
        self.delivery_predictor = DeliveryPredictor(predictor_config, base_dir=Path(__file__).resolve().parents[2])

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
                timeout_ms = max(1, int(payload.get("timeout_ms", self.default_receive_timeout_ms)))
                endpoint = payload.get("endpoint")
                protocol = payload.get("protocol")
                receive_constraints = ensure_mapping(payload.get("constraints"), field_name="constraints", allow_none=True)
                return self.receive(
                    channel=channel,
                    timeout_ms=timeout_ms,
                    endpoint=str(endpoint) if endpoint is not None else None,
                    protocol=str(protocol) if protocol is not None else None,
                    constraints=receive_constraints,
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
        """
        Select route, apply policy checks, send payload, and finalize lifecycle.

        Delivery contract:

        * Once ``stream.relay`` has returned, the message *is delivered*. Post-delivery
          bookkeeping (lifecycle completion, reliability/metrics accounting, shared-memory
          persistence, events) is best-effort: a failure there is logged and surfaced in
          ``result["bookkeeping_warnings"]`` but never turns the delivery into a failure or a
          retry (which would duplicate the message).
        * Retries are bounded by ``max_attempts`` (capped by ``max_delivery_attempts_ceiling``),
          by the reliability layer, and by an optional overall ``deadline_ms`` (constraints or
          envelope, falling back to ``default_deadline_ms``). Each attempt's timeout is clamped to
          the remaining budget and a retry is not started if the budget cannot cover it.
        * Failure bookkeeping always runs (failure metrics, dead-letter, persisted result, events),
          even if the reliability layer itself raises (for example ``FailoverExhaustedError``).
        * If the call is interrupted (``KeyboardInterrupt``, ``SystemExit``, task cancellation) the
          delivery is marked failed so its idempotency claim is released, then the interrupt is
          re-raised.
        """
        if not self.enabled:
            return {"status": "disabled", "agent": self.name, "reason": "network agent disabled"}
        self._ensure_open("relay")

        self._mark_agent_activity(operation="relay", stage="start")
        started = time.monotonic()
        request_envelope = ensure_mapping(envelope, field_name="envelope")
        constraint_map = ensure_mapping(constraints, field_name="constraints", allow_none=True)
        state: Optional[_RelayState] = None
        attempt = 1

        try:
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
            deadline_ms = self._resolve_deadline_ms(request_envelope, constraint_map)
            state = _RelayState(
                begin=begin,
                envelope=canonical_envelope,
                constraints=constraint_map,
                message_id=delivery.get("message_id") or canonical_envelope.get("message_id"),
                correlation_id=canonical_envelope.get("correlation_id"),
                started=started,
                max_attempts=self._resolve_max_attempts(delivery.get("max_attempts")),
                deadline_ms=deadline_ms,
                idempotent=self._resolve_idempotent(canonical_envelope),
            )

            error: Optional[NetworkError] = None
            while attempt <= state.max_attempts:
                result, retry, error = self._run_attempt(state, attempt)
                if result is not None:
                    return result
                if retry:
                    attempt += 1
                    continue
                break

            return self._raise_relay_failure(state, attempt, error)
        except BaseException as exc:
            if not isinstance(exc, Exception) and state is not None:
                self._abort_delivery(state, exc, attempt)
            raise
        finally:
            self._mark_agent_activity(operation="relay", stage="end")

    # ------------------------------------------------------------------
    # Relay internals
    # ------------------------------------------------------------------
    def _run_attempt(self, state: "_RelayState", attempt: int) -> Tuple[Optional[dict], bool, Optional[NetworkError]]:
        """Run one delivery attempt. Returns ``(result, retry, error)``; ``result`` is set on delivery."""
        attempt_started = time.monotonic()
        envelope = state.envelope
        route_resolution: Optional[Dict[str, Any]] = None
        selected_route: Dict[str, Any] = {}
        plan: Optional[_AttemptPlan] = None
        state.last_plan = None

        try:
            remaining_ms = state.remaining_ms()
            if remaining_ms is not None and remaining_ms < self.min_attempt_budget_ms:
                raise DeliveryTimeoutError(
                    "Overall delivery deadline exhausted before the attempt could start.",
                    context={
                        "operation": "relay",
                        "endpoint": envelope.get("endpoint"),
                        "channel": envelope.get("channel"),
                        "protocol": envelope.get("protocol"),
                        "attempt": attempt,
                        "max_attempts": state.max_attempts,
                        "correlation_id": state.correlation_id,
                    },
                    details={"deadline_ms": state.deadline_ms, "remaining_ms": round(remaining_ms, 3)},
                )

            policy_result = self.policy.assert_allowed(
                envelope=envelope,
                endpoint=envelope.get("endpoint"),
                protocol=envelope.get("protocol"),
                channel=envelope.get("channel"),
                constraints=state.constraints,
            )

            # Let circuits whose open timeout has elapsed offer a probe before routes are chosen.
            self._best_effort(state, "reliability.refresh_circuits", self.reliability.refresh_circuits)
            route_resolution = self.stream.resolve_route(
                protocol=envelope.get("protocol"),
                channel=envelope.get("channel"),
                endpoint=envelope.get("endpoint"),
                region=envelope.get("region"),
                operation=envelope.get("operation"),
                constraints=state.constraints,
            )
            state.last_route_resolution = route_resolution
            selected_route = ensure_mapping(route_resolution.get("selected"), field_name="selected_route")

            if state.message_id:
                self.lifecycle.mark_selected(
                    str(state.message_id),
                    metadata={"route": selected_route, "attempt": attempt},
                )

            admission = self.reliability.admit_request(
                endpoint=selected_route.get("endpoint") or envelope.get("endpoint"),
                channel=selected_route.get("channel") or envelope.get("channel"),
                protocol=selected_route.get("protocol") or envelope.get("protocol"),
                route=selected_route.get("route") or envelope.get("route"),
                current_route=selected_route,
                candidates=route_resolution.get("candidates"),
                request_context={"operation": envelope.get("operation")},
                attempt=attempt,
                correlation_id=state.correlation_id,
                message_id=state.message_id,
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
                        "max_attempts": state.max_attempts,
                        "correlation_id": state.correlation_id,
                    },
                    details={"admission": admission},
                )

            # Planned here (not earlier) so the prediction is about the endpoint that will actually be used.
            plan = self._plan_attempt(state, attempt, state.remaining_ms(), selected_route)
            plan.sent_at = time.monotonic()
            stream_result = self.stream.relay(
                envelope.get("payload"),
                envelope=envelope,
                protocol=selected_route.get("protocol") or envelope.get("protocol"),
                channel=selected_route.get("channel") or envelope.get("channel"),
                endpoint=selected_route.get("endpoint") or envelope.get("endpoint"),
                operation=envelope.get("operation"),
                constraints=state.constraints,
                timeout_ms=plan.timeout_ms,
                metadata=envelope.get("metadata"),
                await_response=bool(envelope.get("await_response", False)),
                receive_timeout_ms=envelope.get("receive_timeout_ms"),
            )
            plan.transport_ms = (time.monotonic() - plan.sent_at) * 1000.0
        except Exception as exc:  # noqa: BLE001 - final boundary normalization.
            if plan is not None and plan.sent_at and plan.transport_ms is None:
                plan.transport_ms = (time.monotonic() - plan.sent_at) * 1000.0
            retry, final_error = self._handle_attempt_failure(
                state,
                exc,
                attempt=attempt,
                attempt_started=attempt_started,
                route_resolution=route_resolution,
                selected_route=selected_route,
            )
            return None, retry, final_error

        # The message has been handed to the transport: from here on nothing may cause a retry.
        result = self._finish_delivery(
            state,
            attempt=attempt,
            attempt_started=attempt_started,
            policy_result=policy_result,
            route_resolution=route_resolution,
            selected_route=selected_route,
            stream_result=stream_result,
            plan=plan,
        )
        return result, False, None

    def _finish_delivery(
        self,
        state: "_RelayState",
        *,
        attempt: int,
        attempt_started: float,
        policy_result: Mapping[str, Any],
        route_resolution: Mapping[str, Any],
        selected_route: Mapping[str, Any],
        stream_result: Mapping[str, Any],
        plan: "_AttemptPlan",
    ) -> dict:
        envelope = state.envelope
        message_id = str(state.message_id) if state.message_id else None
        now = time.monotonic()
        attempt_latency_ms = max(0.0, (now - attempt_started) * 1000.0)
        elapsed_ms = max(0.0, (now - state.started) * 1000.0)
        endpoint = selected_route.get("endpoint") or envelope.get("endpoint")
        channel = selected_route.get("channel") or envelope.get("channel")
        protocol = selected_route.get("protocol") or envelope.get("protocol")
        route_name = selected_route.get("route") or envelope.get("route")

        if message_id:
            self._best_effort(state, "lifecycle.mark_sent", self.lifecycle.mark_sent, message_id, metadata={"attempt": attempt})
            self._best_effort(
                state,
                "lifecycle.complete",
                self.lifecycle.complete,
                message_id,
                response_snapshot={"stream": stream_result, "attempt": attempt},
                metadata={"status": "delivered"},
            )

        success_route = ensure_mapping(stream_result.get("route"), field_name="stream_result.route", allow_none=True)
        self._best_effort(
            state,
            "reliability.record_success",
            self.reliability.record_success,
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route=route_name,
            current_route=selected_route,
            previous_route=success_route.get("secondary") if success_route else None,
            used_failover=bool(success_route.get("used_secondary", False)) if success_route else None,
            correlation_id=state.correlation_id,
            message_id=state.message_id,
            metadata={"attempt": attempt},
        )
        self._best_effort(
            state,
            "metrics.record_success",
            self.metrics.record_success,
            channel=channel or "http",
            endpoint=endpoint,
            protocol=protocol,
            route=route_name,
            operation=envelope.get("operation"),
            latency_ms=attempt_latency_ms,
            retry_count=max(0, attempt - 1),
            delivery_state="acked",
            message_id=state.message_id,
            correlation_id=state.correlation_id,
            metadata={"policy": policy_result.get("decision")},
        )

        result = {
            "status": "delivered",
            "attempt": attempt,
            "max_attempts": state.max_attempts,
            "lifecycle": state.begin,
            "policy": policy_result,
            "route": route_resolution,
            "stream": stream_result,
            "correlation_id": state.correlation_id,
            "message_id": state.message_id,
            "elapsed_ms": round(elapsed_ms, 3),
            "attempt_latency_ms": round(attempt_latency_ms, 3),
        }
        if plan.decision is not None:
            result["delivery_prediction"] = plan.decision
        if state.warnings:
            result["bookkeeping_warnings"] = list(state.warnings)

        self._best_effort(state, "persist_result", self._persist_agent_result, result)
        self._best_effort(state, "agent_activity", self._mark_agent_activity, operation="relay", stage="success")
        self._best_effort(
            state,
            "coordination_event",
            self._record_coordination_event,
            event_type="relay.delivered",
            payload={
                "message_id": state.message_id,
                "correlation_id": state.correlation_id,
                "attempt": attempt,
                "max_attempts": state.max_attempts,
                "selected_route": selected_route,
                "policy": policy_result.get("decision"),
            },
        )
        self._record_delivery_outcome(
            state,
            attempt=attempt,
            plan=plan,
            selected_route=selected_route,
            success=True,
            latency_ms=attempt_latency_ms,
            error=None,
        )
        return result

    def _handle_attempt_failure(
        self,
        state: "_RelayState",
        exc: Exception,
        *,
        attempt: int,
        attempt_started: float,
        route_resolution: Optional[Mapping[str, Any]],
        selected_route: Mapping[str, Any],
    ) -> Tuple[bool, NetworkError]:
        """
        Record one failed attempt and decide whether to retry.

        Returns ``(retry, error)``. This method never raises ``Exception``: every bookkeeping
        step is isolated so the original delivery error is what the caller ultimately sees.
        """
        envelope = state.envelope
        message_id = str(state.message_id) if state.message_id else None
        route = dict(selected_route or {})
        endpoint = route.get("endpoint") or envelope.get("endpoint")
        channel = route.get("channel") or envelope.get("channel")
        protocol = route.get("protocol") or envelope.get("protocol")
        route_name = route.get("route") or envelope.get("route")
        attempt_latency_ms = max(0.0, (time.monotonic() - attempt_started) * 1000.0)
        source = "network_error" if isinstance(exc, NetworkError) else "exception_boundary"

        if isinstance(exc, NetworkError):
            error: NetworkError = exc
        else:
            error = normalize_network_exception(
                exc,
                operation="network_agent_relay",
                endpoint=envelope.get("endpoint"),
                protocol=envelope.get("protocol"),
                channel=envelope.get("channel"),
                route=envelope.get("route"),
                correlation_id=state.correlation_id,
                attempt=attempt,
                max_attempts=state.max_attempts,
                metadata={"agent": self.name},
            )

        attributable = self._is_endpoint_attributable(error)
        state.last_error_class = classify_error(error)
        if message_id:
            self._best_effort(state, "lifecycle.fail", self.lifecycle.fail, message_id, error, metadata={"attempt": attempt})

        reliability, final_error = self._evaluate_failure(
            state,
            error,
            attributable=attributable,
            attempt=attempt,
            endpoint=endpoint,
            channel=channel,
            protocol=protocol,
            route_name=route_name,
            route=route,
            route_resolution=route_resolution,
            source=source,
        )
        state.last_reliability = reliability

        self._best_effort(
            state,
            "metrics.record_failure",
            self.metrics.record_failure,
            channel=channel or "http",
            # Local errors (policy denial, invalid payload, duplicate key...) say nothing about the
            # remote endpoint, so they must not degrade its health or trip its circuit breaker.
            endpoint=endpoint if attributable else None,
            protocol=protocol,
            route=route_name,
            operation=envelope.get("operation"),
            error=error,
            latency_ms=attempt_latency_ms,
            retry_count=max(0, attempt - 1),
            delivery_state="failed",
            message_id=state.message_id,
            correlation_id=state.correlation_id,
            metadata={"attempt": attempt, **({"source": source} if source != "network_error" else {})},
        )
        self._record_delivery_outcome(
            state,
            attempt=attempt,
            plan=state.last_plan,
            selected_route=route,
            success=False,
            latency_ms=attempt_latency_ms,
            error=error,
        )

        retry = attempt < state.max_attempts and final_error is None and self._should_retry(reliability)
        reason = "retry_exhausted_or_not_retryable" if source == "network_error" else "exception_boundary"
        sleep_ms = 0
        if retry:
            retry, sleep_ms = self._check_retry_budget(state, reliability)
            if not retry:
                reason = "deadline_exhausted"
        if retry:
            retry = self._gate_retry(state, attempt=attempt, error=error, selected_route=route)
            if not retry:
                reason = "predicted_retry_ineffective"
        if retry and message_id:
            try:
                self.lifecycle.plan_retry(message_id, error, metadata={"attempt": attempt})
            except Exception as plan_exc:  # noqa: BLE001
                # Never re-send a message whose lifecycle/idempotency state cannot be tracked.
                logger.warning("NetworkAgent could not plan retry for %s: %s", message_id, plan_exc)
                state.warnings.append("lifecycle.plan_retry")
                retry = False
                reason = "retry_planning_failed"

        if retry:
            self._sleep_for_retry(reliability, planned_ms=sleep_ms)
            return True, error

        if message_id:
            self._best_effort(
                state,
                "lifecycle.dead_letter",
                self.lifecycle.dead_letter,
                message_id,
                error,
                metadata={"attempt": attempt, "reason": reason},
            )
        outcome = final_error if final_error is not None else error
        if outcome is not error and outcome.__cause__ is None:
            outcome.__cause__ = error
        self._best_effort(state, "agent_activity", self._mark_agent_activity, operation="relay", stage="failure")
        self._best_effort(
            state,
            "coordination_event",
            self._record_coordination_event,
            event_type="relay.failed_network_error" if source == "network_error" else "relay.failed_exception_boundary",
            payload={
                "message_id": state.message_id,
                "correlation_id": state.correlation_id,
                "attempt": attempt,
                "max_attempts": state.max_attempts,
                "route": route,
                "reliability": reliability,
                "reason": reason,
                "delivery_prediction": state.last_retry_prediction,
                "error": json_safe(error.to_dict()),
            },
        )
        return False, outcome

    def _evaluate_failure(
        self,
        state: "_RelayState",
        error: NetworkError,
        *,
        attributable: bool,
        attempt: int,
        endpoint: Any,
        channel: Any,
        protocol: Any,
        route_name: Any,
        route: Mapping[str, Any],
        route_resolution: Optional[Mapping[str, Any]],
        source: str,
    ) -> Tuple[Dict[str, Any], Optional[NetworkError]]:
        """Ask the reliability layer for a recovery decision without letting it abort the relay."""
        envelope = state.envelope
        if not attributable:
            return (
                {
                    "action": "fail",
                    "should_retry": False,
                    "reason": "local, non-transport error: retrying cannot help and the endpoint is not at fault",
                    "source": "network_agent",
                },
                None,
            )
        try:
            decision = self.reliability.record_failure(
                error,
                attempt=attempt,
                max_attempts=state.max_attempts,
                endpoint=endpoint,
                channel=channel,
                protocol=protocol,
                route=route_name,
                current_route=route or None,
                candidates=(route_resolution or {}).get("candidates") if route_resolution else None,
                request_context={"operation": envelope.get("operation")},
                correlation_id=state.correlation_id,
                message_id=state.message_id,
                idempotent=state.idempotent,
                status_code=getattr(error, "status_code", None),
                metadata={"agent": self.name, **({"source": source} if source != "network_error" else {})},
                retry_profile=self._resolve_retry_profile_for_route(selected_route=route, envelope=envelope),
            )
            return decision, None
        except NetworkError as reliability_exc:
            # e.g. FailoverExhaustedError: the reliability layer reached a terminal verdict by raising.
            return (
                {
                    "action": "fail",
                    "should_retry": False,
                    "reason": reliability_exc.message,
                    "source": "reliability_exception",
                    "error": json_safe(reliability_exc.to_dict()),
                },
                reliability_exc,
            )
        except Exception as reliability_exc:  # noqa: BLE001
            logger.warning("NetworkAgent reliability evaluation failed: %s", reliability_exc)
            state.warnings.append("reliability.record_failure")
            return (
                {
                    "action": "fail",
                    "should_retry": False,
                    "reason": "reliability evaluation unavailable",
                    "source": "reliability_unavailable",
                },
                None,
            )

    def _raise_relay_failure(self, state: "_RelayState", attempt: int, error: Optional[NetworkError]) -> dict:
        final_error: BaseException = error or RuntimeError("relay failed without explicit error")
        failure_payload = {
            "status": "failed",
            "attempt": attempt,
            "max_attempts": state.max_attempts,
            "message_id": state.message_id,
            "correlation_id": state.correlation_id,
            "route": state.last_route_resolution,
            "reliability": state.last_reliability,
            "error": json_safe(final_error.to_dict() if isinstance(final_error, NetworkError) else final_error),
            "elapsed_ms": round(max(0.0, (time.monotonic() - state.started) * 1000.0), 3),
        }
        if state.warnings:
            failure_payload["bookkeeping_warnings"] = list(state.warnings)
        self._best_effort(state, "persist_result", self._persist_agent_result, failure_payload)
        self._best_effort(
            state,
            "coordination_event",
            self._record_coordination_event,
            event_type="relay.failed",
            payload={
                "message_id": state.message_id,
                "correlation_id": state.correlation_id,
                "attempt": attempt,
                "max_attempts": state.max_attempts,
                "route": state.last_route_resolution,
                "reliability": state.last_reliability,
                "error": failure_payload.get("error"),
            },
        )
        if isinstance(final_error, NetworkError):
            raise final_error
        raise normalize_network_exception(
            final_error,
            operation="network_agent_relay",
            endpoint=state.envelope.get("endpoint"),
            protocol=state.envelope.get("protocol"),
            channel=state.envelope.get("channel"),
            route=state.envelope.get("route"),
            correlation_id=state.correlation_id,
            message_id=state.message_id,  # type: ignore
            metadata={"agent": self.name},
        )

    def _abort_delivery(self, state: "_RelayState", exc: BaseException, attempt: int) -> None:
        """Release lifecycle/idempotency state when relay is interrupted by a non-Exception."""
        if not state.message_id:
            return
        try:
            self.lifecycle.fail(
                str(state.message_id),
                RuntimeError(f"relay interrupted by {type(exc).__name__}"),
                metadata={"attempt": attempt, "reason": "interrupted"},
            )
        except Exception as abort_exc:  # noqa: BLE001
            logger.warning("NetworkAgent could not release delivery %s after interrupt: %s", state.message_id, abort_exc)
        try:
            self._mark_agent_activity(operation="relay", stage="failure")
        except Exception:  # noqa: BLE001
            pass

    def _best_effort(self, state: Optional["_RelayState"], label: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a bookkeeping step; log and record (never raise) on failure."""
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NetworkAgent bookkeeping step '%s' failed: %s", label, exc)
            self._bookkeeping_errors += 1
            if state is not None:
                state.warnings.append(label)
            return None

    def _ensure_open(self, operation: str) -> None:
        if self._network_agent_closed or self._closing:
            raise SessionClosedError(
                "NetworkAgent has been shut down and no longer accepts communication requests.",
                context={"operation": operation},
                retryable=False,
                retry_disposition=RetryDisposition.NEVER,
            )

    def _resolve_max_attempts(self, requested: Any) -> int:
        """Caller/envelope supplied attempt counts are clamped: they must not enable retry storms."""
        try:
            value = int(requested) if requested else self.max_delivery_attempts
        except (TypeError, ValueError):
            value = self.max_delivery_attempts
        return max(1, min(value, self.max_attempts_ceiling))

    def _resolve_deadline_ms(self, envelope: Mapping[str, Any], constraints: Mapping[str, Any]) -> Optional[float]:
        raw = constraints.get("deadline_ms")
        if raw is None:
            raw = envelope.get("deadline_ms")
        if raw is None:
            raw = self.default_deadline_ms or None
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float("nan")
        if not (value > 0.0) or value == float("inf"):
            raise PayloadValidationError(
                "deadline_ms must be a positive, finite number of milliseconds.",
                context={"operation": "relay"},
                details={"received": repr(raw)[:64]},
            )
        return min(value, float(self.max_deadline_ms))

    @staticmethod
    def _resolve_idempotent(envelope: Mapping[str, Any]) -> bool:
        """Honour an explicit ``idempotent`` flag (envelope or metadata); default stays True."""
        for source in (envelope, envelope.get("metadata") if isinstance(envelope.get("metadata"), Mapping) else {}):
            if "idempotent" in source:
                value = source["idempotent"]
                if isinstance(value, str):
                    return value.strip().lower() not in {"false", "0", "no", "off"}
                return bool(value)
        return True

    @staticmethod
    def _is_endpoint_attributable(error: BaseException) -> bool:
        """Whether a failure says anything about the remote endpoint's health."""
        return not isinstance(error, _LOCAL_ERROR_TYPES)

    def _check_retry_budget(self, state: "_RelayState", reliability: Optional[Mapping[str, Any]]) -> Tuple[bool, int]:
        sleep_ms = self._planned_sleep_ms(reliability)
        remaining_ms = state.remaining_ms()
        if remaining_ms is None:
            return True, sleep_ms
        if remaining_ms - sleep_ms < self.min_attempt_budget_ms:
            return False, 0
        return True, sleep_ms

    def _payload_bytes(self, state: "_RelayState") -> int:
        if state.payload_bytes is None:
            try:
                state.payload_bytes = int(estimate_payload_size(state.envelope.get("payload")))
            except Exception:  # noqa: BLE001 - a feature only; never worth failing a delivery over
                state.payload_bytes = 0
        return state.payload_bytes

    def _delivery_context(
        self,
        state: "_RelayState",
        attempt: int,
        requested_timeout_ms: Optional[int],
        remaining_ms: Optional[float],
        selected_route: Optional[Mapping[str, Any]],
    ) -> DeliveryContext:
        route = selected_route or {}
        endpoint = route.get("endpoint") or state.envelope.get("endpoint") or ""
        return DeliveryContext(
            endpoint=str(endpoint),
            attempt=attempt,
            max_attempts=state.max_attempts,
            requested_timeout_ms=requested_timeout_ms,
            remaining_ms=remaining_ms,
            payload_bytes=self._payload_bytes(state),
            protocol=str(route.get("protocol") or state.envelope.get("protocol") or "http"),
            idempotent=state.idempotent,
            last_error_class=state.last_error_class,
        )

    def _plan_attempt(
        self,
        state: "_RelayState",
        attempt: int,
        remaining_ms: Optional[float],
        selected_route: Optional[Mapping[str, Any]] = None,
    ) -> "_AttemptPlan":
        """
        Choose the transport timeout for one attempt.

        The deterministic part clamps the caller's timeout to the remaining deadline budget. The
        DeliveryPredictor may then *shorten* it (never lengthen, never below its floor) when it is in
        ``active`` mode; in ``shadow`` mode it only records what it would have done.
        """
        requested = state.envelope.get("timeout_ms")
        timeout_ms: Optional[int] = None if requested is None else int(requested)
        if remaining_ms is not None:
            base = timeout_ms if timeout_ms is not None else int(getattr(self.stream, "default_request_timeout_ms", self.default_receive_timeout_ms))
            timeout_ms = max(1, min(base, int(remaining_ms)))

        plan = _AttemptPlan(timeout_ms=timeout_ms)
        if self.delivery_predictor.mode != "off":
            context = self._delivery_context(state, attempt, timeout_ms, remaining_ms, selected_route)
            prediction = self.delivery_predictor.plan(context)
            plan.context = context
            plan.prediction = prediction
            plan.decision = prediction.to_dict()
            if prediction.timeout_ms is not None and (timeout_ms is None or prediction.timeout_ms <= timeout_ms):
                plan.timeout_ms = prediction.timeout_ms
        state.last_plan = plan
        return plan

    def _gate_retry(self, state: "_RelayState", *, attempt: int, error: NetworkError, selected_route: Mapping[str, Any]) -> bool:
        """
        Last veto before a retry that policy and reliability already permit.

        Deterministic default: allow. The DeliveryPredictor can veto only on evidence (enough endpoint
        history, consecutive endpoint failures, a low predicted success probability) and only in
        ``active`` mode.
        """
        if self.delivery_predictor.mode == "off":
            return True
        plan = state.last_plan
        requested = plan.context.requested_timeout_ms if plan is not None and plan.context is not None else None
        context = self._delivery_context(state, attempt + 1, requested, state.remaining_ms(), selected_route)
        decision = self.delivery_predictor.allow_retry(context)
        state.last_retry_prediction = decision.to_dict()
        return bool(decision.allow_retry)

    def _record_delivery_outcome(
        self,
        state: "_RelayState",
        *,
        attempt: int,
        plan: Optional["_AttemptPlan"],
        selected_route: Mapping[str, Any],
        success: bool,
        latency_ms: float,
        error: Optional[NetworkError],
    ) -> None:
        """Feed the attempt outcome back to the predictor (transport time only; local errors are not evidence)."""
        if plan is None or plan.context is None or not plan.sent_at:
            return  # the attempt failed before anything was sent
        if error is not None and not self._is_endpoint_attributable(error):
            return
        transport_ms = plan.transport_ms if plan.transport_ms is not None else latency_ms
        self.delivery_predictor.observe(
            plan.context.endpoint,
            ok=success,
            latency_ms=transport_ms,
            error=error,
            decision=plan.prediction,
            context=plan.context,
        )

    def receive(
        self,
        channel: str,
        timeout_ms: int = 5000,
        *,
        endpoint: Optional[str] = None,
        protocol: Optional[str] = None,
        constraints: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """
        Receive and normalize inbound message from a channel.

        When an explicit ``endpoint`` is given the same destination/protocol/TLS policy that guards
        ``relay`` is applied first (``enforce_policy_on_receive``), because receiving may open an
        outbound connection to a caller-supplied destination.
        """
        if not self.enabled:
            return {"status": "disabled", "agent": self.name, "reason": "network agent disabled"}
        self._ensure_open("receive")

        self._mark_agent_activity(operation="receive", stage="start")
        try:
            return self._receive_impl(channel, timeout_ms, endpoint=endpoint, protocol=protocol, constraints=constraints)
        finally:
            self._mark_agent_activity(operation="receive", stage="end")

    def _receive_impl(
        self,
        channel: str,
        timeout_ms: int,
        *,
        endpoint: Optional[str],
        protocol: Optional[str],
        constraints: Optional[Mapping[str, Any]],
    ) -> dict:
        normalized_channel = normalize_protocol_name(channel)
        resolved_protocol = normalize_protocol_name(protocol) if protocol is not None else normalized_channel
        constraint_map = ensure_mapping(constraints, field_name="constraints", allow_none=True)
        started = time.monotonic()
        try:
            if endpoint is not None and self.enforce_policy_on_receive:
                self.policy.assert_allowed(
                    envelope={
                        "channel": normalized_channel,
                        "protocol": resolved_protocol,
                        "endpoint": endpoint,
                        "operation": "receive",
                    },
                    endpoint=endpoint,
                    protocol=resolved_protocol,
                    channel=normalized_channel,
                    constraints=constraint_map,
                )

            stream_result = self.stream.receive(
                channel=normalized_channel,
                protocol=resolved_protocol,
                endpoint=endpoint,
                constraints=constraint_map or None,
                timeout_ms=max(1, int(timeout_ms)),
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
                timeout_ms=max(1, int(timeout_ms)),
                metadata={"source": "network_agent.receive"},
            )
        except Exception as exc:  # noqa: BLE001 - final boundary normalization.
            normalized = exc if isinstance(exc, NetworkError) else normalize_network_exception(
                exc,
                operation="network_agent_receive",
                endpoint=endpoint,
                protocol=resolved_protocol,
                channel=normalized_channel,
                metadata={"agent": self.name},
            )
            attributable = self._is_endpoint_attributable(normalized)
            self._best_effort(
                None,
                "metrics.record_failure",
                self.metrics.record_failure,
                channel=normalized_channel,
                endpoint=endpoint if attributable else None,
                protocol=resolved_protocol,
                operation="receive",
                error=normalized,
                latency_ms=max(0.0, (time.monotonic() - started) * 1000.0),
                delivery_state="receive_failed",
                metadata={"source": "network_agent.receive"},
            )
            failure_payload = {
                "status": "failed",
                "operation": "receive",
                "channel": normalized_channel,
                "protocol": resolved_protocol,
                "endpoint": endpoint,
                "error": json_safe(normalized.to_dict()),
            }
            self._best_effort(None, "persist_result", self._persist_agent_result, failure_payload)
            self._best_effort(None, "agent_activity", self._mark_agent_activity, operation="receive", stage="failure")
            self._best_effort(None, "coordination_event", self._record_coordination_event, event_type="receive.failed", payload=failure_payload)
            if normalized is exc:
                raise
            raise normalized from exc

        # The inbound message has been received and registered: telemetry must not discard it.
        delivery = ensure_mapping(lifecycle_result.get("delivery"), field_name="lifecycle.delivery", allow_none=True)
        message_id = delivery.get("message_id") or envelope.get("message_id")
        correlation_id = delivery.get("correlation_id") or envelope.get("correlation_id")

        self._best_effort(
            None,
            "metrics.record_success",
            self.metrics.record_success,
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
        self._best_effort(None, "persist_result", self._persist_agent_result, response)
        self._best_effort(None, "agent_activity", self._mark_agent_activity, operation="receive", stage="success")
        self._best_effort(
            None,
            "coordination_event",
            self._record_coordination_event,
            event_type="receive.delivered",
            payload={
                "message_id": message_id,
                "correlation_id": correlation_id,
                "channel": normalized_channel,
                "protocol": resolved_protocol,
                "endpoint": endpoint,
            },
        )
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
            "delivery_predictor": self.delivery_predictor.get_snapshot(),
            "shared_memory_keys": {
                "health": self.health_shared_key,
                "last_result": self.last_result_shared_key,
                "history": self.history_shared_key,
            },
        }
        self.shared_memory.set(self.health_shared_key, sanitize_for_logging(health))
        self._record_coordination_event(
            event_type="health.snapshot",
            payload={"enabled": self.enabled, "metrics": health.get("metrics")},
        )
        return health

    def _resolve_retry_profile_for_route(self, *, selected_route: Mapping[str, Any], envelope: Mapping[str, Any]) -> str:
        """Resolve retry profile by explicit envelope hints and channel/protocol defaults."""
        override = envelope.get("retry_profile") or envelope.get("reliability_retry_profile")
        if isinstance(override, str) and override.strip():
            return override.strip()

        channel = str(selected_route.get("channel") or envelope.get("channel") or "").strip().lower()
        protocol = str(selected_route.get("protocol") or envelope.get("protocol") or "").strip().lower()

        if channel and channel in self.retry_profile_by_channel:
            value = self.retry_profile_by_channel.get(channel)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if protocol and protocol in self.retry_profile_by_channel:
            value = self.retry_profile_by_channel.get(protocol)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if channel == "queue" or protocol in {"queue", "amqp", "amqps", "mq", "sqs", "kafka", "pubsub"}:
            return self.queue_retry_profile
        return self.default_retry_profile

    def _should_retry(self, reliability_payload: Optional[Mapping[str, Any]]) -> bool:
        payload = ensure_mapping(reliability_payload, field_name="reliability_payload", allow_none=True)
        if not payload:
            return False
        action = str(payload.get("action", "")).strip().lower()
        if action in {"retry", "failover"}:
            return True
        return bool(payload.get("should_retry", False))

    def _planned_sleep_ms(self, reliability_payload: Optional[Mapping[str, Any]]) -> int:
        """
        Backoff before the next attempt, in ms (0 when ``retry_sleep_enabled`` is off).

        ``RetryPolicy`` publishes its jittered exponential backoff as ``wait_ms`` (it already folds in a
        server ``Retry-After``); ``retry_after_ms`` is only a legacy fallback. Reading only the latter
        meant the computed backoff was never applied.
        """
        if not self.retry_sleep_enabled:
            return 0
        payload = ensure_mapping(reliability_payload, field_name="reliability_payload", allow_none=True)
        retry_payload = ensure_mapping(payload.get("retry"), field_name="retry_payload", allow_none=True)
        decision = ensure_mapping(retry_payload.get("decision"), field_name="retry_decision", allow_none=True)
        wait_ms = decision.get("wait_ms")
        if wait_ms is None:
            wait_ms = decision.get("retry_after_ms")
        try:
            wait = int(wait_ms or 0)
        except (TypeError, ValueError):
            wait = 0
        return min(max(0, wait), self.max_retry_sleep_ms)

    def _sleep_for_retry(self, reliability_payload: Optional[Mapping[str, Any]], *, planned_ms: Optional[int] = None) -> None:
        sleep_ms = self._planned_sleep_ms(reliability_payload) if planned_ms is None else max(0, int(planned_ms))
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    @staticmethod
    def _summarize_result(result: Mapping[str, Any]) -> Dict[str, Any]:
        """Compact, scalar-only view of a result for histories (full results are ~30 KB each)."""
        route = result.get("route") if isinstance(result.get("route"), Mapping) else {}
        selected = route.get("selected") if isinstance(route.get("selected"), Mapping) else {}
        error = result.get("error") if isinstance(result.get("error"), Mapping) else {}
        keys = ("status", "operation", "message_id", "correlation_id", "attempt", "max_attempts",
                "elapsed_ms", "attempt_latency_ms", "channel", "protocol", "endpoint")
        summary = {key: result.get(key) for key in keys if isinstance(result.get(key), (str, int, float, bool))}
        for key in ("endpoint", "channel", "protocol"):
            if key not in summary and isinstance(selected.get(key), (str, int, float, bool)):
                summary[key] = selected.get(key)
        if error:
            summary["error_code"] = error.get("code")
            summary["error_type"] = error.get("type")
        if result.get("bookkeeping_warnings"):
            summary["bookkeeping_warnings"] = list(result["bookkeeping_warnings"])
        return summary

    def _persist_agent_result(self, result: Mapping[str, Any]) -> None:
        snapshot = sanitize_for_logging(json_safe(result))
        summary = sanitize_for_logging(json_safe(self._summarize_result(result)))
        with self._state_lock:
            self.shared_memory.set(self.last_result_shared_key, snapshot)
            history = self.shared_memory.get(self.history_shared_key) or []
            if not isinstance(history, list):
                history = [history]
            history.append(summary)
            if len(history) > self.max_history:
                history = history[-self.max_history :]
            self.shared_memory.set(self.history_shared_key, history)
            self._update_coordination_state(last_result=summary, history_size=len(history))

    def _record_coordination_event(self, *, event_type: str, payload: Mapping[str, Any]) -> None:
        """
        Push a structured network-agent event into shared memory and publish it
        to subscribers when pub/sub is supported by SharedMemory.
        """
        event = sanitize_for_logging(
            json_safe(
                {
                    "event_type": str(event_type),
                    "agent": self.name,
                    "occurred_at_ms": int(time.time() * 1000),
                    "payload": ensure_mapping(payload, field_name="payload", allow_none=True),
                }
            )
        )
        with self._state_lock:
            history = self.shared_memory.get(self.event_stream_key) or []
            if not isinstance(history, list):
                history = [history]
            history.append(event)
            if len(history) > self.max_event_stream:
                history = history[-self.max_event_stream :]
            self.shared_memory.set(self.event_stream_key, history)

        if hasattr(self.shared_memory, "publish"):
            try:
                self.shared_memory.publish(self.publish_topic, event)
                self.shared_memory.publish(
                    self.task_events_topic,
                    {
                        "event": str(event_type),
                        "agent": self.name,
                        "payload": ensure_mapping(payload, field_name="payload", allow_none=True),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("NetworkAgent publish failed: %s", exc)

        self._update_coordination_state(last_event=event, event_stream_size=len(history))

    def _update_coordination_state(self, *, last_result: Optional[Mapping[str, Any]] = None, history_size: Optional[int] = None, last_event: Optional[Mapping[str, Any]] = None, event_stream_size: Optional[int] = None) -> None:
        """
        Keep a compact, deterministic state block that other agents can poll for
        handoffs without scanning large histories.
        """
        with self._state_lock:
            self._update_coordination_state_locked(
                last_result=last_result, history_size=history_size,
                last_event=last_event, event_stream_size=event_stream_size,
            )

    def _update_coordination_state_locked(self, *, last_result: Optional[Mapping[str, Any]] = None, history_size: Optional[int] = None, last_event: Optional[Mapping[str, Any]] = None, event_stream_size: Optional[int] = None) -> None:
        state = self.shared_memory.get(self.coordination_state_key) or {}
        if not isinstance(state, dict):
            state = {"previous_state": json_safe(state)}

        state.update(
            {
                "agent": self.name,
                "updated_at_ms": int(time.time() * 1000),
                "enabled": self.enabled,
                "shared_memory_keys": {
                    "health": self.health_shared_key,
                    "last_result": self.last_result_shared_key,
                    "history": self.history_shared_key,
                    "coordination_state": self.coordination_state_key,
                    "event_stream": self.event_stream_key,
                },
                "network_capabilities": {
                    "supports_relay": True,
                    "supports_receive": True,
                    "supports_health_snapshot": True,
                    "max_delivery_attempts": self.max_delivery_attempts,
                    "default_receive_timeout_ms": self.default_receive_timeout_ms,
                },
            }
        )

        if last_result is not None:
            state["last_result"] = sanitize_for_logging(json_safe(last_result))
        if history_size is not None:
            state["result_history_size"] = int(history_size)
        if last_event is not None:
            state["last_event"] = sanitize_for_logging(json_safe(last_event))
        if event_stream_size is not None:
            state["event_stream_size"] = int(event_stream_size)

        self.shared_memory.set(self.coordination_state_key, state)

    def _mark_agent_activity(self, *, operation: str, stage: str) -> None:
        """
        Align with cross-agent shared-memory conventions used by TaskRouter and
        execution/collaboration flows (`agent_stats`, `agent:<name>`, `active_agent`).
        """
        with self._state_lock:
            self._mark_agent_activity_locked(operation=operation, stage=stage)

    def _mark_agent_activity_locked(self, *, operation: str, stage: str) -> None:
        now = time.time()
        stats = self.shared_memory.get(self.agent_stats_key, {}) or {}
        if not isinstance(stats, dict):
            stats = {}
        row = stats.setdefault(
            self.name,
            {"successes": 0, "failures": 0, "active_tasks": 0, "last_seen": 0.0},
        )
        row["last_seen"] = float(now)
        row["last_operation"] = str(operation)
        row["last_stage"] = str(stage)
        if stage == "start":
            row["active_tasks"] = max(0, int(row.get("active_tasks", 0)) + 1)
            self.shared_memory.set(self.active_agent_key, self.name)
        elif stage == "end":
            row["active_tasks"] = max(0, int(row.get("active_tasks", 0)) - 1)
        elif stage == "success":
            row["successes"] = int(row.get("successes", 0)) + 1
        elif stage == "failure":
            row["failures"] = int(row.get("failures", 0)) + 1
        stats[self.name] = row
        self.shared_memory.set(self.agent_stats_key, stats)

        heartbeat = self.shared_memory.get(self.agent_presence_key, {}) or {}
        if not isinstance(heartbeat, dict):
            heartbeat = {"status": "active"}
        heartbeat["name"] = self.name
        heartbeat["status"] = "active"
        heartbeat["last_seen"] = float(now) # type: ignore
        heartbeat["operation"] = str(operation)
        heartbeat["stage"] = str(stage)
        self.shared_memory.set(self.agent_presence_key, heartbeat)

    def _resolve_shared_memory_key(self, config_key: str, nested_key: str, default: str) -> str:
        """
        Resolve shared-memory key configuration while supporting both:
        - direct string values, and
        - nested mappings like: {network_agent: "health"}.
        """
        raw_value = self.network_config.get(config_key, default)
        if isinstance(raw_value, Mapping):
            nested_value = raw_value.get("network_agent", raw_value.get(nested_key, default))
            resolved = str(nested_value).strip()
            return resolved or default
        if raw_value is None:
            return default
        resolved = str(raw_value).strip()
        return resolved or default

    def shutdown(self) -> None:
        """
        Close NetworkAgent-owned communication resources.
    
        NetworkStream manages stream sessions and delegates adapter
        shutdown to the injected NetworkAdapters manager.
    
        NetworkMemory and external SharedMemory are not closed here.
        """
        if self._network_agent_closed:
            return
        # Refuse new relay()/receive() calls from here on; a failed shutdown can be retried.
        self._closing = True

        errors = []
        try:
            self.stream.close_stream(reason="network_agent_shutdown")
    
        except Exception as exc:
            errors.append(exc)
    
            # Fallback only when stream-level cleanup fails.
            try:
                self.adapters.close_all(reason="network_agent_shutdown_fallback")
    
            except Exception as adapter_exc:
                errors.append(adapter_exc)
    
        if errors:
            raise RuntimeError("NetworkAgent shutdown failed") from errors[0]
    
        self._network_agent_closed = True
    
    def close(self) -> None:
        self.shutdown()


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
    agent.network_memory = NetworkMemory()
    agent.adapters = NetworkAdapters(memory=agent.network_memory)
    agent.policy = NetworkPolicy(memory=agent.network_memory)
    agent.stream = NetworkStream(memory=agent.network_memory, adapters=agent.adapters)
    agent.lifecycle = NetworkLifecycle(memory=agent.network_memory)
    agent.reliability = NetworkReliability(memory=agent.network_memory)
    agent.metrics = NetworkMetrics(memory=agent.network_memory)

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

        def _log_message(self, fmt: str, *args: Any) -> None:
            return

    class _SmokeHTTPServer:
        def __init__(self) -> None:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SmokeHTTPHandler)
            self.thread = Thread(target=self.server.serve_forever, daemon=True)

        @property
        def endpoint(self) -> str:
            host, port = self.server.server_address # type: ignore
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
                "constraints": {"tls_required": False, "allow_loopback_hosts": True},
            }
        )
        printer.pretty("NETWORK RECEIVE", receive_result, "success")

        health_result = agent.perform_task({"operation": "health"})
        printer.pretty("NETWORK HEALTH", health_result, "success")
    finally:
        smoke_server.stop()

    print("\n=== Successfully ran the Network Agent smoke block ===\n")