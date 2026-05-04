from __future__ import annotations

"""
Production-grade task router for SLAI's collaborative runtime.

This module owns the task-to-agent execution boundary. It coordinates the
registry, routing strategy, reliability manager, shared-memory runtime stats,
and optional policy/contract validation hooks while preserving the original
TaskRouter public API.

Responsibilities
----------------
- Resolve eligible agents from AgentRegistry for a task type.
- Rank candidates through a pluggable BaseRouterStrategy.
- Execute candidates through ReliabilityManager with controlled fallback.
- Maintain shared-memory agent stats and heartbeat records.
- Emit route attempts, audit events, task events, and snapshots for operations.
- Optionally evaluate task contracts and policy decisions before execution.
- Raise structured collaboration errors instead of untyped RuntimeError at the
  router boundary.

Design principles
-----------------
1. Stable public API: ``TaskRouter``, ``route()``, ``_get_stats()``,
   ``_set_stats()``, ``_set_active_tasks()``, ``_record_success()``,
   ``_record_failure()``, and ``_touch_agent_heartbeat()`` remain available.
2. Separation of concerns: the router does not discover agents, decide circuit
   transitions, score agents internally, or own contract/policy registries.
3. Helper/error integration: normalization, redaction, task envelopes, audit
   events, stats updates, result payloads, and collaboration errors are routed
   through the collaborative utility layer.
4. Config-backed behavior: task-router tuning belongs in
   ``collaborative_config.yaml`` under ``task_router`` and interoperates with
   existing ``task_routing`` settings.
5. Safe degradation: candidate failures are isolated per attempt and the router
   proceeds to the next ranked agent when configured to do so.
"""

import threading
import time

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.collaboration_error import *
from .utils.collaborative_helpers import *
from .reliability import ReliabilityManager, RetryPolicy
from .router_strategy import BaseRouterStrategy, build_router_strategy
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Task Router")
printer = PrettyPrinter()

AgentMeta = Dict[str, Any]
AgentMap = Dict[str, AgentMeta]
StatsMap = Dict[str, Dict[str, Any]]
RankedAgent = Tuple[str, AgentMeta, float]
RouteHook = Callable[[Dict[str, Any]], None]


class RouteEventType(str, Enum):
    """Normalized task-router audit event labels."""

    ROUTER_INITIALIZED = "router_initialized"
    ROUTE_STARTED = "route_started"
    CONTRACT_REJECTED = "contract_rejected"
    POLICY_DENIED = "policy_denied"
    POLICY_REVIEW_REQUIRED = "policy_review_required"
    AGENTS_RESOLVED = "agents_resolved"
    AGENTS_RANKED = "agents_ranked"
    AGENT_SKIPPED = "agent_skipped"
    ATTEMPT_STARTED = "attempt_started"
    ATTEMPT_SUCCEEDED = "attempt_succeeded"
    ATTEMPT_FAILED = "attempt_failed"
    ROUTE_SUCCEEDED = "route_succeeded"
    ROUTE_FAILED = "route_failed"


@dataclass(frozen=True)
class TaskRouterConfig:
    """Runtime configuration for ``TaskRouter``.

    ``task_router`` settings are authoritative for router behavior while
    selected ``task_routing`` values remain supported for backward
    compatibility with existing collaborative configuration.
    """

    strategy: str = "weighted"
    stats_key: str = DEFAULT_AGENT_STATS_KEY
    audit_enabled: bool = True
    audit_key: str = "collaboration:task_router_events"
    audit_max_events: int = DEFAULT_MAX_AUDIT_EVENTS
    task_events_enabled: bool = True
    task_events_key: str = DEFAULT_TASK_EVENT_KEY
    task_events_max_events: int = DEFAULT_MAX_AUDIT_EVENTS
    route_history_limit: int = 1000
    include_ranking_report: bool = True
    update_agent_stats: bool = True
    publish_heartbeats: bool = True
    require_agent_instance: bool = True
    execution_method: str = "execute"
    fail_fast_on_unavailable: bool = False
    continue_on_agent_failure: bool = True
    max_candidate_attempts: Optional[int] = None
    normalize_task_payloads: bool = True
    contract_validation_enabled: bool = False
    policy_evaluation_enabled: bool = False
    deny_on_policy_review: bool = True
    fail_closed_on_policy_error: bool = False
    fail_closed_on_contract_error: bool = True
    record_success_results: bool = False
    result_preview_length: int = 500
    fallback_plan_enabled: bool = False
    fallback_task_data_key: str = "fallback_from"
    status_key: str = "collaboration:task_router_status"
    publish_status: bool = True
    redact_audit_payloads: bool = True

    @classmethod
    def from_config(
        cls,
        task_router_config: Optional[Mapping[str, Any]] = None,
        task_routing_config: Optional[Mapping[str, Any]] = None,
    ) -> "TaskRouterConfig":
        router = dict(task_router_config or {})
        routing = dict(task_routing_config or {})
        nested = routing.get("task_router") if isinstance(routing.get("task_router"), Mapping) else {}
        source = merge_mappings(routing, nested, router, deep=True, drop_none=True)
        max_attempts = source.get("max_candidate_attempts")
        return cls(
            strategy=str(source.get("strategy", cls.strategy)).strip() or cls.strategy,
            stats_key=str(source.get("stats_key", cls.stats_key)).strip() or cls.stats_key,
            audit_enabled=coerce_bool(source.get("audit_enabled"), default=cls.audit_enabled),
            audit_key=str(source.get("audit_key", cls.audit_key)).strip() or cls.audit_key,
            audit_max_events=coerce_int(source.get("audit_max_events"), default=cls.audit_max_events, minimum=1),
            task_events_enabled=coerce_bool(source.get("task_events_enabled"), default=cls.task_events_enabled),
            task_events_key=str(source.get("task_events_key", cls.task_events_key)).strip() or cls.task_events_key,
            task_events_max_events=coerce_int(source.get("task_events_max_events"), default=cls.task_events_max_events, minimum=1),
            route_history_limit=coerce_int(source.get("route_history_limit"), default=cls.route_history_limit, minimum=1),
            include_ranking_report=coerce_bool(source.get("include_ranking_report"), default=cls.include_ranking_report),
            update_agent_stats=coerce_bool(source.get("update_agent_stats"), default=cls.update_agent_stats),
            publish_heartbeats=coerce_bool(source.get("publish_heartbeats"), default=cls.publish_heartbeats),
            require_agent_instance=coerce_bool(source.get("require_agent_instance"), default=cls.require_agent_instance),
            execution_method=str(source.get("execution_method", cls.execution_method)).strip() or cls.execution_method,
            fail_fast_on_unavailable=coerce_bool(source.get("fail_fast_on_unavailable"), default=cls.fail_fast_on_unavailable),
            continue_on_agent_failure=coerce_bool(source.get("continue_on_agent_failure"), default=cls.continue_on_agent_failure),
            max_candidate_attempts=None if max_attempts is None else coerce_int(max_attempts, default=0, minimum=0),
            normalize_task_payloads=coerce_bool(source.get("normalize_task_payloads"), default=cls.normalize_task_payloads),
            contract_validation_enabled=coerce_bool(source.get("contract_validation_enabled"), default=cls.contract_validation_enabled),
            policy_evaluation_enabled=coerce_bool(source.get("policy_evaluation_enabled"), default=cls.policy_evaluation_enabled),
            deny_on_policy_review=coerce_bool(source.get("deny_on_policy_review"), default=cls.deny_on_policy_review),
            fail_closed_on_policy_error=coerce_bool(source.get("fail_closed_on_policy_error"), default=cls.fail_closed_on_policy_error),
            fail_closed_on_contract_error=coerce_bool(source.get("fail_closed_on_contract_error"), default=cls.fail_closed_on_contract_error),
            record_success_results=coerce_bool(source.get("record_success_results"), default=cls.record_success_results),
            result_preview_length=coerce_int(source.get("result_preview_length"), default=cls.result_preview_length, minimum=1),
            fallback_plan_enabled=coerce_bool(source.get("fallback_plan_enabled"), default=cls.fallback_plan_enabled),
            fallback_task_data_key=str(source.get("fallback_task_data_key", cls.fallback_task_data_key)).strip() or cls.fallback_task_data_key,
            status_key=str(source.get("status_key", cls.status_key)).strip() or cls.status_key,
            publish_status=coerce_bool(source.get("publish_status"), default=cls.publish_status),
            redact_audit_payloads=coerce_bool(source.get("redact_audit_payloads"), default=cls.redact_audit_payloads),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteAttemptRecord:
    """Serializable record for one candidate-agent execution attempt."""

    attempt_id: str
    route_id: str
    task_type: str
    agent_name: str
    rank: int
    score: float
    status: str
    started_at: float
    finished_at: float
    duration_ms: float
    error: Optional[Dict[str, Any]] = None
    result_fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self), drop_empty=True)
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class RouteRecord:
    """Bounded task-router history record."""

    route_id: str
    task_id: str
    task_type: str
    status: str
    started_at: float
    finished_at: float
    duration_ms: float
    selected_agent: Optional[str] = None
    candidate_count: int = 0
    ranked_count: int = 0
    attempts: Tuple[Dict[str, Any], ...] = ()
    ranking_report: Optional[Dict[str, Any]] = None
    contract_validation: Optional[Dict[str, Any]] = None
    policy_evaluation: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: generate_correlation_id("route"))

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self), drop_empty=True)
        return redact_mapping(payload) if redact else payload


class TaskRouter:
    """Capability-based router backed by AgentRegistry with strategy and reliability controls."""

    def __init__(
        self,
        registry=None,
        shared_memory=None,
        strategy: Optional[BaseRouterStrategy] = None,
        reliability_manager: Optional[ReliabilityManager] = None,
        *,
        contract_registry: Optional[Any] = None,
        policy_engine: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
        route_hooks: Optional[Iterable[RouteHook]] = None,
    ):
        self.registry = registry
        self.shared_memory = shared_memory
        self.config = load_global_config()
        self.router_config = get_config_section("task_router") or {}
        self.task_routing_config = get_config_section("task_routing") or {}
        self.runtime_config = TaskRouterConfig.from_config(
            merge_mappings(self.router_config, config or {}, deep=True, drop_none=True),
            self.task_routing_config,
        )
        self.strategy = strategy or build_router_strategy(self.runtime_config.strategy, config=self.task_routing_config)
        self.reliability_manager = reliability_manager or ReliabilityManager(shared_memory=shared_memory)
        self.contract_registry = contract_registry
        self.policy_engine = policy_engine
        self._lock = threading.RLock()
        self._route_history: Deque[Dict[str, Any]] = deque(maxlen=self.runtime_config.route_history_limit)
        self._last_route: Optional[Dict[str, Any]] = None
        self._route_hooks: List[RouteHook] = list(route_hooks or [])
        self._started_at = epoch_seconds()
        self._routes_started = 0
        self._routes_succeeded = 0
        self._routes_failed = 0
        self._attempts_started = 0
        self._attempts_succeeded = 0
        self._attempts_failed = 0

        self._audit(
            RouteEventType.ROUTER_INITIALIZED.value,
            "Task Router initialized.",
            severity="info",
            metadata={"config": self.runtime_config.to_dict(), "strategy": getattr(self.strategy, "name", type(self.strategy).__name__)},
        )
        self._publish_status()
        logger.info("Task Router initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def route(self, task_type: str, task_data: Dict[str, Any]) -> Any:
        """Route a task to the best available agent and return the agent result.

        The method intentionally preserves the original return contract: the
        successful agent result is returned directly rather than wrapped.
        Operational metadata is available through ``last_route()``,
        ``route_history()``, shared-memory audit keys, and status snapshots.
        """

        started_at = epoch_seconds()
        start_ms = monotonic_ms()
        route_id = generate_uuid("route", length=24)
        normalized_task_type = normalize_task_type(task_type)
        task_mapping = self._normalize_task_data(task_data)
        envelope = build_task_envelope(
            task_type=normalized_task_type,
            payload=task_mapping,
            required_capabilities=task_mapping.get("required_capabilities"),
            priority=task_mapping.get("priority"),
            retry_limit=task_mapping.get("retry_limit"),
            deadline_at=task_mapping.get("deadline_at"),
            timeout_seconds=task_mapping.get("timeout_seconds"),
            correlation_id=task_mapping.get("correlation_id"),
            task_id=task_mapping.get("task_id"),
            source=task_mapping.get("source", "task_router"),
            tags=task_mapping.get("tags"),
            metadata={"route_id": route_id},
        )
        validation = validate_task_envelope(envelope)
        if not validation.valid:
            error = self._routing_error(
                f"Invalid task envelope for '{normalized_task_type}'.",
                task_type=normalized_task_type,
                context={"validation": validation.to_dict(), "route_id": route_id},
            )
            self._finalize_route_failure(route_id, envelope, started_at, start_ms, error=error, attempts=[], metadata={"stage": "envelope_validation"})
            raise error

        with self._lock:
            self._routes_started += 1

        self._audit(
            RouteEventType.ROUTE_STARTED.value,
            f"Routing task '{normalized_task_type}'.",
            task=envelope,
            metadata={"route_id": route_id, "task_id": envelope.task_id},
        )
        self._publish_task_event(RouteEventType.ROUTE_STARTED.value, envelope, metadata={"route_id": route_id})

        contract_result = self._validate_contract(envelope)
        policy_result = self._evaluate_policy(envelope)

        eligible_agents = self._resolve_agents(normalized_task_type, envelope)
        if not eligible_agents:
            error = self._no_capable_agent_error(normalized_task_type, envelope, available_agents={})
            self._finalize_route_failure(
                route_id,
                envelope,
                started_at,
                start_ms,
                error=error,
                attempts=[],
                contract_validation=contract_result,
                policy_evaluation=policy_result,
                metadata={"stage": "agent_resolution"},
            )
            raise error

        ranked, ranking_report = self._rank_agents(eligible_agents, envelope)
        if not ranked:
            error = self._no_capable_agent_error(normalized_task_type, envelope, available_agents=eligible_agents)
            self._finalize_route_failure(
                route_id,
                envelope,
                started_at,
                start_ms,
                error=error,
                attempts=[],
                ranking_report=ranking_report,
                contract_validation=contract_result,
                policy_evaluation=policy_result,
                metadata={"stage": "ranking"},
            )
            raise error

        self._audit(
            RouteEventType.AGENTS_RANKED.value,
            f"Ranked {len(ranked)} candidate agent(s) for '{normalized_task_type}'.",
            task=envelope,
            metadata={"route_id": route_id, "ranked_agents": [name for name, _, _ in ranked]},
        )

        attempts: List[Dict[str, Any]] = []
        last_error: Optional[BaseException] = None
        candidate_limit = self.runtime_config.max_candidate_attempts or len(ranked)

        for rank, (agent_name, agent_meta, score) in enumerate(ranked[:candidate_limit], start=1):
            if envelope.expired:
                last_error = self._routing_error(
                    f"Task '{normalized_task_type}' expired before route completion.",
                    task_type=normalized_task_type,
                    context={"route_id": route_id, "task_id": envelope.task_id, "deadline_at": envelope.deadline_at},
                )
                break

            normalized_agent_name = normalize_agent_name(agent_name)
            if not self.reliability_manager.is_available(normalized_agent_name):
                last_error = self._routing_error(
                    f"Agent '{normalized_agent_name}' unavailable due to open circuit.",
                    task_type=normalized_task_type,
                    context={"route_id": route_id, "agent_name": normalized_agent_name, "rank": rank},
                )
                attempt = self._build_skipped_attempt(
                    route_id=route_id,
                    task_type=normalized_task_type,
                    agent_name=normalized_agent_name,
                    rank=rank,
                    score=score,
                    error=last_error,
                    reason="circuit_unavailable",
                )
                attempts.append(attempt)
                self._audit(RouteEventType.AGENT_SKIPPED.value, f"Skipped unavailable agent '{normalized_agent_name}'.", task=envelope, agent_name=normalized_agent_name, error=last_error)
                if self.runtime_config.fail_fast_on_unavailable:
                    break
                continue

            attempt_started = epoch_seconds()
            attempt_start_ms = monotonic_ms()
            self._audit(RouteEventType.ATTEMPT_STARTED.value, f"Attempting agent '{normalized_agent_name}'.", task=envelope, agent_name=normalized_agent_name, metadata={"rank": rank, "score": score})
            self._set_active_tasks(normalized_agent_name, +1)
            with self._lock:
                self._attempts_started += 1
            try:
                operation = self._build_agent_operation(normalized_agent_name, agent_meta, envelope)
                result = self.reliability_manager.execute(
                    normalized_agent_name,
                    operation,
                    metadata={"route_id": route_id, "task_id": envelope.task_id, "task_type": normalized_task_type, "rank": rank, "score": score},
                )
                self._record_success(normalized_agent_name, result=result)
                result_fingerprint = stable_hash(json_safe(result), length=16)
                attempt_record = RouteAttemptRecord(
                    attempt_id=generate_uuid("route_attempt", length=24),
                    route_id=route_id,
                    task_type=normalized_task_type,
                    agent_name=normalized_agent_name,
                    rank=rank,
                    score=float(score),
                    status="success",
                    started_at=attempt_started,
                    finished_at=epoch_seconds(),
                    duration_ms=elapsed_ms(attempt_start_ms),
                    result_fingerprint=result_fingerprint,
                    metadata={"capabilities": list(extract_agent_capabilities(agent_meta))},
                ).to_dict()
                attempts.append(attempt_record)
                with self._lock:
                    self._attempts_succeeded += 1
                    self._routes_succeeded += 1
                route_record = self._build_route_record(
                    route_id=route_id,
                    envelope=envelope,
                    status="success",
                    started_at=started_at,
                    duration_ms=elapsed_ms(start_ms),
                    selected_agent=normalized_agent_name,
                    candidate_count=len(eligible_agents),
                    ranked_count=len(ranked),
                    attempts=attempts,
                    ranking_report=ranking_report,
                    contract_validation=contract_result,
                    policy_evaluation=policy_result,
                    metadata={"result_fingerprint": result_fingerprint},
                )
                self._record_route(route_record)
                self._audit(RouteEventType.ATTEMPT_SUCCEEDED.value, f"Agent '{normalized_agent_name}' completed task '{normalized_task_type}'.", task=envelope, agent_name=normalized_agent_name, metadata={"route_id": route_id, "duration_ms": attempt_record.get("duration_ms")})
                self._audit(RouteEventType.ROUTE_SUCCEEDED.value, f"Task '{normalized_task_type}' routed successfully.", task=envelope, agent_name=normalized_agent_name, metadata={"route_id": route_id, "selected_agent": normalized_agent_name})
                self._publish_task_event(RouteEventType.ROUTE_SUCCEEDED.value, envelope, agent_name=normalized_agent_name, metadata={"route_id": route_id})
                self._publish_status()
                self._emit_hooks(route_record)
                return result
            except BaseException as exc:  # noqa: BLE001 - route boundary intentionally isolates candidate failures.
                last_error = exc
                self._record_failure(normalized_agent_name, error=exc)
                attempt_record = RouteAttemptRecord(
                    attempt_id=generate_uuid("route_attempt", length=24),
                    route_id=route_id,
                    task_type=normalized_task_type,
                    agent_name=normalized_agent_name,
                    rank=rank,
                    score=float(score),
                    status="failed",
                    started_at=attempt_started,
                    finished_at=epoch_seconds(),
                    duration_ms=elapsed_ms(attempt_start_ms),
                    error=exception_to_error_payload(exc, action="task_route_attempt").get("error"),
                    metadata={"capabilities": list(extract_agent_capabilities(agent_meta))},
                ).to_dict()
                attempts.append(attempt_record)
                with self._lock:
                    self._attempts_failed += 1
                self._audit(RouteEventType.ATTEMPT_FAILED.value, f"Agent '{normalized_agent_name}' failed for task '{normalized_task_type}'.", task=envelope, agent_name=normalized_agent_name, error=exc, metadata={"route_id": route_id, "rank": rank})
                logger.exception("Agent '%s' failed for task '%s'", normalized_agent_name, normalized_task_type)
                if not self.runtime_config.continue_on_agent_failure:
                    break
            finally:
                self._set_active_tasks(normalized_agent_name, -1)

        terminal_error = self._routing_error(
            f"All agents failed for task type '{normalized_task_type}'.",
            task_type=normalized_task_type,
            context={
                "route_id": route_id,
                "task_id": envelope.task_id,
                "attempt_count": len(attempts),
                "last_error": exception_to_error_payload(last_error, action="task_route").get("error") if last_error else None,
            },
            cause=last_error,
        )
        self._finalize_route_failure(
            route_id,
            envelope,
            started_at,
            start_ms,
            error=terminal_error,
            attempts=attempts,
            ranking_report=ranking_report,
            contract_validation=contract_result,
            policy_evaluation=policy_result,
            metadata={"stage": "execution", "candidate_count": len(eligible_agents), "ranked_count": len(ranked)},
        )
        raise terminal_error

    def explain_route(self, task_type: str, task_data: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return candidate and ranking diagnostics without executing agents."""

        normalized_task_type = normalize_task_type(task_type)
        envelope = build_task_envelope(task_type=normalized_task_type, payload=self._normalize_task_data(task_data or {}))
        agents = self._resolve_agents(normalized_task_type, envelope)
        ranked, report = self._rank_agents(agents, envelope)
        return redact_mapping(
            prune_none(
                {
                    "task_type": normalized_task_type,
                    "candidate_count": len(agents),
                    "ranked_count": len(ranked),
                    "ranked_agents": [{"rank": idx, "agent_name": name, "score": score, "capabilities": list(extract_agent_capabilities(meta))} for idx, (name, meta, score) in enumerate(ranked, start=1)],
                    "ranking_report": report,
                    "stats": self._get_stats(),
                },
                drop_empty=True,
            )
        )

    def route_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = list(self._route_history)
        if limit is None:
            return items
        return items[-max(0, int(limit)):]

    def last_route(self) -> Optional[Dict[str, Any]]:
        return self._last_route

    def register_hook(self, hook: RouteHook) -> None:
        if not callable(hook):
            raise self._routing_error("TaskRouter hook must be callable.", context={"received_type": type(hook).__name__})
        with self._lock:
            self._route_hooks.append(hook)

    def unregister_hook(self, hook: RouteHook) -> bool:
        with self._lock:
            if hook in self._route_hooks:
                self._route_hooks.remove(hook)
                return True
        return False

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            stats = self._get_stats()
            reliability_status = self.reliability_manager.status() if self.reliability_manager is not None else {}
            return redact_mapping(
                {
                    "component": "task_router",
                    "captured_at": epoch_seconds(),
                    "captured_at_utc": utc_timestamp(),
                    "uptime_seconds": round(epoch_seconds() - self._started_at, 6),
                    "routes_started": self._routes_started,
                    "routes_succeeded": self._routes_succeeded,
                    "routes_failed": self._routes_failed,
                    "attempts_started": self._attempts_started,
                    "attempts_succeeded": self._attempts_succeeded,
                    "attempts_failed": self._attempts_failed,
                    "strategy": getattr(self.strategy, "name", type(self.strategy).__name__),
                    "config": self.runtime_config.to_dict(),
                    "stats": stats,
                    "reliability_status": reliability_status,
                    "history_size": len(self._route_history),
                    "last_route": self._last_route,
                }
            )

    def health_report(self) -> Dict[str, Any]:
        snapshot = self.snapshot()
        failure_ratio = snapshot["routes_failed"] / snapshot["routes_started"] if snapshot.get("routes_started") else 0.0
        status = "degraded" if failure_ratio >= 0.25 and snapshot.get("routes_started", 0) >= 4 else "healthy"
        if self.registry is None:
            status = "unavailable"
        return {
            "status": status,
            "captured_at": snapshot.get("captured_at"),
            "captured_at_utc": snapshot.get("captured_at_utc"),
            "summary": {
                "routes_started": snapshot.get("routes_started", 0),
                "routes_succeeded": snapshot.get("routes_succeeded", 0),
                "routes_failed": snapshot.get("routes_failed", 0),
                "attempts_failed": snapshot.get("attempts_failed", 0),
                "history_size": snapshot.get("history_size", 0),
            },
        }

    def clear_history(self) -> None:
        with self._lock:
            self._route_history.clear()
            self._last_route = None

    # ------------------------------------------------------------------
    # Existing internal API retained
    # ------------------------------------------------------------------
    def _get_stats(self) -> Dict[str, Dict[str, Any]]:
        if self.shared_memory is None:
            return {}
        return get_agent_stats(self.shared_memory, stats_key=self.runtime_config.stats_key)

    def _set_stats(self, stats: Dict[str, Dict[str, Any]]) -> None:
        if self.shared_memory is not None:
            set_agent_stats(self.shared_memory, stats, stats_key=self.runtime_config.stats_key)

    def _set_active_tasks(self, agent_name: str, delta: int) -> None:
        if not self.runtime_config.update_agent_stats:
            return
        if self.shared_memory is not None:
            set_agent_active_delta(self.shared_memory, agent_name, int(delta), stats_key=self.runtime_config.stats_key)
        if self.runtime_config.publish_heartbeats:
            self._touch_agent_heartbeat(agent_name, epoch_seconds())

    def _record_success(self, agent_name: str, result: Any = None) -> None:
        if self.runtime_config.update_agent_stats and self.shared_memory is not None:
            stats = record_agent_success(self.shared_memory, agent_name, stats_key=self.runtime_config.stats_key)
            if self.runtime_config.record_success_results:
                row = stats.get(normalize_agent_name(agent_name), {})
                row["last_result_fingerprint"] = stable_hash(json_safe(result), length=16)
                self._set_stats(stats)
        if self.runtime_config.publish_heartbeats:
            self._touch_agent_heartbeat(agent_name, epoch_seconds())

    def _record_failure(self, agent_name: str, error: Optional[BaseException] = None) -> None:
        if self.runtime_config.update_agent_stats and self.shared_memory is not None:
            stats = record_agent_failure(self.shared_memory, agent_name, stats_key=self.runtime_config.stats_key)
            if error is not None:
                row = stats.get(normalize_agent_name(agent_name), {})
                row["last_error"] = exception_to_error_payload(error, action="task_router").get("error")
                self._set_stats(stats)
        if self.runtime_config.publish_heartbeats:
            self._touch_agent_heartbeat(agent_name, epoch_seconds())

    def _touch_agent_heartbeat(self, agent_name: str, timestamp: float) -> None:
        if self.shared_memory is None:
            return
        touch_agent_heartbeat(self.shared_memory, agent_name, timestamp=float(timestamp))

    # ------------------------------------------------------------------
    # Routing internals
    # ------------------------------------------------------------------
    def _normalize_task_data(self, task_data: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if task_data is None:
            return {}
        if not isinstance(task_data, Mapping):
            raise self._routing_error("task_data must be a mapping.", context={"received_type": type(task_data).__name__})
        if self.runtime_config.normalize_task_payloads:
            return normalize_task_payload(task_data, allow_none=True)
        return dict(task_data)

    def _resolve_agents(self, task_type: str, envelope: Any) -> AgentMap:
        if self.registry is None:
            raise self._routing_error("TaskRouter requires a registry.", task_type=task_type, context={"task_id": getattr(envelope, "task_id", None)})
        try:
            agents = self.registry.get_agents_by_task(task_type)
        except BaseException as exc:
            raise self._routing_error(
                f"Failed to resolve agents for task type '{task_type}'.",
                task_type=task_type,
                context={"task_id": getattr(envelope, "task_id", None)},
                cause=exc,
            ) from exc
        if not isinstance(agents, Mapping):
            raise self._routing_error("registry.get_agents_by_task must return a mapping.", task_type=task_type, context={"received_type": type(agents).__name__})
        normalized: AgentMap = {}
        for name, meta in agents.items():
            normalized_name = normalize_agent_name(name)
            normalized_meta = ensure_mapping(meta, field_name=f"agents[{normalized_name}]", allow_none=True)
            normalized_meta.setdefault("capabilities", list(extract_agent_capabilities(normalized_meta)))
            normalized[normalized_name] = normalized_meta
        self._audit(RouteEventType.AGENTS_RESOLVED.value, f"Resolved {len(normalized)} candidate agent(s) for '{task_type}'.", task=envelope, metadata={"agents": list(normalized.keys())})
        return normalized

    def _rank_agents(self, eligible_agents: AgentMap, envelope: Any) -> Tuple[List[RankedAgent], Optional[Dict[str, Any]]]:
        task_data = self._task_data_for_strategy(envelope)
        stats = self._get_stats()
        try:
            ranked = self.strategy.rank_agents(eligible_agents, stats=stats, task_data=task_data)
            ranking_report = None
            if self.runtime_config.include_ranking_report and hasattr(self.strategy, "last_report") and getattr(self.strategy, "last_report") is not None:
                report = getattr(self.strategy, "last_report")
                if hasattr(report, "to_dict"):
                    ranking_report = report.to_dict()
                elif isinstance(report, Mapping):
                    ranking_report = dict(report)
            return self._normalize_ranked_agents(ranked), ranking_report
        except BaseException as exc:
            raise self._routing_error("Router strategy failed to rank agents.", task_type=getattr(envelope, "task_type", "unknown"), context={"candidate_count": len(eligible_agents)}, cause=exc) from exc

    def _normalize_ranked_agents(self, ranked: Any) -> List[RankedAgent]:
        if ranked is None:
            return []
        normalized: List[RankedAgent] = []
        for index, item in enumerate(list(ranked), start=1):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 3:
                raise self._routing_error("Router strategy returned an invalid ranking item.", context={"index": index, "item": json_safe(item)})
            name, meta, score = item[0], item[1], item[2]
            normalized.append((normalize_agent_name(name), ensure_mapping(meta, field_name=f"ranked[{index}].meta", allow_none=True), coerce_float(score, default=0.0)))
        return normalized

    def _task_data_for_strategy(self, envelope: Any) -> Dict[str, Any]:
        data = merge_mappings(
            getattr(envelope, "payload", {}),
            {
                "task_type": getattr(envelope, "task_type", None),
                "required_capabilities": list(getattr(envelope, "required_capabilities", ()) or ()),
                "priority": getattr(envelope, "priority", None),
                "task_id": getattr(envelope, "task_id", None),
                "correlation_id": getattr(envelope, "correlation_id", None),
            },
            deep=True,
            drop_none=True,
        )
        return data

    def _build_agent_operation(self, agent_name: str, agent_meta: Mapping[str, Any], envelope: Any) -> Callable[[], Any]:
        instance = agent_meta.get("instance")
        if instance is None:
            if self.runtime_config.require_agent_instance:
                raise self._routing_error(
                    f"Agent '{agent_name}' has no executable instance.",
                    task_type=getattr(envelope, "task_type", "unknown"),
                    context={"agent_name": agent_name, "meta_keys": list(agent_meta.keys())},
                )
            instance = agent_meta.get("class")
        method = getattr(instance, self.runtime_config.execution_method, None)
        if not callable(method):
            raise self._routing_error(
                f"Agent '{agent_name}' does not expose callable '{self.runtime_config.execution_method}'.",
                task_type=getattr(envelope, "task_type", "unknown"),
                context={"agent_name": agent_name, "instance_type": type(instance).__name__},
            )
        task_payload = self._task_data_for_agent(envelope)
        return lambda: method(task_payload)

    def _task_data_for_agent(self, envelope: Any) -> Dict[str, Any]:
        payload = dict(getattr(envelope, "payload", {}) or {})
        payload.setdefault("task_type", getattr(envelope, "task_type", None))
        payload.setdefault("task_id", getattr(envelope, "task_id", None))
        payload.setdefault("correlation_id", getattr(envelope, "correlation_id", None))
        payload.setdefault("priority", getattr(envelope, "priority", None))
        payload.setdefault("required_capabilities", list(getattr(envelope, "required_capabilities", ()) or ()))
        return prune_none(payload, drop_empty=True)

    def _validate_contract(self, envelope: Any) -> Optional[Dict[str, Any]]:
        if not self.runtime_config.contract_validation_enabled or self.contract_registry is None:
            return None
        try:
            result = self.contract_registry.validate(envelope.task_type, envelope.payload)
            result_dict = contract_validation_to_dict(result) or {"valid": True}
            if not contract_is_valid(result):
                error = self._routing_error(
                    f"Task contract rejected payload for '{envelope.task_type}'.",
                    task_type=envelope.task_type,
                    context={"task_id": envelope.task_id, "contract_validation": result_dict},
                )
                self._audit(RouteEventType.CONTRACT_REJECTED.value, "Task contract validation failed.", task=envelope, error=error, metadata={"validation": result_dict})
                raise error
            return result_dict
        except BaseException as exc:
            if self.runtime_config.fail_closed_on_contract_error:
                if isinstance(exc, Exception):
                    raise
                raise self._routing_error("Task contract validation failed.", task_type=envelope.task_type, context={"task_id": envelope.task_id}, cause=exc) from exc
            logger.warning("Task contract validation failed open for task '%s': %s", envelope.task_type, exc)
            return {"valid": True, "warning": str(exc), "fail_open": True}

    def _evaluate_policy(self, envelope: Any) -> Optional[Dict[str, Any]]:
        if not self.runtime_config.policy_evaluation_enabled or self.policy_engine is None:
            return None
        try:
            evaluation = self.policy_engine.evaluate(envelope.payload, agent_meta={}, context={"task_type": envelope.task_type, "task_id": envelope.task_id, "envelope": envelope.to_dict(redact=True)})
            evaluation_dict = policy_evaluation_to_dict(evaluation) or {"decision": "allow"}
            if not policy_allows(evaluation):
                if policy_requires_review(evaluation) and not self.runtime_config.deny_on_policy_review:
                    self._audit(RouteEventType.POLICY_REVIEW_REQUIRED.value, "Policy requested review; router continuing because deny_on_policy_review is false.", task=envelope, metadata={"policy": evaluation_dict})
                    return evaluation_dict
                event_type = RouteEventType.POLICY_REVIEW_REQUIRED.value if policy_requires_review(evaluation) else RouteEventType.POLICY_DENIED.value
                error = self._routing_error(
                    f"Policy blocked task '{envelope.task_type}'.",
                    task_type=envelope.task_type,
                    context={"task_id": envelope.task_id, "policy_evaluation": evaluation_dict},
                )
                self._audit(event_type, "Policy blocked task before routing.", task=envelope, error=error, metadata={"policy": evaluation_dict})
                raise error
            return evaluation_dict
        except BaseException as exc:
            if self.runtime_config.fail_closed_on_policy_error or isinstance(exc, CollaborationError if isinstance(CollaborationError, type) else Exception):
                raise
            logger.warning("Policy evaluation failed open for task '%s': %s", envelope.task_type, exc)
            return {"decision": "allow", "warning": str(exc), "fail_open": True}

    # ------------------------------------------------------------------
    # Records, events, and errors
    # ------------------------------------------------------------------
    def _build_skipped_attempt(self, *, route_id: str, task_type: str, agent_name: str, rank: int, score: float, error: BaseException, reason: str) -> Dict[str, Any]:
        now = epoch_seconds()
        return RouteAttemptRecord(
            attempt_id=generate_uuid("route_attempt", length=24),
            route_id=route_id,
            task_type=task_type,
            agent_name=agent_name,
            rank=rank,
            score=float(score),
            status="skipped",
            started_at=now,
            finished_at=now,
            duration_ms=0.0,
            error=exception_to_error_payload(error, action="task_route_attempt").get("error"),
            metadata={"reason": reason},
        ).to_dict()

    def _build_route_record(
        self,
        *,
        route_id: str,
        envelope: Any,
        status: str,
        started_at: float,
        duration_ms: float,
        selected_agent: Optional[str] = None,
        candidate_count: int = 0,
        ranked_count: int = 0,
        attempts: Optional[Sequence[Mapping[str, Any]]] = None,
        ranking_report: Optional[Mapping[str, Any]] = None,
        contract_validation: Optional[Mapping[str, Any]] = None,
        policy_evaluation: Optional[Mapping[str, Any]] = None,
        error: Optional[BaseException] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        finished_at = epoch_seconds()
        return RouteRecord(
            route_id=route_id,
            task_id=envelope.task_id,
            task_type=envelope.task_type,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            selected_agent=selected_agent,
            candidate_count=candidate_count,
            ranked_count=ranked_count,
            attempts=tuple(dict(item) for item in (attempts or ())),
            ranking_report=dict(ranking_report) if isinstance(ranking_report, Mapping) else None,
            contract_validation=dict(contract_validation) if isinstance(contract_validation, Mapping) else None,
            policy_evaluation=dict(policy_evaluation) if isinstance(policy_evaluation, Mapping) else None,
            error=exception_to_error_payload(error, action="task_route").get("error") if error is not None else None,
            metadata=normalize_metadata(metadata, drop_none=True),
            correlation_id=envelope.correlation_id,
        ).to_dict()

    def _record_route(self, route_record: Mapping[str, Any]) -> None:
        record = dict(route_record)
        with self._lock:
            self._route_history.append(record)
            self._last_route = record
        self._publish_status()

    def _finalize_route_failure(
        self,
        route_id: str,
        envelope: Any,
        started_at: float,
        start_ms: float,
        *,
        error: BaseException,
        attempts: Sequence[Mapping[str, Any]],
        ranking_report: Optional[Mapping[str, Any]] = None,
        contract_validation: Optional[Mapping[str, Any]] = None,
        policy_evaluation: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            self._routes_failed += 1
        route_record = self._build_route_record(
            route_id=route_id,
            envelope=envelope,
            status="failed",
            started_at=started_at,
            duration_ms=elapsed_ms(start_ms),
            attempts=attempts,
            ranking_report=ranking_report,
            contract_validation=contract_validation,
            policy_evaluation=policy_evaluation,
            error=error,
            metadata=metadata,
        )
        self._record_route(route_record)
        self._audit(RouteEventType.ROUTE_FAILED.value, f"Task '{getattr(envelope, 'task_type', 'unknown')}' route failed.", task=envelope, error=error, metadata={"route_id": route_id})
        self._publish_task_event(RouteEventType.ROUTE_FAILED.value, envelope, error=error, metadata={"route_id": route_id})
        self._emit_hooks(route_record)
        return route_record

    def _audit(self, event_type: str, message: str, *, severity: str = "info", task: Optional[Any] = None, agent_name: Optional[str] = None, error: Optional[BaseException] = None, metadata: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not self.runtime_config.audit_enabled:
            return None
        event = build_audit_event(
            event_type,
            message,
            severity=severity,
            component="task_router",
            task=task,
            agent_name=agent_name,
            error=error,
            metadata=metadata,
        )
        if self.shared_memory is not None:
            append_audit_event(self.shared_memory, event, key=self.runtime_config.audit_key, max_events=self.runtime_config.audit_max_events)
        return event

    def _publish_task_event(self, event_type: str, envelope: Any, *, agent_name: Optional[str] = None, error: Optional[BaseException] = None, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if not self.runtime_config.task_events_enabled or self.shared_memory is None:
            return
        event = build_audit_event(
            event_type,
            f"Task router event: {event_type}",
            component="task_router",
            task=envelope,
            agent_name=agent_name,
            error=error,
            metadata=metadata,
        )
        append_audit_event(self.shared_memory, event, key=self.runtime_config.task_events_key, max_events=self.runtime_config.task_events_max_events)

    def _publish_status(self) -> None:
        if not self.runtime_config.publish_status or self.shared_memory is None:
            return
        status = {
            "component": "task_router",
            "updated_at": epoch_seconds(),
            "updated_at_utc": utc_timestamp(),
            "routes_started": self._routes_started,
            "routes_succeeded": self._routes_succeeded,
            "routes_failed": self._routes_failed,
            "attempts_started": self._attempts_started,
            "attempts_succeeded": self._attempts_succeeded,
            "attempts_failed": self._attempts_failed,
            "strategy": getattr(self.strategy, "name", type(self.strategy).__name__),
        }
        memory_set(self.shared_memory, self.runtime_config.status_key, status)

    def _emit_hooks(self, route_record: Mapping[str, Any]) -> None:
        for hook in list(self._route_hooks):
            try:
                hook(dict(route_record))
            except Exception as exc:
                logger.warning("TaskRouter route hook failed: %s", exc)

    def _routing_error(
        self,
        message: str,
        *,
        task_type: str = "unknown",
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> Exception:
        normalized_context = normalize_metadata(context, drop_none=True)
        routing_cls = globals().get("RoutingFailureError")
        if isinstance(routing_cls, type) and issubclass(routing_cls, Exception):
            try:
                error = routing_cls(task_type, message)
                with_context_method = getattr(error, "with_context", None)
                if callable(with_context_method):
                    with_context_method(**normalized_context)
                if cause is not None and getattr(error, "__cause__", None) is None:
                    error.__cause__ = cause
                return error
            except TypeError:
                pass
        return make_collaboration_exception(
            "RoutingFailureError",
            message,
            task_type=task_type,
            reason=message,
            context=normalized_context,
            cause=cause,
        )

    def _no_capable_agent_error(self, task_type: str, envelope: Any, *, available_agents: Mapping[str, Any]) -> Exception:
        required = list(getattr(envelope, "required_capabilities", ()) or ())
        available = {name: list(extract_agent_capabilities(meta)) for name, meta in available_agents.items()}
        context = {"task_id": getattr(envelope, "task_id", None), "task_type": task_type, "available_agents": available}
        no_capable_cls = globals().get("NoCapableAgentError")
        if isinstance(no_capable_cls, type) and issubclass(no_capable_cls, Exception):
            try:
                error = no_capable_cls(task_type, required)
                with_context_method = getattr(error, "with_context", None)
                if callable(with_context_method):
                    with_context_method(**context)
                return error
            except TypeError:
                pass
        return make_collaboration_exception(
            "NoCapableAgentError",
            f"No capable agent found for task type '{task_type}'.",
            task_type=task_type,
            required_capabilities=required,
            available_agents=available,
            context=context,
        )


if __name__ == "__main__":
    print("\n=== Running Task Router with Real Modules ===\n")
    printer.status("TEST", "Task Router initialized", "info")
    from .shared_memory import SharedMemory
    from .registry import AgentRegistry
    from ..base_agent import BaseAgent
    from .task_contracts import TaskContractRegistry
    from .policy_engine import PolicyEngine, PolicyDecision

    # ------------------------------------------------------------------
    # Real SharedMemory
    # ------------------------------------------------------------------
    memory = SharedMemory()
    memory.clear_all()

    # ------------------------------------------------------------------
    # Real AgentRegistry (auto_discover=False)
    # ------------------------------------------------------------------
    # Define concrete agents that inherit from BaseAgent and provide execute
    class TranslateAgent(BaseAgent):
        capabilities = ["translate"]

        def __init__(self, shared_memory=None, agent_factory=None, config=None):
            super().__init__(shared_memory, agent_factory, config)
            self.name = "TranslateAgent"

        def execute(self, task_data): # type: ignore
            return {"agent": self.name, "task": task_data, "status": "ok"}

    class FailingAgent(BaseAgent):
        capabilities = ["translate"]

        def __init__(self, shared_memory=None, agent_factory=None, config=None):
            super().__init__(shared_memory, agent_factory, config)
            self.name = "FailingAgent"

        def execute(self, task_data): # type: ignore
            raise RuntimeError("FailingAgent always fails")

    class SummarizeAgent(BaseAgent):
        capabilities = ["summarize"]

        def __init__(self, shared_memory=None, agent_factory=None, config=None):
            super().__init__(shared_memory, agent_factory, config)
            self.name = "SummarizeAgent"

        def execute(self, task_data): # type: ignore
            return {"agent": self.name, "task": task_data, "status": "summarized"}

    # Create registry (without auto-discovery)
    registry = AgentRegistry(shared_memory=memory, auto_discover=False)

    # Register instances (agent_factory=None is fine)
    registry.register("TranslateAgent", agent_instance=TranslateAgent(shared_memory=memory, agent_factory=None))
    registry.register("FailingAgent", agent_instance=FailingAgent(shared_memory=memory, agent_factory=None))
    registry.register("SummarizeAgent", agent_instance=SummarizeAgent(shared_memory=memory, agent_factory=None))

    # ------------------------------------------------------------------
    # Real TaskContractRegistry
    # ------------------------------------------------------------------
    contract_registry = TaskContractRegistry(shared_memory=memory, load_configured=False)

    # Register a simple contract for "translate" tasks
    contract_registry.register_contract(
        "translate",
        required_fields=["text"],
        field_types={"text": (str,)},
        allow_unknown_fields=True,
    )

    # ------------------------------------------------------------------
    # Real PolicyEngine
    # ------------------------------------------------------------------
    policy_engine = PolicyEngine(shared_memory=memory, load_config_rules=False)

    # Add a simple deny rule for testing
    policy_engine.add_simple_rule(
        rule_id="deny_blocked",
        description="Deny tasks that have 'blocked' = True",
        effect=PolicyDecision.DENY,
        priority=10,
        predicate=lambda task, agent, ctx: task.get("blocked") is True,
    )

    # ------------------------------------------------------------------
    # Real ReliabilityManager
    # ------------------------------------------------------------------
    reliability_manager = ReliabilityManager(shared_memory=memory)

    # ------------------------------------------------------------------
    # TaskRouter with all real components
    # ------------------------------------------------------------------
    router = TaskRouter(
        registry=registry,
        shared_memory=memory,
        contract_registry=contract_registry,
        policy_engine=policy_engine,
        reliability_manager=reliability_manager,
        config={
            "contract_validation_enabled": True,
            "policy_evaluation_enabled": True,
            "audit_enabled": True,
            "task_events_enabled": True,
            "continue_on_agent_failure": True,
            "update_agent_stats": True,
            "publish_heartbeats": True,
        },
    )

    # ------------------------------------------------------------------
    # Test route with real agents
    # ------------------------------------------------------------------
    result = router.route("translate", {"text": "hello", "source": "smoke"})
    assert result["agent"] == "TranslateAgent", f"Expected TranslateAgent, got {result}"
    assert result["task"]["text"] == "hello", result

    # Check agent stats in shared memory
    stats = memory.get("agent_stats")
    assert stats["TranslateAgent"]["successes"] >= 1, stats # type: ignore
    assert stats["FailingAgent"]["failures"] == 0, stats   # type: ignore # FailingAgent not used yet

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    explanation = router.explain_route("translate", {"text": "hello"})
    assert explanation["candidate_count"] == 2, explanation  # TranslateAgent and FailingAgent
    assert explanation["ranked_count"] == 2, explanation

    # ------------------------------------------------------------------
    # Route history
    # ------------------------------------------------------------------
    history = router.route_history()
    assert history and history[-1]["status"] == "success", history
    assert router.last_route()["selected_agent"] == "TranslateAgent", router.last_route() # type: ignore

    # ------------------------------------------------------------------
    # Health and snapshot
    # ------------------------------------------------------------------
    health = router.health_report()
    assert health["status"] in {"healthy", "degraded"}, health
    snapshot = router.snapshot()
    assert snapshot["routes_succeeded"] >= 1, snapshot

    # ------------------------------------------------------------------
    # Policy denial test
    # ------------------------------------------------------------------
    try:
        router.route("translate", {"blocked": True, "text": "x"})
        raise AssertionError("Expected policy failure")
    except Exception as exc:
        assert "Policy blocked" in str(exc) or "deny" in str(exc).lower(), exc

    # ------------------------------------------------------------------
    # Contract validation failure
    # ------------------------------------------------------------------
    try:
        router.route("translate", {"source": "missing-text"})   # missing 'text' field
        raise AssertionError("Expected contract failure")
    except Exception as exc:
        assert "contract" in str(exc).lower() or "missing" in str(exc).lower(), exc

    # ------------------------------------------------------------------
    # No capable agent
    # ------------------------------------------------------------------
    try:
        router.route("missing", {"text": "x"})
        raise AssertionError("Expected missing-task failure")
    except Exception as exc:
        assert "No capable agent" in str(exc) or "No agents" in str(exc), exc

    # ------------------------------------------------------------------
    # Verify shared-memory status and audit events
    # ------------------------------------------------------------------
    assert memory.get("collaboration:task_router_status"), "Task router status not published"
    audit_events = memory.get("collaboration:task_router_events", [])
    assert audit_events, "No audit events recorded"

    print("\n=== Test ran successfully ===\n")