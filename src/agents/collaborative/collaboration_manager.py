from __future__ import annotations

"""
Production-grade collaboration manager for SLAI's collaborative runtime.

This module owns the top-level orchestration facade for the collaborative
multi-agent subsystem. It composes the registry, task contracts, policy engine,
routing strategy, reliability manager, task router, shared-memory statistics,
and operational telemetry while preserving the original CollaborationManager
public API.

Responsibilities
----------------
- Provide a stable facade for registering agents and running collaborative tasks.
- Compose AgentRegistry, TaskContractRegistry, PolicyEngine, RouterStrategy,
  ReliabilityManager, and TaskRouter without duplicating their internals.
- Enforce manager-level load controls before routing begins.
- Maintain bounded manager execution history and lifecycle/audit events.
- Publish manager status, health snapshots, and metrics into SharedMemory.
- Export stats and snapshots for reports and offline diagnostics.
- Use collaboration errors and collaborative helpers consistently at subsystem
  boundaries.

Design principles
-----------------
1. Stable public API: ``CollaborationManager``, ``max_load``,
   ``register_agent()``, ``run_task()``, ``get_system_load()``,
   ``list_agents()``, ``get_agent_stats()``, ``get_reliability_status()``, and
   ``export_stats_to_json()`` remain available.
2. Direct local imports: project-local config, error, helper, registry,
   reliability, router, policy, and contract imports remain explicit and
   unwrapped.
3. Separation of concerns: this manager composes subsystems; it does not own
   discovery internals, policy rule semantics, contract validation internals,
   strategy scoring, circuit-breaker transitions, or low-level shared-memory
   storage.
4. Config-backed behavior: manager tuning belongs in ``collaborative_config.yaml``
   under ``collaboration`` and optional ``collaboration_manager`` overrides.
5. Operational transparency: all diagnostics are JSON-safe, redacted by default,
   and suitable for logs, shared memory, tests, and incident reports.
"""

import json
import os
import threading
import time

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from .policy_engine import PolicyEngine
from .registry import AgentRegistry
from .reliability import CircuitBreakerConfig, ReliabilityManager, RetryPolicy
from .router_strategy import BaseRouterStrategy, build_router_strategy
from .task_contracts import TaskContractRegistry
from .task_router import TaskRouter
from .utils.config_loader import get_config_section, load_global_config
from .utils.collaboration_error import *
from .utils.collaborative_helpers import *
from .utils.collaborative_helpers import (
    DEFAULT_AGENT_STATS_KEY,
    DEFAULT_AGENT_TASK_MULTIPLIER,
    DEFAULT_MAX_AUDIT_EVENTS,
)
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Collaboration Manager")
printer = PrettyPrinter()


class CollaborationManagerEventType(str, Enum):
    """Normalized collaboration-manager lifecycle/audit event labels."""

    MANAGER_INITIALIZED = "manager_initialized"
    AGENT_REGISTERED = "agent_registered"
    AGENTS_REGISTERED = "agents_registered"
    TASK_STARTED = "task_started"
    TASK_SUCCEEDED = "task_succeeded"
    TASK_FAILED = "task_failed"
    TASK_RETRYING = "task_retrying"
    TASK_REJECTED_OVERLOAD = "task_rejected_overload"
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"
    SNAPSHOT_EXPORTED = "snapshot_exported"
    STATS_EXPORTED = "stats_exported"
    MANAGER_SHUTDOWN = "manager_shutdown"


@dataclass(frozen=True)
class CollaborationManagerConfig:
    """Runtime configuration for ``CollaborationManager``.

    ``collaboration`` remains the primary compatibility section. Optional
    ``collaboration_manager`` values can override top-level defaults without
    disturbing older configs.
    """

    max_concurrent_tasks: int = 100
    load_factor: float = 0.75
    thread_pool_workers: int = 10
    health_check_interval: float = 60.0
    stats_key: str = DEFAULT_AGENT_STATS_KEY
    audit_enabled: bool = True
    audit_key: str = "collaboration:manager_events"
    audit_max_events: int = DEFAULT_MAX_AUDIT_EVENTS
    task_history_limit: int = 1000
    status_key: str = "collaboration:manager_status"
    publish_status: bool = True
    auto_discover_agents: bool = True
    initialize_policy_engine: bool = True
    initialize_contract_registry: bool = True
    enforce_load_limit: bool = True
    max_load_agent_task_multiplier: int = DEFAULT_AGENT_TASK_MULTIPLIER
    default_task_retries: int = 1
    run_task_uses_envelope: bool = True
    include_task_type_in_payload: bool = False
    export_pretty_json: bool = True
    shutdown_wait: bool = True
    redact_audit_payloads: bool = True

    @classmethod
    def from_config(
        cls,
        collaboration_config: Optional[Mapping[str, Any]] = None,
        manager_config: Optional[Mapping[str, Any]] = None,
        task_routing_config: Optional[Mapping[str, Any]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "CollaborationManagerConfig":
        collaboration = dict(collaboration_config or {})
        manager = dict(manager_config or {})
        routing = dict(task_routing_config or {})
        source = merge_mappings(collaboration, manager, overrides or {}, deep=True, drop_none=True)
        retry_policy: Mapping[str, Any] = {}
        retry_policy_candidate = routing.get("retry_policy")
        if isinstance(retry_policy_candidate, Mapping):
            retry_policy = retry_policy_candidate
        return cls(
            max_concurrent_tasks=coerce_int(source.get("max_concurrent_tasks"), default=cls.max_concurrent_tasks, minimum=1),
            load_factor=coerce_float(source.get("load_factor"), default=cls.load_factor, minimum=0.0),
            thread_pool_workers=coerce_int(source.get("thread_pool_workers"), default=cls.thread_pool_workers, minimum=1),
            health_check_interval=coerce_float(source.get("health_check_interval"), default=cls.health_check_interval, minimum=0.0),
            stats_key=str(source.get("stats_key", cls.stats_key)).strip() or cls.stats_key,
            audit_enabled=coerce_bool(source.get("audit_enabled"), default=cls.audit_enabled),
            audit_key=str(source.get("audit_key", cls.audit_key)).strip() or cls.audit_key,
            audit_max_events=coerce_int(source.get("audit_max_events"), default=cls.audit_max_events, minimum=1),
            task_history_limit=coerce_int(source.get("task_history_limit"), default=cls.task_history_limit, minimum=1),
            status_key=str(source.get("status_key", cls.status_key)).strip() or cls.status_key,
            publish_status=coerce_bool(source.get("publish_status"), default=cls.publish_status),
            auto_discover_agents=coerce_bool(source.get("auto_discover_agents"), default=cls.auto_discover_agents),
            initialize_policy_engine=coerce_bool(source.get("initialize_policy_engine"), default=cls.initialize_policy_engine),
            initialize_contract_registry=coerce_bool(source.get("initialize_contract_registry"), default=cls.initialize_contract_registry),
            enforce_load_limit=coerce_bool(source.get("enforce_load_limit"), default=cls.enforce_load_limit),
            max_load_agent_task_multiplier=coerce_int(source.get("max_load_agent_task_multiplier"), default=cls.max_load_agent_task_multiplier, minimum=1),
            default_task_retries=coerce_int(source.get("default_task_retries", retry_policy.get("max_attempts", cls.default_task_retries)), default=cls.default_task_retries, minimum=1),
            run_task_uses_envelope=coerce_bool(source.get("run_task_uses_envelope"), default=cls.run_task_uses_envelope),
            include_task_type_in_payload=coerce_bool(source.get("include_task_type_in_payload"), default=cls.include_task_type_in_payload),
            export_pretty_json=coerce_bool(source.get("export_pretty_json"), default=cls.export_pretty_json),
            shutdown_wait=coerce_bool(source.get("shutdown_wait"), default=cls.shutdown_wait),
            redact_audit_payloads=coerce_bool(source.get("redact_audit_payloads"), default=cls.redact_audit_payloads),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManagerTaskRecord:
    """Bounded history record for one manager-level task execution."""

    execution_id: str
    task_id: str
    task_type: str
    status: str
    started_at: float
    finished_at: float
    duration_ms: float
    attempts: int
    selected_agent: Optional[str] = None
    result_fingerprint: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    route: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: generate_correlation_id("manager-task"))

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self), drop_empty=True)
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class BatchTaskRecord:
    """Summary for one batch execution through ``run_tasks``."""

    batch_id: str
    status: str
    started_at: float
    finished_at: float
    duration_ms: float
    total: int
    successes: int
    failures: int
    results: Tuple[Dict[str, Any], ...] = ()
    errors: Tuple[Dict[str, Any], ...] = ()
    correlation_id: str = field(default_factory=lambda: generate_correlation_id("manager-batch"))

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self), drop_empty=True)
        return redact_mapping(payload) if redact else payload


class CollaborationManager:
    """Coordinates collaborative execution through registry, contracts, policy, router, and reliability."""

    def __init__(
        self,
        shared_memory=None,
        *,
        registry: Optional[Any] = None,
        router: Optional[TaskRouter] = None,
        strategy: Optional[BaseRouterStrategy] = None,
        reliability_manager: Optional[ReliabilityManager] = None,
        policy_engine: Optional[Any] = None,
        contract_registry: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
        auto_discover: Optional[bool] = None,
    ):
        self.shared_memory = shared_memory
        self.config = load_global_config()
        self.manager_config = get_config_section("collaboration") or {}
        self.collaboration_manager_config = get_config_section("collaboration_manager") or {}
        self.routing_config = get_config_section("task_routing") or {}
        self.reliability_config = get_config_section("reliability") or {}
        self.runtime_config = CollaborationManagerConfig.from_config(
            self.manager_config,
            self.collaboration_manager_config,
            self.routing_config,
            overrides=config,
        )
        if auto_discover is not None:
            self.runtime_config = CollaborationManagerConfig.from_config(
                self.manager_config,
                self.collaboration_manager_config,
                self.routing_config,
                overrides=merge_mappings(config or {}, {"auto_discover_agents": bool(auto_discover)}),
            )

        self._lock = threading.RLock()
        self._started_at = epoch_seconds()
        self._shutdown = False
        self._task_history: Deque[Dict[str, Any]] = deque(maxlen=self.runtime_config.task_history_limit)
        self._last_task: Optional[Dict[str, Any]] = None
        self._batch_history: Deque[Dict[str, Any]] = deque(maxlen=max(10, self.runtime_config.task_history_limit // 10))
        self._executor = ThreadPoolExecutor(max_workers=self.runtime_config.thread_pool_workers)
        # Compatibility alias retained from the original implementation.
        self.executor = self._executor

        self.max_concurrent_tasks = self.runtime_config.max_concurrent_tasks
        self.load_factor = self.runtime_config.load_factor

        self.registry = registry or AgentRegistry(
            shared_memory=self.shared_memory,
            auto_discover=self.runtime_config.auto_discover_agents,
        )
        self.contract_registry = contract_registry if contract_registry is not None else self._build_contract_registry()
        self.policy_engine = policy_engine if policy_engine is not None else self._build_policy_engine()
        self.reliability_manager = reliability_manager or self._build_reliability_manager()
        strategy_name = str(self.routing_config.get("strategy", self.runtime_config.to_dict().get("strategy", "weighted")) or "weighted")
        self.strategy = strategy or build_router_strategy(strategy_name, config=self.routing_config)
        self.router = router or self._build_task_router()

        self._record_manager_event(
            CollaborationManagerEventType.MANAGER_INITIALIZED.value,
            "Collaboration manager initialized.",
            severity="info",
            metadata={
                "config": self.runtime_config.to_dict(),
                "agent_count": self._safe_agent_count(),
                "has_policy_engine": self.policy_engine is not None,
                "has_contract_registry": self.contract_registry is not None,
            },
        )
        self._publish_status()
        logger.info("Collaboration Manager initialized")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_contract_registry(self) -> Optional[Any]:
        if not self.runtime_config.initialize_contract_registry:
            return None
        return TaskContractRegistry()

    def _build_policy_engine(self) -> Optional[Any]:
        if not self.runtime_config.initialize_policy_engine:
            return None
        return PolicyEngine()

    def _build_reliability_manager(self) -> ReliabilityManager:
        if hasattr(RetryPolicy, "from_config"):
            retry_policy = RetryPolicy.from_config(self.reliability_config, self.routing_config)  # type: ignore[attr-defined]
        else:
            retry_policy_config = self.routing_config.get("retry_policy")
            retry_source = retry_policy_config if isinstance(retry_policy_config, Mapping) else {}
            retry_policy = RetryPolicy(
                max_attempts=coerce_int(retry_source.get("max_attempts"), default=1, minimum=1),
                backoff_factor=coerce_float(retry_source.get("backoff_factor"), default=0.0, minimum=0.0),
                max_backoff_seconds=coerce_float(self.reliability_config.get("max_backoff_seconds"), default=2.0, minimum=0.0),
                jitter_seconds=coerce_float(self.reliability_config.get("jitter_seconds"), default=0.0, minimum=0.0),
            )

        if hasattr(CircuitBreakerConfig, "from_config"):
            breaker_config = CircuitBreakerConfig.from_config(self.reliability_config)  # type: ignore[attr-defined]
        else:
            breaker_config = CircuitBreakerConfig(
                failure_threshold=coerce_int(self.reliability_config.get("failure_threshold"), default=3, minimum=1),
                recovery_timeout_seconds=coerce_float(self.reliability_config.get("recovery_timeout_seconds"), default=5.0, minimum=0.0),
                half_open_success_threshold=coerce_int(self.reliability_config.get("half_open_success_threshold"), default=1, minimum=1),
            )

        try:
            return ReliabilityManager(retry_policy=retry_policy, breaker_config=breaker_config, shared_memory=self.shared_memory)
        except TypeError:
            return ReliabilityManager(retry_policy=retry_policy, breaker_config=breaker_config)

    def _build_task_router(self) -> TaskRouter:
        try:
            return TaskRouter(
                registry=self.registry,
                shared_memory=self.shared_memory,
                strategy=self.strategy,
                reliability_manager=self.reliability_manager,
                contract_registry=self.contract_registry,
                policy_engine=self.policy_engine,
            )
        except TypeError:
            return TaskRouter(
                registry=self.registry,
                shared_memory=self.shared_memory,
                strategy=self.strategy,
                reliability_manager=self.reliability_manager,
            )

    # ------------------------------------------------------------------
    # Core public API retained from original module
    # ------------------------------------------------------------------
    @property
    def max_load(self) -> int:
        """Return manager-level maximum active task load."""

        agent_count = max(1, self._safe_agent_count())
        calculated = int(agent_count * self.runtime_config.max_load_agent_task_multiplier * self.load_factor)
        return min(self.max_concurrent_tasks, max(1, calculated))

    def register_agent(self, agent_name: str, agent_instance: Any, capabilities, *, version: Any = 1.0, metadata: Optional[Mapping[str, Any]] = None):
        """Register a single agent with the underlying registry.

        The original method accepted ``agent_name``, ``agent_instance`` and
        ``capabilities``. Optional version/metadata are additive and backward
        compatible.
        """

        name = normalize_agent_name(agent_name)
        caps = normalize_capabilities(capabilities)
        registration = {
            "name": name,
            "meta": {
                "class": type(agent_instance),
                "instance": agent_instance,
                "capabilities": list(caps),
                "version": version,
                "metadata": normalize_metadata(metadata, drop_none=True),
            },
        }
        self.registry.batch_register([registration])
        touch_agent_heartbeat(
            self.shared_memory,
            name,
            capabilities=caps,
            version=version,
            metadata={"registered_by": "collaboration_manager", **normalize_metadata(metadata, drop_none=True)},
        )
        self._record_manager_event(
            CollaborationManagerEventType.AGENT_REGISTERED.value,
            f"Registered agent '{name}'.",
            agent_name=name,
            metadata={"capabilities": list(caps), "version": version},
        )
        self._publish_status()

    def run_task(self, task_type: str, task_data: Dict[str, Any], retries: int = 1) -> Any:
        """Run a collaborative task through the configured TaskRouter.

        Manager-level retries wrap router-level fallback. Router/reliability
        retries remain responsible for per-agent attempts.
        """

        if self._shutdown:
            raise _manager_exception(
                "Collaboration manager has been shut down and cannot run tasks.",
                context={"task_type": task_type},
            )

        envelope = self._build_envelope(task_type, task_data, retries=retries)
        self._check_load_or_raise(envelope)

        started_at = epoch_seconds()
        start_ms = monotonic_ms()
        execution_id = generate_uuid("manager_exec", length=24)
        attempts_allowed = max(1, coerce_int(retries, default=self.runtime_config.default_task_retries, minimum=1))
        last_error: Optional[BaseException] = None
        route_output: Any = None

        self._record_manager_event(
            CollaborationManagerEventType.TASK_STARTED.value,
            f"Task '{envelope.task_type}' started.",
            task=envelope,
            metadata={"execution_id": execution_id, "attempts_allowed": attempts_allowed},
        )

        for attempt in range(1, attempts_allowed + 1):
            try:
                route_payload = self._route_payload(envelope, attempt=attempt, execution_id=execution_id)
                route_output = self.router.route(envelope.task_type, route_payload)
                record = self._build_task_record(
                    execution_id=execution_id,
                    envelope=envelope,
                    status="success",
                    started_at=started_at,
                    duration_ms=elapsed_ms(start_ms),
                    attempts=attempt,
                    result=route_output,
                )
                self._store_task_record(record)
                self._record_manager_event(
                    CollaborationManagerEventType.TASK_SUCCEEDED.value,
                    f"Task '{envelope.task_type}' completed successfully.",
                    task=envelope,
                    metadata={"execution_id": execution_id, "attempt": attempt, "result_fingerprint": record.get("result_fingerprint")},
                )
                self._publish_status()
                return route_output
            except Exception as exc:  # noqa: BLE001 - manager boundary intentionally normalizes downstream failures.
                last_error = exc
                if attempt < attempts_allowed:
                    self._record_manager_event(
                        CollaborationManagerEventType.TASK_RETRYING.value,
                        f"Task '{envelope.task_type}' attempt {attempt} failed; retrying.",
                        task=envelope,
                        error=exc,
                        metadata={"execution_id": execution_id, "attempt": attempt},
                        severity="warning",
                    )
                    continue
                break

        assert last_error is not None
        record = self._build_task_record(
            execution_id=execution_id,
            envelope=envelope,
            status="failed",
            started_at=started_at,
            duration_ms=elapsed_ms(start_ms),
            attempts=attempts_allowed,
            error=last_error,
        )
        self._store_task_record(record)
        self._record_manager_event(
            CollaborationManagerEventType.TASK_FAILED.value,
            f"Task '{envelope.task_type}' failed after {attempts_allowed} manager attempt(s).",
            task=envelope,
            error=last_error,
            metadata={"execution_id": execution_id, "attempts": attempts_allowed},
            severity="error",
        )
        self._publish_status()
        raise last_error

    def get_system_load(self) -> int:
        """Return active task count from shared-memory agent stats."""

        return get_system_load_from_stats(self.get_agent_stats())

    def list_agents(self):
        return self.registry.list_agents()

    def get_agent_stats(self):
        return get_agent_stats(self.shared_memory, stats_key=self.runtime_config.stats_key)

    def get_reliability_status(self):
        if hasattr(self.reliability_manager, "status"):
            return self.reliability_manager.status()
        return {}

    def export_stats_to_json(self, filename="report/agent_stats.json"):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(self.get_agent_stats()), handle, indent=2 if self.runtime_config.export_pretty_json else None, ensure_ascii=False)
        self._record_manager_event(
            CollaborationManagerEventType.STATS_EXPORTED.value,
            f"Agent stats exported to {path}.",
            metadata={"filename": str(path)},
        )

    # ------------------------------------------------------------------
    # Additive production APIs
    # ------------------------------------------------------------------
    def register_agents(self, agents: Iterable[Mapping[str, Any]]) -> None:
        """Register multiple agents using registry batch registration shape."""

        registrations = build_agent_batch_registrations(agents)
        self.registry.batch_register(registrations)
        for item in registrations:
            meta = item.get("meta", {})
            touch_agent_heartbeat(
                self.shared_memory,
                item.get("name"),
                capabilities=meta.get("capabilities"),
                version=meta.get("version"),
                metadata={"registered_by": "collaboration_manager"},
            )
        self._record_manager_event(
            CollaborationManagerEventType.AGENTS_REGISTERED.value,
            f"Registered {len(registrations)} agent(s).",
            metadata={"count": len(registrations), "agents": [item.get("name") for item in registrations]},
        )
        self._publish_status()

    def run_task_async(self, task_type: str, task_data: Dict[str, Any], retries: int = 1) -> Future:
        """Submit ``run_task`` to the manager executor and return a Future."""

        if self._shutdown:
            raise _manager_exception("Cannot submit async task after manager shutdown.", context={"task_type": task_type})
        return self._executor.submit(self.run_task, task_type, task_data, retries)

    def run_tasks(self, tasks: Iterable[Mapping[str, Any]], *, retries: Optional[int] = None, parallel: bool = False) -> Dict[str, Any]:
        """Run a batch of task mappings.

        Each task mapping should contain ``task_type`` and either ``payload`` or
        ``task_data``. Results are normalized so batch callers can inspect partial
        failures without losing successful outputs.
        """

        items = [ensure_mapping(item, field_name="task") for item in tasks]
        batch_id = generate_uuid("batch", length=24)
        started_at = epoch_seconds()
        start_ms = monotonic_ms()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        self._record_manager_event(
            CollaborationManagerEventType.BATCH_STARTED.value,
            f"Batch '{batch_id}' started.",
            metadata={"batch_id": batch_id, "count": len(items), "parallel": parallel},
        )

        def _run_one(index: int, item: Mapping[str, Any]) -> Dict[str, Any]:
            task_type = normalize_task_type(item.get("task_type") or item.get("type") or item.get("operation"))
            payload = ensure_mapping(item.get("payload", item.get("task_data", item.get("data", {}))), field_name="task.payload", allow_none=True)
            result = self.run_task(task_type, payload, retries=retries or self.runtime_config.default_task_retries)
            return success_result(action="run_task", message="Task completed", data={"index": index, "task_type": task_type, "result": json_safe(result)})

        if parallel:
            future_map = {self._executor.submit(_run_one, index, item): (index, item) for index, item in enumerate(items)}
            for future in as_completed(future_map):
                index, item = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    error_payload = exception_to_error_payload(exc, action="run_task")
                    error_payload.setdefault("data", {})["index"] = index
                    error_payload["data"]["task"] = json_safe(item)
                    errors.append(error_payload)
        else:
            for index, item in enumerate(items):
                try:
                    results.append(_run_one(index, item))
                except Exception as exc:
                    error_payload = exception_to_error_payload(exc, action="run_task")
                    error_payload.setdefault("data", {})["index"] = index
                    error_payload["data"]["task"] = json_safe(item)
                    errors.append(error_payload)

        status = "success" if not errors else "partial_success" if results else "failed"
        record = BatchTaskRecord(
            batch_id=batch_id,
            status=status,
            started_at=started_at,
            finished_at=epoch_seconds(),
            duration_ms=elapsed_ms(start_ms),
            total=len(items),
            successes=len(results),
            failures=len(errors),
            results=tuple(results),
            errors=tuple(errors),
        ).to_dict()
        self._batch_history.append(record)
        self._record_manager_event(
            CollaborationManagerEventType.BATCH_COMPLETED.value,
            f"Batch '{batch_id}' completed with status '{status}'.",
            metadata={"batch": record},
            severity="info" if not errors else "warning",
        )
        self._publish_status()
        return record

    def explain_task(self, task_type: str, task_data: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return a routing explanation without executing the task when supported."""

        task_type = normalize_task_type(task_type)
        payload = normalize_task_payload(task_data or {}, allow_none=True)
        eligible = self.registry.get_agents_by_task(task_type)
        stats = self.get_agent_stats()
        if hasattr(self.router, "explain_route"):
            try:
                return self.router.explain_route(task_type, payload)  # type: ignore[attr-defined]
            except TypeError:
                pass
        if hasattr(self.strategy, "explain"):
            return self.strategy.explain(eligible, stats, {"task_type": task_type, **payload})  # type: ignore[attr-defined]
        ranked = self.strategy.rank_agents(eligible, stats=stats, task_data={"task_type": task_type, **payload})
        return {
            "task_type": task_type,
            "candidate_count": len(eligible),
            "ranked_agents": [
                {"rank": index + 1, "agent_name": name, "score": score, "capabilities": json_safe(meta.get("capabilities"))}
                for index, (name, meta, score) in enumerate(ranked)
            ],
        }

    def snapshot(self, *, include_history: bool = True) -> Dict[str, Any]:
        """Return a redacted operational snapshot of the manager and subsystems."""

        agents = self.list_agents()
        stats = self.get_agent_stats()
        reliability = self.get_reliability_status()
        heartbeats = read_agent_heartbeats(self.shared_memory, agents.keys() if isinstance(agents, Mapping) else None)
        router_snapshot = self.router.snapshot() if hasattr(self.router, "snapshot") else {}
        registry_snapshot = self.registry.snapshot() if hasattr(self.registry, "snapshot") else snapshot_registry(self.registry, shared_memory=self.shared_memory)
        reliability_snapshot = self.reliability_manager.snapshot(include_history=False) if hasattr(self.reliability_manager, "snapshot") else reliability
        policy_snapshot = {}
        if self.policy_engine is not None:
            if hasattr(self.policy_engine, "summary"):
                policy_snapshot = self.policy_engine.summary()
            elif hasattr(self.policy_engine, "list_rules"):
                policy_snapshot = self.policy_engine.list_rules()
        
        contract_snapshot = {}
        if self.contract_registry is not None:
            if hasattr(self.contract_registry, "summary"):
                contract_snapshot = self.contract_registry.summary()
            elif hasattr(self.contract_registry, "list_contracts"):
                contract_snapshot = self.contract_registry.list_contracts()
        
        payload = {
            "component": "collaboration_manager",
            "status": "shutdown" if self._shutdown else "running",
            "started_at": self._started_at,
            "uptime_seconds": round(epoch_seconds() - self._started_at, 6),
            "captured_at": epoch_seconds(),
            "captured_at_utc": utc_timestamp(),
            "config": self.runtime_config.to_dict(),
            "load": self.build_load_snapshot().to_dict(),
            "agents": json_safe(agents),
            "agent_stats": stats,
            "heartbeats": heartbeats,
            "registry": registry_snapshot,
            "router": router_snapshot,
            "reliability": reliability_snapshot,
            "policy": policy_snapshot,
            "task_contracts": contract_snapshot,
            "shared_memory": memory_snapshot(self.shared_memory, include_metrics=True, include_keys=False),
            "last_task": self._last_task,
        }
        if include_history:
            payload["task_history"] = list(self._task_history)
            payload["batch_history"] = list(self._batch_history)
        return redact_mapping(prune_none(payload, drop_empty=True))

    def health_report(self) -> Dict[str, Any]:
        """Return a compact health report for external probes."""

        agents = self.list_agents()
        stats = self.get_agent_stats()
        reliability = self.get_reliability_status()
        heartbeats = read_agent_heartbeats(self.shared_memory, agents.keys() if isinstance(agents, Mapping) else None)
        report = build_health_report(
            agents=agents if isinstance(agents, Mapping) else {},
            stats=stats,
            heartbeats=heartbeats,
            reliability_status=reliability if isinstance(reliability, Mapping) else {},
            shared_memory=self.shared_memory,
        )
        report["manager"] = {
            "status": "shutdown" if self._shutdown else "running",
            "system_load": self.get_system_load(),
            "max_load": self.max_load,
            "history_size": len(self._task_history),
            "uptime_seconds": round(epoch_seconds() - self._started_at, 6),
        }
        return redact_mapping(report)

    def build_load_snapshot(self) -> LoadSnapshot:
        agents = self.list_agents()
        stats = self.get_agent_stats()
        heartbeats = read_agent_heartbeats(self.shared_memory, agents.keys() if isinstance(agents, Mapping) else None)
        return build_load_snapshot(
            agents=agents if isinstance(agents, Mapping) else {},
            stats=stats,
            heartbeats=heartbeats,
            max_concurrent_tasks=self.max_concurrent_tasks,
            load_factor=self.load_factor,
        )

    def get_task_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = list(self._task_history)
        if limit is not None:
            return items[-max(0, int(limit)):]
        return items

    def get_last_task(self) -> Optional[Dict[str, Any]]:
        return self._last_task

    def get_batch_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = list(self._batch_history)
        if limit is not None:
            return items[-max(0, int(limit)):]
        return items

    def export_snapshot_to_json(self, filename: str = "report/collaboration_manager_snapshot.json", *, include_history: bool = True) -> Path:
        path = export_json_file(filename, self.snapshot(include_history=include_history), pretty=self.runtime_config.export_pretty_json)
        self._record_manager_event(
            CollaborationManagerEventType.SNAPSHOT_EXPORTED.value,
            f"Collaboration manager snapshot exported to {path}.",
            metadata={"filename": str(path)},
        )
        return path

    def reset_reliability(self, key: Optional[str] = None) -> None:
        if hasattr(self.reliability_manager, "reset"):
            self.reliability_manager.reset(key)  # type: ignore[attr-defined]
        self._publish_status()

    def shutdown(self, *, wait: Optional[bool] = None) -> None:
        """Shut down the manager executor and publish final status."""

        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        should_wait = self.runtime_config.shutdown_wait if wait is None else bool(wait)
        self._executor.shutdown(wait=should_wait)
        self._record_manager_event(
            CollaborationManagerEventType.MANAGER_SHUTDOWN.value,
            "Collaboration manager shut down.",
            severity="info",
        )
        self._publish_status()

    def __enter__(self) -> "CollaborationManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_envelope(self, task_type: str, task_data: Mapping[str, Any], *, retries: int) -> TaskEnvelope:
        payload = normalize_task_payload(task_data, allow_none=True) if self.runtime_config.run_task_uses_envelope else dict(task_data or {})
        return build_task_envelope(
            task_type=task_type,
            payload=payload,
            retry_limit=retries,
            source="collaboration_manager",
            metadata={"manager": "CollaborationManager"},
        )

    def _route_payload(self, envelope: TaskEnvelope, *, attempt: int, execution_id: str) -> Dict[str, Any]:
        payload = dict(envelope.payload)
        if self.runtime_config.include_task_type_in_payload:
            payload.setdefault("task_type", envelope.task_type)
            payload.setdefault("task_id", envelope.task_id)
            payload.setdefault("correlation_id", envelope.correlation_id)
        payload.setdefault("_manager_attempt", attempt)
        payload.setdefault("_manager_execution_id", execution_id)
        return payload

    def _check_load_or_raise(self, envelope: TaskEnvelope) -> None:
        if not self.runtime_config.enforce_load_limit:
            return
        current_load = self.get_system_load()
        maximum = self.max_load
        if current_load < maximum:
            return
        error = _overload_error(
            f"System load exceeded ({current_load}/{maximum})",
            current_load=current_load,
            max_load=maximum,
            context={"task_type": envelope.task_type, "task_id": envelope.task_id},
        )
        self._record_manager_event(
            CollaborationManagerEventType.TASK_REJECTED_OVERLOAD.value,
            f"Task '{envelope.task_type}' rejected due to overload.",
            task=envelope,
            error=error,
            severity="warning",
            metadata={"current_load": current_load, "max_load": maximum},
        )
        self._publish_status()
        raise error

    def _build_task_record(
        self,
        *,
        execution_id: str,
        envelope: TaskEnvelope,
        status: str,
        started_at: float,
        duration_ms: float,
        attempts: int,
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> Dict[str, Any]:
        finished_at = epoch_seconds()
        route_snapshot = self.router.last_route() if hasattr(self.router, "last_route") and callable(getattr(self.router, "last_route")) else getattr(self.router, "_last_route", None)
        selected_agent = None
        if isinstance(route_snapshot, Mapping):
            selected_agent = route_snapshot.get("selected_agent")
        record = ManagerTaskRecord(
            execution_id=execution_id,
            task_id=envelope.task_id,
            task_type=envelope.task_type,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            attempts=attempts,
            selected_agent=str(selected_agent) if selected_agent else None,
            result_fingerprint=stable_hash(json_safe(result), length=16) if result is not None else None,
            error=exception_to_error_payload(error, action="run_task").get("error") if error is not None else None,
            route=dict(route_snapshot) if isinstance(route_snapshot, Mapping) else None,
            metadata={"envelope": envelope.to_dict()},
            correlation_id=envelope.correlation_id,
        ).to_dict(redact=self.runtime_config.redact_audit_payloads)
        return record

    def _store_task_record(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            payload = dict(record)
            self._task_history.append(payload)
            self._last_task = payload

    def _safe_agent_count(self) -> int:
        try:
            agents = self.list_agents()
            return len(agents) if hasattr(agents, "__len__") else 0
        except Exception:
            return 0

    def _record_manager_event(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        task: Optional[Any] = None,
        agent_name: Optional[str] = None,
        error: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = build_audit_event(
            event_type,
            message,
            severity=severity,
            component="collaboration_manager",
            task=task,
            agent_name=agent_name,
            error=error,
            metadata=metadata,
            state={"system_load": self.get_system_load() if hasattr(self, "shared_memory") else 0, "max_load": self.max_load if hasattr(self, "registry") else None},
        )
        if self.runtime_config.audit_enabled:
            append_audit_event(
                self.shared_memory,
                event,
                key=self.runtime_config.audit_key,
                max_events=self.runtime_config.audit_max_events,
            )
        return event

    def _publish_status(self) -> None:
        if not self.runtime_config.publish_status:
            return
        status = {
            "component": "collaboration_manager",
            "status": "shutdown" if self._shutdown else "running",
            "system_load": self.get_system_load() if hasattr(self, "shared_memory") else 0,
            "max_load": self.max_load if hasattr(self, "registry") else 0,
            "agent_count": self._safe_agent_count(),
            "task_history_size": len(getattr(self, "_task_history", [])),
            "batch_history_size": len(getattr(self, "_batch_history", [])),
            "updated_at": epoch_seconds(),
            "updated_at_utc": utc_timestamp(),
        }
        memory_set(self.shared_memory, self.runtime_config.status_key, status)


def _manager_exception(message: str, *, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> Exception:
    return make_collaboration_exception(
        "RoutingFailureError",
        message,
        context=normalize_metadata(context, drop_none=True),
        cause=cause,
    )


def _overload_error(message: str, *, current_load: int, max_load: int, context: Optional[Mapping[str, Any]] = None) -> Exception:
    merged_context = merge_mappings(context, {"current_load": current_load, "max_load": max_load}, deep=True, drop_none=True)
    try:
        return OverloadError(message, context=merged_context, current_load=current_load, max_load=max_load) # type: ignore
    except TypeError:
        try:
            return OverloadError(message, context=merged_context) # type: ignore
        except TypeError:
            return OverloadError(message) # type: ignore


if __name__ == "__main__":
    print("\n=== Running Collaboration Manager ===\n")
    printer.status("TEST", "Collaboration Manager initialized", "info")

    class _Memory:
        def __init__(self):
            self._store: Dict[str, Any] = {}

        def get(self, key, default=None):
            return self._store.get(key, default)

        def set(self, key, value, **kwargs):
            self._store[key] = value
            return True

        def delete(self, key):
            self._store.pop(key, None)
            return True

        def append(self, key, value, **kwargs):
            current = self._store.setdefault(key, [])
            if not isinstance(current, list):
                current = [current]
            current.append(value)
            self._store[key] = current
            return time.time()

        def get_all_keys(self):
            return list(self._store.keys())

        def metrics(self):
            return {"item_count": len(self._store)}

    class _Agent:
        capabilities = ["translate"]

        def __init__(self, name: str, fail: bool = False):
            self.name = name
            self.fail = fail

        def execute(self, data):
            if self.fail:
                raise RuntimeError(f"{self.name} failed")
            return {"ok": True, "agent": self.name, "payload": data}

    class _Registry:
        def __init__(self):
            self._agents: Dict[str, Dict[str, Any]] = {}

        def batch_register(self, agents):
            for item in agents:
                self._agents[item["name"]] = item["meta"]

        def get_agents_by_task(self, task_type):
            return {
                name: meta
                for name, meta in self._agents.items()
                if task_type in meta.get("capabilities", [])
            }

        def list_agents(self):
            return {name: list(meta.get("capabilities", [])) for name, meta in self._agents.items()}

    memory = _Memory()
    registry = _Registry()
    manager = CollaborationManager(
        shared_memory=memory,
        registry=registry,
        config={
            "auto_discover_agents": False,
            "initialize_policy_engine": False,
            "initialize_contract_registry": False,
            "thread_pool_workers": 2,
            "max_concurrent_tasks": 10,
            "load_factor": 1.0,
            "max_load_agent_task_multiplier": 5,
        },
    )

    manager.register_agent("FailFirst", _Agent("FailFirst", fail=True), ["translate"])
    manager.register_agent("Echo", _Agent("Echo"), ["translate"])

    result = manager.run_task("translate", {"text": "hola"}, retries=1)
    assert result["ok"] is True
    assert result["agent"] == "Echo"

    stats = manager.get_agent_stats()
    assert stats["FailFirst"]["failures"] >= 1
    assert stats["Echo"]["successes"] >= 1

    explanation = manager.explain_task("translate", {"text": "ciao"})
    assert explanation.get("candidate_count", 0) >= 1 or explanation.get("ranked_agents") is not None

    future = manager.run_task_async("translate", {"text": "bonjour"})
    async_result = future.result(timeout=5)
    assert async_result["ok"] is True

    batch = manager.run_tasks([
        {"task_type": "translate", "payload": {"text": "uno"}},
        {"task_type": "missing", "payload": {"text": "dos"}},
    ])
    assert batch["total"] == 2
    assert batch["successes"] == 1
    assert batch["failures"] == 1

    snapshot = manager.snapshot()
    assert snapshot["component"] == "collaboration_manager"
    health = manager.health_report()
    assert "manager" in health

    export_path = manager.export_snapshot_to_json("/tmp/collaboration_manager_snapshot_test.json")
    assert Path(export_path).exists()
    manager.export_stats_to_json("/tmp/collaboration_manager_stats_test.json")
    assert Path("/tmp/collaboration_manager_stats_test.json").exists()

    # Force overload path.
    max_load = manager.max_load
    memory.set(DEFAULT_AGENT_STATS_KEY, {"Echo": {"active_tasks": max_load}})
    try:
        manager.run_task("translate", {"text": "boom"})
        raise AssertionError("Expected overload")
    except Exception as exc:
        assert "OVERLOAD" in str(exc).upper() or "load" in str(exc).lower()

    manager.shutdown()
    assert manager.snapshot(include_history=False)["status"] == "shutdown"

    print("\n=== Test ran successfully ===\n")
