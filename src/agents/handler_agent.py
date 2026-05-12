from __future__ import annotations

__version__ = "2.2.0"

"""
Production Handler Agent

The HandlerAgent is the orchestration facade over the Handler subsystem. It is
responsible for turning a raw failure event into a normalized failure, selecting
and applying a recovery strategy, recording operational memory, and producing
learning/escalation artifacts.

Memory boundary
---------------
- HandlerMemory is the authoritative operational memory API for HandlerAgent.
- SharedMemory is the collaboration channel. HandlerMemory may mirror bounded
  events into SharedMemory when the subsystem configuration enables mirroring.
"""

import contextlib
import time

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .base_agent import BaseAgent
from .base.issue_handler import (
    IssueHandler,
    handle_common_dependency_error,
    handle_memory_error,
    handle_network_error,
    handle_resource_constraint,
    handle_runtime_error,
    handle_timeout_error,
    handle_unicode_emoji_error,
)
from .base.utils.main_config_loader import get_config_section, load_global_config
from .handler import *
from .handler.utils.handler_error import *
from .handler.utils.handler_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Handler Agent")
printer = PrettyPrinter()

RecoveryCallable = Callable[[Any, Any, Mapping[str, Any], IssueHandler], Any]


@dataclass(frozen=True)
class HandlerAgentExecution:
    """Compact top-level execution envelope returned by HandlerAgent."""

    status: str
    normalized_failure: Mapping[str, Any]
    recovery_result: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    postmortem: Mapping[str, Any]
    correlation_id: str
    duration_seconds: float
    calls: int
    schema: str = "handler.agent.execution.v2"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "schema": self.schema,
                "status": self.status,
                "normalized_failure": dict(self.normalized_failure),
                "recovery_result": dict(self.recovery_result),
                "telemetry": dict(self.telemetry),
                "postmortem": dict(self.postmortem),
                "correlation_id": self.correlation_id,
                "duration_seconds": round(self.duration_seconds, 6),
                "calls": self.calls,
                "metadata": dict(self.metadata),
            },
            drop_none=True,
            drop_empty=True,
        )


class HandlerAgent(BaseAgent):
    """
    Production cross-agent reliability layer.

    Responsibilities:
    - normalize raw failures into Handler failure payloads
    - coordinate policy, SLA, intelligence, strategy selection, retry, recovery,
      checkpointing, telemetry, postmortems, and escalation
    - use IssueHandler recovery functions instead of duplicating issue-specific
      remediation logic
    - keep SharedMemory as a collaboration fabric and HandlerMemory as the
      authoritative operational memory API
    """

    DEFAULT_RECOVERY_STRATEGIES: Tuple[str, ...] = (
        "network",
        "timeout",
        "memory",
        "runtime",
        "dependency",
        "resource",
        "unicode",
    )
    DEFAULT_SHARED_LAST_RESULT_KEY = "handler:agent:last_execution"
    DEFAULT_SHARED_STATUS_KEY = "handler:agent:status"

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.config = load_global_config()
        agent_cfg = get_config_section("handler_agent")
        self.agent_config: Dict[str, Any] = deep_merge(agent_cfg if isinstance(agent_cfg, Mapping) else {}, config or {})

        self.error_policy = HandlerErrorPolicy.from_mapping(
            self.agent_config.get("error_policy") if isinstance(self.agent_config.get("error_policy"), Mapping) else None
        )
        self._load_handler_agent_config()

        self.memory = HandlerMemory(shared_memory=self.shared_memory, error_policy=self.error_policy)
        self.issue_handler = IssueHandler()

        self.policy = HandlerPolicy(memory=self.memory, error_policy=self.error_policy)
        self.adaptive_retry_policy = AdaptiveRetryPolicy(memory=self.memory, error_policy=self.error_policy)
        self.strategy_selector = ProbabilisticStrategySelector(memory=self.memory, error_policy=self.error_policy)
        self.sla_policy = SLARecoveryPolicy(memory=self.memory, error_policy=self.error_policy)
        self.escalation_manager = EscalationManager(memory=self.memory, error_policy=self.error_policy)
        self.failure_intelligence = FailureIntelligence(memory=self.memory, error_policy=self.error_policy)

        self.recovery_strategies: Dict[str, RecoveryCallable] = self._build_recovery_strategies()
        self.calls = 0

        logger.info(
            "HandlerAgent initialized | memory=HandlerMemory shared_memory=%s strategies=%s",
            type(self.shared_memory).__name__,
            sorted(self.recovery_strategies.keys()),
        )
        self._publish_status("initialized", {"strategies": sorted(self.recovery_strategies.keys())})

    def _load_handler_agent_config(self) -> None:
        """Load only agent-level runtime settings from agents_config.yaml."""
        cfg = self.agent_config
        self.enabled = coerce_bool(cfg.get("enabled"), default=True)
        self.enable_recovery = coerce_bool(cfg.get("enable_recovery"), default=True)
        self.enable_observability = coerce_bool(cfg.get("enable_observability"), default=True)
        self.enable_learning_loop = coerce_bool(cfg.get("enable_learning_loop"), default=True)
        self.enable_escalation = coerce_bool(cfg.get("enable_escalation"), default=True)
        self.enable_lightweight_fallback = coerce_bool(cfg.get("enable_lightweight_fallback"), default=True)
        self.honor_retry_delay = coerce_bool(cfg.get("honor_retry_delay"), default=False)
        self.publish_last_execution_to_shared_memory = coerce_bool(cfg.get("publish_last_execution_to_shared_memory"), default=True)
        self.publish_status_to_shared_memory = coerce_bool(cfg.get("publish_status_to_shared_memory"), default=True)
        self.include_result_payload = coerce_bool(cfg.get("include_result_payload"), default=True)
        self.include_failed_checkpoint_state = coerce_bool(cfg.get("include_failed_checkpoint_state"), default=False)
        self.telemetry_buffer_size = coerce_int(cfg.get("telemetry_buffer_size"), 1000, minimum=10, maximum=100_000)
        self.postmortem_buffer_size = coerce_int(cfg.get("postmortem_buffer_size"), 1000, minimum=10, maximum=100_000)
        self.checkpoint_max_age_seconds = coerce_int(cfg.get("checkpoint_max_age_seconds"), 600, minimum=1, maximum=86_400)
        self.max_strategy_exceptions = coerce_int(cfg.get("max_strategy_exceptions"), 3, minimum=1, maximum=100)
        self.default_failure_source = coerce_str(cfg.get("default_failure_source"), default="handler_agent")
        self.fallback_strategy = normalize_identifier(cfg.get("fallback_strategy", "runtime"), default="runtime")
        self.recovery_strategy_order = tuple(
            strategy
            for strategy in coerce_list(cfg.get("recovery_strategy_order"), default=self.DEFAULT_RECOVERY_STRATEGIES, split_strings=True)
            if normalize_identifier(strategy, default="")
        )
        self.shared_last_result_key = coerce_str(cfg.get("shared_last_result_key"), default=self.DEFAULT_SHARED_LAST_RESULT_KEY)
        self.shared_status_key = coerce_str(cfg.get("shared_status_key"), default=self.DEFAULT_SHARED_STATUS_KEY)

    def _build_recovery_strategies(self) -> Dict[str, RecoveryCallable]:
        registry: Dict[str, RecoveryCallable] = {
            "network": handle_network_error,
            "timeout": handle_timeout_error,
            "memory": handle_memory_error,
            "runtime": handle_runtime_error,
            "dependency": handle_common_dependency_error,
            "resource": handle_resource_constraint,
            "unicode": handle_unicode_emoji_error,
        }
        ordered: Dict[str, RecoveryCallable] = {}
        for strategy_name in self.recovery_strategy_order:
            normalized = normalize_identifier(strategy_name, default="")
            if normalized in registry:
                ordered[normalized] = registry[normalized]
        for strategy_name, handler in registry.items():
            ordered.setdefault(strategy_name, handler)
        return ordered

    def _profile_hot_path(self, path: str, started: float, **extra: Any) -> None:
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        payload = {
            "event_type": "hot_path_profile",
            "path": path,
            "duration_ms": round(duration_ms, 4),
        }
        # Add extra fields that are not None
        payload.update({k: v for k, v in extra.items() if v is not None})
        with contextlib.suppress(Exception):
            self.memory.append_telemetry(payload)

    def perform_task(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute the full Handler recovery pipeline."""
        started = monotonic_timestamp()
        self.calls += 1
        context: Dict[str, Any] = {}

        try:
            if not isinstance(task_data, Mapping):
                raise ValidationError(
                    "HandlerAgent expected task_data to be a mapping",
                    context={"actual_type": type(task_data).__name__},
                    code="HANDLER_AGENT_TASK_MAPPING_REQUIRED",
                    policy=self.error_policy,
                )

            task = coerce_mapping(task_data)
            context = extract_task_context(task, default=task.get("context"))
            context.setdefault("correlation_id", generate_correlation_id("handler-agent"))
            context.setdefault("handler_call", self.calls)

            normalized = self.failure_normalization(
                error=task.get("error"),
                error_info=task.get("error_info"),
                context=context,
                normalized_failure=task.get("normalized_failure"),
            )

            if not self.enabled or not self.enable_recovery:
                recovery_result = self._disabled_recovery_result(normalized_failure=normalized, context=context)
            else:
                recovery_result = self.recovery(
                    target_agent=task.get("target_agent"),
                    task_data=task.get("task_data", task.get("payload")),
                    normalized_failure=normalized,
                    context=context,
                )

            telemetry = self.observability(normalized_failure=normalized, recovery_result=recovery_result, context=context)
            postmortem = self.learning_loop(
                normalized_failure=normalized,
                recovery_result=recovery_result,
                telemetry=telemetry,
                context=context,
            )
            status = self._execution_status(recovery_result)

            execution = HandlerAgentExecution(
                status=status,
                normalized_failure=normalized,
                recovery_result=recovery_result,
                telemetry=telemetry,
                postmortem=postmortem,
                correlation_id=str(context.get("correlation_id")),
                duration_seconds=elapsed_seconds(started),
                calls=self.calls,
                metadata={
                    "target_agent": context.get("agent"),
                    "task_id": context.get("task_id"),
                    "route": context.get("route"),
                },
            ).to_dict()
            self._publish_last_execution(execution)
            return execution

        except HandlerError as exc:
            return self._error_execution(exc, context=context, started=started)
        except Exception as exc:
            handler_error = HandlerError.from_exception(
                exc,
                error_type=HandlerErrorType.RECOVERY,
                severity=FailureSeverity.HIGH,
                retryable=False,
                context=context,
                source="handler_agent.perform_task",
                code="HANDLER_AGENT_PERFORM_TASK_FAILED",
                policy=self.error_policy,
            )
            return self._error_execution(handler_error, context=context, started=started)

    def reset_calls(self) -> None:
        self.calls = 0

    def failure_normalization(
        self,
        error: Optional[BaseException] = None,
        error_info: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        normalized_failure: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert raw exceptions/error payloads into the canonical Handler failure schema."""
        context_map = coerce_mapping(context)
        if isinstance(normalized_failure, Mapping):
            failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
            if not failure.get("correlation_id") and context_map.get("correlation_id"):
                failure["correlation_id"] = context_map.get("correlation_id")
            return failure

        return build_normalized_failure(
            error=error,
            error_info=error_info,
            context=context_map,
            policy=self.error_policy,
            source=self.default_failure_source,
            correlation_id=context_map.get("correlation_id"),
        )

    def recovery(
        self,
        target_agent: Any,
        task_data: Any,
        normalized_failure: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Coordinate SLA, policy, strategy, retry, checkpoint, fallback, and escalation."""
        context_map = coerce_mapping(context)
        failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
        target_agent_name = get_agent_name(target_agent, context=context_map)
        context_map.setdefault("agent", target_agent_name)
        context_map.setdefault("correlation_id", failure.get("correlation_id") or generate_correlation_id("handler-recovery"))

        telemetry_history = self._telemetry_history()
        insight = self._analyze_failure(failure=failure, context=context_map, telemetry_history=telemetry_history)
        insight_payload = self._insight_to_dict(insight)
        sla = self.sla_policy.evaluate(context=context_map, normalized_failure=failure)

        if target_agent is None:
            return self._failed_recovery(
                normalized_failure=failure,
                context=context_map,
                strategy="none",
                recommendation="target_agent_missing",
                attempts=0,
                max_retries=0,
                sla=sla,
                insight=insight_payload,
                last_result={"status": "failed", "reason": "target_agent_missing"},
                checkpoint_id=None,
            )

        policy_started = time.perf_counter()
        policy_decision = self.policy.evaluate_attempt(
            target_agent_name,
            normalized_failure=failure,
            context=context_map,
            attempted_retries=0,
        )
        policy_payload = self._payload(policy_decision)
        self._profile_hot_path("handler.policy.evaluate_attempt", policy_started, target_agent=target_agent_name, allowed=policy_payload.get("allowed"))
        if not coerce_bool(policy_payload.get("allowed"), default=True):
            return self._failed_recovery(
                normalized_failure=failure,
                context=context_map,
                strategy="policy_guardrail",
                recommendation=policy_payload.get("reason", "policy_blocked_attempt"),
                attempts=0,
                max_retries=0,
                sla=sla,
                insight=insight_payload,
                last_result={"status": "failed", "reason": policy_payload.get("reason"), "policy_decision": policy_payload},
                checkpoint_id=None,
                extra={"policy_decision": policy_payload, "breaker": self.policy.breaker_status(target_agent_name)},
            )

        selection = self._select_strategy(failure=failure, context=context_map, telemetry_history=telemetry_history, insight=insight_payload)
        strategy_key = normalize_identifier(selection.get("selected_strategy"), default=self.fallback_strategy)
        strategy_distribution = summarize_strategy_distribution(selection.get("distribution"))
        strategy_handler = self.recovery_strategies.get(strategy_key) or self.recovery_strategies.get(self.fallback_strategy) or handle_runtime_error

        retry_decision = self.adaptive_retry_policy.decide(
            normalized_failure=failure,
            telemetry_history=telemetry_history,
            context=context_map,
            sla=sla,
        )
        retry_payload = self._payload(retry_decision)
        max_retries = min(
            coerce_int(retry_payload.get("retries"), 0, minimum=0),
            coerce_int(sla.get("recommended_attempts"), 0, minimum=0),
        )

        checkpoint_id = self._save_pre_recovery_checkpoint(
            target_agent_name=target_agent_name,
            task_data=task_data,
            context=context_map,
            normalized_failure=failure,
            strategy_key=strategy_key,
            strategy_distribution=strategy_distribution,
            sla=sla,
            retry_decision=retry_payload,
            insight=insight_payload,
        )

        error_info = build_error_info(failure)
        attempts = 0
        strategy_exceptions = 0
        last_result: Dict[str, Any] = {"status": "failed", "reason": "unattempted"}
        executed_strategies: List[str] = []

        while self._can_continue_recovery(
            target_agent_name=target_agent_name,
            attempts=attempts,
            max_retries=max_retries,
            normalized_failure=failure,
            context=context_map,
            sla=sla,
        ):
            if self.honor_retry_delay:
                self._sleep_before_attempt(attempts=attempts, retry_payload=retry_payload, failure=failure)

            raw_result = self._execute_strategy(
                strategy_key=strategy_key,
                strategy_handler=strategy_handler,
                target_agent=target_agent,
                task_data=task_data,
                error_info=error_info,
            )
            attempts += 1
            executed_strategies.append(strategy_key)
            last_result = raw_result

            if self._is_recovered(raw_result):
                self.policy.record_success(target_agent_name, context=context_map, reset_failure_window=False)
                recovered = self._recovery_payload(
                    status="recovered",
                    strategy=strategy_key,
                    attempts=attempts,
                    max_retries=max_retries,
                    result=raw_result.get("result", raw_result),
                    checkpoint_id=checkpoint_id,
                    sla=sla,
                    strategy_distribution=strategy_distribution,
                    insight=insight_payload,
                    policy_decision=policy_payload,
                    retry_decision=retry_payload,
                    selection=selection,
                    executed_strategies=executed_strategies,
                )
                return recovered

            if raw_result.get("exception"):
                strategy_exceptions += 1
                if strategy_exceptions >= self.max_strategy_exceptions:
                    break

            self.policy.record_failure(
                target_agent_name,
                normalized_failure=failure,
                context=context_map,
                reason=raw_result.get("reason", "strategy_failed"),
            )

            if raw_result.get("not_applicable") and strategy_key != self.fallback_strategy and self.fallback_strategy in self.recovery_strategies:
                strategy_key = self.fallback_strategy
                strategy_handler = self.recovery_strategies[strategy_key]

        fallback_result = self._attempt_lightweight_fallback(
            target_agent=target_agent,
            task_data=task_data,
            target_agent_name=target_agent_name,
            strategy_key=strategy_key,
            strategy_distribution=strategy_distribution,
            checkpoint_id=checkpoint_id,
            attempts=attempts,
            sla=sla,
            context=context_map,
            insight=insight_payload,
            policy_decision=policy_payload,
            retry_decision=retry_payload,
            selection=selection,
            executed_strategies=executed_strategies,
        )
        if fallback_result is not None:
            return fallback_result

        restored_state = self.memory.restore_checkpoint(checkpoint_id) if checkpoint_id else None
        escalation_started = time.perf_counter()
        failed = self._failed_recovery(
            normalized_failure=failure,
            context=context_map,
            strategy=strategy_key,
            recommendation="escalate_to_planning_or_evaluation",
            attempts=attempts,
            max_retries=max_retries,
            sla=sla,
            insight=insight_payload,
            last_result=last_result,
            checkpoint_id=checkpoint_id,
            strategy_distribution=strategy_distribution,
            extra={
                "retry_decision": retry_payload,
                "policy_decision": policy_payload,
                "strategy_selection": selection,
                "checkpoint_restored": restored_state is not None,
                "restored_state": restored_state if self.include_failed_checkpoint_state else None,
                "executed_strategies": executed_strategies,
            },
        )
        self._profile_hot_path("handler.escalation.failed_recovery", escalation_started, target_agent=target_agent_name, attempts=attempts)
        return failed

    def observability(
        self,
        normalized_failure: Mapping[str, Any],
        recovery_result: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write structured telemetry through HandlerMemory."""
        context_map = coerce_mapping(context)
        failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
        recovery = normalize_recovery_result(recovery_result)

        if not self.enable_observability:
            return {
                "event_type": "handler_recovery",
                "status": "skipped",
                "reason": "observability_disabled",
                "timestamp": utc_timestamp(),
                "failure": failure,
                "recovery": recovery,
            }

        try:
            insight_payload = coerce_mapping(recovery_result.get("insight"))
            if not insight_payload:
                insight = self._analyze_failure(failure=failure, context=context_map, telemetry_history=self._telemetry_history())
                insight_payload = self._insight_to_dict(insight)

            telemetry_event = self.memory.append_recovery_telemetry(
                failure=failure,
                recovery=recovery,
                context=context_map,
                insight=insight_payload,
                sla=coerce_mapping(recovery.get("sla")),
                strategy_distribution=coerce_mapping(recovery.get("strategy_distribution")),
                correlation_id=failure.get("correlation_id") or context_map.get("correlation_id"),
            )
            logger.info(
                "HandlerAgent telemetry emitted | failure=%s severity=%s recovery=%s strategy=%s",
                failure.get("type"),
                failure.get("severity"),
                recovery.get("status"),
                recovery.get("strategy"),
            )
            return telemetry_event
        except HandlerError as exc:
            logger.warning("HandlerAgent observability failed: %s", exc)
            return exc.to_telemetry_dict()
        except Exception as exc:
            err = TelemetryError(
                "HandlerAgent failed to emit telemetry",
                cause=exc,
                context={"failure_type": failure.get("type")},
                code="HANDLER_AGENT_TELEMETRY_FAILED",
                policy=self.error_policy,
            )
            logger.warning("HandlerAgent observability failed: %s", err)
            return err.to_telemetry_dict()

    def learning_loop(
        self,
        normalized_failure: Mapping[str, Any],
        recovery_result: Mapping[str, Any],
        telemetry: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Emit postmortem artifact through HandlerMemory for adaptive/learning loops."""
        if not self.enable_learning_loop:
            return {"status": "skipped", "reason": "learning_loop_disabled", "timestamp": utc_timestamp()}

        try:
            return self.memory.append_postmortem(
                normalized_failure=normalized_failure,
                recovery_result=recovery_result,
                telemetry=telemetry,
                context=context,
            )
        except HandlerError as exc:
            logger.warning("HandlerAgent postmortem failed: %s", exc)
            return exc.to_telemetry_dict()
        except Exception as exc:
            err = TelemetryError(
                "HandlerAgent failed to append postmortem",
                cause=exc,
                context={"telemetry_type": type(telemetry).__name__},
                code="HANDLER_AGENT_POSTMORTEM_FAILED",
                policy=self.error_policy,
            )
            logger.warning("HandlerAgent postmortem failed: %s", err)
            return err.to_telemetry_dict()

    def health(self) -> Dict[str, Any]:
        """Return a compact health payload for dashboards/tests."""
        return {
            "status": "ok" if self.enabled else "disabled",
            "timestamp": utc_timestamp(),
            "calls": self.calls,
            "memory": self.memory.health() if hasattr(self.memory, "health") else {},
            "policy": self.policy.health() if hasattr(self.policy, "health") else {},
            "components": {
                "memory": type(self.memory).__name__,
                "policy": type(self.policy).__name__,
                "adaptive_retry_policy": type(self.adaptive_retry_policy).__name__,
                "strategy_selector": type(self.strategy_selector).__name__,
                "sla_policy": type(self.sla_policy).__name__,
                "escalation_manager": type(self.escalation_manager).__name__,
                "failure_intelligence": type(self.failure_intelligence).__name__,
                "issue_handler": type(self.issue_handler).__name__,
            },
            "strategies": sorted(self.recovery_strategies.keys()),
        }

    def _disabled_recovery_result(self, *, normalized_failure: Mapping[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
        return self._recovery_payload(
            status="skipped",
            strategy="none",
            attempts=0,
            max_retries=0,
            result={"reason": "handler_agent_disabled"},
            checkpoint_id=None,
            sla={"can_retry": False, "mode": "disabled", "remaining_seconds": 0.0},
            strategy_distribution={},
            insight={},
            policy_decision={},
            retry_decision={},
            selection={},
            executed_strategies=[],
            recommendation="handler_agent_disabled",
        )

    def _select_strategy(
        self,
        *,
        failure: Mapping[str, Any],
        context: Mapping[str, Any],
        telemetry_history: Sequence[Mapping[str, Any]],
        insight: Mapping[str, Any],
    ) -> Dict[str, Any]:
        selection = self.strategy_selector.decide(
            normalized_failure=failure,
            telemetry_history=telemetry_history,
            context=context,
            insight=insight,
            available_strategies=tuple(self.recovery_strategies.keys()),
        )
        return self._payload(selection)

    def _save_pre_recovery_checkpoint(
        self,
        *,
        target_agent_name: str,
        task_data: Any,
        context: Mapping[str, Any],
        normalized_failure: Mapping[str, Any],
        strategy_key: str,
        strategy_distribution: Mapping[str, Any],
        sla: Mapping[str, Any],
        retry_decision: Mapping[str, Any],
        insight: Mapping[str, Any],
    ) -> str:
        return self.memory.save_checkpoint(
            label="pre_recovery",
            state={
                "task_data": task_data,
                "context": dict(context),
                "target_agent_name": target_agent_name,
                "failure": dict(normalized_failure),
            },
            metadata={
                "task_id": context.get("task_id"),
                "agent": target_agent_name,
                "route": context.get("route"),
                "correlation_id": context.get("correlation_id"),
                "strategy": strategy_key,
                "strategy_distribution": dict(strategy_distribution),
                "failure_type": normalized_failure.get("type"),
                "context_hash": normalized_failure.get("context_hash"),
                "sla": dict(sla),
                "retry_decision": dict(retry_decision),
                "insight_signature": insight.get("signature"),
            },
            ttl_seconds=self.checkpoint_max_age_seconds,
            correlation_id=str(context.get("correlation_id")),
        )

    def _can_continue_recovery(
        self,
        *,
        target_agent_name: str,
        attempts: int,
        max_retries: int,
        normalized_failure: Mapping[str, Any],
        context: Mapping[str, Any],
        sla: Mapping[str, Any],
    ) -> bool:
        if max_retries <= 0 or not coerce_bool(sla.get("can_retry"), default=False):
            return False
        decision = self.policy.evaluate_attempt(
            target_agent_name,
            normalized_failure=normalized_failure,
            context=context,
            attempted_retries=attempts,
            max_retries=max_retries,
        )
        return bool(decision.allowed)

    def _execute_strategy(
        self,
        *,
        strategy_key: str,
        strategy_handler: RecoveryCallable,
        target_agent: Any,
        task_data: Any,
        error_info: Mapping[str, Any],
    ) -> Dict[str, Any]:
        try:
            raw_result = strategy_handler(target_agent, task_data, error_info, self.issue_handler)
            payload = self._payload(raw_result)
            recovered = self._is_recovered(payload)
            status = "recovered" if recovered else str(payload.get("status") or "failed").lower()
            not_applicable = status in {"not_applicable", "skipped"} or payload.get("handled") is False
            return compact_dict(
                {
                    "status": status,
                    "strategy": strategy_key,
                    "handled": payload.get("handled"),
                    "recovered": payload.get("recovered", recovered),
                    "reason": payload.get("reason") or payload.get("message"),
                    "result": payload.get("result", payload if self.include_result_payload else None),
                    "attempts": payload.get("attempts"),
                    "handler_name": payload.get("handler_name"),
                    "issue_key": payload.get("issue_key"),
                    "not_applicable": not_applicable,
                },
                drop_none=True,
                drop_empty=True,
            )
        except Exception as exc:
            err = RecoveryError(
                "HandlerAgent recovery strategy raised an exception",
                cause=exc,
                context={"strategy": strategy_key, "error_type": error_info.get("error_type")},
                code="HANDLER_AGENT_STRATEGY_EXCEPTION",
                policy=self.error_policy,
            )
            return {
                "status": "failed",
                "strategy": strategy_key,
                "reason": "strategy_exception",
                "exception": err.to_telemetry_dict(),
                "recovered": False,
            }

    def _attempt_lightweight_fallback(
        self,
        *,
        target_agent: Any,
        task_data: Any,
        target_agent_name: str,
        strategy_key: str,
        strategy_distribution: Mapping[str, Any],
        checkpoint_id: Optional[str],
        attempts: int,
        sla: Mapping[str, Any],
        context: Mapping[str, Any],
        insight: Mapping[str, Any],
        policy_decision: Mapping[str, Any],
        retry_decision: Mapping[str, Any],
        selection: Mapping[str, Any],
        executed_strategies: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if not self.enable_lightweight_fallback:
            return None
        if not callable(getattr(target_agent, "use_lightweight_mode", None)):
            return None
        if coerce_float(sla.get("remaining_seconds"), 0.0) <= 0:
            return None

        enabled = False
        try:
            target_agent.use_lightweight_mode(True)
            enabled = True
            fallback = target_agent.perform_task(task_data)
            self.policy.record_success(target_agent_name, context=context, reset_failure_window=False)
            return self._recovery_payload(
                status="recovered",
                strategy=f"{strategy_key}+fallback_mode",
                attempts=attempts + 1,
                max_retries=attempts + 1,
                result=fallback,
                checkpoint_id=checkpoint_id,
                sla=sla,
                strategy_distribution=strategy_distribution,
                insight=insight,
                policy_decision=policy_decision,
                retry_decision=retry_decision,
                selection=selection,
                executed_strategies=list(executed_strategies) + ["fallback_mode"],
                recommendation="recovered_with_lightweight_fallback",
            )
        except Exception as exc:
            self.policy.record_failure(
                target_agent_name,
                normalized_failure={"type": type(exc).__name__, "message": str(exc), "severity": "medium", "retryable": False},
                context=context,
                reason="lightweight_fallback_failed",
            )
            return None
        finally:
            if enabled:
                with contextlib.suppress(Exception):
                    target_agent.use_lightweight_mode(False)

    def _failed_recovery(
        self,
        *,
        normalized_failure: Mapping[str, Any],
        context: Mapping[str, Any],
        strategy: str,
        recommendation: str,
        attempts: int,
        max_retries: int,
        sla: Mapping[str, Any],
        insight: Mapping[str, Any],
        last_result: Mapping[str, Any],
        checkpoint_id: Optional[str],
        strategy_distribution: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
        result = self._recovery_payload(
            status="failed",
            strategy=strategy,
            attempts=attempts,
            max_retries=max_retries,
            result=None,
            checkpoint_id=checkpoint_id,
            sla=sla,
            strategy_distribution=strategy_distribution or {},
            insight=insight,
            policy_decision=coerce_mapping((extra or {}).get("policy_decision")),
            retry_decision=coerce_mapping((extra or {}).get("retry_decision")),
            selection=coerce_mapping((extra or {}).get("strategy_selection")),
            executed_strategies=coerce_list((extra or {}).get("executed_strategies")),
            recommendation=recommendation,
            last_result=last_result,
            extra=extra,
        )
        if self.enable_escalation:
            result["escalation"] = self.escalation_manager.build_handoff_payload(
                normalized_failure=failure,
                recovery_result=result,
                context=context,
                strategy_distribution=strategy_distribution or result.get("strategy_distribution"),
                sla=sla,
                insight=insight,
            )
        return result

    def _recovery_payload(
        self,
        *,
        status: str,
        strategy: str,
        attempts: int,
        max_retries: int,
        result: Any,
        checkpoint_id: Optional[str],
        sla: Mapping[str, Any],
        strategy_distribution: Mapping[str, Any],
        insight: Mapping[str, Any],
        policy_decision: Mapping[str, Any],
        retry_decision: Mapping[str, Any],
        selection: Mapping[str, Any],
        executed_strategies: Sequence[str],
        recommendation: Optional[str] = None,
        last_result: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = compact_dict(
            {
                "status": status,
                "strategy": strategy,
                "strategy_distribution": summarize_strategy_distribution(strategy_distribution),
                "attempts": coerce_int(attempts, 0, minimum=0),
                "max_retries": coerce_int(max_retries, 0, minimum=0),
                "result": make_json_safe(result) if self.include_result_payload else None,
                "checkpoint_id": checkpoint_id,
                "checkpoint_restored": False,
                "sla": dict(sla),
                "insight": dict(insight),
                "policy_decision": dict(policy_decision),
                "retry_decision": dict(retry_decision),
                "strategy_selection": dict(selection),
                "executed_strategies": list(executed_strategies),
                "recommendation": recommendation,
                "last_result": dict(last_result) if isinstance(last_result, Mapping) else None,
            },
            drop_none=True,
            drop_empty=True,
        )
        payload.update(coerce_mapping(extra))
        return normalize_recovery_result(payload) | {k: v for k, v in payload.items() if k not in normalize_recovery_result(payload)}

    def _analyze_failure(
        self,
        *,
        failure: Mapping[str, Any],
        context: Mapping[str, Any],
        telemetry_history: Sequence[Mapping[str, Any]],
    ) -> FailureInsight:
        return self.failure_intelligence.analyze(
            normalized_failure=failure,
            context=context,
            telemetry_history=[dict(event) for event in telemetry_history if isinstance(event, Mapping)],
        )

    @staticmethod
    def _payload(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            payload = value.to_dict()
            return dict(payload) if isinstance(payload, Mapping) else {"value": payload}
        return {"value": make_json_safe(value)}

    @staticmethod
    def _insight_to_dict(insight: Any) -> Dict[str, Any]:
        if hasattr(insight, "to_dict") and callable(insight.to_dict):
            payload = insight.to_dict()
            return dict(payload) if isinstance(payload, Mapping) else {}
        return dict(insight) if isinstance(insight, Mapping) else {}

    @staticmethod
    def _is_recovered(result: Any) -> bool:
        payload = HandlerAgent._payload(result)
        status = str(payload.get("status", "")).lower()
        if payload.get("recovered") is True:
            return True
        if status in {"recovered", "ok", "success", "degraded"}:
            return True
        if status in {"failed", "error", "not_applicable", "skipped"}:
            return False
        return is_recovered_result(payload)

    @staticmethod
    def _execution_status(recovery_result: Mapping[str, Any]) -> str:
        status = str(recovery_result.get("status", "failed")).lower()
        if status == "recovered":
            return "ok"
        if status == "degraded":
            return "degraded"
        if status == "skipped":
            return "skipped"
        return "failed"

    def _sleep_before_attempt(self, *, attempts: int, retry_payload: Mapping[str, Any], failure: Mapping[str, Any]) -> None:
        schedule = coerce_list(retry_payload.get("delay_schedule"))
        if attempts >= len(schedule):
            return
        delay = coerce_float(schedule[attempts], 0.0, minimum=0.0, maximum=60.0)
        if delay > 0:
            logger.debug("HandlerAgent sleeping before retry | delay=%s failure=%s", delay, failure.get("context_hash"))
            time.sleep(delay)

    def _telemetry_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        history_limit = coerce_int(limit, self.telemetry_buffer_size, minimum=1, maximum=self.telemetry_buffer_size)
        if hasattr(self.memory, "recent_telemetry") and callable(self.memory.recent_telemetry):
            return self.memory.recent_telemetry(limit=history_limit)
        value = shared_memory_get(self.shared_memory, "handler:telemetry", default=[])
        return [dict(event) for event in coerce_list(value)[-history_limit:] if isinstance(event, Mapping)]

    def _publish_last_execution(self, execution: Mapping[str, Any]) -> None:
        if not self.publish_last_execution_to_shared_memory:
            return
        shared_memory_set(self.shared_memory, self.shared_last_result_key, make_json_safe(execution))
        self._publish_status("last_execution", {"status": execution.get("status"), "correlation_id": execution.get("correlation_id")})

    def _publish_status(self, status: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if not self.publish_status_to_shared_memory:
            return
        shared_memory_set(
            self.shared_memory,
            self.shared_status_key,
            {
                "status": status,
                "timestamp": utc_timestamp(),
                "calls": getattr(self, "calls", 0),
                "metadata": make_json_safe(metadata or {}),
            },
        )

    def _error_execution(self, error: HandlerError, *, context: Mapping[str, Any], started: float) -> Dict[str, Any]:
        failure = error.to_failure_payload()
        recovery = {
            "status": "failed",
            "strategy": "handler_agent_guardrail",
            "recommendation": error.action_value,
            "attempts": 0,
            "error": error.to_telemetry_dict(),
            "sla": {"can_retry": False, "mode": "fail_fast", "remaining_seconds": 0.0},
        }
        telemetry = self.observability(normalized_failure=failure, recovery_result=recovery, context=context)
        postmortem = self.learning_loop(normalized_failure=failure, recovery_result=recovery, telemetry=telemetry, context=context)
        execution = HandlerAgentExecution(
            status="failed",
            normalized_failure=failure,
            recovery_result=recovery,
            telemetry=telemetry,
            postmortem=postmortem,
            correlation_id=str(context.get("correlation_id") or error.correlation_id or generate_correlation_id("handler-error")),
            duration_seconds=elapsed_seconds(started),
            calls=self.calls,
            metadata={"error_type": error.error_type, "code": error.code},
        ).to_dict()
        self._publish_last_execution(execution)
        return execution


if __name__ == "__main__":
    print("\n=== Running Handler Agent ===\n")
    printer.status("TEST", "Handler Agent initialized", "info")

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    factory = AgentFactory()
    execution_config = get_config_section("handler_agent")

    agent = HandlerAgent(
        shared_memory=shared_memory,
        agent_factory=factory,
        config=execution_config,
    )

    class DemoTargetAgent:
        def __init__(self, name: str = "demo_target"):
            self.name = name
            self.failures_remaining = 0
            self.lightweight_mode = False
            self.calls = 0

        def use_lightweight_mode(self, enabled: bool) -> None:
            self.lightweight_mode = bool(enabled)

        def perform_task(self, task_data: Any) -> Dict[str, Any]:
            self.calls += 1
            if self.failures_remaining > 0:
                self.failures_remaining -= 1
                raise TimeoutError("Simulated timeout with Authorization: Bearer token-123")
            return {
                "ok": True,
                "task": task_data,
                "lightweight_mode": self.lightweight_mode,
                "calls": self.calls,
            }

    target = DemoTargetAgent()

    print("\n* * * * * Phase 1: failure normalization * * * * *\n")
    normalized = agent.failure_normalization(
        error=TimeoutError("Connection timed out while calling upstream with password=SuperSecret123"),
        context={"route": "demo_route", "agent": target.name, "task_id": "demo-001", "correlation_id": "corr-handler-agent-test"},
    )
    printer.pretty("Normalized failure", normalized, "success")

    print("\n* * * * * Phase 2: recovery success within policy/SLA budget * * * * *\n")
    target.failures_remaining = 1
    recovery = agent.recovery(
        target_agent=target,
        task_data={"operation": "compute", "payload": {"x": 7, "y": 5}},
        normalized_failure=normalized,
        context={
            "route": "demo_route",
            "agent": target.name,
            "task_id": "demo-002",
            "correlation_id": "corr-handler-agent-test",
            "sla": {"max_recovery_seconds": 15},
        },
    )
    printer.pretty("Recovery result", recovery, "success" if recovery.get("status") == "recovered" else "error")

    print("\n* * * * * Phase 3: end-to-end perform_task * * * * *\n")
    target.failures_remaining = 3
    full = agent.perform_task(
        {
            "target_agent": target,
            "task_data": {"operation": "sync", "payload": {"record": 42}},
            "error": TimeoutError("Network timeout during sync with api_key=sk-test-123"),
            "context": {
                "route": "sync_route",
                "agent": target.name,
                "task_id": "demo-003",
                "correlation_id": "corr-handler-agent-test-e2e",
                "sla": {"latency_budget_ms": 4000},
            },
        }
    )
    printer.pretty("perform_task", full, "success" if full.get("status") in {"ok", "failed", "degraded"} else "error")

    health = agent.health()
    serialized = stable_json_dumps({"normalized": normalized, "recovery": recovery, "full": full, "health": health})

    assert "context_hash" in normalized
    assert recovery.get("status") in {"recovered", "failed", "degraded"}
    assert "strategy_distribution" in recovery
    assert "sla" in recovery
    assert full.get("recovery_result", {}).get("sla") is not None
    assert agent.memory.health()["status"] == "ok"
    assert "SuperSecret123" not in serialized
    assert "sk-test-123" not in serialized
    assert "token-123" not in serialized

    printer.pretty("Handler health", health, "success")
    print("\n=== Test ran successfully ===\n")
