from __future__ import annotations

__version__ = "2.1.0"

import hashlib
import json
import time

from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from .base.issue_handler import (IssueHandler,
    handle_common_dependency_error,
    handle_memory_error,
    handle_network_error,
    handle_resource_constraint,
    handle_runtime_error,
    handle_timeout_error,
    handle_unicode_emoji_error,
)
from .base.utils.main_config_loader import get_config_section, load_global_config
from .handler.adaptive_retry_policy import AdaptiveRetryPolicy
from .handler.escalation_manager import EscalationManager
from .handler.handler_policy import HandlerPolicy
from .handler.sla_policy import SLARecoveryPolicy
from .handler.strategy_selector import ProbabilisticStrategySelector
from .handler.failure_intelligence import FailureIntelligence
from .handler.utils.handler_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Handler Agent")
printer = PrettyPrinter()

class HandlerAgent(BaseAgent):
    """Cross-agent reliability layer for failure normalization and recovery orchestration."""
    _SEVERITY_CRITICAL_TAGS = ("critical", "fatal", "security", "data loss")
    _SEVERITY_HIGH_TAGS = ("oom", "outofmemory", "memory", "runtime", "dependency")
    _SEVERITY_MEDIUM_TAGS = ("timeout", "network", "connection")
    _RETRYABLE_TAGS = ("timeout", "network", "connection", "resource busy", "temporary")

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.config = load_global_config()
        self.agent_config = get_config_section("handler_agent")
        if config:
            self.agent_config.update(config)

        self.policy = HandlerPolicy()
        self.adaptive_retry_policy = AdaptiveRetryPolicy()
        self.strategy_selector = ProbabilisticStrategySelector()
        self.sla_policy = SLARecoveryPolicy()
        self.escalation_manager = EscalationManager()
        self.failure_intelligence = FailureIntelligence()
        self.issue_handler = IssueHandler()

        self.recovery_strategies = {
            "network": handle_network_error,
            "timeout": handle_timeout_error,
            "memory": handle_memory_error,
            "runtime": handle_runtime_error,
            "dependency": handle_common_dependency_error,
            "resource": handle_resource_constraint,
            "unicode": handle_unicode_emoji_error,
        }
        self.calls = 0

        logger.info("HandlerAgent initialized")

    def perform_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Entry-point that executes normalization, recovery, telemetry, and postmortem emission."""
        self.calls += 1
        error = task_data.get("error")
        error_info = task_data.get("error_info")
        target_agent = task_data.get("target_agent")
        target_task_data = task_data.get("task_data")
        context = task_data.get("context", {})

        normalized = self.failure_normalization(error=error, error_info=error_info, context=context)
        recovery_result = self.recovery(
            target_agent=target_agent,
            task_data=target_task_data,
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

        return {
            "status": "ok" if recovery_result.get("status") == "recovered" else "failed",
            "normalized_failure": normalized,
            "recovery_result": recovery_result,
            "telemetry": telemetry,
            "postmortem": postmortem,
        }
    
    def reset_calls(self):      # optional, for test isolation
        self.calls = 0

    def failure_normalization(
        self,
        error: Optional[Exception] = None,
        error_info: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert raw exceptions/error payloads into canonical failure schema."""
        context = context or {}
        error_info = error_info or {}

        error_type = error_info.get("error_type")
        error_message = error_info.get("error_message")

        if error is not None:
            error_type = error_type or type(error).__name__
            error_message = error_message or str(error)

        error_type = error_type or "UnknownError"
        error_message = error_message or "No error message provided"

        lowered = f"{error_type} {error_message}".lower()

        if any(tag in lowered for tag in self._SEVERITY_CRITICAL_TAGS):
            severity = "critical"
        elif any(tag in lowered for tag in self._SEVERITY_HIGH_TAGS):
            severity = "high"
        elif any(tag in lowered for tag in self._SEVERITY_MEDIUM_TAGS):
            severity = "medium"
        else:
            severity = "low"

        retryable = (
            any(tag in lowered for tag in self._RETRYABLE_TAGS)
            and "invalid" not in lowered
        )

        fingerprint_payload = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context,
        }
        context_hash = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

        normalized = {
            "type": error_type,
            "message": error_message,
            "severity": severity,
            "retryable": retryable,
            "context_hash": context_hash,
            "timestamp": utc_timestamp(),
        }

        return normalized

    def observability(
        self,
        normalized_failure: Dict[str, Any],
        recovery_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write structured telemetry to logs + shared memory for adaptation/evaluation."""
        context = context or {}
        telemetry_history = self._telemetry_history()
        insight = self.failure_intelligence.analyze(
            normalized_failure=normalized_failure,
            context=context,
            telemetry_history=telemetry_history,
        )
        telemetry_event = {
            "event_type": "handler_recovery",
            "timestamp": time.time(),
            "failure": normalize_failure_payload(normalized_failure),
            "recovery": recovery_result,
            "insight": insight.to_dict(),
            "context": {
                "route": context.get("route"),
                "agent": context.get("agent"),
                "task_id": context.get("task_id"),
            },
            "sla": recovery_result.get("sla", {}),
            "strategy_distribution": recovery_result.get("strategy_distribution", {}),
        }

        key = "handler:telemetry"
        if hasattr(self.shared_memory, "get") and hasattr(self.shared_memory, "set"):
            current = self.shared_memory.get(key) or []
            max_items = coerce_int(self.agent_config.get("telemetry_buffer_size", 1000), 1000, minimum=10, maximum=10000)
            bounded_append(current, telemetry_event, max_items=max_items)
            self.shared_memory.set(key, current)

        logger.info(
            "[HandlerAgent][Observability] failure_type=%s severity=%s recovered=%s",
            normalized_failure.get("type"),
            normalized_failure.get("severity"),
            recovery_result.get("status") == "recovered",
        )

        return telemetry_event

    def learning_loop(
        self,
        normalized_failure: Dict[str, Any],
        recovery_result: Dict[str, Any],
        telemetry: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Emit postmortem artifact for Learning/Adaptive agents."""
        context = context or {}

        postmortem = {
            "timestamp": utc_timestamp(),
            "failure_type": normalized_failure.get("type"),
            "severity": normalized_failure.get("severity"),
            "retryable": normalized_failure.get("retryable"),
            "context_hash": normalized_failure.get("context_hash"),
            "recovery_status": recovery_result.get("status"),
            "strategy": recovery_result.get("strategy"),
            "task_id": context.get("task_id"),
            "telemetry_ref": telemetry.get("timestamp"),
            "failure_signature": telemetry.get("insight", {}).get("signature"),
            "recommendation": telemetry.get("insight", {}).get("recommendation", recovery_result.get("recommendation", "collect_more_signals")),
        }

        if hasattr(self.shared_memory, "get") and hasattr(self.shared_memory, "set"):
            key = "handler:postmortems"
            current = self.shared_memory.get(key) or []
            current.append(postmortem)
            self.shared_memory.set(key, current[-1000:])

        return postmortem

    def recovery(
        self,
        target_agent: Any,
        task_data: Any,
        normalized_failure: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Select and apply recovery strategy using adaptive retry, probabilistic routing, and SLA policy."""
        context = context or {}
        target_agent_name = getattr(target_agent, "name", context.get("agent", "unknown_agent"))
        telemetry_history = self._telemetry_history()
        sla = self.sla_policy.evaluate(context=context, normalized_failure=normalized_failure)

        if target_agent is None:
            failed = {
                "status": "failed",
                "strategy": "none",
                "recommendation": "target_agent_missing",
                "sla": sla,
            }
            failed["escalation"] = self.escalation_manager.build_handoff_payload(
                normalized_failure=normalized_failure,
                recovery_result=failed,
                context=context,
                sla=sla,
            )
            return failed

        if not self.policy.can_attempt(target_agent_name):
            failed = {
                "status": "failed",
                "strategy": "circuit_breaker",
                "recommendation": "defer_and_retry_later",
                "breaker": self.policy.breaker_status(target_agent_name),
                "sla": sla,
            }
            failed["escalation"] = self.escalation_manager.build_handoff_payload(
                normalized_failure=normalized_failure,
                recovery_result=failed,
                context=context,
                sla=sla,
            )
            return failed

        selection = self.strategy_selector.select(normalized_failure=normalized_failure, telemetry_history=telemetry_history)
        strategy_key = selection["selected_strategy"]
        strategy_distribution = selection["distribution"]
        handler = self.recovery_strategies.get(strategy_key, handle_runtime_error)

        max_retries_adaptive = self.adaptive_retry_policy.retries_for_fingerprint(
            fingerprint=normalized_failure.get("context_hash", ""),
            severity=normalized_failure.get("severity", "low"),
            retryable=bool(normalized_failure.get("retryable")),
            telemetry_history=telemetry_history,
        )
        max_retries = min(max_retries_adaptive, int(sla.get("recommended_attempts", 0)))

        error_info = {
            "error_type": normalized_failure.get("type", "UnknownError"),
            "error_message": normalized_failure.get("message", ""),
        }

        checkpoint_id = f"handler:checkpoint:{int(time.time() * 1000)}"
        if hasattr(self.shared_memory, "set"):
            self.shared_memory.set(
                checkpoint_id,
                {
                    "label": "pre_recovery",
                    "state": {"task_data": task_data, "context": context, "target_agent_name": target_agent_name},
                    "metadata": {
                        "strategy": strategy_key,
                        "strategy_distribution": strategy_distribution,
                        "failure_type": normalized_failure.get("type"),
                        "sla": sla,
                    },
                },
                ttl=int(self.agent_config.get("checkpoint_max_age_seconds", 600)),
            )

        retries = 0
        last_result = {"status": "failed", "reason": "unattempted"}

        while self.policy.retries_allowed(retries, max_retries=max_retries) and sla.get("can_retry", False):
            result = handler(target_agent, task_data, error_info, self.issue_handler)
            last_result = result if isinstance(result, dict) else {"status": "recovered", "result": result}

            if not (isinstance(last_result, dict) and last_result.get("status") == "failed"):
                self.policy.record_success(target_agent_name)
                return {
                    "status": "recovered",
                    "strategy": strategy_key,
                    "strategy_distribution": strategy_distribution,
                    "result": result,
                    "checkpoint_id": checkpoint_id,
                    "attempts": retries + 1,
                    "sla": sla,
                }

            retries += 1

        if hasattr(target_agent, "use_lightweight_mode") and sla.get("remaining_seconds", 0.0) > 0:
            try:
                target_agent.use_lightweight_mode(True)
                fallback = target_agent.perform_task(task_data)
                target_agent.use_lightweight_mode(False)
                self.policy.record_success(target_agent_name)
                return {
                    "status": "recovered",
                    "strategy": f"{strategy_key}+fallback_mode",
                    "strategy_distribution": strategy_distribution,
                    "result": fallback,
                    "checkpoint_id": checkpoint_id,
                    "attempts": retries + 1,
                    "sla": sla,
                }
            except Exception:
                try:
                    target_agent.use_lightweight_mode(False)
                except Exception:
                    pass

        restored_state = self.shared_memory.get(checkpoint_id) if hasattr(self.shared_memory, "get") else None
        self.policy.record_failure(target_agent_name)
        failed = {
            "status": "failed",
            "strategy": strategy_key,
            "strategy_distribution": strategy_distribution,
            "attempts": retries,
            "max_retries": max_retries,
            "last_result": last_result,
            "checkpoint_restored": restored_state is not None,
            "checkpoint_id": checkpoint_id,
            "recommendation": "escalate_to_planning_or_evaluation",
            "sla": sla,
        }
        failed["escalation"] = self.escalation_manager.build_handoff_payload(
            normalized_failure=normalized_failure,
            recovery_result=failed,
            context=context,
            strategy_distribution=strategy_distribution,
            sla=sla,
        )
        return failed

    def _telemetry_history(self) -> list[Dict[str, Any]]:
        if hasattr(self.shared_memory, "get"):
            return self.shared_memory.get("handler:telemetry") or []
        return []


if __name__ == "__main__":
    print("\n=== Running Handler Agent ===\n")
    printer.status("TEST", "Starting Handler Agent tests", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory
    from .adaptive_agent import AdaptiveAgent

    memory = SharedMemory()
    factory = AgentFactory()
    execution_config = get_config_section("handler_agent")

    agent = HandlerAgent(
        shared_memory=memory,
        agent_factory=factory,
        config=execution_config,
    )

    class DemoTargetAgent:
        def __init__(self, name="demo_target"):
            self.name = name
            self.failures_remaining = 0
            self.lightweight_mode = False

        def use_lightweight_mode(self, enabled: bool):
            self.lightweight_mode = enabled

        def perform_task(self, task_data):
            if self.failures_remaining > 0:
                self.failures_remaining -= 1
                raise TimeoutError("Simulated timeout from DemoTargetAgent")

            return {
                "ok": True,
                "task": task_data,
                "lightweight_mode": self.lightweight_mode,
            }

    target = DemoTargetAgent()

    print("\n* * * * * Phase 1: failure_normalization * * * * *\n")
    normalized = agent.failure_normalization(
        error=TimeoutError("Connection timed out while calling upstream"),
        context={"route": "demo_route", "agent": target.name, "task_id": "demo-001"},
    )
    printer.pretty("Normalized failure", normalized, "success")

    print("\n* * * * * Phase 2: recovery success within retries * * * * *\n")
    target.failures_remaining = 1
    recovery = agent.recovery(
        target_agent=target,
        task_data={"operation": "compute", "payload": {"x": 7, "y": 5}},
        normalized_failure=normalized,
        context={
            "route": "demo_route",
            "agent": target.name,
            "task_id": "demo-002",
            "sla": {"max_recovery_seconds": 15},
        },
    )
    printer.pretty("Recovery result", recovery, "success" if recovery.get("status") == "recovered" else "error")

    print("\n* * * * * Phase 3: end-to-end perform_task with SLA + escalation fields * * * * *\n")
    target.failures_remaining = 3
    full = agent.perform_task(
        {
            "target_agent": target,
            "task_data": {"operation": "sync", "payload": {"record": 42}},
            "error": TimeoutError("Network timeout during sync"),
            "context": {
                "route": "sync_route",
                "agent": target.name,
                "task_id": "demo-003",
                "sla": {"latency_budget_ms": 4000},
            },
        }
    )
    printer.pretty("perform_task", full, "success" if full.get("status") in {"ok", "failed"} else "error")

    assert "strategy_distribution" in recovery
    assert "sla" in recovery
    assert full.get("recovery_result", {}).get("sla") is not None

    print("\n=== All tests completed successfully! ===\n")
