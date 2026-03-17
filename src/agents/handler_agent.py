import hashlib
import json
import time

from typing import Any, Dict, Optional

from src.agents.base_agent import BaseAgent
from src.agents.base.issue_handler import (
    handle_common_dependency_error,
    handle_memory_error,
    handle_network_error,
    handle_resource_constraint,
    handle_runtime_error,
    handle_timeout_error,
    handle_unicode_emoji_error,
)
from src.agents.handler.handler_memory import HandlerMemory
from src.agents.handler.handler_policy import HandlerPolicy
from src.agents.handler.utils.config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Handler Agent")
printer = PrettyPrinter

class HandlerAgent(BaseAgent):
    """Cross-agent reliability layer for failure normalization and recovery orchestration."""

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.config = load_global_config()
        self.handler_config = get_config_section("handler_agent")
        if config:
            self.handler_config.update(config)

        self.memory = HandlerMemory(config=self.handler_config)
        self.policy = HandlerPolicy(config=self.handler_config)
        self.recovery_strategies = {
            "network": handle_network_error,
            "timeout": handle_timeout_error,
            "memory": handle_memory_error,
            "runtime": handle_runtime_error,
            "dependency": handle_common_dependency_error,
            "resource": handle_resource_constraint,
            "unicode": handle_unicode_emoji_error,
        }

        logger.info("HandlerAgent initialized")

    def perform_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Entry-point that executes normalization, recovery, telemetry, and postmortem emission."""
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

        if any(tag in lowered for tag in ["critical", "fatal", "security", "data loss"]):
            severity = "critical"
        elif any(tag in lowered for tag in ["oom", "outofmemory", "memory", "runtime", "dependency"]):
            severity = "high"
        elif any(tag in lowered for tag in ["timeout", "network", "connection"]):
            severity = "medium"
        else:
            severity = "low"

        retryable = any(tag in lowered for tag in ["timeout", "network", "connection", "resource busy", "temporary"]) \
            and "invalid" not in lowered

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
            "timestamp": time.time(),
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
        telemetry_event = {
            "event_type": "handler_recovery",
            "timestamp": time.time(),
            "failure": normalized_failure,
            "recovery": recovery_result,
            "context": {
                "route": context.get("route"),
                "agent": context.get("agent"),
                "task_id": context.get("task_id"),
            },
        }

        self.memory.append_telemetry(telemetry_event)

        key = "handler:telemetry"
        if hasattr(self.shared_memory, "get") and hasattr(self.shared_memory, "set"):
            current = self.shared_memory.get(key) or []
            current.append(telemetry_event)
            max_items = self.handler_config.get("telemetry_buffer_size", 1000)
            self.shared_memory.set(key, current[-max_items:])

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
            "timestamp": time.time(),
            "failure_type": normalized_failure.get("type"),
            "severity": normalized_failure.get("severity"),
            "retryable": normalized_failure.get("retryable"),
            "context_hash": normalized_failure.get("context_hash"),
            "recovery_status": recovery_result.get("status"),
            "strategy": recovery_result.get("strategy"),
            "recommendation": recovery_result.get("recommendation", "collect_more_signals"),
            "task_id": context.get("task_id"),
            "telemetry_ref": telemetry.get("timestamp"),
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
        """Select and apply recovery strategy using existing issue/recovery primitives."""
        context = context or {}
        target_agent_name = getattr(target_agent, "name", context.get("agent", "unknown_agent"))

        if target_agent is None:
            return {
                "status": "failed",
                "strategy": "none",
                "recommendation": "target_agent_missing",
            }

        if not self.policy.can_attempt(target_agent_name):
            return {
                "status": "failed",
                "strategy": "circuit_breaker",
                "recommendation": "defer_and_retry_later",
                "breaker": self.policy.breaker_status(target_agent_name),
            }

        error_type = (normalized_failure.get("type") or "").lower()
        error_message = (normalized_failure.get("message") or "").lower()
        error_info = {
            "error_type": normalized_failure.get("type", "UnknownError"),
            "error_message": normalized_failure.get("message", ""),
        }

        strategy_key = "runtime"
        if any(x in error_type or x in error_message for x in ["network", "connection", "http", "socket"]):
            strategy_key = "network"
        elif "timeout" in error_type or "timed out" in error_message:
            strategy_key = "timeout"
        elif any(x in error_type or x in error_message for x in ["memory", "outofmemory", "cuda"]):
            strategy_key = "memory"
        elif any(x in error_message for x in ["no module named", "cannot import name", "dll load failed"]):
            strategy_key = "dependency"
        elif any(x in error_message for x in ["resource", "gpu", "cpu", "busy"]):
            strategy_key = "resource"
        elif any(x in error_type for x in ["unicodeencodeerror", "unicodedecodeerror"]):
            strategy_key = "unicode"

        handler = self.recovery_strategies.get(strategy_key, handle_runtime_error)

        checkpoint_id = self.memory.save_checkpoint(
            label="pre_recovery",
            state={"task_data": task_data, "context": context, "target_agent_name": target_agent_name},
            metadata={"strategy": strategy_key, "failure_type": normalized_failure.get("type")},
        )

        retries = 0
        last_result = {"status": "failed", "reason": "unattempted"}

        while self.policy.retries_allowed(retries):
            result = handler(target_agent, task_data, error_info)
            last_result = result if isinstance(result, dict) else {"status": "recovered", "result": result}

            if not (isinstance(last_result, dict) and last_result.get("status") == "failed"):
                self.policy.record_success(target_agent_name)
                return {
                    "status": "recovered",
                    "strategy": strategy_key,
                    "result": result,
                    "checkpoint_id": checkpoint_id,
                    "attempts": retries + 1,
                }

            retries += 1

        # Fallback mode: if possible, enable lightweight mode and retry once.
        if hasattr(target_agent, "use_lightweight_mode"):
            try:
                target_agent.use_lightweight_mode(True)
                fallback = target_agent.perform_task(task_data)
                target_agent.use_lightweight_mode(False)
                self.policy.record_success(target_agent_name)
                return {
                    "status": "recovered",
                    "strategy": f"{strategy_key}+fallback_mode",
                    "result": fallback,
                    "checkpoint_id": checkpoint_id,
                    "attempts": retries + 1,
                }
            except Exception:
                try:
                    target_agent.use_lightweight_mode(False)
                except Exception:
                    pass

        restored_state = self.memory.restore_checkpoint(checkpoint_id)
        self.policy.record_failure(target_agent_name)
        return {
            "status": "failed",
            "strategy": strategy_key,
            "attempts": retries,
            "last_result": last_result,
            "checkpoint_restored": restored_state is not None,
            "checkpoint_id": checkpoint_id,
            "recommendation": "escalate_to_planning_or_evaluation",
        }


if __name__ == "__main__":
    print("\n=== Running Handler Agent ===\n")
    printer.status("TEST", "Starting Handler Agent tests", "info")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory

    memory = SharedMemory()
    factory = AgentFactory()
    execution_config = get_config_section('handler_agent')
    agent_type="handler"

    agent = HandlerAgent(
        shared_memory=memory,
        agent_factory=factory,
        config=execution_config
    )
    print("\n=== All tests completed successfully! ===\n")
