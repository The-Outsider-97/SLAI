import time

from typing import Any, Dict, Optional

from .utils.config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Escalation Manager")
printer = PrettyPrinter


class EscalationManager:
    """Builds typed escalation payloads using severity/retryability matrix."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.matrix = config.get(
            "escalation_matrix",
            {
                "critical": {"retryable": "safety_agent", "non_retryable": "planning_agent"},
                "high": {"retryable": "planning_agent", "non_retryable": "evaluation_agent"},
                "medium": {"retryable": "handler_agent", "non_retryable": "planning_agent"},
                "low": {"retryable": "handler_agent", "non_retryable": "handler_agent"},
            },
        )

        logger.info("Escalation Manager initialized")

    def build_handoff_payload(
        self,
        normalized_failure: Dict[str, Any],
        recovery_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        strategy_distribution: Optional[Dict[str, float]] = None,
        sla: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        severity = normalized_failure.get("severity", "low")
        retryable = bool(normalized_failure.get("retryable"))
        retry_key = "retryable" if retryable else "non_retryable"

        escalate_to = self.matrix.get(severity, self.matrix.get("medium", {})).get(retry_key, "planning_agent")

        return {
            "handoff_type": "handler_escalation.v1",
            "timestamp": time.time(),
            "target_agent": escalate_to,
            "reason": recovery_result.get("recommendation", "recovery_exhausted"),
            "failure": {
                "type": normalized_failure.get("type"),
                "message": normalized_failure.get("message"),
                "severity": severity,
                "retryable": retryable,
                "context_hash": normalized_failure.get("context_hash"),
            },
            "recovery": {
                "status": recovery_result.get("status"),
                "strategy": recovery_result.get("strategy"),
                "attempts": recovery_result.get("attempts", 0),
            },
            "strategy_distribution": strategy_distribution or {},
            "sla": sla or {},
            "context": {
                "task_id": context.get("task_id"),
                "route": context.get("route"),
                "agent": context.get("agent"),
            },
        }
