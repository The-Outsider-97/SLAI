import time

from typing import Any, Dict, Optional

from src.agents.handler.utils.config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SLA Recovery Policy")
printer = PrettyPrinter


class SLARecoveryPolicy:
    """Evaluates SLA budget and constrains recovery attempts/mode."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = load_global_config()
        handler_cfg = get_config_section("handler_agent")
        if config:
            handler_cfg.update(config)

        self.default_recovery_budget_seconds = float(handler_cfg.get("default_recovery_budget_seconds", 30.0))
        self.default_max_attempts = int(handler_cfg.get("default_sla_max_attempts", 2))

        logger.info("SLA Recovery Policy initialized")

    def evaluate(self, context: Optional[Dict[str, Any]], normalized_failure: Dict[str, Any]) -> Dict[str, Any]:
        context = context or {}
        now = time.time()
        sla = context.get("sla", {}) if isinstance(context.get("sla", {}), dict) else {}

        deadline_ts = sla.get("deadline_ts")
        explicit_budget = sla.get("max_recovery_seconds")
        latency_budget_ms = sla.get("latency_budget_ms")

        if isinstance(deadline_ts, (int, float)):
            remaining_seconds = max(0.0, float(deadline_ts) - now)
        elif isinstance(explicit_budget, (int, float)):
            remaining_seconds = max(0.0, float(explicit_budget))
        elif isinstance(latency_budget_ms, (int, float)):
            remaining_seconds = max(0.0, float(latency_budget_ms) / 1000.0)
        else:
            remaining_seconds = self.default_recovery_budget_seconds

        severity = (normalized_failure.get("severity") or "low").lower()
        retryable = bool(normalized_failure.get("retryable"))

        if not retryable or remaining_seconds <= 0:
            recommended_attempts = 0
            mode = "degrade"
        elif remaining_seconds < 3:
            recommended_attempts = 1
            mode = "fast_failover"
        elif severity == "critical":
            recommended_attempts = 1
            mode = "conservative"
        else:
            recommended_attempts = self.default_max_attempts
            mode = "standard"

        return {
            "remaining_seconds": remaining_seconds,
            "recommended_attempts": max(0, recommended_attempts),
            "mode": mode,
            "can_retry": recommended_attempts > 0,
            "priority": context.get("priority", "normal"),
        }
