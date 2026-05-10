from typing import Any, Dict, List, Optional

from .utils.config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Adaptive Retry Policy")
printer = PrettyPrinter()


class AdaptiveRetryPolicy:
    """Adaptive retry policy tuned by fingerprint-level historical outcomes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = load_global_config()
        policy_cfg = get_config_section("policy")
        handler_cfg = get_config_section("handler_agent")

        merged = {}
        merged.update(policy_cfg)
        merged.update(handler_cfg)
        if config:
            merged.update(config)

        self.base_max_retries = int(merged.get("max_retries", 2))
        self.min_retries = int(merged.get("adaptive_retry_min", 0))
        self.max_retries = int(merged.get("adaptive_retry_max", max(self.base_max_retries, 4)))
        self.min_samples = int(merged.get("adaptive_retry_min_samples", 3))

        logger.info("Adaptive Retry Policy initialized")

    def retries_for_fingerprint(
        self,
        fingerprint: str,
        severity: str,
        retryable: bool,
        telemetry_history: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        if not retryable:
            return 0

        telemetry_history = telemetry_history or []
        stats = self._fingerprint_stats(fingerprint=fingerprint, telemetry_history=telemetry_history)

        retries = self.base_max_retries

        if severity == "critical":
            retries = max(self.min_retries, retries - 1)
        elif severity == "low":
            retries = min(self.max_retries, retries + 1)

        total = stats["total"]
        success_rate = stats["success_rate"]

        if total >= self.min_samples:
            if success_rate >= 0.70:
                retries = min(self.max_retries, retries + 1)
            elif success_rate <= 0.20:
                retries = max(self.min_retries, retries - 1)

        return max(self.min_retries, min(self.max_retries, retries))

    @staticmethod
    def _fingerprint_stats(fingerprint: str, telemetry_history: List[Dict[str, Any]]) -> Dict[str, float | int]:
        matched = []
        for event in telemetry_history:
            failure = event.get("failure", {}) if isinstance(event, dict) else {}
            if failure.get("context_hash") == fingerprint:
                matched.append(event)

        total = len(matched)
        recovered = 0
        for event in matched:
            recovery = event.get("recovery", {}) if isinstance(event, dict) else {}
            if recovery.get("status") == "recovered":
                recovered += 1

        success_rate = (recovered / total) if total else 0.0
        return {"total": total, "recovered": recovered, "success_rate": success_rate}
