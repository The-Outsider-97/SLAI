from typing import Any, Dict, List, Optional

from src.agents.handler.utils.config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Probabilistic Strategy Selector")
printer = PrettyPrinter

class ProbabilisticStrategySelector:
    """Chooses recovery strategy using priors plus empirical success rates."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = load_global_config()
        handler_cfg = get_config_section("handler_agent")
        if config:
            handler_cfg.update(config)

        self.priors = handler_cfg.get(
            "strategy_priors",
            {
                "network": 0.20,
                "timeout": 0.20,
                "memory": 0.15,
                "runtime": 0.25,
                "dependency": 0.10,
                "resource": 0.07,
                "unicode": 0.03,
            },
        )

        logger.info("Probabilistic Strategy Selector initialized")

    def select(
        self,
        normalized_failure: Dict[str, Any],
        telemetry_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        telemetry_history = telemetry_history or []
        candidates = self._infer_candidates(normalized_failure)

        scores: Dict[str, float] = {}
        for strategy in candidates:
            prior = float(self.priors.get(strategy, 0.01))
            success_rate = self._strategy_success_rate(strategy=strategy, telemetry_history=telemetry_history)
            scores[strategy] = max(0.001, (0.65 * prior) + (0.35 * success_rate))

        norm = sum(scores.values()) or 1.0
        distribution = {k: (v / norm) for k, v in scores.items()}
        selected = max(distribution.items(), key=lambda x: x[1])[0]

        return {
            "selected_strategy": selected,
            "distribution": distribution,
            "candidates": candidates,
        }

    @staticmethod
    def _strategy_success_rate(strategy: str, telemetry_history: List[Dict[str, Any]]) -> float:
        matched = []
        for event in telemetry_history:
            recovery = event.get("recovery", {}) if isinstance(event, dict) else {}
            if recovery.get("strategy", "").split("+")[0] == strategy:
                matched.append(event)

        if not matched:
            return 0.5

        recovered = 0
        for event in matched:
            recovery = event.get("recovery", {}) if isinstance(event, dict) else {}
            if recovery.get("status") == "recovered":
                recovered += 1

        return recovered / len(matched)

    @staticmethod
    def _infer_candidates(normalized_failure: Dict[str, Any]) -> List[str]:
        error_type = (normalized_failure.get("type") or "").lower()
        error_message = (normalized_failure.get("message") or "").lower()

        candidates = ["runtime"]

        if any(x in error_type or x in error_message for x in ["network", "connection", "http", "socket"]):
            candidates.append("network")
        if "timeout" in error_type or "timed out" in error_message:
            candidates.append("timeout")
        if any(x in error_type or x in error_message for x in ["memory", "outofmemory", "cuda"]):
            candidates.append("memory")
        if any(x in error_message for x in ["no module named", "cannot import name", "dll load failed"]):
            candidates.append("dependency")
        if any(x in error_message for x in ["resource", "gpu", "cpu", "busy"]):
            candidates.append("resource")
        if any(x in error_type for x in ["unicodeencodeerror", "unicodedecodeerror"]):
            candidates.append("unicode")

        # keep order + uniqueness
        return list(dict.fromkeys(candidates))
