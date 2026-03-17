from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Router Strategy")
printer = PrettyPrinter

@dataclass
class RouterScoreWeights:
    success_rate: float = 1.0
    load_penalty: float = 0.25


class BaseRouterStrategy:
    name = "base"

    def rank_agents(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Dict[str, Dict[str, Any]],
        task_data: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        raise NotImplementedError


class WeightedRouterStrategy(BaseRouterStrategy):
    name = "weighted"

    def __init__(self, weights: RouterScoreWeights | None = None):
        self.weights = weights or RouterScoreWeights()

        logger.info("Weighted Router Strategy initialized")

    def rank_agents(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Dict[str, Dict[str, Any]],
        task_data: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        ranked: List[Tuple[str, Dict[str, Any], float]] = []
        for name, agent in agents.items():
            meta = stats.get(name, {})
            successes = float(meta.get("successes", 0.0))
            failures = float(meta.get("failures", 0.0))
            active_tasks = float(meta.get("active_tasks", 0.0))
            total = successes + failures
            success_rate = successes / total if total else 1.0
            score = (self.weights.success_rate * success_rate) - (self.weights.load_penalty * active_tasks)
            ranked.append((name, agent, score))
        return sorted(ranked, key=lambda row: row[2], reverse=True)


class LeastLoadedRouterStrategy(BaseRouterStrategy):
    name = "least_loaded"

    def rank_agents(
        self,
        agents: Dict[str, Dict[str, Any]],
        stats: Dict[str, Dict[str, Any]],
        task_data: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        ranked: List[Tuple[str, Dict[str, Any], float]] = []
        for name, agent in agents.items():
            active_tasks = float((stats.get(name, {}) or {}).get("active_tasks", 0.0))
            score = -active_tasks
            ranked.append((name, agent, score))
        return sorted(ranked, key=lambda row: row[2], reverse=True)


def build_router_strategy(name: str, config: Dict[str, Any] | None = None) -> BaseRouterStrategy:
    config = config or {}
    lowered = (name or "weighted").strip().lower()
    if lowered == LeastLoadedRouterStrategy.name:
        return LeastLoadedRouterStrategy()

    weights = RouterScoreWeights(
        success_rate=float(config.get("weight_success_rate", 1.0)),
        load_penalty=float(config.get("weight_load_penalty", 0.25)),
    )
    return WeightedRouterStrategy(weights=weights)
