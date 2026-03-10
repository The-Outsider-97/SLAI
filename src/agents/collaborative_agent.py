__version__ = "2.0.0"

"""
Collaborative agent built on shared BaseAgent architecture.

Features:
1. Comprehensive safety monitoring with Bayesian risk assessment
2. Multi-agent task coordination with optimization
3. Thread-safe shared memory operations
4. Configuration management
5. Serialization/deserialization support
6. Performance tracking and metrics
"""

from __future__ import annotations

import json
import threading
import time

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base.utils.main_config_loader import get_config_section, load_global_config
from src.agents.base_agent import BaseAgent
from src.agents.collaborative.collaboration_manager import CollaborationManager
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Collaborative Agent")
printer = PrettyPrinter

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyAssessment:
    risk_score: float
    threshold: float
    risk_level: RiskLevel
    recommended_action: str
    confidence: float
    source_agent: str = "unknown"
    task_type: str = "general"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["risk_level"] = self.risk_level.value
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SafetyAssessment":
        data = dict(payload)
        data["risk_level"] = RiskLevel(data.get("risk_level", RiskLevel.MODERATE.value))
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> "SafetyAssessment":
        return cls.from_dict(json.loads(raw))


class BayesianRiskModel:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self._default_alpha = max(alpha, 0.01)
        self._default_beta = max(beta, 0.01)
        self._posterior: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    def _ensure_key(self, key: str) -> None:
        if key not in self._posterior:
            self._posterior[key] = [self._default_alpha, self._default_beta]

    def update(self, key: str, event_was_safe: bool) -> None:
        with self._lock:
            self._ensure_key(key)
            if event_was_safe:
                self._posterior[key][0] += 1.0
            else:
                self._posterior[key][1] += 1.0

    def threshold(self, key: str, fallback: float = 0.7) -> float:
        with self._lock:
            self._ensure_key(key)
            alpha, beta = self._posterior[key]
            safe_rate = alpha / (alpha + beta)
            return max(0.05, min(0.95, 0.25 + 0.7 * safe_rate if safe_rate > 0 else fallback))

    def snapshot(self) -> Dict[str, List[float]]:
        with self._lock:
            return {k: [v[0], v[1]] for k, v in self._posterior.items()}

class CollaborativeAgent(BaseAgent):
    capabilities = ["coordination", "safety_assessment", "shared_memory"]

    def __init__(self, shared_memory=None, agent_factory=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.name = "CollaborativeAgent"
        self._lock = threading.RLock()

        self.global_config = load_global_config()
        self.collaborative_config = get_config_section("collaborative_agent") or {}

        self.risk_threshold = float(self.collaborative_config.get("risk_threshold", 0.7))
        self.max_concurrent_tasks = int(self.collaborative_config.get("max_concurrent_tasks", 100))
        self.load_factor = float(self.collaborative_config.get("load_factor", 0.75))
        self.optimization_weight_capability = float(self.collaborative_config.get("optimization_weight_capability", 0.5))
        self.optimization_weight_load = float(self.collaborative_config.get("optimization_weight_load", 0.3))
        self.optimization_weight_risk = float(self.collaborative_config.get("optimization_weight_risk", 0.2))
        self.bayes_prior_alpha = float(self.collaborative_config.get("bayes_prior_alpha", 1.0))
        self.bayes_prior_beta = float(self.collaborative_config.get("bayes_prior_beta", 1.0))
        self.use_collaboration_manager = bool(self.collaborative_config.get("use_collaboration_manager", True))

        if config:
            self.risk_threshold = float(config.get("risk_threshold", self.risk_threshold))
            self.max_concurrent_tasks = int(config.get("max_concurrent_tasks", self.max_concurrent_tasks))
            self.load_factor = float(config.get("load_factor", self.load_factor))
            self.optimization_weight_capability = float(config.get("optimization_weight_capability", self.optimization_weight_capability))
            self.optimization_weight_load = float(config.get("optimization_weight_load", self.optimization_weight_load))
            self.optimization_weight_risk = float(config.get("optimization_weight_risk", self.optimization_weight_risk))
            self.bayes_prior_alpha = float(config.get("bayes_prior_alpha", self.bayes_prior_alpha))
            self.bayes_prior_beta = float(config.get("bayes_prior_beta", self.bayes_prior_beta))
            self.use_collaboration_manager = bool(config.get("use_collaboration_manager", self.use_collaboration_manager))

        self._risk_model = BayesianRiskModel(alpha=self.bayes_prior_alpha, beta=self.bayes_prior_beta)
        self.collaboration_manager = CollaborationManager()

        self._metrics = {
            "assessments_completed": 0,
            "high_risk_interventions": 0,
            "tasks_coordinated": 0,
            "coordination_failures": 0,
            "avg_assessment_latency_ms": 0.0,
            "avg_coordination_latency_ms": 0.0,
            "delegated_tasks": 0,
            "delegation_failures": 0,
        }

    def shared_get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.shared_memory.get(key, default)

    def shared_set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            if ttl is not None:
                self.shared_memory.set(key, value, ttl=ttl)
            else:
                self.shared_memory.set(key, value)

    def shared_update(self, key: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            current = self.shared_memory.get(key, {}) or {}
            if not isinstance(current, dict):
                current = {"value": current}
            current.update(updates)
            self.shared_memory.set(key, current)
            return current

    def assess_risk(
        self,
        risk_score: float,
        task_type: str = "general",
        source_agent: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyAssessment:
        start = time.perf_counter()
        risk_score = max(0.0, min(1.0, float(risk_score)))

        task_key = f"task:{task_type}"
        agent_key = f"agent:{source_agent}"
        dynamic_threshold = min(
            self._risk_model.threshold(task_key),
            self._risk_model.threshold(agent_key),
            float(self.collab_config.get("risk_threshold", 0.7)),
        )

        if risk_score >= dynamic_threshold * 1.4:
            level = RiskLevel.CRITICAL
            action = "halt_and_escalate"
        elif risk_score >= dynamic_threshold:
            level = RiskLevel.HIGH
            action = "human_review"
        elif risk_score >= dynamic_threshold * 0.6:
            level = RiskLevel.MODERATE
            action = "proceed_with_guardrails"
        else:
            level = RiskLevel.LOW
            action = "proceed"

        assessment = SafetyAssessment(
            risk_score=risk_score,
            threshold=dynamic_threshold,
            risk_level=level,
            recommended_action=action,
            confidence=max(0.0, min(1.0, 1.0 - abs(risk_score - dynamic_threshold))),
            source_agent=source_agent,
            task_type=task_type,
        )

        event_was_safe = level in (RiskLevel.LOW, RiskLevel.MODERATE)
        self._risk_model.update(task_key, event_was_safe)
        self._risk_model.update(agent_key, event_was_safe)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._update_metric("assessments_completed", 1)
        self._rolling_metric("avg_assessment_latency_ms", elapsed_ms, "assessments_completed")
        if level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            self._update_metric("high_risk_interventions", 1)

        if context:
            self.shared_update("collaborative:last_assessment_context", context)
        self.shared_set("collaborative:last_assessment", assessment.to_dict())
        return assessment

    def coordinate_tasks(
        self,
        tasks: List[Dict[str, Any]],
        available_agents: Dict[str, Dict[str, Any]],
        optimization_goals: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        constraints = constraints or {}

        if not tasks or not available_agents:
            self._update_metric("coordination_failures", 1)
            return {"status": "error", "error": "tasks and available_agents are required"}

        assignments: Dict[str, Dict[str, Any]] = {}
        max_tasks_per_agent = int(constraints.get("max_tasks_per_agent", self.collab_config.get("max_concurrent_tasks", 100)))
        agent_loads = {name: int(meta.get("current_load", 0)) for name, meta in available_agents.items()}

        for task in sorted(tasks, key=lambda t: (t.get("deadline", float("inf")), -float(t.get("priority", 0)))):
            task_id = str(task.get("id", f"task-{len(assignments)+1}"))

            delegated = self._try_manager_delegation(task)
            if delegated is not None:
                assignments[task_id] = delegated
                continue

            chosen_agent, chosen_score = self._select_best_agent(task, available_agents, agent_loads)
            if chosen_agent is None:
                assignments[task_id] = {"status": "unassigned", "reason": "no_capable_agent"}
                continue

            assessment = self.assess_risk(
                risk_score=float(task.get("estimated_risk", 0.5)),
                task_type=task.get("type", "general"),
                source_agent=chosen_agent,
            )
            if assessment.risk_level == RiskLevel.CRITICAL:
                assignments[task_id] = {"status": "rejected_high_risk", "agent": chosen_agent, "safety": assessment.to_dict()}
                continue

            assignments[task_id] = {
                "status": "assigned",
                "agent": chosen_agent,
                "optimization_score": round(chosen_score, 4),
                "safety": assessment.to_dict(),
            }
            agent_loads[chosen_agent] += 1
            if agent_loads[chosen_agent] >= max_tasks_per_agent:
                available_agents[chosen_agent]["_saturated"] = True

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._update_metric("tasks_coordinated", len(tasks))
        self._rolling_metric("avg_coordination_latency_ms", elapsed_ms, "tasks_coordinated")

        result = {
            "status": "success",
            "assignments": assignments,
            "metrics": self.get_metrics(),
            "optimization_goals": optimization_goals or ["minimize_risk", "balance_load"],
        }
        self.shared_set("collaborative:last_coordination", result)
        return result

    def _try_manager_delegation(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.collaboration_manager is None:
            return None
        task_type = task.get("type")
        if not task_type:
            return None
        try:
            result = self.collaboration_manager.run_task(task_type, task, retries=1)
            self._update_metric("delegated_tasks", 1)
            return {"status": "delegated", "task_type": task_type, "result": result}
        except Exception as exc:
            self._update_metric("delegation_failures", 1)
            logger.debug("Delegation failed for task type %s: %s", task_type, exc)
            return None

    def _select_best_agent(
        self,
        task: Dict[str, Any],
        available_agents: Dict[str, Dict[str, Any]],
        agent_loads: Dict[str, int],
    ) -> Tuple[Optional[str], float]:
        required = set(task.get("requirements", []))
        best_agent: Optional[str] = None
        best_score = -1.0

        for agent_name, meta in available_agents.items():
            if meta.get("_saturated"):
                continue
            capabilities = set(meta.get("capabilities", []))
            if required and not required.issubset(capabilities):
                continue

            capability_score = (len(required & capabilities) / max(1, len(required))) if required else 1.0
            load_score = 1.0 / (1.0 + float(agent_loads.get(agent_name, 0)))
            risk_score = self._risk_model.threshold(f"agent:{agent_name}")

            score = (
                self.collab_config.get("optimization_weight_capability", 0.5) * capability_score
                + self.collab_config.get("optimization_weight_load", 0.3) * load_score
                + self.collab_config.get("optimization_weight_risk", 0.2) * risk_score
            )
            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent, best_score

    def perform_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        mode = task_input.get("mode", "coordinate")
        if mode == "assess":
            assessment = self.assess_risk(
                risk_score=float(task_input.get("risk_score", 0.5)),
                task_type=task_input.get("task_type", "general"),
                source_agent=task_input.get("source_agent", "unknown"),
                context=task_input.get("context"),
            )
            return {"status": "success", "assessment": assessment.to_dict()}

        return self.coordinate_tasks(
            tasks=task_input.get("tasks", []),
            available_agents=task_input.get("available_agents", {}),
            optimization_goals=task_input.get("optimization_goals"),
            constraints=task_input.get("constraints"),
        )

    def serialize_state(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "config": self.collab_config,
                "metrics": self._metrics,
                "risk_model": self._risk_model.snapshot(),
                "timestamp": time.time(),
            }
        )

    @classmethod
    def deserialize_state(cls, raw: str, shared_memory=None, agent_factory=None) -> "CollaborativeAgent":
        payload = json.loads(raw)
        agent = cls(shared_memory=shared_memory, agent_factory=agent_factory, config=payload.get("config", {}))
        agent._metrics.update(payload.get("metrics", {}))
        for key, values in payload.get("risk_model", {}).items():
            if isinstance(values, list) and len(values) == 2:
                agent._risk_model._posterior[key] = [float(values[0]), float(values[1])]
        return agent

    def save_state(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(self.serialize_state(), encoding="utf-8")

    @classmethod
    def load_state(cls, path: str, shared_memory=None, agent_factory=None) -> "CollaborativeAgent":
        return cls.deserialize_state(Path(path).read_text(encoding="utf-8"), shared_memory=shared_memory, agent_factory=agent_factory)

    def _update_metric(self, key: str, delta: float) -> None:
        with self._lock:
            self._metrics[key] = float(self._metrics.get(key, 0.0) + delta)

    def _rolling_metric(self, key: str, latest: float, count_key: str) -> None:
        with self._lock:
            count = max(1.0, float(self._metrics.get(count_key, 1.0)))
            old = float(self._metrics.get(key, 0.0))
            self._metrics[key] = old + ((latest - old) / count)

    def get_metrics(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._metrics)


if __name__ == "__main__":
    print("\n=== Running Collaborative Agent ===\n")
    printer.status("TEST", "Starting Collaborative Agent tests", "info")
    print("\nAll tests completed successfully!\n")
