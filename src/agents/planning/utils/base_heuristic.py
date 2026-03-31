import numpy as np
import time
import json

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Heuristic Selector")
printer = PrettyPrinter

class BaseHeuristics(ABC):
    """Abstract base class for all planning heuristics."""

    @abstractmethod
    def predict_success_prob(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        method_stats: Dict[Tuple[str, str], Dict[str, int]],
        method_id: str
    ) -> float:
        """Predict success probability for a given method."""
        pass

    @abstractmethod
    def select_best_method(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        method_stats: Dict[Tuple[str, str], Dict[str, int]]
    ) -> Tuple[Optional[str], float]:
        """Select the best method from candidates."""
        pass

    # -------------------------------------------------------------------------
    # Common feature extraction helpers (thread‑safe, with error handling)
    # -------------------------------------------------------------------------
    def extract_base_features(self, task: Dict[str, Any], world_state: Dict[str, Any],
                              method_stats: Dict, method_id: str) -> Dict[str, float]:
        """Extract a set of common features. Can be overridden."""
        return {
            'task_depth': self._calculate_task_depth(task),
            'goal_overlap': self._calculate_goal_overlap(task, world_state),
            'method_failure_rate': self._calculate_method_failure_rate(task, method_stats, method_id),
            'state_diversity': self._calculate_state_diversity(world_state)
        }

    @staticmethod
    def _calculate_task_depth(task: Dict[str, Any], max_depth: int = 20) -> float:
        depth = 0
        current = task
        while current:
            if isinstance(current, dict):
                parent = current.get("parent")
            else:
                parent = getattr(current, "parent", None)
            if parent:
                depth += 1
                current = parent
            else:
                break
        return depth / max_depth  # Normalized

    @staticmethod
    def _calculate_goal_overlap(task: Dict[str, Any], world_state: Dict[str, Any]) -> float:
        goal_state = task.get("goal_state", {})
        if isinstance(goal_state, str):
            try:
                goal_state = json.loads(goal_state)
            except json.JSONDecodeError:
                logger.error("Failed to parse goal_state JSON")
                return 0.0
        if not isinstance(goal_state, dict):
            logger.error(f"goal_state not dict: {type(goal_state)}")
            return 0.0
        if not isinstance(world_state, dict):
            logger.error(f"world_state not dict: {type(world_state)}")
            return 0.0
        if not goal_state:
            return 0.0
        common = set(goal_state.keys()) & set(world_state.keys())
        return len(common) / len(goal_state)

    @staticmethod
    def _calculate_method_failure_rate(task: Dict[str, Any], method_stats: Dict,
                                       method_id: str) -> float:
        key = (task.get("name"), method_id)
        stats = method_stats.get(key, {'success': 1, 'total': 2})
        if stats['total'] == 0:
            return 1.0
        return 1 - (stats['success'] / stats['total'])

    @staticmethod
    def _calculate_state_diversity(world_state: Dict[str, Any]) -> float:
        values = [float(v) for v in world_state.values() if isinstance(v, (int, float))]
        return float(np.std(values)) if values else 0.0

    # Optional temporal features
    def extract_temporal_features(self, task: Dict[str, Any]) -> Dict[str, float]:
        return {
            'time_since_creation': self._time_since_creation(task),
            'deadline_proximity': self._deadline_proximity(task)
        }

    @staticmethod
    def _time_since_creation(task: Dict[str, Any]) -> float:
        creation = task.get("creation_time")
        if not creation:
            return 0.0
        if isinstance(creation, str):
            creation = datetime.fromisoformat(creation)
        return (datetime.now() - creation).total_seconds() / 3600.0  # hours

    @staticmethod
    def _deadline_proximity(task: Dict[str, Any]) -> float:
        creation = task.get("creation_time")
        deadline = task.get("deadline")
        if not creation or not deadline:
            return 0.0
        if isinstance(creation, str):
            creation = datetime.fromisoformat(creation)
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)
        total = (deadline - creation).total_seconds()
        if total <= 0:
            return 0.0
        elapsed = (datetime.now() - creation).total_seconds()
        return min(1.0, elapsed / total)


if __name__ == "__main__":
    print("\n=== Running Base Heuristics Test ===\n")
    printer.status("Init", "Base Heuristics initialized", "success")

    class DummyHeuristics(BaseHeuristics):
        def predict_success_prob(self, *args, **kwargs):
            return 0.5
    
        def select_best_method(self, *args, **kwargs):
            return None, 0.5
    
    base = DummyHeuristics()

    task = {}
    state = {}
    stats = {}
    id = "fewfef123"

    print(f"Base Features: {base.extract_base_features(task=task, world_state=state, method_stats=stats, method_id=id)}")
    print("\n=== Successfully Ran Base Heuristics ===\n")