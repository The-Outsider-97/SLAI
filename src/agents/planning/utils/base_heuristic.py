
import numpy as np
import time
import json

from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Heuristic Selector")
printer = PrettyPrinter

class BaseHeuristics:
    def predict_success_prob(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        method_stats: Dict[Tuple[str, str], Dict[str, int]],
        method_id: str
    ) -> float:
        raise NotImplementedError("Subclasses must implement predict_success_prob")
    
    def select_best_method(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        method_stats: Dict[Tuple[str, str], Dict[str, int]]
    ) -> Tuple[Optional[str], float]:
        raise NotImplementedError("Subclasses must implement select_best_method")

    def extract_base_features(self, task, world_state, method_stats, method_id: str) -> dict:
        return {
            'task_depth': self._calculate_task_depth(task),
            'goal_overlap': self._calculate_goal_overlap(task, world_state),
            'method_failure_rate': self._calculate_method_failure_rate(
                task, method_stats, method_id
            ),
            'state_diversity': self._calculate_state_diversity(world_state)
        }

    @staticmethod
    def _calculate_task_depth(task, max_depth=20):
        printer.status("BASE", "Task depth calculation succesfully initialized", "info")

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
        return depth / max_depth  # Normalized value

    @staticmethod
    def _calculate_goal_overlap(task, world_state):
        printer.status("BASE", "Goal overlap calculation succesfully initialized", "info")

        goal_state = task.get("goal_state", {})
        
        # Parse JSON string if needed
        if isinstance(goal_state, str):
            try:
                goal_state = json.loads(goal_state)
            except json.JSONDecodeError:
                logger.error("Failed to parse goal_state JSON string")
                return 0.0
                
        # Validate goal_state type
        if not isinstance(goal_state, dict):
            logger.error(f"Invalid goal_state type: expected dict, got {type(goal_state)}")
            return 0.0
        
        # Validate world_state type
        if not isinstance(world_state, dict):
            logger.error(f"Invalid world_state type: expected dict, got {type(world_state)}")
            return 0.0
            
        if not goal_state:
            return 0.0
            
        common_keys = set(goal_state.keys()) & set(world_state.keys())
        return len(common_keys) / len(goal_state)

    @staticmethod
    def _calculate_method_failure_rate(task, method_stats):
        printer.status("BASE", "Failure rate calculation succesfully initialized", "info")
    
        key = (task.get("name"), task.get("selected_method"))
        stats = method_stats.get(key, {'success': 1, 'total': 2})
        return 1 - (stats['success'] / stats['total']) if stats['total'] > 0 else 1.0

    @staticmethod
    def _calculate_state_diversity(world_state):
        printer.status("BASE", "State diversity calculation succesfully initialized", "info")

        state_vals = [float(v) for v in world_state.values() 
                     if isinstance(v, (int, float))]
        return np.std(state_vals) if state_vals else 0

    def extract_temporal_features(self, task) -> dict:
        return {
            'time_since_creation': self._time_since_creation(task),
            'deadline_proximity': self._deadline_proximity(task)
        }

    @staticmethod
    def _time_since_creation(task):
        creation_time = task.get("creation_time")
        if not creation_time: 
            return 0.0
        if isinstance(creation_time, str):
            creation_time = datetime.fromisoformat(creation_time)
        return (datetime.now() - creation_time).total_seconds() / 3600  # Hours

    @staticmethod
    def _deadline_proximity(task):
        printer.status("BASE", "Deadline proximation", "info")

        creation_time = task.get("creation_time")
        deadline = task.get("deadline")
        if not creation_time or not deadline: 
            return 0.0
        if isinstance(creation_time, str):
            creation_time = datetime.fromisoformat(creation_time)
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)
        total_time = (deadline - creation_time).total_seconds()
        elapsed = (datetime.now() - creation_time).total_seconds()
        return elapsed / total_time if total_time > 0 else 0.0
