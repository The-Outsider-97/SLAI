import os
import time
import joblib
import numpy as np

from typing import Dict, Any, Tuple, Optional

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.decision_tree_heuristic import DecisionTreeHeuristic
from src.agents.planning.gradient_boosting_heuristic import GradientBoostingHeuristic
from src.agents.planning.reinforcement_learning_heuristic import ReinforcementLearningHeuristic
from src.agents.planning.planning_memory import PlanningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Heuristic Selector")
printer = PrettyPrinter

class HeuristicSelector:
    def __init__(self):
        self.config = load_global_config()

        self.selector_config = get_config_section('heuristic_selector')
        self.performance_log_path = self.selector_config.get('performance_log_path')
        self.speed_weight = self.selector_config.get('speed_weight')
        self.accuracy_weight = self.selector_config.get('accuracy_weight')
        self.speed_threshold = self.selector_config.get('speed_threshold')
        self.max_dt_depth = self.selector_config.get('max_dt_depth')
        self.min_rl_sequence_length = self.selector_config.get('min_rl_sequence_length')
        self.heuristic_priority = self.selector_config.get('heuristic_priority')
        self.time_budget = self.selector_config.get('time_budget')

        self.memory = PlanningMemory()

        self.heuristics = {
            "DT": DecisionTreeHeuristic(),
            "GB": GradientBoostingHeuristic(),
            "RL": ReinforcementLearningHeuristic()
        }
        self.heuristic_performance = {key: {"speed": 0.0, "accuracy": 0.5} for key in self.heuristics}
        self.last_used = {key: 0 for key in self.heuristics}
        # Load state from memory
        self.load_state()
        self.load_performance_stats()

        logger.info(f"Heuristic Selector initialized with {len(self.heuristics)} heuristics")

    def load_state(self):
        """Load state from memory or initialize defaults"""
        state = self.memory.base_state.get('heuristic_selector', {})
        
        self.heuristic_performance = state.get(
            'performance', 
            {key: {"speed": 0.0, "accuracy": 0.5} for key in self.heuristics}
        )
        
        self.last_used = state.get(
            'last_used', 
            {key: 0 for key in self.heuristics}
        )
        
        # Initialize heuristics with memory reference
        self.heuristics = {
            "DT": DecisionTreeHeuristic(),
            "GB": GradientBoostingHeuristic(),
            "RL": ReinforcementLearningHeuristic()
        }

    def save_state(self):
        """Persist current state to memory"""
        self.memory.base_state['heuristic_selector'] = {
            'performance': self.heuristic_performance,
            'last_used': self.last_used
        }
        # Create checkpoint if agent is attached
        if hasattr(self.memory, 'agent') and self.memory.agent:
            self.memory.save_checkpoint(label="heuristic_update")

    def load_performance_stats(self):
        """Load historical performance metrics if available"""
        try:
            if os.path.exists(self.selector_config.get('performance_log_path')):
                self.heuristic_performance = joblib.load(self.selector_config.get('performance_log_path'))
        except Exception as e:
            print(f"Failed to load performance stats: {str(e)}")
    
    def save_performance_stats(self):
        """Save current performance metrics"""
        try:
            joblib.dump(self.heuristic_performance, self.selector_config.get('performance_log_path'))
        except Exception as e:
            print(f"Failed to save performance stats: {str(e)}")
    
    def update_performance(self, heuristic_name: str, speed: float, accuracy: float):
        """Update heuristic performance metrics with exponential smoothing"""
        printer.status("INIT", "Updating performance", "info")

        old_speed = self.heuristic_performance[heuristic_name]["speed"]
        old_accuracy = self.heuristic_performance[heuristic_name]["accuracy"]
        
        # Update with 30% weight to new measurement
        self.heuristic_performance[heuristic_name]["speed"] = 0.7 * old_speed + 0.3 * speed
        self.heuristic_performance[heuristic_name]["accuracy"] = 0.7 * old_accuracy + 0.3 * accuracy
        
        self.save_performance_stats()
        self.save_state() 
    
    def select_heuristic(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: list,
        time_budget= 0.5
    ) -> Tuple[str, Any]:
        """Select the best heuristic based on task characteristics and constraints"""
        printer.status("INIT", "Selecting heuristic", "info")

        if self.memory.is_sequential_task(task, self.selector_config.get('min_rl_sequence_length')):
            if self._heuristic_available("RL", time_budget):
                return "RL", self.heuristics["RL"]

        # Check for sequential task pattern (RL preferred)
        if self._is_sequential_task(task):
            if self._heuristic_available("RL", time_budget):
                return "RL", self.heuristics["RL"]
        
        # Check for deep task hierarchy (DT preferred)
        task_depth = self._calculate_task_depth(task)
        if task_depth > self.selector_config.get('max_dt_depth'):
            if self._heuristic_available("DT", time_budget):
                return "DT", self.heuristics["DT"]
        
        # Check for resource-constrained environment (GB preferred)
        if "cpu_available" in world_state and world_state["cpu_available"] < 0.3:
            if self._heuristic_available("GB", time_budget):
                return "GB", self.heuristics["GB"]
        
        # Fallback to performance-based selection
        return self._select_by_performance(time_budget)
    
    def _is_sequential_task(self, task: Dict[str, Any]) -> bool:
        """Determine if task is part of a sequence"""
        printer.status("INIT", "Determine task sequence", "info")

        current = task
        sequence_length = 1
        
        # Traverse up through parent tasks
        while "parent" in current and current["parent"] is not None:
            sequence_length += 1
            current = current["parent"]
            if sequence_length >= self.selector_config.get('min_rl_sequence_length'):
                return True
        
        return False
    
    def _calculate_task_depth(self, task: Dict[str, Any]) -> int:
        """Calculate depth of task hierarchy"""
        depth = 0
        current = task
        while "parent" in current and current["parent"] is not None:
            depth += 1
            current = current["parent"]
        return depth
    
    def _heuristic_available(self, heuristic_name: str, time_budget: float) -> bool:
        """Check if heuristic is trained and meets time constraints"""
        heuristic = self.heuristics[heuristic_name]
        avg_speed = self.heuristic_performance[heuristic_name]["speed"]
        
        # Check if trained and meets time budget with 20% buffer
        return heuristic.trained and (avg_speed * 1.2 < time_budget)
    
    def _select_by_performance(self, time_budget: float) -> Tuple[str, Any]:
        """Select heuristic based on performance metrics"""
        candidates = []
        
        for name, heuristic in self.heuristics.items():
            if not heuristic.trained:
                continue
                
            perf = self.heuristic_performance[name]
            speed_score = max(0, 1 - (perf["speed"] / time_budget)) if time_budget > 0 else 1
            accuracy_score = perf["accuracy"]
            
            # Calculate weighted score
            score = (
                self.selector_config.get('accuracy_weight') * accuracy_score +
                self.selector_config.get('speed_weight') * speed_score
            )
            candidates.append((score, name, heuristic))
        
        if not candidates:
            # Fallback to DT with default prediction
            return "DT", self.heuristics["DT"]
        
        # Return heuristic with highest score
        candidates.sort(reverse=True)
        return candidates[0][1], candidates[0][2]
    
    def predict_success_prob(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        method_stats: Dict[Tuple[str, str], Dict[str, int]],
        method_id: str,
        time_budget: float = 0.5
    ) -> float:
        """Predict success probability using the best heuristic"""
        start_time = time.time()
        
        # Select appropriate heuristic
        heuristic_name, heuristic = self.select_heuristic(
            task, world_state, [method_id], time_budget
        )
        
        # Make prediction
        if heuristic_name == "RL":
            prob = heuristic.predict_success_prob(task, world_state, method_stats, method_id)
        else:
            # Save current method and set to target method
            original_method = task.get("selected_method")
            task["selected_method"] = method_id
            
            # Get prediction
            prob = heuristic.predict_success_prob(task, world_state, method_stats)
            
            # Restore original method
            if original_method:
                task["selected_method"] = original_method
            else:
                del task["selected_method"]
        
        # Update performance metrics
        elapsed = time.time() - start_time
        # Accuracy would require knowing the actual outcome
        # In real use, this would be updated after task execution
        self.update_performance(heuristic_name, elapsed, 0.5)
        
        return prob
    
    def select_best_method(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: list,
        method_stats: Dict[Tuple[str, str], Dict[str, int]],
        time_budget: float = 0.5
    ) -> Tuple[Optional[str], float]:
        """Select the best method using the optimal heuristic"""
        start_time = time.time()
        
        # Select appropriate heuristic
        heuristic_name, heuristic = self.select_heuristic(
            task, world_state, candidate_methods, time_budget
        )
        
        # Select best method
        if heuristic_name == "RL":
            method, prob = heuristic.select_method(task, world_state, candidate_methods, method_stats)
        elif heuristic_name == "DT":
            method, prob = heuristic.select_best_method(task, world_state, candidate_methods, method_stats)
        else:  # GB
            # GB doesn't take method_stats in select_best_method in the provided code
            # This is a workaround for the interface inconsistency
            method, prob = heuristic.select_best_method(task, world_state, candidate_methods)
        
        # Update performance metrics
        elapsed = time.time() - start_time
        # Accuracy would be updated after execution based on actual outcome
        self.update_performance(heuristic_name, elapsed, 0.5)
        
        return method, prob

if __name__ == "__main__":
    print("\n=== Testing Heuristic Selector ===\n")
    selector = HeuristicSelector()
    
    # Create test task
    task = {
        "name": "sequential_navigation",
        "priority": 0.9,
        "goal_state": {"position": "target"},
        "parent": {
            "name": "parent_task",
            "parent": {
                "name": "root_task",
                "parent": None
            }
        },
        "creation_time": "2023-10-05T12:00:00",
        "deadline": "2023-10-05T12:30:00"
    }
    
    world_state = {
        "position": "start",
        "cpu_available": 0.4,
        "memory_available": 0.8,
        "battery_level": 0.9
    }
    
    method_stats = {
        ("navigation", "A*"): {"success": 15, "total": 20},
        ("navigation", "RRT"): {"success": 12, "total": 18},
        ("navigation", "D*"): {"success": 8, "total": 15}
    }
    
    candidate_methods = ["A*", "RRT", "D*"]
    
    print("\nTest 1: Sequential task selection (should prefer RL)")
    heuristic_name, _ = selector.select_heuristic(task, world_state, candidate_methods)
    print(f"Selected heuristic: {heuristic_name}")
    
    print("\nTest 2: Method selection")
    best_method, confidence = selector.select_best_method(
        task, world_state, candidate_methods, method_stats
    )
    print(f"Recommended method: {best_method} (confidence: {confidence:.2f})")
    
    print("\nTest 3: Success probability prediction")
    for method in candidate_methods:
        prob = selector.predict_success_prob(
            task, world_state, method_stats, method
        )
        print(f"Success probability for {method}: {prob:.2f}")
    
    print("\n=== Heuristic Selector Tests Complete ===\n")
