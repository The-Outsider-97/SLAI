
import time
import random

from typing import Dict, Any, Optional, Tuple

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.utils.execution_error import ExecutionError, ActionFailureError
from src.agents.execution.execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Recovery")
printer = PrettyPrinter

class ExecutionRecovery:
    """
    A robust execution recovery system that intelligently handles failures and adapts plans.
    """
    def __init__(self):
        self.config = load_global_config()
        self.max_retries = self.config.get('max_retries')

        self.recovery_config = get_config_section("execution_recovery")
        self.max_history = self.recovery_config.get("max_failure_history")
        self.recovery_idle_time = self.recovery_config.get("recovery_idle_time")
        self.sensitive_keys = self.recovery_config.get("sensitive_keys", [])

        self.memory = ExecutionMemory()
        self.task_coordinator = None
        
        # Recovery strategy registry
        self.recovery_strategies = {
            "move_to": self._handle_move_failure,
            "pick_object": self._handle_pick_failure,
            "idle": self._handle_general_failure,
            "default": self._handle_general_failure
        }
        
        # Failure tracking for learning
        self.failure_history = []
        
        logger.info("Execution Recovery initialized")

    def handle_failure(self, action_name: str, error: Exception, context: Dict) -> Tuple[bool, Dict]:
        """
        Main failure handling entry point. Determines recovery strategy and
        returns modified context for retry.
        """
        printer.status("RECOVERY", f"Handling {action_name} failure", "warning")
        
        # Log failure for future learning
        self._log_failure(action_name, error, context)
        
        # Get specialized handler or use default
        handler = self.recovery_strategies.get(action_name, self.recovery_strategies["default"])
        recovery_success, modified_context = handler(action_name, error, context)
        
        if recovery_success:
            logger.info(f"Recovery strategy for {action_name} succeeded")
        else:
            logger.error(f"All recovery attempts for {action_name} failed")
        
        return recovery_success, modified_context

    def _handle_move_failure(self, action_name: str, error: Exception, context: Dict) -> Tuple[bool, Dict]:
        """Specialized recovery for movement failures"""
        printer.status("RECOVERY", "Movement failure recovery", "warning")
        
        # Strategy 1: Increase pathfinding tolerance
        modified_context = context.copy()
        modified_context["replan_threshold"] = context.get("replan_threshold", 0.05) * 1.5
        modified_context["avoidance_radius"] = context.get("avoidance_radius", 1.0) * 1.2
        
        logger.warning(f"Retrying movement with relaxed parameters: "
                      f"replan_threshold={modified_context['replan_threshold']:.2f}, "
                      f"avoidance_radius={modified_context['avoidance_radius']:.2f}")
        
        # Strategy 2: Fallback to simplified pathfinding
        if isinstance(error, ActionFailureError) and "No path found" in str(error):
            modified_context["path_simplify"] = True
            logger.warning("Using simplified pathfinding algorithm")
        
        # Strategy 3: Backtrack to last checkpoint
        if "path_blocked" in str(error).lower():
            return self._backtrack_to_checkpoint("pre_movement", context)
        
        return True, modified_context

    def _handle_pick_failure(self, action_name: str, error: Exception, context: Dict) -> Tuple[bool, Dict]:
        """Specialized recovery for object picking failures"""
        printer.status("RECOVERY", "Pick failure recovery", "warning")
        
        modified_context = context.copy()
        
        # Strategy 1: Adjust position and retry
        if "too far" in str(error).lower():
            modified_context["approach_distance"] = context.get("min_distance", 0.5) * 0.8
            logger.warning(f"Retrying pickup from closer distance: {modified_context['approach_distance']:.2f}")
            return True, modified_context
        
        # Strategy 2: Use alternative grasp method
        if "grasp failed" in str(error).lower():
            modified_context["grasp_strategy"] = "alternative"
            modified_context["grasp_time"] = context.get("grasp_time", 1.0) * 1.3
            logger.warning("Using alternative grasp strategy with extended time")
            return True, modified_context
        
        # Strategy 3: Backtrack to pre-pick state
        return self._backtrack_to_checkpoint("pre_pick", context)

    def _handle_general_failure(self, action_name: str, error: Exception, context: Dict) -> Tuple[bool, Dict]:
        """Generic failure recovery strategies"""
        printer.status("RECOVERY", "General failure recovery", "warning")
        
        # Strategy 1: Simple retry with delay
        modified_context = context.copy()
        modified_context["retry_count"] = context.get("retry_count", 0) + 1
        
        if modified_context["retry_count"] <= self.max_retries:
            delay = random.uniform(0.5, 2.0)
            logger.warning(f"Retry #{modified_context['retry_count']} after {delay:.1f}s delay")
            time.sleep(delay)
            return True, modified_context
        
        # Strategy 2: Fallback to idle to recover energy
        if "energy" in str(error).lower():
            modified_context["force_idle"] = True
            modified_context["idle_duration"] = self.recovery_idle_time
            logger.warning("Insufficient energy - falling back to idle recovery")
            return True, modified_context
        
        # Strategy 3: Backtrack to last known good state
        return self._backtrack_to_checkpoint("pre_action", context)

    def _backtrack_to_checkpoint(self, checkpoint_type: str, context: Dict) -> Tuple[bool, Dict]:
        """Restore state from a previously saved checkpoint"""
        printer.status("RECOVERY", f"Backtracking to {checkpoint_type} checkpoint", "warning")
        
        # Find most recent relevant checkpoint
        checkpoints = self.memory.find_checkpoints(tag=checkpoint_type, max_age=300)
        if not checkpoints:
            logger.error(f"No {checkpoint_type} checkpoints available for recovery")
            return False, context
        
        # Restore from latest checkpoint
        latest_checkpoint = checkpoints[0]
        restored_state = self.memory.restore_checkpoint(latest_checkpoint['id'])
        
        if not restored_state:
            logger.error(f"Failed to restore from checkpoint {latest_checkpoint['id']}")
            return False, context
        
        logger.info(f"Restored state from checkpoint {latest_checkpoint['id']} "
                   f"created at {latest_checkpoint['created']}")
        
        # Update context with restored state
        modified_context = context.copy()
        modified_context.update(restored_state)
        modified_context["recovery_mode"] = True
        modified_context["recovery_source"] = latest_checkpoint['id']
        
        # Add failure to history for learning
        self.failure_history.append({
            "action": "backtrack",
            "checkpoint": latest_checkpoint['id'],
            "timestamp": time.time()
        })
        
        return True, modified_context

    def _log_failure(self, action_name: str, error: Exception, context: Dict):
        """Record failure details for analysis and learning"""
        failure_record = {
            "timestamp": time.time(),
            "action": action_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context_snapshot": {
                k: v for k, v in context.items() 
                if k not in self.recovery_config.get("sensitive_keys", ["password", "token"])
            },
            "recovery_attempted": False
        }
        
        # Maintain fixed-size history
        if len(self.failure_history) >= self.max_history:
            self.failure_history.pop(0)
            
        self.failure_history.append(failure_record)
        
        logger.debug(f"Logged failure for {action_name}: {type(error).__name__}")

    def analyze_failure_patterns(self):
        """Analyze historical failures to improve recovery strategies"""
        printer.status("RECOVERY", "Analyzing failure patterns", "info")
        
        if not self.failure_history:
            return {"status": "no_data"}
        
        # Simple pattern detection (can be enhanced with ML)
        action_failures = {}
        common_errors = {}
        
        for record in self.failure_history:
            action = record["action"]
            error_type = record["error_type"]
            
            action_failures[action] = action_failures.get(action, 0) + 1
            common_errors[error_type] = common_errors.get(error_type, 0) + 1
        
        # Identify most frequent failure per action
        action_error_map = {}
        for record in self.failure_history:
            action = record["action"]
            error_type = record["error_type"]
            if action not in action_error_map:
                action_error_map[action] = {}
            action_error_map[action][error_type] = action_error_map[action].get(error_type, 0) + 1
        
        # Generate recommendations
        recommendations = []
        for action, errors in action_error_map.items():
            max_error = max(errors.items(), key=lambda x: x[1])
            recommendations.append({
                "action": action,
                "most_common_error": max_error[0],
                "count": max_error[1],
                "suggestion": self._generate_error_suggestion(action, max_error[0])
            })
        
        return {
            "total_failures": len(self.failure_history),
            "actions_by_failure": action_failures,
            "common_errors": common_errors,
            "recommendations": recommendations
        }

    def _generate_error_suggestion(self, action: str, error_type: str) -> str:
        """Generate improvement suggestions based on error patterns"""
        suggestions = {
            "move_to": {
                "ActionFailureError": "Implement obstacle mapping and dynamic path recalibration",
                "TimeoutError": "Increase timeout thresholds or optimize pathfinding algorithm"
            },
            "pick_object": {
                "ActionFailureError": "Implement grasp quality assessment and tool assistance",
                "InvalidContextError": "Enhance perception system for better object detection"
            },
            "idle": {
                "ActionInterruptionError": "Implement protected rest periods for critical recovery"
            }
        }
        
        default_suggestions = {
            "ActionFailureError": "Review action implementation and add additional validation checks",
            "TimeoutError": "Increase time allowances or optimize action execution",
            "InvalidContextError": "Improve context gathering and validation mechanisms",
            "default": "Conduct root cause analysis and implement targeted improvements"
        }
        
        # Get action-specific or default suggestion
        action_suggestions = suggestions.get(action, {})
        return action_suggestions.get(error_type, default_suggestions.get(error_type, default_suggestions["default"]))

    def create_recovery_checkpoint(self, checkpoint_type: str, state: Dict):
        """Create a recovery point with expiration time"""
        printer.status("RECOVERY", f"Creating {checkpoint_type} checkpoint", "info")
        
        # Filter out large state elements
        filtered_state = {
            k: v for k, v in state.items() 
            if k not in ["map_data", "object_state"]
        }
        
        tags = [checkpoint_type, "recovery_point"]
        metadata = {
            "agent_state": filtered_state,
            "timestamp": time.time()
        }
        
        # Create checkpoint
        checkpoint_id = self.memory.create_checkpoint(
            filtered_state, 
            tags=tags,
            metadata=metadata
        )
        
        logger.info(f"Created recovery checkpoint {checkpoint_id} for {checkpoint_type}")
        return checkpoint_id

    def get_recovery_report(self) -> Dict:
        """Generate a recovery readiness report"""
        printer.status("RECOVERY", "Generating recovery report", "info")
        
        checkpoints = self.memory.find_checkpoints(tag="recovery_point")
        recovery_points = [c for c in checkpoints if "recovery_point" in c.get("tags", [])]
        
        return {
            "recovery_checkpoints": len(recovery_points),
            "recent_checkpoint": recovery_points[0]["created"] if recovery_points else "None",
            "failure_history_size": len(self.failure_history),
            "readiness_score": min(100, len(recovery_points) * 20),
            "recommended_actions": self.analyze_failure_patterns().get("recommendations", [])
        }

if __name__ == "__main__":
    print("\n=== Running Execution Recovery Tests ===\n")
    printer.status("TEST", "Starting Execution Recovery tests", "info")

    recovery = ExecutionRecovery()
    
    # Test 1: Handle movement failure
    context = {"current_position": (0, 0), "destination": (10, 10)}
    error = ActionFailureError("move_to", "Path blocked by obstacle")
    success, new_context = recovery.handle_failure("move_to", error, context)
    printer.pretty("Movement Recovery", success, "success" if success else "error")
    
    # Test 2: Handle pick failure
    context = {"object_position": (1, 1), "current_position": (1.2, 1.2)}
    error = ActionFailureError("pick_object", "Too far from object")
    success, new_context = recovery.handle_failure("pick_object", error, context)
    printer.pretty("Pick Recovery", success, "success" if success else "error")
    
    # Test 3: Analyze failure patterns
    report = recovery.analyze_failure_patterns()
    printer.pretty("Failure Analysis", report, "info")
    
    print("\n=== All recovery tests completed ===")