import threading
import time
from typing import Dict, Any, List, Optional, Tuple, Callable

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.planning_memory import PlanningMemory
from src.agents.planning.planning_types import Task, TaskStatus
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Executor")
printer = PrettyPrinter

class PlanningExecutor:
    def __init__(self):
        self.config = load_global_config()
        self.executor_config = get_config_section('planning_executor')
        self.state_checks_enabled = self.executor_config.get('state_checks_enabled')
        self.precondition_checks_enabled = self.executor_config.get('precondition_checks_enabled')
        self.snapshot_interval = self.executor_config.get('snapshot_interval')
        self.max_deviation_threshold = self.executor_config.get('max_deviation_threshold')
        self.check_interval = self.executor_config.get('check_interval')
        self.divergence_threshold = self.executor_config.get('divergence_threshold')

        self.agent = {}
        self.plan: List[Task] = []
        self.projections: Dict[str, Any] = {}

        self.memory = PlanningMemory()
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.event_handlers = {}
        self.expected_state_cache = {}
        self.current_task_index: int = 0
        self.state_deviation_count = 0

        self._lock = threading.RLock()
        self._is_running = False        
        self.register_default_events()

    def register_default_events(self):
        """Register default event handlers"""
        self.register_event('precondition_violation', self.handle_precondition_violation)
        self.register_event('state_deviation', self.handle_state_deviation)
        self.register_event('resource_violation', self.handle_resource_violation)
        self.register_event('temporal_violation', self.handle_temporal_violation)

    def register_event(self, event_type: str, handler: Callable):
        """Register a custom event handler"""
        printer.status("EXEC", "Registering event...", "info")

        self.event_handlers[event_type] = handler

    def start_monitoring(self, plan: List[Task], expected_states: Dict[str, Any]):
        """Start monitoring the execution of a plan"""
        printer.status("EXEC", "Starting monitor...", "info")

        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitor already running. Stopping previous instance.")
            self.stop_monitoring()

        with self._lock:
            self.plan = plan
            self.projections = expected_states
            self.current_task_index = 0
            self._is_running = True

        self.stop_event.clear()
        self.expected_state_cache = expected_states
        self.state_deviation_count = 0
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(plan,),
            daemon=False
        )
        self.monitor_thread.start()
        logger.info("Execution monitoring started")
        printer.status("EXEC-MONITOR", "Monitoring started", "info")

    def stop_monitoring(self):
        """Stop the monitoring process"""
        printer.status("EXEC", "Stopping monitor...", "info")

        if not self._is_running:
            return

        self._is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=self.check_interval * 2)

        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Execution monitoring stopped")
        printer.status("EXEC-MONITOR", "Monitoring stopped", "info")

    def _monitor_loop(self, plan: List[Task]):
        """Main monitoring loop running in a separate thread"""
        printer.status("EXEC", "Looping monitor...", "info")

        while not self.stop_event.is_set():
            try:
                # Capture state snapshot for later analysis
                self.memory.save_checkpoint(label="monitor_snapshot")
                
                # Check upcoming task preconditions
                if self.precondition_checks_enabled:
                    self._check_upcoming_preconditions(plan)
                
                # Verify state consistency
                if self.state_checks_enabled:
                    self._verify_state_consistency()
                
                # Monitor for perception events
                self._check_perception_events()
                
                time.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"Monitor loop error: {str(e)}", exc_info=True)
                time.sleep(1)  # Prevent tight error loop

    def _check_upcoming_preconditions(self, plan: List[Task]):
        """Check preconditions of upcoming tasks in the plan"""
        printer.status("EXEC", "Looping monitor...", "info")

        current_index = self._get_current_task_index(plan)
        if current_index is None:
            return
            
        lookahead = min(current_index + self.executor_config.get('precondition_lookahead', 3), len(plan))
        for i in range(current_index, lookahead):
            task = plan[i]
            if task.status != TaskStatus.PENDING:
                continue
                
            if not task.check_preconditions(self.agent.world_state):
                logger.warning(f"Precondition violation detected for task: {task.name}")
                self.trigger_event('precondition_violation', task)

    def _verify_state_consistency(self):
        """Compare actual world state with expected state"""
        printer.status("EXEC", "Verifying state...", "info")

        if not self.expected_state_cache:
            return
            
        # Calculate state deviation
        deviation_score = self._calculate_state_deviation(
            self.agent.world_state, 
            self.expected_state_cache
        )
        
        if deviation_score > self.max_deviation_threshold:
            self.state_deviation_count += 1
            if self.state_deviation_count > self.executor_config.get('max_tolerable_deviations', 2):
                logger.error(f"Significant state deviation detected: {deviation_score:.2f}")
                self.trigger_event('state_deviation', deviation_score)
        else:
            self.state_deviation_count = max(0, self.state_deviation_count - 1)

    def _check_state_divergence(self):
        """Compares the actual world state to the projected state."""
        last_completed_task = self.plan[self.current_task_index - 1]
        expected_state = self.projections.get(last_completed_task.name)
        
        if not expected_state:
            return  # No projection available for this task

        actual_state = self.agent.world_state
        divergence = self._calculate_state_divergence(expected_state, actual_state)

        if divergence > self.divergence_threshold:
            error_msg = f"State divergence ({divergence}) exceeds threshold ({self.divergence_threshold}) after task '{last_completed_task.name}'."
            logger.warning(error_msg)
            printer.status("EXEC-MONITOR", error_msg, "error")
            self._is_running = False # Stop monitoring
            # Trigger a general replan because the world is not as expected
            self.agent.replan_from_execution_failure(None, "state_divergence")

    def _calculate_state_divergence(self, state1: Dict, state2: Dict) -> int:
        """Calculates a simple divergence score between two states."""
        divergence_count = 0
        all_keys = set(state1.keys()) | set(state2.keys())
        
        for key in all_keys:
            val1 = state1.get(key)
            val2 = state2.get(key)
            # This comparison works for primitive types and the custom 'Any' type
            if val1 != val2:
                divergence_count += 1
                logger.debug(f"State divergence on key '{key}': Expected '{val1}', Got '{val2}'")

        return divergence_count

    def _calculate_state_deviation(self, actual: Dict, expected: Dict) -> float:
        """Calculate normalized deviation between actual and expected states"""
        total = 0
        matches = 0
        
        for key, exp_value in expected.items():
            total += 1
            if key in actual:
                if isinstance(exp_value, (int, float)) and isinstance(actual[key], (int, float)):
                    # Handle numeric comparisons
                    diff = abs(exp_value - actual[key])
                    range_val = self.executor_config.get(f'range.{key}', 1.0)
                    if diff < 0.05 * range_val:  # 5% tolerance
                        matches += 1
                elif actual[key] == exp_value:
                    matches += 1
        
        return 1.0 - (matches / total) if total > 0 else 0.0

    def _check_perception_events(self):
        """Check perception system for relevant events"""
        perception = self.agent.shared_memory.get('perception')
        if not perception:
            return
            
        events = perception.get_recent_events(
            types=['object_change', 'agent_movement', 'environment_change'],
            max_events=5
        )
        
        for event in events:
            if self._is_event_relevant(event):
                logger.info(f"Relevant perception event detected: {event['type']}")
                self.trigger_event('perception_event', event)

    def _is_event_relevant(self, event: Dict) -> bool:
        """Determine if an event is relevant to current plan"""
        # Check if event affects any pending tasks
        for task in self.agent.current_plan:
            if task.status == TaskStatus.PENDING:
                if 'affected_objects' in event:
                    if any(obj in task.required_objects for obj in event['affected_objects']):
                        return True
                if 'location' in event and hasattr(task, 'location'):
                    if event['location'] == task.location:
                        return True
        return False

    def _get_current_task_index(self, plan: List[Task]) -> Optional[int]:
        """Find the index of the currently executing task"""
        for i, task in enumerate(plan):
            if task.status == TaskStatus.EXECUTING:
                return i
        return None

    def trigger_event(self, event_type: str, *args, **kwargs):
        """Trigger an event handler"""
        handler = self.event_handlers.get(event_type)
        if handler:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {str(e)}", exc_info=True)
        else:
            logger.warning(f"No handler registered for event: {event_type}")

    def handle_precondition_violation(self, task: Task):
        """Default handler for precondition violations"""
        logger.warning(f"Handling precondition violation for task: {task.name}")
        self.agent.replan_from_execution_failure(task, "precondition_violation")

    def handle_state_deviation(self, deviation_score: float):
        """Default handler for state deviations"""
        logger.warning(f"Handling state deviation: {deviation_score:.2f}")
        self.agent.replan_from_execution_failure(None, "state_deviation")

    def handle_resource_violation(self, resource: str, usage: float, limit: float):
        """Default handler for resource violations"""
        logger.warning(f"Resource violation: {resource} usage {usage} > limit {limit}")
        self.agent.adjust_for_resource_violation(resource, usage, limit)

    def handle_temporal_violation(self, task: Task, time_delta: float):
        """Default handler for temporal violations"""
        logger.warning(f"Temporal violation: {task.name} is {time_delta:.1f}s behind schedule")
        self.agent.adjust_for_temporal_violation(task, time_delta)


if __name__ == "__main__":
    print("\n=== Running Planning Executor ===\n")
    printer.status("TEST", "Starting Planning Executor tests", "info")

    executor = PlanningExecutor()

    print(executor)

    print("\n* * * * * Phase 2 * * * * *\n")
    event_type = "custom_test_event"
    def handler(*args, **kwargs):
        print("Custom event triggered!", args, kwargs)    

    executor.register_event(event_type=event_type, handler=handler)
    executor.trigger_event("custom_test_event", "example_argument", key="value")

    print("\n* * * * * Phase 3 - monitor * * * * *\n")
    plan = [type("Task", (), {
        "name": "move_to_location",
        "status": "pending",
        "required_objects": [],
        "check_preconditions": lambda self, ws: True
    })()]
    
    expected_states = {
        "move_to_location": {
            "position": (5, 5),
            "energy": 10.0
        }
    }

    executor.agent = type("Agent", (), {
        "world_state": {"position": (5, 5), "energy": 10.0},
        "current_plan": [],
        "shared_memory": {"perception": None},
        "replan_from_execution_failure": lambda self, task, reason: print(f"Replan: {reason}"),
        "adjust_for_resource_violation": lambda self, res, usage, lim: print("Resource fix"),
        "adjust_for_temporal_violation": lambda self, task, dt: print("Temporal fix")
    })()
    
    monitor = executor.start_monitoring(plan=plan, expected_states=expected_states)
    printer.pretty("EXECUTION", monitor, "success")
    print("\n=== All tests completed successfully! ===\n")
