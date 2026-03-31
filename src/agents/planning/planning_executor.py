"""
Planning Executor – Production‑ready plan execution monitoring.

This module provides a thread‑safe executor that monitors the execution of a plan,
checking preconditions, state consistency, and triggering events when violations occur.
It integrates with PlanningMemory for snapshots and the planning agent for replanning.
"""

import threading
import time

from typing import Any, Callable, Dict, List, Optional, Union

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.planning_memory import PlanningMemory
from src.agents.planning.planning_types import Task, TaskStatus, TaskType
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Executor")
printer = PrettyPrinter

class PlanningExecutor:
    """
    Monitors plan execution, checks preconditions and state consistency,
    and triggers events on violations.
    """

    def __init__(self) -> None:
        """Initialize the executor with configuration."""
        self.config = load_global_config()
        self.executor_config = get_config_section("planning_executor")

        self.state_checks_enabled = self.executor_config.get('state_checks_enabled')
        self.precondition_checks_enabled = self.executor_config.get('precondition_checks_enabled')
        self.snapshot_interval = self.executor_config.get('snapshot_interval')
        self.max_deviation_threshold = self.executor_config.get('max_deviation_threshold')
        self.check_interval = self.executor_config.get('check_interval')
        self.divergence_threshold = self.executor_config.get('divergence_threshold')
        self.precondition_lookahead = self.executor_config.get('precondition_lookahead')
        self.max_tolerable_deviations = self.executor_config.get('max_tolerable_deviations', 2)

        # Runtime state
        self.agent: Optional[Any] = None  # Reference to the planning agent
        self.plan: List[Task] = []
        self.expected_states: Dict[str, Dict[str, Any]] = {}  # Task name -> expected state after task
        self.current_task_index: int = 0
        self._lock = threading.RLock()
        self._is_running = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._state_deviation_count = 0

        # Memory for snapshots
        self.memory = PlanningMemory()

        # Event handlers
        self._event_handlers: Dict[str, Callable] = {}
        self._register_default_events()

        logger.info("Planning Executor successfully initialized")

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------
    def _register_default_events(self) -> None:
        """Register the default event handlers."""
        self.register_event("precondition_violation", self._handle_precondition_violation)
        self.register_event("state_deviation", self._handle_state_deviation)
        self.register_event("resource_violation", self._handle_resource_violation)
        self.register_event("temporal_violation", self._handle_temporal_violation)

    def register_event(self, event_type: str, handler: Callable):
        """Register a custom event handler"""
        printer.status("EXEC", "Registering event...", "info")

        with self._lock:
            self._event_handlers[event_type] = handler
        logger.debug(f"Registered handler for event '{event_type}'")

    def trigger_event(self, event_type: str, *args: Any, **kwargs: Any) -> None:
        """
        Trigger an event, invoking its registered handler.

        Args:
            event_type: Event identifier.
            *args, **kwargs: Passed to the handler.
        """
        with self._lock:
            handler = self._event_handlers.get(event_type)
        if handler:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event handler for '{event_type}': {e}", exc_info=True)
        else:
            logger.warning(f"No handler registered for event: {event_type}")

    # -------------------------------------------------------------------------
    # Default event handlers (these call back to the agent)
    # -------------------------------------------------------------------------
    def _handle_precondition_violation(self, task: Task) -> None:
        """Default handler: ask the agent to replan due to precondition failure."""
        logger.warning(f"Handling precondition violation for task: {task.name}")
        if self.agent and hasattr(self.agent, "replan_from_execution_failure"):
            self.agent.replan_from_execution_failure(task, "precondition_violation")

    def _handle_state_deviation(self, deviation_score: float) -> None:
        """Default handler: replan when state deviates too much."""
        logger.warning(f"Handling state deviation: {deviation_score:.2f}")
        if self.agent and hasattr(self.agent, "replan_from_execution_failure"):
            self.agent.replan_from_execution_failure(None, "state_deviation")

    def _handle_resource_violation(self, resource: str, usage: float, limit: float) -> None:
        """Default handler: ask agent to adjust resource usage."""
        logger.warning(f"Resource violation: {resource} usage {usage} > limit {limit}")
        if self.agent and hasattr(self.agent, "adjust_for_resource_violation"):
            self.agent.adjust_for_resource_violation(resource, usage, limit)

    def _handle_temporal_violation(self, task: Task, time_delta: float) -> None:
        """Default handler: ask agent to adjust schedule."""
        logger.warning(f"Temporal violation: {task.name} is {time_delta:.1f}s behind schedule")
        if self.agent and hasattr(self.agent, "adjust_for_temporal_violation"):
            self.agent.adjust_for_temporal_violation(task, time_delta)

    # -------------------------------------------------------------------------
    # Public control methods
    # -------------------------------------------------------------------------
    def start_monitoring(self, plan: List[Task], expected_states: Dict[str, Dict[str, Any]]) -> None:
        """
        Start monitoring the execution of a plan.

        Args:
            plan: List of tasks in the plan (order of execution).
            expected_states: Map from task name to expected world state after that task.
        """
        if self._is_running:
            logger.warning("Monitoring already running. Stopping previous instance.")
            self.stop_monitoring()

        with self._lock:
            self.plan = plan
            self.expected_states = expected_states
            self.current_task_index = 0
            self._state_deviation_count = 0
            self._is_running = True
            self._stop_event.clear()

            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="PlanningExecutorMonitor",
                daemon=False,
            )
            self._monitor_thread.start()

        logger.info("Execution monitoring started")
        printer.status("EXEC-MONITOR", "Monitoring started", "info")

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if not self._is_running:
            return

        logger.info("Stopping execution monitoring...")
        self._is_running = False
        self._stop_event.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
            if self._monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop in time")
            else:
                logger.info("Monitor thread stopped")

        with self._lock:
            self._monitor_thread = None

        printer.status("EXEC-MONITOR", "Monitoring stopped", "info")

    def set_plan(self, plan: List[Task]) -> None:
        """
        Update the plan being monitored (call while monitoring is active).
        This resets internal state.
        """
        with self._lock:
            self.plan = plan
            self.current_task_index = 0
            self._state_deviation_count = 0
        logger.info(f"Plan updated to {len(plan)} tasks")

    def set_expected_states(self, expected_states: Dict[str, Dict[str, Any]]) -> None:
        """Update the expected state projections."""
        with self._lock:
            self.expected_states = expected_states

    # -------------------------------------------------------------------------
    # Monitoring loop
    # -------------------------------------------------------------------------
    def _monitor_loop(self) -> None:
        """Main monitoring loop, runs in a separate thread."""
        logger.debug("Monitor loop started")
        last_snapshot_time = 0.0

        while not self._stop_event.is_set():
            try:
                # Optionally save a snapshot (if enough time has passed)
                now = time.time()
                if now - last_snapshot_time >= self.snapshot_interval:
                    self._save_snapshot()
                    last_snapshot_time = now

                # Check preconditions of upcoming tasks
                if self.precondition_checks_enabled:
                    self._check_upcoming_preconditions()

                # Verify state consistency
                if self.state_checks_enabled:
                    self._verify_state_consistency()

                # Wait before next iteration
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)
                time.sleep(1.0)  # Prevent tight error loop

        logger.debug("Monitor loop exited")

    def _save_snapshot(self) -> None:
        """Save a checkpoint of the current state."""
        try:
            self.memory.save_checkpoint(label="monitor_snapshot")
        except Exception as e:
            logger.warning(f"Failed to save monitor snapshot: {e}")

    # -------------------------------------------------------------------------
    # Precondition checks
    # -------------------------------------------------------------------------
    def _check_upcoming_preconditions(self) -> None:
        """Check preconditions of upcoming tasks in the plan."""
        # Find the currently executing task
        current_idx = self._get_current_task_index()
        if current_idx is None:
            # No executing task; maybe not started yet or finished
            return

        # Determine tasks to check up to lookahead
        end_idx = min(current_idx + self.precondition_lookahead, len(self.plan))
        for i in range(current_idx, end_idx):
            task = self.plan[i]
            if task.status != TaskStatus.PENDING:
                continue

            # Get world state from agent if available, else empty
            world_state = getattr(self.agent, "world_state", {}) if self.agent else {}
            if not task.check_preconditions(world_state):
                logger.warning(f"Precondition violation for task: {task.name}")
                self.trigger_event("precondition_violation", task)

    def _get_current_task_index(self) -> Optional[int]:
        """
        Find the index of the currently executing task in the plan.

        Returns:
            Index of the first task with status EXECUTING, or None.
        """
        with self._lock:
            for i, task in enumerate(self.plan):
                if task.status == TaskStatus.EXECUTING:
                    return i
            return None

    # -------------------------------------------------------------------------
    # State consistency checks
    # -------------------------------------------------------------------------
    def _verify_state_consistency(self) -> None:
        """Compare actual world state with expected state after the last completed task."""
        # Determine the last completed task
        last_completed_idx = self._get_last_completed_index()
        if last_completed_idx is None:
            return

        last_task = self.plan[last_completed_idx]
        expected_state = self.expected_states.get(last_task.name)
        if not expected_state:
            # No expected state for this task – skip
            return

        actual_state = getattr(self.agent, "world_state", {}) if self.agent else {}
        deviation = self._calculate_state_deviation(actual_state, expected_state)

        if deviation > self.max_deviation_threshold:
            self._state_deviation_count += 1
            if self._state_deviation_count > self.max_tolerable_deviations:
                logger.error(
                    f"Significant state deviation detected after task '{last_task.name}': "
                    f"{deviation:.2f} (threshold {self.max_deviation_threshold})"
                )
                self.trigger_event("state_deviation", deviation)
                # Reset counter after triggering to avoid repeated triggers
                self._state_deviation_count = 0
        else:
            # Gradually reduce counter when state is good
            self._state_deviation_count = max(0, self._state_deviation_count - 1)

    def _get_last_completed_index(self) -> Optional[int]:
        """
        Find the index of the last task that has completed (status SUCCESS or FAILED).

        Returns:
            Index of the last completed task, or None if none completed.
        """
        with self._lock:
            last_idx = -1
            for i, task in enumerate(self.plan):
                if task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
                    last_idx = i
            return last_idx if last_idx >= 0 else None

    def _calculate_state_deviation(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """
        Calculate a normalized deviation score between actual and expected state.

        Args:
            actual: Current world state dictionary.
            expected: Expected world state dictionary.

        Returns:
            Float between 0 and 1, where 0 = perfect match, 1 = completely different.
        """
        total_keys = set(actual.keys()) | set(expected.keys())
        if not total_keys:
            return 0.0

        mismatches = 0
        for key in total_keys:
            actual_val = actual.get(key)
            expected_val = expected.get(key)

            # Simple equality check; can be extended for numeric tolerance
            if actual_val != expected_val:
                mismatches += 1
                logger.debug(f"State mismatch on '{key}': expected {expected_val}, got {actual_val}")

        return mismatches / len(total_keys)

    # -------------------------------------------------------------------------
    # Optional: method to check state divergence (kept for compatibility)
    # -------------------------------------------------------------------------
    def _check_state_divergence(self) -> None:
        """
        Legacy method to check divergence between expected and actual state.
        Called automatically by the monitor loop if enabled.
        Currently not used, but kept for API compatibility.
        """
        # Implementation can be added later if needed
        pass

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """
        Return a summary of the current monitoring status.

        Returns:
            Dictionary with current task index, plan length, deviation count, etc.
        """
        with self._lock:
            return {
                "is_running": self._is_running,
                "plan_length": len(self.plan),
                "current_task_index": self._get_current_task_index(),
                "state_deviation_count": self._state_deviation_count,
            }


# -------------------------------------------------------------------------
# Test
# -------------------------------------------------------------------------
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
    # Create a dummy task using the updated Task class
    class DummyAgent:
        world_state = {"position": (5, 5), "energy": 10.0}
        def replan_from_execution_failure(self, task, reason):
            print(f"Replan: {reason}")
        def adjust_for_resource_violation(self, resource, usage, limit):
            print("Resource fix")
        def adjust_for_temporal_violation(self, task, dt):
            print("Temporal fix")

    dummy_task = Task(
        name="move_to_location",
        task_type=TaskType.PRIMITIVE,
        preconditions=[lambda ws: True],
    )

    plan = [dummy_task]
    expected_states = {
        "move_to_location": {"position": (5, 5), "energy": 10.0}
    }

    executor.agent = DummyAgent()
    executor.start_monitoring(plan=plan, expected_states=expected_states)

    # Let it run for a couple of seconds
    time.sleep(3)
    executor.stop_monitoring()

    print("\n=== All tests completed successfully! ===\n")