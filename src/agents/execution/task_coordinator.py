
import time
import pickle

from enum import Enum
from typing import List, Dict, Any, Optional, Set

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Task Coordinator")
printer = PrettyPrinter

class TaskState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskCoordinator:
    def __init__(self):
        self.config = load_global_config()
        self.task_config = get_config_section("task_coordinator")
        self.default_timeout = self.task_config.get("default_timeout")  # 5 minutes default
        self.max_retries =  self.task_config.get("max_retries")
        self.tasks: List[Dict[str, Any]] = []
        self.task_history: List[Dict[str, Any]] = []
        self.completed_tasks: Set[str] = set()

        self.memory = ExecutionMemory()
        self.state_key = "task_coordinator_state"
        self._load_state()

        # Task dependency tracking
        self.task_dependencies: Dict[str, List[str]] = {}
        self.task_dependents: Dict[str, List[str]] = {}
        
        logger.info("Task Coordinator initialized")

    @property
    def _memory(self):
        return self.memory

    def _load_state(self):
        """Load state from persistent memory"""
        printer.status("TASK", "Loading state...", "info")

        state = self.memory.get_cache(self.state_key)
        if state and isinstance(state, bytes):
            try:
                state = pickle.loads(state)
            except pickle.UnpicklingError:
                logger.error("Could not unpickle task coordinator state. Starting fresh.")
                state = {}
        
        # Handle different state types
        if not isinstance(state, dict):
            if state is None:
                state = {}
            else:
                logger.error(f"Unexpected state type: {type(state)}. Resetting state.")
                state = {}
        
        self.tasks = state.get("tasks", [])
        self.task_history = state.get("task_history", [])
        self.completed_tasks = set(state.get("completed_tasks", []))
        self.task_dependencies = state.get("task_dependencies", {})
        self.task_dependents = state.get("task_dependents", {})

    def assign_task(self, task: dict):
        """
        Accepts a task and adds it to the queue with enhanced validation.
        """
        printer.status("TASK", "Assigning task...", "info")

        if not task or not task.get("name"):
            logger.warning("Invalid task provided to assign")
            return False

        # Check for duplicate tasks
        if any(t["name"] == task["name"] for t in self.tasks):
            logger.warning(f"Task '{task['name']}' already exists in queue")
            return False

        # Set default values
        task.setdefault("priority", 0)
        task.setdefault("dependencies", [])
        task.setdefault("state", TaskState.PENDING.value)
        task.setdefault("timeout", self.default_timeout)
        task.setdefault("progress", 0.0)
        task.setdefault("retries", 0)
        task.setdefault("max_retries",self.max_retries)
        task.setdefault("created_at", time.time())
        task["assignee"] = None
        task["started_at"] = None
        task["completed_at"] = None

        # Register dependencies
        for dep in task["dependencies"]:
            self.task_dependencies.setdefault(task["name"], []).append(dep)
            self.task_dependents.setdefault(dep, []).append(task["name"])

        self.tasks.append(task)
        self._save_state()
        logger.info(f"Assigned new task: {task['name']} (Priority: {task['priority']})")
        return True

    def _save_state(self):
        """Save current state to persistent memory"""
        printer.status("TASK", "Saving task...", "info")

        state = {
            "tasks": self.tasks,
            "task_history": self.task_history,
            "completed_tasks": list(self.completed_tasks),
            "task_dependencies": self.task_dependencies,
            "task_dependents": self.task_dependents
        }
        self.memory.set_cache(self.state_key, state, ttl=None)

    def get_next_task(self) -> Optional[Dict]:
        """
        Return the highest priority task that's ready for execution,
        considering dependencies and current state.
        """
        printer.status("TASK", "Getting next task...", "info")

        if not self.tasks:
            logger.debug("No tasks in queue")
            return None

        # Filter tasks that are ready to run
        ready_tasks = []
        for task in self.tasks:
            if task["state"] == TaskState.PENDING.value and self._dependencies_satisfied(task):
                ready_tasks.append(task)

        if not ready_tasks:
            logger.debug("No tasks with satisfied dependencies")
            return None

        # Sort by priority (highest first) and creation time (oldest first)
        ready_tasks.sort(key=lambda t: (-t["priority"], t["created_at"]))
        return ready_tasks[0]

    def start_task(self, task_name: str, assignee: str) -> bool:
        """
        Mark a task as in progress and assign it to an agent.
        """
        printer.status("TASK", "Starting task...", "info")

        task = self._find_task(task_name)
        if not task:
            logger.warning(f"Task '{task_name}' not found for starting")
            return False
            
        if task["state"] != TaskState.PENDING.value:
            logger.warning(f"Cannot start task '{task_name}' in state {task['state']}")
            return False
            
        task["state"] = TaskState.IN_PROGRESS.value
        task["assignee"] = assignee
        task["started_at"] = time.time()
        task["retries"] += 1
        logger.info(f"Started task '{task_name}' assigned to {assignee}")
        return True

    def update_task_progress(self, task_name: str, progress: float) -> bool:
        """
        Update the progress of a task (0.0 to 1.0).
        """
        printer.status("TASK", "Updating task...", "info")

        task = self._find_task(task_name)
        if not task:
            return False
            
        if task["state"] != TaskState.IN_PROGRESS.value:
            logger.warning(f"Cannot update progress for task '{task_name}' in state {task['state']}")
            return False
            
        task["progress"] = max(0.0, min(1.0, progress))
        return True

    def pause_task(self, task_name: str) -> bool:
        task = self._find_task(task_name)
        if not task:
            return False
            
        if task["state"] == TaskState.IN_PROGRESS.value:
            task["state"] = TaskState.PAUSED.value
            logger.info(f"Paused task '{task_name}'")
            return True
        return False

    def complete_task(self, task_name: str) -> bool:
        """
        Mark a task as successfully completed.
        """
        printer.status("TASK", "Completing task...", "info")

        task = self._find_task(task_name)
        if not task:
            return False
            
        task["state"] = TaskState.COMPLETED.value
        task["progress"] = 1.0
        task["completed_at"] = time.time()
        self.completed_tasks.add(task_name)
        logger.info(f"Completed task '{task_name}'")
        
        # Archive completed task
        self.task_history.append(task)
        self.tasks = [t for t in self.tasks if t["name"] != task_name]
        
        # Resolve dependencies for dependent tasks
        self._resolve_dependencies(task_name)
        self._save_state()
        return True

    def fail_task(self, task_name: str, reason: str = "") -> bool:
        """
        Mark a task as failed with an optional reason.
        """
        printer.status("TASK", "Failed task", "info")

        task = self._find_task(task_name)
        if not task:
            return False
            
        task["state"] = TaskState.FAILED.value
        task["failure_reason"] = reason
        logger.error(f"Task '{task_name}' failed: {reason}")
        
        # Handle retries or escalate failure
        if task["retries"] < task["max_retries"]:
            logger.info(f"Rescheduling task '{task_name}' for retry")
            task["state"] = TaskState.PENDING.value
            task["assignee"] = None
            task["started_at"] = None
        else:
            logger.error(f"Task '{task_name}' has exceeded maximum retries")
            self.task_history.append(task)
            self.tasks = [t for t in self.tasks if t["name"] != task_name]
            
            # Cancel dependent tasks that can't proceed
            self._cancel_dependent_tasks(task_name)
        return True

    def cancel_task(self, task_name: str) -> bool:
        """
        Cancel a task and its dependents.
        """
        printer.status("TASK", "Cancelled task", "info")

        task = self._find_task(task_name)
        if not task:
            return False
            
        task["state"] = TaskState.CANCELLED.value
        logger.warning(f"Cancelled task '{task_name}'")
        
        # Archive cancelled task
        self.task_history.append(task)
        self.tasks = [t for t in self.tasks if t["name"] != task_name]
        
        # Cancel dependent tasks
        self._cancel_dependent_tasks(task_name)
        return True

    def validate_task_completion(self, task_name: str) -> bool:
        """
        Validate if a task is fully resolved and all effects are applied.
        """
        printer.status("TASK", "Validating task", "info")

        # Check if task exists in history
        task = next((t for t in self.task_history if t["name"] == task_name), None)
        if not task:
            return False
            
        # Check if task was actually completed
        if task["state"] != TaskState.COMPLETED.value:
            return False
            
        # Check if all postconditions are met (could be extended with actual world state checks)
        return True

    def check_timeouts(self):
        """
        Check for tasks that have exceeded their timeout and mark them as failed.
        Should be called periodically.
        """
        printer.status("TASK", "Checking timeouts...", "info")

        current_time = time.time()
        for task in self.tasks:
            if (task["state"] == TaskState.IN_PROGRESS.value and 
                task["started_at"] and 
                current_time - task["started_at"] > task["timeout"]):
                self.fail_task(task["name"], "Task timed out")

    def has_pending_tasks(self) -> bool:
        """Check if there are tasks ready for execution"""
        printer.status("TASK", "Checking task...", "info")

        return any(t["state"] == TaskState.PENDING.value for t in self.tasks)

    def _find_task(self, task_name: str) -> Optional[Dict]:
        """Find a task by name in the active task list"""
        printer.status("TASK", "Finding task...", "info")

        return next((t for t in self.tasks if t["name"] == task_name), None)

    def _dependencies_satisfied(self, task: Dict) -> bool:
        """Check if all dependencies for a task are satisfied"""
        printer.status("TASK", "Checking all dependency", "info")

        for dep in task.get("dependencies", []):
            if dep not in self.completed_tasks:
                return False
        return True

    def _resolve_dependencies(self, task_name: str):
        """Resolve dependencies when a task completes"""
        printer.status("TASK", "Resolving dependency", "info")

        # For any tasks that depended on this task, check if they're now ready
        for dependent in self.task_dependents.get(task_name, []):
            dep_task = self._find_task(dependent)
            if dep_task and self._dependencies_satisfied(dep_task):
                logger.debug(f"Task '{dependent}' dependencies now satisfied")

    def _cancel_dependent_tasks(self, task_name: str):
        """Cancel tasks that depend on a failed/cancelled task"""
        printer.status("TASK", "Cancelling task", "info")

        for dependent in self.task_dependents.get(task_name, []):
            dep_task = self._find_task(dependent)
            if dep_task:
                self.cancel_task(dep_task["name"])

    def retry_task(self, task_name: str):
        printer.status("TASK", "Retry task", "info")

        task = self._find_task(task_name)
        if task:
            task['state'] = TaskState.PENDING.value
            task['retry_count'] = task.get('retry_count', 0) + 1
            logger.info(f"Task '{task_name}' marked for retry (attempt {task['retry_count']})")

if __name__ == "__main__":
    print("\n=== Running Execution Task Coordinator ===\n")
    printer.status("TEST", "Starting Task Coordinator tests", "info")

    coordinator = TaskCoordinator()
    
    # Test task assignment
    coordinator.assign_task({"name": "scan_room", "priority": 2})
    coordinator.assign_task({
        "name": "fetch_item", 
        "priority": 5, 
        "dependencies": ["scan_room"],
        "timeout": 120
    })
    
    # Test getting next task
    printer.pretty("Next Task", coordinator.get_next_task()["name"], "success")
    
    # Test dependency resolution
    coordinator.complete_task("scan_room")
    printer.pretty("Next Task After Complete", coordinator.get_next_task()["name"], "success")
    
    # Test task failure and retry
    coordinator.fail_task("fetch_item", "Object not found")
    printer.pretty("Task After Failure", coordinator._find_task("fetch_item")["state"], "info")
    
    # Test validation
    printer.pretty("Validation", coordinator.validate_task_completion("scan_room"), "success")
