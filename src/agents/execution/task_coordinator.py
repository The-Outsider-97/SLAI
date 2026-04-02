import copy
import time
import pickle

from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .utils.config_loader import load_global_config, get_config_section
from .utils.execution_error import DeadlockError
from .execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Task Coordinator")
printer = PrettyPrinter

class TaskState(Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskCoordinator:
    """
    Production-ready task coordination layer with:
    - durable state persistence via ExecutionMemory
    - dependency indexing and cycle detection
    - retries, timeout handling, and lifecycle transitions
    - deadlock visibility and bounded task history
    """

    TERMINAL_STATES = {
        TaskState.COMPLETED.value,
        TaskState.FAILED.value,
        TaskState.CANCELLED.value,
    }

    def __init__(self, memory: Optional[ExecutionMemory] = None):
        self.config = load_global_config()
        self.task_config = get_config_section("task_coordinator") or {}

        self.default_timeout = int(self.task_config.get("default_timeout", 300))
        self.max_retries = int(self.task_config.get("max_retries", 3))
        self.state_ttl = int(self.task_config.get("state_ttl", 30 * 86400))
        self.history_limit = int(self.task_config.get("history_limit", 1000))
        self.deadlock_timeout = int(self.task_config.get("deadlock_timeout", max(60, self.default_timeout)))

        self.tasks: List[Dict[str, Any]] = []
        self.task_history: List[Dict[str, Any]] = []
        self.completed_tasks: Set[str] = set()
        self.task_dependencies: Dict[str, List[str]] = {}
        self.task_dependents: Dict[str, List[str]] = {}

        self.memory = memory or ExecutionMemory()
        self.state_key = "task_coordinator_state"
        self.state_namespace = "task_coordinator"

        self._load_state()
        logger.info("Task Coordinator initialized")

    @property
    def _memory(self):
        return self.memory

    def _load_state(self) -> None:
        printer.status("TASK", "Loading state...", "info")
        state = self.memory.get_cache(self.state_key, namespace=self.state_namespace, default={})

        if not isinstance(state, dict):
            logger.error("Unexpected persisted task state type: %s. Resetting.", type(state))
            state = {}

        self.tasks = list(state.get("tasks", []))
        self.task_history = list(state.get("task_history", []))[-self.history_limit :]
        self.completed_tasks = set(state.get("completed_tasks", []))
        self.task_dependencies = dict(state.get("task_dependencies", {}))
        self.task_dependents = dict(state.get("task_dependents", {}))

        if not self.task_dependencies or not self.task_dependents:
            self._rebuild_dependency_indexes()

        for task in self.tasks:
            self._refresh_task_block_state(task)

    def _save_state(self) -> None:
        printer.status("TASK", "Saving task...", "info")
        state = {
            "tasks": self.tasks,
            "task_history": self.task_history[-self.history_limit :],
            "completed_tasks": sorted(self.completed_tasks),
            "task_dependencies": self.task_dependencies,
            "task_dependents": self.task_dependents,
            "updated_at": time.time(),
        }
        self.memory.set_cache(
            self.state_key,
            state,
            namespace=self.state_namespace,
            ttl=self.state_ttl,
        )

    def assign_task(self, task: Dict[str, Any]) -> bool:
        printer.status("TASK", "Assigning task...", "info")

        if not task or not task.get("name"):
            logger.warning("Invalid task provided to assign")
            return False

        task_name = str(task["name"])
        if self._find_task(task_name) is not None:
            logger.warning("Task '%s' already exists in queue", task_name)
            return False

        normalized = copy.deepcopy(task)
        normalized["name"] = task_name
        normalized.setdefault("priority", 0)
        normalized.setdefault("dependencies", [])
        normalized.setdefault("state", TaskState.PENDING.value)
        normalized.setdefault("timeout", self.default_timeout)
        normalized.setdefault("progress", 0.0)
        normalized.setdefault("retries", 0)
        normalized.setdefault("max_retries", self.max_retries)
        normalized.setdefault("created_at", time.time())
        normalized.setdefault("updated_at", normalized["created_at"])
        normalized.setdefault("postconditions", [])
        normalized.setdefault("metadata", {})
        normalized["assignee"] = normalized.get("assignee")
        normalized["started_at"] = normalized.get("started_at")
        normalized["completed_at"] = normalized.get("completed_at")
        normalized["failure_reason"] = normalized.get("failure_reason")

        normalized["dependencies"] = self._normalize_dependencies(task_name, normalized.get("dependencies", []))
        self._register_task_dependencies(task_name, normalized["dependencies"])

        self.tasks.append(normalized)
        if self._has_dependency_cycle():
            self.tasks.pop()
            self._unregister_task_dependencies(task_name, normalized["dependencies"])
            logger.error("Task '%s' introduces a dependency cycle; rejecting", task_name)
            return False

        self._refresh_task_block_state(normalized)
        self._save_state()
        logger.info("Assigned new task: %s (priority=%s)", task_name, normalized["priority"])
        return True

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        printer.status("TASK", "Getting next task...", "info")
        self.check_timeouts()

        ready_tasks = []
        for task in self.tasks:
            self._refresh_task_block_state(task)
            if task["state"] == TaskState.PENDING.value and self._dependencies_satisfied(task):
                ready_tasks.append(task)

        if not ready_tasks:
            logger.debug("No tasks currently ready for execution")
            return None

        ready_tasks.sort(
            key=lambda task: (
                -int(task.get("priority", 0)),
                float(task.get("deadline", float("inf"))),
                float(task.get("created_at", time.time())),
            )
        )
        return copy.deepcopy(ready_tasks[0])

    def start_task(self, task_name: str, assignee: str) -> bool:
        printer.status("TASK", "Starting task...", "info")
        task = self._find_task(task_name)
        if not task:
            logger.warning("Task '%s' not found for starting", task_name)
            return False

        if task["state"] not in {TaskState.PENDING.value, TaskState.PAUSED.value}:
            logger.warning("Cannot start task '%s' in state %s", task_name, task["state"])
            return False

        if not self._dependencies_satisfied(task):
            self._refresh_task_block_state(task)
            logger.warning("Cannot start task '%s'; dependencies are not satisfied", task_name)
            return False

        if task["state"] == TaskState.PENDING.value:
            task["retries"] = int(task.get("retries", 0)) + 1
        task["state"] = TaskState.IN_PROGRESS.value
        task["assignee"] = assignee
        task["started_at"] = time.time()
        task["updated_at"] = task["started_at"]
        self._save_state()
        logger.info("Started task '%s' assigned to %s", task_name, assignee)
        return True

    def update_task_progress(self, task_name: str, progress: float) -> bool:
        printer.status("TASK", "Updating task...", "info")
        task = self._find_task(task_name)
        if not task:
            return False
        if task["state"] != TaskState.IN_PROGRESS.value:
            logger.warning("Cannot update progress for task '%s' in state %s", task_name, task["state"])
            return False

        task["progress"] = max(0.0, min(1.0, float(progress)))
        task["updated_at"] = time.time()
        self._save_state()
        return True

    def pause_task(self, task_name: str) -> bool:
        task = self._find_task(task_name)
        if not task:
            return False
        if task["state"] == TaskState.IN_PROGRESS.value:
            task["state"] = TaskState.PAUSED.value
            task["updated_at"] = time.time()
            self._save_state()
            logger.info("Paused task '%s'", task_name)
            return True
        return False

    def resume_task(self, task_name: str) -> bool:
        task = self._find_task(task_name)
        if not task:
            return False
        if task["state"] == TaskState.PAUSED.value:
            task["state"] = TaskState.PENDING.value
            task["updated_at"] = time.time()
            self._save_state()
            logger.info("Resumed task '%s'", task_name)
            return True
        return False

    def complete_task(self, task_name: str) -> bool:
        printer.status("TASK", "Completing task...", "info")
        task = self._find_task(task_name)
        if not task:
            return False

        task["state"] = TaskState.COMPLETED.value
        task["progress"] = 1.0
        task["completed_at"] = time.time()
        task["updated_at"] = task["completed_at"]
        self.completed_tasks.add(task_name)
        logger.info("Completed task '%s'", task_name)

        self._archive_task(task)
        self._remove_active_task(task_name)
        self._resolve_dependencies(task_name)
        self._save_state()
        return True

    def _is_retryable_failure(self, task: Dict[str, Any], reason: str) -> bool:
        reason_lc = (reason or "").lower()
    
        if "timed out" in reason_lc:
            return True
        if "temporary" in reason_lc or "transient" in reason_lc:
            return True
    
        terminal_markers = [
            "object not found",
            "invalid target",
            "missing dependency",
            "permission denied",
            "unsupported",
        ]
        if any(marker in reason_lc for marker in terminal_markers):
            return False
    
        return bool(task.get("retry_on_failure", True))
    
    
    def fail_task(self, task_name: str, reason: str = "") -> bool:
        printer.status("TASK", "Failed task", "info")
        task = self._find_task(task_name)
        if not task:
            return False
    
        task["state"] = TaskState.FAILED.value
        task["failure_reason"] = reason
        task["updated_at"] = time.time()
        logger.error("Task '%s' failed: %s", task_name, reason)
    
        retryable = self._is_retryable_failure(task, reason)
        retries = int(task.get("retries", 0))
        max_retries = int(task.get("max_retries", self.max_retries))
    
        if retryable and retries < max_retries:
            logger.info("Rescheduling task '%s' for retry", task_name)
            task["state"] = TaskState.PENDING.value
            task["assignee"] = None
            task["started_at"] = None
            task["updated_at"] = time.time()
            self._refresh_task_block_state(task)
        else:
            logger.error(
                "Task '%s' will not be retried (retryable=%s, retries=%s/%s)",
                task_name, retryable, retries, max_retries
            )
            self._archive_task(task)
            self._remove_active_task(task_name)
            self._cancel_dependent_tasks(task_name, cascade=True)
    
        self._save_state()
        return True

    def cancel_task(self, task_name: str, cascade: bool = True) -> bool:
        printer.status("TASK", "Cancelled task", "info")
        task = self._find_task(task_name)
        if not task:
            return False

        task["state"] = TaskState.CANCELLED.value
        task["updated_at"] = time.time()
        logger.warning("Cancelled task '%s'", task_name)

        self._archive_task(task)
        self._remove_active_task(task_name)
        if cascade:
            self._cancel_dependent_tasks(task_name, cascade=cascade)
        self._save_state()
        return True

    def retry_task(self, task_name: str) -> bool:
        printer.status("TASK", "Retry task", "info")
        task = self._find_task(task_name)
        if not task:
            return False

        if int(task.get("retries", 0)) >= int(task.get("max_retries", self.max_retries)):
            logger.warning("Task '%s' cannot be retried; max retries reached", task_name)
            return False

        task["state"] = TaskState.PENDING.value
        task["failure_reason"] = None
        task["assignee"] = None
        task["started_at"] = None
        task["updated_at"] = time.time()
        self._refresh_task_block_state(task)
        self._save_state()
        logger.info("Task '%s' marked for retry", task_name)
        return True

    def validate_task_completion(self, task_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        printer.status("TASK", "Validating task", "info")
        task = next((task for task in self.task_history if task["name"] == task_name), None)
        if not task:
            return False
        if task["state"] != TaskState.COMPLETED.value:
            return False

        if context:
            for postcondition in task.get("postconditions", []):
                if not context.get(postcondition, False):
                    return False
        return True

    def check_timeouts(self) -> List[str]:
        printer.status("TASK", "Checking timeouts...", "info")
        current_time = time.time()
        timed_out: List[str] = []

        for task in list(self.tasks):
            if task["state"] == TaskState.IN_PROGRESS.value and task.get("started_at"):
                timeout = float(task.get("timeout", self.default_timeout))
                if current_time - float(task["started_at"]) > timeout:
                    timed_out.append(task["name"])
                    self.fail_task(task["name"], "Task timed out")

        return timed_out

    def has_pending_tasks(self) -> bool:
        printer.status("TASK", "Checking task...", "info")
        for task in self.tasks:
            self._refresh_task_block_state(task)
        return any(task["state"] in {TaskState.PENDING.value, TaskState.BLOCKED.value} for task in self.tasks)

    def detect_deadlocks(self) -> Optional[DeadlockError]:
        active = [task for task in self.tasks if task["state"] in {TaskState.PENDING.value, TaskState.BLOCKED.value}]
        if not active:
            return None

        if self._has_dependency_cycle():
            involved = sorted(task["name"] for task in active)
            return DeadlockError(involved)

        stalled = []
        now = time.time()
        for task in active:
            age = now - float(task.get("updated_at", task.get("created_at", now)))
            if age >= self.deadlock_timeout and not self._dependencies_satisfied(task):
                stalled.append(task["name"])

        if stalled:
            return DeadlockError(sorted(stalled))
        return None

    def get_task_snapshot(self, task_name: str) -> Optional[Dict[str, Any]]:
        task = self._find_task(task_name)
        return copy.deepcopy(task) if task else None

    def list_tasks(self, include_history: bool = False) -> List[Dict[str, Any]]:
        tasks = copy.deepcopy(self.tasks)
        if include_history:
            tasks.extend(copy.deepcopy(self.task_history))
        return tasks

    def summary(self) -> Dict[str, Any]:
        return {
            "active_tasks": len(self.tasks),
            "history_tasks": len(self.task_history),
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len([t for t in self.tasks if t["state"] == TaskState.PENDING.value]),
            "blocked_tasks": len([t for t in self.tasks if t["state"] == TaskState.BLOCKED.value]),
            "in_progress_tasks": len([t for t in self.tasks if t["state"] == TaskState.IN_PROGRESS.value]),
        }

    def _archive_task(self, task: Dict[str, Any]) -> None:
        archived = copy.deepcopy(task)
        self.task_history.append(archived)
        if len(self.task_history) > self.history_limit:
            self.task_history = self.task_history[-self.history_limit :]

    def _remove_active_task(self, task_name: str) -> None:
        self.tasks = [task for task in self.tasks if task["name"] != task_name]

    def _find_task(self, task_name: str) -> Optional[Dict[str, Any]]:
        return next((task for task in self.tasks if task["name"] == task_name), None)

    def _normalize_dependencies(self, task_name: str, dependencies: List[Any]) -> List[str]:
        normalized: List[str] = []
        for dependency in dependencies or []:
            dependency_name = str(dependency)
            if dependency_name == task_name:
                raise ValueError(f"Task '{task_name}' cannot depend on itself")
            if dependency_name not in normalized:
                normalized.append(dependency_name)
        return normalized

    def _dependencies_satisfied(self, task: Dict[str, Any]) -> bool:
        for dependency in task.get("dependencies", []):
            if dependency not in self.completed_tasks:
                return False
        return True

    def _refresh_task_block_state(self, task: Dict[str, Any]) -> None:
        if task["state"] == TaskState.PENDING.value and not self._dependencies_satisfied(task):
            task["state"] = TaskState.BLOCKED.value
        elif task["state"] == TaskState.BLOCKED.value and self._dependencies_satisfied(task):
            task["state"] = TaskState.PENDING.value

    def _register_task_dependencies(self, task_name: str, dependencies: List[str]) -> None:
        self.task_dependencies[task_name] = list(dependencies)
        for dependency in dependencies:
            self.task_dependents.setdefault(dependency, [])
            if task_name not in self.task_dependents[dependency]:
                self.task_dependents[dependency].append(task_name)

    def _unregister_task_dependencies(self, task_name: str, dependencies: List[str]) -> None:
        self.task_dependencies.pop(task_name, None)
        for dependency in dependencies:
            dependents = self.task_dependents.get(dependency, [])
            if task_name in dependents:
                dependents.remove(task_name)
            if not dependents and dependency in self.task_dependents:
                del self.task_dependents[dependency]

    def _resolve_dependencies(self, task_name: str) -> None:
        for dependent_name in self.task_dependents.get(task_name, []):
            dependent = self._find_task(dependent_name)
            if dependent:
                self._refresh_task_block_state(dependent)
                if dependent["state"] == TaskState.PENDING.value:
                    logger.debug("Task '%s' dependencies are now satisfied", dependent_name)

    def _cancel_dependent_tasks(self, task_name: str, cascade: bool = True) -> None:
        for dependent_name in list(self.task_dependents.get(task_name, [])):
            dependent = self._find_task(dependent_name)
            if dependent:
                self.cancel_task(dependent["name"], cascade=cascade)

    def _rebuild_dependency_indexes(self) -> None:
        self.task_dependencies = {}
        self.task_dependents = {}
        for task in self.tasks:
            dependencies = self._normalize_dependencies(task["name"], task.get("dependencies", []))
            task["dependencies"] = dependencies
            self._register_task_dependencies(task["name"], dependencies)

    def _has_dependency_cycle(self) -> bool:
        graph = {task["name"]: list(task.get("dependencies", [])) for task in self.tasks}
        visited: Set[str] = set()
        visiting: Set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for dependency in graph.get(node, []):
                if dependency in graph and visit(dependency):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in list(graph.keys()))


if __name__ == "__main__":
    print("\n=== Running Execution Task Coordinator ===\n")
    printer.status("TEST", "Starting Task Coordinator tests", "info")

    coordinator = TaskCoordinator()
    coordinator.assign_task({"name": "scan_room", "priority": 2})
    coordinator.assign_task(
        {
            "name": "fetch_item",
            "priority": 5,
            "dependencies": ["scan_room"],
            "timeout": 120,
        }
    )

    next_task = coordinator.get_next_task()
    printer.pretty("Next Task", next_task["name"] if next_task else None, "success")
    coordinator.complete_task("scan_room")
    next_task = coordinator.get_next_task()
    printer.pretty("Next Task After Complete", next_task["name"] if next_task else None, "success")
    coordinator.fail_task("fetch_item", "Object not found")
    fetch_item = coordinator._find_task("fetch_item")
    printer.pretty("Task After Failure", fetch_item["state"] if fetch_item else None, "info")
    printer.pretty("Summary", coordinator.summary(), "info")
