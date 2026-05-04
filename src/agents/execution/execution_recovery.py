import copy
import random
import time

from typing import Any, Callable, Dict, List, Optional, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.execution_error import (ActionFailureError, CorruptedContextStateError,
                                    DeadlockError, ExecutionError,
                                    ExecutionLoopLockError, InvalidContextError,
                                    MissingActionHandlerError, StaleCheckpointError,
                                    TimeoutError as ExecutionTimeoutError,
                                    UnreachableTargetError)
from .execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Recovery")
printer = PrettyPrinter

class ExecutionRecovery:
    """
    Production-ready execution recovery system.

    Responsibilities:
    - classify and sanitize failures
    - select recovery strategies per action/error type
    - persist failure history for later analysis
    - create and restore bounded recovery checkpoints
    - coordinate with TaskCoordinator when failure escalation is needed
    """

    DEFAULT_EXCLUDED_STATE_KEYS = {"map_data", "object_state", "raw_sensor_stream"}

    def __init__(
        self,
        memory: Optional[ExecutionMemory] = None,
        task_coordinator: Optional[Any] = None,
    ):
        self.config = load_global_config()
        self.recovery_config = get_config_section("execution_recovery") or {}

        self.max_retries = max(0, int(self.config.get("max_retries", 3)))
        self.max_history = max(1, int(self.recovery_config.get("max_failure_history", 100)))
        self.recovery_idle_time = float(self.recovery_config.get("recovery_idle_time", 5.0))
        self.sensitive_keys = set(self.recovery_config.get("sensitive_keys", ["password", "token"]))
        self.failure_loop_threshold = max(2, int(self.recovery_config.get("failure_loop_threshold", 3)))
        self.recent_window_seconds = max(30, int(self.recovery_config.get("recent_window_seconds", 900)))
        self.base_retry_delay = float(self.recovery_config.get("base_retry_delay", 0.25))
        self.max_retry_delay = float(self.recovery_config.get("max_retry_delay", 1.5))
        self.checkpoint_ttl = int(self.recovery_config.get("checkpoint_ttl", 3600))
        self.checkpoint_max_age = int(self.recovery_config.get("checkpoint_max_age", 1800))
        self.history_cache_ttl = int(self.recovery_config.get("history_cache_ttl", 86400))

        self.memory = memory or ExecutionMemory()
        self.task_coordinator = task_coordinator

        self.history_namespace = "execution_recovery"
        self.history_key = "failure_history"

        self.failure_history: List[Dict[str, Any]] = []
        self._load_history()

        self.recovery_strategies: Dict[str, Callable[[str, Exception, Dict[str, Any]], Tuple[bool, Dict[str, Any]]]] = {
            "move_to": self._handle_move_failure,
            "pick_object": self._handle_pick_failure,
            "place_object": self._handle_place_failure,
            "idle": self._handle_idle_failure,
            "default": self._handle_general_failure,
        }

        logger.info("Execution Recovery initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attach_task_coordinator(self, task_coordinator: Any) -> None:
        self.task_coordinator = task_coordinator

    def register_strategy(
        self,
        action_name: str,
        handler: Callable[[str, Exception, Dict[str, Any]], Tuple[bool, Dict[str, Any]]],
    ) -> None:
        self.recovery_strategies[action_name] = handler

    def handle_failure(self, action_name: str, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Main recovery entry point.

        Returns:
            (recovery_success, modified_context)
        """
        printer.status("RECOVERY", f"Handling {action_name} failure", "info")
    
        safe_context = copy.deepcopy(context or {})
        error = error if isinstance(error, Exception) else Exception(str(error))
    
        record = self._log_failure(action_name, error, safe_context)
        handler = self.recovery_strategies.get(action_name, self.recovery_strategies["default"])
    
        try:
            recovery_success, modified_context = handler(action_name, error, safe_context)
        except Exception as strategy_error:
            logger.error("Recovery strategy for %s failed unexpectedly: %s", action_name, strategy_error)
            recovery_success, modified_context = False, safe_context
    
        printer.status(
            "RECOVERY",
            f"{action_name} recovery {'succeeded' if recovery_success else 'failed'}",
            "success" if recovery_success else "warning",
        )

        modified_context = copy.deepcopy(modified_context or safe_context)
        modified_context.setdefault("recovery_metadata", {})
        modified_context["recovery_metadata"].update(
            {
                "failed_action": action_name,
                "failure_error": type(error).__name__,
                "failure_message": str(error),
                "recovery_success": recovery_success,
                "failure_id": record["failure_id"],
                "attempt_index": record["attempt_index"],
                "timestamp": time.time(),
            }
        )

        loop_detected = self._detect_failure_loop(action_name, error)
        if loop_detected and recovery_success:
            modified_context["recovery_metadata"]["loop_detected"] = True
            modified_context["force_idle"] = True
            modified_context.setdefault("idle_duration", self.recovery_idle_time)
            modified_context.setdefault("retry_count", safe_context.get("retry_count", 0) + 1)

        record["recovery_attempted"] = True
        record["recovery_success"] = recovery_success
        record["recovery_strategy"] = getattr(handler, "__name__", "custom")
        record["loop_detected"] = loop_detected
        self._save_history()

        if recovery_success:
            logger.info("Recovery strategy for %s succeeded", action_name)
        else:
            logger.error("All recovery attempts for %s failed", action_name)

        return recovery_success, modified_context

    def create_recovery_checkpoint(
        self,
        checkpoint_type: str,
        state: Dict[str, Any],
        extra_tags: Optional[List[str]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Create a bounded recovery checkpoint with filtered state."""
        printer.status("RECOVERY", f"Creating {checkpoint_type} checkpoint", "info")
        filtered_state = self._filter_checkpoint_state(state)
        tags = [checkpoint_type, "recovery_point"] + list(extra_tags or [])
        metadata = {
            "timestamp": time.time(),
            "checkpoint_type": checkpoint_type,
            "state_keys": sorted(filtered_state.keys()),
        }
        checkpoint_id = self.memory.create_checkpoint(
            filtered_state,
            tags=sorted(set(tags)),
            metadata=metadata,
            ttl=ttl or self.checkpoint_ttl,
        )
        logger.info("Created recovery checkpoint %s for %s", checkpoint_id, checkpoint_type)
        return checkpoint_id

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Aggregate recent failure history and generate recommendations."""
        printer.status("RECOVERY", "Analyzing failure patterns", "info")
        if not self.failure_history:
            return {
                "status": "no_data",
                "total_failures": 0,
                "actions_by_failure": {},
                "common_errors": {},
                "recommendations": [],
            }

        actions_by_failure: Dict[str, int] = {}
        common_errors: Dict[str, int] = {}
        action_error_map: Dict[str, Dict[str, int]] = {}

        for record in self.failure_history:
            action = record["action"]
            error_type = record["error_type"]
            actions_by_failure[action] = actions_by_failure.get(action, 0) + 1
            common_errors[error_type] = common_errors.get(error_type, 0) + 1
            action_error_map.setdefault(action, {})
            action_error_map[action][error_type] = action_error_map[action].get(error_type, 0) + 1

        recommendations = []
        for action, errors in sorted(action_error_map.items()):
            common_error, count = max(errors.items(), key=lambda item: item[1])
            recommendations.append(
                {
                    "action": action,
                    "most_common_error": common_error,
                    "count": count,
                    "suggestion": self._generate_error_suggestion(action, common_error),
                }
            )

        return {
            "status": "ok",
            "total_failures": len(self.failure_history),
            "actions_by_failure": actions_by_failure,
            "common_errors": common_errors,
            "recommendations": recommendations,
        }

    def get_recovery_report(self) -> Dict[str, Any]:
        printer.status("RECOVERY", "Generating recovery report", "info")
        checkpoints = self.memory.find_checkpoints(tag="recovery_point", max_age=self.checkpoint_max_age)
        recommendations = self.analyze_failure_patterns().get("recommendations", [])
        readiness = 0
        readiness += min(40, len(checkpoints) * 10)
        readiness += min(30, max(0, 30 - len(recommendations) * 5))
        readiness += min(30, max(0, self.max_history - len(self.failure_history)) / max(1, self.max_history) * 30)

        return {
            "recovery_checkpoints": len(checkpoints),
            "recent_checkpoint": checkpoints[0]["created"] if checkpoints else None,
            "failure_history_size": len(self.failure_history),
            "readiness_score": round(min(100.0, readiness), 1),
            "recommended_actions": recommendations,
        }

    def get_recent_failures(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.failure_history[-max(1, limit):]

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------
    def _handle_move_failure(self, action_name: str, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        printer.status("RECOVERY", "Movement failure recovery", "warning")
        modified = self._base_retry_context(context)

        modified["replan_threshold"] = float(context.get("replan_threshold", 0.05)) * 1.5
        modified["avoidance_radius"] = float(context.get("avoidance_radius", 1.0)) * 1.2
        modified["path_update_interval"] = max(0.25, float(context.get("path_update_interval", 2.0)) * 0.75)

        message = str(error).lower()
        if isinstance(error, (UnreachableTargetError, ExecutionTimeoutError)) or "no path" in message:
            modified["path_simplify"] = True
            modified["allow_partial_path"] = True
            modified["movement_timeout_multiplier"] = 1.5

        if "unreachable" in message:
            success, restored = self._backtrack_to_checkpoint(["pre_movement", "pre_action"], context)
            if success and self._movement_preconditions_changed(context, restored):
                restored["recovery_mode"] = True
                return True, restored
            modified.setdefault("disallowed_actions", [])
            if action_name not in modified["disallowed_actions"]:
                modified["disallowed_actions"].append(action_name)
            modified["recovery_reason"] = "unreachable_target"
            return False, modified

        if "blocked" in message or "obstacle" in message:
            success, restored = self._backtrack_to_checkpoint(["pre_movement", "pre_action"], context)
            if success:
                restored.update(
                    {
                        "replan_threshold": modified["replan_threshold"],
                        "avoidance_radius": modified["avoidance_radius"],
                        "recovery_mode": True,
                    }
                )
                return True, restored

        return True, modified

    @staticmethod
    def _movement_preconditions_changed(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
        keys = ("current_position", "destination", "map_data", "obstacles", "dynamic_objects")
        return any(before.get(key) != after.get(key) for key in keys)

    def _handle_pick_failure(self, action_name: str, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        printer.status("RECOVERY", "Pick failure recovery", "warning")
        modified = self._base_retry_context(context)
        message = str(error).lower()

        if "too far" in message or "distance" in message:
            modified["approach_distance"] = max(0.05, float(context.get("min_distance", 0.5)) * 0.8)
            modified["requires_reposition"] = True
            return True, modified

        if "grasp failed" in message or "slip" in message:
            modified["grasp_strategy"] = "alternative"
            modified["grasp_time"] = float(context.get("grasp_time", 1.0)) * 1.3
            modified["grasp_force_scale"] = 1.1
            return True, modified

        if "holding" in message and context.get("holding_object"):
            modified["confirm_held_object"] = True
            return True, modified

        return self._backtrack_to_checkpoint(["pre_pick", "pre_action"], context)

    def _handle_place_failure(self, action_name: str, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        printer.status("RECOVERY", "Place failure recovery", "warning")
        modified = self._base_retry_context(context)
        message = str(error).lower()

        if "not holding" in message:
            modified["requires_inventory_reconcile"] = True
            return False, modified

        if "too far" in message or "distance" in message:
            modified["requires_reposition"] = True
            modified["approach_distance"] = max(0.05, float(context.get("min_distance", 0.5)) * 0.8)
            return True, modified

        if "placement" in message or "collision" in message:
            modified["placement_strategy"] = "offset_retry"
            modified["placement_offset"] = context.get("placement_offset", 0.1)
            return True, modified

        return self._backtrack_to_checkpoint(["pre_place", "pre_action"], context)

    def _handle_idle_failure(self, action_name: str, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        printer.status("RECOVERY", "Idle failure recovery", "warning")
        modified = self._base_retry_context(context)
        modified["idle_duration"] = max(self.recovery_idle_time, float(context.get("idle_duration", self.recovery_idle_time)))
        modified["protected_idle"] = True
        return True, modified

    def _handle_general_failure(self, action_name: str, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        printer.status("RECOVERY", "General failure recovery", "warning")
        modified = self._base_retry_context(context)

        if isinstance(error, (InvalidContextError, CorruptedContextStateError)):
            modified["requires_context_refresh"] = True
            return self._backtrack_to_checkpoint(["pre_action"], context)

        if isinstance(error, (ExecutionLoopLockError, DeadlockError)):
            if self.task_coordinator and context.get("task_name"):
                try:
                    self.task_coordinator.fail_task(context["task_name"], "deadlock detected during recovery")
                except Exception as exc:
                    logger.warning("Task coordinator escalation failed: %s", exc)
            modified["force_idle"] = True
            modified["idle_duration"] = self.recovery_idle_time
            return True, modified

        if isinstance(error, StaleCheckpointError):
            modified["invalidate_cached_state"] = True
            return True, modified

        if "energy" in str(error).lower() or context.get("energy", 0.0) <= 0:
            modified["force_idle"] = True
            modified["idle_duration"] = self.recovery_idle_time
            modified["recovery_reason"] = "energy_recovery"
            return True, modified

        retry_count = int(modified.get("retry_count", 0))
        if retry_count <= self.max_retries:
            self._bounded_delay(retry_count)
            return True, modified

        return self._backtrack_to_checkpoint(["pre_action"], context)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _base_retry_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        modified = copy.deepcopy(context)
        modified["retry_count"] = int(context.get("retry_count", 0)) + 1
        modified["recovery_mode"] = True
        modified.setdefault("recovery_actions", [])
        return modified

    def _bounded_delay(self, retry_count: int) -> None:
        delay = min(self.max_retry_delay, self.base_retry_delay * (2 ** max(0, retry_count - 1)))
        delay = random.uniform(self.base_retry_delay, max(self.base_retry_delay, delay))
        logger.warning("Recovery retry backoff %.2fs", delay)
        time.sleep(delay)

    def _backtrack_to_checkpoint(
        self,
        checkpoint_types: List[str],
        context: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        checkpoint_types = checkpoint_types if isinstance(checkpoint_types, list) else [checkpoint_types]
        for checkpoint_type in checkpoint_types:
            printer.status("RECOVERY", f"Backtracking to {checkpoint_type} checkpoint", "warning")
            checkpoints = self.memory.find_checkpoints(tag=checkpoint_type, max_age=self.checkpoint_max_age, limit=5)
            if not checkpoints:
                continue

            for checkpoint in checkpoints:
                restored_state = self.memory.restore_checkpoint(checkpoint["id"])
                if not restored_state or not isinstance(restored_state, dict):
                    continue

                modified = copy.deepcopy(context)
                modified.update(restored_state)
                modified["recovery_mode"] = True
                modified["recovery_source"] = checkpoint["id"]
                modified.setdefault("recovery_actions", []).append("backtrack")
                return True, modified

        logger.error("No usable checkpoints available for recovery from %s", checkpoint_types)
        return False, copy.deepcopy(context)

    def _filter_checkpoint_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        excluded = set(self.DEFAULT_EXCLUDED_STATE_KEYS) | set(self.sensitive_keys)
        filtered: Dict[str, Any] = {}
        for key, value in (state or {}).items():
            if key in excluded:
                continue
            filtered[key] = value
        return filtered

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for key, value in (context or {}).items():
            if key in self.sensitive_keys:
                sanitized[key] = "***"
            elif key in self.DEFAULT_EXCLUDED_STATE_KEYS:
                sanitized[key] = "<omitted>"
            else:
                sanitized[key] = value
        return sanitized

    def _failure_id(self, action_name: str, error: Exception) -> str:
        return f"{action_name}:{type(error).__name__}:{int(time.time() * 1000)}"

    def _log_failure(self, action_name: str, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "failure_id": self._failure_id(action_name, error),
            "timestamp": time.time(),
            "action": action_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context_snapshot": self._sanitize_context(context),
            "attempt_index": int(context.get("retry_count", 0)) + 1,
            "recovery_attempted": False,
            "recovery_success": False,
        }
        if len(self.failure_history) >= self.max_history:
            self.failure_history.pop(0)
        self.failure_history.append(record)
        logger.debug("Logged failure for %s: %s", action_name, type(error).__name__)
        self._save_history()
        return record

    def _detect_failure_loop(self, action_name: str, error: Exception) -> bool:
        cutoff = time.time() - self.recent_window_seconds
        recent = [
            record
            for record in self.failure_history
            if record["timestamp"] >= cutoff
            and record["action"] == action_name
            and record["error_type"] == type(error).__name__
        ]
        return len(recent) >= self.failure_loop_threshold

    def _generate_error_suggestion(self, action: str, error_type: str) -> str:
        suggestions = {
            "move_to": {
                "ActionFailureError": "Tighten obstacle modeling and replan against real obstacle sources.",
                "TimeoutError": "Review timeout policy and reduce replan churn before extending duration.",
                "UnreachableTargetError": "Add fallback waypoints and goal relaxation for blocked targets.",
            },
            "pick_object": {
                "ActionFailureError": "Improve grasp scoring and retry with alternate grasp poses.",
                "InvalidContextError": "Verify object perception and required manipulation context fields.",
            },
            "place_object": {
                "ActionFailureError": "Validate place pose occupancy and release sequencing before actuation.",
            },
            "idle": {
                "ActionInterruptionError": "Protect recovery idles from non-critical interruptions.",
            },
        }
        defaults = {
            "ActionFailureError": "Review the action logic and add stronger validation before execution.",
            "TimeoutError": "Investigate resource contention, loop guards, and timeout calibration.",
            "InvalidContextError": "Refresh or reconstruct execution context before retry.",
            "default": "Perform root-cause analysis and add targeted recovery handling for this failure class.",
        }
        return suggestions.get(action, {}).get(error_type, defaults.get(error_type, defaults["default"]))

    def _load_history(self) -> None:
        state = self.memory.get_cache(self.history_key, namespace=self.history_namespace, default={})
        if isinstance(state, dict):
            history = state.get("failure_history", [])
            if isinstance(history, list):
                self.failure_history = history[-self.max_history :]

    def _save_history(self) -> None:
        state = {
            "failure_history": self.failure_history[-self.max_history :],
            "updated_at": time.time(),
        }
        self.memory.set_cache(
            self.history_key,
            state,
            namespace=self.history_namespace,
            ttl=self.history_cache_ttl,
        )


if __name__ == "__main__":
    print("\n=== Running Execution Recovery Tests ===\n")
    printer.status("TEST", "Starting Execution Recovery tests", "info")

    recovery = ExecutionRecovery()

    context = {"current_position": (0, 0), "destination": (10, 10)}
    recovery.create_recovery_checkpoint("pre_movement", context)
    error = ActionFailureError("move_to", "Path blocked by obstacle")
    success, new_context = recovery.handle_failure("move_to", error, context)
    printer.pretty("Movement Recovery", success, "success" if success else "error")
    printer.pretty("Movement Context", new_context, "info")

    context = {"object_position": (1, 1), "current_position": (1.2, 1.2)}
    error = ActionFailureError("pick_object", "Too far from object")
    success, new_context = recovery.handle_failure("pick_object", error, context)
    printer.pretty("Pick Recovery", success, "success" if success else "error")

    report = recovery.analyze_failure_patterns()
    printer.pretty("Failure Analysis", report, "info")

    print("\n=== All recovery tests completed ===")
