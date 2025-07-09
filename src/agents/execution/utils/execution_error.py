
import random
import time
import json
import hashlib
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Error")
printer = PrettyPrinter

class ExecutionErrorType(Enum):
    TIMEOUT = "Task Execution Timeout"
    INVALID_CONTEXT = "Missing or Invalid Execution Context"
    DEADLOCK = "Task Coordination Deadlock"
    TASK_REJECTION = "Task Rejected by Executor"
    ACTION_FAILURE = "Action Failed During Execution"
    RESOURCE_STARVATION = "Execution Resource Starvation"
    MEMORY_OVERFLOW = "Execution Memory Overflow"
    INTERRUPTED_EXECUTION = "Execution Interrupted Unexpectedly"
    ILLEGAL_STATE_TRANSITION = "Illegal Action State Transition"
    STALE_CHECKPOINT = "Outdated or Invalid Checkpoint Access"
    COOKIE_MISMATCH = "Execution Cookie Verification Failed"

class ExecutionError(Exception):
    def __init__(self,
                 error_type: ExecutionErrorType,
                 message: str,
                 severity: str = "medium",
                 context: Optional[Dict[str, Any]] = None,
                 execution_agent_state: Optional[Dict] = None,
                 remediation_guidance: Optional[str] = None):
        super().__init__(message)
        self.context = context or {}
        self.config = load_global_config()
        self.error_config = get_config_section('execution_error')
        self.forensic_hash_algorithm = self.error_config.get('forensic_hash_algorithm')
        self.error_id_hash_algorithm = self.error_config.get('error_id_hash_algorithm')
        self.forensic_hash_salt = self.error_config.get('forensic_hash_salt')
        self.error_id_length = self.error_config.get('error_id_length')
        self.report_format = self.error_config.get('report_format')
        self.error_type = error_type
        self.severity = severity
        self.execution_agent_state = execution_agent_state or {}
        self.remediation_guidance = remediation_guidance
        self.timestamp = time.time()
        self.error_id = self._generate_error_id()
        
        # Generate forensic hash
        self.forensic_hash = self._generate_forensic_hash()

    def _generate_error_id(self) -> str:
        """Generates a unique ID for the error instance."""
        algorithm = self.error_id_hash_algorithm
        length = self.error_id_length
        
        # Get hash function
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        
        # Generate hash
        return hash_func(
            f"{self.timestamp}{random.getrandbits(64)}".encode()
        ).hexdigest()[:length]

    def _generate_forensic_hash(self) -> str:
        """Generate forensic hash with configurable algorithm and salt"""
        algorithm = self.forensic_hash_algorithm
        salt = self.forensic_hash_salt
        
        # Get hash function
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        
        # Prepare data
        context_str = json.dumps(self.context, sort_keys=True, default=str)
        state_str = json.dumps(self.execution_agent_state, sort_keys=True, default=str)
        data = f"{salt}{self.timestamp}{self.error_id}{str(self)}{self.error_type.value}{self.severity}{context_str}{state_str}".encode()
        
        return hash_func(data).hexdigest()

    def to_audit_format(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "error_type": self.error_type.value,
            "severity": self.severity,
            "message": str(self),
            "forensic_hash": self.forensic_hash,
            "context": self.context,
            "execution_agent_state_snapshot": self.execution_agent_state,
            "remediation_guidance": self.remediation_guidance
        }

    def generate_report(self) -> str:
        """Generate comprehensive security incident report"""
        report_format = self.report_format
        
        if report_format == 'json':
            return json.dumps(self.to_audit_format(), indent=2)
        
        # Default to markdown format
        audit_data = self.to_audit_format()
        
        # Format timestamp
        dt = datetime.fromtimestamp(audit_data['timestamp'])
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        report = [
            "# Security Incident Report",
            f"**Generated**: {formatted_time}",
            f"**Error ID**: `{audit_data['error_id']}`",
            f"**Error Type**: {audit_data['error_type']}",
            f"**Severity**: {audit_data['severity'].upper()}",
            "---",
            f"**Message**: {audit_data['message']}",
        ]
        
        # Include forensic hash if configured
        if self.config.get('include_forensic_hash', True):
            report.append(f"**Forensic Hash**: `{audit_data['forensic_hash']}`")
        
        # Include context details if configured
        if self.config.get('include_context', True):
            context_str = json.dumps(audit_data['context'], indent=2)
            report.append("## Context Details")
            report.append(f"```json\n{context_str}\n```")
        
        # Include safety agent state if configured
        if self.config.get('include_safety_agent_state', True):
            state_str = json.dumps(audit_data['safety_agent_state_snapshot'], indent=2)
            report.append("## Safety Agent State")
            report.append(f"```json\n{state_str}\n```")
        
        # Include remediation guidance if configured
        if self.config.get('include_remediation_guidance', True):
            report.append("## Remediation Guidance")
            report.append(audit_data['remediation_guidance'] or "No specific guidance provided.")
        
        return "\n".join(report)

    def __str__(self) -> str:
        return f"[{self.error_type.name} - {self.severity.upper()}] {super().__str__()}"


# --- Specific Error Types ---

class TimeoutError(ExecutionError):
    def __init__(self, task_name: str, allowed_time: float, actual_time: float):
        super().__init__(
            ExecutionErrorType.TIMEOUT,
            f"Task '{task_name}' exceeded allowed time limit ({actual_time:.2f}s > {allowed_time}s).",
            severity="high",
            context={
                "task_name": task_name,
                "allowed_time": allowed_time,
                "actual_time": actual_time
            },
            remediation_guidance="Optimize task or increase timeout threshold. Investigate resource contention or inefficiencies."
        )

class InvalidContextError(ExecutionError):
    def __init__(self, action_name: str, missing_fields: list):
        super().__init__(
            ExecutionErrorType.INVALID_CONTEXT,
            f"Action '{action_name}' failed due to missing context fields: {', '.join(missing_fields)}.",
            severity="medium",
            context={
                "action_name": action_name,
                "missing_fields": missing_fields
            },
            remediation_guidance="Ensure all required context parameters are present and valid before execution."
        )

class DeadlockError(ExecutionError):
    def __init__(self, involved_tasks: list):
        super().__init__(
            ExecutionErrorType.DEADLOCK,
            "Potential deadlock detected among tasks.",
            severity="critical",
            context={
                "involved_tasks": involved_tasks
            },
            remediation_guidance="Check task dependencies and coordination logic. Introduce timeout/abort fallback mechanisms."
        )

class ActionFailureError(ExecutionError):
    def __init__(self, action_name: str, reason: str):
        super().__init__(
            ExecutionErrorType.ACTION_FAILURE,
            f"Action '{action_name}' failed with error: {reason}",
            severity="high",
            context={
                "action_name": action_name,
                "failure_reason": reason
            },
            remediation_guidance="Review action logic and exception handling. Enable retry logic if safe."
        )

class CookieMismatchError(ExecutionError):
    def __init__(self, expected_cookie: str, received_cookie: str):
        super().__init__(
            ExecutionErrorType.COOKIE_MISMATCH,
            "Execution cookie verification failed.",
            severity="critical",
            context={
                "expected_cookie": expected_cookie,
                "received_cookie": received_cookie
            },
            remediation_guidance="Check execution memory synchronization and consistency mechanisms. Invalidate corrupted states."
        )

class StaleCheckpointError(ExecutionError):
    def __init__(self, checkpoint_id: str, last_updated: str):
        super().__init__(
            ExecutionErrorType.STALE_CHECKPOINT,
            f"Attempted to access stale checkpoint: '{checkpoint_id}'",
            severity="medium",
            context={
                "checkpoint_id": checkpoint_id,
                "last_updated": last_updated
            },
            remediation_guidance="Ensure proper checkpoint tagging and expiry. Avoid reuse of outdated execution state."
        )

class ActionInterruptionError(ExecutionError):
    """Raised when an action is intentionally interrupted"""
    level = "MEDIUM"
    code = "ACTION_INTERRUPTION"

class InvalidGridReferenceError(ExecutionError):
    def __init__(self, location: tuple, grid_bounds: tuple):
        super().__init__(
            ExecutionErrorType.ACTION_FAILURE,
            f"Location {location} is outside valid grid bounds {grid_bounds}.",
            severity="high",
            context={
                "location": location,
                "grid_bounds": grid_bounds
            },
            remediation_guidance="Ensure coordinate validation and boundary checks before action execution."
        )

class CorruptedContextStateError(ExecutionError):
    def __init__(self, corrupted_keys: list, context_snapshot: dict):
        super().__init__(
            ExecutionErrorType.INVALID_CONTEXT,
            "Execution context appears corrupted or inconsistent.",
            severity="critical",
            context={
                "corrupted_keys": corrupted_keys,
                "context_snapshot": context_snapshot
            },
            remediation_guidance="Validate context lifecycle and access pattern. Consider deep copying before mutation."
        )

class ExecutionLoopLockError(ExecutionError):
    def __init__(self, repeated_action: str, repeat_count: int):
        super().__init__(
            ExecutionErrorType.DEADLOCK,
            f"Detected execution loop: '{repeated_action}' repeated {repeat_count} times with no progression.",
            severity="critical",
            context={
                "repeated_action": repeated_action,
                "repeat_count": repeat_count
            },
            remediation_guidance="Introduce exit criteria or timeout guards in strategy execution."
        )

class MissingActionHandlerError(ExecutionError):
    def __init__(self, task_name: str):
        super().__init__(
            ExecutionErrorType.TASK_REJECTION,
            f"No action handler registered for task '{task_name}'.",
            severity="high",
            context={
                "task_name": task_name
            },
            remediation_guidance="Register a handler for this task or review task-to-action mappings."
        )

class InvalidTaskTransitionError(ExecutionError):
    def __init__(self, current_state: str, attempted_action: str):
        super().__init__(
            ExecutionErrorType.ILLEGAL_STATE_TRANSITION,
            f"Cannot perform '{attempted_action}' from state '{current_state}'.",
            severity="medium",
            context={
                "current_state": current_state,
                "attempted_action": attempted_action
            },
            remediation_guidance="Verify task prerequisites and update task coordinator logic."
        )

class UnreachableTargetError(ExecutionError):
    def __init__(self, action_name: str, target: tuple, agent_pos: tuple):
        super().__init__(
            ExecutionErrorType.ACTION_FAILURE,
            f"Target {target} unreachable from current position {agent_pos}.",
            severity="high",
            context={
                "action": action_name,
                "target": target,
                "agent_position": agent_pos
            },
            remediation_guidance="Re-evaluate pathfinding or replanning strategy. Consider alternate goal or movement strategy."
        )
