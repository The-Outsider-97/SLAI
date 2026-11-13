
import random
import time
import json
import hashlib

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any

from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Execution Error")
printer = PrettyPrinter()

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
    COMMAND_REJECTION = "Command Rejected by Gateway"

class SoftInterrupt(Exception):
    """Raised to gracefully interrupt an action without treating it as a hard failure."""
    def __init__(self, reason: str = "external_interrupt", data: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.data = data or {}
        super().__init__(f"SoftInterrupt: {reason}")

    def __str__(self) -> str:
        return f"SoftInterrupt(reason={self.reason}, data={self.data})"

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
        hash_func = getattr(hashlib, algorithm or 'sha256', hashlib.sha256)

        # Generate hash
        return hash_func(
            f"{self.timestamp}{random.getrandbits(64)}".encode()
        ).hexdigest()[:length]

    def _generate_forensic_hash(self) -> str:
        """Generate forensic hash with configurable algorithm and salt"""
        algorithm = self.forensic_hash_algorithm
        salt = self.forensic_hash_salt

        # Get hash function
        hash_func = getattr(hashlib, algorithm or 'sha256', hashlib.sha256)

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

class CorruptedExecutionStateError(ExecutionError):
    def __init__(self, corrupted_state: dict):
        super().__init__(
            ExecutionErrorType.INVALID_CONTEXT,
            "Execution state appears corrupted or inconsistent.",
            severity="critical",
            context={
                "corrupted_state": corrupted_state
            },
            remediation_guidance="Validate execution state lifecycle and access pattern. Consider restoring from a known good checkpoint."
        )

class ExecutionMemoryOverflowError(ExecutionError):
    def __init__(self, memory_limit: int, current_usage: int):
        super().__init__(
            ExecutionErrorType.MEMORY_OVERFLOW,
            f"Execution memory overflow: limit {memory_limit} bytes, current usage {current_usage} bytes.",
            severity="critical",
            context={
                "memory_limit": memory_limit,
                "current_usage": current_usage
            },
            remediation_guidance="Optimize memory usage or increase memory allocation. Consider purging stale data."
        )

class ExecutionTimeoutError(ExecutionError):
    def __init__(self, task_name: str, timeout_duration: float):
        super().__init__(
            ExecutionErrorType.TIMEOUT,
            f"Task '{task_name}' exceeded timeout duration of {timeout_duration} seconds.",
            severity="high",
            context={
                "task_name": task_name,
                "timeout_duration": timeout_duration
            },
            remediation_guidance="Review task execution time and optimize performance. Consider increasing timeout threshold if safe."
        )


class CommandRejected(ExecutionError):
    """Invalid command, capability, idempotency conflict, or permission failure.

    Raised by the command gateway, execution boundary, and (indirectly) the
    execution guard when a command cannot be dispatched. Distinct from
    ``ActionFailureError`` (which is raised by concrete actions) and from
    ``InvalidContextError`` (which is raised by context validation). This class
    is the canonical failure type for the *dispatch pipeline* itself.

    Stable reason codes allow callers to branch on failure class without parsing
    strings. Every rejection path in the gateway/boundary/guard is expected to
    supply one of the ``UPPER_SNAKE`` codes below.
    """

    # -- Stable reason codes ------------------------------------------------
    INVALID_COMMAND = "invalid_command"
    INVALID_PARAMETERS = "invalid_parameters"
    INVALID_ACTION_NAME = "invalid_action_name"
    INVALID_TASK_ID = "invalid_task_id"
    INVALID_CAPABILITY = "invalid_capability"
    UNREGISTERED_CAPABILITY = "unregistered_capability"
    CAPABILITY_ALREADY_REGISTERED = "capability_already_registered"
    AUTHORIZATION_DENIED = "authorization_denied"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    DEADLINE_EXPIRED = "deadline_expired"
    VERSION_CONFLICT = "version_conflict"
    VERSION_CHECK_UNSUPPORTED = "version_check_unsupported"
    REPLAY_DETECTED = "replay_detected"
    INDETERMINATE_OUTCOME = "indeterminate_outcome"
    POSTCONDITION_FAILED = "postcondition_failed"
    RECONCILIATION_DENIED = "reconciliation_denied"
    DISPATCH_STATE_ERROR = "dispatch_state_error"
    NO_ACTIVE_TASK = "no_active_task"
    TASK_ALREADY_ACTIVE = "task_already_active"
    GATEWAY_UNAVAILABLE = "gateway_unavailable"
    UNEXPECTED_ERROR = "unexpected_error"

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "unknown",
        stage: str = "gateway",
        operation: Optional[str] = None,
        resource: Optional[str] = None,
        command_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        principal_id: Optional[str] = None,
        retryable: bool = False,
        severity: str = "high",
        details: Optional[Dict[str, Any]] = None,
        remediation_guidance: Optional[str] = None,
        execution_agent_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        reason_code = str(reason_code or "unknown")
        stage = str(stage or "gateway")

        context: Dict[str, Any] = {
            "reason_code": reason_code,
            "stage": stage,
            "retryable": bool(retryable),
        }
        if operation is not None:
            context["operation"] = str(operation)
        if resource is not None:
            context["resource"] = str(resource)
        if command_id is not None:
            context["command_id"] = str(command_id)
        if idempotency_key is not None:
            context["idempotency_key"] = str(idempotency_key)
        if principal_id is not None:
            context["principal_id"] = str(principal_id)
        if details:
            context["details"] = dict(details)

        super().__init__(
            ExecutionErrorType.COMMAND_REJECTION,
            message,
            severity=severity,
            context=context,
            execution_agent_state=execution_agent_state,
            remediation_guidance=(
                remediation_guidance
                or self._default_remediation(reason_code, retryable)
            ),
        )
        self.reason_code = reason_code
        self.stage = stage
        self.operation = operation
        self.resource = resource
        self.command_id = command_id
        self.idempotency_key = idempotency_key
        self.principal_id = principal_id
        self.retryable = bool(retryable)
        self.details = dict(details or {})

    @staticmethod
    def _default_remediation(reason_code: str, retryable: bool) -> str:
        suggestions = {
            CommandRejected.INVALID_COMMAND: "Validate the command structure before dispatch.",
            CommandRejected.INVALID_PARAMETERS: "Ensure parameters are a bounded, JSON-compatible, string-keyed mapping.",
            CommandRejected.INVALID_ACTION_NAME: "Use a registered, well-formed action name.",
            CommandRejected.INVALID_TASK_ID: "Use a nonempty task identifier up to 256 chars.",
            CommandRejected.INVALID_CAPABILITY: "Inspect the capability registration for invalid effect/risk fields.",
            CommandRejected.UNREGISTERED_CAPABILITY: "Register the capability before dispatching.",
            CommandRejected.CAPABILITY_ALREADY_REGISTERED: "Do not re-register an existing capability; use the original handler.",
            CommandRejected.AUTHORIZATION_DENIED: "Verify principal identity and current permissions.",
            CommandRejected.IDEMPOTENCY_CONFLICT: "Use a new idempotency key or reconcile the earlier command.",
            CommandRejected.DEADLINE_EXPIRED: "Retry with a fresh deadline if the effect is still desired.",
            CommandRejected.VERSION_CONFLICT: "Refresh the resource version and retry.",
            CommandRejected.VERSION_CHECK_UNSUPPORTED: "Provide a version_reader on the capability or omit expected_version.",
            CommandRejected.REPLAY_DETECTED: "Do not re-execute; inspect prior outcome via reconciliation.",
            CommandRejected.INDETERMINATE_OUTCOME: "Reconcile using authoritative external evidence.",
            CommandRejected.POSTCONDITION_FAILED: "Investigate the capability verifier and the handler output.",
            CommandRejected.RECONCILIATION_DENIED: "Reconciliation requires trusted administrator and evidence.",
            CommandRejected.DISPATCH_STATE_ERROR: "Inspect gateway/boundary state; this indicates a pipeline invariant breach.",
            CommandRejected.NO_ACTIVE_TASK: "Call start_task() with a trusted request context first.",
            CommandRejected.TASK_ALREADY_ACTIVE: "Finish the active task before starting a new one.",
            CommandRejected.GATEWAY_UNAVAILABLE: "Attach a trusted command gateway to the boundary.",
            CommandRejected.UNEXPECTED_ERROR: "Inspect the underlying exception and its stage details.",
        }
        base = suggestions.get(reason_code, "Review the command, context, and gateway configuration.")
        if retryable:
            return base + " This failure is considered retryable."
        return base

    def to_audit_format(self) -> Dict[str, Any]:
        data = super().to_audit_format()
        data["command_rejection"] = {
            "reason_code": self.reason_code,
            "stage": self.stage,
            "operation": self.operation,
            "resource": self.resource,
            "command_id": self.command_id,
            "idempotency_key": self.idempotency_key,
            "principal_id": self.principal_id,
            "retryable": self.retryable,
            "details": self.details,
        }
        return data

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} [code={self.reason_code}, stage={self.stage}]"


__all__ = [
    "ExecutionError",
    "ExecutionErrorType",
    "SoftInterrupt",
    "TimeoutError",
    "InvalidContextError",
    "DeadlockError",
    "ActionFailureError",
    "CookieMismatchError",
    "StaleCheckpointError",
    "ActionInterruptionError",
    "InvalidGridReferenceError",
    "CorruptedContextStateError",
    "ExecutionLoopLockError",
    "MissingActionHandlerError",
    "InvalidTaskTransitionError",
    "UnreachableTargetError",
    "CorruptedExecutionStateError",
    "ExecutionMemoryOverflowError",
    "ExecutionTimeoutError",
    "CommandRejected",
]