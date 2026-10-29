"""Trusted adapter between the legacy Execution Agent and the command gateway.

One boundary per agent instance. The host must pass a gateway backed by trusted
identity/authorization; otherwise effectful dispatch fails closed. The private
pending context is never derived from Command parameters or an authorization
claim inside a user/task dictionary.
"""
from __future__ import annotations

import copy
import math
import threading
import time

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from .command_gateway import Capability, Command, CommandGateway, CommandRejected, Outcome
from .execution_guard import authorized_action, build_action_operation, validate_action_name
from .utils.execution_error import *
from .utils.execution_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Execution Boundary")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_RESOURCE_PREFIX = "execution-task/"
MAX_TASK_ID_LENGTH = 256

HIGH_RISK_ACTIONS = frozenset({
    "motor", "ackermann", "stop", "spin", "navigate",
    "follow_path", "gripper", "joint", "led", "sequence",
})


# ---------------------------------------------------------------------------
# Dispatch statistics
# ---------------------------------------------------------------------------

@dataclass
class DispatchStats:
    """Counters tracking dispatch lifecycle events for diagnostics."""
    tasks_started: int = 0
    tasks_finished: int = 0
    dispatches_attempted: int = 0
    dispatches_succeeded: int = 0
    dispatches_rejected: int = 0
    dispatches_failed: int = 0
    dispatch_replays: int = 0
    last_error: str = ""
    last_error_at: float = 0.0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "tasks_started": self.tasks_started,
            "tasks_finished": self.tasks_finished,
            "dispatches_attempted": self.dispatches_attempted,
            "dispatches_succeeded": self.dispatches_succeeded,
            "dispatches_rejected": self.dispatches_rejected,
            "dispatches_failed": self.dispatches_failed,
            "dispatch_replays": self.dispatch_replays,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _validate_task_id(task_id: str) -> str:
    """Validate a task identifier for use in resource/command IDs."""
    if not isinstance(task_id, str):
        raise CommandRejected("task_id must be a string")
    stripped = task_id.strip()
    if not stripped:
        raise CommandRejected("task_id must be nonempty")
    if len(stripped) > MAX_TASK_ID_LENGTH:
        raise CommandRejected(
            f"task_id exceeds maximum length ({MAX_TASK_ID_LENGTH})"
        )
    if any(ch in stripped for ch in ("\n", "\r", "\x00")):
        raise CommandRejected("task_id contains forbidden control characters")
    return stripped


def build_capability_name(action_name: str) -> str:
    """Return the gateway capability name for an action."""
    return build_action_operation(action_name)


def build_resource_id(task_id: str) -> str:
    """Return the gateway resource identifier for a task."""
    return f"{TASK_RESOURCE_PREFIX}{task_id}"


def build_command_id(task_id: str, step: int, action_name: str) -> str:
    """Return the deterministic gateway command identifier."""
    return f"{task_id}:{int(step)}:{action_name}"


def build_idempotency_key(task_id: str, step: int) -> str:
    """Return the idempotency key that scopes a dispatch to one task step."""
    return f"{task_id}:{int(step)}"


def classify_risk(action_name: str) -> str:
    """Classify the risk class recorded on the capability registration."""
    return "critical" if action_name in HIGH_RISK_ACTIONS else "moderate"


def prepare_private_context(
    context: Mapping[str, Any],
    robot: Any,
) -> Dict[str, Any]:
    """Deep-copy the dispatch context and inject only the guarded robot.

    Strips any robot reference the untrusted task/command supplied, then
    inserts the boundary-owned guarded robot (if any). The resulting dict is
    the *only* context handed to action constructors.
    """
    if not isinstance(context, Mapping):
        raise CommandRejected("action context must be a mapping")
    sanitized: Dict[str, Any] = {
        key: copy.deepcopy(value)
        for key, value in context.items()
        if key != "robot"
    }
    if robot is not None:
        sanitized["robot"] = robot
        sanitized["robot_ready"] = True
    else:
        sanitized.pop("robot", None)
        sanitized["robot_ready"] = False
    return sanitized


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------

class ExecutionBoundary:
    """Per-agent stateful, serial command bridge for registered actions.

    The boundary is the *only* location where an action is allowed to obtain
    an authorized execution scope. It enforces:

    * a single active task per boundary;
    * a monotonically-increasing ``step`` counter used as the idempotency key;
    * a frozen action registry (unknown actions cannot be dispatched);
    * a private context that strips untrusted robot references.
    """

    def __init__(
        self,
        gateway: Optional[CommandGateway],
        action_registry: Mapping[str, type],
        *,
        apply_action: Callable[[Any], None],
        robot: Any = None,
    ) -> None:
        if not callable(apply_action):
            raise TypeError("apply_action must be callable")
        self.gateway = gateway
        self._registry = self._freeze_registry(action_registry)
        self._apply_action = apply_action
        self.robot = robot
        self._lock = threading.RLock()
        self._task_id: Optional[str] = None
        self._request_context: Any = None
        self._step = 0
        self._pending: Optional[tuple] = None
        self._stats = DispatchStats()

        if gateway is not None:
            self._register_capabilities()
        else:
            logger.warning(
                "ExecutionBoundary created without a gateway; "
                "all effectful dispatch will fail closed"
            )

    # -- Construction helpers ----------------------------------------------

    @staticmethod
    def _freeze_registry(
        action_registry: Mapping[str, type],
    ) -> Dict[str, type]:
        """Validate and copy the injected action registry."""
        if not isinstance(action_registry, Mapping):
            raise TypeError("action_registry must be a mapping")
        frozen: Dict[str, type] = {}
        for name, cls in action_registry.items():
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "action registry keys must be nonempty strings"
                )
            if not isinstance(cls, type):
                raise TypeError(
                    f"action class for {name!r} must be a class, "
                    f"got {type(cls).__name__}"
                )
            frozen[name] = cls
        return frozen

    def _register_capabilities(self) -> None:
        """Register every frozen action as a gateway capability."""
        assert self.gateway is not None
        for name in self._registry:
            capability = Capability(
                name=build_capability_name(name),
                handler=self._perform_authorized,
                effect="external",
                risk=classify_risk(name),
            )
            try:
                self.gateway.register(capability)
            except CommandRejected:
                logger.exception(
                    "Failed to register capability for action=%s", name
                )
                raise
        logger.info(
            "Registered %d action capabilities with the command gateway",
            len(self._registry),
        )

    # -- Lifecycle ---------------------------------------------------------

    def start_task(self, task_id: str, *, request_context: Any) -> None:
        """Open a task scope; only one task may be active at a time."""
        if self.gateway is None:
            raise PermissionError(
                "trusted execution gateway required for task start"
            )
        if request_context is None:
            raise PermissionError("trusted request context required for task start")
        task_id = _validate_task_id(task_id)
        with self._lock:
            if self._task_id is not None:
                raise CommandRejected(
                    "another authorized task is already active "
                    f"({self._task_id!r})"
                )
            self._task_id = task_id
            self._request_context = request_context
            self._step = 0
            self._pending = None
            self._stats.tasks_started += 1
            logger.info(
                "Boundary task started task_id=%s step=0", task_id
            )

    def finish_task(self) -> None:
        """Close the task scope; safe to call when idle."""
        with self._lock:
            task_id = self._task_id
            if task_id is None:
                logger.debug("finish_task called with no active task")
                return
            self._pending = None
            self._task_id = None
            self._request_context = None
            self._step = 0
            self._stats.tasks_finished += 1
            logger.info("Boundary task finished task_id=%s", task_id)

    # -- Dispatch ----------------------------------------------------------

    def dispatch(
        self,
        action_name: str,
        context: Mapping[str, Any],
        *,
        deadline: Optional[float] = None,
    ) -> None:
        """Dispatch one non-retriable operation; never return success on replay."""
        try:
            action_name = validate_action_name(action_name)
        except ValueError as exc:
            raise CommandRejected(str(exc)) from exc

        if action_name not in self._registry:
            raise CommandRejected(
                f"unregistered action: {action_name!r}"
            )
        if not isinstance(context, Mapping):
            raise CommandRejected("action context must be a mapping")
        if deadline is not None:
            if (not isinstance(deadline, (int, float))
                    or not math.isfinite(float(deadline))):
                raise CommandRejected(
                    "deadline must be a finite unix timestamp"
                )

        with self._lock:
            if self.gateway is None:
                raise PermissionError(
                    "trusted command dispatch not established"
                )
            if self._task_id is None or self._request_context is None:
                raise PermissionError("no active authorized task")

            task_id = self._task_id
            step = self._step
            command = Command(
                command_id=build_command_id(task_id, step, action_name),
                idempotency_key=build_idempotency_key(task_id, step),
                operation=build_capability_name(action_name),
                resource=build_resource_id(task_id),
                parameters={"action": action_name, "step": step},
                deadline=float(deadline) if deadline is not None else None,
            )
            self._pending = (action_name, dict(context))
            self._stats.dispatches_attempted += 1
            logger.debug(
                "Dispatching action=%s task_id=%s step=%d command_id=%s",
                action_name,
                task_id,
                step,
                command.command_id,
            )
            try:
                result = self.gateway.execute(
                    command, request_context=self._request_context
                )
            finally:
                self._pending = None

            self._process_dispatch_result(action_name, result)
            self._step += 1

    def _process_dispatch_result(self, action_name: str, result: Any) -> None:
        """Classify a dispatch outcome and raise on any non-success status."""
        if getattr(result, "replayed", False):
            self._stats.dispatch_replays += 1

        if (
            result.status == Outcome.SUCCEEDED
            and not getattr(result, "replayed", False)
        ):
            self._stats.dispatches_succeeded += 1
            logger.debug(
                "Dispatch succeeded action=%s status=%s",
                action_name,
                result.status.value,
            )
            return

        if result.status in (Outcome.DENIED, Outcome.REJECTED):
            self._stats.dispatches_rejected += 1
        else:
            self._stats.dispatches_failed += 1

        reason = result.reason or "duplicate or unconfirmed outcome"
        self._stats.last_error = f"{result.status.value}: {reason}"
        self._stats.last_error_at = time.time()
        logger.error(
            "Dispatch rejected action=%s status=%s reason=%s replayed=%s",
            action_name,
            result.status.value,
            reason,
            getattr(result, "replayed", False),
        )
        raise CommandRejected(
            f"dispatch {result.status.value}: {reason}"
        )

    # -- Handler (invoked from the gateway) --------------------------------

    def _perform_authorized(self, command: Command) -> bool:
        """Gateway-side handler; the only place that opens an action scope."""
        pending = self._pending
        if pending is None or self._task_id is None:
            logger.error(
                "Handler invoked without pending action command_id=%s",
                getattr(command, "command_id", "<unknown>"),
            )
            raise PermissionError("missing trusted pending command")

        name, context = pending
        expected_operation = build_capability_name(name)
        expected_resource = build_resource_id(self._task_id)
        expected_parameters = {"action": name, "step": self._step}

        if (
            command.operation != expected_operation
            or command.resource != expected_resource
            or command.parameters != expected_parameters
        ):
            logger.error(
                "Command mismatch action=%s expected_step=%d command_id=%s",
                name,
                self._step,
                getattr(command, "command_id", "<unknown>"),
            )
            raise PermissionError("command mismatches prepared action")

        action_type = self._registry.get(name)
        if action_type is None:
            raise CommandRejected(f"action {name!r} is not registered")

        private_context = prepare_private_context(context, self.robot)

        try:
            action = action_type(context=private_context)
        except Exception:
            logger.exception(
                "Action construction failed action=%s", name
            )
            raise

        # Action retry logic is subordinate to the durable dispatch journal.
        action.retry_attempts = 0

        with authorized_action(
            command.operation,
            command.command_id,
            task_id=self._task_id,
            step=self._step,
        ):
            try:
                executed = action.execute()
            except Exception:
                logger.exception(
                    "Action execution raised action=%s", name
                )
                raise
            if not executed:
                raise RuntimeError(
                    "action failed or was interrupted; "
                    "effects require reconciliation"
                )
            self._apply_action(action)
        return True

    # -- Diagnostics -------------------------------------------------------

    def supports_action(self, action_name: str) -> bool:
        """Return whether an action is currently registered and dispatchable."""
        if not isinstance(action_name, str):
            return False
        return action_name in self._registry

    def get_active_task_id(self) -> Optional[str]:
        """Return the currently active task identifier, if any."""
        with self._lock:
            return self._task_id

    def get_current_step(self) -> int:
        """Return the next step index that will be dispatched."""
        with self._lock:
            return self._step

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of dispatch statistics."""
        with self._lock:
            return self._stats.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        """Return a full diagnostic snapshot for operational inspection."""
        with self._lock:
            return {
                "gateway_attached": self.gateway is not None,
                "robot_attached": self.robot is not None,
                "active_task_id": self._task_id,
                "step": self._step,
                "registered_actions": sorted(self._registry),
                "stats": self._stats.snapshot(),
            }

    def __repr__(self) -> str:
        with self._lock:
            task = self._task_id or "<idle>"
            gateway_state = "attached" if self.gateway else "absent"
            return (
                f"ExecutionBoundary(gateway={gateway_state}, "
                f"task={task}, step={self._step}, "
                f"actions={len(self._registry)})"
            )


__all__ = [
    "DispatchStats",
    "ExecutionBoundary",
    "HIGH_RISK_ACTIONS",
    "TASK_RESOURCE_PREFIX",
    "build_capability_name",
    "build_command_id",
    "build_idempotency_key",
    "build_resource_id",
    "classify_risk",
    "prepare_private_context",
]