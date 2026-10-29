"""Trusted adapter between the legacy Execution Agent and the command gateway.

One boundary per agent instance. The host must pass a gateway backed by trusted
identity/authorization; otherwise effectful dispatch fails closed. The private
pending context is never derived from Command parameters or an authorization
claim inside a user/task dictionary.
"""
from __future__ import annotations

import copy
from threading import RLock
from time import time
from typing import Any, Callable, Mapping

from .command_gateway import Capability, Command, CommandGateway, CommandRejected, Outcome
from .execution_guard import authorized_action


class ExecutionBoundary:
    """Per-agent stateful, serial command bridge for registered actions."""

    def __init__(self, gateway: CommandGateway | None, action_registry: Mapping[str, type],
                 *, apply_action: Callable[[Any], None], robot: Any = None) -> None:
        self.gateway = gateway
        self._registry = dict(action_registry)  # freeze initial trusted action classes
        self._apply_action = apply_action
        self.robot = robot
        self._lock = RLock()
        self._task_id: str | None = None
        self._request_context: Any = None
        self._step = 0
        self._pending: tuple[str, dict[str, Any]] | None = None
        if gateway is not None:
            for name in self._registry:
                gateway.register(Capability(
                    name=f"execution.action.{name}",
                    handler=self._perform_authorized,
                    effect="external",  # conservative: side effects may be irreversible
                    risk="critical" if name in {"motor", "ackermann", "stop", "spin", "navigate", "follow_path", "gripper", "joint", "led", "sequence"} else "moderate",
                ))

    def start_task(self, task_id: str, *, request_context: Any) -> None:
        if self.gateway is None or request_context is None:
            raise PermissionError("trusted execution gateway and request context required")
        if not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 256:
            raise CommandRejected("explicit stable task identifier required")
        with self._lock:
            if self._task_id is not None:
                raise CommandRejected("another authorized task is already active")
            self._task_id = task_id
            self._request_context = request_context
            self._step = 0

    def finish_task(self) -> None:
        with self._lock:
            self._pending = None
            self._task_id = None
            self._request_context = None
            self._step = 0

    def dispatch(self, action_name: str, context: Mapping[str, Any], *, deadline: float | None = None) -> None:
        """Dispatch one non-retriable operation; never return success on replay."""
        with self._lock:
            if self.gateway is None or self._task_id is None or self._request_context is None:
                raise PermissionError("trusted command dispatch not established")
            if action_name not in self._registry:
                raise CommandRejected("unregistered action")
            command = Command(
                command_id=f"{self._task_id}:{self._step}:{action_name}",
                idempotency_key=f"{self._task_id}:{self._step}",
                operation=f"execution.action.{action_name}",
                resource=f"execution-task/{self._task_id}",
                parameters={"action": action_name, "step": self._step},
                deadline=deadline,
            )
            self._pending = (action_name, dict(context))
            try:
                result = self.gateway.execute(command, request_context=self._request_context)
            finally:
                self._pending = None
            if result.status != Outcome.SUCCEEDED or result.replayed:
                # Even a prior successful dispatch may have already mutated state.
                # Never re-execute or silently treat a replay as the current step.
                raise CommandRejected(f"dispatch {result.status.value}: {result.reason or 'duplicate or unconfirmed outcome'}")
            self._step += 1

    def _perform_authorized(self, command: Command) -> bool:
        pending = self._pending
        if pending is None or self._task_id is None:
            raise PermissionError("missing trusted pending command")
        name, context = pending
        if (command.operation != f"execution.action.{name}"
                or command.resource != f"execution-task/{self._task_id}"
                or command.parameters != {"action": name, "step": self._step}):
            raise PermissionError("command mismatches prepared action")
        action_type = self._registry.get(name)
        if action_type is None:
            raise CommandRejected("action is not registered")
        private_context = copy.deepcopy({k: v for k, v in context.items() if k != "robot"})
        if self.robot is not None:
            private_context["robot"] = self.robot
            private_context["robot_ready"] = True
        else:
            private_context.pop("robot", None)
            private_context["robot_ready"] = False
        action = action_type(context=private_context)
        # Action retry logic is subordinate to the durable dispatch journal.
        action.retry_attempts = 0
        with authorized_action(command.operation, command.command_id):
            if not action.execute():
                raise RuntimeError("action failed or was interrupted; effects require reconciliation")
            self._apply_action(action)
        return True