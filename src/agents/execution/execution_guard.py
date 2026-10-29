"""Internal execution-scoping guard for registered SLAI actions.

This is defense in depth against accidental or command-data-originated bypass.
It is NOT a Python sandbox: malicious code with access to privileged objects and
interpreter imports must be isolated at the process/OS capability boundary.
Only trusted gateway handlers should enter an execution scope.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_CURRENT: ContextVar[tuple[str, str] | None] = ContextVar("slai_authorized_action", default=None)


@contextmanager
def authorized_action(operation: str, command_id: str) -> Iterator[None]:
    """Set an operation-scoped context for the duration of one gateway dispatch.

    Do not expose this helper to model-selected tools or untrusted extensions.
    """
    if not operation or not command_id:
        raise PermissionError("execution scope requires a registered operation and command")
    token = _CURRENT.set((operation, command_id))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def require_authorized_action(action_name: str) -> None:
    context = _CURRENT.get()
    if context is None or context[0] != f"execution.action.{action_name}":
        raise PermissionError(f"action {action_name!r} requires a trusted command gateway dispatch")


def require_authorized_robot() -> None:
    if _CURRENT.get() is None:
        raise PermissionError("robot operation requires trusted gateway dispatch")


class GuardedRobotAdapter:
    """Narrow facade for approved robot capability calls from a gateway scope.

    The raw adapter must not be independently supplied to untrusted code; this
    guard cannot secure calls made to an independently held raw adapter reference.
    """

    __slots__ = ("__adapter",)

    def __init__(self, adapter: object) -> None:
        object.__setattr__(self, "_GuardedRobotAdapter__adapter", adapter)

    def supports_capability(self, name: str) -> bool:
        adapter = object.__getattribute__(self, "_GuardedRobotAdapter__adapter")
        checker = getattr(adapter, "supports_capability", None)
        return bool(checker(name)) if callable(checker) else callable(getattr(adapter, name, None))

    def __getattr__(self, name: str):
        if name.startswith("_") or name in {"__class__", "__dict__"}:
            raise AttributeError(name)
        adapter = object.__getattribute__(self, "_GuardedRobotAdapter__adapter")
        method = getattr(adapter, name, None)
        if not callable(method) or not self.supports_capability(name):
            raise AttributeError(name)

        def guarded(*args, **kwargs):
            require_authorized_robot()
            return method(*args, **kwargs)
        return guarded