"""Internal execution-scoping guard for registered SLAI actions.

This is defense in depth against accidental or command-data-originated bypass.
It is NOT a Python sandbox: malicious code with access to privileged objects and
interpreter imports must be isolated at the process/OS capability boundary.
Only trusted gateway handlers should enter an execution scope.
"""
from __future__ import annotations

import threading
import time

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .utils.execution_error import *
from .utils.execution_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Execution Guard")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Operation namespace constants
# ---------------------------------------------------------------------------

ACTION_OPERATION_PREFIX = "execution.action."
ROBOT_OPERATION_PREFIX = "execution.robot."
SENSOR_OPERATION_PREFIX = "execution.sensor."
ADMIN_OPERATION_PREFIX = "execution.admin."

_ACTION_NAME_MAX_LENGTH = 128
_FORBIDDEN_NAME_CHARS = ("\n", "\r", "\x00")


# ---------------------------------------------------------------------------
# Scope metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScopeInfo:
    """Immutable snapshot of the currently active execution scope.

    Exposed to callers that need audit-level provenance (task, step, trace).
    Never hand this object to untrusted model-selected tools.
    """
    operation: str
    command_id: str
    action_name: Optional[str] = None
    task_id: Optional[str] = None
    step: int = 0
    trace_id: str = ""
    entered_at: float = field(default_factory=time.time)

    @property
    def is_action_scope(self) -> bool:
        return self.operation.startswith(ACTION_OPERATION_PREFIX)

    @property
    def is_robot_scope(self) -> bool:
        return self.operation.startswith(ROBOT_OPERATION_PREFIX)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "command_id": self.command_id,
            "action_name": self.action_name,
            "task_id": self.task_id,
            "step": self.step,
            "trace_id": self.trace_id,
            "entered_at": self.entered_at,
        }


_CURRENT: ContextVar[Optional[ScopeInfo]] = ContextVar("slai_authorized_action", default=None)


# ---------------------------------------------------------------------------
# Scope statistics (thread-safe, diagnostic only)
# ---------------------------------------------------------------------------

_scope_stats_lock = threading.RLock()
_scope_stats: dict[str, int] = {
    "scopes_entered": 0,
    "scopes_exited": 0,
    "denials": 0,
    "nested_rejections": 0,
    "robot_calls_denied": 0,
}


def _record_scope_event(kind: str) -> None:
    with _scope_stats_lock:
        if kind in _scope_stats:
            _scope_stats[kind] += 1


def get_scope_stats() -> dict[str, int]:
    """Return a snapshot of guard statistics for diagnostics."""
    with _scope_stats_lock:
        return dict(_scope_stats)


def reset_scope_stats() -> None:
    """Reset guard statistics; intended for tests and diagnostics."""
    with _scope_stats_lock:
        for key in _scope_stats:
            _scope_stats[key] = 0


# ---------------------------------------------------------------------------
# Operation-ID and name validation helpers
# ---------------------------------------------------------------------------

def validate_action_name(action_name: str) -> str:
    """Validate an action name for use inside an operation identifier.

    Raises ``ValueError`` with a precise reason rather than silently
    truncating or replacing characters (fail-closed).
    """
    if not isinstance(action_name, str):
        raise ValueError("action_name must be a string")
    stripped = action_name.strip()
    if not stripped:
        raise ValueError("action_name must be nonempty")
    if len(stripped) > _ACTION_NAME_MAX_LENGTH:
        raise ValueError(
            f"action_name exceeds maximum length ({_ACTION_NAME_MAX_LENGTH})"
        )
    if any(ch in stripped for ch in _FORBIDDEN_NAME_CHARS):
        raise ValueError("action_name contains forbidden control characters")
    return stripped


def build_action_operation(action_name: str) -> str:
    """Return the canonical operation identifier for an action."""
    return f"{ACTION_OPERATION_PREFIX}{validate_action_name(action_name)}"


def parse_operation(operation: str) -> Optional[str]:
    """Extract the action name from an action operation, or return ``None``."""
    if not isinstance(operation, str):
        return None
    if operation.startswith(ACTION_OPERATION_PREFIX):
        remainder = operation[len(ACTION_OPERATION_PREFIX):]
        return remainder or None
    return None


# ---------------------------------------------------------------------------
# Scope lifecycle
# ---------------------------------------------------------------------------

@contextmanager
def authorized_action(
    operation: str,
    command_id: str,
    *,
    task_id: Optional[str] = None,
    step: int = 0,
    trace_id: str = "",
) -> Iterator[ScopeInfo]:
    """Open an execution scope for the duration of one gateway dispatch.

    Only a trusted gateway handler should enter this scope. Do not expose
    this helper (or the resulting :class:`ScopeInfo`) to model-selected tools
    or untrusted extension code.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise PermissionError("execution scope requires a registered operation")
    if not isinstance(command_id, str) or not command_id.strip():
        raise PermissionError("execution scope requires a trusted command identifier")

    existing = _CURRENT.get()
    if existing is not None:
        # Reject nested scopes: they hide misuse and complicate provenance.
        _record_scope_event("nested_rejections")
        logger.error(
            "Nested execution scope rejected: active=%s incoming=%s",
            existing.operation,
            operation,
        )
        raise PermissionError(f"nested execution scope rejected (active={existing.operation!r})")

    info = ScopeInfo(
        operation=operation,
        command_id=command_id,
        action_name=parse_operation(operation),
        task_id=task_id,
        step=int(step),
        trace_id=str(trace_id),
    )
    token = _CURRENT.set(info)
    _record_scope_event("scopes_entered")
    logger.debug(
        "Entered execution scope operation=%s command_id=%s task_id=%s step=%s",
        operation,
        command_id,
        task_id,
        step,
    )
    try:
        yield info
    finally:
        _CURRENT.reset(token)
        _record_scope_event("scopes_exited")
        logger.debug("Exited execution scope operation=%s", operation)


def current_scope() -> Optional[ScopeInfo]:
    """Return the active scope snapshot, or ``None`` when outside a scope."""
    return _CURRENT.get()


def is_within_scope() -> bool:
    """Return whether the calling context is inside a dispatch scope."""
    return _CURRENT.get() is not None


def require_within_scope() -> ScopeInfo:
    """Return the current scope, raising ``PermissionError`` when absent."""
    info = _CURRENT.get()
    if info is None:
        _record_scope_event("denials")
        logger.warning("Operation requires an execution scope; none active")
        raise PermissionError("operation requires a trusted gateway dispatch scope")
    return info


# ---------------------------------------------------------------------------
# Capability requirements
# ---------------------------------------------------------------------------

def require_authorized_action(action_name: str) -> None:
    """Require that ``action_name`` is executing inside its own action scope.

    Any mismatch (no scope, wrong action, wrong namespace) is a hard fail.
    """
    expected = build_action_operation(action_name)
    info = _CURRENT.get()
    if info is None or info.operation != expected:
        _record_scope_event("denials")
        actual = info.operation if info else "<no scope>"
        logger.warning(
            "Unauthorized action access: action=%s expected_operation=%s actual=%s",
            action_name,
            expected,
            actual,
        )
        raise PermissionError(f"action {action_name!r} requires a trusted command gateway dispatch")


def require_authorized_robot() -> ScopeInfo:
    """Require that a robot capability call is occurring from a scope."""
    info = _CURRENT.get()
    if info is None:
        _record_scope_event("denials")
        _record_scope_event("robot_calls_denied")
        logger.warning("Unauthorized robot access: no execution scope active")
        raise PermissionError("robot operation requires trusted gateway dispatch")
    return info


def require_operation(operation_prefix: str) -> ScopeInfo:
    """Require that the active scope's operation begins with ``operation_prefix``.

    Useful for future sensor/admin capabilities that are not action-bound.
    """
    if not isinstance(operation_prefix, str) or not operation_prefix:
        raise ValueError("operation_prefix must be a nonempty string")
    info = _CURRENT.get()
    if info is None or not info.operation.startswith(operation_prefix):
        _record_scope_event("denials")
        actual = info.operation if info else "<no scope>"
        logger.warning(
            "Operation scope mismatch: required_prefix=%s actual=%s",
            operation_prefix,
            actual,
        )
        raise PermissionError(
            f"operation requires scope prefix {operation_prefix!r}"
        )
    return info


# ---------------------------------------------------------------------------
# Guarded robot adapter
# ---------------------------------------------------------------------------

class GuardedRobotAdapter:
    """Narrow facade for approved robot capability calls inside a gateway scope.

    The raw adapter must not be independently supplied to untrusted code; this
    guard cannot secure calls made to an independently held raw adapter
    reference.
    """

    __slots__ = ("__adapter", "__capabilities_cache")

    def __init__(self, adapter: object) -> None:
        if adapter is None:
            raise TypeError("adapter cannot be None")
        object.__setattr__(self, "_GuardedRobotAdapter__adapter", adapter)
        object.__setattr__(self, "_GuardedRobotAdapter__capabilities_cache", {})

    # -- Capability introspection -------------------------------------------

    def supports_capability(self, name: str) -> bool:
        if not isinstance(name, str) or not name:
            return False
        cache = self.__capabilities_cache
        if name in cache:
            return cache[name]
        adapter = self.__adapter
        checker = getattr(adapter, "supports_capability", None)
        if callable(checker):
            supported = bool(checker(name))
        else:
            supported = callable(getattr(adapter, name, None))
        cache[name] = supported
        return supported

    def __getattr__(self, name: str):
        if name.startswith("_") or name in {"__class__", "__dict__"}:
            raise AttributeError(name)
        adapter = self.__adapter
        method = getattr(adapter, name, None)
        if not callable(method) or not self.supports_capability(name):
            raise AttributeError(name)

        def guarded(*args, **kwargs):
            require_authorized_robot()
            try:
                return method(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Guarded robot capability %s raised %s: %s",
                    name,
                    type(exc).__name__,
                    exc,
                )
                raise

        return guarded

    def __repr__(self) -> str:
        return f"GuardedRobotAdapter({type(self.__adapter).__name__})"


__all__ = [
    "ACTION_OPERATION_PREFIX",
    "ROBOT_OPERATION_PREFIX",
    "SENSOR_OPERATION_PREFIX",
    "ADMIN_OPERATION_PREFIX",
    "ScopeInfo",
    "GuardedRobotAdapter",
    "authorized_action",
    "build_action_operation",
    "current_scope",
    "get_scope_stats",
    "is_within_scope",
    "parse_operation",
    "require_authorized_action",
    "require_authorized_robot",
    "require_operation",
    "require_within_scope",
    "reset_scope_stats",
    "validate_action_name",
]