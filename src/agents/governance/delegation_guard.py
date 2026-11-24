"""Governed delegation guard (audit finding 4.6, GDPR Art. 25 / 32, EU AI Act Art. 14 & 15).

Delegation between agents previously resolved ``getattr(agent, action)`` for any string,
so a caller could invoke private methods, dunder methods, lifecycle internals or injected
attributes. This guard makes an action delegable only if it is

1. a plain public identifier (no leading underscore, no dunder, bounded length),
2. a function defined on the target's class (not an instance attribute, property or
   dynamically attached callable), and
3. not on the non-delegable list (lifecycle/infrastructure methods), and
4. approved by the installed authoriser (e.g. the Governance Agent capability registry),
   when one is installed.

Stdlib only; usable from the orchestrator, BaseAgent and the command gateway.
"""
from __future__ import annotations

import inspect
import re
from threading import RLock
from typing import Any, Callable, Optional

_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

#: Infrastructure/lifecycle methods are never remotely delegable, whatever the registry says.
NON_DELEGABLE = frozenset({
    "shutdown", "close", "stop", "kill", "terminate", "reset", "register", "unregister",
    "set_config", "load_config", "save_state", "load_state", "attach_providers",
})

Authorizer = Callable[[str, str, str], None]

_lock = RLock()
_authorizer: Optional[Authorizer] = None


class DelegationDenied(PermissionError):
    """Raised when a delegated action is not permitted."""


def install_authorizer(authorizer: Optional[Authorizer]) -> None:
    """Install (or clear) the process-wide authoriser ``(caller, target, action) -> None``.

    The authoriser must raise to deny. Exceptions from a faulty authoriser deny as well.
    """
    global _authorizer
    with _lock:
        _authorizer = authorizer


def check_delegable(target: Any, *, target_name: str, action: Any, caller: str = "unknown") -> Callable[..., Any]:
    """Return the bound method for ``action`` or raise :class:`DelegationDenied`."""
    if not isinstance(action, str) or not _IDENT.match(action):
        raise DelegationDenied(f"Action {action!r} is not a valid public action name")
    if action in NON_DELEGABLE:
        raise DelegationDenied(f"Action '{action}' is infrastructure and cannot be delegated")
    cls_attr = inspect.getattr_static(type(target), action, None)
    if cls_attr is None or isinstance(cls_attr, property) or not inspect.isfunction(
        cls_attr.__func__ if isinstance(cls_attr, (classmethod, staticmethod)) else cls_attr
    ):
        raise DelegationDenied(f"'{target_name}' exposes no delegable action '{action}'")
    with _lock:
        authorizer = _authorizer
    if authorizer is not None:
        try:
            authorizer(caller, target_name, action)
        except DelegationDenied:
            raise
        except Exception as exc:  # fail closed: a broken authoriser denies
            raise DelegationDenied(f"Delegation authoriser failed closed: {type(exc).__name__}") from exc
    return getattr(target, action)
