"""Thread-safe registry primitives for the SLAI optimization subsystem.

The registry layer contains several distinct registries (solver adapters,
mathematical capabilities, and scalarization strategies).  They share the same
concurrency and duplicate-handling semantics through :class:`RegistryStore`
without forcing every registry entry to inherit from a common domain type.

``BaseRegistry`` remains the solver-adapter registry base used by
``SolverRegistry`` and preserves the public API that existed before the generic
store was introduced.
"""
from __future__ import annotations

__version__ = "2.3.0"

import threading

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from ..solvers.base_solver import BaseSolver
from ..utils.optimization_errors import RegistryError
from logs.logger import configure_logging, get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Base Registry")
printer = PrettyPrinter()


T = TypeVar("T")


class RegistryStore(Generic[T]):
    """Small, thread-safe named-object store used by optimization registries.

    The store deliberately owns only registry mechanics: deterministic name
    ordering, replacement semantics, and locking.  Domain validation belongs
    to the registry that uses the store.
    """

    def __init__(self) -> None:
        self._items: dict[str, T] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise RegistryError("registry key must be a non-empty string", context={"name": name})
        return name.strip()

    def register(self, name: str, item: T, *, replace: bool = False) -> T:
        key = self._validate_name(name)
        with self._lock:
            if key in self._items and not replace:
                raise RegistryError("registry key is already registered", context={"name": key})
            self._items[key] = item
            return item

    def unregister(self, name: str) -> bool:
        key = self._validate_name(name)
        with self._lock:
            return self._items.pop(key, None) is not None

    def get(self, name: str) -> T:
        key = self._validate_name(name)
        with self._lock:
            try:
                return self._items[key]
            except KeyError as exc:
                raise RegistryError("unknown registry key", context={"name": key}) from exc

    def contains(self, name: str) -> bool:
        key = self._validate_name(name)
        with self._lock:
            return key in self._items

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._items))

    def values(self) -> tuple[T, ...]:
        with self._lock:
            return tuple(self._items[name] for name in sorted(self._items))

    def items(self) -> tuple[tuple[str, T], ...]:
        with self._lock:
            return tuple((name, self._items[name]) for name in sorted(self._items))

    def update(self, name: str, updater: Callable[[T], T]) -> T:
        """Atomically replace one item using ``updater`` while holding the store lock."""
        key = self._validate_name(name)
        if not callable(updater):
            raise RegistryError("registry updater must be callable", context={"name": key})
        with self._lock:
            try:
                current = self._items[key]
            except KeyError as exc:
                raise RegistryError("unknown registry key", context={"name": key}) from exc
            updated = updater(current)
            self._items[key] = updated
            return updated

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    solver: BaseSolver
    priority: int = 100
    enabled: bool = True
    metadata: dict[str, Any] | None = None


class BaseRegistry:
    """Thread-safe base for registries of concrete ``BaseSolver`` adapters."""

    def __init__(self) -> None:
        self._store: RegistryStore[RegistryEntry] = RegistryStore()

    def register(
        self,
        solver: BaseSolver,
        *,
        priority: int = 100,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> RegistryEntry:
        if not isinstance(solver, BaseSolver):
            raise RegistryError("solver registry accepts BaseSolver instances only", context={"type": type(solver).__name__})
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise RegistryError("solver priority must be an integer", context={"priority": priority})
        entry = RegistryEntry(solver, priority, bool(enabled), dict(metadata or {}))
        return self._store.register(solver.solver_name, entry, replace=replace)

    def unregister(self, solver_name: str) -> bool:
        return self._store.unregister(solver_name)

    def get_entry(self, solver_name: str) -> RegistryEntry:
        try:
            return self._store.get(solver_name)
        except RegistryError as exc:
            raise RegistryError("unknown solver", context={"solver": solver_name}) from exc

    def get(self, solver_name: str) -> BaseSolver:
        return self.get_entry(solver_name).solver

    def names(self) -> tuple[str, ...]:
        return self._store.names()

    def entries(self) -> tuple[RegistryEntry, ...]:
        return self._store.values()

    def set_enabled(self, solver_name: str, enabled: bool) -> None:
        try:
            self._store.update(
                solver_name,
                lambda entry: RegistryEntry(entry.solver, entry.priority, bool(enabled), entry.metadata),
            )
        except RegistryError as exc:
            raise RegistryError("unknown solver", context={"solver": solver_name}) from exc


__all__ = ["BaseRegistry", "RegistryEntry", "RegistryStore"]
