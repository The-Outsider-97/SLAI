"""Optional solver backend adapters and deterministic backend registry."""

from __future__ import annotations

import importlib
import importlib.util
import threading
import time

from collections.abc import Callable
from typing import Any

from logs.logger import get_logger

from .solver import (
    BOOL_SORT,
    NamedConstraint,
    SolverBackend,
    SolverCapabilities,
    SolverRequest,
    SolverResponse,
    SolverStatus,
    SortKind,
    Term,
    TermOp,
    collect_free_symbols,
)
from ..utils.verification_errors import (
    SolverBackendError,
    SolverUnavailableError,
    VerificationError,
)


logger = get_logger("Verification Solver Backends")


_SUPPORTED_OPERATORS = frozenset(TermOp)


class Z3Backend:
    """Optional Z3 adapter.

    Z3 is discovered and imported only when this backend is selected.  ``Any``
    is intentionally confined to this third-party boundary because Z3's Python
    native expression hierarchy is not part of SLAI's type system.
    """

    @property
    def name(self) -> str:
        return "z3"

    @property
    def available(self) -> bool:
        try:
            return importlib.util.find_spec("z3") is not None
        except (ImportError, ValueError):
            return False

    @property
    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            sorts=frozenset({SortKind.BOOL, SortKind.INT, SortKind.REAL}),
            operators=_SUPPORTED_OPERATORS,
            quantifiers=True,
            models=True,
            unsat_cores=True,
            assumptions=True,
        )

    def _load(self) -> Any:
        if not self.available:
            raise SolverUnavailableError(
                "Z3 backend is not installed",
                context={"backend": self.name, "optional_package": "z3-solver"},
            )
        try:
            return importlib.import_module("z3")
        except (ImportError, ModuleNotFoundError) as exc:
            raise SolverUnavailableError(
                "Z3 backend could not be imported",
                context={"backend": self.name},
                cause=exc,
            ) from exc

    def check(self, request: SolverRequest) -> SolverResponse:
        if not isinstance(request, SolverRequest):
            raise SolverBackendError("Z3 backend received an invalid SolverRequest")
        z3 = self._load()
        started = time.monotonic()
        logger.debug(
            "Starting solver check | backend=%s | constraints=%d | assumptions=%d | quantified=%s",
            self.name,
            len(request.constraints),
            len(request.assumptions),
            request.quantified,
        )
        try:
            solver = z3.Solver()
            if request.timeout_seconds is not None:
                solver.set(timeout=max(1, int(request.timeout_seconds * 1000)))

            free_cache: dict[tuple[str, SortKind], Any] = {}
            tracked_names: dict[str, str] = {}

            def native_sort(kind: SortKind) -> Any:
                if kind is SortKind.BOOL:
                    return z3.BoolSort()
                if kind is SortKind.INT:
                    return z3.IntSort()
                if kind is SortKind.REAL:
                    return z3.RealSort()
                raise SolverBackendError(
                    "Z3 adapter received an unsupported sort",
                    context={"sort": kind.value},
                )

            def native_symbol(term: Term, local: dict[tuple[str, SortKind], Any]) -> Any:
                key = (term.name or "", term.sort.kind)
                if key in local:
                    return local[key]
                if key not in free_cache:
                    free_cache[key] = z3.Const(term.name, native_sort(term.sort.kind))
                return free_cache[key]

            def translate(term: Term, local: dict[tuple[str, SortKind], Any] | None = None) -> Any:
                local = {} if local is None else local
                op = term.op
                if op is TermOp.SYMBOL:
                    return native_symbol(term, local)
                if op is TermOp.BOOL_VAL:
                    return z3.BoolVal(term.value)
                if op is TermOp.INT_VAL:
                    return z3.IntVal(term.value)
                if op is TermOp.REAL_VAL:
                    return z3.RealVal(term.value)

                if op in {TermOp.FORALL, TermOp.EXISTS}:
                    nested = dict(local)
                    variables: list[Any] = []
                    for variable in term.bound_variables:
                        native = z3.Const(variable.name, native_sort(variable.sort.kind))
                        nested[(variable.name or "", variable.sort.kind)] = native
                        variables.append(native)
                    body = translate(term.args[0], nested)
                    return z3.ForAll(variables, body) if op is TermOp.FORALL else z3.Exists(variables, body)

                args = [translate(child, local) for child in term.args]
                if op is TermOp.NOT:
                    return z3.Not(args[0])
                if op is TermOp.AND:
                    return z3.And(*args)
                if op is TermOp.OR:
                    return z3.Or(*args)
                if op is TermOp.XOR:
                    expression = args[0]
                    for item in args[1:]:
                        expression = z3.Xor(expression, item)
                    return expression
                if op is TermOp.IMPLIES:
                    return z3.Implies(args[0], args[1])
                if op is TermOp.EQ:
                    return z3.And(*(args[index] == args[index + 1] for index in range(len(args) - 1)))
                if op is TermOp.DISTINCT:
                    return z3.Distinct(*args)
                if op is TermOp.LT:
                    return args[0] < args[1]
                if op is TermOp.LE:
                    return args[0] <= args[1]
                if op is TermOp.GT:
                    return args[0] > args[1]
                if op is TermOp.GE:
                    return args[0] >= args[1]
                if op is TermOp.ADD:
                    return z3.Sum(*args)
                if op is TermOp.SUB:
                    result = args[0]
                    for item in args[1:]:
                        result = result - item
                    return result
                if op is TermOp.MUL:
                    result = args[0]
                    for item in args[1:]:
                        result = result * item
                    return result
                if op is TermOp.DIV:
                    result = args[0]
                    for item in args[1:]:
                        result = result / item
                    return result
                if op is TermOp.NEG:
                    return -args[0]
                if op is TermOp.ITE:
                    return z3.If(args[0], args[1], args[2])
                raise SolverBackendError(
                    "Z3 adapter received an unsupported operator",
                    context={"operator": op.value},
                )

            native_constraints: list[Any] = []
            assumption_literals: list[Any] = []
            all_named = request.constraints + request.assumptions
            if request.produce_unsat_core:
                for index, named in enumerate(all_named):
                    marker_name = f"__slai_verify_{index}"
                    marker = z3.Bool(marker_name)
                    tracked_names[marker_name] = named.name
                    solver.add(z3.Implies(marker, translate(named.term)))
                    assumption_literals.append(marker)
            else:
                native_constraints.extend(translate(item.term) for item in request.constraints)
                solver.add(*native_constraints)
                assumption_literals.extend(translate(item.term) for item in request.assumptions)

            native_result = solver.check(*assumption_literals)
            elapsed = time.monotonic() - started
            if native_result == z3.sat:
                model_payload: dict[str, str] = {}
                if request.produce_model:
                    model = solver.model()
                    free_symbols: list[Term] = []
                    seen: set[tuple[str, SortKind]] = set()
                    for named in all_named:
                        for item in collect_free_symbols(named.term):
                            key = (item.name or "", item.sort.kind)
                            if key not in seen:
                                seen.add(key)
                                free_symbols.append(item)
                    for item in free_symbols:
                        native = free_cache.get((item.name or "", item.sort.kind))
                        if native is not None:
                            value = model.eval(native, model_completion=True)
                            model_payload[item.name or ""] = str(value)
                logger.debug("Solver completed | backend=%s | status=sat | elapsed=%.6f", self.name, elapsed)
                return SolverResponse(
                    status=SolverStatus.SAT,
                    backend=self.name,
                    model=model_payload,
                    elapsed_seconds=elapsed,
                )
            if native_result == z3.unsat:
                core: tuple[str, ...] = ()
                if request.produce_unsat_core:
                    names: list[str] = []
                    for literal in solver.unsat_core():
                        raw = str(literal)
                        if raw in tracked_names:
                            names.append(tracked_names[raw])
                    core = tuple(names)
                logger.debug("Solver completed | backend=%s | status=unsat | elapsed=%.6f", self.name, elapsed)
                return SolverResponse(
                    status=SolverStatus.UNSAT,
                    backend=self.name,
                    unsat_core=core,
                    elapsed_seconds=elapsed,
                )
            reason = str(solver.reason_unknown() or "solver returned unknown")
            logger.warning(
                "Solver returned UNKNOWN | backend=%s | elapsed=%.6f | reason=%s",
                self.name,
                elapsed,
                reason,
            )
            return SolverResponse(
                status=SolverStatus.UNKNOWN,
                backend=self.name,
                reason_unknown=reason,
                elapsed_seconds=elapsed,
            )
        except VerificationError:
            raise
        except Exception as exc:
            raise SolverBackendError(
                "Z3 backend failed during satisfiability checking",
                context={"backend": self.name},
                cause=exc,
            ) from exc


BackendFactory = Callable[[], SolverBackend]


class BackendRegistry:
    """Small explicit, thread-safe registry with deterministic preference order."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}
        self._lock = threading.RLock()

    def register(self, name: str, factory: BackendFactory, *, replace: bool = False) -> None:
        normalized = str(name).strip().lower()
        if not normalized:
            raise SolverUnavailableError("backend registry name must be non-empty")
        if not callable(factory):
            raise SolverUnavailableError(
                "backend factory must be callable",
                context={"backend": normalized},
            )
        with self._lock:
            if normalized in self._factories and not replace:
                raise SolverUnavailableError(
                    "backend is already registered",
                    context={"backend": normalized},
                )
            self._factories[normalized] = factory
        logger.debug("Solver backend registered | backend=%s | replace=%s", normalized, replace)

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))

    def create(self, name: str) -> SolverBackend:
        normalized = str(name).strip().lower()
        with self._lock:
            factory = self._factories.get(normalized)
        if factory is None:
            raise SolverUnavailableError(
                "requested solver backend is not registered",
                context={"backend": normalized, "registered": self.names()},
            )
        try:
            backend = factory()
        except VerificationError:
            raise
        except Exception as exc:
            raise SolverUnavailableError(
                "solver backend factory failed",
                context={"backend": normalized},
                cause=exc,
            ) from exc
        if not isinstance(backend, SolverBackend):
            raise SolverUnavailableError(
                "registered backend does not implement SolverBackend",
                context={"backend": normalized},
            )
        if not backend.available:
            logger.warning("Solver backend unavailable | backend=%s", normalized)
            raise SolverUnavailableError(
                "requested solver backend is unavailable",
                context={"backend": normalized},
            )
        logger.debug("Solver backend created | backend=%s", normalized)
        return backend

    def select(self, preferred: tuple[str, ...]) -> SolverBackend:
        attempts: list[str] = []
        for name in preferred:
            normalized = str(name).strip().lower()
            attempts.append(normalized)
            try:
                backend = self.create(normalized)
                logger.info("Selected verification solver backend | backend=%s", normalized)
                return backend
            except SolverUnavailableError:
                continue
        raise SolverUnavailableError(
            "no preferred solver backend is available",
            context={"preferred_backends": attempts, "registered": self.names()},
        )


def default_backend_registry() -> BackendRegistry:
    """Return a fresh registry; no mutable solver registry is shared globally."""
    registry = BackendRegistry()
    registry.register("z3", Z3Backend)
    return registry


__all__ = [
    "BackendRegistry",
    "Z3Backend",
    "default_backend_registry",
]
