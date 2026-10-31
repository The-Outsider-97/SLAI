"""Thread-safe empirical memory for mathematical optimization evidence.

This module stores reproducibility, warm-start, and solver-performance facts;
it does not store planning policy or model-tuning trials. Solver ranking may
consume these observations only *after* compatibility and availability gates.
That ordering follows the algorithm-selection framing of Rice (1976) and
Kerschke et al. (2019). Runtime/robustness summaries are inspired by Dolan &
Moré (2002), while remaining intentionally lightweight enough to be embedded
inside SLAI.

Persistence is versioned and atomic. Files created by the earlier list-only
prototype are still accepted, so hardening the subsystem does not invalidate
existing local evidence.
"""
from __future__ import annotations

__version__ = "2.3.0"

import json
import math
import os
import threading
import time

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .utils.config_loader import get_config_section, load_global_config
from .utils.optimization_errors import OptimizationValidationError
from .utils.optimization_helpers import stable_fingerprint
from .optimization_math import classify_problem
from .optimization_types import OptimizationProblem, Solution, SolverStatus
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Memory")
printer = PrettyPrinter()

_MEMORY_SCHEMA_VERSION = 1


def _signature_value(value: Any, *, depth: int = 0) -> Any:
    """Return a deterministic, bounded representation for callable signatures."""
    if depth > 4:
        return {"type": type(value).__qualname__}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}", "value": value.value}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (tuple, list)):
        return [_signature_value(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _signature_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _callable_signature(function: Any) -> Mapping[str, Any] | None:
    """Describe callable identity without embedding process-specific addresses."""
    if function is None:
        return None
    payload: dict[str, Any] = {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", getattr(function, "__name__", type(function).__qualname__)),
    }
    code = getattr(function, "__code__", None)
    if code is not None:
        payload["code"] = {
            "bytecode": code.co_code.hex(),
            "consts": _signature_value(code.co_consts),
            "names": tuple(code.co_names),
            "varnames": tuple(code.co_varnames),
            "argcount": code.co_argcount,
            "kwonlyargcount": code.co_kwonlyargcount,
        }
    defaults = getattr(function, "__defaults__", None)
    if defaults:
        payload["defaults"] = _signature_value(defaults)
    closure = getattr(function, "__closure__", None)
    if closure:
        payload["closure"] = [_signature_value(cell.cell_contents) for cell in closure]
    return payload


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert run evidence to deterministic JSON-safe data.

    Optimization decision values are normally primitive numbers/strings. For a
    custom finite-domain object, audit persistence retains a tagged type/repr
    rather than failing the entire save. Such tagged values are deliberately
    not reconstructed as executable Python objects or warm starts on load.
    """
    if depth > 8:
        return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}"}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OptimizationValidationError("optimization memory cannot persist NaN or infinity")
        return value
    if isinstance(value, Enum):
        return value.value
    # Preserve NumPy-like scalar/array diagnostics without importing another
    # dependency into the memory layer.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _json_safe(scalar, depth=depth + 1)
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            listed = list_method()
        except (TypeError, ValueError):
            listed = value
        if listed is not value:
            return _json_safe(listed, depth=depth + 1)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth=depth + 1) for item in value]
    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "__repr__": repr(value),
    }


def _restorable_warm_start(value: Any) -> bool:
    """Return whether a persisted decision mapping can safely be reused."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_restorable_warm_start(item) for item in value)
    if isinstance(value, Mapping):
        if "__type__" in value and "__repr__" in value:
            return False
        return all(isinstance(key, str) and _restorable_warm_start(item) for key, item in value.items())
    return False


@dataclass(frozen=True, slots=True)
class OptimizationRunRecord:
    """Immutable normalized evidence for one solver execution."""

    run_id: str
    timestamp: float
    problem_name: str
    problem_fingerprint: str
    problem_class: str
    solver_name: str
    solver_version: str | None
    status: str
    termination_reason: str
    success: bool
    feasibility: str
    optimality: str
    objective_values: Mapping[str, float]
    max_constraint_violation: float | None
    primal_bound: float | None
    dual_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    runtime_seconds: float | None
    iteration_count: int | None
    node_count: int | None
    function_evaluation_count: int | None
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    primal_solution: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SolverStatistics:
    solver_name: str
    runs: int
    successes: int
    optimal_runs: int
    success_rate: float
    optimal_rate: float
    median_runtime_seconds: float | None
    mean_runtime_seconds: float | None
    median_relative_gap: float | None
    numerical_failures: int
    timeouts: int


class OptimizationMemory:
    """Bounded, thread-safe optimization evidence store with atomic persistence."""

    def __init__(self, config: Mapping[str, Any] | None = None, template: Mapping[str, Any] | None = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("optimization_memory", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.memory_config = section
        self.max_records = int(section.get("max_records", 5000))
        self.stats_window = int(section.get("stats_window", 250))
        self.autosave = bool(section.get("autosave", False))
        raw_path = section.get("persistence_path")
        self.persistence_path = Path(raw_path).expanduser().resolve() if raw_path else None
        if self.max_records <= 0 or self.stats_window <= 0:
            raise OptimizationValidationError("optimization memory limits must be positive")
        self._records: deque[OptimizationRunRecord] = deque(maxlen=self.max_records)
        self._warm_starts: dict[str, Mapping[str, Any]] = {}
        self._lock = threading.RLock()
        if self.persistence_path and self.persistence_path.exists():
            self.load(self.persistence_path)

    @staticmethod
    def problem_fingerprint(problem: OptimizationProblem) -> str:
        """Fingerprint mathematical structure and callback semantics.

        Callback bytecode/defaults/closure values are included when available,
        preventing distinct black-box problems from colliding merely because
        both contain a callable.
        """
        payload = {
            "name": problem.name,
            "problem_class": classify_problem(problem, validate_declared=True).value,
            "known_convex": problem.known_convex,
            "known_smooth": problem.known_smooth,
            "deterministic": problem.deterministic,
            "variables": [
                {
                    "name": variable.name,
                    "kind": variable.domain.kind.value,
                    "lower": variable.domain.lower,
                    "upper": variable.domain.upper,
                    "values": [_signature_value(value) for value in variable.domain.values],
                    "shape": variable.shape,
                }
                for variable in problem.variables
            ],
            "objectives": [
                {
                    "name": objective.name,
                    "sense": objective.sense.value,
                    "linear": dict(objective.linear),
                    "quadratic": {f"{a}|{b}": value for (a, b), value in objective.quadratic.items()},
                    "constant": objective.constant,
                    "function": _callable_signature(objective.function),
                    "gradient": _callable_signature(objective.gradient),
                    "hessian": _callable_signature(objective.hessian),
                }
                for objective in problem.objectives
            ],
            "constraints": [
                {
                    "name": constraint.name,
                    "sense": constraint.sense.value,
                    "rhs": constraint.rhs,
                    "linear": dict(constraint.linear),
                    "function": _callable_signature(constraint.function),
                    "jacobian": _callable_signature(constraint.jacobian),
                    "tolerance": constraint.tolerance,
                }
                for constraint in problem.constraints
            ],
        }
        return stable_fingerprint(payload)

    def record(
        self,
        problem: OptimizationProblem,
        result: Solution,
        *,
        options: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OptimizationRunRecord:
        """Append one normalized run and update its reusable warm start."""
        fingerprint = self.problem_fingerprint(problem)
        run_id = stable_fingerprint(
            {"fingerprint": fingerprint, "solver": result.solver_name, "timestamp_ns": time.time_ns()}
        )[:24]
        record = OptimizationRunRecord(
            run_id=run_id,
            timestamp=time.time(),
            problem_name=problem.name,
            problem_fingerprint=fingerprint,
            problem_class=classify_problem(problem, validate_declared=True).value,
            solver_name=result.solver_name,
            solver_version=result.solver_version,
            status=result.status.value,
            termination_reason=result.termination_reason.value,
            success=result.success,
            feasibility=result.feasibility.value,
            optimality=result.optimality.value,
            objective_values=dict(result.objective_values),
            max_constraint_violation=result.max_constraint_violation,
            primal_bound=result.primal_bound,
            dual_bound=result.dual_bound,
            absolute_gap=result.absolute_gap,
            relative_gap=result.relative_gap,
            runtime_seconds=result.runtime_seconds,
            iteration_count=result.iteration_count,
            node_count=result.node_count,
            function_evaluation_count=result.function_evaluation_count,
            options=dict(options or {}),
            metadata={**dict(result.metadata), **dict(metadata or {})},
            primal_solution=None if result.primal_solution is None else dict(result.primal_solution),
            warnings=tuple(result.warnings),
            diagnostics=dict(result.diagnostics),
        )
        with self._lock:
            self._records.append(record)
            if result.success and result.primal_solution is not None:
                self._warm_starts[fingerprint] = dict(result.primal_solution)
            if self.autosave and self.persistence_path is not None:
                self.save(self.persistence_path)
        return record

    def history(
        self,
        *,
        solver_name: str | None = None,
        problem_class: str | None = None,
        problem_fingerprint: str | None = None,
        limit: int | None = None,
    ) -> tuple[OptimizationRunRecord, ...]:
        with self._lock:
            records = list(self._records)
        if solver_name is not None:
            records = [record for record in records if record.solver_name == solver_name]
        if problem_class is not None:
            records = [record for record in records if record.problem_class == problem_class]
        if problem_fingerprint is not None:
            records = [record for record in records if record.problem_fingerprint == problem_fingerprint]
        if limit is not None:
            if limit <= 0:
                return ()
            records = records[-limit:]
        return tuple(records)

    def warm_start(self, problem: OptimizationProblem) -> Mapping[str, Any] | None:
        with self._lock:
            point = self._warm_starts.get(self.problem_fingerprint(problem))
            return None if point is None else dict(point)

    def solver_statistics(self, solver_name: str, *, problem_class: str | None = None) -> SolverStatistics:
        records = list(self.history(solver_name=solver_name, problem_class=problem_class, limit=self.stats_window))
        runs = len(records)
        if runs == 0:
            return SolverStatistics(solver_name, 0, 0, 0, 0.0, 0.0, None, None, None, 0, 0)
        successes = sum(record.success for record in records)
        optimal_runs = sum(record.status == SolverStatus.OPTIMAL.value for record in records)
        runtimes = sorted(record.runtime_seconds for record in records if record.runtime_seconds is not None)
        gaps = sorted(record.relative_gap for record in records if record.relative_gap is not None)

        def median(values: Sequence[float]) -> float | None:
            if not values:
                return None
            middle = len(values) // 2
            return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0

        return SolverStatistics(
            solver_name=solver_name,
            runs=runs,
            successes=successes,
            optimal_runs=optimal_runs,
            success_rate=successes / runs,
            optimal_rate=optimal_runs / runs,
            median_runtime_seconds=median(runtimes),
            mean_runtime_seconds=(sum(runtimes) / len(runtimes)) if runtimes else None,
            median_relative_gap=median(gaps),
            numerical_failures=sum(record.status == SolverStatus.NUMERICAL_FAILURE.value for record in records),
            timeouts=sum(record.status == SolverStatus.TIMEOUT.value for record in records),
        )

    def historical_score(self, solver_name: str, *, problem_class: str | None = None) -> float:
        stats = self.solver_statistics(solver_name, problem_class=problem_class)
        if stats.runs == 0:
            return 0.0
        reliability = 0.7 * stats.success_rate + 0.3 * stats.optimal_rate
        runtime_penalty = (
            0.0
            if stats.median_runtime_seconds is None
            else min(1.0, math.log1p(max(0.0, stats.median_runtime_seconds)) / 10.0)
        )
        failure_penalty = min(
            0.5,
            (stats.numerical_failures + stats.timeouts) / max(1, stats.runs) * 0.5,
        )
        return max(-1.0, min(1.0, reliability - 0.2 * runtime_penalty - failure_penalty))

    def best_known(self, problem: OptimizationProblem) -> OptimizationRunRecord | None:
        records = [
            record
            for record in self.history(problem_fingerprint=self.problem_fingerprint(problem))
            if record.success
        ]
        if not records:
            return None
        if len(problem.objectives) != 1:
            return records[-1]
        objective = problem.objectives[0]
        values = [(record.objective_values.get(objective.name), record) for record in records]
        values = [(value, record) for value, record in values if value is not None]
        if not values:
            return records[-1]
        reverse = objective.sense.value == "maximize"
        return sorted(values, key=lambda item: item[0], reverse=reverse)[0][1]

    def save(self, path: Path | str | None = None) -> Path:
        """Atomically persist bounded evidence using a versioned JSON schema."""
        target = Path(path).expanduser().resolve() if path is not None else self.persistence_path
        if target is None:
            raise OptimizationValidationError("no optimization memory persistence path configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = [_json_safe(asdict(record)) for record in self._records]
        payload = {
            "schema_version": _MEMORY_SCHEMA_VERSION,
            "created_at": time.time(),
            "records": records,
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise OptimizationValidationError(
                "failed to persist optimization memory",
                context={"path": str(target), "error": str(exc)},
            ) from exc
        return target

    @staticmethod
    def _record_from_payload(item: Mapping[str, Any]) -> OptimizationRunRecord:
        """Normalize current and legacy persisted records."""
        data = dict(item)
        data.setdefault("options", {})
        data.setdefault("metadata", {})
        data.setdefault("primal_solution", None)
        data.setdefault("warnings", ())
        data.setdefault("diagnostics", {})
        data["warnings"] = tuple(str(value) for value in data.get("warnings", ()) or ())
        primal = data.get("primal_solution")
        if primal is not None and not isinstance(primal, Mapping):
            data["primal_solution"] = None
        return OptimizationRunRecord(**data)

    def load(self, path: Path | str) -> int:
        """Load evidence, accepting both the current schema and legacy list files."""
        target = Path(path).expanduser().resolve()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OptimizationValidationError(
                "failed to load optimization memory",
                context={"path": str(target), "error": str(exc)},
            ) from exc

        if isinstance(raw, list):
            raw_records = raw
        elif isinstance(raw, Mapping):
            version = raw.get("schema_version")
            if version != _MEMORY_SCHEMA_VERSION:
                raise OptimizationValidationError(
                    "unsupported optimization memory schema version",
                    context={"schema_version": version, "supported": _MEMORY_SCHEMA_VERSION},
                )
            raw_records = raw.get("records")
            if not isinstance(raw_records, list):
                raise OptimizationValidationError("optimization memory records must be a list")
        else:
            raise OptimizationValidationError("optimization memory file must contain a mapping or legacy list")

        records: list[OptimizationRunRecord] = []
        for item in raw_records:
            if not isinstance(item, Mapping):
                raise OptimizationValidationError("optimization memory record must be a mapping")
            try:
                records.append(self._record_from_payload(item))
            except (TypeError, ValueError) as exc:
                raise OptimizationValidationError(
                    "invalid optimization memory record",
                    context={"record": dict(item), "error": str(exc)},
                ) from exc

        retained = records[-self.max_records :]
        with self._lock:
            self._records.clear()
            self._warm_starts.clear()
            for record in retained:
                self._records.append(record)
                if (
                    record.success
                    and record.primal_solution is not None
                    and _restorable_warm_start(record.primal_solution)
                ):
                    self._warm_starts[record.problem_fingerprint] = dict(record.primal_solution)
        return len(records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._warm_starts.clear()


__all__ = ["OptimizationMemory", "OptimizationRunRecord", "SolverStatistics"]
