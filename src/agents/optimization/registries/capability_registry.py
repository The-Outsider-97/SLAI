"""Mathematical capability registry for optimization solver adapters.

This registry is deliberately separate from solver *selection*.  It is the
hard-compatibility catalogue: which mathematical structures a registered
solver can represent and execute.  ``SolverRegistry`` may rank only candidates
that this registry declares compatible.

Academic basis
--------------
Rice (1976), *The Algorithm Selection Problem*, separates problem features from
algorithm choice.  Kerschke et al. (2019) formalize feature-based algorithm
selection and algorithm portfolios.  In that spirit, this module records
algorithm capability independently from empirical preference or configuration.

The registry does not benchmark solvers, tune solver options, or choose a
winner.  Those concerns remain with ``SolverRegistry`` and
``OptimizationMemory``.
"""
from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..optimization_math import problem_features
from ..optimization_types import OptimizationProblem, ProblemClass, ProblemFeatures, SolverCapability
from ..utils.optimization_errors import RegistryError
from .base_registry import RegistryStore


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """One solver's declared mathematical capability contract."""

    solver_name: str
    capability: SolverCapability
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.solver_name, str) or not self.solver_name.strip():
            raise RegistryError("capability solver_name must be non-empty")
        if not isinstance(self.capability, SolverCapability):
            raise RegistryError(
                "capability record requires SolverCapability",
                context={"solver": self.solver_name, "type": type(self.capability).__name__},
            )
        if self.capability.name != self.solver_name:
            raise RegistryError(
                "solver capability name must match registry solver name",
                context={"solver": self.solver_name, "capability_name": self.capability.name},
            )
        object.__setattr__(self, "solver_name", self.solver_name.strip())
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class CapabilityAssessment:
    solver_name: str
    compatible: bool
    incompatibilities: tuple[str, ...]
    enabled: bool
    capability: SolverCapability


@dataclass(frozen=True, slots=True)
class CapabilityAudit:
    """Coverage snapshot for a requested set of optimization problem classes."""

    coverage: Mapping[ProblemClass, tuple[str, ...]]
    uncovered: tuple[ProblemClass, ...]
    disabled_only: Mapping[ProblemClass, tuple[str, ...]]
    duplicate_signatures: tuple[tuple[str, ...], ...]

    @property
    def complete(self) -> bool:
        return not self.uncovered


class CapabilityRegistry:
    """Thread-safe catalogue of solver capability declarations.

    Compatibility is a hard mathematical filter.  Availability, priority and
    historical performance are intentionally absent from this registry.
    """

    def __init__(self) -> None:
        self._store: RegistryStore[CapabilityRecord] = RegistryStore()

    def register(
        self,
        solver_name: str,
        capability: SolverCapability,
        *,
        enabled: bool = True,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> CapabilityRecord:
        record = CapabilityRecord(
            solver_name=solver_name,
            capability=capability,
            enabled=enabled,
            metadata=dict(metadata or {}),
        )
        return self._store.register(record.solver_name, record, replace=replace)

    def unregister(self, solver_name: str) -> bool:
        return self._store.unregister(solver_name)

    def get(self, solver_name: str) -> CapabilityRecord:
        try:
            return self._store.get(solver_name)
        except RegistryError as exc:
            raise RegistryError("unknown solver capability", context={"solver": solver_name}) from exc

    def names(self, *, include_disabled: bool = True) -> tuple[str, ...]:
        if include_disabled:
            return self._store.names()
        return tuple(record.solver_name for record in self.records() if record.enabled)

    def records(self) -> tuple[CapabilityRecord, ...]:
        return self._store.values()

    def set_enabled(self, solver_name: str, enabled: bool) -> None:
        try:
            self._store.update(
                solver_name,
                lambda current: CapabilityRecord(
                    solver_name=current.solver_name,
                    capability=current.capability,
                    enabled=enabled,
                    metadata=current.metadata,
                ),
            )
        except RegistryError as exc:
            raise RegistryError("unknown solver capability", context={"solver": solver_name}) from exc

    @staticmethod
    def _features(problem_or_features: OptimizationProblem | ProblemFeatures) -> ProblemFeatures:
        if isinstance(problem_or_features, ProblemFeatures):
            return problem_or_features
        if isinstance(problem_or_features, OptimizationProblem):
            return problem_features(problem_or_features)
        raise RegistryError(
            "capability assessment requires OptimizationProblem or ProblemFeatures",
            context={"type": type(problem_or_features).__name__},
        )

    def assess(
        self,
        solver_name: str,
        problem_or_features: OptimizationProblem | ProblemFeatures,
    ) -> CapabilityAssessment:
        record = self.get(solver_name)
        features = self._features(problem_or_features)
        reasons = record.capability.incompatibilities(features)
        return CapabilityAssessment(
            solver_name=record.solver_name,
            compatible=not reasons,
            incompatibilities=reasons,
            enabled=record.enabled,
            capability=record.capability,
        )

    def compatible(
        self,
        problem_or_features: OptimizationProblem | ProblemFeatures,
        *,
        include_disabled: bool = False,
    ) -> tuple[CapabilityRecord, ...]:
        features = self._features(problem_or_features)
        matches: list[CapabilityRecord] = []
        for record in self.records():
            if not include_disabled and not record.enabled:
                continue
            if not record.capability.incompatibilities(features):
                matches.append(record)
        return tuple(matches)

    def supporting(
        self,
        problem_class: ProblemClass,
        *,
        include_disabled: bool = False,
    ) -> tuple[CapabilityRecord, ...]:
        if not isinstance(problem_class, ProblemClass):
            raise RegistryError("supporting() requires ProblemClass", context={"value": repr(problem_class)})
        return tuple(
            record
            for record in self.records()
            if (include_disabled or record.enabled) and record.capability.supports(problem_class)
        )

    def support_matrix(self, *, include_disabled: bool = True) -> dict[str, dict[str, Any]]:
        """Return a stable, JSON-friendly mathematical support summary."""
        matrix: dict[str, dict[str, Any]] = {}
        for record in self.records():
            if not include_disabled and not record.enabled:
                continue
            cap = record.capability
            matrix[record.solver_name] = {
                "enabled": record.enabled,
                "problem_classes": tuple(sorted(problem.value for problem in cap.problem_classes)),
                "continuous": cap.supports_continuous,
                "integer": cap.supports_integer,
                "binary": cap.supports_binary,
                "finite_domain": cap.supports_finite_domain,
                "semi_continuous": cap.supports_semi_continuous,
                "semi_integer": cap.supports_semi_integer,
                "quadratic_objective": cap.supports_quadratic_objective,
                "callable_objective": cap.supports_callable_objective,
                "nonlinear_constraints": cap.supports_nonlinear_constraints,
                "multiple_objectives": cap.supports_multiple_objectives,
                "black_box": cap.supports_black_box,
                "warm_start": cap.supports_warm_start,
                "requires_finite_bounds": cap.requires_finite_bounds,
                "deterministic": cap.deterministic,
                "backend": cap.backend,
            }
        return matrix

    @staticmethod
    def _signature(capability: SolverCapability) -> tuple[Any, ...]:
        return (
            tuple(sorted(item.value for item in capability.problem_classes)),
            capability.supports_continuous,
            capability.supports_integer,
            capability.supports_binary,
            capability.supports_finite_domain,
            capability.supports_semi_continuous,
            capability.supports_semi_integer,
            capability.supports_quadratic_objective,
            capability.supports_callable_objective,
            capability.supports_nonlinear_constraints,
            capability.supports_multiple_objectives,
            capability.supports_black_box,
            capability.supports_warm_start,
            capability.requires_finite_bounds,
            capability.deterministic,
        )

    def audit(
        self,
        required_classes: Iterable[ProblemClass] | None = None,
    ) -> CapabilityAudit:
        """Audit direct-solver class coverage and duplicate capability signatures.

        ``MULTIOBJECTIVE`` is not assumed to require a native solver because the
        subsystem can transform such problems through ``ScalarizationRegistry``.
        ``UNKNOWN`` is likewise excluded from the default audit because it is a
        classification state, not a mathematical solver class.
        """
        required = tuple(
            required_classes
            if required_classes is not None
            else (
                ProblemClass.LP,
                ProblemClass.IP,
                ProblemClass.MILP,
                ProblemClass.QP,
                ProblemClass.NLP,
                ProblemClass.CP,
                ProblemClass.COMBINATORIAL,
                ProblemClass.BLACK_BOX,
            )
        )
        if any(not isinstance(problem_class, ProblemClass) for problem_class in required):
            raise RegistryError("required_classes must contain only ProblemClass members")

        coverage: dict[ProblemClass, tuple[str, ...]] = {}
        disabled_only: dict[ProblemClass, tuple[str, ...]] = {}
        uncovered: list[ProblemClass] = []
        for problem_class in required:
            active = tuple(record.solver_name for record in self.supporting(problem_class))
            all_records = tuple(record.solver_name for record in self.supporting(problem_class, include_disabled=True))
            coverage[problem_class] = active
            if not active:
                uncovered.append(problem_class)
                if all_records:
                    disabled_only[problem_class] = all_records

        signatures: dict[tuple[Any, ...], list[str]] = {}
        for record in self.records():
            signatures.setdefault(self._signature(record.capability), []).append(record.solver_name)
        duplicate_signatures = tuple(
            tuple(sorted(names))
            for names in signatures.values()
            if len(names) > 1
        )
        duplicate_signatures = tuple(sorted(duplicate_signatures))

        return CapabilityAudit(
            coverage=coverage,
            uncovered=tuple(uncovered),
            disabled_only=disabled_only,
            duplicate_signatures=duplicate_signatures,
        )


__all__ = [
    "CapabilityAssessment",
    "CapabilityAudit",
    "CapabilityRecord",
    "CapabilityRegistry",
]
