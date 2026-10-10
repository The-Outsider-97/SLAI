"""
Mathematical-domain type definitions for the SLAI optimization subsystem.

This module describes *what* an optimization problem is. It does not solve
anything.

The types below provide a solver-agnostic vocabulary consistent with
continuous optimization (Boyd & Vandenberghe, 2004), numerical optimization
(Nocedal & Wright, 2006), and algebraic modeling (Hart et al., 2017). A generic
problem is written

.. math::

    \\min_{x \\in \\mathcal{X}}\\ f(x)
    \\quad \\text{s.t.}\\quad
    g_i(x) \\le 0,\\quad h_j(x) = 0,

and the objects in this module are the descriptive counterparts of
:math:`\\mathcal{X}`, :math:`f`, :math:`g_i`, :math:`h_j`, the solver
capabilities that may address them, and the results a solver may return.

Nothing in this module performs dominance computation, derivative evaluation,
simplex pivoting, interior-point iteration, or any other algorithmic step.
"""

from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .utils.optimization_errors import OptimizationValidationError
from .utils.optimization_helpers import to_finite_float
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Optimization Types")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Private validation helpers
# ---------------------------------------------------------------------------
#
# These helpers centralise the input contract shared by every value object in
# this module. They are deliberately *not* exported: callers should rely on the
# dataclass constructors, which raise ``OptimizationValidationError`` with a
# structured ``context`` mapping.
# ---------------------------------------------------------------------------


def _validate_non_empty_string(value: Any, field_name: str) -> str:
    """Return ``value`` if it is a non-empty string, else raise."""
    if not isinstance(value, str) or not value.strip():
        raise OptimizationValidationError(
            f"{field_name} must be a non-empty string",
            context={"field": field_name, "value": repr(value)},
        )
    return value


def _validate_finite(value: Any, field_name: str) -> float:
    """Return ``value`` as a finite float, rejecting bools, NaN and infinities."""
    result = to_finite_float(value)
    if result is None:
        raise OptimizationValidationError(
            f"{field_name} must be a finite real number",
            context={"field": field_name, "value": repr(value)},
        )
    return result


def _validate_optional_finite(value: Any, field_name: str) -> float | None:
    """Return ``None`` for ``None``; otherwise validate as a finite float."""
    if value is None:
        return None
    return _validate_finite(value, field_name)


def _validate_positive_int(value: Any, field_name: str) -> int:
    """Return ``value`` if it is a strictly positive ``int`` (not ``bool``)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise OptimizationValidationError(
            f"{field_name} must be a positive integer",
            context={"field": field_name, "value": repr(value)},
        )
    if value <= 0:
        raise OptimizationValidationError(
            f"{field_name} must be a positive integer",
            context={"field": field_name, "value": value},
        )
    return value


def _validate_shape(value: Any, field_name: str) -> tuple[int, ...]:
    """Return ``value`` if it is a tuple of non-negative integers."""
    if not isinstance(value, tuple):
        raise OptimizationValidationError(
            f"{field_name} must be a tuple of non-negative integers",
            context={"field": field_name, "type": type(value).__name__},
        )
    for index, dimension in enumerate(value):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise OptimizationValidationError(
                f"{field_name}[{index}] must be a non-negative integer",
                context={"field": field_name, "index": index, "value": repr(dimension)},
            )
    return value


def _validate_mapping(value: Any, field_name: str) -> Mapping[Any, Any]:
    """Return ``value`` if it is a mapping."""
    if not isinstance(value, Mapping):
        raise OptimizationValidationError(
            f"{field_name} must be a mapping",
            context={"field": field_name, "type": type(value).__name__},
        )
    return value


def _validate_finite_mapping(value: Any, field_name: str) -> dict[Any, float]:
    """Return a shallow copy of ``value`` with every value validated as finite."""
    mapping = _validate_mapping(value, field_name)
    validated: dict[Any, float] = {}
    for key, item in mapping.items():
        finite = to_finite_float(item)
        if finite is None:
            raise OptimizationValidationError(
                f"{field_name}[{key!r}] must be a finite real number",
                context={"field": field_name, "key": repr(key), "value": repr(item)},
            )
        validated[key] = finite
    return validated


def _validate_tuple_of(value: Any, expected_type: type, field_name: str) -> None:
    """Require ``value`` to be a tuple whose members are all ``expected_type``."""
    if not isinstance(value, tuple):
        raise OptimizationValidationError(
            f"{field_name} must be a tuple of {expected_type.__name__} instances",
            context={"field": field_name, "type": type(value).__name__},
        )
    for index, item in enumerate(value):
        if not isinstance(item, expected_type):
            raise OptimizationValidationError(
                f"{field_name} must contain only {expected_type.__name__} instances",
                context={
                    "field": field_name,
                    "index": index,
                    "type": type(item).__name__,
                },
            )


def _validate_unique_names(items: Any, field_name: str, kind: str) -> None:
    """Ensure ``items`` (each exposing ``.name``) do not repeat a name."""
    seen: set[str] = set()
    for item in items:
        name = item.name
        if name in seen:
            raise OptimizationValidationError(
                f"duplicate {kind} name in {field_name}: {name!r}",
                context={"field": field_name, "kind": kind, "name": name},
            )
        seen.add(name)


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------


class DomainKind(Enum):
    """
    The mathematical kind of a decision-variable domain.

    * ``REAL`` — :math:`\\mathcal{X} \\subseteq \\mathbb{R}`
    * ``INTEGER`` — :math:`\\mathcal{X} \\subseteq \\mathbb{Z}`
    * ``BINARY`` — :math:`\\mathcal{X} = \\{0, 1\\}`
    * ``SEMI_CONTINUOUS`` — :math:`\\mathcal{X} = \\{0\\} \\cup [\\ell, u]`
    * ``SEMI_INTEGER`` — :math:`\\mathcal{X} = \\{0\\} \\cup \\{\\ell, \\dots, u\\}`
    * ``CUSTOM`` — an opaque domain whose structure is solver- or model-defined

    The taxonomy is deliberately solver-agnostic; the concrete
    ``kind``/bounds mapping is the caller's responsibility.
    """

    REAL = "real"
    INTEGER = "integer"
    BINARY = "binary"
    SEMI_CONTINUOUS = "semi_continuous"
    SEMI_INTEGER = "semi_integer"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class VariableDomain:
    """
    Mathematical domain of a decision variable.

    A domain is a subset of an ambient space; when the variable is a
    ``shape``-indexed array of length :math:`n` the domain of the whole
    variable is the Cartesian product :math:`\\mathcal{X}_1 \\times \\dots
    \\times \\mathcal{X}_n`.

    The fields are descriptive only. This class performs no projection, no
    rounding and no interval arithmetic; it merely records bounds, strictness,
    discreteness and (optionally) a finite cardinality.

    References
    ----------
    Boyd & Vandenberghe (2004), §4.2 (convex sets and domains);
    Nocedal & Wright (2006), §12.1 (bound-constrained formulations).
    """

    kind: DomainKind = DomainKind.REAL
    lower: float | None = None
    upper: float | None = None
    strict_lower: bool = False
    strict_upper: bool = False
    cardinality: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DomainKind):
            raise OptimizationValidationError(
                "kind must be a DomainKind member",
                context={"kind": repr(self.kind)},
            )

        lower = _validate_optional_finite(self.lower, "lower")
        upper = _validate_optional_finite(self.upper, "upper")

        if not isinstance(self.strict_lower, bool):
            raise OptimizationValidationError(
                "strict_lower must be a bool",
                context={"strict_lower": repr(self.strict_lower)},
            )
        if not isinstance(self.strict_upper, bool):
            raise OptimizationValidationError(
                "strict_upper must be a bool",
                context={"strict_upper": repr(self.strict_upper)},
            )

        cardinality = self.cardinality
        if cardinality is not None:
            cardinality = _validate_positive_int(cardinality, "cardinality")

        if lower is not None and upper is not None:
            if lower > upper:
                raise OptimizationValidationError(
                    "lower bound must not exceed upper bound",
                    context={"lower": lower, "upper": upper},
                )
            if lower == upper and (self.strict_lower or self.strict_upper):
                raise OptimizationValidationError(
                    "strict bounds make the interval empty",
                    context={"lower": lower, "upper": upper},
                )

        if self.strict_lower and lower is None:
            raise OptimizationValidationError(
                "strict_lower requires a finite lower bound",
                context={"kind": self.kind.value},
            )
        if self.strict_upper and upper is None:
            raise OptimizationValidationError(
                "strict_upper requires a finite upper bound",
                context={"kind": self.kind.value},
            )

        if self.kind is DomainKind.BINARY:
            if lower is not None and lower != 0.0:
                raise OptimizationValidationError(
                    "binary domain lower bound must be 0 when provided",
                    context={"lower": lower},
                )
            if upper is not None and upper != 1.0:
                raise OptimizationValidationError(
                    "binary domain upper bound must be 1 when provided",
                    context={"upper": upper},
                )
            if self.strict_lower or self.strict_upper:
                raise OptimizationValidationError(
                    "binary domains cannot use strict bounds",
                )
            if cardinality is not None and cardinality != 2:
                raise OptimizationValidationError(
                    "binary domain cardinality must be 2 when provided",
                    context={"cardinality": cardinality},
                )

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "cardinality", cardinality)

    # -- constructors ------------------------------------------------------

    @classmethod
    def real(
        cls,
        lower: float | None = None,
        upper: float | None = None,
        *,
        strict_lower: bool = False,
        strict_upper: bool = False,
    ) -> "VariableDomain":
        """Continuous domain :math:`\\mathcal{X} \\subseteq \\mathbb{R}`."""
        return cls(DomainKind.REAL, lower, upper, strict_lower, strict_upper)

    @classmethod
    def integer(
        cls,
        lower: float | None = None,
        upper: float | None = None,
        *,
        strict_lower: bool = False,
        strict_upper: bool = False,
    ) -> "VariableDomain":
        """Integer domain :math:`\\mathcal{X} \\subseteq \\mathbb{Z}`."""
        return cls(DomainKind.INTEGER, lower, upper, strict_lower, strict_upper)

    @classmethod
    def binary(cls) -> "VariableDomain":
        """Binary domain :math:`\\mathcal{X} = \\{0, 1\\}` with cardinality 2."""
        return cls(DomainKind.BINARY, 0.0, 1.0, False, False, 2)

    @classmethod
    def nonnegative(cls) -> "VariableDomain":
        """Continuous half-line :math:`\\mathcal{X} = [0, \\infty)`."""
        return cls(DomainKind.REAL, 0.0, None)

    @classmethod
    def bounds(
        cls,
        lower: float,
        upper: float,
        *,
        strict_lower: bool = False,
        strict_upper: bool = False,
    ) -> "VariableDomain":
        """Continuous box domain :math:`\\mathcal{X} = [\\ell, u]`."""
        return cls(DomainKind.REAL, lower, upper, strict_lower, strict_upper)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Variable:
    """
    A decision variable of an optimization problem.

    A variable :math:`x` is a typed symbol ranging over ``domain``. When
    ``shape`` is non-empty the variable denotes an array of independent
    components; when ``index_set`` is provided it records the categorical
    labels of those components (or of a single indexed component).

    Instances are immutable descriptions; they carry no value. Concrete values
    belong to :class:`Solution` or :class:`ParetoPoint`.
    """

    name: str
    domain: VariableDomain
    shape: tuple[int, ...] = ()
    index_set: tuple[Any, ...] | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")

        if not isinstance(self.domain, VariableDomain):
            raise OptimizationValidationError(
                "domain must be a VariableDomain instance",
                context={"type": type(self.domain).__name__},
            )

        shape = _validate_shape(self.shape, "shape")

        if self.index_set is not None and not isinstance(self.index_set, tuple):
            raise OptimizationValidationError(
                "index_set must be a tuple or None",
                context={"type": type(self.index_set).__name__},
            )

        if self.description is not None and not isinstance(self.description, str):
            raise OptimizationValidationError(
                "description must be a string or None",
                context={"type": type(self.description).__name__},
            )

        _validate_mapping(self.metadata, "metadata")

        object.__setattr__(self, "shape", shape)


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


class ObjectiveSense(Enum):
    """Direction of an objective function :math:`f`."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True, slots=True)
class Objective:
    """
    An objective function participating in an optimization problem.

    Formally this describes :math:`f : \\mathcal{X} \\to \\mathbb{R}` together
    with its sense (``min`` or ``max``) and the decision variables it depends
    on. The ``expression`` payload is intentionally opaque: it may be a string,
    a symbolic expression tree, a callable, or a backend-specific object.

    The class stores only descriptive structure. It does not evaluate, compile
    or differentiate the objective.
    """

    name: str
    sense: ObjectiveSense
    expression: Any
    variables: tuple[Variable, ...] = ()
    constant: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")

        if not isinstance(self.sense, ObjectiveSense):
            raise OptimizationValidationError(
                "sense must be an ObjectiveSense member",
                context={"sense": repr(self.sense)},
            )

        _validate_tuple_of(self.variables, Variable, "variables")
        _validate_unique_names(self.variables, "variables", "variable")

        constant = _validate_finite(self.constant, "constant")
        _validate_mapping(self.metadata, "metadata")

        object.__setattr__(self, "constant", constant)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class ConstraintSense(Enum):
    """
    Canonical sense of a scalar constraint.

    Used when a constraint is expressed in the classical inequality form
    rather than as a two-sided interval:

    * ``LEQ`` — :math:`g(x) \\le u`
    * ``GEQ`` — :math:`g(x) \\ge \\ell`
    * ``EQ``  — :math:`h(x) = \\ell = u`
    """

    LEQ = "leq"
    GEQ = "geq"
    EQ = "eq"


@dataclass(frozen=True, slots=True)
class Constraint:
    """
    A constraint on the decision variables.

    The constraint is encoded as a two-sided bound on an opaque ``body``
    expression:

    .. math::

        \\ell \\;\\le\\; \\text{body}(x) \\;\\le\\; u

    Exactly one of ``lower`` / ``upper`` may be ``None`` for a one-sided
    constraint; both being ``None`` is rejected. The class performs no
    feasibility check — that is the role of a solver — and no symbolic
    manipulation of ``body``.
    """

    name: str
    body: Any
    lower: float | None = None
    upper: float | None = None
    variables: tuple[Variable, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")

        lower = _validate_optional_finite(self.lower, "lower")
        upper = _validate_optional_finite(self.upper, "upper")

        if lower is None and upper is None:
            raise OptimizationValidationError(
                "constraint requires at least one of lower or upper",
                context={"name": self.name},
            )
        if lower is not None and upper is not None and lower > upper:
            raise OptimizationValidationError(
                "constraint lower bound must not exceed upper bound",
                context={"name": self.name, "lower": lower, "upper": upper},
            )

        _validate_tuple_of(self.variables, Variable, "variables")
        _validate_unique_names(self.variables, "variables", "variable")

        _validate_mapping(self.metadata, "metadata")

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def sense(self) -> ConstraintSense | None:
        """
        Classify the constraint as ``LEQ``, ``GEQ`` or ``EQ`` when possible.

        A genuinely two-sided constraint with distinct bounds returns
        ``None`` because it does not correspond to a single canonical sense.
        """
        if self.lower is None and self.upper is not None:
            return ConstraintSense.LEQ
        if self.lower is not None and self.upper is None:
            return ConstraintSense.GEQ
        if self.lower is not None and self.upper is not None and self.lower == self.upper:
            return ConstraintSense.EQ
        return None


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


class ProblemClass(Enum):
    """
    Solver-agnostic classification of an optimization problem.

    The taxonomy follows the standard mathematical-programming categories
    (Boyd & Vandenberghe, 2004; Nocedal & Wright, 2006):

    * ``LP`` — linear program
    * ``MILP`` — mixed-integer linear program
    * ``QP`` — quadratic program
    * ``MIQP`` — mixed-integer quadratic program
    * ``QCQP`` — quadratically constrained quadratic program
    * ``SOCP`` — second-order cone program
    * ``SDP`` — semidefinite program
    * ``NLP`` — nonlinear program
    * ``MINLP`` — mixed-integer nonlinear program
    * ``CONVEX`` — a convex program outside the above named families
    * ``NONCONVEX`` — a nonconvex program outside the above named families
    * ``MULTI_OBJECTIVE`` — a vector-valued objective problem
    * ``UNKNOWN`` — not yet classified
    """

    LP = "lp"
    MILP = "milp"
    QP = "qp"
    MIQP = "miqp"
    QCQP = "qcqp"
    SOCP = "socp"
    SDP = "sdp"
    NLP = "nlp"
    MINLP = "minlp"
    CONVEX = "convex"
    NONCONVEX = "nonconvex"
    MULTI_OBJECTIVE = "multi_objective"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class OptimizationProblem:
    """
    A mathematical optimization problem.

    The object gathers the descriptive parts of

    .. math::

        \\min_{x \\in \\mathcal{X}}\\ f(x)
        \\quad \\text{s.t.}\\quad
        g_i(x) \\le 0,\\; h_j(x) = 0,

    namely variables, objectives, constraints and an optional problem class.
    It performs **no** analysis: no convexity test, no dual construction, no
    substitution, no reduction. Any such work belongs to an algorithm module.
    """

    name: str
    variables: tuple[Variable, ...]
    objectives: tuple[Objective, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    problem_class: ProblemClass = ProblemClass.UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")

        _validate_tuple_of(self.variables, Variable, "variables")
        _validate_tuple_of(self.objectives, Objective, "objectives")
        _validate_tuple_of(self.constraints, Constraint, "constraints")

        _validate_unique_names(self.variables, "variables", "variable")
        _validate_unique_names(self.objectives, "objectives", "objective")
        _validate_unique_names(self.constraints, "constraints", "constraint")

        if not isinstance(self.problem_class, ProblemClass):
            raise OptimizationValidationError(
                "problem_class must be a ProblemClass member",
                context={"problem_class": repr(self.problem_class)},
            )

        _validate_mapping(self.metadata, "metadata")

    @property
    def objective(self) -> Objective | None:
        """Return the single objective when exactly one is present, else None."""
        if len(self.objectives) == 1:
            return self.objectives[0]
        return None

    @property
    def is_multi_objective(self) -> bool:
        """``True`` when the problem declares more than one objective."""
        return len(self.objectives) > 1


# ---------------------------------------------------------------------------
# Solver capability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolverCapability:
    """
    A description of what a solver can handle.

    This is a declarative record used for routing and compatibility checks.
    It does not invoke the solver, and it does not attempt to derive
    capabilities from the solver's behaviour.
    """

    solver_name: str
    supported_problem_classes: frozenset[ProblemClass]
    supports_continuous: bool = True
    supports_integer: bool = False
    supports_binary: bool = False
    supports_nonlinear: bool = False
    supports_multiobjective: bool = False
    supports_duals: bool = False
    supports_warm_start: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.solver_name, "solver_name")

        classes = self.supported_problem_classes
        if not isinstance(classes, frozenset):
            try:
                classes = frozenset(classes)
            except TypeError as exc:  # pragma: no cover - defensive
                raise OptimizationValidationError(
                    "supported_problem_classes must be an iterable of ProblemClass",
                    context={"type": type(self.supported_problem_classes).__name__},
                ) from exc
            object.__setattr__(self, "supported_problem_classes", classes)

        for entry in classes:
            if not isinstance(entry, ProblemClass):
                raise OptimizationValidationError(
                    "supported_problem_classes must contain only ProblemClass members",
                    context={"value": repr(entry)},
                )

        for field_name in (
            "supports_continuous",
            "supports_integer",
            "supports_binary",
            "supports_nonlinear",
            "supports_multiobjective",
            "supports_duals",
            "supports_warm_start",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise OptimizationValidationError(
                    f"{field_name} must be a bool",
                    context={"field": field_name, "value": repr(getattr(self, field_name))},
                )

        _validate_mapping(self.metadata, "metadata")

    def supports(self, problem_class: ProblemClass) -> bool:
        """Return ``True`` when ``problem_class`` is declared as supported."""
        if not isinstance(problem_class, ProblemClass):
            raise OptimizationValidationError(
                "problem_class must be a ProblemClass member",
                context={"problem_class": repr(problem_class)},
            )
        return problem_class in self.supported_problem_classes


# ---------------------------------------------------------------------------
# Solver / solution status enums
# ---------------------------------------------------------------------------


class SolverStatus(Enum):
    """Lifecycle state of a solver instance."""

    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"
    UNKNOWN = "unknown"


class FeasibilityStatus(Enum):
    """
    Feasibility classification of a candidate point or solution.

    ``NEARLY_FEASIBLE`` denotes a point that violates the constraint system by
    more than numerical noise but within a solver-declared feasibility
    tolerance.
    """

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"
    NEARLY_FEASIBLE = "nearly_feasible"


class OptimalityStatus(Enum):
    """
    Optimality classification of a solution.

    The distinction between ``GLOBALLY_OPTIMAL`` and ``LOCALLY_OPTIMAL``
    matters for nonconvex problems where first-order stationarity does not
    imply global optimality (Nocedal & Wright, 2006, §12.4).
    """

    OPTIMAL = "optimal"
    GLOBALLY_OPTIMAL = "globally_optimal"
    LOCALLY_OPTIMAL = "locally_optimal"
    SUBOPTIMAL = "suboptimal"
    UNBOUNDED = "unbounded"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


class TerminationReason(Enum):
    """Why a solver stopped."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"
    OBJECTIVE_LIMIT = "objective_limit"
    FEASIBLE_SOLUTION_FOUND = "feasible_solution_found"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    USER_INTERRUPT = "user_interrupt"
    NUMERICAL_ERROR = "numerical_error"
    SOLVER_ERROR = "solver_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Solutions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Solution:
    """
    The result of an optimization run.

    A solution records concrete variable values, objective values, and the
    status information produced by a solver. Solver-internal state (iterates,
    dual vectors, factorisations) is deliberately excluded: if needed, it
    belongs in ``metadata``.

    The class performs no post-processing, no feasibility repair, and no
    validation against a :class:`OptimizationProblem`.
    """

    variable_values: Mapping[str, Any]
    objective_values: Mapping[str, float] = field(default_factory=dict)
    feasibility: FeasibilityStatus = FeasibilityStatus.UNKNOWN
    optimality: OptimalityStatus = OptimalityStatus.UNKNOWN
    solver_status: SolverStatus = SolverStatus.UNKNOWN
    termination_reason: TerminationReason = TerminationReason.UNKNOWN
    constraint_violations: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _validate_mapping(self.variable_values, "variable_values")

        objective_values = _validate_finite_mapping(self.objective_values, "objective_values")
        constraint_violations = _validate_finite_mapping(
            self.constraint_violations, "constraint_violations"
        )

        if not isinstance(self.feasibility, FeasibilityStatus):
            raise OptimizationValidationError(
                "feasibility must be a FeasibilityStatus member",
                context={"feasibility": repr(self.feasibility)},
            )
        if not isinstance(self.optimality, OptimalityStatus):
            raise OptimizationValidationError(
                "optimality must be an OptimalityStatus member",
                context={"optimality": repr(self.optimality)},
            )
        if not isinstance(self.solver_status, SolverStatus):
            raise OptimizationValidationError(
                "solver_status must be a SolverStatus member",
                context={"solver_status": repr(self.solver_status)},
            )
        if not isinstance(self.termination_reason, TerminationReason):
            raise OptimizationValidationError(
                "termination_reason must be a TerminationReason member",
                context={"termination_reason": repr(self.termination_reason)},
            )

        _validate_mapping(self.metadata, "metadata")

        object.__setattr__(self, "objective_values", objective_values)
        object.__setattr__(self, "constraint_violations", constraint_violations)


# ---------------------------------------------------------------------------
# Pareto concepts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """
    A feasible point in a multiobjective optimization context.

    For a vector objective :math:`f(x) = (f_1(x), \\dots, f_k(x))`, a Pareto
    point is a point :math:`x` such that no other feasible point strictly
    improves every objective. This class records the raw coordinates only;
    dominance is a property of a *set* of points and is computed elsewhere
    (see :class:`ParetoFront`).
    """

    variables: Mapping[str, Any]
    objectives: Mapping[str, float]
    feasible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _validate_mapping(self.variables, "variables")
        objectives = _validate_finite_mapping(self.objectives, "objectives")

        if not isinstance(self.feasible, bool):
            raise OptimizationValidationError(
                "feasible must be a bool",
                context={"feasible": repr(self.feasible)},
            )

        _validate_mapping(self.metadata, "metadata")

        object.__setattr__(self, "objectives", objectives)


@dataclass(frozen=True, slots=True)
class ParetoFront:
    """
    A container for Pareto-optimal points.

    The front is the image of the Pareto set under the objective map. This
    class is a passive container: it stores points, exposes them through the
    standard container protocol, and performs no dominance filtering, crowding
    distance computation, or hypervolume estimation.
    """

    points: tuple[ParetoPoint, ...] = ()
    problem: OptimizationProblem | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _validate_tuple_of(self.points, ParetoPoint, "points")

        if self.problem is not None and not isinstance(self.problem, OptimizationProblem):
            raise OptimizationValidationError(
                "problem must be an OptimizationProblem or None",
                context={"type": type(self.problem).__name__},
            )

        _validate_mapping(self.metadata, "metadata")

    def __iter__(self) -> Iterator[ParetoPoint]:
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int | slice) -> ParetoPoint | tuple[ParetoPoint, ...]:
        return self.points[index]

    def __bool__(self) -> bool:
        return bool(self.points)

    @property
    def is_empty(self) -> bool:
        """``True`` when the front contains no points."""
        return not self.points


__all__ = [
    "Constraint",
    "ConstraintSense",
    "DomainKind",
    "FeasibilityStatus",
    "Objective",
    "ObjectiveSense",
    "OptimalityStatus",
    "OptimizationProblem",
    "ParetoFront",
    "ParetoPoint",
    "ProblemClass",
    "Solution",
    "SolverCapability",
    "SolverStatus",
    "TerminationReason",
    "Variable",
    "VariableDomain",
]