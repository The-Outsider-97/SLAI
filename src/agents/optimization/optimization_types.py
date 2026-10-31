"""Canonical solver-independent mathematical optimization contracts.

The model follows the constrained-programming form

    minimize/maximize f(x)
    subject to g_i(x) <= 0, h_j(x) = 0, x in X

and deliberately separates mathematical declaration from concrete solver
backends.  The design is informed by Boyd & Vandenberghe (2004), Nocedal &
Wright (2006), and solver-independent algebraic modeling principles described
by Bynum et al. (2021).
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .utils.optimization_errors import OptimizationValidationError
from .utils.optimization_helpers import require_finite, to_finite_float


Point = Mapping[str, Any]
ScalarCallable = Callable[[Point], float]
GradientCallable = Callable[[Point], Mapping[str, float] | Sequence[float]]
HessianCallable = Callable[[Point], Sequence[Sequence[float]]]
JacobianCallable = Callable[[Point], Mapping[str, float] | Sequence[float]]


class DomainKind(str, Enum):
    REAL = "real"
    INTEGER = "integer"
    BINARY = "binary"
    FINITE = "finite"
    SEMI_CONTINUOUS = "semi_continuous"
    SEMI_INTEGER = "semi_integer"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class VariableDomain:
    """Domain of a scalar decision variable.

    ``FINITE`` stores explicit values.  ``INTEGER`` and ``SEMI_INTEGER`` use
    numeric bounds.  ``CUSTOM`` is descriptive only and is rejected by generic
    built-in solvers unless a compatible adapter explicitly opts in.
    """

    kind: DomainKind = DomainKind.REAL
    lower: float | None = None
    upper: float | None = None
    strict_lower: bool = False
    strict_upper: bool = False
    cardinality: int | None = None
    values: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DomainKind):
            raise OptimizationValidationError("kind must be a DomainKind", context={"kind": repr(self.kind)})
        if not isinstance(self.strict_lower, bool) or not isinstance(self.strict_upper, bool):
            raise OptimizationValidationError("strict bound flags must be bool")

        lower = None if self.lower is None else require_finite(self.lower, "lower")
        upper = None if self.upper is None else require_finite(self.upper, "upper")
        values = tuple(self.values)

        if lower is not None and upper is not None and lower > upper:
            raise OptimizationValidationError("lower bound must not exceed upper bound", context={"lower": lower, "upper": upper})
        if lower is not None and upper is not None and lower == upper and (self.strict_lower or self.strict_upper):
            raise OptimizationValidationError("strict bounds define an empty domain")
        if self.strict_lower and lower is None:
            raise OptimizationValidationError("strict_lower requires a finite lower bound")
        if self.strict_upper and upper is None:
            raise OptimizationValidationError("strict_upper requires a finite upper bound")

        if self.kind is DomainKind.BINARY:
            lower, upper, values = 0.0, 1.0, (0, 1)
        elif self.kind is DomainKind.FINITE:
            if not values:
                raise OptimizationValidationError("FINITE domain requires at least one explicit value")
            fingerprints = [repr(v) for v in values]
            if len(set(fingerprints)) != len(fingerprints):
                raise OptimizationValidationError("finite-domain values must be unique")
        elif values:
            raise OptimizationValidationError("explicit values are only valid for BINARY or FINITE domains")

        if self.kind in {DomainKind.INTEGER, DomainKind.SEMI_INTEGER}:
            for label, bound in (("lower", lower), ("upper", upper)):
                if bound is not None and not float(bound).is_integer():
                    raise OptimizationValidationError(
                        f"{label} bound for integer domains must be integral",
                        context={label: bound},
                    )
        if self.kind in {DomainKind.SEMI_CONTINUOUS, DomainKind.SEMI_INTEGER}:
            if lower is None or upper is None or lower <= 0:
                raise OptimizationValidationError("semi-continuous/integer domains require 0 < lower <= upper")

        cardinality = self.cardinality
        if cardinality is not None:
            if isinstance(cardinality, bool) or not isinstance(cardinality, int) or cardinality <= 0:
                raise OptimizationValidationError("cardinality must be a positive integer")
        if values:
            if cardinality is not None and cardinality != len(values):
                raise OptimizationValidationError("cardinality does not match explicit finite-domain values")
            cardinality = len(values)

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "cardinality", cardinality)

    @classmethod
    def real(
        cls,
        lower: float | None = None,
        upper: float | None = None,
        *,
        strict_lower: bool = False,
        strict_upper: bool = False,
    ) -> "VariableDomain":
        return cls(DomainKind.REAL, lower, upper, strict_lower, strict_upper)

    @classmethod
    def integer(cls, lower: int | None = None, upper: int | None = None) -> "VariableDomain":
        return cls(DomainKind.INTEGER, lower, upper)

    @classmethod
    def binary(cls) -> "VariableDomain":
        return cls(DomainKind.BINARY)

    @classmethod
    def finite(cls, values: Sequence[Any]) -> "VariableDomain":
        return cls(DomainKind.FINITE, values=tuple(values))

    @classmethod
    def semi_continuous(cls, lower: float, upper: float) -> "VariableDomain":
        return cls(DomainKind.SEMI_CONTINUOUS, lower, upper)

    @classmethod
    def semi_integer(cls, lower: int, upper: int) -> "VariableDomain":
        return cls(DomainKind.SEMI_INTEGER, lower, upper)

    @classmethod
    def nonnegative(cls) -> "VariableDomain":
        return cls(DomainKind.REAL, 0.0, None)

    @property
    def discrete(self) -> bool:
        return self.kind in {DomainKind.INTEGER, DomainKind.BINARY, DomainKind.FINITE, DomainKind.SEMI_INTEGER}

    @property
    def numeric(self) -> bool:
        return self.kind is not DomainKind.CUSTOM and (
            self.kind is not DomainKind.FINITE
            or all(to_finite_float(v) is not None for v in self.values)
        )


@dataclass(frozen=True, slots=True)
class Variable:
    name: str
    domain: VariableDomain = field(default_factory=VariableDomain)
    initial: Any | None = None
    shape: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise OptimizationValidationError("variable name must be a non-empty string")
        if not isinstance(self.domain, VariableDomain):
            raise OptimizationValidationError("domain must be VariableDomain", context={"variable": self.name})
        if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in self.shape):
            raise OptimizationValidationError("shape dimensions must be positive integers", context={"variable": self.name})
        if self.initial is not None:
            self.validate_value(self.initial)

    def validate_value(self, value: Any, *, tolerance: float = 1e-10) -> Any:
        domain = self.domain
        if domain.kind is DomainKind.FINITE:
            if value not in domain.values:
                raise OptimizationValidationError(
                    "value is outside finite domain",
                    context={"variable": self.name, "value": repr(value)},
                )
            return value
        finite = to_finite_float(value)
        if finite is None:
            raise OptimizationValidationError(
                "numeric variable value must be finite",
                context={"variable": self.name, "value": repr(value)},
            )
        if domain.kind in {DomainKind.INTEGER, DomainKind.BINARY, DomainKind.SEMI_INTEGER}:
            rounded = round(finite)
            if abs(finite - rounded) > tolerance:
                raise OptimizationValidationError("integer-domain value is not integral", context={"variable": self.name, "value": finite})
            finite = float(rounded)
        if domain.kind is DomainKind.BINARY and finite not in (0.0, 1.0):
            raise OptimizationValidationError("binary variable value must be 0 or 1", context={"variable": self.name})
        if domain.kind in {DomainKind.SEMI_CONTINUOUS, DomainKind.SEMI_INTEGER} and finite == 0.0:
            return int(finite) if domain.kind is DomainKind.SEMI_INTEGER else finite
        if domain.lower is not None:
            if domain.strict_lower and not finite > domain.lower:
                raise OptimizationValidationError("value violates strict lower bound", context={"variable": self.name})
            if not domain.strict_lower and finite < domain.lower - tolerance:
                raise OptimizationValidationError("value violates lower bound", context={"variable": self.name})
        if domain.upper is not None:
            if domain.strict_upper and not finite < domain.upper:
                raise OptimizationValidationError("value violates strict upper bound", context={"variable": self.name})
            if not domain.strict_upper and finite > domain.upper + tolerance:
                raise OptimizationValidationError("value violates upper bound", context={"variable": self.name})
        if domain.kind in {DomainKind.INTEGER, DomainKind.BINARY, DomainKind.SEMI_INTEGER}:
            return int(round(finite))
        return finite


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    @classmethod
    def coerce(cls, value: "ObjectiveSense | str") -> "ObjectiveSense":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {
            "min": cls.MINIMIZE,
            "minimum": cls.MINIMIZE,
            "minimize": cls.MINIMIZE,
            "max": cls.MAXIMIZE,
            "maximum": cls.MAXIMIZE,
            "maximize": cls.MAXIMIZE,
        }
        if normalized not in aliases:
            raise OptimizationValidationError("invalid objective sense", context={"sense": value})
        return aliases[normalized]


@dataclass(frozen=True, slots=True)
class Objective:
    name: str
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    function: ScalarCallable | None = None
    linear: Mapping[str, float] = field(default_factory=dict)
    quadratic: Mapping[tuple[str, str], float] = field(default_factory=dict)
    constant: float = 0.0
    weight: float = 1.0
    priority: int = 0
    gradient: GradientCallable | None = None
    hessian: HessianCallable | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise OptimizationValidationError("objective name must be non-empty")
        object.__setattr__(self, "sense", ObjectiveSense.coerce(self.sense))
        constant = require_finite(self.constant, "constant")
        weight = require_finite(self.weight, "weight")
        if weight < 0:
            raise OptimizationValidationError("objective weight must be non-negative")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise OptimizationValidationError("objective priority must be an integer")
        if self.function is None and not self.linear and not self.quadratic:
            raise OptimizationValidationError("objective requires callable, linear, or quadratic representation")
        linear: dict[str, float] = {}
        for key, value in self.linear.items():
            if not isinstance(key, str) or not key:
                raise OptimizationValidationError("linear objective coefficient key must be variable name")
            linear[key] = require_finite(value, f"objective[{self.name}].linear[{key}]")
        quadratic: dict[tuple[str, str], float] = {}
        for key, value in self.quadratic.items():
            if not isinstance(key, tuple) or len(key) != 2 or any(not isinstance(k, str) or not k for k in key):
                raise OptimizationValidationError("quadratic coefficient key must be (variable, variable)")
            quadratic[key] = require_finite(value, f"objective[{self.name}].quadratic[{key}]")
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", quadratic)

    @property
    def algebraic(self) -> bool:
        return self.function is None

    @property
    def nonlinear(self) -> bool:
        return self.function is not None or bool(self.quadratic)


class ConstraintSense(str, Enum):
    LE = "<="
    GE = ">="
    EQ = "=="

    @classmethod
    def coerce(cls, value: "ConstraintSense | str") -> "ConstraintSense":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip()
        aliases = {"<=": cls.LE, "<": cls.LE, ">=": cls.GE, ">": cls.GE, "=": cls.EQ, "==": cls.EQ}
        if normalized not in aliases:
            raise OptimizationValidationError("invalid constraint sense", context={"sense": value})
        return aliases[normalized]


@dataclass(frozen=True, slots=True)
class Constraint:
    name: str
    sense: ConstraintSense = ConstraintSense.LE
    rhs: float = 0.0
    function: ScalarCallable | None = None
    linear: Mapping[str, float] = field(default_factory=dict)
    tolerance: float = 1e-8
    jacobian: JacobianCallable | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise OptimizationValidationError("constraint name must be non-empty")
        object.__setattr__(self, "sense", ConstraintSense.coerce(self.sense))
        rhs = require_finite(self.rhs, "rhs")
        tolerance = require_finite(self.tolerance, "tolerance")
        if tolerance < 0:
            raise OptimizationValidationError("constraint tolerance must be non-negative")
        if self.function is None and not self.linear:
            raise OptimizationValidationError("constraint requires callable or linear coefficients")
        linear: dict[str, float] = {}
        for key, value in self.linear.items():
            if not isinstance(key, str) or not key:
                raise OptimizationValidationError("constraint coefficient key must be variable name")
            linear[key] = require_finite(value, f"constraint[{self.name}].linear[{key}]")
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "linear", linear)

    @property
    def nonlinear(self) -> bool:
        return self.function is not None


class ProblemClass(str, Enum):
    LP = "lp"
    IP = "ip"
    MILP = "milp"
    QP = "qp"
    NLP = "nlp"
    CP = "cp"
    COMBINATORIAL = "combinatorial"
    BLACK_BOX = "black_box"
    MULTIOBJECTIVE = "multiobjective"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class OptimizationProblem:
    name: str
    variables: tuple[Variable, ...]
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...] = ()
    problem_class: ProblemClass = ProblemClass.UNKNOWN
    known_convex: bool | None = None
    known_smooth: bool | None = None
    derivatives_available: bool = False
    deterministic: bool = True
    expected_evaluation_cost: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise OptimizationValidationError("problem name must be non-empty")
        if not isinstance(self.problem_class, ProblemClass):
            raise OptimizationValidationError("problem_class must be ProblemClass")
        if not self.variables:
            raise OptimizationValidationError("problem requires at least one variable")
        if not self.objectives:
            raise OptimizationValidationError("problem requires at least one objective")
        if any(not isinstance(v, Variable) for v in self.variables):
            raise OptimizationValidationError("variables must contain only Variable objects")
        if any(not isinstance(o, Objective) for o in self.objectives):
            raise OptimizationValidationError("objectives must contain only Objective objects")
        if any(not isinstance(c, Constraint) for c in self.constraints):
            raise OptimizationValidationError("constraints must contain only Constraint objects")
        if len({v.name for v in self.variables}) != len(self.variables):
            raise OptimizationValidationError("variable names must be unique")
        if len({o.name for o in self.objectives}) != len(self.objectives):
            raise OptimizationValidationError("objective names must be unique")
        if len({c.name for c in self.constraints}) != len(self.constraints):
            raise OptimizationValidationError("constraint names must be unique")

        variable_names = {v.name for v in self.variables}
        for objective in self.objectives:
            references = set(objective.linear)
            references.update(name for pair in objective.quadratic for name in pair)
            unknown = references - variable_names
            if unknown:
                raise OptimizationValidationError(
                    "objective references unknown variables",
                    context={"objective": objective.name, "unknown": sorted(unknown)},
                )
        for constraint in self.constraints:
            unknown = set(constraint.linear) - variable_names
            if unknown:
                raise OptimizationValidationError(
                    "constraint references unknown variables",
                    context={"constraint": constraint.name, "unknown": sorted(unknown)},
                )

        if self.expected_evaluation_cost is not None:
            cost = require_finite(self.expected_evaluation_cost, "expected_evaluation_cost")
            if cost < 0:
                raise OptimizationValidationError("expected_evaluation_cost must be non-negative")
            object.__setattr__(self, "expected_evaluation_cost", cost)
        if len(self.objectives) > 1 and self.problem_class is not ProblemClass.MULTIOBJECTIVE:
            object.__setattr__(self, "problem_class", ProblemClass.MULTIOBJECTIVE)

    @property
    def objective(self) -> Objective | None:
        return self.objectives[0] if len(self.objectives) == 1 else None

    @property
    def is_multi_objective(self) -> bool:
        return len(self.objectives) > 1

    @property
    def dimensionality(self) -> int:
        return len(self.variables)

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)


@dataclass(frozen=True, slots=True)
class ProblemFeatures:
    problem_class: ProblemClass
    dimensionality: int
    objective_count: int
    continuous_variables: int
    integer_variables: int
    binary_variables: int
    finite_domain_variables: int
    semi_continuous_variables: int
    semi_integer_variables: int
    custom_domain_variables: int
    has_quadratic_objective: bool
    has_callable_objective: bool
    has_nonlinear_constraints: bool
    has_equality_constraints: bool
    has_unbounded_variables: bool
    known_convex: bool | None
    known_smooth: bool | None
    derivatives_available: bool
    deterministic: bool
    expected_evaluation_cost: float | None
    warm_start_available: bool

    @property
    def has_discrete_variables(self) -> bool:
        return any((self.integer_variables, self.binary_variables, self.finite_domain_variables, self.semi_integer_variables))

    @property
    def finite_domain_only(self) -> bool:
        return self.dimensionality == self.finite_domain_variables + self.binary_variables + self.integer_variables


@dataclass(frozen=True, slots=True)
class SolverCapability:
    name: str
    problem_classes: frozenset[ProblemClass]
    supports_continuous: bool = True
    supports_integer: bool = False
    supports_binary: bool = False
    supports_finite_domain: bool = False
    supports_semi_continuous: bool = False
    supports_semi_integer: bool = False
    supports_quadratic_objective: bool = False
    supports_callable_objective: bool = False
    supports_nonlinear_constraints: bool = False
    supports_multiple_objectives: bool = False
    supports_black_box: bool = False
    supports_warm_start: bool = False
    requires_finite_bounds: bool = False
    deterministic: bool = True
    backend: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise OptimizationValidationError("solver capability name must be non-empty")
        if not isinstance(self.problem_classes, frozenset) or any(not isinstance(p, ProblemClass) for p in self.problem_classes):
            raise OptimizationValidationError("problem_classes must be frozenset[ProblemClass]")

    def supports(self, problem_class: ProblemClass) -> bool:
        return problem_class in self.problem_classes

    def incompatibilities(self, features: ProblemFeatures) -> tuple[str, ...]:
        reasons: list[str] = []
        if features.problem_class not in self.problem_classes:
            reasons.append(f"problem_class:{features.problem_class.value}")
        if features.continuous_variables and not self.supports_continuous:
            reasons.append("continuous_variables")
        if features.integer_variables and not self.supports_integer:
            reasons.append("integer_variables")
        if features.binary_variables and not self.supports_binary:
            reasons.append("binary_variables")
        if features.finite_domain_variables and not self.supports_finite_domain:
            reasons.append("finite_domain_variables")
        if features.semi_continuous_variables and not self.supports_semi_continuous:
            reasons.append("semi_continuous_variables")
        if features.semi_integer_variables and not self.supports_semi_integer:
            reasons.append("semi_integer_variables")
        if features.custom_domain_variables:
            reasons.append("custom_domain_variables")
        if features.has_quadratic_objective and not self.supports_quadratic_objective:
            reasons.append("quadratic_objective")
        if features.has_callable_objective and not self.supports_callable_objective:
            reasons.append("callable_objective")
        if features.has_nonlinear_constraints and not self.supports_nonlinear_constraints:
            reasons.append("nonlinear_constraints")
        if features.objective_count > 1 and not self.supports_multiple_objectives:
            reasons.append("multiple_objectives")
        if self.requires_finite_bounds and features.has_unbounded_variables:
            reasons.append("finite_bounds_required")
        return tuple(reasons)


class SolverStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    LIMIT_REACHED = "limit_reached"
    TIMEOUT = "timeout"
    NUMERICAL_FAILURE = "numerical_failure"
    INTERRUPTED = "interrupted"
    SOLVER_UNAVAILABLE = "solver_unavailable"
    UNSUPPORTED = "unsupported_problem"
    ERROR = "error"


class FeasibilityStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


class OptimalityStatus(str, Enum):
    GLOBAL = "globally_optimal"
    LOCAL = "locally_optimal"
    PARETO = "pareto_optimal_or_nondominated"
    NONE = "none"
    UNKNOWN = "unknown"


class TerminationReason(str, Enum):
    OPTIMAL = "optimal"
    CONVERGED = "converged"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    ITERATION_LIMIT = "iteration_limit"
    EVALUATION_LIMIT = "evaluation_limit"
    NODE_LIMIT = "node_limit"
    TIME_LIMIT = "time_limit"
    NUMERICAL_FAILURE = "numerical_failure"
    INTERRUPTED = "interrupted"
    SOLVER_UNAVAILABLE = "solver_unavailable"
    UNSUPPORTED = "unsupported_problem"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Solution:
    solver_name: str
    problem_class: ProblemClass
    status: SolverStatus
    termination_reason: TerminationReason
    success: bool
    primal_solution: Mapping[str, Any] | None = None
    objective_values: Mapping[str, float] = field(default_factory=dict)
    feasibility: FeasibilityStatus = FeasibilityStatus.UNKNOWN
    optimality: OptimalityStatus = OptimalityStatus.UNKNOWN
    max_constraint_violation: float | None = None
    primal_bound: float | None = None
    dual_bound: float | None = None
    absolute_gap: float | None = None
    relative_gap: float | None = None
    iteration_count: int | None = None
    node_count: int | None = None
    function_evaluation_count: int | None = None
    runtime_seconds: float | None = None
    solver_version: str | None = None
    warnings: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.solver_name, str) or not self.solver_name:
            raise OptimizationValidationError("solution.solver_name must be non-empty")
        if not isinstance(self.problem_class, ProblemClass):
            raise OptimizationValidationError("solution.problem_class must be ProblemClass")
        if not isinstance(self.status, SolverStatus) or not isinstance(self.termination_reason, TerminationReason):
            raise OptimizationValidationError("solution status/termination must use canonical enums")
        if not isinstance(self.feasibility, FeasibilityStatus) or not isinstance(self.optimality, OptimalityStatus):
            raise OptimizationValidationError("solution feasibility/optimality must use canonical enums")
        for name, value in self.objective_values.items():
            if not isinstance(name, str) or to_finite_float(value) is None:
                raise OptimizationValidationError("objective_values must contain finite numeric values")
        for field_name in (
            "max_constraint_violation",
            "primal_bound",
            "dual_bound",
            "absolute_gap",
            "relative_gap",
            "runtime_seconds",
        ):
            value = getattr(self, field_name)
            if value is not None and to_finite_float(value) is None:
                raise OptimizationValidationError(f"solution.{field_name} must be finite when present")
        for field_name in ("absolute_gap", "relative_gap", "runtime_seconds", "max_constraint_violation"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise OptimizationValidationError(f"solution.{field_name} must be non-negative")
        for field_name in ("iteration_count", "node_count", "function_evaluation_count"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise OptimizationValidationError(f"solution.{field_name} must be a non-negative integer")
        if self.success and self.primal_solution is None:
            raise OptimizationValidationError("successful solution requires primal_solution")
        terminal_failure_statuses = {
            SolverStatus.INFEASIBLE,
            SolverStatus.UNBOUNDED,
            SolverStatus.INFEASIBLE_OR_UNBOUNDED,
            SolverStatus.SOLVER_UNAVAILABLE,
            SolverStatus.UNSUPPORTED,
        }
        if self.status in terminal_failure_statuses and self.success:
            raise OptimizationValidationError(
                "terminal failure status cannot be marked successful",
                context={"status": self.status.value},
            )
        if self.status is SolverStatus.OPTIMAL and not self.success:
            raise OptimizationValidationError("OPTIMAL status must be successful")
        if self.success and self.feasibility is FeasibilityStatus.INFEASIBLE:
            raise OptimizationValidationError("successful solution cannot be explicitly infeasible")
        if self.status is SolverStatus.OPTIMAL and self.optimality not in {OptimalityStatus.GLOBAL, OptimalityStatus.LOCAL}:
            raise OptimizationValidationError("OPTIMAL status requires explicit local/global optimality semantics")
        expected_termination = {
            SolverStatus.INFEASIBLE: TerminationReason.INFEASIBLE,
            SolverStatus.UNBOUNDED: TerminationReason.UNBOUNDED,
            SolverStatus.INFEASIBLE_OR_UNBOUNDED: TerminationReason.INFEASIBLE_OR_UNBOUNDED,
            SolverStatus.TIMEOUT: TerminationReason.TIME_LIMIT,
            SolverStatus.SOLVER_UNAVAILABLE: TerminationReason.SOLVER_UNAVAILABLE,
            SolverStatus.UNSUPPORTED: TerminationReason.UNSUPPORTED,
        }.get(self.status)
        if expected_termination is not None and self.termination_reason is not expected_termination:
            raise OptimizationValidationError(
                "solver status and termination reason are inconsistent",
                context={
                    "status": self.status.value,
                    "termination_reason": self.termination_reason.value,
                    "expected": expected_termination.value,
                },
            )

    @property
    def objective_value(self) -> float | None:
        return next(iter(self.objective_values.values())) if len(self.objective_values) == 1 else None


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    decision: Mapping[str, Any]
    objective_values: Mapping[str, float]
    feasible: bool = True
    violation: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        violation = require_finite(self.violation, "violation")
        if violation < 0:
            raise OptimizationValidationError("Pareto point violation must be non-negative")
        for name, value in self.objective_values.items():
            if not isinstance(name, str) or to_finite_float(value) is None:
                raise OptimizationValidationError("Pareto point objectives must be finite")
        object.__setattr__(self, "violation", violation)


@dataclass(frozen=True, slots=True)
class ParetoFront:
    points: tuple[ParetoPoint, ...]
    objective_names: tuple[str, ...]
    approximate: bool = False
    algorithm: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.objective_names)) != len(self.objective_names):
            raise OptimizationValidationError("Pareto objective_names must be unique")
        for point in self.points:
            if tuple(point.objective_values.keys()) != self.objective_names:
                if set(point.objective_values) != set(self.objective_names):
                    raise OptimizationValidationError("Pareto point objective set does not match front")

    def __iter__(self) -> Iterator[ParetoPoint]:
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int | slice) -> ParetoPoint | tuple[ParetoPoint, ...]:
        return self.points[index]

    def __bool__(self) -> bool:
        return bool(self.points)


@dataclass(frozen=True, slots=True)
class SolverAttempt:
    solver_name: str
    available: bool
    compatible: bool
    preference_score: float | None = None
    incompatibilities: tuple[str, ...] = ()
    status: SolverStatus | None = None
    termination_reason: TerminationReason | None = None
    runtime_seconds: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.solver_name, str) or not self.solver_name:
            raise OptimizationValidationError("solver attempt name must be non-empty")
        if self.preference_score is not None and to_finite_float(self.preference_score) is None:
            raise OptimizationValidationError("solver attempt preference_score must be finite")
        if self.runtime_seconds is not None:
            runtime = to_finite_float(self.runtime_seconds)
            if runtime is None or runtime < 0:
                raise OptimizationValidationError("solver attempt runtime_seconds must be non-negative and finite")
        if self.status is not None and not isinstance(self.status, SolverStatus):
            raise OptimizationValidationError("solver attempt status must be SolverStatus")
        if self.termination_reason is not None and not isinstance(self.termination_reason, TerminationReason):
            raise OptimizationValidationError("solver attempt termination_reason must be TerminationReason")


@dataclass(frozen=True, slots=True)
class SolverSelection:
    selected_solver: str | None
    attempts: tuple[SolverAttempt, ...]
    requested_solver: str | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(attempt, SolverAttempt) for attempt in self.attempts):
            raise OptimizationValidationError("solver selection attempts must contain SolverAttempt objects")
        names = [attempt.solver_name for attempt in self.attempts]
        if len(names) != len(set(names)):
            raise OptimizationValidationError("solver selection attempts must have unique solver names")
        if self.selected_solver is not None and self.selected_solver not in names:
            raise OptimizationValidationError(
                "selected solver must be represented in selection attempts",
                context={"selected_solver": self.selected_solver, "attempts": names},
            )


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    solution: Solution | None
    pareto_front: ParetoFront | None = None
    selection: SolverSelection | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.solution is None and self.pareto_front is None:
            raise OptimizationValidationError("OptimizationResult requires solution or Pareto front")


__all__ = [
    "Constraint",
    "ConstraintSense",
    "DomainKind",
    "FeasibilityStatus",
    "GradientCallable",
    "HessianCallable",
    "JacobianCallable",
    "Objective",
    "ObjectiveSense",
    "OptimalityStatus",
    "OptimizationProblem",
    "OptimizationResult",
    "ParetoFront",
    "ParetoPoint",
    "Point",
    "ProblemClass",
    "ProblemFeatures",
    "ScalarCallable",
    "Solution",
    "SolverAttempt",
    "SolverCapability",
    "SolverSelection",
    "SolverStatus",
    "TerminationReason",
    "Variable",
    "VariableDomain",
]
