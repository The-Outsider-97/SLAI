# stem_types.py
"""
Production-ready STEM domain types.

This module defines the shared scientific value objects for the STEM subsystem.
It deliberately contains only class-specific behaviour. All reusable validation,
conversion, comparison, uncertainty-combination, and serialization helpers are
provided by ``stem_helpers.py``. All exceptions are provided by
``stem_errors.py``.
"""

from __future__ import annotations

__version__ = "2.3.0"

import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union, cast

from .utils.config_loader import get_config_section, load_global_config
from .utils.stem_errors import *
from .utils.stem_helpers import (
    affine_from_base,
    affine_to_base,
    combine_exponents,
    combine_standard_uncertainties,
    ensure_finite_number,
    ensure_mapping,
    ensure_non_empty_string,
    ensure_non_negative,
    ensure_number,
    ensure_positive,
    ensure_sequence,
    format_dimension,
    isclose as helper_isclose,
    json_safe,
    normalize_exponents,
    power_exponents,
)


_DOMAIN_VALIDATION_ERROR = cast(type[STEMValidationError], STEMDomainError)
_EQUATION_VALIDATION_ERROR = cast(type[STEMValidationError], STEMEquationError)
_PHYSICAL_CONSTANT_VALIDATION_ERROR = cast(type[STEMValidationError], STEMPhysicalConstantError)
_SOLVER_VALIDATION_ERROR = cast(type[STEMValidationError], STEMSolverError)


def _load_stem_types_config() -> Dict[str, Any]:
    """Load optional STEM type defaults without making configuration mandatory."""
    try:
        config = load_global_config()
        return dict(get_config_section("stem_types", config=config) or {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Dimensions and units
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Dimension:
    """
    Algebraic dimension vector.

    Exponents are stored as a normalized mapping from base-dimension symbol to
    finite exponent. Multiplication, division, and exponentiation combine
    exponents according to Kennedy-style dimension algebra.
    """

    exponents: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = normalize_exponents(self.exponents, error_cls=STEMDimensionError)
        object.__setattr__(self, "exponents", normalized)

    @property
    def is_dimensionless(self) -> bool:
        return not self.exponents

    def __mul__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(combine_exponents(self.exponents, other.exponents, sign=1))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(combine_exponents(self.exponents, other.exponents, sign=-1))

    def __pow__(self, power: float) -> "Dimension":
        return Dimension(power_exponents(self.exponents, power))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self.exponents == other.exponents

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.exponents.items())))

    def to_dict(self) -> Dict[str, float]:
        return dict(self.exponents)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dimension":
        return cls(data)

    def __str__(self) -> str:
        return format_dimension(self.exponents) or "1"


@dataclass(frozen=True, eq=False)
class Unit:
    """
    Unit of measure with linear or affine conversion to a base unit.

    ``to_base`` implements ``base = value * scale + offset``. Therefore affine
    units such as degree Celsius can use ``scale=1.0`` and ``offset=273.15``.
    """

    symbol: str
    name: Optional[str] = None
    dimension: Dimension = field(default_factory=Dimension)
    scale: float = 1.0
    offset: float = 0.0
    system: str = "SI"

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.symbol, "unit.symbol")
        if self.name is not None:
            ensure_non_empty_string(self.name, "unit.name")
        ensure_positive(self.scale, "unit.scale", allow_zero=False, error_cls=STEMUnitError)
        ensure_finite_number(self.offset, "unit.offset", error_cls=STEMUnitError)
        if not isinstance(self.dimension, Dimension):
            raise STEMUnitError(
                "unit.dimension must be a Dimension instance",
                context={"received": type(self.dimension).__name__},
            )
        ensure_non_empty_string(self.system, "unit.system")

    @property
    def is_affine(self) -> bool:
        return self.offset != 0.0

    def to_base(self, value: float) -> float:
        ensure_number(value, "value", error_cls=STEMUnitError)
        return affine_to_base(float(value), self.scale, self.offset)

    def from_base(self, base: float) -> float:
        ensure_number(base, "base", error_cls=STEMUnitError)
        return affine_from_base(float(base), self.scale, self.offset)

    def convert(self, value: float, target: "Unit") -> float:
        if not isinstance(target, Unit):
            raise STEMUnitError(
                "target must be a Unit",
                context={"received": type(target).__name__},
            )
        if self.dimension != target.dimension:
            raise STEMUnitError(
                "Cannot convert between units with different dimensions",
                context={"from": self.symbol, "to": target.symbol},
            )
        return target.from_base(self.to_base(value))

    def __mul__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        if self.is_affine or other.is_affine:
            raise STEMUnitError("Affine units cannot be multiplied directly")
        return Unit(
            f"{self.symbol}*{other.symbol}",
            dimension=self.dimension * other.dimension,
            scale=self.scale * other.scale,
        )

    def __truediv__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        if self.is_affine or other.is_affine:
            raise STEMUnitError("Affine units cannot be divided directly")
        return Unit(
            f"{self.symbol}/{other.symbol}",
            dimension=self.dimension / other.dimension,
            scale=self.scale / other.scale,
        )

    def __pow__(self, power: float) -> "Unit":
        if self.is_affine:
            raise STEMUnitError("Affine units cannot be exponentiated directly")
        return Unit(
            f"{self.symbol}^{power}",
            dimension=self.dimension ** power,
            scale=self.scale ** power,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return (
            self.symbol == other.symbol
            and self.name == other.name
            and self.dimension == other.dimension
            and self.scale == other.scale
            and self.offset == other.offset
            and self.system == other.system
        )

    def __hash__(self) -> int:
        return hash((self.symbol, self.dimension, self.scale, self.offset, self.system))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "dimension": self.dimension.to_dict(),
            "scale": self.scale,
            "offset": self.offset,
            "system": self.system,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Unit":
        return cls(
            symbol=str(data["symbol"]),
            name=data.get("name"),
            dimension=Dimension.from_dict(data.get("dimension", {})),
            scale=float(data.get("scale", 1.0)),
            offset=float(data.get("offset", 0.0)),
            system=str(data.get("system", "SI")),
        )

    def __str__(self) -> str:
        return self.symbol


# ---------------------------------------------------------------------------
# Precision and tolerance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrecisionPolicy:
    """IEEE-754-aware precision, rounding, and comparison policy."""

    significant_digits: Optional[int] = None
    decimal_places: Optional[int] = None
    rounding_mode: str = "ROUND_HALF_EVEN"
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-9
    nan_policy: str = "equal"

    def __post_init__(self) -> None:
        if self.significant_digits is not None:
            ensure_positive(
                self.significant_digits,
                "significant_digits",
                allow_zero=False,
                error_cls=STEMValidationError,
            )
            object.__setattr__(self, "significant_digits", int(self.significant_digits))

        if self.decimal_places is not None:
            ensure_non_negative(
                self.decimal_places,
                "decimal_places",
                allow_zero=True,
                error_cls=STEMValidationError,
            )
            object.__setattr__(self, "decimal_places", int(self.decimal_places))

        allowed_rounding = {
            "ROUND_HALF_EVEN",
            "ROUND_HALF_UP",
            "ROUND_HALF_DOWN",
            "ROUND_UP",
            "ROUND_DOWN",
            "ROUND_CEILING",
            "ROUND_FLOOR",
        }
        if self.rounding_mode not in allowed_rounding:
            raise STEMValidationError(
                "Invalid IEEE-754 rounding mode",
                context={"rounding_mode": self.rounding_mode, "allowed": sorted(allowed_rounding)},
            )

        ensure_non_negative(
            self.absolute_tolerance,
            "absolute_tolerance",
            allow_zero=True,
            error_cls=STEMValidationError,
        )
        ensure_non_negative(
            self.relative_tolerance,
            "relative_tolerance",
            allow_zero=True,
            error_cls=STEMValidationError,
        )

        if self.nan_policy not in {"equal", "never", "always"}:
            raise STEMValidationError(
                "nan_policy must be one of 'equal', 'never', or 'always'",
                context={"nan_policy": self.nan_policy},
            )

    @classmethod
    def default(cls) -> "PrecisionPolicy":
        cfg = _load_stem_types_config().get("precision", {})
        return cls(
            significant_digits=cfg.get("significant_digits"),
            decimal_places=cfg.get("decimal_places"),
            rounding_mode=cfg.get("rounding_mode", "ROUND_HALF_EVEN"),
            absolute_tolerance=cfg.get("absolute_tolerance", 1e-12),
            relative_tolerance=cfg.get("relative_tolerance", 1e-9),
            nan_policy=cfg.get("nan_policy", "equal"),
        )

    def isclose(self, a: float, b: float) -> bool:
        return helper_isclose(
            a,
            b,
            rel_tol=self.relative_tolerance,
            abs_tol=self.absolute_tolerance,
            nan_policy=self.nan_policy,
        )

    def round(self, value: float) -> float:
        ensure_number(value, "value", error_cls=STEMValidationError)
        value_f = float(value)

        if self.decimal_places is not None:
            return round(value_f, self.decimal_places)

        if self.significant_digits is not None:
            if value_f == 0.0 or not math.isfinite(value_f):
                return value_f
            digits = self.significant_digits - int(math.floor(math.log10(abs(value_f)))) - 1
            return round(value_f, digits)

        return value_f


@dataclass(frozen=True)
class Tolerance:
    """Tolerance interval with absolute/relative bounds and optional policy."""

    lower: Optional[float] = None
    upper: Optional[float] = None
    absolute: Optional[float] = None
    relative: Optional[float] = None
    inclusive_min: bool = True
    inclusive_max: bool = True
    policy: Optional[PrecisionPolicy] = None

    def __post_init__(self) -> None:
        if self.lower is not None:
            ensure_finite_number(self.lower, "lower", error_cls=STEMValidationError)
        if self.upper is not None:
            ensure_finite_number(self.upper, "upper", error_cls=STEMValidationError)
        if self.absolute is not None:
            ensure_non_negative(self.absolute, "absolute", allow_zero=True, error_cls=STEMValidationError)
        if self.relative is not None:
            ensure_non_negative(self.relative, "relative", allow_zero=True, error_cls=STEMValidationError)
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise STEMValidationError("lower must be <= upper")
        if self.policy is not None and not isinstance(self.policy, PrecisionPolicy):
            raise STEMValidationError("policy must be a PrecisionPolicy")

    def contains(self, value: float, nominal: Optional[float] = None) -> bool:
        ensure_number(value, "value", error_cls=STEMValidationError)
        v = float(value)

        if self.lower is not None:
            if self.inclusive_min:
                if v < self.lower:
                    return False
            elif v <= self.lower:
                return False

        if self.upper is not None:
            if self.inclusive_max:
                if v > self.upper:
                    return False
            elif v >= self.upper:
                return False

        if nominal is not None:
            ensure_number(nominal, "nominal", error_cls=STEMValidationError)
            n = float(nominal)
            if self.absolute is not None and abs(v - n) > self.absolute:
                return False
            if self.relative is not None:
                denom = abs(n) if n != 0.0 else 1.0
                if abs(v - n) / denom > self.relative:
                    return False

        return True


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
class UncertaintyType(str, Enum):
    TYPE_A = "type_a"
    TYPE_B = "type_b"


class Distribution(str, Enum):
    NORMAL = "normal"
    RECTANGULAR = "rectangular"
    TRIANGULAR = "triangular"
    U_SHAPED = "u_shaped"


@dataclass(frozen=True)
class Uncertainty:
    """JCGM-style standard uncertainty with optional coverage and dof."""

    value: float
    coverage_factor: float = 1.0
    distribution: Distribution = Distribution.NORMAL
    uncertainty_type: UncertaintyType = UncertaintyType.TYPE_B
    degrees_freedom: Optional[float] = None
    unit: Optional[Unit] = None

    def __post_init__(self) -> None:
        ensure_non_negative(self.value, "uncertainty.value", allow_zero=True, error_cls=STEMUncertaintyError)
        ensure_positive(
            self.coverage_factor,
            "uncertainty.coverage_factor",
            allow_zero=False,
            error_cls=STEMUncertaintyError,
        )

        if not isinstance(self.distribution, Distribution):
            try:
                object.__setattr__(self, "distribution", Distribution(self.distribution))
            except Exception as exc:
                raise STEMUncertaintyError(
                    "Invalid uncertainty distribution",
                    context={"distribution": self.distribution},
                    cause=exc,
                ) from exc

        if not isinstance(self.uncertainty_type, UncertaintyType):
            try:
                object.__setattr__(self, "uncertainty_type", UncertaintyType(self.uncertainty_type))
            except Exception as exc:
                raise STEMUncertaintyError(
                    "Invalid uncertainty type",
                    context={"uncertainty_type": self.uncertainty_type},
                    cause=exc,
                ) from exc

        if self.degrees_freedom is not None:
            ensure_positive(
                self.degrees_freedom,
                "uncertainty.degrees_freedom",
                allow_zero=False,
                error_cls=STEMUncertaintyError,
            )

        if self.unit is not None and not isinstance(self.unit, Unit):
            raise STEMUncertaintyError("uncertainty.unit must be a Unit")

    def standard(self) -> float:
        return float(self.value)

    def expanded(self) -> float:
        return float(self.value) * float(self.coverage_factor)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "coverage_factor": self.coverage_factor,
            "distribution": self.distribution.value,
            "uncertainty_type": self.uncertainty_type.value,
            "degrees_freedom": self.degrees_freedom,
            "unit": self.unit.to_dict() if self.unit else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Uncertainty":
        unit_data = data.get("unit")
        unit = Unit.from_dict(unit_data) if isinstance(unit_data, Mapping) else None
        return cls(
            value=float(data["value"]),
            coverage_factor=float(data.get("coverage_factor", 1.0)),
            distribution=Distribution(data.get("distribution", Distribution.NORMAL.value)),
            uncertainty_type=UncertaintyType(data.get("uncertainty_type", UncertaintyType.TYPE_B.value)),
            degrees_freedom=data.get("degrees_freedom"),
            unit=unit,
        )


@dataclass
class UncertaintyBudget:
    """Combined uncertainty budget using RSS/correlation and Welch-Satterthwaite."""

    components: Sequence[Uncertainty]
    correlations: Optional[Mapping[Tuple[int, int], float]] = None
    coverage_factor: float = 2.0
    combined: Optional[float] = None
    expanded: Optional[float] = None
    effective_degrees_freedom: Optional[float] = None

    def __post_init__(self) -> None:
        ensure_sequence(
            self.components,
            "components",
            error_cls=cast(type[STEMValidationError], STEMUncertaintyError),
        )
        if not self.components:
            raise STEMUncertaintyError("UncertaintyBudget requires at least one component")

        for index, component in enumerate(self.components):
            if not isinstance(component, Uncertainty):
                raise STEMUncertaintyError(
                    f"components[{index}] must be an Uncertainty",
                    context={"received": type(component).__name__},
                )

        ensure_positive(
            self.coverage_factor,
            "coverage_factor",
            allow_zero=False,
            error_cls=STEMUncertaintyError,
        )

        values = [component.standard() for component in self.components]
        dof = [component.degrees_freedom for component in self.components]
        combined, effective_dof = combine_standard_uncertainties(
            values,
            correlations=self.correlations,
            dof=dof,
        )

        self.combined = combined
        self.expanded = combined * float(self.coverage_factor)
        self.effective_degrees_freedom = effective_dof


# ---------------------------------------------------------------------------
# Solver status and results
# ---------------------------------------------------------------------------
class ConvergenceStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    MAX_ITERATIONS = "max_iterations"
    STALLED = "stalled"
    FAILED = "failed"


@dataclass
class NumericResult:
    """Generic numeric result wrapper."""

    value: Optional[float]
    unit: Optional[Unit] = None
    uncertainty: Optional[Uncertainty] = None
    status: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value is not None:
            ensure_number(self.value, "value", error_cls=STEMValidationError)
        if self.unit is not None and not isinstance(self.unit, Unit):
            raise STEMValidationError("unit must be a Unit")
        if self.uncertainty is not None and not isinstance(self.uncertainty, Uncertainty):
            raise STEMValidationError("uncertainty must be an Uncertainty")
        ensure_mapping(self.metadata, "metadata", error_cls=STEMValidationError)

    def ok(self) -> bool:
        return self.value is not None

    def failed(self) -> bool:
        return not self.ok()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit.to_dict() if self.unit else None,
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "status": self.status,
            "metadata": json_safe(self.metadata),
        }


@dataclass
class SolverResult:
    """Structured solver outcome with consistency checks."""

    solution: Any
    residual: Optional[float]
    iterations: int
    converged: bool
    status: ConvergenceStatus
    message: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    runtime: Optional[float] = None

    def __post_init__(self) -> None:
        ensure_non_negative(self.iterations, "iterations", allow_zero=True, error_cls=STEMSolverError)
        self.iterations = int(self.iterations)

        if self.residual is not None:
            ensure_finite_number(
                self.residual,
                "residual",
                allow_nan=False,
                allow_inf=False,
                error_cls=STEMSolverError,
            )

        if not isinstance(self.status, ConvergenceStatus):
            try:
                self.status = ConvergenceStatus(self.status)
            except Exception as exc:
                raise STEMSolverError(
                    "Invalid convergence status",
                    context={"status": self.status},
                    cause=exc,
                ) from exc

        if self.converged and self.status != ConvergenceStatus.CONVERGED:
            raise STEMSolverError("converged=True requires status CONVERGED")
        if not self.converged and self.status == ConvergenceStatus.CONVERGED:
            raise STEMSolverError("converged=False cannot have status CONVERGED")

        ensure_mapping(self.diagnostics, "diagnostics", error_cls=_SOLVER_VALIDATION_ERROR)

        if self.runtime is not None:
            ensure_non_negative(self.runtime, "runtime", allow_zero=True, error_cls=STEMSolverError)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution": json_safe(self.solution),
            "residual": self.residual,
            "iterations": self.iterations,
            "converged": self.converged,
            "status": self.status.value,
            "message": self.message,
            "diagnostics": json_safe(self.diagnostics),
            "runtime": self.runtime,
        }


# ---------------------------------------------------------------------------
# Boundary and initial conditions
# ---------------------------------------------------------------------------
class BoundaryConditionKind(str, Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"
    INITIAL = "initial"


@dataclass(frozen=True, eq=False)
class BoundaryCondition:
    """Boundary or initial condition specification."""

    kind: BoundaryConditionKind
    location: Any
    value: Optional[float] = None
    function: Optional[Callable[..., Any]] = None
    unit: Optional[Unit] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryConditionKind):
            try:
                object.__setattr__(self, "kind", BoundaryConditionKind(self.kind))
            except Exception as exc:
                raise STEMValidationError(
                    "Invalid boundary condition kind",
                    context={"kind": self.kind},
                    cause=exc,
                ) from exc

        if self.value is None and self.function is None:
            raise STEMValidationError("BoundaryCondition requires value or function")

        if self.value is not None:
            ensure_number(self.value, "value", error_cls=STEMValidationError)

        if self.function is not None and not callable(self.function):
            raise STEMValidationError("function must be callable")

        if self.unit is not None and not isinstance(self.unit, Unit):
            raise STEMValidationError("unit must be a Unit")

        ensure_mapping(self.parameters, "parameters", error_cls=STEMValidationError)


@dataclass(frozen=True, eq=False)
class InitialCondition:
    """Initial condition for time-dependent problems."""

    time: float = 0.0
    state: Any = None
    value: Optional[float] = None
    unit: Optional[Unit] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_finite_number(self.time, "time", error_cls=STEMValidationError)
        if float(self.time) < 0.0:
            raise STEMValidationError("time must be non-negative")

        if self.state is None and self.value is None:
            raise STEMValidationError("InitialCondition requires state or value")

        if self.value is not None:
            ensure_number(self.value, "value", error_cls=STEMValidationError)

        if self.unit is not None and not isinstance(self.unit, Unit):
            raise STEMValidationError("unit must be a Unit")

        ensure_mapping(self.metadata, "metadata", error_cls=STEMValidationError)


# ---------------------------------------------------------------------------
# Domains and equations
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Domain:
    """Coordinate domain with ordered bounds and optional mesh."""

    bounds: Sequence[Tuple[float, float]]
    dimension: Dimension = field(default_factory=Dimension)
    coordinate_system: str = "cartesian"
    mesh: Optional[Any] = None

    def __post_init__(self) -> None:
        ensure_sequence(self.bounds, "bounds", allow_str=False, error_cls=_DOMAIN_VALIDATION_ERROR)

        normalized_bounds = []
        for index, bound in enumerate(self.bounds):
            if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                raise STEMDomainError(
                    f"bounds[{index}] must be a (lower, upper) pair",
                    context={"bounds": self.bounds},
                )

            lower = ensure_finite_number(
                bound[0],
                f"bounds[{index}].lower",
                error_cls=STEMDomainError,
            )
            upper = ensure_finite_number(
                bound[1],
                f"bounds[{index}].upper",
                error_cls=STEMDomainError,
            )

            if lower > upper:
                raise STEMDomainError(f"bounds[{index}] lower must be <= upper")

            normalized_bounds.append((lower, upper))

        object.__setattr__(self, "bounds", tuple(normalized_bounds))

        if not isinstance(self.dimension, Dimension):
            raise STEMDomainError("dimension must be a Dimension")

        ensure_non_empty_string(
            self.coordinate_system,
            "coordinate_system",
            error_cls=_DOMAIN_VALIDATION_ERROR,
        )

    def contains(self, point: Sequence[float]) -> bool:
        ensure_sequence(point, "point", allow_str=False, error_cls=_DOMAIN_VALIDATION_ERROR)

        if len(point) != len(self.bounds):
            raise STEMDomainError(
                "point dimensionality does not match domain bounds",
                context={"point_dim": len(point), "domain_dim": len(self.bounds)},
            )

        for value, (lower, upper) in zip(point, self.bounds):
            coordinate = ensure_finite_number(value, "point coordinate", error_cls=STEMDomainError)
            if coordinate < lower or coordinate > upper:
                return False

        return True

    def volume(self) -> float:
        volume = 1.0
        for lower, upper in self.bounds:
            volume *= upper - lower
        return volume


@dataclass
class Equation:
    """Equation represented by a safe expression or a callable."""

    name: str
    expression: Optional[str] = None
    callable: Optional[Callable[..., Any]] = None
    variables: Sequence[str] = field(default_factory=tuple)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    residual: Optional[Callable[..., Any]] = None
    domain: Optional[Domain] = None

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.name, "name", error_cls=_EQUATION_VALIDATION_ERROR)

        provided = int(self.expression is not None) + int(self.callable is not None)
        if provided != 1:
            raise STEMEquationError("Exactly one of expression or callable must be provided")

        if self.expression is not None:
            ensure_non_empty_string(
                self.expression,
                "expression",
                error_cls=_EQUATION_VALIDATION_ERROR,
            )

        if self.callable is not None and not callable(self.callable):
            raise STEMEquationError("callable must be callable")

        ensure_sequence(self.variables, "variables", allow_str=False, error_cls=_EQUATION_VALIDATION_ERROR)
        ensure_mapping(self.parameters, "parameters", error_cls=_EQUATION_VALIDATION_ERROR)

        if self.residual is not None and not callable(self.residual):
            raise STEMEquationError("residual must be callable")

        if self.domain is not None and not isinstance(self.domain, Domain):
            raise STEMEquationError("domain must be a Domain")

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        if self.callable is not None:
            return self.callable(*args, **kwargs)

        env = dict(self.parameters)
        env.update(kwargs)

        try:
            return eval(self.expression, {"__builtins__": {}}, env)  # type: ignore[arg-type]
        except Exception as exc:
            raise STEMEquationError(
                "Failed to evaluate expression",
                context={"expression": self.expression},
                cause=exc,
            ) from exc

    def residual_for(self, *args: Any, **kwargs: Any) -> Any:
        if self.residual is not None:
            return self.residual(*args, **kwargs)
        return self.evaluate(*args, **kwargs)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicalConstant:
    """Physical constant with unit, uncertainty, exactness, and source."""

    name: str
    symbol: str
    value: float
    unit: Unit
    uncertainty: Optional[Uncertainty] = None
    exact: bool = False
    source: Optional[str] = None

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.name, "name", error_cls=_PHYSICAL_CONSTANT_VALIDATION_ERROR)
        ensure_non_empty_string(self.symbol, "symbol", error_cls=_PHYSICAL_CONSTANT_VALIDATION_ERROR)
        ensure_finite_number(self.value, "value", error_cls=STEMPhysicalConstantError)

        if not isinstance(self.unit, Unit):
            raise STEMPhysicalConstantError("unit must be a Unit")

        if self.uncertainty is not None and not isinstance(self.uncertainty, Uncertainty):
            raise STEMPhysicalConstantError("uncertainty must be an Uncertainty")

        if self.exact and self.uncertainty is not None:
            raise STEMPhysicalConstantError("Exact constants must not have uncertainty")

        if self.source is not None:
            ensure_non_empty_string(self.source, "source", error_cls=_PHYSICAL_CONSTANT_VALIDATION_ERROR)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "value": self.value,
            "unit": self.unit.to_dict(),
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "exact": self.exact,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Quantity:
    """Magnitude with unit, optional uncertainty, and optional precision policy."""

    magnitude: float
    unit: Unit
    uncertainty: Optional[Uncertainty] = None
    precision: Optional[PrecisionPolicy] = None

    def __post_init__(self) -> None:
        ensure_number(self.magnitude, "magnitude", error_cls=STEMValidationError)
        object.__setattr__(self, "magnitude", float(self.magnitude))

        if not isinstance(self.unit, Unit):
            raise STEMValidationError("unit must be a Unit")

        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, Uncertainty):
                raise STEMValidationError("uncertainty must be an Uncertainty")
            if (
                self.uncertainty.unit is not None
                and self.uncertainty.unit.dimension != self.unit.dimension
            ):
                raise STEMValidationError(
                    "uncertainty unit dimension must match quantity unit dimension"
                )

        if self.precision is not None and not isinstance(self.precision, PrecisionPolicy):
            raise STEMValidationError("precision must be a PrecisionPolicy")

    def convert(self, target_unit: Unit) -> "Quantity":
        if not isinstance(target_unit, Unit):
            raise STEMUnitError("target_unit must be a Unit")

        converted_magnitude = self.unit.convert(self.magnitude, target_unit)
        converted_uncertainty = self.uncertainty

        if self.uncertainty is not None:
            uncertainty_value = self.uncertainty.value

            if self.unit.is_affine or target_unit.is_affine:
                scale = self.unit.scale / target_unit.scale
                uncertainty_value = uncertainty_value * scale
            else:
                uncertainty_value = self.unit.convert(uncertainty_value, target_unit)

            converted_uncertainty = Uncertainty(
                value=abs(uncertainty_value),
                coverage_factor=self.uncertainty.coverage_factor,
                distribution=self.uncertainty.distribution,
                uncertainty_type=self.uncertainty.uncertainty_type,
                degrees_freedom=self.uncertainty.degrees_freedom,
                unit=target_unit,
            )

        return Quantity(
            converted_magnitude,
            target_unit,
            converted_uncertainty,
            self.precision,
        )

    def __add__(self, other: "Quantity") -> "Quantity":
        if not isinstance(other, Quantity):
            return NotImplemented

        other_converted = other.convert(self.unit)
        magnitude = self.magnitude + other_converted.magnitude
        uncertainty = self._combine_uncertainties(other_converted)
        return Quantity(magnitude, self.unit, uncertainty, self.precision or other_converted.precision)

    def __sub__(self, other: "Quantity") -> "Quantity":
        if not isinstance(other, Quantity):
            return NotImplemented

        other_converted = other.convert(self.unit)
        magnitude = self.magnitude - other_converted.magnitude
        uncertainty = self._combine_uncertainties(other_converted)
        return Quantity(magnitude, self.unit, uncertainty, self.precision or other_converted.precision)

    def __mul__(self, other: Union["Quantity", int, float]) -> "Quantity":
        if isinstance(other, Quantity):
            new_unit = self.unit * other.unit
            magnitude = self.magnitude * other.magnitude
            uncertainty = self._combine_relative_uncertainties(other)
            return Quantity(magnitude, new_unit, uncertainty, self.precision or other.precision)

        ensure_number(other, "scalar", error_cls=STEMValidationError)
        return Quantity(self.magnitude * float(other), self.unit, self.uncertainty, self.precision)

    def __truediv__(self, other: Union["Quantity", int, float]) -> "Quantity":
        if isinstance(other, Quantity):
            new_unit = self.unit / other.unit
            magnitude = self.magnitude / other.magnitude
            uncertainty = self._combine_relative_uncertainties(other, divide=True)
            return Quantity(magnitude, new_unit, uncertainty, self.precision or other.precision)

        ensure_number(other, "scalar", error_cls=STEMValidationError)
        return Quantity(self.magnitude / float(other), self.unit, self.uncertainty, self.precision)

    def _combine_uncertainties(self, other: "Quantity") -> Optional[Uncertainty]:
        if self.uncertainty is None and other.uncertainty is None:
            return None

        u1 = self.uncertainty.standard() if self.uncertainty else 0.0
        u2 = other.uncertainty.standard() if other.uncertainty else 0.0
        combined, dof = combine_standard_uncertainties([u1, u2])

        return Uncertainty(value=combined, unit=self.unit, degrees_freedom=dof)

    def _combine_relative_uncertainties(self, other: "Quantity", divide: bool = False) -> Optional[Uncertainty]:
        if self.uncertainty is None and other.uncertainty is None:
            return None

        rel1 = (
            self.uncertainty.standard() / abs(self.magnitude)
            if self.uncertainty and self.magnitude != 0.0
            else 0.0
        )
        rel2 = (
            other.uncertainty.standard() / abs(other.magnitude)
            if other.uncertainty and other.magnitude != 0.0
            else 0.0
        )

        combined_rel = math.sqrt(rel1 ** 2 + rel2 ** 2)

        if divide:
            magnitude = abs(self.magnitude / other.magnitude) if other.magnitude != 0.0 else 0.0
            unit = self.unit / other.unit
        else:
            magnitude = abs(self.magnitude * other.magnitude)
            unit = self.unit * other.unit

        return Uncertainty(value=combined_rel * magnitude, unit=unit)

    def isclose(self, other: "Quantity", policy: Optional[PrecisionPolicy] = None) -> bool:
        if not isinstance(other, Quantity):
            return False

        other_converted = other.convert(self.unit)
        active_policy = policy or self.precision or other_converted.precision or PrecisionPolicy.default()
        return active_policy.isclose(self.magnitude, other_converted.magnitude)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "magnitude": self.magnitude,
            "unit": self.unit.to_dict(),
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "precision": {
                "significant_digits": self.precision.significant_digits,
                "decimal_places": self.precision.decimal_places,
                "rounding_mode": self.precision.rounding_mode,
                "absolute_tolerance": self.precision.absolute_tolerance,
                "relative_tolerance": self.precision.relative_tolerance,
                "nan_policy": self.precision.nan_policy,
            }
            if self.precision
            else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Quantity":
        unit_data = data.get("unit")
        if isinstance(unit_data, Mapping):
            unit = Unit.from_dict(unit_data)
        else:
            unit = Unit(symbol=str(unit_data or "1"))

        uncertainty_data = data.get("uncertainty")
        uncertainty = (
            Uncertainty.from_dict(uncertainty_data)
            if isinstance(uncertainty_data, Mapping)
            else None
        )

        return cls(
            magnitude=float(data["magnitude"]),
            unit=unit,
            uncertainty=uncertainty,
        )

    def __str__(self) -> str:
        if self.uncertainty is not None:
            return f"{self.magnitude} ± {self.uncertainty.expanded()} {self.unit.symbol}"
        return f"{self.magnitude} {self.unit.symbol}"


__all__ = [
    "NumericResult",
    "Quantity",
    "Unit",
    "Dimension",
    "PrecisionPolicy",
    "Tolerance",
    "Uncertainty",
    "UncertaintyBudget",
    "SolverResult",
    "ConvergenceStatus",
    "BoundaryCondition",
    "InitialCondition",
    "Equation",
    "Domain",
    "PhysicalConstant",
    "UncertaintyType",
    "Distribution",
    "BoundaryConditionKind",
]
