"""Shared scientific value objects and result contracts for SLAI STEM.

The contracts keep mathematical values, units, uncertainty, tolerances, and
solver diagnostics explicit so domain modules do not return incompatible
ad-hoc dictionaries.
"""
from __future__ import annotations

__version__ = "2.3.0"

import ast
import math

from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

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
    ensure_positive,
    ensure_sequence,
    format_dimension,
    isclose as helper_isclose,
    json_safe,
    normalize_exponents,
    power_exponents,
)


def _type_config() -> Dict[str, Any]:
    try:
        cfg = load_global_config()
        return dict(get_config_section("stem_types", config=cfg) or {})
    except Exception:
        return {}


@dataclass(frozen=True, eq=False)
class Dimension:
    exponents: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exponents", normalize_exponents(self.exponents, error_cls=STEMDimensionError))

    @property
    def is_dimensionless(self) -> bool:
        return not self.exponents

    def __mul__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(combine_exponents(self.exponents, other.exponents))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(combine_exponents(self.exponents, other.exponents, -1))

    def __pow__(self, power: float) -> "Dimension":
        return Dimension(power_exponents(self.exponents, power))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Dimension) and self.exponents == other.exponents

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.exponents.items())))

    def to_dict(self) -> Dict[str, float]:
        return dict(self.exponents)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dimension":
        return cls(data)

    def __str__(self) -> str:
        return format_dimension(self.exponents)


@dataclass(frozen=True, eq=False)
class Unit:
    symbol: str
    name: Optional[str] = None
    dimension: Dimension = field(default_factory=Dimension)
    scale: float = 1.0
    offset: float = 0.0
    system: str = "SI"

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.symbol, "unit.symbol", error_cls=STEMUnitError)
        if self.name is not None:
            ensure_non_empty_string(self.name, "unit.name", error_cls=STEMUnitError)
        ensure_positive(self.scale, "unit.scale", error_cls=STEMUnitError)
        ensure_finite_number(self.offset, "unit.offset", error_cls=STEMUnitError)
        if not isinstance(self.dimension, Dimension):
            raise STEMUnitError("unit.dimension must be a Dimension")
        ensure_non_empty_string(self.system, "unit.system", error_cls=STEMUnitError)

    @property
    def is_affine(self) -> bool:
        return self.offset != 0.0

    def to_base(self, value: float) -> float:
        return affine_to_base(value, self.scale, self.offset)

    def from_base(self, base: float) -> float:
        return affine_from_base(base, self.scale, self.offset)

    def convert(self, value: float, target: "Unit") -> float:
        if not isinstance(target, Unit):
            raise STEMUnitError("target must be a Unit")
        if self.dimension != target.dimension:
            raise STEMUnitError("Cannot convert incompatible units", context={"from": self.symbol, "to": target.symbol})
        return target.from_base(self.to_base(value))

    def __mul__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        if self.is_affine or other.is_affine:
            raise STEMUnitError("Affine units cannot be multiplied")
        return Unit(f"{self.symbol}*{other.symbol}", dimension=self.dimension * other.dimension, scale=self.scale * other.scale, system=self.system)

    def __truediv__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        if self.is_affine or other.is_affine:
            raise STEMUnitError("Affine units cannot be divided")
        return Unit(f"{self.symbol}/{other.symbol}", dimension=self.dimension / other.dimension, scale=self.scale / other.scale, system=self.system)

    def __pow__(self, power: float) -> "Unit":
        if self.is_affine:
            raise STEMUnitError("Affine units cannot be exponentiated")
        p = float(power)
        return Unit(f"{self.symbol}^{p:g}", dimension=self.dimension ** p, scale=self.scale ** p, system=self.system)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unit) and (
            self.symbol, self.dimension, self.scale, self.offset, self.system
        ) == (other.symbol, other.dimension, other.scale, other.offset, other.system)

    def __hash__(self) -> int:
        return hash((self.symbol, self.dimension, self.scale, self.offset, self.system))

    def to_dict(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "name": self.name, "dimension": self.dimension.to_dict(), "scale": self.scale, "offset": self.offset, "system": self.system}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Unit":
        return cls(str(data["symbol"]), data.get("name"), Dimension.from_dict(data.get("dimension", {})), float(data.get("scale", 1.0)), float(data.get("offset", 0.0)), str(data.get("system", "SI")))

    def __str__(self) -> str:
        return self.symbol


@dataclass(frozen=True)
class PrecisionPolicy:
    significant_digits: Optional[int] = None
    decimal_places: Optional[int] = None
    rounding_mode: str = "ROUND_HALF_EVEN"
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-9
    nan_policy: str = "never"

    def __post_init__(self) -> None:
        if self.significant_digits is not None and self.significant_digits < 1:
            raise STEMValidationError("significant_digits must be >= 1")
        if self.decimal_places is not None and self.decimal_places < 0:
            raise STEMValidationError("decimal_places must be >= 0")
        ensure_non_negative(self.absolute_tolerance, "absolute_tolerance")
        ensure_non_negative(self.relative_tolerance, "relative_tolerance")
        if self.nan_policy not in {"never", "equal", "always"}:
            raise STEMValidationError("Invalid nan_policy")

    @classmethod
    def default(cls) -> "PrecisionPolicy":
        cfg = dict(_type_config().get("precision", {}))
        return cls(
            significant_digits=cfg.get("significant_digits"),
            decimal_places=cfg.get("decimal_places"),
            rounding_mode=str(cfg.get("rounding_mode", "ROUND_HALF_EVEN")),
            absolute_tolerance=float(cfg.get("absolute_tolerance", 1e-12)),
            relative_tolerance=float(cfg.get("relative_tolerance", 1e-9)),
            nan_policy=str(cfg.get("nan_policy", "never")),
        )

    def isclose(self, a: float, b: float) -> bool:
        return helper_isclose(a, b, rel_tol=self.relative_tolerance, abs_tol=self.absolute_tolerance, nan_policy=self.nan_policy)

    def round(self, value: float) -> float:
        v = ensure_finite_number(value, "value")
        modes = {
            "ROUND_HALF_EVEN": ROUND_HALF_EVEN, "ROUND_HALF_UP": ROUND_HALF_UP,
            "ROUND_HALF_DOWN": ROUND_HALF_DOWN, "ROUND_UP": ROUND_UP,
            "ROUND_DOWN": ROUND_DOWN, "ROUND_CEILING": ROUND_CEILING, "ROUND_FLOOR": ROUND_FLOOR,
        }
        if self.rounding_mode not in modes:
            raise STEMValidationError("Unsupported rounding_mode", context={"rounding_mode": self.rounding_mode})
        if self.decimal_places is not None:
            quantum = Decimal(1).scaleb(-self.decimal_places)
            return float(Decimal(str(v)).quantize(quantum, rounding=modes[self.rounding_mode]))
        if self.significant_digits is not None and v != 0.0:
            exponent = math.floor(math.log10(abs(v))) - self.significant_digits + 1
            quantum = Decimal(1).scaleb(exponent)
            return float(Decimal(str(v)).quantize(quantum, rounding=modes[self.rounding_mode]))
        return v


@dataclass(frozen=True)
class Tolerance:
    lower: Optional[float] = None
    upper: Optional[float] = None
    absolute: Optional[float] = None
    relative: Optional[float] = None
    inclusive_min: bool = True
    inclusive_max: bool = True
    policy: Optional[PrecisionPolicy] = None

    def __post_init__(self) -> None:
        if self.lower is not None:
            ensure_finite_number(self.lower, "lower")
        if self.upper is not None:
            ensure_finite_number(self.upper, "upper")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise STEMValidationError("lower must be <= upper")
        if self.absolute is not None:
            ensure_non_negative(self.absolute, "absolute")
        if self.relative is not None:
            ensure_non_negative(self.relative, "relative")

    def contains(self, value: float, nominal: Optional[float] = None) -> bool:
        v = ensure_finite_number(value, "value")
        if self.lower is not None and (v < self.lower if self.inclusive_min else v <= self.lower):
            return False
        if self.upper is not None and (v > self.upper if self.inclusive_max else v >= self.upper):
            return False
        if nominal is not None:
            n = ensure_finite_number(nominal, "nominal")
            if self.absolute is not None and abs(v - n) > self.absolute:
                return False
            if self.relative is not None and abs(v - n) > self.relative * max(abs(n), 1.0):
                return False
        return True


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
    value: float
    coverage_factor: float = 1.0
    distribution: Distribution = Distribution.NORMAL
    uncertainty_type: UncertaintyType = UncertaintyType.TYPE_B
    degrees_freedom: Optional[float] = None
    unit: Optional[Unit] = None

    def __post_init__(self) -> None:
        ensure_non_negative(self.value, "uncertainty.value", error_cls=STEMUncertaintyError)
        ensure_positive(self.coverage_factor, "uncertainty.coverage_factor", error_cls=STEMUncertaintyError)
        if not isinstance(self.distribution, Distribution):
            object.__setattr__(self, "distribution", Distribution(self.distribution))
        if not isinstance(self.uncertainty_type, UncertaintyType):
            object.__setattr__(self, "uncertainty_type", UncertaintyType(self.uncertainty_type))
        if self.degrees_freedom is not None:
            ensure_positive(self.degrees_freedom, "uncertainty.degrees_freedom", error_cls=STEMUncertaintyError)
        if self.unit is not None and not isinstance(self.unit, Unit):
            raise STEMUncertaintyError("uncertainty.unit must be a Unit")

    def standard(self) -> float:
        return float(self.value)

    def expanded(self) -> float:
        return float(self.value) * float(self.coverage_factor)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "coverage_factor": self.coverage_factor, "distribution": self.distribution.value, "uncertainty_type": self.uncertainty_type.value, "degrees_freedom": self.degrees_freedom, "unit": self.unit.to_dict() if self.unit else None}


@dataclass
class UncertaintyBudget:
    components: Sequence[Uncertainty]
    correlations: Optional[Mapping[Tuple[int, int], float]] = None
    coverage_factor: float = 2.0
    combined: float = field(init=False)
    expanded: float = field(init=False)
    effective_degrees_freedom: Optional[float] = field(init=False)

    def __post_init__(self) -> None:
        if not self.components or not all(isinstance(item, Uncertainty) for item in self.components):
            raise STEMUncertaintyError("components must contain at least one Uncertainty")
        values = [item.standard() for item in self.components]
        dofs = [item.degrees_freedom for item in self.components]
        self.combined, self.effective_degrees_freedom = combine_standard_uncertainties(values, self.correlations, dofs)
        self.expanded = self.combined * ensure_positive(self.coverage_factor, "coverage_factor", error_cls=STEMUncertaintyError)


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
    value: Any
    unit: Optional[Unit] = None
    uncertainty: Optional[Uncertainty] = None
    method: Optional[str] = None
    precision: Optional[PrecisionPolicy] = None
    absolute_error: Optional[float] = None
    relative_error: Optional[float] = None
    residual: Optional[float] = None
    condition_estimate: Optional[float] = None
    warnings: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.unit is not None and not isinstance(self.unit, Unit):
            raise STEMValidationError("unit must be a Unit")
        if self.uncertainty is not None and not isinstance(self.uncertainty, Uncertainty):
            raise STEMValidationError("uncertainty must be an Uncertainty")
        for name in ("absolute_error", "relative_error", "residual", "condition_estimate"):
            candidate = getattr(self, name)
            if candidate is not None:
                ensure_non_negative(candidate, name, error_cls=STEMValidationError)
        ensure_mapping(self.metadata, "metadata", error_cls=STEMValidationError)

    def ok(self) -> bool:
        return self.value is not None

    def failed(self) -> bool:
        return not self.ok()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": json_safe(self.value), "unit": self.unit.to_dict() if self.unit else None,
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "method": self.method, "absolute_error": self.absolute_error, "relative_error": self.relative_error,
            "residual": self.residual, "condition_estimate": self.condition_estimate,
            "warnings": list(self.warnings), "metadata": json_safe(self.metadata),
        }


@dataclass
class SolverResult:
    solution: Any
    residual: Optional[float]
    iterations: int
    converged: bool
    status: ConvergenceStatus
    message: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    runtime: Optional[float] = None

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise STEMSolverError("iterations must be non-negative")
        if not isinstance(self.status, ConvergenceStatus):
            self.status = ConvergenceStatus(self.status)
        if self.converged != (self.status == ConvergenceStatus.CONVERGED):
            raise STEMSolverError("converged flag and status disagree")
        if self.residual is not None:
            ensure_finite_number(self.residual, "residual", error_cls=STEMSolverError)
        ensure_mapping(self.diagnostics, "diagnostics", error_cls=STEMSolverError)

    def to_dict(self) -> Dict[str, Any]:
        return {"solution": json_safe(self.solution), "residual": self.residual, "iterations": self.iterations, "converged": self.converged, "status": self.status.value, "message": self.message, "diagnostics": json_safe(self.diagnostics), "runtime": self.runtime}


class BoundaryConditionKind(str, Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"
    INITIAL = "initial"


@dataclass(frozen=True)
class BoundaryCondition:
    kind: BoundaryConditionKind
    location: Any
    value: Optional[float] = None
    function: Optional[Callable[..., Any]] = None
    unit: Optional[Unit] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryConditionKind):
            object.__setattr__(self, "kind", BoundaryConditionKind(self.kind))
        if self.value is None and self.function is None:
            raise STEMValidationError("BoundaryCondition requires value or function")
        if self.function is not None and not callable(self.function):
            raise STEMValidationError("function must be callable")


@dataclass(frozen=True)
class InitialCondition:
    time: float = 0.0
    state: Any = None
    value: Optional[float] = None
    unit: Optional[Unit] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_finite_number(self.time, "time")
        if self.state is None and self.value is None:
            raise STEMValidationError("InitialCondition requires state or value")


@dataclass(frozen=True)
class Domain:
    bounds: Sequence[Tuple[float, float]]
    dimension: Dimension = field(default_factory=Dimension)
    coordinate_system: str = "cartesian"
    mesh: Optional[Any] = None

    def __post_init__(self) -> None:
        normalized = []
        for i, bound in enumerate(self.bounds):
            if len(bound) != 2:
                raise STEMValidationError(f"bounds[{i}] must be a pair")
            lo, hi = ensure_finite_number(bound[0], "lower"), ensure_finite_number(bound[1], "upper")
            if lo > hi:
                raise STEMValidationError("domain lower bound must be <= upper bound")
            normalized.append((lo, hi))
        object.__setattr__(self, "bounds", tuple(normalized))

    def contains(self, point: Sequence[float]) -> bool:
        if len(point) != len(self.bounds):
            raise STEMValidationError("point dimensionality does not match domain")
        return all(lo <= float(v) <= hi for v, (lo, hi) in zip(point, self.bounds))

    def volume(self) -> float:
        return math.prod(hi - lo for lo, hi in self.bounds)


_ALLOWED_EXPR_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Call,
)
_ALLOWED_EXPR_FUNCS = {name: getattr(math, name) for name in ("sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh", "fabs")}


@dataclass
class Equation:
    name: str
    expression: Optional[str] = None
    callable: Optional[Callable[..., Any]] = None
    variables: Sequence[str] = field(default_factory=tuple)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    residual: Optional[Callable[..., Any]] = None
    domain: Optional[Domain] = None

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.name, "name", error_cls=STEMEquationError)
        if (self.expression is None) == (self.callable is None):
            raise STEMEquationError("Exactly one of expression or callable must be supplied")
        if self.expression is not None:
            tree = ast.parse(self.expression, mode="eval")
            for node in ast.walk(tree):
                if not isinstance(node, _ALLOWED_EXPR_NODES):
                    raise STEMEquationError("Unsupported expression syntax", context={"node": type(node).__name__})
                if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_EXPR_FUNCS):
                    raise STEMEquationError("Unsupported expression function")

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        if self.callable is not None:
            return self.callable(*args, **kwargs)
        env = dict(_ALLOWED_EXPR_FUNCS)
        env.update(self.parameters)
        env.update(kwargs)
        for name, value in zip(self.variables, args):
            env[name] = value
        try:
            return eval(compile(ast.parse(self.expression or "", mode="eval"), "<stem-equation>", "eval"), {"__builtins__": {}}, env)
        except Exception as exc:
            raise STEMEquationError("Failed to evaluate equation", context={"equation": self.name}, cause=exc) from exc

    def residual_for(self, *args: Any, **kwargs: Any) -> Any:
        return self.residual(*args, **kwargs) if self.residual is not None else self.evaluate(*args, **kwargs)


@dataclass(frozen=True)
class PhysicalConstant:
    name: str
    symbol: str
    value: float
    unit: Unit
    uncertainty: Optional[Uncertainty] = None
    exact: bool = False
    source: Optional[str] = None

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.name, "name", error_cls=STEMPhysicalConstantError)
        ensure_non_empty_string(self.symbol, "symbol", error_cls=STEMPhysicalConstantError)
        ensure_finite_number(self.value, "value", error_cls=STEMPhysicalConstantError)
        if not isinstance(self.unit, Unit):
            raise STEMPhysicalConstantError("unit must be a Unit")
        if self.exact and self.uncertainty is not None:
            raise STEMPhysicalConstantError("Exact constants cannot have uncertainty")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "symbol": self.symbol, "value": self.value, "unit": self.unit.to_dict(), "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None, "exact": self.exact, "source": self.source}


@dataclass(frozen=True, eq=False)
class Quantity:
    magnitude: float
    unit: Unit
    uncertainty: Optional[Uncertainty] = None
    precision: Optional[PrecisionPolicy] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "magnitude", ensure_finite_number(self.magnitude, "magnitude"))
        if not isinstance(self.unit, Unit):
            raise STEMValidationError("unit must be a Unit")
        if self.uncertainty is not None and self.uncertainty.unit is not None and self.uncertainty.unit.dimension != self.unit.dimension:
            raise STEMValidationError("uncertainty unit dimension must match quantity")

    def convert(self, target_unit: Unit) -> "Quantity":
        magnitude = self.unit.convert(self.magnitude, target_unit)
        uncertainty = self.uncertainty
        if uncertainty is not None:
            # uncertainty is a difference, so offsets never apply
            converted_u = abs(uncertainty.value * self.unit.scale / target_unit.scale)
            uncertainty = Uncertainty(converted_u, uncertainty.coverage_factor, uncertainty.distribution, uncertainty.uncertainty_type, uncertainty.degrees_freedom, target_unit)
        return Quantity(magnitude, target_unit, uncertainty, self.precision)

    def __add__(self, other: "Quantity") -> "Quantity":
        if not isinstance(other, Quantity):
            return NotImplemented
        rhs = other.convert(self.unit)
        u = self._rss(rhs)
        return Quantity(self.magnitude + rhs.magnitude, self.unit, u, self.precision or rhs.precision)

    def __sub__(self, other: "Quantity") -> "Quantity":
        if not isinstance(other, Quantity):
            return NotImplemented
        rhs = other.convert(self.unit)
        return Quantity(self.magnitude - rhs.magnitude, self.unit, self._rss(rhs), self.precision or rhs.precision)

    def __mul__(self, other: Union["Quantity", int, float]) -> "Quantity":
        if isinstance(other, Quantity):
            magnitude = self.magnitude * other.magnitude
            return Quantity(magnitude, self.unit * other.unit, self._relative_uncertainty(other, magnitude, self.unit * other.unit), self.precision or other.precision)
        return Quantity(self.magnitude * float(other), self.unit, self.uncertainty, self.precision)

    def __truediv__(self, other: Union["Quantity", int, float]) -> "Quantity":
        if isinstance(other, Quantity):
            if other.magnitude == 0.0:
                raise STEMValidationError("Division by zero quantity")
            magnitude = self.magnitude / other.magnitude
            return Quantity(magnitude, self.unit / other.unit, self._relative_uncertainty(other, magnitude, self.unit / other.unit), self.precision or other.precision)
        if float(other) == 0.0:
            raise STEMValidationError("Division by zero")
        return Quantity(self.magnitude / float(other), self.unit, self.uncertainty, self.precision)

    def _rss(self, other: "Quantity") -> Optional[Uncertainty]:
        if self.uncertainty is None and other.uncertainty is None:
            return None
        combined = math.hypot(self.uncertainty.value if self.uncertainty else 0.0, other.uncertainty.value if other.uncertainty else 0.0)
        return Uncertainty(combined, unit=self.unit)

    def _relative_uncertainty(self, other: "Quantity", result_magnitude: float, result_unit: Unit) -> Optional[Uncertainty]:
        if self.uncertainty is None and other.uncertainty is None:
            return None
        rel_a = 0.0 if self.uncertainty is None or self.magnitude == 0 else self.uncertainty.value / abs(self.magnitude)
        rel_b = 0.0 if other.uncertainty is None or other.magnitude == 0 else other.uncertainty.value / abs(other.magnitude)
        return Uncertainty(abs(result_magnitude) * math.hypot(rel_a, rel_b), unit=result_unit)

    def isclose(self, other: "Quantity", policy: Optional[PrecisionPolicy] = None) -> bool:
        rhs = other.convert(self.unit)
        return (policy or self.precision or rhs.precision or PrecisionPolicy.default()).isclose(self.magnitude, rhs.magnitude)

    def to_dict(self) -> Dict[str, Any]:
        return {"magnitude": self.magnitude, "unit": self.unit.to_dict(), "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None}

    def __str__(self) -> str:
        return f"{self.magnitude:g} {self.unit.symbol}" if self.uncertainty is None else f"{self.magnitude:g} ± {self.uncertainty.expanded():g} {self.unit.symbol}"


__all__ = [
    "Dimension", "Unit", "PrecisionPolicy", "Tolerance", "UncertaintyType", "Distribution", "Uncertainty",
    "UncertaintyBudget", "ConvergenceStatus", "NumericResult", "SolverResult", "BoundaryConditionKind",
    "BoundaryCondition", "InitialCondition", "Domain", "Equation", "PhysicalConstant", "Quantity",
]
