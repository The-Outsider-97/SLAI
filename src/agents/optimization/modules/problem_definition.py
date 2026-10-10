"""
Problem definition for the Optimization agent.

Turns a declarative description of an optimisation problem (plain data from config, a tool call or an LLM)
into a validated, immutable ``OptimizationProblem``, and provides the runtime services around it: checking
assignments, evaluating candidates, sampling starting points and encoding to/from algorithm-friendly vectors.

A problem consists of
- Variables: continuous, integer, binary or categorical decision variables with bounds/choices/step.
- Objectives: ``pareto_efficiency.Objective`` (name, direction, weight, bounds, tolerance), reused as-is so a
  problem plugs straight into ``ParetoEfficiency``.
- Constraints: relations such as ``"x + 2*y <= 10"``, or an expression/sense/bound triple, or a Python callable.
  Constraints that only touch variables are *variable constraints* (cheap, checked before evaluation);
  constraints that touch objectives or metrics are *output constraints* (checked after evaluation).
- Metrics: extra outputs the evaluator reports that constraints may use (e.g. ``latency``).
- Formulas: optional analytic expressions for objectives/metrics, so simple problems need no evaluator.
- Limits / reference point: optional decision preferences forwarded to ``ParetoEfficiency``.

Example
    problem_definition = ProblemDefinition()
    problem = problem_definition.define({
        "name": "widget",
        "variables": {"width": [1, 10], "units": {"type": "integer", "lower": 1, "upper": 50},
                      "material": ["steel", "alloy"]},
        "objectives": {"cost": {"direction": "min", "formula": "2*width + 0.5*units"},
                       "strength": "max"},
        "metrics": ["weight"],
        "constraints": ["width * units <= 200", {"name": "light", "expression": "weight <= 40"}],
    })
    batch = problem_definition.evaluate_many(problem, problem_definition.sample(problem, 20), my_evaluator)
    front = ParetoEfficiency(config=problem.to_pareto_config()).analyze(batch.candidates)

Conventions
- Names (variables, objectives, metrics) must be valid identifiers, unique across all three groups, and must not
  collide with expression functions/constants (``sqrt``, ``pi`` ...), because constraints and formulas refer to them.
- Strict relations ``<``/``>`` are treated as ``<=``/``>=``; optimisation cannot distinguish them.
- Expressions may only reference numeric variables; use a callable constraint to involve categorical ones.
- Violations are summed into ``Candidate.violation`` (0 = feasible), which Pareto uses for constrained dominance.
- Errors come from ``optimization_errors`` and numeric/validation primitives from ``optimization_helpers``.
"""
from __future__ import annotations

__version__ = "2.3.0"

import keyword
import math
import random
import re
import threading
import uuid

from collections import Counter
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from .pareto_efficiency import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Problem Definition")
printer = PrettyPrinter()

CONTINUOUS = "continuous"
INTEGER = "integer"
BINARY = "binary"
CATEGORICAL = "categorical"

DOMAIN_POLICIES: Tuple[str, ...] = ("reject", "clip")
SAMPLING_METHODS: Tuple[str, ...] = ("random", "latin_hypercube")
MAX_EXPRESSION_LENGTH = 500

_KIND_ALIASES: Dict[str, str] = {
    "continuous": CONTINUOUS, "float": CONTINUOUS, "real": CONTINUOUS,
    "integer": INTEGER, "int": INTEGER,
    "binary": BINARY, "bool": BINARY, "boolean": BINARY,
    "categorical": CATEGORICAL, "category": CATEGORICAL, "choice": CATEGORICAL, "enum": CATEGORICAL,
}
_SENSES: Dict[str, str] = {"<=": "<=", "<": "<=", ">=": ">=", ">": ">=", "==": "==", "=": "=="}
_RELATION = re.compile(r"(<=|>=|==|<|>)")
_RESERVED_NAMES = frozenset(EXPRESSION_FUNCTIONS) | frozenset(EXPRESSION_CONSTANTS)
_SPEC_KEYS = frozenset({"name", "description", "variables", "objectives", "constraints", "metrics", "formulas",
                        "limits", "reference_point", "metadata"})
_GRID_SLACK = 1e-9

Evaluator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _check_name(label: str, name: Any) -> str:
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name in _RESERVED_NAMES:
        raise OptimizationValidationError(
            f"{label} name must be an identifier that is not a Python keyword or expression function/constant",
            context={"name": name, "reserved": sorted(_RESERVED_NAMES)})
    return name


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variable:
    """A decision variable.

    ``step`` defines a grid ``lower + k*step`` (integers default to 1, binaries are integers on [0, 1]).
    ``initial`` defaults to the first choice, or the (snapped) midpoint for numeric kinds.
    """

    name: str
    kind: str = CONTINUOUS
    lower: Optional[float] = None
    upper: Optional[float] = None
    choices: Tuple[Any, ...] = ()
    initial: Any = None
    step: Optional[float] = None

    def __post_init__(self) -> None:
        name = _check_name("Variable", self.name)
        kind = _KIND_ALIASES.get(str(self.kind).strip().lower())
        if kind is None:
            raise OptimizationValidationError(f"Variable '{name}' has unknown kind {self.kind!r}", context={"allowed": sorted(set(_KIND_ALIASES.values()))})
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)

        if kind == CATEGORICAL:
            if any(x is not None for x in (self.lower, self.upper, self.step)):
                raise OptimizationValidationError(f"Categorical variable '{name}' cannot have lower/upper/step")
            if not isinstance(self.choices, (list, tuple)) or not self.choices:
                raise OptimizationValidationError(f"Categorical variable '{name}' needs a non-empty list of choices", context={"choices": self.choices})
            choices = tuple(self.choices)
            if not all(isinstance(c, (str, int, float)) and (not isinstance(c, float) or math.isfinite(c)) for c in choices) \
                    or len(set(choices)) != len(choices):
                raise OptimizationValidationError(f"Choices of '{name}' must be unique strings, numbers or booleans", context={"choices": choices})
            object.__setattr__(self, "choices", choices)
            initial = choices[0] if self.initial is None else self.coerce(self.initial)
        else:
            if self.choices:
                raise OptimizationValidationError(f"Variable '{name}' is {kind}; only categorical variables take choices")
            lower, upper = (0.0, 1.0) if kind == BINARY else (to_finite_float(self.lower), to_finite_float(self.upper))
            if kind == BINARY and any(x is not None and to_finite_float(x) != b for x, b in ((self.lower, 0.0), (self.upper, 1.0))):
                raise OptimizationValidationError(f"Binary variable '{name}' is fixed to [0, 1]")
            if lower is None or upper is None or lower >= upper:
                raise OptimizationValidationError(f"Variable '{name}' needs finite bounds with lower < upper", context={"lower": self.lower, "upper": self.upper})
            step = to_finite_float(self.step) if self.step is not None else (1.0 if kind in (INTEGER, BINARY) else None)
            invalid_integer_step = kind == INTEGER and (
                step is None or not float(lower).is_integer() or not float(upper).is_integer() or not float(step).is_integer()
            )
            if (self.step is not None and (step is None or step <= 0)) or invalid_integer_step:
                raise OptimizationValidationError(f"Variable '{name}' has an invalid step (or non-integral integer bounds)",
                                                  context={"lower": lower, "upper": upper, "step": self.step})
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
            object.__setattr__(self, "step", step)
            initial = self.coerce((lower + upper) / 2.0, clip=True) if self.initial is None else self.coerce(self.initial)
        object.__setattr__(self, "initial", initial)

    @property
    def numeric(self) -> bool:
        return self.kind != CATEGORICAL

    @property
    def levels(self) -> Optional[int]:
        """Number of distinct values, or None for an ungridded continuous variable."""
        if self.kind == CATEGORICAL:
            return len(self.choices)
        if self.step is None:
            return None
        return self._top_index() + 1

    def _top_index(self) -> int:
        return math.floor((self.upper - self.lower) / self.step + _GRID_SLACK)  # type: ignore[operator]

    def coerce(self, value: Any, *, clip: bool = False) -> Any:
        """Return ``value`` as a canonical member of the domain (int/float/choice).

        With ``clip`` out-of-range numbers are clamped and off-grid numbers snapped; otherwise they raise.
        Categorical values must always be one of the choices.
        """
        if self.kind == CATEGORICAL:
            if value in self.choices:
                return self.choices[self.choices.index(value)]
            raise OptimizationValidationError(f"'{value}' is not a choice of variable '{self.name}'",
                                              context={"variable": self.name, "choices": self.choices})
        if isinstance(value, bool) and self.kind in (INTEGER, BINARY):
            value = int(value)
        number = to_finite_float(value)
        if number is None:
            raise OptimizationValidationError(f"Variable '{self.name}' needs a finite number, got {value!r}", context={"variable": self.name})
        slack = _GRID_SLACK * max(1.0, abs(self.lower), abs(self.upper))  # type: ignore[arg-type]
        if not clip and not (self.lower - slack <= number <= self.upper + slack):  # type: ignore[operator]
            raise OptimizationValidationError(f"Variable '{self.name}'={number} is outside [{self.lower}, {self.upper}]", context={"variable": self.name})
        number = min(max(number, self.lower), self.upper)  # type: ignore[type-var]
        if self.step is not None:
            index = max(0, min(round((number - self.lower) / self.step), self._top_index()))  # type: ignore[operator]
            snapped = self.lower + index * self.step  # type: ignore[operator]
            if not clip and abs(snapped - number) > _GRID_SLACK * max(1.0, self.step):
                raise OptimizationValidationError(f"Variable '{self.name}'={number} is not on its grid (step {self.step})", context={"variable": self.name})
            number = snapped
        return int(round(number)) if self.kind in (INTEGER, BINARY) else number

    def to_unit(self, value: Any) -> float:
        """Map a value to [0, 1]; gridded and categorical domains map to the centre of their bin."""
        value = self.coerce(value, clip=True) if self.numeric else self.coerce(value)
        if self.kind == CATEGORICAL:
            return (self.choices.index(value) + 0.5) / len(self.choices)
        if self.step is not None:
            return (round((value - self.lower) / self.step) + 0.5) / self.levels  # type: ignore[operator]
        return (value - self.lower) / (self.upper - self.lower)  # type: ignore[operator]

    def from_unit(self, u: float) -> Any:
        """Inverse of ``to_unit``; ``u`` is clamped to [0, 1] and every bin is equally likely."""
        u = min(max(float(u), 0.0), 1.0)
        levels = self.levels
        if levels is not None:
            index = min(int(u * levels), levels - 1)
            return self.choices[index] if self.kind == CATEGORICAL else self.coerce(self.lower + index * self.step, clip=True)  # type: ignore[operator]
        return self.lower + u * (self.upper - self.lower)  # type: ignore[operator]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "lower": self.lower, "upper": self.upper,
                "choices": list(self.choices), "initial": self.initial, "step": self.step}


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Constraint:
    """A constraint ``g(values) <sense> bound`` or a relation string.

    Forms (exactly one of ``expression`` / ``function``):
    - ``Constraint(expression="x + y <= 10")``: a relation; ``sense`` and ``bound`` are derived.
    - ``Constraint(expression="x + y", sense="<=", bound=10)``.
    - ``Constraint(function=fn, sense="<=", bound=10, uses=("x", "cost"))``: ``fn(values) -> float``; ``uses``
      names what it reads so the problem can tell variable constraints from output constraints.

    ``scale`` divides the violation so differently-sized constraints are comparable when summed.
    """

    name: str = ""
    expression: Optional[str] = None
    sense: Optional[str] = None
    bound: float = 0.0
    function: Optional[Callable[[Mapping[str, Any]], float]] = None
    uses: Tuple[str, ...] = ()
    tolerance: Optional[float] = None
    scale: float = 1.0
    references: FrozenSet[str] = field(init=False, default=frozenset())
    _evaluate: Optional[Callable[[Mapping[str, Any]], float]] = field(init=False, default=None, repr=False)
    _sense: str = field(init=False, default="<=", repr=False)
    _bound: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if (self.expression is None) == (self.function is None):
            raise OptimizationValidationError("Constraint needs exactly one of `expression` or `function`", context={"name": self.name})
        scale, tolerance = to_finite_float(self.scale), (None if self.tolerance is None else to_finite_float(self.tolerance))
        if scale is None or scale <= 0 or (self.tolerance is not None and (tolerance is None or tolerance < 0)):
            raise OptimizationValidationError("Constraint scale must be > 0 and tolerance >= 0", context={"scale": self.scale, "tolerance": self.tolerance})
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "tolerance", tolerance)

        if self.expression is not None:
            text = self.expression.strip() if isinstance(self.expression, str) else ""
            if self.sense is None:
                parts = _RELATION.split(text)
                if len(parts) != 3 or not parts[0].strip() or not parts[2].strip():
                    raise OptimizationValidationError("A relation needs exactly one of <=, >=, ==, <, >", context={"expression": self.expression})
                sense, bound, source = _SENSES[parts[1]], 0.0, f"({parts[0]}) - ({parts[2]})"
            else:
                sense, bound, source = self._normalise_sense(), self._normalise_bound(), text
            evaluate, references = compile_expression(source, max_length=MAX_EXPRESSION_LENGTH)
            label = text
        else:
            if not callable(self.function):
                raise OptimizationValidationError("Constraint `function` must be callable", context={"name": self.name})
            if self.sense is None:
                raise OptimizationValidationError("A callable constraint needs a `sense`", context={"name": self.name})
            sense, bound, evaluate = self._normalise_sense(), self._normalise_bound(), self.function
            references = frozenset(self.uses)
            label = getattr(self.function, "__name__", "callable")
        object.__setattr__(self, "_sense", sense)
        object.__setattr__(self, "_bound", bound)
        object.__setattr__(self, "_evaluate", evaluate)
        object.__setattr__(self, "references", frozenset(references))
        object.__setattr__(self, "name", self.name.strip() if self.name and self.name.strip() else f"constraint_{stable_fingerprint(label)[:8]}")

    def _normalise_sense(self) -> str:
        sense = _SENSES.get(str(self.sense).strip())
        if sense is None:
            raise OptimizationValidationError(f"Unknown constraint sense {self.sense!r}", context={"allowed": sorted(_SENSES)})
        return sense

    def _normalise_bound(self) -> float:
        bound = to_finite_float(self.bound)
        if bound is None:
            raise OptimizationValidationError("Constraint bound must be a finite number", context={"bound": self.bound})
        return bound

    def violation(self, values: Mapping[str, Any], default_tolerance: float) -> float:
        """Non-negative, scaled violation (0 = satisfied within tolerance)."""
        try:
            raw = to_finite_float(self._evaluate(values))  # type: ignore[misc]
        except OptimizationEvaluationError:
            raise
        except Exception as exc:  # expression arithmetic errors, or anything a user callable raises
            raise OptimizationEvaluationError(f"Constraint '{self.name}' could not be evaluated: {exc}", context={"constraint": self.name}) from exc
        if raw is None:
            raise OptimizationEvaluationError(f"Constraint '{self.name}' produced a non-finite value", context={"constraint": self.name})
        gap = raw - self._bound if self._sense == "<=" else self._bound - raw if self._sense == ">=" else abs(raw - self._bound)
        tolerance = self.tolerance if self.tolerance is not None else default_tolerance
        return max(0.0, gap - tolerance) / self.scale

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"name": self.name, "tolerance": self.tolerance, "scale": self.scale}
        if self.expression is not None:
            data["expression"] = self.expression
            if self.sense is not None:
                data.update(sense=self._sense, bound=self._bound)
        else:  # callables are described, not serialised; from_dict rejects them
            data.update(function=getattr(self.function, "__name__", "callable"), sense=self._sense, bound=self._bound, uses=list(self.uses))
        return data


# ---------------------------------------------------------------------------
# Spec coercion (pure functions)
# ---------------------------------------------------------------------------
def _numeric(x: Any) -> bool:
    return to_finite_float(x) is not None


def _build_variable(name: str, spec: Any) -> Variable:
    if isinstance(spec, Variable):
        return spec
    if isinstance(spec, str):
        return Variable(name=name, kind=spec)
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(_numeric(x) for x in spec):
            return Variable(name=name, kind=CONTINUOUS, lower=spec[0], upper=spec[1])
        if spec and all(isinstance(x, str) for x in spec):
            return Variable(name=name, kind=CATEGORICAL, choices=tuple(spec))
        raise OptimizationValidationError(f"Variable '{name}': a list must be [lower, upper] or a list of strings; use {{'choices': [...]}} otherwise", context={"spec": spec})
    if isinstance(spec, Mapping):
        data = dict(spec)
        data.pop("name", None)
        if "type" in data:
            data["kind"] = data.pop("type")
        if "bounds" in data:
            bounds = data.pop("bounds")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise OptimizationValidationError(f"Variable '{name}': bounds must be [lower, upper]", context={"bounds": bounds})
            data["lower"], data["upper"] = bounds
        if "choices" in data:
            data["choices"] = tuple(data["choices"]) if isinstance(data["choices"], (list, tuple)) else data["choices"]
            data.setdefault("kind", CATEGORICAL)
        try:
            return Variable(name=name, **data)
        except TypeError as exc:
            raise OptimizationValidationError(f"Invalid variable specification for '{name}': {exc}", context={"allowed": [f.name for f in fields(Variable)]}) from exc
    raise OptimizationValidationError(f"Variable '{name}' has an unsupported specification", context={"type": type(spec).__name__})


def _coerce_variables(raw: Any) -> Tuple[Variable, ...]:
    """``{name: spec}`` or ``[Variable | {name, ...}]``; spec = kind string, [lo, hi], [str, ...] or a mapping."""
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return tuple(_build_variable(str(name), spec) for name, spec in raw.items())
    if isinstance(raw, (list, tuple)):
        items = []
        for spec in raw:
            if isinstance(spec, Variable):
                items.append(spec)
            elif isinstance(spec, Mapping) and "name" in spec:
                items.append(_build_variable(str(spec["name"]), spec))
            else:
                raise OptimizationValidationError("Each variable must be a Variable or a mapping with a `name`", context={"spec": spec})
        return tuple(items)
    raise OptimizationValidationError("variables must be a mapping or a list", context={"type": type(raw).__name__})


def _build_constraint(name: str, spec: Any) -> Constraint:
    if isinstance(spec, Constraint):
        return spec
    if isinstance(spec, str):
        return Constraint(name=name, expression=spec)
    if isinstance(spec, Mapping):
        data = dict(spec)
        if "function" in data:
            raise OptimizationValidationError("Callable constraints cannot be restored from data; construct Constraint(function=...) directly", context={"constraint": name})
        data.setdefault("name", name)
        try:
            return Constraint(**data)
        except TypeError as exc:
            raise OptimizationValidationError(f"Invalid constraint specification: {exc}", context={"spec": data}) from exc
    raise OptimizationValidationError("A constraint must be a relation string, a mapping or a Constraint", context={"type": type(spec).__name__})


def _coerce_constraints(raw: Any) -> Tuple[Constraint, ...]:
    """``["x + y <= 10", {...}, Constraint]`` or ``{name: relation | {...}}``."""
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return tuple(_build_constraint(str(name), spec) for name, spec in raw.items())
    if isinstance(raw, (list, tuple)):
        return tuple(_build_constraint("", spec) for spec in raw)
    raise OptimizationValidationError("constraints must be a list or a mapping", context={"type": type(raw).__name__})


def _split_formulas(raw: Any) -> Tuple[Any, Dict[str, str]]:
    """Pull ``formula`` entries out of objective specs so ``coerce_objectives`` sees plain Objective specs."""
    formulas: Dict[str, str] = {}
    if isinstance(raw, Mapping):
        cleaned: Any = {}
        for name, spec in raw.items():
            if isinstance(spec, Mapping) and "formula" in spec:
                spec = dict(spec)
                formulas[str(name)] = spec.pop("formula")
            cleaned[name] = spec
        return cleaned, formulas
    if isinstance(raw, (list, tuple)):
        cleaned = []
        for spec in raw:
            if isinstance(spec, Mapping) and "formula" in spec:
                spec = dict(spec)
                formulas[str(spec.get("name"))] = spec.pop("formula")
            cleaned.append(spec)
        return cleaned, formulas
    return raw, formulas


@dataclass
class _Parts:
    """A parsed (and possibly partial) spec; the unit that templates are merged on."""

    name: str = ""
    description: str = ""
    variables: Dict[str, Variable] = field(default_factory=dict)
    objectives: Dict[str, Objective] = field(default_factory=dict)
    constraints: Dict[str, Constraint] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)
    formulas: Dict[str, str] = field(default_factory=dict)
    limits: Dict[str, float] = field(default_factory=dict)
    reference_point: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _index(items: Sequence[Any], label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if item.name in out:
            raise OptimizationValidationError(f"Duplicate {label} name '{item.name}'", context={"name": item.name})
        out[item.name] = item
    return out


def _parse_spec(spec: Mapping[str, Any]) -> _Parts:
    if not isinstance(spec, Mapping):
        raise OptimizationValidationError("A problem specification must be a mapping", context={"type": type(spec).__name__})
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise OptimizationValidationError("Problem specification contains unsupported keys", context={"unknown": sorted(unknown), "allowed": sorted(_SPEC_KEYS)})
    objectives_raw, formulas = _split_formulas(spec.get("objectives"))
    extra_formulas = spec.get("formulas") or {}
    if not isinstance(extra_formulas, Mapping):
        raise OptimizationValidationError("formulas must be a mapping of output name to expression")
    formulas.update({str(k): v for k, v in extra_formulas.items()})
    if not all(isinstance(v, str) for v in formulas.values()):
        raise OptimizationValidationError("Every formula must be an expression string", context={"formulas": formulas})
    metrics = spec.get("metrics") or []
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, (list, tuple)):
        raise OptimizationValidationError("metrics must be a list of names", context={"metrics": metrics})
    if not isinstance(spec.get("metadata") or {}, Mapping):
        raise OptimizationValidationError("metadata must be a mapping")
    return _Parts(
        name=str(spec.get("name") or ""),
        description=str(spec.get("description") or ""),
        variables=_index(_coerce_variables(spec.get("variables")), "variable"),
        objectives=_index(_coerce_objectives(objectives_raw), "objective"),
        constraints=_index(_coerce_constraints(spec.get("constraints")), "constraint"),
        metrics=list(dict.fromkeys(str(m) for m in metrics)),
        formulas=formulas,
        limits=_coerce_value_map(spec.get("limits"), "limits"),
        reference_point=_coerce_value_map(spec.get("reference_point"), "reference_point"),
        metadata=dict(spec.get("metadata") or {}),
    )


def _merge_parts(base: _Parts, over: _Parts) -> _Parts:
    """``over`` wins. Named items merge by name; redefining an output drops the template's formula for it."""
    return _Parts(
        name=over.name,
        description=over.description or base.description,
        variables={**base.variables, **over.variables},
        objectives={**base.objectives, **over.objectives},
        constraints={**base.constraints, **over.constraints},
        metrics=list(dict.fromkeys([*base.metrics, *over.metrics])),
        formulas={**{k: v for k, v in base.formulas.items() if k not in over.objectives and k not in over.metrics}, **over.formulas},
        limits={**base.limits, **over.limits},
        reference_point={**base.reference_point, **over.reference_point},
        metadata={**base.metadata, **over.metadata},
    )


# ---------------------------------------------------------------------------
# The validated problem
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class OptimizationProblem:
    """Immutable, validated problem. Build it with ``ProblemDefinition.define`` or construct it directly."""

    name: str
    variables: Tuple[Variable, ...]
    objectives: Tuple[Objective, ...]
    constraints: Tuple[Constraint, ...] = ()
    metrics: Tuple[str, ...] = ()
    formulas: Mapping[str, str] = field(default_factory=dict)
    limits: Mapping[str, float] = field(default_factory=dict)
    reference_point: Mapping[str, float] = field(default_factory=dict)
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    variable_constraints: Tuple[Constraint, ...] = field(init=False, default=())
    output_constraints: Tuple[Constraint, ...] = field(init=False, default=())
    formula_functions: Mapping[str, Callable[[Mapping[str, Any]], float]] = field(init=False, default_factory=dict, repr=False)
    fingerprint: str = field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _check_name("Problem", self.name) if self.name.isidentifier() else self._check_label(self.name))
        for attr in ("variables", "objectives", "constraints", "metrics"):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        if not self.variables or not self.objectives:
            raise OptimizationValidationError(f"Problem '{self.name}' needs at least one variable and one objective")

        variable_names = [v.name for v in self.variables]
        objective_names = [o.name for o in self.objectives]
        outputs = objective_names + list(self.metrics)
        all_names = variable_names + outputs
        for name in objective_names + list(self.metrics):
            _check_name("Objective/metric", name)
        if len(set(all_names)) != len(all_names):
            raise OptimizationValidationError("Variable, objective and metric names must be unique across all three",
                                              context={"duplicates": sorted(n for n, c in Counter(all_names).items() if c > 1)})
        numeric = {v.name for v in self.variables if v.numeric}

        formula_functions: Dict[str, Callable[[Mapping[str, Any]], float]] = {}
        for output, text in self.formulas.items():
            if output not in outputs:
                raise OptimizationValidationError(f"Formula given for unknown objective/metric '{output}'", context={"outputs": outputs})
            function, refs = compile_expression(text, max_length=MAX_EXPRESSION_LENGTH)
            if not refs <= numeric:
                raise OptimizationValidationError(f"Formula for '{output}' may only use numeric variables", context={"invalid": sorted(refs - numeric)})
            formula_functions[output] = function

        output_set = set(outputs)
        for constraint in self.constraints:
            if not constraint.references <= numeric | output_set:
                raise OptimizationValidationError(
                    f"Constraint '{constraint.name}' refers to unknown or non-numeric names",
                    context={"invalid": sorted(constraint.references - (numeric | output_set))})
        names = [c.name for c in self.constraints]
        if len(set(names)) != len(names):
            raise OptimizationValidationError("Constraint names must be unique", context={"names": names})

        if not set(self.limits) <= set(objective_names):
            raise OptimizationValidationError("limits may only name objectives", context={"invalid": sorted(set(self.limits) - set(objective_names))})
        if self.reference_point and set(self.reference_point) != set(objective_names):
            raise OptimizationValidationError("reference_point must give a value for every objective", context={"objectives": objective_names})

        object.__setattr__(self, "variable_constraints", tuple(c for c in self.constraints if not c.references & output_set))
        object.__setattr__(self, "output_constraints", tuple(c for c in self.constraints if c.references & output_set))
        object.__setattr__(self, "formula_functions", formula_functions)
        object.__setattr__(self, "fingerprint", stable_fingerprint(self.to_dict()))

    @staticmethod
    def _check_label(name: str) -> str:
        """Problem names are labels (e.g. 'problem-1a2b'), not expression names, so any non-empty string is fine."""
        if not isinstance(name, str) or not name.strip():
            raise OptimizationValidationError("Problem name must be a non-empty string", context={"name": name})
        return name.strip()

    # ---- introspection
    @property
    def variable_names(self) -> Tuple[str, ...]:
        return tuple(v.name for v in self.variables)

    @property
    def objective_names(self) -> Tuple[str, ...]:
        return tuple(o.name for o in self.objectives)

    @property
    def output_names(self) -> Tuple[str, ...]:
        return self.objective_names + self.metrics

    @property
    def is_multi_objective(self) -> bool:
        return len(self.objectives) > 1

    @property
    def is_constrained(self) -> bool:
        return bool(self.constraints)

    @property
    def search_space_size(self) -> Optional[int]:
        """Number of distinct assignments, or None when any variable is continuous without a step."""
        levels = [v.levels for v in self.variables]
        return None if any(n is None for n in levels) else math.prod(levels)  # type: ignore[arg-type]

    def variable(self, name: str) -> Variable:
        for v in self.variables:
            if v.name == name:
                return v
        raise OptimizationValidationError(f"Unknown variable '{name}'", context={"variables": list(self.variable_names)})

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name, "fingerprint": self.fingerprint, "description": self.description,
            "n_variables": len(self.variables), "variable_kinds": dict(Counter(v.kind for v in self.variables)),
            "objectives": {o.name: o.direction for o in self.objectives}, "multi_objective": self.is_multi_objective,
            "metrics": list(self.metrics), "analytic_outputs": sorted(self.formulas),
            "n_constraints": len(self.constraints), "n_variable_constraints": len(self.variable_constraints),
            "n_output_constraints": len(self.output_constraints), "search_space_size": self.search_space_size,
        }

    # ---- vector encoding for optimisation algorithms
    def default_point(self) -> Dict[str, Any]:
        return {v.name: v.initial for v in self.variables}

    def encode(self, assignment: Mapping[str, Any]) -> List[float]:
        """Assignment -> unit-hypercube vector in variable order (values are clipped/snapped into the domain)."""
        missing = [v.name for v in self.variables if v.name not in assignment]
        if missing:
            raise OptimizationValidationError("Assignment is missing variables", context={"missing": missing})
        return [v.to_unit(assignment[v.name]) for v in self.variables]

    def decode(self, vector: Sequence[float]) -> Dict[str, Any]:
        """Unit-hypercube vector -> valid assignment (out-of-range entries are clamped)."""
        if len(vector) != len(self.variables):
            raise OptimizationValidationError("Vector length must equal the number of variables", context={"expected": len(self.variables), "received": len(vector)})
        return {v.name: v.from_unit(u) for v, u in zip(self.variables, vector)}

    # ---- serialisation and Pareto integration
    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name, "description": self.description,
            "variables": [v.to_dict() for v in self.variables],
            "objectives": [o.to_dict() for o in self.objectives],
            "constraints": [c.to_dict() for c in self.constraints],
            "metrics": list(self.metrics), "formulas": dict(self.formulas),
            "limits": dict(self.limits), "reference_point": dict(self.reference_point), "metadata": dict(self.metadata),
        }
        return data

    def to_pareto_config(self) -> Dict[str, Any]:
        """Config for ``ParetoEfficiency``: objectives plus any limits / reference point."""
        config: Dict[str, Any] = {"objectives": [o.to_dict() for o in self.objectives]}
        if self.limits:
            config["limits"] = dict(self.limits)
        if self.reference_point:
            config["reference_point"] = dict(self.reference_point)
        return config


# ---------------------------------------------------------------------------
# Settings and result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProblemSettings:
    """Runtime settings, read from the ``problem_definition`` config section."""

    constraint_tolerance: float = 1e-9          # slack allowed before a constraint counts as violated
    domain_policy: str = "reject"               # out-of-domain assignments: "reject" (raise) or "clip" (repair)
    max_variables: int = 10000
    max_constraints: int = 1000
    max_problems: int = 100
    default_seed: int = 0
    skip_infeasible_evaluations: bool = False   # don't call the evaluator when variable constraints already fail
    penalty_value: float = 1e9                  # objective fill for skipped evaluations (worst bound wins if set)
    sample_attempt_factor: int = 50             # feasible_only sampling gives up after n * factor extra draws

    def __post_init__(self) -> None:
        validate_settings("problem_definition", self, (
            ("constraint_tolerance", is_number(self.constraint_tolerance, 0), "a number >= 0"),
            ("domain_policy", self.domain_policy in DOMAIN_POLICIES, f"one of {DOMAIN_POLICIES}"),
            ("max_variables", is_int(self.max_variables, 1), "an integer >= 1"),
            ("max_constraints", is_int(self.max_constraints, 0), "an integer >= 0"),
            ("max_problems", is_int(self.max_problems, 1), "an integer >= 1"),
            ("default_seed", is_int(self.default_seed, 0), "an integer >= 0"),
            ("skip_infeasible_evaluations", isinstance(self.skip_infeasible_evaluations, bool), "a boolean"),
            ("penalty_value", is_number(self.penalty_value, 0) and self.penalty_value > 0, "a number > 0"),
            ("sample_attempt_factor", is_int(self.sample_attempt_factor, 1), "an integer >= 1"),
        ))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ProblemSettings":
        return settings_from_mapping(cls, config, section="problem_definition", ignore=("problems",))


@dataclass(frozen=True)
class ConstraintReport:
    total_violation: float
    violations: Mapping[str, float]             # every checked constraint (0.0 = satisfied)
    pending: Tuple[str, ...]                    # output constraints that could not be checked (no outputs given)
    satisfied: bool                             # True only if nothing is violated and nothing is pending


@dataclass(frozen=True)
class EvaluationBatch:
    candidates: Tuple[Candidate, ...]
    failures: Mapping[str, str]                 # candidate id (or "index:<i>") -> reason, when on_error="skip"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ProblemDefinition:
    """Builds, registers and serves optimisation problems.

    ``template`` is a partial spec merged *under* every problem defined here (shared variables, objectives,
    constraints ...). It may not carry a ``name``. ``config`` / the ``problem_definition`` config section may
    override any ``ProblemSettings`` field and may hold ``problems``: ``{name: spec}`` registered at start-up.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None, template: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.pd_config = dict(get_config_section("problem_definition", config=self.config) or {})
        if config:
            self.pd_config.update(dict(config))
        self.settings = ProblemSettings.from_config(self.pd_config)

        self._template = _Parts()
        if template:
            try:
                self._template = _parse_spec(template)
            except OptimizationValidationError as exc:
                raise OptimizationTemplateError(f"Invalid template: {exc}") from exc
            if self._template.name:
                raise OptimizationTemplateError("A template cannot carry a problem name")

        self._lock = threading.RLock()
        self._problems: Dict[str, OptimizationProblem] = {}
        for name, spec in (self.pd_config.get("problems") or {}).items():
            self.define({**spec, "name": spec.get("name", name)})
        logger.debug("ProblemDefinition initialised (%d predefined problems)", len(self._problems))

    # ------------------------------------------------------------------ defining and registering
    def build(self, spec: Mapping[str, Any]) -> OptimizationProblem:
        """Validate ``spec`` (merged over the template) into a problem without registering it."""
        parts = _merge_parts(self._template, _parse_spec(spec))
        problem = OptimizationProblem(
            name=parts.name or "problem",
            variables=tuple(parts.variables.values()), objectives=tuple(parts.objectives.values()),
            constraints=tuple(parts.constraints.values()), metrics=tuple(parts.metrics), formulas=parts.formulas,
            limits=parts.limits, reference_point=parts.reference_point, description=parts.description, metadata=parts.metadata,
        )
        if not parts.name:
            problem = replace(problem, name=f"problem-{problem.fingerprint[:8]}")
        if len(problem.variables) > self.settings.max_variables or len(problem.constraints) > self.settings.max_constraints:
            raise OptimizationValidationError("Problem exceeds configured size limits",
                                              context={"variables": len(problem.variables), "max_variables": self.settings.max_variables,
                                                       "constraints": len(problem.constraints), "max_constraints": self.settings.max_constraints})
        return problem

    def define(self, spec: Union[Mapping[str, Any], OptimizationProblem], *, replace_existing: bool = False) -> OptimizationProblem:
        """``build`` + ``register`` in one step."""
        problem = spec if isinstance(spec, OptimizationProblem) else self.build(spec)
        return self.register(problem, replace_existing=replace_existing)

    def register(self, problem: OptimizationProblem, *, replace_existing: bool = False) -> OptimizationProblem:
        """Store a problem by name. Re-registering an identical problem is a no-op; a different one needs ``replace_existing``."""
        with self._lock:
            existing = self._problems.get(problem.name)
            if existing is not None:
                if existing.fingerprint == problem.fingerprint:
                    return existing
                if not replace_existing:
                    raise OptimizationValidationError(f"A different problem named '{problem.name}' is already registered", context={"name": problem.name})
            elif len(self._problems) >= self.settings.max_problems:
                raise OptimizationValidationError("Problem registry is full", context={"max_problems": self.settings.max_problems})
            self._problems[problem.name] = problem
            logger.debug("registered problem '%s' (%s)", problem.name, problem.fingerprint[:8])
            return problem

    def get(self, name: str) -> OptimizationProblem:
        with self._lock:
            if name not in self._problems:
                raise OptimizationValidationError(f"Unknown problem '{name}'", context={"known": sorted(self._problems)})
            return self._problems[name]

    def remove(self, name: str) -> bool:
        with self._lock:
            return self._problems.pop(name, None) is not None

    @property
    def problems(self) -> Dict[str, OptimizationProblem]:
        with self._lock:
            return dict(self._problems)

    def _resolve(self, problem: Union[str, OptimizationProblem]) -> OptimizationProblem:
        return problem if isinstance(problem, OptimizationProblem) else self.get(problem)

    # ------------------------------------------------------------------ assignments and constraints
    def validate_assignment(self, problem: Union[str, OptimizationProblem], assignment: Mapping[str, Any],
                            *, policy: Optional[str] = None) -> Dict[str, Any]:
        """Return a canonical, complete assignment. Unknown or missing variables always raise;
        out-of-domain values raise under ``reject`` and are repaired under ``clip``."""
        prob = self._resolve(problem)
        policy = policy or self.settings.domain_policy
        if policy not in DOMAIN_POLICIES:
            raise OptimizationValidationError(f"Unknown domain policy '{policy}'", context={"allowed": DOMAIN_POLICIES})
        if not isinstance(assignment, Mapping):
            raise OptimizationValidationError("An assignment must be a mapping of variable name to value", context={"type": type(assignment).__name__})
        unknown, missing = set(assignment) - set(prob.variable_names), [n for n in prob.variable_names if n not in assignment]
        if unknown or missing:
            raise OptimizationValidationError("Assignment does not match the problem's variables",
                                              context={"unknown": sorted(unknown), "missing": missing})
        return {v.name: v.coerce(assignment[v.name], clip=policy == "clip") for v in prob.variables}

    def _violations(self, constraints: Sequence[Constraint], namespace: Mapping[str, Any]) -> Dict[str, float]:
        return {c.name: c.violation(namespace, self.settings.constraint_tolerance) for c in constraints}

    def _outputs_namespace(self, problem: OptimizationProblem, outputs: Mapping[str, Any]) -> Dict[str, float]:
        namespace: Dict[str, float] = {}
        for name, value in outputs.items():
            number = to_finite_float(value)
            if name in problem.output_names and number is None:
                raise OptimizationEvaluationError(f"Output '{name}' is not a finite number", context={"output": name, "value": value})
            if number is not None:
                namespace[name] = number
        return namespace

    def check_constraints(self, problem: Union[str, OptimizationProblem], assignment: Mapping[str, Any],
                          outputs: Optional[Mapping[str, Any]] = None, *, policy: Optional[str] = None) -> ConstraintReport:
        """Constraint violations for an assignment. Output constraints need ``outputs`` (objective/metric values)
        and are listed as ``pending`` without them."""
        prob = self._resolve(problem)
        values = self.validate_assignment(prob, assignment, policy=policy)
        violations = self._violations(prob.variable_constraints, values)
        pending: Tuple[str, ...] = ()
        if outputs is None:
            pending = tuple(c.name for c in prob.output_constraints)
        else:
            violations.update(self._violations(prob.output_constraints, {**values, **self._outputs_namespace(prob, outputs)}))
        total = sum(violations.values())
        return ConstraintReport(total, violations, pending, total == 0.0 and not pending)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, problem: Union[str, OptimizationProblem], assignment: Mapping[str, Any],
                 evaluator: Optional[Evaluator] = None, *, candidate_id: Optional[str] = None,
                 policy: Optional[str] = None, skip_infeasible: Optional[bool] = None,
                 metadata: Optional[Mapping[str, Any]] = None) -> Candidate:
        """Evaluate one assignment into a ``Candidate`` ready for ``ParetoEfficiency``.

        Objectives/metrics with a formula are computed analytically; the rest come from ``evaluator(values)``,
        which must return a mapping containing every remaining objective and metric (extra keys are kept in
        ``metadata["extra_outputs"]``). ``Candidate.violation`` is the summed violation of all constraints.
        With ``skip_infeasible`` (default: setting) an assignment that already violates a variable constraint is
        not evaluated; its objectives are filled with the worst bound (or ``penalty_value``) and
        ``metadata["evaluated"]`` is False.
        """
        prob = self._resolve(problem)
        values = self.validate_assignment(prob, assignment, policy=policy)
        cid = candidate_id or f"{prob.name}-{uuid.uuid4().hex[:8]}"
        violations = self._violations(prob.variable_constraints, values)
        skip = self.settings.skip_infeasible_evaluations if skip_infeasible is None else skip_infeasible

        extra: Dict[str, Any] = {}
        if skip and sum(violations.values()) > 0:
            outputs = {o.name: o.sign * (o.worst_bound if o.worst_bound is not None else self.settings.penalty_value)
                       for o in prob.objectives}
            evaluated = False
        else:
            outputs, extra = self._compute_outputs(prob, values, evaluator, cid)
            violations.update(self._violations(prob.output_constraints, {**values, **outputs}))
            evaluated = True

        info = {"problem": prob.name, "fingerprint": prob.fingerprint, "evaluated": evaluated,
                "constraint_violations": {k: v for k, v in violations.items() if v > 0},
                "metrics": {m: outputs[m] for m in prob.metrics if m in outputs}, "extra_outputs": extra}
        info.update(metadata or {})
        return Candidate(id=cid, objectives={n: outputs[n] for n in prob.objective_names}, variables=values,
                         violation=sum(violations.values()), metadata=info)

    @staticmethod
    def _compute_outputs(problem: OptimizationProblem, values: Mapping[str, Any], evaluator: Optional[Evaluator],
                         cid: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
        external: Mapping[str, Any] = {}
        needed = [n for n in problem.output_names if n not in problem.formula_functions]
        if needed:
            if evaluator is None:
                raise OptimizationEvaluationError(f"Problem '{problem.name}' needs an evaluator for: {needed}", context={"outputs": needed})
            try:
                external = evaluator(dict(values))
            except OptimizationEvaluationError:
                raise
            except Exception as exc:
                raise OptimizationEvaluationError(f"Evaluator failed for '{cid}': {exc}", context={"candidate": cid}) from exc
            if not isinstance(external, Mapping):
                raise OptimizationEvaluationError("Evaluator must return a mapping of output name to value", context={"type": type(external).__name__})
        outputs: Dict[str, float] = {}
        for name in problem.output_names:
            if name in problem.formula_functions:
                try:
                    value: Optional[float] = to_finite_float(problem.formula_functions[name](values))
                except (ArithmeticError, ValueError) as exc:
                    raise OptimizationEvaluationError(f"Formula for '{name}' failed: {exc}", context={"candidate": cid, "output": name}) from exc
            else:
                value = to_finite_float(external.get(name))
            if value is None:
                raise OptimizationEvaluationError(f"Output '{name}' is missing or not a finite number for '{cid}'", context={"candidate": cid, "output": name})
            outputs[name] = value
        return outputs, {k: v for k, v in external.items() if k not in problem.output_names}

    def evaluate_many(self, problem: Union[str, OptimizationProblem], assignments: Sequence[Mapping[str, Any]],
                      evaluator: Optional[Evaluator] = None, *, on_error: str = "raise", **kwargs: Any) -> EvaluationBatch:
        """Evaluate many assignments. With ``on_error="skip"`` bad assignments/evaluations are recorded in
        ``failures`` instead of aborting the batch."""
        if on_error not in ("raise", "skip"):
            raise OptimizationValidationError("on_error must be 'raise' or 'skip'", context={"on_error": on_error})
        prob = self._resolve(problem)
        candidates: List[Candidate] = []
        failures: Dict[str, str] = {}
        for index, assignment in enumerate(assignments):
            try:
                candidates.append(self.evaluate(prob, assignment, evaluator, **kwargs))
            except (OptimizationEvaluationError, OptimizationValidationError) as exc:
                if on_error == "raise":
                    raise
                failures[kwargs.get("candidate_id") or f"index:{index}"] = str(exc)
        if failures:
            logger.warning("evaluate_many('%s'): %d of %d evaluations failed", prob.name, len(failures), len(assignments))
        return EvaluationBatch(tuple(candidates), failures)

    # ------------------------------------------------------------------ sampling
    def sample(self, problem: Union[str, OptimizationProblem], n: int, *, seed: Optional[int] = None,
               method: str = "random", feasible_only: bool = False) -> List[Dict[str, Any]]:
        """Draw ``n`` assignments (seeded, reproducible).

        ``latin_hypercube`` stratifies every variable into ``n`` bins. ``feasible_only`` rejects draws that violate
        *variable* constraints (output constraints need an evaluation) and tops up with random draws; it raises
        ``OptimizationEvaluationError`` if ``n`` feasible points are not found within ``n * sample_attempt_factor`` extra draws.
        """
        prob = self._resolve(problem)
        if not is_int(n, 0):
            raise OptimizationValidationError("n must be an integer >= 0", context={"n": n})
        if method not in SAMPLING_METHODS:
            raise OptimizationValidationError(f"Unknown sampling method '{method}'", context={"allowed": SAMPLING_METHODS})
        rng = random.Random(self.settings.default_seed if seed is None else seed)
        dimension = len(prob.variables)

        def decode(unit: Sequence[float]) -> Dict[str, Any]:
            return prob.decode(unit)

        if method == "latin_hypercube" and n > 0:
            columns = []
            for _ in range(dimension):
                strata = list(range(n))
                rng.shuffle(strata)
                columns.append([(s + rng.random()) / n for s in strata])
            drafts = [decode(row) for row in zip(*columns)]
        else:
            drafts = [decode([rng.random() for _ in range(dimension)]) for _ in range(n)]
        if not feasible_only or not prob.variable_constraints:
            return drafts

        def feasible(values: Mapping[str, Any]) -> bool:
            try:
                return all(v == 0.0 for v in self._violations(prob.variable_constraints, values).values())
            except OptimizationEvaluationError:
                return False  # undefined constraint arithmetic (e.g. sqrt of a negative) counts as infeasible

        accepted = [d for d in drafts if feasible(d)]
        attempts = n * self.settings.sample_attempt_factor
        while len(accepted) < n and attempts > 0:
            attempts -= 1
            draft = decode([rng.random() for _ in range(dimension)])
            if feasible(draft):
                accepted.append(draft)
        if len(accepted) < n:
            raise OptimizationEvaluationError(f"Found only {len(accepted)} of {n} feasible samples; the constraints may be too tight or contradictory",
                                              context={"problem": prob.name, "found": len(accepted), "requested": n})
        return accepted

    # ------------------------------------------------------------------ reporting
    def report(self, problem: Union[str, OptimizationProblem]) -> None:
        """Print the problem for humans (``problem.describe()`` is the structured equivalent)."""
        prob = self._resolve(problem)
        printer.section_header(f"Problem {prob.name}")
        printer.table(["variable", "kind", "domain", "initial"],
                      [[v.name, v.kind, f"{list(v.choices)}" if v.kind == CATEGORICAL else f"[{v.lower}, {v.upper}]" + (f" step {v.step}" if v.step and v.kind == CONTINUOUS else ""), v.initial]
                       for v in prob.variables], title="Variables")
        printer.table(["objective", "direction", "weight", "source"],
                      [[o.name, o.direction, o.weight, "formula" if o.name in prob.formulas else "evaluator"] for o in prob.objectives],
                      title="Objectives")
        if prob.constraints:
            printer.table(["constraint", "stage", "definition"],
                          [[c.name, "variables" if c in prob.variable_constraints else "outputs", c.expression or "callable"] for c in prob.constraints],
                          title="Constraints")
        printer.pretty("Problem summary", prob.describe())


__all__ = [
    "ProblemDefinition",
    "ProblemSettings",
    "OptimizationProblem",
    "Variable",
    "Constraint",
    "ConstraintReport",
    "EvaluationBatch",
    "CONTINUOUS",
    "INTEGER",
    "BINARY",
    "CATEGORICAL",
    "DOMAIN_POLICIES",
    "SAMPLING_METHODS",
]