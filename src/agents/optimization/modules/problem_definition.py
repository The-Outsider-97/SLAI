"""Declarative problem construction and evaluation for SLAI optimization.

This module turns plain mappings, safe analytic expressions and Python callables
into the canonical solver-independent objects from ``optimization_types``.  It
also provides deterministic registration, assignment validation, constraint
reports, batch evaluation and mixed-domain sampling.  It does not solve
problems; solver execution remains in ``optimization/solvers``.

The model/solver separation follows Bynum et al. (2021), while constrained
problem semantics follow Boyd & Vandenberghe (2004) and Nocedal & Wright
(2006).
"""
from __future__ import annotations

__version__ = "2.3.0"

import copy
import math
import random
import re
import threading
import yaml # type: ignore

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from ..utils.temp_loader import load_template
from ..optimization_math import *
from ..optimization_types import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Problem Definition")
printer = PrettyPrinter()


# Backward-compatible domain aliases retained from the pre-hardening API.
# Internally the subsystem uses ``DomainKind``; these names remain data-level
# constants so existing callers do not need to change imports.
CONTINUOUS = "continuous"
INTEGER = "integer"
BINARY = "binary"
CATEGORICAL = "categorical"
DOMAIN_POLICIES = ("reject", "clip")
SAMPLING_METHODS = ("random", "latin_hypercube")

_RELATION = re.compile(r"(<=|>=|==|<|>)")
_KIND_ALIASES = {
    "real": DomainKind.REAL,
    "continuous": DomainKind.REAL,
    "float": DomainKind.REAL,
    "integer": DomainKind.INTEGER,
    "int": DomainKind.INTEGER,
    "binary": DomainKind.BINARY,
    "bool": DomainKind.BINARY,
    "boolean": DomainKind.BINARY,
    "finite": DomainKind.FINITE,
    "categorical": DomainKind.FINITE,
    "category": DomainKind.FINITE,
    "choice": DomainKind.FINITE,
    "semi_continuous": DomainKind.SEMI_CONTINUOUS,
    "semicontinuous": DomainKind.SEMI_CONTINUOUS,
    "semi_integer": DomainKind.SEMI_INTEGER,
    "semiinteger": DomainKind.SEMI_INTEGER,
    "custom": DomainKind.CUSTOM,
}


@dataclass(frozen=True, slots=True)
class ProblemSettings:
    feasibility_tolerance: float = 1e-8
    sample_attempt_factor: int = 50
    max_expression_length: int = 1000

    def __post_init__(self) -> None:
        if self.feasibility_tolerance < 0:
            raise OptimizationValidationError("feasibility_tolerance must be non-negative")
        if self.sample_attempt_factor <= 0 or self.max_expression_length <= 0:
            raise OptimizationValidationError("problem-definition integer settings must be positive")


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    feasible: bool
    violations: Mapping[str, float]
    max_violation: float


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    assignment: Mapping[str, Any]
    objective_values: Mapping[str, float]
    feasible: bool
    violations: Mapping[str, float]
    max_violation: float


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    records: tuple[EvaluationRecord, ...]

    @property
    def feasible_count(self) -> int:
        return sum(1 for record in self.records if record.feasible)

    @property
    def infeasible_count(self) -> int:
        return len(self.records) - self.feasible_count


Evaluator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _build_domain(spec: Any) -> VariableDomain:
    if isinstance(spec, VariableDomain):
        return spec
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(to_finite_float(item) is not None for item in spec):
            return VariableDomain.real(float(spec[0]), float(spec[1]))
        return VariableDomain.finite(tuple(spec))
    if isinstance(spec, str):
        kind = _KIND_ALIASES.get(spec.strip().lower())
        if kind is None:
            raise OptimizationValidationError("unknown variable domain type", context={"type": spec})
        if kind is DomainKind.BINARY:
            return VariableDomain.binary()
        if kind is DomainKind.REAL:
            return VariableDomain.real()
        if kind is DomainKind.INTEGER:
            return VariableDomain.integer()
        raise OptimizationValidationError("domain type requires additional parameters", context={"type": spec})
    if not isinstance(spec, Mapping):
        raise OptimizationValidationError("variable specification must be mapping, bounds pair, finite values, or domain type")

    kind_raw = str(spec.get("type", spec.get("kind", "real"))).strip().lower()
    kind = _KIND_ALIASES.get(kind_raw)
    if kind is None:
        raise OptimizationValidationError("unknown variable domain type", context={"type": kind_raw})
    if kind is DomainKind.BINARY:
        return VariableDomain.binary()
    if kind is DomainKind.FINITE:
        values = spec.get("values", spec.get("choices", spec.get("domain", ())))
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise OptimizationValidationError("finite domain requires a sequence of values")
        return VariableDomain.finite(tuple(values))
    lower = spec.get("lower", spec.get("lb"))
    upper = spec.get("upper", spec.get("ub"))
    strict_lower = bool(spec.get("strict_lower", False))
    strict_upper = bool(spec.get("strict_upper", False))
    if kind is DomainKind.REAL:
        return VariableDomain.real(lower, upper, strict_lower=strict_lower, strict_upper=strict_upper)
    if kind is DomainKind.INTEGER:
        return VariableDomain.integer(lower, upper)
    if kind is DomainKind.SEMI_CONTINUOUS:
        if lower is None or upper is None:
            raise OptimizationValidationError("semi-continuous domain requires lower and upper bounds")
        return VariableDomain.semi_continuous(float(lower), float(upper))
    if kind is DomainKind.SEMI_INTEGER:
        if lower is None or upper is None:
            raise OptimizationValidationError("semi-integer domain requires lower and upper bounds")
        return VariableDomain.semi_integer(lower, upper)
    return VariableDomain(DomainKind.CUSTOM, lower=lower, upper=upper)


def _build_variable(name: str, spec: Any) -> Variable:
    if isinstance(spec, Variable):
        if spec.name != name:
            raise OptimizationValidationError("variable mapping key/name mismatch", context={"key": name, "variable": spec.name})
        return spec
    if isinstance(spec, Mapping):
        domain = _build_domain(spec)
        initial = spec.get("initial", spec.get("warm_start"))
        metadata = dict(spec.get("metadata", {})) if isinstance(spec.get("metadata", {}), Mapping) else {}
        shape_raw = spec.get("shape", ())
        shape = tuple(int(item) for item in shape_raw) if shape_raw else ()
        return Variable(name=name, domain=domain, initial=initial, shape=shape, metadata=metadata)
    return Variable(name=name, domain=_build_domain(spec))


def _parse_quadratic(raw: Any) -> dict[tuple[str, str], float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise OptimizationValidationError("quadratic coefficients must be a mapping")
    result: dict[tuple[str, str], float] = {}
    for key, value in raw.items():
        if isinstance(key, str):
            if "*" in key:
                left, right = [part.strip() for part in key.split("*", 1)]
            elif "," in key:
                left, right = [part.strip() for part in key.split(",", 1)]
            else:
                left = right = key.strip()
            pair = (left, right)
        elif isinstance(key, tuple) and len(key) == 2:
            pair = (str(key[0]), str(key[1]))
        else:
            raise OptimizationValidationError("quadratic key must be tuple or 'x*y' string", context={"key": repr(key)})
        result[pair] = require_finite(value, f"quadratic[{pair}]")
    return result


def _build_objective(name: str, spec: Any, *, max_expression_length: int) -> Objective:
    if isinstance(spec, Objective):
        if spec.name != name:
            raise OptimizationValidationError("objective mapping key/name mismatch", context={"key": name, "objective": spec.name})
        return spec
    if isinstance(spec, str):
        normalized = spec.strip().lower()
        if normalized in {"min", "minimize", "max", "maximize"}:
            raise OptimizationValidationError("objective direction alone is insufficient; supply formula/linear/quadratic/function")
        function, _ = compile_expression(spec, max_length=max_expression_length)
        return Objective(name=name, function=function)
    if callable(spec):
        return Objective(name=name, function=cast(ScalarCallable, spec))
    if not isinstance(spec, Mapping):
        raise OptimizationValidationError("objective specification must be mapping, expression string, callable, or Objective")
    sense = ObjectiveSense.coerce(spec.get("sense", spec.get("direction", "minimize")))
    function = spec.get("function")
    formula = spec.get("formula", spec.get("expression"))
    if function is not None and not callable(function):
        raise OptimizationValidationError("objective function must be callable")
    if formula is not None:
        if function is not None:
            raise OptimizationValidationError("objective cannot define both function and formula")
        function, references = compile_expression(str(formula), max_length=max_expression_length)
        metadata = dict(spec.get("metadata", {})) if isinstance(spec.get("metadata", {}), Mapping) else {}
        metadata["expression"] = str(formula)
        metadata["expression_references"] = tuple(sorted(references))
    else:
        metadata = dict(spec.get("metadata", {})) if isinstance(spec.get("metadata", {}), Mapping) else {}
    linear_raw = spec.get("linear", {})
    if not isinstance(linear_raw, Mapping):
        raise OptimizationValidationError("objective.linear must be a mapping")
    return Objective(
        name=name,
        sense=sense,
        function=cast(ScalarCallable | None, function),
        linear={str(key): require_finite(value, f"linear[{key}]") for key, value in linear_raw.items()},
        quadratic=_parse_quadratic(spec.get("quadratic")),
        constant=spec.get("constant", 0.0),
        weight=spec.get("weight", 1.0),
        priority=int(spec.get("priority", 0)),
        gradient=spec.get("gradient") if callable(spec.get("gradient")) else None,
        hessian=spec.get("hessian") if callable(spec.get("hessian")) else None,
        metadata=metadata,
    )


def _constraint_from_relation(name: str, source: str, *, tolerance: float, max_expression_length: int) -> Constraint:
    match = _RELATION.search(source)
    if match is None:
        raise OptimizationValidationError("constraint expression must contain <=, >=, ==, <, or >", context={"constraint": source})
    left_text = source[: match.start()].strip()
    right_text = source[match.end() :].strip()
    if not left_text or not right_text:
        raise OptimizationValidationError("constraint relation must have both sides", context={"constraint": source})
    left_fn, left_refs = compile_expression(left_text, max_length=max_expression_length)
    right_fn, right_refs = compile_expression(right_text, max_length=max_expression_length)

    def difference(point: Mapping[str, Any]) -> float:
        return left_fn(point) - right_fn(point)

    return Constraint(
        name=name,
        sense=ConstraintSense.coerce(match.group(1)),
        rhs=0.0,
        function=difference,
        tolerance=tolerance,
        metadata={"expression": source, "expression_references": tuple(sorted(left_refs | right_refs))},
    )


def _build_constraint(index: int, spec: Any, *, default_tolerance: float, max_expression_length: int) -> Constraint:
    if isinstance(spec, Constraint):
        return spec
    if isinstance(spec, str):
        return _constraint_from_relation(f"constraint_{index}", spec, tolerance=default_tolerance, max_expression_length=max_expression_length)
    if not isinstance(spec, Mapping):
        raise OptimizationValidationError("constraint specification must be mapping, relation string, or Constraint")
    name = str(spec.get("name", f"constraint_{index}"))
    tolerance = spec.get("tolerance", default_tolerance)
    expression = spec.get("expression", spec.get("formula"))
    if expression is not None and _RELATION.search(str(expression)):
        relation = _constraint_from_relation(name, str(expression), tolerance=float(tolerance), max_expression_length=max_expression_length)
        return relation
    function = spec.get("function")
    if function is not None and not callable(function):
        raise OptimizationValidationError("constraint function must be callable")
    metadata = dict(spec.get("metadata", {})) if isinstance(spec.get("metadata", {}), Mapping) else {}
    if expression is not None:
        if function is not None:
            raise OptimizationValidationError("constraint cannot define both function and formula")
        function, references = compile_expression(str(expression), max_length=max_expression_length)
        metadata["expression"] = str(expression)
        metadata["expression_references"] = tuple(sorted(references))
    linear_raw = spec.get("linear", {})
    if not isinstance(linear_raw, Mapping):
        raise OptimizationValidationError("constraint.linear must be a mapping")
    return Constraint(
        name=name,
        sense=ConstraintSense.coerce(spec.get("sense", "<=")),
        rhs=spec.get("rhs", spec.get("bound", 0.0)),
        function=cast(ScalarCallable | None, function),
        linear={str(key): require_finite(value, f"constraint[{name}].linear[{key}]") for key, value in linear_raw.items()},
        tolerance=tolerance,
        jacobian=spec.get("jacobian") if callable(spec.get("jacobian")) else None,
        metadata=metadata,
    )


_CATALOG_VERSION = "1.0"
_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})
_SUPPORTED_TEMPLATE_VERSIONS = frozenset({"1.0"})
_TEMPLATE_PARAMETER_TYPES = frozenset({"any", "boolean", "integer", "mapping", "number", "sequence", "string"})
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.]*)\}")
_FULL_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_.]*)\}$")
_FORBIDDEN_TEMPLATE_KEYS = frozenset({
    "backend", "fallback_solver", "max_iterations", "population_size",
    "solver", "solver_name", "solver_priority",
})


def _template_error(message: str, **context: Any) -> OptimizationTemplateError:
    return OptimizationTemplateError(
        message,
        context={key: value for key, value in context.items() if value is not None},
    )


def _load_template_catalog(filename: str, root_key: str) -> Mapping[str, Any]:
    """Parse canonical YAML text loaded through the existing template loader."""
    raw = load_template(filename)
    if not isinstance(raw, str):
        raise _template_error("Optimization YAML template asset must load as text", template_file=filename)
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise _template_error("Invalid Optimization template YAML", template_file=filename) from exc
    if not isinstance(document, Mapping):
        raise _template_error("Optimization template YAML root must be a mapping", template_file=filename)
    if document.get("format_version") != _CATALOG_VERSION:
        raise _template_error(
            "Unsupported Optimization template catalog version",
            template_file=filename,
            expected=_CATALOG_VERSION,
            received=document.get("format_version"),
        )
    payload = document.get(root_key)
    if not isinstance(payload, Mapping):
        raise _template_error("Optimization template catalog is missing its required mapping", template_file=filename, required_key=root_key)
    return copy.deepcopy(payload)


def _mapping(value: Any, label: str, *, template: str | None = None, schema: str | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _template_error(f"{label} must be a mapping", template=template, schema=schema, received_type=type(value).__name__)
    return value


def _reject_unknown_fields(value: Mapping[str, Any], allowed: set[str], label: str, *, template: str | None = None, schema: str | None = None) -> None:
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise _template_error(f"{label} contains unsupported fields", template=template, schema=schema, fields=unknown)


def _validate_schema(schema_name: str, raw: Any) -> Mapping[str, Any]:
    schema = _mapping(raw, "schema", schema=schema_name)
    _reject_unknown_fields(schema, {"schema_version", "description", "parameters", "problem", "validation"}, "schema", schema=schema_name)
    if schema.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
        raise _template_error("Unsupported optimization schema version", schema=schema_name, received=schema.get("schema_version"))
    if not isinstance(schema.get("description"), str) or not str(schema["description"]).strip():
        raise _template_error("Optimization schema requires a description", schema=schema_name)
    parameters = _mapping(schema.get("parameters", {}), "schema.parameters", schema=schema_name)
    _mapping(schema.get("problem", {}), "schema.problem", schema=schema_name)
    validation = _mapping(schema.get("validation", {}), "schema.validation", schema=schema_name)
    if validation.get("unknown_parameters", "reject") != "reject":
        raise _template_error("Only the 'reject' unknown-parameter policy is supported", schema=schema_name, policy=validation.get("unknown_parameters"))
    for name, raw_spec in parameters.items():
        if not isinstance(name, str) or not name.strip():
            raise _template_error("Schema parameter names must be non-empty strings", schema=schema_name)
        spec = _mapping(raw_spec, "parameter specification", schema=schema_name)
        _reject_unknown_fields(
            spec,
            {"type", "required", "default", "minimum", "maximum", "required_keys", "values_type", "description"},
            "parameter specification",
            schema=schema_name,
        )
        parameter_type = str(spec.get("type", "any")).strip().lower()
        if parameter_type not in _TEMPLATE_PARAMETER_TYPES:
            raise _template_error("Unsupported template parameter type", schema=schema_name, parameter=name, parameter_type=parameter_type)
        if not isinstance(spec.get("required", False), bool):
            raise _template_error("parameter.required must be boolean", schema=schema_name, parameter=name)
        if spec.get("required", False) and "default" in spec:
            raise _template_error("Required template parameter cannot define a default", schema=schema_name, parameter=name)
    return schema


def _find_forbidden_template_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_TEMPLATE_KEYS:
                return normalized
            found = _find_forbidden_template_key(child)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = _find_forbidden_template_key(child)
            if found:
                return found
    return None


def _validate_template(template_name: str, raw: Any) -> Mapping[str, Any]:
    template = _mapping(raw, "template", template=template_name)
    _reject_unknown_fields(template, {"template_version", "schema", "description", "problem", "metadata"}, "template", template=template_name)
    if template.get("template_version") not in _SUPPORTED_TEMPLATE_VERSIONS:
        raise _template_error("Unsupported optimization template version", template=template_name, received=template.get("template_version"))
    schema_name = template.get("schema")
    if not isinstance(schema_name, str) or not schema_name.strip():
        raise _template_error("Optimization template requires a schema reference", template=template_name)
    if not isinstance(template.get("description"), str) or not str(template["description"]).strip():
        raise _template_error("Optimization template requires a description", template=template_name, schema=schema_name)
    problem = _mapping(template.get("problem"), "template.problem", template=template_name, schema=schema_name)
    _mapping(template.get("metadata", {}), "template.metadata", template=template_name, schema=schema_name)
    forbidden = _find_forbidden_template_key(problem)
    if forbidden:
        raise _template_error(
            "Optimization templates may not contain solver/backend configuration",
            template=template_name,
            schema=schema_name,
            offending_key=forbidden,
        )
    return template


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "any":
        return True
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and to_finite_float(value) is not None
    if expected == "string":
        return isinstance(value, str)
    if expected == "mapping":
        return isinstance(value, Mapping)
    return expected == "sequence" and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _validate_parameter(schema_name: str, name: str, value: Any, spec: Mapping[str, Any]) -> None:
    expected = str(spec.get("type", "any")).strip().lower()
    if not _type_matches(value, expected):
        raise _template_error("Template parameter has wrong type", schema=schema_name, parameter=name, expected_type=expected, received_type=type(value).__name__)
    if expected in {"number", "integer"}:
        numeric = to_finite_float(value)
        if numeric is None:
            raise _template_error("Template numeric parameter must be finite", schema=schema_name, parameter=name)
        minimum = to_finite_float(spec.get("minimum")) if "minimum" in spec else None
        maximum = to_finite_float(spec.get("maximum")) if "maximum" in spec else None
        if "minimum" in spec and (minimum is None or numeric < minimum):
            raise _template_error("Template parameter is below its minimum", schema=schema_name, parameter=name, minimum=spec.get("minimum"))
        if "maximum" in spec and (maximum is None or numeric > maximum):
            raise _template_error("Template parameter exceeds its maximum", schema=schema_name, parameter=name, maximum=spec.get("maximum"))
    required_keys = spec.get("required_keys")
    if required_keys is not None:
        if not isinstance(value, Mapping) or not isinstance(required_keys, Sequence) or isinstance(required_keys, (str, bytes, bytearray)):
            raise _template_error("parameter.required_keys requires mapping data and a key sequence", schema=schema_name, parameter=name)
        missing = [str(key) for key in required_keys if key not in value]
        if missing:
            raise _template_error("Template mapping parameter is missing required keys", schema=schema_name, parameter=name, missing_keys=missing)
    values_type = spec.get("values_type")
    if values_type is not None:
        item_type = str(values_type).strip().lower()
        if item_type not in {"boolean", "integer", "number", "string"}:
            raise _template_error("Unsupported parameter values_type", schema=schema_name, parameter=name, values_type=item_type)
        values = value.values() if isinstance(value, Mapping) else value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else None
        if values is None:
            raise _template_error("parameter.values_type requires mapping or sequence data", schema=schema_name, parameter=name)
        for item in values:
            if not _type_matches(item, item_type):
                raise _template_error("Template parameter contains value of wrong type", schema=schema_name, parameter=name, expected_value_type=item_type, received_type=type(item).__name__)


def _bind_parameters(schema_name: str, schema: Mapping[str, Any], parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = {} if parameters is None else parameters
    if not isinstance(supplied, Mapping):
        raise _template_error("Template parameters must be a mapping", schema=schema_name, received_type=type(supplied).__name__)
    specs = _mapping(schema.get("parameters", {}), "schema.parameters", schema=schema_name)
    unknown = sorted(str(key) for key in supplied if key not in specs)
    if unknown:
        raise _template_error("Unknown template parameters are not allowed", schema=schema_name, unknown_parameters=unknown)
    bound: dict[str, Any] = {}
    for name, raw_spec in specs.items():
        spec = _mapping(raw_spec, "parameter specification", schema=schema_name)
        if name in supplied:
            value = copy.deepcopy(supplied[name])
        elif "default" in spec:
            value = copy.deepcopy(spec["default"])
        elif spec.get("required", False):
            raise _template_error("Missing required template parameter", schema=schema_name, parameter=name)
        else:
            continue
        _validate_parameter(schema_name, str(name), value, spec)
        bound[str(name)] = value
    return bound


def _resolve_parameter(parameters: Mapping[str, Any], path: str, *, template: str, schema: str) -> Any:
    value: Any = parameters
    for segment in path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            raise _template_error("Template placeholder cannot be resolved", template=template, schema=schema, placeholder=path, missing_segment=segment)
        value = value[segment]
    return copy.deepcopy(value)


def _substitute(value: Any, parameters: Mapping[str, Any], *, template: str, schema: str) -> Any:
    if isinstance(value, str):
        full = _FULL_PLACEHOLDER.fullmatch(value)
        if full:
            return _resolve_parameter(parameters, full.group(1), template=template, schema=schema)
        rendered = value
        for match in tuple(_PLACEHOLDER.finditer(value)):
            replacement = _resolve_parameter(parameters, match.group(1), template=template, schema=schema)
            if isinstance(replacement, (Mapping, list, tuple, set)):
                raise _template_error("Container parameters require a complete placeholder value", template=template, schema=schema, placeholder=match.group(1))
            rendered = rendered.replace(match.group(0), str(replacement))
        if "${" in rendered:
            raise _template_error("Malformed template placeholder", template=template, schema=schema, value=rendered)
        return rendered
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _substitute(child, parameters, template=template, schema=schema) for key, child in value.items()}
    if isinstance(value, list):
        return [_substitute(child, parameters, template=template, schema=schema) for child in value]
    if isinstance(value, tuple):
        return tuple(_substitute(child, parameters, template=template, schema=schema) for child in value)
    return copy.deepcopy(value)


def _structure(spec: Any) -> str:
    if isinstance(spec, Mapping):
        if spec.get("formula") is not None or spec.get("expression") is not None or spec.get("function") is not None:
            return "expression"
        if spec.get("quadratic"):
            return "quadratic"
        if spec.get("linear"):
            return "linear"
    return "expression" if isinstance(spec, str) else "unknown"


def _collection(value: Any, label: str, *, template: str, schema: str) -> list[Any]:
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise _template_error(f"Template {label} must be a mapping or sequence", template=template, schema=schema)


def _check_collection_contract(label: str, raw: Any, contract: Mapping[str, Any], *, template: str, schema: str, structure: bool = False) -> list[Any]:
    items = _collection(raw, label, template=template, schema=schema)
    minimum, maximum = contract.get("min_items"), contract.get("max_items")
    for field, bound in (("min_items", minimum), ("max_items", maximum)):
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, int) or bound < 0):
            raise _template_error(f"schema.problem.{label}.{field} must be a non-negative integer", schema=schema)
    if minimum is not None and len(items) < minimum:
        raise _template_error(f"Template has too few {label}", template=template, schema=schema, minimum=minimum, received=len(items))
    if maximum is not None and len(items) > maximum:
        raise _template_error(f"Template has too many {label}", template=template, schema=schema, maximum=maximum, received=len(items))
    allowed = contract.get("allowed_structures") if structure else None
    if allowed is not None:
        if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes, bytearray)):
            raise _template_error(f"schema.problem.{label}.allowed_structures must be a sequence", schema=schema)
        allowed_set = {str(item).strip().lower() for item in allowed}
        for item in items:
            actual = _structure(item)
            if actual not in allowed_set:
                raise _template_error(f"Template {label[:-1]} structure is not allowed by its schema", template=template, schema=schema, structure=actual, allowed_structures=sorted(allowed_set))
    return items


def _validate_problem_contract(template_name: str, schema_name: str, schema: Mapping[str, Any], problem: Mapping[str, Any]) -> None:
    contract = _mapping(schema.get("problem", {}), "schema.problem", schema=schema_name)
    _reject_unknown_fields(contract, {"class", "variables", "objectives", "constraints"}, "schema.problem", schema=schema_name)
    expected_class = contract.get("class")
    actual_class = problem.get("problem_class")
    if expected_class is not None and (not isinstance(actual_class, str) or actual_class.strip().lower() != str(expected_class).strip().lower()):
        raise _template_error("Template problem class does not match its schema", template=template_name, schema=schema_name, expected_class=expected_class, received_class=actual_class)

    variable_contract = _mapping(contract.get("variables", {}), "schema.problem.variables", schema=schema_name)
    _reject_unknown_fields(variable_contract, {"min_items", "max_items", "allowed_domains"}, "schema.problem.variables", schema=schema_name)
    variables = _check_collection_contract("variables", problem.get("variables", {}), variable_contract, template=template_name, schema=schema_name)
    allowed_domains = variable_contract.get("allowed_domains")
    if allowed_domains is not None:
        if not isinstance(allowed_domains, Sequence) or isinstance(allowed_domains, (str, bytes, bytearray)):
            raise _template_error("schema.problem.variables.allowed_domains must be a sequence", schema=schema_name)
        allowed = {str(item).strip().lower() for item in allowed_domains}
        for variable in variables:
            domain = _build_domain(variable)
            if domain.kind.value not in allowed:
                raise _template_error("Template variable domain is not allowed by its schema", template=template_name, schema=schema_name, domain=domain.kind.value, allowed_domains=sorted(allowed))

    objectives = problem.get("objectives")
    if objectives is None and "objective" in problem:
        objectives = {"objective": problem["objective"]}
    objective_contract = _mapping(contract.get("objectives", {}), "schema.problem.objectives", schema=schema_name)
    _reject_unknown_fields(objective_contract, {"min_items", "max_items", "allowed_structures"}, "schema.problem.objectives", schema=schema_name)
    _check_collection_contract("objectives", objectives, objective_contract, template=template_name, schema=schema_name, structure=True)

    constraint_contract = _mapping(contract.get("constraints", {}), "schema.problem.constraints", schema=schema_name)
    _reject_unknown_fields(constraint_contract, {"min_items", "max_items", "allowed_structures"}, "schema.problem.constraints", schema=schema_name)
    _check_collection_contract("constraints", problem.get("constraints", ()), constraint_contract, template=template_name, schema=schema_name, structure=True)


class ProblemDefinition:
    def __init__(self, config: Mapping[str, Any] | None = None, template: Mapping[str, Any] | None = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("problem_definition", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.settings = ProblemSettings(
            feasibility_tolerance=require_finite(section.get("feasibility_tolerance", 1e-8), "feasibility_tolerance"),
            sample_attempt_factor=int(section.get("sample_attempt_factor", 50)),
            max_expression_length=int(section.get("max_expression_length", 1000)),
        )
        self.template = dict(template or {})
        self._problems: dict[str, OptimizationProblem] = {}
        self._lock = threading.RLock()

    def from_template(
        self,
        name: str,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> OptimizationProblem:
        """Instantiate a named read-only blueprint as a canonical ``OptimizationProblem``."""
        if not isinstance(name, str) or not name.strip():
            raise _template_error("Template name must be a non-empty string", template=name)
        template_name = name.strip()
        schemas = _load_template_catalog("schemas.yaml", "schemas")
        templates = _load_template_catalog("instances.yaml", "templates")
        if template_name not in templates:
            raise _template_error("Unknown Optimization problem template", template=template_name)

        template = _validate_template(template_name, templates[template_name])
        schema_name = str(template["schema"]).strip()
        if schema_name not in schemas:
            raise _template_error("Optimization template references an unknown schema", template=template_name, schema=schema_name)
        schema = _validate_schema(schema_name, schemas[schema_name])
        bound = _bind_parameters(schema_name, schema, parameters)
        problem_spec = _substitute(copy.deepcopy(template["problem"]), bound, template=template_name, schema=schema_name)
        if not isinstance(problem_spec, Mapping):
            raise _template_error("Template substitution did not produce a problem mapping", template=template_name, schema=schema_name)
        _validate_problem_contract(template_name, schema_name, schema, problem_spec)

        spec = dict(problem_spec)
        metadata = dict(spec.get("metadata", {})) if isinstance(spec.get("metadata", {}), Mapping) else {}
        if "template_provenance" in metadata:
            raise _template_error("Template problem metadata reserves 'template_provenance'", template=template_name, schema=schema_name)
        metadata["template_provenance"] = {
            "template": template_name,
            "template_version": template["template_version"],
            "schema": schema_name,
            "schema_version": schema["schema_version"],
        }
        if template.get("metadata"):
            metadata["template_metadata"] = copy.deepcopy(dict(template["metadata"]))
        spec["metadata"] = metadata

        try:
            return self.build(spec)
        except OptimizationValidationError as exc:
            exc.add_context(template=template_name, schema=schema_name)
            raise

    def build(self, spec: Mapping[str, Any]) -> OptimizationProblem:
        if not isinstance(spec, Mapping):
            raise OptimizationValidationError("problem specification must be a mapping")
        name = str(spec.get("name", "optimization_problem")).strip()
        if not name:
            raise OptimizationValidationError("problem name must be non-empty")

        raw_variables = spec.get("variables", {})
        if isinstance(raw_variables, Mapping):
            variables = tuple(_build_variable(str(key), value) for key, value in raw_variables.items())
        elif isinstance(raw_variables, Sequence) and not isinstance(raw_variables, (str, bytes)):
            variables_list: list[Variable] = []
            for item in raw_variables:
                if isinstance(item, Variable):
                    variables_list.append(item)
                elif isinstance(item, Mapping) and "name" in item:
                    variables_list.append(_build_variable(str(item["name"]), item))
                else:
                    raise OptimizationValidationError("variable sequence members require name/spec mappings")
            variables = tuple(variables_list)
        else:
            raise OptimizationValidationError("variables must be mapping or sequence")

        raw_objectives = spec.get("objectives")
        if raw_objectives is None and "objective" in spec:
            raw_objectives = {"objective": spec["objective"]}
        if isinstance(raw_objectives, Mapping):
            objectives = tuple(
                _build_objective(str(key), value, max_expression_length=self.settings.max_expression_length)
                for key, value in raw_objectives.items()
            )
        elif isinstance(raw_objectives, Sequence) and not isinstance(raw_objectives, (str, bytes)):
            objective_items: list[Objective] = []
            for item in raw_objectives:
                if isinstance(item, Objective):
                    objective_items.append(item)
                elif isinstance(item, Mapping) and "name" in item:
                    objective_items.append(_build_objective(str(item["name"]), item, max_expression_length=self.settings.max_expression_length))
                else:
                    raise OptimizationValidationError("objective sequence members require name/spec mappings")
            objectives = tuple(objective_items)
        else:
            raise OptimizationValidationError("objectives must be supplied as mapping or sequence")

        raw_constraints = spec.get("constraints", ())
        if not isinstance(raw_constraints, Sequence) or isinstance(raw_constraints, (str, bytes)):
            raise OptimizationValidationError("constraints must be a sequence")
        constraints = tuple(
            _build_constraint(
                index,
                item,
                default_tolerance=self.settings.feasibility_tolerance,
                max_expression_length=self.settings.max_expression_length,
            )
            for index, item in enumerate(raw_constraints)
        )

        metadata = dict(spec.get("metadata", {})) if isinstance(spec.get("metadata", {}), Mapping) else {}
        problem = OptimizationProblem(
            name=name,
            variables=variables,
            objectives=objectives,
            constraints=constraints,
            known_convex=spec.get("known_convex"),
            known_smooth=spec.get("known_smooth"),
            derivatives_available=bool(spec.get("derivatives_available", False)),
            deterministic=bool(spec.get("deterministic", True)),
            expected_evaluation_cost=spec.get("expected_evaluation_cost"),
            metadata=metadata,
        )
        problem = with_classification(problem)
        if "problem_class" in spec:
            requested = str(spec["problem_class"]).strip().lower()
            if requested != problem.problem_class.value:
                try:
                    requested_class = ProblemClass(requested)
                except ValueError as exc:
                    raise OptimizationValidationError("unknown problem_class", context={"problem_class": requested}) from exc
                deliberate_discrete = requested_class in {ProblemClass.CP, ProblemClass.COMBINATORIAL} and all(
                    variable.domain.kind in {DomainKind.INTEGER, DomainKind.BINARY, DomainKind.FINITE, DomainKind.SEMI_INTEGER}
                    for variable in problem.variables
                )
                deliberate_black_box = requested_class is ProblemClass.BLACK_BOX
                if not (deliberate_discrete or deliberate_black_box):
                    raise OptimizationValidationError(
                        "explicit problem_class conflicts with inferred mathematical structure",
                        context={"requested": requested, "inferred": problem.problem_class.value},
                    )
                problem = OptimizationProblem(
                    name=problem.name,
                    variables=problem.variables,
                    objectives=problem.objectives,
                    constraints=problem.constraints,
                    problem_class=requested_class,
                    known_convex=problem.known_convex,
                    known_smooth=problem.known_smooth,
                    derivatives_available=problem.derivatives_available,
                    deterministic=problem.deterministic,
                    expected_evaluation_cost=problem.expected_evaluation_cost,
                    metadata=problem.metadata,
                )
        self._validate_expression_references(problem)
        return problem

    @staticmethod
    def _validate_expression_references(problem: OptimizationProblem) -> None:
        variables = set(problem.variable_names)
        for objective in problem.objectives:
            refs = set(objective.metadata.get("expression_references", ()))
            unknown = refs - variables
            if unknown:
                raise OptimizationValidationError("objective expression references unknown variables", context={"objective": objective.name, "unknown": sorted(unknown)})
        for constraint in problem.constraints:
            refs = set(constraint.metadata.get("expression_references", ()))
            unknown = refs - variables
            if unknown:
                raise OptimizationValidationError("constraint expression references unknown variables", context={"constraint": constraint.name, "unknown": sorted(unknown)})

    def define(self, spec: Mapping[str, Any] | OptimizationProblem, *, replace_existing: bool = False) -> OptimizationProblem:
        problem = spec if isinstance(spec, OptimizationProblem) else self.build(spec)
        return self.register(problem, replace_existing=replace_existing)

    def register(self, problem: OptimizationProblem, *, replace_existing: bool = False) -> OptimizationProblem:
        if not isinstance(problem, OptimizationProblem):
            raise OptimizationValidationError("register requires OptimizationProblem")
        with self._lock:
            if problem.name in self._problems and not replace_existing:
                raise OptimizationValidationError("problem is already registered", context={"problem": problem.name})
            self._problems[problem.name] = problem
        return problem

    def get(self, name: str) -> OptimizationProblem:
        with self._lock:
            try:
                return self._problems[name]
            except KeyError as exc:
                raise OptimizationValidationError("unknown optimization problem", context={"problem": name}) from exc

    def remove(self, name: str) -> bool:
        with self._lock:
            return self._problems.pop(name, None) is not None

    def problems(self) -> dict[str, OptimizationProblem]:
        with self._lock:
            return dict(self._problems)

    def _resolve(self, problem: OptimizationProblem | str) -> OptimizationProblem:
        return self.get(problem) if isinstance(problem, str) else problem

    def validate_assignment(
        self,
        problem: OptimizationProblem | str,
        assignment: Mapping[str, Any],
        *,
        allow_extra: bool = False,
    ) -> dict[str, Any]:
        model = self._resolve(problem)
        if not isinstance(assignment, Mapping):
            raise OptimizationValidationError("assignment must be a mapping")
        expected = set(model.variable_names)
        actual = set(assignment)
        missing = expected - actual
        extra = actual - expected
        if missing or (extra and not allow_extra):
            raise OptimizationValidationError(
                "assignment variable mismatch",
                context={"missing": sorted(missing), "extra": sorted(extra)},
            )
        validated: dict[str, Any] = {}
        for variable in model.variables:
            validated[variable.name] = variable.validate_value(assignment[variable.name])
        if allow_extra:
            for key in extra:
                validated[key] = assignment[key]
        return validated

    def check_constraints(self, problem: OptimizationProblem | str, assignment: Mapping[str, Any]) -> ConstraintReport:
        model = self._resolve(problem)
        validated = self.validate_assignment(model, assignment)
        violations = constraint_violations(model, validated)
        maximum = max(violations.values(), default=0.0)
        feasible = is_feasible(model, validated, tolerance=self.settings.feasibility_tolerance)
        return ConstraintReport(feasible, violations, maximum)

    def evaluate(
        self,
        problem: OptimizationProblem | str,
        assignment: Mapping[str, Any],
        evaluator: Evaluator | None = None,
    ) -> EvaluationRecord:
        model = self._resolve(problem)
        validated = self.validate_assignment(model, assignment)
        namespace: dict[str, Any] = dict(validated)
        if evaluator is not None:
            try:
                extra = evaluator(validated)
            except Exception as exc:
                raise OptimizationEvaluationError("external evaluator failed", cause=exc) from exc
            if not isinstance(extra, Mapping):
                raise OptimizationEvaluationError("external evaluator must return a mapping")
            namespace.update(extra)
        values = objective_values(model, namespace)
        violations = constraint_violations(model, namespace)
        maximum = max(violations.values(), default=0.0)
        feasible = is_feasible(model, namespace, tolerance=self.settings.feasibility_tolerance)
        return EvaluationRecord(validated, values, feasible, violations, maximum)

    def evaluate_many(
        self,
        problem: OptimizationProblem | str,
        assignments: Sequence[Mapping[str, Any]],
        evaluator: Evaluator | None = None,
    ) -> EvaluationBatch:
        model = self._resolve(problem)
        records = tuple(self.evaluate(model, assignment, evaluator=evaluator) for assignment in assignments)
        return EvaluationBatch(records)

    def sample(
        self,
        problem: OptimizationProblem | str,
        n: int,
        *,
        seed: int | None = None,
        method: str = "random",
        feasible_only: bool = True,
    ) -> list[dict[str, Any]]:
        model = self._resolve(problem)
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise OptimizationValidationError("sample size must be a positive integer")
        if method not in {"random", "latin_hypercube"}:
            raise OptimizationValidationError("sampling method must be random or latin_hypercube")
        rng = random.Random(seed)
        lhs_units: dict[str, list[float]] = {}
        if method == "latin_hypercube":
            for variable in model.variables:
                strata = [(i + rng.random()) / n for i in range(n)]
                rng.shuffle(strata)
                lhs_units[variable.name] = strata

        def draw(variable: Variable, sample_index: int) -> Any:
            domain = variable.domain
            u = rng.random() if method == "random" else lhs_units[variable.name][sample_index % n]
            if domain.kind is DomainKind.FINITE:
                index = min(int(u * len(domain.values)), len(domain.values) - 1)
                return domain.values[index]
            if domain.kind is DomainKind.BINARY:
                return int(u >= 0.5)
            if domain.kind in {DomainKind.INTEGER, DomainKind.SEMI_INTEGER}:
                if domain.lower is None or domain.upper is None:
                    raise OptimizationValidationError("sampling integer variables requires finite bounds")
                values = list(range(int(domain.lower), int(domain.upper) + 1))
                if domain.kind is DomainKind.SEMI_INTEGER:
                    values = [0, *values]
                return values[min(int(u * len(values)), len(values) - 1)]
            if domain.kind in {DomainKind.REAL, DomainKind.SEMI_CONTINUOUS}:
                if domain.lower is None or domain.upper is None:
                    raise OptimizationValidationError("sampling continuous variables requires finite bounds")
                if domain.kind is DomainKind.SEMI_CONTINUOUS and u < 1.0 / (len(model.variables) + 1):
                    return 0.0
                return domain.lower + u * (domain.upper - domain.lower)
            raise OptimizationValidationError("cannot sample custom domain", context={"variable": variable.name})

        samples: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = max(n, n * self.settings.sample_attempt_factor)
        while len(samples) < n and attempts < max_attempts:
            sample_index = attempts % n
            attempts += 1
            point = {variable.name: draw(variable, sample_index) for variable in model.variables}
            if not feasible_only or is_feasible(model, point, tolerance=self.settings.feasibility_tolerance):
                samples.append(point)
        if len(samples) < n:
            logger.warning(
                "Optimization sampling returned fewer feasible points than requested | problem=%s | requested=%d | returned=%d | attempts=%d",
                model.name,
                n,
                len(samples),
                attempts,
            )
        return samples

    @staticmethod
    def describe(problem: OptimizationProblem) -> dict[str, Any]:
        return {
            "name": problem.name,
            "problem_class": problem.problem_class.value,
            "variables": [
                {
                    "name": variable.name,
                    "domain": variable.domain.kind.value,
                    "lower": variable.domain.lower,
                    "upper": variable.domain.upper,
                    "values": list(variable.domain.values),
                    "initial": variable.initial,
                }
                for variable in problem.variables
            ],
            "objectives": [
                {
                    "name": objective.name,
                    "sense": objective.sense.value,
                    "weight": objective.weight,
                    "priority": objective.priority,
                    "linear_terms": len(objective.linear),
                    "quadratic_terms": len(objective.quadratic),
                    "callable": objective.function is not None,
                }
                for objective in problem.objectives
            ],
            "constraints": [
                {
                    "name": constraint.name,
                    "sense": constraint.sense.value,
                    "rhs": constraint.rhs,
                    "linear_terms": len(constraint.linear),
                    "callable": constraint.function is not None,
                    "tolerance": constraint.tolerance,
                }
                for constraint in problem.constraints
            ],
            "known_convex": problem.known_convex,
            "known_smooth": problem.known_smooth,
            "derivatives_available": problem.derivatives_available,
            "deterministic": problem.deterministic,
        }

    def report(self, problem: OptimizationProblem | str) -> None:
        model = self._resolve(problem)
        logger.info(
            "Optimization problem | name=%s | class=%s | variables=%d | objectives=%d | constraints=%d",
            model.name,
            model.problem_class.value,
            len(model.variables),
            len(model.objectives),
            len(model.constraints),
        )


__all__ = [
    "BINARY",
    "CATEGORICAL",
    "CONTINUOUS",
    "Constraint",
    "ConstraintReport",
    "DOMAIN_POLICIES",
    "EvaluationBatch",
    "EvaluationRecord",
    "Evaluator",
    "INTEGER",
    "Objective",
    "OptimizationProblem",
    "ProblemDefinition",
    "ProblemSettings",
    "SAMPLING_METHODS",
    "Variable",
]
