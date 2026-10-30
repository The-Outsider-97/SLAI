"""Dimensionally safe, bounded parametric evaluation for the general-purpose STEM subsystem.

Does not execute Python expressions. Only a small mathematical AST is interpreted.
This is a deterministic evaluator, not a CAS or general nonlinear constraint solver.
Numerical differentiation delegates to STEM NumericalMethods.
"""
from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .math.numerical_methods import NumericalMethods
from .stem_types import Dimension, Quantity, Unit
from .units.unit_system import UnitSystem
from .utils.stem_errors import STEMEquationError, STEMUnitError, STEMValidationError

_NAME = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_FUNCTIONS = frozenset({"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "fabs", "sinh", "cosh", "tanh"})
_CONSTANTS = {"pi": math.pi, "e": math.e}


class ParametricMath:
    """Interprets scalar quantity expressions with explicit units and dependencies.

    Expression examples: ``0.5*w + 0.2*m``, ``pi/2* sin(pi*t/(2*T))**2``.
    Bare numbers are dimensionless. Inputs can be numbers, Quantity, or
    ``{'value': number, 'unit': symbol}``. Units are looked up in UnitSystem.
    The constructor does not mutate the shared UnitSystem configuration.
    """

    MAX_SOURCE = 2048
    MAX_NODES = 256
    MAX_DEPTH = 32
    MAX_PARAMETERS = 256
    MAX_BATCH = 2048
    MAX_EXPONENT = 64
    MAX_ABS_MAGNITUDE = 1e150

    def __init__(self, units: UnitSystem | None = None,
                 numerical_methods: NumericalMethods | None = None) -> None:
        self.units = units if units is not None else UnitSystem()
        self.numerical = numerical_methods if numerical_methods is not None else NumericalMethods()
        self.one = Unit("1", "dimensionless", Dimension())

    def _quantity(self, value: Any) -> Quantity:
        if isinstance(value, Quantity):
            return value
        if isinstance(value, bool):
            raise STEMValidationError("Boolean is not a mathematical quantity")
        if isinstance(value, (int, float)):
            x = float(value)
            if not math.isfinite(x) or abs(x) > self.MAX_ABS_MAGNITUDE:
                raise STEMValidationError("Numeric value outside supported range")
            return Quantity(x, self.one)
        if isinstance(value, Mapping):
            if set(value) - {"value", "unit"} or "value" not in value:
                raise STEMValidationError("Quantity mapping accepts only value and unit")
            raw = self._quantity(value["value"])
            if not raw.unit.dimension.is_dimensionless:
                raise STEMValidationError("Nested quantities are not supported")
            unit = value.get("unit", "1")
            target = self.one if unit == "1" else self.units.lookup(str(unit))
            return Quantity(raw.magnitude, target)
        raise STEMValidationError("Unsupported quantity input type", context={"type": type(value).__name__})

    def _parse(self, expression: str) -> ast.Expression:
        if not isinstance(expression, str) or not expression.strip() or len(expression) > self.MAX_SOURCE:
            raise STEMEquationError("Expression must be non-empty and under the source limit")
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, RecursionError) as exc:
            raise STEMEquationError("Invalid mathematical expression", cause=exc) from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > self.MAX_NODES:
            raise STEMEquationError("Expression exceeds AST node budget")
        for node in nodes:
            if not isinstance(node, (ast.Expression, ast.Constant, ast.Name, ast.Load,
                                      ast.BinOp, ast.UnaryOp, ast.Call, ast.Add, ast.Sub,
                                      ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)):
                raise STEMEquationError("Forbidden mathematical syntax", context={"node": type(node).__name__})
            if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or
                    not isinstance(node.value, (float, int)) or
                    not math.isfinite(float(node.value)) or
                    abs(float(node.value)) > self.MAX_ABS_MAGNITUDE):
                raise STEMEquationError("Unsupported numeric literal")
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or
                (len(node.args) != 1 and not (node.func.id == "log" and len(node.args) == 2)) or bool(node.keywords)
            ):
                raise STEMEquationError("Only allowlisted one-argument functions are supported")
        return tree

    def _names(self, tree: ast.Expression) -> set[str]:
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - _FUNCTIONS

    def _unit_name(self, name: str) -> Quantity | None:
        if name == "rad":
            return Quantity(1.0, Unit("rad", "radian", Dimension()))
        if name == "deg":
            return Quantity(1.0, Unit("deg", "degree", Dimension(), math.pi / 180.0))
        try:
            return Quantity(1.0, self.units.lookup(name))
        except STEMUnitError:
            return None

    def _calculate(self, node: ast.AST, env: Mapping[str, Quantity], depth: int = 0) -> Quantity:
        if depth > self.MAX_DEPTH:
            raise STEMEquationError("Expression depth exceeds limit")
        if isinstance(node, ast.Constant):
            return self._quantity(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in _CONSTANTS:
                return self._quantity(_CONSTANTS[node.id])
            result = self._unit_name(node.id)
            if result is not None:
                return result
            raise STEMEquationError("Unknown parameter or unit", context={"name": node.id})
        if isinstance(node, ast.UnaryOp):
            value = self._calculate(node.operand, env, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Call):
            arg = self._calculate(node.args[0], env, depth + 1)
            if node.func.id == "sqrt":
                return arg ** 0.5
            if node.func.id in ("abs", "fabs"):
                return Quantity(abs(arg.magnitude), arg.unit)
            if not arg.unit.dimension.is_dimensionless:
                raise STEMEquationError("Mathematical function requires dimensionless argument")
            # Base units handle degree/radian conversion before trig functions.
            x = arg.unit.to_base(arg.magnitude)
            if node.func.id == "log" and len(node.args) == 2:
                base = self._calculate(node.args[1], env, depth + 1)
                if not base.unit.dimension.is_dimensionless:
                    raise STEMEquationError("Logarithm base must be dimensionless")
                try:
                    return self._quantity(math.log(x, base.unit.to_base(base.magnitude)))
                except (ValueError, OverflowError, ZeroDivisionError) as exc:
                    raise STEMEquationError("Logarithm domain error", cause=exc) from exc
            try:
                f = {"sin": math.sin, "cos": math.cos, "tan": math.tan,
                     "exp": math.exp, "log": math.log, "abs": abs,
                     "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
                     "sqrt": math.sqrt}[node.func.id]
                y = f(x)
            except (ValueError, OverflowError, ZeroDivisionError) as exc:
                raise STEMEquationError("Function domain or range error", cause=exc) from exc
            return self._quantity(y)
        if isinstance(node, ast.BinOp):
            lhs = self._calculate(node.left, env, depth + 1)
            rhs = self._calculate(node.right, env, depth + 1)
            try:
                if isinstance(node.op, ast.Add): return lhs + rhs
                if isinstance(node.op, ast.Sub): return lhs - rhs
                if isinstance(node.op, ast.Mult): return lhs * rhs
                if isinstance(node.op, ast.Div): return lhs / rhs
                if isinstance(node.op, ast.Pow):
                    if not rhs.unit.dimension.is_dimensionless:
                        raise STEMEquationError("Exponent must be dimensionless")
                    power = rhs.unit.to_base(rhs.magnitude)
                    if abs(power) > self.MAX_EXPONENT:
                        raise STEMEquationError("Exponent exceeds permitted magnitude")
                    if lhs.magnitude == 0 and power < 0:
                        raise STEMEquationError("Zero cannot have a negative exponent")
                    if lhs.magnitude < 0 and not power.is_integer():
                        raise STEMEquationError("Complex-valued powers are not supported")
                    if lhs.unit.is_affine:
                        raise STEMEquationError("Affine unit cannot be exponentiated")
                    return Quantity(lhs.magnitude ** power, lhs.unit ** power)
            except (ZeroDivisionError, ValueError, OverflowError) as exc:
                raise STEMEquationError("Invalid arithmetic operation", cause=exc) from exc
        raise STEMEquationError("Unsupported AST node")

    def evaluate_expression(self, expression: str, parameters: Mapping[str, Any] | None = None) -> Quantity:
        tree = self._parse(expression)
        params = {} if parameters is None else parameters
        if not isinstance(params, Mapping) or len(params) > self.MAX_PARAMETERS:
            raise STEMValidationError("Invalid parameter mapping or parameter limit exceeded")
        env: dict[str, Quantity] = {}
        for name, value in params.items():
            if not isinstance(name, str) or not _NAME.fullmatch(name) or name in _FUNCTIONS or name in _CONSTANTS or (name != "T" and self._unit_name(name) is not None):
                raise STEMValidationError("Invalid or reserved parameter name", context={"name": str(name)})
            env[name] = self._quantity(value)
        return self._calculate(tree.body, env)

    def evaluate(self, expressions: Mapping[str, str],
                 parameters: Mapping[str, Any] | None = None) -> dict[str, Quantity]:
        """Topologically evaluate derived parameters and detect cycles/missing symbols."""
        if not isinstance(expressions, Mapping) or len(expressions) > self.MAX_PARAMETERS:
            raise STEMValidationError("Invalid expression mapping or graph too large")
        raw = parameters or {}
        if not isinstance(raw, Mapping):
            raise STEMValidationError("parameters must be mapping")
        if set(expressions) & set(raw):
            raise STEMEquationError("Derived and independent parameter names overlap")
        trees = {}
        dependencies = {}
        for name, expression in expressions.items():
            if not isinstance(name, str) or not _NAME.fullmatch(name) or name in _CONSTANTS or name in _FUNCTIONS or (name != "T" and self._unit_name(name) is not None):
                raise STEMEquationError("Invalid or reserved derived parameter name")
            tree = self._parse(expression)
            trees[name] = tree
            dependencies[name] = {n for n in self._names(tree) if n in expressions}
        env = {}
        for name, value in raw.items():
            if not isinstance(name, str) or not _NAME.fullmatch(name) or name in _FUNCTIONS or name in _CONSTANTS or (name != "T" and self._unit_name(name) is not None):
                raise STEMValidationError("Invalid or reserved independent parameter name")
            env[name] = self._quantity(value)
        status: dict[str, int] = {}
        result: dict[str, Quantity] = {}

        def visit(name: str) -> None:
            state = status.get(name, 0)
            if state == 1:
                raise STEMEquationError("Circular dependency detected", context={"parameter": name})
            if state == 2:
                return
            status[name] = 1
            for dep in sorted(dependencies[name]):
                visit(dep)
            value = self._calculate(trees[name].body, {**env, **result})
            result[name] = value
            status[name] = 2

        for name in expressions:
            visit(name)
        return result

    def evaluate_serializable(self, expressions: Mapping[str, str],
                              parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {name: quantity.to_dict() for name, quantity in self.evaluate(expressions, parameters).items()}

    def evaluate_series(self, expression: str, parameters: Mapping[str, Any],
                        variable: str, samples: Sequence[Any]) -> list[dict[str, Any]]:
        if not isinstance(samples, (tuple, list)) or not 1 <= len(samples) <= self.MAX_BATCH:
            raise STEMValidationError("samples must be a bounded nonempty sequence")
        if variable not in parameters:
            raise STEMEquationError("Series variable must be declared in parameters")
        tree = self._parse(expression)  # Parse once for the entire series.
        if not isinstance(parameters, Mapping) or len(parameters) > self.MAX_PARAMETERS:
            raise STEMValidationError("Invalid parameters")
        base = {}
        for key, value in parameters.items():
            if not isinstance(key, str) or not _NAME.fullmatch(key) or key in _FUNCTIONS or key in _CONSTANTS or (key != "T" and self._unit_name(key) is not None):
                raise STEMValidationError("Invalid or reserved independent parameter name")
            base[key] = self._quantity(value)
        output = []
        for value in samples:
            state = dict(base)
            state[variable] = self._quantity(value)
            output.append(self._calculate(tree.body, state).to_dict())
        return output

    def validate_constraints(self, constraints: Sequence[str],
                             parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate inequalities/equalities against explicit physical dimensions.

        This checks feasibility only; it does NOT solve constraints or optimize.
        Comparisons operate on coherent base values to avoid unit bias.
        """
        if not isinstance(constraints, (list, tuple)) or len(constraints) > 128:
            raise STEMValidationError("constraints must be a bounded sequence")
        outcomes = []
        for expression in constraints:
            if not isinstance(expression, str):
                raise STEMValidationError("constraint must be a string")
            matched = re.fullmatch(r"(.+?)\s*(<=|>=|==|<|>)\s*(.+)", expression)
            if not matched:
                raise STEMEquationError("Constraint requires exactly one comparison operator")
            lhs = self.evaluate_expression(matched.group(1).strip(), parameters)
            rhs = self.evaluate_expression(matched.group(3).strip(), parameters)
            if lhs.unit.dimension != rhs.unit.dimension:
                raise STEMUnitError("Incompatible dimensions in constraint")
            a, b = lhs.unit.to_base(lhs.magnitude), rhs.unit.to_base(rhs.magnitude)
            op = matched.group(2)
            passed = {"<": a < b, "<=": a <= b, "==": math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12),
                      ">=": a >= b, ">": a > b}[op]
            outcomes.append({"constraint": expression, "passed": passed,
                             "difference_in_base_units": a - b})
        return {"feasible": all(item["passed"] for item in outcomes), "constraints": outcomes}

    def derivative(self, expression: str, parameters: Mapping[str, Any], variable: str,
                   step: float | None = None) -> dict[str, Any]:
        """Numerical derivative: delegated to NumericalMethods.central_difference.

        This provides a pointwise approximation, not a symbolic proof. User must
        choose an interior point where the function is differentiable.
        """
        if variable not in parameters:
            raise STEMEquationError("Missing differentiation variable")
        independent = self._quantity(parameters[variable])
        if independent.unit.is_affine:
            raise STEMEquationError("Cannot differentiate with respect to affine temperature")
        x = independent.magnitude
        base_result = self.evaluate_expression(expression, parameters)
        if base_result.unit.is_affine:
            raise STEMEquationError("Cannot differentiate affine temperature output")

        def f(v: float) -> float:
            state = dict(parameters)
            state[variable] = Quantity(v, independent.unit)
            result = self.evaluate_expression(expression, state)
            if result.unit.dimension != base_result.unit.dimension:
                raise STEMEquationError("Derivative changed output dimension")
            return result.unit.to_base(result.magnitude)

        derivative_base = self.numerical.central_difference(f, x, h=step)
        # Numerical derivative divides output base units by one native x unit;
        # account for that native unit scale when reporting the SI-derived unit.
        unit = Unit("d(" + str(base_result.unit.dimension) + ")/d(" + str(independent.unit.dimension) + ")",
                    dimension=base_result.unit.dimension / independent.unit.dimension)
        val = derivative_base / independent.unit.scale
        return {"value": val, "unit": unit.to_dict(), "method": "central_difference",
                "status": "approximate", "step": step}


__all__ = ["ParametricMath"]
