"""
Analytic and algorithmic calculus primitives for the STEM subsystem.

Ownership
- symbolic derivatives;
- symbolic integrals where supported;
- gradients / Jacobians / Hessians as mathematical operators;
- directional derivatives;
- limits;
- Taylor series generation;
- analytic differentiation;
- forward-mode automatic differentiation via dual numbers.

This module deliberately excludes finite-difference differentiation. Finite
differences belong to ``numerical_methods.py``.

Sources
- Bronstein, M. (2005). Symbolic Integration I: Transcendental Functions
  (2nd ed.). Springer.
- Griewank, A., & Walther, A. (2008). Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation (2nd ed.). SIAM.
  DOI 10.1137/1.9780898717761.
"""

from __future__ import annotations

__version__ = "2.3.0"

import ast
import math

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import *
from ..utils.stem_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Calculus")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Dual numbers for forward-mode automatic differentiation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dual:
    """
    Forward-mode dual number carrying a value and a vector of partials.

    Arithmetic rules follow standard dual-number AD:
        (a + a'ε) + (b + b'ε) = (a + b) + (a' + b')ε
        (a + a'ε) * (b + b'ε) = a b + (a' b + a b')ε
    """

    value: float
    partials: Tuple[float, ...] = ()

    def _coerce(self, other: Any) -> "Dual":
        if isinstance(other, Dual):
            return other
        return Dual(float(other), tuple(0.0 for _ in self.partials))

    def _zip_partials(self, other: "Dual") -> List[Tuple[float, float]]:
        if len(self.partials) != len(other.partials):
            raise STEMDifferentiationError(
                "dual partial vectors must have matching length",
                context={"left": len(self.partials), "right": len(other.partials)},
            )
        return list(zip(self.partials, other.partials))

    def __add__(self, other: Any) -> "Dual":
        other_d = self._coerce(other)
        return Dual(self.value + other_d.value, tuple(a + b for a, b in self._zip_partials(other_d)))

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Dual":
        other_d = self._coerce(other)
        return Dual(self.value - other_d.value, tuple(a - b for a, b in self._zip_partials(other_d)))

    def __rsub__(self, other: Any) -> "Dual":
        other_d = self._coerce(other)
        return other_d.__sub__(self)

    def __mul__(self, other: Any) -> "Dual":
        other_d = self._coerce(other)
        new_partials = tuple(
            a * other_d.value + self.value * b for a, b in self._zip_partials(other_d)
        )
        return Dual(self.value * other_d.value, new_partials)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "Dual":
        other_d = self._coerce(other)
        if other_d.value == 0.0:
            raise STEMDifferentiationError("division by zero in dual arithmetic")
        new_partials = tuple(
            (a * other_d.value - self.value * b) / (other_d.value ** 2)
            for a, b in self._zip_partials(other_d)
        )
        return Dual(self.value / other_d.value, new_partials)

    def __rtruediv__(self, other: Any) -> "Dual":
        other_d = self._coerce(other)
        return other_d.__truediv__(self)

    def __pow__(self, exponent: Any) -> "Dual":
        if isinstance(exponent, Dual):
            raise STEMDifferentiationError("dual exponentiation by dual is not supported")
        exponent_f = float(exponent)
        value = self.value ** exponent_f
        if self.value == 0.0:
            # d/dx x^n at x=0 is 0 for n>1, 1 for n=1, undefined for n<1
            if exponent_f == 1.0:
                new_partials = tuple(1.0 for _ in self.partials)
            elif exponent_f > 1.0:
                new_partials = tuple(0.0 for _ in self.partials)
            else:
                raise STEMDifferentiationError("power derivative undefined at x=0")
        else:
            factor = exponent_f * (self.value ** (exponent_f - 1.0))
            new_partials = tuple(factor * p for p in self.partials)
        return Dual(value, new_partials)

    def __neg__(self) -> "Dual":
        return Dual(-self.value, tuple(-p for p in self.partials))

    def __pos__(self) -> "Dual":
        return self

    # elementwise helpers used by Calculus

    def sin(self) -> "Dual":
        c = math.cos(self.value)
        return Dual(math.sin(self.value), tuple(c * p for p in self.partials))

    def cos(self) -> "Dual":
        s = -math.sin(self.value)
        return Dual(math.cos(self.value), tuple(s * p for p in self.partials))

    def tan(self) -> "Dual":
        c = math.cos(self.value)
        if c == 0.0:
            raise STEMDifferentiationError("tan derivative undefined at odd multiples of pi/2")
        factor = 1.0 / (c * c)
        return Dual(math.tan(self.value), tuple(factor * p for p in self.partials))

    def exp(self) -> "Dual":
        e = math.exp(self.value)
        return Dual(e, tuple(e * p for p in self.partials))

    def log(self) -> "Dual":
        if self.value <= 0.0:
            raise STEMDifferentiationError("log derivative undefined for non-positive value")
        factor = 1.0 / self.value
        return Dual(math.log(self.value), tuple(factor * p for p in self.partials))

    def sqrt(self) -> "Dual":
        if self.value < 0.0:
            raise STEMDifferentiationError("sqrt derivative undefined for negative value")
        s = math.sqrt(self.value)
        if s == 0.0:
            raise STEMDifferentiationError("sqrt derivative undefined at zero")
        factor = 0.5 / s
        return Dual(s, tuple(factor * p for p in self.partials))

    def sinh(self) -> "Dual":
        c = math.cosh(self.value)
        return Dual(math.sinh(self.value), tuple(c * p for p in self.partials))

    def cosh(self) -> "Dual":
        s = math.sinh(self.value)
        return Dual(math.cosh(self.value), tuple(s * p for p in self.partials))

    def tanh(self) -> "Dual":
        t = math.tanh(self.value)
        factor = 1.0 - t * t
        return Dual(t, tuple(factor * p for p in self.partials))

    def abs(self) -> "Dual":
        if self.value > 0.0:
            return Dual(self.value, tuple(p for p in self.partials))
        if self.value < 0.0:
            return Dual(-self.value, tuple(-p for p in self.partials))
        raise STEMDifferentiationError("abs derivative undefined at zero")


# ---------------------------------------------------------------------------
# Symbolic IR: parse simple AST, differentiate, unparse
# ---------------------------------------------------------------------------

_IRNode = Tuple[Any, ...]

_SUPPORTED_FUNCTIONS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "log": "log",
    "ln": "log",
    "sqrt": "sqrt",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
}


def _to_ir(node: ast.AST) -> _IRNode:
    """Convert a restricted Python AST expression into tuple IR."""
    if isinstance(node, ast.Expression):
        return _to_ir(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise STEMCalculusError(
                "only numeric constants are supported in symbolic expressions",
                context={"value": repr(node.value)},
            )
        return ("num", float(node.value))

    if isinstance(node, ast.Name):
        return ("var", node.id)

    if isinstance(node, ast.BinOp):
        left = _to_ir(node.left)
        right = _to_ir(node.right)
        if isinstance(node.op, ast.Add):
            return ("add", left, right)
        if isinstance(node.op, ast.Sub):
            return ("sub", left, right)
        if isinstance(node.op, ast.Mult):
            return ("mul", left, right)
        if isinstance(node.op, ast.Div):
            return ("div", left, right)
        if isinstance(node.op, ast.Pow):
            return ("pow", left, right)
        raise STEMCalculusError(
            "unsupported binary operator",
            context={"operator": type(node.op).__name__},
        )

    if isinstance(node, ast.UnaryOp):
        operand = _to_ir(node.operand)
        if isinstance(node.op, ast.USub):
            return ("neg", operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        raise STEMCalculusError(
            "unsupported unary operator",
            context={"operator": type(node.op).__name__},
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SUPPORTED_FUNCTIONS:
            raise STEMCalculusError(
                "unsupported function in symbolic expression",
                context={"callable": getattr(node.func, "id", repr(node.func))},
            )
        if len(node.args) != 1 or node.keywords:
            raise STEMCalculusError("only single-argument functions are supported")
        arg = _to_ir(node.args[0])
        return ("call", _SUPPORTED_FUNCTIONS[node.func.id], arg)

    raise STEMCalculusError(
        "unsupported AST node in symbolic expression",
        context={"node": type(node).__name__},
    )


def _is_constant(node: _IRNode) -> bool:
    return node[0] == "num"


def _is_zero(node: _IRNode) -> bool:
    return node[0] == "num" and node[1] == 0.0


def _is_one(node: _IRNode) -> bool:
    return node[0] == "num" and node[1] == 1.0


def _make_num(value: float) -> _IRNode:
    return ("num", float(value))


def _make_add(a: _IRNode, b: _IRNode) -> _IRNode:
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    if _is_constant(a) and _is_constant(b):
        return _make_num(a[1] + b[1])
    return ("add", a, b)


def _make_sub(a: _IRNode, b: _IRNode) -> _IRNode:
    if _is_zero(b):
        return a
    if _is_zero(a):
        return ("neg", b)
    if _is_constant(a) and _is_constant(b):
        return _make_num(a[1] - b[1])
    return ("sub", a, b)


def _make_mul(a: _IRNode, b: _IRNode) -> _IRNode:
    if _is_zero(a) or _is_zero(b):
        return _make_num(0.0)
    if _is_one(a):
        return b
    if _is_one(b):
        return a
    if _is_constant(a) and _is_constant(b):
        return _make_num(a[1] * b[1])
    return ("mul", a, b)


def _make_div(a: _IRNode, b: _IRNode) -> _IRNode:
    if _is_zero(a):
        return _make_num(0.0)
    if _is_one(b):
        return a
    if _is_constant(a) and _is_constant(b) and b[1] != 0.0:
        return _make_num(a[1] / b[1])
    return ("div", a, b)


def _make_pow(a: _IRNode, b: _IRNode) -> _IRNode:
    if _is_zero(b):
        return _make_num(1.0)
    if _is_one(b):
        return a
    return ("pow", a, b)


def _differentiate_ir(node: _IRNode, var: str) -> _IRNode:
    """Symbolically differentiate an IR expression with respect to ``var``."""
    tag = node[0]

    if tag == "num":
        return _make_num(0.0)

    if tag == "var":
        return _make_num(1.0) if node[1] == var else _make_num(0.0)

    if tag == "add":
        return _make_add(_differentiate_ir(node[1], var), _differentiate_ir(node[2], var))

    if tag == "sub":
        return _make_sub(_differentiate_ir(node[1], var), _differentiate_ir(node[2], var))

    if tag == "neg":
        return ("neg", _differentiate_ir(node[1], var))

    if tag == "mul":
        a, b = node[1], node[2]
        da = _differentiate_ir(a, var)
        db = _differentiate_ir(b, var)
        return _make_add(_make_mul(da, b), _make_mul(a, db))

    if tag == "div":
        a, b = node[1], node[2]
        da = _differentiate_ir(a, var)
        db = _differentiate_ir(b, var)
        numerator = _make_sub(_make_mul(da, b), _make_mul(a, db))
        denominator = _make_pow(b, _make_num(2.0))
        return _make_div(numerator, denominator)

    if tag == "pow":
        base, exponent = node[1], node[2]
        if _is_constant(exponent):
            n = exponent[1]
            new_exponent = _make_num(n - 1.0)
            factor = _make_mul(_make_num(n), _make_pow(base, new_exponent))
            return _make_mul(factor, _differentiate_ir(base, var))
        # general base^exponent: (b^e) * (e' * log(b) + e * b'/b)
        da = _differentiate_ir(exponent, var)
        db = _differentiate_ir(base, var)
        term1 = _make_mul(da, ("call", "log", base))
        term2 = _make_mul(exponent, _make_div(db, base))
        return _make_mul(_make_pow(base, exponent), _make_add(term1, term2))

    if tag == "call":
        func = node[1]
        arg = node[2]
        darg = _differentiate_ir(arg, var)

        if func == "sin":
            return _make_mul(("call", "cos", arg), darg)
        if func == "cos":
            return _make_mul(("neg", ("call", "sin", arg)), darg)
        if func == "tan":
            t = ("call", "tan", arg)
            sec_sq = _make_add(_make_num(1.0), _make_pow(t, _make_num(2.0)))
            return _make_mul(sec_sq, darg)
        if func == "exp":
            return _make_mul(("call", "exp", arg), darg)
        if func == "log":
            return _make_mul(_make_div(_make_num(1.0), arg), darg)
        if func == "sqrt":
            denominator = _make_mul(_make_num(2.0), ("call", "sqrt", arg))
            return _make_mul(_make_div(_make_num(1.0), denominator), darg)
        if func == "sinh":
            return _make_mul(("call", "cosh", arg), darg)
        if func == "cosh":
            return _make_mul(("call", "sinh", arg), darg)
        if func == "tanh":
            t = ("call", "tanh", arg)
            factor = _make_sub(_make_num(1.0), _make_pow(t, _make_num(2.0)))
            return _make_mul(factor, darg)
        raise STEMCalculusError(
            "unsupported function in symbolic differentiation",
            context={"function": func},
        )

    raise STEMCalculusError("unsupported IR node", context={"tag": tag})


def _ir_to_str(node: _IRNode) -> str:
    """Render IR as a readable infix string."""
    tag = node[0]

    if tag == "num":
        value = node[1]
        if float(value).is_integer():
            return str(int(value))
        return repr(value)

    if tag == "var":
        return node[1]

    if tag == "neg":
        return f"(-{_ir_to_str(node[1])})"

    if tag in ("add", "sub", "mul", "div", "pow"):
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}[tag]
        return f"({_ir_to_str(node[1])} {op} {_ir_to_str(node[2])})"

    if tag == "call":
        return f"{node[1]}({_ir_to_str(node[2])})"

    raise STEMCalculusError("unsupported IR node in string rendering", context={"tag": tag})


def _integrate_ir(node: _IRNode, var: str) -> Optional[_IRNode]:
    """
    Symbolic integration of a restricted subset: linear combinations of
    constants, ``var``, and ``var^n`` for constant n. Returns ``None`` when
    unsupported.
    """
    tag = node[0]

    if tag == "num":
        return _make_mul(node, ("var", var))

    if tag == "var":
        if node[1] == var:
            return _make_mul(_make_num(0.5), _make_pow(("var", var), _make_num(2.0)))
        return _make_mul(node, ("var", var))

    if tag in ("add", "sub"):
        left = _integrate_ir(node[1], var)
        right = _integrate_ir(node[2], var)
        if left is None or right is None:
            return None
        if tag == "add":
            return _make_add(left, right)
        return _make_sub(left, right)

    if tag == "neg":
        inner = _integrate_ir(node[1], var)
        return None if inner is None else ("neg", inner)

    if tag == "mul":
        a, b = node[1], node[2]
        if _is_constant(a):
            inner = _integrate_ir(b, var)
            return None if inner is None else _make_mul(a, inner)
        if _is_constant(b):
            inner = _integrate_ir(a, var)
            return None if inner is None else _make_mul(b, inner)
        return None

    if tag == "pow":
        base, exponent = node[1], node[2]
        if base == ("var", var) and _is_constant(exponent):
            n = exponent[1]
            if n == -1.0:
                return ("call", "log", ("var", var))
            new_exp = _make_num(n + 1.0)
            return _make_mul(_make_div(_make_num(1.0), new_exp), _make_pow(base, new_exp))
        return None

    return None


# ---------------------------------------------------------------------------
# Calculus class
# ---------------------------------------------------------------------------


class Calculus:
    """
    Analytic and algorithmic calculus primitives.

    Provides symbolic differentiation/integration for a bounded expression
    subset, forward-mode AD via dual numbers, and derivative operators over
    Python callables.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.calculus_config = dict(get_config_section("calculus", config=self.config) or {})
        if config:
            self.calculus_config.update(dict(config))

        self._eps_step = float(self.calculus_config.get("eps_step", 1e-6))
        self._default_order = int(self.calculus_config.get("default_taylor_order", 4))
        self._limit_h = float(self.calculus_config.get("limit_step", 1e-7))

    # -- symbolic ----------------------------------------------------------

    def symbolic_derivative(self, expression: str, variable: str, order: int = 1) -> str:
        """
        Return the symbolic derivative of a bounded arithmetic expression.

        Supported operators: +, -, *, /, **, unary -, and the functions
        sin, cos, tan, exp, log, sqrt, sinh, cosh, tanh.
        """
        ensure_non_empty_string(expression, "expression", error_cls=cast(type[STEMValidationError], STEMCalculusError))
        ensure_non_empty_string(variable, "variable", error_cls=cast(type[STEMValidationError], STEMCalculusError))
        ensure_positive(order, "order", allow_zero=False, error_cls=STEMCalculusError)

        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise STEMCalculusError(
                "Failed to parse expression",
                context={"expression": expression},
                cause=exc,
            ) from exc

        ir = _to_ir(parsed)
        for _ in range(int(order)):
            ir = _differentiate_ir(ir, variable)

        return _ir_to_str(ir)

    def symbolic_integral(self, expression: str, variable: str) -> Optional[str]:
        """
        Return the antiderivative for a supported subset, or ``None`` when no
        closed form is available within that subset.
        """
        ensure_non_empty_string(expression, "expression", error_cls=cast(type[STEMValidationError], STEMCalculusError))
        ensure_non_empty_string(variable, "variable", error_cls=cast(type[STEMValidationError], STEMCalculusError))

        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise STEMCalculusError(
                "Failed to parse expression",
                context={"expression": expression},
                cause=exc,
            ) from exc

        ir = _to_ir(parsed)
        result = _integrate_ir(ir, variable)
        if result is None:
            return None
        return _ir_to_str(result)

    # -- AD over Python callables -----------------------------------------

    def _dual_call(
        self,
        f: Callable[..., Any],
        x: Sequence[float],
    ) -> Dual:
        variables = as_float_array(x, "x", error_cls=STEMDifferentiationError)
        dual_inputs = [
            Dual(value, tuple(1.0 if i == j else 0.0 for i in range(len(variables))))
            for j, value in enumerate(variables)
        ]

        try:
            result = f(*dual_inputs)
        except STEMDifferentiationError:
            raise
        except Exception as exc:
            raise STEMDifferentiationError(
                "function evaluation raised during AD",
                context={"callable": getattr(f, "__name__", repr(f))},
                cause=exc,
            ) from exc

        if not isinstance(result, Dual):
            raise STEMDifferentiationError(
                "function must return a Dual value during AD",
                context={"received_type": type(result).__name__},
            )
        return result

    def gradient(
        self,
        f: Callable[..., Any],
        x: Sequence[float],
        method: str = "ad",
    ) -> List[float]:
        """Gradient of a scalar-valued function at ``x`` via forward-mode AD."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        variables = as_float_array(x, "x", error_cls=STEMDifferentiationError)

        if method != "ad":
            raise STEMDifferentiationError(
                "calculus.gradient only supports forward-mode AD; use numerical_methods for finite differences",
                context={"method": method},
            )

        result = self._dual_call(f, variables)
        return [float(p) for p in result.partials]

    def jacobian(
        self,
        f: Callable[..., Any],
        x: Sequence[float],
        method: str = "ad",
    ) -> List[List[float]]:
        """Jacobian of a vector-valued function at ``x`` via forward-mode AD."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        variables = as_float_array(x, "x", error_cls=STEMDifferentiationError)

        if method != "ad":
            raise STEMDifferentiationError(
                "calculus.jacobian only supports forward-mode AD; use numerical_methods for finite differences",
                context={"method": method},
            )

        dual_inputs = [
            Dual(value, tuple(1.0 if i == j else 0.0 for i in range(len(variables))))
            for j, value in enumerate(variables)
        ]

        try:
            outputs = f(*dual_inputs)
        except STEMDifferentiationError:
            raise
        except Exception as exc:
            raise STEMDifferentiationError(
                "function evaluation raised during Jacobian AD",
                context={"callable": getattr(f, "__name__", repr(f))},
                cause=exc,
            ) from exc

        if not isinstance(outputs, (list, tuple)):
            raise STEMDifferentiationError(
                "vector-valued functions must return a list or tuple of Dual values"
            )

        rows: List[List[float]] = []
        for index, output in enumerate(outputs):
            if not isinstance(output, Dual):
                raise STEMDifferentiationError(
                    f"jacobian output[{index}] must be a Dual",
                    context={"received_type": type(output).__name__},
                )
            rows.append([float(p) for p in output.partials])
        return rows

    def hessian(
        self,
        f: Callable[..., Any],
        x: Sequence[float],
        method: str = "ad",
    ) -> List[List[float]]:
        """Hessian of a scalar-valued function at ``x`` via nested AD."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        variables = as_float_array(x, "x", error_cls=STEMDifferentiationError)

        if method != "ad":
            raise STEMDifferentiationError(
                "calculus.hessian only supports forward-mode AD; use numerical_methods for finite differences",
                context={"method": method},
            )

        n = len(variables)
        hessian = [[0.0] * n for _ in range(n)]

        for i in range(n):
            grad_i = self._second_partial(f, variables, i)
            for j in range(n):
                hessian[i][j] = grad_i[j]

        # symmetrize (numerical AD is symmetric in exact arithmetic)
        for i in range(n):
            for j in range(i + 1, n):
                avg = 0.5 * (hessian[i][j] + hessian[j][i])
                hessian[i][j] = avg
                hessian[j][i] = avg

        return hessian

    def _second_partial(
        self,
        f: Callable[..., Any],
        x: Sequence[float],
        seed: int,
    ) -> List[float]:
        n = len(x)

        def grad_component(*dvals: Any) -> Dual:
            # dvals are Duals; build a Dual whose partial vector is a full
            # basis vector so we can extract second partials.

            def inner_f(*outer_dvals: Any) -> Dual:
                return f(*outer_dvals)

            # first derivative in variable ``seed`` at the current dual point
            first = self._dual_call_partial(inner_f, dvals, seed)
            return first

        # Build a full AD pass with dual partials = identity to get the first
        # gradient; then extract partials of the ``seed`` component.
        result = self._dual_call(f, x)
        if not isinstance(result, Dual):
            raise STEMDifferentiationError("function must return a Dual value")

        # For second derivatives we re-evaluate with a nested Dual trick.
        # Inner pass: partials = I (n x n). Outer pass: seed column j.
        seeds_out: List[float] = []
        for j in range(n):
            # build dual inputs whose partials are the identity
            dual_inputs = [
                Dual(value, tuple(1.0 if i == k else 0.0 for i in range(n)))
                for k, value in enumerate(x)
            ]
            # warp so that only partial ``seed`` carries a nontrivial outer dual
            warped = []
            for k, dual in enumerate(dual_inputs):
                warped.append(
                    Dual(dual, tuple(1.0 if k == j and i == seed else 0.0 for i in range(n))) # type: ignore
                )

            try:
                result = f(*warped)
            except STEMDifferentiationError:
                raise
            except Exception as exc:
                raise STEMDifferentiationError(
                    "function evaluation raised during Hessian AD",
                    cause=exc,
                ) from exc

            if not isinstance(result, Dual):
                raise STEMDifferentiationError("function must return a Dual value")

            seeds_out.append(float(result.partials[j]))

        # second partials = d/dx_j (∂f/∂x_seed) evaluated at x, from _second_partial_inner
        row: List[float] = [0.0] * n
        for j in range(n):
            row[j] = self._second_partial_value(f, x, seed, j)
        return row

    def _second_partial_value(
        self,
        f: Callable[..., Any],
        x: Sequence[float],
        i: int,
        j: int,
    ) -> float:
        """Central second-difference-free estimate using two nested duals."""
        # Build outer duals: values are Dual (inner duals), each with two
        # components representing the seed direction of variable i and j.
        outer_inputs: List[Dual] = []
        for k, value in enumerate(x):
            inner = Dual(value, tuple(1.0 if m == i else 0.0 for m in range(len(x))))
            outer_partials = tuple(1.0 if k == j and m == i else 0.0 for m in range(len(x)))
            outer_inputs.append(Dual(inner, outer_partials)) # type: ignore

        try:
            result = f(*outer_inputs)
        except STEMDifferentiationError:
            raise
        except Exception as exc:
            raise STEMDifferentiationError(
                "function evaluation raised during nested Hessian AD",
                cause=exc,
            ) from exc

        # result is a Dual whose value is a Dual; extract mixed partial
        if not isinstance(result, Dual) or not isinstance(result.value, Dual):
            raise STEMDifferentiationError("nested Hessian requires scalar-valued function")

        return float(result.value.partials[j])

    def _dual_call_partial(
        self,
        f: Callable[..., Any],
        dual_inputs: Sequence[Any],
        seed: int,
    ) -> Dual:
        try:
            result = f(*dual_inputs)
        except Exception as exc:
            raise STEMDifferentiationError(
                "internal AD helper failed",
                cause=exc,
            ) from exc
        if not isinstance(result, Dual):
            raise STEMDifferentiationError("function must return a Dual value")
        return result

    # -- derivative operators ---------------------------------------------

    def directional_derivative(self, f: Callable[..., Any], x: Sequence[float], direction: Sequence[float]) -> float:
        """Directional derivative ``∇f(x) · v`` via forward-mode AD."""
        gradient = self.gradient(f, x)
        vector = as_float_array(direction, "direction", error_cls=STEMDifferentiationError)
        if len(vector) != len(gradient):
            raise STEMDifferentiationError(
                "direction dimensionality mismatch",
                context={"gradient_dim": len(gradient), "direction_dim": len(vector)},
            )
        return float(sum(g * v for g, v in zip(gradient, vector)))

    def limit(self, f: Callable[[float], float], x: float, direction: str = "both") -> Optional[float]:
        """
        Estimate the limit of a scalar function at ``x``.

        Uses symmetric or one-sided sampling at the configured step size.
        Returns ``None`` when the sampled values disagree beyond tolerance.
        """
        validate_callable(f, "f", error_cls=STEMValidationError)
        x_f = ensure_finite_number(x, "x", error_cls=STEMCalculusError)
        if direction not in {"both", "left", "right"}:
            raise STEMCalculusError(
                "direction must be 'both', 'left', or 'right'",
                context={"direction": direction},
            )

        h = self._limit_h
        tolerance = float(self.calculus_config.get("limit_tolerance", 1e-4))

        def sample(point: float) -> float:
            try:
                return float(f(point))
            except Exception as exc:
                raise STEMCalculusError(
                    "function evaluation failed during limit computation",
                    context={"point": point},
                    cause=exc,
                ) from exc

        if direction == "left":
            return sample(x_f - h)
        if direction == "right":
            return sample(x_f + h)

        left = sample(x_f - h)
        right = sample(x_f + h)
        if abs(left - right) > tolerance * max(1.0, abs(left), abs(right)):
            return None
        return 0.5 * (left + right)

    def taylor_series(
        self,
        f: Callable[[float], float],
        x0: float,
        order: int,
        variable: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """
        Compute the Taylor polynomial coefficients of a scalar function.

        Returns a mapping with ``variable``, ``center``, ``order``,
        ``coefficients`` (list indexed by derivative order), and an
        ``evaluate`` callable for the resulting polynomial.
        """
        validate_callable(f, "f", error_cls=STEMValidationError)
        x0_f = ensure_finite_number(x0, "x0", error_cls=STEMCalculusError)
        order_i = int(ensure_positive(order, "order", allow_zero=True, error_cls=STEMCalculusError))
        var_name = variable or "x"

        coefficients: List[float] = []
        for k in range(order_i + 1):
            coefficients.append(self._nth_derivative_value(f, x0_f, k))

        def evaluate(x: float) -> float:
            x_f = float(x)
            total = 0.0
            for k, coef in enumerate(coefficients):
                total += (coef / math.factorial(k)) * (x_f - x0_f) ** k
            return total

        return {
            "variable": var_name,
            "center": x0_f,
            "order": order_i,
            "coefficients": coefficients,
            "evaluate": evaluate,
        }

    def _nth_derivative_value(self, f: Callable[[float], float], x0: float, k: int) -> float:
        if k == 0:
            try:
                return float(f(x0))
            except Exception as exc:
                raise STEMDifferentiationError(
                    "function evaluation failed in Taylor series",
                    context={"x": x0, "order": k},
                    cause=exc,
                ) from exc

        # Repeated AD: encode f as a function of a dual, then take k-th partial.
        def g(*args: Any) -> Any:
            return f(args[0])

        # For a scalar function, k-th derivative via repeated dual composition.
        value = x0
        # iterative: d^k/dx^k f(x0) via nested duals is awkward; use AD with
        # a "jet" of length k+1 using a small local polynomial carrier instead.
        return self._derivative_via_jet(f, x0, k)

    def _derivative_via_jet(self, f: Callable[[Any], Any], x0: float, k: int) -> float:
        """
        Evaluate the k-th derivative of a scalar function using a truncated
        Taylor jet (dual numbers of length k+1) evaluated via a local
        polynomial carrier that mimics dual arithmetic up to order k.
        """
        # Local carrier: list of length k+1 where entry i is the coefficient
        # of eps^i. Multiplication is truncated at order k.
        order = k

        class Jet:
            __slots__ = ("coeffs",)

            def __init__(self, coeffs: List[float]) -> None:
                self.coeffs = list(coeffs) + [0.0] * (order + 1 - len(coeffs))
                self.coeffs = self.coeffs[: order + 1]

            @staticmethod
            def constant(value: float) -> "Jet":
                return Jet([value])

            @staticmethod
            def variable(value: float) -> "Jet":
                base = [0.0] * (order + 1)
                base[0] = value
                if order >= 1:
                    base[1] = 1.0
                return Jet(base)

            def __add__(self, other: Any) -> "Jet":
                other = other if isinstance(other, Jet) else Jet.constant(float(other))
                return Jet([a + b for a, b in zip(self.coeffs, other.coeffs)])

            __radd__ = __add__

            def __neg__(self) -> "Jet":
                return Jet([-c for c in self.coeffs])

            def __sub__(self, other: Any) -> "Jet":
                other = other if isinstance(other, Jet) else Jet.constant(float(other))
                return Jet([a - b for a, b in zip(self.coeffs, other.coeffs)])

            def __rsub__(self, other: Any) -> "Jet":
                return (-self).__add__(other)

            def __mul__(self, other: Any) -> "Jet":
                other = other if isinstance(other, Jet) else Jet.constant(float(other))
                result = [0.0] * (order + 1)
                for i, a in enumerate(self.coeffs):
                    if a == 0.0:
                        continue
                    for j, b in enumerate(other.coeffs):
                        if i + j > order:
                            break
                        result[i + j] += a * b
                return Jet(result)

            __rmul__ = __mul__

            def __truediv__(self, other: Any) -> "Jet":
                other = other if isinstance(other, Jet) else Jet.constant(float(other))
                if other.coeffs[0] == 0.0:
                    raise STEMDifferentiationError("jet division by zero constant term")
                inv = [0.0] * (order + 1)
                inv[0] = 1.0 / other.coeffs[0]
                for n in range(1, order + 1):
                    s = 0.0
                    for k in range(1, n + 1):
                        s += other.coeffs[k] * inv[n - k]
                    inv[n] = -s / other.coeffs[0]
                return self.__mul__(Jet(inv))

            def __pow__(self, exponent: Any) -> "Jet":
                if not isinstance(exponent, (int, float)):
                    raise STEMDifferentiationError("jet power requires a scalar exponent")
                e = float(exponent)
                if e == 0.0:
                    return Jet.constant(1.0)
                if e == int(e) and e >= 0:
                    result = Jet.constant(1.0)
                    for _ in range(int(e)):
                        result = result * self
                    return result
                # general via exp(e * log(self))
                return (self.log() * e).exp()

            def exp(self) -> "Jet":
                c0 = self.coeffs[0]
                e0 = math.exp(c0)
                result = [0.0] * (order + 1)
                # f' = f, f(0) = e0
                result[0] = e0
                for n in range(1, order + 1):
                    s = 0.0
                    for k in range(1, n + 1):
                        s += k * self.coeffs[k] * result[n - k]
                    result[n] = s / n
                return Jet(result)

            def log(self) -> "Jet":
                c0 = self.coeffs[0]
                if c0 <= 0.0:
                    raise STEMDifferentiationError("jet log requires positive constant term")
                result = [0.0] * (order + 1)
                result[0] = math.log(c0)
                # f' = self' / self
                inv = (Jet.constant(1.0) / self).coeffs
                deriv = [0.0] * (order + 1)
                for n in range(1, order + 1):
                    deriv[n - 1] = n * self.coeffs[n]
                conv = [0.0] * (order + 1)
                for i in range(order + 1):
                    for j in range(order + 1 - i):
                        conv[i + j] += deriv[i] * inv[j]
                for n in range(1, order + 1):
                    result[n] = conv[n - 1] / n
                return Jet(result)

            def sin(self) -> "Jet":
                return _jet_sin(self, order)

            def cos(self) -> "Jet":
                return _jet_cos(self, order)

            def tan(self) -> "Jet":
                return self.sin() / self.cos()

            def sinh(self) -> "Jet":
                return (self.exp() - (-self).exp()) / 2.0

            def cosh(self) -> "Jet":
                return (self.exp() + (-self).exp()) / 2.0

            def tanh(self) -> "Jet":
                return self.sinh() / self.cosh()

            def sqrt(self) -> "Jet":
                return self ** 0.5

        def _jet_sin(z: "Jet", order_inner: int) -> "Jet":
            # sin' = cos, cos' = -sin, evaluated via recurrence on derivatives
            c0 = z.coeffs[0]
            result = [0.0] * (order_inner + 1)
            sin_c = math.sin(c0)
            cos_c = math.cos(c0)
            # Taylor of sin(z(x)) about x=0 uses chain rule; easier to recurse
            # via Faà di Bruno but for our bounded use we exploit linearity:
            # sin(z) = sin(z0 + w) = sin(z0) cos(w) + cos(z0) sin(w)
            w = Jet([z.coeffs[i] if i > 0 else 0.0 for i in range(order_inner + 1)])
            sin_w = _jet_sin_small(w, order_inner)
            cos_w = _jet_cos_small(w, order_inner)
            return sin_c * cos_w + cos_c * sin_w

        def _jet_cos(z: "Jet", order_inner: int) -> "Jet":
            c0 = z.coeffs[0]
            sin_c = math.sin(c0)
            cos_c = math.cos(c0)
            w = Jet([z.coeffs[i] if i > 0 else 0.0 for i in range(order_inner + 1)])
            sin_w = _jet_sin_small(w, order_inner)
            cos_w = _jet_cos_small(w, order_inner)
            return cos_c * cos_w - sin_c * sin_w

        def _jet_sin_small(w: "Jet", order_inner: int) -> "Jet":
            # w has zero constant term; series sin(w) = w - w^3/3! + w^5/5! - ...
            term = w
            total = w
            k = 3
            w_pow = w * w * w
            while k <= order_inner:
                sign = 1.0 if ((k - 1) // 2) % 2 == 0 else -1.0
                total = total + term * w_pow * (sign / math.factorial(k)) * math.factorial(k - 3)
                term = term * w_pow
                k += 2
            # Above shortcut is messy; fall back to direct Taylor with factorial.
            # Recompute properly:
            total = Jet.constant(0.0)
            fact = 1.0
            power = w
            sign = 1.0
            for n in range(1, order_inner + 1, 2):
                fact = math.factorial(n)
                total = total + power * (sign / fact)
                power = power * w * w
                sign = -sign
            return total

        def _jet_cos_small(w: "Jet", order_inner: int) -> "Jet":
            total = Jet.constant(1.0)
            power = w * w
            sign = -1.0
            for n in range(2, order_inner + 1, 2):
                fact = math.factorial(n)
                total = total + power * (sign / fact)
                power = power * w * w
                sign = -sign
            return total

        jet_x = Jet.variable(x0)

        def wrapped(z: "Jet") -> "Jet":
            try:
                result = f(z)
            except STEMDifferentiationError:
                raise
            except Exception as exc:
                raise STEMDifferentiationError(
                    "function evaluation failed during jet differentiation",
                    context={"x": x0, "order": k},
                    cause=exc,
                ) from exc
            if not isinstance(result, Jet):
                # scalar function of a Jet may return a plain float
                try:
                    return Jet.constant(float(result))
                except Exception as exc:
                    raise STEMDifferentiationError(
                        "jet function must return a numeric or jet value",
                        cause=exc,
                    ) from exc
            return result

        out = wrapped(jet_x)
        if k >= len(out.coeffs):
            raise STEMDifferentiationError(
                "requested derivative order exceeds jet order",
                context={"k": k, "jet_order": len(out.coeffs) - 1},
            )
        return float(out.coeffs[k]) * math.factorial(k)

    # -- autodiff ----------------------------------------------------------

    def autodiff(self, f: Callable[..., Any]) -> Callable[[Sequence[float]], Tuple[float, List[float]]]:
        """
        Return a callable that evaluates ``f`` and its gradient at a point.

        The returned callable has signature ``g(x) -> (value, gradient)``.
        """
        validate_callable(f, "f", error_cls=STEMValidationError)

        def wrapped(x: Sequence[float]) -> Tuple[float, List[float]]:
            result = self._dual_call(f, x)
            return float(result.value), [float(p) for p in result.partials]

        return wrapped


__all__ = ["Calculus", "Dual"]