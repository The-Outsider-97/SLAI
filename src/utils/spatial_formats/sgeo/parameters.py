"""Strict, CPU-only expression evaluation and dependency-graph resolution for parametric CAD dimensions.

This is deliberately NOT Python evaluation. It accepts an allowlist of AST nodes, evaluates a bounded dependency DAG and rejects
incompatible dimensions. Internal values are SI: lengths in metres, angles in radians, masses in kilograms, times in seconds,
temperatures in kelvin (``common.units`` converts at the boundary).

Typed parameters
    Numeric kinds carry a physical dimension (length, area, volume, angle, mass, time, velocity, acceleration, force, pressure,
    energy, power, density, second moment of area, frequency, angular velocity, absolute temperature and temperature difference)
    plus plain ``number``/``integer``/``boolean``. Data kinds (``string``, ``enumeration``, ``material_ref``, ``element_ref``,
    ``geometry_ref``, ``vector``, ``coordinate``, ``transform``) are validated values, never operands of arithmetic.
    Absolute temperatures are affine: ``T1 - T2`` is a difference, ``T + dT`` is a temperature, ``T1 + T2`` and ``2 * T`` are errors.

Dependency graph
    :class:`ParameterGraph` exposes direct/transitive dependencies and dependents, a deterministic topological order, cycle paths and an
    incremental evaluator that recomputes only the changed parameters and their descendants. :class:`ParameterEvaluator` adds a
    fingerprint-keyed cache on top of it.

Numerics
    ``ceil``/``floor``/``//`` use a relative tolerance of 1e-9 so that exact quotients are not perturbed by representation error
    (``floor(0.3 / 0.1)`` is 3, not 2); ``round`` rounds half away from zero.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import heapq
import json
import math
import io
import re
import threading
import tokenize
from collections import OrderedDict
from dataclasses import dataclass

from ..common.helpers import finite
from ..common.errors import ValidationError, ResourceLimitError
from ..common import units as U
from ..common.ids import require_id
from ..common.transforms import require_transform

MAX_EXPR_LENGTH = 512
MAX_AST_NODES = 128
MAX_EXPRESSION_DEPTH = 32
MAX_PARAMETERS = 128
MAX_CHOICES = 64
MAX_STRING_VALUE = 1024
_REL_TOL = 1e-9

# Dimension exponents: (length, angle, mass, time, temperature). Two-element tuples (length, angle) are still accepted and padded.
NDIM = 5
ZERO = (0, 0, 0, 0, 0)
_DIMENSIONS = {
    "number": ZERO, "integer": ZERO, "boolean": ZERO,
    "length": (1, 0, 0, 0, 0), "area": (2, 0, 0, 0, 0), "volume": (3, 0, 0, 0, 0), "angle": (0, 1, 0, 0, 0),
    "mass": (0, 0, 1, 0, 0), "time": (0, 0, 0, 1, 0), "frequency": (0, 0, 0, -1, 0), "angular_velocity": (0, 1, 0, -1, 0),
    "velocity": (1, 0, 0, -1, 0), "acceleration": (1, 0, 0, -2, 0), "force": (1, 0, 1, -2, 0),
    "pressure": (-1, 0, 1, -2, 0), "energy": (2, 0, 1, -2, 0), "power": (2, 0, 1, -3, 0), "density": (-3, 0, 1, 0, 0),
    "second_moment": (4, 0, 0, 0, 0), "temperature": (0, 0, 0, 0, 1), "temperature_difference": (0, 0, 0, 0, 1),
}
PARAMETER_DIMENSIONS = _DIMENSIONS
#: Types that existed before the typed-parameter extension (a document using only these needs no extra capability).
LEGACY_PARAMETER_TYPES = frozenset({"number", "integer", "boolean", "length", "area", "volume", "angle"})
DATA_TYPES = frozenset({"string", "enumeration", "material_ref", "element_ref", "geometry_ref", "vector", "coordinate", "transform"})
PARAMETER_TYPES = frozenset(_DIMENSIONS) | DATA_TYPES
_NONNEGATIVE = frozenset({"length", "area", "volume", "mass", "density", "second_moment", "temperature"})
SCOPES = ("type", "instance")
_SPEC_FIELDS = {"type", "default", "min", "max", "expression", "read_only", "description", "scope", "choices", "size", "dimension"}

CONSTANTS = {"pi": math.pi, "tau": math.tau}
CONSTRUCTORS = {name: (dim, name == "temperature") for name, dim in _DIMENSIONS.items()
                if name not in ("number", "integer", "boolean", "frequency", "angular_velocity")}
FUNCTIONS = {
    "abs": ("abs(x)", "Absolute value; keeps the unit."),
    "min": ("min(a, b, ...)", "Smallest argument; all arguments need the same unit."),
    "max": ("max(a, b, ...)", "Largest argument; all arguments need the same unit."),
    "clamp": ("clamp(x, low, high)", "x limited to [low, high]; same unit for all three."),
    "sqrt": ("sqrt(x)", "Square root of a non-negative value; the unit's exponents must be even (area -> length)."),
    "sin": ("sin(a)", "Sine of an angle."), "cos": ("cos(a)", "Cosine of an angle."), "tan": ("tan(a)", "Tangent of an angle."),
    "asin": ("asin(x)", "Arc sine of a number in [-1, 1]; returns an angle."),
    "acos": ("acos(x)", "Arc cosine of a number in [-1, 1]; returns an angle."),
    "atan": ("atan(x)", "Arc tangent of a number; returns an angle."),
    "atan2": ("atan2(y, x)", "Angle of the point (x, y); y and x need the same unit and must not both be zero."),
    "exp": ("exp(x)", "e to the power of a dimensionless x."),
    "ln": ("ln(x)", "Natural logarithm of a positive dimensionless number."),
    "log10": ("log10(x)", "Base-10 logarithm of a positive dimensionless number."),
    "round": ("round(x) or round(x, step)", "Round half away from zero to a whole number, or to a multiple of step."),
    "roundup": ("roundup(x) or roundup(x, step)", "Round up (toward +infinity) to a whole number or a multiple of step."),
    "rounddown": ("rounddown(x) or rounddown(x, step)", "Round down (toward -infinity) to a whole number or a multiple of step."),
    "ceil": ("ceil(x)", "Same as roundup(x)."), "floor": ("floor(x)", "Same as rounddown(x)."),
    "mod": ("mod(a, b)", "Remainder of a / b with the sign of b; same unit, b not zero."),
    "if": ("if(condition, a, b)", "a when the Boolean condition is true, else b. Both branches must have the same unit (checked even for the branch not taken); only the chosen branch supplies the value."),
    "log": ("log(x)", "Base-10 logarithm of a positive dimensionless number, as in Revit formulas (ln is the natural logarithm)."),
    "and": ("and(a, b, ...)", "True when every Boolean argument is true (also written a and b)."),
    "or": ("or(a, b, ...)", "True when any Boolean argument is true (also written a or b)."),
}
for _n, (_d, _a) in CONSTRUCTORS.items():
    FUNCTIONS[_n] = ("%s(x)" % _n, "Gives the plain number x the unit '%s' (SI)%s." % (_n, "; absolute reading in kelvin" if _a else ""))
_ALLOWED_CALLS = frozenset(FUNCTIONS) | {"choose"}
# A parameter may share a name with a function (``area`` and ``area(x)`` live in different namespaces) and shadows a constant of the same name.
RESERVED_NAMES = frozenset({"True", "False", "None", "and", "or", "not", "if"})


def function_catalog() -> list:
    """Functions and constants for formula-editor autocomplete: [{name, kind, signature, description}], sorted by name."""
    rows = [{"name": n, "kind": "function", "signature": s, "description": d} for n, (s, d) in FUNCTIONS.items()]
    rows += [{"name": n, "kind": "constant", "signature": n, "description": "The constant %r." % v} for n, v in CONSTANTS.items()]
    return sorted(rows, key=lambda r: r["name"])


# ------------------------------------------------------------------------------------------------------------------ quantities
@dataclass(frozen=True)
class Quantity:
    value: float | bool
    dimension: tuple = ZERO
    absolute: bool = False          # absolute temperature (affine); everything else is linear

    def __post_init__(self):
        d = tuple(self.dimension)
        if len(d) < NDIM:
            d = d + (0,) * (NDIM - len(d))
        if len(d) != NDIM:
            raise ValidationError("invalid dimension vector")
        object.__setattr__(self, "dimension", d)

    def as_number(self) -> float:
        if isinstance(self.value, bool):
            raise ValidationError("boolean cannot be used in arithmetic")
        if not math.isfinite(self.value) or abs(self.value) > 1e12:
            raise ValidationError("nonfinite or unbounded parameter result")
        return float(self.value)


_NAMES = {}
for _t, _d in _DIMENSIONS.items():
    if _t not in ("integer", "boolean", "temperature_difference"):
        _NAMES.setdefault(_d, _t)
_NAMES[(0, 0, 0, 0, 1)] = "temperature_difference"


def dimension_name(dimension, absolute: bool = False) -> str:
    """A readable name for a dimension vector (``length``, ``pressure``, ``length^2*time^-1``...)."""
    d = tuple(dimension) + (0,) * (NDIM - len(dimension))
    if absolute:
        return "temperature"
    if d in _NAMES:
        return "dimensionless" if d == ZERO else _NAMES[d]
    labels = ("length", "angle", "mass", "time", "temperature")
    return "*".join(l if e == 1 else "%s^%d" % (l, e) for l, e in zip(labels, d) if e) or "dimensionless"


def _quantity(value, dimension=ZERO, absolute=False):
    if isinstance(value, bool):
        if tuple(dimension) != ZERO[:len(tuple(dimension))] and any(dimension):
            raise ValidationError("boolean requires dimensionless type")
        return Quantity(value)
    if not isinstance(value, (int, float)) or abs(value) > 1e12 or not math.isfinite(value):
        raise ValidationError("expression result exceeds numeric limits")
    return Quantity(float(value), dimension, absolute)


def _boolean(q: Quantity) -> bool:
    if not isinstance(q.value, bool):
        raise ValidationError("conditional requires a Boolean expression")
    return q.value


class _DimensionError(ValidationError):
    """A unit mismatch: reported wherever it occurs, including in a branch that ``if`` does not take."""


def _mismatch(a: Quantity, b: Quantity) -> ValidationError:
    return _DimensionError("incompatible physical dimensions: %s vs %s" % (dimension_name(a.dimension, a.absolute),
                                                                          dimension_name(b.dimension, b.absolute)),
                           context={"expected": dimension_name(a.dimension, a.absolute), "actual": dimension_name(b.dimension, b.absolute)})


def _compatible(a: Quantity, b: Quantity) -> None:
    if a.dimension != b.dimension or a.absolute != b.absolute:
        raise _mismatch(a, b)
    a.as_number()
    b.as_number()


def _tol_floor(x: float) -> int:
    return math.floor(x + _REL_TOL * max(1.0, abs(x)))


def _tol_ceil(x: float) -> int:
    return math.ceil(x - _REL_TOL * max(1.0, abs(x)))


def _round_half_away(x: float) -> int:
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def _no_affine(q: Quantity, what: str) -> None:
    if q.absolute:
        raise ValidationError("an absolute temperature cannot be used in %s; use a temperature difference" % what)


# ------------------------------------------------------------------------------------------------------------------- parsing
# ------------------------------------------------------------------------------------------------------- Revit-style syntax
# Formulas may be written the way Revit family formulas are: ``^`` for powers, a single ``=`` for equality, ``and(a, b)`` / ``or(a, b)`` as functions,
# ``log`` for the base-10 logarithm and unit-suffixed literals (``800 mm``, ``90 deg``). Everything is rewritten to the checked subset below before it is parsed,
# so the dimension rules are the same for every spelling; a unit literal is an ordinary dimensioned constant, never a silent conversion of a plain number.
_UNIT_LITERALS: dict[str, tuple] = {}
for _kind, _units in U.QUANTITY_UNITS.items():
    if _kind in ("number", "integer", "temperature", "temperature_difference") or _kind not in CONSTRUCTORS:
        continue
    for _symbol, _factor in _units.items():
        if _symbol.isidentifier() and not _symbol.startswith("_"):
            _UNIT_LITERALS.setdefault(_symbol, (_kind, _factor))
_BOOL_FUNCTIONS = {"and": ast.And, "or": ast.Or}
_OPERAND_END = (tokenize.NUMBER, tokenize.NAME)


def _revit_syntax(expression: str) -> str:
    """Rewrite Revit spellings to the checked subset; text that is already in that subset comes back unchanged in meaning."""
    try:
        tokens = [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(expression).readline)
                  if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.COMMENT)]
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ValidationError("invalid parameter expression syntax") from exc
    out: list = []
    i = 0
    while i < len(tokens):
        kind, text = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else (None, "")
        prev = out[-1] if out else (None, "")
        if kind == tokenize.OP and text == "^":
            out.append((tokenize.OP, "**"))
        elif kind == tokenize.OP and text == "=":
            out.append((tokenize.OP, "=="))
        elif kind == tokenize.NAME and text in _BOOL_FUNCTIONS and nxt == (tokenize.OP, "(") and not (
                prev[0] == tokenize.NUMBER or (prev[0] == tokenize.NAME and prev[1] not in ("and", "or", "not", "if", "in")) or prev == (tokenize.OP, ")")):
            out.append((tokenize.NAME, "__%s__" % text))
        elif kind == tokenize.NUMBER and nxt[0] == tokenize.NAME and nxt[1] in _UNIT_LITERALS and not (
                i + 2 < len(tokens) and tokens[i + 2] == (tokenize.OP, "(")):
            ptype, factor = _UNIT_LITERALS[nxt[1]]
            try:
                value = float(ast.literal_eval(text)) * factor
            except (ValueError, SyntaxError) as exc:
                raise ValidationError("invalid number before a unit") from exc
            out.extend([(tokenize.NAME, ptype), (tokenize.OP, "("), (tokenize.NUMBER, repr(value)), (tokenize.OP, ")")])
            i += 1
        else:
            out.append((kind, text))
        i += 1
    return " ".join(text for _, text in out)


class _CallForms(ast.NodeTransformer):
    """``__and__(a, b)`` / ``__or__(a, b)`` become Boolean operations and ``log`` becomes ``log10``; everything else is left to the allowlist."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in ("__and__", "__or__"):
                if node.keywords or len(node.args) < 2:
                    raise ValidationError("%s needs at least two arguments" % node.func.id.strip("_"))
                return ast.copy_location(ast.BoolOp(ast.And() if node.func.id == "__and__" else ast.Or(), node.args), node)
            if node.func.id == "log":
                node.func.id = "log10"
        return node


def _checked_expression(expression: str) -> ast.Expression:
    if not isinstance(expression, str) or not 0 < len(expression) <= MAX_EXPR_LENGTH:
        raise ValidationError("invalid expression length")
    # 'if(a,b,c)' is conventional in formulas but 'if' is reserved in Python.
    source = re.sub(r"\bif\s*\(", "choose(", expression)
    source = _revit_syntax(source)
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValidationError("invalid parameter expression syntax") from exc
    tree = ast.fix_missing_locations(_CallForms().visit(tree))
    count = 0
    stack = [(tree, 0)]
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp,
               ast.Compare, ast.BoolOp, ast.IfExp, ast.Call, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod, ast.UAdd, ast.USub, ast.Eq,
               ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.And,
               ast.Or, ast.Not, ast.Load)
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_EXPRESSION_DEPTH:
            raise ResourceLimitError("parameter expression complexity exceeded")
        if not isinstance(node, allowed):
            raise ValidationError("forbidden AST node: " + type(node).__name__)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords or node.func.id not in _ALLOWED_CALLS:
                raise ValidationError("function is not allowlisted")
        if isinstance(node, ast.Constant) and (isinstance(node.value, str) or node.value is None or isinstance(node.value, (bytes, complex))):
            raise ValidationError("only numeric/boolean literals are supported")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


def _names_read(expression: str) -> frozenset[str]:
    tree = _checked_expression(expression)
    functions = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return frozenset(n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in functions)


def expression_dependencies(expression: str) -> frozenset[str]:
    """The names a formula reads (function names are not dependencies; neither are the constants ``pi`` and ``tau``)."""
    return frozenset(n for n in _names_read(expression) if n not in CONSTANTS)


# ------------------------------------------------------------------------------------------------------------------ evaluation
def evaluate_expression(expression: str, variables: dict[str, Quantity]) -> Quantity:
    tree = _checked_expression(expression)

    def calc(n):
        if isinstance(n, ast.Constant):
            return _quantity(n.value)
        if isinstance(n, ast.Name):
            if n.id in variables:
                return variables[n.id]
            if n.id in CONSTANTS:
                return Quantity(CONSTANTS[n.id])
            raise ValidationError("unknown parameter: " + n.id, context={"unknown": n.id})
        if isinstance(n, ast.Expression):
            return calc(n.body)
        if isinstance(n, ast.IfExp):
            return calc(n.body if _boolean(calc(n.test)) else n.orelse)
        if isinstance(n, ast.UnaryOp):
            v = calc(n.operand)
            if isinstance(n.op, ast.Not):
                return Quantity(not _boolean(v))
            _no_affine(v, "a sign change")
            return _quantity(-v.as_number() if isinstance(n.op, ast.USub) else v.as_number(), v.dimension)
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                for v in n.values:
                    if not _boolean(calc(v)):
                        return Quantity(False)
                return Quantity(True)
            for v in n.values:
                if _boolean(calc(v)):
                    return Quantity(True)
            return Quantity(False)
        if isinstance(n, ast.Compare):
            left = calc(n.left)
            for op, rhs in zip(n.ops, n.comparators):
                right = calc(rhs)
                if isinstance(left.value, bool) or isinstance(right.value, bool):
                    if not (isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(left.value, bool) and isinstance(right.value, bool)):
                        raise ValidationError("Booleans can only be compared for equality with Booleans")
                    good = (left.value == right.value) == isinstance(op, ast.Eq)
                else:
                    _compatible(left, right)
                    a, b = left.as_number(), right.as_number()
                    tol = _REL_TOL * max(1.0, abs(a), abs(b))      # values that differ only by rounding noise compare equal (0.1 + 0.2 = 0.3)
                    good = ((isinstance(op, ast.Lt) and a < b - tol) or (isinstance(op, ast.LtE) and a <= b + tol)
                            or (isinstance(op, ast.Gt) and a > b + tol) or (isinstance(op, ast.GtE) and a >= b - tol)
                            or (isinstance(op, ast.Eq) and abs(a - b) <= tol) or (isinstance(op, ast.NotEq) and abs(a - b) > tol))
                if not good:
                    return Quantity(False)
                left = right
            return Quantity(True)
        if isinstance(n, ast.BinOp):
            a, b = calc(n.left), calc(n.right)
            if isinstance(n.op, (ast.Add, ast.Sub)):
                sub = isinstance(n.op, ast.Sub)
                if a.dimension != b.dimension:
                    raise _mismatch(a, b)
                x, y = a.as_number(), b.as_number()
                if a.absolute or b.absolute:                                           # affine temperature rules
                    if a.absolute and b.absolute:
                        if not sub:
                            raise ValidationError("two absolute temperatures cannot be added; add a temperature difference")
                        return _quantity(x - y, a.dimension, False)
                    if b.absolute:
                        raise ValidationError("an absolute temperature cannot be subtracted from or added to a difference this way"
                                              if sub else "a temperature difference plus an absolute temperature: write absolute + difference")
                    return _quantity(x + (-y if sub else y), a.dimension, True)
                return _quantity(x + (-y if sub else y), a.dimension)
            if isinstance(n.op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                _no_affine(a, "a product, quotient, remainder or power")
                _no_affine(b, "a product, quotient, remainder or power")
            if isinstance(n.op, ast.Mult):
                return _quantity(a.as_number() * b.as_number(), tuple(x + y for x, y in zip(a.dimension, b.dimension)))
            if isinstance(n.op, ast.Div):
                denominator = b.as_number()
                if denominator == 0:
                    raise ValidationError("division by zero")
                return _quantity(a.as_number() / denominator, tuple(x - y for x, y in zip(a.dimension, b.dimension)))
            if isinstance(n.op, (ast.FloorDiv, ast.Mod)):
                _compatible(a, b)
                if b.as_number() == 0:
                    raise ValidationError("division by zero")
                q = _tol_floor(a.as_number() / b.as_number())
                if isinstance(n.op, ast.FloorDiv):
                    return _quantity(float(q))
                return _quantity(a.as_number() - b.as_number() * q, a.dimension)
            if isinstance(n.op, ast.Pow):
                if b.dimension != ZERO:
                    raise ValidationError("exponent must be dimensionless")
                exponent = b.as_number()
                if not exponent.is_integer() or not -4 <= exponent <= 4:
                    raise ValidationError("exponents must be bounded integers")
                if a.as_number() == 0 and exponent < 0:
                    raise ValidationError("zero to negative power")
                return _quantity(a.as_number() ** int(exponent), tuple(x * int(exponent) for x in a.dimension))
        if isinstance(n, ast.Call):
            return call(n)
        raise ValidationError("unsupported expression construct")

    def call(n):
        name = n.func.id
        if name == "choose":
            if len(n.args) != 3:
                raise ValidationError("if requires three arguments")
            taken = _boolean(calc(n.args[0]))
            chosen = calc(n.args[1] if taken else n.args[2])
            # The branch that is not taken must still have the same unit, otherwise the result's type would depend on the data. A domain error in it (the
            # division an ``if`` exists to guard) is not a type error and is ignored; a unit mismatch inside it is reported.
            try:
                other = calc(n.args[2] if taken else n.args[1])
            except _DimensionError:
                raise
            except (ValidationError, ResourceLimitError, ArithmeticError):
                other = None
            if other is not None:
                if isinstance(other.value, bool) != isinstance(chosen.value, bool):
                    raise ValidationError("the branches of if must both be Boolean or both be numbers")
                if other.dimension != chosen.dimension or other.absolute != chosen.absolute:
                    raise _mismatch(chosen, other)
            return chosen
        args = [calc(arg) for arg in n.args]
        if name in {"min", "max", "clamp"}:
            if len(args) < 1 or (name == "clamp" and len(args) != 3):
                raise ValidationError("invalid function arity")
            if name == "clamp":
                for arg in args:
                    _no_affine(arg, name)
            for arg in args[1:]:
                _compatible(args[0], arg)
            args[0].as_number()
            if name == "clamp":
                if args[1].as_number() > args[2].as_number():
                    raise ValidationError("clamp minimum exceeds maximum")
                value = max(args[1].as_number(), min(args[0].as_number(), args[2].as_number()))
            else:
                value = (min if name == "min" else max)(x.as_number() for x in args)
            return _quantity(value, args[0].dimension, args[0].absolute)
        if name in ("round", "roundup", "rounddown", "ceil", "floor"):
            if len(args) not in (1, 2) or (len(args) == 2 and name in ("ceil", "floor")):
                raise ValidationError("invalid function arity")
            x = args[0]
            _no_affine(x, name)
            quotient = x.as_number()
            if len(args) == 2:
                step = args[1]
                _compatible(x, step)
                if step.as_number() <= 0:
                    raise ValidationError("rounding step must be positive")
                quotient = quotient / step.as_number()
            rounder = _round_half_away if name == "round" else _tol_ceil if name in ("roundup", "ceil") else _tol_floor
            result = rounder(quotient)
            return _quantity(float(result) * (args[1].as_number() if len(args) == 2 else 1.0), x.dimension)
        if name == "mod":
            if len(args) != 2:
                raise ValidationError("invalid function arity")
            a, b = args
            _no_affine(a, "mod")
            _compatible(a, b)
            if b.as_number() == 0:
                raise ValidationError("division by zero")
            return _quantity(a.as_number() - b.as_number() * _tol_floor(a.as_number() / b.as_number()), a.dimension)
        if name == "atan2":
            if len(args) != 2:
                raise ValidationError("invalid function arity")
            y, x = args
            _compatible(y, x)
            if y.as_number() == 0 and x.as_number() == 0:
                raise ValidationError("atan2 is undefined for (0, 0)")
            return _quantity(math.atan2(y.as_number(), x.as_number()), (0, 1, 0, 0, 0))
        if len(args) != 1:
            raise ValidationError("invalid function arity")
        arg = args[0]
        if name in CONSTRUCTORS:
            if arg.dimension != ZERO or arg.absolute or isinstance(arg.value, bool):
                raise ValidationError("unit constructor requires dimensionless scalar")
            dim, absolute = CONSTRUCTORS[name]
            return _quantity(arg.as_number(), dim, absolute)
        _no_affine(arg, name)
        x = arg.as_number()
        if name == "abs":
            return _quantity(abs(x), arg.dimension)
        if name == "sqrt":
            if any(e % 2 for e in arg.dimension) or x < 0:
                raise ValidationError("sqrt requires even dimensions and nonnegative input")
            return _quantity(math.sqrt(x), tuple(e // 2 for e in arg.dimension))
        if name in ("sin", "cos", "tan"):
            if arg.dimension != _DIMENSIONS["angle"]:
                raise ValidationError("trigonometry requires an angle (radians)")
            if name == "tan" and abs(math.cos(x)) < 1e-12:
                raise ValidationError("tan is undefined at odd multiples of 90 degrees")
            return _quantity(getattr(math, name)(x))
        if name in ("asin", "acos", "atan", "exp", "ln", "log10"):
            if arg.dimension != ZERO:
                raise ValidationError("%s requires a dimensionless number" % name)
            if name in ("asin", "acos") and not -1.0 <= x <= 1.0:
                raise ValidationError("%s requires an argument in [-1, 1]" % name)
            if name in ("ln", "log10") and x <= 0:
                raise ValidationError("%s requires a positive argument" % name)
            if name == "exp" and x > 50:
                raise ValidationError("exp argument too large")
            fn = {"asin": math.asin, "acos": math.acos, "atan": math.atan, "exp": math.exp, "ln": math.log, "log10": math.log10}[name]
            return _quantity(fn(x), _DIMENSIONS["angle"] if name in ("asin", "acos", "atan") else ZERO)
        raise ValidationError("unsupported expression construct")

    try:
        return calc(tree)
    except (OverflowError, ZeroDivisionError, ValueError) as exc:
        if isinstance(exc, (ValidationError, ResourceLimitError)):
            raise
        raise ValidationError("expression numerical failure") from exc


# ----------------------------------------------------------------------------------------------------------------- the graph
def _located(exc, name, expression):
    """The same failure, naming the parameter (and formula) it happened in."""
    base = exc.args[0] if exc.args else str(exc)
    where = "parameter '%s'" % name + (" = %s" % expression if expression else "")
    context = dict(getattr(exc, "context", None) or {})
    context.update({"parameter": name, "expression": expression, "reason": base})
    return type(exc)("%s (%s)" % (base, where), context=context)


def _spec_error(name, message, **context):
    return ValidationError("%s (parameter '%s')" % (message, name), context={"parameter": name, "reason": message, **context})


def _canonical(spec) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)


class ParameterGraph:
    """A validated parameter table with its dependency graph.

    ``dependencies[n]`` are the parameters ``n``'s formula reads; ``dependents[n]`` the ones that read ``n``. Construction validates
    the table (names, field set, types, formula syntax, unknown or non-numeric references) and rejects cycles with the path.
    """

    def __init__(self, specifications: dict):
        if not isinstance(specifications, dict) or len(specifications) > MAX_PARAMETERS:
            raise ValidationError("parameter table exceeds bounds")
        self.specs = specifications
        deps: dict[str, frozenset] = {}
        for name, spec in specifications.items():
            self._check_spec(name, spec)
            if "expression" in spec:
                try:
                    deps[name] = frozenset(n for n in _names_read(spec["expression"]) if n in specifications or n not in CONSTANTS)
                except (ValidationError, ResourceLimitError) as exc:
                    raise _located(exc, name, spec["expression"]) from exc
                unknown = sorted(deps[name] - specifications.keys())
                if unknown:
                    raise _spec_error(name, "unknown expression dependencies: " + ", ".join(unknown), unknown=unknown, expression=spec["expression"])
                bad = sorted(d for d in deps[name] if specifications[d]["type"] in DATA_TYPES)
                if bad:
                    raise _spec_error(name, "a formula cannot read %s (not a numeric or Boolean parameter)" % ", ".join(bad), operands=bad,
                                      expression=spec["expression"])
            else:
                deps[name] = frozenset()
        self.dependencies = deps
        dependents: dict[str, set] = {n: set() for n in specifications}
        for n, ds in deps.items():
            for d in ds:
                dependents[d].add(n)
        self.dependents = {n: frozenset(s) for n, s in dependents.items()}
        self.order = self._topological_order()

    # -- validation -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_spec(name, spec):
        if not isinstance(name, str) or not name.isidentifier() or len(name) > 64:
            raise ValidationError("invalid parameter name")
        if name in RESERVED_NAMES:
            raise _spec_error(name, "the name is reserved for a function or constant")
        if not isinstance(spec, dict) or not isinstance(spec.get("type"), str) or spec["type"] not in PARAMETER_TYPES:
            raise ValidationError("invalid parameter definition", context={"parameter": name})
        if set(spec) - _SPEC_FIELDS:
            raise ValidationError("unknown parameter definition field", context={"parameter": name, "fields": sorted(set(spec) - _SPEC_FIELDS)})
        typ = spec["type"]
        if type(spec.get("read_only", False)) is not bool:
            raise ValidationError("read_only must be Boolean")
        if not isinstance(spec.get("description", ""), str) or len(spec.get("description", "")) > 4096:
            raise ValidationError("invalid parameter description")
        if spec.get("scope", "instance") not in SCOPES:
            raise _spec_error(name, "scope must be one of " + ", ".join(SCOPES))
        if "expression" in spec and "default" in spec:
            raise ValidationError("derived parameter cannot also declare a default", context={"parameter": name})
        if "expression" in spec and typ in DATA_TYPES:
            raise _spec_error(name, "a %s parameter cannot have a formula" % typ)
        if typ == "boolean":
            if "min" in spec or "max" in spec:
                raise ValidationError("Boolean parameters cannot have numeric bounds")
        elif typ in DATA_TYPES:
            if "min" in spec or "max" in spec:
                raise _spec_error(name, "%s parameters cannot have numeric bounds" % typ)
        else:
            low = finite(spec.get("min", 0 if typ in _NONNEGATIVE else -1e12))
            high = finite(spec.get("max", 1e12))
            if low > high or (typ in _NONNEGATIVE and low < 0):
                raise ValidationError("invalid parameter bounds", context={"parameter": name})
        if typ == "enumeration":
            choices = spec.get("choices")
            if (not isinstance(choices, list) or not 1 <= len(choices) <= MAX_CHOICES or len(set(choices)) != len(choices)
                    or not all(isinstance(c, str) and 0 < len(c) <= 64 for c in choices)):
                raise _spec_error(name, "an enumeration needs 1-%d distinct text choices" % MAX_CHOICES)
        elif "choices" in spec:
            raise _spec_error(name, "only an enumeration has choices")
        if typ in ("vector", "coordinate"):
            size = spec.get("size", 3)
            if type(size) is not int or not 2 <= size <= 16 or (typ == "coordinate" and size != 3):
                raise _spec_error(name, "invalid vector size")
            dim = spec.get("dimension", "length" if typ == "coordinate" else "number")
            if dim not in _DIMENSIONS or dim in ("boolean",):
                raise _spec_error(name, "invalid vector dimension")
        elif "size" in spec or "dimension" in spec:
            raise _spec_error(name, "size and dimension apply to vectors only")

    def _topological_order(self) -> tuple:
        pending = {n: len(ds) for n, ds in self.dependencies.items()}
        ready = [n for n, c in pending.items() if c == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            n = heapq.heappop(ready)
            order.append(n)
            for m in self.dependents[n]:
                pending[m] -= 1
                if pending[m] == 0:
                    heapq.heappush(ready, m)
        if len(order) != len(self.specs):
            cycle = self._find_cycle(set(self.specs) - set(order))
            raise ValidationError("cyclic parameter dependencies: " + " -> ".join(cycle), context={"cycle": cycle, "parameter": cycle[0]})
        return tuple(order)

    def _find_cycle(self, remaining: set) -> list:
        node = min(remaining)
        seen = {}
        path = []
        while node not in seen:
            seen[node] = len(path)
            path.append(node)
            node = min(d for d in self.dependencies[node] if d in remaining)
        return path[seen[node]:] + [node]

    # -- queries --------------------------------------------------------------------------------------------------------
    def transitive_dependencies(self, names) -> frozenset:
        return self._closure(names, self.dependencies)

    def transitive_dependents(self, names) -> frozenset:
        """Every parameter that (directly or indirectly) reads any of ``names`` (the names themselves excluded)."""
        return self._closure(names, self.dependents)

    @staticmethod
    def _closure(names, edges) -> frozenset:
        seen, stack = set(), list(names)
        while stack:
            for m in edges.get(stack.pop(), ()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return frozenset(seen)

    def impact(self, changed) -> tuple:
        """The changed parameters and everything downstream, in evaluation order (the dirty set of an edit)."""
        changed = set(changed)
        unknown = changed - self.specs.keys()
        if unknown:
            raise ValidationError("unknown parameter: " + ", ".join(sorted(unknown)), context={"unknown": sorted(unknown)})
        dirty = changed | self.transitive_dependents(changed)
        return tuple(n for n in self.order if n in dirty)

    def edges(self) -> list:
        """(reader, read) pairs: ``(a, b)`` means a's formula reads b. Sorted, for display and tests."""
        return sorted((n, d) for n, ds in self.dependencies.items() for d in ds)

    def fingerprint(self) -> str:
        """A hash of the canonical table; a cache is valid only while it is unchanged."""
        return hashlib.sha256(_canonical(self.specs).encode()).hexdigest()

    # -- evaluation -----------------------------------------------------------------------------------------------------
    def _check_overrides(self, overrides):
        if not isinstance(overrides, dict) or set(overrides) - set(self.specs):
            raise ValidationError("invalid parameter overrides", context={"unknown": sorted(set(overrides) - set(self.specs))} if isinstance(overrides, dict) else None)
        for name in overrides:
            spec = self.specs[name]
            if "expression" in spec:
                raise _spec_error(name, "derived parameters cannot be overridden")
            if spec.get("read_only", False):
                raise _spec_error(name, "read-only parameter override")

    def evaluate(self, overrides: dict | None = None, *, previous: dict | None = None, changed=None) -> dict:
        """Resolve every parameter, transactionally (a failure returns nothing).

        With ``previous`` (a prior full result for the same table) and ``changed`` (names whose override or definition changed) only
        the changed parameters and their dependents are recomputed; the rest are reused. Callers that pass a ``previous`` from a
        different table get wrong answers: use :class:`ParameterEvaluator`, which keys the reuse on :meth:`fingerprint`.
        """
        return self.evaluate_with_report(overrides, previous=previous, changed=changed)[0]

    def evaluate_with_report(self, overrides: dict | None = None, *, previous: dict | None = None, changed=None):
        """``(values, recomputed)``: the resolved values and the names that were actually recomputed, in order."""
        overrides = {} if overrides is None else overrides
        self._check_overrides(overrides)
        if previous is not None and changed is not None and set(previous) == set(self.specs):
            todo = self.impact(changed)
        else:
            previous, todo = None, self.order
        todo_set = set(todo)
        values: dict = {}
        for name in self.order:
            spec = self.specs[name]
            if name not in todo_set:
                values[name] = self._requantify(spec, previous[name])
                continue
            try:
                values[name] = self._resolve(name, spec, overrides, values)
            except (ValidationError, ResourceLimitError) as exc:
                if (exc.context or {}).get("parameter") == name:
                    raise
                raise _located(exc, name, spec.get("expression")) from exc
        return {n: self._plain(self.specs[n], v) for n, v in values.items()}, tuple(todo)

    @staticmethod
    def _requantify(spec, value):
        typ = spec["type"]
        if typ in DATA_TYPES:
            return value
        if typ == "boolean":
            return Quantity(value)
        return Quantity(value, _DIMENSIONS[typ], typ == "temperature")

    @staticmethod
    def _plain(spec, value):
        return value if spec["type"] in DATA_TYPES else value.value

    def _resolve(self, name, spec, overrides, values):
        typ = spec["type"]
        if typ in DATA_TYPES:
            return _data_value(name, spec, overrides.get(name, spec.get("default")))
        if "expression" in spec:
            q = evaluate_expression(spec["expression"], {k: v for k, v in values.items() if not isinstance(v, (list, str, type(None)))})
            raw = q.value
        else:
            raw = overrides.get(name, spec.get("default"))
            q = _quantity(raw, _DIMENSIONS[typ], typ == "temperature") if typ != "boolean" else Quantity(raw)
        if typ == "boolean":
            if not isinstance(raw, bool):
                raise ValidationError("Boolean parameter requires true/false")
            return Quantity(raw)
        if isinstance(raw, bool) and typ != "number":
            raise ValidationError("a Boolean cannot be used as a %s" % typ)
        if typ == "integer" and (not isinstance(raw, (int, float)) or isinstance(raw, bool) or not float(raw).is_integer()):
            raise ValidationError("integer parameter requires integral value")
        if q.dimension != _DIMENSIONS[typ] or q.absolute != (typ == "temperature"):
            raise ValidationError("parameter dimension mismatch: %s (expected %s, got %s)" % (
                name, dimension_name(_DIMENSIONS[typ], typ == "temperature"), dimension_name(q.dimension, q.absolute)),
                context={"expected": dimension_name(_DIMENSIONS[typ], typ == "temperature"), "actual": dimension_name(q.dimension, q.absolute)})
        number = q.as_number()
        low = spec.get("min", 0 if typ in _NONNEGATIVE else -1e12)
        high = spec.get("max", 1e12)
        if number < low or number > high:
            raise ValidationError("parameter outside bounds: " + name, context={"value": number, "min": low, "max": high})
        if typ in _NONNEGATIVE and number < 0:
            raise ValidationError("%s cannot be negative" % typ)
        return Quantity(int(number) if typ == "integer" else number, q.dimension, q.absolute)


def _data_value(name, spec, raw):
    typ = spec["type"]
    if typ == "string":
        if not isinstance(raw, str) or len(raw) > MAX_STRING_VALUE:
            raise ValidationError("string parameter requires text of at most %d characters" % MAX_STRING_VALUE)
        return raw
    if typ == "enumeration":
        if raw not in spec["choices"]:
            raise ValidationError("enumeration value must be one of: " + ", ".join(spec["choices"]), context={"choices": list(spec["choices"])})
        return raw
    if typ in ("material_ref", "element_ref", "geometry_ref"):
        if raw is None:
            return None
        return require_id(raw)
    if typ in ("vector", "coordinate"):
        size = spec.get("size", 3)
        if not isinstance(raw, (list, tuple)) or len(raw) != size or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in raw):
            raise ValidationError("%s parameter requires %d numbers" % (typ, size))
        return [float(finite(x)) for x in raw]
    if typ == "transform":
        return list(require_transform(raw))
    raise ValidationError("unsupported parameter type")


_GRAPH_CACHE: "OrderedDict[str, ParameterGraph]" = OrderedDict()
_GRAPH_CACHE_SIZE = 64
_GRAPH_CACHE_LOCK = threading.Lock()


def _graph_for(specifications: dict) -> "ParameterGraph":
    """A validated graph for ``specifications``, reused when an identical table was seen recently.

    Validating a table and building its graph costs as much as evaluating it, and scene validation resolves every instance of a
    family against the same table. The key is the table's canonical JSON, and the cached graph owns a private copy, so a caller
    changing its dictionary afterwards cannot change what a later call sees. A table that fails validation is never cached.
    """
    try:
        key = json.dumps(specifications, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return ParameterGraph(specifications)
    with _GRAPH_CACHE_LOCK:
        graph = _GRAPH_CACHE.get(key)
        if graph is not None:
            _GRAPH_CACHE.move_to_end(key)
            return graph
    graph = ParameterGraph(copy.deepcopy(specifications))
    with _GRAPH_CACHE_LOCK:
        _GRAPH_CACHE[key] = graph
        while len(_GRAPH_CACHE) > _GRAPH_CACHE_SIZE:
            _GRAPH_CACHE.popitem(last=False)
    return graph


_RESULT_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_RESULT_CACHE_SIZE = 8192
_RESULT_CACHE_LOCK = threading.Lock()


def _copy_values(values: dict) -> dict:
    """A copy of resolved values (numbers, strings, booleans, and lists of those) that does not share a list with the original; cheaper than deepcopy."""
    return {k: ([list(x) if isinstance(x, list) else x for x in v] if isinstance(v, list) else v) for k, v in values.items()}


def evaluate_parameters(specifications: dict, overrides: dict | None = None) -> dict:
    """Resolve a typed DAG transactionally; failures produce no partial values.

    The evaluation is a pure function of the table and the overrides, and scene validation resolves every instance of a family on every
    edit, so the resolved values are remembered for a table and a set of overrides seen recently (a bounded cache, never holding a failure).
    Each caller gets its own copy of the values.
    """
    try:
        key = json.dumps([specifications, overrides or {}], sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return _graph_for(specifications).evaluate(overrides)
    with _RESULT_CACHE_LOCK:
        hit = _RESULT_CACHE.get(key)
        if hit is not None:
            _RESULT_CACHE.move_to_end(key)
            return _copy_values(hit)
    values = _graph_for(specifications).evaluate(overrides)
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE[key] = _copy_values(values)
        while len(_RESULT_CACHE) > _RESULT_CACHE_SIZE:
            _RESULT_CACHE.popitem(last=False)
    return values


class ParameterEvaluator:
    """A fingerprint-keyed cache around :class:`ParameterGraph`.

    ``evaluate(specifications, overrides)`` returns the same values as :func:`evaluate_parameters`. When the previous call used an
    identical table, only the overrides that changed (and their dependents) are recomputed; a changed table (any definition,
    formula, bound...) invalidates the whole cache. ``recomputed`` names what the last call actually evaluated.
    """

    def __init__(self):
        self._graph = None
        self._fingerprint = None
        self._overrides = None
        self._values = None
        self.recomputed: tuple = ()
        self.full_evaluations = 0
        self.incremental_evaluations = 0
        self.cache_hits = 0

    def evaluate(self, specifications: dict, overrides: dict | None = None) -> dict:
        overrides = {} if overrides is None else overrides
        graph = ParameterGraph(specifications)
        fp = graph.fingerprint()
        if fp == self._fingerprint and self._values is not None:
            names = set(overrides) | set(self._overrides)
            changed = {n for n in names if overrides.get(n, _MISSING) != self._overrides.get(n, _MISSING)}
            if not changed:
                self.cache_hits += 1
                self.recomputed = ()
                return dict(self._values)
            values, recomputed = graph.evaluate_with_report(overrides, previous=self._values, changed=changed)
            self.incremental_evaluations += 1
        else:
            values, recomputed = graph.evaluate_with_report(overrides)
            self.full_evaluations += 1
        self._graph, self._fingerprint, self._overrides, self._values = graph, fp, dict(overrides), values
        self.recomputed = recomputed
        return dict(values)


_MISSING = object()


def check_formula(specifications: dict, name: str, expression: str, *, type: str = "number", overrides: dict | None = None) -> dict:
    """Validate a formula for ``name`` against the table and preview its value, without changing anything.

    Returns ``{"ok", "parameter", "expression", "dependencies", "dependents", "value", "dimension"}`` or, on failure,
    ``{"ok": False, "error": {"code", "message", "reason", ...}}`` naming the parameter, the formula and what is wrong. A cycle
    reports its path; a dimension error reports expected and actual.
    """
    out = {"ok": False, "parameter": name, "expression": expression, "dependencies": [], "dependents": [], "value": None, "dimension": None}
    try:
        base = specifications.get(name)
        spec = {k: v for k, v in (base or {}).items() if k in ("type", "min", "max", "description", "read_only", "scope")}
        spec.setdefault("type", type)
        spec["expression"] = expression
        graph = ParameterGraph({**specifications, name: spec})
        values = graph.evaluate({k: v for k, v in (overrides or {}).items() if k != name})
        deps = graph.dependencies[name]
        out.update(ok=True, value=values[name], dependencies=sorted(deps), dependents=sorted(graph.transitive_dependents({name})),
                   dimension=dimension_name(_DIMENSIONS[spec["type"]], spec["type"] == "temperature") if spec["type"] in _DIMENSIONS else spec["type"])
    except (ValidationError, ResourceLimitError) as exc:
        out["error"] = {"code": exc.code, "message": exc.args[0] if exc.args else str(exc), **{k: v for k, v in (exc.context or {}).items()}}
    return out
