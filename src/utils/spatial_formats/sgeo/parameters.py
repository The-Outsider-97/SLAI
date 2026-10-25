"""Strict, CPU-only expression evaluation for parametric CAD dimensions.

This is deliberately NOT Python evaluation. It accepts an allowlist of AST
nodes, evaluates a bounded dependency DAG, and rejects incompatible dimensions.
Lengths are expressed in the family/document's declared units, angles in radians.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

from ..common.errors import ValidationError, ResourceLimitError

MAX_EXPR_LENGTH = 512
MAX_AST_NODES = 128
MAX_EXPRESSION_DEPTH = 32
MAX_PARAMETERS = 128

# Length, angle exponents; quantities with squared/cubed length are derived.
_DIMENSIONS = {
    "number": (0, 0), "integer": (0, 0), "boolean": (0, 0),
    "length": (1, 0), "area": (2, 0), "volume": (3, 0), "angle": (0, 1),
}


@dataclass(frozen=True)
class Quantity:
    value: float | bool
    dimension: tuple[int, int] = (0, 0)

    def as_number(self) -> float:
        if isinstance(self.value, bool):
            raise ValidationError("boolean cannot be used in arithmetic")
        if not math.isfinite(self.value) or abs(self.value) > 1e12:
            raise ValidationError("nonfinite or unbounded parameter result")
        return float(self.value)


def _quantity(value, dimension=(0, 0)):
    if isinstance(value, bool):
        if dimension != (0, 0):
            raise ValidationError("boolean requires dimensionless type")
        return Quantity(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e12:
        raise ValidationError("expression result exceeds numeric limits")
    return Quantity(float(value), dimension)


def _boolean(q: Quantity) -> bool:
    if not isinstance(q.value, bool):
        raise ValidationError("conditional requires a Boolean expression")
    return q.value


def _compatible(a: Quantity, b: Quantity) -> None:
    if a.dimension != b.dimension:
        raise ValidationError("incompatible physical dimensions")
    a.as_number()
    b.as_number()


def _checked_expression(expression: str) -> ast.Expression:
    if not isinstance(expression, str) or not 0 < len(expression) <= MAX_EXPR_LENGTH:
        raise ValidationError("invalid expression length")
    # 'if(a,b,c)' is conventional in formulas but 'if' is reserved in Python.
    source = re.sub(r"\bif\s*\(", "choose(", expression)
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValidationError("invalid parameter expression syntax") from exc
    count = 0
    stack = [(tree, 0)]
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp,
               ast.Compare, ast.BoolOp, ast.IfExp, ast.Call, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub, ast.Eq,
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
            if not isinstance(node.func, ast.Name) or node.keywords or node.func.id not in {
                "abs", "min", "max", "sqrt", "sin", "cos", "choose", "clamp", "length", "angle",
            }:
                raise ValidationError("function is not allowlisted")
        if isinstance(node, ast.Constant) and (isinstance(node.value, str) or node.value is None):
            raise ValidationError("only numeric/boolean literals are supported")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


def expression_dependencies(expression: str) -> frozenset[str]:
    tree = _checked_expression(expression)
    function_names = {"abs", "min", "max", "sqrt", "sin", "cos", "choose", "clamp", "length", "angle"}
    return frozenset(n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in function_names)


def evaluate_expression(expression: str, variables: dict[str, Quantity]) -> Quantity:
    tree = _checked_expression(expression)

    def calc(n):
        if isinstance(n, ast.Constant):
            return _quantity(n.value)
        if isinstance(n, ast.Name):
            try:
                return variables[n.id]
            except KeyError as exc:
                raise ValidationError("unknown parameter: " + n.id) from exc
        if isinstance(n, ast.Expression):
            return calc(n.body)
        if isinstance(n, ast.IfExp):
            return calc(n.body if _boolean(calc(n.test)) else n.orelse)
        if isinstance(n, ast.UnaryOp):
            v = calc(n.operand)
            if isinstance(n.op, ast.Not):
                return Quantity(not _boolean(v))
            return _quantity(-v.as_number() if isinstance(n.op, ast.USub) else v.as_number(), v.dimension)
        if isinstance(n, ast.BoolOp):
            # Short-circuit evaluation.
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
                _compatible(left, right)
                a, b = left.as_number(), right.as_number()
                good = ((isinstance(op, ast.Lt) and a < b)
                        or (isinstance(op, ast.LtE) and a <= b)
                        or (isinstance(op, ast.Gt) and a > b)
                        or (isinstance(op, ast.GtE) and a >= b)
                        or (isinstance(op, ast.Eq) and a == b)
                        or (isinstance(op, ast.NotEq) and a != b))
                if not good:
                    return Quantity(False)
                left = right
            return Quantity(True)
        if isinstance(n, ast.BinOp):
            a, b = calc(n.left), calc(n.right)
            if isinstance(n.op, (ast.Add, ast.Sub)):
                _compatible(a, b)
                return _quantity(a.as_number() + (b.as_number() if isinstance(n.op, ast.Add)
                                                   else -b.as_number()), a.dimension)
            if isinstance(n.op, ast.Mult):
                return _quantity(a.as_number()*b.as_number(),
                                 tuple(x+y for x,y in zip(a.dimension, b.dimension)))
            if isinstance(n.op, ast.Div):
                denominator = b.as_number()
                if denominator == 0:
                    raise ValidationError("division by zero")
                return _quantity(a.as_number()/denominator,
                                 tuple(x-y for x,y in zip(a.dimension, b.dimension)))
            if isinstance(n.op, ast.Pow):
                if b.dimension != (0,0):
                    raise ValidationError("exponent must be dimensionless")
                exponent = b.as_number()
                if not exponent.is_integer() or not -4 <= exponent <= 4:
                    raise ValidationError("exponents must be bounded integers")
                if a.as_number() == 0 and exponent < 0:
                    raise ValidationError("zero to negative power")
                return _quantity(a.as_number() ** int(exponent),
                                 tuple(x*int(exponent) for x in a.dimension))
        if isinstance(n, ast.Call):
            name=n.func.id
            if name=="choose":
                if len(n.args)!=3:
                    raise ValidationError("choose requires three arguments")
                return calc(n.args[1] if _boolean(calc(n.args[0])) else n.args[2])
            args=[calc(arg) for arg in n.args]
            if name in {"min", "max", "clamp"}:
                if len(args) < 1 or (name=="clamp" and len(args)!=3):
                    raise ValidationError("invalid function arity")
                for arg in args[1:]:
                    _compatible(args[0],arg)
                fn = min if name=="min" else max
                value=(fn(x.as_number() for x in args) if name!="clamp"
                       else max(args[1].as_number(),min(args[0].as_number(),args[2].as_number())))
                return _quantity(value,args[0].dimension)
            if len(args)!=1:
                raise ValidationError("invalid function arity")
            arg=args[0]
            if name in ("length","angle"):
                if arg.dimension!=(0,0):
                    raise ValidationError("unit constructor requires dimensionless scalar")
                return _quantity(arg.as_number(),(1,0) if name=="length" else (0,1))
            if name=="abs":
                return _quantity(abs(arg.as_number()),arg.dimension)
            if name=="sqrt":
                if any(x%2 for x in arg.dimension) or arg.as_number()<0:
                    raise ValidationError("sqrt requires even dimensions and nonnegative input")
                return _quantity(math.sqrt(arg.as_number()),tuple(x//2 for x in arg.dimension))
            if name in ("sin","cos"):
                if arg.dimension!=(0,1):
                    raise ValidationError("trigonometry requires an angle (radians)")
                return _quantity((math.sin if name=="sin" else math.cos)(arg.as_number()))
        raise ValidationError("unsupported expression construct")

    try:
        return calc(tree)
    except (OverflowError, ZeroDivisionError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("expression numerical failure") from exc


def evaluate_parameters(specifications: dict, overrides: dict | None = None) -> dict:
    """Resolve a typed DAG transactionally; failures produce no partial values."""
    if not isinstance(specifications,dict) or len(specifications)>MAX_PARAMETERS:
        raise ValidationError("parameter table exceeds bounds")
    overrides={} if overrides is None else overrides
    if not isinstance(overrides,dict) or set(overrides)-set(specifications):
        raise ValidationError("invalid parameter overrides")
    deps={}
    for name,spec in specifications.items():
        if not isinstance(name,str) or not name.isidentifier() or len(name)>64:
            raise ValidationError("invalid parameter name")
        if not isinstance(spec,dict) or spec.get("type") not in _DIMENSIONS:
            raise ValidationError("invalid parameter definition")
        if "expression" in spec and name in overrides:
            raise ValidationError("derived parameters cannot be overridden")
        if spec.get("read_only",False) and name in overrides:
            raise ValidationError("read-only parameter override")
        if "expression" in spec:
            deps[name]=expression_dependencies(spec["expression"])
            if deps[name] - specifications.keys():
                raise ValidationError("unknown expression dependencies")
        else:
            deps[name]=frozenset()
    ordering=[]
    state={}
    for start in specifications:
        if state.get(start)==2:
            continue
        stack=[(start,iter(sorted(deps[start])))]
        state[start]=1
        while stack:
            name,iterator=stack[-1]
            dependency=next(iterator,None)
            if dependency is None:
                state[name]=2
                ordering.append(name)
                stack.pop()
            elif state.get(dependency)==1:
                raise ValidationError("cyclic parameter dependencies")
            elif state.get(dependency)!=2:
                state[dependency]=1
                stack.append((dependency,iter(sorted(deps[dependency]))))
    values={}
    for name in ordering:
        spec=specifications[name]
        typ=spec["type"]
        if "expression" in spec:
            q=evaluate_expression(spec["expression"],values)
            raw=q.value
        else:
            raw=overrides.get(name,spec.get("default"))
            q=_quantity(raw,_DIMENSIONS[typ]) if typ!="boolean" else Quantity(raw)
        if typ=="boolean":
            if not isinstance(raw,bool):
                raise ValidationError("Boolean parameter requires true/false")
            values[name]=Quantity(raw)
            continue
        if typ=="integer" and (not isinstance(raw,(int,float)) or isinstance(raw,bool)
                               or not float(raw).is_integer()):
            raise ValidationError("integer parameter requires integral value")
        if q.dimension != _DIMENSIONS[typ]:
            raise ValidationError("parameter dimension mismatch: "+name)
        number=q.as_number()
        if number < spec.get("min",-1e12) or number > spec.get("max",1e12):
            raise ValidationError("parameter outside bounds: "+name)
        if typ=="length" and number<0:
            raise ValidationError("length cannot be negative")
        values[name]=Quantity(int(number) if typ=="integer" else number,q.dimension)
    return {name:q.value for name,q in values.items()}
