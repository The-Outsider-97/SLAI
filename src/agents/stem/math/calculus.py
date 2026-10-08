"""Symbolic and algorithmic calculus for SLAI STEM.

Grounding: Bronstein (symbolic integration) and Griewank & Walther
(algorithmic differentiation). Numerical finite differences are intentionally
owned by :mod:`numerical_methods`.
"""
from __future__ import annotations

__version__ = "2.3.0"

import ast
import math

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import STEMCalculusError, STEMDifferentiationError
from ..utils.stem_helpers import as_float_array, ensure_finite_number, ensure_non_empty_string, validate_callable
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Calculus")
printer = PrettyPrinter()


@dataclass(frozen=True)
class Dual:
    real: float
    partials: Tuple[float, ...]

    def _coerce(self, other: Any) -> "Dual":
        return other if isinstance(other, Dual) else Dual(float(other), (0.0,) * len(self.partials))

    def _zip(self, other: "Dual") -> List[Tuple[float, float]]:
        if len(self.partials) != len(other.partials):
            raise STEMDifferentiationError("Dual dimensions do not match")
        return list(zip(self.partials, other.partials))

    def __add__(self, other: Any) -> "Dual":
        o = self._coerce(other); return Dual(self.real + o.real, tuple(a + b for a, b in self._zip(o)))
    __radd__ = __add__

    def __sub__(self, other: Any) -> "Dual":
        o = self._coerce(other); return Dual(self.real - o.real, tuple(a - b for a, b in self._zip(o)))
    def __rsub__(self, other: Any) -> "Dual": return self._coerce(other).__sub__(self)

    def __mul__(self, other: Any) -> "Dual":
        o = self._coerce(other); return Dual(self.real * o.real, tuple(a * o.real + self.real * b for a, b in self._zip(o)))
    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "Dual":
        o = self._coerce(other)
        if o.real == 0.0: raise STEMDifferentiationError("Dual division by zero")
        d = o.real * o.real
        return Dual(self.real / o.real, tuple((a * o.real - self.real * b) / d for a, b in self._zip(o)))
    def __rtruediv__(self, other: Any) -> "Dual": return self._coerce(other).__truediv__(self)

    def __pow__(self, exponent: Any) -> "Dual":
        if isinstance(exponent, Dual):
            if self.real <= 0.0: raise STEMDifferentiationError("Variable exponent requires positive base")
            return (exponent * self.log()).exp()
        p = float(exponent)
        value = self.real ** p
        derivative = p * (self.real ** (p - 1.0)) if not (self.real == 0.0 and p < 1.0) else math.nan
        return Dual(value, tuple(derivative * a for a in self.partials))

    def __neg__(self) -> "Dual": return Dual(-self.real, tuple(-a for a in self.partials))
    def __pos__(self) -> "Dual": return self
    def sin(self) -> "Dual": return Dual(math.sin(self.real), tuple(math.cos(self.real) * a for a in self.partials))
    def cos(self) -> "Dual": return Dual(math.cos(self.real), tuple(-math.sin(self.real) * a for a in self.partials))
    def tan(self) -> "Dual":
        c = math.cos(self.real); return Dual(math.tan(self.real), tuple(a / (c*c) for a in self.partials))
    def exp(self) -> "Dual":
        v = math.exp(self.real); return Dual(v, tuple(v * a for a in self.partials))
    def log(self) -> "Dual":
        if self.real <= 0.0: raise STEMDifferentiationError("log domain error")
        return Dual(math.log(self.real), tuple(a / self.real for a in self.partials))
    def sqrt(self) -> "Dual": return self ** 0.5
    def sinh(self) -> "Dual": return Dual(math.sinh(self.real), tuple(math.cosh(self.real)*a for a in self.partials))
    def cosh(self) -> "Dual": return Dual(math.cosh(self.real), tuple(math.sinh(self.real)*a for a in self.partials))
    def tanh(self) -> "Dual":
        v=math.tanh(self.real); return Dual(v, tuple((1-v*v)*a for a in self.partials))
    def abs(self) -> "Dual":
        if self.real == 0.0: raise STEMDifferentiationError("abs is non-differentiable at zero")
        sign = 1.0 if self.real > 0 else -1.0
        return Dual(abs(self.real), tuple(sign*a for a in self.partials))


_ALLOWED_FUNCS = {"sin","cos","tan","exp","log","sqrt","sinh","cosh","tanh"}


def _expr(node: ast.AST) -> str:
    if isinstance(node, ast.Constant): return repr(float(node.value)) if isinstance(node.value, (int,float)) else repr(node.value)
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return f"-({_expr(node.operand)})"
    if isinstance(node, ast.BinOp):
        op = {ast.Add:"+", ast.Sub:"-", ast.Mult:"*", ast.Div:"/", ast.Pow:"**"}.get(type(node.op))
        if not op: raise STEMCalculusError("Unsupported symbolic operator")
        return f"({_expr(node.left)} {op} {_expr(node.right)})"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS and len(node.args)==1:
        return f"{node.func.id}({_expr(node.args[0])})"
    raise STEMCalculusError("Unsupported symbolic expression", context={"node":type(node).__name__})


def _d(node: ast.AST, var: str) -> str:
    if isinstance(node, ast.Constant): return "0.0"
    if isinstance(node, ast.Name): return "1.0" if node.id == var else "0.0"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return f"-({_d(node.operand,var)})"
    if isinstance(node, ast.BinOp):
        u,v=_expr(node.left),_expr(node.right); du,dv=_d(node.left,var),_d(node.right,var)
        if isinstance(node.op,ast.Add): return f"({du} + {dv})"
        if isinstance(node.op,ast.Sub): return f"({du} - {dv})"
        if isinstance(node.op,ast.Mult): return f"(({du})*({v}) + ({u})*({dv}))"
        if isinstance(node.op,ast.Div): return f"((({du})*({v}) - ({u})*({dv})) / (({v})**2))"
        if isinstance(node.op,ast.Pow):
            if isinstance(node.right,ast.Constant) and isinstance(node.right.value,(int,float)):
                p=float(node.right.value); return f"({p}*(({u})**({p-1.0}))*({du}))"
            return f"((({u})**({v}))*(({dv})*log({u}) + ({v})*({du})/({u})))"
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and len(node.args)==1:
        u=_expr(node.args[0]); du=_d(node.args[0],var); name=node.func.id
        factors={"sin":f"cos({u})","cos":f"-sin({u})","tan":f"1/(cos({u})**2)","exp":f"exp({u})","log":f"1/({u})","sqrt":f"1/(2*sqrt({u}))","sinh":f"cosh({u})","cosh":f"sinh({u})","tanh":f"1-(tanh({u})**2)"}
        if name in factors: return f"(({factors[name]})*({du}))"
    raise STEMCalculusError("Unsupported symbolic derivative")


class Calculus:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.calculus_config = dict(get_config_section("calculus", config=self.config) or {})
        if config: self.calculus_config.update(dict(config))

    def symbolic_derivative(self, expression: str, variable: str, order: int = 1) -> str:
        ensure_non_empty_string(expression,"expression"); ensure_non_empty_string(variable,"variable")
        if order < 0: raise STEMCalculusError("order must be non-negative")
        current = expression
        for _ in range(order):
            tree=ast.parse(current,mode="eval"); current=_d(tree.body,variable)
        return current

    def symbolic_integral(self, expression: str, variable: str) -> Optional[str]:
        """Integrate a conservative exact subset; return ``None`` if unsupported."""
        tree=ast.parse(expression,mode="eval").body
        def integ(node: ast.AST) -> Optional[str]:
            if isinstance(node,ast.Constant):
                if isinstance(node.value,(int,float)):
                    return f"({float(node.value)}*{variable})"
                return None
            if isinstance(node,ast.Name):
                if node.id==variable: return f"(({variable}**2)/2.0)"
                return f"({node.id}*{variable})"
            if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub)):
                a,b=integ(node.left),integ(node.right)
                if a is None or b is None:return None
                return f"({a} {'+' if isinstance(node.op,ast.Add) else '-'} {b})"
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):
                if isinstance(node.left,ast.Constant) and isinstance(node.left.value,(int,float)):
                    right=integ(node.right); return None if right is None else f"({float(node.left.value)}*({right}))"
                if isinstance(node.right,ast.Constant) and isinstance(node.right.value,(int,float)):
                    left=integ(node.left); return None if left is None else f"({float(node.right.value)}*({left}))"
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow) and isinstance(node.left,ast.Name) and node.left.id==variable and isinstance(node.right,ast.Constant) and isinstance(node.right.value,(int,float)):
                p=float(node.right.value)
                return f"(log({variable}))" if p==-1.0 else f"(({variable}**{p+1.0})/{p+1.0})"
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and len(node.args)==1 and isinstance(node.args[0],ast.Name) and node.args[0].id==variable:
                return {"sin":f"-cos({variable})","cos":f"sin({variable})","exp":f"exp({variable})"}.get(node.func.id)
            return None
        return integ(tree)

    def _dual_call(self, f: Callable[..., Any], x: Sequence[float]) -> Dual:
        values=as_float_array(x,"x")
        duals=[Dual(v,tuple(1.0 if i==j else 0.0 for j in range(len(values)))) for i,v in enumerate(values)]
        result=f(*duals)
        if not isinstance(result,Dual): raise STEMDifferentiationError("Function did not propagate Dual values")
        return result

    def gradient(self, f: Callable[..., Any], x: Sequence[float], method: str = "ad") -> List[float]:
        if method != "ad": raise STEMDifferentiationError("Only algorithmic differentiation is owned by Calculus")
        return list(self._dual_call(validate_callable(f,"f"),x).partials)

    def jacobian(self, f: Callable[..., Any], x: Sequence[float], method: str = "ad") -> List[List[float]]:
        if method != "ad": raise STEMDifferentiationError("Only algorithmic differentiation is owned by Calculus")
        values=as_float_array(x,"x"); duals=[Dual(v,tuple(1.0 if i==j else 0.0 for j in range(len(values)))) for i,v in enumerate(values)]
        result=f(*duals)
        seq=result if isinstance(result,(list,tuple)) else [result]
        if not all(isinstance(item,Dual) for item in seq): raise STEMDifferentiationError("Jacobian function must propagate Dual outputs")
        return [list(item.partials) for item in seq]

    def hessian(self, f: Callable[..., Any], x: Sequence[float], method: str = "ad") -> List[List[float]]:
        # Second-order central differences of the AD gradient: deterministic and
        # stable enough for the shared calculus contract without duplicating the
        # general finite-difference API in numerical_methods.
        values=as_float_array(x,"x"); n=len(values)
        h=float(self.calculus_config.get("hessian_step",1e-5))
        out=[[0.0]*n for _ in range(n)]
        for j in range(n):
            xp=values[:]; xm=values[:]; xp[j]+=h; xm[j]-=h
            gp=self.gradient(f,xp,method); gm=self.gradient(f,xm,method)
            for i in range(n): out[i][j]=(gp[i]-gm[i])/(2*h)
        return out

    def directional_derivative(self, f: Callable[..., Any], x: Sequence[float], direction: Sequence[float]) -> float:
        grad=self.gradient(f,x); d=as_float_array(direction,"direction")
        if len(grad)!=len(d): raise STEMDifferentiationError("direction dimension mismatch")
        norm=math.sqrt(math.fsum(v*v for v in d))
        if norm==0.0: raise STEMDifferentiationError("direction must be non-zero")
        return math.fsum(g*v/norm for g,v in zip(grad,d))

    def limit(self, f: Callable[[float], float], x: float, direction: str = "both") -> Optional[float]:
        validate_callable(f,"f"); x0=ensure_finite_number(x,"x")
        if direction not in {"left","right","both"}: raise STEMCalculusError("direction must be left, right, or both")
        scale=max(1.0,abs(x0)); hs=[scale*10.0**(-k) for k in range(3,10)]
        def side(sign:float)->float:
            vals=[ensure_finite_number(f(x0+sign*h),"f(x)",error_cls=STEMCalculusError) for h in hs]
            return vals[-1]
        left=side(-1.0) if direction in {"left","both"} else None; right=side(1.0) if direction in {"right","both"} else None
        if direction=="left": return left
        if direction=="right": return right
        if left is not None and right is not None and math.isclose(left,right,rel_tol=1e-7,abs_tol=1e-9): return 0.5*(left+right)
        return None

    def taylor_series(self, f: Callable[[float], float], x0: float, order: int, variable: Optional[str] = None) -> Mapping[str, Any]:
        if order < 0: raise STEMCalculusError("order must be non-negative")
        x0=ensure_finite_number(x0,"x0"); validate_callable(f,"f")
        coeffs=[ensure_finite_number(f(x0),"f(x0)")]
        h=float(self.calculus_config.get("taylor_step",1e-4))
        # recursion on finite differences is restricted to this diagnostic helper
        current=f
        for k in range(1,order+1):
            prev=current
            current=lambda z, prev=prev: (prev(z+h)-prev(z-h))/(2*h)
            coeffs.append(float(current(x0))/math.factorial(k))
        def evaluate(x: float)->float:
            dx=float(x)-x0
            return math.fsum(c*(dx**k) for k,c in enumerate(coeffs))
        return {"center":x0,"order":order,"coefficients":coeffs,"variable":variable or "x","evaluate":evaluate}

    def autodiff(self, f: Callable[..., Any]) -> Callable[[Sequence[float]], Tuple[float, List[float]]]:
        validate_callable(f,"f")
        def wrapped(x: Sequence[float]) -> Tuple[float,List[float]]:
            result=self._dual_call(f,x); return result.real,list(result.partials)
        return wrapped


__all__ = ["Calculus", "Dual"]
