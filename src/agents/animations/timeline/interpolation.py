"""Deterministic scalar and unit quaternion interpolation (scalar-first w,x,y,z)."""
from math import acos, cos, sin, sqrt, isfinite

def lerp(a: float, b: float, alpha: float) -> float:
    if not all(isfinite(float(x)) for x in (a, b, alpha)) or not 0 <= alpha <= 1:
        raise ValueError("invalid interpolation inputs")
    return a + (b - a) * alpha

def slerp(a, b, alpha: float):
    if not isfinite(float(alpha)) or not 0 <= alpha <= 1 or len(a) != 4 or len(b) != 4:
        raise ValueError("invalid quaternion interpolation")
    def norm(q):
        if not all(isfinite(float(x)) for x in q):
            raise ValueError("quaternion must be finite")
        magnitude = sqrt(sum(float(x) ** 2 for x in q))
        if magnitude < 1e-15:
            raise ValueError("zero quaternion")
        return tuple(float(x) / magnitude for x in q)
    q0, q1 = norm(a), norm(b)
    dot = sum(x*y for x,y in zip(q0,q1))
    if dot < 0:
        q1 = tuple(-x for x in q1)
        dot = -dot
    if dot > .9995:
        return norm(tuple(lerp(x,y,alpha) for x,y in zip(q0,q1)))
    angle = acos(min(1.0,max(-1.0,dot)))
    sine = sin(angle)
    return norm(tuple((sin((1-alpha)*angle)/sine)*x + (sin(alpha*angle)/sine)*y for x,y in zip(q0,q1)))
