"""Deterministic geometric constraint verification for explicitly supported 2D primitives.

This verifier does not claim to solve unconstrained geometry systems.
Constraint references are stable SDRW entity UUIDs; failures are reported.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import atan2, hypot, pi
from ..common.errors import ValidationError
from ..common.helpers import finite, vector
from ..common.ids import require_id

@dataclass(frozen=True)
class ConstraintResult:
    kind:str
    satisfied:bool
    residual:float
    tolerance:float
    entity_ids:tuple[str,...]

def check_constraints(entities, constraints, *, absolute_tolerance=1e-8, relative_tolerance=1e-8):
    from .entities import validate_entity
    abs_tol=finite(absolute_tolerance,"absolute_tolerance")
    rel_tol=finite(relative_tolerance,"relative_tolerance")
    if abs_tol<0 or rel_tol<0:
        raise ValidationError("negative geometric tolerance")
    if not isinstance(entities,list) or not isinstance(constraints,list) or len(constraints)>10_000:
        raise ValidationError("invalid constraint input")
    mapping={}
    for entity in entities:
        validate_entity(entity)
        if entity["id"] in mapping:
            raise ValidationError("duplicate constraint entity ID")
        mapping[entity["id"]]=entity

    def segment(oid):
        entity=mapping[oid]
        if entity["kind"]!="line":
            raise ValidationError("orientation constraints require line entities")
        a,b=entity["geometry"]["start"],entity["geometry"]["end"]
        direction=(b[0]-a[0],b[1]-a[1])
        length=hypot(*direction)
        return a,b,direction,length

    results=[]
    for constraint in constraints:
        if not isinstance(constraint,dict) or set(constraint)-{"kind","entities","value"}:
            raise ValidationError("unsupported constraint fields")
        kind=constraint.get("kind")
        if kind not in {"parallel","perpendicular","horizontal","vertical","equal_length",
                        "distance","coincident","angle"}:
            raise ValidationError("unsupported geometric constraint")
        ids=constraint.get("entities")
        expected=1 if kind in {"horizontal","vertical"} else 2
        if not isinstance(ids,list) or len(ids)!=expected:
            raise ValidationError("invalid constraint arity")
        ids=tuple(require_id(v) for v in ids)
        if any(v not in mapping for v in ids):
            raise ValidationError("broken constraint reference")
        a,b,u,ulen=segment(ids[0]) if kind not in {"coincident","distance"} else (None,None,None,None)
        if kind in {"parallel","perpendicular","equal_length","angle"}:
            _,_,v,vlen=segment(ids[1])
            cross=u[0]*v[1]-u[1]*v[0]
            dot=u[0]*v[0]+u[1]*v[1]
            if kind=="parallel":
                residual=abs(cross)/(ulen*vlen)
                target=0.0
            elif kind=="perpendicular":
                residual=abs(dot)/(ulen*vlen)
                target=0.0
            elif kind=="equal_length":
                residual=abs(ulen-vlen)
                target=max(ulen,vlen)
            else:
                from math import acos
                angle=finite(constraint.get("value"),"target angle")
                if not 0<=angle<=pi:
                    raise ValidationError("angle constraint must be between 0 and pi radians")
                residual=abs(atan2(abs(cross),dot)-angle)
                target=pi
        elif kind in {"horizontal","vertical"}:
            residual=abs(u[1] if kind=="horizontal" else u[0])
            target=ulen
        else:
            def point(oid):
                item=mapping[oid]
                if item["kind"]!="point":
                    raise ValidationError("point constraints require point entities")
                return vector(item["geometry"]["position"],2)
            p,q=point(ids[0]),point(ids[1])
            measured=hypot(p[0]-q[0],p[1]-q[1])
            if kind=="distance":
                desired=finite(constraint.get("value"),"distance")
                if desired<0:
                    raise ValidationError("distance cannot be negative")
                residual=abs(measured-desired)
                target=max(measured,desired)
            else:
                residual=measured
                target=0.0
        tol=abs_tol+rel_tol*target
        results.append(ConstraintResult(kind,residual<=tol,residual,tol,ids))
    return results
