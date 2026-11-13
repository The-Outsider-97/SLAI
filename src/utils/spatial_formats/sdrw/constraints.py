"""Deterministic geometric constraint verification for explicitly supported 2D primitives.

This verifier does not claim to solve unconstrained geometry systems.
Constraint references are stable SDRW entity UUIDs; failures are reported.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from math import atan2, hypot, pi

from ..common.errors import (ConflictError, DependencyUnavailableError, ReferenceError,
                             ResourceLimitError, ValidationError)
from ..common.helpers import finite, vector
from ..common.ids import require_id

@dataclass(frozen=True)
class ConstraintResult:
    kind:str
    satisfied:bool
    residual:float
    tolerance:float
    entity_ids:tuple[str,...]


@dataclass(frozen=True)
class ConstraintSolution:
    """Proposed native geometry, with a verified satisfaction report."""
    geometries: dict[str, dict]
    results: tuple[ConstraintResult, ...]
    iterations: int
    max_residual: float
    converged: bool


CONSTRAINT_METADATA_KEY = "slai.constraints.v1"


def validate_stored_constraints(entities, payload):
    """Check version, IDs and semantics of declarative persistent constraints."""
    if not isinstance(payload, dict) or set(payload) != {"version", "items"} or payload["version"] != 1:
        raise ValidationError("unsupported stored SDRW constraint contract")
    constraints = payload["items"]
    if not isinstance(constraints, list) or len(constraints) > 256:
        raise ResourceLimitError("stored constraints must be a bounded array")
    mapping = {e["id"]: e for e in entities}
    involved = set()
    for c in constraints:
        if not isinstance(c, dict) or not isinstance(c.get("entities"), list):
            raise ValidationError("invalid stored constraint reference")
        for ident in c["entities"]:
            require_id(ident)
            if ident not in mapping:
                raise ReferenceError("stored constraint references a missing entity")
            involved.add(ident)
    if len(involved) > 128:
        raise ResourceLimitError("stored constraints reference too many entities")
    return check_constraints([mapping[i] for i in involved], constraints)

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
            raise ReferenceError("broken constraint reference")

        if kind in {"parallel","perpendicular","equal_length","angle"}:
            _,_,u,ulen=segment(ids[0])
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
                angle=finite(constraint.get("value"),"target angle")
                if not 0<=angle<=pi:
                    raise ValidationError("angle constraint must be between 0 and pi radians")
                residual=abs(atan2(abs(cross),dot)-angle)
                target=pi
        elif kind in {"horizontal","vertical"}:
            _,_,u,ulen=segment(ids[0])
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


def solve_constraints(entities, constraints, *, fixed_ids=(), absolute_tolerance=1e-8,
                      relative_tolerance=1e-8, max_iterations=300) -> ConstraintSolution:
    """Least-change numerical solve for SDRW point/line constraint systems.

    A failed or underdetermined solve never edits source. Fixed entities retain
    their original geometry. The caller must explicitly commit the proposal;
    that commit is an ordinary revision-checked SDRW transaction.
    """
    if type(max_iterations) is not int or not 1 <= max_iterations <= 2_000:
        raise ValidationError("max_iterations must be 1..2000")
    check_constraints(entities, constraints, absolute_tolerance=absolute_tolerance,
                      relative_tolerance=relative_tolerance)
    if not isinstance(fixed_ids, (tuple, list, set)):
        raise ValidationError("fixed_ids must be a collection of entity UUIDs")
    fixed = {require_id(ident) for ident in fixed_ids}
    source = {e["id"]: e for e in entities}
    if fixed - set(source):
        raise ReferenceError("fixed entity is absent from the drawing")
    involved = {ident for c in constraints for ident in c["entities"]}
    if len(involved) > 128 or len(constraints) > 256:
        raise ResourceLimitError("constraint solver has a 128-entity/256-constraint budget")
    writable = sorted(involved - fixed)
    if not constraints:
        return ConstraintSolution({}, (), 0, 0.0, True)
    # One solver on every machine: the neutral constraint engine (numpy only). The solver that used to be chosen when scipy happened to be installed did not
    # converge to the tolerance on simple systems and "solved" a contradictory one by collapsing a line to a point, so results depended on what was installed.
    return _solve_with_engine(entities, constraints, fixed, writable, source, absolute_tolerance, relative_tolerance, max_iterations)


# ---------------------------------------------------------------------------------------------------- engine-backed analysis (numpy only)
def _engine_input(entities, constraints, fixed_ids):
    """SDRW points and lines in the neutral form of ``common.constraint_engine``. Other kinds are left out and show up as unsupported references."""
    ents = {}
    for e in entities:
        if e["kind"] == "point":
            ents[e["id"]] = {"kind": "point", "coords": list(e["geometry"]["position"]), "name": e.get("name")}
        elif e["kind"] == "line":
            ents[e["id"]] = {"kind": "segment", "coords": list(e["geometry"]["start"]) + list(e["geometry"]["end"]), "name": e.get("name")}
    cons = []
    for c in constraints:
        refs = list(c["entities"])
        if c["kind"] in ("distance", "coincident"):                  # SDRW stores these between points; a line is its start point (the verifier's own reading)
            refs = [r if ents.get(r, {}).get("kind") != "segment" else {"entity": r, "part": "start"} for r in refs]
        cons.append({"kind": c["kind"], "refs": refs, **({"value": c["value"]} if "value" in c else {})})
    return ents, cons


def analyze_constraints(entities, constraints, *, fixed_ids=(), absolute_tolerance=1e-8, max_iterations=200):
    """Degrees of freedom, redundancy and conflicts of a point/line constraint system, with the stored references checked (see the engine's docs).
    Read-only: nothing is solved into the document."""
    from ..common import constraint_engine as CE
    for c in constraints:
        if not isinstance(c, dict) or not isinstance(c.get("entities"), list):
            raise ValidationError("invalid constraint")
    ents, cons = _engine_input(entities, constraints, fixed_ids)
    fixed = [require_id(i) for i in fixed_ids]
    return CE.analyze(ents, cons, fixed=[i for i in fixed if i in ents], tolerance=absolute_tolerance, max_iterations=max_iterations)


def _solve_with_engine(entities, constraints, fixed, writable, source, absolute_tolerance, relative_tolerance, max_iterations):
    from ..common import constraint_engine as CE
    from .entities import validate_entity
    report = analyze_constraints(entities, constraints, fixed_ids=fixed, absolute_tolerance=absolute_tolerance, max_iterations=max_iterations)
    if report["unresolved"]:
        raise ReferenceError("a constraint reference could not be resolved", context={"unresolved": report["unresolved"]})
    if report["status"] == "conflicting":
        raise ConflictError("constraint solver could not satisfy the system within tolerance",
                            context={"max_residual": report["max_residual"], "iterations": report["iterations"], "converged": report["converged"],
                                     "conflicting": report["conflicting"]})
    candidates = deepcopy(entities)
    proposals = {}
    for entity in candidates:
        ident = entity["id"]
        if ident not in report["solution"]:
            continue
        values = [finite(float(v), "solved coordinate") for v in report["solution"][ident]]
        if entity["kind"] == "point":
            entity["geometry"]["position"] = values
        else:
            entity["geometry"]["start"], entity["geometry"]["end"] = values[:2], values[2:]
        validate_entity(entity)
        if entity["kind"] == "line":                                  # a system can be "solved" by collapsing a line to a point: that is not a drawing
            old_g, new_g = source[ident]["geometry"], entity["geometry"]
            was, now = hypot(old_g["end"][0] - old_g["start"][0], old_g["end"][1] - old_g["start"][1]), hypot(new_g["end"][0] - new_g["start"][0], new_g["end"][1] - new_g["start"][1])
            if now < max(1e-9, 1e-6 * was):
                raise ConflictError("the only solution collapses a line to a point; the constraints are contradictory for a real line",
                                    context={"entity": ident, "constraints_conflict": True})
        if entity["geometry"] != source[ident]["geometry"]:
            proposals[ident] = deepcopy(entity["geometry"])
    results = tuple(check_constraints(candidates, constraints, absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance))
    if not all(r.satisfied for r in results):
        raise ConflictError("constraint solver could not satisfy the system within tolerance",
                            context={"max_residual": max(r.residual for r in results), "iterations": report["iterations"], "converged": report["converged"]})
    return ConstraintSolution(proposals, results, int(report["iterations"]), max(r.residual for r in results), bool(report["converged"]))
