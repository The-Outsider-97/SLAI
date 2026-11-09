"""A dependency-light (numpy only) geometric constraint engine: solve, degrees-of-freedom analysis, redundancy/conflict diagnosis, stable references.

It works on a neutral description so it serves 2D drawings and 3D scenes alike and knows nothing about either format:

  entities     ``{id: {"kind": "point"|"segment", "coords": [...], "name": optional}}``; a point has 2 or 3 coordinates, a segment twice that
               (start then end). Entities in ``fixed`` keep their coordinates and are constants of the system.
  constraints  ``{"kind": ..., "refs": [...], "value": ...}``. A *reference* is an entity id, or ``{"entity": id, "part": "start"|"end",
               "fingerprint": "..."}`` or ``{"name": "..."}``.

Kinds: coincident, distance, fixed (to ``value`` = coordinates), parallel, perpendicular, equal_length, angle (radians, 0..pi, unsigned), horizontal,
vertical (2D), point_on_line.

Honest limits, which the results repeat: the analysis is *local* (the Jacobian at the solution found), so a degree of freedom reported is a
direction in which the solution can move to first order; the solver finds the solution nearest the input that its damped Gauss-Newton reaches, not a
global one; a system with several solutions (a flipped triangle) returns one of them.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from .errors import DependencyUnavailableError, ValidationError

POINT_LIKE = {"coincident": (0, 1), "distance": (0, 1), "fixed": (0,), "point_on_line": (0,)}
SEGMENTS = {"parallel": (0, 1), "perpendicular": (0, 1), "equal_length": (0, 1), "angle": (0, 1), "horizontal": (0,), "vertical": (0,), "point_on_line": (1,)}
ARITY = {"coincident": 2, "distance": 2, "fixed": 1, "parallel": 2, "perpendicular": 2, "equal_length": 2, "angle": 2, "horizontal": 1, "vertical": 1, "point_on_line": 2}
KINDS = tuple(ARITY)
MAX_ENTITIES, MAX_CONSTRAINTS = 400, 1200


def _np():
    try:
        import numpy as np
    except ImportError as exc:                                      # pragma: no cover - numpy is a hard requirement of SGEO already
        raise DependencyUnavailableError("constraint solving requires numpy") from exc
    return np


def fingerprint(entity: Mapping[str, Any]) -> str:
    """What a reference remembers of its target: the kind and arity, so an id that now names something else is detected as stale."""
    return hashlib.sha1(f"{entity['kind']}:{len(entity['coords'])}".encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------------------------------------------------- references
def resolve(ref: Any, entities: Mapping[str, Mapping[str, Any]], *, need: str) -> dict[str, Any]:
    """Resolve one reference to ``{"state", "entity", "part", "message"}``. States: resolved, missing, stale, ambiguous, unsupported."""
    part = None
    if isinstance(ref, str):
        key = ref
    elif isinstance(ref, Mapping):
        part = ref.get("part")
        if "entity" in ref:
            key = ref["entity"]
        elif "name" in ref:
            hits = sorted(i for i, e in entities.items() if e.get("name") == ref["name"])
            if not hits:
                return {"state": "missing", "message": f"no entity is named '{ref['name']}'"}
            if len(hits) > 1:
                return {"state": "ambiguous", "message": f"{len(hits)} entities are named '{ref['name']}'", "candidates": hits}
            key = hits[0]
        else:
            return {"state": "unsupported", "message": "a reference needs an entity id or a name"}
    else:
        return {"state": "unsupported", "message": "a reference is an id or an object"}
    ent = entities.get(key)
    if ent is None:
        return {"state": "missing", "message": f"entity {key} is not in the system"}
    if isinstance(ref, Mapping) and ref.get("fingerprint") not in (None, fingerprint(ent)):
        return {"state": "stale", "message": f"entity {key} is not what the reference was made against (kind or size changed)"}
    if ent["kind"] == "point":
        if part is not None:
            return {"state": "unsupported", "message": "a point has no start or end"}
        return {"state": "resolved", "entity": key, "part": None}
    if need == "point":
        if part not in ("start", "end"):
            return {"state": "unsupported", "message": f"a segment needs 'start' or 'end' where a point is expected ({key})"}
        return {"state": "resolved", "entity": key, "part": part}
    if part is not None:
        return {"state": "unsupported", "message": "this constraint takes the whole segment"}
    return {"state": "resolved", "entity": key, "part": None}


def _needs(kind: str, i: int) -> str:
    if kind == "point_on_line":
        return "point" if i == 0 else "segment"
    return "point" if kind in ("coincident", "distance", "fixed") else "segment"


# ----------------------------------------------------------------------------------------------------------------------- system
class _System:
    def __init__(self, entities: Mapping[str, Mapping[str, Any]], constraints: Sequence[Mapping[str, Any]], fixed: Sequence[str]) -> None:
        np = _np()
        if not isinstance(entities, Mapping) or len(entities) > MAX_ENTITIES:
            raise ValidationError(f"a constraint system holds at most {MAX_ENTITIES} entities")
        if not isinstance(constraints, (list, tuple)) or len(constraints) > MAX_CONSTRAINTS:
            raise ValidationError(f"a constraint system holds at most {MAX_CONSTRAINTS} constraints")
        self.entities = {}
        for ident, e in entities.items():
            dim = {"point": 1, "segment": 2}.get(e.get("kind"))
            coords = e.get("coords")
            if dim is None or not isinstance(coords, (list, tuple)) or len(coords) not in (2 * dim, 3 * dim) or not all(isinstance(c, (int, float)) and math.isfinite(c) for c in coords):
                raise ValidationError(f"entity {ident} is not a point or segment with finite 2D or 3D coordinates")
            self.entities[ident] = {"kind": e["kind"], "coords": [float(c) for c in coords], "name": e.get("name")}
        self.fixed = set(fixed)
        if self.fixed - set(self.entities):
            raise ValidationError("a fixed entity is not in the system")
        self.unresolved: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []                     # resolved constraints
        for idx, c in enumerate(constraints):
            if not isinstance(c, Mapping):
                raise ValidationError("a constraint is an object")
            kind = c.get("kind")
            refs = c.get("refs", c.get("entities"))
            if kind not in ARITY:
                raise ValidationError(f"unsupported constraint kind '{kind}'", context={"kinds": list(KINDS)})
            if not isinstance(refs, (list, tuple)) or len(refs) != ARITY[kind]:
                raise ValidationError(f"{kind} takes {ARITY[kind]} reference(s)")
            value = c.get("value")
            if kind in ("distance", "angle") and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValidationError(f"{kind} needs a non-negative finite value")
            if kind == "angle" and value > math.pi + 1e-12:
                raise ValidationError("an angle is between 0 and pi (radians)")
            if kind == "fixed" and (not isinstance(value, (list, tuple)) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)):
                raise ValidationError("fixed needs the coordinates to hold")
            resolved = [resolve(r, self.entities, need=_needs(kind, i)) for i, r in enumerate(refs)]
            bad = [r for r in resolved if r["state"] != "resolved"]
            if bad:
                self.unresolved.append({"index": idx, "kind": kind, "state": bad[0]["state"], "message": bad[0]["message"], **({"candidates": bad[0]["candidates"]} if "candidates" in bad[0] else {})})
                continue
            self.rows.append({"index": idx, "kind": kind, "refs": resolved, "value": value})
        involved = sorted({r["entity"] for row in self.rows for r in row["refs"]})
        self.involved = involved
        self.names = [i for i in involved if i not in self.fixed]                  # entities with unknowns
        self.slices, n = {}, 0
        for i in self.names:
            k = len(self.entities[i]["coords"])
            self.slices[i] = (n, n + k)
            n += k
        self.n = n
        self.x0 = np.array([c for i in self.names for c in self.entities[i]["coords"]], dtype=float)

    # -- geometry
    def _vec(self, ref: Mapping[str, Any], x: Any):
        np = _np()
        ident = ref["entity"]
        if ident in self.slices:
            a, b = self.slices[ident]
            v = x[a:b]
        else:
            v = np.array(self.entities[ident]["coords"], dtype=float)
        if self.entities[ident]["kind"] == "point":
            return v
        d = len(v) // 2
        if ref["part"] == "start":
            return v[:d]
        if ref["part"] == "end":
            return v[d:]
        return v[:d], v[d:]

    def block(self, row: Mapping[str, Any], x: Any) -> Any:
        """The residual vector of one constraint at ``x``."""
        np = _np()
        kind, refs = row["kind"], row["refs"]
        if kind in ("coincident", "distance"):
            p, q = self._vec(refs[0], x), self._vec(refs[1], x)
            if len(p) != len(q):
                return np.array([1e3])
            return (p - q) if kind == "coincident" else np.array([np.linalg.norm(p - q) - row["value"]])
        if kind == "fixed":
            p = self._vec(refs[0], x)
            target = np.array(row["value"], dtype=float)
            return p - target if len(target) == len(p) else np.array([1e3])
        if kind in ("horizontal", "vertical"):
            a, b = self._vec(refs[0], x)
            if len(a) != 2:
                return np.array([1e3])
            return np.array([b[1] - a[1] if kind == "horizontal" else b[0] - a[0]])
        if kind == "point_on_line":
            p = self._vec(refs[0], x)
            a, b = self._vec(refs[1], x)
            u, w = b - a, p - a
            lu = np.linalg.norm(u)
            if lu < 1e-12 or len(p) != len(a):
                return np.array([1e3])
            return _cross(u, w) / lu
        (a, b), (c, d) = self._vec(refs[0], x), self._vec(refs[1], x)
        u, v = b - a, d - c
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if len(u) != len(v):
            return np.array([1e3])
        if kind == "equal_length":
            return np.array([lu - lv])
        if lu < 1e-12 or lv < 1e-12:
            return np.array([1e3])
        if kind == "parallel":
            return _cross(u, v) / (lu * lv)
        if kind == "perpendicular":
            return np.array([float(np.dot(u, v)) / (lu * lv)])
        return np.array([math.atan2(float(np.linalg.norm(_cross(u, v, as_vector=True))), float(np.dot(u, v))) - row["value"]])

    def residual(self, x: Any) -> Any:
        np = _np()
        parts = [self.block(r, x) for r in self.rows]
        return np.concatenate(parts) if parts else np.zeros(0)

    def jacobian(self, x: Any, base: Any = None) -> Any:
        np = _np()
        f0 = self.residual(x) if base is None else base
        J = np.zeros((len(f0), self.n))
        for j in range(self.n):
            h = 1e-6 * max(1.0, abs(x[j]))
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (self.residual(xp) - self.residual(xm)) / (2 * h)
        return J


def _cross(u: Any, v: Any, as_vector: bool = False) -> Any:
    np = _np()
    if len(u) == 2:
        c = np.array([u[0] * v[1] - u[1] * v[0]])
        return c
    return np.cross(u, v)


# ----------------------------------------------------------------------------------------------------------------------- solve
def _solve_system(sys_: _System, tolerance: float, max_iterations: int) -> tuple[Any, int, float, bool]:
    np = _np()
    x = sys_.x0.copy()
    if sys_.n == 0 or not sys_.rows:
        r = sys_.residual(x)
        return x, 0, float(np.max(np.abs(r))) if len(r) else 0.0, True
    lam, f = 1e-3, sys_.residual(x)
    cost = float(f @ f)
    it = 0
    for it in range(1, max_iterations + 1):
        if float(np.max(np.abs(f))) <= tolerance * 1e-2:
            break
        J = sys_.jacobian(x, f)
        A, g = J.T @ J, J.T @ f
        accepted = False
        for _ in range(12):
            step = np.linalg.solve(A + lam * np.eye(sys_.n), -g)
            trial = x + step
            ft = sys_.residual(trial)
            ct = float(ft @ ft)
            if ct < cost:
                x, f, cost, accepted = trial, ft, ct, True
                lam = max(lam * 0.3, 1e-14)
                break
            lam *= 10.0
        if not accepted or float(np.max(np.abs(step))) < 1e-15:
            break
    worst = float(np.max(np.abs(f))) if len(f) else 0.0
    return x, it, worst, worst <= tolerance


def _rank(J: Any) -> tuple[int, Any, Any]:
    np = _np()
    if J.size == 0:
        return 0, np.zeros(0), np.zeros((J.shape[1], J.shape[1])) if J.ndim == 2 else np.zeros((0, 0))
    u, s, vt = np.linalg.svd(J, full_matrices=True)
    tol = 1e-7 * (s[0] if len(s) else 1.0)
    r = int((s > max(tol, 1e-12)).sum())
    return r, s, vt


def analyze(entities: Mapping[str, Mapping[str, Any]], constraints: Sequence[Mapping[str, Any]], *, fixed: Sequence[str] = (), tolerance: float = 1e-8,
            max_iterations: int = 200) -> dict[str, Any]:
    """Solve (without changing the input) and report: status, degrees of freedom (overall and per entity), redundant and conflicting constraints,
    unresolved references, and the solved coordinates."""
    np = _np()
    sys_ = _System(entities, constraints, fixed)
    x, iterations, worst, converged = _solve_system(sys_, tolerance, max_iterations)
    J = sys_.jacobian(x)
    rank, svals, vt = _rank(J) if sys_.n else (0, np.zeros(0), np.zeros((0, 0)))
    dof = sys_.n - rank
    per_entity = {}
    if dof > 0:
        null = vt[rank:]
        for ident, (a, b) in sys_.slices.items():
            w = float(np.sum(null[:, a:b] ** 2))
            if w > 1e-9:
                per_entity[ident] = {"free_dof_weight": round(w, 9)}
    # constraint by constraint, in order: does it add independent rows?
    basis: list[Any] = []
    per_constraint, redundant, conflicting = [], [], []
    at = 0
    f_all = sys_.residual(x)
    for row in sys_.rows:
        m = len(sys_.block(row, x))
        rows = J[at:at + m]
        resid = f_all[at:at + m]
        at += m
        added = 0
        for r in rows:
            v = r.copy()
            for b in basis:
                v -= float(v @ b) * b
            n0, n1 = float(np.linalg.norm(r)), float(np.linalg.norm(v))
            if n0 < 1e-12:
                continue
            if n1 / n0 > 1e-6:
                basis.append(v / n1)
                added += 1
        satisfied = float(np.max(np.abs(resid))) <= max(tolerance, 1e-8) if len(resid) else True
        status = "ok" if added == m else ("redundant" if satisfied else "conflicting")
        if added < m and satisfied is False:
            status = "conflicting"
        if status == "redundant":
            redundant.append(row["index"])
        if not satisfied:
            conflicting.append(row["index"])
        per_constraint.append({"index": row["index"], "kind": row["kind"], "independent_equations": added, "equations": m, "residual": float(np.max(np.abs(resid))) if len(resid) else 0.0,
                               "satisfied": satisfied, "status": status})
    degenerate = []
    for ident, (a, b) in sys_.slices.items():
        e = sys_.entities[ident]
        if e["kind"] == "segment":
            d = len(e["coords"]) // 2
            was = float(np.linalg.norm(np.array(e["coords"][d:]) - np.array(e["coords"][:d])))
            now = float(np.linalg.norm(x[a + d:b] - x[a:a + d]))
            if was > 0 and now < max(1e-9, 1e-6 * was):
                degenerate.append(ident)                       # solved only by collapsing a segment to a point: contradictory for a real segment
    if conflicting or not converged or degenerate:
        status = "conflicting"
    elif dof > 0:
        status = "under_constrained"
    elif redundant:
        status = "over_constrained_consistent"
    else:
        status = "well_constrained"
    solved = {i: [float(v) for v in x[a:b]] for i, (a, b) in sys_.slices.items()}
    return {"status": status, "converged": bool(converged), "iterations": iterations, "max_residual": worst, "variables": sys_.n, "equations": int(len(f_all)), "rank": rank,
            "dof": int(dof), "free_entities": per_entity, "constraints": per_constraint, "redundant": redundant, "conflicting": sorted(set(conflicting)), "degenerate": degenerate,
            "unresolved": sys_.unresolved, "solution": solved,
            "singular_values": [float(s) for s in svals[: min(12, len(svals))]],
            "scope": "local analysis at the solution found (first-order degrees of freedom); one solution is returned even if several exist"}
