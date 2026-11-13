"""A small, deterministic, dependency-free optimizer for bounded parametric problems (real, integer and choice variables, inequality constraints,
one or several objectives). It is application-independent: the caller supplies ``evaluate(point) -> {"objectives": [...], "violations": [...]}``.

What it does: seeded Latin-hypercube sampling of the design space, then a pattern (coordinate) search from the best feasible point with a shrinking step.
Several objectives are scalarised with the given weights for the search; the *Pareto front* of every feasible point evaluated is reported as well.

What it does not claim: it is not a global optimizer and proves nothing about optimality or infeasibility. ``status`` says what was observed:

  converged        a feasible point was found and the local search stopped because no neighbour improved it at the smallest step
  budget_exhausted a feasible point was found but the evaluation budget ended first
  infeasible       no feasible point was found in the budget (the least-violating point is returned, flagged infeasible); this is *not* a proof
  unbounded        an objective evaluated to a non-finite or absurdly large value at a feasible point (the problem as posed is not bounded)
  failed           every evaluation raised or returned invalid values
  timeout          the optional wall-clock limit ended the search first (the best point so far is returned, flagged; a timed run is not reproducible)

Same inputs and seed give the same result (the result carries a hash to check that).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

from .errors import ValidationError

MAX_BUDGET = 20_000
BACKEND = {"name": "spadra-deterministic-lhs-pattern-search", "version": "1", "kind": "derivative-free, local; no optimality or infeasibility proof"}
MAX_VARIABLES = 64
HUGE = 1e15


def _rng(seed: int):
    """A tiny deterministic generator (xorshift64*), so results do not depend on the Python or numpy version."""
    state = (int(seed) * 0x9E3779B97F4A7C15 + 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF or 1

    def nxt() -> float:
        nonlocal state
        state ^= state >> 12
        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 27
        return (((state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF) >> 11) / float(1 << 53)
    return nxt


def check_variables(variables: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(variables, Mapping) or not 1 <= len(variables) <= MAX_VARIABLES:
        raise ValidationError(f"an optimization has 1 to {MAX_VARIABLES} variables")
    out = {}
    for name in sorted(variables):
        v = variables[name]
        kind = v.get("kind", "real")
        if kind == "choice":
            ch = v.get("choices")
            if not isinstance(ch, (list, tuple)) or not 1 <= len(ch) <= 64:
                raise ValidationError(f"variable {name}: choices must list 1 to 64 values")
            out[name] = {"kind": "choice", "choices": list(ch)}
            continue
        if kind not in ("real", "integer"):
            raise ValidationError(f"variable {name}: kind must be real, integer or choice")
        lo, hi = v.get("min"), v.get("max")
        if not all(isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b) for b in (lo, hi)) or lo > hi:
            raise ValidationError(f"variable {name}: min and max must be finite numbers with min <= max")
        step = v.get("step")
        if step is not None and (not isinstance(step, (int, float)) or step <= 0):
            raise ValidationError(f"variable {name}: step must be positive")
        out[name] = {"kind": kind, "min": float(lo), "max": float(hi), "step": None if step is None else float(step)}
        if kind == "integer":
            out[name].update(min=math.ceil(lo), max=math.floor(hi), step=max(1.0, float(step or 1)))
            if out[name]["min"] > out[name]["max"]:
                raise ValidationError(f"variable {name}: no integer between min and max")
    return out


def _snap(spec: Mapping[str, Any], x: float) -> float:
    x = min(max(x, spec["min"]), spec["max"])
    if spec["kind"] == "integer":
        return float(round(x))
    if spec.get("step"):
        x = spec["min"] + round((x - spec["min"]) / spec["step"]) * spec["step"]
        x = min(x, spec["max"])
    return x


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def optimize(variables: Mapping[str, Mapping[str, Any]], evaluate: Callable[[dict[str, Any]], Mapping[str, Any]], *, senses: Sequence[str] = ("min",),
             weights: Sequence[float] | None = None, budget: int = 400, seed: int = 0, tolerance: float = 1e-9, feasibility_tolerance: float = 1e-9,
             start: Mapping[str, Any] | None = None, time_limit: float | None = None) -> dict[str, Any]:
    started = time.monotonic()
    if time_limit is not None and not (isinstance(time_limit, (int, float)) and not isinstance(time_limit, bool) and 0 < time_limit <= 3600):
        raise ValidationError("time_limit must be a number of seconds in (0, 3600]")
    deadline = None if time_limit is None else started + float(time_limit)
    timed_out = False
    specs = check_variables(variables)
    if not isinstance(budget, int) or not 1 <= budget <= MAX_BUDGET:
        raise ValidationError(f"budget must be 1..{MAX_BUDGET} evaluations")
    if not senses or any(s not in ("min", "max") for s in senses):
        raise ValidationError("each objective's sense is 'min' or 'max'")
    weights = list(weights or [1.0] * len(senses))
    if len(weights) != len(senses) or any(not isinstance(w, (int, float)) or w < 0 or not math.isfinite(w) for w in weights) or sum(weights) <= 0:
        raise ValidationError("weights must be non-negative, one per objective, and not all zero")
    sign = [1.0 if s == "min" else -1.0 for s in senses]
    names = sorted(specs)
    rand = _rng(seed)
    log: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    failures = 0
    unbounded = False

    def key(point: Mapping[str, Any]) -> str:
        return json.dumps([point[n] for n in names], sort_keys=True)

    def score(rec: Mapping[str, Any]) -> tuple:
        v = sum(rec["violations"])
        feasible = v <= feasibility_tolerance
        s = sum(w * sg * o for w, sg, o in zip(weights, sign, rec["objectives"]))
        return (0 if feasible else 1, s if feasible else v)

    def run(point: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal failures, unbounded, timed_out
        k = key(point)
        if k in cache:
            return cache[k]
        if len(log) >= budget:
            return None
        if deadline is not None and time.monotonic() > deadline:
            timed_out = True
            return None
        try:
            out = evaluate(dict(point))
            objs = [float(o) for o in out["objectives"]]
            viol = [max(0.0, float(v)) for v in out.get("violations", [])]
            if len(objs) != len(senses) or not all(map(math.isfinite, viol)):
                raise ValueError("an evaluation returned the wrong number of objectives or a non-finite violation")
            rec = {"point": dict(point), "objectives": objs, "violations": viol, "valid": True, "message": out.get("message", "")}
            if not all(math.isfinite(o) for o in objs) or any(abs(o) > HUGE for o in objs):
                rec["valid"] = False
                if sum(viol) <= feasibility_tolerance:
                    unbounded = True
        except Exception as exc:                                            # a point the model cannot build is an infeasible point, with its reason
            failures += 1
            rec = {"point": dict(point), "objectives": [math.inf] * len(senses), "violations": [math.inf], "valid": False, "message": f"{type(exc).__name__}: {exc}"[:300]}
        rec["index"] = len(log)
        cache[k] = rec
        log.append(rec)
        return rec

    def sample() -> dict[str, Any]:
        p = {}
        for n in names:
            s = specs[n]
            u = rand()
            p[n] = s["choices"][min(int(u * len(s["choices"])), len(s["choices"]) - 1)] if s["kind"] == "choice" else _snap(s, s["min"] + u * (s["max"] - s["min"]))
        return p

    # 1. the start point (if given) and the centre, then a Latin hypercube
    if start:
        run({n: (start[n] if specs[n]["kind"] == "choice" else _snap(specs[n], float(start[n]))) for n in names if n in start} | {n: (specs[n]["choices"][0] if specs[n]["kind"] == "choice" else _snap(specs[n], (specs[n]["min"] + specs[n]["max"]) / 2)) for n in names if n not in start})
    run({n: (specs[n]["choices"][0] if specs[n]["kind"] == "choice" else _snap(specs[n], (specs[n]["min"] + specs[n]["max"]) / 2)) for n in names})
    n_lhs = max(1, int(budget * 0.5) - len(log))
    strata = {n: sorted(range(n_lhs), key=lambda _i: rand()) for n in names}
    for i in range(n_lhs):
        p = {}
        for n in names:
            s = specs[n]
            u = (strata[n][i] + rand()) / n_lhs
            p[n] = s["choices"][min(int(u * len(s["choices"])), len(s["choices"]) - 1)] if s["kind"] == "choice" else _snap(s, s["min"] + u * (s["max"] - s["min"]))
        if run(p) is None:
            break
    # 2. pattern search from the best point
    converged = False
    ok = [r for r in log if r["valid"] or sum(r["violations"]) <= feasibility_tolerance]
    if ok and len(log) < budget and not timed_out:
        best = min(ok, key=score)
        frac = {n: 0.25 for n in names}
        while len(log) < budget and not timed_out:
            improved = False
            for n in names:
                s = specs[n]
                if s["kind"] == "choice":
                    candidates = [c for c in s["choices"] if c != best["point"][n]]
                else:
                    width = (s["max"] - s["min"]) * frac[n]
                    unit = s["step"] or 0.0
                    d = max(width, unit) if s["kind"] == "integer" or unit else width
                    candidates = [_snap(s, best["point"][n] + d), _snap(s, best["point"][n] - d)]
                    candidates = [c for c in candidates if c != best["point"][n]]
                for c in candidates:
                    trial = dict(best["point"])
                    trial[n] = c
                    rec = run(trial)
                    if rec is None:
                        break
                    if score(rec) < score(best) and (score(best)[0] != score(rec)[0] or abs(score(rec)[1] - score(best)[1]) > tolerance * max(1.0, abs(score(best)[1]))):
                        best, improved = rec, True
            if len(log) >= budget or timed_out:
                break
            if not improved:
                done = True
                for n in names:
                    s = specs[n]
                    if s["kind"] == "choice":
                        continue
                    floor = (s["step"] or (s["max"] - s["min"]) * 1e-9)
                    if (s["max"] - s["min"]) * frac[n] > floor * 1.0001 and s["max"] > s["min"]:
                        frac[n] /= 2
                        done = False
                if done:
                    converged = True
                    break
    feasible = [r for r in log if r["valid"] and sum(r["violations"]) <= feasibility_tolerance]
    if failures == len(log):
        status, best_rec = ("timeout" if timed_out else "failed"), None          # a limit that ended the search before anything succeeded is a timeout, not a model failure
    elif unbounded:
        status, best_rec = "unbounded", None
    elif feasible:
        best_rec = min(feasible, key=score)
        status = "timeout" if timed_out else "converged" if converged else "budget_exhausted"
    else:
        best_rec = min((r for r in log if math.isfinite(sum(r["violations"]))), key=lambda r: sum(r["violations"]), default=None)
        status = "timeout" if timed_out else "infeasible"
    pareto = []
    if len(senses) > 1:
        vecs = [[sg * o for sg, o in zip(sign, r["objectives"])] for r in feasible]
        for i, r in enumerate(feasible):
            if not any(_dominates(vecs[j], vecs[i]) for j in range(len(feasible)) if j != i):
                pareto.append(r)
        pareto.sort(key=lambda r: (r["objectives"], key(r["point"])))
    result = {"status": status, "best": best_rec, "feasible_found": len(feasible), "evaluations": len(log), "failed_evaluations": failures, "budget": budget, "seed": seed,
              "senses": list(senses), "weights": weights, "pareto": pareto[:200], "variables": specs,
              "history": log[:200], "proved": False,
              "note": "search result, not a proof of optimality or of infeasibility"}
    termination = {"converged": "no neighbour improved the best point at the smallest step size", "budget_exhausted": "the evaluation budget ended before the local search converged",
                   "infeasible": "no feasible point was found within the budget (not a proof that none exists)", "unbounded": "an objective was non-finite or absurdly large at a feasible point",
                   "failed": "every evaluation raised or returned invalid values", "timeout": "the wall-clock limit ended the search"}[status]
    result.update({"backend": dict(BACKEND), "termination": termination, "runtime_ms": round((time.monotonic() - started) * 1000.0, 3),
                   "tolerances": {"objective": tolerance, "feasibility": feasibility_tolerance, "violation_measure": "sum over constraints of the relative violation"},
                   "time_limit_s": time_limit, "timed_out": timed_out})
    result["result_sha256"] = hashlib.sha256(json.dumps([status, best_rec and best_rec["point"], len(log), seed], sort_keys=True, default=str).encode()).hexdigest()
    return result


def _distance(specs: Mapping[str, Mapping[str, Any]], names: Sequence[str], a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Distance between two points in the unit cube of the variables (choices count 0 if equal, 1 if not); 0 for identical points, 1 at most."""
    total = 0.0
    for n in names:
        s = specs[n]
        if s["kind"] == "choice":
            total += 0.0 if a[n] == b[n] else 1.0
        else:
            span = s["max"] - s["min"]
            total += 0.0 if span <= 0 else ((a[n] - b[n]) / span) ** 2
    return math.sqrt(total / len(names))


def alternatives(variables: Mapping[str, Mapping[str, Any]], evaluate: Callable[[dict[str, Any]], Mapping[str, Any]], *, senses: Sequence[str] = ("min",),
                 weights: Sequence[float] | None = None, count: int = 4, candidates: int = 300, seed: int = 0, top_fraction: float = 0.5,
                 feasibility_tolerance: float = 1e-9) -> dict[str, Any]:
    """A small set of *different* good designs, deterministic for a given seed.

    Candidates are a seeded Latin hypercube; each is evaluated, infeasible or unbuildable ones are dropped, the rest are ranked by the weighted objective, and from
    the best ``top_fraction`` of them ``count`` are chosen greedily: the best first, then each time the one farthest from those already chosen (max-min distance in
    the unit cube of the variables). So the options are all reasonably good and as different as the sample allows. They are samples: nothing here says no better
    or more different design exists, and no learned model is involved."""
    started = time.monotonic()
    specs = check_variables(variables)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 20:
        raise ValidationError("count must be 1 to 20 alternatives")
    if not isinstance(candidates, int) or isinstance(candidates, bool) or not count <= candidates <= MAX_BUDGET:
        raise ValidationError(f"candidates must be at least the number of alternatives and at most {MAX_BUDGET}")
    if not isinstance(top_fraction, (int, float)) or isinstance(top_fraction, bool) or not 0 < top_fraction <= 1:
        raise ValidationError("top_fraction must be above 0 and at most 1")
    if not senses or any(s not in ("min", "max") for s in senses):
        raise ValidationError("each objective's sense is 'min' or 'max'")
    weights = list(weights or [1.0] * len(senses))
    if len(weights) != len(senses) or any(not isinstance(w, (int, float)) or w < 0 or not math.isfinite(w) for w in weights) or sum(weights) <= 0:
        raise ValidationError("weights must be non-negative, one per objective, and not all zero")
    sign = [1.0 if s == "min" else -1.0 for s in senses]
    names = sorted(specs)
    rand = _rng(seed)
    strata = {n: sorted(range(candidates), key=lambda _i: rand()) for n in names}
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    failures = 0
    for i in range(candidates):
        point = {}
        for n in names:
            s = specs[n]
            u = (strata[n][i] + rand()) / candidates
            point[n] = s["choices"][min(int(u * len(s["choices"])), len(s["choices"]) - 1)] if s["kind"] == "choice" else _snap(s, s["min"] + u * (s["max"] - s["min"]))
        k = json.dumps([point[n] for n in names], sort_keys=True)
        if k in seen:                                                       # snapping to a step can repeat a point; evaluating it twice adds nothing
            continue
        seen.add(k)
        try:
            out = evaluate(dict(point))
            objs = [float(o) for o in out["objectives"]]
            viol = [max(0.0, float(v)) for v in out.get("violations", [])]
            if len(objs) != len(senses) or not all(map(math.isfinite, objs + viol)):
                raise ValueError("wrong number of objectives or a non-finite value")
            records.append({"point": point, "objectives": objs, "violations": viol, "key": k})
        except Exception:
            failures += 1
    evaluated = len(seen)
    feasible = [r for r in records if sum(r["violations"]) <= feasibility_tolerance]

    def score(r: Mapping[str, Any]) -> float:
        return sum(w * sg * o for w, sg, o in zip(weights, sign, r["objectives"]))
    ranked = sorted(feasible, key=lambda r: (score(r), r["key"]))
    chosen: list[dict[str, Any]] = []
    if ranked:
        pool = ranked[:max(count, int(math.ceil(len(ranked) * top_fraction)))]
        chosen = [dict(pool[0], distance=None)]
        rest = pool[1:]
        while len(chosen) < count and rest:
            far = [min(_distance(specs, names, r["point"], c["point"]) for c in chosen) for r in rest]
            best = max(range(len(rest)), key=lambda i: (far[i], -i))        # farthest from the chosen; ties go to the better-ranked
            chosen.append(dict(rest[best], distance=round(far[best], 9)))
            rest.pop(best)
    rank_of = {r["key"]: i + 1 for i, r in enumerate(ranked)}
    out_alts = [{"point": c["point"], "objectives": c["objectives"], "violations": c["violations"], "rank": rank_of[c["key"]], "distance": c["distance"]} for c in chosen]
    status = "failed" if failures == evaluated else "infeasible" if not feasible else "ok" if len(out_alts) == count else "fewer_than_requested"
    result = {"status": status, "alternatives": out_alts, "requested": count, "evaluations": evaluated, "feasible_found": len(feasible), "failed_evaluations": failures,
              "seed": seed, "senses": list(senses), "weights": weights, "top_fraction": top_fraction, "variables": specs, "proved": False,
              "note": "deterministic samples of the design space; not a proof that no better or more different design exists"}
    result.update({"backend": {**BACKEND, "name": "spadra-deterministic-lhs-maxmin-sampling", "kind": "seeded sampling; no optimality or infeasibility proof"},
                   "runtime_ms": round((time.monotonic() - started) * 1000.0, 3), "tolerances": {"feasibility": feasibility_tolerance}})
    result["result_sha256"] = hashlib.sha256(json.dumps([status, [a["point"] for a in out_alts], evaluated, seed], sort_keys=True, default=str).encode()).hexdigest()
    return result
