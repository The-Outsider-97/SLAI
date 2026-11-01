"""Native SGEO articulated motion: bounded joint validation, FK and constrained IK.

Joint transforms are local deltas after the authoritative static object transform.
Scene/world transform functions remain authoritative; no parallel scene graph is
created. IK uses SLAI's existing OptimizationSolver (SciPy nonlinear adapter).
Only fixed, revolute and prismatic joints are supported in this version.
"""
from __future__ import annotations

from copy import deepcopy
from math import hypot

from ..common.errors import ReferenceError, ValidationError, ResourceLimitError
from ..common.helpers import finite, vector
from ..common.ids import require_canonical_id, require_id
from ..common.transforms import IDENTITY, compose, from_axis_angle, from_translation

MAX_JOINTS = 256


def validate_joints(joints, objects) -> dict:
    """Validate native joint declarations and return object-id indexed joints."""
    if not isinstance(joints, list):
        raise ValidationError("joints must be a list")
    if len(joints) > MAX_JOINTS:
        raise ResourceLimitError("too many animation joints")
    by_object = {}
    identifiers = set()
    for joint in joints:
        if not isinstance(joint, dict) or set(joint) - {
            "id", "object_id", "kind", "axis", "lower", "upper", "value"
        } or not {"id", "object_id", "kind"} <= set(joint):
            raise ValidationError("invalid joint fields")
        jid, oid = require_canonical_id(joint["id"]), require_canonical_id(joint["object_id"])
        if jid in identifiers or oid in by_object:
            raise ValidationError("duplicate joint id or object")
        identifiers.add(jid)
        if oid not in objects:
            raise ReferenceError("joint refers to a missing object", context={"object_id": oid})
        kind = joint["kind"]
        if not isinstance(kind, str) or kind not in {"fixed", "revolute", "prismatic"}:
            raise ValidationError("unsupported joint kind")
        if kind == "fixed":
            if set(joint) - {"id", "object_id", "kind"}:
                raise ValidationError("fixed joint must not declare movement parameters")
        else:
            axis = vector(joint.get("axis"), 3, "joint axis")
            if hypot(*axis) < 1e-12:
                raise ValidationError("joint axis must be nonzero")
            lo = finite(joint.get("lower"), "joint lower limit", limit=1e9)
            hi = finite(joint.get("upper"), "joint upper limit", limit=1e9)
            val = finite(joint.get("value", 0), "joint default value", limit=1e9)
            if lo > hi or not lo <= val <= hi:
                raise ValidationError("invalid joint limits or default value")
        by_object[oid] = joint
    return by_object


def joint_transform(joint: dict, value: float | None = None) -> tuple:
    """One joint's local transform; relies on canonical SGEO transform helpers."""
    if joint["kind"] == "fixed":
        from ..common.transforms import IDENTITY
        return IDENTITY
    value = finite(joint.get("value", 0) if value is None else value,
                   "joint value", limit=1e9)
    if not joint["lower"] <= value <= joint["upper"]:
        raise ValidationError("joint value violates hard joint limits",
                              context={"joint_id": joint["id"], "value": value})
    axis = vector(joint["axis"], 3, "joint axis")
    norm = hypot(*axis)
    direction = tuple(x/norm for x in axis)
    if joint["kind"] == "revolute":
        return from_axis_angle(direction, value)
    return from_translation(*(x*value for x in direction))


def forward_kinematics(scene, joints: list, values: dict | None = None) -> dict:
    """Return detached world transforms after applying a validated joint pose."""
    objects = {obj["id"]: obj for obj in scene.objects}
    by_object = validate_joints(joints, objects)
    overrides = {} if values is None else values
    if not isinstance(overrides, dict) or set(overrides) - set(by_object):
        raise ValidationError("joint values must map declared object IDs")
    candidate = scene._detached_copy()
    for obj in candidate.objects:
        joint = by_object.get(obj["id"])
        if joint is None:
            continue
        transform = joint_transform(joint, overrides.get(obj["id"]))
        obj["transform"] = list(compose(obj["transform"], transform))
    candidate.validate()
    return candidate.world_transforms()


def solve_inverse_kinematics(
    scene, joints: list, end_effector: str, target,
    *, max_iterations: int = 200, tolerance: float = 1e-5,
) -> dict:
    """Bounded local IK using SLAI Optimization; no authored-state mutation.

    A non-convergent solve returns numerically_unresolved rather than falsely
    declaring mathematical infeasibility. Joint-value proposals are committed
    only by a separate explicit animation editing transaction.
    """
    oid = require_id(end_effector)
    goal = vector(target, 3, "end-effector target")
    tol = finite(tolerance, "IK tolerance", limit=1)
    if tol <= 0:
        raise ValidationError("IK tolerance must be positive")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 2000:
        raise ValidationError("invalid IK iteration limit")
    objects = {obj["id"]: obj for obj in scene.objects}
    if oid not in objects:
        raise ReferenceError("unknown IK end effector")
    by_object = validate_joints(joints, objects)
    chain = []
    cursor = oid
    while cursor is not None:
        if cursor in by_object and by_object[cursor]["kind"] != "fixed":
            chain.append(cursor)
        cursor = objects[cursor].get("parent")
    chain.reverse()
    if not chain:
        raise ValidationError("end effector has no movable joints in its hierarchy")
    if len(chain) > 32:
        raise ResourceLimitError("IK chain exceeds supported 32 degrees of freedom")
    originals = {item: by_object[item].get("value", 0) for item in chain}
    ancestry = []
    cursor = oid
    while cursor is not None:
        ancestry.append(cursor)
        cursor = objects[cursor].get("parent")
    ancestry.reverse()
    # IK evaluates a single kinematic chain; re-validating/deep-copying an
    # entire SGEO scene at every optimization iterate is unnecessary.
    # Callers supply a revision-checked detached scene snapshot.
    baseline = tuple((obj_id, tuple(objects[obj_id]["transform"]), by_object.get(obj_id))
                     for obj_id in ancestry)

    def pose(point):
        return {item: point["j%d" % i] for i, item in enumerate(chain)}

    def location(point):
        q = pose(point)
        transform = IDENTITY
        for obj_id, local, joint in baseline:
            transform = compose(transform, local)
            if joint is not None:
                transform = compose(transform, joint_transform(joint, q.get(obj_id)))
        return (transform[3], transform[7], transform[11])

    def residual(point):
        xyz = location(point)
        return tuple(xyz[i] - goal[i] for i in range(3))

    initial = {"j%d" % i: originals[item] for i,item in enumerate(chain)}
    def result(point, *, status, solver_status=None, iterations=None):
        errors = residual(point)
        error = hypot(*errors)
        return {
            "status": "solved" if error <= tol else status,
            "solver_status": solver_status, "iterations": iterations,
            "end_effector": oid, "target": list(goal),
            "actual": list(location(point)), "position_error": error,
            "tolerance": tol, "joint_values": pose(point),
        }
    first = result(initial, status="unsatisfied")
    if first["status"] == "solved":
        return first

    from src.agents.optimization.optimization_types import (
        Variable, VariableDomain, Objective, Constraint, ConstraintSense,
        OptimizationProblem, ProblemClass,
    )
    from src.agents.optimization.optimization_solver import OptimizationSolver
    variables = tuple(
        Variable(name="j%d" % i,
                 domain=VariableDomain.real(by_object[item]["lower"], by_object[item]["upper"]),
                 initial=originals[item])
        for i,item in enumerate(chain)
    )
    # A 2-DOF planar arm cannot satisfy 3 independent positional equalities
    # in SLSQP; minimize Cartesian residuals subject to joint limits instead.
    # Always check measured end-effector accuracy before reporting success.
    def objective_value(point):
        distance_squared = sum(component**2 for component in residual(point))
        regularization = 1e-10 * sum(
            (point[name]-value)**2 for name, value in initial.items()
        )
        return distance_squared + regularization

    objective = Objective(name="end_effector_error", function=objective_value)
    problem = OptimizationProblem(
        name="sgeo_inverse_kinematics",
        variables=variables, objectives=(objective,),
        constraints=(), problem_class=ProblemClass.NLP,
        deterministic=True,
    )
    from src.agents.optimization.utils.optimization_errors import (
        OptimizationError, SolverUnavailableError, OptimizationConfigurationError,
    )

    # Repeatable, bounded initializations escape stationary stretched-arm poses
    # where the squared Cartesian residual has vanishing first derivative.
    names = list(initial)
    seeds = [list(initial.values())]
    for direction in (1, -1):
        seed = []
        for i, object_id in enumerate(chain):
            joint = by_object[object_id]
            lo, hi = joint["lower"], joint["upper"]
            delta = min(0.25, (hi-lo)/8)
            seed.append(max(lo, min(hi, initial[names[i]] + direction*delta)))
        if seed not in seeds:
            seeds.append(seed)
    middle = [(by_object[oid]["lower"] + by_object[oid]["upper"])/2
              for oid in chain]
    if middle not in seeds:
        seeds.append(middle)

    best = first
    attempts = 0
    solver = OptimizationSolver()
    for seed in seeds[:4]:
        attempts += 1
        try:
            optimization = solver.solve_single_detailed(
                problem, solver="scipy_nonlinear",
                options={"x0": seed, "scipy_options": {"maxiter": max_iterations},
                         "time_limit_seconds": 5.0},
                allow_fallback=False,
            )
            solution = optimization.solution
        except SolverUnavailableError as exc:
            return {**best, "status": "unsupported",
                    "solver_status": type(exc).__name__,
                    "attempts": attempts}
        except OptimizationConfigurationError as exc:
            return {**best, "status": "invalid",
                    "solver_status": type(exc).__name__,
                    "attempts": attempts}
        except (OptimizationError, ArithmeticError, RuntimeError) as exc:
            best = {**best, "solver_status": type(exc).__name__}
            continue
        if solution is None or solution.primal_solution is None:
            continue
        try:
            proposal = result(
                solution.primal_solution, status="numerically_unresolved",
                solver_status=solution.status.value,
                iterations=solution.iteration_count,
            )
        except ValidationError:
            continue
        if proposal["position_error"] < best["position_error"]:
            best = proposal
        if best["status"] == "solved":
            return {**best, "attempts": attempts}
    return {**best, "attempts": attempts}


def retarget_joint_clip(source_scene, target_scene, source_clip_id, joint_map, *,
                        translation_scales=None):
    """Non-destructively retarget compatible scalar joint tracks to a new clip.

    Explicit source/target object correspondence is required. Revolute joint
    angles are transferred as default-pose deltas, correcting sign for exactly
    opposite local axes. Prismatic displacement may be scaled per joint.
    This intentionally rejects incompatible joint frames/topologies rather
    than claiming general character retargeting or contact preservation.
    """
    from ..common.ids import new_id
    from .animation import ANIMATION_V2_KEY
    source_scene.validate()
    target_scene.validate()
    source = source_scene.metadata.get(ANIMATION_V2_KEY)
    target = target_scene.metadata.get(ANIMATION_V2_KEY)
    if source is None or target is None:
        raise ValidationError("joint clip retargeting requires native v2 scenes")
    require_canonical_id(source_clip_id)
    if not isinstance(joint_map, dict) or not joint_map or len(joint_map) > MAX_JOINTS:
        raise ValidationError("joint_map must map 1..256 source IDs to target IDs")
    scales = {} if translation_scales is None else translation_scales
    if not isinstance(scales, dict) or set(scales) - set(joint_map):
        raise ValidationError("translation_scales must reference mapped source joints")
    source_joints = validate_joints(source.get("joints", []),
                                   {o["id"]: o for o in source_scene.objects})
    target_joints = validate_joints(target.get("joints", []),
                                   {o["id"]: o for o in target_scene.objects})
    if len(set(joint_map.values())) != len(joint_map):
        raise ValidationError("joint correspondence must be one-to-one")
    corrections = {}
    for sid, tid in joint_map.items():
        require_canonical_id(sid)
        require_canonical_id(tid)
        sj, tj = source_joints.get(sid), target_joints.get(tid)
        if sj is None or tj is None or sj["kind"] != tj["kind"] or sj["kind"] == "fixed":
            raise ValidationError("retargeting requires matching movable joint kinds")
        sa, ta = vector(sj["axis"], 3), vector(tj["axis"], 3)
        sn, tn = hypot(*sa), hypot(*ta)
        dot = sum(a*b for a,b in zip(sa,ta)) / (sn*tn)
        if abs(abs(dot)-1.0) > 1e-6:
            raise ValidationError("joint axes differ; explicit frame conversion is required")
        sign = 1 if dot > 0 else -1
        factor = finite(scales.get(sid, 1), "joint displacement scale", limit=1e6)
        if factor <= 0 or (sj["kind"] != "prismatic" and factor != 1):
            raise ValidationError("scaling is supported only for prismatic joint displacement")
        corrections[sid] = (tid, sj, tj, sign, factor)
    clips = [c for c in source.get("clips", []) if c["id"] == source_clip_id]
    if len(clips) != 1:
        raise ReferenceError("unknown source animation clip")
    original = clips[0]
    if not original["tracks"] or any(t["property"] != "joint" for t in original["tracks"]):
        raise ValidationError("retargeting currently supports joint-only source clips")
    if len(original["tracks"]) != len(joint_map):
        raise ValidationError("every source clip joint requires an explicit mapping")
    # SGEO clip tracks are instance-bound, not object-bound. A joint-only clip
    # may be transferred only when all its tracks share one target binding.
    # Therefore the supported subset is a single-joint clip. Multi-joint
    # motion uses multiple clips and an explicit choreographic relationship.
    if len(original["tracks"]) != 1 or len(joint_map) != 1:
        raise ValidationError("joint clips are single-target; retarget multi-joint rigs through individual clips")
    sid, (tid, sj, tj, sign, factor) = next(iter(corrections.items()))
    converted = deepcopy(original)
    converted["id"] = new_id()
    for track in converted["tracks"]:
        for keyframe in track["keyframes"]:
            current = finite(keyframe["value"], "retargeted joint value", limit=1e9)
            transformed = tj.get("value", 0) + sign * factor * (current - sj.get("value", 0))
            if not tj["lower"] <= transformed <= tj["upper"]:
                raise ValidationError("retargeted joint exceeds target joint limits")
            keyframe["value"] = transformed
            for name in ("in_tangent", "out_tangent"):
                if name in keyframe:
                    keyframe[name] = sign * factor * keyframe[name]
    return {"clip": converted, "target_object_id": tid,
            "source_clip_id": source_clip_id, "status": "proposed",
            "mapping": {sid: tid}}



__all__ = ["MAX_JOINTS", "validate_joints", "joint_transform",
           "forward_kinematics", "solve_inverse_kinematics", "retarget_joint_clip"]
