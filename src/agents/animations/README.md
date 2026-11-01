# SLAI Animation — native SGEO v1/v2 and agent integration

The Animation Agent coordinates native SGEO motion. **SGEO is authoritative**
for versioned animation data, scene transforms, constraints, evaluation,
serialization and atomic revisions. The agent does not maintain another
scene, animation evaluator, optimization backend or editing transaction log.
This document describes the implemented native capabilities and their explicit
limits. It is not a declaration of production readiness until regression tests
have executed successfully on supported Python environments.

## Native schema and compatibility

Existing `slai.animation.v1` remains supported without changing its original
fields, interpolation or evaluation semantics. `Scene.set_animation(spec)`
preserves the currently active schema. An intentional upgrade uses
`Scene.set_animation(spec, version=2)` or the lossless
`Scene.upgrade_animation(expected_revision=...)` operation, which stores the **exclusive** native
metadata key `slai.animation.v2`. Passing `None` removes the active
animation. Native SGEO `.sgeo` packages preserve the metadata key and
declare `animation-v1` or `animation-v2` in the manifest `requires` list,
separate from the overall file-format version (`0.2.0`).

V2 `{"duration": seconds, "tracks": [...]}` retains seconds as the time
unit. Track identity is the pair `(object_id, property)`; time keys must
increase strictly inside the animation duration. Supported properties:

| Property | Value | Interpolation |
| --- | --- | --- |
| `translation` | XYZ vector | step, linear, cubic |
| `rotation_angle` | scalar radians plus axis | step, linear, cubic |
| `rotation_quaternion` | scalar-first (w,x,y,z) | step, slerp |
| `scale` | XYZ vector | step, linear, cubic |
| `visibility` | boolean | step |
| `parameter.<name>` | independent numeric family parameter | step, linear, cubic (integer: step) |
| `joint` | angle (radians) or prismatic distance in SGEO units | step, linear, cubic |

Cubic tangents are explicit `in_tangent` and `out_tangent` on **every**
keyframe. They represent time derivatives, not normalized-segment tangents.
Native spline extrema are checked for joint/family bounds and singular scale.
Transforms use the shared right-handed, +Z-up, row-major affine convention.
Unspecified interpolation is linear; values outside the keyed interval are
clamped. `Scene.evaluate(time)` returns a detached `EvaluatedScene`, not a
mutation of authored SGEO data.

## Reusable clips and time-dependent dependencies

V2 optionally carries `clips` and `instances`. Each clip is a reusable
`{"id": uuid, "duration": seconds, "tracks": [...]}`; clip tracks omit
`object_id`. Each instance binds one clip to a compatible SGEO object:

```python
{"id": instance_uuid, "clip_id": clip_uuid,
 "object_id": object_uuid, "start": 0.0, "speed": 1.0}
```

An instance is active within its own bounded time window; inactive tracks do
not override authored base properties. The same clip may bind to several
objects without duplicating its authored keyframes. Multiple simultaneous
writers to the same target/property are rejected in this foundation, rather
than silently layered. Sequential, non-overlapping instances can address the
same target/property deterministically. Multi-layer blend policies and clip transitions are
planned as a separate, explicitly specified capability.

V2 `drivers` are acyclic, declarative scalar mechanical couplings:

```python
{"source_object_id": source_uuid, "source_property": "rotation_angle",
 "target_object_id": target_uuid, "target_property": "translation",
 "factor": 0.05, "offset": 0.0, "axis": [1, 0, 0]}
```

Drivers support scalar `rotation_angle`, `joint`, and independent numeric
family parameters as sources and `rotation_angle`, `joint`, or
axis-directed `translation` as targets. Duplicate targets, missing source
tracks and dependency cycles are rejected. A coupled *feedback* mechanism
requires a proper simultaneous equation system and is not modeled as a DAG.
The DAG is evaluated in a deterministic order; this implementation does not
promise compiled incremental graph caches.

## Constraints and articulated kinematics

Native v2 `constraints` are bounded records with UUID `id`, `kind`,
`object_id`, `movable` (boolean), optional positive `tolerance` and
optional `start`/`end` activation times. The present constraint solver
supports **root-object world positions** for these kinds:

- `position`: `target: [x, y, z]`.
- `distance`: `other_id` and nonnegative `distance`.
- `bounds`: `minimum` and `maximum` world-space vectors.
- `attachment`: `other_id` and relative `offset` vector.
- `alignment`: `other_id` and nonzero `axis`.

All are hard constraints. Compatible position-only systems can be solved
analytically, with contradictory tolerance intervals classified infeasible.
Isolated distance, bounds, attachment and alignment constraints also use
analytical projections, avoiding unnecessary nonlinear solver overhead.
Coupled problems use SLAI's existing `OptimizationSolver` nonlinear adapter
(SciPy). A result is accepted **only after its physical-space residuals meet
all individual tolerances**. Solver failure does not prove mathematical
infeasibility; the diagnostic reports `numerically_unresolved`. Invalid or
unsatisfied animated states raise SGEO `ValidationError` with a structured
`constraint_report` context. Solves are bounded by iteration and wall-time
budgets and operate only on detached scene data. Unsupported child-frame,
contact, look-at, and collision constraints are not misrepresented as solved.

V2 `joints` consist of fixed, revolute and prismatic declarations with UUID
`id`, `object_id`, `kind`, and, for movable joints, `axis`,
`lower`, `upper`, and optional default `value`. Exactly one joint
may attach to an object. Native `joint` tracks bind animated values to these
declarations, with enforced limits and shared SGEO transforms. Joint FK uses
the same scene hierarchy, never a separate skeleton database.

`Scene.forward_kinematics(values=None)` returns detached world transforms.
`Scene.solve_inverse_kinematics(end_effector, target, ...)` performs
position-target IK for one end effector, using a bounded local nonlinear
optimization against supported joint limits. Deterministic restarts reduce
stagnation at singular poses. The returned structured proposal identifies
`solved`, `numerically_unresolved`, `unsupported`, or `invalid` as appropriate,
with the measured target error,
iterations and proposed joint values. It never silently commits motion or
claims global optimality.

`Scene.apply_inverse_kinematics(..., expected_revision=...)` optionally
commits a **static, unanimated** verified pose through the existing
`set_animation` atomic revision mechanism. This intentionally refuses
assemblies with authored tracks, drivers, constraints or clip instances;
otherwise the solved static pose might not match the actual evaluated motion.
Other animation edits continue through the normal agent batch editor.

## Application-independent agent operations

The existing Animation Agent/SGEO adapter exposes the native provider
operations `forward_kinematics`, `solve_ik` (nonmutating) and `apply_ik`
(static pose commit) alongside established evaluation, edit, history and
playback operations. `apply_ik` requires an optimistic
`expected_revision`; completed edits are recorded in bounded
`AnimationMemory` as operational history, not a second authoritative undo
store.

```python
from src.utils.spatial_formats.sgeo.scene import Scene
from src.utils.spatial_formats.adapters.slai_animation import SGEOAnimationAdapter

scene = Scene()
adapter = SGEOAnimationAdapter(scene)
# Add scene objects and a valid slai.animation.v2 joint declaration first.
# Then bind adapter through the existing AnimationAgent.bind_scene(...)
# and dispatch:
# {"operation": "solve_ik", "scene_id": scene.document_id,
#  "payload": {"end_effector": object_uuid, "target": [1, 1, 0]}}
```

## Verification and remaining boundaries

Run from the repository root, with the repository's environment and
the existing optimization dependencies (NumPy and SciPy) installed:

```sh
python -m compileall -q src/utils/spatial_formats src/agents/animations src/agents/animation_agent.py
python -m pytest tests/spatial_formats/test_sgeo_animation_v2.py tests/spatial_formats/test_sgeo_animation_p0_p2.py tests/spatial_formats/test_sgeo_temporal_and_box_boolean.py tests/agents/animations -v -W error
python -m pytest tests/spatial_formats -v -W error
```

No Blender-like viewport, rendered playback, animation layers, skeletal
skinning, arbitrary graph feedback, dynamic physics, orientation IK, full
contact/collision solving, learned motion synthesis, or authoritative
undo/redo subsystem is claimed here. Those need appropriate native models
and separate verification. All user-visible success claims depend on actual
executed tests and numerical regression evidence.
