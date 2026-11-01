# SLAI Animation — initial integration foundation

This first implementation reuses SGEO's validated `slai.animation.v1` tracks and read-only scene evaluation. It does **not** claim full native animation, IK, rigging, retargeting, choreography, or learned motion. Those are deferred until validated and implemented.

Implemented: typed request, read-only evaluation, revision-checked edits to SGEO animation, simple bounded edit metadata, timebase/interpolation/playback utilities, config binding, and BaseAgent facade.

## Scope
SGEO is authoritative for scene animation data and transactions. Spatial handles spatial queries. Simulation handles physical modeling. The facade refuses unsupported operations. UI integration is intentionally absent.

## Quickstart
Pass an SGEO Scene to `AnimationAgent.bind_scene(scene)`, then call `perform_task({"operation": "evaluate", "scene_id": scene.document_id, "payload": {"time": 0.5}})`. Write operations require the existing SGEO v1 track schema.

Caveat: scene bindings are process-local, not a persisted scene registry; deployment needs authorization and lifecycle integration. Direct scene mutation outside this facade can invalidate history. This is a foundation, **not** production-ready.


## SGEO v2 native composition (implementation branch)

The native `slai.animation.v2` metadata document retains `duration` and
`tracks` and optionally accepts `clips`, `instances`, `layers` and
`events`. The original `slai.animation.v1` schema is unchanged.

- A clip has `id`, `duration` and `tracks`. Each track's `object_id`
  is a clip-local source binding, not necessarily a scene object.
- An instance has `clip_id` and `bindings`, mapping every clip-local
  source identifier to an existing SGEO scene object. Optional `layer_id`,
  `start`, `speed` (nonzero, negative for reverse), `loop` and
  `enabled` determine when the clip contributes.
- A layer has `id`, optional `weight` between 0 and 1, and `enabled`.
  Layers currently use **ordered override blending only**; additive
  blending, masks, transitions and arbitrary time-warp curves are not
  implemented. Layer weights blend translation, scale and scalar parameters
  componentwise; quaternion tracks use SLERP. Visibility is discrete.
- Events have `name` and `time`. `animation_events` retrieves markers
  without executing any side effects.
- `generate_linear_motion` adds a deterministic translation track to an
  existing v2 animation via the existing atomic animation-command path.
  It does not perform obstacle avoidance, inverse kinematics or dynamics.

The Animation Agent supports the existing `edit_batch` operation for
`add_clip`, `update_clip`, `delete_clip`, `add_instance`,
`update_instance`, `delete_instance`, `add_layer`, `update_layer`,
`delete_layer`, `add_event` and `delete_event`. Each operation uses
the corresponding singular payload key (`clip`, `instance`, `layer`,
`event`) or a zero-based `<singular>_index`. The SGEO validator
remains authoritative for references, type compatibility and resource limits.

### Unimplemented capabilities

Skeletal deformation, rigging, motion retargeting, choreography, nonlinear
constraints, physically validated trajectories, AI-learned motion, arbitrary
time warping and production-grade undo/redo are **not implemented** by these
changes. A successful native animation evaluation does not imply dynamic
feasibility. Production use requires the complete relevant regression suite
and representative workload benchmarks to pass.
