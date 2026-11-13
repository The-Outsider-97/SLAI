# SLAI Animation — initial integration foundation

This first implementation reuses SGEO's validated `slai.animation.v1` tracks and read-only scene evaluation. It does **not** claim full native animation, IK, rigging, retargeting, choreography, or learned motion. Those are deferred until validated and implemented.

Implemented: typed request, read-only evaluation, revision-checked edits to SGEO animation, simple bounded edit metadata, timebase/interpolation/playback utilities, config binding, and BaseAgent facade.

## Scope
SGEO is authoritative for scene animation data and transactions. Spatial handles spatial queries. Simulation handles physical modeling. The facade refuses unsupported operations. UI integration is intentionally absent.

## Quickstart
Pass an SGEO Scene to `AnimationAgent.bind_scene(scene)`, then call `perform_task({"operation": "evaluate", "scene_id": scene.document_id, "payload": {"time": 0.5}})`. Write operations require the existing SGEO v1 track schema.

Caveat: scene bindings are process-local, not a persisted scene registry; deployment needs authorization and lifecycle integration. Direct scene mutation outside this facade can invalidate history. This is a foundation, **not** production-ready.
