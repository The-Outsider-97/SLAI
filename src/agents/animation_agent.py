"""General-purpose SLAI Animation Agent: transactional orchestration, not a second animation engine.

This is an orchestration facade for the existing Animation subsystem and scene
providers.  The provider owns scene state, animation documents, geometry,
read-only evaluations and revision commits.  ``AnimationMemory`` owns *only*
bounded operational edit summaries. ``BaseAgent.shared_memory`` owns inter-agent
result/event publication. Neither memory is an authoritative scene store.

Important boundaries:
* A bound scene is a trusted process-local provider; callers cannot inject a
  Python scene through the request envelope.
* All edits flow through ``animation_commands`` and a provider-owned reentrant
  lock, not direct mutation of SGEO metadata.
* Evaluations delegate to ``animation_evaluation``; returned summaries contain
  identifiers/timestamps, never duplicated scene geometry.
* No transparent retry of edits; repeated commits are not idempotent.
* The AnimationAgent does not claim learned generation, IK, physics, skinning,
  retargeting or full motion synthesis when no backend exists.
* This facade deliberately does NOT advertise durable checkpoint support:
  scene state belongs to the provider, and the operational memory is ephemeral.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import Any, cast

from src.utils.spatial_formats.sgeo.animation import ANIMATION_KEY, ANIMATION_V2_KEY # type: ignore
from .base_agent import BaseAgent
from .base.utils.main_config_loader import get_config_section
from .animations.animation_commands import SUPPORTED_EDITS, apply_animation_batch, update_animation
from .animations.animation_evaluation import (
    evaluate_animation,
    evaluate_animation_frames,
    evaluate_animation_many,
)
from .animations.animation_memory import AnimationEditRecord, AnimationMemory
from .animations.animation_protocols import (
    get_animation_edit_lock,
    is_animation_scene_provider,
    is_read_only_animation_scene_provider,
)
from .animations.animation_types import AnimationRequest, EvaluationResult
from .animations.timeline.playback import Playback
from .animations.timeline.timebase import Timebase
from .animations.utils.animation_errors import *
from .animations.utils.animation_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Animation Agent")
printer = PrettyPrinter()

_READ_OPERATIONS = frozenset({
    "evaluate", "evaluate_many", "evaluate_frames", "get_animation", "recent_edits", "edit_counts",
    "capabilities", "status", "playback_status", "forward_kinematics", "solve_ik",
})
_TRANSPORT_OPERATIONS = frozenset({"play", "pause", "stop", "seek", "advance", "set_playback_rate", "set_loop"})
_BATCH_OPERATION = "edit_batch"
_OPERATIONS = frozenset(SUPPORTED_EDITS | _READ_OPERATIONS | _TRANSPORT_OPERATIONS | {_BATCH_OPERATION})


@dataclass(frozen=True, slots=True)
class _AgentPolicy:
    """Validated agent configuration; values are loaded by BaseAgent from agents_config.yaml."""

    enabled: bool
    publish_results: bool
    publish_errors: bool
    publish_events: bool
    shared_memory_ttl_seconds: int
    max_bound_scenes: int
    max_recent_edits: int
    max_result_objects: int
    max_result_samples: int
    max_history_records: int
    max_request_id_length: int
    result_key_prefix: str
    error_key_prefix: str
    event_channel: str

    @classmethod
    def load(cls, overrides: Mapping[str, Any] | None = None) -> "_AgentPolicy":
        # This is the BaseAgent config binding (agents_config.yaml), never the
        # Animation subsystem configuration facade. The agent does not inspect
        # or load any subsystem YAML, directly or indirectly.
        raw = get_config_section("animation_agent")
        if not isinstance(raw, Mapping):
            raise AnimationConfigurationError("animation_agent must be a mapping in agents_config.yaml")
        values = dict(raw)
        if overrides is not None:
            if not isinstance(overrides, Mapping):
                raise AnimationConfigurationError("animation agent overrides must be a mapping")
            unknown = set(overrides) - set(cls.__dataclass_fields__)
            if unknown:
                raise AnimationConfigurationError(
                    "unknown animation agent configuration overrides", context={"unknown": sorted(unknown)}
                )
            values.update(overrides)
        unknown = set(values) - set(cls.__dataclass_fields__)
        missing = set(cls.__dataclass_fields__) - set(values)
        if unknown or missing:
            raise AnimationConfigurationError(
                "invalid animation_agent configuration keys",
                context={"missing": sorted(missing), "unknown": sorted(unknown)},
            )
        for key in ("enabled", "publish_results", "publish_errors", "publish_events"):
            if type(values[key]) is not bool:
                raise AnimationConfigurationError(f"animation_agent.{key} must be bool")
        for key in (
            "shared_memory_ttl_seconds", "max_bound_scenes", "max_recent_edits", "max_result_objects",
            "max_result_samples", "max_history_records", "max_request_id_length",
        ):
            if type(values[key]) is not int or values[key] < 1:
                raise AnimationConfigurationError(f"animation_agent.{key} must be a positive integer")
        for key in ("result_key_prefix", "error_key_prefix", "event_channel"):
            try:
                ensure_non_empty_string(values[key], key, strip=True)
            except AnimationValidationError as exc:
                raise AnimationConfigurationError(f"animation_agent.{key} must be non-empty", cause=exc) from exc
            if len(values[key]) > 128:
                raise AnimationConfigurationError(f"animation_agent.{key} is too long")
        return cls(**values)


class AnimationAgent(BaseAgent):
    """SLAI agent facade over deterministic animation providers.

    Public lifecycle:
      1. Construct with the usual BaseAgent dependencies.
      2. ``bind_scene(scene)`` with a trusted, validated scene provider.
      3. ``perform_task(AnimationRequest | mapping)`` or inherited ``execute``.
      4. ``unbind_scene(scene_id)`` when the external owner releases a scene.

    ``shared_memory`` is inherited from BaseAgent; the agent never creates a
    second, unrelated SharedMemory instance. AnimationMemory is an independent,
    bounded, non-durable operational history. No scene geometry is stored there.
    """

    AGENT_KEY = "animation_agent"
    CHECKPOINTING_SUPPORTED = False
    execution_is_idempotent = False

    def __init__(
        self,
        shared_memory: Any = None,
        agent_factory: Any = None,
        config: Mapping[str, Any] | None = None,
        *,
        checkpoint_manager: Any = None,
    ) -> None:
        # BaseAgent owns SharedMemory, generic configuration, runtime lifecycle,
        # logging, execution health and checkpoint manager integration.
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            checkpoint_manager=checkpoint_manager,
        )
        self._policy = _AgentPolicy.load(config)
        self.agent_config = {key: getattr(self._policy, key) for key in _AgentPolicy.__dataclass_fields__}
        # An explicit capacity prevents AnimationMemory's standalone default
        # configuration loader from being invoked by the agent.
        self.animation_memory = AnimationMemory(capacity=self._policy.max_recent_edits)
        self._scene_lock = RLock()
        self._scenes: dict[str, Any] = {}
        self._playbacks: dict[str, Playback] = {}
        self._stats_lock = RLock()
        self._counts: Counter[str] = Counter()
        self._failures: Counter[str] = Counter()
        self._last_request_id: str | None = None
        self._last_latency_ms: float | None = None
        logger.info(
            "Animation Agent initialized | enabled=%s memory_capacity=%d",
            self._policy.enabled, self.animation_memory.capacity,
        )

    # ------------------------------------------------------------------
    # Trusted scene-provider binding: not exposed as a remote operation.
    # ------------------------------------------------------------------
    def bind_scene(self, scene: Any) -> str:
        """Bind one trusted, externally-owned scene without copying its state."""
        if not is_read_only_animation_scene_provider(scene):
            raise AnimationUnsupportedOperationError("scene does not implement read-only animation protocol")
        scene_id = ensure_non_empty_string(getattr(scene, "document_id", None), "document_id", strip=True).strip()
        if scene.revision < 1:
            raise AnimationStateError("scene revision must be positive")
        with self._scene_lock:
            registered = self._scenes.get(scene_id)
            if registered is not None and registered is not scene:
                raise AnimationStateError(
                    "different scene provider already bound under this document_id",
                    context={"scene_id": scene_id},
                )
            if registered is None and len(self._scenes) >= self._policy.max_bound_scenes:
                raise AnimationStateError("maximum bound scene count reached")
            self._scenes[scene_id] = scene
        logger.info("animation scene bound | scene=%s revision=%d", scene_id, scene.revision)
        return scene_id

    def unbind_scene(self, scene_id: str) -> bool:
        """Forget the binding/playback controller; leave provider and edit history intact."""
        key = ensure_non_empty_string(scene_id, "scene_id", strip=True).strip()
        with self._scene_lock:
            existed = self._scenes.pop(key, None) is not None
            self._playbacks.pop(key, None)
        if existed:
            logger.info("animation scene unbound | scene=%s", key)
        return existed

    def _scene(self, scene_id: str | None) -> Any:
        if scene_id is None:
            raise AnimationValidationError("scene_id is required for this operation")
        with self._scene_lock:
            scene = self._scenes.get(scene_id)
        if scene is None:
            raise AnimationStateError("scene is not bound", context={"scene_id": scene_id})
        return scene

    # ------------------------------------------------------------------
    # Public native evaluation API for in-process consumers. perform_task
    # returns bounded metadata, not duplicated scene snapshots.
    # ------------------------------------------------------------------
    def evaluate_state(self, scene_id: str, seconds: float, *, expected_revision: int | None = None) -> Any:
        """Return the provider's detached evaluated state, without serialization."""
        if not self._policy.enabled:
            raise AnimationStateError("Animation Agent is disabled by animation_agent.enabled")
        return evaluate_animation(self._scene(scene_id), seconds, expected_revision=expected_revision)

    # ------------------------------------------------------------------
    # BaseAgent request dispatch
    # ------------------------------------------------------------------
    def perform_task(self, task_data: AnimationRequest | Mapping[str, Any]) -> dict[str, Any]:
        """Execute one validated request; errors propagate with their domain taxonomy.

        Animation commands are non-idempotent; BaseAgent must not transparently
        retry them. The request envelope never carries an untrusted scene object.
        """
        if isinstance(task_data, AnimationRequest):
            request = task_data
        else:
            request = AnimationRequest.from_mapping(task_data)
        if len(request.request_id) > self._policy.max_request_id_length:
            raise AnimationValidationError("request_id exceeds animation_agent.max_request_id_length")
        op = request.operation
        started = perf_counter()
        succeeded = False
        try:
            if op not in _OPERATIONS:
                raise AnimationUnsupportedOperationError(
                    "animation operation is not implemented", context={"operation": op}
                )
            if op not in {"capabilities", "status"} and not self._policy.enabled:
                raise AnimationStateError("Animation Agent is disabled by animation_agent.enabled")
            if op == "capabilities":
                result = self._capabilities(request)
            elif op == "status":
                result = self._status(request)
            elif op == "recent_edits":
                result = self._recent_edits(request)
            elif op == "edit_counts":
                result = self._edit_counts(request)
            else:
                scene = self._scene(request.scene_id)
                if op == "evaluate":
                    result = self._evaluate_one(request, scene)
                elif op == "evaluate_many":
                    result = self._evaluate_many(request, scene)
                elif op == "evaluate_frames":
                    result = self._evaluate_frames(request, scene)
                elif op == "get_animation":
                    result = self._get_animation(request, scene)
                elif op in {"forward_kinematics", "solve_ik"}:
                    result = self._kinematics(request, scene)
                elif op == _BATCH_OPERATION or op in SUPPORTED_EDITS:
                    result = self._edit(request, scene)
                else:
                    result = self._playback_operation(request, scene)
            succeeded = True
            self._publish_result(request, result)
            return result
        except Exception as exc:
            logger.warning(
                "animation request failed | operation=%s request_id=%s error_type=%s",
                op, request.request_id, type(exc).__name__,
            )
            self._publish_failure(request, exc)
            raise
        finally:
            with self._stats_lock:
                self._counts[op] += 1
                if not succeeded:
                    self._failures[op] += 1
                self._last_request_id = request.request_id
                self._last_latency_ms = (perf_counter() - started) * 1000.0

    def predict(self, state: Any, context: Any = None) -> dict[str, Any]:
        """Preserve BaseAgent/factory predict-style dispatch without a second parser."""
        return self.perform_task(state)

    # ------------------------------------------------------------------
    # Scene reads
    # ------------------------------------------------------------------
    def _evaluate_one(self, request: AnimationRequest, scene: Any) -> dict[str, Any]:
        time = request.payload.get("time")
        if isinstance(time, bool) or not isinstance(time, (int, float)):
            raise AnimationValidationError("evaluate requires a numeric time")
        scene_id = request.scene_id
        if scene_id is None:
            raise AnimationValidationError("scene_id is required for evaluate")
        snapshot = evaluate_animation(scene, float(time), expected_revision=request.expected_revision)
        objects = snapshot.ids()
        if not isinstance(objects, (list, tuple)):
            raise AnimationStateError("evaluated scene ids() must return an ordered sequence")
        ids = tuple(objects)
        if any(not isinstance(identifier, str) for identifier in ids):
            raise AnimationStateError("evaluated scene returned non-string object identifiers")
        truncated = len(ids) > self._policy.max_result_objects
        result = EvaluationResult(
            request_id=request.request_id,
            scene_id=scene_id,
            revision=snapshot.revision,
            time=snapshot.time,
            object_ids=ids[: self._policy.max_result_objects],
        ).to_dict()
        if truncated:
            result.update({"objects_truncated": True, "object_count": len(ids)})
        return result

    def _batch_result(self, request: AnimationRequest, batch: Any) -> dict[str, Any]:
        # Batch states stay with their existing scene provider; only summary
        # information crosses this agent's structured response boundary.
        samples = [
            {"time": state.time, "revision": state.revision}
            for state in batch.states[: self._policy.max_result_samples]
        ]
        return {
            "status": "ok", "request_id": request.request_id, "revision": batch.revision,
            "sample_count": batch.sample_count, "elapsed_ms": batch.elapsed_ms,
            "samples": samples, "samples_truncated": batch.sample_count > len(samples),
        }

    def _evaluate_many(self, request: AnimationRequest, scene: Any) -> dict[str, Any]:
        payload = request.payload
        if "times" not in payload:
            raise AnimationValidationError("evaluate_many requires times")
        lock_scene = payload.get("lock_scene", False)
        batch = evaluate_animation_many(
            scene, payload["times"], expected_revision=request.expected_revision,
            lock_scene=lock_scene,
        )
        return self._batch_result(request, batch)

    def _evaluate_frames(self, request: AnimationRequest, scene: Any) -> dict[str, Any]:
        payload = request.payload
        if "start_frame" not in payload or "end_frame" not in payload:
            raise AnimationValidationError("evaluate_frames requires start_frame and end_frame")
        num, den = payload.get("fps_num"), payload.get("fps_den")
        if (num is None) != (den is None):
            raise AnimationValidationError("fps_num and fps_den must be supplied together")
        timebase = Timebase(cast(int, num), cast(int, den)) if num is not None else None
        batch = evaluate_animation_frames(
            scene, payload["start_frame"], payload["end_frame"],
            timebase=timebase, expected_revision=request.expected_revision,
            lock_scene=payload.get("lock_scene", False),
        )
        return self._batch_result(request, batch)

    def _get_animation(self, request: AnimationRequest, scene: Any) -> dict[str, Any]:
        if not is_animation_scene_provider(scene):
            raise AnimationUnsupportedOperationError("scene does not expose animation metadata")
        lock = get_animation_edit_lock(scene)
        if lock is None:
            raise AnimationUnsupportedOperationError("consistent animation reads require provider edit lock")
        with lock:
            if request.expected_revision is not None and scene.revision != request.expected_revision:
                raise AnimationRevisionConflictError("scene revision mismatch")
            revision = scene.revision
            document = deepcopy(scene.metadata.get(ANIMATION_V2_KEY, scene.metadata.get(ANIMATION_KEY)))
        return {"status": "ok", "request_id": request.request_id, "revision": revision, "animation": document}

    def _kinematics(self, request: AnimationRequest, scene: Any) -> dict[str, Any]:
        """Route FK/IK to the authoritative SGEO scene, never solve in agent."""
        fk = getattr(scene, "forward_kinematics", None)
        ik = getattr(scene, "solve_inverse_kinematics", None)
        if not callable(fk) or not callable(ik):
            raise AnimationUnsupportedOperationError(
                "bound provider does not expose the native articulated SGEO operations"
            )
        scene_id = request.scene_id
        if request.operation == "forward_kinematics":
            lock = get_animation_edit_lock(scene)
            if lock is None:
                raise AnimationUnsupportedOperationError(
                    "consistent kinematics requires the provider's edit lock"
                )
            with lock:
                if request.expected_revision is not None and scene.revision != request.expected_revision:
                    raise AnimationRevisionConflictError("scene revision mismatch")
                values = request.payload.get("values")
                matrices = fk(values)
                revision = scene.revision
            keys = sorted(matrices)
            limit = self._policy.max_result_objects
            return {
                "status": "ok", "request_id": request.request_id,
                "scene_id": scene_id, "revision": revision,
                "object_count": len(keys),
                "transforms_truncated": len(keys) > limit,
                "transforms": {key: list(matrices[key]) for key in keys[:limit]},
            }
        payload = request.payload
        if "end_effector" not in payload or "target" not in payload:
            raise AnimationValidationError("solve_ik requires end_effector and target")
        result = ik(
            payload["end_effector"], payload["target"],
            expected_revision=request.expected_revision,
            max_iterations=payload.get("max_iterations", 200),
            tolerance=payload.get("tolerance", 1e-5),
        )
        return {"request_id": request.request_id, "scene_id": scene_id, **result}

    # ------------------------------------------------------------------
    # Transactional writes; SGEO/provider is authoritative
    # ------------------------------------------------------------------
    def _edit(self, request: AnimationRequest, scene: Any) -> dict[str, Any]:
        if not is_animation_scene_provider(scene):
            raise AnimationUnsupportedOperationError("scene is read-only")
        lock = get_animation_edit_lock(scene)
        if lock is None:
            raise AnimationUnsupportedOperationError("scene does not expose a reentrant edit lock")
        # Outer lock gives an accurate revision_before for operational memory.
        # Commands acquires the *same* provider-owned reentrant lock to stage
        # and commit the edit. No duplicated transaction manager is created.
        with lock:
            before = scene.revision
            if request.operation == _BATCH_OPERATION:
                if "commands" not in request.payload:
                    raise AnimationValidationError("edit_batch requires commands")
                result = apply_animation_batch(
                    scene, request.payload["commands"],
                    expected_revision=request.expected_revision,
                    dry_run=request.dry_run,
                )
            else:
                result = update_animation(
                    scene, request.operation, request.payload,
                    expected_revision=request.expected_revision,
                    dry_run=request.dry_run,
                )
            after = result["revision"]
        response = {"status": "ok", "request_id": request.request_id, **result}
        if not request.dry_run:
            scene_id = request.scene_id
            if scene_id is None:
                raise AnimationValidationError("scene_id is required for this operation")
            # Memory failure must NOT turn an already committed provider edit
            # into an exception that callers might retry. BaseAgent provides a
            # standard degraded-health channel for such secondary failures.
            self._run_optional_runtime_operation(
                "telemetry", "animation_memory.record",
                lambda: self.animation_memory.record(
                    AnimationEditRecord(request.operation, scene_id, before, after)
                ),
            )
            self._run_optional_runtime_operation(
                "telemetry", "playback.reconcile",
                lambda: self._refresh_playback_duration(scene_id, scene),
            )
        return response

    # ------------------------------------------------------------------
    # Headless playback state.  No wall-clock scheduler / rendering loop.
    # ------------------------------------------------------------------
    def _playback_for(
        self, scene_id: str, scene: Any, *, expected_revision: int | None = None
    ) -> Playback:
        if not is_animation_scene_provider(scene):
            raise AnimationUnsupportedOperationError("playback needs scene animation metadata")
        lock = get_animation_edit_lock(scene)
        if lock is None:
            raise AnimationUnsupportedOperationError("playback requires consistent animation metadata access")
        with lock:
            if expected_revision is not None and scene.revision != expected_revision:
                raise AnimationRevisionConflictError("scene revision mismatch")
            spec = scene.metadata.get(ANIMATION_V2_KEY, scene.metadata.get(ANIMATION_KEY))
            if not isinstance(spec, Mapping):
                raise AnimationStateError("scene has no animation document for playback")
            duration = ensure_finite_non_negative(spec.get("duration"), "duration")
            if duration <= 0:
                raise AnimationStateError("animation duration must be positive")
        current = self._playbacks.get(scene_id)
        if current is None:
            current = Playback(duration=duration)
            self._playbacks[scene_id] = current
        elif current.duration != duration:
            current.duration = duration
            current.seek(min(current.position, duration))
        return current

    def _refresh_playback_duration(self, scene_id: str, scene: Any) -> None:
        # A successful clear_animation removes a playback controller.  Other
        # edits reconcile duration on the next explicit playback request.
        if not is_animation_scene_provider(scene):
            return
        with self._scene_lock:
            if scene_id in self._playbacks and not any(key in scene.metadata for key in (ANIMATION_KEY, ANIMATION_V2_KEY)):
                del self._playbacks[scene_id]

    def _playback_operation(self, request: AnimationRequest, scene: Any) -> dict[str, Any]:
        op = request.operation
        payload = request.payload
        if request.dry_run and op != "playback_status":
            raise AnimationUnsupportedOperationError("dry_run is not supported for playback mutations")
        scene_id = request.scene_id
        if scene_id is None:
            raise AnimationValidationError("playback requires scene_id")
        with self._scene_lock:
            player = self._playback_for(
                scene_id, scene, expected_revision=request.expected_revision
            )
            if op == "play":
                player.play()
            elif op == "pause":
                player.pause()
            elif op == "stop":
                player.stop()
            elif op == "seek":
                if "time" not in payload:
                    raise AnimationValidationError("seek requires time")
                player.seek(payload["time"])
            elif op == "advance":
                if "elapsed_seconds" not in payload:
                    raise AnimationValidationError("advance requires elapsed_seconds")
                player.advance(payload["elapsed_seconds"])
            elif op == "set_playback_rate":
                if "rate" not in payload:
                    raise AnimationValidationError("set_playback_rate requires rate")
                player.set_rate(payload["rate"])
            elif op == "set_loop":
                if type(payload.get("loop")) is not bool:
                    raise AnimationValidationError("set_loop requires boolean loop")
                player.loop = payload["loop"]
            elif op != "playback_status":
                raise AnimationUnsupportedOperationError("unsupported playback operation")
            return {
                "status": "ok", "request_id": request.request_id, "operation": op,
                "scene_id": request.scene_id, "time": player.position,
                "duration": player.duration, "rate": player.rate,
                "playing": player.playing, "loop": player.loop,
            }

    # ------------------------------------------------------------------
    # Bounded operational memory, capabilities, and health
    # ------------------------------------------------------------------
    def _recent_edits(self, request: AnimationRequest) -> dict[str, Any]:
        raw_limit = request.payload.get("limit", self._policy.max_history_records)
        limit = ensure_positive_integer(raw_limit, "limit")
        if limit > self._policy.max_history_records:
            raise AnimationValidationError("history limit exceeds animation_agent.max_history_records")
        edits = self.animation_memory.recent(scene_id=request.scene_id, limit=limit)
        return {"status": "ok", "request_id": request.request_id,
                "count": len(edits), "edits": [item.to_dict() for item in edits]}

    def _edit_counts(self, request: AnimationRequest) -> dict[str, Any]:
        if request.scene_id is None:
            counts = self.animation_memory.operation_counts()
        else:
            counts = Counter(item.operation for item in self.animation_memory.recent(scene_id=request.scene_id))
        return {"status": "ok", "request_id": request.request_id, "counts": dict(counts)}

    def _capabilities(self, request: AnimationRequest) -> dict[str, Any]:
        return {
            "status": "ok", "request_id": request.request_id,
            "enabled": self._policy.enabled,
            "operations": sorted(_OPERATIONS),
            "scene_binding": "trusted-process-local",
            "scene_state_owner": "provider",
            "operational_memory": "AnimationMemory",
            "cross_agent_memory": "BaseAgent.shared_memory",
            "checkpointing": False,
            "unsupported": ["learned_motion", "skeletal_rigging", "retargeting", "choreography", "physics_simulation"],
        }

    def _status(self, request: AnimationRequest) -> dict[str, Any]:
        with self._scene_lock:
            bound = tuple(sorted(self._scenes))
        with self._stats_lock:
            counts = dict(self._counts)
            failures = dict(self._failures)
            last_request_id = self._last_request_id
            last_latency_ms = self._last_latency_ms
        return {
            "status": "ok", "request_id": request.request_id, "enabled": self._policy.enabled,
            "bound_scenes": bound, "memory_records": len(self.animation_memory),
            "requests_by_operation": counts, "failures_by_operation": failures,
            "last_request_id": last_request_id, "last_latency_ms": last_latency_ms,
            "runtime": self.runtime_status(),
        }

    # ------------------------------------------------------------------
    # Bounded SharedMemory telemetry (never publish animation documents).
    # ------------------------------------------------------------------
    def _shared_publish(self, request: AnimationRequest, payload: dict[str, Any], *, kind: str) -> None:
        event = {
            "schema": "slai.animation.agent.event.v1", "agent": self.name,
            "agent_id": self.agent_id, "request_id": request.request_id,
            "operation": request.operation, "kind": kind, "scene_id": request.scene_id,
            **payload,
        }
        policy = self._policy
        if policy.publish_events:
            self._run_optional_runtime_operation(
                "communication", "shared_memory.publish",
                lambda: self.shared_memory.publish(policy.event_channel, event),
            )
        if (kind == "ok" and policy.publish_results) or (kind == "error" and policy.publish_errors):
            prefix = policy.result_key_prefix if kind == "ok" else policy.error_key_prefix
            self._run_optional_runtime_operation(
                "communication", "shared_memory.set",
                lambda: self.shared_memory.set(
                    f"{prefix}:{request.request_id}", event,
                    ttl=policy.shared_memory_ttl_seconds,
                ),
            )

    def _publish_result(self, request: AnimationRequest, result: Mapping[str, Any]) -> None:
        # Curated allowlist: no raw keyframes, scene snapshots, histories or input.
        allowed = ("revision", "time", "duration", "sample_count", "operations_applied", "would_change")
        summary = {key: result[key] for key in allowed if key in result and type(result[key]) in (int, float, bool)}
        self._shared_publish(request, summary, kind="ok")

    def _publish_failure(self, request: AnimationRequest, error: Exception) -> None:
        self._shared_publish(request, {"error_type": type(error).__name__}, kind="error")


__all__ = ["AnimationAgent"]
