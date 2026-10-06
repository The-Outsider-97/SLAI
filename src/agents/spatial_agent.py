"""SLAI v2.3 Spatial Agent orchestration façade.

SpatialAgent exposes deterministic spatial state and computation to the SLAI
multi-agent runtime while preserving the system boundary:

    Perception -> interpreted observations -> SpatialAgent -> spatial subsystem
    SpatialAgent -> structured spatial results -> Planning

The Agent never interprets raw sensor modalities and never chooses actions,
routes, goals, or navigation policies. Geometry, topology, transforms,
registration, occupancy, collision, indexing, visibility, and relation
semantics are delegated to ``src.agents.spatial``.

Academic semantics are inherited from the completed subsystem (Cohn & Renz;
Kuipers; Egenhofer/RCC; Lynch & Park; Barfoot/Solà; Elfes/OctoMap;
Bentley/Guttman/Samet; Besl & McKay/Horn; de Berg/Ericson/Gottschalk).
"""
from __future__ import annotations

__version__ = "2.3.0"

import time
import uuid
import numpy as np  # type: ignore

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .base_agent import BaseAgent
from .base.utils.base_errors import *
from .base.utils.base_helpers import coerce_bool, coerce_int
from .base.utils.main_config_loader import get_config_section
from .spatial import (
    GeometryKind,
    RelationKind,
    SpatialBounds,
    SpatialCompute,
    SpatialEntity,
    SpatialIndex,
    SpatialQueries,
    SpatialQueryResult,
    SpatialRelations,
)
from .spatial.modules.mapping import AlignmentResult, MapRecord, Mapping as SpatialMapping
from .spatial.modules.transform import RigidTransform
from .spatial.utils.spatial_errors import *
from .spatial.utils.spatial_helpers import *
from .spatial.world.geometry import AABB, Mesh, OBB, Plane, PointCloud, Polygon, Ray, Segment, Sphere, Triangle
from .spatial.world.occupancy import ESDFVolume, OccupancyGrid2D, OctreeOccupancy, TSDFVolume, VoxelGrid
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Agent")
printer = PrettyPrinter()


class SpatialAgent(BaseAgent):
    """System-level orchestration façade for SLAI spatial intelligence."""

    AGENT_KEY = "spatial_agent"
    STATE_UPDATED_TOPIC = "spatial_agent:state_updated"
    CHECKPOINT_SCHEMA = "slai.spatial-agent.state.v3"
    CHECKPOINTING_SUPPORTED = True

    _POINT_RELATIONS = frozenset(
        {
            RelationKind.NEAR.value,
            RelationKind.FAR.value,
            RelationKind.LEFT_OF.value,
            RelationKind.RIGHT_OF.value,
            RelationKind.ABOVE.value,
            RelationKind.BELOW.value,
            RelationKind.IN_FRONT_OF.value,
            RelationKind.BEHIND.value,
        }
    )

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Mapping[str, Any] | None = None,
        *,
        checkpoint_manager: Any = None,
    ) -> None:
        # BaseAgent owns the base_agent configuration section. Spatial-specific
        # runtime overrides deliberately do not leak into BaseAgent config.
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            checkpoint_manager=checkpoint_manager,
        )

        self.agent_config: dict[str, Any] = dict(get_config_section(self.AGENT_KEY) or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise BaseConfigurationError(
                    "SpatialAgent config override must be a mapping",
                    component=self.name,
                    context={"type": type(config).__name__},
                )
            self.agent_config.update(dict(config))

        self._load_agent_config()

        # Shared subsystem composition. SpatialQueries creates a coherent query,
        # index and relation view; SpatialCompute uses the subsystem-owned
        # default state and its own public transform/collision façades.
        self.queries = SpatialQueries()
        self.index: SpatialIndex = self.queries.index
        self.relations: SpatialRelations = self.queries.relations
        self.compute = SpatialCompute()
        self.mapping = SpatialMapping()

        self._relation_handlers: dict[str, Callable[..., bool]] = {
            RelationKind.INTERSECTS.value: self.relations.intersects,
            RelationKind.DISJOINT.value: self.relations.disjoint,
            RelationKind.TOUCHES.value: self.relations.touches,
            RelationKind.OVERLAPS.value: self.relations.overlaps,
            RelationKind.CONTAINS.value: self.relations.contains,
            RelationKind.INSIDE.value: self.relations.inside,
            RelationKind.EQUALS.value: self.relations.equals,
            RelationKind.CROSSES.value: self.relations.crosses,
            RelationKind.CONNECTED.value: self.relations.connected,
            RelationKind.NEAR.value: self.relations.near,
            RelationKind.FAR.value: self.relations.far,
            RelationKind.LEFT_OF.value: self.relations.left_of,
            RelationKind.RIGHT_OF.value: self.relations.right_of,
            RelationKind.ABOVE.value: self.relations.above,
            RelationKind.BELOW.value: self.relations.below,
            RelationKind.IN_FRONT_OF.value: self.relations.in_front_of,
            RelationKind.BEHIND.value: self.relations.behind,
        }

        self._operation_handlers: dict[str, Callable[[Mapping[str, Any]], Any]] = {
            "capabilities": self._handle_capabilities,
            "status": self._handle_status,
            "upsert_entity": self._handle_upsert_entity,
            "get_entity": self._handle_get_entity,
            "remove_entity": self._handle_remove_entity,
            "distance": self._handle_distance,
            "bounds": self._handle_bounds,
            "register_frame": self._handle_register_frame,
            "set_frame_parent": self._handle_set_frame_parent,
            "set_frame_transform": self._handle_set_frame_transform,
            "remove_frame": self._handle_remove_frame,
            "resolve_transform": self._handle_resolve_transform,
            "transform_point": self._handle_transform_point,
            "transform_vector": self._handle_transform_vector,
            "transform_pose": self._handle_transform_pose,
            "transform_crs": self._handle_transform_crs,
            "relation": self._handle_relation,
            "nearest": self._handle_nearest,
            "k_nearest": self._handle_k_nearest,
            "within_radius": self._handle_within_radius,
            "within_bounds": self._handle_within_bounds,
            "relation_query": self._handle_relation_query,
            "visibility": self._handle_visibility,
            "collision": self._handle_collision,
            "register_map": self._handle_register_map,
            "get_map": self._handle_get_map,
            "remove_map": self._handle_remove_map,
            "map_summary": self._handle_map_summary,
            "align_point_clouds": self._handle_align_point_clouds,
            "occupancy_lookup": self._handle_occupancy_lookup,
            "occupancy_update": self._handle_occupancy_update,
        }
        self._operation_aliases = {
            "register_entity": "upsert_entity",
            "update_entity": "upsert_entity",
            "transform": "transform_point",
            "nearest_neighbor": "nearest",
            "knn": "k_nearest",
            "visible_from": "visibility",
            "align": "align_point_clouds",
        }
        self._validate_agent_config()

        self._stats: dict[str, Any] = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "per_operation": Counter(),
            "last_operation": None,
            "last_request_id": None,
            "last_latency_ms": None,
        }
        self._last_failure: dict[str, Any] | None = None

        logger.info(
            "Spatial Agent initialized | operations=%d | publish_shared_memory=%s",
            len(self.allowed_operations),
            self.publish_shared_memory,
        )
        self._publish_event("initialized", {"operations": len(self.allowed_operations)})

    # ------------------------------------------------------------------
    # Agent configuration (agents_config.yaml through Base infrastructure)
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any) -> Any:
        return self.agent_config.get(key, default)

    def _load_agent_config(self) -> None:
        self.enabled = coerce_bool(self._cfg("enabled", True), True)
        self.publish_shared_memory = coerce_bool(self._cfg("publish_shared_memory", True), True)
        self.fail_on_shared_memory_error = coerce_bool(self._cfg("fail_on_shared_memory_error", False), False)
        self.shared_memory_ttl_seconds = coerce_int(self._cfg("shared_memory_ttl_seconds", 3600),
            3600,
            minimum=0,
            maximum=31_536_000,
        )
        self.max_query_results = coerce_int(self._cfg("max_query_results", 128), 128, minimum=1, maximum=100_000)
        self.max_result_items = coerce_int(self._cfg("max_result_items", 256), 256, minimum=8, maximum=100_000)
        self.max_obstacles = coerce_int(self._cfg("max_obstacles", 512), 512, minimum=0, maximum=100_000)
        self.max_points_per_request = coerce_int(self._cfg("max_points_per_request", 100_000), 100_000, minimum=3, maximum=5_000_000)
        self.result_key_prefix = str(self._cfg("result_key_prefix", "spatial_agent.result")).strip()
        self.event_channel = str(self._cfg("event_channel", "spatial_agent.events")).strip()
        configured = self._cfg("allowed_operations", None)
        if configured is None:
            self._configured_allowed_operations: set[str] | None = None
        elif isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
            self._configured_allowed_operations = {str(item).strip().lower() for item in configured if str(item).strip()}
        else:
            raise BaseConfigurationError("spatial_agent.allowed_operations must be a sequence of operation names", component=self.name)

    def _validate_agent_config(self) -> None:
        if not self.result_key_prefix:
            raise BaseConfigurationError("spatial_agent.result_key_prefix must be non-empty", component=self.name)
        if not self.event_channel:
            raise BaseConfigurationError("spatial_agent.event_channel must be non-empty", component=self.name,)
        supported = set(self._operation_handlers)
        if self._configured_allowed_operations is None:
            self.allowed_operations = frozenset(supported)
            return
        unknown = self._configured_allowed_operations - supported
        if unknown:
            raise BaseConfigurationError(
                "spatial_agent.allowed_operations contains unsupported operations",
                component=self.name,
                context={"unknown": sorted(unknown)},
            )
        self.allowed_operations = frozenset(self._configured_allowed_operations)

    # ------------------------------------------------------------------
    # BaseAgent task boundary
    # ------------------------------------------------------------------

    def perform_task(self, task_data: Any) -> dict[str, Any]:
        """Validate and dispatch one structured spatial orchestration request."""
        if not isinstance(task_data, Mapping):
            raise BaseValidationError(
                "SpatialAgent expects a mapping request",
                component=self.name,
                operation="perform_task",
                context={"received_type": type(task_data).__name__},
            )
        if not self.enabled:
            return {
                "status": "disabled",
                "agent": self.name,
                "operation": str(task_data.get("operation", "")),
                "result": None,
            }

        payload = dict(task_data)
        operation = self._normalize_operation(payload.get("operation", payload.get("op")))
        handler = self._operation_handlers.get(operation)
        if handler is None or operation not in self.allowed_operations:
            logger.warning("SpatialAgent rejected unsupported operation | operation=%s", operation)
            raise SpatialQueryError(
                "unsupported SpatialAgent operation",
                context={
                    "operation": operation,
                    "supported": sorted(self.allowed_operations),
                },
            )

        request_id = str(payload.get("request_id") or uuid.uuid4().hex).strip()
        if not request_id:
            raise BaseValidationError("request_id must be non-empty", component=self.name, operation=operation)

        started = time.perf_counter()
        self._record_request_start(operation, request_id)
        try:
            result = handler(payload)
            response = {
                "status": "ok",
                "agent": self.name,
                "request_id": request_id,
                "operation": operation,
                "result": self._safe_result(result),
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record_request_failure(operation, request_id, latency_ms, exc)
            self._publish_event(
                "request_failed",
                {
                    "request_id": request_id,
                    "operation": operation,
                    "error_type": type(exc).__name__,
                },
            )
            logger.exception("SpatialAgent request failed | id=%s | operation=%s", request_id, operation)
            raise

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record_request_success(operation, request_id, latency_ms)
        self._publish_result(request_id, response)
        self._publish_completion_event(operation, request_id, response)
        logger.debug(
            "SpatialAgent request completed | id=%s | operation=%s | latency_ms=%.3f",
            request_id,
            operation,
            latency_ms,
        )
        return response

    def predict(self, state: Any, context: Any = None) -> dict[str, Any]:
        """Compatibility route for BaseAgent/factory predict-style dispatch."""
        if not isinstance(state, Mapping):
            raise BaseValidationError("SpatialAgent.predict requires a mapping request", component=self.name)
        payload = dict(state)
        if context is not None and "context" not in payload:
            payload["context"] = context
        return self.perform_task(payload)

    def act(self, task_data: Any, context: Any = None) -> dict[str, Any]:
        """Compatibility route; Spatial reports state and never selects actions."""
        return self.predict(task_data, context=context)

    def capabilities(self) -> dict[str, Any]:
        """Return the bounded public capability contract of this Agent."""
        return {
            "agent": self.name,
            "operations": tuple(sorted(self.allowed_operations)),
            "checkpointing_supported": self.supports_checkpointing,
            "shared_memory_publication": self.publish_shared_memory,
            "perception": False,
            "planning": False,
            "action_selection": False,
            "raw_sensor_interpretation": False,
        }

    # ------------------------------------------------------------------
    # Entity/state orchestration
    # ------------------------------------------------------------------

    def _handle_upsert_entity(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        entity_payload = payload.get("entity", payload)
        existing = None
        if isinstance(entity_payload, Mapping) and entity_payload.get("entity_id"):
            existing = self.index.get(str(entity_payload["entity_id"]))
        entity = self._coerce_entity(entity_payload, existing=existing)
        stored = self.index.upsert(entity)
        return {"entity": self._entity_summary(stored), "created": existing is None}

    def _handle_get_entity(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        entity = self._require_entity(self._required_str(payload, "entity_id"))
        return {"entity": self._entity_summary(entity)}

    def _handle_remove_entity(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        entity_id = self._required_str(payload, "entity_id")
        removed = self.index.remove(entity_id)
        return {
            "removed": removed is not None,
            "entity": self._entity_summary(removed) if removed is not None else None,
        }

    # ------------------------------------------------------------------
    # Deterministic calculations / frames
    # ------------------------------------------------------------------

    def _handle_distance(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source_value = self._required(payload, "source")
        target_value = self._required(payload, "target")
        source, source_frame, source_id = self._position_operand(source_value, frame_hint=self._optional_str(payload, "source_frame"))
        target, target_frame, target_id = self._position_operand(target_value, frame_hint=self._optional_str(payload, "target_frame"))
        if source.shape != target.shape:
            raise SpatialValidationError(
                "distance operands must have matching dimensions",
                context={"source_dim": len(source), "target_dim": len(target)},
            )
        result_frame = source_frame or target_frame
        if source_frame and target_frame and source_frame != target_frame:
            if len(source) != 3:
                raise SpatialFrameError(
                    "cross-frame distance requires 3D coordinates",
                    context={"source_frame": source_frame, "target_frame": target_frame},
                )
            target = self.compute.transform_point(target, target_frame, source_frame)
            result_frame = source_frame
        metric = str(payload.get("metric", "euclidean")).strip().lower()
        value = self.compute.distance(source, target, metric=metric)
        return {
            "distance": value,
            "metric": metric,
            "frame_id": result_frame,
            "source_entity": source_id,
            "target_entity": target_id,
        }

    def _handle_bounds(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = self._required(payload, "value")
        if isinstance(value, str):
            value = self._require_entity(value)
        elif isinstance(value, Mapping):
            value = self._coerce_geometry(value)
        bounds = self.compute.bounds(value)
        return {"bounds": self._geometry_payload(bounds)}

    def _handle_register_frame(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        frame_id = self._required_str(payload, "frame_id")
        parent_id = self._optional_str(payload, "parent_id")
        transform_payload = payload.get("transform")
        transform = None
        if transform_payload is not None:
            if parent_id is None:
                raise SpatialValidationError("parent_id is required when registering a frame transform")
            transform = self._coerce_transform(
                transform_payload,
                default_source=frame_id,
                default_target=parent_id,
            )
        self.compute.register_frame(
            frame_id,
            parent_id=parent_id,
            transform_to_parent=transform,
            metadata=self._mapping(payload.get("metadata")),
        )
        return {"frame_id": frame_id, "parent_id": parent_id}

    def _handle_set_frame_parent(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        frame_id = self._required_str(payload, "frame_id")
        parent_id = self._required_str(payload, "parent_id")
        transform = self._coerce_transform(
            self._required(payload, "transform"),
            default_source=frame_id,
            default_target=parent_id,
        )
        self.compute.set_frame_parent(frame_id, parent_id, transform)
        return {
            "frame_id": frame_id,
            "parent_id": parent_id,
            "transform": transform.to_dict(),
        }

    def _handle_set_frame_transform(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        transform = self._coerce_transform(self._required(payload, "transform"))
        self.compute.set_frame_transform(transform)
        return {"transform": transform.to_dict()}

    def _handle_remove_frame(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        frame_id = self._required_str(payload, "frame_id")
        recursive = coerce_bool(payload.get("recursive", False), False)
        self.compute.remove_frame(frame_id, recursive=recursive)
        return {"removed": True, "frame_id": frame_id, "recursive": recursive}

    def _handle_resolve_transform(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = self._required_str(payload, "source_frame")
        target = self._required_str(payload, "target_frame")
        return {"transform": self.compute.resolve_transform(source, target).to_dict()}

    def _handle_transform_point(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = self._required_str(payload, "source_frame")
        target = self._required_str(payload, "target_frame")
        result = self.compute.transform_point(self._required(payload, "point"), source, target)
        return {"point": result.tolist(), "source_frame": source, "target_frame": target}

    def _handle_transform_vector(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = self._required_str(payload, "source_frame")
        target = self._required_str(payload, "target_frame")
        result = self.compute.transform_vector(self._required(payload, "vector"), source, target)
        return {"vector": result.tolist(), "source_frame": source, "target_frame": target}

    def _handle_transform_pose(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        pose = self._coerce_transform(self._required(payload, "pose"))
        target = self._required_str(payload, "target_frame")
        return {"pose": self.compute.transform_pose(pose, target).to_dict()}

    def _handle_transform_crs(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = self._required_str(payload, "source_crs")
        target = self._required_str(payload, "target_crs")
        coordinates = self.compute.transform_crs(
            self._required(payload, "coordinates"), source, target
        )
        return {
            "coordinates": coordinates.tolist(),
            "source_crs": source,
            "target_crs": target,
        }

    # ------------------------------------------------------------------
    # Relations / queries
    # ------------------------------------------------------------------

    def _handle_relation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        relation = self._required_str(payload, "relation").lower()
        first = self._entity_operand(self._required(payload, "source"))
        second = self._entity_operand(self._required(payload, "target"))

        if relation in {"all", "relate"}:
            if first.frame_id != second.frame_id:
                raise SpatialFrameError(
                    "full relation evaluation requires entities in a common frame",
                    context={"source_frame": first.frame_id, "target_frame": second.frame_id},
                )
            relationships = self.relations.relate(first, second)
            return {"relationships": [item.to_dict() for item in relationships]}

        if relation == RelationKind.DISTANCE.value:
            return self._handle_distance({"source": first, "target": second})

        handler = self._relation_handlers.get(relation)
        if handler is None:
            raise SpatialQueryError(
                "unsupported spatial relation",
                context={"relation": relation, "supported": sorted(self._relation_handlers)},
            )

        if first.frame_id != second.frame_id:
            if relation not in self._POINT_RELATIONS:
                raise SpatialFrameError(
                    "cross-frame geometric/topological relations require geometry in a common frame",
                    context={
                        "relation": relation,
                        "source_frame": first.frame_id,
                        "target_frame": second.frame_id,
                    },
                )
            second = self.compute.transform_entity(second, first.frame_id)

        kwargs: dict[str, Any] = {}
        if relation in {RelationKind.NEAR.value, RelationKind.FAR.value} and "threshold" in payload:
            kwargs["threshold"] = float(payload["threshold"])
        value = bool(handler(first, second, **kwargs))
        return {
            "relation": relation,
            "value": value,
            "source": first.entity_id,
            "target": second.entity_id,
            "frame_id": first.frame_id,
        }

    def _handle_nearest(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        point, frame_id = self._query_point(payload)
        exclude = self._string_set(payload.get("exclude_ids"))
        return self._query_payload(self.queries.nearest(point, frame_id=frame_id, exclude_ids=exclude))

    def _handle_k_nearest(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        point, frame_id = self._query_point(payload)
        k = int(self._required(payload, "k"))
        if k < 1 or k > self.max_query_results:
            raise SpatialQueryError(
                "k is outside SpatialAgent query bounds",
                context={"k": k, "max_query_results": self.max_query_results},
            )
        return self._query_payload(self.queries.k_nearest(point, k, frame_id=frame_id))

    def _handle_within_radius(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        point, frame_id = self._query_point(payload)
        radius = float(self._required(payload, "radius"))
        return self._query_payload(self.queries.within_radius(point, radius, frame_id=frame_id))

    def _handle_within_bounds(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        bounds = self._coerce_aabb(self._required(payload, "bounds"))
        frame_id = self._optional_str(payload, "frame_id")
        return self._query_payload(self.queries.within_bounds(bounds, frame_id=frame_id))

    def _handle_relation_query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        entity = self._required(payload, "entity")
        relation = self._required_str(payload, "relation")
        candidates = payload.get("candidates")
        normalized_candidates = None
        if candidates is not None:
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                raise SpatialValidationError("candidates must be a sequence")
            normalized_candidates = tuple(self._entity_operand(item) for item in candidates)
        result = self.queries.query_relation(
            self._entity_operand(entity),
            relation,
            candidates=normalized_candidates,
        )
        return self._query_payload(result)

    def _handle_visibility(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        observer = self._entity_operand(self._required(payload, "observer"))
        target = self._entity_operand(self._required(payload, "target"))
        if target.frame_id != observer.frame_id:
            target = self.compute.transform_entity(target, observer.frame_id)

        obstacles = payload.get("obstacles")
        normalized_obstacles = None
        if obstacles is not None:
            if not isinstance(obstacles, Sequence) or isinstance(obstacles, (str, bytes)):
                raise SpatialValidationError("obstacles must be a sequence")
            if len(obstacles) > self.max_obstacles:
                raise SpatialValidationError(
                    "obstacle count exceeds SpatialAgent limit",
                    context={"count": len(obstacles), "max_obstacles": self.max_obstacles},
                )
            normalized: list[Any] = []
            for item in obstacles:
                if isinstance(item, str):
                    entity = self._require_entity(item)
                    if entity.frame_id != observer.frame_id:
                        raise SpatialFrameError(
                            "visibility obstacle geometry is not in the observer frame",
                            context={"entity_id": item, "frame_id": entity.frame_id},
                        )
                    normalized.append(entity)
                elif isinstance(item, SpatialEntity):
                    if item.frame_id != observer.frame_id:
                        raise SpatialFrameError(
                            "visibility obstacle geometry is not in the observer frame",
                            context={"entity_id": item.entity_id, "frame_id": item.frame_id},
                        )
                    normalized.append(item)
                else:
                    normalized.append(self._coerce_geometry(item))
            normalized_obstacles = tuple(normalized)

        return self._query_payload(self.queries.visible_from(observer, target, obstacles=normalized_obstacles))

    # ------------------------------------------------------------------
    # Collision / map / occupancy orchestration
    # ------------------------------------------------------------------

    def _handle_collision(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        first, first_frame = self._collision_operand(self._required(payload, "source"), self._optional_str(payload, "source_frame"))
        second, second_frame = self._collision_operand(self._required(payload, "target"), self._optional_str(payload, "target_frame"))
        if first_frame and second_frame and first_frame != second_frame:
            raise SpatialFrameError(
                "collision geometry must be expressed in a common frame",
                context={"source_frame": first_frame, "target_frame": second_frame},
            )
        result = self.compute.collision_state(first, second)
        details = dict(result.details or {})
        if "candidate_triangle_pairs" in details:
            pairs = tuple(details["candidate_triangle_pairs"])
            details["candidate_triangle_pairs"] = pairs[: self.max_result_items]
            details["candidate_triangle_pairs_truncated"] = len(pairs) > self.max_result_items
        return {
            "collides": result.collides,
            "distance": result.distance,
            "broad_phase_only": result.broad_phase_only,
            "details": to_json_safe(details),
            "frame_id": first_frame or second_frame,
        }

    def _handle_register_map(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        map_id = self._required_str(payload, "map_id")
        frame_id = str(payload.get("frame_id", "world")).strip() or "world"
        representation = self._coerce_map_representation(self._required(payload, "representation"), frame_id=frame_id)
        record = self.mapping.register(
            map_id,
            representation,
            frame_id=frame_id,
            metadata=self._mapping(payload.get("metadata")),
        )
        return {"map": self._map_summary(record)}

    def _handle_get_map(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {"map": self._map_summary(self.mapping.get(self._required_str(payload, "map_id")))}

    def _handle_remove_map(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = self.mapping.remove(self._required_str(payload, "map_id"))
        return {"removed": True, "map": self._map_summary(record)}

    def _handle_map_summary(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        summary = self.mapping.summary()
        return {
            "maps": list(summary[: self.max_result_items]),
            "truncated": len(summary) > self.max_result_items,
        }

    def _handle_align_point_clouds(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = self._resolve_point_cloud(self._required(payload, "source"))
        target = self._resolve_point_cloud(self._required(payload, "target"))
        self._validate_point_count(source)
        self._validate_point_count(target)
        kwargs = {
            key: payload[key]
            for key in (
                "max_iterations",
                "tolerance",
                "max_correspondence_distance",
                "trim_fraction",
            )
            if key in payload
        }
        result = self.mapping.align_point_clouds(source, target, **kwargs)
        return self._alignment_payload(result)

    def _handle_occupancy_lookup(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = self.mapping.get(self._required_str(payload, "map_id"))
        representation = record.representation
        if isinstance(representation, OccupancyGrid2D):
            x, y = self._grid2d_index(representation, payload)
            return {
                "map_id": record.map_id,
                "index": [x, y],
                "probability": representation.probability(x, y),
                "state": representation.state(x, y).value,
                "frame_id": record.frame_id,
            }
        if isinstance(representation, VoxelGrid):
            index = self._voxel_index(representation, payload)
            return {
                "map_id": record.map_id,
                "index": list(index),
                "probability": representation.probability(index),
                "state": representation.state(index).value,
                "frame_id": record.frame_id,
            }
        if isinstance(representation, OctreeOccupancy):
            point = self._required(payload, "point")
            return {
                "map_id": record.map_id,
                "probability": representation.probability(point),
                "frame_id": record.frame_id,
            }
        if isinstance(representation, TSDFVolume):
            index = self._triple_index(self._required(payload, "index"))
            voxel = representation.get(index)
            return {
                "map_id": record.map_id,
                "index": list(index),
                "voxel": None
                if voxel is None
                else {"distance": voxel.distance, "weight": voxel.weight},
                "frame_id": record.frame_id,
            }
        if isinstance(representation, ESDFVolume):
            index = self._triple_index(self._required(payload, "index"))
            return {
                "map_id": record.map_id,
                "index": list(index),
                "distance": representation.distance(index),
                "frame_id": record.frame_id,
            }
        raise SpatialOccupancyError(
            "registered map is not an occupancy/distance-field representation",
            context={"map_id": record.map_id, "type": type(representation).__name__},
        )

    def _handle_occupancy_update(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = self.mapping.get(self._required_str(payload, "map_id"))
        representation = record.representation
        if isinstance(representation, OccupancyGrid2D):
            x, y = self._grid2d_index(representation, payload)
            probability = representation.update_probability(
                x,
                y,
                float(self._required(payload, "observation_probability")),
                prior_probability=float(payload.get("prior_probability", 0.5)),
            )
            return {"map_id": record.map_id, "index": [x, y], "probability": probability}
        if isinstance(representation, VoxelGrid):
            index = self._voxel_index(representation, payload)
            probability = representation.update_probability(
                index,
                float(self._required(payload, "observation_probability")),
                prior_probability=float(payload.get("prior_probability", 0.5)),
            )
            return {"map_id": record.map_id, "index": list(index), "probability": probability}
        if isinstance(representation, OctreeOccupancy):
            point = self._required(payload, "point")
            probability = representation.update_probability(
                point,
                float(self._required(payload, "observation_probability")),
                prior_probability=float(payload.get("prior_probability", 0.5)),
            )
            return {"map_id": record.map_id, "probability": probability}
        if isinstance(representation, TSDFVolume):
            index = self._triple_index(self._required(payload, "index"))
            voxel = representation.integrate(
                index,
                float(self._required(payload, "signed_distance")),
                weight=float(payload.get("weight", 1.0)),
            )
            return {
                "map_id": record.map_id,
                "index": list(index),
                "distance": voxel.distance,
                "weight": voxel.weight,
            }
        if isinstance(representation, ESDFVolume):
            index = self._triple_index(self._required(payload, "index"))
            representation.set_distance(index, float(self._required(payload, "distance")))
            return {
                "map_id": record.map_id,
                "index": list(index),
                "distance": representation.distance(index),
            }
        raise SpatialOccupancyError(
            "registered map does not support occupancy updates",
            context={"map_id": record.map_id, "type": type(representation).__name__},
        )

    def _handle_capabilities(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.capabilities()

    def _handle_status(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            stats = {
                **self._stats,
                "per_operation": dict(self._stats["per_operation"]),
            }
        frames = self.compute.frame_snapshot()
        maps = self.mapping.summary()
        return {
            "enabled": self.enabled,
            "stats": stats,
            "frames": list(frames[: self.max_result_items]),
            "frames_truncated": len(frames) > self.max_result_items,
            "map_count": len(maps),
            "last_failure": dict(self._last_failure) if self._last_failure else None,
        }

    # ------------------------------------------------------------------
    # Request normalization helpers
    # ------------------------------------------------------------------

    def _normalize_operation(self, operation: Any) -> str:
        if not isinstance(operation, str) or not operation.strip():
            raise BaseValidationError(
                "SpatialAgent request requires a non-empty operation",
                component=self.name,
                operation="perform_task",
            )
        normalized = operation.strip().lower()
        return self._operation_aliases.get(normalized, normalized)

    @staticmethod
    def _required(payload: Mapping[str, Any], field: str) -> Any:
        if field not in payload or payload[field] is None:
            raise SpatialValidationError(
                f"{field} is required",
                context={"field": field},
            )
        return payload[field]

    @classmethod
    def _required_str(cls, payload: Mapping[str, Any], field: str) -> str:
        return validate_identifier(str(cls._required(payload, field)), name=field)

    @staticmethod
    def _optional_str(payload: Mapping[str, Any], field: str) -> str | None:
        value = payload.get(field)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise SpatialValidationError("metadata must be a mapping")
        return dict(value)

    @staticmethod
    def _string_set(value: Any) -> set[str] | None:
        if value is None:
            return None
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise SpatialValidationError("exclude_ids must be a sequence")
        return {validate_identifier(str(item), name="exclude_id") for item in value}

    def _require_entity(self, entity_id: str) -> SpatialEntity:
        entity = self.index.get(entity_id)
        if entity is None:
            raise SpatialValidationError(
                "unknown spatial entity",
                context={"entity_id": entity_id},
            )
        return entity

    def _entity_operand(self, value: Any) -> SpatialEntity:
        if isinstance(value, SpatialEntity):
            return value
        if isinstance(value, str):
            return self._require_entity(value)
        if isinstance(value, Mapping):
            if set(value).issubset({"entity_id"}) and value.get("entity_id"):
                return self._require_entity(str(value["entity_id"]))
            return self._coerce_entity(value)
        raise SpatialValidationError(
            "entity operand must be an entity id, SpatialEntity, or entity mapping"
        )

    def _coerce_entity(self, value: Any, *, existing: SpatialEntity | None = None) -> SpatialEntity:
        if isinstance(value, SpatialEntity):
            return value
        if not isinstance(value, Mapping):
            raise SpatialValidationError("entity must be a SpatialEntity or mapping")
        payload = dict(value)
        entity_id = validate_identifier(
            str(payload.get("entity_id") or (existing.entity_id if existing else "")),
            name="entity_id",
        )
        position_value = payload.get("position", existing.position if existing else None)
        if position_value is None:
            raise SpatialValidationError("entity.position is required")
        position_array = np.asarray(position_value, dtype=float)
        if position_array.ndim != 1 or position_array.size == 0 or not np.all(np.isfinite(position_array)):
            raise SpatialValidationError("entity.position must be a finite one-dimensional coordinate")
        frame_id = str(payload.get("frame_id", existing.frame_id if existing else "world")).strip()
        if not frame_id:
            raise SpatialValidationError("entity.frame_id must be non-empty")
        frame_changed = existing is not None and frame_id != existing.frame_id
        assert existing is not None
        if frame_changed and existing.geometry is not None and "geometry" not in payload:
            raise SpatialFrameError(
                "changing the frame of a geometry-bearing entity requires geometry expressed in the new frame",
                context={
                    "entity_id": existing.entity_id,
                    "source_frame": existing.frame_id,
                    "target_frame": frame_id,
                },
            )
        if frame_changed and existing.bounds is not None and "bounds" not in payload:
            raise SpatialFrameError(
                "changing the frame of a bounded entity requires bounds expressed in the new frame",
                context={
                    "entity_id": existing.entity_id,
                    "source_frame": existing.frame_id,
                    "target_frame": frame_id,
                },
            )
        geometry = existing.geometry if existing is not None else None
        if "geometry" in payload:
            geometry = self._coerce_geometry(payload["geometry"]) if payload["geometry"] is not None else None
        geometry_kind = existing.geometry_kind if existing is not None else GeometryKind.UNKNOWN
        if "geometry_kind" in payload:
            raw_geometry_kind = payload["geometry_kind"]
            try:
                geometry_kind = (
                    raw_geometry_kind
                    if isinstance(raw_geometry_kind, GeometryKind)
                    else GeometryKind(str(raw_geometry_kind).strip().lower())
                )
            except ValueError as exc:
                raise SpatialValidationError("unsupported geometry_kind", cause=exc) from exc
        elif geometry is not None:
            geometry_kind = self._infer_geometry_kind(geometry)
        elif "geometry" in payload:
            geometry_kind = GeometryKind.UNKNOWN
        bounds = existing.bounds if existing is not None else None
        if "bounds" in payload:
            bounds = self._coerce_spatial_bounds(payload["bounds"], frame_id=frame_id)
        metadata = dict(existing.metadata) if existing is not None else {}
        metadata.update(self._mapping(payload.get("metadata")))
        revision = existing.revision + 1 if existing is not None else int(payload.get("revision", 0))
        return SpatialEntity(
            entity_id=entity_id,
            position=tuple(float(item) for item in position_array),
            frame_id=frame_id,
            geometry=geometry,
            geometry_kind=geometry_kind,
            bounds=bounds,
            metadata=metadata,
            revision=revision,
        )

    def _position_operand(self, value: Any, *, frame_hint: str | None) -> tuple[np.ndarray, str | None, str | None]:
        if isinstance(value, str):
            entity = self._require_entity(value)
            return np.asarray(entity.position, dtype=float), entity.frame_id, entity.entity_id
        if isinstance(value, SpatialEntity):
            return np.asarray(value.position, dtype=float), value.frame_id, value.entity_id
        if isinstance(value, Mapping) and "position" in value:
            if value.get("entity_id"):
                entity = self._coerce_entity(value)
                return np.asarray(entity.position, dtype=float), entity.frame_id, entity.entity_id
            try:
                point = np.asarray(value["position"], dtype=float)
            except (TypeError, ValueError) as exc:
                raise SpatialValidationError("position operand must be numeric", cause=exc) from exc
            if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
                raise SpatialValidationError("position operand must be a finite one-dimensional coordinate")
            frame_id = str(value.get("frame_id") or frame_hint or "").strip() or None
            return point, frame_id, None
        try:
            point = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise SpatialValidationError("position operand must be numeric", cause=exc) from exc
        if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
            raise SpatialValidationError("position operand must be a finite one-dimensional coordinate")
        return point, frame_hint, None

    def _query_point(self, payload: Mapping[str, Any]) -> tuple[np.ndarray, str | None]:
        point = np.asarray(self._required(payload, "point"), dtype=float)
        if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
            raise SpatialQueryError("query point must be a finite one-dimensional coordinate")
        source_frame = self._optional_str(payload, "source_frame")
        index_frame = self._optional_str(payload, "frame_id") or source_frame
        if source_frame and index_frame and source_frame != index_frame:
            if point.size != 3:
                raise SpatialFrameError("cross-frame spatial query requires a 3D point")
            point = self.compute.transform_point(point, source_frame, index_frame)
        return point, index_frame

    def _collision_operand(self, value: Any, frame_hint: str | None) -> tuple[Any, str | None]:
        if isinstance(value, str):
            entity = self._require_entity(value)
            return entity, entity.frame_id
        if isinstance(value, SpatialEntity):
            return value, value.frame_id
        return self._coerce_geometry(value), frame_hint

    def _coerce_transform(
        self,
        value: Any,
        *,
        default_source: str | None = None,
        default_target: str | None = None,
    ) -> RigidTransform:
        if isinstance(value, RigidTransform):
            return value
        if not isinstance(value, Mapping):
            raise SpatialValidationError("transform must be a RigidTransform or mapping")
        source = str(value.get("source_frame") or default_source or "").strip()
        target = str(value.get("target_frame") or default_target or "").strip()
        if not source or not target:
            raise SpatialValidationError("transform requires source_frame and target_frame")
        if "matrix" in value:
            return RigidTransform.from_matrix(source, target, value["matrix"])
        rotation = value.get("rotation", np.eye(3))
        translation = value.get("translation", (0.0, 0.0, 0.0))
        return RigidTransform(source, target, rotation, translation)

    def _coerce_spatial_bounds(self, value: Any, *, frame_id: str) -> SpatialBounds:
        if isinstance(value, SpatialBounds):
            return value
        if isinstance(value, AABB):
            return SpatialBounds(tuple(value.minimum), tuple(value.maximum), frame_id=frame_id)
        if not isinstance(value, Mapping):
            raise SpatialValidationError("bounds must be SpatialBounds, AABB, or mapping")
        lower = value.get("lower", value.get("minimum"))
        upper = value.get("upper", value.get("maximum"))
        if lower is None or upper is None:
            raise SpatialValidationError("bounds require lower/minimum and upper/maximum")
        return SpatialBounds(
            tuple(float(item) for item in lower),
            tuple(float(item) for item in upper),
            frame_id=str(value.get("frame_id", frame_id)),
        )

    def _coerce_aabb(self, value: Any) -> AABB:
        if isinstance(value, AABB):
            return value
        if isinstance(value, SpatialBounds):
            return AABB(np.asarray(value.lower, dtype=float), np.asarray(value.upper, dtype=float))
        if not isinstance(value, Mapping):
            raise SpatialValidationError("AABB must be an AABB, SpatialBounds, or mapping")
        lower = value.get("minimum", value.get("lower"))
        upper = value.get("maximum", value.get("upper"))
        if lower is None or upper is None:
            raise SpatialValidationError("AABB requires minimum/lower and maximum/upper")
        return AABB(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))

    def _coerce_geometry(self, value: Any) -> Any:
        if isinstance(value, (AABB, OBB, Sphere, Mesh, PointCloud, Triangle, Polygon, Segment, Ray, Plane, np.ndarray)):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
            array = np.asarray(value, dtype=float)
            if array.ndim == 1:
                return array
        if not isinstance(value, Mapping):
            raise SpatialValidationError(
                "geometry must be a supported geometry object or structured mapping",
                context={"type": type(value).__name__},
            )
        kind = str(value.get("type", value.get("geometry_kind", ""))).strip().lower()
        if kind == "aabb":
            return self._coerce_aabb(value)
        if kind == "sphere":
            return Sphere(np.asarray(self._required(value, "center"), dtype=float), float(self._required(value, "radius")))
        if kind == "obb":
            return OBB(
                np.asarray(self._required(value, "center"), dtype=float),
                np.asarray(self._required(value, "half_extents"), dtype=float),
                np.asarray(self._required(value, "rotation"), dtype=float),
            )
        if kind == "point_cloud":
            return self._coerce_point_cloud(value)
        if kind == "mesh":
            vertices = np.asarray(self._required(value, "vertices"), dtype=float)
            faces = np.asarray(self._required(value, "faces"), dtype=int)
            self._validate_geometry_size(vertices, faces)
            return Mesh(vertices, faces, frame_id=str(value.get("frame_id", "world")))
        if kind == "triangle":
            return Triangle(np.asarray(value["a"], dtype=float), np.asarray(value["b"], dtype=float), np.asarray(value["c"], dtype=float))
        if kind == "polygon":
            return Polygon(np.asarray(self._required(value, "vertices"), dtype=float))
        if kind == "segment":
            return Segment(np.asarray(value["start"], dtype=float), np.asarray(value["end"], dtype=float))
        if kind == "ray":
            return Ray(np.asarray(value["origin"], dtype=float), np.asarray(value["direction"], dtype=float))
        if kind == "plane":
            return Plane(np.asarray(value["point"], dtype=float), np.asarray(value["normal"], dtype=float))
        if kind == "point":
            point = np.asarray(self._required(value, "coordinates"), dtype=float)
            if point.ndim != 1 or not np.all(np.isfinite(point)):
                raise SpatialValidationError("point geometry must be a finite vector")
            return point
        raise SpatialValidationError("unsupported geometry type", context={"geometry_type": kind})

    @staticmethod
    def _infer_geometry_kind(value: Any) -> GeometryKind:
        mapping = {
            AABB: GeometryKind.AABB,
            OBB: GeometryKind.OBB,
            Sphere: GeometryKind.SPHERE,
            Mesh: GeometryKind.MESH,
            PointCloud: GeometryKind.POINT_CLOUD,
            Triangle: GeometryKind.TRIANGLE,
            Polygon: GeometryKind.POLYGON,
            Segment: GeometryKind.SEGMENT,
            Ray: GeometryKind.RAY,
            Plane: GeometryKind.PLANE,
        }
        for cls, kind in mapping.items():
            if isinstance(value, cls):
                return kind
        if isinstance(value, np.ndarray) and value.ndim == 1:
            return GeometryKind.POINT
        return GeometryKind.UNKNOWN

    def _coerce_point_cloud(self, value: Any, *, default_frame: str = "world") -> PointCloud:
        if isinstance(value, PointCloud):
            return value
        if not isinstance(value, Mapping):
            points = np.asarray(value, dtype=float)
            cloud = PointCloud(points, frame_id=default_frame)
        else:
            points = np.asarray(self._required(value, "points"), dtype=float)
            cloud = PointCloud(points, frame_id=str(value.get("frame_id", default_frame)))
        self._validate_point_count(cloud)
        return cloud

    def _validate_point_count(self, cloud: PointCloud) -> None:
        if len(cloud) > self.max_points_per_request:
            raise SpatialMappingError(
                "point cloud exceeds SpatialAgent request limit",
                context={"points": len(cloud), "max_points": self.max_points_per_request},
            )

    def _validate_geometry_size(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        if len(vertices) > self.max_points_per_request or len(faces) > self.max_points_per_request:
            raise SpatialValidationError(
                "mesh exceeds SpatialAgent request size limit",
                context={
                    "vertices": len(vertices),
                    "faces": len(faces),
                    "max_items": self.max_points_per_request,
                },
            )

    def _coerce_map_representation(self, value: Any, *, frame_id: str) -> Any:
        if not isinstance(value, Mapping):
            return value
        kind = str(value.get("type", "")).strip().lower()
        if kind == "point_cloud":
            return self._coerce_point_cloud(value, default_frame=frame_id)
        if kind == "occupancy_grid_2d":
            return OccupancyGrid2D(
                int(self._required(value, "width")),
                int(self._required(value, "height")),
                resolution=float(value.get("resolution", 1.0)),
                origin=value.get("origin", (0.0, 0.0)),
                free_threshold=float(value.get("free_threshold", 0.35)),
                occupied_threshold=float(value.get("occupied_threshold", 0.65)),
            )
        if kind in {"voxel_grid", "occupancy_grid_3d"}:
            return VoxelGrid(
                resolution=float(value.get("resolution", 1.0)),
                origin=value.get("origin", (0.0, 0.0, 0.0)),
                free_threshold=float(value.get("free_threshold", 0.35)),
                occupied_threshold=float(value.get("occupied_threshold", 0.65)),
            )
        if kind == "octree_occupancy":
            return OctreeOccupancy(
                self._required(value, "minimum"),
                self._required(value, "maximum"),
                max_depth=int(value.get("max_depth", 8)),
            )
        if kind == "tsdf":
            return TSDFVolume(
                voxel_size=float(value.get("voxel_size", 0.1)),
                truncation_distance=float(value.get("truncation_distance", 0.3)),
                origin=value.get("origin", (0.0, 0.0, 0.0)),
            )
        if kind == "esdf":
            return ESDFVolume(
                voxel_size=float(value.get("voxel_size", 0.1)),
                origin=value.get("origin", (0.0, 0.0, 0.0)),
            )
        raise SpatialValidationError("unsupported structured map representation", context={"type": kind})

    def _resolve_point_cloud(self, value: Any) -> PointCloud:
        if isinstance(value, str):
            record = self.mapping.get(value)
            if not isinstance(record.representation, PointCloud):
                raise SpatialMappingError(
                    "map does not contain a PointCloud representation",
                    context={"map_id": value, "type": type(record.representation).__name__},
                )
            return record.representation
        return self._coerce_point_cloud(value)

    @staticmethod
    def _grid2d_index(grid: OccupancyGrid2D, payload: Mapping[str, Any]) -> tuple[int, int]:
        if "index" in payload:
            index = tuple(int(item) for item in payload["index"])
            if len(index) != 2:
                raise SpatialOccupancyError("2D occupancy index must have two coordinates")
            return index[0], index[1]
        return grid.world_to_grid(SpatialAgent._required(payload, "point"))

    @staticmethod
    def _voxel_index(grid: VoxelGrid, payload: Mapping[str, Any]) -> tuple[int, int, int]:
        if "index" in payload:
            return SpatialAgent._triple_index(payload["index"])
        return grid.world_to_voxel(SpatialAgent._required(payload, "point"))

    @staticmethod
    def _triple_index(value: Any) -> tuple[int, int, int]:
        try:
            index = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise SpatialOccupancyError("voxel index must contain three integers", cause=exc) from exc
        if len(index) != 3:
            raise SpatialOccupancyError("voxel index must contain three integers")
        return index[0], index[1], index[2]

    # ------------------------------------------------------------------
    # Result shaping / observability / SharedMemory
    # ------------------------------------------------------------------

    def _query_payload(self, result: SpatialQueryResult) -> dict[str, Any]:
        count = len(result.entity_ids)
        limit = min(count, self.max_query_results)
        return {
            "query": result.query,
            "entity_ids": list(result.entity_ids[:limit]),
            "distances": list(result.distances[:limit]) if result.distances else [],
            "relationships": [item.to_dict() for item in result.relationships[:limit]],
            "metadata": to_json_safe(result.metadata),
            "truncated": count > limit,
            "total_matches": count,
        }

    @staticmethod
    def _entity_summary(entity: SpatialEntity | None) -> dict[str, Any] | None:
        if entity is None:
            return None
        return {
            "entity_id": entity.entity_id,
            "position": list(entity.position),
            "frame_id": entity.frame_id,
            "geometry_kind": entity.geometry_kind.value,
            "geometry_type": type(entity.geometry).__name__ if entity.geometry is not None else None,
            "bounds": entity.bounds.to_dict() if entity.bounds else None,
            "metadata": to_json_safe(entity.metadata),
            "timestamp": entity.timestamp,
            "revision": entity.revision,
        }

    @staticmethod
    def _geometry_payload(geometry: Any) -> Any:
        if isinstance(geometry, AABB):
            return {"type": "aabb", "minimum": geometry.minimum.tolist(), "maximum": geometry.maximum.tolist()}
        if isinstance(geometry, Sphere):
            return {"type": "sphere", "center": geometry.center.tolist(), "radius": geometry.radius}
        if isinstance(geometry, OBB):
            return {
                "type": "obb",
                "center": geometry.center.tolist(),
                "half_extents": geometry.half_extents.tolist(),
                "rotation": geometry.rotation.tolist(),
            }
        return {"type": type(geometry).__name__}

    @staticmethod
    def _alignment_payload(result: AlignmentResult) -> dict[str, Any]:
        return result.to_dict()

    @staticmethod
    def _map_summary(record: MapRecord) -> dict[str, Any]:
        payload = record.to_dict()
        representation = record.representation
        if isinstance(representation, PointCloud):
            payload["point_count"] = len(representation)
        elif isinstance(representation, (VoxelGrid, OctreeOccupancy)):
            payload["stored_cells"] = len(representation)
        elif isinstance(representation, OccupancyGrid2D):
            payload["shape"] = [representation.height, representation.width]
        return payload

    def _safe_result(self, result: Any) -> Any:
        return to_json_safe(result, max_depth=10)

    def _publish_result(self, request_id: str, response: Mapping[str, Any]) -> None:
        if not self.publish_shared_memory:
            return
        key = f"{self.result_key_prefix}:{request_id}"
        ttl = None if self.shared_memory_ttl_seconds <= 0 else self.shared_memory_ttl_seconds
        try:
            self.shared_memory.set(
                key,
                self._safe_result(response),
                ttl=ttl,
                tags=["spatial", "result"],
                metadata={
                    "agent": self.name,
                    "agent_id": self.agent_id,
                    "schema": "spatial_agent.result.v1",
                },
            )
        except Exception as exc:
            self._handle_shared_memory_error("set", key, exc)

    def _publish_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if not self.publish_shared_memory:
            return
        message = {
            "schema": "spatial_agent.event.v1",
            "event": event,
            "agent": self.name,
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "payload": self._safe_result(dict(payload)),
        }
        try:
            self.shared_memory.publish(self.event_channel, message)
        except Exception as exc:
            self._handle_shared_memory_error("publish", self.event_channel, exc)

    def _publish_completion_event(
        self,
        operation: str,
        request_id: str,
        response: Mapping[str, Any],
    ) -> None:
        if operation in {"upsert_entity", "remove_entity", "register_frame", "set_frame_parent", "set_frame_transform", "remove_frame", "register_map", "remove_map", "occupancy_update"}:
            event = "state_updated"
        elif operation.startswith("transform_") or operation == "resolve_transform":
            event = "transform_completed"
        elif operation == "relation":
            event = "relation_evaluated"
        else:
            event = "query_completed"
        self._publish_event(
            event,
            {
                "request_id": request_id,
                "operation": operation,
                "status": response.get("status"),
            },
        )

    def _handle_shared_memory_error(self, operation: str, key: str, exc: BaseException) -> None:
        self._mark_runtime_degraded("communication", f"shared_memory.{operation}", exc)
        if self.fail_on_shared_memory_error:
            raise BaseStateError(
                "SpatialAgent shared-memory operation failed",
                component=self.name,
                operation=f"shared_memory.{operation}",
                context={"key": key, "error_type": type(exc).__name__},
                cause=exc,
            ) from exc
        logger.warning(
            "SpatialAgent SharedMemory %s degraded | key=%s | error=%s",
            operation,
            key,
            type(exc).__name__,
        )

    def _record_request_start(self, operation: str, request_id: str) -> None:
        with self._lock:
            self._stats["requests_total"] += 1
            self._stats["per_operation"][operation] += 1
            self._stats["last_operation"] = operation
            self._stats["last_request_id"] = request_id
        self._metric_counter("spatial.requests_total")
        self._metric_counter(f"spatial.operation.{operation}")

    def _record_request_success(
        self,
        operation: str,
        request_id: str,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._stats["requests_successful"] += 1
            self._stats["last_latency_ms"] = latency_ms
            self._last_failure = None
        self._metric_counter("spatial.requests_successful")
        self._metric_value("spatial.request_latency_ms", latency_ms, unit="ms", operation=operation)
        self._mark_runtime_recovered("spatial", operation)

    def _record_request_failure(
        self,
        operation: str,
        request_id: str,
        latency_ms: float,
        exc: BaseException,
    ) -> None:
        failure = {
            "request_id": request_id,
            "operation": operation,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        with self._lock:
            self._stats["requests_failed"] += 1
            self._stats["last_latency_ms"] = latency_ms
            self._last_failure = failure
        self._metric_counter("spatial.requests_failed")
        self._metric_value("spatial.request_latency_ms", latency_ms, unit="ms", operation=operation)
        self._mark_runtime_degraded("spatial", operation, exc, retryable=False)

    def _metric_counter(self, name: str) -> None:
        metric_store = self.metric_store
        increment_counter = getattr(metric_store, "increment_counter", None)
        if increment_counter is None:
            return
        self._run_optional_runtime_operation(
            "telemetry",
            "spatial_agent.metric_counter",
            lambda: increment_counter(name, category="spatial_agent"),
        )

    def _metric_value(self, name: str, value: float, *, unit: str, operation: str) -> None:
        metric_store = self.metric_store
        record_value = getattr(metric_store, "record_value", None)
        if record_value is None:
            return
        self._run_optional_runtime_operation(
            "telemetry",
            "spatial_agent.metric_value",
            lambda: record_value(
                name,
                value,
                category="spatial_agent",
                unit=unit,
                metadata={"operation": operation},
            ),
        )

    # ------------------------------------------------------------------
    # Checkpointing: Agent-owned orchestration state only
    # ------------------------------------------------------------------

    def checkpoint_step(self) -> int:
        with self._lock:
            return int(self._stats["requests_total"])

    def checkpoint_metrics(self) -> Mapping[str, Any]:
        with self._lock:
            total = int(self._stats["requests_total"])
            successful = int(self._stats["requests_successful"])
            failed = int(self._stats["requests_failed"])
        return {
            "requests_total": total,
            "requests_successful": successful,
            "requests_failed": failed,
            "success_rate": (successful / total) if total else 1.0,
        }

    def _export_checkpoint_state(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "schema": self.CHECKPOINT_SCHEMA,
                "stats": {
                    **self._stats,
                    "per_operation": dict(self._stats["per_operation"]),
                },
                "last_failure": dict(self._last_failure) if self._last_failure else None,
            }

    def _import_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise BaseValidationError(
                "SpatialAgent checkpoint state must be a mapping",
                component=self.name,
            )
        schema = state.get("schema")
        if schema not in {None, self.CHECKPOINT_SCHEMA}:
            raise BaseStateError(
                "SpatialAgent checkpoint schema is incompatible",
                component=self.name,
                context={"received": schema, "expected": self.CHECKPOINT_SCHEMA},
            )
        stats = state.get("stats", {})
        if not isinstance(stats, Mapping):
            raise BaseValidationError(
                "SpatialAgent checkpoint stats must be a mapping",
                component=self.name,
            )
        restored = {
            "requests_total": max(0, int(stats.get("requests_total", 0))),
            "requests_successful": max(0, int(stats.get("requests_successful", 0))),
            "requests_failed": max(0, int(stats.get("requests_failed", 0))),
            "per_operation": Counter(
                {
                    str(key): max(0, int(value))
                    for key, value in dict(stats.get("per_operation", {})).items()
                }
            ),
            "last_operation": stats.get("last_operation"),
            "last_request_id": stats.get("last_request_id"),
            "last_latency_ms": stats.get("last_latency_ms"),
        }
        failure = state.get("last_failure")
        if failure is not None and not isinstance(failure, Mapping):
            raise BaseValidationError(
                "SpatialAgent checkpoint last_failure must be a mapping or null",
                component=self.name,
            )
        with self._lock:
            self._stats = restored
            self._last_failure = dict(failure) if isinstance(failure, Mapping) else None
        logger.info("Spatial Agent checkpoint state restored | requests=%d", restored["requests_total"])


__all__ = ["SpatialAgent"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Spatial Agent ===\n")
    printer.status("TEST", "Spatial Agent module import successful", "info")
