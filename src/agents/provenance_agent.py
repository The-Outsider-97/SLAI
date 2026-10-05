"""SLAI v2.3 Provenance Agent orchestration façade.

The Agent coordinates the completed provenance subsystem across SLAI while
preserving a strict evidence-only boundary:

    Provenance records/reconstructs derivation evidence; it does not judge it.

Subsystem algorithms and durable facts remain under ``src.agents.provenance``.
SharedMemory is used only for lightweight cross-agent coordination/reference
publication. ``ProvenanceMemory`` is local checkpoint/lifecycle provenance
state. ``CheckpointManager`` remains BaseAgent's generic checkpoint authority.

The orchestration model is informed by OPM/W3C PROV, PrIMe, PASS, ModelDB,
PROV-ML, ReproZip, Software Heritage, and in-toto.  Those foundations motivate
stable identities, graph-oriented ancestry, multi-parent derivation, and
checkpoint/custody reconstruction without introducing trust or quality scoring.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any, Optional

from .base_agent import BaseAgent
from .base.utils.base_errors import BaseConfigurationError, BaseStateError
from .base.utils.base_helpers import coerce_bool, coerce_int
from .base.utils.main_config_loader import get_config_section
from .runtime_contracts import RuntimeLifecycle
from .provenance.provenance_custody import ProvenanceCustody
from .provenance.provenance_lineage import ProvenanceLineage
from .provenance.provenance_memory import ProvenanceMemory
from .provenance.provenance_store import ProvenanceStore
from .provenance.provenance_types import CheckpointRecord, ProvenanceEntity
from .provenance.lineage.artifact_lineage import ArtifactLineage
from .provenance.lineage.dataset_lineage import DatasetLineage
from .provenance.lineage.dependancy_lineage import DependencyLineage
from .provenance.lineage.model_lineage import ModelLineage
from .provenance.lineage.transformation_lineage import TransformationLineage
from .provenance.modules.lineage_graph import LineageGraph
from .provenance.modules.reproducibility import Reproducibility
from .provenance.modules.source_registry import SourceRegistry
from .provenance.utils.provenance_errors import *
from .provenance.utils.provenance_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Agent")
printer = PrettyPrinter()


class ProvenanceAgent(BaseAgent):
    """System-wide orchestration façade for SLAI provenance infrastructure."""

    AGENT_KEY = "provenance_agent"
    STATE_UPDATED_TOPIC = "provenance_agent:state_updated"
    CHECKPOINT_SCHEMA = "slai.provenance-agent.state.v3"
    CHECKPOINTING_SUPPORTED = True

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Optional[Mapping[str, Any]] = None, *, checkpoint_manager: Any = None) -> None:
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, checkpoint_manager=checkpoint_manager)
        self.agent_config: dict[str, Any] = dict(get_config_section(self.AGENT_KEY) or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise BaseConfigurationError(
                    "ProvenanceAgent config override must be a mapping",
                    component=self.name,
                    context={"type": type(config).__name__}
                    )
            self.agent_config.update(dict(config))
        self._load_agent_config()
        self._validate_agent_config()

        # One store instance is injected into every service so durable facts and
        # indexes share a single authority. No service is recreated per request.
        self.provenance_store = ProvenanceStore()
        self.local_memory = ProvenanceMemory(store=self.provenance_store, max_checkpoints=self.local_memory_max_checkpoints)
        # Backward-compatible attribute retained; it is the same object, not a
        # second memory layer.
        self.provenance_memory = self.local_memory

        self.provenance_lineage = ProvenanceLineage(store=self.provenance_store)
        self.provenance_custody = ProvenanceCustody(store=self.provenance_store)
        self.source_registry = SourceRegistry(store=self.provenance_store)
        self.lineage_graph = LineageGraph(store=self.provenance_store)
        self.reproducibility = Reproducibility(store=self.provenance_store)
        self.artifact_lineage = ArtifactLineage(store=self.provenance_store)
        self.dataset_lineage = DatasetLineage(store=self.provenance_store)
        self.dependency_lineage = DependencyLineage(store=self.provenance_store)
        self.model_lineage = ModelLineage(store=self.provenance_store)
        self.transformation_lineage = TransformationLineage(store=self.provenance_store)

        self.processed_events = 0
        self.failed_events = 0
        self.last_event_id: Optional[str] = None
        self.last_event_timestamp: Optional[str] = None
        self._last_agent_checkpoint_id: Optional[str] = None
        self._closed = False
        self._event_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

        self.operational_state = "ready"
        self._publish_state()
        logger.info(
            "ProvenanceAgent initialized | publish_shared_memory=%s | max_query_depth=%s | local_checkpoints=%s",
            self.publish_shared_memory,
            self.max_query_depth,
            self.local_memory_max_checkpoints,
        )

    # ------------------------------------------------------------------
    # Agent configuration -- agents_config.yaml only
    # ------------------------------------------------------------------
    def _cfg(self, key: str, default: Any) -> Any:
        return self.agent_config.get(key, default)

    def _load_agent_config(self) -> None:
        self.enabled = coerce_bool(self._cfg("enabled", True), True)
        self.publish_shared_memory = coerce_bool(self._cfg("publish_shared_memory", True), True)
        self.publish_state_updates = coerce_bool(self._cfg("publish_state_updates", True), True)
        self.fail_on_shared_memory_error = coerce_bool(self._cfg("fail_on_shared_memory_error", False), False)
        self.strict_event_validation = coerce_bool(self._cfg("strict_event_validation", True), True)
        self.shared_memory_ttl_seconds = coerce_int(self._cfg("shared_memory_ttl_seconds", 86400),
            86400,
            minimum=0,
            maximum=31_536_000,
        )
        self.max_query_depth = coerce_int(self._cfg("max_query_depth", 64), 64, minimum=1, maximum=10_000)
        self.max_query_results = coerce_int(self._cfg("max_query_results", 2048),
            2048,
            minimum=1,
            maximum=1_000_000,
        )
        self.max_subgraph_depth = coerce_int(self._cfg("max_subgraph_depth", 16), 16, minimum=1, maximum=10_000)
        self.local_memory_max_checkpoints = coerce_int(
            self._cfg("local_memory_max_checkpoints", 1000),
            1000,
            minimum=1,
            maximum=1_000_000,
        )
        self.event_cache_size = coerce_int(self._cfg("event_cache_size", 256), 256, minimum=0, maximum=100_000)
        self.record_agent_checkpoints = coerce_bool(self._cfg("record_agent_checkpoints", True), True)
        self.event_channel = str(self._cfg("event_channel", "provenance.events") or "").strip()
        self.state_key = str(self._cfg("state_key", "provenance_agent.state") or "").strip()
        self.latest_reference_key = str(self._cfg("latest_reference_key", "provenance_agent.latest_reference") or "").strip()

    def _validate_agent_config(self) -> None:
        for field_name in ("event_channel", "state_key", "latest_reference_key"):
            if not getattr(self, field_name):
                raise BaseConfigurationError(
                    f"provenance_agent.{field_name} must be a non-empty string",
                    component=self.name,
                    context={"field": field_name},
                )
        if self.publish_shared_memory:
            missing = [
                method
                for method in ("set", "publish")
                if not callable(getattr(self.shared_memory, method, None))
            ]
            if missing:
                raise BaseConfigurationError(
                    "SharedMemory does not satisfy ProvenanceAgent's coordination contract",
                    component=self.name,
                    context={"missing_methods": missing},
                )

    def _ensure_enabled(self, operation: str) -> None:
        if not self.enabled:
            raise BaseConfigurationError(
                "ProvenanceAgent is disabled by configuration",
                component=self.name,
                operation=operation,
            )
        if self._closed:
            raise BaseStateError(
                "ProvenanceAgent is shut down",
                component=self.name,
                operation=operation,
            )

    # ------------------------------------------------------------------
    # SharedMemory coordination -- lightweight references only
    # ------------------------------------------------------------------
    def _shared_memory_failure(self, operation: str, key: str, exc: BaseException) -> None:
        self._mark_runtime_degraded("coordination", f"shared_memory.{operation}", exc)
        if self.fail_on_shared_memory_error:
            raise BaseStateError(
                "ProvenanceAgent shared-memory operation failed",
                component=self.name,
                operation=f"shared_memory.{operation}",
                context={"key": key, "error_type": type(exc).__name__},
                cause=exc,
            ) from exc
        logger.warning(
            "ProvenanceAgent SharedMemory %s degraded | key=%s | error=%s",
            operation,
            key,
            type(exc).__name__,
        )

    def _shared_set(self, key: str, payload: Mapping[str, Any], *, tags: Iterable[str]) -> None:
        if not self.publish_shared_memory:
            return
        ttl = None if self.shared_memory_ttl_seconds <= 0 else self.shared_memory_ttl_seconds
        try:
            self.shared_memory.set(
                key,
                canonicalize_value(payload),
                ttl=ttl,
                tags=list(tags),
                metadata={
                    "agent": self.name,
                    "agent_id": self.agent_id,
                    "schema": "provenance_agent.reference.v1",
                },
            )
            self._mark_runtime_recovered("coordination", "shared_memory.set")
        except Exception as exc:
            self._shared_memory_failure("set", key, exc)

    def _shared_publish(self, channel: str, payload: Mapping[str, Any]) -> None:
        if not self.publish_shared_memory:
            return
        try:
            self.shared_memory.publish(channel, canonicalize_value(payload))
            self._mark_runtime_recovered("coordination", "shared_memory.publish")
        except Exception as exc:
            self._shared_memory_failure("publish", channel, exc)

    def _publish_reference(self, event_type: str, reference: Mapping[str, Any]) -> None:
        if not self.publish_shared_memory:
            return
        payload = {
            "schema": "provenance_agent.event.v1",
            "event_type": str(event_type),
            "agent_id": self.agent_id,
            "timestamp": get_current_timestamp(),
            "reference": canonicalize_value(reference),
        }
        self._shared_publish(self.event_channel, payload)
        self._shared_set(
            self.latest_reference_key,
            payload,
            tags=("provenance", "reference", str(event_type)),
        )

    def _publish_state(self) -> None:
        if not (self.publish_shared_memory and self.publish_state_updates):
            return
        state = self.provenance_state()
        self._shared_set(self.state_key, state, tags=("provenance", "state"))
        self._shared_publish(self.STATE_UPDATED_TOPIC, state)

    # ------------------------------------------------------------------
    # Compact Agent state
    # ------------------------------------------------------------------
    def provenance_state(self) -> dict[str, Any]:
        with self._lock:
            local = self.local_memory.snapshot(include_checkpoints=False) if hasattr(self, "local_memory") else {}
            return {
                "schema": "slai.provenance-agent.runtime-state.v1",
                "agent_id": self.agent_id,
                "operational_state": self.operational_state,
                "processed_events": int(self.processed_events),
                "failed_events": int(self.failed_events),
                "last_event_id": self.last_event_id,
                "last_event_timestamp": self.last_event_timestamp,
                "last_agent_checkpoint_id": self._last_agent_checkpoint_id,
                "local_memory_revision": local.get("revision"),
                "known_local_checkpoints": local.get("checkpoint_count", 0),
                "persistence_enabled": local.get("persist", True),
            }

    def _mark_success(self, event_type: str, reference: Mapping[str, Any]) -> None:
        timestamp = get_current_timestamp()
        event_id = str(reference.get("event_id") or reference.get("record_id") or reference.get("artifact_id") or reference.get("source_id") or stable_provenance_id("agent-event", event_type, reference))
        with self._lock:
            self.processed_events += 1
            self.last_event_id = event_id
            self.last_event_timestamp = timestamp
        self._publish_reference(event_type, reference)
        self._publish_state()

    def _mark_failure(self, event_type: str, exc: BaseException) -> None:
        with self._lock:
            self.failed_events += 1
            self.last_event_timestamp = get_current_timestamp()
        logger.warning(
            "Provenance operation rejected | operation=%s | error=%s",
            event_type,
            type(exc).__name__,
        )
        self._publish_state()

    # ------------------------------------------------------------------
    # Public provenance capture API
    # ------------------------------------------------------------------
    def register_source(self, source_id: str, source_info: Mapping[str, Any]) -> dict[str, Any]:
        self._ensure_enabled("register_source")
        if not isinstance(source_info, Mapping):
            raise ProvenanceValidationError("source_info must be a mapping")
        # Preserve stable identity across retries. SourceRegistry supplies a
        # timestamp when omitted; reusing the persisted timestamp keeps an
        # otherwise identical retried registration idempotent.
        payload = dict(source_info)
        existing = self.provenance_store.get_source(source_id, strict=False)
        if (
            existing is not None
            and "registered_at" not in payload
            and "timestamp" not in payload
        ):
            payload["registered_at"] = existing.get("registered_at")
        try:
            result = self.source_registry.register_source(source_id, payload)
        except ProvenanceError as exc:
            self._mark_failure("source_registration", exc)
            raise
        self._mark_success("source_registered", {"source_id": result["source_id"]})
        return result

    def register_artifact(
        self,
        artifact_id: str,
        *,
        artifact_type: str = "artifact",
        label: Optional[str] = None,
        source_id: Optional[str] = None,
        digest: Optional[str] = None,
        created_at: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        self._ensure_enabled("register_artifact")
        # A missing timestamp is creation metadata, not a retry nonce. Reuse
        # the authoritative value when the stable artifact already exists.
        existing = self.provenance_store.get_entity(artifact_id, strict=False)
        effective_created_at = created_at
        if effective_created_at is None and existing is not None:
            effective_created_at = existing.get("created_at")
        try:
            entity = ProvenanceEntity(
                entity_id=artifact_id,
                entity_type=artifact_type,
                label=label,
                source_id=source_id,
                digest=digest,
                created_at=effective_created_at or get_current_timestamp(),
                metadata=normalize_metadata(metadata),
            )
            result = self.provenance_store.save_entity(entity)
        except ProvenanceError as exc:
            self._mark_failure("artifact_registration", exc)
            raise
        self._mark_success(
            "artifact_registered",
            {"artifact_id": result["entity_id"], "artifact_type": result["entity_type"]},
        )
        return result

    def record_derivation(
        self,
        artifact_id: str,
        parent_artifact_id: Optional[str] = None,
        transformation: str = "derived",
        timestamp: Optional[str] = None,
        *,
        parent_artifact_ids: Optional[Iterable[str]] = None,
        transformation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        source_ids: Optional[Iterable[str]] = None,
        model_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        self._ensure_enabled("record_derivation")
        try:
            result = self.provenance_lineage.record_lineage(
                artifact_id,
                parent_artifact_id,
                transformation,
                timestamp,
                parent_artifact_ids=parent_artifact_ids,
                transformation_id=transformation_id,
                agent_id=agent_id,
                source_ids=source_ids,
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                metadata=metadata,
            )
        except ProvenanceError as exc:
            self._mark_failure("derivation", exc)
            raise
        self._mark_success(
            "derivation_recorded",
            {"artifact_id": artifact_id, "record_id": result["record_id"]},
        )
        return result

    def record_transformation(self, transformation_id: str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_enabled("record_transformation")
        try:
            result = self.transformation_lineage.record_transformation_lineage(
                transformation_id, **kwargs
            )
        except ProvenanceError as exc:
            self._mark_failure("transformation", exc)
            raise
        self._mark_success("transformation_recorded", {"transformation_id": transformation_id})
        return result

    def record_dependency(self, artifact_id: str, dependency_id: str, relationship: str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_enabled("record_dependency")
        try:
            result = self.dependency_lineage.record_dependency_lineage(artifact_id, dependency_id, relationship, **kwargs)
        except ProvenanceError as exc:
            self._mark_failure("dependency", exc)
            raise
        self._mark_success(
            "dependency_recorded",
            {
                "artifact_id": artifact_id,
                "dependency_id": dependency_id,
                "record_id": result["record_id"],
            },
        )
        return result

    def record_dataset_lineage(
        self,
        dataset_id: str,
        parent_dataset_id: Optional[str | Iterable[str]] = None,
        transformation: Optional[str] = "derived",
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._ensure_enabled("record_dataset_lineage")
        try:
            result = self.dataset_lineage.record_dataset_lineage(dataset_id, parent_dataset_id, transformation, **kwargs)
        except ProvenanceError as exc:
            self._mark_failure("dataset_lineage", exc)
            raise
        self._mark_success(
            "dataset_lineage_recorded",
            {"dataset_id": dataset_id, "record_id": result["record_id"]},
        )
        return result

    def record_model_lineage(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_enabled("record_model_lineage")
        try:
            result = self.model_lineage.record_model_lineage(model_id, **kwargs)
            checkpoint = result.get("checkpoint")
            if isinstance(checkpoint, Mapping):
                # The store already owns the durable fact; local memory adds only
                # the checkpoint lifecycle/index reference.
                self.local_memory.save_checkpoint(checkpoint)
        except ProvenanceError as exc:
            self._mark_failure("model_lineage", exc)
            raise
        reference = {"model_id": model_id}
        checkpoint = result.get("checkpoint")
        if isinstance(checkpoint, Mapping) and checkpoint.get("checkpoint_id"):
            reference["checkpoint_id"] = str(checkpoint["checkpoint_id"])
        self._mark_success("model_lineage_recorded", reference)
        return result

    def record_checkpoint(self, checkpoint: CheckpointRecord | Mapping[str, Any] | str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_enabled("record_checkpoint")
        try:
            result = self.local_memory.save_checkpoint(checkpoint, **kwargs)
        except ProvenanceError as exc:
            self._mark_failure("checkpoint", exc)
            raise
        self._mark_success("checkpoint_recorded", {"checkpoint_id": result["checkpoint_id"]})
        return result

    def record_custody(self, artifact_id: str, new_custodian: str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_enabled("record_custody")
        try:
            result = self.provenance_custody.transfer_custody(artifact_id, new_custodian, **kwargs)
        except ProvenanceError as exc:
            self._mark_failure("custody", exc)
            raise
        self._mark_success("custody_recorded", {"artifact_id": artifact_id, "event_id": result["event_id"]})
        return result

    # ------------------------------------------------------------------
    # Public query API -- delegates to subsystem; no graph engine here
    # ------------------------------------------------------------------
    def track_artifact(self, artifact_id: str) -> dict[str, Any]:
        self._ensure_enabled("track_artifact")
        return self.provenance_store.get_provenance(artifact_id)

    def get_provenance(self, artifact_id: str) -> dict[str, Any]:
        return self.track_artifact(artifact_id)

    def get_lineage(self, artifact_id: str, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
        self._ensure_enabled("get_lineage")
        records = self.provenance_lineage.get_lineage(artifact_id)
        bound = self.max_query_results if limit is None else min(max(int(limit), 0), self.max_query_results)
        return records[:bound]

    def get_ancestors(self, artifact_id: str, *, max_depth: Optional[int] = None) -> list[str]:
        self._ensure_enabled("get_ancestors")
        depth = self.max_query_depth if max_depth is None else min(int(max_depth), self.max_query_depth)
        if depth < 1:
            raise ProvenanceValidationError("max_depth must be >= 1")
        result = self.lineage_graph.ancestors(artifact_id, max_depth=depth)
        return result[: self.max_query_results]

    def get_descendants(self, artifact_id: str, *, max_depth: Optional[int] = None) -> list[str]:
        self._ensure_enabled("get_descendants")
        depth = self.max_query_depth if max_depth is None else min(int(max_depth), self.max_query_depth)
        if depth < 1:
            raise ProvenanceValidationError("max_depth must be >= 1")
        result = self.lineage_graph.descendants(artifact_id, max_depth=depth)
        return result[: self.max_query_results]

    def get_derivation_path(self, source_id: str, output_id: str) -> list[str]:
        self._ensure_enabled("get_derivation_path")
        path = self.lineage_graph.derivation_path(source_id, output_id)
        if len(path) > self.max_query_results:
            raise ProvenanceValidationError(
                "derivation path exceeds ProvenanceAgent query result bound",
                context={"max_query_results": self.max_query_results},
            )
        return path

    def get_lineage_graph(self, artifact_id: str, *, depth: Optional[int] = None, include_descendants: bool = False) -> dict[str, Any]:
        self._ensure_enabled("get_lineage_graph")
        query_depth = self.max_subgraph_depth if depth is None else min(int(depth), self.max_subgraph_depth)
        if query_depth < 1:
            raise ProvenanceValidationError("depth must be >= 1")
        return self.lineage_graph.subgraph(
            artifact_id,
            depth=query_depth,
            include_descendants=bool(include_descendants),
        )

    def chain_of_custody(self, artifact_id: str) -> list[dict[str, Any]]:
        """Retrieve custody history without mutating provenance state."""

        self._ensure_enabled("chain_of_custody")
        return self.provenance_custody.chain_of_custody(artifact_id)

    def get_custody_history(self, artifact_id: str) -> list[dict[str, Any]]:
        return self.chain_of_custody(artifact_id)

    def get_reproducibility_report(self, artifact_id: str) -> dict[str, Any]:
        self._ensure_enabled("get_reproducibility_report")
        return self.reproducibility.reproducibility_report(artifact_id)

    # ------------------------------------------------------------------
    # Normalized cross-agent event boundary
    # ------------------------------------------------------------------
    def record_event(self, event: Mapping[str, Any]) -> Any:
        """Normalize one generic provenance event into the typed subsystem API.

        Event shape::

            {
                "event_type": "derivation",
                "event_id": "optional-stable-id",
                "payload": {...}
            }

        If ``payload`` is omitted, all fields other than ``event_type`` and
        ``event_id`` are treated as the payload.  Unknown event types and
        malformed payloads are rejected; ambiguous Quality/Observability events
        are never coerced into provenance facts.
        """

        self._ensure_enabled("record_event")
        if not isinstance(event, Mapping):
            raise ProvenanceValidationError("provenance event must be a mapping")
        raw_type = event.get("event_type", event.get("type"))
        if raw_type is None:
            raise ProvenanceValidationError("provenance event requires event_type")
        event_type = str(raw_type).strip().lower()
        if not event_type:
            raise ProvenanceValidationError("event_type must be non-empty")

        raw_payload = event.get("payload")
        if raw_payload is None:
            payload = {
                str(key): value
                for key, value in event.items()
                if key not in {"event_type", "type", "event_id"}
            }
        elif isinstance(raw_payload, Mapping):
            payload = dict(raw_payload)
        else:
            raise ProvenanceValidationError("event payload must be a mapping")

        canonical_payload = canonicalize_value(payload)
        event_id = str(
            event.get("event_id")
            or stable_provenance_id("event", event_type, canonical_payload)
        )
        fingerprint = stable_provenance_id("event-fingerprint", event_type, canonical_payload)

        with self._lock:
            cached = self._event_cache.get(event_id)
            if cached is not None:
                if cached["fingerprint"] != fingerprint:
                    raise ProvenanceConflictError(
                        "event_id was reused with conflicting provenance payload",
                        context={"event_id": event_id, "event_type": event_type},
                    )
                self._event_cache.move_to_end(event_id)
                return canonicalize_value(cached["result"])

        try:
            result = self._dispatch_event(event_type, payload)
        except ProvenanceError:
            raise

        if self.event_cache_size > 0:
            with self._lock:
                self._event_cache[event_id] = {
                    "fingerprint": fingerprint,
                    "result": canonicalize_value(result),
                }
                self._event_cache.move_to_end(event_id)
                while len(self._event_cache) > self.event_cache_size:
                    self._event_cache.popitem(last=False)
        return result

    def _dispatch_event(self, event_type: str, payload: Mapping[str, Any]) -> Any:
        aliases = {
            "source": "source",
            "source_registration": "source",
            "artifact": "artifact",
            "artifact_registration": "artifact",
            "derivation": "derivation",
            "lineage": "derivation",
            "transformation": "transformation",
            "dependency": "dependency",
            "dataset": "dataset",
            "dataset_lineage": "dataset",
            "model": "model",
            "model_lineage": "model",
            "checkpoint": "checkpoint",
            "custody": "custody",
        }
        normalized = aliases.get(event_type)
        if normalized is None:
            raise ProvenanceValidationError(
                "unsupported provenance event type",
                context={"event_type": event_type},
            )

        data = dict(payload)
        if normalized == "source":
            source_id = data.pop("source_id", None)
            if source_id is None:
                raise ProvenanceValidationError("source event requires source_id")
            source_info = data.pop("source_info", None)
            if source_info is None:
                source_info = data
            elif data and self.strict_event_validation:
                raise ProvenanceValidationError(
                    "source event cannot mix source_info with extra top-level payload fields"
                )
            if not isinstance(source_info, Mapping):
                raise ProvenanceValidationError("source_info must be a mapping")
            return self.register_source(str(source_id), source_info)

        if normalized == "artifact":
            artifact_id = data.pop("artifact_id", None)
            if artifact_id is None:
                raise ProvenanceValidationError("artifact event requires artifact_id")
            return self.register_artifact(str(artifact_id), **data)

        if normalized == "derivation":
            artifact_id = data.pop("artifact_id", None)
            if artifact_id is None:
                raise ProvenanceValidationError("derivation event requires artifact_id")
            return self.record_derivation(str(artifact_id), **data)

        if normalized == "transformation":
            transformation_id = data.pop("transformation_id", None)
            if transformation_id is None:
                raise ProvenanceValidationError("transformation event requires transformation_id")
            return self.record_transformation(str(transformation_id), **data)

        if normalized == "dependency":
            artifact_id = data.pop("artifact_id", None)
            dependency_id = data.pop("dependency_id", None)
            relationship = data.pop("relationship", None)
            if artifact_id is None or dependency_id is None or relationship is None:
                raise ProvenanceValidationError(
                    "dependency event requires artifact_id, dependency_id, and relationship"
                )
            return self.record_dependency(
                str(artifact_id), str(dependency_id), str(relationship), **data
            )

        if normalized == "dataset":
            dataset_id = data.pop("dataset_id", None)
            if dataset_id is None:
                raise ProvenanceValidationError("dataset event requires dataset_id")
            return self.record_dataset_lineage(str(dataset_id), **data)

        if normalized == "model":
            model_id = data.pop("model_id", None)
            if model_id is None:
                raise ProvenanceValidationError("model event requires model_id")
            return self.record_model_lineage(str(model_id), **data)

        if normalized == "checkpoint":
            checkpoint = data.pop("checkpoint", None)
            if checkpoint is None:
                checkpoint = data.pop("checkpoint_id", None)
            if checkpoint is None:
                raise ProvenanceValidationError("checkpoint event requires checkpoint or checkpoint_id")
            return self.record_checkpoint(checkpoint, **data)

        artifact_id = data.pop("artifact_id", None)
        new_custodian = data.pop("new_custodian", data.pop("owner", None))
        if artifact_id is None or new_custodian is None:
            raise ProvenanceValidationError(
                "custody event requires artifact_id and new_custodian"
            )
        return self.record_custody(str(artifact_id), str(new_custodian), **data)

    # ------------------------------------------------------------------
    # BaseAgent task/factory compatibility
    # ------------------------------------------------------------------
    def perform_task(self, task_data: Any) -> Any:
        if not isinstance(task_data, Mapping):
            raise ProvenanceValidationError("ProvenanceAgent task_data must be a mapping")
        operation = str(task_data.get("operation", "record_event")).strip().lower()
        payload = task_data.get("payload", task_data)
        if not isinstance(payload, Mapping):
            raise ProvenanceValidationError("ProvenanceAgent task payload must be a mapping")
        data = dict(payload)
        data.pop("operation", None)

        if operation == "record_event":
            return self.record_event(data)
        if operation in {"track_artifact", "get_provenance"}:
            return self.track_artifact(str(data["artifact_id"]))
        if operation == "get_lineage":
            artifact_id = str(data.pop("artifact_id"))
            return self.get_lineage(artifact_id, **data)
        if operation == "get_ancestors":
            artifact_id = str(data.pop("artifact_id"))
            return self.get_ancestors(artifact_id, **data)
        if operation == "get_descendants":
            artifact_id = str(data.pop("artifact_id"))
            return self.get_descendants(artifact_id, **data)
        if operation == "get_derivation_path":
            source_id = str(data.pop("source_id"))
            output_id = str(data.pop("output_id"))
            return self.get_derivation_path(source_id, output_id)
        if operation == "get_lineage_graph":
            artifact_id = str(data.pop("artifact_id"))
            return self.get_lineage_graph(artifact_id, **data)
        if operation in {"chain_of_custody", "get_custody_history"}:
            return self.chain_of_custody(str(data["artifact_id"]))
        if operation == "get_reproducibility_report":
            return self.get_reproducibility_report(str(data["artifact_id"]))
        raise ProvenanceValidationError(
            "unsupported ProvenanceAgent task operation",
            context={"operation": operation},
        )

    def predict(self, state: Any, context: Any = None) -> Any:
        if not isinstance(state, Mapping):
            raise ProvenanceValidationError("ProvenanceAgent.predict requires a mapping")
        payload = dict(state)
        if context is not None and "context" not in payload:
            payload["context"] = context
        return self.perform_task(payload)

    def act(self, task_data: Any, context: Any = None) -> Any:
        return self.predict(task_data, context=context)

    def capabilities(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "capture": (
                "source",
                "artifact",
                "derivation",
                "transformation",
                "dependency",
                "dataset_lineage",
                "model_lineage",
                "checkpoint",
                "custody",
            ),
            "queries": (
                "provenance",
                "lineage",
                "ancestors",
                "descendants",
                "derivation_path",
                "lineage_graph",
                "custody_history",
                "reproducibility",
            ),
            "shared_memory": self.publish_shared_memory,
            "checkpointing": self.checkpointing_enabled,
        }

    # ------------------------------------------------------------------
    # BaseAgent durable checkpoint integration
    # ------------------------------------------------------------------
    def checkpoint_step(self) -> Optional[int]:
        return int(self.processed_events)

    def checkpoint_metrics(self) -> Mapping[str, Any]:
        return {
            "processed_events": int(self.processed_events),
            "failed_events": int(self.failed_events),
        }

    def _export_checkpoint_state(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "schema_version": self.CHECKPOINT_SCHEMA,
                "processed_events": int(self.processed_events),
                "failed_events": int(self.failed_events),
                "last_event_id": self.last_event_id,
                "last_event_timestamp": self.last_event_timestamp,
                "last_agent_checkpoint_id": self._last_agent_checkpoint_id,
                # Reference only: local memory remains independently persisted.
                "local_memory": self.local_memory.snapshot(include_checkpoints=False),
            }

    def _import_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ProvenanceValidationError("ProvenanceAgent checkpoint state must be a mapping")
        if state.get("schema_version") != self.CHECKPOINT_SCHEMA:
            raise ProvenanceValidationError(
                "unsupported ProvenanceAgent checkpoint schema",
                context={"schema_version": state.get("schema_version")},
            )
        try:
            processed_events = int(state.get("processed_events", 0))
            failed_events = int(state.get("failed_events", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProvenanceValidationError(
                "ProvenanceAgent checkpoint counters must be integers",
                cause=exc,
            ) from exc
        if processed_events < 0 or failed_events < 0:
            raise ProvenanceValidationError("ProvenanceAgent checkpoint counters cannot be negative")

        local_reference = state.get("local_memory")
        if local_reference is not None:
            if not isinstance(local_reference, Mapping):
                raise ProvenanceValidationError(
                    "ProvenanceAgent local_memory checkpoint reference must be a mapping"
                )
            local_schema = local_reference.get("schema_version")
            if local_schema not in {None, ProvenanceMemory.MANIFEST_SCHEMA}:
                raise ProvenanceValidationError(
                    "unsupported ProvenanceMemory checkpoint-reference schema",
                    context={"schema_version": local_schema},
                )

        # Reload the independently persisted local checkpoint-provenance index;
        # never deserialize the durable ProvenanceStore into BaseAgent state.
        self.local_memory.restore()
        with self._lock:
            self.processed_events = processed_events
            self.failed_events = failed_events
            self.last_event_id = (
                str(state["last_event_id"]) if state.get("last_event_id") else None
            )
            self.last_event_timestamp = (
                str(state["last_event_timestamp"])
                if state.get("last_event_timestamp")
                else None
            )
            self._last_agent_checkpoint_id = (
                str(state["last_agent_checkpoint_id"])
                if state.get("last_agent_checkpoint_id")
                else None
            )
            self.operational_state = "ready"
        self._publish_state()

    def save_checkpoint(
        self,
        version: Optional[str] = None,
        *,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        overwrite: Optional[bool] = None,
    ) -> Any:
        result = super().save_checkpoint(
            version,
            reason=reason,
            metadata=metadata,
            overwrite=overwrite,
        )
        if not self.record_agent_checkpoints:
            return result

        record = getattr(result, "record", None)
        checkpoint_id = getattr(record, "checkpoint_id", None)
        if checkpoint_id is None:
            raise ProvenanceStorageError(
                "CheckpointManager save result does not expose checkpoint identity",
                context={"result_type": type(result).__name__},
            )
        try:
            configuration_id = stable_provenance_id("agent-config", self.agent_config)
            provenance_record = self.local_memory.save_checkpoint(
                str(checkpoint_id),
                parent_checkpoint_id=self._last_agent_checkpoint_id,
                configuration_id=configuration_id,
                code_version=__version__,
                framework_versions={
                    "checkpoint_schema": getattr(record, "schema_version", None),
                },
                metadata={
                    "agent": self.name,
                    "agent_id": self.agent_id,
                    "checkpoint_version": getattr(record, "version", None),
                    "checkpoint_path": str(getattr(record, "path", "")),
                    "reason": reason,
                },
                timestamp=getattr(record, "created_at", None),
            )
        except ProvenanceError as exc:
            self._mark_runtime_degraded("persistence", "checkpoint.provenance", exc)
            raise ProvenanceStorageError(
                "SLAI checkpoint committed but provenance recording failed",
                context={"checkpoint_id": str(checkpoint_id)},
                cause=exc,
            ) from exc

        with self._lock:
            self._last_agent_checkpoint_id = provenance_record["checkpoint_id"]
        self._mark_runtime_recovered("persistence", "checkpoint.provenance")
        self._publish_reference(
            "agent_checkpoint_recorded",
            {"checkpoint_id": provenance_record["checkpoint_id"]},
        )
        self._publish_state()
        return result

    def restore_checkpoint(
        self,
        version: Optional[str] = None,
        *,
        verify_integrity: bool = True,
    ) -> Any:
        result = super().restore_checkpoint(version, verify_integrity=verify_integrity)
        record = getattr(result, "record", None)
        checkpoint_id = getattr(record, "checkpoint_id", None)
        if checkpoint_id:
            with self._lock:
                self._last_agent_checkpoint_id = str(checkpoint_id)
        self.local_memory.restore()
        self.operational_state = "ready"
        self._publish_reference(
            "agent_checkpoint_restored",
            {"checkpoint_id": str(checkpoint_id) if checkpoint_id else None},
        )
        self._publish_state()
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        lifecycle = self._runtime_status.lifecycle
        if lifecycle in {RuntimeLifecycle.STOPPING, RuntimeLifecycle.STOPPED}:
            return
        self._transition_runtime_lifecycle(RuntimeLifecycle.STOPPING)
        self.operational_state = "stopping"
        self._publish_state()

        failure: Optional[BaseException] = None
        try:
            self.local_memory.flush()
            self.local_memory.close()
            self._mark_runtime_recovered("persistence", "provenance_memory.shutdown")
        except Exception as exc:
            failure = exc
            self._mark_runtime_degraded(
                "persistence", "provenance_memory.shutdown", exc, retryable=False
            )
            logger.error("ProvenanceAgent local-memory shutdown failed: %s", exc)
        finally:
            with self._lock:
                self._closed = True
                self.operational_state = "stopped"
            self._transition_runtime_lifecycle(RuntimeLifecycle.STOPPED)
            self._publish_state()

        if failure is not None:
            if isinstance(failure, ProvenanceError):
                raise failure
            raise ProvenanceStorageError(
                "ProvenanceAgent shutdown persistence failed", cause=failure
            ) from failure


__all__ = ["ProvenanceAgent"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Provenance Agent ===\n")
    printer.status("TEST", "Provenance Agent initialized", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    config = {"publish_shared_memory": False}

    agent = ProvenanceAgent(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
    printer.status("START", agent, "info")
