"""
Shared lineage contract for all provenance specializations.

W3C PROV-DM and PROV Constraints define the semantic contract: derivations are
explicit causal relationships between entities, may have multiple parents, and
must not form inconsistent cycles.  Moreau et al. (2011) motivates the graph
representation used by all specialized lineage modules.
"""

from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Iterable, Mapping
from typing import Any, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.provenance_helpers import *
from ..provenance_store import ProvenanceStore
from ..provenance_types import (
    LineageRecord,
    ProvenanceActivity,
    ProvenanceAgentRef,
    ProvenanceEntity,
    ProvenanceRelation,
    ProvenanceRelationType,
)
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Base Lineage")
printer = PrettyPrinter()


class BaseLineage:
    """Common multi-parent derivation recorder used by specialized lineage types."""

    LINEAGE_TYPE = "artifact"

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        store: Optional[ProvenanceStore] = None,
    ) -> None:
        self.config = load_global_config()
        self.lineage_config = dict(
            config or get_config_section("lineage", config=self.config, default={})
        )
        self.store = store or ProvenanceStore()

    def record_lineage(
        self,
        artifact_id: str,
        parent_artifact_id: Optional[str | Iterable[str]] = None,
        transformation: Optional[str] = None,
        timestamp: Optional[str] = None,
        *,
        parent_artifact_ids: Optional[Iterable[str]] = None,
        transformation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        source_ids: Optional[Iterable[str]] = None,
        model_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        lineage_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record one immutable derivation event with zero or more parents."""

        artifact = require_identifier(artifact_id, field_name="artifact_id")
        combined_parents = []
        if parent_artifact_id is not None:
            if isinstance(parent_artifact_id, str):
                combined_parents.append(parent_artifact_id)
            else:
                combined_parents.extend(parent_artifact_id)
        if parent_artifact_ids is not None:
            combined_parents.extend(parent_artifact_ids)
        parents = normalize_id_sequence(combined_parents, field_name="parent_artifact_ids")
        sources = normalize_id_sequence(source_ids, field_name="source_ids")
        event_timestamp = timestamp or get_current_timestamp()
        effective_type = lineage_type or self.LINEAGE_TYPE
        normalized_metadata = normalize_metadata(metadata)

        effective_transformation_id = transformation_id
        synthesized_activity = effective_transformation_id is None and bool(transformation)
        if synthesized_activity:
            effective_transformation_id = stable_provenance_id(
                "activity",
                effective_type,
                artifact,
                parents,
                transformation,
                event_timestamp,
            )

        record_id = stable_provenance_id(
            "lineage",
            effective_type,
            artifact,
            parents,
            effective_transformation_id,
            transformation,
            event_timestamp,
            agent_id,
            sources,
            model_id,
            checkpoint_id,
        )
        if agent_id:
            self.store.save_agent(ProvenanceAgentRef(agent_id=agent_id, agent_type="slai_component"))
        if synthesized_activity and effective_transformation_id is not None:
            self.store.save_activity(
                ProvenanceActivity(
                    activity_id=effective_transformation_id,
                    activity_type=str(transformation),
                    started_at=event_timestamp,
                    ended_at=event_timestamp,
                    agent_id=agent_id,
                    metadata={"synthetic_from_lineage": True},
                )
            )

        record = LineageRecord(
            record_id=record_id,
            artifact_id=artifact,
            parent_artifact_ids=parents,
            transformation_id=effective_transformation_id,
            transformation=transformation,
            timestamp=event_timestamp,
            agent_id=agent_id,
            source_ids=sources,
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            lineage_type=effective_type,
            metadata=normalized_metadata,
        )
        persisted = self.store.save_lineage_record(record)
        self._persist_relations(record)
        return persisted

    def _persist_relations(self, record: LineageRecord) -> None:
        for parent_id in record.parent_artifact_ids:
            relation = ProvenanceRelation(
                relation_id=stable_provenance_id(
                    "relation",
                    ProvenanceRelationType.WAS_DERIVED_FROM.value,
                    record.artifact_id,
                    parent_id,
                    record.record_id,
                ),
                relation_type=ProvenanceRelationType.WAS_DERIVED_FROM,
                subject_id=record.artifact_id,
                object_id=parent_id,
                activity_id=record.transformation_id,
                timestamp=record.timestamp,
                metadata={"lineage_record_id": record.record_id},
            )
            self.store.save_relation(relation)

        if record.transformation_id:
            relation = ProvenanceRelation(
                relation_id=stable_provenance_id(
                    "relation",
                    ProvenanceRelationType.WAS_GENERATED_BY.value,
                    record.artifact_id,
                    record.transformation_id,
                    record.record_id,
                ),
                relation_type=ProvenanceRelationType.WAS_GENERATED_BY,
                subject_id=record.artifact_id,
                object_id=record.transformation_id,
                activity_id=record.transformation_id,
                timestamp=record.timestamp,
                metadata={"lineage_record_id": record.record_id},
            )
            self.store.save_relation(relation)

        if record.agent_id:
            relation = ProvenanceRelation(
                relation_id=stable_provenance_id(
                    "relation",
                    ProvenanceRelationType.WAS_ATTRIBUTED_TO.value,
                    record.artifact_id,
                    record.agent_id,
                    record.record_id,
                ),
                relation_type=ProvenanceRelationType.WAS_ATTRIBUTED_TO,
                subject_id=record.artifact_id,
                object_id=record.agent_id,
                activity_id=record.transformation_id,
                timestamp=record.timestamp,
                metadata={"lineage_record_id": record.record_id},
            )
            self.store.save_relation(relation)

    def ensure_entity(
        self,
        entity_id: str,
        *,
        entity_type: str = "artifact",
        source_id: Optional[str] = None,
        digest: Optional[str] = None,
        created_at: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Register an entity once without mutating an existing immutable identity."""

        existing = self.store.get_entity(entity_id, strict=False)
        if existing is not None:
            return existing
        entity = ProvenanceEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            source_id=source_id,
            digest=digest,
            created_at=created_at or get_current_timestamp(),
            metadata=metadata or {},
        )
        return self.store.save_entity(entity)

    def get_lineage(self, artifact_id: str) -> list[dict[str, Any]]:
        """Return all direct derivation records for an artifact."""

        return self.store.get_lineage_records(artifact_id)


__all__ = ["BaseLineage"]


if __name__ == "__main__":
    configure_logging()
    printer.status("PROVENANCE", "base lineage contract loaded", "success")
