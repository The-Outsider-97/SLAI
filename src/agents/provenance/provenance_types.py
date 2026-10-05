"""
Typed provenance domain model for SLAI v2.3.

The model follows W3C PROV-DM's Entity/Activity/Agent and relationship semantics
without requiring RDF.  Records are frozen dataclasses so persisted provenance
facts are treated as immutable events.  SLAI-specific record types cover model,
checkpoint, dataset, dependency, custody, and reproducibility metadata while
remaining evidence-only: no trust, quality, or correctness judgment is stored.

Academic basis: W3C PROV-DM/PROV Constraints; Moreau et al. (2011); Cheney,
Chiticariu & Tan (2009); Buneman, Khanna & Tan (2001); PROV-ML/ModelDB for ML
lineage extensions.
"""

from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

from .utils.provenance_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Provenance Types")
printer = PrettyPrinter()


TRecord = TypeVar("TRecord", bound="ProvenanceRecord")


class ProvenanceRelationType(str, Enum):
    """PROV-compatible relationship names used by the lightweight SLAI model."""

    USED = "used"
    WAS_GENERATED_BY = "wasGeneratedBy"
    WAS_DERIVED_FROM = "wasDerivedFrom"
    WAS_ASSOCIATED_WITH = "wasAssociatedWith"
    WAS_ATTRIBUTED_TO = "wasAttributedTo"
    SPECIALIZATION_OF = "specializationOf"
    ALTERNATE_OF = "alternateOf"
    DEPENDS_ON = "dependsOn"


class ProvenanceRecord:
    """Mixin for deterministic record serialization."""

    def to_dict(self) -> Dict[str, Any]:
        dataclass_fields = getattr(self, "__dataclass_fields__", {})
        payload = {item.name: getattr(self, item.name) for item in dataclass_fields.values()}
        normalized = canonicalize_value(payload)
        return dict(normalized)

    @classmethod
    def from_dict(cls: Type[TRecord], payload: Mapping[str, Any]) -> TRecord:
        return cls(**dict(payload))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProvenanceEntity(ProvenanceRecord):
    entity_id: str
    entity_type: str = "artifact"
    label: Optional[str] = None
    source_id: Optional[str] = None
    digest: Optional[str] = None
    created_at: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", require_identifier(self.entity_id, field_name="entity_id"))
        object.__setattr__(self, "entity_type", require_identifier(self.entity_type, field_name="entity_type"))
        if self.source_id is not None:
            object.__setattr__(self, "source_id", require_identifier(self.source_id, field_name="source_id"))
        object.__setattr__(self, "created_at", normalize_timestamp(self.created_at))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class ProvenanceActivity(ProvenanceRecord):
    activity_id: str
    activity_type: str
    started_at: str = field(default_factory=get_current_timestamp)
    ended_at: Optional[str] = None
    agent_id: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity_id", require_identifier(self.activity_id, field_name="activity_id"))
        object.__setattr__(self, "activity_type", require_identifier(self.activity_type, field_name="activity_type"))
        object.__setattr__(self, "started_at", normalize_timestamp(self.started_at))
        if self.ended_at is not None:
            object.__setattr__(self, "ended_at", normalize_timestamp(self.ended_at))
        if self.agent_id is not None:
            object.__setattr__(self, "agent_id", require_identifier(self.agent_id, field_name="agent_id"))
        object.__setattr__(self, "parameters", normalize_metadata(self.parameters))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class ProvenanceAgentRef(ProvenanceRecord):
    agent_id: str
    agent_type: str = "slai_component"
    label: Optional[str] = None
    version: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", require_identifier(self.agent_id, field_name="agent_id"))
        object.__setattr__(self, "agent_type", require_identifier(self.agent_type, field_name="agent_type"))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class ProvenanceRelation(ProvenanceRecord):
    relation_id: str
    relation_type: ProvenanceRelationType | str
    subject_id: str
    object_id: str
    activity_id: Optional[str] = None
    timestamp: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_id", require_identifier(self.relation_id, field_name="relation_id"))
        relation_type = (
            self.relation_type
            if isinstance(self.relation_type, ProvenanceRelationType)
            else ProvenanceRelationType(str(self.relation_type))
        )
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "subject_id", require_identifier(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "object_id", require_identifier(self.object_id, field_name="object_id"))
        if self.activity_id is not None:
            object.__setattr__(self, "activity_id", require_identifier(self.activity_id, field_name="activity_id"))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class ProvenanceAttribute(ProvenanceRecord):
    name: str
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_identifier(self.name, field_name="attribute_name"))
        object.__setattr__(self, "value", canonicalize_value(self.value))


@dataclass(frozen=True)
class DerivationRecord(ProvenanceRecord):
    record_id: str
    artifact_id: str
    parent_artifact_ids: Tuple[str, ...] = ()
    transformation_id: Optional[str] = None
    transformation: Optional[str] = None
    timestamp: str = field(default_factory=get_current_timestamp)
    agent_id: Optional[str] = None
    source_ids: Tuple[str, ...] = ()
    model_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    lineage_type: str = "artifact"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", require_identifier(self.record_id, field_name="record_id"))
        object.__setattr__(self, "artifact_id", require_identifier(self.artifact_id, field_name="artifact_id"))
        parents = normalize_id_sequence(self.parent_artifact_ids, field_name="parent_artifact_ids")
        if self.artifact_id in parents:
            from .utils.provenance_errors import ProvenanceGraphError

            raise ProvenanceGraphError(
                "an artifact cannot be derived from itself",
                context={"artifact_id": self.artifact_id},
            )
        object.__setattr__(self, "parent_artifact_ids", parents)
        object.__setattr__(self, "source_ids", normalize_id_sequence(self.source_ids, field_name="source_ids"))
        if self.transformation_id is not None:
            object.__setattr__(self, "transformation_id", require_identifier(self.transformation_id, field_name="transformation_id"))
        if self.agent_id is not None:
            object.__setattr__(self, "agent_id", require_identifier(self.agent_id, field_name="agent_id"))
        if self.model_id is not None:
            object.__setattr__(self, "model_id", require_identifier(self.model_id, field_name="model_id"))
        if self.checkpoint_id is not None:
            object.__setattr__(self, "checkpoint_id", require_identifier(self.checkpoint_id, field_name="checkpoint_id"))
        object.__setattr__(self, "lineage_type", require_identifier(self.lineage_type, field_name="lineage_type"))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    @property
    def parent_artifact_id(self) -> Optional[str]:
        """Backward-compatible singular parent accessor."""

        return self.parent_artifact_ids[0] if self.parent_artifact_ids else None


@dataclass(frozen=True)
class LineageRecord(DerivationRecord):
    """Canonical SLAI derivation record."""


@dataclass(frozen=True)
class UsageRecord(ProvenanceRecord):
    record_id: str
    activity_id: str
    entity_id: str
    timestamp: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("record_id", "activity_id", "entity_id"):
            object.__setattr__(self, field_name, require_identifier(getattr(self, field_name), field_name=field_name))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class GenerationRecord(ProvenanceRecord):
    record_id: str
    entity_id: str
    activity_id: str
    timestamp: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("record_id", "entity_id", "activity_id"):
            object.__setattr__(self, field_name, require_identifier(getattr(self, field_name), field_name=field_name))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class AssociationRecord(ProvenanceRecord):
    record_id: str
    activity_id: str
    agent_id: str
    timestamp: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("record_id", "activity_id", "agent_id"):
            object.__setattr__(self, field_name, require_identifier(getattr(self, field_name), field_name=field_name))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class AttributionRecord(ProvenanceRecord):
    record_id: str
    entity_id: str
    agent_id: str
    timestamp: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("record_id", "entity_id", "agent_id"):
            object.__setattr__(self, field_name, require_identifier(getattr(self, field_name), field_name=field_name))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class CustodyRecord(ProvenanceRecord):
    event_id: str
    artifact_id: str
    previous_custodian: Optional[str]
    new_custodian: str
    activity: Optional[str] = None
    timestamp: str = field(default_factory=get_current_timestamp)
    artifact_digest: Optional[str] = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", require_identifier(self.event_id, field_name="event_id"))
        object.__setattr__(self, "artifact_id", require_identifier(self.artifact_id, field_name="artifact_id"))
        if self.previous_custodian is not None:
            object.__setattr__(self, "previous_custodian", require_identifier(self.previous_custodian, field_name="previous_custodian"))
        object.__setattr__(self, "new_custodian", require_identifier(self.new_custodian, field_name="new_custodian"))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))
        object.__setattr__(self, "context", normalize_metadata(self.context))

    @property
    def owner(self) -> str:
        """Backward-compatible name for the resulting custodian."""

        return self.new_custodian


@dataclass(frozen=True)
class SourceRecord(ProvenanceRecord):
    source_id: str
    source_type: str = "unknown"
    locator: Optional[str] = None
    digest: Optional[str] = None
    external_identifier: Optional[str] = None
    registered_at: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", require_identifier(self.source_id, field_name="source_id"))
        object.__setattr__(self, "source_type", require_identifier(self.source_type, field_name="source_type"))
        object.__setattr__(self, "registered_at", normalize_timestamp(self.registered_at))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class DatasetRecord(ProvenanceRecord):
    dataset_id: str
    version: Optional[str] = None
    parent_dataset_ids: Tuple[str, ...] = ()
    source_ids: Tuple[str, ...] = ()
    created_at: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", require_identifier(self.dataset_id, field_name="dataset_id"))
        object.__setattr__(self, "parent_dataset_ids", normalize_id_sequence(self.parent_dataset_ids, field_name="parent_dataset_ids"))
        object.__setattr__(self, "source_ids", normalize_id_sequence(self.source_ids, field_name="source_ids"))
        object.__setattr__(self, "created_at", normalize_timestamp(self.created_at))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class ModelRecord(ProvenanceRecord):
    model_id: str
    version: Optional[str] = None
    parent_model_ids: Tuple[str, ...] = ()
    checkpoint_ids: Tuple[str, ...] = ()
    architecture: Optional[str] = None
    code_version: Optional[str] = None
    created_at: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", require_identifier(self.model_id, field_name="model_id"))
        object.__setattr__(self, "parent_model_ids", normalize_id_sequence(self.parent_model_ids, field_name="parent_model_ids"))
        object.__setattr__(self, "checkpoint_ids", normalize_id_sequence(self.checkpoint_ids, field_name="checkpoint_ids"))
        object.__setattr__(self, "created_at", normalize_timestamp(self.created_at))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class CheckpointRecord(ProvenanceRecord):
    checkpoint_id: str
    model_id: Optional[str] = None
    parent_checkpoint_id: Optional[str] = None
    training_run_id: Optional[str] = None
    dataset_ids: Tuple[str, ...] = ()
    configuration_id: Optional[str] = None
    code_version: Optional[str] = None
    framework_versions: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: Optional[str] = None
    created_at: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_id", require_identifier(self.checkpoint_id, field_name="checkpoint_id"))
        for field_name in (
            "model_id",
            "parent_checkpoint_id",
            "training_run_id",
            "configuration_id",
            "artifact_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_identifier(value, field_name=field_name))
        object.__setattr__(self, "dataset_ids", normalize_id_sequence(self.dataset_ids, field_name="dataset_ids"))
        object.__setattr__(self, "framework_versions", normalize_metadata(self.framework_versions))
        object.__setattr__(self, "created_at", normalize_timestamp(self.created_at))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class DependencyRecord(ProvenanceRecord):
    record_id: str
    artifact_id: str
    dependency_id: str
    relationship: str = "depends_on"
    version: Optional[str] = None
    digest: Optional[str] = None
    timestamp: str = field(default_factory=get_current_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("record_id", "artifact_id", "dependency_id", "relationship"):
            object.__setattr__(self, field_name, require_identifier(getattr(self, field_name), field_name=field_name))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class TransformationRecord(ProvenanceRecord):
    transformation_id: str
    transformation_type: str
    input_ids: Tuple[str, ...] = ()
    output_ids: Tuple[str, ...] = ()
    agent_id: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=get_current_timestamp)
    parent_transformation_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transformation_id", require_identifier(self.transformation_id, field_name="transformation_id"))
        object.__setattr__(self, "transformation_type", require_identifier(self.transformation_type, field_name="transformation_type"))
        object.__setattr__(self, "input_ids", normalize_id_sequence(self.input_ids, field_name="input_ids"))
        object.__setattr__(self, "output_ids", normalize_id_sequence(self.output_ids, field_name="output_ids"))
        object.__setattr__(self, "parent_transformation_ids", normalize_id_sequence(self.parent_transformation_ids, field_name="parent_transformation_ids"))
        if self.agent_id is not None:
            object.__setattr__(self, "agent_id", require_identifier(self.agent_id, field_name="agent_id"))
        object.__setattr__(self, "parameters", normalize_metadata(self.parameters))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class ReproducibilityReport(ProvenanceRecord):
    artifact_id: str
    source_identity_known: bool
    source_available: bool
    code_version_known: bool
    model_checkpoint_known: bool
    dataset_version_known: bool
    configuration_known: bool
    dependencies_known: bool
    transformations_complete: bool
    environment_known: bool
    reproducible: bool
    completeness: float
    required_requirements: Tuple[str, ...] = ()
    missing_requirements: Tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", require_identifier(self.artifact_id, field_name="artifact_id"))
        if not 0.0 <= float(self.completeness) <= 1.0:
            from .utils.provenance_errors import ProvenanceValidationError

            raise ProvenanceValidationError(
                "reproducibility completeness must be between 0 and 1",
                context={"completeness": self.completeness},
            )
        object.__setattr__(self, "required_requirements", tuple(str(item) for item in self.required_requirements))
        object.__setattr__(self, "missing_requirements", tuple(str(item) for item in self.missing_requirements))
        object.__setattr__(self, "evidence", normalize_metadata(self.evidence))


@dataclass(frozen=True)
class ProvenanceBundle(ProvenanceRecord):
    bundle_id: str
    entities: Tuple[ProvenanceEntity, ...] = ()
    activities: Tuple[ProvenanceActivity, ...] = ()
    agents: Tuple[ProvenanceAgentRef, ...] = ()
    relations: Tuple[ProvenanceRelation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", require_identifier(self.bundle_id, field_name="bundle_id"))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True)
class ProvenanceDocument(ProvenanceRecord):
    document_id: str
    bundles: Tuple[ProvenanceBundle, ...] = ()
    schema_version: str = "slai.provenance.v1"
    created_at: str = field(default_factory=get_current_timestamp)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", require_identifier(self.document_id, field_name="document_id"))
        object.__setattr__(self, "schema_version", require_identifier(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "created_at", normalize_timestamp(self.created_at))


class ProvenanceTypes:
    """Compatibility namespace exposing the canonical provenance record types."""

    Entity = ProvenanceEntity
    Activity = ProvenanceActivity
    AgentRef = ProvenanceAgentRef
    Relation = ProvenanceRelation
    Lineage = LineageRecord
    Source = SourceRecord
    Dataset = DatasetRecord
    Model = ModelRecord
    Checkpoint = CheckpointRecord
    Dependency = DependencyRecord
    Transformation = TransformationRecord
    Custody = CustodyRecord
    Reproducibility = ReproducibilityReport


__all__ = [
    "AssociationRecord",
    "AttributionRecord",
    "CheckpointRecord",
    "CustodyRecord",
    "DatasetRecord",
    "DependencyRecord",
    "DerivationRecord",
    "GenerationRecord",
    "LineageRecord",
    "ModelRecord",
    "ProvenanceActivity",
    "ProvenanceAgentRef",
    "ProvenanceAttribute",
    "ProvenanceBundle",
    "ProvenanceDocument",
    "ProvenanceEntity",
    "ProvenanceRecord",
    "ProvenanceRelation",
    "ProvenanceRelationType",
    "ProvenanceTypes",
    "ReproducibilityReport",
    "SourceRecord",
    "TransformationRecord",
    "UsageRecord",
]


if __name__ == "__main__":
    configure_logging()
    printer.status("PROVENANCE", "provenance domain types loaded", "success")
