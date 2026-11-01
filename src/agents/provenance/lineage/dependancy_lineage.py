"""
Dependency provenance for artifacts, models, checkpoints, and training runs.

ReproZip, in-toto, and reproducible-build literature motivate explicit
artifact -> dependency identity/version relationships.  This module records
those relationships only; it does not install, resolve, or vulnerability-scan
dependencies.

The historical filename ``dependancy_lineage.py`` is retained for SLAI import
compatibility.  The class name remains correctly spelled ``DependencyLineage``.
"""

from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Mapping
from typing import Any, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.provenance_helpers import *
from ..provenance_store import ProvenanceStore
from ..provenance_types import DependencyRecord, ProvenanceRelation, ProvenanceRelationType
from .base_lineage import BaseLineage
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Dependency Lineage")
printer = PrettyPrinter()


class DependencyLineage(BaseLineage):
    LINEAGE_TYPE = "dependency"

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        store: Optional[ProvenanceStore] = None,
    ) -> None:
        super().__init__(config=config, store=store)
        self.config = load_global_config()
        self.dependency_lineage_config = get_config_section(
            "dependency_lineage", config=self.config, default={}
        )

    def record_dependency_lineage(
        self,
        artifact_id: str,
        dependency_id: str,
        relationship: str,
        timestamp: Optional[str] = None,
        *,
        version: Optional[str] = None,
        digest: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Record one immutable dependency relationship."""

        event_timestamp = timestamp or get_current_timestamp()
        record = DependencyRecord(
            record_id=stable_provenance_id(
                "dependency",
                artifact_id,
                dependency_id,
                relationship,
                version,
                digest,
                event_timestamp,
            ),
            artifact_id=artifact_id,
            dependency_id=dependency_id,
            relationship=relationship,
            version=version,
            digest=digest,
            timestamp=event_timestamp,
            metadata=metadata or {},
        )
        self.ensure_entity(artifact_id, entity_type="artifact")
        self.ensure_entity(
            dependency_id,
            entity_type="dependency",
            digest=digest,
            metadata={"version": version} if version is not None else {},
        )
        persisted = self.store.save_dependency_record(record)
        relation = ProvenanceRelation(
            relation_id=stable_provenance_id(
                "relation",
                ProvenanceRelationType.DEPENDS_ON.value,
                artifact_id,
                dependency_id,
                record.record_id,
            ),
            relation_type=ProvenanceRelationType.DEPENDS_ON,
            subject_id=artifact_id,
            object_id=dependency_id,
            timestamp=event_timestamp,
            metadata={"dependency_record_id": record.record_id, "relationship": relationship},
        )
        self.store.save_relation(relation)
        return persisted

    def dependencies_for(self, artifact_id: str) -> list[dict[str, Any]]:
        return self.store.get_dependencies(artifact_id)


__all__ = ["DependencyLineage"]


if __name__ == "__main__":
    configure_logging()
    printer.status("PROVENANCE", "dependency lineage loaded", "success")
