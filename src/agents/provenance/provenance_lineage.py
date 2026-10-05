"""
Facade over specialized provenance lineage recorders.

Lineage is represented as a DAG rather than a single-parent tree, reflecting
Open Provenance Model/W3C PROV causal graphs and compositional multi-input
provenance.  This module coordinates lineage recorders only; it does not score
or evaluate their evidence.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Iterable, Mapping
from typing import Any, Optional

from .lineage import ArtifactLineage, DatasetLineage, DependencyLineage, ModelLineage, TransformationLineage
from .modules.lineage_graph import LineageGraph
from .utils.config_loader import get_config_section, load_global_config
from .provenance_store import ProvenanceStore
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Lineage")
printer = PrettyPrinter()


class ProvenanceLineage:
    """Coordinate all lineage specializations through one shared store."""

    def __init__(self, store: Optional[ProvenanceStore] = None) -> None:
        self.config = load_global_config()
        self.lineage_config = get_config_section("provenance_lineage", config=self.config, default={})
        self.provenance_store = store or ProvenanceStore()
        self.artifact = ArtifactLineage(store=self.provenance_store)
        self.dataset = DatasetLineage(store=self.provenance_store)
        self.dependency = DependencyLineage(store=self.provenance_store)
        self.model = ModelLineage(store=self.provenance_store)
        self.transformation = TransformationLineage(store=self.provenance_store)
        self.graph = LineageGraph(store=self.provenance_store)

    def record_lineage(
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
        return self.artifact.record_artifact_lineage(
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

    def get_lineage(self, artifact_id: str) -> list[dict[str, Any]]:
        return self.provenance_store.get_lineage_records(artifact_id)

    def get_lineage_graph(self, artifact_id: str) -> dict[str, Any]:
        return self.graph.get_lineage_graph(artifact_id)


__all__ = ["ProvenanceLineage"]

if __name__ == "__main__":
    configure_logging()
    printer.status("SMOKE", "ProvenanceLineage module loaded", "success")
