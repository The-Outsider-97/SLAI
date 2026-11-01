"""
Artifact ancestry for documents, outputs, checkpoints, files, and intermediates.

Buneman et al. (2001), PASS (Muniswamy-Reddy et al., 2006), and Software
Heritage's content-addressed identity motivate explicit artifact identity plus
immutable ancestry/history without introducing quality judgments.
"""

from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Iterable, Mapping
from typing import Any, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..provenance_store import ProvenanceStore
from .base_lineage import BaseLineage
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Artifact Lineage")
printer = PrettyPrinter()


class ArtifactLineage(BaseLineage):
    LINEAGE_TYPE = "artifact"

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        store: Optional[ProvenanceStore] = None,
    ) -> None:
        super().__init__(config=config, store=store)
        self.config = load_global_config()
        self.artifact_lineage_config = get_config_section(
            "artifact_lineage", config=self.config, default={}
        )

    def record_artifact_lineage(
        self,
        artifact_id: str,
        parent_artifact_id: Optional[str | Iterable[str]],
        transformation: Optional[str],
        timestamp: Optional[str] = None,
        *,
        parent_artifact_ids: Optional[Iterable[str]] = None,
        artifact_type: str = "artifact",
        transformation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        source_ids: Optional[Iterable[str]] = None,
        model_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        digest: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Record artifact ancestry while preserving multi-parent derivation."""

        self.ensure_entity(
            artifact_id,
            entity_type=artifact_type,
            digest=digest,
            created_at=timestamp,
            metadata=metadata,
        )
        parents = []
        if parent_artifact_id is not None:
            parents.extend([parent_artifact_id] if isinstance(parent_artifact_id, str) else parent_artifact_id)
        if parent_artifact_ids is not None:
            parents.extend(parent_artifact_ids)
        for parent_id in parents:
            self.ensure_entity(parent_id, entity_type="artifact")

        return self.record_lineage(
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
            lineage_type=self.LINEAGE_TYPE,
        )


__all__ = ["ArtifactLineage"]


if __name__ == "__main__":
    configure_logging()
    printer.status("PROVENANCE", "artifact lineage loaded", "success")
