"""
Dataset derivation lineage for SLAI training and corpus pipelines.

The module records creation history, source relationships, transformations, and
ancestry informed by Datasheets for Datasets, Longpre et al. (2024), and the
Buneman provenance model.  It intentionally does not assess dataset quality or
fitness for use.
"""

from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Iterable, Mapping
from typing import Any, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.provenance_helpers import normalize_id_sequence
from ..provenance_store import ProvenanceStore
from ..provenance_types import DatasetRecord
from .base_lineage import BaseLineage
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Dataset Lineage")
printer = PrettyPrinter()


class DatasetLineage(BaseLineage):
    LINEAGE_TYPE = "dataset"

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        store: Optional[ProvenanceStore] = None,
    ) -> None:
        super().__init__(config=config, store=store)
        self.config = load_global_config()
        self.dataset_lineage_config = get_config_section(
            "dataset_lineage", config=self.config, default={}
        )

    def record_dataset_lineage(
        self,
        dataset_id: str,
        parent_dataset_id: Optional[str | Iterable[str]],
        transformation: Optional[str],
        timestamp: Optional[str] = None,
        *,
        parent_dataset_ids: Optional[Iterable[str]] = None,
        version: Optional[str] = None,
        transformation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        source_ids: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Record raw-source -> derived-dataset ancestry without quality scoring."""

        parents = []
        if parent_dataset_id is not None:
            parents.extend([parent_dataset_id] if isinstance(parent_dataset_id, str) else parent_dataset_id)
        if parent_dataset_ids is not None:
            parents.extend(parent_dataset_ids)
        normalized_parents = normalize_id_sequence(parents, field_name="parent_dataset_ids")
        normalized_sources = normalize_id_sequence(source_ids, field_name="source_ids")
        dataset_metadata = dict(metadata or {})
        if version is not None:
            dataset_metadata.setdefault("dataset_version", version)

        self.ensure_entity(
            dataset_id,
            entity_type="dataset",
            created_at=timestamp,
            metadata=dataset_metadata,
        )
        for parent_id in normalized_parents:
            self.ensure_entity(parent_id, entity_type="dataset")

        # DatasetRecord captures dataset-specific metadata; the entity remains the
        # canonical graph node.  The record is embedded as lineage metadata so no
        # parallel dataset database is introduced.
        dataset_record = DatasetRecord(
            dataset_id=dataset_id,
            version=version,
            parent_dataset_ids=normalized_parents,
            source_ids=normalized_sources,
            created_at=timestamp or self.ensure_entity(dataset_id)["created_at"],
            metadata=dataset_metadata,
        )
        lineage_metadata = dict(dataset_metadata)
        lineage_metadata["dataset_record"] = dataset_record.to_dict()

        return self.record_lineage(
            dataset_id,
            normalized_parents,
            transformation,
            timestamp,
            transformation_id=transformation_id,
            agent_id=agent_id,
            source_ids=normalized_sources,
            metadata=lineage_metadata,
            lineage_type=self.LINEAGE_TYPE,
        )


__all__ = ["DatasetLineage"]


if __name__ == "__main__":
    configure_logging()
    printer.status("PROVENANCE", "dataset lineage loaded", "success")
