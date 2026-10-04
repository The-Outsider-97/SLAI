from __future__ import annotations

from datetime import datetime
from typing import Mapping, Optional

from .utils.config_loader import load_global_config, get_config_section
from .utils.provenance_errors import *
from .utils.provenance_helpers import *
from .lineage import *
from .provenance_types import *
from .provenance_store import ProvenanceStore
from .provenance_memory import ProvenanceMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Lineage")
printer = PrettyPrinter()

class ProvenanceLineage:
    def __init__(self):
        self.config = load_global_config()
        raw_section = self.config.get("provenance_lineage")
        if not isinstance(raw_section, Mapping):
            raise ProvenanceConfigurationError("provenance_lineage configuration must be a mapping")
        self.lineage_config = get_config_section("provenance_lineage", config=self.config) or {}

        self.lineage_records = {}
        self.provenance_store = ProvenanceStore()
        self.provenance_memory = ProvenanceMemory()

        self.artifact = ArtifactLineage()
        self.dataset = DatasetLineage()
        self.dependency = DependencyLineage()
        self.model = ModelLineage()
        self.transformation = TransformationLineage()

    def record_lineage(self, artifact_id: str, parent_artifact_id: str, transformation: str, timestamp: Optional[str] = None):
        """
        Record the lineage of an artifact.

        Args:
            artifact_id (str): The unique identifier of the artifact.
            parent_artifact_id (str): The unique identifier of the parent artifact.
            transformation (str): Description of the transformation applied to the parent artifact.
            timestamp (Optional[str]): The timestamp of the lineage record. If not provided, the current time will be used.

        Returns:
            dict: A dictionary representing the lineage record.
        """
        if not timestamp:
            timestamp = get_current_timestamp(self)

        lineage_record = {
            "artifact_id": artifact_id,
            "parent_artifact_id": parent_artifact_id,
            "transformation": transformation,
            "timestamp": timestamp
        }

        self.lineage_records[artifact_id] = lineage_record
        self.provenance_store.save_lineage_record(lineage_record)
        return lineage_record

    def get_lineage(self, artifact_id: str):
        """
        Retrieve the lineage record for a given artifact.

        Args:
            artifact_id (str): The unique identifier of the artifact.
            
        """
        return self.lineage_records.get(artifact_id, None)


__all__ = ["ProvenanceLineage"]

if __name__ == "__main__":
    pass