"""
Provenance Custody manages the ownership and control of artifacts.

sources:
- Turner (2006), Selective and intelligent imaging using digital evidence bags.
- Torres-Arias et al. (2019), in-toto: Providing farm-to-table guarantees for bits and bytes.

It borrows the chain representation principles, but not turn SLAI provenance into a forensic or security subsystem.
"""

from __future__ import annotations

from typing import Mapping, Optional
from datetime import datetime

from .utils.config_loader import load_global_config, get_config_section
from .utils.provenance_errors import *
from .utils.provenance_helpers import *
from .provenance_store import ProvenanceStore
from .provenance_memory import ProvenanceMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Custody")
printer = PrettyPrinter()

class ProvenanceCustody:
    def __init__(self):
        self.config = load_global_config()
        raw_section = self.config.get("provenance_custody")
        if not isinstance(raw_section, Mapping):
            raise ProvenanceConfigurationError("provenance_custody configuration must be a mapping")
        self.custody_config = get_config_section("provenance_custody", config=self.config) or {}

        self.custody_records = {}
        self.provenance_store = ProvenanceStore()
        self.provenance_memory = ProvenanceMemory()

    def record_custody(self, artifact_id: str, owner: str, *, timestamp: Optional[str] = None):
        """
        Record the custody of an artifact.

        Args:
            artifact_id (str): The unique identifier of the artifact.
            owner (str): The owner of the artifact.
            timestamp (Optional[str]): The timestamp of the custody record. If not provided, the current time will be used.

        Returns:
            dict: A dictionary representing the custody record.
        """
        if not timestamp:
            timestamp = get_current_timestamp(self)

        custody_record = {
            "artifact_id": artifact_id,
            "owner": owner,
            "timestamp": timestamp
        }

        self.custody_records[artifact_id] = custody_record
        self.provenance_store.save_custody_record(custody_record)
        return custody_record


__all__ = ["ProvenanceCustody"]

if __name__ == "__main__":
    pass