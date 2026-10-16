"""
Immutable chain-of-custody events for SLAI artifacts.

The design borrows continuity/event-chain principles from digital evidence bags
and in-toto while remaining provenance infrastructure, not a forensic or
security-scanning subsystem.  Custody is append-only: ownership transitions are
recorded as events and never overwrite prior custodians.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Mapping
from typing import Any, Optional

from .utils.config_loader import get_config_section, load_global_config
from .utils.provenance_errors import ProvenanceCustodyError
from .utils.provenance_helpers import *
from .provenance_store import ProvenanceStore
from .provenance_types import CustodyRecord
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Custody")
printer = PrettyPrinter()


class ProvenanceCustody:
    """Maintain append-only custody transitions for artifacts."""

    def __init__(self, store: Optional[ProvenanceStore] = None) -> None:
        self.config = load_global_config()
        self.custody_config = get_config_section("provenance_custody", config=self.config, default={})
        self.provenance_store = store or ProvenanceStore()

    def current_custodian(self, artifact_id: str) -> Optional[str]:
        history = self.provenance_store.get_custody_history(artifact_id)
        return str(history[-1]["new_custodian"]) if history else None

    def transfer_custody(
        self,
        artifact_id: str,
        new_custodian: str,
        *,
        previous_custodian: Optional[str] = None,
        activity: Optional[str] = None,
        timestamp: Optional[str] = None,
        artifact_digest: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        artifact = require_identifier(artifact_id, field_name="artifact_id")
        new_owner = require_identifier(new_custodian, field_name="new_custodian")
        current = self.current_custodian(artifact)
        if previous_custodian is not None:
            expected = require_identifier(previous_custodian, field_name="previous_custodian")
            if current is not None and current != expected:
                raise ProvenanceCustodyError(
                    "custody transition does not continue from the recorded custodian",
                    context={
                        "artifact_id": artifact,
                        "recorded_custodian": current,
                        "supplied_previous_custodian": expected,
                    },
                )
            previous = expected
        else:
            previous = current

        when = timestamp or get_current_timestamp()
        event_id = stable_provenance_id("custody", artifact, previous, new_owner, activity, when, artifact_digest)
        record = CustodyRecord(
            event_id=event_id,
            artifact_id=artifact,
            previous_custodian=previous,
            new_custodian=new_owner,
            activity=activity,
            timestamp=when,
            artifact_digest=artifact_digest,
            context=normalize_metadata(context),
        )
        return self.provenance_store.save_custody_record(record)

    def record_custody(self, artifact_id: str, owner: str, *, timestamp: Optional[str] = None) -> dict[str, Any]:
        """Backward-compatible alias that appends a transition to ``owner``."""
        return self.transfer_custody(artifact_id, owner, timestamp=timestamp)

    def chain_of_custody(self, artifact_id: str) -> list[dict[str, Any]]:
        return self.provenance_store.get_custody_history(artifact_id)

    history = chain_of_custody


__all__ = ["ProvenanceCustody"]

if __name__ == "__main__":
    configure_logging()
    printer.status("SMOKE", "ProvenanceCustody module loaded", "success")
