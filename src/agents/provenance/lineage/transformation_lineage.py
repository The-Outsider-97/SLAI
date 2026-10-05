"""
First-class transformation/activity provenance.

W3C PROV-DM models transformations as Activities.  SLAI therefore records a
stable transformation identity, its inputs/outputs, participating component,
parameters, ancestry, and timestamp instead of relying on free-form strings.
"""
from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Iterable, Mapping, Optional

from ..utils.config_loader import get_config_section
from ..utils.provenance_helpers import *
from ..provenance_types import ProvenanceActivity, ProvenanceAgentRef, TransformationRecord
from .base_lineage import BaseLineage
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Transformation Lineage")
printer = PrettyPrinter()


class TransformationLineage(BaseLineage):
    """Record transformation ancestry and input/output derivation facts."""

    def __init__(self, config: Optional[Any] = None, *, store=None):
        super().__init__(config=config, store=store)
        self.transformation_lineage_config = get_config_section("transformation_lineage", config=self.config, default={})

    def record_transformation_lineage(
        self,
        transformation_id: str,
        parent_transformation_id: Optional[str] = None,
        description: str = "transform",
        timestamp: Optional[str] = None,
        *,
        transformation_type: Optional[str] = None,
        parent_transformation_ids: Optional[Iterable[str]] = None,
        input_ids: Optional[Iterable[str]] = None,
        output_ids: Optional[Iterable[str]] = None,
        agent_id: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        parents = list(normalize_id_sequence(parent_transformation_ids, field_name="parent_transformation_ids"))
        if parent_transformation_id and parent_transformation_id not in parents:
            parents.insert(0, parent_transformation_id)
        inputs = normalize_id_sequence(input_ids, field_name="input_ids")
        outputs = normalize_id_sequence(output_ids, field_name="output_ids")
        when = timestamp or get_current_timestamp()
        kind = transformation_type or description or "transform"

        for entity_id in inputs:
            self.ensure_entity(entity_id)
        for entity_id in outputs:
            self.ensure_entity(entity_id)

        if agent_id:
            self.store.save_agent(ProvenanceAgentRef(agent_id=agent_id, agent_type="slai_component"))

        activity = ProvenanceActivity(
            activity_id=transformation_id,
            activity_type=kind,
            started_at=when,
            ended_at=when,
            agent_id=agent_id,
            parameters=normalize_metadata(parameters),
            metadata=normalize_metadata(metadata),
        )
        self.store.save_activity(activity)

        record = TransformationRecord(
            transformation_id=transformation_id,
            transformation_type=kind,
            input_ids=inputs,
            output_ids=outputs,
            agent_id=agent_id,
            parameters=normalize_metadata(parameters),
            parent_transformation_ids=tuple(parents),
            timestamp=when,
            metadata={**normalize_metadata(metadata), "description": description},
        )
        self.store.save_transformation_record(record)

        derived = []
        if outputs:
            for output_id in outputs:
                parent_ids = tuple(item for item in inputs if item != output_id)
                if parent_ids:
                    derived.append(
                        self.record_lineage(
                            output_id,
                            parent_ids,
                            transformation_id,
                            timestamp=when,
                            lineage_type="transformation",
                            transformation_id=transformation_id,
                            agent_id=agent_id,
                            metadata=metadata,
                        )
                    )

        return {"transformation": record.to_dict(), "lineage": derived}


__all__ = ["TransformationLineage"]

if __name__ == "__main__":
    configure_logging()
    printer.status("SMOKE", "TransformationLineage module loaded", "success")
