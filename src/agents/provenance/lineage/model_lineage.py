"""
Model and checkpoint lineage for SLAI.

Academically informed by ModelDB and PROV-ML: a model/checkpoint is treated as
an immutable provenance entity whose ancestry points to parent models,
training datasets, configuration/code identities, and the training activity
that produced it.  Performance evaluation remains outside this module.
"""
from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Iterable, Mapping, Optional

from ..utils.config_loader import get_config_section
from ..utils.provenance_helpers import *
from ..provenance_types import CheckpointRecord, ModelRecord
from .base_lineage import BaseLineage
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Model Lineage")
printer = PrettyPrinter()


class ModelLineage(BaseLineage):
    """Record model/version ancestry and optional checkpoint provenance."""

    def __init__(self, config: Optional[Any] = None, *, store=None):
        super().__init__(config=config, store=store)
        self.model_lineage_config = get_config_section("model_lineage", config=self.config, default={})

    def record_model_lineage(
        self,
        model_id: str,
        parent_model_id: Optional[str] = None,
        transformation: str = "train",
        timestamp: Optional[str] = None,
        *,
        parent_model_ids: Optional[Iterable[str]] = None,
        model_version: Optional[str] = None,
        architecture: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        parent_checkpoint_id: Optional[str] = None,
        training_run_id: Optional[str] = None,
        training_dataset_ids: Optional[Iterable[str]] = None,
        configuration_id: Optional[str] = None,
        code_version: Optional[str] = None,
        framework_versions: Optional[Mapping[str, str]] = None,
        artifact_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        parents = list(normalize_id_sequence(parent_model_ids, field_name="parent_model_ids"))
        if parent_model_id and parent_model_id not in parents:
            parents.insert(0, parent_model_id)
        datasets = normalize_id_sequence(training_dataset_ids, field_name="training_dataset_ids")
        when = timestamp or get_current_timestamp()

        self.ensure_entity(
            model_id,
            entity_type="model",
            metadata={"version": model_version, "architecture": architecture},
        )
        for parent in parents:
            self.ensure_entity(parent, entity_type="model")
        for dataset_id in datasets:
            self.ensure_entity(dataset_id, entity_type="dataset")

        model_metadata = dict(normalize_metadata(metadata))
        model_metadata.update({
            key: value
            for key, value in {
                "training_run_id": training_run_id,
                "training_dataset_ids": list(datasets),
                "configuration_id": configuration_id,
            }.items()
            if value not in (None, [], ())
        })
        model_record = ModelRecord(
            model_id=model_id,
            parent_model_ids=tuple(parents),
            checkpoint_ids=(checkpoint_id,) if checkpoint_id else (),
            version=model_version,
            architecture=architecture,
            code_version=code_version,
            created_at=when,
            metadata=model_metadata,
        )

        lineage_metadata = dict(normalize_metadata(metadata))
        lineage_metadata["model"] = model_record.to_dict()
        if framework_versions:
            lineage_metadata["framework_versions"] = dict(framework_versions)

        lineage = self.record_lineage(
            model_id,
            parents,
            transformation,
            timestamp=when,
            lineage_type="model",
            agent_id=agent_id,
            checkpoint_id=checkpoint_id,
            metadata=lineage_metadata,
        )

        checkpoint = None
        if checkpoint_id:
            checkpoint = CheckpointRecord(
                checkpoint_id=checkpoint_id,
                model_id=model_id,
                parent_checkpoint_id=parent_checkpoint_id,
                training_run_id=training_run_id,
                dataset_ids=datasets,
                configuration_id=configuration_id,
                code_version=code_version,
                framework_versions=dict(framework_versions or {}),
                artifact_id=artifact_id or model_id,
                created_at=when,
                metadata=normalize_metadata(metadata),
            )
            self.store.save_checkpoint_record(checkpoint)

        return {
            "lineage": lineage,
            "model": model_record.to_dict(),
            "checkpoint": checkpoint.to_dict() if checkpoint else None,
        }


__all__ = ["ModelLineage"]

if __name__ == "__main__":
    configure_logging()
    printer.status("SMOKE", "ModelLineage module loaded", "success")
