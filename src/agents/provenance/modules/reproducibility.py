"""Provenance completeness checks for computational reproducibility.

This module follows Sandve et al., ReproZip, and reproducible-build principles:
it asks whether enough identity, configuration, dependency, environment, and
transformation provenance exists to reconstruct an artifact.  The completeness
ratio is not a quality/trust/performance score.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.provenance_errors import ProvenanceNotFoundError, ProvenanceReproducibilityError
from ..utils.provenance_helpers import require_identifier
from ..provenance_store import ProvenanceStore
from ..provenance_types import ReproducibilityReport
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Reproducibility")
printer = PrettyPrinter()

_REQUIREMENTS = (
    "source_identity_known",
    "source_available",
    "code_version_known",
    "model_checkpoint_known",
    "dataset_version_known",
    "configuration_known",
    "dependencies_known",
    "transformations_complete",
    "environment_known",
)
_DEFAULT_REQUIRED = (
    "source_identity_known",
    "configuration_known",
    "dependencies_known",
    "transformations_complete",
    "environment_known",
)


class Reproducibility:
    """Assess whether recorded provenance is sufficient for reconstruction."""

    def __init__(self, store: Optional[ProvenanceStore] = None) -> None:
        self.config = load_global_config()
        self.reproducibility_config = get_config_section(
            "reproducibility", config=self.config, default={}
        )
        self.store = store or ProvenanceStore()
        configured = self.reproducibility_config.get("required_requirements", _DEFAULT_REQUIRED)
        if not isinstance(configured, (list, tuple, set, frozenset)):
            configured = _DEFAULT_REQUIRED
        self.required_requirements = tuple(str(item) for item in configured if str(item) in _REQUIREMENTS)
        if not self.required_requirements:
            self.required_requirements = _DEFAULT_REQUIRED

    def _provenance(self, artifact_id: str) -> dict[str, Any]:
        artifact = require_identifier(artifact_id, field_name="artifact_id")
        try:
            return self.store.get_provenance(artifact)
        except ProvenanceNotFoundError:
            raise
        except Exception as exc:
            raise ProvenanceReproducibilityError(
                "failed to collect provenance for reproducibility assessment",
                context={"artifact_id": artifact},
                cause=exc,
            ) from exc

    @staticmethod
    def _metadata(provenance: Mapping[str, Any]) -> dict[str, Any]:
        entity = provenance.get("entity")
        if isinstance(entity, Mapping) and isinstance(entity.get("metadata"), Mapping):
            return dict(entity["metadata"])
        return {}

    def source_identity_known(self, artifact_id: str) -> bool:
        p = self._provenance(artifact_id)
        if p.get("sources"):
            return True
        entity = p.get("entity") or {}
        if isinstance(entity, Mapping) and entity.get("source_id"):
            return self.store.get_source(str(entity["source_id"]), strict=False) is not None
        return False

    def source_available(self, artifact_id: str) -> bool:
        p = self._provenance(artifact_id)
        sources = list(p.get("sources") or [])
        entity = p.get("entity") or {}
        if not sources and isinstance(entity, Mapping) and entity.get("source_id"):
            record = self.store.get_source(str(entity["source_id"]), strict=False)
            if record:
                sources.append(record)
        if not sources:
            return False
        for source in sources:
            metadata = source.get("metadata") or {}
            if isinstance(metadata, Mapping) and metadata.get("available") is True:
                continue
            locator = source.get("locator")
            if not locator:
                return False
            parsed = urlparse(str(locator))
            if parsed.scheme in {"http", "https", "ftp", "s3", "gs"}:
                # Provenance does not perform network availability probing.  A remote
                # source is available only if availability was recorded explicitly.
                return False
            path = Path(parsed.path if parsed.scheme == "file" else str(locator)).expanduser()
            if not path.exists():
                return False
        return True

    def code_version_known(self, artifact_id: str) -> bool:
        p = self._provenance(artifact_id)
        metadata = self._metadata(p)
        if any(metadata.get(key) for key in ("code_version", "code_commit", "source_revision")):
            return True
        return any(checkpoint.get("code_version") for checkpoint in p.get("checkpoints") or [])

    def model_checkpoint_known(self, artifact_id: str) -> bool:
        p = self._provenance(artifact_id)
        if p.get("checkpoints"):
            return True
        return any(record.get("checkpoint_id") for record in p.get("lineage") or [])

    def dataset_version_known(self, artifact_id: str) -> bool:
        p = self._provenance(artifact_id)
        metadata = self._metadata(p)
        if any(metadata.get(key) for key in ("dataset_version", "data_version")):
            return True
        entity = p.get("entity") or {}
        if isinstance(entity, Mapping) and entity.get("entity_type") == "dataset" and metadata.get("version"):
            return True
        for checkpoint in p.get("checkpoints") or []:
            for dataset_id in checkpoint.get("dataset_ids") or []:
                dataset = self.store.get_entity(str(dataset_id), strict=False)
                if dataset and isinstance(dataset.get("metadata"), Mapping):
                    if dataset["metadata"].get("version") or dataset["metadata"].get("dataset_version"):
                        return True
        return False

    def configuration_known(self, artifact_id: str) -> bool:
        p = self._provenance(artifact_id)
        metadata = self._metadata(p)
        if any(metadata.get(key) is not None for key in ("configuration_id", "configuration_digest", "configuration")):
            return True
        if any(checkpoint.get("configuration_id") for checkpoint in p.get("checkpoints") or []):
            return True
        # A transformation with explicitly recorded parameters is sufficient for
        # that transformation's local configuration, but not for arbitrary roots.
        transformations = p.get("transformations") or []
        return bool(transformations) and all(bool(item.get("parameters")) for item in transformations)

    def dependencies_known(self, artifact_id: str) -> bool:
        p = self._provenance(artifact_id)
        metadata = self._metadata(p)
        if metadata.get("dependencies_required") is False or metadata.get("dependencies_complete") is True:
            return True
        return bool(p.get("dependencies"))

    def transformations_complete(self, artifact_id: str) -> bool:
        p = self._provenance(artifact_id)
        lineage = p.get("lineage") or []
        if not lineage:
            metadata = self._metadata(p)
            # Source/root entities are not required to have a generating transform.
            return bool(metadata.get("provenance_root") or metadata.get("root") or p.get("sources"))
        for record in lineage:
            transformation_id = record.get("transformation_id")
            descriptive = record.get("transformation")
            if not transformation_id and not descriptive:
                return False
            if transformation_id and self.store.get_transformation(str(transformation_id), strict=False) is None:
                # Legacy/descriptive lineage may reference an activity instead of a
                # TransformationRecord; check the activity table before declaring it missing.
                activities = self.store.snapshot().get("activities", {})
                if str(transformation_id) not in activities:
                    return False
        return True

    def environment_known(self, artifact_id: str) -> bool:
        p = self._provenance(artifact_id)
        metadata = self._metadata(p)
        if any(metadata.get(key) for key in ("environment", "environment_id", "environment_digest", "runtime_environment")):
            return True
        return any(bool(checkpoint.get("framework_versions")) for checkpoint in p.get("checkpoints") or [])

    def _assessment(self, artifact_id: str) -> dict[str, bool]:
        return {
            "source_identity_known": self.source_identity_known(artifact_id),
            "source_available": self.source_available(artifact_id),
            "code_version_known": self.code_version_known(artifact_id),
            "model_checkpoint_known": self.model_checkpoint_known(artifact_id),
            "dataset_version_known": self.dataset_version_known(artifact_id),
            "configuration_known": self.configuration_known(artifact_id),
            "dependencies_known": self.dependencies_known(artifact_id),
            "transformations_complete": self.transformations_complete(artifact_id),
            "environment_known": self.environment_known(artifact_id),
        }

    def reproducibility_report(self, artifact_id: str) -> dict[str, Any]:
        artifact = require_identifier(artifact_id, field_name="artifact_id")
        checks = self._assessment(artifact)
        missing = tuple(name for name in self.required_requirements if not checks[name])
        satisfied = len(self.required_requirements) - len(missing)
        completeness = satisfied / len(self.required_requirements) if self.required_requirements else 1.0
        report = ReproducibilityReport(
            artifact_id=artifact,
            **checks,
            reproducible=not missing,
            completeness=completeness,
            required_requirements=self.required_requirements,
            missing_requirements=missing,
            evidence={
                "lineage_records": len(self.store.get_lineage_records(artifact)),
                "dependencies": len(self.store.get_dependencies(artifact)),
                "checkpoints": len(self._provenance(artifact).get("checkpoints") or []),
                "sources": [item.get("source_id") for item in self._provenance(artifact).get("sources") or []],
            },
        )
        return report.to_dict()

    def check_reproducibility(self, artifact_id: str) -> bool:
        return bool(self.reproducibility_report(artifact_id)["reproducible"])

    def reproducibility_score(self, artifact_id: str) -> float:
        """Backward-compatible provenance-completeness ratio, not a quality score."""
        return float(self.reproducibility_report(artifact_id)["completeness"])


__all__ = ["Reproducibility"]

if __name__ == "__main__":
    configure_logging()
    printer.status("SMOKE", "Reproducibility module loaded", "success")
