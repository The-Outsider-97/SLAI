"""Artifact-local claims and an opt-in bridge to the real Provenance subsystem.

The format owns claim representation. ProvenanceStore and ProvenanceLineage own
persistence and ancestry. Importing this file never initializes an agent.
Neither an observation nor a content digest authenticates its stated author.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from ..common.metadata import require_claim, require_provenance
from ..common.helpers import digest, canonical_json
from ..common.ids import require_id
from ..common.errors import ValidationError
from ..sgeo.scene import Scene
from ..sdrw.drawing import Drawing

def make_observation(subject,predicate,*,evidence,confidence=1.0,agent=""):
    require_id(subject)
    if not isinstance(predicate,str) or not predicate or len(predicate)>255:
        raise ValidationError("invalid spatial claim predicate")
    if not isinstance(agent,str) or len(agent)>255:
        raise ValidationError("invalid agent label")
    return require_claim({"subject":subject,"predicate":predicate,"evidence":evidence,
                          "confidence":confidence,"agent":agent,
                          "timestamp":datetime.now(timezone.utc).isoformat()})

def claim_digest(claim):
    return digest(canonical_json(require_claim(claim)))


class ProvenanceStoreService(Protocol):
    """Persistence contract supplied by SLAI's ProvenanceStore."""

    def save_entity(self, record: Mapping[str, Any]) -> dict[str, Any]: ...

    def get_entity(self, entity_id: str, *, strict: bool = True) -> dict[str, Any] | None: ...


class ProvenanceLineageService(Protocol):
    """Lineage contract supplied by SLAI's ProvenanceLineage."""

    def record_lineage(self, artifact_id: str, parent_artifact_id: str | None = None,
                       transformation: str = "derived", timestamp: str | None = None,
                       *, parent_artifact_ids: Any = None, transformation_id: str | None = None,
                       agent_id: str | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]: ...

    def get_lineage(self, artifact_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class ProvenanceArtifactAdapter:
    """Register spatial artifacts and observations with the real PROV store.

    Saved identities are immutable and content-addressed. Existing conflicting
    records never get overwritten. Writes occur only after explicit calls.
    """

    store: ProvenanceStoreService
    lineage: ProvenanceLineageService

    def record_document(self, document: Scene | Drawing, *, sha256: str,
                        artifact_id: str | None = None) -> dict[str, Any]:
        """Persist a *caller-hashed package* identity, not a guessed model hash.

        ``sha256`` must have been calculated over the exact published/loaded
        bytes by the caller. This adapter cannot verify a package it was not
        given; the provenance entry records the caller's assertion as such.
        """
        document.validate()
        require_provenance({"digest": sha256})
        format_suffix = "sgeo" if isinstance(document, Scene) else "sdrw"
        stable_id = artifact_id or f"{format_suffix}:{document.document_id}:r{document.revision}:{sha256}"
        existing = self.store.get_entity(stable_id, strict=False)
        if existing is not None:
            if (existing.get("digest") != sha256
                    or existing.get("metadata", {}).get("document_id") != document.document_id
                    or existing.get("metadata", {}).get("revision") != document.revision
                    or existing.get("metadata", {}).get("format_id") != f"slai.{format_suffix}"):
                raise ValidationError("spatial artifact id refers to a different revision or digest")
            return existing
        return self.store.save_entity({
            "entity_id": stable_id,
            "entity_type": "spatial_artifact",
            "label": document.name,
            "digest": sha256,
            "metadata": {
                "format_id": f"slai.{format_suffix}",
                "document_id": document.document_id,
                "revision": document.revision,
                "digest_scope": "external_package_bytes",
                "digest_verified_by_adapter": False,
            },
        })

    def record_scene(self, scene: Scene, *, sha256: str,
                     artifact_id: str | None = None) -> dict[str, Any]:
        """Backward-compatible SGEO entry point for :meth:`record_document`."""
        return self.record_document(scene, sha256=sha256, artifact_id=artifact_id)

    def record_observation(self, claim: Mapping[str, Any]) -> dict[str, Any]:
        """Persist a validated claim as a *claim entity*, not a verified fact."""
        normalized = require_claim(dict(claim))
        fingerprint = claim_digest(normalized)
        claim_id = f"spatial_observation:{fingerprint}"
        existing = self.store.get_entity(claim_id, strict=False)
        if existing is not None:
            if (existing.get("digest") != fingerprint
                    or existing.get("metadata", {}).get("claim") != normalized):
                raise ValidationError("spatial observation id conflicts with stored content")
            return existing
        payload: dict[str, Any] = {
            "entity_id": claim_id,
            "entity_type": "spatial_observation",
            "digest": fingerprint,
            "metadata": {"claim": normalized, "independently_verified": False},
        }
        if "timestamp" in normalized:
            payload["created_at"] = normalized["timestamp"]
        return self.store.save_entity(payload)

    def record_derivation(self, artifact_id: str, parents: tuple[str, ...], *,
                          transformation: str, transformation_id: str | None = None,
                          agent_id: str | None = None, timestamp: str | None = None,
                          metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Delegate DAG validation, cycle detection and storage to Provenance."""
        return self.lineage.record_lineage(
            artifact_id,
            transformation=transformation,
            timestamp=timestamp,
            parent_artifact_ids=parents,
            transformation_id=transformation_id,
            agent_id=agent_id,
            metadata=metadata,
        )

    def lineage_for(self, artifact_id: str) -> list[dict[str, Any]]:
        return self.lineage.get_lineage(artifact_id)
