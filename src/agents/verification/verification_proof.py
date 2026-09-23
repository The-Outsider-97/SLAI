"""Proof/certificate evidence contracts for formal verification.

This module represents evidence that was actually produced by a solver,
checker, or caller.  It never synthesizes proof text and it never upgrades a
Verification status merely because an artifact is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from src.tuning.utils.tuning_helpers import stable_fingerprint # type: ignore
from .utils.verification_errors import MalformedSpecificationError


@dataclass(frozen=True, slots=True)
class ProofArtifact:
    """Metadata for proof/certificate material actually available.

    ``reference`` identifies an externally stored artifact when applicable and
    ``digest`` preserves a backend- or caller-supplied integrity digest.  The
    subsystem does not claim that a certificate is independently checked merely
    because this record exists.
    """

    artifact_type: str
    backend: str | None = None
    format: str | None = None
    reference: str | None = None
    digest: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        artifact_type = str(self.artifact_type).strip()
        if not artifact_type:
            raise MalformedSpecificationError("proof artifact_type must be non-empty")
        object.__setattr__(self, "artifact_type", artifact_type)

        for name in ("backend", "format", "reference", "digest"):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = str(raw).strip()
            if not value:
                raise MalformedSpecificationError(
                    f"proof {name} must be non-empty when provided"
                )
            object.__setattr__(self, name, value)

        normalized: dict[str, str] = {}
        for raw_key, raw_value in self.metadata.items():
            key = str(raw_key).strip()
            value = str(raw_value).strip()
            if not key or not value:
                raise MalformedSpecificationError(
                    "proof metadata keys and values must be non-empty strings"
                )
            normalized[key] = value
        object.__setattr__(self, "metadata", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-friendly proof metadata."""

        return {
            "artifact_type": self.artifact_type,
            "backend": self.backend,
            "format": self.format,
            "reference": self.reference,
            "digest": self.digest,
            "metadata": dict(sorted(self.metadata.items())),
        }

    @property
    def fingerprint(self) -> str:
        """Fingerprint artifact metadata without pretending to verify contents."""

        return stable_fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProofArtifactSet:
    """Immutable ordered collection of distinct proof/certificate artifacts."""

    artifacts: tuple[ProofArtifact, ...]

    def __post_init__(self) -> None:
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise MalformedSpecificationError("proof artifact set must not be empty")
        if any(not isinstance(item, ProofArtifact) for item in artifacts):
            raise MalformedSpecificationError(
                "proof artifact set may contain only ProofArtifact objects"
            )
        fingerprints = [item.fingerprint for item in artifacts]
        if len(fingerprints) != len(set(fingerprints)):
            raise MalformedSpecificationError("proof artifact set contains duplicates")
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, object]:
        return {"artifacts": [item.to_dict() for item in self.artifacts]}


__all__ = ["ProofArtifact", "ProofArtifactSet"]
