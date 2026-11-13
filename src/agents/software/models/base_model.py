"""Abstract contract shared by every Software program model.

A *program model* is an immutable, content-addressed representation of a
software artifact (one source file, or a whole repository snapshot).  Models are
facts: they hold what was parsed, never judgements about it.  Content addressing
(``digest``) is what lets the subsystem cache, compare and diff models without
trusting object identity.
"""

from __future__ import annotations

__version__ = "2.4.0"

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..software_types import Serializable
from ..utils.software_helpers import to_plain
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Base Model")


@dataclass(frozen=True, slots=True)
class ModelDescriptor(Serializable):
    """Registry-facing description of a model."""

    model_id: str
    model_type: str
    digest: str
    summary: Mapping[str, Any] = field(default_factory=dict)


class BaseSoftwareModel(ABC):
    """Common identity, equality and serialization semantics."""

    model_type: ClassVar[str] = "model"
    schema_version: ClassVar[str] = "slai.software.model.v1"

    @property
    @abstractmethod
    def digest(self) -> str:
        """Content digest identifying the modelled artifact."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-meaningful stable identifier (path, snapshot digest, ...)."""

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """Compact JSON-safe description (counts, status), never bulk content."""

    def descriptor(self) -> ModelDescriptor:
        return ModelDescriptor(
            model_id=self.model_id,
            model_type=self.model_type,
            digest=self.digest,
            summary=to_plain(self.summary()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "model_type": self.model_type,
            "model_id": self.model_id,
            "digest": self.digest,
            **to_plain(self.summary()),
        }

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BaseSoftwareModel)
            and self.model_type == other.model_type
            and self.digest == other.digest
        )

    def __hash__(self) -> int:
        return hash((self.model_type, self.digest))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.model_id!r} digest={self.digest[:12]}>"


__all__ = ["BaseSoftwareModel", "ModelDescriptor"]
