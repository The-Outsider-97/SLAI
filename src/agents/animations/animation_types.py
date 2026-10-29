"""Validated animation request contracts (application-neutral)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4

@dataclass(frozen=True, slots=True)
class AnimationRequest:
    operation: str
    scene_id: str | None = None
    expected_revision: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a nonempty string")
        if self.expected_revision is not None and (type(self.expected_revision) is not int or self.expected_revision < 1):
            raise ValueError("expected_revision must be a positive integer")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be nonempty")
