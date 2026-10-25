"""Structured, machine-readable fidelity declarations for interchange."""
from dataclasses import asdict, dataclass, field
from typing import Literal
from ..common.helpers import json_safe

Accuracy = Literal["exact", "approximate", "projection", "inferred", "unsupported"]

@dataclass(frozen=True)
class ConversionReport:
    source_digest: str
    target_digest: str | None
    adapter: str
    adapter_version: str
    accuracy_class: Accuracy
    object_mapping: dict[str, str] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    approximations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self):
        return json_safe(asdict(self))
