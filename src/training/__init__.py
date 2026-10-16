"""SLAI offline training/curriculum integration package.

Public symbols are resolved lazily so importing a lightweight training utility
(e.g. ``src.training.lantra_corpus``) does not initialize agent runtimes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "LantraCurriculumBuilder": (".curriculum_builder", "LantraCurriculumBuilder"),
    "CurriculumBuildResult": (".enrichment_contracts", "CurriculumBuildResult"),
    "CurriculumConfig": (".enrichment_contracts", "CurriculumConfig"),
    "CurriculumError": (".enrichment_contracts", "CurriculumError"),
    "KnowledgeFact": (".enrichment_contracts", "KnowledgeFact"),
    "SourceDocument": (".enrichment_contracts", "SourceDocument"),
    "SourceSegment": (".enrichment_contracts", "SourceSegment"),
    "KnowledgeAdapter": (".knowledge_adapter", "KnowledgeAdapter"),
    "PerceptionAdapter": (".perception_adapter", "PerceptionAdapter"),
    "ReasoningAdapter": (".reasoning_adapter", "ReasoningAdapter"),
    "CanonicalLantraSourceAdapter": (".source_adapter", "CanonicalLantraSourceAdapter"),
    "TrainingQualityGate": (".training_quality_gate", "TrainingQualityGate"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
