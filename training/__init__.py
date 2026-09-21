"""SLAI training integration package.

This package contains offline training/curriculum orchestration only. Agent
algorithms remain owned by their respective subsystems.
"""

from .curriculum_builder import LantraCurriculumBuilder
from .enrichment_contracts import (
    CurriculumBuildResult,
    CurriculumConfig,
    CurriculumError,
    KnowledgeFact,
    SourceDocument,
    SourceSegment,
)
from .knowledge_adapter import KnowledgeAdapter
from .perception_adapter import PerceptionAdapter
from .reasoning_adapter import ReasoningAdapter
from .source_adapter import CanonicalLantraSourceAdapter
from .training_quality_gate import TrainingQualityGate

__all__ = [
    "LantraCurriculumBuilder",
    "CurriculumBuildResult",
    "CurriculumConfig",
    "CurriculumError",
    "KnowledgeFact",
    "SourceDocument",
    "SourceSegment",
    "KnowledgeAdapter",
    "PerceptionAdapter",
    "ReasoningAdapter",
    "CanonicalLantraSourceAdapter",
    "TrainingQualityGate",
]
