from .compression_engine import CompressionReport, CompressionEngine
from .conversion_engine import ConversionArtifact, MergeArtifact, ConversionEngine
from .detection_engine import FormatSpec, DetectionResult, DetectionEngine
from .integrity_engine import ExpansionLimits, ExpansionGuard, IntegrityEngine
from .job_engine import JobEngine
from .parser_engine import ParserEngine, _HTMLTextExtractor
from .reader_memory import ReaderMemoryPaths, ReaderMemoryEntry, ReaderMemory
from .recovery_engine import RecoveryDecision, RecoveryEngineProfile, RecoveryEngine
from .salvage_engine import SalvageReport, SalvageEngine

__all__ = [
    "ParserEngine",
    "_HTMLTextExtractor",
    # Conversion
    "ConversionArtifact",
    "MergeArtifact",
    "ConversionEngine",
    # Recovery
    "RecoveryDecision",
    "RecoveryEngineProfile",
    "RecoveryEngine",
    # Memory
    "ReaderMemoryPaths",
    "ReaderMemoryEntry",
    "ReaderMemory",
    # Foundations
    "FormatSpec",
    "DetectionResult",
    "DetectionEngine",
    "ExpansionLimits",
    "ExpansionGuard",
    "IntegrityEngine",
    "JobEngine",
    # Capabilities
    "CompressionReport",
    "CompressionEngine",
    "SalvageReport",
    "SalvageEngine",
]

__version__ = "2.4.0"
