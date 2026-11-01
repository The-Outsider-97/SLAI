"""Knowledge subsystem package exports.

Keep import-time package initialization lightweight. Shared Knowledge utility
modules and the standalone assurance package must not eagerly construct the
KnowledgeOrchestrator or import its optional runtime dependencies.
"""
from __future__ import annotations

__all__ = ["KnowledgeOrchestrator"]


def __getattr__(name: str):
    if name != "KnowledgeOrchestrator":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .knowledge_orchestrator import KnowledgeOrchestrator
    globals()[name] = KnowledgeOrchestrator
    return KnowledgeOrchestrator
