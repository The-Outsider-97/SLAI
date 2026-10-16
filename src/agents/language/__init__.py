"""SLAI language subsystem public API.

The package surface is lazy by design. Importing one language submodule must not
initialize every NLP/NLG/dialogue component or acquire optional native runtimes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {}


def _register(module_name: str, *names: str) -> None:
    for name in names:
        _EXPORTS[name] = (module_name, name)


_register(
    ".language_memory",
    "MemoryKind", "MemoryScope", "MemoryRole", "MemoryQuery", "MemoryRecord",
    "MemoryMatch", "MemorySnapshot", "LanguageMemoryStats", "LanguageMemoryConfig",
    "LanguageMemory",
)
_register(
    ".dialogue_context",
    "DialogueRole", "ConversationPhase", "DialogueContextConfig", "DialogueMessage",
    "DialogueTurn", "SlotValue", "IntentTrace", "UnresolvedIssueRecord",
    "DialogueContextStats", "DialogueContextSnapshot", "DialogueContext",
)
_register(
    ".grammar_processor",
    "GrammarSeverity", "InputToken", "DiagnosticGrammarIssue", "GrammarIssue",
    "SentenceGrammarAnalysis", "GrammarAnalysisResult", "GrammarProcessorStats",
    "GrammarProcessor",
)
_register(
    ".orthography_processor",
    "OrthographyToken", "OrthographyEdit", "OrthographyProcessingResult",
    "OrthographyProcessorStats", "OrthographyProcessor",
)
_register(
    ".nlg_engine",
    "NLGTemplate", "NLGTemplateSet", "NLGContextPacket", "NLGRenderAttempt",
    "NLGGenerationResult", "NLGEngineStats", "NLGEngine",
)
_register(
    ".nlp_engine",
    "Entity", "Token", "SentenceAnalysis", "NLPAnalysisResult", "NLPEngineStats",
    "NLPEngine",
)
_register(
    ".nlu_engine",
    "IntentMatchSource", "EntitySource", "NLUSeverity", "NLUIssue", "NLUInputToken",
    "WordlistEntry", "IntentPattern", "EntityPattern", "IntentCandidate",
    "EntityMention", "NLUAnalysisResult", "NLUStats", "Wordlist", "NLUEngine",
    "EnhancedNLU",
)

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
