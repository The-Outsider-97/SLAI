from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ReasoningMemory": (".reasoning_memory", "ReasoningMemory"),
    "MemorySample": (".reasoning_memory", "MemorySample"),
    "Transition": (".reasoning_memory", "Transition"),
    "ReasoningCache": (".reasoning_cache", "ReasoningCache"),
    "CacheEntry": (".reasoning_cache", "CacheEntry"),
    "CacheCounters": (".reasoning_cache", "CacheCounters"),
    "ReasoningTypes": (".reasoning_types", "ReasoningTypes"),
    "RuleEngine": (".rule_engine", "RuleEngine"),
    "ValidationEngine": (".validation", "ValidationEngine"),
    "HybridProbabilisticModels": (".hybrid_probabilistic_models", "HybridProbabilisticModels"),
    "HybridStrategySpec": (".hybrid_probabilistic_models", "HybridStrategySpec"),
    "HybridBuildReport": (".hybrid_probabilistic_models", "HybridBuildReport"),
    "ProbabilisticModels": (".probabilistic_models", "ProbabilisticModels"),
    "NetworkSelectionDecision": (".probabilistic_models", "NetworkSelectionDecision"),
    "InferenceTrace": (".probabilistic_models", "InferenceTrace"),
    "LearningCycleReport": (".probabilistic_models", "LearningCycleReport"),
}


__all__ = list(_EXPORTS) # type: ignore


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)

    if target is None:
        raise AttributeError(
            f"module {__name__!r} "
            f"has no attribute {name!r}"
        )

    module_name, attribute_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)

    globals()[name] = value

    return value


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(__all__)
    )