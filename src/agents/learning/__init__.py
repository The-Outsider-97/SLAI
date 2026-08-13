"""Lazy public API for the learning subsystem.

Importing a concrete learning module must not initialize every learning agent.
In particular, ``python -m src.agents.learning.dqn`` should not import the
factory and RSI modules before DQN itself is executed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np  # type: ignore


# NumPy 2.x compatibility for legacy Gym code paths that still check np.bool8.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_  # type: ignore[attr-defined]


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "DQNAgent": (".dqn", "DQNAgent"),
    "DecentralizedMAMLFleet": (".maml_rl", "DecentralizedMAMLFleet"),
    "EvolutionaryTrainer": (".dqn", "EvolutionaryTrainer"),
    "LearningFactory": (".learning_factory", "LearningFactory"),
    "LearningMemory": (".learning_memory", "LearningMemory"),
    "MAMLAgent": (".maml_rl", "MAMLAgent"),
    "RLAgent": (".rl_agent", "RLAgent"),
    "RSIAgent": (".rsi", "RSIAgent"),
    "SLAIEnv": (".slaienv", "SLAIEnv"),
    "SkillLibrary": (".rsi", "SkillLibrary"),
    "StrategySelector": (".strategy_selector", "StrategySelector"),
    "SumTree": (".learning_memory", "SumTree"),
    "UnifiedDQNAgent": (".dqn", "UnifiedDQNAgent"),
}

__all__ = sorted(_EXPORTS) # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str):
    """Resolve one exported learning symbol on first use."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = __import__(f"{__name__}{module_name}", fromlist=[attribute_name])
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
