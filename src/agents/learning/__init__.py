import numpy as np # type: ignore

# Global NumPy 2.x compatibility shim for legacy Gym code paths that still check np.bool8.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from .dqn import DQNAgent, EvolutionaryTrainer, UnifiedDQNAgent
from .learning_factory import LearningFactory
from .learning_memory import SumTree, LearningMemory
from .maml_rl import MAMLAgent, DecentralizedMAMLFleet
from .rl_agent import RLAgent
from .rsi import RSIAgent, SkillLibrary
from .slaienv import SLAIEnv
from .strategy_selector import StrategySelector

__all__ = [
    # Learning Memory
    "SumTree",
    "LearningMemory",
    # DQN
    "DQNAgent",
    "EvolutionaryTrainer",
    "UnifiedDQNAgent",
    # MAML
    "MAMLAgent",
    "DecentralizedMAMLFleet",
    # RLA
    "RLAgent",
    # RSI
    "RSIAgent",
    "SkillLibrary",
    # SLAIEnv
    "SLAIEnv",
    # Selector
    "StrategySelector",
    # Factory
    "LearningFactory",
]