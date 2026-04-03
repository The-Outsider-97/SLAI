"""Safe package exports for the learning/network stack.

Optional modules are imported lazily so missing ancillary components do not break
package import for the core network modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Iterable

__all__ = []


def _export(module_name: str, names: Iterable[str]) -> None:
    try:
        module = import_module(f".{module_name}", __name__)
    except Exception:
        return

    for name in names:
        if hasattr(module, name):
            globals()[name] = getattr(module, name)
            __all__.append(name)


_export(
    "neural_network",
    [
        "Softmax",
        "Loss",
        "CrossEntropyLoss",
        "MSELoss",
        "Optimizer",
        "SGD",
        "SGDMomentum",
        "Adam",
        "NeuralNetwork",
    ],
)
_export(
    "policy_network",
    [
        "PolicyNetwork",
        "NoveltyDetector",
        "create_policy_network",
        "create_policy_optimizer",
    ],
)
_export("multi_task_learner", ["MultiTaskLearner"])
_export("recovery_system", ["RecoverySystem"])
_export("rl_engine", ["StateProcessor", "ExplorationStrategies", "QTableOptimizer"])
_export("state_processor", ["StateProcessor"])
