"""
Sources:
- Puterman (1994) for state-transition trajectory semantics.
- Hafner et al. (2020) for trajectories imagined through learned world models.
- Ha & Schmidhuber (2018) as supporting literature for simulated trajectories inside learned world models.

It supports:
- single deterministic rollout
- N stochastic rollouts
- branched rollout
- intervention rollout
- perturbed rollout
- world-model rollout
"""

from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, Dict, Mapping, Optional

from .base_engine import BaseEngine
from ..utils.config_loader import *
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Rollout Engine")
printer = PrettyPrinter()

@dataclass(frozen=True)
class CounterfactualRollout:
    pass


class RolloutEngine(BaseEngine):
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        self.config = load_global_config()
        self.ro_config: Dict[str, Any] = dict(get_config_section("rollout_engine") or {})
        if config:
            self.ro_config.update(dict(config))

        logger.info("")


__all__ = ["RolloutEngine"]