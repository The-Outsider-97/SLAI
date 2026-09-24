"""
This is the mathematical heart of the subsystem.
 
Sources:
- Puterman (1994) provides the stochastic state-transition basis.
- Zeigler et al. (2000), Theory of Modeling and Simulation.
- Van Tendeloo & Vangheluwe (2018). Discrete Event System Specification Modeling and Simulation. Winter Simulation Conference. DOI: 10.1109/WSC.2018.8632372.

"""

from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, Dict, Mapping, Optional

from ...base.modules.math_science import *
from ..utils.config_loader import *
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("State Transition")
printer = PrettyPrinter()


@abstractmethod
def transition():
    ...


class StateTransition:
    def __int__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.state_config: Dict[str, Any] = dict(get_config_section("state_transition") or {})
        if config:
            self.state_config.update(dict(config))

        next_state = transition(
            state,
            action,
            intervention=None,
            parameters=None,
            rng=None,
        )
        self.state = next_state
        logger.info("")

__all__ = ["StateTransition"]