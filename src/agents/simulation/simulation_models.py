"""
This is the model abstraction/registry layer.

Sources:
- Zeigler et al. (2000), Theory of Modeling and Simulation.
- Banks, Carson, Nelson & Nicol (2010), Discrete-Event System Simulation, 5th ed.

"""

from __future__ import annotations

__version__ = "2.3.0"

import threading
import re

from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Any, Dict, Mapping, Protocol, Optional

from .models.stochastic_model import *
from .models.world_model import *
from .utils.config_loader import *
from .utils.simulation_errors import *
from .utils.simulation_helpers import *
from .simulation_memory import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Simulation Models")
printer = PrettyPrinter()


class SimulationModel(Protocol):
    config: Optional[Mapping[str, Any]]
    model_config: Dict[str, Any]
    memory: Optional[SimulationMemory]
    world: Optional[WorldModel]
    stochastic: Optional[StochasticModel]
    _lock: threading.RLock

    def __init__(self, config, *, memory, world, stochastic) -> None:
        self.config = load_global_config()
        self.model_config: Dict[str, Any] = dict(get_config_section("simulation_model") or {})
        if config:
            self.model_config.update(dict(config))

        self.memory = memory
        self.world = world
        self.stochastic = stochastic
        self._lock = threading.RLock()

        logger.info("All models successfully registered")
    
    def initialize(...): ... # type: ignore

    def transition(...): ... # type: ignore

    def observe(...): ... # type: ignore

    def snapshot(...): ... # type: ignore


__all__ = ["SimulationModel"]