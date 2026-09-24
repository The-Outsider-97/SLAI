"""
This is the formal vocabulary of the subsystem, grounded primarily in Zeigler and Puterman.

Sources:
- Zeigler et al. (2000), Theory of Modeling and Simulation.
- Puterman (1994), Markov Decision Processes.

"""

from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .utils.config_loader import *
from .utils.simulation_errors import *
from .utils.simulation_helpers import *
from .simulation_memory import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Simulation Types")
printer = PrettyPrinter()


@dataclass(frozen=True)
class SimulationVocab:
    SimulationState: str
    Transition: str
    TransitionKernel: str
    Trajectory: str
    TrajectoryStep: str
    SimulationRequest: str
    SimulationResult: str
    SimulationRun: str
    SimulationSeed: str
    Intervention: str
    Perturbation: str
    Scenario: str
    ScenarioBranch: str
    EnvironmentState: str
    ModelParameters: str
    StochasticParameters: str
    SimulationStatistics: str
    TerminationCondition: str 


class SimulationTypes:
    def __int__(
            self,
            config: Optional[Mapping[str, Any]] = None,
            *,
            memory: Optional[SimulationMemory] = None) -> None:
        self.config = load_global_config()
        self.types_config: Dict[str, Any] = dict(get_config_section("simulation_types") or {})
        if config:
            self.types_config.update(dict(config))

        self.memory = memory

__all__ = [
    "SimulationVocab",
    "SimulationTypes",
]