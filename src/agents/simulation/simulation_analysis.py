"""
This perform simulation-output analysis, not Evaluation. 
 
Sources:
- Law, A. M. (1983). Statistical Analysis of Simulation Output Data. Operations Research, 31(6), 983–1029. DOI: 10.1287/opre.31.6.983.
- Heidelberger, P., & Welch, P. D. (1983). Simulation Run Length Control in the Presence of an Initial Transient. Operations Research, 31(6), 1109–1144. DOI: 10.1287/opre.31.6.1109.
- Saltelli et al. (2008). Global Sensitivity Analysis: The Primer. Wiley. DOI: 10.1002/9780470725184.
- Sobol, I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. Mathematics and Computers in Simulation, 55, 271–280. DOI: 10.1016/S0378-4754(00)00270-6.
- Morris, M. D. (1991). Factorial Sampling Plans for Preliminary Computational Experiments. Technometrics, 33(2), 161–174. DOI: 10.1080/00401706.1991.10484804.

It performs:
- uncertainty summaries
- confidence intervals
- distribution summaries
- branch frequencies
- trajectory divergence
- sensitivity indices
- perturbation effects
- Monte Carlo convergence diagnostics
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

logger = get_logger("Simulation Analysis")
printer = PrettyPrinter()


class SimulationAnalysis:
    def __int__(
            self,
            config: Optional[Mapping[str, Any]] = None,
            *,
            memory: Optional[SimulationMemory] = None) -> None:
        self.config = load_global_config()
        self.analysis_config: Dict[str, Any] = dict(get_config_section("simulation_analysis") or {})
        if config:
            self.analysis_config.update(dict(config))

        self.memory = memory

__all__ = [
    "SimulationAnalysis",
]