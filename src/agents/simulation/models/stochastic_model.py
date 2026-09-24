"""
It owns stochastic environment dynamics, not uncertainty evaluation.
 
Sources:
- Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. Journal of Physical Chemistry, 81(25), 2340–2361. DOI: 10.1021/j100540a008.
- Robert, C. P., & Casella, G. (2004). Monte Carlo Statistical Methods (2nd ed.). Springer. DOI: 10.1007/978-1-4757-4145-2.

functionalities:
- transition distributions
- noise processes
- random disturbances
- stochastic event generation
- probability kernels
- parameter distributions
- process noise
- random event timing
"""

from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..utils.config_loader import *
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Stochastic Model")
printer = PrettyPrinter()


class StochasticModel:
    def __int__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.sm_config: Dict[str, Any] = dict(get_config_section("stochastic_model") or {})
        if config:
            self.sm_config.update(dict(config))

        logger.info("")

__all__ = ["StochasticModel"]