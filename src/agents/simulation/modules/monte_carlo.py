"""
This is the mathematical heart of the subsystem.
 
Sources:
- Robert & Casella (2004), Monte Carlo Statistical Methods.
- McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. Technometrics, 21(2), 239–245. DOI: 10.1080/00401706.1979.10489755.
- JCGM 101:2008 — Propagation of distributions using a Monte Carlo method. DOI: 10.59161/JCGM101-2008.

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

logger = get_logger("Monte Carlo")
printer = PrettyPrinter()


class MonteCarlo:
    def __int__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.mc_config: Dict[str, Any] = dict(get_config_section("monte_carlo") or {})
        if config:
            self.mc_config.update(dict(config))

        logger.info("")

__all__ = ["MonteCarlo"]