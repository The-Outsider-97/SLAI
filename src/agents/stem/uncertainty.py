"""

There are three different concepts that SLAI should not conflate:
    numerical error
            ↓
    finite precision / approximation / truncation

    measurement uncertainty
            ↓
    uncertainty attached to measured input quantities

    epistemic/probabilistic uncertainty
            ↓
    belief about what is true
The first two belong in STEM. The third generally belongs to Reasoning.

sources:
- JCGM 100:2008 — Guide to the Expression of Uncertainty in Measurement (GUM).
- 2026 Amendment 1 on nonlinearity in measurement models
- JCGM GUM-6:2020 — Developing and Using Measurement Models.
- JCGM 101:2008 — Propagation of Distributions Using a Monte Carlo Method.
"""

from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Dict, Mapping, Optional

from ..base.modules.math_science import *
from .utils.config_loader import load_global_config, get_config_section
from .utils.stem_errors import *
from .utils.stem_helpers import *
from .stem_memory import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("SLAI Uncertainty")
printer = PrettyPrinter()


def standard_uncertainty():
    pass 


def combined_standard_uncertainty():
    pass 


def expanded_uncertainty():
    pass 


def covariance_propagation():
    pass 


def jacobian_propagation():
    pass 


def sensitivity_coefficients():
    pass 


def coverage_interval():
    pass 


def monte_carlo_propagation():
    """should accept an explicit seed and record it in result metadata.""" 


def numerical_error_budget():
    pass 




class Uncertainty():
    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None):
        self.config: Dict[str, Any] = load_global_config()
        self.uncertainty_config = dict(get_config_section("stem_uncertainty", config=self.config) or {})
        if config:
            self.uncertainty_config.update(dict(config))

        self.memory = memory

__all__ = ["Uncertainty"]