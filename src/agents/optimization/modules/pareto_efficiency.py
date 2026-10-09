"""

sources:
- Pareto Optimal

Core concepts:
- Optimal Solution: A choice where no other available option can improve one criterion without harming another.
- Pareto Frontier (Pareto Front): The set or curve of all Pareto-optimal trade-off solutions plotted in objective space.
- Dominance: Solution A "dominates" solution B if A is strictly better than B in at least one objective and no worse in all others.
- Pareto Improvement: A change that makes at least one objective better off without making any other objective worse off. 

Real-World Applications
- Engineering & Design: Balancing conflicting factors like minimizing manufacturing cost while maximizing product strength or fuel efficiency.
- Economics & Welfare: Distributing goods or resources so that no individual can be made better off without hurting someone else.
- Power Systems & Logistics: Optimizing delivery routes, energy production costs, or emissions reduction using tools like evolutionary algorithms.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from ..optimization_types import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Pareto Efficiency")
printer = PrettyPrinter()


class ParetoEfficiency:
    def __init__(self, config: Optional[Mapping[str, Any]] = None, template: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.pe_config = dict(get_config_section("pareto_efficiency", config=self.config) or {})
        if config:
            self.pe_config.update(dict(config))



__all__ = ["ParetoEfficiency"]