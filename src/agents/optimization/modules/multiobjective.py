"""

sources:
- Miettinen (1998), Nonlinear Multiobjective Optimization.
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197. DOI 10.1109/4235.996017.

hierarchy should remain:
    multiobjective.py
        Pareto semantics
        dominance
        scalarization
        front representation

    solvers/
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.temp_loader import load_template, TEMPLATE_ROOT
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from ..optimization_types import *
from .pareto_efficiency import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Multiobjective")
printer = PrettyPrinter()


class Multiobjective:
    def __init__(self, config: Optional[Mapping[str, Any]] = None, template: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.multiobjective_config = dict(get_config_section("multiobjective", config=self.config) or {})
        if config:
            self.multiobjective_config.update(dict(config))

        self.template: Dict[str, Any] = load_template("multiobjective")
        self.multi_temp = {}
        if template:
            self.multi_temp.update(dict(template))


__all__ = ["Multiobjective"]