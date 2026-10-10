"""
Optimization memory's job is storing facts such as:
problem fingerprint
problem classification
selected solver
solver version
parameters/options
termination state
objective value
constraint violations
bounds
optimality gap
runtime
iterations/nodes/evaluations
Pareto set
warm-start state

The academic basis comes more from reproducible solver benchmarking.

sources:
- Dolan, E. D., & Moré, J. J. (2002). Benchmarking optimization software with performance profiles. Mathematical Programming, 91, 201–213. DOI 10.1007/s101070100263.
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from .utils.config_loader import get_config_section, load_global_config
from .utils.temp_loader import load_template, TEMPLATE_ROOT
from .utils.optimization_errors import *
from .utils.optimization_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Memory")
printer = PrettyPrinter()


class OptimizationMemory:
    def __init__(self, config: Optional[Mapping[str, Any]] = None, template: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.memory_config = dict(get_config_section("optimization_memory", config=self.config) or {})
        if config:
            self.memory_config.update(dict(config))

        self.template: Dict[str, Any] = load_template("memory")
        self.memory_temp = {}
        if template:
            self.memory_temp.update(dict(template))


__all__ = ["OptimizationMemory"]
