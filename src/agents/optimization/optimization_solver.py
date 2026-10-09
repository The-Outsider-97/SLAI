"""
Handels the optimization/solvers/ submodules
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from .utils.config_loader import get_config_section, load_global_config
from .utils.optimization_errors import *
from .utils.optimization_helpers import *
from .solvers import CombinatorialSolver, LinearSolver, NonLinearSolver, QuadraticSolver, BlackBoxSolver
from .optimization_types import *
from .optimization_memory import OptimizationMemory
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Solver")
printer = PrettyPrinter()


class OptimizationSolver:
    def __init__(self, config: Optional[Mapping[str, Any]] = None, memory: Optional[Any[OptimizationMemory]]=None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.solver_config = dict(get_config_section("optimization_solver", config=self.config) or {})
        if config:
            self.solver_config.update(dict(config))

        self.memory = memory


__all__ = ["OptimizationSolver"]