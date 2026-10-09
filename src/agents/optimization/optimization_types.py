"""
This module defines mathematical-domain types rather than algorithms

sources:
- Boyd & Vandenberghe (2004) for continuous/convex terminology, Nocedal & Wright (2006)
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from .utils.config_loader import get_config_section, load_global_config
from .utils.optimization_errors import *
from .utils.optimization_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Types")
printer = PrettyPrinter()


class OptimizationProblem:
    pass


class Variable:
    pass


class VariableDomain:
    pass


class Objective:
    pass


class Constraint:
    pass


class ProblemClass:
    pass


class SolverCapability:
    pass


class SolverStatus:
    pass


class Solution:
    pass


class FeasibilityStatus:
    pass


class OptimalityStatus:
    pass


class ParetoPoint:
    pass


class ParetoFront:
    pass


class TerminationReason:
    pass


__all__ = [
    "OptimizationProblem",
    "Variable",
    "VariableDomain",
    "Objective",
    "Constraint",
    "ProblemClass",
    "SolverCapability",
    "SolverStatus",
    "Solution",
    "FeasibilityStatus",
    "OptimalityStatus",
    "ParetoPoint",
    "ParetoFront",
    "TerminationReason",
]