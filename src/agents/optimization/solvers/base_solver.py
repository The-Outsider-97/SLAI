"""
Base
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Base Optimization Solver")
printer = PrettyPrinter()


class BaseSolver:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.base_config = dict(get_config_section("base_solver", config=self.config) or {})
        if config:
            self.base_config.update(dict(config))


__all__ = ["BaseSolver"]