"""
Handels the optimization/registries/ submodules
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from .utils.config_loader import get_config_section, load_global_config
from .utils.optimization_errors import *
from .utils.optimization_helpers import *
from .registries import 
from .optimization_types import *
from .optimization_memory import OptimizationMemory
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Registries")
printer = PrettyPrinter()


class OptimizationRegistries:
    def __init__(self, config: Optional[Mapping[str, Any]] = None, memory: Optional[Any[OptimizationMemory]]=None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.registry_config = dict(get_config_section("optimization_registries", config=self.config) or {})
        if config:
            self.registry_config.update(dict(config))

        self.memory = memory


__all__ = ["OptimizationRegistries"]