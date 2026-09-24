"""
This provide the generic execution semantics for the simulation models.
It uses Sargent's model-verification framework to define what metadata the engine needs to expose downstream to VerificationAgent,
without having base_engine.py perform the actual formal verification itself.
 
Sources:
- Zeigler et al. (2000) for model/simulator separation and hierarchical simulation.
- Banks et al. (2010) for general discrete-event simulation execution
- Law (2015) for generic simulation methodology.

flow:
initialize state
       ↓
while not terminated:
       ↓
obtain supplied action/intervention
       ↓
apply transition
       ↓
advance simulation clock
       ↓
record state
       ↓
emit trajectory step
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

logger = get_logger("Base Engine")
printer = PrettyPrinter()


class BaseEngine:
    def __int__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.base_config: Dict[str, Any] = dict(get_config_section("base_engine") or {})
        if config:
            self.base_config.update(dict(config))

        logger.info("")

__all__ = ["BaseEngine"]