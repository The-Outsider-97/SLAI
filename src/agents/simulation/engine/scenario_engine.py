"""
It owns scenario construction and branching execution.
For example:

                    State S0
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           nominal   wet-road   sensor-noise
             │         │         │
          rollout    rollout    rollout
             │         │         │
             ▼         ▼         ▼

Sources:
- Schoemaker, P. J. H. (1995). Scenario Planning: A Tool for Strategic Thinking. Sloan Management Review, 36, 25–40.
- Morris (1991). For systematic parameter perturbation
- Saltelli et al. (2008). For global sensitivity and scenarios
- McKay et al. (1979). For sampling different environment parameter combinations

"""

from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, Dict, Mapping, Optional

from .base_engine import BaseEngine
from ..utils.config_loader import *
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Scenario Engine")
printer = PrettyPrinter()


class ScenarioEngine(BaseEngine):
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        self.config = load_global_config()
        self.scenario_config: Dict[str, Any] = dict(get_config_section("scenario_engine") or {})
        if config:
            self.scenario_config.update(dict(config))

        logger.info("")


__all__ = ["ScenarioEngine"]