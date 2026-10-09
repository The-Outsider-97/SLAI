"""
OptimizationAgent is solver-oriented rather than ML-tuning-oriented. It finds the best feasible solution to a defined mathematical objective.

Important architectural boundary:
    src/tuning/
        optimize SLAI/model configuration

    OptimizationAgent
        optimize arbitrary problem variables
So Bayesian/grid hyperparameter search should remain in src/tuning/.

Cross-agent relationship:

                    Knowledge
                        │
                        ▼
                    Reasoning
                        │
             ┌──────────┴───────────┐
             ▼                      ▼
           STEM                  Spatial
             │                      │
             └──────────┬───────────┘
                        ▼
                   Optimization
                        │
                        ▼
                     Planning
                        │
                        ▼
                   Simulation
                        │
                        ▼
                  Verification
                        │
                        ▼
                      Safety
                        │
                        ▼
                    Execution


"""
from __future__ import annotations

__version__ = "2.3.0"

import hashlib
import json
import random
import time
import uuid

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple

from .base.utils.config_contract import ConfigContractError, assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .base_agent import BaseAgent
from .optimization import *
from .optimization.utils.optimization_errors import *
from .optimization.utils.optimization_helpers import *
from .optimization.utils.temp_loader import list_templates, load_template
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Agent")
printer = PrettyPrinter()


class OptimizationAgent(BaseAgent):
    """Validated orchestration boundary over the deterministic STEM subsystem."""

    AGENT_KEY = "optimization_agent"
    CHECKPOINTING_SUPPORTED = True
    CHECKPOINT_SCHEMA = "slai.optimization_agent.state.v1"
    _ALLOWED_CONFIG_KEYS = {}
    _TASK_KEYS = {}

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        checkpoint_manager: Any = None,
    ) -> None:
        # Agent-level STEM options are intentionally not injected into
        # BaseAgent's ``base_agent`` config namespace.
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            checkpoint_manager=checkpoint_manager,
        )

        self._stem_lock = RLock()
        self._metrics_lock = RLock()
        self.config = load_global_config()
        self.global_config = self.config
        self.agent_config: Dict[str, Any] = dict(get_config_section(self.AGENT_KEY) or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise OptimizationConfigurationError(
                    "OptimizationAgent config override must be a mapping",
                    component=self.name,
                )
            self.agent_config.update(dict(config))

        try:
            assert_valid_config_contract(
                global_config=self.config,
                agent_key=self.AGENT_KEY,
                agent_config=self.agent_config,
                agent_allowed_keys=self._ALLOWED_CONFIG_KEYS,
                required_agent_keys={"enabled", "supported_domains", "default_domain"},
                require_global_keys=False,
                require_agent_section=True,
                warn_unknown_global_keys=False,
                logger=logger,
            )
        except ConfigContractError as exc:
            raise OptimizationConfigurationError(
                "OptimizationAgent configuration violates the SLAI agent contract",
                component=self.name,
                cause=exc,
            ) from exc
        self._load_agent_config()
        self._validate_agent_config()


        logger.info(
            "OptimizationAgent initialized | domains=%s | shared_publish=%s | checkpointing=%s",
            ",".join(self.supported_domains),
            self.shared_memory_enabled and self.publish_shared_results,
            self.supports_checkpointing,
        )


__all__ = ["OptimizationAgent"]


if __name__ == "__main__":
    configure_logging()
    printer.status("TEST", "OptimizationAgent production façade validation", "info")

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    agent = OptimizationAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
        config={"publish_shared_results": False},
    )
    printer.status("SUCCESS", "", "success")
