"""
SLAI STEM Agent. Top-level orchestration facade for the formal STEM subsystem. SLAI's deterministic quantitative scientific-computing authority.

It fills the gap between probabilistic reasoning and deterministic calculation.


This is different from Reasoning:
ReasoningAgent
P(A|B) = ?
Does A imply B?
What explanation fits evidence?

STEMAgent
Solve Ax=b
Integrate f(x)
Solve dy/dt
Compute eigenvalues
Propagate numerical error


It becomes the deterministic mathematical backend used by:
Reasoning
Planning
Simulation
Optimization
Spatial
Experiment
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeAlias, cast

from .base_agent import BaseAgent
from .base.utils.base_errors import *
from .base.utils.base_helpers import *
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import *
from .stem import *
from .stem.utils.stem_errors import *
from .stem.utils.stem_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("STEM Agent")
printer = PrettyPrinter()


class STEMAgent(BaseAgent):
    """
    """

    CHECKPOINTING_SUPPORTED = False

    def __init__(
        self,
        shared_memory: object,
        agent_factory: object,
        config: Mapping[str, object] | None = None,
        *,
        checkpoint_manager: object | None = None,
    ) -> None:
        # Agent-level overrides are deliberately not passed to BaseAgent.
        # BaseAgent owns its own base_agent configuration section; STEMAgent owns only the stem_agent section below.
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            checkpoint_manager=checkpoint_manager,
        )

        self.agent_config: dict[str, object] = dict(get_config_section("stem_agent") or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise BaseConfigurationError(
                    "STEMAgent config override must be a mapping",
                    component=self.name,
                )
            self.agent_config.update(dict(config))

        logger.info(
            "STEM Agent initialized | shared_memory=%s | publish_evidence=%s",
            self.publish_shared_memory,
            self.publish_evidence_metadata,
        )


__all__ = [
    "STEMAgent",
]


if __name__ == "__main__":
    print("\n=== Running STEM Agent ===\n")
    printer.status("TEST", "STEM Agent initialized", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    config = {"publish_shared_memory": False}

    agent = STEMAgent(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
    printer.status("START", agent, "info")

    print("\n=== Test ran successfully ===\n")