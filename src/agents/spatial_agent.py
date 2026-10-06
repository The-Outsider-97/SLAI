"""
SLAI v2.3 Spatial Agent orchestration façade.

The Agent introduces a proper world-space representation layer.

PerceptionAgent
    detects / embeds things
    "I detected object A and object B"
              ↓
SpatialAgent
    determines spatial structure
    A = (2.1, 3.7, 0.5)
    B = (4.8, 1.2, 0.5)

    distance(A,B)
    intersects(A,B)
    inside(A,room)
    visibility(A,B)
    transform(camera → world)
              ↓
PlanningAgent
    decides what to do in that structure
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Any, Optional, Type

from .base_agent import BaseAgent
from .base.utils.main_config_loader import get_config_section
from .runtime_contracts import RuntimeLifecycle
from .spatial import *
from .spatial.utils.spatial_errors import *
from .spatial.utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Agent")
printer = PrettyPrinter()


class SpatialAgent(BaseAgent):
    """System-wide orchestration façade for SLAI spatial infrastructure."""

    AGENT_KEY = "spatial_agent"
    STATE_UPDATED_TOPIC = "spatial_agent:state_updated"
    CHECKPOINT_SCHEMA = "slai.spatial-agent.state.v3"
    CHECKPOINTING_SUPPORTED = True

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Mapping[str, Any] | None = None, *, checkpoint_manager: Any = None) -> None:
        super().__init__(shared_memory, agent_factory, config, checkpoint_manager=checkpoint_manager)
        self.agent_config: dict[str, Any] = dict(get_config_section(self.AGENT_KEY) or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise BaseConfigurationError(
                    "SpatialAgent config override must be a mapping",
                    component=self.name,
                    context={"type": type(config).__name__}
                    )
            self.agent_config.update(dict(config))

        self.relations = SpatialRelations()
        self.queries = SpatialQueries()
        self.index = SpatialIndex()
        self.compute = SpatialCompute()
        # SpatialMemory() is only use within these 4 modules, the Agent should remain unaware of its exisitance


__all__ = ["SpatialAgent"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Spatial Agent ===\n")
    printer.status("TEST", "Spatial Agent initialized", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    config = {"publish_shared_memory": False}

    agent = SpatialAgent(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
    printer.status("START", agent, "info")

