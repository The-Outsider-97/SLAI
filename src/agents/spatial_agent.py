"""
SLAI v2.3 Spatial Agent orchestration façade.

The Agent introduces a proper world-space representation layer.

Perception
    observations / detections
           ↓
SpatialAgent
    structured world-space representation
    coordinate systems
    geometry
    topology
    relations
    maps
    indices
    spatial predicates
           ↓
Planning
    consumes spatial state

sources:
- Cohn, A. G., & Renz, J. (2008). Qualitative Spatial Representation and Reasoning. In Handbook of Knowledge Representation.
- Kuipers, B. (2000). “The Spatial Semantic Hierarchy.” Artificial Intelligence, 119(1–2), 191–233.
- Lynch, K. M., & Park, F. C. (2017). Modern Robotics: Mechanics, Planning, and Control.


A point cloud received from Perception should be interpreted spatially as a geometric dataset:

Perception:
raw sensor → detected/filtered point cloud

Spatial:
point cloud → spatial geometry
            → index
            → registration
            → surface/volume representation
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

