"""
Production Simulation Agent orchestration for SLAI.

"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any

from typing import Any, Mapping

__version__ = "2.3.0"


from .base_agent import BaseAgent
from .base.utils.base_errors import BaseRuntimeError, BaseValidationError
from .base.utils.base_helpers import *
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .simulation import *
from .simulation.utils.simulation_helpers import *
from .simulation.utils.simulation_errors import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Simulation Agent")
printer = PrettyPrinter()

MODULE_VERSION = __version__
ASSESSMENT_SCHEMA_VERSION = "simulation_agent.assessment.v4"


class SimulationAgent(BaseAgent):
    """Simulation orchestration boundary."""
    def __init__(self, shared_memory: Any, agent_factory: Any, config: Mapping[str, Any] | None = None, *, checkpoint_manager: Any = None) -> None:
        super().__init__(shared_memory, agent_factory, config, checkpoint_manager=checkpoint_manager)
        self.config = load_global_config()
        self.simulation_config: Dict[str, Any] = dict(get_config_section("simulation_agent") or {})
        if config:
            self.simulation_config.update(dict(config))
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        self._publish_event("initialized", self.health_check())
        logger.info("Simulation Agent initialized")
        printer.status("INIT", "Simulation Agent initialized", "success")

    def _publish_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if not self.publish_lifecycle_events or self.shared_memory is None:
            return
        publisher = getattr(self.shared_memory, "publish", None)
        if callable(publisher):
            try:
                publisher(self.event_channel, {"event": event, "payload": json_safe(payload), "timestamp": time_module.time()})
            except Exception as exc:
                logger.debug("SimulationAgent lifecycle publish failed: %s", exc)