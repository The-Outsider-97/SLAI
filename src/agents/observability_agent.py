"""
The Observability Agent converts multi-agent runtime behavior into actionable operational intelligence:
traces, bottlenecks, error families, saturation warnings, and incident summaries.

Interfaces and dependencies
Inputs:
- Logs/events from all agents
- Execution state transitions
- Queue and scheduler metrics

Outputs:
- Alert severity and incident status
- Root-cause hypotheses
- Remediation recommendations

KPIs
- Mean time to detect (MTTD)
- Mean time to resolve (MTTR)
- Alert precision (signal/noise ratio)
- Recurring incident rate
- User-facing degraded response rate

Failure modes & mitigations
- Alert fatigue: dedupe, rate-limit, and contextual suppression.
- Blind spots: enforce telemetry contract in BaseAgent hooks.
- High cardinality metrics: bounded label strategy and rollups.
"""

from __future__ import annotations

from .base_agent import BaseAgent
from .base.utils.main_config_loader import load_global_config, get_config_section
from .observability import ObservabilityCapacity, ObservabilityIntelligence, ObservabilityTracing, ObservabilityPerformance
from .observability.utils.observability_error import ObservabilityError
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Observability Agent")
printer = PrettyPrinter


class ObservabilityAgent(BaseAgent):
    """
    """

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.shared_memory = shared_memory or self.shared_memory
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.observability_config = get_config_section("observability_agent")
        if config:
            self.observability_config.update(dict(config))

        self.enabled = bool(self.observability_config.get("enabled", True))