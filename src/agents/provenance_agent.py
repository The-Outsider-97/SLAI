from __future__ import annotations
__version__ = "2.2.0"

"""
Provenance Agent is a system-wide authority for artifact lineage and chain-of-custody.

The Knowledge subsystem already stores provenance for inferred knowledge, and Quality assesses source trust.
But neither is a system-wide lineage authority.

The Provenance Agent answers the following questions:
Where exactly did this result come from?

Across the whole SLAI system.

The agent reconstructs:

final-output-42
 ├── reasoning-event-103
 │    ├── fact-51
 │    │    └── document-18
 │    │         └── source URL
 │    └── rule-7
 └── model-version
      └── checkpoint

Important distinction
ObservabilityAgent = what happened operationally?
ProvenanceAgent    = where did this artifact/information originate?

QualityAgent       = can this data be trusted/used?
ProvenanceAgent    = how did this data reach this result?

KnowledgeAgent     = what information do we know?
ProvenanceAgent    = how was that information derived?

Observability already explicitly owns execution tracing and incident intelligence, not artifact lineage.

This is a powerful addition for academic rigor as well.
"""

from typing import Any, Mapping, Optional

from .base_agent import BaseAgent
from .base.utils.main_config_loader import get_config_section, load_global_config
from .provenance import *
from .provenance.provenance_types import *
from .provenance.utils.provenance_errors import *
from .provenance.utils.provenance_helpers import *
from .runtime_contracts import RuntimeLifecycle
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Agent")
printer = PrettyPrinter()

class ProvenanceAgent(BaseAgent):
    """
    The Provenance Agent is responsible for tracking the lineage and chain-of-custody of artifacts within the SLAI system.
    It reconstructs the provenance of artifacts, providing a clear understanding of their origins and transformations.
    """

    AGENT_KEY = "provenance_agent"
    STATE_UPDATED_TOPIC = "provenance_agent:state_updated"
    CHECKPOINT_SCHEMA = "slai.provenance-agent.state.v2"

    _ALLOWED_CONFIG_KEYS = {}
    CHECKPOINTING_SUPPORTED = True

    def __init__(self, shared_memory, agent_factory, config: Optional[Mapping[str, Any]]=None, checkpoint_manager: Any = None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config, checkpoint_manager=checkpoint_manager, **kwargs)
        self.adaptive_agent = None
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.provenance_agent_config = get_config_section("provenance_agent") or {}

        self.runtime_lifecycle = RuntimeLifecycle()
        self.provenance_store = ProvenanceStore()
        self.provenance_lineage = ProvenanceLineage()
        self.provenance_custody = ProvenanceCustody()
        self.provenance_memory = ProvenanceMemory()

    def track_artifact(self, artifact_id: str):
        """
        Track the provenance of a given artifact by its ID.

        Args:
            artifact_id (str): The unique identifier of the artifact to track.

        Returns:
            dict: A dictionary representing the provenance information of the artifact.
        """
        try:
            provenance_info = self.provenance_store.get_provenance(artifact_id)
            return provenance_info
        except Exception as e:
            logger.error(f"Failed to track artifact {artifact_id}: {e}")
            raise ProvenanceTrackingError(f"Could not track artifact {artifact_id}") from e

    def chain_of_custody(self, artifact_id: str):
        """
        Retrieve the chain-of-custody for a given artifact.

        Args:
            artifact_id (str): The unique identifier of the artifact.

        Returns:
            dict: A dictionary representing the chain-of-custody information of the artifact.
        """
        try:
            custody_info = self.provenance_custody.record_custody(artifact_id, owner="current_owner")
            return custody_info
        except Exception as e:
            logger.error(f"Failed to retrieve chain-of-custody for artifact {artifact_id}: {e}")
            raise ProvenanceCustodyError(f"Could not retrieve chain-of-custody for artifact {artifact_id}") from e


if __name__ == "__main__":
    print("\n=== Running Provenance Agent ===\n")
    printer.status("TEST", "Provenance Agent initialized", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    memory = SharedMemory()
    factory = AgentFactory()
    provenance_config = get_config_section("provenance_agent")

    agent = ProvenanceAgent(shared_memory=memory, agent_factory=factory, config=provenance_config)
