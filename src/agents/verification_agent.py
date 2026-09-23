"""
Verification Agent facng facade for the subsystem
"""

from __future__ import annotations

__version__ = "2.3.0"

import threading
import time as _time

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from .base_agent import BaseAgent
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .verification.verification_proof import *
from .verification.verification_invariants import *
from .verification.utils.verification_errors import *
from .verification.utils.verification_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Verification Agent")
printer = PrettyPrinter()

class VerificationAgent(BaseAgent):
    """Application-facing verification orchestration boundary for SLAI"""

    AGENT_KEY = "verification_agent"
    _ALLOWED_CONFIG_KEYS = {
        "enabled",
    }

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Mapping[str, Any] | None = None, *, checkpoint_manager: Any = None) -> None:
        super().__init__(shared_memory, agent_factory, config, checkpoint_manager=checkpoint_manager)

        self.shared_memory = shared_memory if shared_memory is not None else self.shared_memory
        self.agent_factory = agent_factory
        self._alignment_lock = threading.RLock()

        # Agent-level configuration comes from the same central contract used by
        # the other SLAI agents. No alignment subsystem config is read here.
        self.config = load_global_config()
        self.global_config = self.config
        self.agent_config: Dict[str, Any] = dict(get_config_section(self.AGENT_KEY, config=self.config) or {})
        if config:
            if not isinstance(config, Mapping):
                raise ConfigurationError(
                    "Verification Agent runtime config override must be a mapping.",
                    context={"actual_type": type(config).__name__},
                )
            self.agent_config.update(dict(config))

        assert_valid_config_contract(
            global_config=self.config,
            agent_key=self.AGENT_KEY,
            agent_config=self.agent_config,
            logger=logger,
            agent_allowed_keys=self._ALLOWED_CONFIG_KEYS,
            require_global_keys=False,
            require_agent_section=False,
            warn_unknown_global_keys=False,
        )

        self._publish_alignment_event(
            "initialized",
            {
                "agent_id": self.agent_id,
                "enabled": self.enabled,
                "shared_memory_enabled": self.publish_to_shared_memory,
            },
        )

        logger.info(
            "Verification Agent initialized | enabled=%s | shared_memory=%s | risk_threshold=%.3f",
            self.enabled,
            self.publish_to_shared_memory,
            self.risk_threshold,
        )

    def _publish_alignment_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        pass