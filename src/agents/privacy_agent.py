"""
The Data Quality Agent is the quality gatekeeper between raw inputs and all downstream components (knowledge ingestion, training loops, inference context, and memory updates).
It continuously measures and enforces data integrity, consistency, and fitness-for-use.

Interfaces and dependencies
Inputs:
- Raw user prompts and uploads
- Reader/browser extracted content
- Memory writes/reads
- External tool invocation payloads

Outputs:
- Allow/modify/block decision
- Sanitized payloads
- Retention/deletion tasks
- Audit event records

KPIs
- PII leakage incident rate
- Redaction precision/recall
- Policy violation prevention count
- Deletion SLA compliance
- Audit completeness score

Failure modes & mitigations
- Over-redaction harming utility: context-aware exceptions and tiered masking.
- Under-redaction risk: ensemble detectors + conservative defaults.
- Policy drift: versioned policy packs and periodic validation.
"""

from __future__ import annotations

__version__ = "2.1.0"

import time
import uuid

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .base_agent import BaseAgent
from .base.utils.main_config_loader import load_global_config, get_config_section
from .privacy import DataID, DataMinimization, DataRetention, PrivacyAuditability, DataConsent
from .privacy.utils.privacy_error import (PrivacyError)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Privacy Agent")
printer = PrettyPrinter

class PrivacyAgent(BaseAgent):
    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.shared_memory = shared_memory or self.shared_memory
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.privacy_config = get_config_section("privacy_agent")
        if config:
            self.privacy_config.update(dict(config))

        self.enabled = bool(self.privacy_config.get("enabled", True))

        # subsystem initialization
        self.data_consent = DataConsent()
        self.data_id = DataID()
        self.data_min = DataMinimization()
        self.data_retention = DataRetention()
        self.private_audit = PrivacyAuditability()