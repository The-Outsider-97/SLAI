"""SLAI governance package: mandatory regulatory controls (GDPR/AVG, EU AI Act, UAVG).

Imports are lazy so importing this package never pulls in torch, the network or other
agents. Everything exported here is stdlib-only except ``GovernanceAgent`` (which lives
in ``src/agents/governance_agent.py`` and depends on ``BaseAgent``).
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "GovernanceCore": ".core",
    "ProcessingContext": ".processing_context",
    "ProcessingDecision": ".processing_context",
    "Decision": ".processing_context",
    "DataClass": ".processing_context",
    "Operation": ".processing_context",
    "LawfulBasis": ".processing_context",
    "GovernanceDenied": ".processing_context",
    "IdentityContext": ".processing_context",
    "PolicyEngine": ".policy_engine",
    "PolicyRegistry": ".policy_registry",
    "CapabilityRegistry": ".policy_registry",
    "Capability": ".policy_registry",
    "ProcessingRegistry": ".processing_registry",
    "ProcessingActivity": ".processing_registry",
    "RiskClassifier": ".risk_classification",
    "AISystemProfile": ".risk_classification",
    "OversightManager": ".human_oversight",
    "DecisionDossier": ".human_oversight",
    "IncidentRegistry": ".incident_registry",
    "EvidenceStore": ".regulatory_evidence",
    "ComplianceEngine": ".compliance_engine",
    "DeploymentProfile": ".compliance_engine",
    "RegulatoryReporter": ".regulatory_reporting",
    "GrantAuthority": ".regulatory_context",
    "AuthorizationGrant": ".regulatory_context",
    "IdentityAuthority": ".utils.identity",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(target, __name__), name)
    globals()[name] = value
    return value
