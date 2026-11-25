"""Versioned policy catalogue and the registered-capability catalogue.

Ownership model
---------------
* Every mandatory policy is defined here, in code, with its legal references.
  Configuration can *add* stricter policies; it can never remove, deactivate or
  weaken a mandatory one (``deactivate`` raises, ``register`` of a lower
  version for an existing id raises).
* Capabilities are the only thing that may be delegated/executed. A capability
  names its owner agent, the exact action, and the data classes it may touch.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from .processing_context import DataClass, fingerprint

__all__ = [
    "Capability",
    "CapabilityRegistry",
    "MandatoryPolicyError",
    "Policy",
    "PolicyRegistry",
    "BUILTIN_POLICIES",
]


class MandatoryPolicyError(PermissionError):
    pass


@dataclass(frozen=True)
class Policy:
    policy_id: str
    version: int
    title: str
    legal_refs: Tuple[str, ...]
    mandatory: bool = True
    owner: str = "governance"
    effective_from: float = 0.0


def _p(pid: str, title: str, *refs: str, owner: str = "governance") -> Policy:
    return Policy(pid, 1, title, tuple(refs), True, owner)


BUILTIN_POLICIES: Tuple[Policy, ...] = (
    _p("GOV-ID-01", "Authenticated, tenant-bound identity for protected processing", "GDPR Art. 5(1)(f)", "GDPR Art. 32"),
    _p("GOV-CLS-02", "Unclassified data is quarantined; non-personal claims must be verified", "GDPR Art. 4(1)", "GDPR Art. 5(2)", owner="privacy"),
    _p("GOV-AI5-03", "AI Act Art. 5 prohibited practices are never executable", "AI Act Art. 5"),
    _p("GOV-LAW-04", "Lawful basis resolved from a registered processing activity", "GDPR Art. 5(1)(a)", "GDPR Art. 6", "GDPR Art. 30", owner="privacy"),
    _p("GOV-PUR-05", "Purpose and operation must be within the registered activity", "GDPR Art. 5(1)(b)", "GDPR Art. 6(4)", owner="privacy"),
    _p("GOV-SPC-06", "Art. 9/10 condition required for special-category, biometric and criminal-offence data", "GDPR Art. 9", "GDPR Art. 10", owner="privacy"),
    _p("GOV-CON-07", "Consent-based processing requires active, specific, withdrawable consent per subject", "GDPR Art. 7", "GDPR Art. 8", owner="privacy"),
    _p("GOV-RTS-08", "Restriction, objection and pending erasure are honoured", "GDPR Art. 17", "GDPR Art. 18", "GDPR Art. 21", owner="privacy"),
    _p("GOV-TRF-09", "Third-country transfers require a registered transfer mechanism", "GDPR Art. 44-49", owner="privacy"),
    _p("GOV-ADM-10", "Significant automated decisions require genuine human intervention", "GDPR Art. 22", "AI Act Art. 14"),
    _p("GOV-LRN-11", "Learning on personal data requires a purpose-compatible activity", "GDPR Art. 5(1)(b)", "GDPR Art. 25", owner="privacy"),
    _p("GOV-AIS-12", "AI systems must have a registered regulatory profile", "AI Act Art. 6", "AI Act Art. 49"),
    _p("GOV-CAP-13", "Only registered capabilities may be executed or delegated", "GDPR Art. 25", "GDPR Art. 32", owner="execution"),
    _p("GOV-HO-14", "Human approval must come from a verified human, bound to the exact request", "AI Act Art. 14", "GDPR Art. 22(3)"),
    _p("GOV-EMG-15", "Emergency termination halts affected processing immediately", "AI Act Art. 14(4)(e)"),
    _p("GOV-ERR-99", "Internal evaluation failure fails closed", "GDPR Art. 25"),
)


class PolicyRegistry:
    def __init__(self, extra: Iterable[Policy] = ()):
        self._lock = RLock()
        self._policies: Dict[str, Policy] = {}
        self._active_version: Optional[str] = None
        for policy in BUILTIN_POLICIES:
            self._policies[policy.policy_id] = policy
        for policy in extra:
            self.register(policy)

    def register(self, policy: Policy) -> None:
        with self._lock:
            existing = self._policies.get(policy.policy_id)
            if existing is not None:
                if existing.mandatory and not policy.mandatory:
                    raise MandatoryPolicyError(f"cannot downgrade mandatory policy {policy.policy_id}")
                if policy.version <= existing.version:
                    raise MandatoryPolicyError(
                        f"policy {policy.policy_id} v{policy.version} does not supersede v{existing.version}"
                    )
            self._policies[policy.policy_id] = policy
            self._active_version = None

    def deactivate(self, policy_id: str) -> None:
        with self._lock:
            policy = self._policies.get(policy_id)
            if policy is None:
                raise KeyError(policy_id)
            if policy.mandatory:
                raise MandatoryPolicyError(f"mandatory policy {policy_id} cannot be deactivated")
            del self._policies[policy_id]
            self._active_version = None

    def get(self, policy_id: str) -> Optional[Policy]:
        return self._policies.get(policy_id)

    def list(self) -> List[Policy]:
        with self._lock:
            return sorted(self._policies.values(), key=lambda p: p.policy_id)

    def active_version(self) -> str:
        """Content digest over every active policy (id+version); changes whenever any policy does."""
        with self._lock:
            if self._active_version is None:
                digest = fingerprint([(p.policy_id, p.version, p.mandatory) for p in self.list()])
                self._active_version = f"slai-gov-{digest[:16]}"
            return self._active_version


@dataclass(frozen=True)
class Capability:
    capability_id: str  # e.g. "privacy.evaluate_privacy"
    owner_agent: str
    action: str
    allowed_data_classes: FrozenSet[DataClass] = field(
        default_factory=lambda: frozenset({DataClass.NON_PERSONAL, DataClass.ANONYMISED})
    )
    requires_human_approval: bool = False
    tenant_bound: bool = False


class CapabilityRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._caps: Dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        if not cap.capability_id or cap.action.startswith("_"):
            raise ValueError("capability needs an id and a public action")
        with self._lock:
            self._caps[cap.capability_id] = cap

    def resolve(self, owner_agent: str, action: str) -> Optional[Capability]:
        cap = self._caps.get(f"{owner_agent}.{action}")
        return cap if cap and cap.owner_agent == owner_agent and cap.action == action else None

    def get(self, capability_id: Optional[str]) -> Optional[Capability]:
        return self._caps.get(capability_id or "")

    def list(self) -> List[Capability]:
        with self._lock:
            return sorted(self._caps.values(), key=lambda c: c.capability_id)
