"""Core vocabulary and request contract for SLAI regulatory governance.

Stdlib-only on purpose: this module (and every mandatory-control module in the
``governance`` package) must import without torch, network access or any other
agent, so mandatory legal controls stay verifiable in isolation.

Design rules
------------
* A ``ProcessingContext`` describes *what is about to happen*. It never carries
  self-asserted legal conclusions: lawful basis is referenced through a
  registered processing activity, never through a free-text string.
* Anything unknown is ``UNCLASSIFIED`` and is treated as personal data until a
  deterministic classifier says otherwise.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "Confidentiality",
    "DataClass",
    "Decision",
    "GovernanceDenied",
    "IdentityContext",
    "LawfulBasis",
    "Operation",
    "ProcessingContext",
    "ProcessingDecision",
    "RuleResult",
    "Art9Condition",
    "Art10Condition",
    "PERSONAL_DATA_CLASSES",
    "SPECIAL_DATA_CLASSES",
    "fingerprint",
    "new_id",
]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def fingerprint(value: Any) -> str:
    """Stable SHA-256 over a JSON-serialisable value (sorted keys, no secrets)."""
    blob = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DataClass(str, Enum):
    """GDPR-oriented classification. Confidentiality is a separate axis."""

    NON_PERSONAL = "non_personal"
    ANONYMISED = "anonymised"  # irreversibly anonymised (Recital 26)
    PERSONAL = "personal"
    PSEUDONYMISED = "pseudonymised"  # still personal data (Art. 4(5), Recital 26)
    SPECIAL_CATEGORY = "special_category"  # Art. 9(1)
    BIOMETRIC = "biometric"  # Art. 9(1) when used for unique identification
    CRIMINAL_OFFENCE = "criminal_offence"  # Art. 10
    UNCLASSIFIED = "unclassified"


class Confidentiality(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECURITY_SENSITIVE = "security_sensitive"


# Classes that are, or must be presumed to be, personal data.
PERSONAL_DATA_CLASSES = frozenset(
    {
        DataClass.PERSONAL,
        DataClass.PSEUDONYMISED,
        DataClass.SPECIAL_CATEGORY,
        DataClass.BIOMETRIC,
        DataClass.CRIMINAL_OFFENCE,
        DataClass.UNCLASSIFIED,
    }
)
SPECIAL_DATA_CLASSES = frozenset({DataClass.SPECIAL_CATEGORY, DataClass.BIOMETRIC})


class LawfulBasis(str, Enum):
    """GDPR Art. 6(1)(a)-(f)."""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class Art9Condition(str, Enum):
    """GDPR Art. 9(2)(a)-(j)."""

    EXPLICIT_CONSENT = "explicit_consent"  # (a)
    EMPLOYMENT_SOCIAL_SECURITY = "employment_social_security"  # (b)
    VITAL_INTERESTS = "vital_interests"  # (c)
    NONPROFIT_MEMBERS = "nonprofit_members"  # (d)
    MANIFESTLY_PUBLIC = "manifestly_made_public"  # (e)
    LEGAL_CLAIMS = "legal_claims"  # (f)
    SUBSTANTIAL_PUBLIC_INTEREST = "substantial_public_interest"  # (g)
    HEALTH_CARE = "health_care"  # (h)
    PUBLIC_HEALTH = "public_health"  # (i)
    RESEARCH_ARCHIVING_STATISTICS = "research_archiving_statistics"  # (j)


class Art10Condition(str, Enum):
    OFFICIAL_AUTHORITY = "official_authority"
    UNION_OR_MEMBER_STATE_LAW = "union_or_member_state_law"


class Operation(str, Enum):
    """Lifecycle stages from the assignment (Collection ... Deletion)."""

    COLLECT = "collect"
    CLASSIFY = "classify"
    PROCESS = "process"
    STORE = "store"
    RETRIEVE = "retrieve"
    SHARE = "share"
    LEARN = "learn"
    RETAIN = "retain"
    RESTRICT = "restrict"
    DELETE = "delete"
    EXPORT = "export"
    TRANSFER = "transfer"
    EXECUTE = "execute"  # invoke a registered capability


class Decision(str, Enum):
    """Interoperable regulatory outcomes (assignment section 8).

    Aggregation takes the *most restrictive* outcome. Mandatory controls always
    outrank learned preferences, optimisation objectives and agent suggestions.

    * ``REDACT``/``RESTRICT`` proceed only on a transformed payload / reduced
      scope; they are NOT plain permissions (see ``permits_execution``).
    """

    ALLOW = "ALLOW"
    ALLOW_WITH_CONTROLS = "ALLOW_WITH_CONTROLS"
    REDACT = "REDACT"
    RESTRICT = "RESTRICT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    QUARANTINE = "QUARANTINE"
    BLOCK = "BLOCK"

    @property
    def severity(self) -> int:
        return _DECISION_SEVERITY[self]

    @property
    def permits_execution(self) -> bool:
        """True only when the request may run exactly as submitted."""
        return self in (Decision.ALLOW, Decision.ALLOW_WITH_CONTROLS)

    @property
    def requires_modification(self) -> bool:
        return self in (Decision.REDACT, Decision.RESTRICT)


_DECISION_SEVERITY = {
    Decision.ALLOW: 0,
    Decision.ALLOW_WITH_CONTROLS: 1,
    Decision.REDACT: 2,
    Decision.RESTRICT: 3,
    Decision.REVIEW_REQUIRED: 4,
    Decision.QUARANTINE: 5,
    Decision.BLOCK: 6,
}


class GovernanceDenied(PermissionError):
    """Raised by ``enforce()`` when processing is not authorised."""

    def __init__(self, decision: "ProcessingDecision"):
        self.decision = decision
        codes = ",".join(r.rule_id for r in decision.blocking_rules) or "none"
        super().__init__(
            f"Governance denied request {decision.request_id}: "
            f"{decision.decision.value} ({codes})"
        )


@dataclass(frozen=True)
class IdentityContext:
    """Authenticated identity. ``verified`` is only ever set by an ``IdentityAuthority``."""

    principal_id: str
    tenant_id: Optional[str]
    verified: bool = False
    actor_type: str = "service"  # "human" | "service" | "agent"
    roles: Tuple[str, ...] = ()
    expires_at: Optional[float] = None
    # Signed assertion this context was derived from. Consumers MUST re-verify the
    # token with the IdentityAuthority; the ``verified`` flag alone is never trusted
    # because any caller can construct a dataclass with verified=True.
    token: Optional[str] = field(default=None, repr=False, compare=False)

    def is_valid(self, now: Optional[float] = None) -> bool:
        if not self.verified or not self.principal_id:
            return False
        if self.expires_at is not None and (now if now is not None else time.time()) >= self.expires_at:
            return False
        return True


@dataclass(frozen=True)
class ProcessingContext:
    """What is about to happen. Immutable; validated on construction."""

    operation: Operation
    purpose: str
    data_class: DataClass = DataClass.UNCLASSIFIED
    classification_verified: bool = False  # True only if set by a deterministic classifier
    confidentiality: Confidentiality = Confidentiality.INTERNAL
    activity_id: Optional[str] = None  # registered ROPA activity (lawful-basis anchor)
    subject_ids: Tuple[str, ...] = ()
    identity: Optional[IdentityContext] = None
    tenant_id: Optional[str] = None  # tenant the resource belongs to
    resource: Optional[str] = None
    capability: Optional[str] = None
    destination_country: Optional[str] = None
    recipients: Tuple[str, ...] = ()
    ai_system_id: Optional[str] = None
    ai_practices: Tuple[str, ...] = ()  # AI Act Art. 5 practice tags observed/declared
    automated_decision: bool = False
    significant_effect: bool = False  # legal or similarly significant effect (GDPR Art. 22)
    subject_is_child: bool = False
    approval_id: Optional[str] = None
    application_id: Optional[str] = None
    agent_id: Optional[str] = None
    controller_id: Optional[str] = None
    processor_id: Optional[str] = None
    provenance_ref: Optional[str] = None
    grant_id: Optional[str] = None  # issued AuthorizationGrant presented by the caller
    request_id: str = field(default_factory=lambda: new_id("gov_req"))
    requested_at: float = field(default_factory=time.time)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, Operation):
            raise TypeError("operation must be an Operation")
        if not isinstance(self.data_class, DataClass):
            raise TypeError("data_class must be a DataClass")
        if not str(self.purpose or "").strip():
            raise ValueError("purpose is required")
        object.__setattr__(self, "purpose", str(self.purpose).strip())
        object.__setattr__(self, "subject_ids", tuple(str(s) for s in self.subject_ids))
        object.__setattr__(self, "recipients", tuple(str(r) for r in self.recipients))
        object.__setattr__(self, "ai_practices", tuple(str(p).lower() for p in self.ai_practices))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @property
    def effective_data_class(self) -> DataClass:
        """Unverified claims of non-personal data are treated as UNCLASSIFIED."""
        if self.data_class in (DataClass.NON_PERSONAL, DataClass.ANONYMISED) and not self.classification_verified:
            return DataClass.UNCLASSIFIED
        return self.data_class

    def fingerprint(self) -> str:
        """Binds approvals to exactly this request (excluding volatile ids/times)."""
        return fingerprint(
            {
                "operation": self.operation.value,
                "purpose": self.purpose,
                "data_class": self.effective_data_class.value,
                "activity_id": self.activity_id,
                "subjects": sorted(self.subject_ids),
                "tenant_id": self.tenant_id,
                "resource": self.resource,
                "capability": self.capability,
                "destination": self.destination_country,
                "recipients": sorted(self.recipients),
                "ai_system_id": self.ai_system_id,
            }
        )


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    decision: Decision
    reason: str
    legal_refs: Tuple[str, ...] = ()
    # Only REVIEW_REQUIRED results may be overridable, and only by a valid human approval.
    overridable: bool = False
    controls: Tuple[str, ...] = ()  # obligations attached to an ALLOW_WITH_CONTROLS outcome

    @property
    def blocking(self) -> bool:
        return not self.decision.permits_execution


@dataclass(frozen=True)
class ProcessingDecision:
    request_id: str
    decision: Decision
    results: Tuple[RuleResult, ...]
    policy_version: str
    context_fingerprint: str
    decided_at: float = field(default_factory=time.time)
    conditions: Tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.decision.permits_execution

    @property
    def blocking_rules(self) -> Tuple[RuleResult, ...]:
        return tuple(r for r in self.results if r.blocking)

    @property
    def requires_modification(self) -> bool:
        return self.decision.requires_modification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "decision": self.decision.value,
            "allowed": self.allowed,
            "policy_version": self.policy_version,
            "context_fingerprint": self.context_fingerprint,
            "decided_at": self.decided_at,
            "conditions": list(self.conditions),
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "decision": r.decision.value,
                    "reason": r.reason,
                    "legal_refs": list(r.legal_refs),
                }
                for r in self.results
            ],
        }
