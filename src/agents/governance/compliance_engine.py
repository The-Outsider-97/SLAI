"""Evidence-driven compliance assessment and release gates.

Contract (assignment 4.10, 15, 18):
* There is NO numeric compliance score and nothing is averaged. A control is in
  exactly one status; a failed or unproven mandatory control blocks the gate.
* Missing evidence is never compliance: no record => ``MISSING``.
* Evidence must be fresh, must have a passing latest result, and where a control
  demands independence it must have been produced by an identity outside the
  Governance Agent (e.g. the Verification Agent or CI).
* A tampered evidence chain invalidates every control.
* Organisational duties (DPO, DPIA, notices, DPAs, FRIA, conformity ...) cannot be
  satisfied by software: without evidence they are ``DEPENDS_ON_ORGANISATION``.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from .regulatory_evidence import EvidenceRecord, EvidenceStore

__all__ = [
    "Assessment",
    "ComplianceEngine",
    "Control",
    "ControlResult",
    "ControlStatus",
    "DeploymentProfile",
    "EvidenceRequirement",
    "DEFAULT_CONTROLS",
]

DAY = 86400.0


class ControlStatus(str, Enum):
    VERIFIED = "implemented_and_verified"
    UNVERIFIED = "implemented_not_fully_verified"
    FAILED = "failed"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    DEPENDS_ON_DEPLOYMENT = "dependent_on_deployment"
    DEPENDS_ON_ORGANISATION = "dependent_on_organisational_action"


@dataclass(frozen=True)
class DeploymentProfile:
    """Facts about a specific deployment; they decide applicability, nothing else."""

    name: str = "default"
    processes_personal_data: bool = True
    uses_processors: bool = False
    transfers_outside_eea: bool = False
    high_risk_ai_in_scope: bool = False
    gpai_provider: bool = False
    public_authority: bool = False
    dpo_required: bool = False
    regulated_decisions_about_persons: bool = False
    production: bool = False
    justification: Mapping[str, str] = field(default_factory=dict)  # control_id -> reason for N/A


@dataclass(frozen=True)
class EvidenceRequirement:
    evidence_type: str
    max_age_days: float = 90.0
    independent: bool = False


@dataclass(frozen=True)
class Control:
    control_id: str
    title: str
    legal_refs: Tuple[str, ...]
    requirements: Tuple[EvidenceRequirement, ...]
    category: str = "technical"  # technical | organisational
    mandatory: bool = True
    applies: Callable[[DeploymentProfile], bool] = lambda p: True


@dataclass(frozen=True)
class ControlResult:
    control: Control
    status: ControlStatus
    findings: Tuple[str, ...]
    evidence_seqs: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_id": self.control.control_id,
            "title": self.control.title,
            "category": self.control.category,
            "mandatory": self.control.mandatory,
            "legal_refs": list(self.control.legal_refs),
            "status": self.status.value,
            "findings": list(self.findings),
            "evidence_seqs": list(self.evidence_seqs),
        }


@dataclass(frozen=True)
class Assessment:
    profile: str
    generated_at: float
    chain_ok: bool
    results: Tuple[ControlResult, ...]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.results:
            out[r.status.value] = out.get(r.status.value, 0) + 1
        return out

    def blocking(self, *, include_organisational: bool) -> List[ControlResult]:
        ok = {ControlStatus.VERIFIED, ControlStatus.NOT_APPLICABLE}
        out = []
        for r in self.results:
            if not r.control.mandatory or r.status in ok:
                continue
            if r.control.category == "organisational" and not include_organisational:
                continue
            out.append(r)
        return out

    def release_allowed(self, *, include_organisational: bool = False) -> bool:
        return self.chain_ok and not self.blocking(include_organisational=include_organisational)


def _tech(cid, title, refs, *reqs, mandatory=True, applies=lambda p: True):
    return Control(cid, title, tuple(refs), tuple(reqs), "technical", mandatory, applies)


def _org(cid, title, refs, *reqs, applies=lambda p: True):
    return Control(cid, title, tuple(refs), tuple(reqs), "organisational", True, applies)


R = EvidenceRequirement
DEFAULT_CONTROLS: Tuple[Control, ...] = (
    _tech("TC-LAW-01", "Lawful-processing authorisation enforced and non-bypassable", ("GDPR Art. 5-6, 25",), R("test_run", 30, True)),
    _tech("TC-SPC-02", "Special-category/criminal-data conditions enforced", ("GDPR Art. 9-10",), R("test_run", 30, True)),
    _tech("TC-CON-03", "Consent validity and withdrawal enforced", ("GDPR Art. 7-8",), R("test_run", 30, True)),
    _tech("TC-DEL-04", "Erasure verified by storage-system evidence", ("GDPR Art. 17, 19",), R("deletion_verification", 90, True), R("test_run", 30, True)),
    _tech("TC-ISO-05", "Tenant/principal isolation with authenticated identity", ("GDPR Art. 5(1)(f), 32",), R("isolation_test", 30, True)),
    _tech("TC-SEC-06", "Persistent-memory encryption, integrity and safe deserialisation", ("GDPR Art. 32",), R("test_run", 30, True)),
    _tech("TC-LOG-07", "Diagnostics are free of unauthorised personal data", ("GDPR Art. 5(1)(c), 25, 32",), R("log_sanitisation_test", 30, True)),
    _tech("TC-DEL-08", "Delegation and execution restricted to registered capabilities", ("GDPR Art. 25, 32",), R("test_run", 30, True)),
    _tech("TC-AI5-09", "AI Act Art. 5 prohibited practices blocked pre-execution", ("AI Act Art. 5",), R("test_run", 30, True)),
    _tech("TC-OVS-10", "Human oversight: human-only, scoped, expiring approvals; emergency stop", ("AI Act Art. 14", "GDPR Art. 22"), R("test_run", 30, True)),
    _tech("TC-POL-11", "Mandatory policy set integrity and change control", ("GDPR Art. 24",), R("policy_integrity", 30, True)),
    _tech("TC-TRF-12", "Third-country transfer controls", ("GDPR Art. 44-49",), R("test_run", 30, True), applies=lambda p: p.processes_personal_data),
    _tech("TC-TRN-13", "Art. 50 transparency: disclosure and content marking", ("AI Act Art. 50",), R("test_run", 30, True)),
    _tech("TC-LRN-14", "Governed learning: eligibility, provenance, independent promotion gate", ("GDPR Art. 5-6, 25", "AI Act Art. 10"), R("training_eligibility", 90, True), R("model_evaluation", 90, True)),
    _tech("TC-INC-15", "Incident lifecycle with Art. 33/34 and AI Act Art. 73 deadline tracking", ("GDPR Art. 33-34", "AI Act Art. 73"), R("test_run", 30, True)),
    _tech("TC-HR-16", "High-risk readiness: risk management, logging, documentation, accuracy", ("AI Act Art. 9-15",), R("ai_system_documentation", 180), R("test_run", 30, True), applies=lambda p: p.high_risk_ai_in_scope),
    _tech("TC-GPA-17", "GPAI documentation, copyright policy, training summary", ("AI Act Art. 53-55",), R("model_documentation", 180), applies=lambda p: p.gpai_provider),
    _org("ORG-ROPA", "Records of processing activities maintained", ("GDPR Art. 30",), R("ropa", 365), applies=lambda p: p.processes_personal_data),
    _org("ORG-NOTICE", "Privacy notices published", ("GDPR Art. 12-14",), R("privacy_notice", 365), applies=lambda p: p.processes_personal_data),
    _org("ORG-DPA", "Processor agreements in place", ("GDPR Art. 28",), R("dpa_contract", 365), applies=lambda p: p.uses_processors),
    _org("ORG-DPO", "Data Protection Officer appointed", ("GDPR Art. 37-39",), R("dpo_appointment", 365), applies=lambda p: p.dpo_required or p.public_authority),
    _org("ORG-DPIA", "DPIA performed where required", ("GDPR Art. 35-36",), R("dpia", 365), applies=lambda p: p.regulated_decisions_about_persons or p.high_risk_ai_in_scope),
    _org("ORG-FRIA", "Fundamental-rights impact assessment where required", ("AI Act Art. 27",), R("fria", 365), applies=lambda p: p.high_risk_ai_in_scope and p.public_authority),
    _org("ORG-LIT", "AI literacy of operators", ("AI Act Art. 4",), R("ai_literacy", 365)),
    _org("ORG-CONF", "Conformity assessment and registration", ("AI Act Art. 43, 49",), R("release_conformity", 365), applies=lambda p: p.high_risk_ai_in_scope),
    _org("ORG-TIA", "Transfer impact assessments / transfer records", ("GDPR Art. 44-49",), R("transfer_record", 365), applies=lambda p: p.transfers_outside_eea),
)


class ComplianceEngine:
    def __init__(
        self,
        store: EvidenceStore,
        *,
        independent_producers: Iterable[str] = ("verification_agent", "ci_pipeline"),
        controls: Sequence[Control] = DEFAULT_CONTROLS,
    ):
        self._store = store
        self._independent = frozenset(independent_producers)
        self._controls = tuple(controls)

    def assess(self, profile: DeploymentProfile, *, now: Optional[float] = None) -> Assessment:
        t = time.time() if now is None else now
        chain_ok, bad = self._store.verify_chain()
        results = [self._assess_control(c, profile, t, chain_ok) for c in self._controls]
        return Assessment(profile.name, t, chain_ok, tuple(results))

    def _assess_control(self, c: Control, p: DeploymentProfile, now: float, chain_ok: bool) -> ControlResult:
        if not c.applies(p):
            reason = p.justification.get(c.control_id, "not triggered by deployment profile")
            return ControlResult(c, ControlStatus.NOT_APPLICABLE, (f"n/a: {reason}",))
        if not chain_ok:
            return ControlResult(c, ControlStatus.UNVERIFIED, ("evidence chain failed integrity verification",))
        records = self._store.records(c.control_id)
        findings: List[str] = []
        seqs: List[int] = []
        worst: Optional[ControlStatus] = None
        for req in c.requirements:
            matching = [r for r in records if r.evidence_type == req.evidence_type]
            if not matching:
                findings.append(f"no '{req.evidence_type}' evidence")
                worst = _worse(worst, ControlStatus.DEPENDS_ON_ORGANISATION if c.category == "organisational" else ControlStatus.MISSING)
                continue
            latest = matching[-1]
            seqs.append(latest.seq)
            if latest.result != "pass":
                findings.append(f"latest '{req.evidence_type}' evidence is '{latest.result}' (seq {latest.seq})")
                worst = _worse(worst, ControlStatus.FAILED)
            elif latest.age(now) > req.max_age_days * DAY:
                findings.append(f"'{req.evidence_type}' evidence is stale ({latest.age(now) / DAY:.0f}d > {req.max_age_days:.0f}d)")
                worst = _worse(worst, ControlStatus.UNVERIFIED)
            elif req.independent and latest.producer not in self._independent:
                findings.append(f"'{req.evidence_type}' not produced by an independent verifier (producer={latest.producer})")
                worst = _worse(worst, ControlStatus.UNVERIFIED)
        if worst is None:
            return ControlResult(c, ControlStatus.VERIFIED, ("all evidence requirements met",), tuple(seqs))
        return ControlResult(c, worst, tuple(findings), tuple(seqs))

    def release_gate(self, profile: DeploymentProfile, *, include_organisational: bool = False, now: Optional[float] = None) -> Tuple[bool, List[str]]:
        a = self.assess(profile, now=now)
        reasons = [f"{r.control.control_id}:{r.status.value}" for r in a.blocking(include_organisational=include_organisational)]
        if not a.chain_ok:
            reasons.insert(0, "evidence_chain_integrity_failed")
        return a.release_allowed(include_organisational=include_organisational), reasons


_SEVERITY = [
    ControlStatus.FAILED,
    ControlStatus.MISSING,
    ControlStatus.DEPENDS_ON_ORGANISATION,
    ControlStatus.UNVERIFIED,
]


def _worse(a: Optional[ControlStatus], b: ControlStatus) -> ControlStatus:
    if a is None:
        return b
    return a if _SEVERITY.index(a) <= _SEVERITY.index(b) else b
