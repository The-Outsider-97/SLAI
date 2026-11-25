"""Policy applicability and deterministic conflict resolution (GOV-P1-05).

What this module is - and is not
--------------------------------
GDPR, the Dutch UAVG and the EU AI Act apply *cumulatively*: the UAVG supplements the
GDPR through its opening clauses and cannot relax it, and the AI Act expressly leaves
the GDPR untouched (Art. 2(7)). There is therefore no instrument that "wins" and lets the
others be ignored. Conflict resolution here answers two different questions:

1. **Outcome** - always the most restrictive result of every applicable rule (unchanged,
   and enforced as an invariant: ``resolve`` can never return a decision less severe
   than the aggregate; see ``ResolvedDecision.decision``).
2. **Governing rule and path** - which rule *explains* the outcome, which permissions it
   overrode, and what resolves it (authorise / conditional / clarify / remediate /
   review / suspend / deny). Ties are broken by a total order
   ``(severity desc, stratum asc, rule_id, reason)`` so the same inputs always produce the
   same explanation, independent of rule registration order.

Strata (lower governs first)
----------------------------
0 SAFETY_STOP         emergency termination, AI Act Art. 5 prohibitions, internal failure (fail closed)
1 RIGHTS_SAFEGUARD    GDPR/UAVG data-subject protections: lawful basis, purpose, special categories,
                      consent (incl. UAVG art. 5 parental representative), restriction/objection/erasure,
                      transfers, automated decisions, identity/tenant isolation (Art. 32)
2 AI_ACT_REQUIREMENT  AI system profile, risk tier, human oversight (Art. 14), conditional-approval controls
3 OPERATIONAL         SLAI capability scoping and gate plumbing
4 EXTENSION           deployment-registered rules that carry no registered basis (can only restrict)

Applicability is descriptive and self-checking: each rule records when it is applicable.
A result emitted by a rule that is *not applicable* to the request is reported as an
``applicability_anomaly`` (a defect in a rule), never used to relax the outcome. A rule id
without a registered basis is reported and treated as stratum 4; it still restricts.

Legal anchors (verified 2026-11-25 against wetten.overheid.nl / EUR-Lex; counsel must still
confirm interpretation): UAVG art. 5 "Toestemming van wettelijk vertegenwoordiger";
UAVG art. 22-23 special categories and national exceptions; AI Act Art. 113(a) as amended
(new Art. 5(1)(ba),(bb),(1a),(1b) apply from 2026-12-02); AI Act Art. 2(7) (GDPR unaffected).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .processing_context import (
    Decision,
    PERSONAL_DATA_CLASSES,
    ProcessingContext,
    RuleResult,
)

__all__ = [
    "BASES",
    "Conflict",
    "Instrument",
    "PolicyBasis",
    "ResolutionKind",
    "ResolvedDecision",
    "Stratum",
    "applicability_facts",
    "resolve",
]

BASIS_VERSION = "policy-basis-1@2026-11-25"


class Instrument(str, Enum):
    GDPR = "GDPR"
    UAVG = "UAVG"
    AI_ACT = "AI_ACT"
    SLAI = "SLAI"


class Stratum(IntEnum):
    SAFETY_STOP = 0
    RIGHTS_SAFEGUARD = 1
    AI_ACT_REQUIREMENT = 2
    OPERATIONAL = 3
    EXTENSION = 4


class ResolutionKind(str, Enum):
    AUTHORIZE = "authorize"
    AUTHORIZE_WITH_CONDITIONS = "authorize_with_conditions"
    REMEDIATE = "remediate"  # proceed only on a transformed payload / reduced scope
    CLARIFY = "clarify"  # facts are missing: supply them and re-submit (nothing is assumed)
    REVIEW = "review"  # a verified human must decide
    SUSPEND = "suspend"  # hold: a dependency (provider, release gate) is unavailable
    DENY = "deny"


_KIND_OF_DECISION = {
    Decision.ALLOW: ResolutionKind.AUTHORIZE,
    Decision.ALLOW_WITH_CONTROLS: ResolutionKind.AUTHORIZE_WITH_CONDITIONS,
    Decision.REDACT: ResolutionKind.REMEDIATE,
    Decision.RESTRICT: ResolutionKind.REMEDIATE,
    Decision.REVIEW_REQUIRED: ResolutionKind.REVIEW,
    Decision.BLOCK: ResolutionKind.DENY,
}
# QUARANTINE is either "facts missing" (clarify) or "something unavailable" (suspend), decided by reason.
_CLARIFY_REASONS = (
    "unclassified",
    "classification",
    "ai_system_has_no_registered_regulatory_profile",
    "destination_unknown_for_disclosure",
    ":unknown",  # a specialist could not reach a verdict: it needs more facts
)

_APPLICABLE: Dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "always": lambda f: True,
    "personal_data": lambda f: bool(f["personal"]),
    "ai_system": lambda f: bool(f["ai_system"]),
    "protected": lambda f: bool(f["protected"]),
    "transfer_like": lambda f: bool(f["transfer_like"]),
    "consent_like": lambda f: bool(f["personal"]),
    "automated_decision": lambda f: bool(f["automated_decision"]) or bool(f["personal"]),
}


@dataclass(frozen=True)
class PolicyBasis:
    rule_id: str
    instruments: Tuple[Instrument, ...]
    stratum: Stratum
    applies_when: str  # key of _APPLICABLE
    description: str
    steps: Tuple[str, ...] = ()  # what resolves a restricting result of this rule

    def applicable(self, facts: Mapping[str, Any]) -> bool:
        return _APPLICABLE[self.applies_when](facts)


def _b(rid, instruments, stratum, applies, desc, *steps) -> PolicyBasis:
    return PolicyBasis(rid, tuple(instruments), stratum, applies, desc, tuple(steps))


_G, _U, _A, _S = Instrument.GDPR, Instrument.UAVG, Instrument.AI_ACT, Instrument.SLAI

BASES: Mapping[str, PolicyBasis] = {
    b.rule_id: b
    for b in (
        _b("GOV-EMG-15", (_A,), Stratum.SAFETY_STOP, "always", "Emergency termination (AI Act Art. 14(4)(e))", "lift the halt through an authorised human oversight action"),
        _b("GOV-AI5-03", (_A,), Stratum.SAFETY_STOP, "ai_system", "Prohibited practices (AI Act Art. 5); not overridable"),
        _b("GOV-ERR-99", (_S, _G), Stratum.SAFETY_STOP, "always", "Evaluation failure fails closed (GDPR Art. 25)", "investigate the failing rule; re-submit after the defect is fixed"),
        _b("GOV-GATE-ERR-01", (_S, _G), Stratum.SAFETY_STOP, "always", "Gate failure fails closed", "investigate the gate error"),
        _b("GOV-ID-01", (_G,), Stratum.RIGHTS_SAFEGUARD, "protected", "Authenticated, tenant-bound identity (GDPR Art. 5(1)(f), 32)", "present a valid, unexpired identity token for the right tenant"),
        _b("GOV-CLS-02", (_G,), Stratum.RIGHTS_SAFEGUARD, "always", "Unverified classification is not trusted (GDPR Art. 4(1), 5(2))", "obtain signed classification evidence from the classification authority"),
        _b("GOV-LAW-04", (_G,), Stratum.RIGHTS_SAFEGUARD, "personal_data", "Lawful basis from a registered activity (GDPR Art. 6, 30)", "register an active processing activity covering this processing"),
        _b("GOV-PUR-05", (_G,), Stratum.RIGHTS_SAFEGUARD, "personal_data", "Purpose limitation (GDPR Art. 5(1)(b), 6(4))", "use a purpose, operation and data class registered for the activity, or register a compatible activity"),
        _b("GOV-SPC-06", (_G, _U), Stratum.RIGHTS_SAFEGUARD, "personal_data", "Art. 9/10 condition (UAVG art. 22-23 national exceptions)", "document the Art. 9(2)/10 condition on the registered activity"),
        _b("GOV-CON-07", (_G, _U), Stratum.RIGHTS_SAFEGUARD, "consent_like", "Consent incl. child (GDPR Art. 7-8; UAVG art. 5 representative)", "obtain valid, specific consent (parental authorisation for children)"),
        _b("GOV-RTS-08", (_G,), Stratum.RIGHTS_SAFEGUARD, "personal_data", "Restriction, objection and erasure (GDPR Art. 17, 18, 21)", "wait for the restriction/objection to end or choose a different purpose"),
        _b("GOV-TRF-09", (_G,), Stratum.RIGHTS_SAFEGUARD, "transfer_like", "Third-country transfers (GDPR Art. 44-49)", "register a transfer mechanism and, where required, supplementary measures"),
        _b("GOV-ADM-10", (_G, _A), Stratum.RIGHTS_SAFEGUARD, "automated_decision", "Automated decisions (GDPR Art. 22; AI Act Art. 14)", "add genuine human intervention or a valid Art. 22(2) exception"),
        _b("GOV-LRN-11", (_G,), Stratum.RIGHTS_SAFEGUARD, "personal_data", "Learning on personal data (GDPR Art. 5(1)(b), 25)", "register an activity that authorises learning for this purpose"),
        _b("GOV-AIS-12", (_A,), Stratum.AI_ACT_REQUIREMENT, "ai_system", "AI system regulatory profile (AI Act Art. 6, 49)", "register an AI system profile; pass the release gate where high-risk obligations apply"),
        _b("GOV-HO-14", (_A, _G), Stratum.AI_ACT_REQUIREMENT, "always", "Human oversight (AI Act Art. 14; GDPR Art. 22(3))", "obtain approval from a verified human bound to this exact request"),
        _b("GOV-CND-01", (_S,), Stratum.AI_ACT_REQUIREMENT, "always", "Unknown control attached to an authorisation", "register the condition in the catalogue"),
        _b("GOV-CND-02", (_S,), Stratum.AI_ACT_REQUIREMENT, "always", "Mandatory pre-condition unmet", "fulfil the pre-condition with signed evidence or an attestation"),
        _b("GOV-CND-03", (_S,), Stratum.AI_ACT_REQUIREMENT, "always", "Runtime condition unmet"),
        _b("GOV-CND-04", (_S,), Stratum.AI_ACT_REQUIREMENT, "always", "Release condition unmet"),
        _b("GOV-COORD-01", (_A, _S), Stratum.AI_ACT_REQUIREMENT, "always", "Specialist agent reported a violation", "resolve the specialist finding and re-submit"),
        _b("GOV-COORD-02", (_A, _S), Stratum.AI_ACT_REQUIREMENT, "always", "Specialist agent raised a concern: human review (AI Act Art. 14)", "obtain approval from a verified human who has seen the specialist finding"),
        _b("GOV-COORD-03", (_A, _S), Stratum.AI_ACT_REQUIREMENT, "always", "A required specialist could not report", "restore the specialist and re-submit"),
        _b("GOV-COORD-04", (_A, _S), Stratum.AI_ACT_REQUIREMENT, "always", "Specialist could not reach a verdict", "supply the facts the specialist needs and re-submit"),
        _b("GOV-COORD-05", (_A, _S), Stratum.AI_ACT_REQUIREMENT, "always", "Coordination evidence is not bound to this request", "re-run coordination for this exact request"),
        _b("GOV-ADP-01", (_S,), Stratum.OPERATIONAL, "always", "Adaptive advisory escalation (can only add human review)", "obtain approval from a verified human, or ask governance to roll the adaptive version back"),
        _b("GOV-CAP-13", (_S, _G), Stratum.OPERATIONAL, "always", "Registered, scoped capability (GDPR Art. 25, 32)", "register the capability or reduce the data class to its scope"),
    )
}


def applicability_facts(ctx: ProcessingContext) -> Dict[str, Any]:
    """Facts derived from the *resolved* context only (never from caller-asserted booleans)."""
    personal = ctx.effective_data_class in PERSONAL_DATA_CLASSES
    return {
        "personal": personal,
        "ai_system": bool(ctx.ai_system_id) or bool(ctx.ai_practices),
        "protected": bool(ctx.tenant_id) or personal,
        "transfer_like": bool(ctx.destination_country) or bool(ctx.recipients),
        "automated_decision": bool(ctx.automated_decision),
    }


@dataclass(frozen=True)
class Conflict:
    """A permission that was overruled by a stricter result."""

    permitting_rule: str
    restricting_rule: str
    governs: str  # always the restricting rule
    why: str


@dataclass(frozen=True)
class ResolvedDecision:
    decision: Decision
    kind: ResolutionKind
    governing: Tuple[str, ...]  # rule ids that determine the outcome, in priority order
    stratum: int
    instruments: Tuple[str, ...]
    steps: Tuple[str, ...]
    conflicts: Tuple[Conflict, ...]
    unregistered_rules: Tuple[str, ...]
    applicability_anomalies: Tuple[str, ...]
    basis_version: str = BASIS_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "kind": self.kind.value,
            "governing": list(self.governing),
            "stratum": self.stratum,
            "instruments": list(self.instruments),
            "steps": list(self.steps),
            "conflicts": [c.__dict__ for c in self.conflicts],
            "unregistered_rules": list(self.unregistered_rules),
            "applicability_anomalies": list(self.applicability_anomalies),
            "basis_version": self.basis_version,
        }


def _basis(rule_id: str) -> Optional[PolicyBasis]:
    return BASES.get(rule_id)


def _stratum(rule_id: str) -> int:
    b = _basis(rule_id)
    return int(b.stratum) if b else int(Stratum.EXTENSION)


def _kind(decision: Decision, governing: Sequence[RuleResult]) -> ResolutionKind:
    if decision is Decision.QUARANTINE:
        reasons = " ".join(r.reason for r in governing)
        return ResolutionKind.CLARIFY if any(tok in reasons for tok in _CLARIFY_REASONS) else ResolutionKind.SUSPEND
    return _KIND_OF_DECISION[decision]


def resolve(
    decision: Decision,
    results: Iterable[RuleResult],
    facts: Optional[Mapping[str, Any]] = None,
) -> ResolvedDecision:
    """Explain ``decision`` (the aggregate of ``results``) deterministically. Never lowers it.

    ``decision`` is what the engine computed. If it is somehow less severe than the most
    restrictive result, the stricter one is returned (fail closed) - the resolver can only
    ever tighten.
    """
    rs: List[RuleResult] = list(results)
    top = max((r.decision for r in rs), key=lambda d: d.severity, default=Decision.ALLOW)
    final = top if top.severity > decision.severity else decision

    ordered = sorted(rs, key=lambda r: (-r.decision.severity, _stratum(r.rule_id), r.rule_id, r.reason))
    governing_results = [r for r in ordered if r.decision is final] if final.severity > Decision.ALLOW.severity else []
    # A governing rule is one that produced the final severity; among those the highest stratum (lowest number) leads.
    governing_ids = tuple(dict.fromkeys(r.rule_id for r in governing_results))

    steps: List[str] = []
    instruments: List[str] = []
    for rid in governing_ids:
        b = _basis(rid)
        if b is None:
            continue
        steps.extend(s for s in b.steps if s not in steps)
        instruments.extend(i.value for i in b.instruments if i.value not in instruments)

    conflicts: List[Conflict] = []
    if governing_results and final.severity > Decision.ALLOW_WITH_CONTROLS.severity:
        lead = governing_results[0].rule_id
        for r in ordered:
            if r.decision.permits_execution:
                conflicts.append(Conflict(r.rule_id, lead, lead, f"{final.value} outranks {r.decision.value} (most restrictive result governs)"))

    unregistered = tuple(sorted({r.rule_id for r in rs if _basis(r.rule_id) is None}))
    anomalies: List[str] = []
    if facts is not None:
        for r in rs:
            b = _basis(r.rule_id)
            if b is not None and not b.applicable(facts):
                anomalies.append(f"{r.rule_id}:not_applicable_but_emitted:{b.applies_when}")

    stratum = min((_stratum(rid) for rid in governing_ids), default=int(Stratum.EXTENSION))
    return ResolvedDecision(
        decision=final,
        kind=_kind(final, governing_results),
        governing=governing_ids,
        stratum=stratum,
        instruments=tuple(instruments),
        steps=tuple(steps),
        conflicts=tuple(conflicts),
        unregistered_rules=unregistered,
        applicability_anomalies=tuple(sorted(set(anomalies))),
    )
