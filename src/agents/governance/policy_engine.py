"""Deterministic, fail-closed evaluation of mandatory regulatory controls.

Every rule is plain code over registered state; no learned component participates.
Aggregation is "most restrictive wins", so extra/advisory rules (analytics,
learned classifiers, agent suggestions) can add restrictions but can never relax
a mandatory outcome. A rule that raises is itself a BLOCK (GOV-ERR-99).

Only ``REVIEW_REQUIRED`` results are overridable, and only by a valid, request-bound
approval from a verified human (see ``human_oversight``). Prohibited practices,
missing lawful basis, tenant violations and the like are never overridable.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence

from .human_oversight import OversightManager, OversightLevel, required_oversight_level
from .lawful_basis import ConsentProvider, RestrictionProvider
from .policy_registry import CapabilityRegistry, PolicyRegistry
from .processing_context import (
    Art9Condition,
    DataClass,
    Decision,
    IdentityContext,
    LawfulBasis,
    Operation,
    PERSONAL_DATA_CLASSES,
    ProcessingContext,
    ProcessingDecision,
    RuleResult,
    SPECIAL_DATA_CLASSES,
)
from .processing_registry import ProcessingActivity, ProcessingRegistry
from .risk_classification import PROHIBITED_PRACTICES, ProfileRegistry, RiskTier
from .utils.identity import IdentityAuthority, IdentityVerificationError

__all__ = ["PolicyEngine", "TransferPolicy"]

EEA = frozenset(
    "AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE IS LI NO".split()
)


@dataclass(frozen=True)
class TransferPolicy:
    """Adequacy decisions change; this list is data, not law. Keep it under change control.

    Last reviewed 2026-11-24 from the Commission's published list as remembered by the
    implementer - **verify against the current Commission list before relying on it**.
    US is only adequate for recipients certified under the EU-US Data Privacy Framework
    (the activity's transfer entry must say ``dpf_certified: true``). Canada is adequate
    for commercial organisations only.
    """

    adequate_countries: FrozenSet[str] = frozenset("AD AR CA FO GG IL IM JP JE NZ KR CH GB UY US".split())
    certified_only: FrozenSet[str] = frozenset({"US"})


_PROTECTED_OPS = frozenset({Operation.DELETE, Operation.EXPORT, Operation.SHARE, Operation.TRANSFER, Operation.LEARN, Operation.EXECUTE})
_RESTRICTION_EXEMPT_OPS = frozenset({Operation.STORE, Operation.DELETE, Operation.RESTRICT, Operation.CLASSIFY})
_NO_ERASURE_PENDING_OPS = frozenset({Operation.PROCESS, Operation.LEARN, Operation.SHARE, Operation.EXPORT, Operation.TRANSFER, Operation.RETRIEVE})
Rule = Callable[[ProcessingContext, "_Env"], List[RuleResult]]


@dataclass
class _Env:
    now: float
    identity: Optional[IdentityContext]
    activity: Optional[ProcessingActivity]


def _r(rule_id: str, decision: Decision, reason: str, *refs: str, overridable: bool = False, controls: Sequence[str] = ()) -> RuleResult:
    return RuleResult(rule_id, decision, reason, tuple(refs), overridable, tuple(controls))


class PolicyEngine:
    def __init__(
        self,
        *,
        policies: PolicyRegistry,
        activities: ProcessingRegistry,
        capabilities: CapabilityRegistry,
        profiles: ProfileRegistry,
        oversight: OversightManager,
        identity: IdentityAuthority,
        consent: Optional[ConsentProvider] = None,
        restrictions: Optional[RestrictionProvider] = None,
        transfers: Optional[TransferPolicy] = None,
        release_gate: Optional[Callable[[str], bool]] = None,
        extra_rules: Sequence[Rule] = (),
        clock: Callable[[], float] = time.time,
    ):
        self.policies = policies
        self.activities = activities
        self.capabilities = capabilities
        self.profiles = profiles
        self.oversight = oversight
        self.identity = identity
        self.consent = consent
        self.restrictions = restrictions
        self.transfers = transfers or TransferPolicy()
        self.release_gate = release_gate
        self._extra = tuple(extra_rules)
        self._clock = clock
        self._rules: Sequence[Rule] = (
            self._emergency,
            self._identity_rule,
            self._classification,
            self._prohibited,
            self._ai_system,
            self._capability,
            self._lawful,
            self._rights_state,
            self._art22,
            self._transfer,
            self._oversight_rule,
        ) + self._extra

    # ------------------------------------------------------------------
    def evaluate(self, ctx: ProcessingContext) -> ProcessingDecision:
        now = self._clock()
        env = _Env(now, self._authenticate(ctx, now), self.activities.get(ctx.activity_id))
        results: List[RuleResult] = []
        for rule in self._rules:
            try:
                results.extend(rule(ctx, env))
            except Exception as exc:  # fail closed, never leak payload content
                results.append(_r("GOV-ERR-99", Decision.BLOCK, f"rule_error:{type(exc).__name__}", "GDPR Art. 25"))
        results = self._apply_human_override(ctx, results, now)
        decision = max((r.decision for r in results), key=lambda d: d.severity, default=Decision.ALLOW)
        controls = tuple(dict.fromkeys(c for r in results for c in r.controls))
        return ProcessingDecision(
            request_id=ctx.request_id,
            decision=decision,
            results=tuple(results),
            policy_version=self.policies.active_version(),
            context_fingerprint=ctx.fingerprint(),
            decided_at=now,
            conditions=controls,
        )

    # ------------------------------------------------------------------ helpers
    def _authenticate(self, ctx: ProcessingContext, now: float) -> Optional[IdentityContext]:
        ident = ctx.identity
        if ident is None or not ident.token:
            return None
        try:
            verified = self.identity.verify(ident.token, now=now)
        except IdentityVerificationError:
            return None
        return verified if verified.principal_id == ident.principal_id else None

    def _personal(self, ctx: ProcessingContext) -> bool:
        return ctx.effective_data_class in PERSONAL_DATA_CLASSES

    def _classified_personal(self, ctx: ProcessingContext) -> bool:
        return self._personal(ctx) and ctx.effective_data_class is not DataClass.UNCLASSIFIED

    @staticmethod
    def _covers(activity: ProcessingActivity, cls: DataClass) -> bool:
        cats = set(activity.data_categories)
        return cls in cats or (cls is DataClass.PSEUDONYMISED and DataClass.PERSONAL in cats)

    # ------------------------------------------------------------------ rules
    def _emergency(self, ctx, env):
        scope = self.oversight.halted_scope(ctx)
        return [_r("GOV-EMG-15", Decision.BLOCK, f"emergency_termination_active:{scope}", "AI Act Art. 14(4)(e)")] if scope else []

    def _identity_rule(self, ctx, env):
        protected = bool(ctx.tenant_id) or self._personal(ctx) or ctx.operation in _PROTECTED_OPS
        if not protected:
            return []
        if env.identity is None:
            return [_r("GOV-ID-01", Decision.BLOCK, "identity_missing_forged_expired_or_unverifiable", "GDPR Art. 32")]
        if ctx.tenant_id and env.identity.tenant_id != ctx.tenant_id:
            return [_r("GOV-ID-01", Decision.BLOCK, "tenant_mismatch", "GDPR Art. 5(1)(f)")]
        return []

    def _classification(self, ctx, env):
        if ctx.effective_data_class is DataClass.UNCLASSIFIED:
            return [_r("GOV-CLS-02", Decision.QUARANTINE, "unclassified_or_unverified_non_personal_claim", "GDPR Art. 4(1)")]
        return []

    def _prohibited(self, ctx, env):
        out: List[RuleResult] = []
        for tag in ctx.ai_practices:
            if tag not in PROHIBITED_PRACTICES:
                out.append(_r("GOV-AI5-03", Decision.QUARANTINE, f"unrecognised_practice_tag:{tag}", "AI Act Art. 5"))
            elif not self.profiles.practice_permitted(ctx.ai_system_id, tag):
                out.append(_r("GOV-AI5-03", Decision.BLOCK, f"prohibited_practice:{tag}", f"AI Act {PROHIBITED_PRACTICES[tag][0]}"))
        cls = self.profiles.classification(ctx.ai_system_id)
        if cls is not None and cls.tier is RiskTier.PROHIBITED:
            out.append(_r("GOV-AI5-03", Decision.BLOCK, "ai_system_classified_prohibited", "AI Act Art. 5"))
        return out

    def _ai_system(self, ctx, env):
        if not ctx.ai_system_id:
            return []
        cls = self.profiles.classification(ctx.ai_system_id)
        if cls is None:
            return [_r("GOV-AIS-12", Decision.QUARANTINE, "ai_system_has_no_registered_regulatory_profile", "AI Act Art. 6")]
        controls = [f"open_question:{q}" for q in cls.open_questions]
        if cls.tier is RiskTier.HIGH_RISK:
            today = datetime.fromtimestamp(env.now, tz=timezone.utc).date()
            due = [o for o in cls.obligations if o.obligation_id.startswith("HR-") and o.status(today) == "applicable"]
            if due and not (self.release_gate and self.release_gate(ctx.ai_system_id)):
                return [_r("GOV-AIS-12", Decision.QUARANTINE, "high_risk_obligations_applicable_and_release_gate_not_passed", "AI Act Art. 8-15, 43")]
            controls.append("high_risk_readiness_controls")
        if cls.tier is RiskTier.TRANSPARENCY:
            controls.append("art50_transparency_obligations")
        return [_r("GOV-AIS-12", Decision.ALLOW_WITH_CONTROLS, f"ai_tier:{cls.tier.value}", "AI Act Art. 6", controls=controls)] if controls else []

    def _capability(self, ctx, env):
        if ctx.operation is not Operation.EXECUTE and not ctx.capability:
            return []
        cap = self.capabilities.get(ctx.capability)
        if cap is None:
            return [_r("GOV-CAP-13", Decision.BLOCK, "capability_not_registered", "GDPR Art. 25")]
        out = []
        if ctx.effective_data_class not in cap.allowed_data_classes:
            out.append(_r("GOV-CAP-13", Decision.BLOCK, "data_class_exceeds_capability_scope", "GDPR Art. 25"))
        if cap.tenant_bound and not ctx.tenant_id:
            out.append(_r("GOV-CAP-13", Decision.BLOCK, "tenant_bound_capability_without_tenant", "GDPR Art. 32"))
        return out

    def _lawful(self, ctx, env):
        if not self._classified_personal(ctx):
            return []
        a = env.activity
        if a is None:
            return [_r("GOV-LAW-04", Decision.BLOCK, "no_registered_processing_activity", "GDPR Art. 6, 30")]
        if not a.is_active(env.now):
            return [_r("GOV-LAW-04", Decision.BLOCK, "processing_activity_not_active", "GDPR Art. 6, 30")]
        out: List[RuleResult] = []
        cls = ctx.effective_data_class
        if ctx.purpose not in a.purposes:
            out.append(_r("GOV-PUR-05", Decision.BLOCK, "purpose_not_in_activity", "GDPR Art. 5(1)(b)"))
        if ctx.operation not in a.operations and ctx.operation is not Operation.CLASSIFY:
            out.append(_r("GOV-PUR-05", Decision.BLOCK, "operation_not_in_activity", "GDPR Art. 5(1)(b)"))
        if not self._covers(a, cls):
            out.append(_r("GOV-PUR-05", Decision.BLOCK, "data_class_not_in_activity", "GDPR Art. 5(1)(c)"))
        if cls in SPECIAL_DATA_CLASSES and a.art9_condition is None:
            out.append(_r("GOV-SPC-06", Decision.BLOCK, "special_category_without_art9_condition", "GDPR Art. 9"))
        if cls is DataClass.CRIMINAL_OFFENCE and a.art10_condition is None:
            out.append(_r("GOV-SPC-06", Decision.BLOCK, "criminal_data_without_art10_condition", "GDPR Art. 10"))
        if ctx.operation is Operation.LEARN and not a.allows_learning:
            out.append(_r("GOV-LRN-11", Decision.BLOCK, "activity_does_not_authorise_learning", "GDPR Art. 5(1)(b), 6(4)"))
        out.extend(self._consent_rules(ctx, a))
        if ctx.recipients and not set(ctx.recipients) <= set(a.recipients):
            out.append(_r("GOV-TRF-09", Decision.BLOCK, "recipient_not_registered_for_activity", "GDPR Art. 28, 30(1)(d)"))
        return out

    @staticmethod
    def _tenant_kw(provider: Any, ctx: ProcessingContext) -> Dict[str, Any]:
        """Tenant-aware providers (the PrivacyLedger) always receive the caller's tenant; a missing
        tenant raises, which the engine converts into a fail-closed BLOCK (GOV-ERR-99)."""
        if not getattr(provider, "tenant_aware", False):
            return {}
        # ctx.tenant_id is cross-checked against the VERIFIED identity by the identity rule (personal data is always protected).
        tenant = ctx.tenant_id
        if not tenant:
            raise ValueError("tenant_id required for tenant-scoped privacy state")
        return {"tenant_id": tenant}

    def _consent_rules(self, ctx, a: ProcessingActivity) -> List[RuleResult]:
        needs = a.basis is LawfulBasis.CONSENT or a.art9_condition is Art9Condition.EXPLICIT_CONSENT
        if not needs:
            return []
        if not ctx.subject_ids:
            return [_r("GOV-CON-07", Decision.BLOCK, "consent_processing_requires_identified_subjects", "GDPR Art. 7")]
        if self.consent is None:
            return [_r("GOV-CON-07", Decision.QUARANTINE, "consent_provider_unavailable", "GDPR Art. 7")]
        out: List[RuleResult] = []
        explicit = a.art9_condition is Art9Condition.EXPLICIT_CONSENT and ctx.effective_data_class in SPECIAL_DATA_CLASSES
        for sid in ctx.subject_ids:
            snap = self.consent.consent_snapshot(sid, ctx.purpose, **self._tenant_kw(self.consent, ctx))
            if not snap.get("active") or snap.get("withdrawn"):
                out.append(_r("GOV-CON-07", Decision.BLOCK, "consent_missing_withdrawn_or_expired", "GDPR Art. 7(3)"))
            elif "purposes" in snap and ctx.purpose not in snap["purposes"]:
                out.append(_r("GOV-CON-07", Decision.BLOCK, "consent_not_specific_to_purpose", "GDPR Art. 7(2), 6(1)(a)"))
            elif explicit and not snap.get("explicit"):
                out.append(_r("GOV-CON-07", Decision.BLOCK, "explicit_consent_required", "GDPR Art. 9(2)(a)"))
            elif ctx.subject_is_child and not snap.get("parental_authorisation"):
                out.append(_r("GOV-CON-07", Decision.BLOCK, "child_consent_without_parental_authorisation", "GDPR Art. 8; UAVG"))
        return out

    def _rights_state(self, ctx, env):
        if not self._classified_personal(ctx) or ctx.operation in (Operation.CLASSIFY, Operation.RESTRICT):
            return []
        if self.restrictions is None:
            return [_r("GOV-RTS-08", Decision.QUARANTINE, "restriction_state_provider_unavailable", "GDPR Art. 18, 21")]
        out: List[RuleResult] = []
        for sid in ctx.subject_ids:
            st = self.restrictions.subject_state(sid, **self._tenant_kw(self.restrictions, ctx))
            if st.get("restricted") and ctx.operation not in _RESTRICTION_EXEMPT_OPS:
                out.append(_r("GOV-RTS-08", Decision.BLOCK, "processing_restricted", "GDPR Art. 18"))
            if ctx.purpose in set(st.get("objected_purposes", ())):
                out.append(_r("GOV-RTS-08", Decision.BLOCK, "subject_objected_to_purpose", "GDPR Art. 21"))
            if st.get("erasure_pending") and ctx.operation in _NO_ERASURE_PENDING_OPS:
                out.append(_r("GOV-RTS-08", Decision.BLOCK, "erasure_pending", "GDPR Art. 17"))
        return out

    def _art22(self, ctx, env):
        if not (ctx.automated_decision and ctx.significant_effect and self._personal(ctx)):
            return []
        a = env.activity
        if a is None or a.art22_exception is None:
            return [_r("GOV-ADM-10", Decision.BLOCK, "solely_automated_significant_decision_without_art22_exception", "GDPR Art. 22(1)-(2)")]
        if ctx.effective_data_class in SPECIAL_DATA_CLASSES and a.art22_exception != "explicit_consent" and a.art22_exception != "law":
            return [_r("GOV-ADM-10", Decision.BLOCK, "art22_4_special_category_conditions_not_met", "GDPR Art. 22(4)")]
        return []  # human-intervention safeguard is enforced by the oversight rule

    def _transfer(self, ctx, env):
        if not self._classified_personal(ctx):
            return []
        dest = (ctx.destination_country or "").strip().upper()
        if not dest:
            if ctx.operation in (Operation.SHARE, Operation.TRANSFER, Operation.EXPORT):
                return [_r("GOV-TRF-09", Decision.QUARANTINE, "destination_unknown_for_disclosure", "GDPR Art. 44")]
            return []
        if dest in EEA:
            return []
        a = env.activity
        entries = [t for t in (a.transfers if a else ()) if str(t.get("country", "")).upper() == dest]
        if not entries:
            return [_r("GOV-TRF-09", Decision.BLOCK, f"no_transfer_mechanism_for:{dest}", "GDPR Art. 44-46")]
        for t in entries:
            mech = t.get("mechanism")
            if mech == "adequacy" and dest in self.transfers.adequate_countries:
                if dest in self.transfers.certified_only and t.get("dpf_certified") is not True:
                    continue
                return [_r("GOV-TRF-09", Decision.ALLOW_WITH_CONTROLS, "adequacy_decision", "GDPR Art. 45", controls=("log_transfer",))]
            if mech in ("scc", "bcr") and t.get("tia_ref"):
                return [_r("GOV-TRF-09", Decision.ALLOW_WITH_CONTROLS, f"{mech}_with_transfer_impact_assessment", "GDPR Art. 46", controls=("log_transfer", "apply_supplementary_measures"))]
            if mech == "art49_derogation" and t.get("derogation_ground"):
                return [_r("GOV-TRF-09", Decision.REVIEW_REQUIRED, "art49_derogation_requires_human_review", "GDPR Art. 49", overridable=True)]
        return [_r("GOV-TRF-09", Decision.BLOCK, f"transfer_mechanism_insufficient_for:{dest}", "GDPR Art. 44-49")]

    def _oversight_rule(self, ctx, env):
        cls = self.profiles.classification(ctx.ai_system_id)
        cap = self.capabilities.get(ctx.capability)
        level = required_oversight_level(
            ctx,
            high_risk_system=bool(cls and cls.tier is RiskTier.HIGH_RISK),
            capability_requires_approval=bool(cap and cap.requires_human_approval),
        )
        if level is OversightLevel.NONE:
            return []
        if level is OversightLevel.NOTIFY:
            return [_r("GOV-HO-14", Decision.ALLOW_WITH_CONTROLS, "oversight_notification", "AI Act Art. 14", controls=("notify_human_oversight",))]
        return [_r("GOV-HO-14", Decision.REVIEW_REQUIRED, f"human_{level.value}_required", "AI Act Art. 14; GDPR Art. 22(3)", overridable=True)]

    # ------------------------------------------------------------------ human override
    def _apply_human_override(self, ctx, results: List[RuleResult], now: float) -> List[RuleResult]:
        if not any(r.decision is Decision.REVIEW_REQUIRED and r.overridable for r in results):
            return results
        ok, reason = self.oversight.verify(ctx.approval_id, ctx, now=now)
        if not ok:
            return results
        out: List[RuleResult] = []
        for r in results:
            if r.decision is Decision.REVIEW_REQUIRED and r.overridable:
                out.append(_r(r.rule_id, Decision.ALLOW_WITH_CONTROLS, f"human_approved:{ctx.approval_id}", *r.legal_refs,
                              controls=(f"consume_approval:{ctx.approval_id}",)))
            else:
                out.append(r)
        return out
