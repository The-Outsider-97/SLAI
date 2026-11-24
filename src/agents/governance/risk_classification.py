"""Legally versioned EU AI Act risk classification (Regulation (EU) 2024/1689).

The ruleset reflects the Act as amended by the Digital Omnibus on AI, Regulation
(EU) 2026/1744 (in force 27 July 2026) as reported by secondary sources on
2026-11-24. ``RULESET_VERSION`` and ``RULESET_VERIFICATION`` state that plainly;
the dates below MUST be re-verified against EUR-Lex before any compliance claim.

Principles
----------
* Classification is per *system profile* (intended purpose, deployment, role),
  never global: autonomy or adaptivity alone does not make a system high-risk.
* Missing facts produce ``open_questions``, not assumptions. GPAI status is
  never inferred from size or repository visibility alone.
* Prohibited practices (Art. 5) are not overridable by approval, learned policy
  or delegation; a documented statutory exception must be present in the
  *registered profile* (never in the request) and complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "AISystemProfile",
    "Obligation",
    "PROHIBITED_PRACTICES",
    "ProfileRegistry",
    "RISK_DATES",
    "RULESET_VERIFICATION",
    "RULESET_VERSION",
    "RiskClassification",
    "RiskClassifier",
    "RiskTier",
]

RULESET_VERSION = "AIA-2024/1689+Omnibus-2026/1744@2026-11-24"
RULESET_VERIFICATION = (
    "EUR-Lex confirms Regulation (EU) 2026/1744 (OJ 24.7.2026, in force 27.7.2026). The application dates below come from several "
    "law-firm summaries and are not yet confirmed against the consolidated text. Counsel must confirm before reliance."
)

RISK_DATES: Dict[str, date] = {
    "prohibitions_and_ai_literacy": date(2025, 2, 2),
    "gpai_obligations": date(2025, 8, 2),
    "gpai_legacy_models_comply_by": date(2027, 8, 2),  # models placed before 2025-08-02
    "commission_gpai_enforcement": date(2026, 8, 2),
    "transparency_art50": date(2026, 8, 2),
    "transparency_art50_2_preexisting_systems": date(2026, 12, 2),
    "high_risk_annex_iii": date(2027, 12, 2),
    "high_risk_annex_i": date(2028, 8, 2),
}

GPAI_SYSTEMIC_COMPUTE_FLOPS = 1e25  # Art. 51(2) presumption threshold


class RiskTier(str, Enum):
    PROHIBITED = "prohibited"
    HIGH_RISK = "high_risk"
    TRANSPARENCY = "transparency"
    MINIMAL = "minimal"
    UNCLASSIFIED = "unclassified"


# tag -> (citation, exception fields that must ALL be truthy in the registered profile)
PROHIBITED_PRACTICES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "harmful_manipulation": ("Art. 5(1)(a)", ()),
    "exploit_vulnerabilities": ("Art. 5(1)(b)", ()),
    "social_scoring": ("Art. 5(1)(c)", ()),
    "criminal_risk_profiling": ("Art. 5(1)(d)", ("supports_human_assessment_on_objective_facts",)),
    "untargeted_face_scraping": ("Art. 5(1)(e)", ()),
    "emotion_recognition_workplace_education": ("Art. 5(1)(f)", ("medical_or_safety_purpose",)),
    "biometric_categorisation_sensitive": ("Art. 5(1)(g)", ("lawful_dataset_labelling_law_enforcement",)),
    "realtime_remote_biometric_id_law_enforcement": (
        "Art. 5(1)(h), 5(2)-(7)",
        ("authorised_objective", "prior_authorisation_ref"),
    ),
    "csam_ncii_generation": ("Art. 5 (added by Reg. (EU) 2026/1744)", ()),
}

ANNEX_III_AREAS = frozenset(
    {
        "biometrics",
        "critical_infrastructure",
        "education",
        "employment",
        "essential_services",  # incl. credit scoring, benefits, life/health insurance pricing
        "law_enforcement",
        "migration_border",
        "justice_democracy",
    }
)
ART6_3_EXCEPTIONS = frozenset(
    {"narrow_procedural_task", "improves_prior_human_activity", "detects_patterns_no_replacement", "preparatory_task"}
)


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    title: str
    legal_ref: str
    applies_from: date
    role: str  # provider | deployer | both

    def status(self, today: date) -> str:
        return "applicable" if today >= self.applies_from else "upcoming"


@dataclass(frozen=True)
class AISystemProfile:
    system_id: str
    name: str
    role: str = "both"  # provider | deployer | both
    intended_purpose: str = ""
    deployed_in_eu: bool = True
    practices: Tuple[str, ...] = ()
    practice_details: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    annex_iii_areas: Tuple[str, ...] = ()
    annex_i_safety_component: bool = False
    annex_i_third_party_conformity: bool = False
    art6_3_exception_claims: Tuple[str, ...] = ()
    art6_3_assessment_ref: str = ""
    performs_profiling_of_persons: bool = False
    interacts_with_natural_persons: bool = False
    interaction_obvious_to_reasonable_person: bool = False
    generates_synthetic_content: bool = False
    synthetic_content_only_assistive_editing: bool = False
    deepfake_content: bool = False
    public_interest_text: bool = False
    emotion_recognition: bool = False
    biometric_categorisation: bool = False
    is_gpai_model: Optional[bool] = None  # None => assessment outstanding
    gpai_generality_evidence_ref: str = ""
    training_compute_flops: Optional[float] = None
    commission_designated_systemic: bool = False
    free_open_source_licence: bool = False
    on_market_before_2026_08_02: bool = False
    gpai_on_market_before_2025_08_02: bool = False


@dataclass(frozen=True)
class RiskClassification:
    system_id: str
    tier: RiskTier
    labels: Tuple[str, ...]
    obligations: Tuple[Obligation, ...]
    open_questions: Tuple[str, ...]
    rationale: Tuple[str, ...]
    prohibited_tags: Tuple[str, ...]
    ruleset_version: str = RULESET_VERSION

    def to_dict(self, today: Optional[date] = None) -> Dict[str, Any]:
        today = today or date.today()
        return {
            "system_id": self.system_id,
            "tier": self.tier.value,
            "labels": list(self.labels),
            "prohibited_tags": list(self.prohibited_tags),
            "open_questions": list(self.open_questions),
            "rationale": list(self.rationale),
            "ruleset_version": self.ruleset_version,
            "ruleset_verification": RULESET_VERIFICATION,
            "obligations": [
                {
                    "id": o.obligation_id,
                    "title": o.title,
                    "legal_ref": o.legal_ref,
                    "applies_from": o.applies_from.isoformat(),
                    "role": o.role,
                    "status": o.status(today),
                }
                for o in self.obligations
            ],
        }


_HIGH_RISK_OBLIGATIONS = (
    ("HR-RISK", "Continuous risk management system", "AI Act Art. 9"),
    ("HR-DATA", "Training/validation/testing data governance", "AI Act Art. 10"),
    ("HR-DOC", "Technical documentation", "AI Act Art. 11"),
    ("HR-LOG", "Automatic record-keeping", "AI Act Art. 12"),
    ("HR-TRANSP", "Transparency and instructions for deployers", "AI Act Art. 13"),
    ("HR-OVERSIGHT", "Effective human oversight", "AI Act Art. 14"),
    ("HR-ACC", "Accuracy, robustness and cybersecurity", "AI Act Art. 15"),
    ("HR-QMS", "Quality management system", "AI Act Art. 17"),
    ("HR-CONF", "Conformity assessment and CE marking", "AI Act Art. 43"),
    ("HR-REG", "Registration in the EU database", "AI Act Art. 49"),
    ("HR-PMM", "Post-market monitoring", "AI Act Art. 72"),
    ("HR-INCIDENT", "Serious-incident reporting", "AI Act Art. 73"),
)
_GPAI_OBLIGATIONS = (
    ("GPAI-DOC", "Technical documentation of the model", "AI Act Art. 53(1)(a)", True),
    ("GPAI-DOWNSTREAM", "Information for downstream providers", "AI Act Art. 53(1)(b)", True),
    ("GPAI-COPYRIGHT", "Copyright compliance policy", "AI Act Art. 53(1)(c)", False),
    ("GPAI-SUMMARY", "Public summary of training content", "AI Act Art. 53(1)(d)", False),
)
_GPAI_SYSTEMIC_OBLIGATIONS = (
    ("GPAISR-EVAL", "Model evaluation incl. adversarial testing", "AI Act Art. 55(1)(a)"),
    ("GPAISR-RISK", "Assess and mitigate systemic risks", "AI Act Art. 55(1)(b)"),
    ("GPAISR-INCIDENT", "Serious-incident tracking and reporting", "AI Act Art. 55(1)(c)"),
    ("GPAISR-CYBER", "Adequate cybersecurity protection", "AI Act Art. 55(1)(d)"),
)


class RiskClassifier:
    """Pure function of ``AISystemProfile`` and ``RISK_DATES``; no learned components."""

    def classify(self, p: AISystemProfile) -> RiskClassification:
        rationale: List[str] = []
        questions: List[str] = []
        labels: List[str] = []
        obligations: List[Obligation] = []
        prohibited: List[str] = []

        unknown = [t for t in p.practices if t not in PROHIBITED_PRACTICES]
        if unknown:
            questions.append(f"unrecognised_practice_tags:{','.join(sorted(unknown))}")
        for tag in p.practices:
            if tag in PROHIBITED_PRACTICES and not self.exception_satisfied(p, tag):
                prohibited.append(tag)
                rationale.append(f"{PROHIBITED_PRACTICES[tag][0]}: prohibited practice '{tag}' without a complete exception")
        if prohibited:
            return RiskClassification(
                p.system_id, RiskTier.PROHIBITED, ("prohibited",), (), tuple(questions), tuple(rationale), tuple(prohibited)
            )

        # --- high-risk -------------------------------------------------
        bad_areas = [a for a in p.annex_iii_areas if a not in ANNEX_III_AREAS]
        if bad_areas:
            questions.append(f"unrecognised_annex_iii_areas:{','.join(sorted(bad_areas))}")
        high_from: Optional[date] = None
        areas = [a for a in p.annex_iii_areas if a in ANNEX_III_AREAS]
        if areas:
            exception_ok = (
                bool(p.art6_3_exception_claims)
                and set(p.art6_3_exception_claims) <= ART6_3_EXCEPTIONS
                and bool(p.art6_3_assessment_ref.strip())
                and not p.performs_profiling_of_persons
            )
            if exception_ok:
                rationale.append("Art. 6(3): Annex III area but documented derogation applies (no profiling); keep assessment on file")
                labels.append("annex_iii_derogation")
            else:
                high_from = RISK_DATES["high_risk_annex_iii"]
                labels.append("high_risk_annex_iii")
                rationale.append(f"Art. 6(2)/Annex III area(s): {', '.join(sorted(areas))}")
        if p.annex_i_safety_component and p.annex_i_third_party_conformity:
            labels.append("high_risk_annex_i")
            rationale.append("Art. 6(1): safety component of an Annex I product requiring third-party conformity assessment")
            when = RISK_DATES["high_risk_annex_i"]
            high_from = when if high_from is None else min(high_from, when)
        if high_from is not None:
            for oid, title, ref in _HIGH_RISK_OBLIGATIONS:
                obligations.append(Obligation(oid, title, ref, high_from, "provider"))
            obligations.append(
                Obligation("HR-DEPLOYER", "Deployer duties incl. oversight, log retention, FRIA where required", "AI Act Art. 26-27", high_from, "deployer")
            )
        tier = RiskTier.HIGH_RISK if high_from is not None else RiskTier.MINIMAL

        # --- Art. 50 transparency -------------------------------------
        t50 = RISK_DATES["transparency_art50"]
        if p.interacts_with_natural_persons and not p.interaction_obvious_to_reasonable_person:
            obligations.append(Obligation("T50-1", "Inform persons they interact with an AI system", "AI Act Art. 50(1)", t50, "provider"))
            labels.append("transparency_interaction")
        if p.generates_synthetic_content and not p.synthetic_content_only_assistive_editing:
            when = RISK_DATES["transparency_art50_2_preexisting_systems"] if p.on_market_before_2026_08_02 else t50
            obligations.append(Obligation("T50-2", "Machine-readable marking and detectability of synthetic content", "AI Act Art. 50(2)", when, "provider"))
            labels.append("transparency_marking")
        if p.emotion_recognition or p.biometric_categorisation:
            obligations.append(Obligation("T50-3", "Inform exposed persons of emotion-recognition/biometric categorisation", "AI Act Art. 50(3)", t50, "deployer"))
            labels.append("transparency_emotion_biometric")
        if p.deepfake_content or p.public_interest_text:
            obligations.append(Obligation("T50-4", "Disclose deepfakes / AI-generated public-interest text", "AI Act Art. 50(4)", t50, "deployer"))
            labels.append("transparency_disclosure")
        if any(l.startswith("transparency") for l in labels) and tier is RiskTier.MINIMAL:
            tier = RiskTier.TRANSPARENCY
        if p.emotion_recognition and "biometrics" not in p.annex_iii_areas:
            questions.append("emotion_recognition_declared_without_annex_iii_biometrics_area:confirm_annex_iii_1(c)_assessment")

        # --- GPAI -------------------------------------------------------
        if p.is_gpai_model is None:
            questions.append("gpai_status_unassessed:needs_generality_evidence_not_size_or_repo_visibility")
        elif p.is_gpai_model:
            if not p.gpai_generality_evidence_ref.strip():
                questions.append("gpai_declared_without_generality_evidence_ref")
            labels.append("gpai")
            systemic = p.commission_designated_systemic or (
                p.training_compute_flops is not None and p.training_compute_flops >= GPAI_SYSTEMIC_COMPUTE_FLOPS
            )
            if p.training_compute_flops is None and not p.commission_designated_systemic:
                questions.append("gpai_training_compute_unknown:systemic_risk_presumption_not_assessable")
            gpai_from = RISK_DATES["gpai_legacy_models_comply_by"] if p.gpai_on_market_before_2025_08_02 else RISK_DATES["gpai_obligations"]
            exempt_docs = p.free_open_source_licence and not systemic  # Art. 53(2)
            for oid, title, ref, is_doc in _GPAI_OBLIGATIONS:
                if is_doc and exempt_docs:
                    rationale.append(f"{ref}: documentation duty waived for free open-source licence without systemic risk")
                    continue
                obligations.append(Obligation(oid, title, ref, gpai_from, "provider"))
            if systemic:
                labels.append("gpai_systemic_risk")
                rationale.append("Art. 51: systemic risk by compute presumption (>=1e25 FLOPs) or Commission designation")
                for oid, title, ref in _GPAI_SYSTEMIC_OBLIGATIONS:
                    obligations.append(Obligation(oid, title, ref, gpai_from, "provider"))

        if tier is RiskTier.MINIMAL and not labels:
            rationale.append("No Art. 5, Annex I/III or Art. 50 trigger in the supplied profile")
        if not p.intended_purpose.strip():
            questions.append("intended_purpose_missing:classification_is_provisional")
            if tier is RiskTier.MINIMAL:
                tier = RiskTier.UNCLASSIFIED
        return RiskClassification(
            p.system_id, tier, tuple(dict.fromkeys(labels)), tuple(obligations), tuple(questions), tuple(rationale), ()
        )

    @staticmethod
    def exception_satisfied(p: AISystemProfile, tag: str) -> bool:
        """A statutory exception counts only if every required field is truthy in the registered profile."""
        required = PROHIBITED_PRACTICES.get(tag, ("", ()))[1]
        if not required:
            return False
        details = p.practice_details.get(tag, {})
        return all(bool(details.get(k)) for k in required)


class ProfileRegistry:
    """In-process registry of regulatory profiles (persisted by the Governance Agent)."""

    def __init__(self, classifier: Optional[RiskClassifier] = None):
        self._classifier = classifier or RiskClassifier()
        self._profiles: Dict[str, AISystemProfile] = {}
        self._classes: Dict[str, RiskClassification] = {}

    def register(self, profile: AISystemProfile) -> RiskClassification:
        result = self._classifier.classify(profile)
        self._profiles[profile.system_id] = profile
        self._classes[profile.system_id] = result
        return result

    def profile(self, system_id: Optional[str]) -> Optional[AISystemProfile]:
        return self._profiles.get(system_id or "")

    def classification(self, system_id: Optional[str]) -> Optional[RiskClassification]:
        return self._classes.get(system_id or "")

    def practice_permitted(self, system_id: Optional[str], tag: str) -> bool:
        profile = self.profile(system_id)
        return bool(profile) and RiskClassifier.exception_satisfied(profile, tag)  # type: ignore[arg-type]

    def all(self) -> List[RiskClassification]:
        return list(self._classes.values())
