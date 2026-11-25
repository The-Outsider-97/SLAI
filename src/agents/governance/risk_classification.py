"""Legally versioned EU AI Act risk classification (Regulation (EU) 2024/1689).

The ruleset reflects the Act as amended by the Digital Omnibus on AI, Regulation
(EU) 2026/1744 (OJ 24.7.2026, in force 27.7.2026). Application dates were checked
on 2026-11-25 against EUR-Lex (Art. 113 as amended) and the Commission's AI Act
Service Desk; ``RULESET_VERIFICATION`` records what was and was not confirmed.
This is an engineering ruleset, not legal advice: counsel must confirm before any
compliance claim.

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
    "PROHIBITED_APPLICABLE_FROM",
    "PROHIBITED_PRACTICES",
    "ProfileError",
    "ProfileRegistry",
    "ProfileStatus",
    "RISK_DATES",
    "RULESET_VERIFICATION",
    "RULESET_VERSION",
    "RiskClassification",
    "RiskClassifier",
    "RiskTier",
]

RULESET_VERSION = "AIA-2024/1689+Omnibus-2026/1744@2026-11-25"
RULESET_VERIFICATION = (
    "Verified 2026-11-25 against EUR-Lex (Reg. (EU) 2026/1744; Art. 113 as amended) and the Commission AI Act Service Desk: "
    "prohibitions/AI literacy 2025-02-02, except new Art. 5(1)(ba),(bb),(1a),(1b) from 2026-12-02; GPAI obligations 2025-08-02 "
    "(legacy models by 2027-08-02); general application, Art. 50 transparency and Commission GPAI enforcement 2026-08-02; "
    "Annex III high-risk 2027-12-02; Annex I high-risk 2028-08-02. NOT independently confirmed from the consolidated text: "
    "the 2026-12-02 date for Art. 50(2) on systems placed on the market before 2026-08-02, the Art. 6(3) derogation wording, "
    "and the mapping of practice tags to article points. Counsel must confirm before reliance."
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
    "new_prohibitions_art5_ba_bb": date(2026, 12, 2),  # Art. 113(a) as amended
}

# Practice tags that become legally applicable after the baseline Art. 5 date. They are
# still treated as prohibited earlier (fail closed): the decision rationale states the
# legal applicability date so the stricter-than-law behaviour is explicit and auditable.
PROHIBITED_APPLICABLE_FROM: Dict[str, date] = {
    "csam_ncii_generation": date(2026, 12, 2),
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
                legal_from = PROHIBITED_APPLICABLE_FROM.get(tag)
                if legal_from is not None and date.today() < legal_from:
                    rationale.append(
                        f"'{tag}' is legally applicable from {legal_from.isoformat()}; blocked earlier by policy (fail closed)"
                    )
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


class ProfileError(PermissionError):
    """An unauthorised, conflicting or malformed change to the regulatory-profile registry."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


class ProfileStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"  # kept and versioned, but nothing may rely on it until restored
    RETIRED = "retired"  # the system is withdrawn; history stays for audit
    TAMPERED = "tampered"  # persisted record failed its integrity check: never trusted, never silently repaired


_PROFILE_ROLES = ("ai_governance_officer", "dpo", "governance_admin")
_GENESIS = "0" * 64
_TUPLE_FIELDS = ("practices", "annex_iii_areas", "art6_3_exception_claims")

_PROFILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile_versions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, system_id TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL,
    profile TEXT NOT NULL, content_digest TEXT NOT NULL, tier TEXT NOT NULL, ruleset_version TEXT NOT NULL,
    actor TEXT NOT NULL, reason TEXT NOT NULL, restored_from INTEGER, at REAL NOT NULL, prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL, signature TEXT NOT NULL, UNIQUE(system_id, version));
CREATE INDEX IF NOT EXISTS pv_system ON profile_versions(system_id, version);
"""


def _profile_to_json(profile: AISystemProfile) -> str:
    import dataclasses
    import json

    return json.dumps(dataclasses.asdict(profile), sort_keys=True, default=str)


def _profile_from_json(blob: str) -> AISystemProfile:
    import json

    data = json.loads(blob)
    for name in _TUPLE_FIELDS:
        data[name] = tuple(data.get(name) or ())
    return AISystemProfile(**data)


class ProfileRegistry:
    """Durable, versioned, tamper-evident registry of regulatory profiles (GOV-P1-03).

    * Every change appends a version (never an update in place); a hash chain plus an HMAC signature per version make removal,
      reordering and edits detectable. A current version that fails verification is ``TAMPERED`` and is never relied upon.
    * Changes need a verified identity holding a profile-administration role (``require_authorization``); suspend / retire /
      restore need a documented reason. Concurrent writers are serialised and may pass ``expected_version``.
    * A change to content, status or classification emits ``profile_changed`` for ``ai_system:<id>`` so running operations are
      reassessed (GOV-P1-06) and approvals bound to the old profile stop matching (GOV-P0-02).
    * ``reclassify`` re-runs the classifier after a ruleset change and records a new version where the outcome changed.

    Without a ``path`` it is an in-memory registry (unit tests, embedding); with one it recovers its state on restart.
    """

    def __init__(self, classifier: Optional[RiskClassifier] = None, *, path: Any = ":memory:", key: Optional[bytes] = None,
                 authority: Any = None, authorised_roles: Any = _PROFILE_ROLES, require_authorization: bool = False,
                 clock: Any = None):
        import threading
        import time as _time

        from .utils.store import open_db

        self._classifier = classifier or RiskClassifier()
        self._lock = threading.RLock()
        self._clock = clock or _time.time
        self._key = key if key and len(key) >= 32 else None
        self._authority = authority
        self._roles = frozenset(authorised_roles)
        self._require_auth = bool(require_authorization)
        self._profiles: Dict[str, AISystemProfile] = {}
        self._classes: Dict[str, RiskClassification] = {}
        self._status: Dict[str, ProfileStatus] = {}
        self._versions: Dict[str, int] = {}
        self.on_change = None  # ``(kind, key, detail)``: set by the core so profile changes reach runtime reassessment
        self._db = open_db(path)
        self._db.executescript(_PROFILE_SCHEMA)
        self.recovery = self._recover()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ queries (public interface kept)
    def profile(self, system_id: Optional[str]) -> Optional[AISystemProfile]:
        with self._lock:
            return self._profiles.get(system_id or "") if self._status.get(system_id or "") is ProfileStatus.ACTIVE else None

    def classification(self, system_id: Optional[str]) -> Optional[RiskClassification]:
        with self._lock:
            return self._classes.get(system_id or "") if self._status.get(system_id or "") is ProfileStatus.ACTIVE else None

    def status(self, system_id: Optional[str]) -> Optional[ProfileStatus]:
        with self._lock:
            return self._status.get(system_id or "")

    def version(self, system_id: Optional[str]) -> int:
        with self._lock:
            return self._versions.get(system_id or "", 0)

    def practice_permitted(self, system_id: Optional[str], tag: str) -> bool:
        profile = self.profile(system_id)
        return bool(profile) and RiskClassifier.exception_satisfied(profile, tag)  # type: ignore[arg-type]

    def all(self) -> List[RiskClassification]:
        with self._lock:
            return [c for sid, c in self._classes.items() if self._status.get(sid) is ProfileStatus.ACTIVE]

    def binding(self, system_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """What a decision about ``system_id`` depends on: the profile content, its classification and its lifecycle state.
        A change to any of them (including suspension) changes this value, so approvals and grants stop matching."""
        profile, cls = self.profile(system_id), self.classification(system_id)
        if profile is None or cls is None:
            st = self.status(system_id)
            return {"system_id": system_id, "status": st.value} if st is not None else None
        import hashlib
        import json

        blob = json.dumps({"profile": _profile_to_json(profile), "tier": cls.tier.value, "labels": list(cls.labels),
                           "ruleset": cls.ruleset_version,
                           "obligations": [(o.obligation_id, o.applies_from.isoformat()) for o in cls.obligations]},
                          sort_keys=True, default=str)
        return {"system_id": system_id, "tier": cls.tier.value, "digest": hashlib.sha256(blob.encode()).hexdigest()[:32]}

    def history(self, system_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT version,status,tier,ruleset_version,actor,reason,restored_from,at,content_digest FROM profile_versions "
                                    "WHERE system_id=? ORDER BY version", (system_id,)).fetchall()
        keys = ("version", "status", "tier", "ruleset_version", "actor", "reason", "restored_from", "at", "content_digest")
        return [dict(zip(keys, r)) for r in rows]

    def verify(self) -> bool:
        """Recompute the whole chain and every signature. False means the persisted registry cannot be trusted as a whole."""
        with self._lock:
            return self._verify_all()[0]

    # ------------------------------------------------------------------ changes
    def register(self, profile: AISystemProfile, actor: Any = None, reason: str = "registration", *, expected_version: Optional[int] = None) -> RiskClassification:
        """Register or update a profile. Identical content is a no-op (no new version). Returns the classification."""
        who = self._authorise(actor)
        result = self._classifier.classify(profile)
        blob = _profile_to_json(profile)
        digest = self._digest(blob, result)
        changed = False
        with self._lock:
            self._begin()
            try:
                st = self._status.get(profile.system_id)
                if st in (ProfileStatus.SUSPENDED, ProfileStatus.RETIRED, ProfileStatus.TAMPERED):
                    raise ProfileError("invalid_state", f"profile {profile.system_id} is {st.value}: resume or restore it explicitly (with a reason)")
                current = self._versions.get(profile.system_id, 0)
                if expected_version is not None and expected_version != current:
                    raise ProfileError("version_conflict", f"profile {profile.system_id} is at version {current}, not {expected_version}")
                prior = self._latest_row(profile.system_id)
                if prior is not None and prior[0] == digest and prior[1] == ProfileStatus.ACTIVE.value:
                    self._db.execute("ROLLBACK")
                    return self._classes[profile.system_id]
                self._append(profile.system_id, current + 1, ProfileStatus.ACTIVE, blob, digest, result.tier.value, result.ruleset_version, who, reason, None)
                self._db.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
            self._profiles[profile.system_id], self._classes[profile.system_id] = profile, result
            self._status[profile.system_id], self._versions[profile.system_id] = ProfileStatus.ACTIVE, current + 1
            changed = True
        if changed:
            self._emit("profile_changed", profile.system_id, result.tier.value)
        return result

    def suspend(self, system_id: str, actor: Any, reason: str) -> None:
        self._transition(system_id, actor, reason, ProfileStatus.SUSPENDED, "profile_suspended", allowed_from=(ProfileStatus.ACTIVE,))

    def retire(self, system_id: str, actor: Any, reason: str) -> None:
        self._transition(system_id, actor, reason, ProfileStatus.RETIRED, "profile_retired",
                         allowed_from=(ProfileStatus.ACTIVE, ProfileStatus.SUSPENDED))

    def resume(self, system_id: str, actor: Any, reason: str) -> None:
        """Lift a suspension: the (unchanged) profile becomes active again under a new version."""
        self._transition(system_id, actor, reason, ProfileStatus.ACTIVE, "profile_resumed", allowed_from=(ProfileStatus.SUSPENDED,))

    def restore(self, system_id: str, actor: Any, reason: str, version: int) -> RiskClassification:
        """Re-instate an earlier version's content as a NEW version (history is never rewritten)."""
        who = self._authorise(actor)
        self._need_reason(reason)
        with self._lock:
            row = self._db.execute("SELECT profile,signature,entry_hash,system_id,version,status,content_digest,tier,ruleset_version,actor,reason,at,prev_hash FROM "
                                   "profile_versions WHERE system_id=? AND version=?", (system_id, version)).fetchone()
            if row is None:
                raise ProfileError("unknown_version", f"no version {version} for {system_id}")
            if not self._row_valid(row):
                raise ProfileError("integrity", "the requested version failed its integrity check")
            profile = _profile_from_json(row[0])
        result = self._classifier.classify(profile)
        blob = _profile_to_json(profile)
        digest = self._digest(blob, result)
        with self._lock:
            self._begin()
            try:
                current = self._versions.get(system_id, 0)
                self._append(system_id, current + 1, ProfileStatus.ACTIVE, blob, digest, result.tier.value, result.ruleset_version, who, reason, version)
                self._db.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
            self._profiles[system_id], self._classes[system_id] = profile, result
            self._status[system_id], self._versions[system_id] = ProfileStatus.ACTIVE, current + 1
        self._emit("profile_changed", system_id, f"restored_from:{version}")
        return result

    def reclassify(self, actor: Any = None, reason: str = "ruleset_change") -> List[str]:
        """Re-run the classifier over every active profile (after ``RULESET_VERSION`` changed). Returns the changed system ids."""
        changed: List[str] = []
        for sid in list(self._profiles):
            if self.status(sid) is not ProfileStatus.ACTIVE:
                continue
            profile = self._profiles[sid]
            before = self._classes[sid]
            after = self._classifier.classify(profile)
            if (after.tier, after.ruleset_version, after.labels) != (before.tier, before.ruleset_version, before.labels) or \
                    [(o.obligation_id, o.applies_from) for o in after.obligations] != [(o.obligation_id, o.applies_from) for o in before.obligations]:
                self.register(profile, actor, reason)
                changed.append(sid)
        return changed

    # ------------------------------------------------------------------ internals
    def _transition(self, system_id: str, actor: Any, reason: str, to: ProfileStatus, kind: str, *, allowed_from: Any) -> None:
        who = self._authorise(actor)
        self._need_reason(reason)
        with self._lock:
            self._begin()
            try:
                st = self._status.get(system_id)
                if st is None:
                    raise ProfileError("unknown_system", f"no profile for {system_id}")
                if st not in allowed_from:
                    raise ProfileError("invalid_transition", f"profile is {st.value}; cannot become {to.value}")
                prior = self._latest_row(system_id)
                profile = self._profiles[system_id]
                cls = self._classes[system_id]
                current = self._versions[system_id]
                self._append(system_id, current + 1, to, _profile_to_json(profile), prior[0] if prior else "", cls.tier.value, cls.ruleset_version, who, reason, None)
                self._db.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
            self._status[system_id], self._versions[system_id] = to, current + 1
        self._emit(kind, system_id, reason[:80])

    def _emit(self, kind: str, system_id: str, detail: str) -> None:
        if self.on_change:
            self.on_change("profile_changed", f"ai_system:{system_id}", f"{kind}:{detail}")

    def _need_reason(self, reason: str) -> None:
        if len((reason or "").strip()) < 10:
            raise ProfileError("reason_required", "a documented reason (at least 10 characters) is required")

    def _authorise(self, actor: Any) -> str:
        if not self._require_auth:
            return getattr(actor, "principal_id", None) or "unauthenticated"
        if self._authority is None or actor is None:
            raise ProfileError("not_authorised", "profile changes require a verified identity")
        from .utils.identity import IdentityVerificationError

        try:
            verified = self._authority.verify(getattr(actor, "token", None), now=self._clock())
        except IdentityVerificationError as exc:
            raise ProfileError("not_authorised", f"identity invalid: {exc.code}") from exc
        if not (self._roles & set(verified.roles)):
            raise ProfileError("not_authorised", "no profile-administration role")
        return verified.principal_id

    def _digest(self, blob: str, cls: RiskClassification) -> str:
        import hashlib
        import json

        return hashlib.sha256(json.dumps({"p": blob, "t": cls.tier.value, "l": list(cls.labels), "r": cls.ruleset_version,
                                          "o": [(o.obligation_id, o.applies_from.isoformat()) for o in cls.obligations]}, sort_keys=True).encode()).hexdigest()

    def _begin(self) -> None:
        self._db.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        try:
            self._db.execute("ROLLBACK")
        except Exception:
            pass

    def _latest_row(self, system_id: str) -> Optional[tuple]:
        return self._db.execute("SELECT content_digest,status FROM profile_versions WHERE system_id=? ORDER BY version DESC LIMIT 1", (system_id,)).fetchone()

    def _sign(self, entry_hash: str) -> str:
        import hashlib
        import hmac

        if self._key is None:
            return ""
        return hmac.new(self._key, f"slai-gov-v1:profile:{entry_hash}".encode(), hashlib.sha256).hexdigest()

    def _entry_hash(self, prev: str, fields: tuple) -> str:
        import hashlib
        import json

        return hashlib.sha256((prev + json.dumps(list(fields), sort_keys=True, default=str)).encode()).hexdigest()

    def _append(self, system_id: str, version: int, status: ProfileStatus, blob: str, digest: str, tier: str, ruleset: str, actor: str,
                reason: str, restored_from: Optional[int]) -> None:
        row = self._db.execute("SELECT entry_hash FROM profile_versions ORDER BY seq DESC LIMIT 1").fetchone()
        prev = row[0] if row else _GENESIS
        at = self._clock()
        entry = self._entry_hash(prev, (system_id, version, status.value, blob, digest, tier, ruleset, actor, reason, restored_from, repr(at)))
        self._db.execute(
            "INSERT INTO profile_versions(system_id,version,status,profile,content_digest,tier,ruleset_version,actor,reason,restored_from,at,prev_hash,entry_hash,signature) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (system_id, version, status.value, blob, digest, tier, ruleset, actor, reason, restored_from, at, prev, entry, self._sign(entry)))

    def _row_valid(self, row: tuple) -> bool:
        profile, signature, entry_hash, system_id, version, status, digest, tier, ruleset, actor, reason, at, prev = row[:13]
        restored = self._db.execute("SELECT restored_from FROM profile_versions WHERE system_id=? AND version=?", (system_id, version)).fetchone()[0]
        import hmac

        expected = self._entry_hash(prev, (system_id, version, status, profile, digest, tier, ruleset, actor, reason, restored, repr(at)))
        return expected == entry_hash and (self._key is None or hmac.compare_digest(signature, self._sign(entry_hash)))

    def _verify_all(self) -> tuple:
        prev, bad = _GENESIS, []
        for r in self._db.execute("SELECT profile,signature,entry_hash,system_id,version,status,content_digest,tier,ruleset_version,actor,reason,at,prev_hash "
                                  "FROM profile_versions ORDER BY seq").fetchall():
            if r[12] != prev or not self._row_valid(r):
                bad.append((r[3], r[4]))
            prev = r[2]
        return (not bad), bad

    def _recover(self) -> Dict[str, Any]:
        """Rebuild the in-memory view from the durable log. A system whose current version fails verification is TAMPERED."""
        with self._lock:
            ok, bad = self._verify_all()
            bad_systems = {sid for sid, _ in bad}
            rows = self._db.execute("SELECT system_id,MAX(version) FROM profile_versions GROUP BY system_id").fetchall()
            for sid, ver in rows:
                self._versions[sid] = ver
                if sid in bad_systems:
                    self._status[sid] = ProfileStatus.TAMPERED
                    continue
                r = self._db.execute("SELECT profile,status FROM profile_versions WHERE system_id=? AND version=?", (sid, ver)).fetchone()
                try:
                    profile = _profile_from_json(r[0])
                    self._profiles[sid], self._classes[sid] = profile, self._classifier.classify(profile)
                    self._status[sid] = ProfileStatus(r[1])
                except Exception:
                    self._status[sid] = ProfileStatus.TAMPERED
            return {"systems": len(rows), "chain_intact": ok, "tampered": sorted(sid for sid, st in self._status.items() if st is ProfileStatus.TAMPERED)}
