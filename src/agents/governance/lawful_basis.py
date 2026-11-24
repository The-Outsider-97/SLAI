"""Evidence validation for GDPR Art. 6 lawful bases and Art. 9/10 conditions.

A basis is never accepted because a string was supplied. Registration of a
processing activity requires the *evidence fields specific to that basis*; the
runtime engine then separately verifies per-subject facts (e.g. active consent)
through providers. What software cannot establish (that a cited legal provision
really exists and covers the processing) is recorded as ``requires_legal_review``
and attributed to a named reviewer rather than silently assumed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol

from .processing_context import Art9Condition, Art10Condition, LawfulBasis

__all__ = [
    "ConsentProvider",
    "RestrictionProvider",
    "validate_basis_evidence",
    "validate_art9_evidence",
    "validate_art10_evidence",
    "BASIS_REQUIRED_FIELDS",
    "ART9_REQUIRED_FIELDS",
]


class ConsentProvider(Protocol):
    def consent_snapshot(self, subject_id: str, purpose: str) -> Mapping[str, Any]:
        """Return at least ``active``(bool). Optional: ``explicit``, ``withdrawn``,
        ``parental_authorisation``, ``purposes`` (list)."""


class RestrictionProvider(Protocol):
    def subject_state(self, subject_id: str) -> Mapping[str, Any]:
        """Return ``restricted``(bool, Art. 18), ``objected_purposes``(list, Art. 21),
        ``erasure_pending``(bool)."""


_COMMON = ("reviewed_by", "reviewed_at")

BASIS_REQUIRED_FIELDS: Dict[LawfulBasis, tuple] = {
    LawfulBasis.CONSENT: ("consent_mechanism_ref", "notice_ref", "withdrawal_mechanism_ref"),
    LawfulBasis.CONTRACT: ("contract_ref", "necessity_assessment_ref"),
    LawfulBasis.LEGAL_OBLIGATION: ("legal_provision", "provision_scope_ref"),
    LawfulBasis.VITAL_INTERESTS: ("emergency_ref", "no_alternative_basis_ref"),
    LawfulBasis.PUBLIC_TASK: ("task_in_law_ref", "authority_attribution_ref"),
    LawfulBasis.LEGITIMATE_INTERESTS: ("lia_ref",),
}

ART9_REQUIRED_FIELDS: Dict[Art9Condition, tuple] = {
    Art9Condition.EXPLICIT_CONSENT: ("explicit_consent_mechanism_ref",),
    Art9Condition.EMPLOYMENT_SOCIAL_SECURITY: ("law_ref",),
    Art9Condition.VITAL_INTERESTS: ("emergency_ref",),
    Art9Condition.NONPROFIT_MEMBERS: ("body_ref", "members_only_attestation_ref"),
    Art9Condition.MANIFESTLY_PUBLIC: ("made_public_by_subject_ref",),
    Art9Condition.LEGAL_CLAIMS: ("claim_ref",),
    Art9Condition.SUBSTANTIAL_PUBLIC_INTEREST: ("law_ref", "safeguards_ref"),
    Art9Condition.HEALTH_CARE: ("law_ref", "professional_secrecy_ref"),
    Art9Condition.PUBLIC_HEALTH: ("law_ref", "safeguards_ref"),
    Art9Condition.RESEARCH_ARCHIVING_STATISTICS: ("law_ref", "safeguards_ref"),
}

_PLACEHOLDERS = {"", "n/a", "na", "none", "null", "todo", "tbd", "yes", "true", "-", "unknown"}


def _present(evidence: Mapping[str, Any], key: str) -> bool:
    value = evidence.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip()
    return text.lower() not in _PLACEHOLDERS and len(text) >= 3


def _missing(evidence: Mapping[str, Any], keys) -> List[str]:
    return [f"missing_or_placeholder:{k}" for k in keys if not _present(evidence, k)]


def validate_basis_evidence(
    basis: LawfulBasis,
    evidence: Mapping[str, Any],
    *,
    controller_is_public_authority: bool = False,
) -> List[str]:
    """Return a list of defects; an empty list means the evidence is structurally sufficient."""
    if not isinstance(basis, LawfulBasis):
        return ["unknown_lawful_basis"]
    if not isinstance(evidence, Mapping):
        return ["evidence_must_be_mapping"]
    defects = _missing(evidence, BASIS_REQUIRED_FIELDS[basis] + _COMMON)

    if basis is LawfulBasis.CONTRACT and evidence.get("data_subject_is_party") is not True:
        defects.append("contract:data_subject_must_be_party_or_precontractual_request")
    if basis is LawfulBasis.VITAL_INTERESTS and evidence.get("subject_incapable_of_consent") is not True:
        defects.append("vital_interests:subject_incapability_not_attested")
    if basis is LawfulBasis.PUBLIC_TASK and evidence.get("controller_vested_with_task") is not True:
        defects.append("public_task:controller_not_vested_with_task")
    if basis is LawfulBasis.LEGITIMATE_INTERESTS:
        if controller_is_public_authority and evidence.get("public_authority_task_processing") is True:
            defects.append("legitimate_interests:not_available_to_public_authorities_performing_tasks")
        for test in ("purpose_test_passed", "necessity_test_passed", "balancing_test_passed"):
            if evidence.get(test) is not True:
                defects.append(f"legitimate_interests:{test}_not_true")
    # Legal-obligation / public-task citations cannot be verified by software;
    # the mandatory ``reviewed_by``/``reviewed_at`` fields attribute that review
    # to a named person instead of assuming it.
    return defects


def validate_art9_evidence(condition: Art9Condition, evidence: Mapping[str, Any]) -> List[str]:
    if not isinstance(condition, Art9Condition):
        return ["unknown_art9_condition"]
    return _missing(evidence or {}, ART9_REQUIRED_FIELDS[condition] + _COMMON)


def validate_art10_evidence(condition: Art10Condition, evidence: Mapping[str, Any]) -> List[str]:
    if not isinstance(condition, Art10Condition):
        return ["unknown_art10_condition"]
    keys = ("law_ref",) if condition is Art10Condition.UNION_OR_MEMBER_STATE_LAW else ("authority_ref",)
    if condition is Art10Condition.UNION_OR_MEMBER_STATE_LAW:
        keys += ("safeguards_ref",)
    return _missing(evidence or {}, keys + _COMMON)
