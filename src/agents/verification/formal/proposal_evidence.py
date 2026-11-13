"""Admission obligations over the *recorded evidence* of a Software patch proposal.

SoftwareAgent publishes each proposal together with the validation evidence it
gathered (syntax check, before/after regression check, unresolved sites,
confidence, and the gates it still has to pass).  This module states a small,
explicit policy over that evidence and evaluates it on the concrete record.

Scope, stated plainly: a ``VERIFIED`` result here means *the recorded evidence
satisfies the admission policy*.  It does not mean the patch preserves program
behaviour, passes tests, or is safe to execute; those remain the job of
formal models, the Safety gate and the Execution/Evaluation stages.  Evidence
that is absent yields ``UNKNOWN``, never ``VERIFIED``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

from ..utils.verification_errors import *
from ..utils.verification_helpers import *
from ..verification_result import VerificationProvenance, VerificationResult
from ..verification_types import VerificationMethod, VerificationScope, VerificationStatus
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Verification Proposal Evidence")

DEFAULT_REQUIRED_GATES: tuple[str, ...] = ("verification", "safety", "execution", "evaluation")

_HOLDS, _VIOLATED, _UNKNOWN = "holds", "violated", "unknown"


@dataclass(frozen=True, slots=True)
class ProposalPolicy:
    """Thresholds of the admission obligations (all conservative by default)."""

    require_unapplied: bool = True
    require_valid_syntax: bool = True
    max_introduced_regressions: int = 0
    max_unresolved_sites: int = 0
    min_confidence: float = 0.0
    required_gates: tuple[str, ...] = DEFAULT_REQUIRED_GATES

    def __post_init__(self) -> None:
        require_non_negative_int(self.max_introduced_regressions, "max_introduced_regressions")
        require_non_negative_int(self.max_unresolved_sites, "max_unresolved_sites")
        confidence = self.min_confidence
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise MalformedSpecificationError("min_confidence must be within [0, 1]", context={"value": repr(confidence)})
        object.__setattr__(self, "min_confidence", float(confidence))
        object.__setattr__(self, "required_gates", tuple(dict.fromkeys(str(item).strip().lower() for item in self.required_gates if str(item).strip())))

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "ProposalPolicy":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise MalformedSpecificationError("proposal policy must be a mapping")
        defaults = cls()
        unknown = set(data) - {"require_unapplied", "require_valid_syntax", "max_introduced_regressions", "max_unresolved_sites", "min_confidence", "required_gates"}
        if unknown:
            raise MalformedSpecificationError("unknown proposal policy keys", context={"keys": sorted(str(key) for key in unknown)})
        gates = data.get("required_gates", defaults.required_gates)
        if isinstance(gates, str) or not hasattr(gates, "__iter__"):
            raise MalformedSpecificationError("required_gates must be a sequence of gate names")
        return cls(
            require_unapplied=bool(data.get("require_unapplied", defaults.require_unapplied)),
            require_valid_syntax=bool(data.get("require_valid_syntax", defaults.require_valid_syntax)),
            max_introduced_regressions=data.get("max_introduced_regressions", defaults.max_introduced_regressions),  # type: ignore[arg-type]
            max_unresolved_sites=data.get("max_unresolved_sites", defaults.max_unresolved_sites),  # type: ignore[arg-type]
            min_confidence=data.get("min_confidence", defaults.min_confidence),  # type: ignore[arg-type]
            required_gates=tuple(str(item) for item in gates),  # type: ignore[union-attr]
        )

    def describe(self) -> tuple[str, ...]:
        return (
            f"require_unapplied={self.require_unapplied}",
            f"require_valid_syntax={self.require_valid_syntax}",
            f"max_introduced_regressions={self.max_introduced_regressions}",
            f"max_unresolved_sites={self.max_unresolved_sites}",
            f"min_confidence={self.min_confidence}",
            f"required_gates={','.join(self.required_gates)}",
        )


Outcome = tuple[str, str]  # (verdict, observed value as text)


def _path(evidence: Mapping[str, object], *keys: str) -> object:
    current: object = evidence
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _not_applied(evidence: Mapping[str, object], policy: ProposalPolicy) -> Outcome:
    applied = evidence.get("applied")
    if not isinstance(applied, bool):
        return _UNKNOWN, "applied flag missing"
    if not policy.require_unapplied:
        return _HOLDS, f"applied={applied} (not required)"
    return (_VIOLATED if applied else _HOLDS), f"applied={applied}"


def _gates_declared(evidence: Mapping[str, object], policy: ProposalPolicy) -> Outcome:
    gates = evidence.get("requires_gates")
    if not isinstance(gates, (list, tuple)):
        return _UNKNOWN, "requires_gates missing"
    declared = {str(item).strip().lower() for item in gates}
    missing = [gate for gate in policy.required_gates if gate not in declared]
    return (_VIOLATED, f"missing gates: {','.join(missing)}") if missing else (_HOLDS, f"declared={','.join(str(g) for g in gates)}")


def _syntax_valid(evidence: Mapping[str, object], policy: ProposalPolicy) -> Outcome:
    ok = _path(evidence, "validation", "syntax", "ok")
    if not isinstance(ok, bool):
        return _UNKNOWN, "syntax validation missing"
    if not policy.require_valid_syntax:
        return _HOLDS, f"syntax.ok={ok} (not required)"
    return (_HOLDS if ok else _VIOLATED), f"syntax.ok={ok}"


def _no_introduced_regressions(evidence: Mapping[str, object], policy: ProposalPolicy) -> Outcome:
    count = _number(_path(evidence, "validation", "static_regressions", "introduced_count"))
    if count is None:
        return _UNKNOWN, "regression check missing"
    return (_HOLDS if count <= policy.max_introduced_regressions else _VIOLATED), f"introduced={int(count)} (max {policy.max_introduced_regressions})"


def _bounded_unresolved_sites(evidence: Mapping[str, object], policy: ProposalPolicy) -> Outcome:
    count = _number(evidence.get("unresolved_sites"))
    if count is None:
        return _UNKNOWN, "unresolved_sites missing"
    return (_HOLDS if count <= policy.max_unresolved_sites else _VIOLATED), f"unresolved={int(count)} (max {policy.max_unresolved_sites})"


def _confidence_floor(evidence: Mapping[str, object], policy: ProposalPolicy) -> Outcome:
    confidence = _number(evidence.get("confidence"))
    if confidence is None:
        return _UNKNOWN, "confidence missing"
    return (_HOLDS if confidence >= policy.min_confidence else _VIOLATED), f"confidence={confidence:g} (min {policy.min_confidence:g})"


def _identity_consistent(evidence: Mapping[str, object], policy: ProposalPolicy) -> Outcome:
    patch_id, base, result = (evidence.get(key) for key in ("patch_id", "base_snapshot_digest", "result_snapshot_digest"))
    if not all(isinstance(item, str) and item.strip() for item in (patch_id, base, result)):
        return _UNKNOWN, "patch_id or snapshot digests missing"
    no_op = _path(evidence, "validation", "no_op")
    if no_op is True or base == result:
        return _VIOLATED, "proposal does not change the repository state"
    return _HOLDS, "base and result snapshots differ"


#: Obligation name → (description, evaluator).  Order is the reporting order.
OBLIGATIONS: dict[str, tuple[str, Callable[[Mapping[str, object], ProposalPolicy], Outcome]]] = {
    "not_applied": ("the proposal has not been applied", _not_applied),
    "gates_declared": ("every downstream gate is declared as still required", _gates_declared),
    "syntax_valid": ("the patched files parse without new syntax errors", _syntax_valid),
    "no_introduced_regressions": ("the before/after static check introduced no new correctness findings", _no_introduced_regressions),
    "bounded_unresolved_sites": ("the number of unresolvable edit sites is within the policy bound", _bounded_unresolved_sites),
    "confidence_floor": ("the proposal's own confidence meets the policy floor", _confidence_floor),
    "identity_consistent": ("the proposal is identified and changes the repository state", _identity_consistent),
}

_ASSUMPTIONS = (
    "the evidence record is exactly as published by SoftwareAgent and was not re-derived",
    "the SoftwareAgent validation steps (syntax, before/after static regression check) were sound",
)
_LIMITATIONS = (
    "establishes the recorded evidence against the admission policy only",
    "does not establish behavioural equivalence, test success or runtime safety of the patch",
)


def check_proposal_evidence(
    evidence: Mapping[str, object],
    policy: ProposalPolicy | None = None,
    *,
    property_name: str = "proposal_evidence",
) -> VerificationResult[object]:
    """Evaluate the admission obligations on one proposal record.

    ``REFUTED`` if any obligation is violated, ``UNKNOWN`` if none is violated
    but some evidence is missing, otherwise ``VERIFIED``.  The per-obligation
    verdicts and observed values are preserved in ``model``.
    """

    if not isinstance(evidence, Mapping):
        raise MalformedSpecificationError("proposal evidence must be a mapping", context={"type": type(evidence).__name__})
    policy = policy or ProposalPolicy()
    verdicts: dict[str, Outcome] = {name: evaluate(evidence, policy) for name, (_, evaluate) in OBLIGATIONS.items()}
    violated = [name for name, (verdict, _) in verdicts.items() if verdict == _VIOLATED]
    unknown = [name for name, (verdict, _) in verdicts.items() if verdict == _UNKNOWN]
    model = {name: f"{verdict}: {observed}" for name, (verdict, observed) in verdicts.items()}
    provenance = VerificationProvenance(method=VerificationMethod.CONTRACT, scope=VerificationScope.COMPLETE, backend="recorded-evidence")
    assumptions = (*_ASSUMPTIONS, *policy.describe())

    if violated:
        status, summary, reason = VerificationStatus.REFUTED, f"Admission obligations violated: {', '.join(violated)}.", None
    elif unknown:
        reason = f"evidence missing for: {', '.join(unknown)}"
        status, summary = VerificationStatus.UNKNOWN, "The recorded evidence is incomplete, so admission is not established."
    else:
        status, summary, reason = VerificationStatus.VERIFIED, f"All {len(verdicts)} admission obligations hold on the recorded evidence.", None
    logger.info("Proposal evidence checked | property=%s | status=%s | violated=%d | unknown=%d", property_name, status.value, len(violated), len(unknown))
    return VerificationResult(
        status=status,
        property_name=property_name,
        summary=summary,
        provenance=provenance,
        assumptions=assumptions,
        limitations=_LIMITATIONS,
        unknown_reason=reason,
        model=model,
    )


__all__ = ["DEFAULT_REQUIRED_GATES", "OBLIGATIONS", "ProposalPolicy", "check_proposal_evidence"]
