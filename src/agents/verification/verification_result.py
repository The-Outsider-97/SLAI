"""Structured verification evidence and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Generic, Mapping, TypeVar

from .verification_proof import ProofArtifact
from .verification_types import*
from .utils.verification_errors import MalformedSpecificationError


TState = TypeVar("TState")


class TraceKind(str, Enum):
    """Purpose of a state/transition trace."""

    COUNTEREXAMPLE = "counterexample"
    WITNESS = "witness"


@dataclass(frozen=True, slots=True)
class TraceStep(Generic[TState]):
    """One state in a structured path.

    ``transition_label`` describes the transition used to reach this step from
    the preceding step and is therefore ``None`` for the first state.
    """

    index: int
    state: TState
    transition_label: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise MalformedSpecificationError("trace step index must be a non-negative integer")
        if self.transition_label is not None and not str(self.transition_label).strip():
            raise MalformedSpecificationError("transition_label must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class VerificationTrace(Generic[TState]):
    """Structured witness or violating path through an explicit transition system."""

    kind: TraceKind
    steps: tuple[TraceStep[TState], ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TraceKind):
            object.__setattr__(self, "kind", TraceKind(self.kind))
        steps = tuple(self.steps)
        if not steps:
            raise MalformedSpecificationError("verification traces must contain at least one state")
        for expected, step in enumerate(steps):
            if step.index != expected:
                raise MalformedSpecificationError(
                    "trace step indices must be contiguous and zero-based",
                    context={"expected": expected, "observed": step.index},
                )
        if not str(self.reason).strip():
            raise MalformedSpecificationError("trace reason must be non-empty")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "reason", str(self.reason).strip())


@dataclass(frozen=True, slots=True)
class VerificationProvenance:
    """Method, backend, scope, and resource facts for one result."""

    method: VerificationMethod
    scope: VerificationScope
    backend: str | None = None
    bounds: ResourceBounds | None = None
    elapsed_seconds: float = 0.0
    explored_states: int | None = None
    explored_transitions: int | None = None
    depth_reached: int | None = None
    quantified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.method, VerificationMethod):
            object.__setattr__(self, "method", VerificationMethod(self.method))
        if not isinstance(self.scope, VerificationScope):
            object.__setattr__(self, "scope", VerificationScope(self.scope))
        if self.elapsed_seconds < 0:
            raise MalformedSpecificationError("elapsed_seconds must be non-negative")
        for name in ("explored_states", "explored_transitions", "depth_reached"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise MalformedSpecificationError(f"{name} must be a non-negative integer or null")


@dataclass(frozen=True, slots=True)
class VerificationResult(Generic[TState]):
    """A non-boolean verification result with preserved formal evidence."""

    status: VerificationStatus
    property_name: str
    summary: str
    provenance: VerificationProvenance
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    unknown_reason: str | None = None
    model: Mapping[str, str] = field(default_factory=dict)
    unsat_core: tuple[str, ...] = ()
    trace: VerificationTrace[TState] | None = None
    proof: ProofArtifact | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatus):
            object.__setattr__(self, "status", VerificationStatus(self.status))
        name = str(self.property_name).strip()
        summary = str(self.summary).strip()
        if not name:
            raise MalformedSpecificationError("property_name must be non-empty")
        if not summary:
            raise MalformedSpecificationError("verification result summary must be non-empty")
        object.__setattr__(self, "property_name", name)
        object.__setattr__(self, "summary", summary)
        assumptions = tuple(str(v).strip() for v in self.assumptions)
        limitations = tuple(str(v).strip() for v in self.limitations)
        unsat_core = tuple(str(v).strip() for v in self.unsat_core)
        if any(not item for item in assumptions):
            raise MalformedSpecificationError("result assumptions must be non-empty strings")
        if any(not item for item in limitations):
            raise MalformedSpecificationError("result limitations must be non-empty strings")
        if any(not item for item in unsat_core):
            raise MalformedSpecificationError("unsat_core entries must be non-empty strings")
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "model", MappingProxyType(dict(self.model)))
        object.__setattr__(self, "unsat_core", unsat_core)
        if self.unknown_reason is not None:
            cleaned_reason = str(self.unknown_reason).strip()
            object.__setattr__(self, "unknown_reason", cleaned_reason or None)

        if self.status is VerificationStatus.UNKNOWN and not self.unknown_reason:
            raise MalformedSpecificationError("UNKNOWN results must preserve an unknown_reason")
        if self.status is VerificationStatus.BOUNDED:
            if self.provenance.scope is not VerificationScope.BOUNDED:
                raise MalformedSpecificationError("BOUNDED results must use BOUNDED provenance scope")
            if self.provenance.bounds is None or self.provenance.bounds.max_depth is None:
                raise MalformedSpecificationError("BOUNDED results must preserve a semantic max_depth bound")
        if self.status is VerificationStatus.VERIFIED and self.trace is not None:
            if self.trace.kind is TraceKind.COUNTEREXAMPLE:
                raise MalformedSpecificationError("VERIFIED results cannot contain a counterexample")
        if self.status is VerificationStatus.REFUTED and self.trace is not None:
            if self.trace.kind is not TraceKind.COUNTEREXAMPLE:
                raise MalformedSpecificationError("REFUTED traces must be counterexamples")

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, JSON-friendly evidence summary.

        Arbitrary state objects in traces are represented with ``repr`` rather
        than invoking user-defined serializers from a formal-result path.
        """

        trace_payload: dict[str, object] | None = None
        if self.trace is not None:
            trace_payload = {
                "kind": self.trace.kind.value,
                "reason": self.trace.reason,
                "steps": [
                    {
                        "index": step.index,
                        "state": repr(step.state),
                        "transition_label": step.transition_label,
                    }
                    for step in self.trace.steps
                ],
            }
        proof_payload: dict[str, object] | None = (
            None if self.proof is None else self.proof.to_dict()
        )
        bounds_payload: dict[str, object] | None = None
        if self.provenance.bounds is not None:
            bounds_payload = {
                "max_states": self.provenance.bounds.max_states,
                "max_transitions": self.provenance.bounds.max_transitions,
                "max_depth": self.provenance.bounds.max_depth,
                "timeout_seconds": self.provenance.bounds.timeout_seconds,
            }
        return {
            "status": self.status.value,
            "property_name": self.property_name,
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "unknown_reason": self.unknown_reason,
            "model": dict(sorted(self.model.items())),
            "unsat_core": list(self.unsat_core),
            "trace": trace_payload,
            "proof": proof_payload,
            "provenance": {
                "method": self.provenance.method.value,
                "scope": self.provenance.scope.value,
                "backend": self.provenance.backend,
                "bounds": bounds_payload,
                "elapsed_seconds": self.provenance.elapsed_seconds,
                "explored_states": self.provenance.explored_states,
                "explored_transitions": self.provenance.explored_transitions,
                "depth_reached": self.provenance.depth_reached,
                "quantified": self.provenance.quantified,
            },
        }


__all__ = [
    "TraceKind",
    "TraceStep",
    "VerificationProvenance",
    "VerificationResult",
    "VerificationTrace",
]
