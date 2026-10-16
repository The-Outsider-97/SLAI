"""Formal specifications, Hoare-style contracts, and proof obligations.

This module represents formal assertions supplied to the Verification subsystem.
It deliberately does not execute programs, score outcomes, or perform generic
logical reasoning.  Universal discharge of an obligation remains the job of a
formal solver/model checker.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar


from ..utils.verification_errors import *
from ..utils.verification_helpers import *
from logs.logger import get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Verification Specifications")


TState = TypeVar("TState")


def _normalize_named_sequence(values: tuple[object, ...], expected_type: type[object], field: str) -> tuple[object, ...]:
    """Require typed formal artifacts with unique, non-empty names."""

    seen: set[str] = set()
    for index, item in enumerate(values):
        if not isinstance(item, expected_type):
            raise MalformedSpecificationError(
                f"{field} must contain {expected_type.__name__} objects",
                context={"field": field, "index": index, "type": type(item).__name__},
            )
        name = require_non_empty_text(getattr(item, "name", ""), f"{field}[{index}].name")
        if name in seen:
            raise MalformedSpecificationError(
                f"{field} must use unique names",
                context={"field": field, "duplicate_name": name},
            )
        seen.add(name)
    return values


@dataclass(frozen=True, slots=True)
class StatePredicate(Generic[TState]):
    """Named Boolean state assertion used by formal contracts and obligations."""

    name: str
    predicate: Callable[[TState], bool]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_non_empty_text(self.name, "predicate name"))
        require_callable(self.predicate, "state predicate")
        object.__setattr__(self, "description", str(self.description).strip())

    def holds(self, state: TState) -> bool:
        """Evaluate the explicit predicate on one supplied state.

        A predicate exception is an evaluation failure, not a formal refutation.
        Returning a non-Boolean value is a malformed predicate contract.
        """

        try:
            result = self.predicate(state)
        except VerificationError:
            raise
        except Exception as exc:
            raise PredicateEvaluationError(
                f"predicate {self.name!r} failed during evaluation",
                context={"predicate": self.name, "state": safe_repr(state)},
                cause=exc,
            ) from exc
        if not isinstance(result, bool):
            raise MalformedSpecificationError(
                f"predicate {self.name!r} must return bool",
                context={"predicate": self.name, "result_type": type(result).__name__},
            )
        return result

    def __call__(self, state: TState) -> bool:
        return self.holds(state)


@dataclass(frozen=True, slots=True)
class Invariant(StatePredicate[TState]):
    """A state predicate intended to hold throughout a transition system."""


@dataclass(frozen=True, slots=True)
class HoareContract(Generic[TState]):
    """Partial-correctness contract ``{P} command {Q}``.

    The object records pre/postconditions and assumptions only. Establishing
    termination is intentionally outside this contract, matching Hoare's
    partial-correctness interpretation.
    """

    name: str
    precondition: StatePredicate[TState]
    postcondition: StatePredicate[TState]
    assumptions: tuple[StatePredicate[TState], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_non_empty_text(self.name, "contract name"))
        if not isinstance(self.precondition, StatePredicate):
            raise MalformedSpecificationError("precondition must be a StatePredicate")
        if not isinstance(self.postcondition, StatePredicate):
            raise MalformedSpecificationError("postcondition must be a StatePredicate")

        assumptions = tuple(self.assumptions)
        _normalize_named_sequence(assumptions, StatePredicate, "contract assumptions" )
        object.__setattr__(self, "assumptions", assumptions)

    def assumptions_hold(self, state: TState) -> bool:
        """Evaluate all explicit assumptions on one concrete state."""

        return all(assumption.holds(state) for assumption in self.assumptions)

    def precondition_holds(self, state: TState) -> bool:
        """Evaluate only the recorded Hoare precondition on one state."""

        return self.precondition.holds(state)

    def postcondition_holds(self, state: TState) -> bool:
        """Evaluate only the recorded Hoare postcondition on one state."""

        return self.postcondition.holds(state)

    def applicable(self, state: TState) -> bool:
        """Whether assumptions and precondition hold for the supplied state.

        This is not a proof that the command establishes the postcondition.
        """

        return self.assumptions_hold(state) and self.precondition_holds(state)


class ProofObligationKind(str, Enum):
    """Closed set of Hoare/invariant proof-obligation roles represented here."""

    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    CONSEQUENCE = "consequence"
    COMPOSITION = "composition"
    ITERATION = "iteration"
    INVARIANT_INITIALIZATION = "invariant_initialization"
    INVARIANT_PRESERVATION = "invariant_preservation"


@dataclass(frozen=True, slots=True)
class ProofObligation(Generic[TState]):
    """Named implication between explicit state predicates.

    ``holds`` checks one concrete state. It is not itself a proof procedure;
    universal discharge belongs to a solver or complete finite-state checker.
    """

    name: str
    kind: ProofObligationKind
    premises: tuple[StatePredicate[TState], ...]
    conclusion: StatePredicate[TState]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name",require_non_empty_text(self.name, "proof obligation name"))
        if not isinstance(self.kind, ProofObligationKind):
            try:
                object.__setattr__(self, "kind", ProofObligationKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise MalformedSpecificationError("invalid proof obligation kind", cause=exc) from exc

        premises = tuple(self.premises)
        _normalize_named_sequence(premises, StatePredicate, "proof obligation premises")
        if not isinstance(self.conclusion, StatePredicate):
            raise MalformedSpecificationError(
                "proof obligation conclusion must be a StatePredicate"
            )
        object.__setattr__(self, "premises", premises)

    def holds(self, state: TState) -> bool:
        """Evaluate ``premises => conclusion`` at one supplied state."""

        if all(premise.holds(state) for premise in self.premises):
            return self.conclusion.holds(state)
        return True


@dataclass(frozen=True, slots=True)
class FormalSpecification(Generic[TState]):
    """Cohesive collection of formal artifacts for one verification subject.

    The container gives the future VerificationAgent a typed specification
    boundary without becoming a generic validation object.  It stores only
    formal assumptions/contracts/invariants/obligations and does not execute or
    score them.
    """

    name: str
    assumptions: tuple[StatePredicate[TState], ...] = ()
    contracts: tuple[HoareContract[TState], ...] = ()
    invariants: tuple[Invariant[TState], ...] = ()
    obligations: tuple[ProofObligation[TState], ...] = ()
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_non_empty_text(self.name, "formal specification name"))
        assumptions = tuple(self.assumptions)
        contracts = tuple(self.contracts)
        invariants = tuple(self.invariants)
        obligations = tuple(self.obligations)

        _normalize_named_sequence(assumptions, StatePredicate, "specification assumptions")
        _normalize_named_sequence(contracts, HoareContract, "specification contracts")
        _normalize_named_sequence(invariants, Invariant, "specification invariants")
        _normalize_named_sequence(obligations, ProofObligation, "specification obligations")

        if not contracts and not invariants and not obligations:
            raise MalformedSpecificationError("formal specification must contain at least one contract, invariant, or proof obligation")

        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "contracts", contracts)
        object.__setattr__(self, "invariants", invariants)
        object.__setattr__(self, "obligations", obligations)
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "tags", normalize_tags(self.tags))

        logger.debug(
            "Formal specification created | name=%s | assumptions=%d | contracts=%d | invariants=%d | obligations=%d",
            self.name,
            len(assumptions),
            len(contracts),
            len(invariants),
            len(obligations),
        )

    def assumptions_hold(self, state: TState) -> bool:
        """Evaluate the specification-level assumptions on one state."""

        return all(assumption.holds(state) for assumption in self.assumptions)

    @property
    def artifact_count(self) -> int:
        return len(self.contracts) + len(self.invariants) + len(self.obligations)


__all__ = [
    "FormalSpecification",
    "HoareContract",
    "Invariant",
    "ProofObligation",
    "ProofObligationKind",
    "StatePredicate",
]
