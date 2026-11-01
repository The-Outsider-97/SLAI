"""Conservative bounded explicit-state reachability and invariant checking."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

from .transition_system import Transition, TransitionSystem
from ..formal.specifications import Invariant, StatePredicate
from ..verification_result import *
from ..verification_types import *
from ..utils.verification_errors import MalformedSpecificationError
from logs.logger import get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Verification Model Checker")


TState = TypeVar("TState", bound=Hashable)


@dataclass(slots=True)
class _SearchOutcome(Generic[TState]):
    found: TState | None
    parents: dict[TState, tuple[TState, Transition[TState]] | None]
    depths: dict[TState, int]
    explored_states: int
    explored_transitions: int
    depth_reached: int
    elapsed_seconds: float
    cutoff: str | None = None


class ModelChecker:
    """Explicit-state safety/reachability checker with hard resource controls."""

    def __init__(self, default_bounds: ResourceBounds | None = None) -> None:
        self.default_bounds = default_bounds or ResourceBounds()
        logger.debug(
            "ModelChecker initialized | max_states=%d | max_transitions=%d | max_depth=%s | timeout=%s",
            self.default_bounds.max_states,
            self.default_bounds.max_transitions,
            self.default_bounds.max_depth,
            self.default_bounds.timeout_seconds,
        )

    @staticmethod
    def _expired(started: float, bounds: ResourceBounds) -> bool:
        return bounds.timeout_seconds is not None and (time.monotonic() - started) >= bounds.timeout_seconds

    def _search(
        self,
        system: TransitionSystem[TState],
        target: StatePredicate[TState],
        bounds: ResourceBounds,
    ) -> _SearchOutcome[TState]:
        started = time.monotonic()
        parents: dict[TState, tuple[TState, Transition[TState]] | None] = {}
        depths: dict[TState, int] = {}
        queue: deque[TState] = deque()

        for state in system.initial_states:
            if state not in parents:
                if len(parents) >= bounds.max_states:
                    return _SearchOutcome(
                        found=None,
                        parents=parents,
                        depths=depths,
                        explored_states=0,
                        explored_transitions=0,
                        depth_reached=0,
                        elapsed_seconds=time.monotonic() - started,
                        cutoff="max_states",
                    )
                parents[state] = None
                depths[state] = 0
                queue.append(state)

        explored_states = 0
        explored_transitions = 0
        depth_reached = 0
        depth_cutoff = False

        while queue:
            if self._expired(started, bounds):
                return _SearchOutcome(
                    None, parents, depths, explored_states, explored_transitions,
                    depth_reached, time.monotonic() - started, "timeout"
                )
            state = queue.popleft()
            explored_states += 1
            depth = depths[state]
            depth_reached = max(depth_reached, depth)
            if target.holds(state):
                return _SearchOutcome(
                    state, parents, depths, explored_states, explored_transitions,
                    depth_reached, time.monotonic() - started
                )

            outgoing = system.outgoing(state)
            if bounds.max_depth is not None and depth >= bounds.max_depth:
                if any(edge.target not in parents for edge in outgoing):
                    depth_cutoff = True
                continue

            for edge in outgoing:
                if self._expired(started, bounds):
                    return _SearchOutcome(
                        None, parents, depths, explored_states, explored_transitions,
                        depth_reached, time.monotonic() - started, "timeout"
                    )
                explored_transitions += 1
                if explored_transitions > bounds.max_transitions:
                    return _SearchOutcome(
                        None, parents, depths, explored_states, explored_transitions - 1,
                        depth_reached, time.monotonic() - started, "max_transitions"
                    )
                successor = edge.target
                if successor in parents:
                    continue
                if len(parents) >= bounds.max_states:
                    return _SearchOutcome(
                        None, parents, depths, explored_states, explored_transitions,
                        depth_reached, time.monotonic() - started, "max_states"
                    )
                parents[successor] = (state, edge)
                depths[successor] = depth + 1
                queue.append(successor)

        return _SearchOutcome(
            None,
            parents,
            depths,
            explored_states,
            explored_transitions,
            depth_reached,
            time.monotonic() - started,
            "max_depth" if depth_cutoff else None,
        )

    @staticmethod
    def _trace(
        outcome: _SearchOutcome[TState],
        target_state: TState,
        kind: TraceKind,
        reason: str,
    ) -> VerificationTrace[TState]:
        reversed_steps: list[tuple[TState, str | None]] = []
        state = target_state
        while True:
            parent = outcome.parents[state]
            if parent is None:
                reversed_steps.append((state, None))
                break
            previous, edge = parent
            reversed_steps.append((state, edge.label))
            state = previous
        ordered = list(reversed(reversed_steps))
        return VerificationTrace(
            kind=kind,
            reason=reason,
            steps=tuple(
                TraceStep(index=index, state=state_value, transition_label=transition_label)
                for index, (state_value, transition_label) in enumerate(ordered)
            ),
        )

    @staticmethod
    def _provenance(
        outcome: _SearchOutcome[TState],
        bounds: ResourceBounds,
        scope: VerificationScope,
        method: VerificationMethod = VerificationMethod.MODEL_CHECKING,
    ) -> VerificationProvenance:
        return VerificationProvenance(
            method=method,
            scope=scope,
            bounds=bounds,
            elapsed_seconds=outcome.elapsed_seconds,
            explored_states=outcome.explored_states,
            explored_transitions=outcome.explored_transitions,
            depth_reached=outcome.depth_reached,
        )

    @staticmethod
    def _inconclusive_result(
        *,
        property_name: str,
        outcome: _SearchOutcome[TState],
        bounds: ResourceBounds,
        complete_summary: str,
    ) -> VerificationResult[TState]:
        if outcome.cutoff == "max_depth":
            return VerificationResult(
                status=VerificationStatus.BOUNDED,
                property_name=property_name,
                summary=complete_summary,
                provenance=ModelChecker._provenance(outcome, bounds, VerificationScope.BOUNDED),
                limitations=(f"search stopped at semantic depth bound {bounds.max_depth}",),
            )
        reason = f"verification resource limit reached: {outcome.cutoff or 'unknown'}"
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            property_name=property_name,
            summary="The explicit-state search did not complete.",
            provenance=ModelChecker._provenance(outcome, bounds, VerificationScope.RESOURCE_LIMITED),
            limitations=(reason,),
            unknown_reason=reason,
        )

    def check_reachability(
        self,
        system: TransitionSystem[TState],
        target: StatePredicate[TState],
        *,
        bounds: ResourceBounds | None = None,
    ) -> VerificationResult[TState]:
        if not isinstance(system, TransitionSystem) or not isinstance(target, StatePredicate):
            raise MalformedSpecificationError("reachability requires a TransitionSystem and StatePredicate")
        active_bounds = bounds or self.default_bounds
        outcome = self._search(system, target, active_bounds)
        logger.debug(
            "Reachability search completed | property=%s | found=%s | cutoff=%s | states=%d | transitions=%d | depth=%d",
            target.name,
            outcome.found is not None,
            outcome.cutoff,
            outcome.explored_states,
            outcome.explored_transitions,
            outcome.depth_reached,
        )
        if outcome.cutoff is not None:
            logger.warning(
                "Reachability search constrained | property=%s | cutoff=%s",
                target.name,
                outcome.cutoff,
            )
        if outcome.found is not None:
            return VerificationResult(
                status=VerificationStatus.SATISFIABLE,
                property_name=target.name,
                summary="A reachable state satisfies the target predicate.",
                provenance=self._provenance(outcome, active_bounds, VerificationScope.COMPLETE),
                trace=self._trace(
                    outcome,
                    outcome.found,
                    TraceKind.WITNESS,
                    f"reachable state satisfies {target.name}",
                ),
            )
        if outcome.cutoff is not None:
            return self._inconclusive_result(
                property_name=target.name,
                outcome=outcome,
                bounds=active_bounds,
                complete_summary="No satisfying state was found within the declared depth bound.",
            )
        return VerificationResult(
            status=VerificationStatus.UNSATISFIABLE,
            property_name=target.name,
            summary="No reachable state satisfies the target predicate in the exhausted finite state space.",
            provenance=self._provenance(outcome, active_bounds, VerificationScope.COMPLETE),
        )

    def check_invariant(
        self,
        system: TransitionSystem[TState],
        invariant: Invariant[TState],
        *,
        bounds: ResourceBounds | None = None,
    ) -> VerificationResult[TState]:
        if not isinstance(invariant, Invariant):
            raise MalformedSpecificationError("check_invariant requires an Invariant")
        negated = StatePredicate(
            name=f"not({invariant.name})",
            predicate=lambda state: not invariant.holds(state),
            description=f"violation of invariant {invariant.name}",
        )
        active_bounds = bounds or self.default_bounds
        outcome = self._search(system, negated, active_bounds)
        logger.debug(
            "Invariant search completed | property=%s | violated=%s | cutoff=%s | states=%d | transitions=%d | depth=%d",
            invariant.name,
            outcome.found is not None,
            outcome.cutoff,
            outcome.explored_states,
            outcome.explored_transitions,
            outcome.depth_reached,
        )
        if outcome.cutoff is not None:
            logger.warning(
                "Invariant search constrained | property=%s | cutoff=%s",
                invariant.name,
                outcome.cutoff,
            )
        if outcome.found is not None:
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                property_name=invariant.name,
                summary="A reachable state violates the invariant.",
                provenance=self._provenance(outcome, active_bounds, VerificationScope.COMPLETE),
                trace=self._trace(
                    outcome,
                    outcome.found,
                    TraceKind.COUNTEREXAMPLE,
                    f"reachable state violates {invariant.name}",
                ),
            )
        if outcome.cutoff is not None:
            return self._inconclusive_result(
                property_name=invariant.name,
                outcome=outcome,
                bounds=active_bounds,
                complete_summary="No invariant violation was found within the declared depth bound.",
            )
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            property_name=invariant.name,
            summary="The invariant holds in every reachable state of the exhausted finite state space.",
            provenance=self._provenance(outcome, active_bounds, VerificationScope.COMPLETE),
        )

    def check_inductive_invariant(
        self,
        system: TransitionSystem[TState],
        invariant: Invariant[TState],
        *,
        bounds: ResourceBounds | None = None,
    ) -> VerificationResult[TState]:
        """Check initialization and one-step inductive preservation over all declared edges."""
        if not isinstance(system, TransitionSystem) or not isinstance(invariant, Invariant):
            raise MalformedSpecificationError("inductive checking requires TransitionSystem and Invariant")
        active_bounds = bounds or self.default_bounds
        started = time.monotonic()
        if len(system.states) > active_bounds.max_states:
            reason = "verification resource limit reached: max_states"
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                property_name=invariant.name,
                summary="Inductive invariant checking did not complete.",
                provenance=VerificationProvenance(
                    method=VerificationMethod.INVARIANT,
                    scope=VerificationScope.RESOURCE_LIMITED,
                    bounds=active_bounds,
                    elapsed_seconds=time.monotonic() - started,
                    explored_states=0,
                    explored_transitions=0,
                    depth_reached=0,
                ),
                limitations=(reason,),
                unknown_reason=reason,
            )

        checked_states = 0
        for initial in system.initial_states:
            if self._expired(started, active_bounds):
                return self._inductive_unknown(invariant, active_bounds, started, checked_states, 0, "timeout")
            checked_states += 1
            if not invariant.holds(initial):
                trace = VerificationTrace(
                    kind=TraceKind.COUNTEREXAMPLE,
                    reason=f"initial state violates {invariant.name}",
                    steps=(TraceStep(0, initial, None),),
                )
                return VerificationResult(
                    status=VerificationStatus.REFUTED,
                    property_name=invariant.name,
                    summary="The invariant fails the initialization obligation.",
                    provenance=VerificationProvenance(
                        method=VerificationMethod.INVARIANT,
                        scope=VerificationScope.COMPLETE,
                        bounds=active_bounds,
                        elapsed_seconds=time.monotonic() - started,
                        explored_states=checked_states,
                        explored_transitions=0,
                        depth_reached=0,
                    ),
                    trace=trace,
                )

        checked_transitions = 0
        seen_sources: set[TState] = set()
        for edge in system.transitions:
            if self._expired(started, active_bounds):
                return self._inductive_unknown(
                    invariant, active_bounds, started, len(seen_sources), checked_transitions, "timeout"
                )
            if checked_transitions >= active_bounds.max_transitions:
                return self._inductive_unknown(
                    invariant, active_bounds, started, len(seen_sources), checked_transitions, "max_transitions"
                )
            checked_transitions += 1
            seen_sources.add(edge.source)
            source_holds = invariant.holds(edge.source)
            if source_holds and not invariant.holds(edge.target):
                trace = VerificationTrace(
                    kind=TraceKind.COUNTEREXAMPLE,
                    reason=f"transition does not preserve {invariant.name}",
                    steps=(
                        TraceStep(0, edge.source, None),
                        TraceStep(1, edge.target, edge.label),
                    ),
                )
                return VerificationResult(
                    status=VerificationStatus.REFUTED,
                    property_name=invariant.name,
                    summary="The invariant fails the inductive preservation obligation.",
                    provenance=VerificationProvenance(
                        method=VerificationMethod.INVARIANT,
                        scope=VerificationScope.COMPLETE,
                        bounds=active_bounds,
                        elapsed_seconds=time.monotonic() - started,
                        explored_states=len(seen_sources),
                        explored_transitions=checked_transitions,
                        depth_reached=1,
                    ),
                    trace=trace,
                )

        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            property_name=invariant.name,
            summary="The invariant holds initially and is preserved by every declared transition.",
            provenance=VerificationProvenance(
                method=VerificationMethod.INVARIANT,
                scope=VerificationScope.COMPLETE,
                bounds=active_bounds,
                elapsed_seconds=time.monotonic() - started,
                explored_states=len(system.states),
                explored_transitions=checked_transitions,
                depth_reached=1 if system.transitions else 0,
            ),
        )

    @staticmethod
    def _inductive_unknown(
        invariant: Invariant[TState],
        bounds: ResourceBounds,
        started: float,
        states: int,
        transitions: int,
        cutoff: str,
    ) -> VerificationResult[TState]:
        reason = f"verification resource limit reached: {cutoff}"
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            property_name=invariant.name,
            summary="Inductive invariant checking did not complete.",
            provenance=VerificationProvenance(
                method=VerificationMethod.INVARIANT,
                scope=VerificationScope.RESOURCE_LIMITED,
                bounds=bounds,
                elapsed_seconds=time.monotonic() - started,
                explored_states=states,
                explored_transitions=transitions,
                depth_reached=1 if transitions else 0,
            ),
            limitations=(reason,),
            unknown_reason=reason,
        )


__all__ = ["ModelChecker"]
