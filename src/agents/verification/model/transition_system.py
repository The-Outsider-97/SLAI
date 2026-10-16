"""Immutable explicit finite transition-system representation.

The representation is deliberately structural.  It models states, initial
states, transitions and atomic-proposition labels, but leaves reachability,
invariant checking and counterexample search to :mod:`model_checker`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Hashable, Mapping, Sequence, TypeVar, cast

from ..utils.verification_errors import InvalidTransitionModelError
from ..utils.verification_helpers import *
from logs.logger import get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Verification Transition System")


TState = TypeVar("TState", bound=Hashable)


@dataclass(frozen=True, slots=True)
class Transition(Generic[TState]):
    """One directed transition in an explicit finite transition relation."""

    source: TState
    target: TState
    label: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_hashable(self.source, "transition source", error_cls=InvalidTransitionModelError)
        ensure_hashable(self.target, "transition target", error_cls=InvalidTransitionModelError)
        if self.label is not None:
            object.__setattr__(
                self,
                "label",
                require_non_empty_text(self.label, "transition label", error_cls=InvalidTransitionModelError),
            )
        normalized = normalize_metadata(
            cast(Mapping[object, object], self.metadata),
            field="transition metadata",
            error_cls=InvalidTransitionModelError,
        )
        object.__setattr__(self, "metadata", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class TransitionSystem(Generic[TState]):
    """Finite transition system with deterministic state and edge ordering.

    A total transition relation is optional because reachability and safety
    checks remain meaningful for finite terminating/deadlock models. Set
    ``require_total=True`` when Kripke-style totality is part of the formal model.
    """

    states: tuple[TState, ...]
    initial_states: tuple[TState, ...]
    transitions: tuple[Transition[TState], ...]
    labels: Mapping[TState, frozenset[str]] = field(default_factory=dict)
    require_total: bool = False
    _adjacency: Mapping[TState, tuple[Transition[TState], ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _reverse_adjacency: Mapping[TState, tuple[Transition[TState], ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _state_index: Mapping[TState, int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _atomic_propositions: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        states = normalize_unique_hashables(
            self.states,
            "transition-system states",
            require_non_empty=True,
            error_cls=InvalidTransitionModelError,
        )
        initial = normalize_unique_hashables(
            self.initial_states,
            "initial states",
            require_non_empty=True,
            error_cls=InvalidTransitionModelError,
        )
        transitions = tuple(self.transitions)

        if not isinstance(self.require_total, bool):
            raise InvalidTransitionModelError("require_total must be boolean")

        state_set = set(states)
        if not set(initial).issubset(state_set):
            unknown = [safe_repr(state) for state in initial if state not in state_set]
            raise InvalidTransitionModelError(
                "every initial state must belong to states",
                context={"unknown_initial_states": unknown[:10]},
            )

        if any(not isinstance(edge, Transition) for edge in transitions):
            raise InvalidTransitionModelError(
                "transitions must contain Transition objects"
            )
        for index, edge in enumerate(transitions):
            if edge.source not in state_set or edge.target not in state_set:
                raise InvalidTransitionModelError(
                    "transition endpoints must belong to the transition-system state set",
                    context={
                        "transition_index": index,
                        "source": safe_repr(edge.source),
                        "target": safe_repr(edge.target),
                    },
                )

        state_index = {state: index for index, state in enumerate(states)}
        indexed_edges = list(enumerate(transitions))
        indexed_edges.sort(
            key=lambda pair: (
                state_index[pair[1].source],
                state_index[pair[1].target],
                pair[1].label or "",
                pair[0],
            )
        )
        transitions = tuple(edge for _, edge in indexed_edges)

        if not isinstance(self.labels, Mapping):
            raise InvalidTransitionModelError("labels must be a state-to-propositions mapping")
        normalized_labels: dict[TState, frozenset[str]] = {}
        atomic_propositions: set[str] = set()
        for state, atoms in self.labels.items():
            if state not in state_set:
                raise InvalidTransitionModelError(
                    "label mapping contains an unknown state",
                    context={"state": safe_repr(state)},
                )
            normalized_atoms = frozenset(
                normalize_string_values(
                    atoms,
                    f"labels[{safe_repr(state)}]",
                    sort=True,
                    unique=True,
                    error_cls=InvalidTransitionModelError,
                )
            )
            normalized_labels[state] = normalized_atoms
            atomic_propositions.update(normalized_atoms)
        for state in states:
            normalized_labels.setdefault(state, frozenset())

        adjacency_lists: dict[TState, list[Transition[TState]]] = {
            state: [] for state in states
        }
        reverse_lists: dict[TState, list[Transition[TState]]] = {
            state: [] for state in states
        }
        for edge in transitions:
            adjacency_lists[edge.source].append(edge)
            reverse_lists[edge.target].append(edge)

        deadlocks = tuple(state for state in states if not adjacency_lists[state])
        if self.require_total and deadlocks:
            raise InvalidTransitionModelError(
                "require_total=True but one or more states have no outgoing transition",
                context={
                    "deadlock_states": [safe_repr(item) for item in deadlocks[:10]],
                    "deadlock_count": len(deadlocks),
                },
            )

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "initial_states", initial)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "labels", MappingProxyType(normalized_labels))
        object.__setattr__(self, "_state_index", MappingProxyType(state_index))
        object.__setattr__(
            self,
            "_adjacency",
            MappingProxyType(
                {state: tuple(adjacency_lists[state]) for state in states}
            ),
        )
        object.__setattr__(
            self,
            "_reverse_adjacency",
            MappingProxyType(
                {state: tuple(reverse_lists[state]) for state in states}
            ),
        )
        object.__setattr__(self, "_atomic_propositions", frozenset(atomic_propositions))

        logger.debug(
            "Transition system initialized | states=%d | initial=%d | transitions=%d | total=%s | propositions=%d",
            len(states),
            len(initial),
            len(transitions),
            not deadlocks,
            len(atomic_propositions),
        )

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    @property
    def atomic_propositions(self) -> frozenset[str]:
        return self._atomic_propositions

    @property
    def deadlock_states(self) -> tuple[TState, ...]:
        return tuple(state for state in self.states if not self._adjacency[state])

    @property
    def is_total(self) -> bool:
        return not self.deadlock_states

    def contains_state(self, state: TState) -> bool:
        try:
            return state in self._state_index
        except TypeError:
            return False

    def _require_known_state(self, state: TState, operation: str) -> None:
        if not self.contains_state(state):
            raise InvalidTransitionModelError(
                f"{operation} requested an unknown state",
                context={"state": safe_repr(state)},
            )

    def outgoing(self, state: TState) -> tuple[Transition[TState], ...]:
        self._require_known_state(state, "outgoing transitions")
        return self._adjacency[state]

    def incoming(self, state: TState) -> tuple[Transition[TState], ...]:
        self._require_known_state(state, "incoming transitions")
        return self._reverse_adjacency[state]

    def successors(self, state: TState) -> tuple[TState, ...]:
        return tuple(edge.target for edge in self.outgoing(state))

    def predecessors(self, state: TState) -> tuple[TState, ...]:
        return tuple(edge.source for edge in self.incoming(state))

    def state_labels(self, state: TState) -> frozenset[str]:
        self._require_known_state(state, "state labels")
        return self.labels[state]

    def transitions_between(
        self,
        source: TState,
        target: TState,
        *,
        label: str | None = None,
    ) -> tuple[Transition[TState], ...]:
        """Return all transitions from ``source`` to ``target`` in model order."""

        self._require_known_state(source, "transition query")
        self._require_known_state(target, "transition query")
        normalized_label = (
            None
            if label is None
            else require_non_empty_text(
                label,
                "transition query label",
                error_cls=InvalidTransitionModelError,
            )
        )
        return tuple(
            edge
            for edge in self._adjacency[source]
            if edge.target == target
            and (normalized_label is None or edge.label == normalized_label)
        )

    def has_transition(
        self,
        source: TState,
        target: TState,
        *,
        label: str | None = None,
    ) -> bool:
        return bool(self.transitions_between(source, target, label=label))

    def is_path(
        self,
        states: Sequence[TState],
        *,
        transition_labels: Sequence[str | None] | None = None,
    ) -> bool:
        """Return whether a finite state sequence follows the transition relation.

        When labels are supplied there must be exactly one label entry per edge.
        ``None`` in the label sequence means that any label is acceptable for the
        corresponding transition.
        """

        path = tuple(states)
        if not path:
            return False
        if any(not self.contains_state(state) for state in path):
            return False

        labels: tuple[str | None, ...] | None = None
        if transition_labels is not None:
            labels = tuple(transition_labels)
            if len(labels) != len(path) - 1:
                raise InvalidTransitionModelError(
                    "transition_labels length must equal len(states) - 1",
                    context={
                        "states": len(path),
                        "transition_labels": len(labels),
                    },
                )
            normalized: list[str | None] = []
            for index, label in enumerate(labels):
                normalized.append(
                    None
                    if label is None
                    else require_non_empty_text(
                        label,
                        f"transition_labels[{index}]",
                        error_cls=InvalidTransitionModelError,
                    )
                )
            labels = tuple(normalized)

        for index, (source, target) in enumerate(zip(path, path[1:])):
            label = None if labels is None else labels[index]
            if not self.has_transition(source, target, label=label):
                return False
        return True


__all__ = ["MetadataValue", "Transition", "TransitionSystem"]
