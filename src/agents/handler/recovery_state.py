from __future__ import annotations

import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .utils.config_loader import get_config_section, load_global_config
from .utils.handler_error import *
from .utils.handler_helpers import *
from .handler_memory import HandlerMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Recovery State")
printer = PrettyPrinter()


class RecoveryState(str, Enum):
    """Canonical states for a recovery attempt."""

    RECEIVED = "received"
    NORMALIZED = "normalized"
    ASSESSED = "assessed"
    RETRYING = "retrying"
    DEGRADED = "degraded"
    HUMAN_REVIEW = "human_review"
    QUARANTINED = "quarantined"
    RECOVERED = "recovered"
    FAILED = "failed"


# Default allowed transitions – can be overridden by config.
_DEFAULT_ALLOWED_TRANSITIONS: Dict[RecoveryState, set[RecoveryState]] = {
    RecoveryState.RECEIVED: {RecoveryState.NORMALIZED, RecoveryState.FAILED},
    RecoveryState.NORMALIZED: {RecoveryState.ASSESSED, RecoveryState.FAILED},
    RecoveryState.ASSESSED: {
        RecoveryState.RETRYING,
        RecoveryState.DEGRADED,
        RecoveryState.HUMAN_REVIEW,
        RecoveryState.QUARANTINED,
        RecoveryState.FAILED,
    },
    RecoveryState.RETRYING: {
        RecoveryState.RETRYING,
        RecoveryState.RECOVERED,
        RecoveryState.DEGRADED,
        RecoveryState.HUMAN_REVIEW,
        RecoveryState.QUARANTINED,
        RecoveryState.FAILED,
    },
    RecoveryState.DEGRADED: {
        RecoveryState.RECOVERED,
        RecoveryState.HUMAN_REVIEW,
        RecoveryState.FAILED,
    },
    RecoveryState.HUMAN_REVIEW: set(),
    RecoveryState.QUARANTINED: set(),
    RecoveryState.RECOVERED: set(),
    RecoveryState.FAILED: set(),
}

# Terminal states – no further transitions allowed.
_TERMINAL_STATES = {
    RecoveryState.HUMAN_REVIEW,
    RecoveryState.QUARANTINED,
    RecoveryState.RECOVERED,
    RecoveryState.FAILED,
}


@dataclass
class RecoveryStateMachine:
    """
    State machine for tracking the progress of a recovery attempt.

    Integrates with:
      - HandlerMemory: logs state transitions as telemetry and optionally saves checkpoints.
      - HandlerError: raises typed errors for invalid transitions.
      - Handler helpers: sanitisation, timestamping, JSON safety.
    """

    state: RecoveryState = RecoveryState.RECEIVED
    history: List[Dict[str, Any]] = field(default_factory=list)
    correlation_id: str = field(default_factory=lambda: generate_correlation_id("recovery"))
    created: float = field(default_factory=utc_timestamp)
    updated: float = field(default_factory=utc_timestamp)

    # Optional dependencies
    memory: Optional[HandlerMemory] = None
    error_policy: Optional[HandlerErrorPolicy] = None
    config: Mapping[str, Any] = field(default_factory=dict)

    # Internal state
    _allowed_transitions: Dict[RecoveryState, set[RecoveryState]] = field(
        default_factory=lambda: _DEFAULT_ALLOWED_TRANSITIONS.copy(),
        repr=False,
    )

    def __post_init__(self) -> None:
        # Load configuration if not provided
        if not self.config:
            global_config = load_global_config()
            self.config = get_config_section("recovery_state", global_config, default={})
        # Merge allowed transitions from config
        custom_transitions = self.config.get("allowed_transitions")
        if isinstance(custom_transitions, dict):
            self._merge_allowed_transitions(custom_transitions)
        # Ensure all states are present (fallback to default)
        for state in RecoveryState:
            if state not in self._allowed_transitions:
                self._allowed_transitions[state] = _DEFAULT_ALLOWED_TRANSITIONS.get(state, set())

    def _merge_allowed_transitions(self, custom: Mapping[str, Union[str, Sequence[str]]]) -> None:
        """Merge user-defined transition rules into the allowed set."""
        for from_state, to_states in custom.items():
            try:
                src = RecoveryState(from_state.lower())
            except ValueError:
                logger.warning(f"Ignoring unknown recovery state '{from_state}' in config")
                continue
            # Convert to_states to set of RecoveryState
            if isinstance(to_states, str):
                to_list = [to_states]
            else:
                to_list = list(to_states)
            targets = set()
            for t in to_list:
                try:
                    targets.add(RecoveryState(t.lower()))
                except ValueError:
                    logger.warning(f"Ignoring unknown target state '{t}' in config transition for '{from_state}'")
            self._allowed_transitions[src] = targets

    def transition(
        self,
        target: RecoveryState,
        *,
        reason: str,
        metadata: Optional[Mapping[str, Any]] = None,
        emit_telemetry: bool = True,
        create_checkpoint: bool = False,
    ) -> None:
        """
        Transition to a new recovery state.

        Args:
            target: The state to move to.
            reason: Human‑readable reason for the transition.
            metadata: Additional context to attach to the history entry.
            emit_telemetry: If True and a HandlerMemory instance is set, append a telemetry event.
            create_checkpoint: If True and a HandlerMemory instance is set, save a checkpoint of the current state.

        Raises:
            ValidationError: If the transition is invalid.
            RecoveryError: If a telemetry or checkpoint operation fails.
        """
        # Validate target is a RecoveryState enum
        if not isinstance(target, RecoveryState):
            raise ValidationError(
                f"Target must be a RecoveryState enum, got {type(target).__name__}",
                code="RECOVERY_INVALID_TARGET",
                context={"target": target},
                policy=self.error_policy,
            )

        # Check allowed transitions
        allowed = self._allowed_transitions.get(self.state, set())
        if target not in allowed and target != self.state:
            raise ValidationError(
                f"Invalid recovery transition: {self.state.value} -> {target.value}",
                code="RECOVERY_INVALID_TRANSITION",
                context={
                    "from": self.state.value,
                    "to": target.value,
                    "allowed": [s.value for s in allowed],
                },
                policy=self.error_policy,
            )

        # Prepare history entry
        previous = self.state
        now = utc_timestamp()
        sanitized_metadata = self._sanitize_metadata(metadata)
        entry = {
            "from": previous.value,
            "to": target.value,
            "reason": reason,
            "metadata": sanitized_metadata,
            "timestamp": now,
        }

        # Update state
        self.state = target
        self.updated = now
        self.history.append(entry)

        # Emit telemetry if configured
        if self.memory and emit_telemetry:
            self._emit_telemetry(entry)

        # Save checkpoint if requested
        if self.memory and create_checkpoint:
            self._save_checkpoint()

        logger.debug(
            "Recovery state transition: %s -> %s (reason: %s)",
            previous.value,
            target.value,
            reason,
        )

    def _sanitize_metadata(self, metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Apply error policy sanitisation to metadata."""
        raw = coerce_mapping(metadata)
        if self.error_policy:
            sanitized = self.error_policy.sanitize_context(raw)
            return make_json_safe(sanitized)  # type: ignore[return-value]
        return make_json_safe(raw)  # type: ignore[return-value]

    def _emit_telemetry(self, entry: Dict[str, Any]) -> None:
        """Append a telemetry event for the state transition."""
        try:
            event = {
                "event_type": "recovery_state_transition",
                "correlation_id": self.correlation_id,
                "timestamp": entry["timestamp"],
                "context": {
                    "from": entry["from"],
                    "to": entry["to"],
                    "reason": entry["reason"],
                },
                "metadata": entry.get("metadata", {}),
            }
            self.memory.append_telemetry(event)  # type: ignore[attr-defined]
        except Exception as exc:
            raise RecoveryError(
                "Failed to emit recovery state telemetry",
                cause=exc,
                context={"correlation_id": self.correlation_id},
                code="RECOVERY_TELEMETRY_FAILED",
                policy=self.error_policy,
            ) from exc

    def _save_checkpoint(self) -> None:
        """Save a checkpoint of the current state machine."""
        try:
            checkpoint_payload = {
                "state": self.state.value,
                "history": self.history[-10:],  # limit history size in checkpoint
                "correlation_id": self.correlation_id,
                "created": self.created,
                "updated": self.updated,
            }
            assert self.memory is not None
            self.memory.save_checkpoint(
                label=f"recovery_state_{self.state.value}",
                state=checkpoint_payload,
                metadata={
                    "correlation_id": self.correlation_id,
                    "state": self.state.value,
                },
                correlation_id=self.correlation_id,
            )
        except Exception as exc:
            raise RecoveryError(
                "Failed to save recovery state checkpoint",
                cause=exc,
                context={"correlation_id": self.correlation_id},
                code="RECOVERY_CHECKPOINT_FAILED",
                policy=self.error_policy,
            ) from exc

    def reset(
        self,
        initial_state: RecoveryState = RecoveryState.RECEIVED,
        *,
        reason: str = "Reset",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Reset the state machine to a fresh initial state."""
        self.state = initial_state
        self.created = utc_timestamp()
        self.updated = self.created
        self.history.clear()
        # Optionally log reset
        self.history.append({
            "from": "reset",
            "to": initial_state.value,
            "reason": reason,
            "metadata": self._sanitize_metadata(metadata),
            "timestamp": self.created,
        })

    def get_current_state(self) -> RecoveryState:
        """Return the current state."""
        return self.state

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the transition history, newest first."""
        if limit is not None and limit > 0:
            return list(reversed(self.history[-limit:]))
        return list(reversed(self.history))

    def is_terminal(self) -> bool:
        """Return True if the current state is terminal (no further transitions)."""
        return self.state in _TERMINAL_STATES

    def is_recovered(self) -> bool:
        """Return True if the state is RECOVERED."""
        return self.state == RecoveryState.RECOVERED

    def is_failed(self) -> bool:
        """Return True if the state is FAILED."""
        return self.state == RecoveryState.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire state machine to a JSON‑safe dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "state": self.state.value,
            "created": self.created,
            "updated": self.updated,
            "history": make_json_safe(self.history),  # type: ignore[return-value]
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *,
        memory: Optional[HandlerMemory] = None,
        error_policy: Optional[HandlerErrorPolicy] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> RecoveryStateMachine:
        """Reconstruct a RecoveryStateMachine from a serialized dictionary."""
        state_str = data.get("state", "received")
        try:
            state = RecoveryState(state_str.lower())
        except ValueError:
            state = RecoveryState.RECEIVED

        machine = cls(
            state=state,
            correlation_id=str(data.get("correlation_id", generate_correlation_id("recovery"))),
            created=coerce_float(data.get("created"), utc_timestamp()),
            updated=coerce_float(data.get("updated"), utc_timestamp()),
            memory=memory,
            error_policy=error_policy,
            config=config or {},
        )
        # Restore history
        raw_history = data.get("history", [])
        if isinstance(raw_history, list):
            machine.history = [dict(entry) for entry in raw_history if isinstance(entry, Mapping)]
        return machine


if __name__ == "__main__":
    print("\n=== Running Recovery State ===\n")
    printer.status("TEST", "Recovery State initialized", "info")
    memory = HandlerMemory()
    machine = RecoveryStateMachine(memory=memory)

    # Transition with telemetry and checkpoint
    machine.transition(
        target=RecoveryState.NORMALIZED,
        reason="Failure normalized",
        metadata={"failure_type": "TimeoutError"},
        create_checkpoint=True
    )

    # Later, retrieve history
    print(machine.get_history(limit=3))

    printer.status("TEST", "Recovery State checks passed", "info")
    print("\n=== Test ran successfully ===\n")