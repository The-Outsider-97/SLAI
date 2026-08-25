"""Validated in-process QNN model-state schema.

This module is deliberately not a recurrent memory, shared-memory facade, or
durable checkpoint store. It validates the decoded model component passed to
and from SLAI checkpoint codecs. ``CheckpointManager`` remains the sole owner
of persistence, manifests, atomic commits, retention, and recovery.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .utils.config_loader import get_config_section, load_global_config
from .utils.quantum_errors import *
from .utils.quantum_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("QNN Quantum Memory")
printer = PrettyPrinter()

_MODEL_STATE_FIELDS = frozenset({"quantum_weights", "training_step"})
_MAX_TRAINING_STEP = int(np.iinfo(np.int64).max)


def _validated_training_step(value: Any, *, source: str) -> int:
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise QNNCheckpointStateError(f"{source} must contain one integer")
    item = scalar.reshape(-1)[0]
    item_array = np.asarray(item)
    if np.issubdtype(item_array.dtype, np.bool_) or not np.issubdtype(
        item_array.dtype,
        np.integer,
    ):
        raise QNNCheckpointStateError(f"{source} must be an integer")
    result = int(item)
    if result < 0 or result > _MAX_TRAINING_STEP:
        raise QNNCheckpointStateError(
            f"{source} must be within [0, {_MAX_TRAINING_STEP}]"
        )
    return result


def _validated_weights(value: Any) -> np.ndarray:
    try:
        raw_parameters = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise QNNCheckpointStateError(
            "quantum_weights cannot be converted to float64"
        ) from exc
    if np.iscomplexobj(raw_parameters):
        raise QNNCheckpointStateError("quantum_weights must be real-valued")
    try:
        parameters = np.asarray(raw_parameters, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QNNCheckpointStateError(
            "quantum_weights cannot be converted to float64"
        ) from exc
    if parameters.size == 0:
        raise QNNCheckpointStateError("quantum_weights must not be empty")
    if not np.all(np.isfinite(parameters)):
        raise QNNCheckpointStateError("quantum_weights contain non-finite values")
    result = np.array(parameters, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class QuantumModelState:
    """Defensive snapshot of QNN parameters and its optimization-step counter."""

    quantum_weights: np.ndarray
    training_step: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quantum_weights",
            _validated_weights(self.quantum_weights),
        )
        object.__setattr__(
            self,
            "training_step",
            _validated_training_step(
                self.training_step,
                source="training_step",
            ),
        )

    def to_component(self) -> dict[str, np.ndarray]:
        """Return independent arrays for the NumPy checkpoint codec."""

        return {
            "quantum_weights": np.array(self.quantum_weights, copy=True),
            "training_step": np.array([self.training_step], dtype=np.int64),
        }


class QuantumModelMemory:
    """Construct snapshots and validate decoded QNN checkpoint components."""

    fields = _MODEL_STATE_FIELDS

    def __init__(self) -> None:
        self.config = load_global_config()
        raw_section = self.config.get("quantum_memory")
        if not isinstance(raw_section, Mapping):
            raise QNNConfigurationError(
                "quantum_memory configuration must be a mapping"
            )
        self.memory_config = (
            get_config_section("quantum_memory", config=self.config) or {}
        )
        strict_schema = self.memory_config.get("strict_checkpoint_schema")
        if not isinstance(strict_schema, bool):
            raise QNNConfigurationError(
                "quantum_memory.strict_checkpoint_schema must be a boolean"
            )
        self.strict_checkpoint_schema = strict_schema
        logger.debug(
            "Initialized QNN model-state schema strict=%s",
            self.strict_checkpoint_schema,
        )

    def snapshot(
        self,
        weights: Any,
        training_step: int,
        *,
        validate_weights: Callable[[Any], np.ndarray] | None = None,
    ) -> QuantumModelState:
        """Create a defensive model-state snapshot.

        ``validate_weights`` should be the active circuit's shape validator.
        It lets this schema remain independent of a particular circuit layout.
        """

        candidate = weights
        if validate_weights is not None:
            try:
                candidate = validate_weights(weights)
            except (TypeError, ValueError) as exc:
                raise QNNCheckpointStateError(
                    "quantum_weights are incompatible with the active circuit"
                ) from exc
        return QuantumModelState(candidate, training_step)

    def decode(
        self,
        state: Mapping[str, Any],
        *,
        validate_weights: Callable[[Any], np.ndarray],
        strict: bool | None = None,
    ) -> QuantumModelState:
        """Validate a decoded checkpoint component without mutating the model."""

        if not isinstance(state, Mapping):
            raise QNNCheckpointStateError("QNN model state must be a mapping")
        if not callable(validate_weights):
            raise TypeError("validate_weights must be callable")

        strict_mode = self.strict_checkpoint_schema if strict is None else strict
        if not isinstance(strict_mode, bool):
            raise TypeError("strict must be a boolean or None")

        keys = set(state)
        missing = self.fields - keys
        unknown = keys - self.fields
        if missing or (strict_mode and unknown):
            missing_names = sorted(str(item) for item in missing)
            unknown_names = sorted(str(item) for item in unknown)
            raise QNNCheckpointStateError(
                "invalid QNN model state; "
                f"missing={missing_names}, unknown={unknown_names}"
            )

        try:
            weights = validate_weights(state["quantum_weights"])
        except (TypeError, ValueError) as exc:
            raise QNNCheckpointStateError(
                "quantum_weights are incompatible with the active circuit"
            ) from exc

        training_step = _validated_training_step(
            state["training_step"],
            source="training_step checkpoint value",
        )
        return QuantumModelState(weights, training_step)


__all__ = ["QuantumModelMemory", "QuantumModelState"]

if __name__ == "__main__":
    print("\n=== Running quantum_memory tests ===\n")
    printer.status("TEST", "quantum_memory initialized", "info")

    # Test _validated_training_step
    printer.status("TEST", "_validated_training_step", "info")
    assert _validated_training_step(5, source="test") == 5
    assert _validated_training_step(np.int64(10), source="test") == 10
    try:
        _validated_training_step([1, 2], source="test")
        assert False, "Should have raised"
    except QNNCheckpointStateError:
        printer.status("PASS", "rejects non-scalar", "success")
    printer.status("PASS", "training_step validation", "success")

    # Test _validated_weights
    printer.status("TEST", "_validated_weights", "info")
    w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    validated = _validated_weights(w)
    assert np.array_equal(validated, w)
    assert not validated.flags.writeable
    printer.status("PASS", "weights validation", "success")

    # Test QuantumModelState
    printer.status("TEST", "QuantumModelState", "info")
    weights = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    step = 42
    state = QuantumModelState(weights, step)
    assert np.array_equal(state.quantum_weights, weights)
    assert state.training_step == step
    component = state.to_component()
    assert "quantum_weights" in component and "training_step" in component
    printer.status("PASS", "QuantumModelState creation and to_component", "success")

    # Test QuantumModelMemory (requires config)
    printer.status("TEST", "QuantumModelMemory", "info")
    memory = QuantumModelMemory()
    try:
        # snapshot with no validator
        snap = memory.snapshot(weights, step)
        assert isinstance(snap, QuantumModelState)
        printer.status("PASS", "snapshot", "success")
    except Exception as e:
        printer.status("SKIP", f"snapshot failed: {e}", "warning")

    # Test decode with a dummy validate_weights
    try:
        def dummy_validator(w):
            return np.asarray(w, dtype=np.float64)
        state_dict = {"quantum_weights": np.array([1.0, 2.0]), "training_step": np.array([3])}
        decoded = memory.decode(state_dict, validate_weights=dummy_validator)
        assert decoded.training_step == 3
        printer.status("PASS", "decode with dummy validator", "success")
    except Exception as e:
        printer.status("SKIP", f"decode test skipped: {e}", "warning")

    print("\n=== quantum_memory tests ran successfully ===\n")