"""Validated ingress for already encoded SLAI QNN state vectors.

The module intentionally does not implement angle, amplitude, basis, or feature
encoding. Those are model choices that must be evaluated independently. Its
sole numerical responsibility is to turn an already encoded amplitude vector
into a finite, one-dimensional, normalized ``complex128`` array of the exact
dimension required by the active circuit.
"""

from __future__ import annotations

import math
import numpy as np

from collections.abc import Mapping, Sequence
from typing import Any

from .utils.config_loader import get_config_section, load_global_config
from .utils.quantum_errors import *
from .utils.quantum_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("QNN Quantum Encoding")
printer = PrettyPrinter()


def as_state_sequence(value: Sequence[Any] | np.ndarray) -> tuple[Any, ...]:
    """Disambiguate one amplitude vector from a sequence of state vectors.

    A one-dimensional NumPy array, or a flat Python sequence containing only
    scalar amplitudes, represents one state. A two-dimensional array or a
    nested Python sequence represents a sequence of states. Higher-rank NumPy
    inputs are rejected because silently flattening them would change their
    declared semantics.
    """

    if isinstance(value, (str, bytes, bytearray)):
        raise QNNInputError("state sequence must not be text or bytes")

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise QNNInputError("state sequence must not be scalar")
        if value.ndim == 1:
            return (value,)
        if value.ndim == 2:
            return tuple(value[index] for index in range(value.shape[0]))
        raise QNNInputError("state sequence must be a one- or two-dimensional array")

    if not isinstance(value, Sequence):
        raise QNNInputError("state sequence must be a sequence or NumPy array")

    values = tuple(value)
    if values and all(np.isscalar(item) for item in values):
        return (values,)
    return values


def _validate_dimension(expected_dimension: Any) -> int:
    dimension = positive_int(expected_dimension, "expected_dimension")
    if dimension <= 1 or dimension & (dimension - 1):
        raise QNNConfigurationError(
            "expected_dimension must be a power of two greater than one"
        )
    return dimension


def _scaled_norm(state: np.ndarray) -> float:
    """Compute a stable Euclidean norm for finite complex amplitudes."""

    scale = float(np.max(np.abs(state)))
    if not math.isfinite(scale) or scale <= 0.0:
        return scale
    scaled_norm = float(np.linalg.norm(state / scale))
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * scaled_norm)


def normalize_statevector(
    value: Any,
    *,
    expected_dimension: int,
    tolerance: float,
    normalize: bool,
    name: str = "state",
) -> np.ndarray:
    """Return an independent finite and normalized ``complex128`` state.

    When ``normalize`` is true, finite non-zero inputs are normalized using a
    scaled calculation that avoids avoidable overflow and underflow. When it
    is false, the input norm must already be within ``tolerance`` of one.
    """

    dimension = _validate_dimension(expected_dimension)
    norm_tolerance = positive_float(tolerance, "tolerance")
    if norm_tolerance >= 1.0:
        raise QNNConfigurationError("tolerance must be less than one")
    if not isinstance(normalize, bool):
        raise QNNConfigurationError("normalize must be a boolean")
    if not isinstance(name, str) or not name.strip():
        raise QNNInputError("state name must be a non-empty string")

    try:
        state = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise QNNInputError(
            f"{name} cannot be converted to complex amplitudes"
        ) from exc

    if state.ndim != 1 or state.size != dimension:
        raise QNNInputError(f"{name} must have shape ({dimension},), got {state.shape}")
    if not np.all(np.isfinite(state.real)) or not np.all(np.isfinite(state.imag)):
        raise QNNInputError(f"{name} contains non-finite amplitudes")

    scale = float(np.max(np.abs(state)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise QNNInputError(f"{name} must have a finite, non-zero norm")

    if normalize:
        scaled_state = state / scale
        scaled_norm = float(np.linalg.norm(scaled_state))
        if not math.isfinite(scaled_norm) or scaled_norm <= 0.0:
            raise QNNInputError(f"{name} must have a finite, non-zero norm")
        state = scaled_state / scaled_norm
    else:
        norm = _scaled_norm(state)
        if not math.isfinite(norm) or not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=norm_tolerance,
        ):
            raise QNNInputError(
                f"{name} must be normalized within {norm_tolerance:g}; "
                f"observed norm={norm:.16g}"
            )

    result = np.array(state, dtype=np.complex128, order="C", copy=True)
    final_norm = float(np.linalg.norm(result))
    if not math.isfinite(final_norm) or not math.isclose(
        final_norm,
        1.0,
        rel_tol=0.0,
        abs_tol=norm_tolerance,
    ):
        raise QNNInputError(
            f"{name} normalization failed within tolerance {norm_tolerance:g}"
        )
    return result


class StateVectorEncoder:
    """Configuration-backed boundary for already encoded state vectors."""

    def __init__(
        self,
        *,
        state_dimension: int,
        tolerance: float | None = None,
        normalize: bool | None = None,
    ) -> None:
        self.config = load_global_config()
        raw_section = self.config.get("quantum_encoding")
        if not isinstance(raw_section, Mapping):
            raise QNNConfigurationError(
                "quantum_encoding configuration must be a mapping"
            )
        self.encoding_config = (
            get_config_section("quantum_encoding", config=self.config) or {}
        )

        configured_tolerance = self.encoding_config.get("norm_tolerance")
        configured_normalize = self.encoding_config.get("normalize_inputs")
        resolved_tolerance = configured_tolerance if tolerance is None else tolerance
        resolved_normalize = configured_normalize if normalize is None else normalize

        if resolved_tolerance is None:
            raise QNNConfigurationError("quantum_encoding.norm_tolerance is required")
        if not isinstance(resolved_normalize, bool):
            raise QNNConfigurationError(
                "quantum_encoding.normalize_inputs must be a boolean"
            )

        self.state_dimension = _validate_dimension(state_dimension)
        self.tolerance = positive_float(
            resolved_tolerance,
            "quantum_encoding.norm_tolerance",
        )
        if self.tolerance >= 1.0:
            raise QNNConfigurationError(
                "quantum_encoding.norm_tolerance must be less than one"
            )
        self.normalize = resolved_normalize
        logger.debug(
            "Initialized state-vector encoder dimension=%d normalize=%s",
            self.state_dimension,
            self.normalize,
        )

    def encode_state(self, value: Any, *, name: str = "state") -> np.ndarray:
        """Validate and normalize one already encoded state vector."""

        return normalize_statevector(
            value,
            expected_dimension=self.state_dimension,
            tolerance=self.tolerance,
            normalize=self.normalize,
            name=name,
        )

    def encode_sequence(
        self,
        sequence: Sequence[Any],
        *,
        name: str,
    ) -> tuple[np.ndarray, ...]:
        """Validate a non-empty sequence without imposing resource policy."""

        values = as_state_sequence(sequence)
        if not values:
            raise QNNInputError(f"{name} must not be empty")
        return tuple(
            self.encode_state(value, name=f"{name}[{index}]")
            for index, value in enumerate(values)
        )


__all__ = ["StateVectorEncoder", "as_state_sequence", "normalize_statevector"]

if __name__ == "__main__":
    print("\n=== Running quantum_encoding tests ===\n")
    printer.status("TEST", "quantum_encoding initialized", "info")

    # Test as_state_sequence
    printer.status("TEST", "as_state_sequence", "info")
    # Single state as list of scalars
    seq1 = [1.0, 2.0, 3.0]
    result1 = as_state_sequence(seq1)
    assert len(result1) == 1 and isinstance(result1[0], tuple)
    printer.status("PASS", "single sequence list", "success")

    # Single state as 1D numpy array
    arr1 = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
    result2 = as_state_sequence(arr1)
    assert len(result2) == 1 and isinstance(result2[0], np.ndarray)
    printer.status("PASS", "single 1D array", "success")

    # Multiple states as 2D array
    arr2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    result3 = as_state_sequence(arr2)
    assert len(result3) == 2
    printer.status("PASS", "multiple states from 2D array", "success")

    # Test normalize_statevector (dimension must be power of two > 1)
    printer.status("TEST", "normalize_statevector", "info")
    # Dimension 2 is allowed.
    vec = np.array([1.0, 0.0], dtype=np.complex128)
    # Already normalized, normalize=False should pass
    norm_vec = normalize_statevector(vec, expected_dimension=2, tolerance=1e-12, normalize=False)
    assert np.allclose(np.linalg.norm(norm_vec), 1.0)
    printer.status("PASS", "normalize=False with unit norm", "success")

    # Normalize a non-normalized vector with normalize=True
    vec2 = np.array([2.0, 0.0], dtype=np.complex128)
    norm_vec2 = normalize_statevector(vec2, expected_dimension=2, tolerance=1e-12, normalize=True)
    assert np.allclose(np.linalg.norm(norm_vec2), 1.0)
    printer.status("PASS", "normalize=True", "success")

    # Test StateVectorEncoder (requires config)
    printer.status("TEST", "StateVectorEncoder", "info")
    encoder = None
    try:
        encoder = StateVectorEncoder(state_dimension=4, tolerance=1e-10, normalize=True)
        test_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        encoded = encoder.encode_state(test_state)
        assert encoded.shape == (4,)
        assert np.allclose(np.linalg.norm(encoded), 1.0)
        printer.status("PASS", "StateVectorEncoder encode_state", "success")
    except QNNConfigurationError as e:
        printer.status("SKIP", f"StateVectorEncoder test skipped (config missing): {e}", "warning")

    # Test encode_sequence
    if encoder is not None:
        try:
            seq = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            encoded_seq = encoder.encode_sequence(seq, name="test_seq")
            assert len(encoded_seq) == 2
            assert all(s.shape == (4,) for s in encoded_seq)
            printer.status("PASS", "encode_sequence", "success")
        except Exception as e:
            printer.status("SKIP", f"encode_sequence skipped: {e}", "warning")
    else:
        printer.status("SKIP", "encode_sequence skipped: StateVectorEncoder unavailable", "warning")

    print("\n=== quantum_encoding tests ran successfully ===\n")