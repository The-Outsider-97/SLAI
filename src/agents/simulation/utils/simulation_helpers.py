"""Simulation-specific helper functions.

Generic serialization, coercion, hashing, and timing primitives are reused from
SLAI's base helper layer.  This module adds only Simulation-domain operations:
explicit RNG ownership, state cloning/validation, probability validation,
state vectorization, bounded timeout checks, and transition-call adaptation.

Reproducibility follows Sandve et al. (2013): seeds, generator state, inputs,
and model/run metadata are explicit rather than hidden in process-global RNGs.
"""

from __future__ import annotations

__version__ = "2.3.0"

import copy
import inspect
import math
import secrets
import time
import numpy as np

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Optional

from ...base.utils.base_helpers import (
    Stopwatch,
    coerce_bool,
    coerce_float,
    coerce_int,
    deep_merge_dicts,
    stable_fingerprint as _base_stable_fingerprint,
    to_json_safe as _base_to_json_safe,
    utc_now_iso,
)
from ...base.modules.numpy_encoder import NumpyEncoder
from .simulation_errors import *
from logs.logger import configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Simulation Helpers")

UINT64_MAX = (1 << 64) - 1


_NUMPY_ENCODER: Optional[NumpyEncoder] = None

def _numpy_encoder() -> NumpyEncoder:
    global _NUMPY_ENCODER
    if _NUMPY_ENCODER is None:
        _NUMPY_ENCODER = NumpyEncoder(
            array_format="tagged",
            record_history=False,
            handle_non_finite="raise",
            serialize_callables=True,
            serialize_custom_objects=True,
        )
    return _NUMPY_ENCODER

def to_json_safe(value: Any) -> Any:
    """NumPy-aware adapter over SLAI's generic base serialization helper.

    The generic base helper remains the owner of serialization semantics; this
    adapter delegates NumPy values to the existing ``NumpyEncoder`` so large
    numeric states are not reduced to truncated ``repr`` strings.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        encoded = _numpy_encoder().default(value)
        return to_json_safe(encoded)
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_safe(item) for item in value]
    return _base_to_json_safe(value)

def stable_fingerprint(value: Any, *, algorithm: str = "sha256", length: int = 16) -> str:
    """Hash the NumPy-aware normalized value using SLAI's base hasher."""
    return _base_stable_fingerprint(to_json_safe(value), algorithm=algorithm, length=length)


def normalize_seed(seed: Optional[int]) -> int:
    """Return a concrete non-negative 64-bit seed.

    ``None`` intentionally means "create a new explicit seed" rather than using
    NumPy's implicit global entropy path.  The generated value can therefore be
    recorded and replayed.
    """

    if seed is None:
        return secrets.randbits(64)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise SimulationValidationError(
            "random seed must be an integer or None",
            context={"seed_type": type(seed).__name__},
        )
    value = int(seed)
    if value < 0 or value > UINT64_MAX:
        raise SimulationValidationError(
            "random seed must be in [0, 2**64 - 1]",
            context={"seed": value},
        )
    return value


def make_rng(seed: Optional[int] = None) -> tuple[np.random.Generator, int]:
    """Create an owned NumPy Generator and return its concrete replay seed."""

    concrete_seed = normalize_seed(seed)
    return np.random.default_rng(concrete_seed), concrete_seed


def spawn_child_seeds(seed: Optional[int], count: int) -> tuple[int, tuple[int, ...]]:
    """Create reproducible independent child streams using ``SeedSequence``."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SimulationValidationError("count must be a positive integer", context={"count": count})
    root_seed = normalize_seed(seed)
    sequence = np.random.SeedSequence(root_seed)
    children: list[int] = []
    for child in sequence.spawn(count):
        raw = child.generate_state(2, dtype=np.uint32)
        child_seed = (int(raw[0]) << 32) | int(raw[1])
        children.append(child_seed)
    return root_seed, tuple(children)


def rng_state_snapshot(rng: np.random.Generator) -> dict[str, Any]:
    if not isinstance(rng, np.random.Generator):
        raise SimulationReproducibilityError(
            "rng must be numpy.random.Generator",
            context={"rng_type": type(rng).__name__},
        )
    return copy.deepcopy(to_json_safe(rng.bit_generator.state))


def restore_rng(state: Mapping[str, Any]) -> np.random.Generator:
    """Restore an explicit NumPy Generator from a recorded bit-generator state."""

    if not isinstance(state, Mapping):
        raise SimulationReproducibilityError("RNG state must be a mapping")
    bit_name = str(state.get("bit_generator", "PCG64"))
    bit_cls = getattr(np.random, bit_name, None)
    if bit_cls is None:
        raise SimulationReproducibilityError(
            "Unsupported NumPy bit generator",
            context={"bit_generator": bit_name},
        )
    try:
        bit_generator = bit_cls()
        bit_generator.state = copy.deepcopy(dict(state))
        return np.random.Generator(bit_generator)
    except Exception as exc:
        raise SimulationReproducibilityError(
            "Could not restore NumPy RNG state",
            context={"bit_generator": bit_name},
            cause=exc,
        ) from exc


def clone_state(value: Any) -> Any:
    """Clone simulation state without allowing rollout-to-rollout contamination."""

    # ``copy.deepcopy`` already handles recursive/cyclic object graphs and
    # NumPy arrays correctly.  Keeping one memoized operation is safer than a
    # hand-written recursive copier that could loop on self-referential states.
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise SimulationStateError(
            "Simulation state could not be cloned safely",
            context={"state_type": type(value).__name__},
            cause=exc,
        ) from exc


def state_element_count(value: Any, *, limit: Optional[int] = None) -> int:
    """Estimate scalar element count and reject cyclic container states."""

    budget = None if limit is None else max(0, int(limit))
    active: set[int] = set()

    def _count(item: Any, running: int) -> int:
        if budget is not None and running > budget:
            return running
        if item is None or isinstance(item, (str, bytes, bool, int, float, complex, np.generic)):
            return running + 1
        if isinstance(item, np.ndarray):
            return running + int(item.size)
        if isinstance(item, (Mapping, list, tuple, set, frozenset)):
            identity = id(item)
            if identity in active:
                raise SimulationStateError("Cyclic simulation state containers are not supported")
            active.add(identity)
            try:
                children = item.values() if isinstance(item, Mapping) else item
                for child in children:
                    running = _count(child, running)
                    if budget is not None and running > budget:
                        break
                return running
            finally:
                active.remove(identity)
        return running + 1

    return _count(value, 0)


def _validate_finite(value: Any, *, path: str = "state", active: Optional[set[int]] = None) -> None:
    if active is None:
        active = set()
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise SimulationNumericalError("Non-finite numeric value in simulation state", context={"path": path, "value": repr(value)})
        return
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise SimulationNumericalError("Non-finite complex value in simulation state", context={"path": path, "value": repr(value)})
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            if not bool(np.all(np.isfinite(value))):
                raise SimulationNumericalError("Non-finite array value in simulation state", context={"path": path, "shape": tuple(int(v) for v in value.shape), "dtype": str(value.dtype)})
        elif value.dtype == object:
            identity = id(value)
            if identity in active:
                raise SimulationStateError("Cyclic object-array simulation state is not supported", context={"path": path})
            active.add(identity)
            try:
                for index, child in enumerate(value.flat):
                    _validate_finite(child, path=f"{path}[{index}]", active=active)
            finally:
                active.remove(identity)
        return
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        identity = id(value)
        if identity in active:
            raise SimulationStateError("Cyclic simulation state containers are not supported", context={"path": path})
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    _validate_finite(child, path=f"{path}.{key}", active=active)
            else:
                for index, child in enumerate(value):
                    _validate_finite(child, path=f"{path}[{index}]", active=active)
        finally:
            active.remove(identity)


def validate_state(state: Any, *, max_state_size: int = 1_000_000, allow_none: bool = False) -> Any:
    """Validate state-size bounds and reject NaN/inf contamination."""

    if state is None and not allow_none:
        raise SimulationStateError("Simulation state must not be None")
    if isinstance(max_state_size, bool) or int(max_state_size) <= 0:
        raise SimulationValidationError("max_state_size must be a positive integer")
    count = state_element_count(state, limit=int(max_state_size))
    if count > int(max_state_size):
        raise SimulationStateError(
            "Simulation state exceeds configured size limit",
            context={"elements": count, "max_state_size": int(max_state_size)},
        )
    _validate_finite(state)
    return state


def validate_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Validate a categorical probability vector without silently repairing it."""

    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise SimulationProbabilityError("probabilities must be a non-empty one-dimensional sequence")
    if not bool(np.all(np.isfinite(array))):
        raise SimulationProbabilityError("probabilities contain NaN or infinity")
    if bool(np.any(array < 0.0)):
        raise SimulationProbabilityError("probabilities must be non-negative")
    total = float(np.sum(array, dtype=np.float64))
    if not math.isfinite(total) or abs(total - 1.0) > float(tolerance):
        raise SimulationProbabilityError(
            "probabilities must sum to 1 within tolerance",
            context={"sum": total, "tolerance": tolerance},
        )
    return array


def normalize_weights(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    """Explicitly normalize non-negative finite weights into probabilities."""

    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not bool(np.all(np.isfinite(array))) or bool(np.any(array < 0.0)):
        raise SimulationProbabilityError("weights must be a non-empty finite non-negative vector")
    total = float(np.sum(array, dtype=np.float64))
    if total <= 0.0:
        raise SimulationProbabilityError("at least one weight must be positive")
    return array / total


def merge_parameters(base: Optional[Mapping[str, Any]], *overrides: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    mappings: list[Mapping[str, Any]] = []
    if base is not None:
        mappings.append(base)
    mappings.extend(item for item in overrides if item is not None)
    return deep_merge_dicts(*mappings) if mappings else {}


def numeric_state_vector(state: Any) -> np.ndarray:
    """Extract a deterministic flat numeric vector for descriptive comparisons."""

    values: list[float] = []

    def _walk(item: Any) -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float, np.integer, np.floating)):
            value = float(item)
            if math.isfinite(value):
                values.append(value)
            return
        if isinstance(item, np.ndarray):
            if np.issubdtype(item.dtype, np.number):
                flat = np.asarray(item, dtype=np.float64).ravel()
                if not bool(np.all(np.isfinite(flat))):
                    raise SimulationNumericalError("Cannot vectorize non-finite state")
                values.extend(float(v) for v in flat)
            return
        if isinstance(item, Mapping):
            for key in sorted(item.keys(), key=lambda value: str(value)):
                _walk(item[key])
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                _walk(child)

    _walk(state)
    return np.asarray(values, dtype=np.float64)


def check_timeout(started_at: float, timeout_seconds: Optional[float], *, operation: str = "simulation") -> None:
    if timeout_seconds is None:
        return
    timeout = float(timeout_seconds)
    if timeout <= 0.0:
        raise SimulationValidationError("timeout_seconds must be > 0 when provided")
    elapsed = time.monotonic() - float(started_at)
    if elapsed > timeout:
        raise SimulationTimeoutError(
            f"{operation} exceeded its timeout",
            context={"elapsed_seconds": elapsed, "timeout_seconds": timeout},
            operation=operation,
        )


def invoke_with_supported_kwargs(function: Callable[..., Any], /, **kwargs: Any) -> Any:
    """Invoke an integration callable using only keyword arguments it accepts.

    The helper keeps Simulation models compatible with narrow user-supplied
    callables while still allowing a full transition contract.  Missing required
    parameters are not hidden; Python raises ``TypeError`` normally.
    """

    if not callable(function):
        raise SimulationValidationError("integration target must be callable", context={"target_type": type(function).__name__})
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    accepts_var_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if accepts_var_kwargs:
        return function(**kwargs)
    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
        and signature.parameters[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return function(**accepted)


def enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


__all__ = [
    "Stopwatch",
    "check_timeout",
    "clone_state",
    "coerce_bool",
    "coerce_float",
    "coerce_int",
    "configure_logging",
    "enum_value",
    "invoke_with_supported_kwargs",
    "make_rng",
    "merge_parameters",
    "normalize_seed",
    "normalize_weights",
    "numeric_state_vector",
    "restore_rng",
    "rng_state_snapshot",
    "spawn_child_seeds",
    "stable_fingerprint",
    "state_element_count",
    "to_json_safe",
    "utc_now_iso",
    "validate_probabilities",
    "validate_state",
]
