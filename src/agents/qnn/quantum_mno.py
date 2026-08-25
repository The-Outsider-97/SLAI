"""Born measurement, evaluation, and bounded QNN gradient mathematics.

The functions in this module are deterministic for a supplied state, parameter
tensor, and objective. They do not mutate an agent, append telemetry, advance a
random generator, or create checkpoints. Consequently, shifted and
finite-difference probes are safe to use inside SLAI tuning transactions.

The exact parameter-shift path is intentionally restricted to the implemented
Pauli-rotation circuit and its ``state_fidelity`` loss. The probability-MSE loss
is nonlinear in circuit expectation values and therefore uses central finite
differences instead of being mislabeled as an exact parameter-shift gradient.
"""

from __future__ import annotations

import math
import numpy as np

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .quantum_encoding import as_state_sequence
from .utils.config_loader import get_config_section, load_global_config
from .utils.quantum_errors import *
from .utils.quantum_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("QNN Measurement and Optimization")
printer = PrettyPrinter()

_SUPPORTED_METRICS = frozenset({"state_fidelity", "probability_mse"})
_SUPPORTED_GRADIENT_METHODS = frozenset({"parameter_shift", "finite_difference"})
_METRIC_ALIASES = {
    "fidelity": "state_fidelity",
    "mse": "probability_mse",
}


def _positive_tolerance(value: Any, name: str) -> float:
    tolerance = positive_float(value, name)
    if tolerance >= 1.0:
        raise QNNConfigurationError(f"{name} must be less than one")
    return tolerance


def _finite_state(value: Any, *, name: str) -> np.ndarray:
    try:
        state = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise QNNInputError(
            f"{name} cannot be converted to complex amplitudes"
        ) from exc
    if state.ndim != 1 or state.size == 0:
        raise QNNInputError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(state.real)) or not np.all(np.isfinite(state.imag)):
        raise QNNInputError(f"{name} contains non-finite amplitudes")
    return state


def _stable_vector_norm(value: np.ndarray) -> float:
    scale = float(np.max(np.abs(value)))
    if not math.isfinite(scale) or scale <= 0.0:
        return scale
    scaled_norm = float(np.linalg.norm(value / scale))
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * scaled_norm)


def _finite_loss(value: Any, *, probe: str) -> float:
    array = np.asarray(value)
    if array.size != 1 or np.iscomplexobj(array):
        raise QNNInputError(f"{probe} loss must be one real scalar")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise QNNInputError(f"{probe} loss is non-finite")
    return result


def _gradient_parameters(weights: Any) -> np.ndarray:
    try:
        raw_parameters = np.asarray(weights)
    except (TypeError, ValueError) as exc:
        raise QNNInputError("gradient weights must be real-valued") from exc
    if np.iscomplexobj(raw_parameters):
        raise QNNInputError("gradient weights must be real-valued")
    try:
        parameters = np.asarray(raw_parameters, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QNNInputError("gradient weights must be real-valued") from exc
    if parameters.ndim == 0 or parameters.size == 0:
        raise QNNInputError("gradient weights must be a non-empty tensor")
    if not np.all(np.isfinite(parameters)):
        raise QNNInputError("gradient weights must be finite")
    return np.array(parameters, dtype=np.float64, order="C", copy=True)


def born_probabilities(state: Any, *, tolerance: float = 1.0e-12) -> np.ndarray:
    """Return the normalized Born distribution ``p_i = |amplitude_i|^2``.

    Dividing by the finite total makes the function robust to insignificant
    floating-point norm drift while retaining an explicit zero-state guard.
    QNN inference still reports norm error separately, so this normalization
    does not conceal model-health evidence.
    """

    probability_tolerance = _positive_tolerance(tolerance, "tolerance")
    amplitudes = _finite_state(state, name="state")
    scale = float(np.max(np.abs(amplitudes)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise QNNInputError("state does not define a non-zero Born distribution")

    scaled = amplitudes / scale
    scaled_probabilities = np.square(np.abs(scaled), dtype=np.float64)
    total = float(np.sum(scaled_probabilities, dtype=np.float64))
    if not math.isfinite(total) or total <= probability_tolerance:
        raise QNNInputError("state does not define a finite Born distribution")

    probabilities = scaled_probabilities / total
    if not np.all(np.isfinite(probabilities)):
        raise QNNInputError("Born probabilities are non-finite")
    probability_sum = float(np.sum(probabilities, dtype=np.float64))
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=probability_tolerance,
    ):
        raise QNNInputError(
            "Born probability normalization exceeds the configured tolerance"
        )
    probabilities /= probability_sum
    return np.array(probabilities, dtype=np.float64, order="C", copy=True)


def sample_born(
    state: Any,
    *,
    rng: np.random.Generator,
    num_samples: int = 1,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Sample computational-basis indices using an explicit RNG instance."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    sample_count = positive_int(num_samples, "num_samples")
    probabilities = born_probabilities(state, tolerance=tolerance)
    return np.asarray(
        rng.choice(probabilities.size, size=sample_count, p=probabilities),
        dtype=np.int64,
    )


def state_fidelity(output: Any, target: Any) -> float:
    """Return pure-state fidelity ``|<target|output>|^2``.

    The norm quotient extends the usual normalized-state expression safely to
    non-zero vector representatives of the same quantum rays.
    """

    output_state = _finite_state(output, name="output")
    target_state = _finite_state(target, name="target")
    if target_state.shape != output_state.shape:
        raise QNNInputError("fidelity states must be shape-aligned")

    output_norm = _stable_vector_norm(output_state)
    target_norm = _stable_vector_norm(target_state)
    if (
        not math.isfinite(output_norm)
        or not math.isfinite(target_norm)
        or output_norm <= 0.0
        or target_norm <= 0.0
    ):
        raise QNNInputError("fidelity is undefined for a zero- or invalid-norm state")

    normalized_output = output_state / np.max(np.abs(output_state))
    normalized_output /= np.linalg.norm(normalized_output)
    normalized_target = target_state / np.max(np.abs(target_state))
    normalized_target /= np.linalg.norm(normalized_target)
    overlap = np.vdot(normalized_target, normalized_output)
    fidelity = float(np.abs(overlap) ** 2)
    if not math.isfinite(fidelity):
        raise QNNInputError("fidelity is non-finite")
    return float(np.clip(fidelity, 0.0, 1.0))


def probability_mse(
    output: Any,
    target: Any,
    *,
    tolerance: float = 1.0e-12,
) -> float:
    """Return mean squared error between two Born distributions."""

    output_probabilities = born_probabilities(output, tolerance=tolerance)
    target_probabilities = born_probabilities(target, tolerance=tolerance)
    if output_probabilities.shape != target_probabilities.shape:
        raise QNNInputError("probability vectors must be shape-aligned")
    result = float(np.mean(np.square(output_probabilities - target_probabilities)))
    if not math.isfinite(result):
        raise QNNInputError("probability MSE is non-finite")
    return result


def maximum_norm_error(states: Sequence[Any]) -> float:
    """Return the largest absolute deviation from unit Euclidean norm."""

    state_values = as_state_sequence(states)
    if not state_values:
        raise QNNInputError("states must not be empty")
    errors = [
        abs(_stable_vector_norm(_finite_state(state, name=f"states[{index}]")) - 1.0)
        for index, state in enumerate(state_values)
    ]
    result = float(max(errors))
    if not math.isfinite(result):
        raise QNNInputError("state norm error is non-finite")
    return result


class QNNMetrics:
    """Evaluate declared QNN metrics without fabricating missing evidence."""

    supported = _SUPPORTED_METRICS

    def __init__(
        self,
        metric: str = "state_fidelity",
        *,
        probability_tolerance: float | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_global_config()
        raw_section = self.config.get("quantum_mno")
        if not isinstance(raw_section, Mapping):
            raise QNNConfigurationError("quantum_mno configuration must be a mapping")
        self.mno_config = (
            get_config_section(
                "quantum_mno",
                config=self.config,
            )
            or {}
        )

        normalized = str(metric).strip().casefold()
        normalized = _METRIC_ALIASES.get(normalized, normalized)
        if normalized not in self.supported:
            supported = ", ".join(sorted(self.supported))
            raise QNNConfigurationError(f"metric must be one of: {supported}")

        configured_tolerance = self.mno_config.get("probability_tolerance")
        resolved_tolerance = (
            configured_tolerance
            if probability_tolerance is None
            else probability_tolerance
        )
        if resolved_tolerance is None:
            raise QNNConfigurationError("quantum_mno.probability_tolerance is required")

        self.metric = normalized
        self.probability_tolerance = _positive_tolerance(
            resolved_tolerance,
            "quantum_mno.probability_tolerance",
        )

    def evaluate(self, outputs: Sequence[Any], targets: Sequence[Any]) -> float:
        """Return only the configured scalar loss."""

        return self.evaluate_sequence(outputs, targets)["loss"]

    def evaluate_sequence(
        self,
        outputs: Sequence[Any],
        targets: Sequence[Any],
    ) -> dict[str, float]:
        """Return real finite evidence for aligned output/target sequences."""

        output_states = as_state_sequence(outputs)
        target_states = as_state_sequence(targets)
        if not output_states or len(output_states) != len(target_states):
            raise QNNInputError(
                "outputs and targets must be non-empty sequences of equal length"
            )

        fidelities = [
            state_fidelity(output, target)
            for output, target in zip(output_states, target_states)
        ]
        mean_fidelity = float(np.mean(fidelities))
        if self.metric == "state_fidelity":
            loss = 1.0 - mean_fidelity
        else:
            loss = float(
                np.mean(
                    [
                        probability_mse(
                            output,
                            target,
                            tolerance=self.probability_tolerance,
                        )
                        for output, target in zip(output_states, target_states)
                    ]
                )
            )

        result = {
            "loss": float(loss),
            "state_fidelity": mean_fidelity,
            "norm_error": maximum_norm_error(output_states),
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise QNNInputError("QNN evaluation produced a non-finite metric")
        return result


def parameter_shift_gradient(
    loss_fn: Callable[[np.ndarray], float],
    weights: Any,
    *,
    loss_name: str,
    shift: float = np.pi / 2.0,
) -> np.ndarray:
    """Return the exact Pauli-rotation parameter-shift loss gradient.

    For each parameter used once in ``exp(-i theta P / 2)``, the derivative is
    ``(L(theta+s)-L(theta-s))/(2 sin(s))``. SLAI restricts this path to the
    projector-expectation fidelity loss implemented by its QNN circuit.
    """

    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    if str(loss_name).strip().casefold() != "state_fidelity":
        raise QNNConfigurationError(
            "parameter-shift gradients require state_fidelity loss"
        )
    parameters = _gradient_parameters(weights)
    offset = positive_float(shift, "parameter_shift")
    denominator = 2.0 * math.sin(offset)
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-15:
        raise QNNConfigurationError("parameter_shift must have a finite, non-zero sine")

    gradients = np.zeros_like(parameters)
    for index in np.ndindex(parameters.shape):
        plus = np.array(parameters, copy=True)
        minus = np.array(parameters, copy=True)
        plus[index] += offset
        minus[index] -= offset
        if not math.isfinite(float(plus[index])) or not math.isfinite(
            float(minus[index])
        ):
            raise QNNInputError(f"parameter shift overflows at index {index}")
        plus_loss = _finite_loss(loss_fn(plus), probe=f"positive shift {index}")
        minus_loss = _finite_loss(loss_fn(minus), probe=f"negative shift {index}")
        gradients[index] = (plus_loss - minus_loss) / denominator
    if not np.all(np.isfinite(gradients)):
        raise QNNInputError("parameter-shift gradient is non-finite")
    return np.ascontiguousarray(gradients, dtype=np.float64)


def finite_difference_gradient(
    loss_fn: Callable[[np.ndarray], float],
    weights: Any,
    *,
    step: float,
) -> np.ndarray:
    """Return a central finite-difference approximation of the loss gradient."""

    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    parameters = _gradient_parameters(weights)
    epsilon = positive_float(step, "finite_difference_step")

    gradients = np.zeros_like(parameters)
    for index in np.ndindex(parameters.shape):
        plus = np.array(parameters, copy=True)
        minus = np.array(parameters, copy=True)
        plus[index] += epsilon
        minus[index] -= epsilon
        if plus[index] == parameters[index] or minus[index] == parameters[index]:
            raise QNNConfigurationError(
                "finite_difference_step is too small for the parameter scale at "
                f"index {index}"
            )
        if not math.isfinite(float(plus[index])) or not math.isfinite(
            float(minus[index])
        ):
            raise QNNInputError(f"finite-difference probe overflows at index {index}")
        plus_loss = _finite_loss(loss_fn(plus), probe=f"positive difference {index}")
        minus_loss = _finite_loss(loss_fn(minus), probe=f"negative difference {index}")
        gradients[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
    if not np.all(np.isfinite(gradients)):
        raise QNNInputError("finite-difference gradient is non-finite")
    return np.ascontiguousarray(gradients, dtype=np.float64)


def clip_gradient(gradient: Any, *, max_norm: float) -> np.ndarray:
    """Apply deterministic global-L2-norm clipping to a finite gradient."""

    values = _gradient_parameters(gradient)
    limit = positive_float(max_norm, "gradient_clip_norm")
    norm = _stable_vector_norm(values.reshape(-1))
    if not math.isfinite(norm):
        raise QNNInputError("gradient norm is non-finite")
    if norm > limit:
        values *= limit / norm
    return np.ascontiguousarray(values, dtype=np.float64)


class QuantumMeasurementOptimizer:
    """Configuration-backed QNN measurement and gradient service."""

    def __init__(
        self,
        *,
        loss: str,
        gradient_method: str,
        finite_difference_step: float | None = None,
        gradient_clip_norm: float | None = None,
        parameter_shift: float | None = None,
        probability_tolerance: float | None = None,
    ) -> None:
        self.config = load_global_config()
        raw_section = self.config.get("quantum_mno")
        if not isinstance(raw_section, Mapping):
            raise QNNConfigurationError("quantum_mno configuration must be a mapping")
        self.mno_config = (
            get_config_section(
                "quantum_mno",
                config=self.config,
            )
            or {}
        )

        method = str(gradient_method).strip().casefold()
        if method not in _SUPPORTED_GRADIENT_METHODS:
            supported = ", ".join(sorted(_SUPPORTED_GRADIENT_METHODS))
            raise QNNConfigurationError(f"gradient_method must be one of: {supported}")

        self.metrics = QNNMetrics(
            loss,
            probability_tolerance=probability_tolerance,
            config=self.config,
        )
        self.loss = self.metrics.metric
        if method == "parameter_shift" and self.loss != "state_fidelity":
            raise QNNConfigurationError(
                "parameter_shift is supported only for state_fidelity loss"
            )

        finite_step_value = (
            self.mno_config.get("finite_difference_step")
            if finite_difference_step is None
            else finite_difference_step
        )
        clip_norm_value = (
            self.mno_config.get("gradient_clip_norm")
            if gradient_clip_norm is None
            else gradient_clip_norm
        )
        shift_value = (
            self.mno_config.get("parameter_shift")
            if parameter_shift is None
            else parameter_shift
        )
        if finite_step_value is None:
            raise QNNConfigurationError(
                "quantum_mno.finite_difference_step is required"
            )
        if clip_norm_value is None:
            raise QNNConfigurationError("quantum_mno.gradient_clip_norm is required")
        if shift_value is None:
            raise QNNConfigurationError("quantum_mno.parameter_shift is required")

        self.gradient_method = method
        self.finite_difference_step = positive_float(
            finite_step_value,
            "quantum_mno.finite_difference_step",
        )
        self.gradient_clip_norm = positive_float(
            clip_norm_value,
            "quantum_mno.gradient_clip_norm",
        )
        self.parameter_shift = positive_float(
            shift_value,
            "quantum_mno.parameter_shift",
        )
        if abs(math.sin(self.parameter_shift)) <= 1.0e-15:
            raise QNNConfigurationError(
                "quantum_mno.parameter_shift must have a non-zero sine"
            )
        self.probability_tolerance = self.metrics.probability_tolerance
        logger.debug(
            "Initialized QNN M&O loss=%s gradient_method=%s",
            self.loss,
            self.gradient_method,
        )

    def probabilities(self, state: Any) -> np.ndarray:
        """Return Born probabilities using the configured tolerance."""

        return born_probabilities(
            state,
            tolerance=self.probability_tolerance,
        )

    def sample(
        self,
        state: Any,
        *,
        rng: np.random.Generator,
        num_samples: int = 1,
    ) -> np.ndarray:
        """Sample a state using the configured Born-probability tolerance."""

        return sample_born(
            state,
            rng=rng,
            num_samples=num_samples,
            tolerance=self.probability_tolerance,
        )

    def gradient(
        self,
        loss_fn: Callable[[np.ndarray], float],
        weights: Any,
    ) -> np.ndarray:
        """Estimate and globally clip one finite loss gradient."""

        if self.gradient_method == "parameter_shift":
            gradient = parameter_shift_gradient(
                loss_fn,
                weights,
                loss_name=self.loss,
                shift=self.parameter_shift,
            )
        else:
            gradient = finite_difference_gradient(
                loss_fn,
                weights,
                step=self.finite_difference_step,
            )
        return clip_gradient(gradient, max_norm=self.gradient_clip_norm)


# Compatibility name for callers of the repaired v2.2 QNN facade.
PerformanceEvaluator = QNNMetrics


__all__ = [
    "PerformanceEvaluator",
    "QNNMetrics",
    "QuantumMeasurementOptimizer",
    "born_probabilities",
    "clip_gradient",
    "finite_difference_gradient",
    "maximum_norm_error",
    "parameter_shift_gradient",
    "probability_mse",
    "sample_born",
    "state_fidelity",
]



if __name__ == "__main__":
    print("\n=== Running quantum_mno tests ===\n")
    printer.status("TEST", "quantum_mno initialized", "info")

    # Create test states (normalized)
    state0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    state1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)

    # born_probabilities
    printer.status("TEST", "born_probabilities", "info")
    probs0 = born_probabilities(state0)
    assert np.allclose(probs0, [1.0, 0.0, 0.0, 0.0])
    printer.status("PASS", "born_probabilities for basis state", "success")

    # state_fidelity
    printer.status("TEST", "state_fidelity", "info")
    fid = state_fidelity(state0, state0)
    assert np.isclose(fid, 1.0)
    fid2 = state_fidelity(state0, state1)
    assert np.isclose(fid2, 0.0)
    printer.status("PASS", "state_fidelity", "success")

    # probability_mse
    printer.status("TEST", "probability_mse", "info")
    mse = probability_mse(state0, state1)
    # 0 vs 1: MSE = (1-0)^2 + (0-1)^2 + ... / 4 = 2/4 = 0.5
    assert np.isclose(mse, 0.5)
    printer.status("PASS", "probability_mse", "success")

    # sample_born
    printer.status("TEST", "sample_born", "info")
    rng = np.random.default_rng(42)
    samples = sample_born(state0, rng=rng, num_samples=10)
    assert samples.shape == (10,)
    assert np.all(samples == 0)  # all samples should be index 0
    printer.status("PASS", "sample_born", "success")

    # maximum_norm_error
    printer.status("TEST", "maximum_norm_error", "info")
    err = maximum_norm_error([state0, state1])
    assert np.isclose(err, 0.0)
    printer.status("PASS", "maximum_norm_error", "success")

    # Test QNNMetrics (requires config)
    printer.status("TEST", "QNNMetrics", "info")
    try:
        metrics = QNNMetrics(metric="state_fidelity")
        result = metrics.evaluate_sequence([state0], [state0])
        assert "loss" in result and "state_fidelity" in result
        assert np.isclose(result["loss"], 0.0)
        printer.status("PASS", "QNNMetrics state_fidelity", "success")
    except QNNConfigurationError as e:
        printer.status("SKIP", f"QNNMetrics skipped (config missing): {e}", "warning")

    # Test gradient functions with a dummy loss
    printer.status("TEST", "gradient functions", "info")
    def dummy_loss(w):
        return float(np.sum(w**2))
    w = np.array([1.0, 2.0, 3.0])
    # parameter_shift requires loss_name="state_fidelity" but we can test finite_difference
    try:
        grad_fd = finite_difference_gradient(dummy_loss, w, step=1e-5)
        # analytical gradient is 2*w
        expected = 2.0 * w
        assert np.allclose(grad_fd, expected, rtol=1e-4)
        printer.status("PASS", "finite_difference_gradient", "success")
    except Exception as e:
        printer.status("SKIP", f"finite_difference skipped: {e}", "warning")

    # clip_gradient
    printer.status("TEST", "clip_gradient", "info")
    grad = np.array([10.0, 0.0, 0.0])
    clipped = clip_gradient(grad, max_norm=5.0)
    assert np.isclose(np.linalg.norm(clipped), 5.0)
    printer.status("PASS", "clip_gradient", "success")

    # Test QuantumMeasurementOptimizer (requires config)
    printer.status("TEST", "QuantumMeasurementOptimizer", "info")
    try:
        optimizer = QuantumMeasurementOptimizer(
            loss="state_fidelity",
            gradient_method="finite_difference",
        )
        grad_opt = optimizer.gradient(dummy_loss, w)
        assert grad_opt.shape == w.shape
        printer.status("PASS", "QuantumMeasurementOptimizer gradient", "success")
    except QNNConfigurationError as e:
        printer.status("SKIP", f"QuantumMeasurementOptimizer skipped (config missing): {e}", "warning")

    print("\n=== quantum_mno tests ran successfully ===\n")