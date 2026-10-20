"""Supervised binary VQC backed by SLAI's existing state-vector circuit.

This is a CPU-only classical simulation. The circuit is injected by QNNAgent:
the module deliberately does not import QNNAgent or duplicate gate execution.
Training and production artifact promotion are separate operations.

Design notes
------------
* Readout is the Born probability ``P(readout qubit == 1)``; it is *not* a
  calibrated success probability and is reported as such in every artifact
  manifest. The readout qubit defaults to the LAST qubit: StateVectorCircuit's
  CNOT entanglers always use the lower-indexed qubit as control, so qubit 0's
  computational-basis marginal can never depend on any other qubit's feature.
  ``reachable_features`` verifies against the real circuit that every feature
  can influence the readout, and ``fit`` refuses to train a silently
  univariate model.
* Every example is angle-encoded with one feature per qubit. Scaling bounds are
  fitted on the training partition only, never on inference or evaluation data.
* All numerical work goes through ``circuit.forward``. When a batch is at least
  as large as the state dimension, the circuit is applied once to each basis
  state and the resulting unitary evaluates the whole batch with one matrix
  product; this is mathematically identical to per-example forward passes (the
  circuit is linear in the state) but far cheaper for training partitions.
* Resource limits (rows, epochs, gradient evaluations, working set, artifact
  size) come from ``quantum_config.yaml`` (``vqc``, ``quantum_policy`` and
  ``quantum_mno`` sections); nothing is hard-coded here.
* Failures raise the typed ``QNN*`` taxonomy (all ``ValueError`` subclasses) and
  are logged once, at the public-method boundary that rejects them.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import tempfile
import zipfile
import numpy as np # type: ignore

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .quantum_mno import clip_gradient
from .utils.config_loader import get_config_section, load_global_config
from .utils.quantum_errors import *
from .utils.quantum_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Variational Quantum Classifier")
printer = PrettyPrinter()


SCHEMA = "slai.qnn.vqc.v1"
_ARCHIVE_MEMBERS = frozenset({"weights.npy", "feature_min.npy", "feature_max.npy"})
_OPTIMIZERS = frozenset({"adam", "sgd"})
_CLASS_WEIGHTS = frozenset({"balanced"})
_MAX_LEARNING_RATE = 1.0
_MIN_IMPROVEMENT = 1e-8
_PROGRESS_EVERY = 10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _resolve_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the QNN configuration mapping, loading the global file if needed."""
    if config is not None:
        if not isinstance(config, Mapping):
            raise QNNConfigurationError("VQC config override must be a mapping")
        return dict(config)
    try:
        return load_global_config()
    except (OSError, TypeError, ValueError) as exc:
        raise QNNConfigurationError(f"unable to load QNN configuration: {exc}") from exc


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(config.get(name), Mapping):
        raise QNNConfigurationError(f"{name} configuration must be a mapping")
    return get_config_section(name, config=dict(config)) or {}


def _required(section: Mapping[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise QNNConfigurationError(f"{section_name}.{key} is required")
    return section[key]


@dataclass(frozen=True, slots=True)
class VQCLimits:
    """Validated resource envelope assembled from the shared QNN configuration."""

    max_qubits: int
    max_training_rows: int
    max_inference_rows: int
    max_training_steps: int
    max_gradient_evaluations: int
    max_working_set_bytes: int
    max_artifact_bytes: int
    max_manifest_bytes: int
    probability_epsilon: float
    norm_tolerance: float
    parameter_shift: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VQCLimits":
        vqc = _section(config, "vqc")
        policy = _section(config, "quantum_policy")
        mno = _section(config, "quantum_mno")

        def integer(section: Mapping[str, Any], section_name: str, key: str) -> int:
            return positive_int(_required(section, section_name, key), f"{section_name}.{key}")

        def real(section: Mapping[str, Any], section_name: str, key: str) -> float:
            return positive_float(_required(section, section_name, key), f"{section_name}.{key}")

        limits = cls(
            max_qubits=integer(vqc, "vqc", "max_qubits"),
            max_training_rows=integer(vqc, "vqc", "max_training_rows"),
            max_inference_rows=integer(policy, "quantum_policy", "max_sequence_length"),
            max_training_steps=integer(policy, "quantum_policy", "max_training_steps"),
            max_gradient_evaluations=integer(policy, "quantum_policy", "max_gradient_evaluations"),
            max_working_set_bytes=integer(policy, "quantum_policy", "max_working_set_bytes"),
            max_artifact_bytes=integer(vqc, "vqc", "max_artifact_bytes"),
            max_manifest_bytes=integer(vqc, "vqc", "max_manifest_bytes"),
            probability_epsilon=real(vqc, "vqc", "probability_epsilon"),
            norm_tolerance=real(vqc, "vqc", "norm_tolerance"),
            parameter_shift=real(mno, "quantum_mno", "parameter_shift"),
        )
        if not 0.0 < limits.probability_epsilon < 0.5:
            raise QNNConfigurationError("vqc.probability_epsilon must be within (0, 0.5)")
        if abs(math.sin(limits.parameter_shift)) < 1e-6:
            raise QNNConfigurationError("quantum_mno.parameter_shift must not be a multiple of pi")
        return limits


def _provenance_hook(value: Any) -> Any:
    """``json_safe`` hook: provenance may carry paths; everything else is rejected."""
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported provenance value {type(value).__name__}")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` next to ``path`` and ``os.replace`` it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _read_bounded(path: Path, limit: int, what: str) -> bytes:
    """Read a whole file, refusing anything above ``limit`` bytes."""
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except FileNotFoundError as exc:
        raise QNNInputError(f"{what} not found: {path.name}") from exc
    except OSError as exc:
        raise QNNInputError(f"{what} is unreadable ({type(exc).__name__}): {path.name}") from exc
    if len(data) > limit:
        raise QNNResourceLimitError(f"{what} exceeds the {limit}-byte limit: {path.name}")
    return data


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class VariationalQuantumClassifier:
    """Immutable inference snapshot; fit() mutates only a training candidate.

    Readout is Born P(readout qubit == 1), not a calibrated success probability.
    Each example has one angle-encoded feature per qubit. Scaling bounds are
    fitted on the training partition, never on inference or evaluation data.
    """

    def __init__(
        self,
        circuit: Any,
        weights: Any,
        feature_min: Any,
        feature_max: Any,
        *,
        config: Mapping[str, Any] | None = None,
        artifact_id: str = "unpromoted",
        bounds_quantile: float = 0.0,
        readout_qubit: int | None = None,
    ) -> None:
        self.config = _resolve_config(config)
        self.limits = VQCLimits.from_config(self.config)
        self.num_qubits = self._require_circuit(circuit, self.limits)
        self.circuit = circuit
        self.weights = np.array(circuit.validate_weights(weights), dtype=np.float64, copy=True)

        shape = (self.num_qubits,)
        self.feature_min = validate_finite(
            validate_shape(as_float64(feature_min, name="feature_min"), shape, name="feature_min"),
            name="feature_min",
        ).copy()
        self.feature_max = validate_finite(
            validate_shape(as_float64(feature_max, name="feature_max"), shape, name="feature_max"),
            name="feature_max",
        ).copy()
        if np.any(self.feature_min > self.feature_max):
            raise QNNInputError("feature_min must not exceed feature_max")

        quantile = float(bounds_quantile)
        if not math.isfinite(quantile) or not 0.0 <= quantile < 0.5:
            raise QNNConfigurationError("bounds_quantile must be within [0, 0.5)")
        self.bounds_quantile = quantile
        if readout_qubit is None:
            readout_qubit = self.num_qubits - 1
        if isinstance(readout_qubit, bool) or not isinstance(readout_qubit, (int, np.integer)) \
                or not 0 <= int(readout_qubit) < self.num_qubits:
            raise QNNConfigurationError(f"readout_qubit must be an integer within [0, {self.num_qubits})")
        self.readout_qubit = int(readout_qubit)
        self.artifact_id = str(artifact_id)
        self.training_steps = 0
        self._dimension = 1 << self.num_qubits

        logger.debug(
            "Initialized VQC id=%s qubits=%d layers=%d parameters=%d readout_qubit=%d",
            self.artifact_id,
            self.num_qubits,
            self.circuit.num_layers,
            self.weights.size,
            self.readout_qubit,
        )

    def __repr__(self) -> str:
        return (
            f"VariationalQuantumClassifier(id={self.artifact_id!r}, qubits={self.num_qubits}, "
            f"layers={self.circuit.num_layers}, training_steps={self.training_steps})"
        )

    @staticmethod
    def _require_circuit(circuit: Any, limits: VQCLimits) -> int:
        missing = [name for name in ("num_qubits", "num_layers", "entanglement") if not hasattr(circuit, name)]
        missing += [name for name in ("forward", "validate_weights") if not callable(getattr(circuit, name, None))]
        if missing:
            raise QNNConfigurationError(f"circuit must provide {', '.join(missing)} (expected a StateVectorCircuit-compatible object)")
        qubits = positive_int(circuit.num_qubits, "circuit.num_qubits")
        if qubits > limits.max_qubits:
            raise QNNResourceLimitError(f"binary VQC supports 1..{limits.max_qubits} qubits, got {qubits}")
        return qubits

    @classmethod
    def training_candidate(
        cls,
        circuit: Any,
        weights: Any,
        training_features: Any,
        *,
        config: Mapping[str, Any] | None = None,
        bounds_quantile: float = 0.0,
        readout_qubit: int | None = None,
    ) -> "VariationalQuantumClassifier":
        """Build a candidate whose scaling bounds are fitted on ``training_features``.

        ``bounds_quantile=0`` uses the exact training min/max. A value ``q`` in
        (0, 0.5) uses the ``q`` / ``1-q`` training quantiles instead, so heavy
        tails do not compress the bulk of the data into a narrow angle range;
        inference clips to the same bounds, so train and serve stay identical.
        """
        resolved = _resolve_config(config)
        limits = VQCLimits.from_config(resolved)
        qubits = cls._require_circuit(circuit, limits)
        quantile = float(bounds_quantile)
        if not math.isfinite(quantile) or not 0.0 <= quantile < 0.5:
            raise QNNConfigurationError("bounds_quantile must be within [0, 0.5)")
        X = _matrix(training_features, qubits, name="training_features", max_rows=limits.max_training_rows) # type: ignore
        if quantile == 0.0:
            lower, upper = X.min(axis=0), X.max(axis=0)
        else:
            lower, upper = np.quantile(X, [quantile, 1.0 - quantile], axis=0)
        return cls(
            circuit, weights, lower, upper, config=resolved,
            bounds_quantile=quantile, readout_qubit=readout_qubit,
        )

    # ------------------------------------------------------------------
    # Encoding and circuit evaluation
    # ------------------------------------------------------------------
    def _encode(self, X: np.ndarray) -> np.ndarray:
        """Angle-encode rows as product states; qubit 0 is the most significant axis."""
        scale = np.where(self.feature_max > self.feature_min, self.feature_max - self.feature_min, 1.0)
        return self._product_states(np.pi * np.clip((X - self.feature_min) / scale, 0.0, 1.0))

    def _product_states(self, angles: np.ndarray) -> np.ndarray:
        states = np.ones((angles.shape[0], 1), dtype=np.complex128)
        for qubit in range(self.num_qubits):
            single = np.stack([np.cos(angles[:, qubit] / 2.0), np.sin(angles[:, qubit] / 2.0)], axis=1).astype(np.complex128)
            states = (states[:, :, None] * single[:, None, :]).reshape(angles.shape[0], -1)
        return states

    def _forward(self, state: np.ndarray, weights: np.ndarray) -> np.ndarray:
        output = np.asarray(self.circuit.forward(state, weights), dtype=np.complex128)
        if output.shape != (self._dimension,):
            raise QNNStateError(f"circuit returned shape {output.shape}, expected ({self._dimension},)")
        return output

    def _evolve(self, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply the circuit to every row of ``states`` via ``circuit.forward``."""
        if states.shape[0] >= self._dimension:
            basis = np.eye(self._dimension, dtype=np.complex128)
            unitary = np.stack([self._forward(basis[k], weights) for k in range(self._dimension)], axis=1)
            return states @ unitary.T
        return np.stack([self._forward(row, weights) for row in states], axis=0)

    def _probabilities(self, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Born ``P(readout qubit == 1)`` per encoded state (norm drift is reported, not hidden)."""
        outputs = self._evolve(states, weights)
        if not np.all(np.isfinite(outputs.real)) or not np.all(np.isfinite(outputs.imag)):
            raise QNNFiniteValueError("circuit produced non-finite amplitudes")
        drift = float(np.max(np.abs(np.linalg.norm(outputs, axis=1) - 1.0)))
        if drift > self.limits.norm_tolerance:
            raise QNNNormError(
                f"circuit output norm drift {drift:.3e} exceeds tolerance {self.limits.norm_tolerance:.1e}"
            )
        marginal = (np.abs(outputs) ** 2).reshape((outputs.shape[0],) + (2,) * self.num_qubits)
        probabilities = np.take(marginal, 1, axis=1 + self.readout_qubit).reshape(outputs.shape[0], -1).sum(axis=1)
        if np.any(probabilities < -1e-9) or np.any(probabilities > 1.0 + 1e-9):
            raise QNNStateError("circuit produced an invalid Born probability")
        return np.clip(probabilities, 0.0, 1.0)

    def _predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out_of_range = np.any((X < self.feature_min) | (X > self.feature_max), axis=1)
        probabilities = self._probabilities(self._encode(X), self.weights)
        logger.debug(
            "VQC predicted %d row(s), %d outside fitted bounds", X.shape[0], int(out_of_range.sum())
        )
        return probabilities, out_of_range

    def predict_proba(self, features: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(P(label=1), outside_fitted_bounds)`` for each feature row.

        Serving path: the batch is capped at ``quantum_policy.max_sequence_length``.
        """
        X = _matrix(features, self.num_qubits, name="features", max_rows=self.limits.max_inference_rows) # type: ignore
        return self._predict(X)

    def predict_proba_batched(self, features: Any) -> tuple[np.ndarray, np.ndarray]:
        """``predict_proba`` for offline evaluation partitions (``vqc.max_training_rows`` cap)."""
        X = _matrix(features, self.num_qubits, name="features", max_rows=self.limits.max_training_rows) # type: ignore
        self._check_working_set(X.shape[0])
        return self._predict(X)

    def _check_working_set(self, rows: int) -> None:
        required = 4 * rows * self._dimension * np.dtype(np.complex128).itemsize
        if required > self.limits.max_working_set_bytes:
            raise QNNResourceLimitError(
                f"VQC working set {required} bytes exceeds limit {self.limits.max_working_set_bytes}"
            )

    def reachable_features(self, *, trials: int = 4, threshold: float = 1e-6) -> np.ndarray:
        """Boolean mask of features that can influence the readout of *this* circuit.

        Probes the injected circuit itself (random weights, two angles per
        feature) instead of re-deriving its entangler topology, so it stays
        correct if the circuit implementation changes.
        """
        rng = np.random.default_rng(0)
        reachable = np.zeros(self.num_qubits, dtype=bool)
        for _ in range(trials):
            weights = np.array(
                self.circuit.validate_weights(rng.uniform(-np.pi, np.pi, self.weights.shape)),
                dtype=np.float64,
            )
            base = rng.uniform(0.2 * np.pi, 0.8 * np.pi, self.num_qubits)
            for qubit in range(self.num_qubits):
                low, high = base.copy(), base.copy()
                low[qubit], high[qubit] = 0.1 * np.pi, 0.9 * np.pi
                pair = self._probabilities(self._product_states(np.stack([low, high])), weights)
                reachable[qubit] |= abs(float(pair[1] - pair[0])) > threshold
        return reachable

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @staticmethod
    def binary_log_loss(
        probabilities: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        epsilon: float = 1e-7,
    ) -> float:
        p = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0 - epsilon)
        y = np.asarray(labels, dtype=np.float64)
        losses = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
        if sample_weight is None:
            return float(np.mean(losses))
        return float(np.mean(np.asarray(sample_weight, dtype=np.float64) * losses))

    @staticmethod
    def _sample_weights(labels: np.ndarray, mode: str | None) -> np.ndarray:
        """Per-example weights with mean 1; ``balanced`` equalises class influence."""
        weights = np.ones_like(labels, dtype=np.float64)
        if mode is None:
            return weights
        positives = int(labels.sum())
        negatives = labels.size - positives
        if positives == 0 or negatives == 0:
            return weights
        weights[labels == 1.0] = labels.size / (2.0 * positives)
        weights[labels == 0.0] = labels.size / (2.0 * negatives)
        return weights

    def _gradient(
        self, states: np.ndarray, chain: np.ndarray
    ) -> np.ndarray:
        """Parameter-shift gradient of the (weighted) log-loss w.r.t. every weight.

        Exact for rotation weights that each appear once as ``exp(-i*theta*P/2)``,
        which holds for StateVectorCircuit's Rx/Ry/Rz layers.
        """
        shift = self.limits.parameter_shift
        denominator = 2.0 * math.sin(shift)
        gradient = np.zeros_like(self.weights)
        for index in np.ndindex(self.weights.shape):
            plus = self.weights.copy()
            minus = self.weights.copy()
            plus[index] += shift
            minus[index] -= shift
            derivative = (
                self._probabilities(states, plus) - self._probabilities(states, minus)
            ) / denominator
            gradient[index] = float(np.mean(chain * derivative))
        return gradient

    def fit(
        self,
        features: Any,
        labels: Any,
        *,
        epochs: int = 100,
        learning_rate: float = 0.1,
        clip_norm: float = 1.0,
        validation_features: Any = None,
        validation_labels: Any = None,
        patience: int = 15,
        optimizer: str = "adam",
        class_weight: str | None = None,
    ) -> dict[str, float | int | bool | str]:
        """Full-batch training with early stopping on validation (or training) loss.

        The best-monitored weights are kept, including the initial weights if no
        epoch improves on them. On any failure the weights and step counter are
        restored, so a failed ``fit`` never leaves a half-updated candidate.
        """
        limits = self.limits
        X = _matrix(features, self.num_qubits, name="training_features", max_rows=limits.max_training_rows) # type: ignore
        y = _labels(labels, len(X), name="training_labels") # type: ignore
        if set(np.unique(y)) != {0.0, 1.0}:
            raise QNNInputError("training split must contain both classes")
        epochs = positive_int(epochs, "epochs")
        if epochs > limits.max_training_steps:
            raise QNNResourceLimitError(
                f"epochs {epochs} exceeds quantum_policy.max_training_steps={limits.max_training_steps}"
            )
        evaluations = epochs * 2 * self.weights.size
        if evaluations > limits.max_gradient_evaluations:
            raise QNNResourceLimitError(
                f"fit needs {evaluations} gradient evaluations; limit is {limits.max_gradient_evaluations}"
            )
        learning_rate = positive_float(learning_rate, "learning_rate")
        if learning_rate > _MAX_LEARNING_RATE:
            raise QNNConfigurationError(f"learning_rate must be within (0, {_MAX_LEARNING_RATE}]")
        clip_norm = positive_float(clip_norm, "clip_norm")
        patience = positive_int(patience, "patience")
        optimizer = str(optimizer).strip().casefold()
        if optimizer not in _OPTIMIZERS:
            raise QNNConfigurationError(f"optimizer must be one of: {', '.join(sorted(_OPTIMIZERS))}")
        if class_weight is not None and class_weight not in _CLASS_WEIGHTS:
            raise QNNConfigurationError(f"class_weight must be None or one of: {', '.join(sorted(_CLASS_WEIGHTS))}")
        if (validation_features is None) != (validation_labels is None):
            raise QNNInputError("validation features and labels must be provided together")
        self._check_working_set(len(X))
        reachable = self.reachable_features()
        if not reachable.all():
            blind = [int(i) for i in np.flatnonzero(~reachable)]
            raise QNNConfigurationError(
                f"readout qubit {self.readout_qubit} cannot depend on feature(s) {blind} with "
                f"entanglement={self.circuit.entanglement!r}; training would silently ignore them"
            )

        V = vy = None
        if validation_features is not None:
            V = _matrix(
                validation_features, self.num_qubits, name="validation_features",
                max_rows=limits.max_training_rows, # type: ignore
            )
            vy = _labels(validation_labels, len(V), name="validation_labels") # type: ignore
            self._check_working_set(len(V))

        snapshot = (self.weights.copy(), self.training_steps)
        logger.info(
            "VQC fit start rows=%d validation_rows=%d parameters=%d epochs=%d optimizer=%s lr=%g class_weight=%s",
            len(X), 0 if V is None else len(V), self.weights.size, epochs, optimizer,
            learning_rate, class_weight,
        )
        try:
            stats = self._fit_loop(
                X, y, V, vy, epochs=epochs, learning_rate=learning_rate, clip_norm=clip_norm,
                patience=patience, optimizer=optimizer, class_weight=class_weight,
            )
        except QNNError as exc:
            self.weights, self.training_steps = snapshot
            logger.error("VQC fit aborted; weights restored: %s", exc)
            raise
        except Exception as exc:
            self.weights, self.training_steps = snapshot
            logger.error("VQC fit aborted; weights restored: %s: %s", type(exc).__name__, exc)
            raise QNNGradientError(f"VQC training failed ({type(exc).__name__}): {exc}") from exc
        except BaseException:
            self.weights, self.training_steps = snapshot
            raise
        logger.info(
            "VQC fit done epochs_run=%d stopped_early=%s train_loss=%.6f best_monitor_loss=%.6f",
            stats["epochs_run"], stats["stopped_early"], stats["train_loss"], stats["best_monitor_loss"],
        )
        return stats

    def _fit_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        V: np.ndarray | None,
        vy: np.ndarray | None,
        *,
        epochs: int,
        learning_rate: float,
        clip_norm: float,
        patience: int,
        optimizer: str,
        class_weight: str | None,
    ) -> dict[str, float | int | bool | str]:
        epsilon = self.limits.probability_epsilon
        states = self._encode(X)
        weights_y = self._sample_weights(y, class_weight)
        monitor_states = states if V is None else self._encode(V)
        monitor_y = y if vy is None else vy
        monitor_weights = weights_y if vy is None else self._sample_weights(vy, class_weight)

        def monitor_loss() -> float:
            return self.binary_log_loss(
                self._probabilities(monitor_states, self.weights), monitor_y,
                sample_weight=monitor_weights, epsilon=epsilon,
            )

        initial_train = self.binary_log_loss(
            self._probabilities(states, self.weights), y, sample_weight=weights_y, epsilon=epsilon
        )
        best_loss = monitor_loss() if V is not None else initial_train
        initial_monitor = best_loss
        best_weights = self.weights.copy()
        best_epoch = 0
        stale = 0
        first_moment = np.zeros_like(self.weights)
        second_moment = np.zeros_like(self.weights)
        epochs_run = 0
        stopped_early = False

        for epoch in range(1, epochs + 1):
            p = np.clip(self._probabilities(states, self.weights), epsilon, 1.0 - epsilon)
            chain = weights_y * (p - y) / (p * (1.0 - p))
            gradient = self._gradient(states, chain)
            if not np.all(np.isfinite(gradient)):
                raise QNNGradientError("non-finite gradient")
            gradient = clip_gradient(gradient, max_norm=clip_norm)

            if optimizer == "adam":
                first_moment = 0.9 * first_moment + 0.1 * gradient
                second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
                corrected_first = first_moment / (1.0 - 0.9 ** epoch)
                corrected_second = second_moment / (1.0 - 0.999 ** epoch)
                update = learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            else:
                update = learning_rate * gradient
            self.weights = np.array(
                self.circuit.validate_weights(self.weights - update), dtype=np.float64, copy=True
            )
            self.training_steps += 1
            epochs_run = epoch

            current = monitor_loss()
            if not math.isfinite(current):
                raise QNNGradientError("non-finite monitor loss")
            if current < best_loss - _MIN_IMPROVEMENT:
                best_loss, best_weights, best_epoch, stale = current, self.weights.copy(), epoch, 0
            else:
                stale += 1
            if epoch % _PROGRESS_EVERY == 0:
                logger.info(
                    "VQC epoch %d/%d monitor_loss=%.6f best=%.6f stale=%d",
                    epoch, epochs, current, best_loss, stale,
                )
            if stale >= patience:
                stopped_early = True
                logger.info("VQC early stop at epoch %d (best epoch %d)", epoch, best_epoch)
                break

        self.weights = best_weights
        final = self.binary_log_loss(
            self._probabilities(states, self.weights), y, sample_weight=weights_y, epsilon=epsilon
        )
        stats: dict[str, float | int | bool | str] = {
            "initial_train_loss": initial_train,
            "train_loss": final,
            "initial_monitor_loss": initial_monitor,
            "best_monitor_loss": best_loss,
            "monitor_split": "train" if V is None else "validation",
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "optimizer": optimizer,
            "readout_qubit": self.readout_qubit,
            "steps": self.training_steps,
        }
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path, *, provenance: Mapping[str, Any] | None = None) -> Path:
        """Atomically write ``<name>.npz`` plus its ``<name>.json`` manifest.

        Size limits are enforced on the in-memory payloads, so an oversized
        artifact never reaches the disk. The manifest carries the SHA-256 that
        ``load`` verifies before any array is parsed.
        """
        target = Path(path)
        try:
            if target.suffix != ".npz":
                raise QNNInputError("VQC artifact path must end in .npz")
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer, weights=self.weights, feature_min=self.feature_min, feature_max=self.feature_max
            )
            data = buffer.getvalue()
            if len(data) > self.limits.max_artifact_bytes:
                raise QNNResourceLimitError("VQC artifact exceeds size limit")
            try:
                safe_provenance = json_safe(dict(provenance or {}), object_hook=_provenance_hook)
            except TypeError as exc:
                raise QNNInputError(f"VQC provenance is not JSON-compatible: {exc}") from exc
            meta = {
                "schema": SCHEMA,
                "sha256": hashlib.sha256(data).hexdigest(),
                "num_qubits": self.num_qubits,
                "num_layers": self.circuit.num_layers,
                "entanglement": self.circuit.entanglement,
                "feature_count": self.num_qubits,
                "classes": [0, 1],
                "training_steps": self.training_steps,
                "bounds_quantile": self.bounds_quantile,
                "readout": f"born_p_qubit{self.readout_qubit}_one",
                "readout_qubit": self.readout_qubit,
                "probabilities_calibrated": False,
                "provenance": safe_provenance,
            }
            try:
                manifest = json.dumps(meta, sort_keys=True, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise QNNInputError(f"VQC manifest is not JSON-serialisable: {exc}") from exc
            if len(manifest) > self.limits.max_manifest_bytes:
                raise QNNResourceLimitError("VQC manifest exceeds size limit")
            # Training is performed on a separate candidate. Promoting this file
            # to the production model remains an explicit deployment decision.
            _atomic_write_bytes(target, data)
            _atomic_write_bytes(target.with_suffix(".json"), manifest)
        except QNNError as exc:
            logger.error("VQC save rejected (%s): %s", target.name, exc)
            raise
        except OSError as exc:
            logger.error("VQC save failed (%s): %s", target.name, exc)
            raise QNNCheckpointStateError(f"unable to write VQC artifact {target.name}: {exc}") from exc
        logger.info(
            "Saved VQC artifact %s sha256=%s steps=%d", target.name, meta["sha256"][:16], self.training_steps
        )
        return target

    @classmethod
    def load(cls, circuit: Any, path: str | Path, *, config: Mapping[str, Any] | None = None) -> "VariationalQuantumClassifier":
        """Verify and load an artifact; every rejection is a typed ``QNNError``."""
        target = Path(path)
        logger.info("Loading VQC artifact %s", target.name)
        try:
            model = cls._load_verified(circuit, target, config)
        except QNNError as exc:
            logger.error("VQC load rejected (%s): %s", target.name, exc)
            raise
        logger.info(
            "Loaded VQC id=%s qubits=%d layers=%d training_steps=%d",
            model.artifact_id, model.num_qubits, circuit.num_layers, model.training_steps,
        )
        return model

    @classmethod
    def _load_verified(cls, circuit: Any, target: Path, config: Mapping[str, Any] | None) -> "VariationalQuantumClassifier":
        resolved = _resolve_config(config)
        limits = VQCLimits.from_config(resolved)
        cls._require_circuit(circuit, limits)
        if target.suffix != ".npz":
            raise QNNInputError("VQC artifact path must end in .npz")

        # Read each file exactly once so the verified bytes are the parsed bytes.
        raw_manifest = _read_bounded(target.with_suffix(".json"), limits.max_manifest_bytes, "VQC manifest")
        raw_artifact = _read_bounded(target, limits.max_artifact_bytes, "VQC artifact")
        try:
            meta = json.loads(raw_manifest.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise QNNCheckpointStateError("VQC manifest is not valid UTF-8 JSON") from exc
        if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
            raise QNNCheckpointStateError("unsupported VQC manifest schema")
        expected = (circuit.num_qubits, circuit.num_layers, circuit.entanglement)
        observed = (meta.get("num_qubits"), meta.get("num_layers"), meta.get("entanglement"))
        if observed != expected:
            raise QNNCheckpointStateError(f"VQC circuit schema mismatch: artifact={observed}, circuit={expected}")
        if meta.get("feature_count") != circuit.num_qubits or meta.get("classes") != [0, 1]:
            raise QNNCheckpointStateError("VQC feature or label schema mismatch")
        digest = hashlib.sha256(raw_artifact).hexdigest()
        recorded = meta.get("sha256")
        if not isinstance(recorded, str) or not hmac.compare_digest(digest, recorded):
            raise QNNCheckpointStateError("VQC artifact checksum mismatch")
        steps = meta.get("training_steps")
        if type(steps) is not int or steps < 1:
            raise QNNCheckpointStateError("VQC artifact lacks verified training steps")
        quantile = meta.get("bounds_quantile", 0.0)
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
            raise QNNCheckpointStateError("VQC manifest bounds_quantile must be numeric")
        readout_qubit = meta.get("readout_qubit")
        if type(readout_qubit) is not int or meta.get("readout") != f"born_p_qubit{readout_qubit}_one":
            raise QNNCheckpointStateError("VQC manifest readout is missing or inconsistent")

        try:
            with zipfile.ZipFile(io.BytesIO(raw_artifact)) as archive:
                infos = archive.infolist()
                names = [entry.filename for entry in infos]
                if len(infos) != 3 or len(set(names)) != 3 or set(names) != _ARCHIVE_MEMBERS:
                    raise QNNCheckpointStateError("unexpected VQC archive entries")
                if sum(entry.file_size for entry in infos) > limits.max_artifact_bytes:
                    raise QNNResourceLimitError("unexpected or oversized VQC archive")
            with np.load(io.BytesIO(raw_artifact), allow_pickle=False) as bundle:
                weights, feature_min, feature_max = (bundle["weights"], bundle["feature_min"], bundle["feature_max"])
        except zipfile.BadZipFile as exc:
            raise QNNCheckpointStateError("VQC artifact is not a valid archive") from exc
        except (ValueError, OSError, KeyError, EOFError) as exc:
            raise QNNCheckpointStateError(f"VQC artifact arrays are unreadable: {exc}") from exc

        model = cls(
            circuit, weights, feature_min, feature_max,
            config=resolved, artifact_id=digest[:16], bounds_quantile=float(quantile),
            readout_qubit=readout_qubit,
        )
        model.training_steps = steps
        return model


__all__ = ["VariationalQuantumClassifier", "VQCLimits", "SCHEMA"]


if __name__ == "__main__":
    print("\n=== Running Variational Quantum Classifier ===\n")
    def _self_test() -> None:
        """Synthetic-data smoke test of the classifier mechanics (not a trained model)."""
        import tempfile as _tempfile
        from src.agents.qnn_agent import StateVectorCircuit # pyright: ignore[reportMissingImports]

        printer.status("TEST", "Variational Quantum Classifier initialized", "info")
        circuit = StateVectorCircuit(num_qubits=2, num_layers=1, entanglement="linear")
        rng = np.random.default_rng(7)
        X = rng.uniform(-1.0, 1.0, (60, 2))
        y = (X[:, 0] + 0.25 * X[:, 1] > 0.0).astype(int)
        weights = rng.uniform(-0.3, 0.3, circuit.weight_shape)
        vqc = VariationalQuantumClassifier.training_candidate(circuit, weights, X)

        stats = vqc.fit(X, y, epochs=25, learning_rate=0.2)
        assert stats["train_loss"] < stats["initial_train_loss"], "training did not reduce loss" # type: ignore
        printer.status("PASS", f"fit reduced loss {stats['initial_train_loss']:.4f} -> {stats['train_loss']:.4f}", "success")

        probabilities, outside = vqc.predict_proba(X[:5])
        assert probabilities.shape == (5,) and not outside.any()
        printer.status("PASS", "predict_proba shape and bounds", "success")

        with _tempfile.TemporaryDirectory() as directory:
            artifact = vqc.save(Path(directory) / "selftest.npz", provenance={"purpose": "self-test"})
            reloaded = VariationalQuantumClassifier.load(circuit, artifact)
            assert np.allclose(reloaded.predict_proba(X[:5])[0], probabilities)
            printer.status("PASS", "save/load round trip", "success")

            artifact.write_bytes(artifact.read_bytes() + b"x")
            try:
                VariationalQuantumClassifier.load(circuit, artifact)
            except QNNCheckpointStateError:
                printer.status("PASS", "tampered artifact rejected", "success")
            else:
                raise AssertionError("tampered artifact was accepted")

        try:
            vqc.predict_proba([[0.0]])
        except QNNDimensionError:
            printer.status("PASS", "wrong feature width rejected", "success")
        else:
            raise AssertionError("wrong feature width was accepted")

    _self_test()
    printer.status("TEST", "Variational Quantum Classifier checks passed", "success")
    print("\n=== Test ran successfully ===\n")
