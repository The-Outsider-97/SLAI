"""Supervised binary VQC backed by SLAI's existing state-vector circuit.

This is a CPU-only classical simulation. The circuit is injected by QNNAgent:
the module deliberately does not import QNNAgent or duplicate gate execution.
Training and production artifact promotion are separate operations.
"""
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "slai.qnn.vqc.v1"
MAX_ARTIFACT_BYTES = 1_048_576
MAX_MANIFEST_BYTES = 65_536


def _matrix(values: Any, width: int, *, name: str) -> np.ndarray:
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric features") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width or not 0 < array.shape[0] <= 128:
        raise ValueError(f"{name} must have shape (1..128, {width})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _labels(values: Any, count: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.shape != (count,) or np.iscomplexobj(labels):
        raise ValueError("labels must align with samples and contain only 0/1")
    try:
        floating = labels.astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("labels must contain only 0/1") from exc
    if not np.isfinite(floating).all() or not np.isin(floating, (0.0, 1.0)).all():
        raise ValueError("labels must contain only 0/1")
    if not np.array_equal(labels, floating):
        raise ValueError("labels must contain only 0/1")
    return floating


class VariationalQuantumClassifier:
    """Immutable inference snapshot; fit() mutates only a training candidate.

    Readout is Born P(qubit-0 == 1), not a calibrated success probability.
    Each example has one angle-encoded feature per qubit. Scaling bounds are
    fitted on the training partition, never on inference or evaluation data.
    """

    def __init__(
        self, circuit: Any, weights: Any, feature_min: Any, feature_max: Any,
        *, artifact_id: str = "unpromoted",
    ) -> None:
        if not 1 <= circuit.num_qubits <= 8:
            raise ValueError("binary VQC supports 1..8 qubits")
        self.circuit = circuit
        self.weights = np.array(circuit.validate_weights(weights), copy=True)
        self.feature_min = np.asarray(feature_min, dtype=np.float64).copy()
        self.feature_max = np.asarray(feature_max, dtype=np.float64).copy()
        shape = (circuit.num_qubits,)
        if self.feature_min.shape != shape or self.feature_max.shape != shape:
            raise ValueError(f"feature bounds must have shape {shape}")
        if not np.isfinite(self.feature_min).all() or not np.isfinite(self.feature_max).all():
            raise ValueError("feature bounds must be finite")
        if np.any(self.feature_min > self.feature_max):
            raise ValueError("feature_min must not exceed feature_max")
        self.artifact_id = str(artifact_id)
        self.training_steps = 0

    @classmethod
    def training_candidate(cls, circuit: Any, weights: Any, training_features: Any):
        X = _matrix(training_features, circuit.num_qubits, name="training_features")
        return cls(circuit, weights, X.min(axis=0), X.max(axis=0))

    def _state(self, x: np.ndarray) -> np.ndarray:
        scale = np.where(self.feature_max > self.feature_min,
                         self.feature_max - self.feature_min, 1.0)
        angles = np.pi * np.clip((x - self.feature_min) / scale, 0.0, 1.0)
        state = np.array([1.0 + 0.0j], dtype=np.complex128)
        for theta in angles:
            state = np.kron(state, np.array([math.cos(theta / 2.0),
                                              math.sin(theta / 2.0)], dtype=np.complex128))
        return state

    def _probability(self, state: np.ndarray, weights: np.ndarray) -> float:
        output = self.circuit.forward(state, weights)
        probability = float(np.sum(np.abs(output.reshape(2, -1)[1]) ** 2))
        if not math.isfinite(probability) or probability < -1e-9 or probability > 1 + 1e-9:
            raise ValueError("circuit produced an invalid Born probability")
        return float(np.clip(probability, 0.0, 1.0))

    def _states(self, X: np.ndarray) -> list[np.ndarray]:
        return [self._state(row) for row in X]

    def predict_proba(self, features: Any) -> tuple[np.ndarray, np.ndarray]:
        X = _matrix(features, self.circuit.num_qubits, name="features")
        out_of_range = np.any(
            (X < self.feature_min) | (X > self.feature_max), axis=1
        )
        states = self._states(X)
        probabilities = np.array([self._probability(s, self.weights) for s in states])
        return probabilities, out_of_range

    @staticmethod
    def binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
        p = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p)))

    def fit(
        self, features: Any, labels: Any, *, epochs: int = 30,
        learning_rate: float = 0.05, clip_norm: float = 1.0,
        validation_features: Any = None, validation_labels: Any = None,
        patience: int = 8,
    ) -> dict[str, float | int]:
        X = _matrix(features, self.circuit.num_qubits, name="training_features")
        y = _labels(labels, len(X))
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("training split must contain both classes")
        if type(epochs) is not int or not 1 <= epochs <= 1000:
            raise ValueError("epochs must be within 1..1000")
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be within (0,1]")
        if not math.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if type(patience) is not int or patience < 1:
            raise ValueError("patience must be a positive integer")
        if (validation_features is None) != (validation_labels is None):
            raise ValueError("validation features and labels must be provided together")
        V = None if validation_features is None else _matrix(
            validation_features, self.circuit.num_qubits, name="validation_features")
        vy = None if V is None else _labels(validation_labels, len(V))
        states = self._states(X)
        validation_states = None if V is None else self._states(V)
        best_weights = self.weights.copy()
        best_loss = float("inf")
        stale = 0
        initial = self.binary_log_loss(
            np.array([self._probability(s, self.weights) for s in states]), y)
        for _ in range(epochs):
            p = np.array([self._probability(s, self.weights) for s in states])
            bounded = np.clip(p, 1e-7, 1 - 1e-7)
            chain = (bounded - y) / (bounded * (1 - bounded))
            gradient = np.zeros_like(self.weights)
            for index in np.ndindex(self.weights.shape):
                plus = self.weights.copy()
                minus = self.weights.copy()
                plus[index] += np.pi / 2
                minus[index] -= np.pi / 2
                dp = np.array([
                    (self._probability(s, plus) - self._probability(s, minus)) / 2.0
                    for s in states
                ])
                gradient[index] = float(np.mean(chain * dp))
            norm = float(np.linalg.norm(gradient))
            if not math.isfinite(norm):
                raise ValueError("non-finite gradient")
            if norm > clip_norm:
                gradient *= clip_norm / norm
            self.weights = self.circuit.validate_weights(self.weights - learning_rate * gradient).copy()
            self.training_steps += 1
            monitor_states = states if validation_states is None else validation_states
            monitor_y = y if vy is None else vy
            current_loss = self.binary_log_loss(
                np.array([self._probability(s, self.weights) for s in monitor_states]), monitor_y)
            if current_loss < best_loss - 1e-8:
                best_loss, best_weights, stale = current_loss, self.weights.copy(), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.weights = best_weights
        final = self.binary_log_loss(
            np.array([self._probability(s, self.weights) for s in states]), y)
        return {"initial_train_loss": initial, "train_loss": final,
                "best_monitor_loss": best_loss, "steps": self.training_steps}

    def save(self, path: str | Path, *, provenance: dict[str, Any] | None = None) -> Path:
        target = Path(path)
        if target.suffix != ".npz":
            raise ValueError("VQC artifact path must end in .npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Training is performed on a separate candidate. Promoting this file
        # to the production model remains an explicit deployment decision.
        with target.open("wb") as handle:
            np.savez_compressed(handle, weights=self.weights,
                                feature_min=self.feature_min, feature_max=self.feature_max)
        data = target.read_bytes()
        if len(data) > MAX_ARTIFACT_BYTES:
            target.unlink(missing_ok=True)
            raise ValueError("VQC artifact exceeds size limit")
        digest = hashlib.sha256(data).hexdigest()
        meta = {
            "schema": SCHEMA, "sha256": digest,
            "num_qubits": self.circuit.num_qubits,
            "num_layers": self.circuit.num_layers,
            "entanglement": self.circuit.entanglement,
            "feature_count": self.circuit.num_qubits,
            "classes": [0, 1], "training_steps": self.training_steps,
            "readout": "born_p_qubit0_one", "probabilities_calibrated": False,
            "provenance": provenance or {},
        }
        manifest = json.dumps(meta, sort_keys=True, allow_nan=False)
        if len(manifest.encode("utf-8")) > MAX_MANIFEST_BYTES:
            target.unlink(missing_ok=True)
            raise ValueError("VQC manifest exceeds size limit")
        target.with_suffix(".json").write_text(manifest, encoding="utf-8")
        return target

    @classmethod
    def load(cls, circuit: Any, path: str | Path) -> "VariationalQuantumClassifier":
        target = Path(path)
        if target.suffix != ".npz" or target.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("invalid or oversized VQC artifact")
        manifest_path = target.with_suffix(".json")
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("oversized VQC manifest")
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
            raise ValueError("unsupported VQC manifest schema")
        if (meta.get("num_qubits"), meta.get("num_layers"), meta.get("entanglement")) != (
            circuit.num_qubits, circuit.num_layers, circuit.entanglement):
            raise ValueError("VQC circuit schema mismatch")
        if meta.get("feature_count") != circuit.num_qubits or meta.get("classes") != [0, 1]:
            raise ValueError("VQC feature or label schema mismatch")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != meta.get("sha256"):
            raise ValueError("VQC artifact checksum mismatch")
        with zipfile.ZipFile(target) as archive:
            infos = archive.infolist()
            if len(infos) != 3 or sum(entry.file_size for entry in infos) > MAX_ARTIFACT_BYTES:
                raise ValueError("unexpected or oversized VQC archive")
            if set(entry.filename for entry in infos) != {
                "weights.npy", "feature_min.npy", "feature_max.npy"}:
                raise ValueError("unexpected VQC archive entries")
        with np.load(target, allow_pickle=False) as bundle:
            model = cls(circuit, bundle["weights"], bundle["feature_min"],
                        bundle["feature_max"], artifact_id=digest[:16])
        steps = meta.get("training_steps")
        if type(steps) is not int or steps < 1:
            raise ValueError("VQC artifact lacks verified training steps")
        model.training_steps = steps
        return model
