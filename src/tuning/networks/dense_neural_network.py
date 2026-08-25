"""Deterministic fully connected neural network used by SLAI tuning.

The name describes the architecture rather than the search strategy.  A dense
network may be tuned by grid search, Bayesian optimization, or any future
strategy without changing this module.

The implementation is NumPy-only and deliberately exposes explicit state and
configuration boundaries.  It supports regression, binary classification,
and multiclass classification; mini-batch Adam; dropout; L2 regularization;
gradient clipping; validation-based early stopping; and resumable state.
"""

from __future__ import annotations

import copy
import math
import numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..utils.tuning_errors import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Dense Neural Network")
printer = PrettyPrinter()


Array = np.ndarray


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TuningConfigError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TuningConfigError(f"{name} must be finite.")
    return numeric


def _stable_sigmoid(values: Array) -> Array:
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _softmax(values: Array) -> Array:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class DenseNetworkConfig:
    task_type: str = "regression"
    learning_rate: float = 1.0e-3
    hidden_activation: str = "relu"
    leaky_relu_slope: float = 0.01
    weight_init_scale: float = 1.0
    gradient_clip_norm: float | None = 5.0
    l2_lambda: float = 1.0e-4
    dropout_rate: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    prediction_threshold: float = 0.5
    stability_epsilon: float = 1.0e-8
    random_state: int | None = None

    def __post_init__(self) -> None:
        task = str(self.task_type).strip().casefold()
        if task not in {
            "regression",
            "binary_classification",
            "multiclass_classification",
        }:
            raise TuningConfigError(
                "task_type must be regression, binary_classification, or "
                "multiclass_classification."
            )
        object.__setattr__(self, "task_type", task)
        activation = str(self.hidden_activation).strip().casefold()
        if activation not in {"relu", "tanh", "leaky_relu"}:
            raise TuningConfigError(
                "hidden_activation must be relu, tanh, or leaky_relu."
            )
        object.__setattr__(self, "hidden_activation", activation)
        positive = {
            "learning_rate": self.learning_rate,
            "leaky_relu_slope": self.leaky_relu_slope,
            "weight_init_scale": self.weight_init_scale,
            "adam_epsilon": self.adam_epsilon,
            "stability_epsilon": self.stability_epsilon,
        }
        for name, raw_value in positive.items():
            value = _finite_float(raw_value, name)
            if value <= 0:
                raise TuningConfigError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        for name in ("l2_lambda", "dropout_rate"):
            value = _finite_float(getattr(self, name), name)
            if value < 0:
                raise TuningConfigError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if self.dropout_rate >= 1:
            raise TuningConfigError("dropout_rate must be less than 1.")
        for name in ("beta1", "beta2", "prediction_threshold"):
            value = _finite_float(getattr(self, name), name)
            if not 0 < value < 1:
                raise TuningConfigError(f"{name} must be within (0, 1).")
            object.__setattr__(self, name, value)
        if self.gradient_clip_norm is not None:
            clip = _finite_float(self.gradient_clip_norm, "gradient_clip_norm")
            if clip <= 0:
                raise TuningConfigError("gradient_clip_norm must be positive.")
            object.__setattr__(self, "gradient_clip_norm", clip)
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, (int, np.integer))
        ):
            raise TuningConfigError("random_state must be an integer or None.")
        if self.random_state is not None:
            object.__setattr__(self, "random_state", int(self.random_state))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DenseNetworkConfig":
        if not isinstance(config, Mapping):
            raise TuningConfigError("dnn configuration must be a mapping.")
        known = set(cls.__dataclass_fields__)
        unknown = set(config) - known - {"training", "monitoring"}
        if unknown:
            raise TuningConfigError(
                "Unknown dense-network configuration fields.",
                details={"unknown_fields": sorted(str(item) for item in unknown)},
            )
        return cls(**{name: config[name] for name in known if name in config})


@dataclass(slots=True)
class DenseTrainingHistory:
    epochs: list[int] = field(default_factory=list)
    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    stopped_early: bool = False
    total_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": list(self.epochs),
            "training_loss": list(self.training_loss),
            "validation_loss": list(self.validation_loss),
            "gradient_norm": list(self.gradient_norm),
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
            "total_steps": self.total_steps,
        }


class DenseNeuralNetwork:
    """Fully connected network with deterministic, resumable optimization."""

    STATE_SCHEMA_VERSION = 1
    CONSTRUCTOR_PARAMETER_NAMES = frozenset(DenseNetworkConfig.__dataclass_fields__)
    FIT_PARAMETER_NAMES = frozenset(
        {
            "epochs",
            "batch_size",
            "shuffle",
            "early_stopping_patience",
            "min_delta",
            "restore_best_weights",
        }
    )

    def __init__(
        self,
        layer_sizes: Sequence[int],
        config: DenseNetworkConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self.layer_sizes = self._validate_layer_sizes(layer_sizes)
        if config is None:
            self.config = DenseNetworkConfig(**overrides)
        else:
            base = (
                config
                if isinstance(config, DenseNetworkConfig)
                else DenseNetworkConfig.from_mapping(config)
            )
            if overrides:
                payload = {
                    name: getattr(base, name)
                    for name in DenseNetworkConfig.__dataclass_fields__
                }
                payload.update(overrides)
                self.config = DenseNetworkConfig(**payload)
            else:
                self.config = base
        self._validate_task_output()
        self.rng = np.random.default_rng(self.config.random_state)
        self.training_steps = 0
        self.last_gradient_norm: float | None = None
        self.last_metrics: dict[str, float] = {}
        self.tuning_fit_parameters: dict[str, Any] = {}
        self.weights: list[Array] = []
        self.biases: list[Array] = []
        self._initialize_parameters()
        self._initialize_optimizer_state()

    @staticmethod
    def _validate_layer_sizes(layer_sizes: Sequence[int]) -> tuple[int, ...]:
        if isinstance(layer_sizes, (str, bytes)):
            raise TuningConfigError("layer_sizes must be a sequence of integers.")
        values = tuple(layer_sizes)
        if len(values) < 2:
            raise TuningConfigError(
                "layer_sizes must include input and output dimensions."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in values
        ):
            raise TuningConfigError("Every layer size must be a positive integer.")
        return tuple(int(value) for value in values)

    def _validate_task_output(self) -> None:
        output_dim = self.layer_sizes[-1]
        if self.config.task_type == "binary_classification" and output_dim != 1:
            raise TuningConfigError(
                "Binary classification requires an output dimension of 1."
            )
        if self.config.task_type == "multiclass_classification" and output_dim < 2:
            raise TuningConfigError(
                "Multiclass classification requires at least two outputs."
            )

    def _initialize_parameters(self) -> None:
        for fan_in, fan_out in zip(
            self.layer_sizes[:-1], self.layer_sizes[1:], strict=True
        ):
            if self.config.hidden_activation in {"relu", "leaky_relu"}:
                scale = math.sqrt(2.0 / fan_in)
            else:
                scale = math.sqrt(1.0 / fan_in)
            scale *= self.config.weight_init_scale
            self.weights.append(
                self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(float)
            )
            self.biases.append(np.zeros((1, fan_out), dtype=float))
        self._assert_finite_parameters("initialization")

    def _initialize_optimizer_state(self) -> None:
        self._weight_m = [np.zeros_like(item) for item in self.weights]
        self._weight_v = [np.zeros_like(item) for item in self.weights]
        self._bias_m = [np.zeros_like(item) for item in self.biases]
        self._bias_v = [np.zeros_like(item) for item in self.biases]

    def _activation(self, values: Array) -> Array:
        if self.config.hidden_activation == "relu":
            return np.maximum(values, 0.0)
        if self.config.hidden_activation == "tanh":
            return np.tanh(values)
        return np.where(
            values > 0.0, values, self.config.leaky_relu_slope * values
        )

    def _activation_derivative(self, values: Array) -> Array:
        if self.config.hidden_activation == "relu":
            return (values > 0.0).astype(float)
        if self.config.hidden_activation == "tanh":
            activated = np.tanh(values)
            return 1.0 - activated**2
        return np.where(values > 0.0, 1.0, self.config.leaky_relu_slope)

    def _forward(
        self, x: Array, *, training: bool
    ) -> tuple[Array, tuple[list[Array], list[Array], list[Array | None]]]:
        activations = [x]
        pre_activations: list[Array] = []
        dropout_masks: list[Array | None] = []
        current = x
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            pre_activation = current @ weight + bias
            pre_activations.append(pre_activation)
            is_output = index == len(self.weights) - 1
            if is_output:
                current = pre_activation
                dropout_masks.append(None)
            else:
                current = self._activation(pre_activation)
                mask: Array | None = None
                if training and self.config.dropout_rate > 0:
                    keep_probability = 1.0 - self.config.dropout_rate
                    mask = (
                        self.rng.random(current.shape) < keep_probability
                    ).astype(float) / keep_probability
                    current = current * mask
                dropout_masks.append(mask)
            activations.append(current)
        if not np.isfinite(current).all():
            raise TuningEvaluationError("Forward pass produced non-finite values.")
        return current, (activations, pre_activations, dropout_masks)

    def _loss_and_output_gradient(
        self, logits: Array, targets: Array
    ) -> tuple[float, Array]:
        count = logits.shape[0]
        if self.config.task_type == "regression":
            residual = logits - targets
            loss = 0.5 * float(np.mean(np.sum(residual**2, axis=1)))
            return loss, residual / count
        if self.config.task_type == "binary_classification":
            loss = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
            return loss, (_stable_sigmoid(logits) - targets) / count
        probabilities = _softmax(logits)
        loss = -float(
            np.mean(
                np.sum(
                    targets
                    * np.log(
                        np.maximum(probabilities, self.config.stability_epsilon)
                    ),
                    axis=1,
                )
            )
        )
        return loss, (probabilities - targets) / count

    def _gradients(
        self,
        output_gradient: Array,
        cache: tuple[list[Array], list[Array], list[Array | None]],
    ) -> tuple[list[Array], list[Array]]:
        activations, pre_activations, dropout_masks = cache
        weight_gradients: list[Array] = [np.empty(0)] * len(self.weights)
        bias_gradients: list[Array] = [np.empty(0)] * len(self.biases)
        delta = output_gradient
        for index in range(len(self.weights) - 1, -1, -1):
            weight_gradients[index] = (
                activations[index].T @ delta
                + self.config.l2_lambda * self.weights[index]
            )
            bias_gradients[index] = np.sum(delta, axis=0, keepdims=True)
            if index > 0:
                delta = delta @ self.weights[index].T
                mask = dropout_masks[index - 1]
                if mask is not None:
                    delta *= mask
                delta *= self._activation_derivative(pre_activations[index - 1])
        return weight_gradients, bias_gradients

    def _clip_gradients(
        self, weight_gradients: list[Array], bias_gradients: list[Array]
    ) -> float:
        norm = math.sqrt(
            sum(float(np.sum(item**2)) for item in weight_gradients)
            + sum(float(np.sum(item**2)) for item in bias_gradients)
        )
        if not math.isfinite(norm):
            raise TuningEvaluationError("Gradient norm is non-finite.")
        clip = self.config.gradient_clip_norm
        if clip is not None and norm > clip:
            scale = clip / max(norm, self.config.stability_epsilon)
            for item in (*weight_gradients, *bias_gradients):
                item *= scale
        self.last_gradient_norm = norm
        return norm

    def _apply_adam(
        self, weight_gradients: list[Array], bias_gradients: list[Array]
    ) -> None:
        self.training_steps += 1
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        correction1 = 1.0 - beta1**self.training_steps
        correction2 = 1.0 - beta2**self.training_steps
        for index in range(len(self.weights)):
            self._weight_m[index] = (
                beta1 * self._weight_m[index]
                + (1.0 - beta1) * weight_gradients[index]
            )
            self._weight_v[index] = (
                beta2 * self._weight_v[index]
                + (1.0 - beta2) * weight_gradients[index] ** 2
            )
            self._bias_m[index] = (
                beta1 * self._bias_m[index]
                + (1.0 - beta1) * bias_gradients[index]
            )
            self._bias_v[index] = (
                beta2 * self._bias_v[index]
                + (1.0 - beta2) * bias_gradients[index] ** 2
            )
            weight_m = self._weight_m[index] / correction1
            weight_v = self._weight_v[index] / correction2
            bias_m = self._bias_m[index] / correction1
            bias_v = self._bias_v[index] / correction2
            self.weights[index] -= self.config.learning_rate * weight_m / (
                np.sqrt(weight_v) + self.config.adam_epsilon
            )
            self.biases[index] -= self.config.learning_rate * bias_m / (
                np.sqrt(bias_v) + self.config.adam_epsilon
            )
        self._assert_finite_parameters("optimizer_step")

    def fit(
        self,
        x_train: Any,
        y_train: Any,
        *,
        validation_data: tuple[Any, Any] | None = None,
        epochs: int = 100,
        batch_size: int = 64,
        shuffle: bool = True,
        early_stopping_patience: int | None = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> dict[str, Any]:
        x = self._validate_features(x_train, "x_train")
        y = self._prepare_targets(y_train, len(x), "y_train")
        options = self._validate_fit_options(
            epochs,
            batch_size,
            shuffle,
            early_stopping_patience,
            min_delta,
            restore_best_weights,
        )
        validation: tuple[Array, Array] | None = None
        if validation_data is not None:
            if not isinstance(validation_data, tuple) or len(validation_data) != 2:
                raise TuningValidationError(
                    "validation_data must be an (x_validation, y_validation) tuple."
                )
            validation_x = self._validate_features(
                validation_data[0], "x_validation"
            )
            validation_y = self._prepare_targets(
                validation_data[1], len(validation_x), "y_validation"
            )
            validation = (validation_x, validation_y)
        elif early_stopping_patience is not None:
            raise TuningValidationError(
                "early_stopping_patience requires validation_data."
            )

        history = DenseTrainingHistory()
        best_state: dict[str, Any] | None = None
        best_loss = math.inf
        epochs_without_improvement = 0
        indices = np.arange(len(x))

        for epoch in range(1, options["epochs"] + 1):
            if options["shuffle"]:
                self.rng.shuffle(indices)
            epoch_loss = 0.0
            epoch_count = 0
            epoch_gradient_norms: list[float] = []
            for start in range(0, len(x), options["batch_size"]):
                batch_indices = indices[start : start + options["batch_size"]]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                logits, cache = self._forward(batch_x, training=True)
                loss, output_gradient = self._loss_and_output_gradient(
                    logits, batch_y
                )
                weight_gradients, bias_gradients = self._gradients(
                    output_gradient, cache
                )
                gradient_norm = self._clip_gradients(
                    weight_gradients, bias_gradients
                )
                self._apply_adam(weight_gradients, bias_gradients)
                epoch_loss += loss * len(batch_indices)
                epoch_count += len(batch_indices)
                epoch_gradient_norms.append(gradient_norm)

            training_loss = epoch_loss / epoch_count
            history.epochs.append(epoch)
            history.training_loss.append(float(training_loss))
            history.gradient_norm.append(float(np.mean(epoch_gradient_norms)))
            if validation is not None:
                validation_loss = self._data_loss(*validation)
                history.validation_loss.append(validation_loss)
                if validation_loss < best_loss - options["min_delta"]:
                    best_loss = validation_loss
                    history.best_epoch = epoch
                    history.best_validation_loss = validation_loss
                    best_state = self.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                patience = options["early_stopping_patience"]
                if patience is not None and epochs_without_improvement >= patience:
                    history.stopped_early = True
                    break

        executed_steps = self.training_steps
        if best_state is not None and options["restore_best_weights"]:
            self.load_state_dict(best_state)
        history.total_steps = executed_steps
        self.last_metrics = self.evaluate(x, y, targets_prepared=True)
        return history.to_dict()

    def fit_for_tuning(
        self,
        x_train: Any,
        y_train: Any,
        x_validation: Any,
        y_validation: Any,
    ) -> dict[str, Any]:
        """Apply the explicit fit options captured by ``from_tuning_params``."""

        return self.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            **dict(self.tuning_fit_parameters),
        )

    def _data_loss(self, x: Array, y: Array) -> float:
        logits, _ = self._forward(x, training=False)
        loss, _ = self._loss_and_output_gradient(logits, y)
        return float(loss)

    def predict_proba(self, x: Any) -> Array:
        features = self._validate_features(x, "x")
        logits, _ = self._forward(features, training=False)
        if self.config.task_type == "binary_classification":
            return _stable_sigmoid(logits)
        if self.config.task_type == "multiclass_classification":
            return _softmax(logits)
        raise TuningValidationError(
            "predict_proba is unavailable for regression."
        )

    def predict(self, x: Any) -> Array:
        features = self._validate_features(x, "x")
        logits, _ = self._forward(features, training=False)
        if self.config.task_type == "regression":
            return logits[:, 0] if logits.shape[1] == 1 else logits
        if self.config.task_type == "binary_classification":
            probabilities = _stable_sigmoid(logits)
            return (
                probabilities[:, 0] >= self.config.prediction_threshold
            ).astype(int)
        return np.argmax(logits, axis=1)

    def evaluate(
        self, x: Any, y: Any, *, targets_prepared: bool = False
    ) -> dict[str, float]:
        features = self._validate_features(x, "x")
        targets = (
            np.asarray(y, dtype=float)
            if targets_prepared
            else self._prepare_targets(y, len(features), "y")
        )
        logits, _ = self._forward(features, training=False)
        if self.config.task_type == "regression":
            residual = logits - targets
            metrics = {
                "mse": float(np.mean(residual**2)),
                "rmse": float(math.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
            }
        elif self.config.task_type == "binary_classification":
            probabilities = _stable_sigmoid(logits)
            predictions = (
                probabilities >= self.config.prediction_threshold
            ).astype(float)
            metrics = {
                "accuracy": float(np.mean(predictions == targets)),
                "log_loss": float(
                    np.mean(np.logaddexp(0.0, logits) - targets * logits)
                ),
                "brier": float(np.mean((probabilities - targets) ** 2)),
            }
        else:
            probabilities = _softmax(logits)
            truth = np.argmax(targets, axis=1)
            metrics = {
                "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == truth)),
                "log_loss": -float(
                    np.mean(
                        np.sum(
                            targets
                            * np.log(
                                np.maximum(
                                    probabilities, self.config.stability_epsilon
                                )
                            ),
                            axis=1,
                        )
                    )
                ),
            }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise TuningEvaluationError("Evaluation produced non-finite metrics.")
        self.last_metrics = metrics
        return dict(metrics)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "layer_sizes": list(self.layer_sizes),
            "config": self.get_config(),
            "weights": [item.copy() for item in self.weights],
            "biases": [item.copy() for item in self.biases],
            "weight_m": [item.copy() for item in self._weight_m],
            "weight_v": [item.copy() for item in self._weight_v],
            "bias_m": [item.copy() for item in self._bias_m],
            "bias_v": [item.copy() for item in self._bias_v],
            "training_steps": self.training_steps,
            "last_gradient_norm": self.last_gradient_norm,
            "last_metrics": dict(self.last_metrics),
            "tuning_fit_parameters": dict(self.tuning_fit_parameters),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TuningValidationError("state must be a mapping.")
        schema_version = state.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != self.STATE_SCHEMA_VERSION
        ):
            raise TuningValidationError("Unsupported dense-network state schema.")
        if tuple(state.get("layer_sizes", ())) != self.layer_sizes:
            raise TuningValidationError("State layer sizes do not match the model.")
        raw_config = state.get("config")
        if not isinstance(raw_config, Mapping) or dict(raw_config) != self.get_config():
            raise TuningValidationError("State configuration does not match the model.")
        collections = {
            "weights": self.weights,
            "biases": self.biases,
            "weight_m": self._weight_m,
            "weight_v": self._weight_v,
            "bias_m": self._bias_m,
            "bias_v": self._bias_v,
        }
        loaded: dict[str, list[Array]] = {}
        for name, reference in collections.items():
            raw_items = state.get(name)
            if not isinstance(raw_items, Sequence) or len(raw_items) != len(reference):
                raise TuningValidationError(f"Invalid state collection {name!r}.")
            try:
                converted = [np.asarray(item, dtype=float).copy() for item in raw_items]
            except (TypeError, ValueError) as exc:
                raise TuningValidationError(
                    f"State collection {name!r} must be numeric."
                ) from exc
            if any(item.shape != expected.shape for item, expected in zip(
                converted, reference, strict=True
            )):
                raise TuningValidationError(
                    f"State collection {name!r} has incompatible shapes."
                )
            if any(not np.isfinite(item).all() for item in converted):
                raise TuningValidationError(
                    f"State collection {name!r} contains non-finite values."
                )
            loaded[name] = converted
        steps = state.get("training_steps", 0)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise TuningValidationError("training_steps must be a non-negative integer.")
        raw_gradient_norm = state.get("last_gradient_norm")
        try:
            gradient_norm = (
                None if raw_gradient_norm is None else float(raw_gradient_norm)
            )
        except (TypeError, ValueError) as exc:
            raise TuningValidationError("last_gradient_norm is invalid.") from exc
        if gradient_norm is not None and (
            not math.isfinite(gradient_norm) or gradient_norm < 0
        ):
            raise TuningValidationError("last_gradient_norm is invalid.")
        raw_metrics = state.get("last_metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise TuningValidationError("last_metrics must be a mapping.")
        try:
            last_metrics = {
                str(name): float(value) for name, value in raw_metrics.items()
            }
        except (TypeError, ValueError) as exc:
            raise TuningValidationError(
                "last_metrics must contain numeric values."
            ) from exc
        if not all(math.isfinite(value) for value in last_metrics.values()):
            raise TuningValidationError("last_metrics contains non-finite values.")
        fit_parameters = state.get("tuning_fit_parameters", {})
        if not isinstance(fit_parameters, Mapping) or any(
            name not in self.FIT_PARAMETER_NAMES for name in fit_parameters
        ):
            raise TuningValidationError("tuning_fit_parameters is invalid.")
        loaded_fit_parameters = dict(fit_parameters)
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise TuningValidationError("State is missing rng_state.")
        loaded_rng_state = copy.deepcopy(dict(rng_state))
        test_rng = np.random.default_rng()
        try:
            test_rng.bit_generator.state = loaded_rng_state
        except (TypeError, ValueError) as exc:
            raise TuningValidationError("rng_state is invalid.") from exc

        # Commit only after every state component has been validated.  This
        # preserves the current model if a checkpoint is malformed.
        self.weights = loaded["weights"]
        self.biases = loaded["biases"]
        self._weight_m = loaded["weight_m"]
        self._weight_v = loaded["weight_v"]
        self._bias_m = loaded["bias_m"]
        self._bias_v = loaded["bias_v"]
        self.training_steps = steps
        self.last_gradient_norm = gradient_norm
        self.last_metrics = last_metrics
        self.tuning_fit_parameters = loaded_fit_parameters
        self.rng.bit_generator.state = loaded_rng_state

    def get_config(self) -> dict[str, Any]:
        return {
            name: getattr(self.config, name)
            for name in DenseNetworkConfig.__dataclass_fields__
        }

    @classmethod
    def from_tuning_params(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        parameters: Mapping[str, Any],
        base_config: Mapping[str, Any] | DenseNetworkConfig | None = None,
        seed: int | None = None,
    ) -> "DenseNeuralNetwork":
        if not isinstance(parameters, Mapping):
            raise TuningConfigError("parameters must be a mapping.")
        allowed = {
            "hidden_layer_sizes",
            *cls.CONSTRUCTOR_PARAMETER_NAMES,
            *cls.FIT_PARAMETER_NAMES,
        }
        unknown = set(parameters) - allowed
        if unknown:
            raise TuningConfigError(
                "Unknown dense-network tuning parameters.",
                details={"unknown_parameters": sorted(str(item) for item in unknown)},
            )
        hidden = cls.parse_hidden_layer_sizes(
            parameters.get("hidden_layer_sizes")
        )
        if base_config is None:
            config_values: dict[str, Any] = {}
        elif isinstance(base_config, DenseNetworkConfig):
            config_values = {
                name: getattr(base_config, name)
                for name in DenseNetworkConfig.__dataclass_fields__
            }
        else:
            config_values = {
                name: value
                for name, value in base_config.items()
                if name in DenseNetworkConfig.__dataclass_fields__
            }
        config_values.update(
            {
                name: value
                for name, value in parameters.items()
                if name in cls.CONSTRUCTOR_PARAMETER_NAMES
            }
        )
        if seed is not None:
            config_values["random_state"] = seed
        model = cls(
            (int(input_dim), *hidden, int(output_dim)),
            DenseNetworkConfig(**config_values),
        )
        training_defaults = (
            base_config.get("training", {})
            if isinstance(base_config, Mapping)
            else {}
        )
        if training_defaults is not None and not isinstance(
            training_defaults, Mapping
        ):
            raise TuningConfigError("dnn.training must be a mapping.")
        model.tuning_fit_parameters = {
            name: value
            for name, value in dict(training_defaults or {}).items()
            if name in cls.FIT_PARAMETER_NAMES
        }
        model.tuning_fit_parameters.update(
            {
                name: value
                for name, value in parameters.items()
                if name in cls.FIT_PARAMETER_NAMES
            }
        )
        return model

    @staticmethod
    def parse_hidden_layer_sizes(value: Any) -> tuple[int, ...]:
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",") if item.strip()]
            values = tuple(int(item) for item in parts)
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            values = (int(value),)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = tuple(int(item) for item in value)
        else:
            raise TuningConfigError(
                "hidden_layer_sizes must be an integer, comma-delimited string, "
                "or sequence of integers."
            )
        if not values or any(item <= 0 for item in values):
            raise TuningConfigError(
                "hidden_layer_sizes must contain positive integers."
            )
        return values

    def _validate_features(self, values: Any, name: str) -> Array:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TuningValidationError(f"{name} must be numeric.") from exc
        if array.ndim != 2 or array.shape[1] != self.layer_sizes[0]:
            raise TuningValidationError(
                f"{name} must have shape (samples, {self.layer_sizes[0]})."
            )
        if len(array) < 1 or not np.isfinite(array).all():
            raise TuningValidationError(
                f"{name} must be non-empty and contain only finite values."
            )
        return array

    def _prepare_targets(self, values: Any, count: int, name: str) -> Array:
        try:
            array = np.asarray(values)
        except Exception as exc:
            raise TuningValidationError(f"{name} cannot be converted to an array.") from exc
        if array.ndim == 0 or array.shape[0] != count:
            raise TuningValidationError(
                f"{name} sample count differs from the feature sample count."
            )
        output_dim = self.layer_sizes[-1]
        if self.config.task_type == "regression":
            numeric = np.asarray(array, dtype=float)
            if numeric.ndim == 1:
                numeric = numeric.reshape(-1, 1)
            if numeric.ndim != 2 or numeric.shape[1] != output_dim:
                raise TuningValidationError(
                    f"{name} must have {output_dim} regression output column(s)."
                )
            result = numeric
        elif self.config.task_type == "binary_classification":
            numeric = np.asarray(array, dtype=float).reshape(-1, 1)
            if not np.isin(numeric, (0.0, 1.0)).all():
                raise TuningValidationError(
                    f"{name} must contain binary labels 0 and 1."
                )
            result = numeric
        else:
            if array.ndim == 2 and array.shape[1] == output_dim:
                numeric = np.asarray(array, dtype=float)
                if not np.allclose(np.sum(numeric, axis=1), 1.0) or np.any(
                    numeric < 0
                ):
                    raise TuningValidationError(
                        f"{name} one-hot targets are invalid."
                    )
                result = numeric
            else:
                labels = np.asarray(array).reshape(-1)
                if not np.issubdtype(labels.dtype, np.integer):
                    raise TuningValidationError(
                        f"{name} multiclass labels must be integers."
                    )
                labels = labels.astype(int)
                if np.any(labels < 0) or np.any(labels >= output_dim):
                    raise TuningValidationError(
                        f"{name} labels must be within [0, {output_dim - 1}]."
                    )
                result = np.eye(output_dim, dtype=float)[labels]
        if not np.isfinite(result).all():
            raise TuningValidationError(f"{name} contains non-finite values.")
        return result

    @staticmethod
    def _validate_fit_options(
        epochs: Any,
        batch_size: Any,
        shuffle: Any,
        patience: Any,
        min_delta: Any,
        restore: Any,
    ) -> dict[str, Any]:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise TuningValidationError("epochs must be a positive integer.")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise TuningValidationError("batch_size must be a positive integer.")
        if not isinstance(shuffle, bool) or not isinstance(restore, bool):
            raise TuningValidationError(
                "shuffle and restore_best_weights must be bool values."
            )
        if patience is not None and (
            isinstance(patience, bool)
            or not isinstance(patience, int)
            or patience < 1
        ):
            raise TuningValidationError(
                "early_stopping_patience must be a positive integer or None."
            )
        delta = _finite_float(min_delta, "min_delta")
        if delta < 0:
            raise TuningValidationError("min_delta must be non-negative.")
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "early_stopping_patience": patience,
            "min_delta": delta,
            "restore_best_weights": restore,
        }

    def _assert_finite_parameters(self, operation: str) -> None:
        if any(
            not np.isfinite(item).all() for item in (*self.weights, *self.biases)
        ):
            raise TuningEvaluationError(
                "Model parameters became non-finite.",
                context=TuningErrorContext(
                    component=self.__class__.__name__, operation=operation
                ),
            )


__all__ = ["DenseNetworkConfig", "DenseNeuralNetwork", "DenseTrainingHistory"]
