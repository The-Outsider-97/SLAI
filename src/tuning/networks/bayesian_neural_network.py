"""Mean-field variational Bayesian neural network for SLAI tuning.

This NumPy implementation follows Bayes-by-Backprop with reparameterized
Gaussian posteriors and a closed-form Gaussian KL term.  It supports the same
supervised task families as ``DenseNeuralNetwork`` while adding Monte Carlo
predictive uncertainty.  Configuration, training, prediction, and mutable
state are explicit; the module does not load global configuration or write
artifacts at import or runtime.
"""

from __future__ import annotations

import copy
import math
import numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from ..utils.tuning_errors import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Bayesian Neural Network")
printer = PrettyPrinter()


Array = np.ndarray


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TuningConfigError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TuningConfigError(f"{name} must be finite.")
    return numeric


def _sigmoid(values: Array) -> Array:
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


def _softplus(values: Array) -> Array:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


@dataclass(frozen=True, slots=True)
class BayesianNetworkConfig:
    task_type: str = "regression"
    learning_rate: float = 5.0e-3
    prior_mu: float = 0.0
    prior_logvar: float = 0.0
    posterior_rho_init: float = -3.0
    likelihood_std: float = 1.0
    hidden_activation: str = "relu"
    leaky_relu_slope: float = 0.01
    weight_init_scale: float = 0.75
    gradient_clip_norm: float | None = 5.0
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
            "likelihood_std": self.likelihood_std,
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
        for name in ("prior_mu", "prior_logvar", "posterior_rho_init"):
            object.__setattr__(
                self, name, _finite_float(getattr(self, name), name)
            )
        try:
            prior_variance = math.exp(self.prior_logvar)
        except OverflowError as exc:
            raise TuningConfigError("prior_logvar produces an invalid variance.") from exc
        if not math.isfinite(prior_variance) or prior_variance <= 0:
            raise TuningConfigError("prior_logvar produces an invalid variance.")
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
    def from_mapping(cls, config: Mapping[str, Any]) -> "BayesianNetworkConfig":
        if not isinstance(config, Mapping):
            raise TuningConfigError("bnn configuration must be a mapping.")
        known = set(cls.__dataclass_fields__)
        unknown = set(config) - known - {"training", "prediction", "monitoring"}
        if unknown:
            raise TuningConfigError(
                "Unknown Bayesian-network configuration fields.",
                details={"unknown_fields": sorted(str(item) for item in unknown)},
            )
        return cls(**{name: config[name] for name in known if name in config})


@dataclass(slots=True)
class BayesianTrainingHistory:
    epochs: list[int] = field(default_factory=list)
    training_negative_elbo: list[float] = field(default_factory=list)
    validation_negative_elbo: list[float] = field(default_factory=list)
    training_data_loss: list[float] = field(default_factory=list)
    kl_per_observation: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_validation_negative_elbo: float | None = None
    stopped_early: bool = False
    total_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": list(self.epochs),
            "training_negative_elbo": list(self.training_negative_elbo),
            "validation_negative_elbo": list(self.validation_negative_elbo),
            "training_data_loss": list(self.training_data_loss),
            "kl_per_observation": list(self.kl_per_observation),
            "gradient_norm": list(self.gradient_norm),
            "best_epoch": self.best_epoch,
            "best_validation_negative_elbo": (
                self.best_validation_negative_elbo
            ),
            "stopped_early": self.stopped_early,
            "total_steps": self.total_steps,
        }


class BayesianNeuralNetwork:
    """Fully connected mean-field Bayesian network trained by variational ELBO."""

    STATE_SCHEMA_VERSION = 1
    CONSTRUCTOR_PARAMETER_NAMES = frozenset(BayesianNetworkConfig.__dataclass_fields__)
    FIT_PARAMETER_NAMES = frozenset(
        {
            "epochs",
            "batch_size",
            "num_samples",
            "validation_num_samples",
            "shuffle",
            "early_stopping_patience",
            "min_delta",
            "restore_best_weights",
        }
    )

    def __init__(
        self,
        layer_sizes: Sequence[int],
        config: BayesianNetworkConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self.layer_sizes = self._validate_layer_sizes(layer_sizes)
        if config is None:
            self.config = BayesianNetworkConfig(**overrides)
        else:
            base = (
                config
                if isinstance(config, BayesianNetworkConfig)
                else BayesianNetworkConfig.from_mapping(config)
            )
            if overrides:
                payload = {
                    name: getattr(base, name)
                    for name in BayesianNetworkConfig.__dataclass_fields__
                }
                payload.update(overrides)
                self.config = BayesianNetworkConfig(**payload)
            else:
                self.config = base
        self._validate_task_output()
        self.rng = np.random.default_rng(self.config.random_state)
        self.training_steps = 0
        self.last_gradient_norm: float | None = None
        self.last_metrics: dict[str, float] = {}
        self.tuning_fit_parameters: dict[str, Any] = {}
        self.weight_mu: list[Array] = []
        self.weight_rho: list[Array] = []
        self.bias_mu: list[Array] = []
        self.bias_rho: list[Array] = []
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
            self.weight_mu.append(
                self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(float)
            )
            self.weight_rho.append(
                np.full(
                    (fan_in, fan_out), self.config.posterior_rho_init, dtype=float
                )
            )
            self.bias_mu.append(np.zeros((1, fan_out), dtype=float))
            self.bias_rho.append(
                np.full((1, fan_out), self.config.posterior_rho_init, dtype=float)
            )
        self._assert_finite_parameters("initialization")

    def _initialize_optimizer_state(self) -> None:
        self._adam_m = {
            name: [np.zeros_like(item) for item in values]
            for name, values in self._parameter_collections().items()
        }
        self._adam_v = {
            name: [np.zeros_like(item) for item in values]
            for name, values in self._parameter_collections().items()
        }

    def _parameter_collections(self) -> dict[str, list[Array]]:
        return {
            "weight_mu": self.weight_mu,
            "weight_rho": self.weight_rho,
            "bias_mu": self.bias_mu,
            "bias_rho": self.bias_rho,
        }

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

    def _sample_parameters(
        self, rng: np.random.Generator
    ) -> tuple[
        list[Array],
        list[Array],
        dict[str, list[Array]],
        dict[str, list[Array]],
    ]:
        samples: dict[str, list[Array]] = {}
        epsilons: dict[str, list[Array]] = {}
        for location_name, rho_name in (
            ("weight_mu", "weight_rho"),
            ("bias_mu", "bias_rho"),
        ):
            locations = getattr(self, location_name)
            rhos = getattr(self, rho_name)
            sampled: list[Array] = []
            noises: list[Array] = []
            for location, rho in zip(locations, rhos, strict=True):
                epsilon = rng.normal(size=location.shape)
                sigma = np.maximum(
                    _softplus(rho), self.config.stability_epsilon
                )
                sampled.append(location + sigma * epsilon)
                noises.append(epsilon)
            samples[location_name] = sampled
            epsilons[rho_name] = noises
        return (
            samples["weight_mu"],
            samples["bias_mu"],
            samples,
            epsilons,
        )

    def _forward_with_parameters(
        self, x: Array, weights: Sequence[Array], biases: Sequence[Array]
    ) -> tuple[Array, tuple[list[Array], list[Array]]]:
        activations = [x]
        pre_activations: list[Array] = []
        current = x
        for index, (weight, bias) in enumerate(
            zip(weights, biases, strict=True)
        ):
            pre_activation = current @ weight + bias
            pre_activations.append(pre_activation)
            current = (
                pre_activation
                if index == len(weights) - 1
                else self._activation(pre_activation)
            )
            activations.append(current)
        if not np.isfinite(current).all():
            raise TuningEvaluationError("Forward pass produced non-finite values.")
        return current, (activations, pre_activations)

    def _data_loss_and_gradient(
        self, logits: Array, targets: Array
    ) -> tuple[float, Array]:
        count = logits.shape[0]
        if self.config.task_type == "regression":
            variance = self.config.likelihood_std**2
            residual = logits - targets
            constant = math.log(2.0 * math.pi * variance)
            loss = 0.5 * float(
                np.mean(
                    np.sum(residual**2 / variance + constant, axis=1)
                )
            )
            return loss, residual / (variance * count)
        if self.config.task_type == "binary_classification":
            loss = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
            return loss, (_sigmoid(logits) - targets) / count
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

    def _sample_gradients(
        self,
        output_gradient: Array,
        cache: tuple[list[Array], list[Array]],
        sampled_weights: Sequence[Array],
    ) -> tuple[list[Array], list[Array]]:
        activations, pre_activations = cache
        weight_gradients: list[Array] = [np.empty(0)] * len(sampled_weights)
        bias_gradients: list[Array] = [np.empty(0)] * len(sampled_weights)
        delta = output_gradient
        for index in range(len(sampled_weights) - 1, -1, -1):
            weight_gradients[index] = activations[index].T @ delta
            bias_gradients[index] = np.sum(delta, axis=0, keepdims=True)
            if index > 0:
                delta = delta @ sampled_weights[index].T
                delta *= self._activation_derivative(pre_activations[index - 1])
        return weight_gradients, bias_gradients

    def _negative_elbo_and_gradients(
        self,
        x: Array,
        y: Array,
        *,
        dataset_size: int,
        num_samples: int,
    ) -> tuple[float, float, float, dict[str, list[Array]]]:
        gradients = {
            name: [np.zeros_like(item) for item in values]
            for name, values in self._parameter_collections().items()
        }
        data_loss = 0.0
        for _ in range(num_samples):
            sampled_weights, sampled_biases, _samples, epsilons = (
                self._sample_parameters(self.rng)
            )
            logits, cache = self._forward_with_parameters(
                x, sampled_weights, sampled_biases
            )
            sample_loss, output_gradient = self._data_loss_and_gradient(logits, y)
            weight_gradient, bias_gradient = self._sample_gradients(
                output_gradient, cache, sampled_weights
            )
            data_loss += sample_loss
            for index in range(len(self.weight_mu)):
                gradients["weight_mu"][index] += weight_gradient[index]
                gradients["bias_mu"][index] += bias_gradient[index]
                gradients["weight_rho"][index] += (
                    weight_gradient[index]
                    * epsilons["weight_rho"][index]
                    * _sigmoid(self.weight_rho[index])
                )
                gradients["bias_rho"][index] += (
                    bias_gradient[index]
                    * epsilons["bias_rho"][index]
                    * _sigmoid(self.bias_rho[index])
                )
        data_loss /= num_samples
        for values in gradients.values():
            for item in values:
                item /= num_samples

        kl_value = 0.0
        prior_variance = math.exp(self.config.prior_logvar)
        prior_std = math.sqrt(prior_variance)
        for location_name, rho_name in (
            ("weight_mu", "weight_rho"),
            ("bias_mu", "bias_rho"),
        ):
            locations = getattr(self, location_name)
            rhos = getattr(self, rho_name)
            for index, (location, rho) in enumerate(
                zip(locations, rhos, strict=True)
            ):
                sigma = np.maximum(
                    _softplus(rho), self.config.stability_epsilon
                )
                kl_value += float(
                    np.sum(
                        np.log(prior_std / sigma)
                        + (
                            sigma**2
                            + (location - self.config.prior_mu) ** 2
                        )
                        / (2.0 * prior_variance)
                        - 0.5
                    )
                )
                location_gradient = (
                    location - self.config.prior_mu
                ) / prior_variance
                sigma_gradient = -1.0 / sigma + sigma / prior_variance
                gradients[location_name][index] += location_gradient / dataset_size
                gradients[rho_name][index] += (
                    sigma_gradient * _sigmoid(rho) / dataset_size
                )
        kl_per_observation = kl_value / dataset_size
        negative_elbo = data_loss + kl_per_observation
        if not all(
            math.isfinite(value)
            for value in (negative_elbo, data_loss, kl_per_observation)
        ):
            raise TuningEvaluationError("Variational objective became non-finite.")
        return negative_elbo, data_loss, kl_per_observation, gradients

    def _clip_gradients(self, gradients: Mapping[str, Sequence[Array]]) -> float:
        norm = math.sqrt(
            sum(
                float(np.sum(item**2))
                for values in gradients.values()
                for item in values
            )
        )
        if not math.isfinite(norm):
            raise TuningEvaluationError("Gradient norm is non-finite.")
        clip = self.config.gradient_clip_norm
        if clip is not None and norm > clip:
            scale = clip / max(norm, self.config.stability_epsilon)
            for values in gradients.values():
                for item in values:
                    item *= scale
        self.last_gradient_norm = norm
        return norm

    def _apply_adam(self, gradients: Mapping[str, Sequence[Array]]) -> None:
        self.training_steps += 1
        correction1 = 1.0 - self.config.beta1**self.training_steps
        correction2 = 1.0 - self.config.beta2**self.training_steps
        parameters = self._parameter_collections()
        for name, values in parameters.items():
            for index, parameter in enumerate(values):
                gradient = gradients[name][index]
                self._adam_m[name][index] = (
                    self.config.beta1 * self._adam_m[name][index]
                    + (1.0 - self.config.beta1) * gradient
                )
                self._adam_v[name][index] = (
                    self.config.beta2 * self._adam_v[name][index]
                    + (1.0 - self.config.beta2) * gradient**2
                )
                first = self._adam_m[name][index] / correction1
                second = self._adam_v[name][index] / correction2
                parameter -= self.config.learning_rate * first / (
                    np.sqrt(second) + self.config.adam_epsilon
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
        num_samples: int = 3,
        validation_num_samples: int = 10,
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
            num_samples,
            validation_num_samples,
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

        history = BayesianTrainingHistory()
        best_state: dict[str, Any] | None = None
        best_loss = math.inf
        epochs_without_improvement = 0
        indices = np.arange(len(x))

        for epoch in range(1, options["epochs"] + 1):
            if options["shuffle"]:
                self.rng.shuffle(indices)
            negative_elbo_total = 0.0
            data_loss_total = 0.0
            kl_total = 0.0
            observed = 0
            gradient_norms: list[float] = []
            for start in range(0, len(x), options["batch_size"]):
                batch_indices = indices[start : start + options["batch_size"]]
                negative_elbo, data_loss, kl, gradients = (
                    self._negative_elbo_and_gradients(
                        x[batch_indices],
                        y[batch_indices],
                        dataset_size=len(x),
                        num_samples=options["num_samples"],
                    )
                )
                gradient_norms.append(self._clip_gradients(gradients))
                self._apply_adam(gradients)
                batch_count = len(batch_indices)
                negative_elbo_total += negative_elbo * batch_count
                data_loss_total += data_loss * batch_count
                kl_total += kl * batch_count
                observed += batch_count

            history.epochs.append(epoch)
            history.training_negative_elbo.append(
                float(negative_elbo_total / observed)
            )
            history.training_data_loss.append(float(data_loss_total / observed))
            history.kl_per_observation.append(float(kl_total / observed))
            history.gradient_norm.append(float(np.mean(gradient_norms)))
            if validation is not None:
                validation_loss = self._negative_elbo_value(
                    *validation,
                    dataset_size=len(x),
                    num_samples=options["validation_num_samples"],
                )
                history.validation_negative_elbo.append(validation_loss)
                if validation_loss < best_loss - options["min_delta"]:
                    best_loss = validation_loss
                    history.best_epoch = epoch
                    history.best_validation_negative_elbo = validation_loss
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
        self.last_metrics = self.evaluate(
            x, y, targets_prepared=True, num_samples=options["validation_num_samples"]
        )
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

    def _negative_elbo_value(
        self,
        x: Array,
        y: Array,
        *,
        dataset_size: int,
        num_samples: int,
    ) -> float:
        data_losses: list[float] = []
        for _ in range(num_samples):
            weights, biases, _samples, _epsilons = self._sample_parameters(self.rng)
            logits, _ = self._forward_with_parameters(x, weights, biases)
            data_loss, _ = self._data_loss_and_gradient(logits, y)
            data_losses.append(data_loss)
        kl = self._kl_divergence() / dataset_size
        value = float(np.mean(data_losses) + kl)
        if not math.isfinite(value):
            raise TuningEvaluationError("Validation negative ELBO is non-finite.")
        return value

    def _kl_divergence(self) -> float:
        prior_variance = math.exp(self.config.prior_logvar)
        prior_std = math.sqrt(prior_variance)
        total = 0.0
        for location_name, rho_name in (
            ("weight_mu", "weight_rho"),
            ("bias_mu", "bias_rho"),
        ):
            for location, rho in zip(
                getattr(self, location_name), getattr(self, rho_name), strict=True
            ):
                sigma = np.maximum(
                    _softplus(rho), self.config.stability_epsilon
                )
                total += float(
                    np.sum(
                        np.log(prior_std / sigma)
                        + (
                            sigma**2
                            + (location - self.config.prior_mu) ** 2
                        )
                        / (2.0 * prior_variance)
                        - 0.5
                    )
                )
        return total

    def predict_distribution(
        self,
        x: Any,
        *,
        num_samples: int = 100,
        seed: int | None = None,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
    ) -> dict[str, Array]:
        features = self._validate_features(x, "x")
        self._validate_prediction_options(
            num_samples, lower_quantile, upper_quantile
        )
        rng = np.random.default_rng(
            self.config.random_state if seed is None else seed
        )
        predictions: list[Array] = []
        for _ in range(num_samples):
            weights, biases, _samples, _epsilons = self._sample_parameters(rng)
            logits, _ = self._forward_with_parameters(features, weights, biases)
            if self.config.task_type == "binary_classification":
                predictions.append(_sigmoid(logits))
            elif self.config.task_type == "multiclass_classification":
                predictions.append(_softmax(logits))
            else:
                predictions.append(logits)
        samples = np.stack(predictions, axis=0)
        mean = np.mean(samples, axis=0)
        epistemic_std = np.std(samples, axis=0, ddof=0)
        result = {
            "mean": mean,
            "epistemic_std": epistemic_std,
            "lower": np.quantile(samples, lower_quantile, axis=0),
            "upper": np.quantile(samples, upper_quantile, axis=0),
        }
        if self.config.task_type == "regression":
            predictive_std = np.sqrt(
                epistemic_std**2 + self.config.likelihood_std**2
            )
            normal = NormalDist()
            result["predictive_std"] = predictive_std
            result["predictive_lower"] = mean + normal.inv_cdf(
                lower_quantile
            ) * predictive_std
            result["predictive_upper"] = mean + normal.inv_cdf(
                upper_quantile
            ) * predictive_std
        return result

    def predict_proba(
        self, x: Any, *, num_samples: int = 100, seed: int | None = None
    ) -> Array:
        if self.config.task_type == "regression":
            raise TuningValidationError(
                "predict_proba is unavailable for regression."
            )
        return self.predict_distribution(
            x, num_samples=num_samples, seed=seed
        )["mean"]

    def predict(
        self, x: Any, *, num_samples: int = 100, seed: int | None = None
    ) -> Array:
        mean = self.predict_distribution(
            x, num_samples=num_samples, seed=seed
        )["mean"]
        if self.config.task_type == "regression":
            return mean[:, 0] if mean.shape[1] == 1 else mean
        if self.config.task_type == "binary_classification":
            return (
                mean[:, 0] >= self.config.prediction_threshold
            ).astype(int)
        return np.argmax(mean, axis=1)

    def evaluate(
        self,
        x: Any,
        y: Any,
        *,
        targets_prepared: bool = False,
        num_samples: int = 100,
        seed: int | None = None,
    ) -> dict[str, float]:
        features = self._validate_features(x, "x")
        targets = (
            np.asarray(y, dtype=float)
            if targets_prepared
            else self._prepare_targets(y, len(features), "y")
        )
        distribution = self.predict_distribution(
            features, num_samples=num_samples, seed=seed
        )
        mean = distribution["mean"]
        if self.config.task_type == "regression":
            residual = mean - targets
            variance = np.maximum(
                distribution["predictive_std"] ** 2,
                self.config.stability_epsilon,
            )
            metrics = {
                "mse": float(np.mean(residual**2)),
                "rmse": float(math.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "predictive_nll": 0.5
                * float(
                    np.mean(
                        np.log(2.0 * math.pi * variance)
                        + residual**2 / variance
                    )
                ),
                "mean_epistemic_std": float(
                    np.mean(distribution["epistemic_std"])
                ),
            }
        elif self.config.task_type == "binary_classification":
            probabilities = np.maximum(
                np.minimum(mean, 1.0 - self.config.stability_epsilon),
                self.config.stability_epsilon,
            )
            predictions = (
                probabilities >= self.config.prediction_threshold
            ).astype(float)
            metrics = {
                "accuracy": float(np.mean(predictions == targets)),
                "log_loss": -float(
                    np.mean(
                        targets * np.log(probabilities)
                        + (1.0 - targets) * np.log(1.0 - probabilities)
                    )
                ),
                "brier": float(np.mean((probabilities - targets) ** 2)),
                "mean_epistemic_std": float(
                    np.mean(distribution["epistemic_std"])
                ),
            }
        else:
            probabilities = np.maximum(mean, self.config.stability_epsilon)
            truth = np.argmax(targets, axis=1)
            metrics = {
                "accuracy": float(np.mean(np.argmax(mean, axis=1) == truth)),
                "log_loss": -float(
                    np.mean(np.sum(targets * np.log(probabilities), axis=1))
                ),
                "mean_epistemic_std": float(
                    np.mean(distribution["epistemic_std"])
                ),
            }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise TuningEvaluationError("Evaluation produced non-finite metrics.")
        self.last_metrics = metrics
        return dict(metrics)

    def state_dict(self) -> dict[str, Any]:
        parameters = self._parameter_collections()
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "layer_sizes": list(self.layer_sizes),
            "config": self.get_config(),
            **{
                name: [item.copy() for item in values]
                for name, values in parameters.items()
            },
            "adam_m": {
                name: [item.copy() for item in values]
                for name, values in self._adam_m.items()
            },
            "adam_v": {
                name: [item.copy() for item in values]
                for name, values in self._adam_v.items()
            },
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
            raise TuningValidationError("Unsupported Bayesian-network state schema.")
        if tuple(state.get("layer_sizes", ())) != self.layer_sizes:
            raise TuningValidationError("State layer sizes do not match the model.")
        raw_config = state.get("config")
        if not isinstance(raw_config, Mapping) or dict(raw_config) != self.get_config():
            raise TuningValidationError("State configuration does not match the model.")
        reference = self._parameter_collections()
        loaded: dict[str, list[Array]] = {}
        for name, expected_values in reference.items():
            loaded[name] = self._load_array_collection(
                state.get(name), expected_values, name
            )
        adam_m = state.get("adam_m")
        adam_v = state.get("adam_v")
        if not isinstance(adam_m, Mapping) or not isinstance(adam_v, Mapping):
            raise TuningValidationError("State is missing Adam optimizer state.")
        loaded_adam_m = {
            name: self._load_array_collection(
                adam_m.get(name), expected, f"adam_m.{name}"
            )
            for name, expected in reference.items()
        }
        loaded_adam_v = {
            name: self._load_array_collection(
                adam_v.get(name), expected, f"adam_v.{name}"
            )
            for name, expected in reference.items()
        }
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
        # preserves the current posterior if a checkpoint is malformed.
        for name, values in loaded.items():
            setattr(self, name, values)
        self._adam_m = loaded_adam_m
        self._adam_v = loaded_adam_v
        self.training_steps = steps
        self.last_gradient_norm = gradient_norm
        self.last_metrics = last_metrics
        self.tuning_fit_parameters = loaded_fit_parameters
        self.rng.bit_generator.state = loaded_rng_state

    @staticmethod
    def _load_array_collection(
        raw: Any, expected: Sequence[Array], name: str
    ) -> list[Array]:
        if not isinstance(raw, Sequence) or len(raw) != len(expected):
            raise TuningValidationError(f"Invalid state collection {name!r}.")
        try:
            values = [np.asarray(item, dtype=float).copy() for item in raw]
        except (TypeError, ValueError) as exc:
            raise TuningValidationError(
                f"State collection {name!r} must be numeric."
            ) from exc
        if any(
            item.shape != reference.shape
            for item, reference in zip(values, expected, strict=True)
        ):
            raise TuningValidationError(
                f"State collection {name!r} has incompatible shapes."
            )
        if any(not np.isfinite(item).all() for item in values):
            raise TuningValidationError(
                f"State collection {name!r} contains non-finite values."
            )
        return values

    def get_config(self) -> dict[str, Any]:
        return {
            name: getattr(self.config, name)
            for name in BayesianNetworkConfig.__dataclass_fields__
        }

    @classmethod
    def from_tuning_params(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        parameters: Mapping[str, Any],
        base_config: Mapping[str, Any] | BayesianNetworkConfig | None = None,
        seed: int | None = None,
    ) -> "BayesianNeuralNetwork":
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
                "Unknown Bayesian-network tuning parameters.",
                details={"unknown_parameters": sorted(str(item) for item in unknown)},
            )
        hidden = cls.parse_hidden_layer_sizes(
            parameters.get("hidden_layer_sizes")
        )
        if base_config is None:
            config_values: dict[str, Any] = {}
        elif isinstance(base_config, BayesianNetworkConfig):
            config_values = {
                name: getattr(base_config, name)
                for name in BayesianNetworkConfig.__dataclass_fields__
            }
        else:
            config_values = {
                name: value
                for name, value in base_config.items()
                if name in BayesianNetworkConfig.__dataclass_fields__
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
            BayesianNetworkConfig(**config_values),
        )
        training_defaults = (
            base_config.get("training", {})
            if isinstance(base_config, Mapping)
            else {}
        )
        if training_defaults is not None and not isinstance(
            training_defaults, Mapping
        ):
            raise TuningConfigError("bnn.training must be a mapping.")
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
        num_samples: Any,
        validation_num_samples: Any,
        shuffle: Any,
        patience: Any,
        min_delta: Any,
        restore: Any,
    ) -> dict[str, Any]:
        for name, value in {
            "epochs": epochs,
            "batch_size": batch_size,
            "num_samples": num_samples,
            "validation_num_samples": validation_num_samples,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TuningValidationError(f"{name} must be a positive integer.")
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
            "num_samples": num_samples,
            "validation_num_samples": validation_num_samples,
            "shuffle": shuffle,
            "early_stopping_patience": patience,
            "min_delta": delta,
            "restore_best_weights": restore,
        }

    @staticmethod
    def _validate_prediction_options(
        num_samples: Any, lower_quantile: Any, upper_quantile: Any
    ) -> None:
        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples < 2
        ):
            raise TuningValidationError("num_samples must be an integer >= 2.")
        lower = _finite_float(lower_quantile, "lower_quantile")
        upper = _finite_float(upper_quantile, "upper_quantile")
        if not 0 < lower < upper < 1:
            raise TuningValidationError(
                "Prediction quantiles must satisfy 0 < lower < upper < 1."
            )

    def _assert_finite_parameters(self, operation: str) -> None:
        if any(
            not np.isfinite(item).all()
            for values in self._parameter_collections().values()
            for item in values
        ):
            raise TuningEvaluationError(
                "Variational parameters became non-finite.",
                context=TuningErrorContext(
                    component=self.__class__.__name__, operation=operation
                ),
            )


__all__ = [
    "BayesianNetworkConfig",
    "BayesianNeuralNetwork",
    "BayesianTrainingHistory",
]
