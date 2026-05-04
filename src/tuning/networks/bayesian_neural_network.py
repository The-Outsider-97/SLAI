from __future__ import annotations

import json
import math
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.tuning_error import (TuningOptimizationError, TuningPersistenceError,
                           TuningConfigError, safe_serialize, wrap_exception,
                           TuningErrorContext, error_boundary, raise_for_condition,
                           TuningEvaluationError, TuningValidationError)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Bayesian Neural Network")
printer = PrettyPrinter

Array = np.ndarray


@dataclass(slots=True)
class BNNTrainingHistory:
    """Structured training history for fit() runs."""

    epochs: List[int] = field(default_factory=list)
    train_elbo: List[float] = field(default_factory=list)
    train_kl: List[float] = field(default_factory=list)
    train_log_likelihood: List[float] = field(default_factory=list)
    validation_elbo: List[float] = field(default_factory=list)
    validation_kl: List[float] = field(default_factory=list)
    validation_log_likelihood: List[float] = field(default_factory=list)
    gradient_norm: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_validation_elbo: Optional[float] = None
    stopped_early: bool = False
    total_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": list(self.epochs),
            "train_elbo": list(self.train_elbo),
            "train_kl": list(self.train_kl),
            "train_log_likelihood": list(self.train_log_likelihood),
            "validation_elbo": list(self.validation_elbo),
            "validation_kl": list(self.validation_kl),
            "validation_log_likelihood": list(self.validation_log_likelihood),
            "gradient_norm": list(self.gradient_norm),
            "best_epoch": self.best_epoch,
            "best_validation_elbo": self.best_validation_elbo,
            "stopped_early": self.stopped_early,
            "total_steps": self.total_steps,
        }


class BayesianNeuralNetwork:
    """Fully-connected Bayesian neural network with diagonal Gaussian posteriors.

    This implementation remains NumPy-only for portability, while providing the
    operational safeguards expected in production code: typed failures,
    numerically safer optimization, configuration-driven defaults, structured
    persistence, and richer training / prediction APIs.
    """

    MODEL_FORMAT_VERSION = "1.0.0"
    SUPPORTED_HIDDEN_ACTIVATIONS = frozenset({"relu", "tanh", "leaky_relu"})

    def __init__(
        self,
        layer_sizes: Sequence[int],
        learning_rate: Optional[float] = None,
        prior_mu: Optional[float] = None,
        prior_logvar: Optional[float] = None,
        random_state: Optional[int] = None,
        logvar_clip_range: Optional[Tuple[float, float]] = None,
        gradient_clip_norm: Optional[float] = None,
        weight_init_scale: Optional[float] = None,
        hidden_activation: Optional[str] = None,
        likelihood_std: Optional[float] = None,
        min_variance: Optional[float] = None,
        stability_epsilon: Optional[float] = None,
        leaky_relu_slope: Optional[float] = None,
    ) -> None:
        self.config = load_global_config() or {}
        self.bnn_config = get_config_section("bnn") or {}

        self.layer_sizes = [int(v) for v in layer_sizes]
        self.learning_rate = float(self._resolve_setting("learning_rate", learning_rate, 0.01))
        self.prior_mu = float(self._resolve_setting("prior_mu", prior_mu, 0.0))
        self.prior_logvar = float(self._resolve_setting("prior_logvar", prior_logvar, 0.0))
        self.random_state = self._resolve_setting("random_state", random_state, None)
        self.logvar_clip_range = self._resolve_logvar_clip_range(logvar_clip_range)
        self.gradient_clip_norm = self._resolve_optional_positive_float(
            "gradient_clip_norm", gradient_clip_norm, 5.0, allow_none=True
        )
        self.weight_init_scale = self._resolve_optional_positive_float(
            "weight_init_scale", weight_init_scale, 1.0, allow_none=False
        )
        self.hidden_activation = str(self._resolve_setting("hidden_activation", hidden_activation, "relu")).strip().lower()
        self.likelihood_std = self._resolve_optional_positive_float(
            "likelihood_std", likelihood_std, 1.0, allow_none=False
        )
        self.min_variance = self._resolve_optional_positive_float(
            "min_variance", min_variance, 1e-6, allow_none=False
        )
        self.stability_epsilon = self._resolve_optional_positive_float(
            "stability_epsilon", stability_epsilon, 1e-8, allow_none=False
        )
        self.leaky_relu_slope = self._resolve_optional_positive_float(
            "leaky_relu_slope", leaky_relu_slope, 0.01, allow_none=False
        )

        self._validate_init_args()

        self.num_layers = len(self.layer_sizes) - 1
        self.rng = np.random.default_rng(self.random_state)
        self.training_steps = 0
        self.last_gradient_norm: Optional[float] = None
        self.last_metrics: Dict[str, float] = {}

        self.weights_mu: List[Array] = []
        self.weights_logvar: List[Array] = []
        self.biases_mu: List[Array] = []
        self.biases_logvar: List[Array] = []
        self._initialize_variational_parameters()
        self._validate_parameter_shapes()
        self._assert_all_parameters_finite(operation="post_initialization")

    def _resolve_setting(self, key: str, explicit: Any, default: Any) -> Any:
        if explicit is not None:
            return explicit
        return self.bnn_config.get(key, default)

    def _resolve_optional_positive_float(
        self,
        key: str,
        explicit: Optional[float],
        default: float,
        *,
        allow_none: bool,
    ) -> Optional[float]:
        value = self._resolve_setting(key, explicit, default)
        if value is None:
            return None if allow_none else float(default)
        return float(value)

    def _resolve_logvar_clip_range(self, explicit: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        value = explicit if explicit is not None else self.bnn_config.get("logvar_clip_range", [-8.0, 4.0])
        if isinstance(value, list):
            value = tuple(value)
        return tuple(value)  # type: ignore[return-value]

    def _context(self, operation: str, **kwargs: Any) -> TuningErrorContext:
        return TuningErrorContext(
            component="BayesianNeuralNetwork",
            operation=operation,
            strategy="variational_inference",
            model_type="bayesian_neural_network",
            random_state=self.random_state,
            config_path=str(self.config.get("__config_path__", "")) or None,
            parameters={
                "layer_sizes": list(self.layer_sizes),
                "learning_rate": self.learning_rate,
                "hidden_activation": self.hidden_activation,
                **{key: value for key, value in kwargs.items() if value is not None},
            },
        )

    def _validate_init_args(self) -> None:
        context = self._context("__init__")
        raise_for_condition(
            len(self.layer_sizes) < 2,
            "layer_sizes must include at least input and output dimensions.",
            error_cls=TuningConfigError,
            context=context,
            details={"layer_sizes": self.layer_sizes},
        )
        raise_for_condition(
            any(size <= 0 for size in self.layer_sizes),
            "All layer sizes must be positive integers.",
            error_cls=TuningConfigError,
            context=context,
            details={"layer_sizes": self.layer_sizes},
        )
        raise_for_condition(
            not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0,
            "learning_rate must be a positive finite float.",
            error_cls=TuningConfigError,
            context=context,
            details={"learning_rate": self.learning_rate},
        )
        raise_for_condition(
            self.hidden_activation not in self.SUPPORTED_HIDDEN_ACTIVATIONS,
            "hidden_activation must be one of: relu, tanh, leaky_relu.",
            error_cls=TuningConfigError,
            context=context,
            details={
                "hidden_activation": self.hidden_activation,
                "supported_values": sorted(self.SUPPORTED_HIDDEN_ACTIVATIONS),
            },
        )
        raise_for_condition(
            not (isinstance(self.logvar_clip_range, tuple) and len(self.logvar_clip_range) == 2),
            "logvar_clip_range must be a two-item tuple or list.",
            error_cls=TuningConfigError,
            context=context,
            details={"logvar_clip_range": safe_serialize(self.logvar_clip_range)},
        )
        logvar_min, logvar_max = self.logvar_clip_range
        raise_for_condition(
            not math.isfinite(logvar_min) or not math.isfinite(logvar_max) or logvar_min >= logvar_max,
            "logvar_clip_range must contain finite values with min < max.",
            error_cls=TuningConfigError,
            context=context,
            details={"logvar_clip_range": list(self.logvar_clip_range)},
        )
        for field_name, value in {
            "prior_mu": self.prior_mu,
            "prior_logvar": self.prior_logvar,
            "weight_init_scale": self.weight_init_scale,
            "likelihood_std": self.likelihood_std,
            "min_variance": self.min_variance,
            "stability_epsilon": self.stability_epsilon,
            "leaky_relu_slope": self.leaky_relu_slope,
        }.items():
            raise_for_condition(
                value is None or not math.isfinite(float(value)),
                f"{field_name} must be finite.",
                error_cls=TuningConfigError,
                context=context,
                details={field_name: value},
            )
        raise_for_condition(
            self.weight_init_scale is None or self.weight_init_scale <= 0.0,
            "weight_init_scale must be strictly positive.",
            error_cls=TuningConfigError,
            context=context,
            details={"weight_init_scale": self.weight_init_scale},
        )
        raise_for_condition(
            self.likelihood_std is None or self.likelihood_std <= 0.0,
            "likelihood_std must be strictly positive.",
            error_cls=TuningConfigError,
            context=context,
            details={"likelihood_std": self.likelihood_std},
        )
        raise_for_condition(
            self.min_variance is None or self.min_variance <= 0.0,
            "min_variance must be strictly positive.",
            error_cls=TuningConfigError,
            context=context,
            details={"min_variance": self.min_variance},
        )
        raise_for_condition(
            self.stability_epsilon is None or self.stability_epsilon <= 0.0,
            "stability_epsilon must be strictly positive.",
            error_cls=TuningConfigError,
            context=context,
            details={"stability_epsilon": self.stability_epsilon},
        )
        raise_for_condition(
            self.leaky_relu_slope is None or self.leaky_relu_slope <= 0.0,
            "leaky_relu_slope must be strictly positive.",
            error_cls=TuningConfigError,
            context=context,
            details={"leaky_relu_slope": self.leaky_relu_slope},
        )
        if self.gradient_clip_norm is not None:
            raise_for_condition(
                not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0,
                "gradient_clip_norm must be a positive finite float when provided.",
                error_cls=TuningConfigError,
                context=context,
                details={"gradient_clip_norm": self.gradient_clip_norm},
            )
        if self.random_state is not None:
            raise_for_condition(
                not isinstance(self.random_state, (int, np.integer)),
                "random_state must be an integer or None.",
                error_cls=TuningConfigError,
                context=context,
                details={"random_state": self.random_state},
            )

    def _initial_weight_scale(self, fan_in: int) -> float:
        if self.hidden_activation == "relu":
            base = math.sqrt(2.0 / fan_in)
        elif self.hidden_activation == "leaky_relu":
            negative_slope = float(self.leaky_relu_slope)
            base = math.sqrt(2.0 / ((1.0 + negative_slope**2) * fan_in))
        else:
            base = math.sqrt(1.0 / fan_in)
        return base * float(self.weight_init_scale)

    def _initialize_variational_parameters(self) -> None:
        try:
            initial_logvar = max(self.logvar_clip_range[0], math.log(max(float(self.min_variance), 1e-6)))
            for layer_idx in range(self.num_layers):
                fan_in = self.layer_sizes[layer_idx]
                fan_out = self.layer_sizes[layer_idx + 1]
                scale = self._initial_weight_scale(fan_in)

                self.weights_mu.append(
                    self.rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out)).astype(float)
                )
                self.weights_logvar.append(np.full((fan_in, fan_out), initial_logvar, dtype=float))
                self.biases_mu.append(np.zeros(fan_out, dtype=float))
                self.biases_logvar.append(np.full(fan_out, initial_logvar, dtype=float))
        except Exception as exc:  # noqa: BLE001
            raise wrap_exception(
                exc,
                message="Failed to initialize Bayesian neural network variational parameters.",
                error_cls=TuningConfigError,
                context=self._context("initialize_parameters"),
                details={"layer_sizes": self.layer_sizes},
            ) from exc

    def _activation(self, values: Array) -> Array:
        if self.hidden_activation == "relu":
            return np.maximum(0.0, values)
        if self.hidden_activation == "tanh":
            return np.tanh(values)
        return np.where(values > 0.0, values, self.leaky_relu_slope * values)

    def _activation_derivative(self, pre_activation: Array) -> Array:
        if self.hidden_activation == "relu":
            return (pre_activation > 0.0).astype(float)
        if self.hidden_activation == "tanh":
            activated = np.tanh(pre_activation)
            return 1.0 - activated**2
        derivative = np.ones_like(pre_activation, dtype=float)
        derivative[pre_activation <= 0.0] = self.leaky_relu_slope
        return derivative

    def _posterior_std(self, logvar: Array) -> Array:
        clipped = np.clip(logvar, *self.logvar_clip_range)
        variance = np.exp(clipped)
        return np.sqrt(np.maximum(variance, self.min_variance))

    def _likelihood_variance(self) -> float:
        return max(float(self.likelihood_std) ** 2, float(self.min_variance))

    def _shape_string(self, value: Array) -> str:
        return "x".join(str(dim) for dim in value.shape)

    def _assert_finite_array(self, value: Array, *, name: str, operation: str) -> None:
        if not np.isfinite(value).all():
            raise TuningEvaluationError(
                f"Non-finite values detected in {name} during {operation}.",
                context=self._context(operation),
                details={
                    "name": name,
                    "shape": self._shape_string(np.asarray(value)),
                },
            )

    def _assert_all_parameters_finite(self, *, operation: str) -> None:
        for collection_name, collection in {
            "weights_mu": self.weights_mu,
            "weights_logvar": self.weights_logvar,
            "biases_mu": self.biases_mu,
            "biases_logvar": self.biases_logvar,
        }.items():
            for layer_idx, parameter in enumerate(collection):
                self._assert_finite_array(
                    np.asarray(parameter, dtype=float),
                    name=f"{collection_name}[{layer_idx}]",
                    operation=operation,
                )

    def _validate_forward_inputs(self, x: Array, weights: Sequence[Array], biases: Sequence[Array]) -> Array:
        x_array = np.asarray(x, dtype=float)
        raise_for_condition(
            x_array.ndim != 2,
            "x must be a 2D array of shape (batch_size, input_dim).",
            error_cls=TuningValidationError,
            context=self._context("forward_validation"),
            details={"x_shape": list(x_array.shape)},
        )
        raise_for_condition(
            x_array.shape[1] != self.layer_sizes[0],
            "Input feature dimension mismatch.",
            error_cls=TuningValidationError,
            context=self._context("forward_validation"),
            details={"expected_input_dim": self.layer_sizes[0], "received_input_dim": int(x_array.shape[1])},
        )
        raise_for_condition(
            len(weights) != self.num_layers or len(biases) != self.num_layers,
            "weights and biases must match network layer count.",
            error_cls=TuningValidationError,
            context=self._context("forward_validation"),
            details={"expected_layers": self.num_layers, "weights_layers": len(weights), "bias_layers": len(biases)},
        )
        for layer_idx in range(self.num_layers):
            expected_weight_shape = (self.layer_sizes[layer_idx], self.layer_sizes[layer_idx + 1])
            expected_bias_shape = (self.layer_sizes[layer_idx + 1],)
            raise_for_condition(
                np.asarray(weights[layer_idx]).shape != expected_weight_shape,
                f"Invalid weight shape at layer {layer_idx}.",
                error_cls=TuningValidationError,
                context=self._context("forward_validation"),
                details={"expected_shape": list(expected_weight_shape), "received_shape": list(np.asarray(weights[layer_idx]).shape)},
            )
            raise_for_condition(
                np.asarray(biases[layer_idx]).shape != expected_bias_shape,
                f"Invalid bias shape at layer {layer_idx}.",
                error_cls=TuningValidationError,
                context=self._context("forward_validation"),
                details={"expected_shape": list(expected_bias_shape), "received_shape": list(np.asarray(biases[layer_idx]).shape)},
            )
        self._assert_finite_array(x_array, name="x", operation="forward_validation")
        return x_array

    def _validate_training_batch(self, x: Array, y: Array, *, operation: str) -> Tuple[Array, Array]:
        x_batch = np.asarray(x, dtype=float)
        y_batch = np.asarray(y, dtype=float)

        raise_for_condition(
            x_batch.ndim != 2,
            "x_batch must be a 2D array.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"x_shape": list(x_batch.shape)},
        )
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)
        raise_for_condition(
            y_batch.ndim != 2,
            "y_batch must be a 1D or 2D array.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"y_shape": list(y_batch.shape)},
        )
        raise_for_condition(
            x_batch.shape[0] != y_batch.shape[0],
            "x_batch and y_batch must have matching sample counts.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"x_rows": int(x_batch.shape[0]), "y_rows": int(y_batch.shape[0])},
        )
        raise_for_condition(
            x_batch.shape[1] != self.layer_sizes[0],
            "Input feature dimension mismatch.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"expected_input_dim": self.layer_sizes[0], "received_input_dim": int(x_batch.shape[1])},
        )
        raise_for_condition(
            y_batch.shape[1] != self.layer_sizes[-1],
            "Target feature dimension mismatch.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"expected_target_dim": self.layer_sizes[-1], "received_target_dim": int(y_batch.shape[1])},
        )
        self._assert_finite_array(x_batch, name="x_batch", operation=operation)
        self._assert_finite_array(y_batch, name="y_batch", operation=operation)
        return x_batch, y_batch

    def _validate_sample_count(self, num_samples: int, *, operation: str) -> int:
        sample_count = int(num_samples)
        raise_for_condition(
            sample_count < 1,
            "num_samples must be >= 1.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"num_samples": num_samples},
        )
        return sample_count

    def _resolve_dataset_size(self, dataset_size: Optional[int], batch_size: int, *, operation: str) -> int:
        resolved = int(dataset_size) if dataset_size is not None else int(batch_size)
        raise_for_condition(
            resolved < batch_size,
            "dataset_size must be >= current batch size.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"dataset_size": resolved, "batch_size": batch_size},
        )
        return resolved

    def _sample_parameters_with_noise(self) -> Tuple[List[Array], List[Array], List[Array], List[Array]]:
        sampled_weights: List[Array] = []
        sampled_biases: List[Array] = []
        epsilons_w: List[Array] = []
        epsilons_b: List[Array] = []

        for layer_idx in range(self.num_layers):
            eps_w = self.rng.standard_normal(self.weights_mu[layer_idx].shape)
            eps_b = self.rng.standard_normal(self.biases_mu[layer_idx].shape)
            std_w = self._posterior_std(self.weights_logvar[layer_idx])
            std_b = self._posterior_std(self.biases_logvar[layer_idx])

            sampled_weights.append(self.weights_mu[layer_idx] + eps_w * std_w)
            sampled_biases.append(self.biases_mu[layer_idx] + eps_b * std_b)
            epsilons_w.append(eps_w)
            epsilons_b.append(eps_b)

        return sampled_weights, sampled_biases, epsilons_w, epsilons_b

    def sample_parameters(self) -> Tuple[List[Array], List[Array]]:
        sampled_weights, sampled_biases, _, _ = self._sample_parameters_with_noise()
        return sampled_weights, sampled_biases

    @error_boundary(
        error_cls=TuningEvaluationError,
        message="Bayesian neural network forward pass failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("forward") if args else None,
        detail_builder=lambda exc, args, kwargs: {"error": exc.__class__.__name__},
    )
    def forward(
        self,
        x: Array,
        weights: Optional[Sequence[Array]] = None,
        biases: Optional[Sequence[Array]] = None,
        *,
        return_cache: bool = False,
    ) -> Array | Tuple[Array, List[Array], List[Array]]:
        weights_seq = list(weights if weights is not None else self.weights_mu)
        biases_seq = list(biases if biases is not None else self.biases_mu)
        x_array = self._validate_forward_inputs(x, weights_seq, biases_seq)

        activation = x_array
        activations: List[Array] = [x_array]
        pre_activations: List[Array] = []

        for layer_idx in range(self.num_layers - 1):
            z = activation @ np.asarray(weights_seq[layer_idx], dtype=float) + np.asarray(biases_seq[layer_idx], dtype=float)
            pre_activations.append(z)
            activation = self._activation(z)
            activations.append(activation)

        outputs = activation @ np.asarray(weights_seq[-1], dtype=float) + np.asarray(biases_seq[-1], dtype=float)
        self._assert_finite_array(outputs, name="outputs", operation="forward")

        if return_cache:
            return outputs, activations, pre_activations
        return outputs

    def _kl_divergence(self, mu: Array, logvar: Array) -> float:
        prior_var = max(math.exp(self.prior_logvar), float(self.min_variance))
        clipped_logvar = np.clip(logvar, *self.logvar_clip_range)
        variance = np.maximum(np.exp(clipped_logvar), self.min_variance)
        divergence = 0.5 * np.sum(
            (variance + (mu - self.prior_mu) ** 2) / prior_var
            - 1.0
            + (self.prior_logvar - clipped_logvar)
        )
        return float(divergence)

    def _total_kl_divergence(self) -> float:
        total = 0.0
        for layer_idx in range(self.num_layers):
            total += self._kl_divergence(self.weights_mu[layer_idx], self.weights_logvar[layer_idx])
            total += self._kl_divergence(self.biases_mu[layer_idx], self.biases_logvar[layer_idx])
        return float(total)

    def _estimate_expected_log_likelihood(self, x: Array, y: Array, num_samples: int) -> float:
        likelihood_variance = self._likelihood_variance()
        batch_size = x.shape[0]
        total = 0.0
        for _ in range(num_samples):
            sampled_weights, sampled_biases = self.sample_parameters()
            outputs = self.forward(x, sampled_weights, sampled_biases)
            residual = y - outputs
            total += float(-0.5 * np.sum((residual**2) / likelihood_variance))
        return total / (num_samples * batch_size)

    @error_boundary(
        error_cls=TuningEvaluationError,
        message="Failed to evaluate Bayesian neural network ELBO.",
        context_builder=lambda exc, args, kwargs: args[0]._context("elbo") if args else None,
        detail_builder=lambda exc, args, kwargs: {"num_samples": kwargs.get("num_samples", args[3] if len(args) > 3 else 1)},
    )
    def elbo(
        self,
        x: Array,
        y: Array,
        num_samples: int = 1,
        *,
        dataset_size: Optional[int] = None,
    ) -> Tuple[float, float]:
        x_batch, y_batch = self._validate_training_batch(x, y, operation="elbo")
        sample_count = self._validate_sample_count(num_samples, operation="elbo")
        resolved_dataset_size = self._resolve_dataset_size(dataset_size, x_batch.shape[0], operation="elbo")

        kl_divergence = self._total_kl_divergence()
        log_likelihood = self._estimate_expected_log_likelihood(x_batch, y_batch, sample_count)
        elbo_value = log_likelihood - (kl_divergence / resolved_dataset_size)
        return float(elbo_value), float(kl_divergence)

    @error_boundary(
        error_cls=TuningEvaluationError,
        message="Failed to evaluate Bayesian neural network metrics.",
        context_builder=lambda exc, args, kwargs: args[0]._context("evaluate") if args else None,
    )
    def evaluate(
        self,
        x: Array,
        y: Array,
        num_samples: int = 20,
        *,
        dataset_size: Optional[int] = None,
    ) -> Dict[str, float]:
        x_batch, y_batch = self._validate_training_batch(x, y, operation="evaluate")
        sample_count = self._validate_sample_count(num_samples, operation="evaluate")
        resolved_dataset_size = self._resolve_dataset_size(dataset_size, x_batch.shape[0], operation="evaluate")

        predictive_mean, predictive_std = self.predict(x_batch, num_samples=sample_count)
        residual = y_batch - predictive_mean
        mse = float(np.mean(residual**2))
        rmse = float(math.sqrt(mse))
        mean_predictive_std = float(np.mean(predictive_std))
        elbo_value, kl_divergence = self.elbo(
            x_batch,
            y_batch,
            num_samples=sample_count,
            dataset_size=resolved_dataset_size,
        )
        metrics = {
            "elbo": float(elbo_value),
            "kl_divergence": float(kl_divergence),
            "log_likelihood": float(elbo_value + kl_divergence / resolved_dataset_size),
            "mse": mse,
            "rmse": rmse,
            "mean_predictive_std": mean_predictive_std,
        }
        self.last_metrics = dict(metrics)
        return metrics

    def _compute_gradients(
        self,
        x: Array,
        y: Array,
        *,
        num_samples: int,
        dataset_size: int,
    ) -> Dict[str, List[Array]]:
        gradients: Dict[str, List[Array]] = {
            "weights_mu": [np.zeros_like(w, dtype=float) for w in self.weights_mu],
            "weights_logvar": [np.zeros_like(w, dtype=float) for w in self.weights_logvar],
            "biases_mu": [np.zeros_like(b, dtype=float) for b in self.biases_mu],
            "biases_logvar": [np.zeros_like(b, dtype=float) for b in self.biases_logvar],
        }

        batch_size = x.shape[0]
        likelihood_variance = self._likelihood_variance()
        prior_variance = max(math.exp(self.prior_logvar), float(self.min_variance))

        for _ in range(num_samples):
            sampled_weights, sampled_biases, epsilons_w, epsilons_b = self._sample_parameters_with_noise()
            outputs, activations, pre_activations = self.forward(
                x,
                sampled_weights,
                sampled_biases,
                return_cache=True,
            )

            delta = (y - outputs) / (likelihood_variance * batch_size * num_samples)
            self._assert_finite_array(delta, name="delta", operation="gradient_computation")

            for layer_idx in range(self.num_layers - 1, -1, -1):
                local_activation = activations[layer_idx]
                d_w = local_activation.T @ delta
                d_b = np.sum(delta, axis=0)

                std_w = self._posterior_std(self.weights_logvar[layer_idx])
                std_b = self._posterior_std(self.biases_logvar[layer_idx])

                gradients["weights_mu"][layer_idx] += d_w
                gradients["weights_logvar"][layer_idx] += d_w * epsilons_w[layer_idx] * 0.5 * std_w
                gradients["biases_mu"][layer_idx] += d_b
                gradients["biases_logvar"][layer_idx] += d_b * epsilons_b[layer_idx] * 0.5 * std_b

                if layer_idx > 0:
                    delta = (delta @ sampled_weights[layer_idx].T) * self._activation_derivative(pre_activations[layer_idx - 1])

        kl_scale = 1.0 / dataset_size
        for layer_idx in range(self.num_layers):
            variance_w = np.maximum(np.exp(np.clip(self.weights_logvar[layer_idx], *self.logvar_clip_range)), self.min_variance)
            variance_b = np.maximum(np.exp(np.clip(self.biases_logvar[layer_idx], *self.logvar_clip_range)), self.min_variance)

            gradients["weights_mu"][layer_idx] -= kl_scale * (self.weights_mu[layer_idx] - self.prior_mu) / prior_variance
            gradients["weights_logvar"][layer_idx] -= kl_scale * 0.5 * (variance_w / prior_variance - 1.0)
            gradients["biases_mu"][layer_idx] -= kl_scale * (self.biases_mu[layer_idx] - self.prior_mu) / prior_variance
            gradients["biases_logvar"][layer_idx] -= kl_scale * 0.5 * (variance_b / prior_variance - 1.0)

        self._assert_gradient_finiteness(gradients)
        return gradients

    def _assert_gradient_finiteness(self, gradients: Mapping[str, Sequence[Array]]) -> None:
        for gradient_name, gradient_values in gradients.items():
            for layer_idx, gradient in enumerate(gradient_values):
                self._assert_finite_array(
                    np.asarray(gradient, dtype=float),
                    name=f"{gradient_name}[{layer_idx}]",
                    operation="gradient_validation",
                )

    def _global_gradient_norm(self, gradients: Mapping[str, Sequence[Array]]) -> float:
        squared_norm = 0.0
        for gradient_values in gradients.values():
            for gradient in gradient_values:
                squared_norm += float(np.sum(np.asarray(gradient, dtype=float) ** 2))
        return float(math.sqrt(max(squared_norm, 0.0)))

    def _clip_gradients(self, gradients: Dict[str, List[Array]]) -> float:
        gradient_norm = self._global_gradient_norm(gradients)
        if self.gradient_clip_norm is not None and gradient_norm > self.gradient_clip_norm:
            scale = self.gradient_clip_norm / (gradient_norm + self.stability_epsilon)
            for gradient_name, gradient_values in gradients.items():
                gradients[gradient_name] = [gradient * scale for gradient in gradient_values]
            gradient_norm = self._global_gradient_norm(gradients)
        self.last_gradient_norm = gradient_norm
        return gradient_norm

    @error_boundary(
        error_cls=TuningOptimizationError,
        message="Bayesian neural network training step failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("train_step") if args else None,
    )
    def train_step(
        self,
        x_batch: Array,
        y_batch: Array,
        num_samples: int = 1,
        *,
        dataset_size: Optional[int] = None,
    ) -> Tuple[float, float]:
        x_valid, y_valid = self._validate_training_batch(x_batch, y_batch, operation="train_step")
        sample_count = self._validate_sample_count(num_samples, operation="train_step")
        resolved_dataset_size = self._resolve_dataset_size(dataset_size, x_valid.shape[0], operation="train_step")

        gradients = self._compute_gradients(
            x_valid,
            y_valid,
            num_samples=sample_count,
            dataset_size=resolved_dataset_size,
        )
        gradient_norm = self._clip_gradients(gradients)

        for layer_idx in range(self.num_layers):
            self.weights_mu[layer_idx] += self.learning_rate * gradients["weights_mu"][layer_idx]
            self.weights_logvar[layer_idx] += self.learning_rate * gradients["weights_logvar"][layer_idx]
            self.biases_mu[layer_idx] += self.learning_rate * gradients["biases_mu"][layer_idx]
            self.biases_logvar[layer_idx] += self.learning_rate * gradients["biases_logvar"][layer_idx]

            self.weights_logvar[layer_idx] = np.clip(self.weights_logvar[layer_idx], *self.logvar_clip_range)
            self.biases_logvar[layer_idx] = np.clip(self.biases_logvar[layer_idx], *self.logvar_clip_range)

        self.training_steps += 1
        self._assert_all_parameters_finite(operation="post_train_step")

        elbo_value, kl_value = self.elbo(
            x_valid,
            y_valid,
            num_samples=max(1, min(sample_count, 5)),
            dataset_size=resolved_dataset_size,
        )
        self.last_metrics = {
            "elbo": float(elbo_value),
            "kl_divergence": float(kl_value),
            "gradient_norm": float(gradient_norm),
        }
        return float(elbo_value), float(kl_value)

    def _resolve_training_defaults(
        self,
        *,
        epochs: Optional[int],
        batch_size: Optional[int],
        num_samples: Optional[int],
        shuffle: Optional[bool],
        early_stopping_patience: Optional[int],
        min_delta: Optional[float],
    ) -> Dict[str, Any]:
        training_cfg = self.bnn_config.get("training", {}) if isinstance(self.bnn_config.get("training", {}), Mapping) else {}
        resolved = {
            "epochs": int(training_cfg.get("epochs", 100) if epochs is None else epochs),
            "batch_size": int(training_cfg.get("batch_size", 32) if batch_size is None else batch_size),
            "num_samples": int(training_cfg.get("num_samples", 3) if num_samples is None else num_samples),
            "shuffle": bool(training_cfg.get("shuffle", True) if shuffle is None else shuffle),
            "early_stopping_patience": (
                training_cfg.get("early_stopping_patience", 10)
                if early_stopping_patience is None
                else early_stopping_patience
            ),
            "min_delta": float(training_cfg.get("min_delta", 1e-4) if min_delta is None else min_delta),
        }
        return resolved

    @error_boundary(
        error_cls=TuningOptimizationError,
        message="Bayesian neural network fit() failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("fit") if args else None,
    )
    def fit(
        self,
        x_train: Array,
        y_train: Array,
        *,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_samples: Optional[int] = None,
        shuffle: Optional[bool] = None,
        validation_data: Optional[Tuple[Array, Array]] = None,
        early_stopping_patience: Optional[int] = None,
        min_delta: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        train_x, train_y = self._validate_training_batch(x_train, y_train, operation="fit")
        defaults = self._resolve_training_defaults(
            epochs=epochs,
            batch_size=batch_size,
            num_samples=num_samples,
            shuffle=shuffle,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
        )

        raise_for_condition(
            defaults["epochs"] < 1,
            "epochs must be >= 1.",
            error_cls=TuningValidationError,
            context=self._context("fit"),
            details={"epochs": defaults["epochs"]},
        )
        raise_for_condition(
            defaults["batch_size"] < 1,
            "batch_size must be >= 1.",
            error_cls=TuningValidationError,
            context=self._context("fit"),
            details={"batch_size": defaults["batch_size"]},
        )
        raise_for_condition(
            defaults["num_samples"] < 1,
            "num_samples must be >= 1.",
            error_cls=TuningValidationError,
            context=self._context("fit"),
            details={"num_samples": defaults["num_samples"]},
        )
        raise_for_condition(
            defaults["min_delta"] < 0.0,
            "min_delta must be >= 0.",
            error_cls=TuningValidationError,
            context=self._context("fit"),
            details={"min_delta": defaults["min_delta"]},
        )

        val_x: Optional[Array] = None
        val_y: Optional[Array] = None
        if validation_data is not None:
            val_x, val_y = self._validate_training_batch(
                validation_data[0],
                validation_data[1],
                operation="fit_validation_data",
            )

        history = BNNTrainingHistory()
        best_state: Optional[Dict[str, Any]] = None
        best_validation_elbo = -math.inf
        patience_counter = 0
        num_rows = train_x.shape[0]

        for epoch in range(1, defaults["epochs"] + 1):
            indices = np.arange(num_rows)
            if defaults["shuffle"]:
                self.rng.shuffle(indices)
            shuffled_x = train_x[indices]
            shuffled_y = train_y[indices]

            epoch_elbos: List[float] = []
            epoch_kls: List[float] = []
            epoch_grad_norms: List[float] = []

            for start in range(0, num_rows, defaults["batch_size"]):
                end = min(start + defaults["batch_size"], num_rows)
                batch_x = shuffled_x[start:end]
                batch_y = shuffled_y[start:end]
                elbo_value, kl_value = self.train_step(
                    batch_x,
                    batch_y,
                    num_samples=defaults["num_samples"],
                    dataset_size=num_rows,
                )
                epoch_elbos.append(float(elbo_value))
                epoch_kls.append(float(kl_value))
                epoch_grad_norms.append(float(self.last_gradient_norm or 0.0))
                history.total_steps += 1

            train_metrics = self.evaluate(
                train_x,
                train_y,
                num_samples=max(5, defaults["num_samples"]),
                dataset_size=num_rows,
            )

            history.epochs.append(epoch)
            history.train_elbo.append(float(train_metrics["elbo"]))
            history.train_kl.append(float(train_metrics["kl_divergence"]))
            history.train_log_likelihood.append(float(train_metrics["log_likelihood"]))
            history.gradient_norm.append(float(np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0))

            if val_x is not None and val_y is not None:
                validation_metrics = self.evaluate(
                    val_x,
                    val_y,
                    num_samples=max(5, defaults["num_samples"]),
                    dataset_size=val_x.shape[0],
                )
                current_val_elbo = float(validation_metrics["elbo"])
                history.validation_elbo.append(current_val_elbo)
                history.validation_kl.append(float(validation_metrics["kl_divergence"]))
                history.validation_log_likelihood.append(float(validation_metrics["log_likelihood"]))

                if current_val_elbo > best_validation_elbo + defaults["min_delta"]:
                    best_validation_elbo = current_val_elbo
                    history.best_validation_elbo = current_val_elbo
                    history.best_epoch = epoch
                    patience_counter = 0
                    best_state = self.to_serializable_dict(include_history=False)
                else:
                    patience_counter += 1
                    if defaults["early_stopping_patience"] is not None and patience_counter >= int(
                        defaults["early_stopping_patience"]
                    ):
                        history.stopped_early = True
                        break
            else:
                history.validation_elbo.append(float("nan"))
                history.validation_kl.append(float("nan"))
                history.validation_log_likelihood.append(float("nan"))

            if verbose:
                logger.info(
                    "BNN epoch %s/%s | train_elbo=%.6f | train_kl=%.6f | grad_norm=%.6f",
                    epoch,
                    defaults["epochs"],
                    history.train_elbo[-1],
                    history.train_kl[-1],
                    history.gradient_norm[-1],
                )

        if history.stopped_early and best_state is not None:
            self._load_from_payload(best_state, validate_shapes=True)

        return history.to_dict()

    @error_boundary(
        error_cls=TuningEvaluationError,
        message="Bayesian neural network prediction failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("predict") if args else None,
    )
    def predict(self, x: Array, num_samples: int = 100) -> Tuple[Array, Array]:
        x_array = np.asarray(x, dtype=float)
        raise_for_condition(
            x_array.ndim != 2,
            "x must be a 2D array of shape (batch_size, input_dim).",
            error_cls=TuningValidationError,
            context=self._context("predict"),
            details={"x_shape": list(x_array.shape)},
        )
        raise_for_condition(
            x_array.shape[1] != self.layer_sizes[0],
            "Input feature dimension mismatch.",
            error_cls=TuningValidationError,
            context=self._context("predict"),
            details={"expected_input_dim": self.layer_sizes[0], "received_input_dim": int(x_array.shape[1])},
        )
        sample_count = self._validate_sample_count(num_samples, operation="predict")
        self._assert_finite_array(x_array, name="x", operation="predict")

        predictions = []
        for _ in range(sample_count):
            sampled_weights, sampled_biases = self.sample_parameters()
            predictions.append(self.forward(x_array, sampled_weights, sampled_biases))

        stacked = np.stack(predictions, axis=0)
        self._assert_finite_array(stacked, name="prediction_stack", operation="predict")
        return np.mean(stacked, axis=0), np.std(stacked, axis=0)

    @error_boundary(
        error_cls=TuningEvaluationError,
        message="Bayesian neural network predictive interval computation failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("predict_interval") if args else None,
    )
    def predict_interval(
        self,
        x: Array,
        *,
        num_samples: int = 200,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
    ) -> Dict[str, Array]:
        raise_for_condition(
            not 0.0 <= lower_quantile < upper_quantile <= 1.0,
            "Quantiles must satisfy 0 <= lower < upper <= 1.",
            error_cls=TuningValidationError,
            context=self._context("predict_interval"),
            details={"lower_quantile": lower_quantile, "upper_quantile": upper_quantile},
        )
        x_array = np.asarray(x, dtype=float)
        sample_count = self._validate_sample_count(num_samples, operation="predict_interval")

        predictions = []
        for _ in range(sample_count):
            sampled_weights, sampled_biases = self.sample_parameters()
            predictions.append(self.forward(x_array, sampled_weights, sampled_biases))

        stacked = np.stack(predictions, axis=0)
        return {
            "mean": np.mean(stacked, axis=0),
            "std": np.std(stacked, axis=0),
            "lower": np.quantile(stacked, lower_quantile, axis=0),
            "upper": np.quantile(stacked, upper_quantile, axis=0),
        }

    def to_serializable_dict(self, *, include_history: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_type": "BayesianNeuralNetwork",
            "model_format_version": self.MODEL_FORMAT_VERSION,
            "layer_sizes": list(self.layer_sizes),
            "learning_rate": self.learning_rate,
            "prior_mu": self.prior_mu,
            "prior_logvar": self.prior_logvar,
            "random_state": self.random_state,
            "logvar_clip_range": list(self.logvar_clip_range),
            "gradient_clip_norm": self.gradient_clip_norm,
            "weight_init_scale": self.weight_init_scale,
            "hidden_activation": self.hidden_activation,
            "likelihood_std": self.likelihood_std,
            "min_variance": self.min_variance,
            "stability_epsilon": self.stability_epsilon,
            "leaky_relu_slope": self.leaky_relu_slope,
            "training_steps": self.training_steps,
            "weights_mu": [weights.tolist() for weights in self.weights_mu],
            "weights_logvar": [weights.tolist() for weights in self.weights_logvar],
            "biases_mu": [bias.tolist() for bias in self.biases_mu],
            "biases_logvar": [bias.tolist() for bias in self.biases_logvar],
            "config_path": self.config.get("__config_path__"),
        }
        if include_history and self.last_metrics:
            payload["last_metrics"] = dict(self.last_metrics)
        return payload

    @error_boundary(
        error_cls=TuningPersistenceError,
        message="Failed to persist Bayesian neural network model.",
        context_builder=lambda exc, args, kwargs: args[0]._context("save") if args else None,
        detail_builder=lambda exc, args, kwargs: {"output": kwargs.get("filename", args[1] if len(args) > 1 else None)},
    )
    def save(self, filename: str) -> None:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_serializable_dict(include_history=True)

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _load_from_payload(self, payload: Mapping[str, Any], *, validate_shapes: bool) -> None:
        try:
            self.weights_mu = [np.asarray(weights, dtype=float) for weights in payload["weights_mu"]]
            self.weights_logvar = [np.asarray(weights, dtype=float) for weights in payload["weights_logvar"]]
            self.biases_mu = [np.asarray(bias, dtype=float) for bias in payload["biases_mu"]]
            self.biases_logvar = [np.asarray(bias, dtype=float) for bias in payload["biases_logvar"]]
            self.training_steps = int(payload.get("training_steps", self.training_steps))
            self.last_metrics = {
                str(key): float(value)
                for key, value in dict(payload.get("last_metrics", {})).items()
                if isinstance(value, (int, float))
            }
            if validate_shapes:
                self._validate_parameter_shapes()
            self._assert_all_parameters_finite(operation="load")
        except KeyError as exc:
            raise TuningPersistenceError(
                "Serialized BNN payload is missing required fields.",
                context=self._context("load"),
                details={"missing_key": str(exc)},
                cause=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise wrap_exception(
                exc,
                message="Failed to restore Bayesian neural network weights from payload.",
                error_cls=TuningPersistenceError,
                context=self._context("load"),
                details={"payload_keys": list(payload.keys())},
            ) from exc

    @classmethod
    @error_boundary(
        error_cls=TuningPersistenceError,
        message="Failed to load Bayesian neural network model.",
        detail_builder=lambda exc, args, kwargs: {"filename": kwargs.get("filename", args[1] if len(args) > 1 else args[0] if args else None)},
    )
    def load(cls, filename: str) -> "BayesianNeuralNetwork":
        input_path = Path(filename)
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        raise_for_condition(
            payload.get("model_type") not in {None, "BayesianNeuralNetwork"},
            "Serialized file does not contain a BayesianNeuralNetwork payload.",
            error_cls=TuningPersistenceError,
            context=TuningErrorContext(
                component="BayesianNeuralNetwork",
                operation="load",
                strategy="variational_inference",
                model_type="bayesian_neural_network",
                output_path=str(input_path),
            ),
            details={"model_type": payload.get("model_type")},
        )

        model = cls(
            layer_sizes=payload["layer_sizes"],
            learning_rate=float(payload.get("learning_rate", 0.01)),
            prior_mu=float(payload.get("prior_mu", 0.0)),
            prior_logvar=float(payload.get("prior_logvar", 0.0)),
            random_state=payload.get("random_state"),
            logvar_clip_range=tuple(payload.get("logvar_clip_range", [-8.0, 4.0])),
            gradient_clip_norm=payload.get("gradient_clip_norm"),
            weight_init_scale=float(payload.get("weight_init_scale", 1.0)),
            hidden_activation=str(payload.get("hidden_activation", "relu")),
            likelihood_std=float(payload.get("likelihood_std", 1.0)),
            min_variance=float(payload.get("min_variance", 1e-6)),
            stability_epsilon=float(payload.get("stability_epsilon", 1e-8)),
            leaky_relu_slope=float(payload.get("leaky_relu_slope", 0.01)),
        )
        model._load_from_payload(payload, validate_shapes=True)
        return model

    def _validate_parameter_shapes(self) -> None:
        if not (
            len(self.weights_mu)
            == len(self.weights_logvar)
            == len(self.biases_mu)
            == len(self.biases_logvar)
            == self.num_layers
        ):
            raise TuningPersistenceError(
                "Parameter collections contain inconsistent layer counts.",
                context=self._context("validate_parameter_shapes"),
                details={
                    "weights_mu": len(self.weights_mu),
                    "weights_logvar": len(self.weights_logvar),
                    "biases_mu": len(self.biases_mu),
                    "biases_logvar": len(self.biases_logvar),
                    "expected_num_layers": self.num_layers,
                },
            )

        for layer_idx in range(self.num_layers):
            expected_weight_shape = (self.layer_sizes[layer_idx], self.layer_sizes[layer_idx + 1])
            expected_bias_shape = (self.layer_sizes[layer_idx + 1],)
            if self.weights_mu[layer_idx].shape != expected_weight_shape or self.weights_logvar[layer_idx].shape != expected_weight_shape:
                raise TuningPersistenceError(
                    f"Invalid weight shape at layer {layer_idx}.",
                    context=self._context("validate_parameter_shapes"),
                    details={
                        "layer_idx": layer_idx,
                        "expected_shape": list(expected_weight_shape),
                        "weights_mu_shape": list(self.weights_mu[layer_idx].shape),
                        "weights_logvar_shape": list(self.weights_logvar[layer_idx].shape),
                    },
                )
            if self.biases_mu[layer_idx].shape != expected_bias_shape or self.biases_logvar[layer_idx].shape != expected_bias_shape:
                raise TuningPersistenceError(
                    f"Invalid bias shape at layer {layer_idx}.",
                    context=self._context("validate_parameter_shapes"),
                    details={
                        "layer_idx": layer_idx,
                        "expected_shape": list(expected_bias_shape),
                        "biases_mu_shape": list(self.biases_mu[layer_idx].shape),
                        "biases_logvar_shape": list(self.biases_logvar[layer_idx].shape),
                    },
                )

    def summary(self) -> Dict[str, Any]:
        parameter_count = sum(parameter.size for parameter in self.weights_mu + self.biases_mu)
        variational_parameter_count = sum(parameter.size for parameter in self.weights_logvar + self.biases_logvar)
        return {
            "model_type": "BayesianNeuralNetwork",
            "layer_sizes": list(self.layer_sizes),
            "num_layers": self.num_layers,
            "parameter_count": int(parameter_count),
            "variational_parameter_count": int(variational_parameter_count),
            "learning_rate": self.learning_rate,
            "prior_mu": self.prior_mu,
            "prior_logvar": self.prior_logvar,
            "hidden_activation": self.hidden_activation,
            "likelihood_std": self.likelihood_std,
            "gradient_clip_norm": self.gradient_clip_norm,
            "training_steps": self.training_steps,
            "last_metrics": dict(self.last_metrics),
        }


__all__ = ["BayesianNeuralNetwork", "BNNTrainingHistory"]

if __name__ == "__main__":
    print("\n=== Running BNN Test ===\n")
    printer.status("Init", "BNN initialized", "success")
    size = [234, 43, 130]
    fan_in = 3

    bnn = BayesianNeuralNetwork(layer_sizes=size)
    print(bnn)

    scale = bnn._initial_weight_scale(fan_in=fan_in)
    printer.pretty("Movement Recovery", scale, "success" if scale else "error")

    print("\n=== Demo test Completed ===\n")