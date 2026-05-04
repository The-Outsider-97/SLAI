from __future__ import annotations

import json
import math
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.tuning_error import (TuningConfigError, wrap_exception, TuningErrorContext,
                           TuningEvaluationError, TuningOptimizationError,
                           TuningPersistenceError, TuningValidationError, error_boundary,
                           raise_for_condition, safe_serialize,)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Grid Neural Network")
printer = PrettyPrinter

Array = np.ndarray


@dataclass(slots=True)
class GNNTrainingHistory:
    """Structured training history for deterministic grid-tuned neural network runs."""

    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    validation_loss: List[float] = field(default_factory=list)
    gradient_norm: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_validation_loss: Optional[float] = None
    stopped_early: bool = False
    total_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": list(self.epochs),
            "train_loss": list(self.train_loss),
            "validation_loss": list(self.validation_loss),
            "gradient_norm": list(self.gradient_norm),
            "learning_rate": list(self.learning_rate),
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
            "total_steps": self.total_steps,
        }


class GridNeuralNetwork:
    """Production-ready fully connected neural network for deterministic training.

    The class is designed to complement the Bayesian neural network already in
    the tuning stack:
    - it keeps the same configuration-loading style
    - it uses the same structured error taxonomy
    - it remains framework-light by using NumPy only
    - it supports direct integration with grid search style evaluators

    Despite the name, this is **not** a graph neural network. The intent is a
    deterministic neural network that is convenient to tune via grid search.
    """

    MODEL_FORMAT_VERSION = "1.0.0"
    SUPPORTED_TASK_TYPES = frozenset({"regression", "binary_classification", "multiclass_classification"})
    SUPPORTED_HIDDEN_ACTIVATIONS = frozenset({"relu", "tanh", "leaky_relu"})
    SUPPORTED_OUTPUT_ACTIVATIONS = frozenset({"linear", "sigmoid", "softmax"})
    SUPPORTED_OPTIMIZERS = frozenset({"adam", "sgd"})
    CONSTRUCTOR_PARAM_KEYS = frozenset(
        {
            "learning_rate",
            "task_type",
            "hidden_activation",
            "output_activation",
            "random_state",
            "weight_init_scale",
            "gradient_clip_norm",
            "l2_lambda",
            "dropout_rate",
            "stability_epsilon",
            "optimizer",
            "beta1",
            "beta2",
            "adam_epsilon",
            "prediction_threshold",
            "leaky_relu_slope",
        }
    )
    FIT_PARAM_KEYS = frozenset(
        {
            "epochs",
            "batch_size",
            "shuffle",
            "early_stopping_patience",
            "min_delta",
            "restore_best_weights",
            "verbose",
        }
    )

    def __init__(self, layer_sizes: Sequence[int],
        task_type: Optional[str] = None,
        learning_rate: Optional[float] = None,
        hidden_activation: Optional[str] = None,
        output_activation: Optional[str] = None,
        random_state: Optional[int] = None,
        weight_init_scale: Optional[float] = None,
        gradient_clip_norm: Optional[float] = None,
        l2_lambda: Optional[float] = None,
        dropout_rate: Optional[float] = None,
        stability_epsilon: Optional[float] = None,
        optimizer: Optional[str] = None,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        adam_epsilon: Optional[float] = None,
        prediction_threshold: Optional[float] = None,
        leaky_relu_slope: Optional[float] = None,
    ) -> None:
        self.config = load_global_config() or {}
        self.gnn_config = get_config_section("gnn") or {}

        self.layer_sizes = [int(value) for value in layer_sizes]
        self.task_type = str(self._resolve_setting("task_type", task_type, "regression")).strip().lower()
        self.learning_rate = float(self._resolve_setting("learning_rate", learning_rate, 0.001))
        self.hidden_activation = str(self._resolve_setting("hidden_activation", hidden_activation, "relu")).strip().lower()
        self.output_activation = self._resolve_output_activation(output_activation)
        self.random_state = self._resolve_setting("random_state", random_state, None)
        self.weight_init_scale = self._resolve_positive_float("weight_init_scale", weight_init_scale, 1.0)
        self.gradient_clip_norm = self._resolve_optional_positive_float(
            "gradient_clip_norm",
            gradient_clip_norm,
            5.0,
            allow_none=True,
        )
        self.l2_lambda = self._resolve_non_negative_float("l2_lambda", l2_lambda, 0.0)
        self.dropout_rate = self._resolve_non_negative_float("dropout_rate", dropout_rate, 0.0)
        self.stability_epsilon = self._resolve_positive_float("stability_epsilon", stability_epsilon, 1e-8)
        self.optimizer = str(self._resolve_setting("optimizer", optimizer, "adam")).strip().lower()
        self.beta1 = self._resolve_positive_float("beta1", beta1, 0.9)
        self.beta2 = self._resolve_positive_float("beta2", beta2, 0.999)
        self.adam_epsilon = self._resolve_positive_float("adam_epsilon", adam_epsilon, 1e-8)
        self.prediction_threshold = self._resolve_positive_float(
            "prediction_threshold",
            prediction_threshold,
            0.5,
        )
        self.leaky_relu_slope = self._resolve_positive_float("leaky_relu_slope", leaky_relu_slope, 0.01)

        self._validate_init_args()

        self.num_layers = len(self.layer_sizes) - 1
        self.rng = np.random.default_rng(self.random_state)
        self.training_steps = 0
        self.last_gradient_norm: Optional[float] = None
        self.last_metrics: Dict[str, float] = {}

        self.weights: List[Array] = []
        self.biases: List[Array] = []
        self._adam_m_weights: List[Array] = []
        self._adam_v_weights: List[Array] = []
        self._adam_m_biases: List[Array] = []
        self._adam_v_biases: List[Array] = []

        self._initialize_parameters()
        self._reset_optimizer_state()
        self._validate_parameter_shapes()
        self._assert_all_parameters_finite(operation="post_initialization")

    def _resolve_setting(self, key: str, explicit: Any, default: Any) -> Any:
        if explicit is not None:
            return explicit
        return self.gnn_config.get(key, default)

    def _resolve_positive_float(self, key: str, explicit: Optional[float], default: float) -> float:
        value = self._resolve_setting(key, explicit, default)
        return float(value)

    def _resolve_optional_positive_float(self, key: str, explicit: Optional[float],
                                         default: float, *, allow_none: bool,) -> Optional[float]:
        value = self._resolve_setting(key, explicit, default)
        if value is None:
            return None if allow_none else float(default)
        return float(value)

    def _resolve_non_negative_float(self, key: str, explicit: Optional[float], default: float) -> float:
        value = self._resolve_setting(key, explicit, default)
        return float(value)

    def _resolve_output_activation(self, explicit: Optional[str]) -> str:
        configured = explicit if explicit is not None else self.gnn_config.get("output_activation", "auto")
        normalized = str(configured).strip().lower()
        if normalized == "auto":
            if self.task_type == "regression":
                return "linear"
            if self.task_type == "binary_classification":
                return "sigmoid"
            return "softmax"
        return normalized

    def _context(self, operation: str, **kwargs: Any) -> TuningErrorContext:
        return TuningErrorContext(
            component="GridNeuralNetwork",
            operation=operation,
            strategy="deterministic_training",
            model_type="grid_neural_network",
            random_state=self.random_state,
            config_path=str(self.config.get("__config_path__", "")) or None,
            parameters={
                "layer_sizes": list(self.layer_sizes),
                "task_type": self.task_type,
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
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
            self.task_type not in self.SUPPORTED_TASK_TYPES,
            "task_type must be one of: regression, binary_classification, multiclass_classification.",
            error_cls=TuningConfigError,
            context=context,
            details={"task_type": self.task_type, "supported_values": sorted(self.SUPPORTED_TASK_TYPES)},
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
            self.output_activation not in self.SUPPORTED_OUTPUT_ACTIVATIONS,
            "output_activation must be one of: linear, sigmoid, softmax.",
            error_cls=TuningConfigError,
            context=context,
            details={
                "output_activation": self.output_activation,
                "supported_values": sorted(self.SUPPORTED_OUTPUT_ACTIVATIONS),
            },
        )
        expected_output = {
            "regression": "linear",
            "binary_classification": "sigmoid",
            "multiclass_classification": "softmax",
        }[self.task_type]
        raise_for_condition(
            self.output_activation != expected_output,
            f"output_activation='{self.output_activation}' is incompatible with task_type='{self.task_type}'.",
            error_cls=TuningConfigError,
            context=context,
            details={"expected_output_activation": expected_output, "received_output_activation": self.output_activation},
        )
        raise_for_condition(
            self.optimizer not in self.SUPPORTED_OPTIMIZERS,
            "optimizer must be one of: adam, sgd.",
            error_cls=TuningConfigError,
            context=context,
            details={"optimizer": self.optimizer, "supported_values": sorted(self.SUPPORTED_OPTIMIZERS)},
        )
        for field_name, value in {
            "learning_rate": self.learning_rate,
            "weight_init_scale": self.weight_init_scale,
            "l2_lambda": self.l2_lambda,
            "dropout_rate": self.dropout_rate,
            "stability_epsilon": self.stability_epsilon,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "adam_epsilon": self.adam_epsilon,
            "prediction_threshold": self.prediction_threshold,
            "leaky_relu_slope": self.leaky_relu_slope,
        }.items():
            raise_for_condition(
                not math.isfinite(float(value)),
                f"{field_name} must be finite.",
                error_cls=TuningConfigError,
                context=context,
                details={field_name: value},
            )
        raise_for_condition(
            self.learning_rate <= 0.0,
            "learning_rate must be strictly positive.",
            error_cls=TuningConfigError,
            context=context,
            details={"learning_rate": self.learning_rate},
        )
        raise_for_condition(
            self.weight_init_scale <= 0.0,
            "weight_init_scale must be strictly positive.",
            error_cls=TuningConfigError,
            context=context,
            details={"weight_init_scale": self.weight_init_scale},
        )
        raise_for_condition(
            self.l2_lambda < 0.0,
            "l2_lambda must be non-negative.",
            error_cls=TuningConfigError,
            context=context,
            details={"l2_lambda": self.l2_lambda},
        )
        raise_for_condition(
            self.dropout_rate < 0.0 or self.dropout_rate >= 1.0,
            "dropout_rate must be in the range [0.0, 1.0).",
            error_cls=TuningConfigError,
            context=context,
            details={"dropout_rate": self.dropout_rate},
        )
        if self.gradient_clip_norm is not None:
            raise_for_condition(
                not math.isfinite(float(self.gradient_clip_norm)) or float(self.gradient_clip_norm) <= 0.0,
                "gradient_clip_norm must be a positive finite float when provided.",
                error_cls=TuningConfigError,
                context=context,
                details={"gradient_clip_norm": self.gradient_clip_norm},
            )
        raise_for_condition(
            self.stability_epsilon <= 0.0,
            "stability_epsilon must be strictly positive.",
            error_cls=TuningConfigError,
            context=context,
            details={"stability_epsilon": self.stability_epsilon},
        )
        raise_for_condition(
            self.adam_epsilon <= 0.0,
            "adam_epsilon must be strictly positive.",
            error_cls=TuningConfigError,
            context=context,
            details={"adam_epsilon": self.adam_epsilon},
        )
        raise_for_condition(
            not (0.0 < self.beta1 < 1.0) or not (0.0 < self.beta2 < 1.0),
            "beta1 and beta2 must be in the range (0.0, 1.0).",
            error_cls=TuningConfigError,
            context=context,
            details={"beta1": self.beta1, "beta2": self.beta2},
        )
        raise_for_condition(
            not (0.0 < self.prediction_threshold < 1.0),
            "prediction_threshold must be in the range (0.0, 1.0).",
            error_cls=TuningConfigError,
            context=context,
            details={"prediction_threshold": self.prediction_threshold},
        )
        if self.task_type == "binary_classification":
            raise_for_condition(
                self.layer_sizes[-1] != 1,
                "binary_classification networks must have output dimension 1.",
                error_cls=TuningConfigError,
                context=context,
                details={"layer_sizes": self.layer_sizes},
            )
        if self.task_type == "multiclass_classification":
            raise_for_condition(
                self.layer_sizes[-1] < 2,
                "multiclass_classification networks must have output dimension >= 2.",
                error_cls=TuningConfigError,
                context=context,
                details={"layer_sizes": self.layer_sizes},
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
            base = math.sqrt(2.0 / ((1.0 + float(self.leaky_relu_slope) ** 2) * fan_in))
        else:
            base = math.sqrt(1.0 / fan_in)
        return base * float(self.weight_init_scale)

    def _initialize_parameters(self) -> None:
        try:
            for layer_idx in range(self.num_layers):
                fan_in = self.layer_sizes[layer_idx]
                fan_out = self.layer_sizes[layer_idx + 1]
                scale = self._initial_weight_scale(fan_in)
                self.weights.append(self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(float))
                self.biases.append(np.zeros(fan_out, dtype=float))
        except Exception as exc:  # noqa: BLE001
            raise wrap_exception(
                exc,
                message="Failed to initialize grid neural network parameters.",
                error_cls=TuningConfigError,
                context=self._context("initialize_parameters"),
                details={"layer_sizes": self.layer_sizes},
            ) from exc

    def _reset_optimizer_state(self) -> None:
        self._adam_m_weights = [np.zeros_like(weight) for weight in self.weights]
        self._adam_v_weights = [np.zeros_like(weight) for weight in self.weights]
        self._adam_m_biases = [np.zeros_like(bias) for bias in self.biases]
        self._adam_v_biases = [np.zeros_like(bias) for bias in self.biases]

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

    def _output_transform(self, values: Array) -> Array:
        if self.output_activation == "linear":
            return values
        if self.output_activation == "sigmoid":
            clipped = np.clip(values, -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-clipped))
        shifted = values - np.max(values, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        denominator = np.sum(exponentials, axis=1, keepdims=True)
        return exponentials / np.maximum(denominator, self.stability_epsilon)

    def _shape_string(self, value: Array) -> str:
        return "x".join(str(dim) for dim in value.shape)

    def _assert_finite_array(self, value: Array, *, name: str, operation: str) -> None:
        if not np.isfinite(value).all():
            raise TuningEvaluationError(
                f"Non-finite values detected in {name} during {operation}.",
                context=self._context(operation),
                details={"name": name, "shape": self._shape_string(np.asarray(value, dtype=float))},
            )

    def _assert_all_parameters_finite(self, *, operation: str) -> None:
        for collection_name, collection in {"weights": self.weights, "biases": self.biases}.items():
            for layer_idx, parameter in enumerate(collection):
                self._assert_finite_array(
                    np.asarray(parameter, dtype=float),
                    name=f"{collection_name}[{layer_idx}]",
                    operation=operation,
                )

    def _validate_feature_array(self, x: Array, *, operation: str) -> Array:
        x_array = np.asarray(x, dtype=float)
        raise_for_condition(
            x_array.ndim != 2,
            "Input features must be a 2D array of shape (batch_size, input_dim).",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"x_shape": list(x_array.shape)},
        )
        raise_for_condition(
            x_array.shape[1] != self.layer_sizes[0],
            "Input feature dimension mismatch.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"expected_input_dim": self.layer_sizes[0], "received_input_dim": int(x_array.shape[1])},
        )
        self._assert_finite_array(x_array, name="x", operation=operation)
        return x_array

    def _prepare_targets(self, y: Array, *, operation: str) -> Tuple[Array, Optional[Array]]:
        if self.task_type == "regression":
            y_array = np.asarray(y, dtype=float)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            raise_for_condition(
                y_array.ndim != 2,
                "Regression targets must be a 1D or 2D numeric array.",
                error_cls=TuningValidationError,
                context=self._context(operation),
                details={"y_shape": list(y_array.shape)},
            )
            raise_for_condition(
                y_array.shape[1] != self.layer_sizes[-1],
                "Regression target dimension mismatch.",
                error_cls=TuningValidationError,
                context=self._context(operation),
                details={"expected_target_dim": self.layer_sizes[-1], "received_target_dim": int(y_array.shape[1])},
            )
            self._assert_finite_array(y_array, name="y", operation=operation)
            return y_array, None

        if self.task_type == "binary_classification":
            y_array = np.asarray(y, dtype=float)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            raise_for_condition(
                y_array.ndim != 2 or y_array.shape[1] != 1,
                "Binary classification targets must be a 1D array or a 2D column vector.",
                error_cls=TuningValidationError,
                context=self._context(operation),
                details={"y_shape": list(y_array.shape)},
            )
            self._assert_finite_array(y_array, name="y", operation=operation)
            raise_for_condition(
                np.any((y_array < 0.0) | (y_array > 1.0)),
                "Binary classification targets must be in the range [0, 1].",
                error_cls=TuningValidationError,
                context=self._context(operation),
                details={"target_min": float(np.min(y_array)), "target_max": float(np.max(y_array))},
            )
            class_indices = (y_array.reshape(-1) >= self.prediction_threshold).astype(int)
            return y_array, class_indices

        raw_y = np.asarray(y)
        if raw_y.ndim == 1:
            class_indices = raw_y.astype(int)
            raise_for_condition(
                class_indices.shape[0] == 0,
                "Multiclass targets cannot be empty.",
                error_cls=TuningValidationError,
                context=self._context(operation),
                details={"y_shape": list(raw_y.shape)},
            )
            raise_for_condition(
                np.any(class_indices < 0) or np.any(class_indices >= self.layer_sizes[-1]),
                "Multiclass class indices are out of range for the configured output dimension.",
                error_cls=TuningValidationError,
                context=self._context(operation),
                details={
                    "num_classes": self.layer_sizes[-1],
                    "target_min": int(np.min(class_indices)),
                    "target_max": int(np.max(class_indices)),
                },
            )
            one_hot = np.zeros((class_indices.shape[0], self.layer_sizes[-1]), dtype=float)
            one_hot[np.arange(class_indices.shape[0]), class_indices] = 1.0
            return one_hot, class_indices

        y_array = np.asarray(raw_y, dtype=float)
        if y_array.ndim == 2 and y_array.shape[1] == 1:
            return self._prepare_targets(y_array.reshape(-1), operation=operation)
        raise_for_condition(
            y_array.ndim != 2,
            "Multiclass targets must be a 1D class-index vector or a 2D one-hot/probability matrix.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"y_shape": list(y_array.shape)},
        )
        raise_for_condition(
            y_array.shape[1] != self.layer_sizes[-1],
            "Multiclass target dimension mismatch.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"expected_target_dim": self.layer_sizes[-1], "received_target_dim": int(y_array.shape[1])},
        )
        self._assert_finite_array(y_array, name="y", operation=operation)
        raise_for_condition(
            np.any(y_array < 0.0),
            "Multiclass target matrices must be non-negative.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"target_min": float(np.min(y_array))},
        )
        row_sums = np.sum(y_array, axis=1)
        raise_for_condition(
            np.any(row_sums <= 0.0),
            "Each multiclass target row must contain positive mass.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"row_sum_min": float(np.min(row_sums))},
        )
        normalized = y_array / np.maximum(row_sums[:, None], self.stability_epsilon)
        class_indices = np.argmax(normalized, axis=1)
        return normalized, class_indices

    def _validate_supervised_batch(self, x: Array, y: Array, *, operation: str) -> Tuple[Array, Array, Optional[Array]]:
        x_array = self._validate_feature_array(x, operation=operation)
        y_array, class_indices = self._prepare_targets(y, operation=operation)
        raise_for_condition(
            x_array.shape[0] != y_array.shape[0],
            "Input features and targets must have matching sample counts.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"x_rows": int(x_array.shape[0]), "y_rows": int(y_array.shape[0])},
        )
        return x_array, y_array, class_indices

    def _validate_parameter_shapes(self) -> None:
        raise_for_condition(
            len(self.weights) != self.num_layers or len(self.biases) != self.num_layers,
            "Parameter collections do not match the configured layer count.",
            error_cls=TuningPersistenceError,
            context=self._context("validate_parameter_shapes"),
            details={"expected_layers": self.num_layers, "weights_layers": len(self.weights), "bias_layers": len(self.biases)},
        )
        for layer_idx in range(self.num_layers):
            expected_weight_shape = (self.layer_sizes[layer_idx], self.layer_sizes[layer_idx + 1])
            expected_bias_shape = (self.layer_sizes[layer_idx + 1],)
            if self.weights[layer_idx].shape != expected_weight_shape:
                raise TuningPersistenceError(
                    f"Invalid weight shape at layer {layer_idx}.",
                    context=self._context("validate_parameter_shapes"),
                    details={
                        "layer_idx": layer_idx,
                        "expected_shape": list(expected_weight_shape),
                        "received_shape": list(self.weights[layer_idx].shape),
                    },
                )
            if self.biases[layer_idx].shape != expected_bias_shape:
                raise TuningPersistenceError(
                    f"Invalid bias shape at layer {layer_idx}.",
                    context=self._context("validate_parameter_shapes"),
                    details={
                        "layer_idx": layer_idx,
                        "expected_shape": list(expected_bias_shape),
                        "received_shape": list(self.biases[layer_idx].shape),
                    },
                )

    @error_boundary(
        error_cls=TuningEvaluationError,
        message="Grid neural network forward pass failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("forward") if args else None,
        detail_builder=lambda exc, args, kwargs: {"error": exc.__class__.__name__},
    )
    def forward(self, x: Array, *,
        training: bool = False,
        return_cache: bool = False,
    ) -> Array | Tuple[Array, List[Array], List[Array], List[Optional[Array]]]:
        x_array = self._validate_feature_array(x, operation="forward")
        activations: List[Array] = [x_array]
        pre_activations: List[Array] = []
        dropout_masks: List[Optional[Array]] = []

        activation = x_array
        keep_probability = 1.0 - self.dropout_rate

        for layer_idx in range(self.num_layers - 1):
            z_values = np.dot(activation, self.weights[layer_idx]) + self.biases[layer_idx]
            pre_activations.append(z_values)
            activation = self._activation(z_values)

            if training and self.dropout_rate > 0.0:
                dropout_mask = (self.rng.random(activation.shape) < keep_probability).astype(float) / keep_probability
                activation = activation * dropout_mask
                dropout_masks.append(dropout_mask)
            else:
                dropout_masks.append(None)

            activations.append(activation)

        output_linear = np.dot(activation, self.weights[-1]) + self.biases[-1]
        pre_activations.append(output_linear)
        output = self._output_transform(output_linear)
        activations.append(output)
        self._assert_finite_array(output, name="network_output", operation="forward")

        if return_cache:
            return output, activations, pre_activations, dropout_masks
        return output

    def _compute_loss(self, predictions: Array, targets: Array) -> float:
        predictions = np.asarray(predictions, dtype=float)
        targets = np.asarray(targets, dtype=float)
        self._assert_finite_array(predictions, name="predictions", operation="compute_loss")
        self._assert_finite_array(targets, name="targets", operation="compute_loss")

        if self.task_type == "regression":
            data_loss = 0.5 * float(np.mean((predictions - targets) ** 2))
        elif self.task_type == "binary_classification":
            clipped = np.clip(predictions, self.stability_epsilon, 1.0 - self.stability_epsilon)
            data_loss = float(-np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped)))
        else:
            clipped = np.clip(predictions, self.stability_epsilon, 1.0)
            data_loss = float(-np.mean(np.sum(targets * np.log(clipped), axis=1)))

        l2_penalty = 0.5 * float(self.l2_lambda) * sum(float(np.sum(weight**2)) for weight in self.weights)
        return float(data_loss + l2_penalty)

    def _compute_metrics(self, predictions: Array, targets: Array,
                         class_indices: Optional[Array],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {"loss": self._compute_loss(predictions, targets)}

        if self.task_type == "regression":
            errors = predictions - targets
            mse = float(np.mean(errors**2))
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(max(mse, 0.0)))
            target_mean = np.mean(targets, axis=0, keepdims=True)
            ss_tot = float(np.sum((targets - target_mean) ** 2))
            ss_res = float(np.sum(errors**2))
            r2 = float("nan") if ss_tot <= self.stability_epsilon else float(1.0 - (ss_res / ss_tot))
            metrics.update({"mse": mse, "rmse": rmse, "mae": mae, "r2": r2})
            return metrics

        if self.task_type == "binary_classification":
            probabilities = np.asarray(predictions, dtype=float).reshape(-1)
            truth = np.asarray(targets, dtype=float).reshape(-1)
            predicted_labels = (probabilities >= self.prediction_threshold).astype(int)
            true_labels = (truth >= self.prediction_threshold).astype(int)
            accuracy = float(np.mean(predicted_labels == true_labels))
            positive_precision_denominator = max(int(np.sum(predicted_labels == 1)), 1)
            positive_recall_denominator = max(int(np.sum(true_labels == 1)), 1)
            true_positive = int(np.sum((predicted_labels == 1) & (true_labels == 1)))
            precision = float(true_positive / positive_precision_denominator)
            recall = float(true_positive / positive_recall_denominator)
            metrics.update({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "brier_score": float(np.mean((probabilities - truth) ** 2)),
            })
            return metrics

        predicted_classes = np.argmax(predictions, axis=1)
        expected_classes = class_indices if class_indices is not None else np.argmax(targets, axis=1)
        accuracy = float(np.mean(predicted_classes == expected_classes))
        metrics.update({"accuracy": accuracy})
        return metrics

    def _backward(self, activations: Sequence[Array], pre_activations: Sequence[Array],
                  dropout_masks: Sequence[Optional[Array]], targets: Array,
                  ) -> Tuple[Dict[str, List[Array]], float]:
        batch_size = targets.shape[0]
        predictions = activations[-1]
        delta = (predictions - targets) / max(batch_size, 1)

        gradients: Dict[str, List[Array]] = {
            "weights": [np.zeros_like(weight) for weight in self.weights],
            "biases": [np.zeros_like(bias) for bias in self.biases],
        }

        for layer_idx in range(self.num_layers - 1, -1, -1):
            gradients["weights"][layer_idx] = np.dot(activations[layer_idx].T, delta) + self.l2_lambda * self.weights[layer_idx]
            gradients["biases"][layer_idx] = np.sum(delta, axis=0)

            if layer_idx > 0:
                delta = np.dot(delta, self.weights[layer_idx].T)
                delta = delta * self._activation_derivative(pre_activations[layer_idx - 1])
                if dropout_masks[layer_idx - 1] is not None:
                    delta = delta * dropout_masks[layer_idx - 1]

        gradient_norm = float(
            math.sqrt(
                sum(float(np.sum(grad**2)) for grad_list in gradients.values() for grad in grad_list)
            )
        )

        if self.gradient_clip_norm is not None and gradient_norm > float(self.gradient_clip_norm):
            scale = float(self.gradient_clip_norm) / max(gradient_norm, self.stability_epsilon)
            for key in gradients:
                gradients[key] = [gradient * scale for gradient in gradients[key]]
            gradient_norm = float(self.gradient_clip_norm)

        return gradients, gradient_norm

    def _apply_gradients(self, gradients: Mapping[str, Sequence[Array]]) -> None:
        self.training_steps += 1

        if self.optimizer == "sgd":
            for layer_idx in range(self.num_layers):
                self.weights[layer_idx] -= self.learning_rate * np.asarray(gradients["weights"][layer_idx], dtype=float)
                self.biases[layer_idx] -= self.learning_rate * np.asarray(gradients["biases"][layer_idx], dtype=float)
            return

        time_step = self.training_steps
        beta1_correction = 1.0 - self.beta1**time_step
        beta2_correction = 1.0 - self.beta2**time_step

        for layer_idx in range(self.num_layers):
            weight_grad = np.asarray(gradients["weights"][layer_idx], dtype=float)
            bias_grad = np.asarray(gradients["biases"][layer_idx], dtype=float)

            self._adam_m_weights[layer_idx] = self.beta1 * self._adam_m_weights[layer_idx] + (1.0 - self.beta1) * weight_grad
            self._adam_v_weights[layer_idx] = self.beta2 * self._adam_v_weights[layer_idx] + (1.0 - self.beta2) * (weight_grad**2)
            self._adam_m_biases[layer_idx] = self.beta1 * self._adam_m_biases[layer_idx] + (1.0 - self.beta1) * bias_grad
            self._adam_v_biases[layer_idx] = self.beta2 * self._adam_v_biases[layer_idx] + (1.0 - self.beta2) * (bias_grad**2)

            m_weight_hat = self._adam_m_weights[layer_idx] / max(beta1_correction, self.stability_epsilon)
            v_weight_hat = self._adam_v_weights[layer_idx] / max(beta2_correction, self.stability_epsilon)
            m_bias_hat = self._adam_m_biases[layer_idx] / max(beta1_correction, self.stability_epsilon)
            v_bias_hat = self._adam_v_biases[layer_idx] / max(beta2_correction, self.stability_epsilon)

            self.weights[layer_idx] -= self.learning_rate * m_weight_hat / (np.sqrt(v_weight_hat) + self.adam_epsilon)
            self.biases[layer_idx] -= self.learning_rate * m_bias_hat / (np.sqrt(v_bias_hat) + self.adam_epsilon)

    @error_boundary(
        error_cls=TuningOptimizationError,
        message="Grid neural network training step failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("train_step") if args else None,
    )
    def train_step(self, x_batch: Array, y_batch: Array) -> float:
        x_valid, y_valid, _ = self._validate_supervised_batch(x_batch, y_batch, operation="train_step")
        predictions, activations, pre_activations, dropout_masks = self.forward(x_valid, training=True, return_cache=True)
        gradients, gradient_norm = self._backward(activations, pre_activations, dropout_masks, y_valid)
        self._apply_gradients(gradients)
        self._validate_parameter_shapes()
        self._assert_all_parameters_finite(operation="train_step")

        self.last_gradient_norm = gradient_norm
        metrics = self._compute_metrics(predictions, y_valid, None)
        self.last_metrics = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.number))}
        return float(metrics["loss"])

    def _resolve_training_defaults(self, *, epochs: Optional[int], batch_size: Optional[int],
                                   shuffle: Optional[bool], early_stopping_patience: Optional[int],
                                   min_delta: Optional[float], restore_best_weights: Optional[bool],
                                   ) -> Dict[str, Any]:
        training_config = self.gnn_config.get("training", {}) if isinstance(self.gnn_config.get("training", {}), Mapping) else {}
        defaults: Dict[str, Any] = {
            "epochs": int(epochs if epochs is not None else training_config.get("epochs", 100)),
            "batch_size": int(batch_size if batch_size is not None else training_config.get("batch_size", 64)),
            "shuffle": bool(shuffle if shuffle is not None else training_config.get("shuffle", True)),
            "early_stopping_patience": (
                int(early_stopping_patience)
                if early_stopping_patience is not None
                else (
                    int(training_config["early_stopping_patience"])
                    if training_config.get("early_stopping_patience") is not None
                    else None
                )
            ),
            "min_delta": float(min_delta if min_delta is not None else training_config.get("min_delta", 1e-4)),
            "restore_best_weights": bool(
                restore_best_weights
                if restore_best_weights is not None
                else training_config.get("restore_best_weights", True)
            ),
        }

        raise_for_condition(
            defaults["epochs"] < 1,
            "epochs must be >= 1.",
            error_cls=TuningValidationError,
            context=self._context("fit"),
            details=defaults,
        )
        raise_for_condition(
            defaults["batch_size"] < 1,
            "batch_size must be >= 1.",
            error_cls=TuningValidationError,
            context=self._context("fit"),
            details=defaults,
        )
        raise_for_condition(
            not math.isfinite(defaults["min_delta"]) or defaults["min_delta"] < 0.0,
            "min_delta must be a finite non-negative float.",
            error_cls=TuningValidationError,
            context=self._context("fit"),
            details=defaults,
        )
        if defaults["early_stopping_patience"] is not None:
            raise_for_condition(
                defaults["early_stopping_patience"] < 1,
                "early_stopping_patience must be >= 1 when provided.",
                error_cls=TuningValidationError,
                context=self._context("fit"),
                details=defaults,
            )
        return defaults

    @error_boundary(
        error_cls=TuningOptimizationError,
        message="Grid neural network fit routine failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("fit") if args else None,
    )
    def fit(self, *, x_train: Array, y_train: Array,
        validation_data: Optional[Tuple[Array, Array]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        early_stopping_patience: Optional[int] = None,
        min_delta: Optional[float] = None,
        restore_best_weights: Optional[bool] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        defaults = self._resolve_training_defaults(
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            restore_best_weights=restore_best_weights,
        )
        train_x, train_y, train_class_indices = self._validate_supervised_batch(x_train, y_train, operation="fit_train")

        val_x: Optional[Array] = None
        val_y: Optional[Array] = None
        val_class_indices: Optional[Array] = None
        if validation_data is not None:
            val_x, val_y, val_class_indices = self._validate_supervised_batch(
                validation_data[0],
                validation_data[1],
                operation="fit_validation",
            )
            raise_for_condition(
                val_x.shape[1] != train_x.shape[1],
                "Validation features must use the same input dimensionality as training features.",
                error_cls=TuningValidationError,
                context=self._context("fit_validation"),
                details={"train_input_dim": int(train_x.shape[1]), "validation_input_dim": int(val_x.shape[1])},
            )

        history = GNNTrainingHistory()
        num_rows = train_x.shape[0]
        best_state: Optional[Dict[str, Any]] = None
        best_monitored_loss = math.inf
        patience_counter = 0

        for epoch in range(1, defaults["epochs"] + 1):
            indices = np.arange(num_rows)
            if defaults["shuffle"]:
                self.rng.shuffle(indices)
            shuffled_x = train_x[indices]
            shuffled_y = train_y[indices]

            epoch_losses: List[float] = []
            epoch_gradient_norms: List[float] = []

            for start in range(0, num_rows, defaults["batch_size"]):
                end = min(start + defaults["batch_size"], num_rows)
                batch_x = shuffled_x[start:end]
                batch_y = shuffled_y[start:end]
                batch_loss = self.train_step(batch_x, batch_y)
                epoch_losses.append(float(batch_loss))
                epoch_gradient_norms.append(float(self.last_gradient_norm or 0.0))
                history.total_steps += 1

            train_predictions = self.forward(train_x)
            train_metrics = self._compute_metrics(train_predictions, train_y, train_class_indices)

            history.epochs.append(epoch)
            history.train_loss.append(float(train_metrics["loss"]))
            history.gradient_norm.append(float(np.mean(epoch_gradient_norms) if epoch_gradient_norms else 0.0))
            history.learning_rate.append(float(self.learning_rate))

            monitored_loss = float(train_metrics["loss"])
            if val_x is not None and val_y is not None:
                validation_predictions = self.forward(val_x)
                validation_metrics = self._compute_metrics(validation_predictions, val_y, val_class_indices)
                monitored_loss = float(validation_metrics["loss"])
                history.validation_loss.append(monitored_loss)
            else:
                history.validation_loss.append(float("nan"))

            if monitored_loss + defaults["min_delta"] < best_monitored_loss:
                best_monitored_loss = monitored_loss
                history.best_validation_loss = monitored_loss if val_x is not None else None
                history.best_epoch = epoch
                patience_counter = 0
                best_state = self.to_serializable_dict(include_history=False)
            else:
                patience_counter += 1
                patience_limit = defaults["early_stopping_patience"]
                if patience_limit is not None and patience_counter >= int(patience_limit):
                    history.stopped_early = True
                    break

            if verbose:
                if val_x is not None:
                    logger.info(
                        "GNN epoch %s/%s | train_loss=%.6f | val_loss=%.6f | grad_norm=%.6f",
                        epoch,
                        defaults["epochs"],
                        history.train_loss[-1],
                        history.validation_loss[-1],
                        history.gradient_norm[-1],
                    )
                else:
                    logger.info(
                        "GNN epoch %s/%s | train_loss=%.6f | grad_norm=%.6f",
                        epoch,
                        defaults["epochs"],
                        history.train_loss[-1],
                        history.gradient_norm[-1],
                    )

        if best_state is not None and defaults["restore_best_weights"]:
            self._load_from_payload(best_state, validate_shapes=True)

        final_predictions = self.forward(train_x)
        self.last_metrics = {
            key: float(value)
            for key, value in self._compute_metrics(final_predictions, train_y, train_class_indices).items()
            if isinstance(value, (int, float, np.number))
        }
        return history.to_dict()

    @error_boundary(
        error_cls=TuningEvaluationError,
        message="Grid neural network evaluation failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("evaluate") if args else None,
    )
    def evaluate(self, x: Array, y: Array) -> Dict[str, float]:
        x_valid, y_valid, class_indices = self._validate_supervised_batch(x, y, operation="evaluate")
        predictions = self.forward(x_valid)
        metrics = self._compute_metrics(predictions, y_valid, class_indices)
        self.last_metrics = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.number))}
        return metrics

    @error_boundary(
        error_cls=TuningEvaluationError,
        message="Grid neural network probability prediction failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("predict_proba") if args else None,
    )
    def predict_proba(self, x: Array) -> Array:
        raise_for_condition(
            self.task_type == "regression",
            "predict_proba is only available for classification tasks.",
            error_cls=TuningValidationError,
            context=self._context("predict_proba"),
            details={"task_type": self.task_type},
        )
        probabilities = np.asarray(self.forward(x), dtype=float)
        self._assert_finite_array(probabilities, name="probabilities", operation="predict_proba")
        return probabilities

    @error_boundary(
        error_cls=TuningEvaluationError,
        message="Grid neural network prediction failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("predict") if args else None,
    )
    def predict(self, x: Array) -> Array:
        outputs = np.asarray(self.forward(x), dtype=float)
        if self.task_type == "regression":
            return outputs
        if self.task_type == "binary_classification":
            return (outputs.reshape(-1, 1) >= self.prediction_threshold).astype(int)
        return np.argmax(outputs, axis=1)

    def to_serializable_dict(self, *, include_history: bool = False, history: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_type": "GridNeuralNetwork",
            "format_version": self.MODEL_FORMAT_VERSION,
            "config_path": self.config.get("__config_path__"),
            "layer_sizes": list(self.layer_sizes),
            "task_type": self.task_type,
            "learning_rate": self.learning_rate,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "random_state": self.random_state,
            "weight_init_scale": self.weight_init_scale,
            "gradient_clip_norm": self.gradient_clip_norm,
            "l2_lambda": self.l2_lambda,
            "dropout_rate": self.dropout_rate,
            "stability_epsilon": self.stability_epsilon,
            "optimizer": self.optimizer,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "adam_epsilon": self.adam_epsilon,
            "prediction_threshold": self.prediction_threshold,
            "leaky_relu_slope": self.leaky_relu_slope,
            "training_steps": self.training_steps,
            "last_gradient_norm": self.last_gradient_norm,
            "last_metrics": safe_serialize(self.last_metrics),
            "weights": [weight.tolist() for weight in self.weights],
            "biases": [bias.tolist() for bias in self.biases],
        }
        if include_history and history is not None:
            payload["training_history"] = safe_serialize(history)
        return payload

    @error_boundary(
        error_cls=TuningPersistenceError,
        message="Grid neural network save failed.",
        context_builder=lambda exc, args, kwargs: args[0]._context("save") if args else None,
    )
    def save(self, filename: str, *, history: Optional[Mapping[str, Any]] = None) -> None:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        persistence_config = self.gnn_config.get("persistence", {}) if isinstance(self.gnn_config.get("persistence", {}), Mapping) else {}
        payload = self.to_serializable_dict(include_history=history is not None, history=history)

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=int(persistence_config.get("indent", 2)),
                sort_keys=bool(persistence_config.get("sort_keys", True)),
            )

    def _load_from_payload(self, payload: Mapping[str, Any], *, validate_shapes: bool = True) -> None:
        try:
            self.weights = [np.asarray(weight, dtype=float) for weight in payload["weights"]]
            self.biases = [np.asarray(bias, dtype=float) for bias in payload["biases"]]
            self.training_steps = int(payload.get("training_steps", 0))
            self.last_gradient_norm = (
                None if payload.get("last_gradient_norm") is None else float(payload.get("last_gradient_norm"))
            )
            self.last_metrics = {
                str(key): float(value)
                for key, value in dict(payload.get("last_metrics", {})).items()
                if isinstance(value, (int, float, np.number))
            }
            self._reset_optimizer_state()
            if validate_shapes:
                self._validate_parameter_shapes()
                self._assert_all_parameters_finite(operation="load")
        except Exception as exc:  # noqa: BLE001
            raise wrap_exception(
                exc,
                message="Failed to load grid neural network payload into model state.",
                error_cls=TuningPersistenceError,
                context=self._context("load_payload"),
                details={"payload_keys": list(payload.keys()) if isinstance(payload, Mapping) else None},
            ) from exc

    @classmethod
    @error_boundary(
        error_cls=TuningPersistenceError,
        message="Grid neural network load failed.",
        context_builder=lambda exc, args, kwargs: TuningErrorContext(
            component="GridNeuralNetwork",
            operation="load",
            strategy="deterministic_training",
            model_type="grid_neural_network",
            output_path=str(args[1]) if len(args) > 1 else str(kwargs.get("filename")),
        ),
    )
    def load(cls, filename: str) -> "GridNeuralNetwork":
        input_path = Path(filename)
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        raise_for_condition(
            payload.get("model_type") != "GridNeuralNetwork",
            "Saved payload does not describe a GridNeuralNetwork.",
            error_cls=TuningPersistenceError,
            context=TuningErrorContext(
                component="GridNeuralNetwork",
                operation="load",
                strategy="deterministic_training",
                model_type="grid_neural_network",
                output_path=str(input_path),
            ),
            details={"payload_model_type": payload.get("model_type")},
        )

        model = cls(
            layer_sizes=payload["layer_sizes"],
            task_type=str(payload.get("task_type", "regression")),
            learning_rate=float(payload.get("learning_rate", 0.001)),
            hidden_activation=str(payload.get("hidden_activation", "relu")),
            output_activation=str(payload.get("output_activation", "linear")),
            random_state=payload.get("random_state"),
            weight_init_scale=float(payload.get("weight_init_scale", 1.0)),
            gradient_clip_norm=payload.get("gradient_clip_norm"),
            l2_lambda=float(payload.get("l2_lambda", 0.0)),
            dropout_rate=float(payload.get("dropout_rate", 0.0)),
            stability_epsilon=float(payload.get("stability_epsilon", 1e-8)),
            optimizer=str(payload.get("optimizer", "adam")),
            beta1=float(payload.get("beta1", 0.9)),
            beta2=float(payload.get("beta2", 0.999)),
            adam_epsilon=float(payload.get("adam_epsilon", 1e-8)),
            prediction_threshold=float(payload.get("prediction_threshold", 0.5)),
            leaky_relu_slope=float(payload.get("leaky_relu_slope", 0.01)),
        )
        model._load_from_payload(payload, validate_shapes=True)
        return model

    def summary(self) -> Dict[str, Any]:
        parameter_count = sum(parameter.size for parameter in self.weights + self.biases)
        return {
            "model_type": "GridNeuralNetwork",
            "layer_sizes": list(self.layer_sizes),
            "num_layers": self.num_layers,
            "parameter_count": int(parameter_count),
            "task_type": self.task_type,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "dropout_rate": self.dropout_rate,
            "l2_lambda": self.l2_lambda,
            "gradient_clip_norm": self.gradient_clip_norm,
            "training_steps": self.training_steps,
            "last_metrics": dict(self.last_metrics),
        }

    @staticmethod
    def infer_output_dim_from_targets(y: Array, task_type: str) -> int:
        if task_type == "regression":
            y_array = np.asarray(y)
            return 1 if y_array.ndim == 1 else int(y_array.shape[1])
        if task_type == "binary_classification":
            return 1
        y_array = np.asarray(y)
        if y_array.ndim == 1:
            return int(np.max(y_array)) + 1
        if y_array.ndim == 2 and y_array.shape[1] == 1:
            return int(np.max(y_array)) + 1
        return int(y_array.shape[1])

    @classmethod
    def build_layer_sizes_from_params(
        cls,
        input_dim: int,
        output_dim: int,
        params: Mapping[str, Any],
    ) -> List[int]:
        hidden_spec = None
        for key in ("hidden_layer_sizes", "hidden_layers", "hidden_units"):
            if key in params:
                hidden_spec = params[key]
                break

        if hidden_spec is None:
            ordered_keys = sorted(
                [key for key in params if key.startswith("hidden_units_")],
                key=lambda value: int(value.split("_")[-1]),
            )
            if ordered_keys:
                hidden_spec = [params[key] for key in ordered_keys]

        if hidden_spec is None:
            raise TuningConfigError(
                "GridNeuralNetwork grid parameters must include hidden layer sizes.",
                context=TuningErrorContext(
                    component="GridNeuralNetwork",
                    operation="build_layer_sizes_from_params",
                    strategy="deterministic_training",
                    model_type="grid_neural_network",
                ),
                details={"params": safe_serialize(dict(params))},
            )

        if isinstance(hidden_spec, str):
            hidden_layers = [int(part.strip()) for part in hidden_spec.split(",") if part.strip()]
        elif isinstance(hidden_spec, (int, np.integer)):
            hidden_layers = [int(hidden_spec)]
        elif isinstance(hidden_spec, Sequence):
            hidden_layers = [int(value) for value in hidden_spec]
        else:
            raise TuningConfigError(
                "Unsupported hidden layer specification for GridNeuralNetwork.",
                context=TuningErrorContext(
                    component="GridNeuralNetwork",
                    operation="build_layer_sizes_from_params",
                    strategy="deterministic_training",
                    model_type="grid_neural_network",
                ),
                details={"hidden_spec": safe_serialize(hidden_spec)},
            )

        raise_for_condition(
            not hidden_layers or any(size <= 0 for size in hidden_layers),
            "Hidden layer sizes must be a non-empty collection of positive integers.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(
                component="GridNeuralNetwork",
                operation="build_layer_sizes_from_params",
                strategy="deterministic_training",
                model_type="grid_neural_network",
            ),
            details={"hidden_layers": hidden_layers},
        )

        return [int(input_dim), *hidden_layers, int(output_dim)]

    @classmethod
    def from_grid_params(
        cls,
        input_dim: int,
        output_dim: int,
        params: Mapping[str, Any],
        *,
        task_type: str = "regression",
    ) -> "GridNeuralNetwork":
        layer_sizes = cls.build_layer_sizes_from_params(input_dim=input_dim, output_dim=output_dim, params=params)
        constructor_kwargs = {
            key: value
            for key, value in dict(params).items()
            if key in cls.CONSTRUCTOR_PARAM_KEYS and key not in {"task_type"}
        }
        return cls(layer_sizes=layer_sizes, task_type=task_type, **constructor_kwargs)

    @classmethod
    def evaluate_grid_candidate(cls, *, params: Mapping[str, Any],
                                x_train: Array, y_train: Array, x_val: Array, y_val: Array,
                                task_type: str = "regression",
                                scoring_metric: Optional[str] = None,
                                fit_kwargs: Optional[Mapping[str, Any]] = None) -> float:
        train_x = np.asarray(x_train, dtype=float)
        val_x = np.asarray(x_val, dtype=float)
        output_dim = cls.infer_output_dim_from_targets(y_train, task_type)
        model = cls.from_grid_params(
            input_dim=int(train_x.shape[1]),
            output_dim=output_dim,
            params=params,
            task_type=task_type,
        )

        runtime_fit_kwargs: Dict[str, Any] = {
            key: value for key, value in dict(params).items() if key in cls.FIT_PARAM_KEYS
        }
        if fit_kwargs:
            runtime_fit_kwargs.update(dict(fit_kwargs))

        model.fit(x_train=train_x, y_train=y_train, validation_data=(val_x, y_val), **runtime_fit_kwargs)
        metrics = model.evaluate(val_x, y_val)

        selected_metric = scoring_metric
        if selected_metric is None:
            grid_defaults = model.gnn_config.get("grid_defaults", {}) if isinstance(model.gnn_config.get("grid_defaults", {}), Mapping) else {}
            selected_metric = str(
                grid_defaults.get(
                    "metric",
                    "r2" if task_type == "regression" else "accuracy",
                )
            ).strip().lower()

        raise_for_condition(
            selected_metric not in metrics,
            f"Requested scoring metric '{selected_metric}' is not available.",
            error_cls=TuningConfigError,
            context=model._context("evaluate_grid_candidate"),
            details={"available_metrics": sorted(metrics.keys()), "requested_metric": selected_metric},
        )

        raw_score = float(metrics[selected_metric])
        maximize_metrics = {"accuracy", "precision", "recall", "r2"}
        if selected_metric in maximize_metrics:
            return raw_score
        return -raw_score


__all__ = ["GridNeuralNetwork", "GNNTrainingHistory"]

if __name__ == "__main__":
    print("\n=== Running GNN Test ===\n")
    printer.status("Init", "GNN initialized", "success")
    size = [234, 43, 130]
    fan_in = 3

    bnn = GridNeuralNetwork(layer_sizes=size)
    print(bnn)

    scale = bnn._initial_weight_scale(fan_in=fan_in)
    printer.pretty("Movement Recovery", scale, "success" if scale else "error")

    print("\n=== Demo test Completed ===\n")