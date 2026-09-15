"""Auditable native neural network for SLAI Safety-Agent threat scoring.

This implementation deliberately remains NumPy/native-Python rather than
introducing a second deep-learning framework into the Safety subsystem.  It is
small enough to inspect, serializes deterministic state, and preserves the
public API consumed by ``cyber_safety.py`` and ``adaptive_security.py``.

Training semantics
------------------
* mini-batch gradients are accumulated mathematically and parameters update
  once per batch;
* sigmoid + binary cross-entropy and softmax + categorical cross-entropy use
  the exact ``prediction - target`` logit gradient (no double derivative);
* inverted-dropout scaling is propagated through the backward pass;
* batch normalization is differentiated fully and its running statistics are
  updated exactly once per batch;
* Adam's timestep advances exactly once per parameter-update step;
* ambiguous bare NumPy training arrays are rejected instead of silently pairing
  their first two rows;
* optional validation early stopping restores the best in-memory weights.

Batch normalization is kept in the historical SLAI order (activation -> BN ->
dropout) to preserve model semantics and persistence compatibility.
"""

from __future__ import annotations

import copy
import json
import math
import random
import shutil
import numpy as np

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ...base.modules.activation_engine import *  # noqa: F401,F403 - retained package compatibility
from ...base.modules.math_science import (
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    leaky_relu,
    leaky_relu_derivative,
    elu,
    elu_derivative,
    swish,
    swish_derivative,
    softmax,
    cross_entropy as cross_entropy_loss_func,
    cross_entropy_derivative,
)
from ..utils.config_loader import get_config_section
from ..utils.security_error import *
from ..utils.safety_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Cyber-Security Neural-Network")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
MODEL_SCHEMA_VERSION = "safety_neural_network.model.v3"  # persistence-compatible
TELEMETRY_SCHEMA_VERSION = "safety_neural_network.telemetry.v3"

SUPPORTED_PROBLEM_TYPES = {"regression", "binary_classification", "multiclass_classification"}
SUPPORTED_LOSSES = {"mse", "cross_entropy"}
SUPPORTED_OPTIMIZERS = {"sgd_momentum_adagrad", "adam"}
SUPPORTED_LR_SCHEDULERS = {None, "none", "step", "exponential", "cosine_annealing"}
SUPPORTED_INITIALIZERS = {
    "uniform_scaled", "he_normal", "lecun_normal", "xavier_uniform",
    "xavier_normal", "small_uniform",
}

ACTIVATION_FUNCTIONS: Dict[str, Tuple[Callable[..., float], Callable[..., float], bool]] = {
    "sigmoid": (sigmoid, sigmoid_derivative, False),
    "relu": (relu, relu_derivative, False),
    "tanh": (tanh, tanh_derivative, False),
    "leaky_relu": (leaky_relu, leaky_relu_derivative, True),
    "elu": (elu, elu_derivative, True),
    "swish": (swish, swish_derivative, False),
    "linear": (lambda x: x, lambda x: 1.0, False),
}

Array = np.ndarray
Dataset = List[Tuple[List[float], List[float]]]


@dataclass
class TrainingEpochRecord:
    epoch: int
    loss: float
    learning_rate: float
    samples: int
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    stopped: bool = False
    timestamp: str = field(default_factory=utc_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingRunSummary:
    run_id: str
    started_at: str
    finished_at: Optional[str] = None
    epochs_requested: int = 0
    epochs_completed: int = 0
    samples_seen: int = 0
    best_validation_loss: Optional[float] = None
    final_loss: Optional[float] = None
    early_stopped: bool = False
    best_weights_restored: bool = False
    model_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, *, context: str = "value") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NeuralNetworkDataError(
            f"Non-numeric neural-network {context} encountered.",
            {"value_repr": safe_repr(value)},
        ) from exc
    if not math.isfinite(result):
        raise NeuralNetworkDataError(
            f"Non-finite neural-network {context} encountered.",
            {"value_repr": safe_repr(value)},
        )
    return result


def _safe_array(values: Any, *, context: str) -> Array:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NeuralNetworkDataError(f"Non-numeric array for {context}.", {"context": context}) from exc
    if not np.all(np.isfinite(arr)):
        raise NeuralNetworkDataError(f"Non-finite numeric state in {context}.", {"shape": list(arr.shape)})
    return arr


def _clip(value: float, limit: Optional[float]) -> float:
    if limit is None:
        return value
    bound = abs(float(limit))
    return max(-bound, min(bound, value))


def _validate_probability_vector(values: Sequence[float], *, context: str) -> None:
    if not values:
        raise NeuralNetworkDataError(f"Empty probability vector for {context}.", {"context": context})
    for idx, value in enumerate(values):
        if value < 0.0 or value > 1.0:
            raise NeuralNetworkDataError(
                f"Probability value out of range for {context}.",
                {"index": idx, "value": value, "context": context},
            )


def mean_squared_error(targets: List[float], outputs: List[float]) -> float:
    if len(targets) != len(outputs):
        raise NeuralNetworkDataError(
            "Targets and outputs must have the same length for MSE.",
            {"target_length": len(targets), "output_length": len(outputs)},
        )
    return 0.5 * sum((target - output) ** 2 for target, output in zip(targets, outputs))


def mean_squared_error_derivative(targets: List[float], outputs: List[float]) -> List[float]:
    if len(targets) != len(outputs):
        raise NeuralNetworkDataError(
            "Targets and outputs must have the same length for MSE derivative.",
            {"target_length": len(targets), "output_length": len(outputs)},
        )
    return [(output - target) for target, output in zip(targets, outputs)]


class Neuron:
    """Inspectable scalar neuron with serializable optimizer state."""

    def __init__(
        self,
        num_inputs: int,
        activation_name: str = "relu",
        initialization_method: str = "he_normal",
        activation_alpha: float = 0.01,
        *,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.num_inputs = coerce_int(num_inputs, -1)
        self.activation_name = str(activation_name).lower()
        self.initialization_method = str(initialization_method).lower()
        self.activation_alpha = coerce_float(activation_alpha, 0.01)
        self.rng = rng or random.Random()
        if self.num_inputs <= 0:
            raise NeuralNetworkDataError("Neuron must have at least one input.", {"num_inputs": num_inputs})
        if self.activation_name not in ACTIVATION_FUNCTIONS:
            raise ConfigurationTamperingError(
                config_file_path="secure_config.yaml:neural_network.layers.activation",
                suspicious_change=f"Unsupported activation: {self.activation_name}",
            )
        if self.initialization_method not in SUPPORTED_INITIALIZERS:
            raise ConfigurationTamperingError(
                config_file_path="secure_config.yaml:neural_network.layers.init",
                suspicious_change=f"Unsupported initializer: {self.initialization_method}",
            )
        self.activation_fn_ptr, self.activation_fn_derivative_ptr, self.activation_needs_alpha = ACTIVATION_FUNCTIONS[self.activation_name]
        self.weights = self._initialize_weights()
        self.bias = self.rng.uniform(-0.1, 0.1)
        self.inputs = [0.0] * self.num_inputs
        self.weighted_sum = 0.0
        self.activation = 0.0
        self.delta = 0.0
        self.reset_optimizer_state()

    def reset_optimizer_state(self) -> None:
        self.velocity_weights = [0.0] * self.num_inputs
        self.velocity_bias = 0.0
        self.cache_weights = [0.0] * self.num_inputs
        self.cache_bias = 0.0
        self.m_weights = [0.0] * self.num_inputs
        self.v_weights = [0.0] * self.num_inputs
        self.m_bias = 0.0
        self.v_bias = 0.0

    def _initialize_weights(self) -> List[float]:
        fan_in = self.num_inputs
        if self.initialization_method == "uniform_scaled":
            limit = 1.0 / math.sqrt(fan_in)
            return [self.rng.uniform(-limit, limit) for _ in range(fan_in)]
        if self.initialization_method == "he_normal":
            std = math.sqrt(2.0 / fan_in)
            return [self.rng.gauss(0.0, std) for _ in range(fan_in)]
        if self.initialization_method == "lecun_normal":
            std = math.sqrt(1.0 / fan_in)
            return [self.rng.gauss(0.0, std) for _ in range(fan_in)]
        if self.initialization_method == "xavier_uniform":
            limit = math.sqrt(6.0 / (fan_in + 1))
            return [self.rng.uniform(-limit, limit) for _ in range(fan_in)]
        if self.initialization_method == "xavier_normal":
            std = math.sqrt(2.0 / (fan_in + 1))
            return [self.rng.gauss(0.0, std) for _ in range(fan_in)]
        return [self.rng.uniform(-0.1, 0.1) for _ in range(fan_in)]

    def _call_activation_fn(self, x: float) -> float:
        return self.activation_fn_ptr(x, self.activation_alpha) if self.activation_needs_alpha else self.activation_fn_ptr(x)

    def _call_activation_fn_derivative(self, x: float) -> float:
        return self.activation_fn_derivative_ptr(x, self.activation_alpha) if self.activation_needs_alpha else self.activation_fn_derivative_ptr(x)

    def _calculate_weighted_sum(self, inputs: List[float]) -> float:
        if len(inputs) != self.num_inputs:
            raise NeuralNetworkDataError("Neuron input dimension mismatch.", {"expected": self.num_inputs, "actual": len(inputs)})
        self.inputs = [_safe_float(value, context="input") for value in inputs]
        self.weighted_sum = sum(w * i for w, i in zip(self.weights, self.inputs)) + self.bias
        return _safe_float(self.weighted_sum, context="weighted_sum")

    def activate(self, inputs: List[float]) -> float:
        z = self._calculate_weighted_sum(inputs)
        self.activation = _safe_float(self._call_activation_fn(z), context="activation")
        return self.activation

    def calculate_delta(self, error_signal_from_downstream: float) -> None:
        signal = _safe_float(error_signal_from_downstream, context="error_signal")
        self.delta = signal * _safe_float(self._call_activation_fn_derivative(self.weighted_sum), context="activation_derivative")
        _safe_float(self.delta, context="delta")

    def calculate_gradients(self, weight_decay_lambda: float = 0.0, gradient_clip_value: Optional[float] = None) -> Tuple[List[float], float]:
        decay = coerce_float(weight_decay_lambda, 0.0, minimum=0.0)
        bound = None if gradient_clip_value is None else abs(coerce_float(gradient_clip_value, 0.0))
        gradients = [
            _clip(_safe_float(self.delta * inp + decay * weight, context="weight_gradient"), bound)
            for inp, weight in zip(self.inputs, self.weights)
        ]
        return gradients, _clip(_safe_float(self.delta, context="bias_gradient"), bound)

    def update_parameters(self, grad_weights: List[float], grad_bias: float, learning_rate: float, momentum_coefficient: float = 0.0, adagrad_epsilon: float = 1e-8) -> None:
        if len(grad_weights) != self.num_inputs:
            raise NeuralNetworkDataError("Gradient dimension mismatch during optimizer update.", {"expected": self.num_inputs, "actual": len(grad_weights)})
        lr = coerce_float(learning_rate, 0.001, minimum=0.0)
        momentum = coerce_float(momentum_coefficient, 0.0, minimum=0.0, maximum=0.999999)
        epsilon = coerce_float(adagrad_epsilon, 1e-8, minimum=1e-12)
        for i, grad in enumerate(grad_weights):
            self.cache_weights[i] += grad * grad
            adjusted = lr / (math.sqrt(self.cache_weights[i]) + epsilon)
            self.velocity_weights[i] = momentum * self.velocity_weights[i] - adjusted * grad
            self.weights[i] = _safe_float(self.weights[i] + self.velocity_weights[i], context="weight")
        self.cache_bias += grad_bias * grad_bias
        adjusted_bias = lr / (math.sqrt(self.cache_bias) + epsilon)
        self.velocity_bias = momentum * self.velocity_bias - adjusted_bias * grad_bias
        self.bias = _safe_float(self.bias + self.velocity_bias, context="bias")

    def to_dict(self, include_optimizer_state: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "num_inputs": self.num_inputs,
            "activation_name": self.activation_name,
            "initialization_method": self.initialization_method,
            "activation_alpha": self.activation_alpha,
            "weights": list(self.weights),
            "bias": self.bias,
        }
        if include_optimizer_state:
            data["optimizer_state"] = {
                "velocity_weights": list(self.velocity_weights),
                "velocity_bias": self.velocity_bias,
                "cache_weights": list(self.cache_weights),
                "cache_bias": self.cache_bias,
                "m_weights": list(self.m_weights),
                "v_weights": list(self.v_weights),
                "m_bias": self.m_bias,
                "v_bias": self.v_bias,
            }
        return data

    def load_state(self, data: Mapping[str, Any], include_optimizer_state: bool = True) -> None:
        weights = [_safe_float(v, context="loaded_weight") for v in data.get("weights", [])]
        if len(weights) != self.num_inputs:
            raise NeuralNetworkPersistenceError(
                "Neuron weight dimension does not match the configured architecture.",
                {"expected_inputs": self.num_inputs, "actual_weights": len(weights)},
            )
        self.weights = weights
        self.bias = _safe_float(data.get("bias", 0.0), context="loaded_bias")
        if include_optimizer_state and isinstance(data.get("optimizer_state"), Mapping):
            state = data["optimizer_state"]
            def vector(name: str) -> List[float]:
                values = [_safe_float(v, context=f"optimizer.{name}") for v in state.get(name, [0.0] * self.num_inputs)]
                if len(values) != self.num_inputs:
                    raise NeuralNetworkPersistenceError("Optimizer state dimension mismatch.", {"name": name, "expected": self.num_inputs, "actual": len(values)})
                return values
            self.velocity_weights = vector("velocity_weights")
            self.cache_weights = vector("cache_weights")
            self.m_weights = vector("m_weights")
            self.v_weights = vector("v_weights")
            self.velocity_bias = _safe_float(state.get("velocity_bias", 0.0), context="optimizer.velocity_bias")
            self.cache_bias = _safe_float(state.get("cache_bias", 0.0), context="optimizer.cache_bias")
            self.m_bias = _safe_float(state.get("m_bias", 0.0), context="optimizer.m_bias")
            self.v_bias = _safe_float(state.get("v_bias", 0.0), context="optimizer.v_bias")

    def __repr__(self) -> str:
        return f"Neuron(Act:{self.activation_name}, Weights:{len(self.weights)}, Bias:{self.bias:.3f})"


class NeuralLayer:
    """Dense layer with optional activation-space batch normalization/dropout."""

    def __init__(
        self,
        num_neurons: int,
        num_inputs_per_neuron: int,
        activation_name: str = "relu",
        initialization_method: str = "he_normal",
        dropout_rate: float = 0.0,
        activation_alpha: float = 0.01,
        use_batch_norm: bool = False,
        bn_momentum: float = 0.9,
        bn_epsilon: float = 1e-5,
        *,
        rng: Optional[random.Random] = None,
        history_max_len: int = 100,
    ) -> None:
        self.num_neurons = coerce_int(num_neurons, -1)
        self.num_inputs_per_neuron = coerce_int(num_inputs_per_neuron, -1)
        self.activation_name = str(activation_name).lower()
        self.initialization_method = str(initialization_method).lower()
        self.dropout_rate = coerce_float(dropout_rate, 0.0, minimum=0.0, maximum=0.95)
        self.activation_alpha = coerce_float(activation_alpha, 0.01)
        self.use_batch_norm = coerce_bool(use_batch_norm, False)
        self.bn_momentum = coerce_float(bn_momentum, 0.9, minimum=0.0, maximum=0.999999)
        self.bn_epsilon = coerce_float(bn_epsilon, 1e-5, minimum=1e-12)
        self.rng = rng or random.Random()
        self.history_max_len = coerce_int(history_max_len, 100, minimum=1)
        if self.num_neurons <= 0 or self.num_inputs_per_neuron <= 0:
            raise NeuralNetworkDataError("Layer dimensions must be positive.", {"num_neurons": num_neurons, "num_inputs_per_neuron": num_inputs_per_neuron})
        self.neurons = [
            Neuron(self.num_inputs_per_neuron, self.activation_name, self.initialization_method, self.activation_alpha, rng=self.rng)
            for _ in range(self.num_neurons)
        ]
        self.is_training = False
        self._dropout_mask: Optional[List[float]] = None
        self.bn_gamma = [1.0] * self.num_neurons
        self.bn_beta = [0.0] * self.num_neurons
        self.running_mean = [0.0] * self.num_neurons
        self.running_variance = [1.0] * self.num_neurons
        self._reset_bn_optimizer_state()
        self.history_activation_mean: List[float] = []
        self.history_activation_variance: List[float] = []

    def _reset_bn_optimizer_state(self) -> None:
        n = self.num_neurons
        self.bn_velocity_gamma = [0.0] * n
        self.bn_velocity_beta = [0.0] * n
        self.bn_cache_gamma = [0.0] * n
        self.bn_cache_beta = [0.0] * n
        self.bn_m_gamma = [0.0] * n
        self.bn_m_beta = [0.0] * n
        self.bn_v_gamma = [0.0] * n
        self.bn_v_beta = [0.0] * n

    def _apply_batch_norm(self, current_sample_activations: List[float], batch_activations_for_stats_T: Optional[List[List[float]]] = None) -> List[float]:
        """Inference-compatible scalar BN path.

        Training uses NeuralNetwork._forward_batch(), which updates running
        statistics once per batch and retains the exact cache required by BN
        backpropagation.  This method intentionally does not mutate running
        statistics repeatedly when batch statistics are supplied.
        """
        if not self.use_batch_norm:
            return current_sample_activations
        normalized = [0.0] * self.num_neurons
        use_batch = bool(self.is_training and batch_activations_for_stats_T and batch_activations_for_stats_T[0])
        for i in range(self.num_neurons):
            if use_batch:
                values = batch_activations_for_stats_T[i]  # type: ignore[index]
                mean = sum(values) / len(values)
                var = sum((value - mean) ** 2 for value in values) / len(values)
            else:
                mean = self.running_mean[i]
                var = max(self.running_variance[i], 0.0)
            xhat = (current_sample_activations[i] - mean) / math.sqrt(var + self.bn_epsilon)
            normalized[i] = self.bn_gamma[i] * xhat + self.bn_beta[i]
        return normalized

    def _apply_dropout(self, activations: List[float]) -> List[float]:
        if not self.is_training or self.dropout_rate <= 0.0:
            self._dropout_mask = None
            return activations
        scale = 1.0 / (1.0 - self.dropout_rate)
        mask = [0.0 if self.rng.random() < self.dropout_rate else scale for _ in activations]
        self._dropout_mask = mask
        return [value * keep for value, keep in zip(activations, mask)]

    def feed_forward_sample(self, inputs: List[float], batch_raw_activations_T: Optional[List[List[float]]] = None) -> List[float]:
        if len(inputs) != self.num_inputs_per_neuron:
            raise NeuralNetworkDataError("Layer input dimension mismatch.", {"expected": self.num_inputs_per_neuron, "actual": len(inputs)})
        raw = [neuron.activate(inputs) for neuron in self.neurons]
        if self.is_training:
            mean = sum(raw) / self.num_neurons
            variance = sum((value - mean) ** 2 for value in raw) / self.num_neurons
            self.history_activation_mean.append(mean)
            self.history_activation_variance.append(variance)
            if len(self.history_activation_mean) > self.history_max_len:
                self.history_activation_mean.pop(0)
                self.history_activation_variance.pop(0)
        normalized = self._apply_batch_norm(raw, batch_raw_activations_T)
        final = self._apply_dropout(normalized)
        for idx, neuron in enumerate(self.neurons):
            neuron.activation = final[idx]
        return final

    def get_raw_activations_for_sample(self, inputs: List[float]) -> List[float]:
        return [neuron._call_activation_fn(neuron._calculate_weighted_sum(inputs)) for neuron in self.neurons]

    def get_layer_stats(self) -> Dict[str, Any]:
        avg_mean = sum(self.history_activation_mean) / len(self.history_activation_mean) if self.history_activation_mean else 0.0
        avg_var = sum(self.history_activation_variance) / len(self.history_activation_variance) if self.history_activation_variance else 0.0
        result: Dict[str, Any] = {"avg_raw_activation_mean": avg_mean, "avg_raw_activation_variance": avg_var, "dropout_rate": self.dropout_rate, "batch_norm": self.use_batch_norm}
        if self.use_batch_norm:
            for i in range(min(3, self.num_neurons)):
                result[f"bn_running_mean_{i}"] = self.running_mean[i]
                result[f"bn_running_variance_{i}"] = self.running_variance[i]
        return result

    def to_dict(self, include_optimizer_state: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "num_neurons": self.num_neurons,
            "num_inputs_per_neuron": self.num_inputs_per_neuron,
            "activation_name": self.activation_name,
            "initialization_method": self.initialization_method,
            "dropout_rate": self.dropout_rate,
            "activation_alpha": self.activation_alpha,
            "use_batch_norm": self.use_batch_norm,
            "bn_momentum": self.bn_momentum,
            "bn_epsilon": self.bn_epsilon,
            "bn_gamma": list(self.bn_gamma),
            "bn_beta": list(self.bn_beta),
            "running_mean": list(self.running_mean),
            "running_variance": list(self.running_variance),
            "neurons": [n.to_dict(include_optimizer_state=include_optimizer_state) for n in self.neurons],
        }
        if include_optimizer_state:
            data["bn_optimizer_state"] = {
                "velocity_gamma": list(self.bn_velocity_gamma),
                "velocity_beta": list(self.bn_velocity_beta),
                "cache_gamma": list(self.bn_cache_gamma),
                "cache_beta": list(self.bn_cache_beta),
                "m_gamma": list(self.bn_m_gamma),
                "m_beta": list(self.bn_m_beta),
                "v_gamma": list(self.bn_v_gamma),
                "v_beta": list(self.bn_v_beta),
            }
        return data

    def load_state(self, data: Mapping[str, Any], include_optimizer_state: bool = True) -> None:
        neurons = data.get("neurons", [])
        if len(neurons) != len(self.neurons):
            raise NeuralNetworkPersistenceError("Layer neuron count does not match configured architecture.", {"expected": len(self.neurons), "actual": len(neurons)})
        def vector(name: str, default: List[float]) -> List[float]:
            values = [_safe_float(v, context=f"layer.{name}") for v in data.get(name, default)]
            if len(values) != self.num_neurons:
                raise NeuralNetworkPersistenceError("Batch-normalization state dimension mismatch.", {"field": name, "expected": self.num_neurons, "actual": len(values)})
            return values
        self.bn_gamma = vector("bn_gamma", self.bn_gamma)
        self.bn_beta = vector("bn_beta", self.bn_beta)
        self.running_mean = vector("running_mean", self.running_mean)
        self.running_variance = vector("running_variance", self.running_variance)
        if any(value < 0.0 for value in self.running_variance):
            raise NeuralNetworkPersistenceError("Batch-normalization running variance cannot be negative.")
        for neuron, state in zip(self.neurons, neurons):
            neuron.load_state(state, include_optimizer_state=include_optimizer_state)
        if include_optimizer_state and isinstance(data.get("bn_optimizer_state"), Mapping):
            state = data["bn_optimizer_state"]
            def opt_vector(name: str) -> List[float]:
                values = [_safe_float(v, context=f"bn_optimizer.{name}") for v in state.get(name, [0.0] * self.num_neurons)]
                if len(values) != self.num_neurons:
                    raise NeuralNetworkPersistenceError("BN optimizer-state dimension mismatch.", {"field": name, "expected": self.num_neurons, "actual": len(values)})
                return values
            self.bn_velocity_gamma = opt_vector("velocity_gamma")
            self.bn_velocity_beta = opt_vector("velocity_beta")
            self.bn_cache_gamma = opt_vector("cache_gamma")
            self.bn_cache_beta = opt_vector("cache_beta")
            self.bn_m_gamma = opt_vector("m_gamma")
            self.bn_m_beta = opt_vector("m_beta")
            self.bn_v_gamma = opt_vector("v_gamma")
            self.bn_v_beta = opt_vector("v_beta")
        elif include_optimizer_state:
            self._reset_bn_optimizer_state()

    def __repr__(self) -> str:
        bn = ", BN" if self.use_batch_norm else ""
        dropout = f", Dropout:{self.dropout_rate}" if self.dropout_rate > 0 else ""
        return f"NeuralLayer({self.num_neurons}N, Act:{self.activation_name}{dropout}{bn})"


class NeuralNetwork:
    """Native MLP used by SLAI cyber/adaptive safety components."""

    def __init__(
        self,
        num_inputs: int,
        layer_config: Optional[List[dict]] = None,
        loss_function_name: Optional[str] = None,
        optimizer_name: Optional[str] = None,
        initialization_method_default: Optional[str] = None,
        problem_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        loaded = get_config_section("neural_network") or {}
        self.nn_config = deep_merge(loaded, config) if config else dict(loaded)
        self.num_inputs = coerce_int(num_inputs, -1)
        self.layer_config = list(layer_config or self.nn_config.get("layers", []))
        self.problem_type = str(problem_type or self.nn_config.get("problem_type", "binary_classification")).lower()
        self.loss_function_name = str(loss_function_name or self.nn_config.get("loss_function_name", "cross_entropy")).lower()
        self.optimizer_name = str(optimizer_name or self.nn_config.get("optimizer_name", "adam")).lower()
        self.initialization_method_default = str(initialization_method_default or self.nn_config.get("initialization_method_default", "he_normal")).lower()
        self.binary_threshold = coerce_float(self.nn_config.get("binary_threshold", 0.5), 0.5, minimum=0.0, maximum=1.0)
        self.layers: List[NeuralLayer] = []
        self.is_training = False
        self.output_layer_activation_is_softmax = False
        self.training_history: List[TrainingEpochRecord] = []
        self.last_training_summary: Optional[TrainingRunSummary] = None
        self.model_id = generate_identifier("nn")
        self.created_at = utc_iso()
        self.updated_at = self.created_at
        seed = coerce_int(self.nn_config.get("deterministic_seed"), 1337)
        self.rng = random.Random(seed)
        if coerce_bool(self.nn_config.get("set_seed_on_init"), True):
            random.seed(seed)
            np.random.seed(seed)
        self._validate_architecture()
        self._configure_loss_function()
        self._configure_optimizer_hyperparameters(initial=True)
        self._build_layers()
        self._configure_optimizer_hyperparameters(initial=False)
        logger.info(
            "Initialized NeuralNetwork %s with %s",
            self.model_id,
            safe_log_payload("nn_init", {"model_id": self.model_id, "num_inputs": self.num_inputs, "num_layers": len(self.layers), "problem_type": self.problem_type, "optimizer": self.optimizer_name, "module_version": MODULE_VERSION}),
        )

    # ------------------------------------------------------------------
    # Architecture / state
    # ------------------------------------------------------------------

    def _validate_architecture(self) -> None:
        if self.num_inputs <= 0:
            raise NeuralNetworkDataError("Network input count must be positive.", {"num_inputs": self.num_inputs})
        if self.problem_type not in SUPPORTED_PROBLEM_TYPES:
            raise ConfigurationTamperingError("secure_config.yaml:neural_network.problem_type", f"Unsupported problem type: {self.problem_type}")
        if self.loss_function_name not in SUPPORTED_LOSSES:
            raise ConfigurationTamperingError("secure_config.yaml:neural_network.loss_function_name", f"Unsupported loss: {self.loss_function_name}")
        if self.optimizer_name not in SUPPORTED_OPTIMIZERS:
            raise ConfigurationTamperingError("secure_config.yaml:neural_network.optimizer_name", f"Unsupported optimizer: {self.optimizer_name}")
        if not self.layer_config:
            raise NeuralNetworkDataError("Layer configuration cannot be empty.", {"config_section": "neural_network.layers"})
        for idx, layer in enumerate(self.layer_config):
            neurons = coerce_int(layer.get("neurons"), -1)
            activation = str(layer.get("activation", "relu")).lower()
            dropout = coerce_float(layer.get("dropout", 0.0), 0.0)
            if neurons <= 0:
                raise ConfigurationTamperingError("secure_config.yaml:neural_network.layers", f"Layer {idx} has invalid neuron count: {neurons}")
            if activation not in set(ACTIVATION_FUNCTIONS).union({"softmax"}):
                raise ConfigurationTamperingError("secure_config.yaml:neural_network.layers.activation", f"Layer {idx} has unsupported activation: {activation}")
            if not 0.0 <= dropout < 1.0:
                raise ConfigurationTamperingError("secure_config.yaml:neural_network.layers.dropout", f"Layer {idx} dropout must be in [0,1): {dropout}")
        output = coerce_int(self.layer_config[-1].get("neurons"), 1)
        if self.problem_type == "binary_classification" and output != 1:
            raise NeuralNetworkDataError("Binary classification requires exactly one output neuron.", {"output_neurons": output})
        if self.problem_type == "multiclass_classification" and output < 2:
            raise NeuralNetworkDataError("Multiclass classification requires at least two output neurons.", {"output_neurons": output})

    def _build_layers(self) -> None:
        current_inputs = self.num_inputs
        self.layers = []
        self.output_layer_activation_is_softmax = False
        for index, config in enumerate(self.layer_config):
            output_layer = index == len(self.layer_config) - 1
            default_activation = "relu"
            if output_layer:
                default_activation = "sigmoid" if self.problem_type == "binary_classification" else "linear"
                if self.problem_type == "multiclass_classification":
                    default_activation = "softmax"
            requested = str(config.get("activation", default_activation)).lower()
            actual = "linear" if output_layer and requested == "softmax" else requested
            if output_layer and requested == "softmax":
                self.output_layer_activation_is_softmax = True
            layer = NeuralLayer(
                num_neurons=coerce_int(config["neurons"], -1),
                num_inputs_per_neuron=current_inputs,
                activation_name=actual,
                initialization_method=str(config.get("init", self.initialization_method_default)).lower(),
                dropout_rate=0.0 if output_layer else coerce_float(config.get("dropout", self.nn_config.get("default_dropout_rate", 0.0)), 0.0),
                activation_alpha=coerce_float(config.get("alpha", self.nn_config.get("default_activation_alpha", 0.01)), 0.01),
                use_batch_norm=False if output_layer else coerce_bool(config.get("batch_norm", self.nn_config.get("default_use_batch_norm", False)), False),
                bn_momentum=coerce_float(config.get("bn_momentum", self.nn_config.get("default_bn_momentum", 0.9)), 0.9),
                bn_epsilon=coerce_float(config.get("bn_epsilon", self.nn_config.get("default_bn_epsilon", 1e-5)), 1e-5),
                rng=self.rng,
                history_max_len=coerce_int(get_nested(self.nn_config, "telemetry.history_limit", 500), 500, minimum=10),
            )
            self.layers.append(layer)
            current_inputs = layer.num_neurons
            logger.info("Configured layer %s: %s", index, layer)

    def _configure_loss_function(self) -> None:
        if self.loss_function_name == "mse":
            self.loss_fn = mean_squared_error
            self.loss_fn_derivative = mean_squared_error_derivative
        elif self.loss_function_name == "cross_entropy":
            self.loss_fn = cross_entropy_loss_func
            self.loss_fn_derivative = cross_entropy_derivative
        else:
            raise ConfigurationTamperingError("secure_config.yaml:neural_network.loss_function_name", f"Unsupported loss: {self.loss_function_name}")

    def _configure_optimizer_hyperparameters(self, *, initial: bool = False) -> None:
        self.adam_beta1 = coerce_float(self.nn_config.get("adam_beta1", 0.9), 0.9, minimum=0.0, maximum=0.999999)
        self.adam_beta2 = coerce_float(self.nn_config.get("adam_beta2", 0.999), 0.999, minimum=0.0, maximum=0.999999)
        self.adam_epsilon = coerce_float(self.nn_config.get("adam_epsilon", 1e-8), 1e-8, minimum=1e-12)
        if initial or not hasattr(self, "adam_global_timestep"):
            self.adam_global_timestep = 0

    def _set_training_mode(self, mode: bool) -> None:
        self.is_training = bool(mode)
        for layer in self.layers:
            layer.is_training = bool(mode)

    def _coerce_sample(self, sample: Sequence[Any], *, expected_length: int, context: str) -> List[float]:
        if len(sample) != expected_length:
            raise NeuralNetworkDataError(f"{context} dimension mismatch.", {"expected": expected_length, "actual": len(sample)})
        return [_safe_float(value, context=f"{context}[{idx}]") for idx, value in enumerate(sample)]

    def _expected_target_length(self) -> int:
        return self.layers[-1].num_neurons

    def _normalize_dataset(
        self,
        data: Union[List[Tuple[List[float], List[float]]], Tuple[np.ndarray, np.ndarray], np.ndarray],
        *,
        context: str,
    ) -> Dataset:
        if isinstance(data, np.ndarray):
            raise NeuralNetworkDataError(
                "A bare NumPy array is ambiguous training data. Pass (X, y) or an iterable of (features, targets) pairs.",
                {"context": context, "shape": list(data.shape)},
            )
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray):
            x, y = data
            if len(x) != len(y):
                raise NeuralNetworkDataError("X and y sample counts differ.", {"x_samples": len(x), "y_samples": len(y), "context": context})
            raw_pairs: Iterable[Any] = zip(x, y)
        else:
            raw_pairs = data  # type: ignore[assignment]
        raw_list = list(raw_pairs)
        maximum = coerce_int(self.nn_config.get("max_training_samples", 250000), 250000, minimum=1)
        if len(raw_list) > maximum:
            raise ResourceExhaustionError("training_samples", len(raw_list), maximum, source_identifier="neural_network.train")
        normalized: Dataset = []
        target_len = self._expected_target_length()
        for idx, pair in enumerate(raw_list):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise NeuralNetworkDataError("Dataset entries must be (features, targets) pairs.", {"index": idx, "context": context})
            features, targets = pair
            feature_values = list(features.tolist()) if isinstance(features, np.ndarray) else list(features)
            target_values = targets.tolist() if isinstance(targets, np.ndarray) else targets
            if not isinstance(target_values, (tuple, list, np.ndarray)):
                target_values = [target_values]
            clean_x = self._coerce_sample(feature_values, expected_length=self.num_inputs, context=f"{context}.features")
            clean_y = self._coerce_sample(list(target_values), expected_length=target_len, context=f"{context}.targets")
            if self.problem_type in {"binary_classification", "multiclass_classification"}:
                _validate_probability_vector(clean_y, context=f"{context}.targets")
                if self.problem_type == "multiclass_classification" and self.loss_function_name == "cross_entropy":
                    total = sum(clean_y)
                    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
                        raise NeuralNetworkDataError("Multiclass cross-entropy targets must sum to 1.", {"target_sum": total, "index": idx})
            normalized.append((clean_x, clean_y))
        return normalized

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def feed_forward_sample(self, inputs: List[float], batch_raw_activations_T_by_layer: Optional[List[List[List[float]]]] = None) -> List[float]:
        clean = self._coerce_sample(inputs, expected_length=self.num_inputs, context="inference.features")
        current = clean
        for idx, layer in enumerate(self.layers):
            stats = batch_raw_activations_T_by_layer[idx] if batch_raw_activations_T_by_layer and idx < len(batch_raw_activations_T_by_layer) else None
            current = layer.feed_forward_sample(current, stats)
            if idx == len(self.layers) - 1 and self.output_layer_activation_is_softmax:
                current = list(softmax([_safe_float(value, context="softmax_logit") for value in current]))
                for neuron_idx, neuron in enumerate(layer.neurons):
                    neuron.activation = current[neuron_idx]
        if coerce_bool(self.nn_config.get("strict_numerics", True), True):
            for value in current:
                _safe_float(value, context="network_output")
        return current

    def _activation_array(self, layer: NeuralLayer, z: Array) -> Array:
        fn = layer.neurons[0]._call_activation_fn
        # frompyfunc avoids assumptions about math_science functions accepting ndarrays
        vectorized = np.frompyfunc(lambda value: float(fn(float(value))), 1, 1)
        return _safe_array(np.asarray(vectorized(z), dtype=np.float64), context="batch_activation")

    def _activation_derivative_array(self, layer: NeuralLayer, z: Array) -> Array:
        fn = layer.neurons[0]._call_activation_fn_derivative
        vectorized = np.frompyfunc(lambda value: float(fn(float(value))), 1, 1)
        return _safe_array(np.asarray(vectorized(z), dtype=np.float64), context="batch_activation_derivative")

    @staticmethod
    def _softmax_array(z: Array) -> Array:
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(shifted)
        denominator = np.sum(exp, axis=1, keepdims=True)
        if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
            raise NeuralNetworkDataError("Invalid softmax normalization denominator.")
        return exp / denominator

    def _forward_batch(self, x: Array, *, training: bool) -> Tuple[Array, List[Dict[str, Any]]]:
        current = _safe_array(x, context="batch.features")
        caches: List[Dict[str, Any]] = []
        for layer_idx, layer in enumerate(self.layers):
            w = np.asarray([neuron.weights for neuron in layer.neurons], dtype=np.float64)
            b = np.asarray([neuron.bias for neuron in layer.neurons], dtype=np.float64)
            z = current @ w.T + b
            output_layer = layer_idx == len(self.layers) - 1
            if output_layer and self.output_layer_activation_is_softmax:
                raw = self._softmax_array(z)
                activation_before_bn = raw
            else:
                activation_before_bn = self._activation_array(layer, z)

            bn_cache: Optional[Dict[str, Array]] = None
            after_bn = activation_before_bn
            if layer.use_batch_norm:
                gamma = np.asarray(layer.bn_gamma, dtype=np.float64)
                beta = np.asarray(layer.bn_beta, dtype=np.float64)
                if training:
                    mean = np.mean(activation_before_bn, axis=0)
                    variance = np.var(activation_before_bn, axis=0)
                    inv_std = 1.0 / np.sqrt(variance + layer.bn_epsilon)
                    xhat = (activation_before_bn - mean) * inv_std
                    after_bn = gamma * xhat + beta
                    layer.running_mean = (layer.bn_momentum * np.asarray(layer.running_mean) + (1.0 - layer.bn_momentum) * mean).tolist()
                    layer.running_variance = (layer.bn_momentum * np.asarray(layer.running_variance) + (1.0 - layer.bn_momentum) * variance).tolist()
                    bn_cache = {"xhat": xhat, "inv_std": inv_std, "gamma": gamma}
                else:
                    mean = np.asarray(layer.running_mean, dtype=np.float64)
                    variance = np.maximum(np.asarray(layer.running_variance, dtype=np.float64), 0.0)
                    after_bn = gamma * ((activation_before_bn - mean) / np.sqrt(variance + layer.bn_epsilon)) + beta

            dropout_mask: Optional[Array] = None
            output = after_bn
            if training and layer.dropout_rate > 0.0 and not output_layer:
                keep_probability = 1.0 - layer.dropout_rate
                random_values = np.asarray([[self.rng.random() for _ in range(layer.num_neurons)] for _ in range(len(current))], dtype=np.float64)
                dropout_mask = (random_values >= layer.dropout_rate).astype(np.float64) / keep_probability
                output = after_bn * dropout_mask
                layer._dropout_mask = dropout_mask[-1].tolist() if len(dropout_mask) else None
            else:
                layer._dropout_mask = None

            if training:
                layer.history_activation_mean.append(float(np.mean(activation_before_bn)))
                layer.history_activation_variance.append(float(np.var(activation_before_bn)))
                if len(layer.history_activation_mean) > layer.history_max_len:
                    layer.history_activation_mean = layer.history_activation_mean[-layer.history_max_len:]
                    layer.history_activation_variance = layer.history_activation_variance[-layer.history_max_len:]

            caches.append({
                "input": current,
                "z": z,
                "activation": activation_before_bn,
                "bn": bn_cache,
                "dropout_mask": dropout_mask,
                "weights": w.copy(),
            })
            current = _safe_array(output, context=f"layer_{layer_idx}.output")
        return current, caches

    # ------------------------------------------------------------------
    # Loss / backward / optimization
    # ------------------------------------------------------------------

    def _calculate_loss_and_output_deltas(self, target_outputs: List[float], predicted_outputs: List[float]) -> Tuple[float, List[float]]:
        if len(target_outputs) != len(predicted_outputs):
            raise NeuralNetworkDataError("Target and prediction dimension mismatch.", {"target_length": len(target_outputs), "prediction_length": len(predicted_outputs)})
        if self.loss_function_name == "cross_entropy":
            loss = _safe_float(self.loss_fn(target_outputs, predicted_outputs), context="loss")
            # For softmax+CE and sigmoid+binary-CE, dL/dz is exactly p-y.
            if self.output_layer_activation_is_softmax or self.layers[-1].activation_name == "sigmoid":
                signals = [_safe_float(p - y, context="cross_entropy_logit_delta") for p, y in zip(predicted_outputs, target_outputs)]
                for neuron, delta in zip(self.layers[-1].neurons, signals):
                    neuron.delta = delta
                return loss, signals
            # Other activation/loss combinations return dL/da and are multiplied
            # by the output activation derivative in _backpropagate compatibility.
            return loss, list(self.loss_fn_derivative(target_outputs, predicted_outputs))
        if self.loss_function_name == "mse":
            return _safe_float(self.loss_fn(target_outputs, predicted_outputs), context="loss"), list(self.loss_fn_derivative(target_outputs, predicted_outputs))
        raise ConfigurationTamperingError("secure_config.yaml:neural_network.loss_function_name", f"Unsupported loss: {self.loss_function_name}")

    def _batch_loss(self, y: Array, p: Array) -> float:
        if self.loss_function_name == "mse":
            return float(0.5 * np.mean(np.sum((p - y) ** 2, axis=1)))
        eps = coerce_float(self.nn_config.get("loss_epsilon", 1e-12), 1e-12, minimum=1e-15)
        clipped = np.clip(p, eps, 1.0 - eps)
        if self.output_layer_activation_is_softmax:
            loss = -np.mean(np.sum(y * np.log(clipped), axis=1))
        elif p.shape[1] == 1:
            loss = -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
        else:
            loss = -np.mean(np.sum(y * np.log(clipped), axis=1))
        return _safe_float(loss, context="batch_loss")

    def _output_delta(self, y: Array, p: Array, cache: Mapping[str, Any]) -> Array:
        if self.loss_function_name == "cross_entropy":
            if self.output_layer_activation_is_softmax or self.layers[-1].activation_name == "sigmoid":
                return p - y  # exact logit gradient: fixes previous double derivative
            d_a = -(y / np.clip(p, 1e-12, None))
            return d_a * self._activation_derivative_array(self.layers[-1], cache["z"])
        d_a = p - y
        if self.output_layer_activation_is_softmax:
            # J_softmax @ dA, vectorized row-wise.
            return p * (d_a - np.sum(d_a * p, axis=1, keepdims=True))
        return d_a * self._activation_derivative_array(self.layers[-1], cache["z"])

    def _backward_batch(self, y: Array, predictions: Array, caches: List[Dict[str, Any]], *, weight_decay_lambda: float, gradient_clip_value: Optional[float]) -> List[Dict[str, Any]]:
        batch_n = max(1, y.shape[0])
        gradients: List[Dict[str, Any]] = [{} for _ in self.layers]
        d_z = self._output_delta(y, predictions, caches[-1])
        clip = None if gradient_clip_value is None else abs(float(gradient_clip_value))

        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            cache = caches[layer_idx]
            input_values: Array = cache["input"]
            old_weights: Array = cache["weights"]
            grad_w = (d_z.T @ input_values) / batch_n
            if weight_decay_lambda:
                grad_w = grad_w + float(weight_decay_lambda) * old_weights
            grad_b = np.mean(d_z, axis=0)
            if clip is not None:
                grad_w = np.clip(grad_w, -clip, clip)
                grad_b = np.clip(grad_b, -clip, clip)
            gradients[layer_idx]["weights"] = grad_w
            gradients[layer_idx]["bias"] = grad_b

            if layer_idx == 0:
                continue
            d_previous_output = d_z @ old_weights
            previous_layer = self.layers[layer_idx - 1]
            previous_cache = caches[layer_idx - 1]

            # Inverted dropout: kept units must carry the same 1/(1-p) scale.
            previous_mask = previous_cache.get("dropout_mask")
            if previous_mask is not None:
                d_previous_output = d_previous_output * previous_mask

            # Full BN backward for historical activation -> BN ordering.
            bn_cache = previous_cache.get("bn")
            if bn_cache is not None:
                xhat: Array = bn_cache["xhat"]
                inv_std: Array = bn_cache["inv_std"]
                gamma: Array = bn_cache["gamma"]
                d_gamma = np.mean(d_previous_output * xhat, axis=0)
                d_beta = np.mean(d_previous_output, axis=0)
                gradients[layer_idx - 1]["bn_gamma"] = d_gamma
                gradients[layer_idx - 1]["bn_beta"] = d_beta
                mean_dy = np.mean(d_previous_output, axis=0)
                mean_dy_xhat = np.mean(d_previous_output * xhat, axis=0)
                d_previous_output = gamma * inv_std * (d_previous_output - mean_dy - xhat * mean_dy_xhat)

            activation_derivative = self._activation_derivative_array(previous_layer, previous_cache["z"])
            d_z = d_previous_output * activation_derivative

        for grad in gradients:
            for value in grad.values():
                if not np.all(np.isfinite(value)):
                    raise NeuralNetworkDataError("Non-finite gradient detected during batch backpropagation.")
        return gradients

    def _optimizer_bounds(self, learning_rate: float) -> float:
        return coerce_float(
            learning_rate,
            0.001,
            minimum=coerce_float(self.nn_config.get("learning_rate_floor", 1e-8), 1e-8),
            maximum=coerce_float(self.nn_config.get("learning_rate_ceiling", 1.0), 1.0),
        )

    def _apply_optimizer_step(self, neuron: Neuron, grad_weights: List[float], grad_bias: float, learning_rate: float, **optimizer_kwargs: Any) -> None:
        lr = self._optimizer_bounds(learning_rate)
        if self.optimizer_name == "sgd_momentum_adagrad":
            neuron.update_parameters(grad_weights, grad_bias, lr, optimizer_kwargs.get("momentum_coefficient", self.nn_config.get("momentum_coefficient", 0.9)), optimizer_kwargs.get("adagrad_epsilon", self.nn_config.get("adagrad_epsilon", 1e-8)))
            return
        if self.optimizer_name == "adam":
            beta1 = coerce_float(optimizer_kwargs.get("adam_beta1", self.adam_beta1), self.adam_beta1, minimum=0.0, maximum=0.999999)
            beta2 = coerce_float(optimizer_kwargs.get("adam_beta2", self.adam_beta2), self.adam_beta2, minimum=0.0, maximum=0.999999)
            epsilon = coerce_float(optimizer_kwargs.get("adam_epsilon", self.adam_epsilon), self.adam_epsilon, minimum=1e-12)
            timestep = max(1, self.adam_global_timestep)
            for i, grad in enumerate(grad_weights):
                neuron.m_weights[i] = beta1 * neuron.m_weights[i] + (1.0 - beta1) * grad
                neuron.v_weights[i] = beta2 * neuron.v_weights[i] + (1.0 - beta2) * grad * grad
                m_hat = neuron.m_weights[i] / (1.0 - beta1 ** timestep)
                v_hat = neuron.v_weights[i] / (1.0 - beta2 ** timestep)
                neuron.weights[i] = _safe_float(neuron.weights[i] - lr * m_hat / (math.sqrt(v_hat) + epsilon), context="adam_weight")
            neuron.m_bias = beta1 * neuron.m_bias + (1.0 - beta1) * grad_bias
            neuron.v_bias = beta2 * neuron.v_bias + (1.0 - beta2) * grad_bias * grad_bias
            m_hat_b = neuron.m_bias / (1.0 - beta1 ** timestep)
            v_hat_b = neuron.v_bias / (1.0 - beta2 ** timestep)
            neuron.bias = _safe_float(neuron.bias - lr * m_hat_b / (math.sqrt(v_hat_b) + epsilon), context="adam_bias")
            return
        raise ConfigurationTamperingError("secure_config.yaml:neural_network.optimizer_name", f"Unsupported optimizer: {self.optimizer_name}")

    def _update_bn_parameters(self, layer: NeuralLayer, grad_gamma: Array, grad_beta: Array, learning_rate: float, **kwargs: Any) -> None:
        if not layer.use_batch_norm:
            return
        lr = self._optimizer_bounds(learning_rate)
        gamma = np.asarray(layer.bn_gamma, dtype=np.float64)
        beta = np.asarray(layer.bn_beta, dtype=np.float64)
        if self.optimizer_name == "adam":
            b1 = coerce_float(kwargs.get("adam_beta1", self.adam_beta1), self.adam_beta1, minimum=0.0, maximum=0.999999)
            b2 = coerce_float(kwargs.get("adam_beta2", self.adam_beta2), self.adam_beta2, minimum=0.0, maximum=0.999999)
            eps = coerce_float(kwargs.get("adam_epsilon", self.adam_epsilon), self.adam_epsilon, minimum=1e-12)
            t = max(1, self.adam_global_timestep)
            m_g, m_b = np.asarray(layer.bn_m_gamma), np.asarray(layer.bn_m_beta)
            v_g, v_b = np.asarray(layer.bn_v_gamma), np.asarray(layer.bn_v_beta)
            m_g = b1 * m_g + (1.0 - b1) * grad_gamma
            m_b = b1 * m_b + (1.0 - b1) * grad_beta
            v_g = b2 * v_g + (1.0 - b2) * (grad_gamma ** 2)
            v_b = b2 * v_b + (1.0 - b2) * (grad_beta ** 2)
            gamma -= lr * (m_g / (1.0 - b1 ** t)) / (np.sqrt(v_g / (1.0 - b2 ** t)) + eps)
            beta -= lr * (m_b / (1.0 - b1 ** t)) / (np.sqrt(v_b / (1.0 - b2 ** t)) + eps)
            layer.bn_m_gamma, layer.bn_m_beta = m_g.tolist(), m_b.tolist()
            layer.bn_v_gamma, layer.bn_v_beta = v_g.tolist(), v_b.tolist()
        else:
            momentum = coerce_float(kwargs.get("momentum_coefficient", self.nn_config.get("momentum_coefficient", 0.9)), 0.9, minimum=0.0, maximum=0.999999)
            eps = coerce_float(kwargs.get("adagrad_epsilon", self.nn_config.get("adagrad_epsilon", 1e-8)), 1e-8, minimum=1e-12)
            vg, vb = np.asarray(layer.bn_velocity_gamma), np.asarray(layer.bn_velocity_beta)
            cg, cb = np.asarray(layer.bn_cache_gamma), np.asarray(layer.bn_cache_beta)
            cg += grad_gamma ** 2
            cb += grad_beta ** 2
            vg = momentum * vg - (lr / (np.sqrt(cg) + eps)) * grad_gamma
            vb = momentum * vb - (lr / (np.sqrt(cb) + eps)) * grad_beta
            gamma += vg
            beta += vb
            layer.bn_velocity_gamma, layer.bn_velocity_beta = vg.tolist(), vb.tolist()
            layer.bn_cache_gamma, layer.bn_cache_beta = cg.tolist(), cb.tolist()
        layer.bn_gamma = _safe_array(gamma, context="bn_gamma").tolist()
        layer.bn_beta = _safe_array(beta, context="bn_beta").tolist()

    def _apply_batch_gradients(self, gradients: List[Dict[str, Any]], learning_rate: float, **optimizer_kwargs: Any) -> None:
        # Exactly one Adam timestep per batch/parameter-update step.
        if self.optimizer_name == "adam":
            self.adam_global_timestep += 1
        for layer, grad in zip(self.layers, gradients):
            grad_w: Array = grad["weights"]
            grad_b: Array = grad["bias"]
            for idx, neuron in enumerate(layer.neurons):
                self._apply_optimizer_step(neuron, grad_w[idx].tolist(), float(grad_b[idx]), learning_rate, **optimizer_kwargs)
            if "bn_gamma" in grad:
                self._update_bn_parameters(layer, grad["bn_gamma"], grad["bn_beta"], learning_rate, **optimizer_kwargs)

    def _backpropagate(self, inputs_sample: List[float], output_layer_error_signals: List[float], learning_rate: float, **optimizer_kwargs: Any) -> None:
        """Compatibility single-sample backpropagation.

        Production training uses the batch implementation above.  This method is
        retained for callers/tests that exercise the historical private API and
        fixes sigmoid+CE and inverted-dropout semantics.
        """
        output_layer = self.layers[-1]
        direct_logit_delta = self.loss_function_name == "cross_entropy" and (self.output_layer_activation_is_softmax or output_layer.activation_name == "sigmoid")
        input_to_output = [n.activation for n in self.layers[-2].neurons] if len(self.layers) > 1 else inputs_sample
        for idx, neuron in enumerate(output_layer.neurons):
            if direct_logit_delta:
                neuron.delta = _safe_float(output_layer_error_signals[idx], context="output_delta")
            else:
                neuron.calculate_delta(output_layer_error_signals[idx])
            neuron.inputs = input_to_output
        # Propagate all deltas before updating any weight so downstream weights
        # are from the same forward state.
        for layer_idx in reversed(range(len(self.layers) - 1)):
            layer = self.layers[layer_idx]
            downstream = self.layers[layer_idx + 1]
            for neuron_idx, neuron in enumerate(layer.neurons):
                signal = sum(d.delta * d.weights[neuron_idx] for d in downstream.neurons)
                mask_scale = layer._dropout_mask[neuron_idx] if layer._dropout_mask else 1.0
                if mask_scale == 0.0:
                    neuron.delta = 0.0
                else:
                    neuron.calculate_delta(signal * mask_scale)
                neuron.inputs = [n.activation for n in self.layers[layer_idx - 1].neurons] if layer_idx > 0 else inputs_sample
        if self.optimizer_name == "adam":
            self.adam_global_timestep += 1
        for layer in self.layers:
            for neuron in layer.neurons:
                grad_w, grad_b = neuron.calculate_gradients(optimizer_kwargs.get("weight_decay_lambda", 0.0), optimizer_kwargs.get("gradient_clip_value"))
                self._apply_optimizer_step(neuron, grad_w, grad_b, learning_rate, **optimizer_kwargs)

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def _apply_lr_schedule(self, initial_lr: float, current_lr: float, epoch: int, total_epochs: int, scheduler_name: Optional[str], decay_rate: Optional[float], decay_steps: Optional[int]) -> float:
        scheduler = None if scheduler_name is None else str(scheduler_name).lower()
        if scheduler in {None, "none"}:
            return current_lr
        if scheduler not in SUPPORTED_LR_SCHEDULERS:
            raise ConfigurationTamperingError("secure_config.yaml:neural_network.lr_scheduler", f"Unsupported scheduler: {scheduler}")
        if decay_rate is None:
            return current_lr
        if scheduler == "step":
            return current_lr * decay_rate if decay_steps and decay_steps > 0 and (epoch + 1) % decay_steps == 0 else current_lr
        if scheduler == "exponential":
            return initial_lr * (decay_rate ** (epoch / max(1, decay_steps or 1)))
        eta_min = decay_rate
        t_max = decay_steps if decay_steps and decay_steps > 0 else total_epochs
        t_cur = epoch % max(1, t_max)
        return eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + math.cos(math.pi * t_cur / max(1, t_max)))

    def train(
        self,
        training_data: Union[List[Tuple[List[float], List[float]]], Tuple[np.ndarray, np.ndarray], np.ndarray],
        epochs: int,
        initial_learning_rate: float,
        batch_size: Optional[int] = 1,
        momentum_coefficient: Optional[float] = None,
        weight_decay_lambda: Optional[float] = None,
        gradient_clip_value: Optional[float] = None,
        adagrad_epsilon: Optional[float] = None,
        adam_beta1: Optional[float] = None,
        adam_beta2: Optional[float] = None,
        adam_epsilon_opt: Optional[float] = None,
        lr_scheduler_name: Optional[str] = None,
        lr_decay_rate: Optional[float] = None,
        lr_decay_steps: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0001,
        validation_data: Optional[Union[List[Tuple[List[float], List[float]]], Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
        verbose: bool = True,
        print_every_n_epochs: Optional[int] = None,
        save_best_model_path: Optional[str] = None,
    ) -> TrainingRunSummary:
        epochs = coerce_int(epochs, 0, minimum=1, maximum=coerce_int(self.nn_config.get("max_epochs", 10000), 10000, minimum=1))
        learning_rate = self._optimizer_bounds(initial_learning_rate)
        effective_batch = coerce_int(batch_size, 1, minimum=1, maximum=coerce_int(self.nn_config.get("max_batch_size", 4096), 4096, minimum=1))
        data = self._normalize_dataset(training_data, context="training")
        if not data:
            raise NeuralNetworkDataError("Training data is empty.", {"operation": "train"})
        validation = self._normalize_dataset(validation_data, context="validation") if validation_data is not None else None
        if save_best_model_path and coerce_bool(self.nn_config.get("save_best_requires_validation"), False) and not validation:
            raise NeuralNetworkDataError("save_best_model_path requires validation data by policy.", {"path_fingerprint": fingerprint(save_best_model_path)})

        opt_kwargs = {
            "momentum_coefficient": momentum_coefficient if momentum_coefficient is not None else self.nn_config.get("momentum_coefficient", 0.9),
            "weight_decay_lambda": weight_decay_lambda if weight_decay_lambda is not None else self.nn_config.get("weight_decay_lambda", 0.0),
            "gradient_clip_value": gradient_clip_value if gradient_clip_value is not None else self.nn_config.get("gradient_clip_value"),
            "adagrad_epsilon": adagrad_epsilon if adagrad_epsilon is not None else self.nn_config.get("adagrad_epsilon", 1e-8),
            "adam_beta1": adam_beta1 if adam_beta1 is not None else self.adam_beta1,
            "adam_beta2": adam_beta2 if adam_beta2 is not None else self.adam_beta2,
            "adam_epsilon": adam_epsilon_opt if adam_epsilon_opt is not None else self.adam_epsilon,
        }
        log_every = print_every_n_epochs or max(1, coerce_int(get_nested(self.nn_config, "telemetry.log_every_n_epochs", 10), 10, minimum=1))
        current_lr = learning_rate
        best_val_loss = float("inf")
        best_state: Optional[List[Dict[str, Any]]] = None
        best_adam_timestep = 0
        epochs_no_improve = 0
        run = TrainingRunSummary(run_id=generate_identifier("nnrun"), started_at=utc_iso(), epochs_requested=epochs)
        self.training_history.clear()
        self._set_training_mode(True)
        if self.optimizer_name == "adam":
            self.adam_global_timestep = 0
        restore_best = coerce_bool(self.nn_config.get("restore_best_weights", True), True)
        try:
            for epoch in range(epochs):
                if coerce_bool(self.nn_config.get("shuffle_training_data", True), True):
                    self.rng.shuffle(data)
                total_loss = 0.0
                samples_processed = 0
                for start in range(0, len(data), effective_batch):
                    batch = data[start:start + effective_batch]
                    if not batch:
                        continue
                    x = np.asarray([features for features, _ in batch], dtype=np.float64)
                    y = np.asarray([targets for _, targets in batch], dtype=np.float64)
                    predictions, caches = self._forward_batch(x, training=True)
                    loss = self._batch_loss(y, predictions)
                    gradients = self._backward_batch(
                        y,
                        predictions,
                        caches,
                        weight_decay_lambda=coerce_float(opt_kwargs["weight_decay_lambda"], 0.0, minimum=0.0),
                        gradient_clip_value=opt_kwargs["gradient_clip_value"],
                    )
                    self._apply_batch_gradients(gradients, current_lr, **opt_kwargs)
                    total_loss += loss * len(batch)
                    samples_processed += len(batch)
                avg_loss = total_loss / max(1, samples_processed)
                val_loss: Optional[float] = None
                val_accuracy: Optional[float] = None
                if validation:
                    metrics = self.evaluate(validation, batch_size=effective_batch)
                    val_loss = metrics.get("loss")
                    val_accuracy = metrics.get("accuracy")
                    if val_loss is not None and val_loss < best_val_loss - early_stopping_min_delta:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        best_state = copy.deepcopy(self.get_weights_biases(include_optimizer_state=True))
                        best_adam_timestep = self.adam_global_timestep
                        if save_best_model_path:
                            self.save_model(save_best_model_path)
                    else:
                        epochs_no_improve += 1

                record = TrainingEpochRecord(epoch=epoch + 1, loss=avg_loss, learning_rate=current_lr, samples=samples_processed, validation_loss=val_loss, validation_accuracy=val_accuracy)
                self.training_history.append(record)
                limit = coerce_int(get_nested(self.nn_config, "telemetry.history_limit", 500), 500, minimum=1)
                if len(self.training_history) > limit:
                    self.training_history = self.training_history[-limit:]
                if verbose and (epoch + 1) % log_every == 0:
                    logger.info("Neural network training telemetry: %s", safe_log_payload("nn_epoch", {"epoch": epoch + 1, "loss": avg_loss, "learning_rate": current_lr, "validation_loss": val_loss, "validation_accuracy": val_accuracy}))
                if early_stopping_patience and validation and epochs_no_improve >= early_stopping_patience:
                    record.stopped = True
                    run.early_stopped = True
                    break
                current_lr = self._apply_lr_schedule(learning_rate, current_lr, epoch, epochs, lr_scheduler_name, lr_decay_rate, lr_decay_steps)

            if validation and restore_best and best_state is not None:
                self.set_weights_biases(best_state, include_optimizer_state=True)
                self.adam_global_timestep = best_adam_timestep
                run.best_weights_restored = True

            run.epochs_completed = len(self.training_history)
            run.samples_seen = sum(item.samples for item in self.training_history)
            run.best_validation_loss = None if best_val_loss == float("inf") else best_val_loss
            run.final_loss = self.training_history[-1].loss if self.training_history else None
            run.finished_at = utc_iso()
            run.model_fingerprint = self.model_fingerprint()
            self.last_training_summary = run
            self.updated_at = utc_iso()
            return run
        except SecurityError:
            raise
        except Exception as exc:
            # wrap_security_exception intentionally has no `message=` argument.
            raise wrap_security_exception(
                exc,
                operation="train",
                error_type=SecurityErrorType.UNSAFE_MODEL_STATE,
                context={"model_id": self.model_id, "epochs_requested": epochs},
                component="neural_network.train",
                severity=SecuritySeverity.CRITICAL,
            ) from exc
        finally:
            self._set_training_mode(False)

    def evaluate(self, test_data: Union[List[Tuple[List[float], List[float]]], Tuple[np.ndarray, np.ndarray], np.ndarray], batch_size: Optional[int] = 32) -> Dict[str, float]:
        data = self._normalize_dataset(test_data, context="evaluation")
        if not data:
            return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        self._set_training_mode(False)
        total_loss = 0.0
        tp = fp = tn = fn = 0
        multiclass_correct = 0
        effective_batch = coerce_int(batch_size, 32, minimum=1, maximum=max(1, len(data)))
        for start in range(0, len(data), effective_batch):
            batch = data[start:start + effective_batch]
            x = np.asarray([features for features, _ in batch], dtype=np.float64)
            y = np.asarray([targets for _, targets in batch], dtype=np.float64)
            outputs, _ = self._forward_batch(x, training=False)
            total_loss += self._batch_loss(y, outputs) * len(batch)
            for prediction, target in zip(outputs, y):
                if self.problem_type == "binary_classification":
                    predicted = 1 if prediction[0] >= self.binary_threshold else 0
                    truth = int(round(float(target[0])))
                    tp += int(predicted == 1 and truth == 1)
                    fp += int(predicted == 1 and truth == 0)
                    tn += int(predicted == 0 and truth == 0)
                    fn += int(predicted == 0 and truth == 1)
                elif self.problem_type == "multiclass_classification":
                    multiclass_correct += int(int(np.argmax(prediction)) == int(np.argmax(target)))
        count = len(data)
        metrics: Dict[str, float] = {"loss": total_loss / count}
        if self.problem_type == "binary_classification":
            accuracy = (tp + tn) / count
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics.update({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn)})
        elif self.problem_type == "multiclass_classification":
            metrics["accuracy"] = multiclass_correct / count
        return metrics

    # ------------------------------------------------------------------
    # Gradient verification (development / CI)
    # ------------------------------------------------------------------

    def gradient_check(self, features: Sequence[float], targets: Sequence[float], *, epsilon: float = 1e-5, tolerance: float = 1e-4, max_parameters: int = 32) -> Dict[str, Any]:
        """Numerically verify weight gradients for a deterministic no-dropout pass.

        Intended for CI/development, not the runtime hot path.  Batch-normalized
        layers are rejected because finite-difference checking of BN requires a
        multi-sample batch contract.
        """
        if any(layer.use_batch_norm or layer.dropout_rate > 0.0 for layer in self.layers):
            raise NeuralNetworkDataError("gradient_check requires batch_norm=False and dropout=0 for all layers.")
        x = np.asarray([self._coerce_sample(features, expected_length=self.num_inputs, context="gradient_check.features")], dtype=np.float64)
        y = np.asarray([self._coerce_sample(targets, expected_length=self._expected_target_length(), context="gradient_check.targets")], dtype=np.float64)
        p, caches = self._forward_batch(x, training=False)
        analytical = self._backward_batch(y, p, caches, weight_decay_lambda=0.0, gradient_clip_value=None)
        checked = 0
        max_relative_error = 0.0
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                for weight_idx in range(len(neuron.weights)):
                    if checked >= max_parameters:
                        break
                    original = neuron.weights[weight_idx]
                    neuron.weights[weight_idx] = original + epsilon
                    plus, _ = self._forward_batch(x, training=False)
                    loss_plus = self._batch_loss(y, plus)
                    neuron.weights[weight_idx] = original - epsilon
                    minus, _ = self._forward_batch(x, training=False)
                    loss_minus = self._batch_loss(y, minus)
                    neuron.weights[weight_idx] = original
                    numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
                    expected = float(analytical[layer_idx]["weights"][neuron_idx, weight_idx])
                    denominator = max(1e-12, abs(numerical) + abs(expected))
                    error = abs(numerical - expected) / denominator
                    max_relative_error = max(max_relative_error, error)
                    checked += 1
                if checked >= max_parameters:
                    break
            if checked >= max_parameters:
                break
        return {"checked_parameters": checked, "max_relative_error": max_relative_error, "tolerance": tolerance, "passed": max_relative_error <= tolerance}

    # ------------------------------------------------------------------
    # Prediction / state / persistence
    # ------------------------------------------------------------------

    def predict(self, inputs: List[float]) -> List[float]:
        self._set_training_mode(False)
        return self.feed_forward_sample(inputs, None)

    def predict_proba(self, inputs: List[float]) -> List[float]:
        outputs = self.predict(inputs)
        if self.problem_type in {"binary_classification", "multiclass_classification"}:
            _validate_probability_vector(outputs, context="prediction")
        return outputs

    def predict_class(self, inputs: List[float]) -> Union[int, List[float]]:
        probabilities = self.predict_proba(inputs)
        if self.problem_type == "binary_classification":
            return 1 if probabilities[0] >= self.binary_threshold else 0
        if self.problem_type == "multiclass_classification":
            return probabilities.index(max(probabilities))
        logger.warning("predict_class called for regression model; returning raw output.")
        return probabilities

    def get_weights_biases(self, *, include_optimizer_state: bool = False) -> List[Dict[str, Any]]:
        return [layer.to_dict(include_optimizer_state=include_optimizer_state) for layer in self.layers]

    def set_weights_biases(self, network_params: List[Dict[str, Any]], *, include_optimizer_state: bool = True) -> None:
        if len(network_params) != len(self.layers):
            raise NeuralNetworkPersistenceError("Network layer count does not match configured architecture.", {"expected": len(self.layers), "actual": len(network_params)})
        for layer, state in zip(self.layers, network_params):
            layer.load_state(state, include_optimizer_state=include_optimizer_state)
        self.updated_at = utc_iso()

    def model_card(self, *, redacted: bool = False) -> Dict[str, Any]:
        card = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "problem_type": self.problem_type,
            "loss_function_name": self.loss_function_name,
            "optimizer_name": self.optimizer_name,
            "num_inputs": self.num_inputs,
            "num_layers": len(self.layers),
            "layers": [{"index": idx, "neurons": layer.num_neurons, "activation": layer.activation_name, "dropout": layer.dropout_rate, "batch_norm": layer.use_batch_norm} for idx, layer in enumerate(self.layers)],
            "last_training_summary": self.last_training_summary.to_dict() if self.last_training_summary else None,
            "fingerprint": self.model_fingerprint(),
        }
        return sanitize_for_logging(card) if redacted else card

    def model_fingerprint(self) -> str:
        payload = {"num_inputs": self.num_inputs, "problem_type": self.problem_type, "loss_function_name": self.loss_function_name, "optimizer_name": self.optimizer_name, "weights": self.get_weights_biases(include_optimizer_state=False)}
        return fingerprint(payload, algorithm=get_nested(self.nn_config, "persistence.signature_hash_algorithm", "sha256"), salt=get_nested(self.nn_config, "persistence.signature_salt", ""), length=32)

    def _serialize_model_state(self) -> Dict[str, Any]:
        include_optimizer = coerce_bool(get_nested(self.nn_config, "persistence.include_optimizer_state", True), True)
        include_history = coerce_bool(get_nested(self.nn_config, "persistence.include_training_history", True), True)
        state: Dict[str, Any] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "saved_at": utc_iso(),
            "model_id": self.model_id,
            "num_inputs": self.num_inputs,
            "layer_config_original": list(self.layer_config),
            "loss_function_name": self.loss_function_name,
            "optimizer_name": self.optimizer_name,
            "problem_type": self.problem_type,
            "initialization_method_default": self.initialization_method_default,
            "binary_threshold": self.binary_threshold,
            "config_used": self.nn_config,
            "trained_weights_biases": self.get_weights_biases(include_optimizer_state=include_optimizer),
            "adam_global_timestep": self.adam_global_timestep,
            "training_history": [record.to_dict() for record in self.training_history] if include_history else [],
            "last_training_summary": self.last_training_summary.to_dict() if self.last_training_summary else None,
        }
        state["model_fingerprint"] = self.model_fingerprint()
        state["signature"] = self._signature_for_payload(state)
        return state

    def _signature_for_payload(self, payload: Mapping[str, Any]) -> str:
        unsigned = dict(payload)
        unsigned.pop("signature", None)
        return hash_text(stable_json(unsigned), algorithm=get_nested(self.nn_config, "persistence.signature_hash_algorithm", "sha256"), salt=get_nested(self.nn_config, "persistence.signature_salt", ""))

    def _verify_model_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise NeuralNetworkPersistenceError("Unsupported neural-network model schema version.", {"schema_version": state.get("schema_version")})
        expected = state.get("signature")
        actual = self._signature_for_payload(state)
        if expected and not constant_time_equals(str(expected), actual):
            raise NeuralNetworkPersistenceError("Model signature verification failed.", {"expected_fingerprint": fingerprint(expected), "actual_fingerprint": fingerprint(actual)})
        if not expected and coerce_bool(get_nested(self.nn_config, "persistence.require_signature_on_load", True), True):
            raise NeuralNetworkPersistenceError("Model signature missing but required by policy.", {"model_id": state.get("model_id")})

    def save_model(self, filepath: str) -> bool:
        path = Path(filepath)
        if not path.is_absolute():
            path = Path(str(get_nested(self.nn_config, "persistence.model_dir", "src/agents/safety/models"))) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._serialize_model_state()
        try:
            if path.exists() and coerce_bool(get_nested(self.nn_config, "persistence.write_backup", True), True):
                backup = path.with_suffix(path.suffix + f".{utc_iso().replace(':', '').replace('-', '')}.bak")
                shutil.copy2(path, backup)
            payload = json.dumps(to_jsonable(state), indent=2, ensure_ascii=False, sort_keys=True)
            if coerce_bool(get_nested(self.nn_config, "persistence.atomic_write", True), True):
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(payload, encoding="utf-8")
                temporary.replace(path)
            else:
                path.write_text(payload, encoding="utf-8")
            logger.info("Cyber-security model saved: %s", safe_log_payload("model_saved", {"path_fingerprint": fingerprint(str(path)), "fingerprint": state.get("model_fingerprint")}))
            return True
        except SecurityError:
            raise
        except Exception as exc:
            raise NeuralNetworkPersistenceError("Failed to save neural-network model.", {"path_fingerprint": fingerprint(str(path))}, cause=exc) from exc

    @classmethod
    def load_model(cls, filepath: str, custom_config_override: Optional[Dict[str, Any]] = None) -> "NeuralNetwork":
        base = get_config_section("neural_network") or {}
        if custom_config_override:
            base = deep_merge(base, custom_config_override)
        path = Path(filepath)
        if not path.is_absolute():
            path = Path(str(get_nested(base, "persistence.model_dir", "src/agents/safety/models"))) / path
        maximum = coerce_int(get_nested(base, "persistence.max_model_file_bytes", 10_485_760), 10_485_760, minimum=1024)
        try:
            if not path.exists():
                raise NeuralNetworkPersistenceError("Model file does not exist.", {"path_fingerprint": fingerprint(str(path))})
            if path.stat().st_size > maximum:
                raise ResourceExhaustionError("model_file_bytes", path.stat().st_size, maximum, source_identifier=fingerprint(str(path)))
            state = parse_json_object(path.read_text(encoding="utf-8"), context="neural_network_model_file")
            loaded_config = deep_merge(state.get("config_used", {}), custom_config_override or {})
            network = cls(
                num_inputs=state["num_inputs"],
                layer_config=state["layer_config_original"],
                loss_function_name=state.get("loss_function_name", "cross_entropy"),
                optimizer_name=state.get("optimizer_name", "adam"),
                initialization_method_default=state.get("initialization_method_default", "he_normal"),
                problem_type=state.get("problem_type", "binary_classification"),
                config=loaded_config,
            )
            network._verify_model_state(state)
            network.model_id = str(state.get("model_id", network.model_id))
            network.binary_threshold = coerce_float(state.get("binary_threshold", network.binary_threshold), network.binary_threshold, minimum=0.0, maximum=1.0)
            network.set_weights_biases(state.get("trained_weights_biases", []), include_optimizer_state=coerce_bool(get_nested(network.nn_config, "persistence.include_optimizer_state", True), True))
            network.adam_global_timestep = coerce_int(state.get("adam_global_timestep", 0), 0, minimum=0)
            history = state.get("training_history", [])
            network.training_history = [TrainingEpochRecord(**record) for record in history if isinstance(record, Mapping)]
            if isinstance(state.get("last_training_summary"), Mapping):
                summary = dict(state["last_training_summary"])
                # Backward compatibility with v3 artifacts predating this field.
                summary.setdefault("best_weights_restored", False)
                network.last_training_summary = TrainingRunSummary(**summary)
            logger.info("Cyber-security model loaded: %s", safe_log_payload("model_loaded", {"path_fingerprint": fingerprint(str(path)), "model_id": network.model_id}))
            return network
        except SecurityError:
            raise
        except Exception as exc:
            raise NeuralNetworkPersistenceError("Failed to load neural-network model.", {"path_fingerprint": fingerprint(str(path))}, cause=exc) from exc

    def __repr__(self) -> str:
        return f"NeuralNetwork(id={self.model_id}, inputs={self.num_inputs}, layers={len(self.layers)}, problem={self.problem_type})"


def _generate_security_data(num_samples: int, rng: Optional[random.Random] = None) -> Dataset:
    rng = rng or random.Random(1337)
    data: Dataset = []
    for _ in range(num_samples):
        malicious = rng.random() < 0.35
        if malicious:
            sample = [rng.uniform(0.55, 1.0), rng.uniform(0.35, 1.0), 1.0 if rng.random() < 0.75 else 0.0, rng.uniform(0.0, 0.25) if rng.random() < 0.6 else rng.uniform(0.75, 1.0)]
            target = [1.0]
        else:
            sample = [rng.uniform(0.0, 0.3), rng.uniform(0.0, 0.55), 1.0 if rng.random() < 0.08 else 0.0, rng.uniform(0.15, 0.75)]
            target = [0.0]
        data.append((sample, target))
    return data

__all__ = [
    # Module metadata
    "MODULE_VERSION",
    "MODEL_SCHEMA_VERSION",
    "TELEMETRY_SCHEMA_VERSION",
    # Supported enums / registries
    "SUPPORTED_PROBLEM_TYPES",
    "SUPPORTED_LOSSES",
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_LR_SCHEDULERS",
    "SUPPORTED_INITIALIZERS",
    "ACTIVATION_FUNCTIONS",
    # Dataclasses
    "TrainingEpochRecord",
    "TrainingRunSummary",
    # Structured errors
    "NeuralNetworkDataError",
    "NeuralNetworkPersistenceError",
    # Loss functions
    "mean_squared_error",
    "mean_squared_error_derivative",
    # Core classes
    "Neuron",
    "NeuralLayer",
    "NeuralNetwork",
]

if __name__ == "__main__":
    print("\n=== Running Neural Network ===\n")
    printer.status("TEST", "Neural Network initialized", "info")
    config = {"deterministic_seed": 2026, "max_epochs": 20, "restore_best_weights": True, "telemetry": {"history_limit": 100, "log_every_n_epochs": 5}, "persistence": {"model_dir": "/tmp/safety_neural_network_models", "require_signature_on_load": True, "write_backup": False}}
    data = _generate_security_data(60, random.Random(2026))
    network = NeuralNetwork(
        num_inputs=4,
        layer_config=[
            {"neurons": 6, "activation": "relu", "dropout": 0.1, "batch_norm": True},
            {"neurons": 3, "activation": "relu", "dropout": 0.1, "batch_norm": True},
            {"neurons": 1, "activation": "sigmoid"},
        ],
        loss_function_name="cross_entropy",
        optimizer_name="adam",
        problem_type="binary_classification",
        config=config,
    )
    summary = network.train(data[:45], epochs=5, initial_learning_rate=0.01, batch_size=8, validation_data=data[45:55], verbose=False)
    metrics = network.evaluate(data[55:])
    assert summary.epochs_completed >= 1
    assert math.isfinite(metrics["loss"])
    assert len(network.predict(data[0][0])) == 1

    # Separate deterministic gradient test without BN/dropout.
    checker = NeuralNetwork(
        num_inputs=2,
        layer_config=[{"neurons": 2, "activation": "tanh", "dropout": 0.0, "batch_norm": False}, {"neurons": 1, "activation": "sigmoid"}],
        loss_function_name="cross_entropy",
        optimizer_name="adam",
        problem_type="binary_classification",
        config={"deterministic_seed": 7},
    )
    check = checker.gradient_check([0.2, -0.4], [1.0], max_parameters=6, tolerance=1e-3)
    assert check["passed"], check
    printer.status("TEST", f"Gradient check max error={check['max_relative_error']:.3e}", "info")
    print("\n=== Test ran successfully ===\n")