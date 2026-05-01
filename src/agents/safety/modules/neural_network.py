"""
Production-grade native neural network for adaptive cyber-security analysis.

This module keeps the original Safety Agent context: a lightweight, inspectable
multi-layer perceptron used by the cyber-security and adaptive-safety subsystem
for threat scoring, anomaly-style classification, and defensive model training.

Design goals:
- keep the native Python/NumPy implementation transparent and auditable;
- retain existing public class names and method names used by subsystem modules;
- use shared safety helpers for hashing, IDs, serialization, redaction, config,
  audit-safe payloads, and error wrapping instead of duplicating utilities;
- use structured security errors for configuration, integrity, unsafe execution,
  data validation, persistence, and model-loading failures;
- keep model files tamper-evident and safe for operational audit;
- support production controls such as deterministic seeding, gradient clipping,
  early stopping, validation telemetry, signed persistence, and bounded runtime.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from ...base.modules.activation_engine import *
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
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.security_error import *
from ..utils.safety_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Cyber-Security Neural-Network")
printer = PrettyPrinter()

MODULE_VERSION = "3.0.0"
MODEL_SCHEMA_VERSION = "safety_neural_network.model.v3"
TELEMETRY_SCHEMA_VERSION = "safety_neural_network.telemetry.v2"

SUPPORTED_PROBLEM_TYPES = {"regression", "binary_classification", "multiclass_classification"}
SUPPORTED_LOSSES = {"mse", "cross_entropy"}
SUPPORTED_OPTIMIZERS = {"sgd_momentum_adagrad", "adam"}
SUPPORTED_LR_SCHEDULERS = {None, "none", "step", "exponential", "cosine_annealing"}
SUPPORTED_INITIALIZERS = {"uniform_scaled", "he_normal", "lecun_normal", "xavier_uniform", "xavier_normal", "small_uniform"}

# --- Activation Functions and their Derivatives (Referenced from math_science) ---
ACTIVATION_FUNCTIONS: Dict[str, Tuple[Callable[..., float], Callable[..., float], bool]] = {
    "sigmoid": (sigmoid, sigmoid_derivative, False),
    "relu": (relu, relu_derivative, False),
    "tanh": (tanh, tanh_derivative, False),
    "leaky_relu": (leaky_relu, leaky_relu_derivative, True),
    "elu": (elu, elu_derivative, True),
    "swish": (swish, swish_derivative, False),
    "linear": (lambda x: x, lambda x: 1.0, False),
}


@dataclass
class TrainingEpochRecord:
    """Audit-safe telemetry for a single training epoch."""

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
    """Compact training summary for model cards, audit, and regression tests."""

    run_id: str
    started_at: str
    finished_at: Optional[str] = None
    epochs_requested: int = 0
    epochs_completed: int = 0
    samples_seen: int = 0
    best_validation_loss: Optional[float] = None
    final_loss: Optional[float] = None
    early_stopped: bool = False
    model_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NeuralNetworkDataError(SecurityError):
    """Raised when model input, target, or training data is unsafe or malformed."""

    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None):
        super().__init__(
            SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
            message,
            severity=SecuritySeverity.HIGH,
            context=context or {},
            component="neural_network",
            response_action=SecurityResponseAction.BLOCK,
            remediation_guidance=(
                "Reject malformed neural-network data before training or inference.",
                "Validate feature dimensions, target dimensions, and numeric finiteness.",
                "Add a regression test for the rejected data shape or numeric condition.",
            ),
        )


class NeuralNetworkPersistenceError(SecurityError):
    """Raised for model save/load/signature failures."""

    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None, *, cause: Optional[BaseException] = None):
        super().__init__(
            SecurityErrorType.MODEL_TAMPERING if "signature" in message.lower() or "integrity" in message.lower() else SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
            message,
            severity=SecuritySeverity.CRITICAL,
            context=context or {},
            component="neural_network.persistence",
            response_action=SecurityResponseAction.QUARANTINE,
            cause=cause,
            remediation_guidance=(
                "Do not load or serve the affected model artifact.",
                "Verify model provenance, path, file permissions, and signature configuration.",
                "Restore from a known-good signed model artifact if needed.",
            ),
        )


def _safe_float(value: Any, *, context: str = "value") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NeuralNetworkDataError(f"Non-numeric neural-network {context} encountered.", {"value_repr": safe_repr(value)}) from exc
    if not math.isfinite(result):
        raise NeuralNetworkDataError(f"Non-finite neural-network {context} encountered.", {"value_repr": safe_repr(value)})
    return result


def _clip(value: float, limit: Optional[float]) -> float:
    if limit is None:
        return value
    limit = abs(float(limit))
    return max(-limit, min(limit, value))


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
    """L = 0.5 * sum((target_i - output_i)^2)."""

    if len(targets) != len(outputs):
        raise NeuralNetworkDataError(
            "Targets and outputs must have the same length for MSE.",
            {"target_length": len(targets), "output_length": len(outputs)},
        )
    return 0.5 * sum((target - output) ** 2 for target, output in zip(targets, outputs))


def mean_squared_error_derivative(targets: List[float], outputs: List[float]) -> List[float]:
    """Derivative of MSE with respect to outputs."""

    if len(targets) != len(outputs):
        raise NeuralNetworkDataError(
            "Targets and outputs must have the same length for MSE derivative.",
            {"target_length": len(targets), "output_length": len(outputs)},
        )
    return [(output - target) for target, output in zip(targets, outputs)]


class Neuron:
    """
    Single processing unit for security feature analysis.

    The neuron intentionally remains simple and inspectable: a weighted sum, an
    activation function, optimizer state, and explicit gradient calculations. It
    raises structured security errors for malformed architecture or non-finite
    numeric state rather than failing with raw ValueError paths.
    """

    def __init__(
        self,
        num_inputs: int,
        activation_name: str = "relu",
        initialization_method: str = "he_normal",
        activation_alpha: float = 0.01,
        *,
        rng: Optional[random.Random] = None,
    ):
        super().__init__()
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
        self.weights: List[float] = self._initialize_weights()
        self.bias: float = self.rng.uniform(-0.1, 0.1)
        self.inputs: List[float] = [0.0] * self.num_inputs
        self.weighted_sum: float = 0.0
        self.activation: float = 0.0
        self.delta: float = 0.0
        self.reset_optimizer_state()

    def reset_optimizer_state(self) -> None:
        self.velocity_weights: List[float] = [0.0] * self.num_inputs
        self.velocity_bias: float = 0.0
        self.cache_weights: List[float] = [0.0] * self.num_inputs
        self.cache_bias: float = 0.0
        self.m_weights: List[float] = [0.0] * self.num_inputs
        self.v_weights: List[float] = [0.0] * self.num_inputs
        self.m_bias: float = 0.0
        self.v_bias: float = 0.0

    def _initialize_weights(self) -> List[float]:
        fan_in = self.num_inputs
        if self.initialization_method == "uniform_scaled":
            limit = 1.0 / math.sqrt(fan_in)
            return [self.rng.uniform(-limit, limit) for _ in range(fan_in)]
        if self.initialization_method == "he_normal":
            stddev = math.sqrt(2.0 / fan_in)
            return [self.rng.gauss(0.0, stddev) for _ in range(fan_in)]
        if self.initialization_method == "lecun_normal":
            stddev = math.sqrt(1.0 / fan_in)
            return [self.rng.gauss(0.0, stddev) for _ in range(fan_in)]
        if self.initialization_method == "xavier_uniform":
            limit = math.sqrt(6.0 / (fan_in + 1))
            return [self.rng.uniform(-limit, limit) for _ in range(fan_in)]
        if self.initialization_method == "xavier_normal":
            stddev = math.sqrt(2.0 / (fan_in + 1))
            return [self.rng.gauss(0.0, stddev) for _ in range(fan_in)]
        return [self.rng.uniform(-0.1, 0.1) for _ in range(fan_in)]

    def _call_activation_fn(self, x: float) -> float:
        return self.activation_fn_ptr(x, self.activation_alpha) if self.activation_needs_alpha else self.activation_fn_ptr(x)

    def _call_activation_fn_derivative(self, x: float) -> float:
        return self.activation_fn_derivative_ptr(x, self.activation_alpha) if self.activation_needs_alpha else self.activation_fn_derivative_ptr(x)

    def _calculate_weighted_sum(self, inputs: List[float]) -> float:
        if len(inputs) != self.num_inputs:
            raise NeuralNetworkDataError(
                "Neuron input dimension mismatch.",
                {"expected": self.num_inputs, "actual": len(inputs)},
            )
        clean_inputs = [_safe_float(value, context="input") for value in inputs]
        self.inputs = clean_inputs
        self.weighted_sum = sum(w * i for w, i in zip(self.weights, clean_inputs)) + self.bias
        if not math.isfinite(self.weighted_sum):
            raise NeuralNetworkDataError("Non-finite weighted sum detected.", {"num_inputs": self.num_inputs})
        return self.weighted_sum

    def activate(self, inputs: List[float]) -> float:
        z = self._calculate_weighted_sum(inputs)
        self.activation = _safe_float(self._call_activation_fn(z), context="activation")
        return self.activation

    def calculate_delta(self, error_signal_from_downstream: float) -> None:
        signal = _safe_float(error_signal_from_downstream, context="error_signal")
        self.delta = signal * _safe_float(self._call_activation_fn_derivative(self.weighted_sum), context="activation_derivative")
        if not math.isfinite(self.delta):
            raise NeuralNetworkDataError("Non-finite neuron delta detected.", {"activation": self.activation})

    def calculate_gradients(
        self,
        weight_decay_lambda: float = 0.0,
        gradient_clip_value: Optional[float] = None,
    ) -> Tuple[List[float], float]:
        weight_decay_lambda = coerce_float(weight_decay_lambda, 0.0, minimum=0.0)
        clip_value = None if gradient_clip_value is None else abs(coerce_float(gradient_clip_value, 0.0))
        grad_weights = []
        for inp, weight in zip(self.inputs, self.weights):
            grad = self.delta * inp + weight_decay_lambda * weight
            grad_weights.append(_clip(_safe_float(grad, context="weight_gradient"), clip_value))
        grad_bias = _clip(_safe_float(self.delta, context="bias_gradient"), clip_value)
        return grad_weights, grad_bias

    def update_parameters(
        self,
        grad_weights: List[float],
        grad_bias: float,
        learning_rate: float,
        momentum_coefficient: float = 0.0,
        adagrad_epsilon: float = 1e-8,
    ) -> None:
        if len(grad_weights) != len(self.weights):
            raise NeuralNetworkDataError(
                "Gradient dimension mismatch during optimizer update.",
                {"expected": len(self.weights), "actual": len(grad_weights)},
            )
        learning_rate = coerce_float(learning_rate, 0.001, minimum=0.0)
        momentum_coefficient = coerce_float(momentum_coefficient, 0.0, minimum=0.0, maximum=0.999999)
        adagrad_epsilon = coerce_float(adagrad_epsilon, 1e-8, minimum=1e-12)
        for i, grad in enumerate(grad_weights):
            self.cache_weights[i] += grad ** 2
            adjusted_lr = learning_rate / (math.sqrt(self.cache_weights[i]) + adagrad_epsilon)
            self.velocity_weights[i] = momentum_coefficient * self.velocity_weights[i] - adjusted_lr * grad
            self.weights[i] = _safe_float(self.weights[i] + self.velocity_weights[i], context="weight")
        self.cache_bias += grad_bias ** 2
        adjusted_lr_bias = learning_rate / (math.sqrt(self.cache_bias) + adagrad_epsilon)
        self.velocity_bias = momentum_coefficient * self.velocity_bias - adjusted_lr_bias * grad_bias
        self.bias = _safe_float(self.bias + self.velocity_bias, context="bias")

    def to_dict(self, include_optimizer_state: bool = True) -> Dict[str, Any]:
        data = {
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
        weights = [_safe_float(value, context="loaded_weight") for value in data.get("weights", [])]
        if len(weights) != self.num_inputs:
            raise ModelTamperingDetectedError(
                model_name="native_mlp_neuron",
                detection_method="weight_dimension_validation",
                expected_hash=str(self.num_inputs),
                actual_hash=str(len(weights)),
            )
        self.weights = weights
        self.bias = _safe_float(data.get("bias", 0.0), context="loaded_bias")
        if include_optimizer_state and isinstance(data.get("optimizer_state"), Mapping):
            state = data["optimizer_state"]
            self.velocity_weights = list(state.get("velocity_weights", [0.0] * self.num_inputs))
            self.velocity_bias = float(state.get("velocity_bias", 0.0))
            self.cache_weights = list(state.get("cache_weights", [0.0] * self.num_inputs))
            self.cache_bias = float(state.get("cache_bias", 0.0))
            self.m_weights = list(state.get("m_weights", [0.0] * self.num_inputs))
            self.v_weights = list(state.get("v_weights", [0.0] * self.num_inputs))
            self.m_bias = float(state.get("m_bias", 0.0))
            self.v_bias = float(state.get("v_bias", 0.0))

    def __repr__(self) -> str:
        return f"Neuron(Act:{self.activation_name}, Weights:{len(self.weights)}, Bias:{self.bias:.3f})"


class NeuralLayer:
    """
    Layer of neurons for hierarchical cyber-security feature extraction.

    The layer supports activation monitoring, inverted dropout, and lightweight
    batch-normalization running statistics while keeping all state serializable.
    """

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
    ):
        super().__init__()
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
            raise NeuralNetworkDataError(
                "Layer dimensions must be positive.",
                {"num_neurons": num_neurons, "num_inputs_per_neuron": num_inputs_per_neuron},
            )

        self.neurons: List[Neuron] = [
            Neuron(
                self.num_inputs_per_neuron,
                self.activation_name,
                self.initialization_method,
                self.activation_alpha,
                rng=self.rng,
            )
            for _ in range(self.num_neurons)
        ]
        self.is_training: bool = False
        self._dropout_mask: Optional[List[float]] = None
        self.bn_gamma: List[float] = [1.0] * self.num_neurons
        self.bn_beta: List[float] = [0.0] * self.num_neurons
        self.running_mean: List[float] = [0.0] * self.num_neurons
        self.running_variance: List[float] = [1.0] * self.num_neurons
        self.history_activation_mean: List[float] = []
        self.history_activation_variance: List[float] = []

    def _update_bn_running_stats(self, batch_activations_T: List[List[float]]) -> None:
        if not self.use_batch_norm or not batch_activations_T or not batch_activations_T[0]:
            return
        num_samples = len(batch_activations_T[0])
        if num_samples <= 0:
            return
        for i in range(self.num_neurons):
            values = batch_activations_T[i]
            mean_i = sum(values) / num_samples
            var_i = sum((a - mean_i) ** 2 for a in values) / num_samples
            self.running_mean[i] = self.bn_momentum * self.running_mean[i] + (1.0 - self.bn_momentum) * mean_i
            self.running_variance[i] = self.bn_momentum * self.running_variance[i] + (1.0 - self.bn_momentum) * max(var_i, 0.0)

    def _apply_batch_norm(
        self,
        current_sample_activations: List[float],
        batch_activations_for_stats_T: Optional[List[List[float]]] = None,
    ) -> List[float]:
        if not self.use_batch_norm:
            return current_sample_activations
    
        normalized = [0.0] * self.num_neurons
    
        # Determine if we can safely use batch statistics
        use_batch_stats = (
            self.is_training
            and batch_activations_for_stats_T is not None
            and batch_activations_for_stats_T
            and batch_activations_for_stats_T[0]
        )
    
        if use_batch_stats:
            # Verify that the first element is a non‑empty sequence
            first = batch_activations_for_stats_T[0] # pyright: ignore[reportOptionalSubscript]
            if not isinstance(first, (list, tuple)) or len(first) == 0:
                use_batch_stats = False
            else:
                num_samples = len(first)
                for i in range(self.num_neurons):
                    batch_values = batch_activations_for_stats_T[i] # pyright: ignore[reportOptionalSubscript]
                    mean_i = sum(batch_values) / num_samples
                    var_i = sum((a - mean_i) ** 2 for a in batch_values) / num_samples
                    norm = (current_sample_activations[i] - mean_i) / math.sqrt(var_i + self.bn_epsilon)
                    normalized[i] = self.bn_gamma[i] * norm + self.bn_beta[i]
                self._update_bn_running_stats(batch_activations_for_stats_T or [])
                return normalized
    
        # Fallback to running statistics (either use_batch_stats was False or we switched to False)
        for i in range(self.num_neurons):
            variance = max(self.running_variance[i], self.bn_epsilon)
            norm = (current_sample_activations[i] - self.running_mean[i]) / math.sqrt(variance + self.bn_epsilon)
            normalized[i] = self.bn_gamma[i] * norm + self.bn_beta[i]
            if self.is_training:
                self.running_mean[i] = self.bn_momentum * self.running_mean[i] + (1.0 - self.bn_momentum) * current_sample_activations[i]
                diff = current_sample_activations[i] - self.running_mean[i]
                self.running_variance[i] = self.bn_momentum * self.running_variance[i] + (1.0 - self.bn_momentum) * diff * diff
    
        return normalized

    def _apply_dropout(self, activations: List[float]) -> List[float]:
        if not self.is_training or self.dropout_rate <= 0.0:
            self._dropout_mask = None
            return activations
        scale = 1.0 / (1.0 - self.dropout_rate)
        self._dropout_mask = []
        output = []
        for value in activations:
            if self.rng.random() < self.dropout_rate:
                self._dropout_mask.append(0.0)
                output.append(0.0)
            else:
                self._dropout_mask.append(scale)
                output.append(value * scale)
        return output

    def feed_forward_sample(self, inputs: List[float], batch_raw_activations_T: Optional[List[List[float]]] = None) -> List[float]:
        if len(inputs) != self.num_inputs_per_neuron:
            raise NeuralNetworkDataError(
                "Layer input dimension mismatch.",
                {"expected": self.num_inputs_per_neuron, "actual": len(inputs)},
            )
        raw_activations = [neuron.activate(inputs) for neuron in self.neurons]
        if self.is_training:
            current_mean = sum(raw_activations) / self.num_neurons
            current_var = sum((a - current_mean) ** 2 for a in raw_activations) / self.num_neurons
            self.history_activation_mean.append(current_mean)
            self.history_activation_variance.append(current_var)
            if len(self.history_activation_mean) > self.history_max_len:
                self.history_activation_mean.pop(0)
                self.history_activation_variance.pop(0)
        normalized = self._apply_batch_norm(raw_activations, batch_raw_activations_T)
        final = self._apply_dropout(normalized)
        for idx, neuron in enumerate(self.neurons):
            neuron.activation = final[idx]
        return final

    def get_raw_activations_for_sample(self, inputs: List[float]) -> List[float]:
        return [neuron._call_activation_fn(neuron._calculate_weighted_sum(inputs)) for neuron in self.neurons]

    def get_layer_stats(self) -> Dict[str, Any]:
        avg_mean = sum(self.history_activation_mean) / len(self.history_activation_mean) if self.history_activation_mean else 0.0
        avg_var = sum(self.history_activation_variance) / len(self.history_activation_variance) if self.history_activation_variance else 0.0
        stats: Dict[str, Any] = {
            "avg_raw_activation_mean": avg_mean,
            "avg_raw_activation_variance": avg_var,
            "dropout_rate": self.dropout_rate,
            "batch_norm": self.use_batch_norm,
        }
        if self.use_batch_norm:
            for i in range(min(3, self.num_neurons)):
                stats[f"bn_running_mean_{i}"] = self.running_mean[i]
                stats[f"bn_running_variance_{i}"] = self.running_variance[i]
        return stats

    def to_dict(self, include_optimizer_state: bool = True) -> Dict[str, Any]:
        return {
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
            "neurons": [neuron.to_dict(include_optimizer_state=include_optimizer_state) for neuron in self.neurons],
        }

    def load_state(self, data: Mapping[str, Any], include_optimizer_state: bool = True) -> None:
        neurons = data.get("neurons", [])
        if len(neurons) != len(self.neurons):
            raise ModelTamperingDetectedError(
                model_name="native_mlp_layer",
                detection_method="neuron_count_validation",
                expected_hash=str(len(self.neurons)),
                actual_hash=str(len(neurons)),
            )
        self.bn_gamma = list(data.get("bn_gamma", self.bn_gamma))
        self.bn_beta = list(data.get("bn_beta", self.bn_beta))
        self.running_mean = list(data.get("running_mean", self.running_mean))
        self.running_variance = list(data.get("running_variance", self.running_variance))
        for neuron, state in zip(self.neurons, neurons):
            neuron.load_state(state, include_optimizer_state=include_optimizer_state)

    def __repr__(self) -> str:
        bn_repr = ", BN" if self.use_batch_norm else ""
        dropout_repr = f", Dropout:{self.dropout_rate}" if self.dropout_rate > 0 else ""
        return f"NeuralLayer({self.num_neurons}N, Act:{self.activation_name}{dropout_repr}{bn_repr})"


class NeuralNetwork:
    """
    Native MLP for adaptive cyber-security tasks.

    The class retains the original interface (`train`, `evaluate`, `predict`,
    `predict_proba`, `predict_class`, `save_model`, `load_model`) while adding
    production controls around validation, telemetry, integrity, and errors.
    """

    def __init__(
        self,
        num_inputs: int,
        layer_config: Optional[List[dict]] = None,
        loss_function_name: Optional[str] = None,
        optimizer_name: Optional[str] = None,
        initialization_method_default: Optional[str] = None,
        problem_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # self.config = load_global_config()
        loaded_config = get_config_section("neural_network") or {}
        if config:
            self.nn_config = deep_merge(loaded_config, config)
        else:
            self.nn_config = loaded_config
        self.num_inputs = coerce_int(num_inputs, -1)
        self.layer_config: List[Dict[str, Any]] = list(layer_config or self.nn_config.get("layers", []))
        self.problem_type = str(problem_type or self.nn_config.get("problem_type", "binary_classification")).lower()
        self.loss_function_name = str(loss_function_name or self.nn_config.get("loss_function_name", "cross_entropy")).lower()
        self.optimizer_name = str(optimizer_name or self.nn_config.get("optimizer_name", "adam")).lower()
        self.initialization_method_default = str(initialization_method_default or self.nn_config.get("initialization_method_default", "he_normal")).lower()
        self.binary_threshold = coerce_float(self.nn_config.get("binary_threshold", 0.5), 0.5, minimum=0.0, maximum=1.0)
        self.layers: List[NeuralLayer] = []
        self.is_training: bool = False
        self.output_layer_activation_is_softmax = False
        self.training_history: List[TrainingEpochRecord] = []
        self.last_training_summary: Optional[TrainingRunSummary] = None
        self.model_id = generate_identifier("nn")
        self.created_at = utc_iso()
        self.updated_at = self.created_at

        seed = self.nn_config.get("deterministic_seed")
        self.rng = random.Random(coerce_int(seed, 1337))
        if coerce_bool(self.nn_config.get("set_seed_on_init"), True):
            random.seed(coerce_int(seed, 1337))
            np.random.seed(coerce_int(seed, 1337))

        self._validate_architecture()
        self._configure_loss_function()
        self._configure_optimizer_hyperparameters(initial=True)
        self._build_layers()
        self._configure_optimizer_hyperparameters(initial=False)
        logger.info("Initialized NeuralNetwork %s with %s", self.model_id, safe_log_payload("nn_init", {"model_id": self.model_id, "num_inputs": self.num_inputs, "num_layers": len(self.layers), "problem_type": self.problem_type, "optimizer": self.optimizer_name}))

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
            raise NeuralNetworkDataError("Layer configuration must be provided and cannot be empty.", {"config_section": "neural_network.layers"})
        for idx, layer in enumerate(self.layer_config):
            neurons = coerce_int(layer.get("neurons"), -1)
            if neurons <= 0:
                raise ConfigurationTamperingError("secure_config.yaml:neural_network.layers", f"Layer {idx} has invalid neuron count: {neurons}")
            activation = str(layer.get("activation", "relu")).lower()
            if activation not in set(ACTIVATION_FUNCTIONS).union({"softmax"}):
                raise ConfigurationTamperingError("secure_config.yaml:neural_network.layers.activation", f"Layer {idx} has unsupported activation: {activation}")
            dropout = coerce_float(layer.get("dropout", 0.0), 0.0)
            if dropout < 0.0 or dropout >= 1.0:
                raise ConfigurationTamperingError("secure_config.yaml:neural_network.layers.dropout", f"Layer {idx} dropout must be in [0, 1): {dropout}")
        output_neurons = coerce_int(self.layer_config[-1].get("neurons"), 1)
        if self.problem_type == "binary_classification" and output_neurons != 1:
            raise NeuralNetworkDataError("Binary classification requires exactly one output neuron.", {"output_neurons": output_neurons})
        if self.problem_type == "multiclass_classification" and output_neurons < 2:
            raise NeuralNetworkDataError("Multiclass classification requires at least two output neurons.", {"output_neurons": output_neurons})

    def _build_layers(self) -> None:
        current_inputs = self.num_inputs
        self.layers = []
        for index, layer_conf in enumerate(self.layer_config):
            is_output = index == len(self.layer_config) - 1
            default_activation = "relu"
            if is_output:
                default_activation = "sigmoid" if self.problem_type == "binary_classification" else "linear"
                if self.problem_type == "multiclass_classification":
                    default_activation = "softmax"
            activation_name = str(layer_conf.get("activation", default_activation)).lower()
            actual_activation = "linear" if is_output and activation_name == "softmax" else activation_name
            if is_output and activation_name == "softmax":
                self.output_layer_activation_is_softmax = True
            init_method = str(layer_conf.get("init", self.initialization_method_default)).lower()
            dropout = 0.0 if is_output else coerce_float(layer_conf.get("dropout", self.nn_config.get("default_dropout_rate", 0.0)), 0.0)
            layer = NeuralLayer(
                num_neurons=coerce_int(layer_conf["neurons"], -1),
                num_inputs_per_neuron=current_inputs,
                activation_name=actual_activation,
                initialization_method=init_method,
                dropout_rate=dropout,
                activation_alpha=coerce_float(layer_conf.get("alpha", self.nn_config.get("default_activation_alpha", 0.01)), 0.01),
                use_batch_norm=False if is_output else coerce_bool(layer_conf.get("batch_norm", self.nn_config.get("default_use_batch_norm", False)), False),
                bn_momentum=coerce_float(layer_conf.get("bn_momentum", self.nn_config.get("default_bn_momentum", 0.9)), 0.9),
                bn_epsilon=coerce_float(layer_conf.get("bn_epsilon", self.nn_config.get("default_bn_epsilon", 1e-5)), 1e-5),
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
        if self.optimizer_name == "adam" and self.layers:
            for layer in self.layers:
                for neuron in layer.neurons:
                    if len(neuron.m_weights) != neuron.num_inputs:
                        neuron.reset_optimizer_state()

    def _set_training_mode(self, mode: bool) -> None:
        self.is_training = bool(mode)
        for layer in self.layers:
            layer.is_training = bool(mode)

    def _coerce_sample(self, sample: Sequence[Any], *, expected_length: int, context: str) -> List[float]:
        if len(sample) != expected_length:
            raise NeuralNetworkDataError(
                f"{context} dimension mismatch.",
                {"expected": expected_length, "actual": len(sample)},
            )
        return [_safe_float(value, context=f"{context}[{idx}]") for idx, value in enumerate(sample)]

    def _expected_target_length(self) -> int:
        return self.layers[-1].num_neurons

    def _normalize_dataset(
        self,
        data: Union[List[Tuple[List[float], List[float]]], Tuple[np.ndarray, np.ndarray], np.ndarray],
        *,
        context: str,
    ) -> List[Tuple[List[float], List[float]]]:
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray):
            raw_pairs = list(zip(data[0], data[1]))
        elif isinstance(data, np.ndarray) and data.ndim >= 2:
            raw_pairs = list(zip(data[0], data[1]))
        else:
            raw_pairs = list(data)  # type: ignore[arg-type]
        max_samples = coerce_int(self.nn_config.get("max_training_samples", 250000), 250000, minimum=1)
        if len(raw_pairs) > max_samples:
            raise ResourceExhaustionError("training_samples", len(raw_pairs), max_samples, source_identifier="neural_network.train")
        normalized: List[Tuple[List[float], List[float]]] = []
        target_len = self._expected_target_length()
        for idx, pair in enumerate(raw_pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise NeuralNetworkDataError("Dataset entries must be (features, targets) pairs.", {"index": idx, "context": context})
            features, targets = pair
            feature_list = list(features.tolist() if hasattr(features, "tolist") else features)
            target_list_raw = targets.tolist() if hasattr(targets, "tolist") else targets
            if not isinstance(target_list_raw, (tuple, list, np.ndarray)):
                target_list_raw = [target_list_raw]
            target_list = list(target_list_raw)
            features_clean = self._coerce_sample(feature_list, expected_length=self.num_inputs, context=f"{context}.features")
            targets_clean = self._coerce_sample(target_list, expected_length=target_len, context=f"{context}.targets")
            if self.problem_type in {"binary_classification", "multiclass_classification"}:
                _validate_probability_vector(targets_clean, context=f"{context}.targets")
            normalized.append((features_clean, targets_clean))
        return normalized

    def feed_forward_sample(self, inputs: List[float], batch_raw_activations_T_by_layer: Optional[List[List[List[float]]]] = None) -> List[float]:
        clean_inputs = self._coerce_sample(inputs, expected_length=self.num_inputs, context="inference.features")
        current_outputs = clean_inputs
        for idx, layer in enumerate(self.layers):
            batch_stats = batch_raw_activations_T_by_layer[idx] if batch_raw_activations_T_by_layer and idx < len(batch_raw_activations_T_by_layer) else None
            current_outputs = layer.feed_forward_sample(current_outputs, batch_stats)
            if idx == len(self.layers) - 1 and self.output_layer_activation_is_softmax:
                current_outputs = [_safe_float(value, context="softmax_logit") for value in current_outputs]
                probabilities = softmax(current_outputs)
                for neuron_idx, neuron in enumerate(self.layers[-1].neurons):
                    neuron.activation = probabilities[neuron_idx]
                current_outputs = probabilities
        if coerce_bool(self.nn_config.get("strict_numerics", True), True):
            for value in current_outputs:
                _safe_float(value, context="network_output")
        return current_outputs

    def _get_batch_raw_activations_for_bn(self, batch_data: List[Tuple[List[float], List[float]]]) -> List[List[List[float]]]:
        all_layers: List[List[List[float]]] = [[[] for _ in range(layer.num_neurons)] for layer in self.layers]
        for features, _ in batch_data:
            current = features
            for layer_idx, layer in enumerate(self.layers):
                raw_outputs = layer.get_raw_activations_for_sample(current)
                for neuron_idx, value in enumerate(raw_outputs):
                    all_layers[layer_idx][neuron_idx].append(value)
                current = raw_outputs
        return all_layers

    def _calculate_loss_and_output_deltas(self, target_outputs: List[float], predicted_outputs: List[float]) -> Tuple[float, List[float]]:
        if len(target_outputs) != len(predicted_outputs):
            raise NeuralNetworkDataError(
                "Target and prediction dimension mismatch.",
                {"target_length": len(target_outputs), "prediction_length": len(predicted_outputs)},
            )
        if self.loss_function_name == "cross_entropy":
            loss = _safe_float(self.loss_fn(target_outputs, predicted_outputs), context="loss")
            if self.output_layer_activation_is_softmax:
                signals = []
                for idx, neuron in enumerate(self.layers[-1].neurons):
                    delta = neuron.activation - target_outputs[idx]
                    neuron.delta = _safe_float(delta, context="softmax_delta")
                    signals.append(neuron.delta)
                return loss, signals
            return loss, list(self.loss_fn_derivative(target_outputs, predicted_outputs))
        if self.loss_function_name == "mse":
            return _safe_float(self.loss_fn(target_outputs, predicted_outputs), context="loss"), list(self.loss_fn_derivative(target_outputs, predicted_outputs))
        raise ConfigurationTamperingError("secure_config.yaml:neural_network.loss_function_name", f"Unsupported loss: {self.loss_function_name}")

    def _apply_optimizer_step(self, neuron: Neuron, grad_weights: List[float], grad_bias: float, learning_rate: float, **optimizer_kwargs: Any) -> None:
        lr = coerce_float(learning_rate, 0.001, minimum=coerce_float(self.nn_config.get("learning_rate_floor", 1e-8), 1e-8), maximum=coerce_float(self.nn_config.get("learning_rate_ceiling", 1.0), 1.0))
        if self.optimizer_name == "sgd_momentum_adagrad":
            neuron.update_parameters(
                grad_weights,
                grad_bias,
                lr,
                optimizer_kwargs.get("momentum_coefficient", self.nn_config.get("momentum_coefficient", 0.9)),
                optimizer_kwargs.get("adagrad_epsilon", self.nn_config.get("adagrad_epsilon", 1e-8)),
            )
            return
        if self.optimizer_name == "adam":
            beta1 = coerce_float(optimizer_kwargs.get("adam_beta1", self.adam_beta1), self.adam_beta1, minimum=0.0, maximum=0.999999)
            beta2 = coerce_float(optimizer_kwargs.get("adam_beta2", self.adam_beta2), self.adam_beta2, minimum=0.0, maximum=0.999999)
            epsilon = coerce_float(optimizer_kwargs.get("adam_epsilon", self.adam_epsilon), self.adam_epsilon, minimum=1e-12)
            timestep = max(1, self.adam_global_timestep)
            for i, grad in enumerate(grad_weights):
                neuron.m_weights[i] = beta1 * neuron.m_weights[i] + (1.0 - beta1) * grad
                neuron.v_weights[i] = beta2 * neuron.v_weights[i] + (1.0 - beta2) * (grad ** 2)
                m_hat = neuron.m_weights[i] / (1.0 - beta1 ** timestep)
                v_hat = neuron.v_weights[i] / (1.0 - beta2 ** timestep)
                neuron.weights[i] = _safe_float(neuron.weights[i] - lr * m_hat / (math.sqrt(v_hat) + epsilon), context="adam_weight")
            neuron.m_bias = beta1 * neuron.m_bias + (1.0 - beta1) * grad_bias
            neuron.v_bias = beta2 * neuron.v_bias + (1.0 - beta2) * (grad_bias ** 2)
            m_hat_bias = neuron.m_bias / (1.0 - beta1 ** timestep)
            v_hat_bias = neuron.v_bias / (1.0 - beta2 ** timestep)
            neuron.bias = _safe_float(neuron.bias - lr * m_hat_bias / (math.sqrt(v_hat_bias) + epsilon), context="adam_bias")
            return
        raise ConfigurationTamperingError("secure_config.yaml:neural_network.optimizer_name", f"Unsupported optimizer: {self.optimizer_name}")

    def _backpropagate(self, inputs_sample: List[float], output_layer_error_signals: List[float], learning_rate: float, **optimizer_kwargs: Any) -> None:
        output_layer = self.layers[-1]
        input_activations_to_output = [n.activation for n in self.layers[-2].neurons] if len(self.layers) > 1 else inputs_sample
        for neuron_idx, neuron in enumerate(output_layer.neurons):
            if not (self.loss_function_name == "cross_entropy" and self.output_layer_activation_is_softmax):
                neuron.calculate_delta(output_layer_error_signals[neuron_idx])
            neuron.inputs = input_activations_to_output
            grad_weights, grad_bias = neuron.calculate_gradients(
                optimizer_kwargs.get("weight_decay_lambda", 0.0),
                optimizer_kwargs.get("gradient_clip_value"),
            )
            self._apply_optimizer_step(neuron, grad_weights, grad_bias, learning_rate, **optimizer_kwargs)

        for layer_idx in reversed(range(len(self.layers) - 1)):
            hidden_layer = self.layers[layer_idx]
            downstream_layer = self.layers[layer_idx + 1]
            input_activations = [n.activation for n in self.layers[layer_idx - 1].neurons] if layer_idx > 0 else inputs_sample
            for neuron_idx, neuron in enumerate(hidden_layer.neurons):
                signal_sum = 0.0
                for downstream_neuron in downstream_layer.neurons:
                    signal_sum += downstream_neuron.delta * downstream_neuron.weights[neuron_idx]
                neuron.calculate_delta(signal_sum)
                if hidden_layer._dropout_mask and hidden_layer._dropout_mask[neuron_idx] == 0.0:
                    neuron.delta = 0.0
                if neuron.delta == 0.0:
                    continue
                neuron.inputs = input_activations
                grad_weights, grad_bias = neuron.calculate_gradients(
                    optimizer_kwargs.get("weight_decay_lambda", 0.0),
                    optimizer_kwargs.get("gradient_clip_value"),
                )
                self._apply_optimizer_step(neuron, grad_weights, grad_bias, learning_rate, **optimizer_kwargs)

    def _apply_lr_schedule(
        self,
        initial_lr: float,
        current_lr: float,
        epoch: int,
        total_epochs: int,
        scheduler_name: Optional[str],
        decay_rate: Optional[float],
        decay_steps: Optional[int],
    ) -> float:
        scheduler = None if scheduler_name is None else str(scheduler_name).lower()
        if scheduler in {None, "none"}:
            return current_lr
        if scheduler not in SUPPORTED_LR_SCHEDULERS:
            raise ConfigurationTamperingError("secure_config.yaml:neural_network.lr_scheduler", f"Unsupported scheduler: {scheduler}")
        if decay_rate is None:
            return current_lr
        if scheduler == "step":
            if decay_steps and decay_steps > 0 and (epoch + 1) % decay_steps == 0:
                return current_lr * decay_rate
            return current_lr
        if scheduler == "exponential":
            effective_steps = decay_steps if decay_steps and decay_steps > 0 else 1
            return initial_lr * (decay_rate ** (epoch / effective_steps))
        if scheduler == "cosine_annealing":
            eta_min = decay_rate
            t_max = decay_steps if decay_steps and decay_steps > 0 else total_epochs
            t_cur = epoch % max(1, t_max)
            return eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + math.cos(math.pi * t_cur / max(1, t_max)))
        return current_lr

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
        learning_rate = coerce_float(initial_learning_rate, 0.001, minimum=coerce_float(self.nn_config.get("learning_rate_floor", 1e-8), 1e-8), maximum=coerce_float(self.nn_config.get("learning_rate_ceiling", 1.0), 1.0))
        effective_batch_size = coerce_int(batch_size, 1, minimum=1, maximum=coerce_int(self.nn_config.get("max_batch_size", 4096), 4096, minimum=1))
        data_list = self._normalize_dataset(training_data, context="training")
        if not data_list:
            raise NeuralNetworkDataError("Training data is empty.", {"operation": "train"})
        validation_list = self._normalize_dataset(validation_data, context="validation") if validation_data is not None else None
        if save_best_model_path and coerce_bool(self.nn_config.get("save_best_requires_validation"), False) and not validation_list:
            raise NeuralNetworkDataError("save_best_model_path requires validation data by policy.", {"save_best_model_path": save_best_model_path})

        opt_kwargs = {
            "momentum_coefficient": momentum_coefficient if momentum_coefficient is not None else self.nn_config.get("momentum_coefficient", 0.9),
            "weight_decay_lambda": weight_decay_lambda if weight_decay_lambda is not None else self.nn_config.get("weight_decay_lambda", 0.0),
            "gradient_clip_value": gradient_clip_value if gradient_clip_value is not None else self.nn_config.get("gradient_clip_value"),
            "adagrad_epsilon": adagrad_epsilon if adagrad_epsilon is not None else self.nn_config.get("adagrad_epsilon", 1e-8),
            "adam_beta1": adam_beta1 if adam_beta1 is not None else self.adam_beta1,
            "adam_beta2": adam_beta2 if adam_beta2 is not None else self.adam_beta2,
            "adam_epsilon": adam_epsilon_opt if adam_epsilon_opt is not None else self.adam_epsilon,
        }
        print_every_n_epochs = print_every_n_epochs or max(1, coerce_int(get_nested(self.nn_config, "telemetry.log_every_n_epochs", 10), 10, minimum=1))
        current_lr = learning_rate
        best_val_loss = float("inf")
        epochs_no_improve = 0
        run = TrainingRunSummary(run_id=generate_identifier("nnrun"), started_at=utc_iso(), epochs_requested=epochs)
        self.training_history.clear()
        self._set_training_mode(True)
        if self.optimizer_name == "adam":
            self.adam_global_timestep = 0
        try:
            for epoch in range(epochs):
                if coerce_bool(self.nn_config.get("shuffle_training_data", True), True):
                    self.rng.shuffle(data_list)
                total_loss = 0.0
                samples_processed = 0
                for batch_idx in range(0, len(data_list), effective_batch_size):
                    batch_data = data_list[batch_idx:batch_idx + effective_batch_size]
                    if not batch_data:
                        continue
                    if self.optimizer_name == "adam":
                        self.adam_global_timestep += 1
                    batch_raw_activations = self._get_batch_raw_activations_for_bn(batch_data) if any(layer.use_batch_norm for layer in self.layers) else None
                    for features, targets in batch_data:
                        outputs = self.feed_forward_sample(features, batch_raw_activations)
                        loss_val, output_signals = self._calculate_loss_and_output_deltas(targets, outputs)
                        total_loss += loss_val
                        samples_processed += 1
                        self._backpropagate(features, output_signals, current_lr, **opt_kwargs)
                avg_loss = total_loss / max(1, samples_processed)
                val_loss: Optional[float] = None
                val_accuracy: Optional[float] = None
                if validation_list:
                    metrics = self.evaluate(validation_list, batch_size=effective_batch_size)
                    val_loss = metrics.get("loss")
                    val_accuracy = metrics.get("accuracy")
                    if val_loss is not None and val_loss < best_val_loss - early_stopping_min_delta:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        if save_best_model_path:
                            self.save_model(save_best_model_path)
                    else:
                        epochs_no_improve += 1
                record = TrainingEpochRecord(
                    epoch=epoch + 1,
                    loss=avg_loss,
                    learning_rate=current_lr,
                    samples=samples_processed,
                    validation_loss=val_loss,
                    validation_accuracy=val_accuracy,
                )
                self.training_history.append(record)
                limit = coerce_int(get_nested(self.nn_config, "telemetry.history_limit", 500), 500, minimum=1)
                if len(self.training_history) > limit:
                    self.training_history = self.training_history[-limit:]
                if verbose and (epoch + 1) % print_every_n_epochs == 0:
                    payload = {
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "learning_rate": current_lr,
                        "validation_loss": val_loss,
                        "validation_accuracy": val_accuracy,
                    }
                    logger.info("Neural network training telemetry: %s", safe_log_payload("nn_epoch", payload))
                if early_stopping_patience and validation_list and epochs_no_improve >= early_stopping_patience:
                    record.stopped = True
                    run.early_stopped = True
                    break
                current_lr = self._apply_lr_schedule(learning_rate, current_lr, epoch, epochs, lr_scheduler_name, lr_decay_rate, lr_decay_steps)
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
            raise wrap_security_exception(
                exc,
                operation="train",
                error_type=SecurityErrorType.UNSAFE_MODEL_STATE,
                message="Unexpected neural-network training failure.", # pyright: ignore[reportCallIssue]
                context={"model_id": self.model_id, "epochs_requested": epochs},
                component="neural_network.train",
                severity=SecuritySeverity.CRITICAL,
            ) from exc
        finally:
            self._set_training_mode(False)

    def evaluate(
        self,
        test_data: Union[List[Tuple[List[float], List[float]]], Tuple[np.ndarray, np.ndarray], np.ndarray],
        batch_size: Optional[int] = 32,
    ) -> Dict[str, float]:
        data_list = self._normalize_dataset(test_data, context="evaluation")
        if not data_list:
            return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        self._set_training_mode(False)
        total_loss = 0.0
        tp = fp = tn = fn = 0
        multiclass_correct = 0
        for features, targets in data_list:
            outputs = self.feed_forward_sample(features, None)
            loss_val, _ = self._calculate_loss_and_output_deltas(targets, outputs)
            total_loss += loss_val
            if self.problem_type == "binary_classification":
                pred_class = 1 if outputs[0] >= self.binary_threshold else 0
                true_class = int(round(targets[0]))
                if pred_class == 1 and true_class == 1:
                    tp += 1
                elif pred_class == 1 and true_class == 0:
                    fp += 1
                elif pred_class == 0 and true_class == 0:
                    tn += 1
                elif pred_class == 0 and true_class == 1:
                    fn += 1
            elif self.problem_type == "multiclass_classification":
                pred_idx = outputs.index(max(outputs))
                true_idx = targets.index(max(targets))
                if pred_idx == true_idx:
                    multiclass_correct += 1
        count = len(data_list)
        metrics: Dict[str, float] = {"loss": total_loss / count}
        if self.problem_type == "binary_classification":
            accuracy = (tp + tn) / count
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            metrics.update({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn)})
        elif self.problem_type == "multiclass_classification":
            metrics.update({"accuracy": multiclass_correct / count})
        return metrics

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
            raise ModelTamperingDetectedError(
                model_name=self.model_id,
                detection_method="layer_count_validation",
                expected_hash=str(len(self.layers)),
                actual_hash=str(len(network_params)),
            )
        for layer, state in zip(self.layers, network_params):
            layer.load_state(state, include_optimizer_state=include_optimizer_state)
        self.updated_at = utc_iso()

    def model_card(self, *, redacted: bool = False) -> Dict[str, Any]:
        card = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "problem_type": self.problem_type,
            "loss_function_name": self.loss_function_name,
            "optimizer_name": self.optimizer_name,
            "num_inputs": self.num_inputs,
            "num_layers": len(self.layers),
            "layers": [
                {
                    "index": idx,
                    "neurons": layer.num_neurons,
                    "activation": layer.activation_name,
                    "dropout": layer.dropout_rate,
                    "batch_norm": layer.use_batch_norm,
                }
                for idx, layer in enumerate(self.layers)
            ],
            "last_training_summary": self.last_training_summary.to_dict() if self.last_training_summary else None,
            "fingerprint": self.model_fingerprint(),
        }
        return sanitize_for_logging(card) if redacted else card

    def model_fingerprint(self) -> str:
        payload = {
            "num_inputs": self.num_inputs,
            "problem_type": self.problem_type,
            "loss_function_name": self.loss_function_name,
            "optimizer_name": self.optimizer_name,
            "weights": self.get_weights_biases(include_optimizer_state=False),
        }
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
        payload_without_signature = dict(payload)
        payload_without_signature.pop("signature", None)
        return hash_text(
            stable_json(payload_without_signature),
            algorithm=get_nested(self.nn_config, "persistence.signature_hash_algorithm", "sha256"),
            salt=get_nested(self.nn_config, "persistence.signature_salt", ""),
        )

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
            default_dir = Path(str(get_nested(self.nn_config, "persistence.model_dir", "src/agents/safety/models")))
            path = default_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = self._serialize_model_state()
        try:
            if path.exists() and coerce_bool(get_nested(self.nn_config, "persistence.write_backup", True), True):
                backup_path = path.with_suffix(path.suffix + f".{utc_iso().replace(':', '').replace('-', '')}.bak")
                shutil.copy2(path, backup_path)
            payload = json.dumps(to_jsonable(state), indent=2, ensure_ascii=False, sort_keys=True)
            if coerce_bool(get_nested(self.nn_config, "persistence.atomic_write", True), True):
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                tmp_path.write_text(payload, encoding="utf-8")
                tmp_path.replace(path)
            else:
                path.write_text(payload, encoding="utf-8")
            logger.info("Cyber-security model saved: %s", safe_log_payload("model_saved", {"path": str(path), "fingerprint": state.get("model_fingerprint")}))
            return True
        except SecurityError:
            raise
        except Exception as exc:
            raise NeuralNetworkPersistenceError("Failed to save neural-network model.", {"path": str(path)}, cause=exc) from exc

    @classmethod
    def load_model(cls, filepath: str, custom_config_override: Optional[Dict[str, Any]] = None) -> "NeuralNetwork":
        base_config = get_config_section("neural_network") or {}
        if custom_config_override:
            base_config = deep_merge(base_config, custom_config_override)
        path = Path(filepath)
        if not path.is_absolute():
            default_dir = Path(str(get_nested(base_config, "persistence.model_dir", "src/agents/safety/models")))
            path = default_dir / path
        max_bytes = coerce_int(get_nested(base_config, "persistence.max_model_file_bytes", 10485760), 10485760, minimum=1024)
        try:
            if not path.exists():
                raise NeuralNetworkPersistenceError("Model file does not exist.", {"path": str(path)})
            if path.stat().st_size > max_bytes:
                raise ResourceExhaustionError("model_file_bytes", path.stat().st_size, max_bytes, source_identifier=str(path))
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
                network.last_training_summary = TrainingRunSummary(**state["last_training_summary"])
            logger.info("Cyber-security model loaded: %s", safe_log_payload("model_loaded", {"path": str(path), "model_id": network.model_id}))
            return network
        except SecurityError:
            raise
        except Exception as exc:
            raise NeuralNetworkPersistenceError("Failed to load neural-network model.", {"path": str(path)}, cause=exc) from exc

    def __repr__(self) -> str:
        return f"NeuralNetwork(id={self.model_id}, inputs={self.num_inputs}, layers={len(self.layers)}, problem={self.problem_type})"


def _generate_security_data(num_samples: int, rng: Optional[random.Random] = None) -> List[Tuple[List[float], List[float]]]:
    rng = rng or random.Random(1337)
    data: List[Tuple[List[float], List[float]]] = []
    for _ in range(num_samples):
        malicious = rng.random() < 0.35
        if malicious:
            failed_logins = rng.uniform(0.55, 1.0)
            traffic_anomaly = rng.uniform(0.35, 1.0)
            unusual_port = 1.0 if rng.random() < 0.75 else 0.0
            session_duration = rng.uniform(0.0, 0.25) if rng.random() < 0.6 else rng.uniform(0.75, 1.0)
            target = [1.0]
        else:
            failed_logins = rng.uniform(0.0, 0.3)
            traffic_anomaly = rng.uniform(0.0, 0.55)
            unusual_port = 1.0 if rng.random() < 0.08 else 0.0
            session_duration = rng.uniform(0.15, 0.75)
            target = [0.0]
        data.append(([failed_logins, traffic_anomaly, unusual_port, session_duration], target))
    return data


if __name__ == "__main__":
    print("\n=== Running Neural Network ===\n")
    printer.status("TEST", "Neural Network initialized", "info")

    test_config = {
        "deterministic_seed": 2026,
        "max_epochs": 50,
        "persistence": {
            "model_dir": "/tmp/safety_neural_network_models",
            "require_signature_on_load": True,
            "write_backup": False,
        },
        "telemetry": {"history_limit": 100, "log_every_n_epochs": 2},
    }

    rng = random.Random(2026)
    dataset = _generate_security_data(60, rng)
    rng.shuffle(dataset)
    train_data = dataset[:40]
    validation_data = dataset[40:50]
    test_data = dataset[50:]

    layer_config = [
        {"neurons": 6, "activation": "relu", "init": "he_normal", "dropout": 0.0, "batch_norm": False},
        {"neurons": 3, "activation": "relu", "init": "he_normal", "dropout": 0.0, "batch_norm": False},
        {"neurons": 1, "activation": "sigmoid"},
    ]

    network = NeuralNetwork(
        num_inputs=4,
        layer_config=layer_config,
        loss_function_name="cross_entropy",
        optimizer_name="adam",
        problem_type="binary_classification",
        config=test_config,
    )
    assert len(network.layers) == 3
    printer.status("TEST", f"Model card: {stable_json(network.model_card(redacted=True))}", "info")

    summary = network.train(
        train_data,
        epochs=3,
        initial_learning_rate=0.01,
        batch_size=8,
        validation_data=validation_data,
        early_stopping_patience=4,
        print_every_n_epochs=1,
        verbose=True,
    )
    assert summary.epochs_completed >= 1
    assert summary.model_fingerprint
    printer.status("TEST", f"Training summary: {stable_json(summary.to_dict())}", "info")

    metrics = network.evaluate(test_data, batch_size=16)
    assert "loss" in metrics and metrics["loss"] >= 0.0
    assert 0.0 <= metrics.get("accuracy", 0.0) <= 1.0
    printer.status("TEST", f"Evaluation metrics: {stable_json(metrics)}", "info")

    benign_event = [0.05, 0.20, 0.0, 0.40]
    malicious_event = [0.90, 0.80, 1.0, 0.10]
    benign_proba = network.predict_proba(benign_event)
    malicious_proba = network.predict_proba(malicious_event)
    assert len(benign_proba) == 1 and len(malicious_proba) == 1
    assert 0.0 <= benign_proba[0] <= 1.0
    assert 0.0 <= malicious_proba[0] <= 1.0
    printer.status("TEST", f"Benign probability: {benign_proba[0]:.4f}", "info")
    printer.status("TEST", f"Malicious probability: {malicious_proba[0]:.4f}", "info")

    model_path = "self_test_intrusion_model.json"
    assert network.save_model(model_path) is True
    loaded = NeuralNetwork.load_model(model_path, custom_config_override=test_config)
    loaded_metrics = loaded.evaluate(test_data, batch_size=16)
    assert "loss" in loaded_metrics
    assert loaded.model_fingerprint() == network.model_fingerprint()
    printer.status("TEST", f"Loaded metrics: {stable_json(loaded_metrics)}", "info")

    try:
        network.predict([1.0, 2.0])
    except SecurityError as exc:
        safe_error = exception_to_safe_dict(exc, operation="dimension_mismatch_self_test", component="neural_network")
        assert safe_error.get("message")
        printer.status("TEST", f"Handled unsafe input: {safe_error.get('error_id', 'no-id')}", "info")
    else:
        raise AssertionError("Expected dimension mismatch to raise SecurityError")

    tampered_path = Path(str(get_nested(test_config, "persistence.model_dir"))) / "tampered_model.json"
    original_path = Path(str(get_nested(test_config, "persistence.model_dir"))) / model_path
    payload = parse_json_object(original_path.read_text(encoding="utf-8"), context="tamper_test_payload")
    payload["model_id"] = "tampered-id"
    tampered_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        NeuralNetwork.load_model(str(tampered_path), custom_config_override=test_config)
    except SecurityError as exc:
        assert exc.blocked or exc.is_critical
        printer.status("TEST", "Tamper detection raised SecurityError", "info")
    else:
        raise AssertionError("Expected tampered model load to raise SecurityError")

    print("\n=== Test ran successfully ===\n")
