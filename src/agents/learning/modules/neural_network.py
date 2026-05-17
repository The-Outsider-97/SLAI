"""Production-ready manual neural network with explicit backpropagation.

The module intentionally keeps the original SLAI surface area intact:
``NeuralNetwork`` remains a lightweight feed-forward network with manual
backpropagation, custom optimizers, explicit weight accessors, and checkpoint
helpers used by DQN, RSI, and the learning factory. Configuration continues to
come from ``learning_config.yaml`` through the existing config loader.
"""

from __future__ import annotations

import math
import time
import torch # pyright: ignore[reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingImports]

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ...base.modules.activation_engine import (Activation, Linear, ReLU,
                                             Sigmoid, Softmax, Tanh,
                                             get_activation)
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.learning_error import *
from ..utils.learning_calculations import *
from ..utils.learning_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Neural Network")
printer = PrettyPrinter()

DEBUG_LEVEL = 10
EPSILON = 1e-12
TensorLike = Union[torch.Tensor, Sequence[float], Sequence[Sequence[float]]]

__all__ = [
    "TensorLike", "Loss", "MSELoss", "HuberLoss", "CrossEntropyLoss",
    "Optimizer", "SGD", "SGDMomentum", "RMSProp", "Adam", "AdamW",
    "NeuralNetwork",
]


def _ensure_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if torch.isnan(tensor).any():
        raise NaNException(f"NaN detected in {name}", location=name)
    if torch.isinf(tensor).any():
        raise InfException(f"Inf detected in {name}", location=name)


def _activation_backward(activation: Activation, z: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    return activation.backward(z, grad_output)


def _clone_tensor_list(values: Optional[List[torch.Tensor]]) -> Optional[List[torch.Tensor]]:
    return None if values is None else [value.detach().clone() for value in values]


class Loss:
    """Base class for manual losses."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(self, y_pred: torch.Tensor, y_true: torch.Tensor, batch_size: int) -> torch.Tensor:
        raise NotImplementedError


class MSELoss(Loss):
    """Mean squared error over all prediction elements."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred: torch.Tensor, y_true: torch.Tensor, batch_size: int) -> torch.Tensor:
        denom = max(int(y_pred.numel()), 1)
        return (2.0 / denom) * (y_pred - y_true)


class HuberLoss(Loss):
    """Huber/Smooth-L1 style loss for robust regression."""

    def __init__(self, delta: float = 1.0):
        validate_positive(delta, "huber_delta", strict=True)
        self.delta = float(delta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_pred - y_true
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.as_tensor(self.delta, device=y_pred.device, dtype=y_pred.dtype))
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic.pow(2) + self.delta * linear)

    def backward(self, y_pred: torch.Tensor, y_true: torch.Tensor, batch_size: int) -> torch.Tensor:
        error = y_pred - y_true
        denom = max(int(y_pred.numel()), 1)
        return torch.clamp(error, -self.delta, self.delta) / denom


class CrossEntropyLoss(Loss):
    """Cross entropy over raw logits and class-index targets."""

    def __init__(self):
        self.softmax = Softmax(dim=-1)
        self._cache: Dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, y_true_indices: torch.Tensor) -> torch.Tensor: # type: ignore
        if logits.ndim != 2:
            raise ObservationShapeError(expected_shape=(None, "classes"), actual_shape=tuple(logits.shape))
        if y_true_indices.ndim != 1:
            raise ObservationShapeError(expected_shape=(logits.shape[0],), actual_shape=tuple(y_true_indices.shape))
        if logits.shape[0] != y_true_indices.shape[0]:
            raise ObservationShapeError(expected_shape=(logits.shape[0],), actual_shape=tuple(y_true_indices.shape))
        _ensure_finite_tensor(logits, "cross_entropy_logits")

        y_true_indices = y_true_indices.to(device=logits.device, dtype=torch.long)
        if torch.any(y_true_indices < 0) or torch.any(y_true_indices >= logits.shape[1]):
            raise InvalidConfigError(
                "Class target out of range",
                config_key="target",
                context={"num_classes": int(logits.shape[1])},
            )
        logsumexp = torch.logsumexp(logits, dim=-1)
        correct_class_logits = logits.gather(1, y_true_indices.unsqueeze(1)).squeeze(1)
        loss = torch.mean(logsumexp - correct_class_logits)
        self._cache["probs"] = self.softmax.forward(logits)
        _ensure_finite_tensor(loss, "cross_entropy_loss")
        return loss

    def backward(self, logits: torch.Tensor, y_true_indices: torch.Tensor, batch_size: int) -> torch.Tensor: # type: ignore
        batch_size = max(int(batch_size), 1)
        probs = self._cache.get("probs")
        if probs is None or probs.shape != logits.shape or probs.device != logits.device:
            probs = self.softmax.forward(logits)
        y_true_indices = y_true_indices.to(device=logits.device, dtype=torch.long)
        y_true_one_hot = torch.zeros_like(probs)
        y_true_one_hot.scatter_(1, y_true_indices.unsqueeze(1), 1.0)
        return (probs - y_true_one_hot) / batch_size


class Optimizer:
    """Base class for the manual optimizers used by ``NeuralNetwork``."""

    def __init__(self, learning_rate: float):
        validate_positive(learning_rate, "learning_rate", strict=True)
        self.learning_rate = float(learning_rate)

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "optimizer_type": type(self).__name__}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.learning_rate = float(state_dict.get("learning_rate", self.learning_rate))


class SGD(Optimizer):
    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        for i in range(len(params_Ws)):
            params_Ws[i].sub_(self.learning_rate * grads_dWs[i])
            params_bs[i].sub_(self.learning_rate * grads_dBs[i])


class SGDMomentum(Optimizer):
    def __init__(self, learning_rate: float, beta: float = 0.9):
        super().__init__(learning_rate)
        validate_in_range(beta, "momentum_beta", 0.0, 1.0, inclusive_high=False)
        self.beta = float(beta)
        self.v_Ws: Optional[List[torch.Tensor]] = None
        self.v_bs: Optional[List[torch.Tensor]] = None

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        if self.v_Ws is None or self.v_bs is None:
            self.v_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.v_bs = [torch.zeros_like(b) for b in params_bs]
        for i in range(len(params_Ws)):
            self.v_Ws[i].mul_(self.beta).add_(grads_dWs[i])
            self.v_bs[i].mul_(self.beta).add_(grads_dBs[i])
            params_Ws[i].sub_(self.learning_rate * self.v_Ws[i])
            params_bs[i].sub_(self.learning_rate * self.v_bs[i])

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "beta": self.beta, "v_Ws": _clone_tensor_list(self.v_Ws), "v_bs": _clone_tensor_list(self.v_bs)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.beta = float(state_dict.get("beta", self.beta))
        self.v_Ws = state_dict.get("v_Ws")
        self.v_bs = state_dict.get("v_bs")


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float, alpha: float = 0.99, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        validate_in_range(alpha, "rmsprop_alpha", 0.0, 1.0, inclusive_high=False)
        validate_positive(epsilon, "rmsprop_epsilon", strict=True)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.avg_Ws: Optional[List[torch.Tensor]] = None
        self.avg_bs: Optional[List[torch.Tensor]] = None

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        if self.avg_Ws is None or self.avg_bs is None:
            self.avg_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.avg_bs = [torch.zeros_like(b) for b in params_bs]
        for i in range(len(params_Ws)):
            self.avg_Ws[i].mul_(self.alpha).addcmul_(grads_dWs[i], grads_dWs[i], value=1.0 - self.alpha)
            self.avg_bs[i].mul_(self.alpha).addcmul_(grads_dBs[i], grads_dBs[i], value=1.0 - self.alpha)
            params_Ws[i].sub_(self.learning_rate * grads_dWs[i] / (torch.sqrt(self.avg_Ws[i]) + self.epsilon))
            params_bs[i].sub_(self.learning_rate * grads_dBs[i] / (torch.sqrt(self.avg_bs[i]) + self.epsilon))

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "alpha": self.alpha, "epsilon": self.epsilon, "avg_Ws": _clone_tensor_list(self.avg_Ws), "avg_bs": _clone_tensor_list(self.avg_bs)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.alpha = float(state_dict.get("alpha", self.alpha))
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
        self.avg_Ws = state_dict.get("avg_Ws")
        self.avg_bs = state_dict.get("avg_bs")


class Adam(Optimizer):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        validate_in_range(beta1, "adam_beta1", 0.0, 1.0, inclusive_high=False)
        validate_in_range(beta2, "adam_beta2", 0.0, 1.0, inclusive_high=False)
        validate_positive(epsilon, "adam_epsilon", strict=True)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.m_Ws: Optional[List[torch.Tensor]] = None
        self.v_Ws: Optional[List[torch.Tensor]] = None
        self.m_bs: Optional[List[torch.Tensor]] = None
        self.v_bs: Optional[List[torch.Tensor]] = None
        self.t = 0

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        self.t += 1
        if self.m_Ws is None or self.v_Ws is None or self.m_bs is None or self.v_bs is None:
            self.m_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.v_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.m_bs = [torch.zeros_like(b) for b in params_bs]
            self.v_bs = [torch.zeros_like(b) for b in params_bs]
        for i in range(len(params_Ws)):
            self.m_Ws[i].mul_(self.beta1).add_(grads_dWs[i], alpha=1.0 - self.beta1)
            self.v_Ws[i].mul_(self.beta2).addcmul_(grads_dWs[i], grads_dWs[i], value=1.0 - self.beta2)
            m_hat_W = self.m_Ws[i] / (1.0 - self.beta1 ** self.t)
            v_hat_W = self.v_Ws[i] / (1.0 - self.beta2 ** self.t)
            params_Ws[i].sub_(self.learning_rate * m_hat_W / (torch.sqrt(v_hat_W) + self.epsilon))
            self.m_bs[i].mul_(self.beta1).add_(grads_dBs[i], alpha=1.0 - self.beta1)
            self.v_bs[i].mul_(self.beta2).addcmul_(grads_dBs[i], grads_dBs[i], value=1.0 - self.beta2)
            m_hat_b = self.m_bs[i] / (1.0 - self.beta1 ** self.t)
            v_hat_b = self.v_bs[i] / (1.0 - self.beta2 ** self.t)
            params_bs[i].sub_(self.learning_rate * m_hat_b / (torch.sqrt(v_hat_b) + self.epsilon))

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(), "beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon,
            "m_Ws": _clone_tensor_list(self.m_Ws), "v_Ws": _clone_tensor_list(self.v_Ws),
            "m_bs": _clone_tensor_list(self.m_bs), "v_bs": _clone_tensor_list(self.v_bs), "t": self.t,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.beta1 = float(state_dict.get("beta1", self.beta1))
        self.beta2 = float(state_dict.get("beta2", self.beta2))
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
        self.m_Ws = state_dict.get("m_Ws")
        self.v_Ws = state_dict.get("v_Ws")
        self.m_bs = state_dict.get("m_bs")
        self.v_bs = state_dict.get("v_bs")
        self.t = int(state_dict.get("t", self.t))


class AdamW(Adam):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
        validate_non_negative(weight_decay, "adamw_weight_decay")
        self.weight_decay = float(weight_decay)

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        if self.weight_decay > 0.0:
            for i in range(len(params_Ws)):
                params_Ws[i].mul_(1.0 - self.learning_rate * self.weight_decay)
        super().step(params_Ws, params_bs, grads_dWs, grads_dBs)

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "weight_decay": self.weight_decay, "optimizer_type": type(self).__name__}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.weight_decay = float(state_dict.get("weight_decay", self.weight_decay))


class NeuralNetwork(nn.Module):
    """Configurable feed-forward neural network with explicit manual backpropagation."""

    SUPPORTED_WEIGHT_INITS = {"auto", "kaiming", "xavier", "orthogonal", "normal", "zeros"}

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        validate_positive(input_dim, "input_dim", strict=True)
        validate_positive(output_dim, "output_dim", strict=True)

        self.config = load_global_config()
        self.nn_config = get_config_section("neural_network") or {}
        if config:
            if not isinstance(config, Mapping):
                raise InvalidConfigError("NeuralNetwork config override must be a mapping", received_value=type(config).__name__)
            self.nn_config.update(dict(config))

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.device_override = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.layer_dims = self._resolve_layer_dims()
        self.num_layers = len(self.layer_dims) - 1

        self.hidden_activation_name = str(self.nn_config.get("hidden_activation", "relu")).lower()
        self.output_activation_name = str(self.nn_config.get("output_activation", "linear")).lower()
        self.loss_function_name = str(self.nn_config.get("loss_function", "mse")).lower()
        self.optimizer_name = str(self.nn_config.get("optimizer", "adam")).lower()
        self.learning_rate = coerce_float(self.nn_config.get("learning_rate", 0.001), minimum=1e-12)
        self.l1_lambda = coerce_float(self.nn_config.get("l1_lambda", 0.0), minimum=0.0)
        self.l2_lambda = coerce_float(self.nn_config.get("l2_lambda", 0.0), minimum=0.0)
        self.max_grad_norm = self.nn_config.get("gradient_clip_norm", 5.0)
        self.max_grad_norm = None if self.max_grad_norm is None else coerce_float(self.max_grad_norm, minimum=1e-12)
        self.gradient_explosion_threshold = coerce_float(self.nn_config.get("gradient_explosion_threshold", 1e4), minimum=1e-12)
        self.weight_init = str(self.nn_config.get("weight_init", "auto")).lower()
        self.use_output_activation_for_loss = coerce_bool(self.nn_config.get("use_output_activation_for_loss", False))
        self.nan_guard = coerce_bool(self.nn_config.get("nan_guard", True), default=True)

        if self.weight_init not in self.SUPPORTED_WEIGHT_INITS:
            raise InvalidConfigError("Unsupported neural network weight_init", config_key="neural_network.weight_init", received_value=self.weight_init)

        self._init_activation_functions()
        self._initialize_weights()
        self._init_loss_function()
        self._init_optimizer()

        self._cache: Dict[str, Any] = {}
        self.dWs: List[torch.Tensor] = []
        self.dBs: List[torch.Tensor] = []
        self.training_steps = 0
        self.last_metrics: Dict[str, float] = {}
        self.loss_stats = RunningStats()

        logger.info(
            "NeuralNetwork initialised | dims=%s hidden_act=%s output_act=%s loss=%s optimizer=%s lr=%s",
            self.layer_dims, self.hidden_activation_name, self.output_activation_name,
            self.loss_function_name, self.optimizer_name, self.learning_rate,
        )

    def _resolve_layer_dims(self) -> List[int]:
        default_dims = [self.input_dim, 128, 64, self.output_dim]
        raw_dims = self.nn_config.get("layer_dims", default_dims)
        if isinstance(raw_dims, int):
            raw_dims = [self.input_dim, int(raw_dims), self.output_dim]
        validate_non_empty_sequence(raw_dims, "neural_network.layer_dims")
        dims = [int(dim) for dim in list(raw_dims)]
        if len(dims) < 2:
            raise InvalidConfigError("layer_dims must define at least input and output dimensions", config_key="neural_network.layer_dims")
        dims[0] = self.input_dim
        dims[-1] = self.output_dim
        for idx, dim in enumerate(dims):
            validate_positive(dim, f"neural_network.layer_dims[{idx}]", strict=True)
        return dims

    def _str_to_activation(self, name_str: str) -> Activation:
        try:
            activation = get_activation(name_str)
        except ValueError as exc:
            raise InvalidConfigError("Unknown activation function", config_key="neural_network.activation", received_value=name_str, cause=exc) from exc
        if not isinstance(activation, Activation):
            raise InvalidConfigError("Activation must implement the Activation interface", received_value=type(activation).__name__)
        return activation

    def _init_activation_functions(self) -> None:
        self.hidden_activations: List[Activation] = [
            self._str_to_activation(self.hidden_activation_name) for _ in range(max(self.num_layers - 1, 0))
        ]
        self.output_activation = self._str_to_activation(self.output_activation_name)

    def _init_loss_function(self) -> None:
        if self.loss_function_name in {"mse", "mean_squared_error"}:
            self.loss_fn = MSELoss()
        elif self.loss_function_name in {"huber", "smooth_l1"}:
            self.loss_fn = HuberLoss(delta=coerce_float(self.nn_config.get("huber_delta", 1.0), minimum=1e-12))
        elif self.loss_function_name in {"cross_entropy", "ce"}:
            if not isinstance(self.output_activation, Linear):
                logger.warning("CrossEntropyLoss uses logits from the final linear transform; output_activation=%s remains available for prediction.", self.output_activation_name)
            self.loss_fn = CrossEntropyLoss()
        else:
            raise InvalidConfigError("Unknown loss function", config_key="neural_network.loss_function", received_value=self.loss_function_name)

    def _init_optimizer(self) -> None:
        if self.optimizer_name == "sgd":
            self.optimizer = SGD(self.learning_rate)
        elif self.optimizer_name in {"momentum", "sgd_momentum"}:
            self.optimizer = SGDMomentum(self.learning_rate, beta=coerce_float(self.nn_config.get("momentum_beta", 0.9), minimum=0.0, maximum=0.999999))
        elif self.optimizer_name == "rmsprop":
            self.optimizer = RMSProp(
                self.learning_rate,
                alpha=coerce_float(self.nn_config.get("rmsprop_alpha", 0.99), minimum=0.0, maximum=0.999999),
                epsilon=coerce_float(self.nn_config.get("rmsprop_epsilon", 1e-8), minimum=1e-16),
            )
        elif self.optimizer_name == "adam":
            self.optimizer = Adam(
                self.learning_rate,
                beta1=coerce_float(self.nn_config.get("adam_beta1", 0.9), minimum=0.0, maximum=0.999999),
                beta2=coerce_float(self.nn_config.get("adam_beta2", 0.999), minimum=0.0, maximum=0.999999),
                epsilon=coerce_float(self.nn_config.get("adam_epsilon", 1e-8), minimum=1e-16),
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = AdamW(
                self.learning_rate,
                beta1=coerce_float(self.nn_config.get("adam_beta1", 0.9), minimum=0.0, maximum=0.999999),
                beta2=coerce_float(self.nn_config.get("adam_beta2", 0.999), minimum=0.0, maximum=0.999999),
                epsilon=coerce_float(self.nn_config.get("adam_epsilon", 1e-8), minimum=1e-16),
                weight_decay=coerce_float(self.nn_config.get("adamw_weight_decay", self.l2_lambda), minimum=0.0),
            )
        else:
            raise InvalidConfigError("Unknown optimizer", config_key="neural_network.optimizer", received_value=self.optimizer_name)

    @property
    def device(self) -> torch.device:
        return self.Ws[0].device

    def _initialize_weights(self) -> None:
        target_device = self.device_override or torch.device("cpu")
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        for i in range(self.num_layers):
            fan_in, fan_out = self.layer_dims[i], self.layer_dims[i + 1]
            activation_name = self.hidden_activation_name if i < self.num_layers - 1 else self.output_activation_name
            weight = torch.empty((fan_in, fan_out), device=target_device, dtype=self.dtype)
            if self.weight_init == "zeros":
                nn.init.zeros_(weight)
            elif self.weight_init == "normal":
                nn.init.normal_(weight, mean=0.0, std=0.02)
            elif self.weight_init == "orthogonal":
                nn.init.orthogonal_(weight)
            elif self.weight_init == "kaiming" or (self.weight_init == "auto" and activation_name in {"relu", "leaky_relu"}):
                nn.init.kaiming_uniform_(weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(weight)
            bias = torch.zeros(fan_out, device=target_device, dtype=self.dtype)
            self.Ws.append(nn.Parameter(weight, requires_grad=False))
            self.bs.append(nn.Parameter(bias, requires_grad=False))

    def _prepare_input(self, X: TensorLike) -> Tuple[torch.Tensor, bool]:
        # Convert input to tensor on the correct device/dtype
        if torch.is_tensor(X):
            tensor = X
        else:
            tensor = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        tensor = tensor.to(device=self.device, dtype=self.dtype) # type: ignore
    
        squeezed = False
        if tensor.ndim == 0:
            raise ObservationShapeError(expected_shape=(self.input_dim,), actual_shape=tuple(tensor.shape))
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        elif tensor.ndim > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.ndim != 2 or tensor.shape[1] != self.input_dim:
            raise ObservationShapeError(expected_shape=(None, self.input_dim), actual_shape=tuple(tensor.shape))
        if self.nan_guard:
            _ensure_finite_tensor(tensor, "network_input")
        return tensor, squeezed

    def _prepare_target(self, y_true: TensorLike, batch_size: int) -> torch.Tensor:
        if torch.is_tensor(y_true):
            y = y_true
        else:
            y = torch.as_tensor(y_true, device=self.device)
        y = y.to(device=self.device) # type: ignore
    
        if isinstance(self.loss_fn, CrossEntropyLoss):
            if y.ndim != 1:
                raise ObservationShapeError(expected_shape=(batch_size,), actual_shape=tuple(y.shape))
            if y.shape[0] != batch_size:
                raise ObservationShapeError(expected_shape=(batch_size,), actual_shape=tuple(y.shape))
            return y.to(dtype=torch.long)
        y = y.to(dtype=self.dtype)
        if y.ndim == 1 and self.output_dim == 1:
            y = y.unsqueeze(1)
        if y.ndim != 2 or y.shape != (batch_size, self.output_dim):
            raise ObservationShapeError(expected_shape=(batch_size, self.output_dim), actual_shape=tuple(y.shape))
        if self.nan_guard:
            _ensure_finite_tensor(y, "target")
        return y

    def forward(self, X: TensorLike) -> torch.Tensor:
        X, squeezed = self._prepare_input(X)
        self._cache = {"inputs": X, "layer_outputs": []}
        current_a = X
        for i in range(self.num_layers):
            z = current_a @ self.Ws[i] + self.bs[i]
            if self.nan_guard:
                _ensure_finite_tensor(z, f"layer_{i}_pre_activation")
            activation = self.hidden_activations[i] if i < self.num_layers - 1 else self.output_activation
            current_a = activation.forward(z)
            if self.nan_guard:
                _ensure_finite_tensor(current_a, f"layer_{i}_activation")
            self._cache["layer_outputs"].append({"z": z, "a": current_a})
        return current_a.squeeze(0) if squeezed else current_a # type: ignore

    def predict_logits(self, X: TensorLike) -> torch.Tensor:
        X, squeezed = self._prepare_input(X)
        current_a = X
        logits = None
        for i in range(self.num_layers):
            z = current_a @ self.Ws[i] + self.bs[i]
            logits = z
            current_a = self.hidden_activations[i].forward(z) if i < self.num_layers - 1 else self.output_activation.forward(z)
        if logits is None:
            raise TrainingError("Network has no layers")
        return logits.squeeze(0) if squeezed else logits

    def regularization_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), device=self.device, dtype=self.dtype)
        if self.l1_lambda > 0.0:
            penalty = penalty + self.l1_lambda * sum(W.abs().sum() for W in self.Ws)
        if self.l2_lambda > 0.0:
            penalty = penalty + 0.5 * self.l2_lambda * sum(W.pow(2).sum() for W in self.Ws)
        return penalty

    def compute_loss(self, y_pred_output: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if not self._cache.get("layer_outputs"):
            raise TrainingError("compute_loss requires a forward pass before loss computation")
        if isinstance(self.loss_fn, CrossEntropyLoss):
            final_layer_z = self._cache["layer_outputs"][-1]["z"]
            data_loss = self.loss_fn.forward(final_layer_z, y_true)
        else:
            data_loss = self.loss_fn.forward(y_pred_output, y_true)
        loss = data_loss + self.regularization_penalty()
        if self.nan_guard:
            _ensure_finite_tensor(loss, "loss")
        return loss

    def backward(self, y_true: torch.Tensor) -> None:
        if not self._cache.get("layer_outputs"):
            raise TrainingError("backward requires a forward pass first")
        m = max(int(y_true.shape[0] if y_true.ndim > 0 else 1), 1)
        self.dWs = [torch.zeros_like(W) for W in self.Ws]
        self.dBs = [torch.zeros_like(b) for b in self.bs]

        final_cache = self._cache["layer_outputs"][-1]
        if isinstance(self.loss_fn, CrossEntropyLoss):
            delta = self.loss_fn.backward(final_cache["z"], y_true, m)
        else:
            dL_daL = self.loss_fn.backward(final_cache["a"], y_true, m)
            delta = _activation_backward(self.output_activation, final_cache["z"], dL_daL)

        for i in reversed(range(self.num_layers)):
            a_prev = self._cache["inputs"] if i == 0 else self._cache["layer_outputs"][i - 1]["a"]
            self.dWs[i] = a_prev.T @ delta
            self.dBs[i] = torch.sum(delta, dim=0)
            if self.l2_lambda > 0.0:
                self.dWs[i] += self.l2_lambda * self.Ws[i]
            if self.l1_lambda > 0.0:
                self.dWs[i] += self.l1_lambda * torch.sign(self.Ws[i])
            if i > 0:
                da_prev = delta @ self.Ws[i].T
                prev_z = self._cache["layer_outputs"][i - 1]["z"]
                delta = _activation_backward(self.hidden_activations[i - 1], prev_z, da_prev)

    def _global_grad_norm(self) -> torch.Tensor:
        total = torch.zeros((), device=self.device, dtype=self.dtype)
        for grad in [*self.dWs, *self.dBs]:
            total = total + grad.pow(2).sum()
        return torch.sqrt(total)

    def _clip_gradients(self) -> float:
        grad_norm = self._global_grad_norm()
        if self.nan_guard:
            _ensure_finite_tensor(grad_norm, "gradient_norm")
        norm_value = float(grad_norm.detach().item())
        if norm_value > self.gradient_explosion_threshold:
            raise GradientExplosionError(norm=norm_value, threshold=self.gradient_explosion_threshold)
        if self.max_grad_norm is not None and norm_value > self.max_grad_norm:
            scale = self.max_grad_norm / (norm_value + EPSILON)
            for i in range(len(self.dWs)):
                self.dWs[i].mul_(scale)
                self.dBs[i].mul_(scale)
            norm_value = float(self._global_grad_norm().detach().item())
        return norm_value

    def update_parameters(self) -> None:
        if not self.dWs or not self.dBs:
            raise TrainingError("No gradients available. Call backward() before update_parameters().")
        with torch.no_grad():
            self.optimizer.step(self.Ws, self.bs, self.dWs, self.dBs)

    def train_step(self, X_batch: TensorLike, y_batch: TensorLike) -> float:
        step_start = time.perf_counter()
        X_prepared, _ = self._prepare_input(X_batch)
        if X_prepared.shape[0] <= 0:
            raise TrainingError("Batch size cannot be zero")
        y_prepared = self._prepare_target(y_batch, X_prepared.shape[0])

        y_pred_output = self.forward(X_prepared)
        loss = self.compute_loss(y_pred_output, y_prepared)
        self.backward(y_prepared)
        grad_norm = self._clip_gradients()
        for i, (dw, db) in enumerate(zip(self.dWs, self.dBs)):
            if self.nan_guard:
                _ensure_finite_tensor(dw, f"dW_{i}")
                _ensure_finite_tensor(db, f"dB_{i}")
        self.update_parameters()

        loss_value = float(loss.detach().item())
        self.training_steps += 1
        self.loss_stats.update(loss_value)
        self.last_metrics = {
            "loss": loss_value,
            "grad_norm": grad_norm,
            "step_time_seconds": time.perf_counter() - step_start,
            "training_steps": float(self.training_steps),
        }
        if logger.isEnabledFor(DEBUG_LEVEL):
            logger.debug("Train step metrics: %s", self.last_metrics)
        return loss_value

    def predict(self, X: TensorLike, return_probabilities: bool = False) -> torch.Tensor:
        if return_probabilities:
            logits = self.predict_logits(X)
            return Softmax(dim=-1).forward(logits)
        return self.forward(X)

    def predict_proba(self, X: TensorLike) -> torch.Tensor:
        return self.predict(X, return_probabilities=True)

    def get_weights(self) -> Dict[str, List[torch.Tensor]]:
        return {"Ws": [W.detach().clone() for W in self.Ws], "bs": [b.detach().clone() for b in self.bs]}

    def set_weights(self, weights_dict: Mapping[str, List[torch.Tensor]]) -> None:
        if "Ws" not in weights_dict or "bs" not in weights_dict:
            raise CheckpointError("<memory>", operation="load", message="weights_dict must contain 'Ws' and 'bs'")
        if len(weights_dict["Ws"]) != self.num_layers or len(weights_dict["bs"]) != self.num_layers:
            raise CheckpointError("<memory>", operation="load", message="Weight/bias layer count mismatch")
        with torch.no_grad():
            for i, (W_new, b_new) in enumerate(zip(weights_dict["Ws"], weights_dict["bs"])):
                W_new = W_new.to(device=self.device, dtype=self.dtype)
                b_new = b_new.to(device=self.device, dtype=self.dtype)
                if W_new.shape != self.Ws[i].shape or b_new.shape != self.bs[i].shape:
                    raise ObservationShapeError(
                        expected_shape={"W": tuple(self.Ws[i].shape), "b": tuple(self.bs[i].shape)},
                        actual_shape={"W": tuple(W_new.shape), "b": tuple(b_new.shape)},
                    )
                self.Ws[i].copy_(W_new)
                self.bs[i].copy_(b_new)

    def get_config(self) -> Dict[str, Any]:
        return {
            "layer_dims": list(self.layer_dims),
            "hidden_activation": self.hidden_activation_name,
            "output_activation": self.output_activation_name,
            "loss_function": self.loss_function_name,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "momentum_beta": self.nn_config.get("momentum_beta", 0.9),
            "adam_beta1": self.nn_config.get("adam_beta1", 0.9),
            "adam_beta2": self.nn_config.get("adam_beta2", 0.999),
            "adam_epsilon": self.nn_config.get("adam_epsilon", 1e-8),
            "rmsprop_alpha": self.nn_config.get("rmsprop_alpha", 0.99),
            "rmsprop_epsilon": self.nn_config.get("rmsprop_epsilon", 1e-8),
            "adamw_weight_decay": self.nn_config.get("adamw_weight_decay", self.l2_lambda),
            "huber_delta": self.nn_config.get("huber_delta", 1.0),
            "l1_lambda": self.l1_lambda,
            "l2_lambda": self.l2_lambda,
            "gradient_clip_norm": self.max_grad_norm,
            "gradient_explosion_threshold": self.gradient_explosion_threshold,
            "weight_init": self.weight_init,
            "nan_guard": self.nan_guard,
        }

    def diagnostics(self) -> Dict[str, Any]:
        weight_norm = math.sqrt(sum(float(W.detach().pow(2).sum().item()) for W in self.Ws))
        bias_norm = math.sqrt(sum(float(b.detach().pow(2).sum().item()) for b in self.bs))
        return {
            "config": self.get_config(),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "num_layers": self.num_layers,
            "parameter_count": int(sum(W.numel() + b.numel() for W, b in zip(self.Ws, self.bs))),
            "weight_norm": weight_norm,
            "bias_norm": bias_norm,
            "training_steps": self.training_steps,
            "last_metrics": dict(self.last_metrics),
            "loss_stats": self.loss_stats.snapshot().__dict__,
        }

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "model_type": type(self).__name__,
            "model_weights": self.get_weights(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.get_config(),
            "training_steps": self.training_steps,
            "last_metrics": dict(self.last_metrics),
        }

    def load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if "model_weights" not in checkpoint:
            raise CheckpointError("<memory>", operation="load", message="NeuralNetwork checkpoint missing model_weights")
        self.set_weights(checkpoint["model_weights"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        self.training_steps = int(checkpoint.get("training_steps", self.training_steps))
        self.last_metrics = dict(checkpoint.get("last_metrics", self.last_metrics))

    def save_weights(self, path: Union[str, Path]) -> None:
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.get_checkpoint(), path)
        except Exception as exc:
            raise CheckpointError(str(path), operation="save", cause=exc) from exc

    def load_weights(self, path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None) -> None:
        path = Path(path)
        try:
            checkpoint = torch.load(path, map_location=map_location)
            self.load_checkpoint(checkpoint)
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(str(path), operation="load", cause=exc) from exc

    def reset_optimizer(self) -> None:
        self._init_optimizer()

    def extra_repr(self) -> str:
        return f"dims={self.layer_dims}, loss={self.loss_function_name}, optimizer={self.optimizer_name}"


if __name__ == "__main__":
    print("\n=== Running Neural Network ===\n")
    printer.status("TEST", "Neural Network initialized", "info")
    torch.manual_seed(7)

    reg = NeuralNetwork(8, 3, config={"layer_dims": [8, 16, 3], "loss_function": "mse", "optimizer": "adam"})
    X = torch.randn(12, 8)
    y = torch.randn(12, 3)
    loss = reg.train_step(X, y)
    assert math.isfinite(loss) and reg.predict(X[:2]).shape == (2, 3)
    printer.status("TEST", "Regression path verified", "success")

    clf_cfg = {"layer_dims": [8, 12, 4], "loss_function": "cross_entropy", "output_activation": "linear", "optimizer": "rmsprop"}
    clf = NeuralNetwork(8, 4, config=clf_cfg)
    labels = torch.randint(0, 4, (12,))
    cls_loss = clf.train_step(X, labels)
    assert math.isfinite(cls_loss) and clf.predict_proba(X[:2]).shape == (2, 4)
    ckpt = Path("neural_network_test.pt")
    clf.save_weights(ckpt)
    restored = NeuralNetwork(8, 4, config=clf_cfg)
    restored.load_weights(ckpt)
    assert torch.allclose(clf.predict_logits(X[:2]), restored.predict_logits(X[:2]), atol=1e-6)
    ckpt.unlink(missing_ok=True)
    printer.status("TEST", "Checkpoint path verified", "success")

    print("\n=== Test ran successfully ===\n")
