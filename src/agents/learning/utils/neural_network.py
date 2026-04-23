"""Production-ready manual neural network with explicit backpropagation."""

from __future__ import annotations

import time
import torch
import torch.nn as nn

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .config_loader import load_global_config, get_config_section
from ...base.modules.activation_engine import (Activation, Linear,
                                             ReLU, Sigmoid, Softmax, Tanh)
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Neural Network")
printer = PrettyPrinter

DEBUG_LEVEL = 10   # numeric value for DEBUG

TensorLike = Union[torch.Tensor, Sequence[float]]


class Loss:
    """Base class for loss functions."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(self, y_pred: torch.Tensor, y_true: torch.Tensor, batch_size: int) -> torch.Tensor:
        raise NotImplementedError


class MSELoss(Loss):
    """Mean squared error loss."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred: torch.Tensor, y_true: torch.Tensor, batch_size: int) -> torch.Tensor:
        batch_size = max(int(batch_size), 1)
        return (2.0 / batch_size) * (y_pred - y_true)


class CrossEntropyLoss(Loss):
    """Cross-entropy loss over raw logits and class-index targets."""

    def __init__(self):
        self.softmax = Softmax(dim=-1)
        self._cache: Dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, y_true_indices: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"CrossEntropyLoss expects 2D logits, got shape {tuple(logits.shape)}")
        if y_true_indices.ndim != 1:
            raise ValueError(
                f"CrossEntropyLoss expects 1D class indices, got shape {tuple(y_true_indices.shape)}"
            )
        if logits.shape[0] != y_true_indices.shape[0]:
            raise ValueError("Batch size mismatch between logits and labels.")

        y_true_indices = y_true_indices.to(device=logits.device, dtype=torch.long)
        logsumexp = torch.logsumexp(logits, dim=-1)
        correct_class_logits = logits.gather(1, y_true_indices.unsqueeze(1)).squeeze(1)
        loss = torch.mean(logsumexp - correct_class_logits)
        self._cache["probs"] = self.softmax.forward(logits)
        return loss

    def backward(self, logits: torch.Tensor, y_true_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        batch_size = max(int(batch_size), 1)
        probs = self._cache.get("probs")
        if probs is None or probs.shape != logits.shape or probs.device != logits.device:
            probs = self.softmax.forward(logits)

        y_true_indices = y_true_indices.to(device=logits.device, dtype=torch.long)
        y_true_one_hot = torch.zeros_like(probs)
        y_true_one_hot.scatter_(1, y_true_indices.unsqueeze(1), 1.0)
        return (probs - y_true_one_hot) / batch_size


class Optimizer:
    """Base class for optimizers."""

    def __init__(self, learning_rate: float):
        self.learning_rate = float(learning_rate)

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.learning_rate = float(state_dict.get("learning_rate", self.learning_rate))


class SGD(Optimizer):
    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        for i in range(len(params_Ws)):
            params_Ws[i].sub_(self.learning_rate * grads_dWs[i])
            params_bs[i].sub_(self.learning_rate * grads_dBs[i])


class SGDMomentum(Optimizer):
    def __init__(self, learning_rate: float, beta: float = 0.9):
        super().__init__(learning_rate)
        self.beta = float(beta)
        self.v_Ws = None
        self.v_bs = None

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        if self.v_Ws is None:
            self.v_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.v_bs = [torch.zeros_like(b) for b in params_bs]

        for i in range(len(params_Ws)):
            self.v_Ws[i].mul_(self.beta).add_(grads_dWs[i])
            self.v_bs[i].mul_(self.beta).add_(grads_dBs[i])
            params_Ws[i].sub_(self.learning_rate * self.v_Ws[i])
            params_bs[i].sub_(self.learning_rate * self.v_bs[i])

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(),
            "beta": self.beta,
            "v_Ws": [v.clone() for v in self.v_Ws] if self.v_Ws is not None else None,
            "v_bs": [v.clone() for v in self.v_bs] if self.v_bs is not None else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.beta = float(state_dict.get("beta", self.beta))
        self.v_Ws = state_dict.get("v_Ws")
        self.v_bs = state_dict.get("v_bs")


class Adam(Optimizer):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.m_Ws = None
        self.v_Ws = None
        self.m_bs = None
        self.v_bs = None
        self.t = 0

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs) -> None:
        self.t += 1
        if self.m_Ws is None:
            self.m_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.v_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.m_bs = [torch.zeros_like(b) for b in params_bs]
            self.v_bs = [torch.zeros_like(b) for b in params_bs]

        for i in range(len(params_Ws)):
            self.m_Ws[i].mul_(self.beta1).add_(grads_dWs[i], alpha=1 - self.beta1)
            self.v_Ws[i].mul_(self.beta2).addcmul_(grads_dWs[i], grads_dWs[i], value=1 - self.beta2)
            m_hat_W = self.m_Ws[i] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_Ws[i] / (1 - self.beta2 ** self.t)
            params_Ws[i].sub_(self.learning_rate * m_hat_W / (torch.sqrt(v_hat_W) + self.epsilon))

            self.m_bs[i].mul_(self.beta1).add_(grads_dBs[i], alpha=1 - self.beta1)
            self.v_bs[i].mul_(self.beta2).addcmul_(grads_dBs[i], grads_dBs[i], value=1 - self.beta2)
            m_hat_b = self.m_bs[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_bs[i] / (1 - self.beta2 ** self.t)
            params_bs[i].sub_(self.learning_rate * m_hat_b / (torch.sqrt(v_hat_b) + self.epsilon))

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(),
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "m_Ws": [m.clone() for m in self.m_Ws] if self.m_Ws is not None else None,
            "v_Ws": [v.clone() for v in self.v_Ws] if self.v_Ws is not None else None,
            "m_bs": [m.clone() for m in self.m_bs] if self.m_bs is not None else None,
            "v_bs": [v.clone() for v in self.v_bs] if self.v_bs is not None else None,
            "t": self.t,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.beta1 = float(state_dict.get("beta1", self.beta1))
        self.beta2 = float(state_dict.get("beta2", self.beta2))
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
        self.m_Ws = state_dict.get("m_Ws")
        self.v_Ws = state_dict.get("v_Ws")
        self.m_bs = state_dict.get("m_bs")
        self.v_bs = state_dict.get("v_bs")
        self.t = int(state_dict.get("t", self.t))


class NeuralNetwork(nn.Module):
    """Configurable feed-forward neural network with explicit manual backpropagation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive integers.")

        self.config = load_global_config()
        self.nn_config = get_config_section("neural_network")
        if config:
            self.nn_config.update(config)

        default_dims = [input_dim, 128, 64, output_dim]
        raw_dims = list(self.nn_config.get("layer_dims", default_dims))
        if len(raw_dims) < 2:
            raise ValueError("layer_dims must define at least an input and output dimension.")

        raw_dims[0] = input_dim
        raw_dims[-1] = output_dim
        if any(int(dim) <= 0 for dim in raw_dims):
            raise ValueError("All layer dimensions must be positive integers.")

        self.layer_dims = [int(dim) for dim in raw_dims]
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.device_override = torch.device(device) if device is not None else None
        self.dtype = dtype

        self.hidden_activation_name = str(self.nn_config.get("hidden_activation", "relu")).lower()
        self.output_activation_name = str(self.nn_config.get("output_activation", "linear")).lower()
        self.loss_function_name = str(self.nn_config.get("loss_function", "mse")).lower()
        self.optimizer_name = str(self.nn_config.get("optimizer", "adam")).lower()
        self.learning_rate = float(self.nn_config.get("learning_rate", 0.001))
        self.num_layers = len(self.layer_dims) - 1
        self.l1_lambda = float(self.nn_config.get("l1_lambda", 0.0))
        self.l2_lambda = float(self.nn_config.get("l2_lambda", 0.0))
        self.max_grad_norm = self.nn_config.get("gradient_clip_norm", 5.0)
        self.max_grad_norm = None if self.max_grad_norm is None else float(self.max_grad_norm)

        self._init_activation_functions()
        self._initialize_weights()
        self._init_loss_function()
        self._init_optimizer()

        self._cache: Dict[str, Any] = {}
        self.dWs: List[torch.Tensor] = []
        self.dBs: List[torch.Tensor] = []

        logger.info(
            "NeuralNetwork initialised: dims=%s hidden_act=%s output_act=%s loss=%s optimizer=%s lr=%s",
            self.layer_dims,
            self.hidden_activation_name,
            self.output_activation_name,
            self.loss_function_name,
            self.optimizer_name,
            self.learning_rate,
        )

    def _str_to_activation(self, name_str: str) -> Activation:
        name_lower = name_str.lower()
        mapping = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh,
            "linear": Linear,
            "identity": Linear,
            "softmax": Softmax,
        }
        if name_lower not in mapping:
            raise ValueError(f"Unknown activation function: {name_str}")
        return mapping[name_lower]() if name_lower != "softmax" else mapping[name_lower](dim=-1)

    def _init_activation_functions(self) -> None:
        self.hidden_activations = []
        if self.num_layers > 1:
            self.hidden_activations = [
                self._str_to_activation(self.hidden_activation_name) for _ in range(self.num_layers - 1)
            ]
        self.output_activation = self._str_to_activation(self.output_activation_name)

    def _init_loss_function(self) -> None:
        if self.loss_function_name == "mse":
            self.loss_fn = MSELoss()
        elif self.loss_function_name == "cross_entropy":
            if not isinstance(self.output_activation, Linear):
                logger.warning(
                    "CrossEntropyLoss expects logits. Overriding non-linear output activation '%s' for loss computation.",
                    self.output_activation_name,
                )
            self.loss_fn = CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function_name}")

    def _initialize_weights(self) -> None:
        target_device = self.device_override or torch.device("cpu")
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()

        for i in range(self.num_layers):
            fan_in = self.layer_dims[i]
            fan_out = self.layer_dims[i + 1]
            activation = self.hidden_activations[i] if i < self.num_layers - 1 else self.output_activation

            weight = torch.empty((fan_in, fan_out), device=target_device, dtype=self.dtype)
            if isinstance(activation, ReLU):
                nn.init.kaiming_uniform_(weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(weight)
            bias = torch.zeros(fan_out, device=target_device, dtype=self.dtype)

            self.Ws.append(nn.Parameter(weight, requires_grad=False))
            self.bs.append(nn.Parameter(bias, requires_grad=False))

    def _init_optimizer(self) -> None:
        if self.optimizer_name == "sgd":
            self.optimizer = SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name == "momentum":
            self.optimizer = SGDMomentum(
                learning_rate=self.learning_rate,
                beta=float(self.nn_config.get("momentum_beta", 0.9)),
            )
        elif self.optimizer_name == "adam":
            self.optimizer = Adam(
                learning_rate=self.learning_rate,
                beta1=float(self.nn_config.get("adam_beta1", 0.9)),
                beta2=float(self.nn_config.get("adam_beta2", 0.999)),
                epsilon=float(self.nn_config.get("adam_epsilon", 1e-8)),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    @property
    def device(self) -> torch.device:
        return self.Ws[0].device

    def _prepare_input(self, X: TensorLike) -> Tuple[torch.Tensor, bool]:
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)

        squeezed = False
        if X.ndim == 1:
            X = X.unsqueeze(0)
            squeezed = True
        if X.ndim != 2:
            raise ValueError(f"Input must be a 2D tensor or a 1D feature vector, got shape {tuple(X.shape)}")
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input feature dimension mismatch. Expected {self.input_dim}, got {X.shape[1]}")
        return X, squeezed

    def _prepare_target(self, y_true: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.loss_function_name == "cross_entropy":
            if y_true.ndim != 1:
                raise ValueError(
                    f"Cross-entropy targets must be a 1D tensor of class indices, got shape {tuple(y_true.shape)}"
                )
            if y_true.shape[0] != batch_size:
                raise ValueError("Target batch size does not match input batch size.")
            return y_true.to(device=self.device, dtype=torch.long)

        if y_true.ndim == 1 and self.output_dim == 1:
            y_true = y_true.unsqueeze(1)
        if y_true.ndim != 2:
            raise ValueError(f"MSE targets must be 2D (or 1D when output_dim == 1), got shape {tuple(y_true.shape)}")
        if y_true.shape != (batch_size, self.output_dim):
            raise ValueError(
                f"Target shape mismatch. Expected {(batch_size, self.output_dim)}, got {tuple(y_true.shape)}"
            )
        return y_true.to(device=self.device, dtype=self.dtype)

    def forward(self, X: TensorLike) -> torch.Tensor:
        X, squeezed = self._prepare_input(X)
        self._cache = {"inputs": X, "layer_outputs": []}

        current_a = X
        for i in range(self.num_layers):
            W, b = self.Ws[i], self.bs[i]
            z = current_a @ W + b
            current_a = self.hidden_activations[i].forward(z) if i < self.num_layers - 1 else self.output_activation.forward(z)
            self._cache["layer_outputs"].append({"z": z, "a": current_a})

        return current_a.squeeze(0) if squeezed else current_a

    def predict_logits(self, X: TensorLike) -> torch.Tensor:
        X, squeezed = self._prepare_input(X)
        current_a = X
        logits = None
        for i in range(self.num_layers):
            z = current_a @ self.Ws[i] + self.bs[i]
            logits = z
            current_a = self.hidden_activations[i].forward(z) if i < self.num_layers - 1 else self.output_activation.forward(z)
        return logits.squeeze(0) if squeezed else logits

    def regularization_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), device=self.device, dtype=self.dtype)
        if self.l1_lambda > 0.0:
            penalty = penalty + self.l1_lambda * sum(W.abs().sum() for W in self.Ws)
        if self.l2_lambda > 0.0:
            penalty = penalty + 0.5 * self.l2_lambda * sum(W.pow(2).sum() for W in self.Ws)
        return penalty

    def compute_loss(self, y_pred_output: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if isinstance(self.loss_fn, CrossEntropyLoss):
            final_layer_z = self._cache["layer_outputs"][-1]["z"]
            data_loss = self.loss_fn.forward(final_layer_z, y_true)
        else:
            data_loss = self.loss_fn.forward(y_pred_output, y_true)
        return data_loss + self.regularization_penalty()

    def backward(self, y_true: torch.Tensor) -> None:
        m = y_true.shape[0] if y_true.ndim > 0 else 1
        m = max(int(m), 1)

        self.dWs = [torch.zeros_like(W) for W in self.Ws]
        self.dBs = [torch.zeros_like(b) for b in self.bs]

        final_layer_cache = self._cache["layer_outputs"][-1]
        if isinstance(self.loss_fn, CrossEntropyLoss):
            delta = self.loss_fn.backward(final_layer_cache["z"], y_true, m)
        else:
            dL_daL = self.loss_fn.backward(final_layer_cache["a"], y_true, m)
            if isinstance(self.output_activation, Softmax):
                delta = self.output_activation.backward(final_layer_cache["z"], dL_daL)
            else:
                delta = dL_daL * self.output_activation.backward(final_layer_cache["z"])

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
                hidden_activation = self.hidden_activations[i - 1]
                prev_z = self._cache["layer_outputs"][i - 1]["z"]
                if isinstance(hidden_activation, Softmax):
                    delta = hidden_activation.backward(prev_z, da_prev)
                else:
                    delta = da_prev * hidden_activation.backward(prev_z)

    def _global_grad_norm(self) -> torch.Tensor:
        if not self.dWs and not self.dBs:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        total = torch.zeros((), device=self.device, dtype=self.dtype)
        for grad in [*self.dWs, *self.dBs]:
            total = total + grad.pow(2).sum()
        return torch.sqrt(total)

    def _clip_gradients(self) -> None:
        if self.max_grad_norm is None:
            return
        grad_norm = self._global_grad_norm()
        if torch.isfinite(grad_norm) and grad_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (grad_norm + 1e-12)
            for i in range(len(self.dWs)):
                self.dWs[i].mul_(scale)
                self.dBs[i].mul_(scale)

    def update_parameters(self) -> None:
        with torch.no_grad():
            self.optimizer.step(self.Ws, self.bs, self.dWs, self.dBs)

    def train_step(self, X_batch: TensorLike, y_batch: torch.Tensor) -> float:
        step_start = time.perf_counter()
        X_batch, _ = self._prepare_input(X_batch)
        if X_batch.shape[0] == 0:
            raise ValueError("Batch size cannot be zero")
        y_batch = self._prepare_target(y_batch, X_batch.shape[0])

        forward_start = time.perf_counter()
        y_pred_output = self.forward(X_batch)
        if not torch.isfinite(y_pred_output).all():
            raise RuntimeError("Non-finite values detected in network output during forward pass.")
        forward_time = time.perf_counter() - forward_start

        loss_start = time.perf_counter()
        loss = self.compute_loss(y_pred_output, y_batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss became non-finite.")
        loss_time = time.perf_counter() - loss_start

        backward_start = time.perf_counter()
        self.backward(y_batch)
        self._clip_gradients()
        for i, (dw, db) in enumerate(zip(self.dWs, self.dBs)):
            if not torch.isfinite(dw).all() or not torch.isfinite(db).all():
                raise RuntimeError(f"Non-finite gradients detected at layer {i}.")
        backward_time = time.perf_counter() - backward_start

        update_start = time.perf_counter()
        self.update_parameters()
        update_time = time.perf_counter() - update_start

        if logger.isEnabledFor(DEBUG_LEVEL):
            logger.debug(
                "Train step timings | forward=%.6fs loss=%.6fs backward=%.6fs update=%.6fs total=%.6fs grad_norm=%.6f",
                forward_time,
                loss_time,
                backward_time,
                update_time,
                time.perf_counter() - step_start,
                float(self._global_grad_norm().item()),
            )

        return float(loss.item())

    def predict(self, X: TensorLike, return_probabilities: bool = False) -> torch.Tensor:
        if return_probabilities and isinstance(self.loss_fn, CrossEntropyLoss) and isinstance(self.output_activation, Linear):
            logits = self.predict_logits(X)
            return Softmax(dim=-1).forward(logits)
        return self.forward(X)

    def predict_proba(self, X: TensorLike) -> torch.Tensor:
        return self.predict(X, return_probabilities=True)

    def get_weights(self) -> Dict[str, List[torch.Tensor]]:
        return {"Ws": [W.detach().clone() for W in self.Ws], "bs": [b.detach().clone() for b in self.bs]}

    def set_weights(self, weights_dict: Dict[str, List[torch.Tensor]]) -> None:
        if "Ws" not in weights_dict or "bs" not in weights_dict:
            raise ValueError("weights_dict must contain 'Ws' and 'bs' keys.")
        if len(weights_dict["Ws"]) != self.num_layers or len(weights_dict["bs"]) != self.num_layers:
            raise ValueError("Mismatch in the number of layers for weights/biases.")

        with torch.no_grad():
            for i, (W_new, b_new) in enumerate(zip(weights_dict["Ws"], weights_dict["bs"])):
                W_new = W_new.to(device=self.device, dtype=self.dtype)
                b_new = b_new.to(device=self.device, dtype=self.dtype)
                if W_new.shape != self.Ws[i].shape:
                    raise ValueError(
                        f"Weight shape mismatch at layer {i}. Expected {tuple(self.Ws[i].shape)}, got {tuple(W_new.shape)}"
                    )
                if b_new.shape != self.bs[i].shape:
                    raise ValueError(
                        f"Bias shape mismatch at layer {i}. Expected {tuple(self.bs[i].shape)}, got {tuple(b_new.shape)}"
                    )
                self.Ws[i].copy_(W_new)
                self.bs[i].copy_(b_new)

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "model_weights": self.get_weights(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "layer_dims": self.layer_dims,
                "hidden_activation": self.hidden_activation_name,
                "output_activation": self.output_activation_name,
                "loss_function": self.loss_function_name,
                "optimizer": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "l1_lambda": self.l1_lambda,
                "l2_lambda": self.l2_lambda,
                "gradient_clip_norm": self.max_grad_norm,
            },
        }

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.set_weights(checkpoint["model_weights"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

    def save_weights(self, path: Union[str, Path]) -> None:
        torch.save(self.get_checkpoint(), Path(path))

    def load_weights(self, path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None) -> None:
        checkpoint = torch.load(Path(path), map_location=map_location)
        self.load_checkpoint(checkpoint)


if __name__ == "__main__":
    print("\n=== Running Neural Network Smoke Test ===\n")

    network = NeuralNetwork(input_dim=64, output_dim=10)
    print(network)

    X = torch.randn(128, 64)
    y_reg = torch.randn(128, 10)
    loss_reg = network.train_step(X, y_reg)
    print(f"Regression train-step loss: {loss_reg:.4f}")

    clf = NeuralNetwork(
        input_dim=64,
        output_dim=5,
        config={"loss_function": "cross_entropy", "output_activation": "linear"},
    )
    X_cls = torch.randn(128, 64)
    y_cls = torch.randint(0, 5, (128,))
    loss_cls = clf.train_step(X_cls, y_cls)
    print(f"Classification train-step loss: {loss_cls:.4f}")

    checkpoint_path = Path("network_checkpoint.pt")
    clf.save_weights(checkpoint_path)
    restored = NeuralNetwork(
        input_dim=64,
        output_dim=5,
        config={"loss_function": "cross_entropy", "output_activation": "linear"},
    )
    restored.load_weights(checkpoint_path)
    print(
        "Checkpoint restore verified:",
        torch.allclose(clf.predict_logits(X_cls[:4]), restored.predict_logits(X_cls[:4]), atol=1e-6),
    )
    print("\n=== Neural Network Smoke Test Complete ===\n")
