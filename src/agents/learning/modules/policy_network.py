"""
Production-ready policy network utilities for the SLAI learning subsystem.

This module provides a configurable PyTorch policy network, optimizer factory,
checkpoint helpers, and a random-network-distillation novelty detector. It keeps
backward-compatible constructor/factory names used by MAML, SLAIEnv, and the
StrategySelector while keeping configuration access centralized through the
existing SLAI learning config loader.
"""

from __future__ import annotations

import math
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import torch.optim as optim  # type: ignore

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from torch.distributions import Categorical  # type: ignore

from ...base.modules.activation_engine import get_activation
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.learning_error import *
from ..utils.learning_calculations import *
from ..utils.learning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Policy Network")
printer = PrettyPrinter()

TensorLike = Union[torch.Tensor, Sequence[float], Sequence[Sequence[float]]]


def _normalise_hidden_sizes(value: Any, *, name: str, default: Sequence[int]) -> List[int]:
    """Validate and normalize a hidden-size value without owning config loading."""
    if value is None:
        value = list(default)
    if isinstance(value, int):
        value = [value]
    validate_non_empty_sequence(value, name)
    sizes: List[int] = []
    for idx, item in enumerate(value):
        validate_positive(item, f"{name}[{idx}]", strict=True)
        sizes.append(int(item))
    return sizes


def _ensure_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if torch.isnan(tensor).any():
        raise NaNException(f"NaN detected in {name}", location=name)
    if torch.isinf(tensor).any():
        raise InfException(f"Inf detected in {name}", location=name)


class PolicyNetwork(nn.Module):
    """Configurable feed-forward policy for discrete or continuous action heads."""

    SUPPORTED_OUTPUTS = {"softmax", "linear", "identity", "none", "tanh", "sigmoid"}
    SUPPORTED_INITS = {"auto", "kaiming", "xavier", "orthogonal", "normal"}

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        hidden_activation: str = "relu",
        output_activation: str = "softmax",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        weight_init: str = "auto",
        gradient_clip_norm: Optional[float] = None,
        config_override: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.config = load_global_config()
        self.pn_config = get_config_section("policy_network") or {}
        if config_override is not None:
            if not isinstance(config_override, Mapping):
                raise InvalidConfigError(
                    "config_override must be a mapping when provided",
                    received_value=type(config_override).__name__,
                )
            self.pn_config.update(dict(config_override))

        layer_dims = self.pn_config.get("layer_dims")
        default_hidden: Sequence[int] = [128, 64]
        if isinstance(layer_dims, Sequence) and not isinstance(layer_dims, (str, bytes)) and len(layer_dims) >= 3:
            default_hidden = [int(dim) for dim in list(layer_dims)[1:-1]]
        elif self.pn_config.get("hidden_size") is not None:
            hidden_size = int(self.pn_config.get("hidden_size", 128))
            default_hidden = [hidden_size, hidden_size]

        if hidden_sizes is None:
            hidden_sizes = self.pn_config.get("hidden_layer_sizes", self.pn_config.get("hidden_sizes", default_hidden))

        hidden_activation = str(self.pn_config.get("hidden_activation", hidden_activation))
        output_activation = str(self.pn_config.get("output_activation", output_activation))
        use_batch_norm = coerce_bool(self.pn_config.get("use_batch_norm", use_batch_norm))
        dropout_rate = coerce_float(self.pn_config.get("dropout_rate", dropout_rate), minimum=0.0, maximum=0.999999)
        l1_lambda = coerce_float(self.pn_config.get("l1_lambda", l1_lambda), minimum=0.0)
        l2_lambda = coerce_float(self.pn_config.get("l2_lambda", l2_lambda), minimum=0.0)
        weight_init = str(self.pn_config.get("weight_init", weight_init))
        gradient_clip_norm = self.pn_config.get("gradient_clip_norm", gradient_clip_norm)
        if gradient_clip_norm is not None:
            validate_positive(gradient_clip_norm, "policy_network.gradient_clip_norm", strict=True)
            gradient_clip_norm = float(gradient_clip_norm)

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = _normalise_hidden_sizes(hidden_sizes, name="policy_network.hidden_layer_sizes", default=default_hidden)
        self.hidden_activation_name = hidden_activation.lower()
        self.output_activation_name = output_activation.lower()
        self.use_batch_norm = bool(use_batch_norm)
        self.dropout_rate = float(dropout_rate)
        self.l1_lambda = float(l1_lambda)
        self.l2_lambda = float(l2_lambda)
        self.weight_init = weight_init.lower()
        self.gradient_clip_norm = gradient_clip_norm

        self._validate_configuration()
        self.backbone, last_dim = self._build_backbone()
        self.output_layer = nn.Linear(last_dim, self.output_dim)
        self.output_activation = self._build_output_activation(self.output_activation_name)
        self._init_weights()

        logger.info(
            "PolicyNetwork initialised | input=%s output=%s hidden=%s output_act=%s batch_norm=%s dropout=%.4f",
            self.input_dim,
            self.output_dim,
            self.hidden_sizes,
            self.output_activation_name,
            self.use_batch_norm,
            self.dropout_rate,
        )

    def _validate_configuration(self) -> None:
        validate_positive(self.input_dim, "input_dim", strict=True)
        validate_positive(self.output_dim, "output_dim", strict=True)
        validate_in_range(self.dropout_rate, "dropout_rate", 0.0, 1.0, inclusive_high=False)
        validate_non_negative(self.l1_lambda, "l1_lambda")
        validate_non_negative(self.l2_lambda, "l2_lambda")
        if self.gradient_clip_norm is not None:
            validate_positive(self.gradient_clip_norm, "gradient_clip_norm", strict=True)
        if self.output_activation_name not in self.SUPPORTED_OUTPUTS:
            raise InvalidConfigError(
                "Unsupported policy output activation",
                config_key="policy_network.output_activation",
                received_value=self.output_activation_name,
                context={"supported": sorted(self.SUPPORTED_OUTPUTS)},
            )
        if self.weight_init not in self.SUPPORTED_INITS:
            raise InvalidConfigError(
                "Unsupported policy weight initialisation",
                config_key="policy_network.weight_init",
                received_value=self.weight_init,
                context={"supported": sorted(self.SUPPORTED_INITS)},
            )

    def _build_backbone(self) -> Tuple[nn.Sequential, int]:
        layers: List[nn.Module] = []
        prev_dim = self.input_dim
        for hdim in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, int(hdim)))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(int(hdim)))
            layers.append(get_activation(self.hidden_activation_name))
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = int(hdim)
        return nn.Sequential(*layers), prev_dim

    @staticmethod
    def _build_output_activation(name: str) -> nn.Module:
        if name in {"linear", "identity", "none"}:
            return nn.Identity()
        if name == "softmax":
            return nn.Softmax(dim=-1)
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        return get_activation(name)

    def _init_weights(self) -> None:
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.weight_init == "auto":
                if self.hidden_activation_name in {"relu", "leaky_relu"}:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
            elif self.weight_init == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif self.weight_init == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif self.weight_init == "orthogonal":
                nn.init.orthogonal_(module.weight)
            elif self.weight_init == "normal":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def device(self) -> torch.device:
        return self.output_layer.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.output_layer.weight.dtype

    def _prepare_input(self, x: TensorLike) -> Tuple[torch.Tensor, bool]:
        if torch.is_tensor(x):
            tensor = x
        else:
            tensor = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        tensor = tensor.to(device=self.device, dtype=self.dtype) # type: ignore
    
        squeezed = False
        if tensor.ndim == 0:
            raise ObservationShapeError(expected_shape=(self.input_dim,), actual_shape=tuple(tensor.shape))
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        elif tensor.ndim > 2:
            if tensor.numel() == self.input_dim:
                tensor = tensor.reshape(1, self.input_dim)
                squeezed = True
            else:
                tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.ndim != 2 or tensor.shape[-1] != self.input_dim:
            raise ObservationShapeError(expected_shape=(None, self.input_dim), actual_shape=tuple(tensor.shape))
        _ensure_finite_tensor(tensor, "policy_input")
        return tensor, squeezed

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        current = x
        for module in self.backbone:
            if isinstance(module, nn.BatchNorm1d) and current.shape[0] == 1 and self.training:
                current = F.batch_norm(
                    current,
                    module.running_mean,
                    module.running_var,
                    module.weight,
                    module.bias,
                    training=False,
                    momentum=module.momentum,
                    eps=module.eps,
                )
            else:
                current = module(current)
        _ensure_finite_tensor(current, "policy_features")
        return current

    def forward_features(self, x: TensorLike) -> torch.Tensor:
        prepared, _ = self._prepare_input(x)
        return self._forward_backbone(prepared)

    def forward_logits(self, x: TensorLike) -> torch.Tensor:
        logits = self.output_layer(self.forward_features(x))
        _ensure_finite_tensor(logits, "policy_logits")
        return logits

    def forward(self, x: TensorLike) -> torch.Tensor:
        prepared, squeezed = self._prepare_input(x)
        logits = self.output_layer(self._forward_backbone(prepared))
        outputs = self.output_activation(logits)
        _ensure_finite_tensor(outputs, "policy_output")
        return outputs.squeeze(0) if squeezed else outputs

    def predict(self, x: TensorLike, deterministic: bool = True) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.sample_action(x, deterministic=deterministic) if self.output_activation_name == "softmax" else self.forward(x)

    def distribution(self, x: TensorLike) -> Categorical:
        if self.output_activation_name != "softmax":
            raise InvalidActionError("Categorical distribution requires output_activation='softmax'")
        return Categorical(logits=self.forward_logits(x))

    def sample_action(self, x: TensorLike, deterministic: bool = False) -> torch.Tensor:
        if self.output_activation_name == "softmax":
            dist = self.distribution(x)
            return dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return self.forward(x)

    def log_prob(self, x: TensorLike, actions: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(x)
        actions = actions.to(device=dist.probs.device, dtype=torch.long)
        return dist.log_prob(actions)

    def entropy(self, x: TensorLike) -> torch.Tensor:
        return self.distribution(x).entropy()

    def evaluate_actions(self, x: TensorLike, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        dist = self.distribution(x)
        actions = actions.to(device=dist.probs.device, dtype=torch.long)
        return {"log_prob": dist.log_prob(actions), "entropy": dist.entropy(), "probs": dist.probs}

    def regularization_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), device=self.device, dtype=self.dtype)
        if self.l1_lambda > 0.0:
            penalty = penalty + self.l1_lambda * sum(param.abs().sum() for param in self.parameters())
        if self.l2_lambda > 0.0:
            penalty = penalty + self.l2_lambda * sum(param.pow(2).sum() for param in self.parameters())
        return penalty

    def policy_gradient_loss(
        self,
        states: TensorLike,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        entropy_coef: float = 0.0,
        normalize_advantages: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        validate_non_negative(entropy_coef, "entropy_coef")
        actions = actions.to(device=self.device, dtype=torch.long).view(-1)
        advantages = advantages.to(device=self.device, dtype=self.dtype).view(-1)
        if normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        dist = self.distribution(states)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        loss = -(log_prob * advantages.detach()).mean() - float(entropy_coef) * entropy + self.regularization_penalty()
        _ensure_finite_tensor(loss, "policy_gradient_loss")
        return loss, {"entropy": float(entropy.detach().item()), "mean_log_prob": float(log_prob.detach().mean().item())}

    def supervised_loss(self, states: TensorLike, target: torch.Tensor, objective: str = "cross_entropy") -> torch.Tensor:
        objective = str(objective).lower()
        if objective in {"cross_entropy", "ce", "classification"}:
            target = target.to(device=self.device, dtype=torch.long).view(-1)
            loss = F.cross_entropy(self.forward_logits(states), target)
        elif objective in {"mse", "regression"}:
            target = target.to(device=self.device, dtype=self.dtype)
            loss = F.mse_loss(self.forward(states), target)
        else:
            raise InvalidConfigError("Unsupported policy supervised objective", received_value=objective)
        loss = loss + self.regularization_penalty()
        _ensure_finite_tensor(loss, "policy_supervised_loss")
        return loss

    def train_step(
        self,
        states: TensorLike,
        target: torch.Tensor,
        optimizer: optim.Optimizer,
        objective: str = "cross_entropy",
        gradient_clip_norm: Optional[float] = None,
    ) -> float:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        loss = self.supervised_loss(states, target, objective=objective)
        loss.backward()
        clip_norm = self.gradient_clip_norm if gradient_clip_norm is None else gradient_clip_norm
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(self.parameters(), float(clip_norm))
        optimizer.step()
        return float(loss.detach().item())

    def get_config(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layer_sizes": list(self.hidden_sizes),
            "hidden_activation": self.hidden_activation_name,
            "output_activation": self.output_activation_name,
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
            "l1_lambda": self.l1_lambda,
            "l2_lambda": self.l2_lambda,
            "weight_init": self.weight_init,
            "gradient_clip_norm": self.gradient_clip_norm,
        }

    def diagnostics(self) -> Dict[str, Any]:
        total_params = int(sum(p.numel() for p in self.parameters()))
        trainable_params = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        grad_norm_sq = 0.0
        for param in self.parameters():
            if param.grad is not None:
                grad_norm_sq += float(param.grad.detach().norm(2).item() ** 2)
        return {
            "config": self.get_config(),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "gradient_norm": math.sqrt(grad_norm_sq),
        }

    def get_checkpoint(self, optimizer: Optional[optim.Optimizer] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        checkpoint = {
            "model_type": type(self).__name__,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
            "diagnostics": self.diagnostics(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            checkpoint["extra"] = to_json_safe(extra)
        return checkpoint

    def load_checkpoint(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, Mapping):
            raise CheckpointError("<memory>", operation="load", message="Policy checkpoint missing state_dict")
        self.load_state_dict(state, strict=strict)

    def save_checkpoint(self, path: Union[str, Path], optimizer: Optional[optim.Optimizer] = None, extra: Optional[Dict[str, Any]] = None) -> Path:
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.get_checkpoint(optimizer=optimizer, extra=extra), path)
            return path
        except Exception as exc:
            raise CheckpointError(str(path), operation="save", cause=exc) from exc

    def load_from_checkpoint(self, path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None, strict: bool = True) -> Dict[str, Any]:
        path = Path(path)
        try:
            checkpoint = torch.load(path, map_location=map_location)
            self.load_checkpoint(checkpoint, strict=strict)
            return checkpoint if isinstance(checkpoint, dict) else {"state_dict": checkpoint}
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(str(path), operation="load", cause=exc) from exc

    def save(self, path: Union[str, Path]) -> None:
        torch.save(self.state_dict(), Path(path))

    def load(self, path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None, strict: bool = True) -> None:
        payload = torch.load(Path(path), map_location=map_location)
        self.load_checkpoint(payload, strict=strict)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, hidden_sizes={self.hidden_sizes}, output={self.output_activation_name}"


class NoveltyDetector(nn.Module):
    """Random-network-distillation novelty detector."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 32,
        learning_rate: float = 1e-3,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        gradient_clip_norm: Optional[float] = None,
        target_update_tau: float = 0.01,
    ) -> None:
        super().__init__()

        self.config = load_global_config()
        self.pn_config = get_config_section("policy_network") or {}
        nd_config = self.pn_config.get("novelty_detector", {})
        if isinstance(nd_config, Mapping):
            feature_dim = int(nd_config.get("feature_dim", feature_dim))
            learning_rate = float(nd_config.get("learning_rate", learning_rate))
            hidden_sizes = nd_config.get("hidden_sizes", hidden_sizes)
            activation = str(nd_config.get("activation", activation))
            gradient_clip_norm = nd_config.get("gradient_clip_norm", gradient_clip_norm)
            target_update_tau = float(nd_config.get("target_update_tau", target_update_tau))

        validate_positive(input_dim, "novelty.input_dim", strict=True)
        validate_positive(feature_dim, "novelty.feature_dim", strict=True)
        validate_positive(learning_rate, "novelty.learning_rate", strict=True)
        if gradient_clip_norm is not None:
            validate_positive(gradient_clip_norm, "novelty.gradient_clip_norm", strict=True)
            gradient_clip_norm = float(gradient_clip_norm)
        validate_probability(target_update_tau, "novelty.target_update_tau")

        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_sizes = _normalise_hidden_sizes(hidden_sizes, name="novelty.hidden_sizes", default=[feature_dim])
        self.activation_name = str(activation).lower()
        self.gradient_clip_norm = gradient_clip_norm
        self.target_update_tau = float(target_update_tau)

        self.predictor = self._build_mlp(trainable=True)
        self.target = self._build_mlp(trainable=False)
        self.target.eval()
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=float(learning_rate))
        self.loss_fn = nn.MSELoss()

    @property
    def device(self) -> torch.device:
        return next(self.predictor.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.predictor.parameters()).dtype

    def _build_mlp(self, trainable: bool) -> nn.Sequential:
        dims = [self.input_dim, *self.hidden_sizes, self.feature_dim]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(int(in_dim), int(out_dim)))
            layers.append(get_activation(self.activation_name))
        layers.append(nn.Linear(int(dims[-2]), int(dims[-1])))
        network = nn.Sequential(*layers)
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if not trainable:
            for param in network.parameters():
                param.requires_grad = False
        return network

    def _prepare_input(self, x: TensorLike) -> torch.Tensor:
        if torch.is_tensor(x):
            tensor = x
        else:
            tensor = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        tensor = tensor.to(device=self.device, dtype=self.dtype) # type: ignore
    
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.ndim != 2 or tensor.shape[-1] != self.input_dim:
            raise ObservationShapeError(expected_shape=(None, self.input_dim), actual_shape=tuple(tensor.shape))
        _ensure_finite_tensor(tensor, "novelty_input")
        return tensor

    def forward(self, x: TensorLike) -> torch.Tensor:
        x = self._prepare_input(x)
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        scores = torch.norm(pred_feat - target_feat, dim=-1)
        _ensure_finite_tensor(scores, "novelty_scores")
        return scores

    def compute_features(self, x: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._prepare_input(x)
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        return pred_feat, target_feat

    @torch.no_grad()
    def update_target(self, tau: Optional[float] = None) -> None:
        tau = self.target_update_tau if tau is None else float(tau)
        validate_probability(tau, "novelty.tau")
        for t_param, p_param in zip(self.target.parameters(), self.predictor.parameters()):
            t_param.mul_(1.0 - tau).add_(p_param, alpha=tau)

    def train_step(self, x: TensorLike) -> float:
        x = self._prepare_input(x)
        self.predictor.train()
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.predictor(x)
        with torch.no_grad():
            target = self.target(x)
        loss = self.loss_fn(pred, target)
        _ensure_finite_tensor(loss, "novelty_loss")
        loss.backward()
        if self.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.predictor.parameters(), float(self.gradient_clip_norm))
        self.optimizer.step()
        return float(loss.detach().item())

    def diagnostics(self, sample: Optional[TensorLike] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "input_dim": self.input_dim,
            "feature_dim": self.feature_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "device": str(self.device),
            "target_update_tau": self.target_update_tau,
        }
        if sample is not None:
            with torch.no_grad():
                scores = self.forward(sample)
            payload.update({"score_mean": float(scores.mean().item()), "score_max": float(scores.max().item())})
        return payload

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "model_type": type(self).__name__,
            "config": self.diagnostics(),
            "predictor_state_dict": self.predictor.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        self.predictor.load_state_dict(checkpoint["predictor_state_dict"], strict=strict)
        self.target.load_state_dict(checkpoint["target_state_dict"], strict=strict)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)


def create_policy_network(
    input_dim: int,
    output_dim: int,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> PolicyNetwork:
    """Create a ``PolicyNetwork`` using the existing learning config loader."""
    network = PolicyNetwork(input_dim=input_dim, output_dim=output_dim, config_override=config)
    return network.to(torch.device(device)) if device is not None else network


def create_policy_optimizer(model: nn.Module, config: Optional[Dict[str, Any]] = None) -> optim.Optimizer:
    """Create an optimizer for a policy network using ``policy_network.optimizer_config``."""
    if config is None:
        full_config = load_global_config()
        config = get_config_section("policy_network", config=full_config) or {}
    optimizer_config = dict(config.get("optimizer_config", {}))
    optimizer_type = str(optimizer_config.get("type", "adam")).lower()
    learning_rate = coerce_float(optimizer_config.get("learning_rate", 1e-3), minimum=1e-12)
    weight_decay = coerce_float(optimizer_config.get("weight_decay", config.get("l2_lambda", 0.0)), minimum=0.0)

    if optimizer_type == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_type == "momentum":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=coerce_float(optimizer_config.get("momentum_beta", 0.9), minimum=0.0, maximum=0.999999),
            weight_decay=weight_decay,
        )
    if optimizer_type == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(
                coerce_float(optimizer_config.get("adam_beta1", 0.9), minimum=0.0, maximum=0.999999),
                coerce_float(optimizer_config.get("adam_beta2", 0.999), minimum=0.0, maximum=0.999999),
            ),
            eps=coerce_float(optimizer_config.get("adam_epsilon", 1e-8), minimum=1e-16),
            weight_decay=weight_decay,
        )
    if optimizer_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(
                coerce_float(optimizer_config.get("adam_beta1", 0.9), minimum=0.0, maximum=0.999999),
                coerce_float(optimizer_config.get("adam_beta2", 0.999), minimum=0.0, maximum=0.999999),
            ),
            eps=coerce_float(optimizer_config.get("adam_epsilon", 1e-8), minimum=1e-16),
            weight_decay=weight_decay,
        )
    if optimizer_type == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            alpha=coerce_float(optimizer_config.get("rmsprop_alpha", 0.99), minimum=0.0, maximum=0.999999),
            eps=coerce_float(optimizer_config.get("rmsprop_epsilon", 1e-8), minimum=1e-16),
            weight_decay=weight_decay,
            momentum=coerce_float(optimizer_config.get("momentum_beta", 0.0), minimum=0.0, maximum=0.999999),
        )
    raise InvalidConfigError(
        "Unsupported policy optimizer type",
        config_key="policy_network.optimizer_config.type",
        received_value=optimizer_type,
        context={"supported": ["sgd", "momentum", "adam", "adamw", "rmsprop"]},
    )


def create_novelty_detector(input_dim: int, config: Optional[Dict[str, Any]] = None, device: Optional[Union[str, torch.device]] = None) -> NoveltyDetector:
    """Create a ``NoveltyDetector`` from the policy-network novelty subsection."""
    if config is None:
        full_config = load_global_config()
        config = get_config_section("policy_network", config=full_config) or {}
    novelty_cfg = config.get("novelty_detector", {}) if isinstance(config.get("novelty_detector", {}), Mapping) else {}
    detector = NoveltyDetector(
        input_dim=input_dim,
        feature_dim=int(novelty_cfg.get("feature_dim", 32)),
        learning_rate=float(novelty_cfg.get("learning_rate", 1e-3)),
        hidden_sizes=novelty_cfg.get("hidden_sizes"),
        activation=str(novelty_cfg.get("activation", config.get("hidden_activation", "relu"))),
        gradient_clip_norm=novelty_cfg.get("gradient_clip_norm", config.get("gradient_clip_norm")),
        target_update_tau=float(novelty_cfg.get("target_update_tau", 0.01)),
    )
    return detector.to(torch.device(device)) if device is not None else detector


if __name__ == "__main__":
    print("\n=== Running Policy Network ===\n")
    printer.status("TEST", "Policy Network initialized", "info")
    torch.manual_seed(7)
    cfg = get_config_section("policy_network") or {}
    net = create_policy_network(4, 3, cfg)
    opt = create_policy_optimizer(net, cfg)
    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    probs = net(x)
    assert probs.shape == (16, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)
    loss = net.train_step(x, y, opt)
    assert math.isfinite(loss)
    assert net.sample_action(x[:4]).shape == (4,)
    ckpt = Path("policy_network_test.pt")
    net.save_checkpoint(ckpt, optimizer=opt)
    restored = create_policy_network(4, 3, cfg)
    restored.load_from_checkpoint(ckpt)
    assert torch.allclose(net.forward_logits(x[:2]), restored.forward_logits(x[:2]), atol=1e-6)
    ckpt.unlink(missing_ok=True)
    nd = create_novelty_detector(4, cfg)
    nd_loss = nd.train_step(x)
    nd.update_target()
    assert nd(x).shape == (16,) and math.isfinite(nd_loss)
    printer.status("TEST", "Policy Network verified", "success")
    print("\n=== Test ran successfully ===\n")
