"""
Policy Network – A standard PyTorch module for policy approximation in RL.

This module defines a configurable feed‑forward network that outputs action
probabilities (for discrete actions) or action parameters (for continuous).
It can be used with any RL algorithm that expects a policy network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .config_loader import load_global_config, get_config_section
from ...base.utils.activation_engine import get_activation
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Policy Network")
printer = PrettyPrinter

TensorLike = Union[torch.Tensor, Sequence[float]]

class PolicyNetwork(nn.Module):
    """Configurable feed-forward policy network for discrete or deterministic continuous actions."""

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
    ):
        super().__init__()

        hidden_sizes = [128, 64] if hidden_sizes is None else list(hidden_sizes)
        self._validate_configuration(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
        )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = hidden_sizes
        self.hidden_activation_name = hidden_activation.lower()
        self.output_activation_name = output_activation.lower()
        self.use_batch_norm = bool(use_batch_norm)
        self.dropout_rate = float(dropout_rate)
        self.l1_lambda = float(l1_lambda)
        self.l2_lambda = float(l2_lambda)
        self.weight_init = weight_init.lower()

        layers: List[nn.Module] = []
        prev_dim = self.input_dim
        for hdim in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, hdim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hdim))
            layers.append(get_activation(self.hidden_activation_name))
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hdim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        self.output_activation = self._build_output_activation(self.output_activation_name)

        self._init_weights()
        logger.info(
            "PolicyNetwork initialised: input=%s output=%s hidden=%s hidden_act=%s output_act=%s batch_norm=%s dropout=%.4f",
            self.input_dim,
            self.output_dim,
            self.hidden_sizes,
            self.hidden_activation_name,
            self.output_activation_name,
            self.use_batch_norm,
            self.dropout_rate,
        )

    @staticmethod
    def _validate_configuration(
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int],
        dropout_rate: float,
        l1_lambda: float,
        l2_lambda: float,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must both be positive integers.")
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive integers.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1).")
        if l1_lambda < 0.0 or l2_lambda < 0.0:
            raise ValueError("Regularisation coefficients must be non-negative.")

    @staticmethod
    def _build_output_activation(name: str) -> nn.Module:
        if name in {"linear", "identity", "none"}:
            return nn.Identity()
        if name == "softmax":
            return nn.Softmax(dim=-1)
        return get_activation(name)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.weight_init == "auto":
                    if self.hidden_activation_name in {"relu", "leaky_relu"}:
                        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    else:
                        nn.init.xavier_uniform_(module.weight)
                elif self.weight_init == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                elif self.weight_init == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    raise ValueError("weight_init must be one of: auto, kaiming, xavier")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _prepare_input(self, x: TensorLike) -> Tuple[torch.Tensor, bool]:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.output_layer.weight.dtype, device=self.output_layer.weight.device)
        else:
            x = x.to(device=self.output_layer.weight.device, dtype=self.output_layer.weight.dtype)

        squeezed = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeezed = True
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input of shape (batch, {self.input_dim}) or ({self.input_dim},), got {tuple(x.shape)}"
            )
        return x, squeezed

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
        return current

    def forward_features(self, x: TensorLike) -> torch.Tensor:
        x, _ = self._prepare_input(x)
        return self._forward_backbone(x)

    def forward_logits(self, x: TensorLike) -> torch.Tensor:
        features = self.forward_features(x)
        return self.output_layer(features)

    def forward(self, x: TensorLike) -> torch.Tensor:
        prepared, squeezed = self._prepare_input(x)
        logits = self.output_layer(self._forward_backbone(prepared))
        outputs = self.output_activation(logits)
        return outputs.squeeze(0) if squeezed else outputs

    def predict(self, x: TensorLike) -> torch.Tensor:
        return self.forward(x)

    def regularization_penalty(self) -> torch.Tensor:
        device = self.output_layer.weight.device
        penalty = torch.zeros((), device=device)
        if self.l1_lambda > 0.0:
            penalty = penalty + self.l1_lambda * sum(param.abs().sum() for param in self.parameters())
        if self.l2_lambda > 0.0:
            penalty = penalty + self.l2_lambda * sum(param.pow(2).sum() for param in self.parameters())
        return penalty

    def distribution(self, x: TensorLike) -> Categorical:
        if self.output_activation_name != "softmax":
            raise RuntimeError("distribution() is only available when output_activation='softmax'.")
        return Categorical(logits=self.forward_logits(x))

    def sample_action(self, x: TensorLike, deterministic: bool = False) -> torch.Tensor:
        if self.output_activation_name == "softmax":
            dist = self.distribution(x)
            return dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        outputs = self.forward(x)
        return outputs if deterministic else outputs

    def log_prob(self, x: TensorLike, actions: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(x)
        actions = actions.to(device=dist.probs.device)
        return dist.log_prob(actions)

    def entropy(self, x: TensorLike) -> torch.Tensor:
        return self.distribution(x).entropy()

    def save(self, path: Union[str, Path]) -> None:
        torch.save(self.state_dict(), Path(path))

    def load(self, path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None, strict: bool = True) -> None:
        state_dict = torch.load(Path(path), map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)


class NoveltyDetector(nn.Module):
    """Predictor-target novelty detector in the style of random-network distillation."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 32,
        learning_rate: float = 1e-3,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        gradient_clip_norm: Optional[float] = None,
    ):
        super().__init__()
        if input_dim <= 0 or feature_dim <= 0:
            raise ValueError("input_dim and feature_dim must be positive integers.")

        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_sizes = list(hidden_sizes) if hidden_sizes is not None else [feature_dim]
        self.activation_name = activation.lower()
        self.gradient_clip_norm = gradient_clip_norm

        self.predictor = self._build_mlp(trainable=True)
        self.target = self._build_mlp(trainable=False)
        self.target.eval()
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_mlp(self, trainable: bool) -> nn.Sequential:
        dims = [self.input_dim, *self.hidden_sizes, self.feature_dim]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(get_activation(self.activation_name))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        network = nn.Sequential(*layers)
        if not trainable:
            for param in network.parameters():
                param.requires_grad = False
        return network

    def _prepare_input(self, x: TensorLike) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.predictor[0].weight.dtype, device=self.predictor[0].weight.device)
        else:
            x = x.to(device=self.predictor[0].weight.device, dtype=self.predictor[0].weight.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input of shape (batch, {self.input_dim}) or ({self.input_dim},), got {tuple(x.shape)}"
            )
        return x

    def forward(self, x: TensorLike) -> torch.Tensor:
        x = self._prepare_input(x)
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        return torch.norm(pred_feat - target_feat, dim=-1)

    def compute_features(self, x: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._prepare_input(x)
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        return pred_feat, target_feat

    @torch.no_grad()
    def update_target(self, tau: float = 0.01) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in the range [0, 1].")
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
        if not torch.isfinite(loss):
            raise RuntimeError("NoveltyDetector loss became non-finite.")
        loss.backward()
        if self.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.predictor.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        return float(loss.item())


# -------------------------------------------------------------------------
# Factories
# -------------------------------------------------------------------------
def create_policy_network(
    input_dim: int,
    output_dim: int,
    config: Optional[Dict[str, Any]] = None,
) -> PolicyNetwork:
    """Create a PolicyNetwork from config, preserving backward compatibility."""
    if config is None:
        full_config = load_global_config()
        config = get_config_section("policy_network")
        if not config:
            config = full_config.get("neural_network", {})

    return PolicyNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=config.get("hidden_layer_sizes", config.get("layer_dims", [input_dim, 128, 64, output_dim])[1:-1]),
        hidden_activation=config.get("hidden_activation", "relu"),
        output_activation=config.get("output_activation", "softmax"),
        use_batch_norm=config.get("use_batch_norm", False),
        dropout_rate=config.get("dropout_rate", 0.0),
        l1_lambda=config.get("l1_lambda", 0.0),
        l2_lambda=config.get("l2_lambda", 0.0),
        weight_init=config.get("weight_init", "auto"),
    )


def create_policy_optimizer(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
) -> optim.Optimizer:
    """Create an optimiser for a policy network using the policy config section."""
    if config is None:
        config = get_config_section("policy_network")

    optimizer_config = config.get("optimizer_config", {})
    optimizer_type = str(optimizer_config.get("type", "adam")).lower()
    learning_rate = float(optimizer_config.get("learning_rate", 1e-3))
    weight_decay = float(optimizer_config.get("weight_decay", config.get("l2_lambda", 0.0)))

    if optimizer_type == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_type == "momentum":
        momentum_beta = float(optimizer_config.get("momentum_beta", 0.9))
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_beta, weight_decay=weight_decay)
    if optimizer_type == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(
                float(optimizer_config.get("adam_beta1", 0.9)),
                float(optimizer_config.get("adam_beta2", 0.999)),
            ),
            eps=float(optimizer_config.get("adam_epsilon", 1e-8)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported policy optimizer type: {optimizer_type}")


if __name__ == "__main__":
    print("Testing PolicyNetwork...")
    net = PolicyNetwork(input_dim=4, output_dim=2)
    dummy = torch.randn(10, 4)
    out = net(dummy)
    print(f"Output shape: {out.shape} (should be [10, 2])")
    print(f"Softmax row sums: {out.sum(dim=1)}")

    nd = NoveltyDetector(input_dim=4, feature_dim=8)
    x = torch.randn(5, 4)
    novelty = nd(x)
    print(f"Novelty scores shape: {novelty.shape}")
    loss = nd.train_step(x)
    print(f"Train loss: {loss:.4f}")
    print("All tests passed.")
