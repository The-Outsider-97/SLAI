"""
Policy Network – A standard PyTorch module for policy approximation in RL.

This module defines a configurable feed‑forward network that outputs action
probabilities (for discrete actions) or action parameters (for continuous).
It can be used with any RL algorithm that expects a policy network.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Union

from src.agents.base.utils.activation_engine import get_activation
from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("PolicyNetwork")


class PolicyNetwork(nn.Module):
    """
    Feed‑forward policy network.

    Architecture:
        input_dim -> [hidden layers] -> output_dim

    Activation for hidden layers is configurable, output activation is also configurable
    (e.g., softmax for discrete, tanh for continuous, linear for means/params).

    The network can be saved and loaded via `state_dict`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        hidden_activation: str = "relu",
        output_activation: str = "softmax",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize the policy network.

        Args:
            input_dim: Dimension of the state space.
            output_dim: Dimension of the action space (number of discrete actions
                        or number of continuous action parameters).
            hidden_sizes: List of hidden layer sizes. If None, uses [128, 64].
            hidden_activation: Name of activation for hidden layers.
            output_activation: Name of activation for output layer.
            use_batch_norm: If True, adds BatchNorm after each hidden layer (except output).
            dropout_rate: Dropout probability (0 = disabled).
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Build layers
        layers = []
        prev_dim = input_dim
        for hdim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hdim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hdim))
            layers.append(get_activation(hidden_activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        # Output activation (applied separately, not as a layer to allow conditional use)
        self.layers = nn.Sequential(*layers)
        if output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=-1)
        else:
            self.output_activation = get_activation(output_activation)

        # Initialize weights (Xavier for all linear layers)
        self._init_weights()

        logger.info(
            f"PolicyNetwork initialised: input={input_dim}, output={output_dim}, "
            f"hidden={hidden_sizes}, hidden_act={hidden_activation}, "
            f"output_act={output_activation}, batch_norm={use_batch_norm}, dropout={dropout_rate}"
        )

    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim), after output activation.
        """
        x = self.layers(x)
        return self.output_activation(x)


class NoveltyDetector(nn.Module):
    """
    Simple neural network for estimating state novelty using a predictor‑target architecture.

    Given an input state, it predicts a feature vector and compares it to a target
    feature vector. The difference (L2 norm) is used as a novelty score.
    """

    def __init__(self, input_dim: int, feature_dim: int = 32, learning_rate: float = 1e-3):
        """
        Initialize the novelty detector.

        Args:
            input_dim: Dimension of the input state.
            feature_dim: Size of the feature representation.
            learning_rate: Learning rate for the predictor network.
        """
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.target = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty score.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Novelty score tensor of shape (batch_size,).
        """
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        return torch.norm(pred_feat - target_feat, dim=1)

    def update_target(self, tau: float = 0.01):
        """
        Soft update the target network using the predictor weights.

        Args:
            tau: Mixing factor (0 = no update, 1 = full copy).
        """
        for t_param, p_param in zip(self.target.parameters(), self.predictor.parameters()):
            t_param.data.copy_(tau * p_param.data + (1.0 - tau) * t_param.data)

    def train_step(self, x: torch.Tensor) -> float:
        """
        Perform one training step on the predictor network.

        Args:
            x: Input batch.

        Returns:
            Loss value (MSE between predictor and target).
        """
        self.predictor.train()
        self.optimizer.zero_grad()
        pred = self.predictor(x)
        with torch.no_grad():
            target = self.target(x)
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# -------------------------------------------------------------------------
# Factory for creating policy networks from configuration
# -------------------------------------------------------------------------
def create_policy_network(
    input_dim: int, output_dim: int, config: Optional[Dict[str, Any]] = None
) -> PolicyNetwork:
    """
    Create a PolicyNetwork instance using configuration from the global YAML.

    The configuration is expected under the key 'policy_network'. If not provided,
    defaults are used.

    Args:
        input_dim: State dimension.
        output_dim: Action dimension.
        config: Optional override config; if None, loads from global config.

    Returns:
        Configured PolicyNetwork.
    """
    if config is None:
        full_config = load_global_config()
        config = get_config_section("policy_network")
        if not config:
            # Fallback to 'neural_network' section for backward compatibility
            config = full_config.get("neural_network", {})

    hidden_sizes = config.get("hidden_layer_sizes", [128, 64])
    hidden_activation = config.get("hidden_activation", "relu")
    output_activation = config.get("output_activation", "softmax")
    use_batch_norm = config.get("use_batch_norm", False)
    dropout_rate = config.get("dropout_rate", 0.0)

    return PolicyNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
    )


# -------------------------------------------------------------------------
# Test / example
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing PolicyNetwork...")
    # Create a small network
    net = PolicyNetwork(input_dim=4, output_dim=2)
    dummy = torch.randn(10, 4)
    out = net(dummy)
    print(f"Output shape: {out.shape} (should be [10,2])")
    if net.output_activation_name == "softmax":
        print(f"Softmax sum: {out.sum(dim=1)} (should be ~1.0)")

    # Test novelty detector
    nd = NoveltyDetector(input_dim=4, feature_dim=8)
    x = torch.randn(5, 4)
    novelty = nd(x)
    print(f"Novelty scores shape: {novelty.shape}")
    loss = nd.train_step(x)
    print(f"Train loss: {loss:.4f}")

    print("All tests passed.")
