import json
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

from src.agents.base.utils.activation_engine import (ReLU, LeakyReLU, ELU, Swish, Mish, GELU, Tanh,
                                                    Sigmoid, Linear, he_init, lecun_normal, Softmax,
                                                    xavier_uniform)
from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Adaptive Neural Network")
printer = PrettyPrinter

class ActivationWrapper(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation.forward(x)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.problem_type = self.config.get('problem_type', 'regression').lower()
        self.dim = self.config.get('final_activation_dim', -1)

        self.nn_config = get_config_section('neural_network')
        self.input_dim = self.nn_config.get('input_dim')
        self.layer_config = self.nn_config.get('layer_config')
        self.final_activation_name = self.nn_config.get(
            'final_activation',
            'sigmoid' if self.problem_type == 'binary_classification' else
            'softmax' if self.problem_type == 'multiclass_classification' else
            'linear'
        )
        self.final_activation = self._get_activation(self.final_activation_name, config={})

        # Get config values
        self.layers = nn.ModuleList()
        self._build_network(self.input_dim, self.layer_config)

        # Only proceed if there are parameters
        if any(p.requires_grad for p in self.parameters()):
            self.optimizer_name = self.nn_config.get('optimizer_name', 'adam').lower()
            self._configure_optimizer()
            self._configure_loss_function()
        else:
            raise RuntimeError("No trainable parameters found in the network.")

        logger.info(f"Neural Network succesfully initialized")

    def _build_network(self, input_dim: int, layer_config: List[Dict[str, Any]]):
        printer.status("INIT", "Network builder succesfully initialized", "info")

        current_dim = input_dim
        for i, layer_conf in enumerate(layer_config):
            # Create linear layer
            neurons = layer_conf['neurons']
            linear = nn.Linear(current_dim, neurons)
    
            # Apply weight initialization
            init_method = layer_conf.get('init', self.nn_config.get('initialization_method_default', 'he_normal'))
            self._init_weights(linear, init_method)
    
            # Add linear layer
            self.layers.append(linear)
    
            # Add activation if applicable
            if i < len(layer_config) - 1 or self.problem_type == 'regression':
                activation_name = layer_conf.get('activation', 'relu')
                activation = self._get_activation(activation_name, layer_conf)
                self.layers.append(ActivationWrapper(activation))
    
            # Dropout
            dropout_rate = layer_conf.get('dropout', 0.0)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
    
            # Batch normalization
            if layer_conf.get('batch_norm', False):
                self.layers.append(nn.BatchNorm1d(neurons))
    
            current_dim = neurons

    def _init_weights(self, layer: nn.Linear, init_method: str):
        printer.status("INIT", "Weight succesfully initialized", "info")

        shape = (layer.out_features, layer.in_features)
        device = layer.weight.device  # Ensure same device
    
        if init_method == 'uniform_scaled':
            limit = 1.0 / math.sqrt(layer.in_features)
            with torch.no_grad():
                layer.weight.copy_(torch.empty(shape, device=device).uniform_(-limit, limit))
        elif init_method == 'he_normal':
            with torch.no_grad():
                layer.weight.copy_(he_init(shape, nonlinearity='relu', device=device))
        elif init_method == 'xavier_uniform':
            with torch.no_grad():
                layer.weight.copy_(xavier_uniform(shape, device=device))
        elif init_method == 'lecun_normal':
            with torch.no_grad():
                layer.weight.copy_(lecun_normal(shape, device=device))
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
    
        nn.init.constant_(layer.bias, 0.0)

    def _get_activation(self, name: str, config: Dict[str, Any]):
        printer.status("INIT", "Activation succesfully initialized", "info")

        alpha = config.get('alpha', 0.01)

        activation_map = {
            'relu': ReLU(),
            'leaky_relu': LeakyReLU(alpha),
            'elu': ELU(alpha),
            'swish': Swish(),
            'mish': Mish(),
            'gelu': GELU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'linear': Linear(),
            'softmax': Softmax(dim=self.dim)
        }
        if name in activation_map:
            return activation_map[name]
        raise ValueError(f"Unsupported activation: {name}")

    def _configure_optimizer(self):
        printer.status("INIT", "Config Optim succesfully initialized", "info")

        lr = self.config.get('parameter_tuner', {}).get('base_learning_rate')
        weight_decay = self.config.get('parameter_tuner', {}).get('weight_decay_lambda')

        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=lr,
                weight_decay=weight_decay
            )
        elif self.optimizer_name == 'sgd_momentum_adagrad':
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _configure_loss_function(self):
        printer.status("INIT", "Loss functions succesfully initialized", "info")

        loss_name = self.nn_config.get('loss_function_name', 'mse').lower()

        if loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_name == 'cross_entropy':
            if self.problem_type == 'binary_classification':
                self.loss_fn = nn.BCEWithLogitsLoss()
            elif self.problem_type == 'multiclass_classification':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError("Cross-entropy requires classification problem")
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        printer.status("INIT", "Forward succesfully initialized", "info")

        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
            
        for layer in self.layers:
            x = layer(x)
            
        # Apply final activation
        return self.final_activation.forward(x)

    def train_network(
        self, 
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        batch_size: Optional[int] = None,
        validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        early_stopping_patience: Optional[int] = None,
        verbose: bool = True
    ):
        printer.status("INIT", "Trainer succesfully initialized", "info")

        self.train()
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            random.shuffle(training_data)
            
            if batch_size is None:
                # Full batch training
                inputs, targets = zip(*training_data)
                inputs = torch.FloatTensor(np.array(inputs))
                targets = self._prepare_targets(targets)
                
                loss = self._train_step(inputs, targets)
                epoch_loss = loss.item()
            else:
                # Mini-batch training
                for i in range(0, len(training_data), batch_size):
                    batch = training_data[i:i+batch_size]
                    inputs, targets = zip(*batch)
                    inputs = torch.FloatTensor(np.array(inputs))
                    targets = self._prepare_targets(targets)
                    
                    loss = self._train_step(inputs, targets)
                    epoch_loss += loss.item()
                epoch_loss /= (len(training_data) / batch_size)
            
            # Validation
            val_loss = None
            if validation_data:
                val_loss = self.evaluate(validation_data)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if verbose:
                val_msg = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}{val_msg}")

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        printer.status("INIT", "Step trainer succesfully initialized", "info")

        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        clip_value = self.config.get('gradient_clip_value', 1.0)
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
        
        self.optimizer.step()
        return loss

    def _prepare_targets(self, targets: List[np.ndarray]) -> torch.Tensor:
        printer.status("INIT", "Prepper succesfully initialized", "info")

        if self.problem_type == 'multiclass_classification':
            # Convert to class indices for CrossEntropyLoss
            return torch.LongTensor([np.argmax(t) for t in targets])
        return torch.FloatTensor(np.array(targets))

    def evaluate(self, test_data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        printer.status("INIT", "Evaluator succesfully initialized", "info")

        self.eval()
        inputs, targets = zip(*test_data)
        inputs = torch.FloatTensor(np.array(inputs))
        targets = self._prepare_targets(targets)
        
        with torch.no_grad():
            outputs = self(inputs)
            loss = self.loss_fn(outputs, targets)
        return loss.item()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        printer.status("INIT", "Predictor succesfully initialized", "info")

        self.eval()
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(inputs)
            outputs = self(inputs_tensor).numpy()
        return outputs

    def predict_class(self, inputs: np.ndarray) -> np.ndarray:
        outputs = self.predict(inputs)
        if self.problem_type == 'binary_classification':
            return (outputs > 0.5).astype(int)
        elif self.problem_type == 'multiclass_classification':
            return np.argmax(outputs, axis=1)
        return outputs

    def save_model(self, filepath: str):
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'layer_config': [layer for layer in self.layers if isinstance(layer, nn.Linear)]
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        checkpoint = torch.load(filepath)
        # Extract layer config from saved model
        layer_config = []
        for layer in checkpoint['layer_config']:
            layer_config.append({
                'neurons': layer.out_features,
                # Add other params as needed
            })
            
        # Create model instance
        input_dim = checkpoint['layer_config'][0].in_features
        model = cls()
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def get_weights_biases(self) -> List[Dict[str, Any]]:
        """Extract weights and biases from all linear layers in the network
        
        Returns:
            List of dictionaries containing:
            - 'weights': 2D array of weights (out_features x in_features)
            - 'bias': 1D array of biases
        """
        weights_biases = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weights = layer.weight.data.cpu().numpy()
                bias = layer.bias.data.cpu().numpy()
                weights_biases.append({
                    'weights': weights,
                    'bias': bias
                })
        return weights_biases

    def set_weights_biases(self, weights_biases: List[Dict[str, Any]]):
        """Set weights and biases for all linear layers in the network
        
        Args:
            weights_biases: List of layer parameters in same order as network layers
                Each item should contain:
                - 'weights': 2D array (out_features x in_features)
                - 'bias': 1D array
        """
        layer_idx = 0
        device = next(self.parameters()).device
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if layer_idx >= len(weights_biases):
                    raise ValueError("Insufficient weight data for network layers")
                
                # Get parameters for current layer
                wb = weights_biases[layer_idx]
                weights = wb['weights']
                bias = wb['bias']
                
                # Validate shape compatibility
                if weights.shape != (layer.out_features, layer.in_features):
                    raise ValueError(f"Layer {layer_idx} weight shape mismatch: "
                                     f"Expected {(layer.out_features, layer.in_features)}, "
                                     f"Got {weights.shape}")
                
                if bias.shape != (layer.out_features,):
                    raise ValueError(f"Layer {layer_idx} bias shape mismatch: "
                                     f"Expected {(layer.out_features,)}, "
                                     f"Got {bias.shape}")
                
                # Set parameters
                with torch.no_grad():
                    layer.weight.data = torch.tensor(
                        weights, dtype=torch.float32, device=device
                    )
                    layer.bias.data = torch.tensor(
                        bias, dtype=torch.float32, device=device
                    )
                
                layer_idx += 1
                
        if layer_idx != len(weights_biases):
            logger.warning("Extra weight data provided but not used")

# BayesianDQN with MC Dropout
class BayesianDQN(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.dropout_rate = self.config.get('dropout_rate')

        self.dqn_config = get_config_section('bayesian_dqn')
        self.num_uncertainty_samples = self.dqn_config.get('num_uncertainty_samples')
        self.uncertainty_threshold = self.dqn_config.get('uncertainty_threshold')
        
        self._enable_dropout()
        logger.info(f"Initialized Bayesian Deep-Q Network with dropout: {self.dropout_rate}")

    def _enable_dropout(self):
        """Enable and configure dropout layers for uncertainty estimation"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
                module.train()  # Always enable dropout for uncertainty estimation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bayesian forward pass with dropout enabled"""
        self._enable_dropout()
        return super().forward(x)

    def estimate_uncertainty(self, state: np.ndarray, num_samples: Optional[int] = None) -> np.ndarray:
        """Estimate prediction uncertainty using Monte Carlo dropout
        Args:
            state: Input state (single sample or batch)
            num_samples: Number of forward passes (default from config)
        Returns:
            Uncertainty estimates (std dev) per output dimension
        """
        num_samples = num_samples or self.num_uncertainty_samples
        self._enable_dropout()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            # Collect multiple predictions
            predictions = torch.stack([self(state_tensor) for _ in range(num_samples)])
            
            # Calculate standard deviation across samples
            std_dev = predictions.std(dim=0).cpu().numpy()
            
            # Apply uncertainty threshold
            if self.uncertainty_threshold:
                std_dev = np.clip(std_dev, 0, self.uncertainty_threshold)
                
            return std_dev

    def get_uncertainty_metrics(self, state: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive uncertainty metrics
        Returns:
            Dictionary with:
            - mean: Mean prediction across samples
            - std: Standard deviation of predictions
            - confidence: 95% confidence interval
            - uncertainty_flag: True if uncertainty exceeds threshold
        """
        num_samples = self.num_uncertainty_samples
        self._enable_dropout()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            predictions = torch.stack([self(state_tensor) for _ in range(num_samples)])
            mean = predictions.mean(dim=0).cpu().numpy()
            std = predictions.std(dim=0).cpu().numpy()
            
            # Calculate 95% confidence interval
            confidence = 1.96 * std / np.sqrt(num_samples)
            
            return {
                'mean': mean,
                'std': std,
                'confidence_interval': (mean - confidence, mean + confidence),
                'uncertainty_flag': np.any(std > self.uncertainty_threshold)
            }

    def predict_with_uncertainty(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get both prediction and uncertainty estimate
        Returns:
            tuple: (prediction, uncertainty)
        """
        prediction = self.predict(inputs)
        uncertainty = self.estimate_uncertainty(inputs)
        return prediction, uncertainty

    def save_model(self, filepath: str):
        """Save BayesianDQN-specific parameters"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'layer_config': [layer for layer in self.layers if isinstance(layer, nn.Linear)],
            'dropout_rate': self.dropout_rate,
            'uncertainty_config': {
                'num_samples': self.num_uncertainty_samples,
                'threshold': self.uncertainty_threshold
            }
        }, filepath)
        logger.info(f"BayesianDQN saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load BayesianDQN with custom parameters"""
        checkpoint = torch.load(filepath)
        model = cls(dropout_rate=checkpoint.get('dropout_rate', 0.1))
        model.load_state_dict(checkpoint['state_dict'])
        
        # Load uncertainty config
        uncertainty_config = checkpoint.get('uncertainty_config', {})
        model.num_uncertainty_samples = uncertainty_config.get('num_samples', 10)
        model.uncertainty_threshold = uncertainty_config.get('threshold', 0.1)
        
        return model

# Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, 
                 actor_layers: Union[List[int], List[Dict[str, Any]]], 
                 critic_layers: Union[List[int], List[Dict[str, Any]]]):
        super().__init__()
        self.config = load_global_config()
        self.initialization_method_default = self.config.get('initialization_method_default')
        self.problem_type = self.config.get('problem_type', 'regression').lower()
        self.dim = self.config.get('final_activation_dim', -1)

        self.acn_config = get_config_section('actor_critic')
        
        # Handle both integer and dictionary layer specifications
        self.actor_layers = self._normalize_layer_config(actor_layers)
        self.critic_layers = self._normalize_layer_config(critic_layers)
        
        self.shared_base = self.acn_config.get('shared_base', False)
        self.shared_layers = self.acn_config.get('shared_layers', [])
        self.continuous_action = self.acn_config.get('continuous_action', False)
        self.initial_std = self.acn_config.get('initial_std', 0.5)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build networks
        if self.shared_base:
            self.base_network = self._build_network(state_dim, self.shared_layers)
            base_output_dim = self.shared_layers[-1]['neurons'] if self.shared_layers else state_dim
            self.actor = self._build_network(base_output_dim, self.actor_layers)
            self.critic = self._build_network(base_output_dim, self.critic_layers)
        else:
            self.actor = self._build_network(state_dim, self.actor_layers)
            self.critic = self._build_network(state_dim, self.critic_layers)
        
        # Policy type configuration
        self.action_std = None
        if self.continuous_action:
            self.action_std = nn.Parameter(torch.ones(1, action_dim) * self.initial_std)
        
        logger.info(f"Initialized Actor-Critic Network | Shared base: {self.shared_base} | Continuous: {self.continuous_action}")

    def _normalize_layer_config(self, layers: Union[List[int], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Convert integer layer specs to dictionary format with defaults"""
        normalized = []
        for layer in layers:
            if isinstance(layer, int):
                normalized.append({
                    'neurons': layer,
                    'activation': 'relu',
                    'batch_norm': False,
                    'dropout': 0.0
                })
            elif isinstance(layer, dict):
                normalized.append(layer)
            else:
                raise TypeError(f"Invalid layer specification type: {type(layer)}")
        return normalized

    def _build_network(self, input_dim: int, layer_config: List[Dict[str, Any]]) -> nn.Sequential:
        """Build a network using the same configuration system as NeuralNetwork"""
        self.layers = []
        current_dim = input_dim
        
        for i, layer_conf in enumerate(layer_config):
            # Create linear layer
            neurons = layer_conf['neurons']
            linear = nn.Linear(current_dim, neurons)
    
            # Apply weight initialization
            init_method = layer_conf.get('init', self.acn_config.get('initialization_method_default', 'he_normal'))
            self._init_weights(linear, init_method)
    
            # Add linear layer
            self.layers.append(linear)
    
            # Add activation if applicable
            if i < len(layer_config) - 1 or self.problem_type == 'regression':
                activation_name = layer_conf.get('activation', 'relu')
                activation = self._get_activation(activation_name, layer_conf)
                self.layers.append(ActivationWrapper(activation))
    
            # Dropout
            dropout_rate = layer_conf.get('dropout', 0.0)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
    
            # Batch normalization
            if layer_conf.get('batch_norm', False):
                self.layers.append(nn.BatchNorm1d(neurons))
    
            current_dim = neurons
            
        return nn.Sequential(*self.layers)

    def _init_weights(self, layer: nn.Linear, init_method: str):
        """Consistent weight initialization with NeuralNetwork"""
        shape = (layer.out_features, layer.in_features)
        device = layer.weight.device  # Ensure same device
    
        if init_method == 'uniform_scaled':
            limit = 1.0 / math.sqrt(layer.in_features)
            with torch.no_grad():
                layer.weight.copy_(torch.empty(shape, device=device).uniform_(-limit, limit))
        elif init_method == 'he_normal':
            with torch.no_grad():
                layer.weight.copy_(he_init(shape, nonlinearity='relu', device=device))
        elif init_method == 'xavier_uniform':
            with torch.no_grad():
                layer.weight.copy_(xavier_uniform(shape, device=device))
        elif init_method == 'lecun_normal':
            with torch.no_grad():
                layer.weight.copy_(lecun_normal(shape, device=device))
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

    def _get_activation(self, name: str, config: Dict[str, Any]) -> nn.Module:
        """Consistent activation setup with NeuralNetwork"""
        alpha = config.get('alpha', 0.01)
    
        activation_map = {
            'relu': ReLU(),
            'leaky_relu': LeakyReLU(alpha),
            'elu': ELU(alpha),
            'swish': Swish(),
            'mish': Mish(),
            'gelu': GELU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'linear': Linear(),
            'softmax': Softmax(dim=self.dim)
        }
        if name in activation_map:
            return activation_map[name]
        raise ValueError(f"Unsupported activation: {name}")

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network only"""
        if self.shared_base:
            x = self.base_network(x)
        return self.actor(x)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network only"""
        if self.shared_base:
            x = self.base_network(x)
        return self.critic(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both networks"""
        if self.shared_base:
            base_out = self.base_network(x)
            return self.actor(base_out), self.critic(base_out)
        return self.actor(x), self.critic(x)

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and get log probability"""
        actor_output = self.actor(state)
        
        if self.continuous_action:
            # For continuous action spaces
            action_mean = actor_output
            action_std = self.action_std.expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
            return action, log_prob
        else:
            # For discrete action spaces
            probs = F.softmax(actor_output, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action, dist.log_prob(action)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value from critic"""
        return self.critic(state)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for policy updates"""
        actor_out, critic_out = self(states)
        
        if self.continuous_action:
            # Continuous actions
            action_std = self.action_std.expand_as(actor_out)
            dist = torch.distributions.Normal(actor_out, action_std)
            log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
            entropy = dist.entropy().mean()
        else:
            # Discrete actions
            probs = F.softmax(actor_out, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
        return log_probs, critic_out.squeeze(-1), entropy

    def save_model(self, filepath: str):
        """Save ActorCritic-specific parameters"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'actor_config': self.actor_layers,
            'critic_config': self.critic_layers,
            'shared_base': self.shared_base,
            'continuous_action': self.continuous_action,
            'action_std': self.action_std.data if self.continuous_action else None
        }, filepath)
        logger.info(f"ActorCritic saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load ActorCritic with custom parameters"""
        checkpoint = torch.load(filepath)
        model = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            actor_config=checkpoint['actor_config'],
            critic_config=checkpoint['critic_config'],
            shared_base=checkpoint['shared_base']
        )
        model.load_state_dict(checkpoint['state_dict'])
        if model.continuous_action and checkpoint['action_std'] is not None:
            model.action_std.data = checkpoint['action_std']
        return model


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("\n=== Running Adaptive Neural Network ===\n")
    printer.status("Init", "Adaptive Neural Network initialized", "success")
    rate=0.01
    input_dim=10
    action_dim=10
    actor_layers=[64, 64]
    critic_layers=[64, 32]

    network = NeuralNetwork()
    dqn = BayesianDQN()
    acn = ActorCriticNetwork(state_dim=input_dim,
            action_dim=action_dim,
            actor_layers= actor_layers,
            critic_layers= critic_layers)
    print(network)
    print(dqn)
    print(acn)

    print("\n* * * * * Phase 2 * * * * *\n")
    network.eval()
    x=torch.randn(1, network.input_dim)

    printer.status("forward", network.forward(x=x), "success")

    print("\n* * * * * Phase 3 * * * * *\n")
    training_data = [(np.random.rand(network.input_dim), np.array([1])) for _ in range(10)]
    epochs=1
    batch_size = 2
    validation_data = None
    early_stopping_patience = None
    verbose = True

    trainer = network.train_network(epochs=epochs,
        training_data=training_data,
        batch_size=batch_size,
        validation_data=validation_data,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose)

    printer.status("forward", trainer, "success")

    print("\n* * * * * Phase 4 * * * * *\n")
    targets=[]

    printer.status("forward", network._prepare_targets(targets=targets), "success")

    print("\n=== Finished Running Adaptive Neural Network ===\n")
