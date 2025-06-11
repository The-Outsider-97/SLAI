import json
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Adaptive Neural Network")
printer = PrettyPrinter

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.nn_config = get_config_section('neural_network')
        self.input_dim = self.nn_config.get('input_dim')
        self.layer_config = self.nn_config.get('layer_config')

        # Get config values
        self.problem_type = self.nn_config.get('problem_type', 'regression').lower()
        self.layers = nn.ModuleList()
        self._build_network(self.input_dim, self.layer_config)

        # Only proceed if there are parameters
        if any(p.requires_grad for p in self.parameters()):
            self.optimizer_name = self.nn_config.get('optimizer_name', 'adam').lower()
            self._configure_optimizer()
            self._configure_loss_function()
        else:
            raise RuntimeError("No trainable parameters found in the network.")

        logger.info(f"Initialized Neural Network: {self}")

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
            
            # Add to layers
            self.layers.append(linear)
            
            # Add activation (except for last layer in classification)
            if i < len(layer_config) - 1 or self.problem_type == 'regression':
                activation_name = layer_conf.get('activation', 'relu')
                self.layers.append(self._get_activation(activation_name, layer_conf))
            
            # Add dropout
            dropout_rate = layer_conf.get('dropout', 0.0)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            
            # Add batch norm
            if layer_conf.get('batch_norm', False):
                self.layers.append(nn.BatchNorm1d(neurons))
            
            current_dim = neurons

    def _init_weights(self, layer: nn.Linear, init_method: str):
        printer.status("INIT", "Weight succesfully initialized", "info")

        if init_method == 'uniform_scaled':
            limit = 1.0 / math.sqrt(layer.in_features)
            nn.init.uniform_(layer.weight, -limit, limit)
        elif init_method == 'he_normal':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(layer.weight)
        elif init_method == 'lecun_normal':
            stddev = math.sqrt(1.0 / layer.in_features)
            nn.init.normal_(layer.weight, 0, stddev)
        nn.init.constant_(layer.bias, 0.0)

    def _get_activation(self, name: str, config: Dict[str, Any]):
        printer.status("INIT", "Activation succesfully initialized", "info")

        alpha = config.get('alpha', 0.01)
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=alpha)
        elif name == 'elu':
            return nn.ELU(alpha=alpha)
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'linear':
            return nn.Identity()
        else:
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
        if self.problem_type == 'binary_classification':
            return torch.sigmoid(x)
        elif self.problem_type == 'multiclass_classification':
            return F.softmax(x, dim=-1)
        return x

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
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self._enable_dropout()

        logger.info(f"Initialized Bayesian Deep-Q Network: {self}")

    def _enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
                module.train()  # Always in train mode to enable dropout

    def estimate_uncertainty(self, state: np.ndarray, num_samples: int = 10) -> np.ndarray:
        self._enable_dropout()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            outputs = [self(state_tensor).numpy() for _ in range(num_samples)]
        return np.std(outputs, axis=0)
    
    def get_weights_biases(self) -> List[Dict[str, Any]]:
        """BayesianDQN-specific implementation with dropout layers ignored"""
        return super().get_weights_biases()

# Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, actor_layers: List[int], critic_layers: List[int]):
        super().__init__()
        self.actor = self._build_network(state_dim, action_dim, actor_layers)
        self.critic = self._build_network(state_dim, 1, critic_layers)

        logger.info(f"Initialized Actor Critic Network: {self}")

    def _build_network(self, input_dim: int, output_dim: int, layer_sizes: List[int]):
        layers = []
        prev_size = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_dim))
        return nn.Sequential(*layers)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("\n=== Running Adaptive Neural Network ===\n")
    printer.status("Init", "Adaptive Neural Network initialized", "success")
    rate=0.01

    network = NeuralNetwork()
    dqn = BayesianDQN(dropout_rate=rate)
    print(network)
    print(dqn)

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
