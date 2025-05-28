
import math
import random
import yaml, json
import numpy as np

from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any, Union

from src.agents.adaptive.utils.math_science import (
        sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative,
        leaky_relu, leaky_relu_derivative, elu, elu_derivative, swish, swish_derivative,
        softmax, cross_entropy as cross_entropy_loss_func, cross_entropy_derivative)
from logs.logger import get_logger

logger = get_logger("Cyber-Security Neural-Network")

CONFIG_PATH = "src/agents/safety/configs/secure_config.yaml"

def load_config(config_path=CONFIG_PATH):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file not found at {config_path}. Returning empty config.")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return {}

def get_merged_config(user_config=None):
    """Merges a base configuration with a user-provided configuration."""
    base_config = load_config()
    if user_config:
        # A simple update; for deep merging, a more sophisticated approach might be needed
        for key, value in user_config.items():
            if isinstance(value, dict) and isinstance(base_config.get(key), dict):
                base_config[key].update(value)
            else:
                base_config[key] = value
    return base_config

# --- Activation Functions and their Derivatives (Referenced from math_science) ---
ACTIVATION_FUNCTIONS: Dict[str, Tuple[Callable, Callable, bool]] = {
    'sigmoid': (sigmoid, sigmoid_derivative, False),
    'relu': (relu, relu_derivative, False),
    'tanh': (tanh, tanh_derivative, False),
    'leaky_relu': (leaky_relu, leaky_relu_derivative, True),
    'elu': (elu, elu_derivative, True),
    'swish': (swish, swish_derivative, False),
    # 'softmax' is handled at the layer/network level, not as a per-neuron activation in this map
    'linear': (lambda x: x, lambda x: 1.0, False) # Identity function
}

# --- Loss Function (Using mean_squared_error for regression or autoencoders) ---
def mean_squared_error(targets: List[float], outputs: List[float]) -> float:
    """L = 0.5 * sum((target_i - output_i)^2)"""
    if len(targets) != len(outputs):
        raise ValueError("Targets and outputs must have the same length for MSE.")
    return 0.5 * sum([(target - output) ** 2 for target, output in zip(targets, outputs)])

def mean_squared_error_derivative(targets: List[float], outputs: List[float]) -> List[float]:
    """ Derivative of MSE w.r.t. outputs: (output_i - target_i) """
    if len(targets) != len(outputs):
        raise ValueError("Targets and outputs must have the same length for MSE derivative.")
    return [(output - target) for target, output in zip(targets, outputs)]


# --- Neuron: The Basic Processing Unit for Security Feature Analysis ---
class Neuron:
    """
    Represents a single neuron in a neural network layer.
    In a cybersecurity context, a neuron processes weighted inputs (features derived
    from network traffic, system logs, etc.) and applies an activation function
    to produce an output, contributing to pattern recognition for threat detection.
    """
    def __init__(self, num_inputs: int,
                 activation_name: str = 'relu',
                 initialization_method: str = 'he_normal',
                 activation_alpha: float = 0.01):
        super().__init__()
        
        if activation_name not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unsupported activation: {activation_name}. Supported: {list(ACTIVATION_FUNCTIONS.keys())}")
        
        self.activation_fn_ptr, self.activation_fn_derivative_ptr, self.activation_needs_alpha = ACTIVATION_FUNCTIONS[activation_name]
        
        self.num_inputs = num_inputs
        self.initialization_method = initialization_method
        self.activation_alpha = activation_alpha

        self.weights: List[float] = self._initialize_weights()
        self.bias: float = random.uniform(-0.1, 0.1)

        self.inputs: List[float] = [0.0] * num_inputs
        self.weighted_sum: float = 0.0
        self.activation: float = 0.0
        self.delta: float = 0.0

        self.velocity_weights: List[float] = [0.0] * num_inputs
        self.velocity_bias: float = 0.0
        self.cache_weights: List[float] = [0.0] * num_inputs
        self.cache_bias: float = 0.0
        self.m_weights: List[float] = [0.0] * num_inputs
        self.v_weights: List[float] = [0.0] * num_inputs
        self.m_bias: float = 0.0
        self.v_bias: float = 0.0

    def _initialize_weights(self) -> List[float]:
        if self.num_inputs == 0: # Should not happen for a connected neuron
            return []
        if self.initialization_method == 'uniform_scaled':
            limit = 1.0 / math.sqrt(self.num_inputs)
            return [random.uniform(-limit, limit) for _ in range(self.num_inputs)]
        elif self.initialization_method == 'he_normal': # Good for ReLU-like activations
            stddev = math.sqrt(2.0 / self.num_inputs)
            return [random.gauss(0, stddev) for _ in range(self.num_inputs)]
        elif self.initialization_method == 'lecun_normal': # Good for Tanh-like activations
            stddev = math.sqrt(1.0 / self.num_inputs)
            return [random.gauss(0, stddev) for _ in range(self.num_inputs)]
        elif self.initialization_method == 'xavier_uniform': # Glorot uniform
            limit = math.sqrt(6.0 / (self.num_inputs + 1)) # Assuming 1 output unit conceptually for this neuron
            return [random.uniform(-limit, limit) for _ in range(self.num_inputs)]
        else: # Fallback to a simple random initialization
            logger.warning(f"Unsupported weight initialization: {self.initialization_method}. Falling back to small random uniform.")
            return [random.uniform(-0.1, 0.1) for _ in range(self.num_inputs)]

    def _call_activation_fn(self, x: float) -> float:
        if self.activation_needs_alpha:
            return self.activation_fn_ptr(x, self.activation_alpha)
        return self.activation_fn_ptr(x)

    def _call_activation_fn_derivative(self, x: float) -> float:
        if self.activation_needs_alpha:
            return self.activation_fn_derivative_ptr(x, self.activation_alpha)
        return self.activation_fn_derivative_ptr(x)

    def _calculate_weighted_sum(self, inputs: List[float]) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError(f"Neuron expected {len(self.weights)} inputs, got {len(inputs)}.")
        self.inputs = inputs 
        self.weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.weighted_sum

    def activate(self, inputs: List[float]) -> float:
        z = self._calculate_weighted_sum(inputs)
        self.activation = self._call_activation_fn(z)
        return self.activation

    def calculate_delta(self, error_signal_from_downstream: float):
        """Calculates the error term (delta) for this neuron. dE/dz."""
        self.delta = error_signal_from_downstream * self._call_activation_fn_derivative(self.weighted_sum)

    def calculate_gradients(self,
                            weight_decay_lambda: float = 0.0,
                            gradient_clip_value: Optional[float] = None
                           ) -> Tuple[List[float], float]:
        """Calculates gradients for weights and bias, including L2 regularization and clipping."""
        grad_weights_loss = [self.delta * inp for inp in self.inputs]
        grad_bias_loss = self.delta 
        
        # L2 Regularization (Weight Decay)
        regularized_grad_weights = [
            gw_loss + weight_decay_lambda * w
            for gw_loss, w in zip(grad_weights_loss, self.weights)
        ]
        # Bias typically not regularized as much, but can be. Here, we don't add decay to bias.
        regularized_grad_bias = grad_bias_loss 

        # Gradient Clipping
        if gradient_clip_value is not None:
            clipped_grad_weights = [
                max(-gradient_clip_value, min(gw, gradient_clip_value))
                for gw in regularized_grad_weights
            ]
            clipped_grad_bias = max(-gradient_clip_value, min(regularized_grad_bias, gradient_clip_value))
        else:
            clipped_grad_weights = regularized_grad_weights
            clipped_grad_bias = regularized_grad_bias
            
        return clipped_grad_weights, clipped_grad_bias

    def update_parameters(self, # Used by SGD_Momentum_Adagrad
                          grad_weights: List[float], grad_bias: float,
                          learning_rate: float,
                          momentum_coefficient: float = 0.0,
                          adagrad_epsilon: float = 1e-8):
        """Updates weights and bias using SGD with Momentum and AdaGrad."""
        # Update Weights
        for i in range(len(self.weights)):
            self.cache_weights[i] += grad_weights[i] ** 2
            adjusted_lr_w = learning_rate / (math.sqrt(self.cache_weights[i]) + adagrad_epsilon)
            
            # Momentum update
            self.velocity_weights[i] = (momentum_coefficient * self.velocity_weights[i] - # Note: common to subtract LR term
                                        adjusted_lr_w * grad_weights[i])
            self.weights[i] += self.velocity_weights[i] 
        
        # Update Bias
        self.cache_bias += grad_bias ** 2
        adjusted_lr_b = learning_rate / (math.sqrt(self.cache_bias) + adagrad_epsilon)
        
        self.velocity_bias = (momentum_coefficient * self.velocity_bias -
                              adjusted_lr_b * grad_bias)
        self.bias += self.velocity_bias

    def __repr__(self) -> str:
        act_name = next((name for name, (fn, _, _) in ACTIVATION_FUNCTIONS.items() if fn == self.activation_fn_ptr), "unknown")
        return (f"Neuron(Act:{act_name}, Weights:{len(self.weights)}, "
                f"Bias:{self.bias:.3f}, Alpha:{self.activation_alpha if self.activation_needs_alpha else 'N/A'})")

# --- Neural Layer: A Collection of Neurons for Hierarchical Feature Extraction ---
class NeuralLayer:
    """
    Represents a layer of neurons. In cybersecurity, layers enable hierarchical
    feature learning, where earlier layers might detect simple patterns (e.g., unusual
    port access) and deeper layers combine these to identify complex threat signatures.
    Includes optional dropout and batch normalization for improved generalization.
    """
    def __init__(self, num_neurons: int, num_inputs_per_neuron: int,
                 activation_name: str = 'relu',
                 initialization_method: str = 'he_normal',
                 dropout_rate: float = 0.0,
                 activation_alpha: float = 0.01,
                 use_batch_norm: bool = False,
                 bn_momentum: float = 0.9,
                 bn_epsilon: float = 1e-5):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.activation_name = activation_name # For saving/loading and repr
        self.neurons: List[Neuron] = [
            Neuron(num_inputs_per_neuron, activation_name, initialization_method, activation_alpha)
            for _ in range(num_neurons)
        ]
        self.is_training: bool = False # Controlled by the NeuralNetwork

        self.dropout_rate = dropout_rate
        self._dropout_mask: Optional[List[float]] = None

        self.use_batch_norm = use_batch_norm
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        if self.use_batch_norm:
            # Parameters for Batch Normalization (gamma, beta are learnable, but often initialized to 1 and 0)
            # For simplicity in this "native" version, we'll use fixed gamma=1, beta=0.
            # These are per-activation, so size is num_neurons.
            self.bn_gamma: List[float] = [1.0] * num_neurons # Scale
            self.bn_beta: List[float] = [0.0] * num_neurons  # Shift
            self.running_mean: List[float] = [0.0] * num_neurons
            self.running_variance: List[float] = [1.0] * num_neurons # Initialize variance to 1

        self.history_activation_mean: List[float] = []
        self.history_activation_variance: List[float] = []
        self.history_max_len = 100

    def _update_bn_running_stats(self, batch_activations_T: List[List[float]]):
        """
        Updates running mean and variance using momentum based on a batch of activations.
        batch_activations_T: Transposed batch activations (List of features, each feature is a list of neuron outputs for that feature across batch)
                             So, batch_activations_T[neuron_idx] is a list of activations for that neuron over the batch.
        """
        if not batch_activations_T or not batch_activations_T[0]: # Empty batch or no neurons
            return

        num_samples_in_batch = len(batch_activations_T[0])
        if num_samples_in_batch == 0: return

        for i in range(self.num_neurons):
            neuron_activations_for_batch = batch_activations_T[i]
            current_batch_mean_i = sum(neuron_activations_for_batch) / num_samples_in_batch
            current_batch_var_i = sum([(a - current_batch_mean_i)**2 for a in neuron_activations_for_batch]) / num_samples_in_batch
            
            self.running_mean[i] = (self.bn_momentum * self.running_mean[i] +
                                    (1 - self.bn_momentum) * current_batch_mean_i)
            self.running_variance[i] = (self.bn_momentum * self.running_variance[i] +
                                        (1 - self.bn_momentum) * current_batch_var_i)


    def _apply_batch_norm(self, current_sample_activations: List[float], batch_activations_for_stats_T: Optional[List[List[float]]] = None) -> List[float]:
        """
        Applies batch normalization.
        current_sample_activations: Activations for the current single sample being processed.
        batch_activations_for_stats_T: Transposed activations of the entire current mini-batch,
                                       used ONLY during training to calculate batch statistics for BN.
                                       If None (e.g. SGD or inference), uses running stats.
        """
        if not self.use_batch_norm:
            return current_sample_activations

        normalized_activations = [0.0] * self.num_neurons

        if self.is_training:
            if batch_activations_for_stats_T and batch_activations_for_stats_T[0]: # Mini-batch training
                num_samples_in_batch = len(batch_activations_for_stats_T[0])
                # Calculate current batch mean and variance for normalization
                for i in range(self.num_neurons):
                    neuron_batch_activations = batch_activations_for_stats_T[i]
                    batch_mean_i = sum(neuron_batch_activations) / num_samples_in_batch
                    batch_var_i = sum([(a - batch_mean_i)**2 for a in neuron_batch_activations]) / num_samples_in_batch
                    
                    # Normalize using batch stats
                    norm_act = (current_sample_activations[i] - batch_mean_i) / math.sqrt(batch_var_i + self.bn_epsilon)
                    normalized_activations[i] = self.bn_gamma[i] * norm_act + self.bn_beta[i]
                
                # Update running stats using this batch's statistics
                self._update_bn_running_stats(batch_activations_for_stats_T)
            else: # SGD (batch_size=1) or if batch stats not provided - use running stats
                for i in range(self.num_neurons):
                    norm_act = (current_sample_activations[i] - self.running_mean[i]) / math.sqrt(self.running_variance[i] + self.bn_epsilon)
                    normalized_activations[i] = self.bn_gamma[i] * norm_act + self.bn_beta[i]
                    
                    # Update running stats with current sample (online update) - less stable than batch updates
                    self.running_mean[i] = self.bn_momentum * self.running_mean[i] + (1-self.bn_momentum) * current_sample_activations[i]
                    self.running_variance[i] = self.bn_momentum * self.running_variance[i] + (1-self.bn_momentum) * (current_sample_activations[i] - self.running_mean[i])**2
        else: # Inference mode - use running stats
            for i in range(self.num_neurons):
                norm_act = (current_sample_activations[i] - self.running_mean[i]) / math.sqrt(self.running_variance[i] + self.bn_epsilon)
                normalized_activations[i] = self.bn_gamma[i] * norm_act + self.bn_beta[i]
        
        return normalized_activations

    def _apply_dropout(self, activations: List[float]) -> List[float]:
        """Applies dropout to the activations (inverted dropout)."""
        if not self.is_training or self.dropout_rate == 0.0:
            self._dropout_mask = None 
            return activations

        self._dropout_mask = [0.0] * self.num_neurons
        scaled_activations = [0.0] * self.num_neurons
        scale_factor = 1.0 / (1.0 - self.dropout_rate) if self.dropout_rate < 1.0 else 0.0

        for i in range(self.num_neurons):
            if random.random() < self.dropout_rate:
                self._dropout_mask[i] = 0.0 # Neuron is dropped
                scaled_activations[i] = 0.0
            else:
                self._dropout_mask[i] = scale_factor # Store the scale factor (or 1.0 if mask just indicates kept/dropped)
                scaled_activations[i] = activations[i] * scale_factor
        return scaled_activations

    def feed_forward_sample(self, inputs: List[float], batch_raw_activations_T: Optional[List[List[float]]] = None) -> List[float]:
        """
        Processes a single sample through the layer.
        If batch_raw_activations_T is provided (during mini-batch training), BN uses it for stats.
        """
        raw_activations = [neuron.activate(inputs) for neuron in self.neurons]

        if self.is_training and self.num_neurons > 0: # Store stats based on raw_activations for layer observation
            current_mean = sum(raw_activations) / self.num_neurons
            current_var = sum([(a - current_mean)**2 for a in raw_activations]) / self.num_neurons
            self.history_activation_mean.append(current_mean)
            self.history_activation_variance.append(current_var)
            if len(self.history_activation_mean) > self.history_max_len:
                self.history_activation_mean.pop(0)
                self.history_activation_variance.pop(0)

        bn_applied_activations = self._apply_batch_norm(raw_activations, batch_raw_activations_T)
        final_activations = self._apply_dropout(bn_applied_activations)

        for i, neuron in enumerate(self.neurons):
            neuron.activation = final_activations[i]
        
        return final_activations

    def get_raw_activations_for_sample(self, inputs: List[float]) -> List[float]:
        """Helper to get raw activations for a sample, used by NN for batch BN stats."""
        return [neuron._call_activation_fn(neuron._calculate_weighted_sum(inputs)) for neuron in self.neurons]


    def get_layer_stats(self) -> Dict[str, Any]:
        """Returns recent average mean/variance of raw activations and BN stats if applicable."""
        avg_mean = sum(self.history_activation_mean) / len(self.history_activation_mean) if self.history_activation_mean else 0.0
        avg_var = sum(self.history_activation_variance) / len(self.history_activation_variance) if self.history_activation_variance else 0.0
        bn_stats = {}
        if self.use_batch_norm:
            # Provide a snapshot of running means/vars for a few neurons for brevity
            num_bn_stats_to_show = min(3, self.num_neurons)
            bn_stats_summary = {
                f"bn_run_mean_neuron{i}": self.running_mean[i] for i in range(num_bn_stats_to_show)
            }
            # bn_stats.update({f"bn_run_var_neuron{i}": self.running_variance[i] for i in range(num_bn_stats_to_show)})
            bn_stats.update(bn_stats_summary)
        return {
            "avg_raw_activation_mean": avg_mean,
            "avg_raw_activation_variance": avg_var,
            **bn_stats
        }

    def __repr__(self) -> str:
        bn_repr = ", BN" if self.use_batch_norm else ""
        dropout_repr = f", Dropout:{self.dropout_rate}" if self.dropout_rate > 0 else ""
        return (f"NeuralLayer({self.num_neurons}N, Act:{self.activation_name}{dropout_repr}{bn_repr})")


# --- Neural Network: The Complete Structure for Adaptive Cyber-Security Analysis ---
class NeuralNetwork:
    """
    A multi-layer perceptron (MLP) designed for adaptive cyber-security tasks
    such as intrusion detection, malware classification, or anomaly detection.
    It supports various activation functions, optimizers, regularization techniques
    (dropout, L2), and batch normalization. The network can be trained iteratively,
    allowing it to "self-improve" by learning from new security event data.
    """
    def __init__(self, num_inputs: int,
                 layer_config: List[dict], # Defines architecture: neurons, activation, dropout, etc. per layer
                 loss_function_name: str = 'cross_entropy', # 'mse' or 'cross_entropy'
                 optimizer_name: str = 'adam', # 'sgd_momentum_adagrad', 'adam'
                 initialization_method_default: str = 'he_normal',
                 problem_type: str = 'binary_classification', # 'regression', 'binary_classification', 'multiclass_classification'
                 config: Optional[Dict[str, Any]] = None): # General NN configs (learning rates, optimizer params)
        super().__init__()
        if not layer_config:
            raise ValueError("Layer configuration must be provided and cannot be empty")
        self.num_inputs = num_inputs
        self.layers: List[NeuralLayer] = []
        self.is_training: bool = False # Global training mode flag
        self.problem_type = problem_type.lower()
        self.loss_function_name = loss_function_name.lower()
        self.optimizer_name = optimizer_name.lower()
        
        self.config = get_merged_config(config) # Merges with base config from file
        self._configure_loss_function()
        self._configure_optimizer_hyperparameters() # Sets up Adam params etc.
        self.output_layer_activation_is_softmax = False # Flag for special softmax handling

        current_num_inputs = num_inputs
        logger.info("Initializing Neural Network Layers for Cyber-Security Task:")
        for i, layer_conf in enumerate(layer_config):
            num_n = layer_conf['neurons']
            
            # Determine default activation based on problem type and layer position
            default_act = 'relu'  # Common default for hidden layers
            if i == len(layer_config) - 1:  # Output layer
                if self.problem_type == 'binary_classification':
                    default_act = 'sigmoid'
                elif self.problem_type == 'multiclass_classification':
                    default_act = 'linear'  # Neurons output weighted sums; softmax applied by NN
                elif self.problem_type == 'regression':
                    default_act = 'linear'
            
            act_name = layer_conf.get('activation', default_act)
            init_method = layer_conf.get('init', initialization_method_default)
            dropout = layer_conf.get('dropout', self.config.get('default_dropout_rate', 0.0))
            act_alpha = layer_conf.get('alpha', self.config.get('default_activation_alpha', 0.01))
            use_bn = layer_conf.get('batch_norm', self.config.get('default_use_batch_norm', False))
            bn_momentum = layer_conf.get('bn_momentum', self.config.get('default_bn_momentum', 0.9))
            bn_epsilon = float(layer_conf.get('bn_epsilon', self.config.get('default_bn_epsilon', 1e-5)))
        
            effective_dropout = dropout if i < len(layer_config) - 1 else 0.0
        
            # Handle softmax activation for output layer
            if i == len(layer_config) - 1 and act_name == 'softmax':
                # For softmax, use linear activation in the layer and apply softmax in feed_forward
                layer = NeuralLayer(
                    num_n, current_num_inputs,
                    activation_name='linear',
                    initialization_method=init_method,
                    dropout_rate=0.0,
                    activation_alpha=act_alpha,
                    use_batch_norm=use_bn,
                    bn_momentum=bn_momentum,
                    bn_epsilon=bn_epsilon
                )
                self.output_layer_activation_is_softmax = True
                logger.info(f"  Layer {i} (Output): {num_n}N, Act:Softmax (applied by NN), NeuronAct:linear, Init:{init_method}, Dropout:OFF, BN:{use_bn}")
            else:
                layer = NeuralLayer(
                    num_n, current_num_inputs,
                    activation_name=act_name,
                    initialization_method=init_method,
                    dropout_rate=effective_dropout,
                    activation_alpha=act_alpha,
                    use_batch_norm=use_bn,
                    bn_momentum=bn_momentum,
                    bn_epsilon=bn_epsilon
                )
                layer_type = 'Output' if i == len(layer_config) - 1 else 'Hidden'
                logger.info(f"  Layer {i} ({layer_type}): {num_n}N, Act:{act_name}, Init:{init_method}, Dropout:{effective_dropout}, BN:{use_bn}, Alpha:{act_alpha if act_name in ['leaky_relu', 'elu'] else 'N/A'}")
        
            self.layers.append(layer)
            current_num_inputs = num_n
        logger.info(f"Initialized Neural Network. Problem: {self.problem_type}, Loss: {self.loss_function_name}, Opt: {self.optimizer_name}")

    def _configure_loss_function(self):
        if self.loss_function_name == 'mse':
            self.loss_fn = mean_squared_error
            self.loss_fn_derivative = mean_squared_error_derivative
        elif self.loss_function_name == 'cross_entropy':
            self.loss_fn = cross_entropy_loss_func # from math_science
            # For CE with Softmax, derivative is handled specially in _calculate_loss_and_output_deltas
            # For CE with Sigmoid (binary), derivative is (pred - true) * (sigmoid_deriv) / (pred * (1-pred)) -> handled by neuron.calculate_delta
            self.loss_fn_derivative = cross_entropy_derivative # (pred - true) for CE+Softmax / (pred-true)/(p*(1-p)) for CE+Sigmoid (dLoss/dActivation)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function_name}")

    def _configure_optimizer_hyperparameters(self):
        self.adam_beta1 = self.config.get('adam_beta1', 0.9)
        self.adam_beta2 = self.config.get('adam_beta2', 0.999)
        self.adam_epsilon = self.config.get('adam_epsilon', 1e-8) # Epsilon for Adam numerical stability
        
        # This initialization should happen after layers are created if Adam is selected
        if self.optimizer_name == 'adam' and self.layers:
            for layer in self.layers:
                for neuron in layer.neurons:
                    neuron.m_weights = [0.0] * neuron.num_inputs
                    neuron.v_weights = [0.0] * neuron.num_inputs
                    neuron.m_bias = 0.0
                    neuron.v_bias = 0.0
            self.adam_global_timestep = 0 # Per-training call, or persistent if desired

    def _set_training_mode(self, mode: bool):
        self.is_training = mode
        for layer in self.layers:
            layer.is_training = mode

    def feed_forward_sample(self, inputs: List[float], batch_raw_activations_T_by_layer: Optional[List[List[List[float]]]] = None) -> List[float]:
        """
        Processes a single sample through the network.
        batch_raw_activations_T_by_layer: If provided (during mini-batch training),
                                          it's a list (per layer) of transposed raw activations for the batch.
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs for network, got {len(inputs)}.")
        
        current_outputs = inputs
        for i, layer in enumerate(self.layers):
            batch_raw_act_T_for_layer = batch_raw_activations_T_by_layer[i] if batch_raw_activations_T_by_layer and i < len(batch_raw_activations_T_by_layer) else None
            current_outputs = layer.feed_forward_sample(current_outputs, batch_raw_act_T_for_layer)
            
            if i == len(self.layers) - 1 and self.output_layer_activation_is_softmax:
                # `current_outputs` are linear outputs from the last layer's neurons
                softmax_probs = softmax(current_outputs) # softmax from math_science
                # Update neuron activations to be the softmax probabilities for consistency
                for neuron_idx, neuron_in_output_layer in enumerate(self.layers[-1].neurons):
                    neuron_in_output_layer.activation = softmax_probs[neuron_idx]
                current_outputs = softmax_probs
                
        return current_outputs
    
    def _get_batch_raw_activations_for_bn(self, batch_data: List[Tuple[List[float], List[float]]]) -> List[List[List[float]]]:
        """
        Helper function to perform a 'dry run' feedforward for a batch to collect
        raw activations from each layer, necessary for batch normalization computations.
        Returns: List (per layer) of List (per neuron in layer) of List (activations over batch).
                 Essentially, [layer][neuron_idx][sample_idx_in_batch]
        """
        all_layers_batch_raw_activations_T = [] # Transposed: [layer][neuron_idx][sample_idx_in_batch]

        # Initialize storage for raw activations
        for layer_idx, layer in enumerate(self.layers):
            # [neuron_outputs_over_batch_for_neuron0, neuron_outputs_over_batch_for_neuron1, ...]
            layer_batch_raw_activations_T = [[] for _ in range(layer.num_neurons)]
            all_layers_batch_raw_activations_T.append(layer_batch_raw_activations_T)

        # Collect raw activations for each sample in the batch
        current_inputs_for_batch = [sample_data[0] for sample_data in batch_data]

        for sample_idx_in_batch in range(len(batch_data)):
            sample_input = current_inputs_for_batch[sample_idx_in_batch]
            
            current_layer_input_for_sample = sample_input
            for layer_idx, layer in enumerate(self.layers):
                # Get raw (pre-BN, pre-dropout) activations for this sample for this layer
                raw_neuron_outputs_for_sample_layer = layer.get_raw_activations_for_sample(current_layer_input_for_sample)
                
                for neuron_idx in range(layer.num_neurons):
                    all_layers_batch_raw_activations_T[layer_idx][neuron_idx].append(raw_neuron_outputs_for_sample_layer[neuron_idx])
                
                # The input to the next layer would be the processed output of this layer,
                # but for raw_activations, we only need the direct neuron output before BN/dropout.
                # For the *next* layer's raw activation calculation, it needs the *output* of *this* layer's neurons
                # (conceptually, after their activation function but before BN/dropout of this layer)
                current_layer_input_for_sample = raw_neuron_outputs_for_sample_layer # This becomes input to next layer's raw activation calc.


        return all_layers_batch_raw_activations_T


    def _calculate_loss_and_output_deltas(self,
                                          target_outputs: List[float],
                                          predicted_outputs: List[float] # These are final outputs (e.g. after softmax)
                                         ) -> Tuple[float, List[float]]:
        """
        Calculates loss and the initial error signals (dE/da_L or dE/dz_L) for output layer neurons.
        Returns: (loss, output_layer_error_signals)
                 output_layer_error_signals: For CE+Softmax, this is (pred_prob - target), which is dE/dz_L.
                                             For MSE or CE+Sigmoid, this is dE/da_L.
        """
        loss = 0.0
        output_layer_error_signals = [0.0] * len(target_outputs) # dE/da for MSE/Sigmoid, dE/dz for Softmax
        output_layer = self.layers[-1]

        if self.loss_function_name == 'cross_entropy':
            loss = self.loss_fn(target_outputs, predicted_outputs) # Uses cross_entropy_loss_func from math_science
            if self.output_layer_activation_is_softmax:
                # For Softmax output with Cross-Entropy loss, dE/dz_j = (p_j - t_j)
                # where z_j is the weighted sum for output neuron j, p_j is its softmax probability.
                # neuron.activation already holds p_j.
                for i in range(len(output_layer.neurons)):
                    output_layer_error_signals[i] = output_layer.neurons[i].activation - target_outputs[i]
                    # This error signal is dE/dz, so store it directly in neuron.delta
                    output_layer.neurons[i].delta = output_layer_error_signals[i]
            else: # E.g. Sigmoid output with Cross-Entropy (binary classification)
                  # dE/da_j = -(t_j/p_j - (1-t_j)/(1-p_j)) or simplified dE/da_j = (p_j-t_j)/(p_j(1-p_j))
                  # The neuron.calculate_delta will then multiply this by da/dz.
                output_layer_error_signals = self.loss_fn_derivative(target_outputs, predicted_outputs) # This is dE/da
        
        elif self.loss_function_name == 'mse':
            loss = self.loss_fn(target_outputs, predicted_outputs)
            # For MSE, dE/da_j = (p_j - t_j)
            output_layer_error_signals = self.loss_fn_derivative(target_outputs, predicted_outputs) # This is dE/da
        else:
            raise NotImplementedError(f"Loss calculation/derivative for {self.loss_function_name} not fully implemented.")
            
        return loss, output_layer_error_signals


    def _backpropagate(self, inputs_sample: List[float], output_layer_error_signals: List[float],
                       learning_rate: float, **optimizer_kwargs):
        """
        Performs backpropagation for a single sample to update network weights and biases.
        output_layer_error_signals: dE/da (for MSE/Sigmoid) or dE/dz (for CE+Softmax) for the output layer.
        """
        # --- Output Layer ---
        output_layer = self.layers[-1]
        # Determine inputs that fed into the output layer
        inputs_to_output_layer = self.layers[-2].neurons if len(self.layers) > 1 else None # List of neuron objects
        input_activations_to_output_layer = [n.activation for n in inputs_to_output_layer] if inputs_to_output_layer else inputs_sample

        for neuron_idx, neuron in enumerate(output_layer.neurons):
            # If CE+Softmax, neuron.delta (dE/dz) was already set in _calculate_loss_and_output_deltas.
            # Otherwise, calculate dE/dz = dE/da * da/dz.
            if not (self.loss_function_name == 'cross_entropy' and self.output_layer_activation_is_softmax):
                # output_layer_error_signals[neuron_idx] is dE/da_L for this neuron
                neuron.calculate_delta(output_layer_error_signals[neuron_idx]) 
            
            # Ensure neuron inputs are correctly set for gradient calculation
            neuron.inputs = input_activations_to_output_layer

            grad_weights, grad_bias = neuron.calculate_gradients(
                optimizer_kwargs.get('weight_decay_lambda', 0.0),
                optimizer_kwargs.get('gradient_clip_value')
            )
            self._apply_optimizer_step(neuron, grad_weights, grad_bias, learning_rate, **optimizer_kwargs)

        # --- Hidden Layers (iterating backwards) ---
        for layer_idx in reversed(range(len(self.layers) - 1)):
            hidden_layer = self.layers[layer_idx]
            downstream_layer = self.layers[layer_idx + 1] # Layer whose errors are known
            
            inputs_to_hidden_layer_objects = self.layers[layer_idx-1].neurons if layer_idx > 0 else None
            input_activations_to_hidden_layer = [n.activation for n in inputs_to_hidden_layer_objects] if inputs_to_hidden_layer_objects else inputs_sample

            for neuron_idx, neuron in enumerate(hidden_layer.neurons):
                # Sum of (delta_from_downstream_neuron * weight_connecting_this_neuron_to_it)
                error_signal_sum_for_activation = 0.0
                for downstream_neuron in downstream_layer.neurons:
                    # downstream_neuron.delta is dE/dz for that neuron
                    # downstream_neuron.weights[neuron_idx] is weight w_kj (k=downstream, j=current)
                    error_signal_sum_for_activation += downstream_neuron.delta * downstream_neuron.weights[neuron_idx]
                
                # This sum is dE/da for the current hidden neuron.
                # Now calculate dE/dz for this hidden neuron: (dE/da) * (da/dz)
                neuron.calculate_delta(error_signal_sum_for_activation)

                # If dropout was applied, scale delta or zero it out
                # Dropout mask is applied during forward pass by scaling activations of kept neurons.
                # During backprop, if a neuron was dropped (mask=0), its delta should be 0.
                # If it was kept, its delta is calculated normally. The scaling is implicitly handled.
                if hasattr(hidden_layer, '_dropout_mask') and hidden_layer._dropout_mask and \
                   hidden_layer._dropout_mask[neuron_idx] == 0.0: # Neuron was dropped
                    neuron.delta = 0.0 
                
                if neuron.delta != 0.0:
                    neuron.inputs = input_activations_to_hidden_layer # Set correct inputs for gradient calc
                    grad_weights, grad_bias = neuron.calculate_gradients(
                        optimizer_kwargs.get('weight_decay_lambda', 0.0),
                        optimizer_kwargs.get('gradient_clip_value')
                    )
                    self._apply_optimizer_step(neuron, grad_weights, grad_bias, learning_rate, **optimizer_kwargs)
                else: # Reset optimizer states if neuron was dropped or delta is zero
                    if hasattr(neuron, 'velocity_weights'): neuron.velocity_weights = [0.0] * neuron.num_inputs
                    if hasattr(neuron, 'velocity_bias'): neuron.velocity_bias = 0.0
                    if self.optimizer_name == 'adam':
                        neuron.m_weights = [0.0] * neuron.num_inputs
                        neuron.v_weights = [0.0] * neuron.num_inputs
                        neuron.m_bias = 0.0
                        neuron.v_bias = 0.0

    def _apply_optimizer_step(self, neuron: Neuron, grad_weights: List[float], grad_bias: float,
                              learning_rate: float, **optimizer_kwargs):
        """Applies a single optimization step to a neuron's parameters based on optimizer_name."""
        if self.optimizer_name == 'sgd_momentum_adagrad':
            neuron.update_parameters(
                grad_weights, grad_bias, learning_rate,
                optimizer_kwargs.get('momentum_coefficient', self.config.get('momentum_coefficient', 0.9)),
                optimizer_kwargs.get('adagrad_epsilon', self.config.get('adagrad_epsilon', 1e-8))
            )
        elif self.optimizer_name == 'adam':
            beta1 = optimizer_kwargs.get('adam_beta1', self.adam_beta1)
            beta2 = optimizer_kwargs.get('adam_beta2', self.adam_beta2)
            epsilon = optimizer_kwargs.get('adam_epsilon', self.adam_epsilon) # This is Adam's own epsilon
            
            # Adam global timestep should be incremented per batch/sample update
            # It's handled in the train loop.
            
            # Update weights
            for i in range(len(neuron.weights)):
                neuron.m_weights[i] = beta1 * neuron.m_weights[i] + (1 - beta1) * grad_weights[i]
                neuron.v_weights[i] = beta2 * neuron.v_weights[i] + (1 - beta2) * (grad_weights[i] ** 2)
                
                m_hat = neuron.m_weights[i] / (1 - beta1 ** self.adam_global_timestep)
                v_hat = neuron.v_weights[i] / (1 - beta2 ** self.adam_global_timestep)
                
                neuron.weights[i] -= learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)

            # Update bias
            neuron.m_bias = beta1 * neuron.m_bias + (1 - beta1) * grad_bias
            neuron.v_bias = beta2 * neuron.v_bias + (1 - beta2) * (grad_bias ** 2)

            m_hat_bias = neuron.m_bias / (1 - beta1 ** self.adam_global_timestep)
            v_hat_bias = neuron.v_bias / (1 - beta2 ** self.adam_global_timestep)

            neuron.bias -= learning_rate * m_hat_bias / (math.sqrt(v_hat_bias) + epsilon)
        else:
            raise ValueError(f"Unsupported optimizer for step: {self.optimizer_name}")

    def train(self, training_data: Union[List[Tuple[List[float], List[float]]], np.ndarray],
              epochs: int, initial_learning_rate: float,
              batch_size: Optional[int] = 1, # Default to SGD if not specified
              # Optimizer specific kwargs from self.config or passed directly
              momentum_coefficient: Optional[float] = None, 
              weight_decay_lambda: Optional[float] = None,
              gradient_clip_value: Optional[float] = None,
              adagrad_epsilon: Optional[float] = None, # For combined optimizer
              adam_beta1: Optional[float] = None, 
              adam_beta2: Optional[float] = None,
              adam_epsilon_opt: Optional[float] = None, # Optimizer specific epsilon
              # LR Decay
              lr_scheduler_name: Optional[str] = None, 
              lr_decay_rate: Optional[float] = None, 
              lr_decay_steps: Optional[int] = None, 
              # Early Stopping
              early_stopping_patience: Optional[int] = None,
              early_stopping_min_delta: float = 0.0001,
              validation_data: Optional[Union[List[Tuple[List[float], List[float]]], np.ndarray]] = None,
              # Other
              verbose: bool = True,
              print_every_n_epochs: Optional[int] = None,
              save_best_model_path: Optional[str] = None):
        """
        Trains the neural network on the provided cybersecurity training data.
        Allows for iterative improvement and adaptation to new threat patterns.
        """
        self._set_training_mode(True)
        if self.optimizer_name == 'adam': # Reset/initialize Adam's global timestep for this training run
            self.adam_global_timestep = 0 

        current_learning_rate = initial_learning_rate

        # Consolidate optimizer kwargs, prioritizing direct args, then self.config, then defaults
        opt_kwargs = {
            'momentum_coefficient': momentum_coefficient if momentum_coefficient is not None else self.config.get('momentum_coefficient', 0.9),
            'weight_decay_lambda': weight_decay_lambda if weight_decay_lambda is not None else self.config.get('weight_decay_lambda', 0.0),
            'gradient_clip_value': gradient_clip_value if gradient_clip_value is not None else self.config.get('gradient_clip_value'),
            'adagrad_epsilon': adagrad_epsilon if adagrad_epsilon is not None else self.config.get('adagrad_epsilon', 1e-8),
            'adam_beta1': adam_beta1 if adam_beta1 is not None else self.adam_beta1, # from self._configure_optimizer_hyperparameters
            'adam_beta2': adam_beta2 if adam_beta2 is not None else self.adam_beta2,
            'adam_epsilon': adam_epsilon_opt if adam_epsilon_opt is not None else self.adam_epsilon, # Use NN's configured Adam epsilon
        }
        
        if print_every_n_epochs is None:
            print_every_n_epochs = max(1, epochs // 20 if epochs >= 20 else 1)
        
        best_val_metric = float('inf') if self.loss_function_name in ['mse', 'cross_entropy'] else float('-inf') # for accuracy etc.
        epochs_no_improve = 0
        if save_best_model_path:
            Path(save_best_model_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n--- Cyber-Security Model Training Started ({self.optimizer_name} optimizer) ---")
        logger.info(f"Epochs: {epochs}, Initial LR: {initial_learning_rate}, Batch Size: {batch_size}")
        logger.info(f"Optimizer Params: {opt_kwargs}")


        # Convert numpy array to list of tuples if necessary
        if isinstance(training_data, np.ndarray) and training_data.ndim >= 2: # Expecting (X,y) pair from np array
             data_list = [(list(x_row), list(y_row) if isinstance(y_row, (list, np.ndarray)) else [y_row]) 
                             for x_row, y_row in zip(training_data[0], training_data[1])]
        elif isinstance(training_data, tuple) and len(training_data) == 2 and isinstance(training_data[0], np.ndarray): # (X_np, y_np)
            data_list = [(list(x_row), list(y_row) if isinstance(y_row, (list, np.ndarray)) else [y_row]) 
                             for x_row, y_row in zip(training_data[0], training_data[1])]
        else: # Assumed to be list of tuples
            data_list = list(training_data)

        if not data_list:
            logger.error("Training data is empty. Aborting training.")
            return

        for epoch in range(epochs):
            random.shuffle(data_list)
            epoch_total_loss = 0.0
            
            num_samples_processed_in_epoch = 0

            effective_batch_size = batch_size if batch_size and batch_size > 0 else len(data_list)

            for batch_idx in range(0, len(data_list), effective_batch_size):
                batch_data = data_list[batch_idx : batch_idx + effective_batch_size]
                if not batch_data: continue

                if self.optimizer_name == 'adam': # Adam timestep increments per effective batch update
                    self.adam_global_timestep += 1 

                # --- Mini-batch Processing ---
                # 1. (Optional, for BN) Pre-calculate raw activations for the batch if BN is used in any layer
                batch_raw_activations_T_by_layer = None
                if any(layer.use_batch_norm for layer in self.layers):
                    batch_raw_activations_T_by_layer = self._get_batch_raw_activations_for_bn(batch_data)

                # 2. Process each sample in the batch: forward, loss, store gradients (or update for SGD)
                # For optimizers like Adam or full batch GD, gradients are typically accumulated.
                # For pure SGD or our current structure that updates per sample:
                for inputs_sample, targets_sample in batch_data:
                    # Forward pass for the current sample
                    outputs = self.feed_forward_sample(inputs_sample, batch_raw_activations_T_by_layer)
                    
                    # Calculate loss and output layer error signals for this sample
                    loss_val, output_err_signals = self._calculate_loss_and_output_deltas(targets_sample, outputs)
                    epoch_total_loss += loss_val
                    num_samples_processed_in_epoch += 1
                    
                    # Backpropagate and update parameters for this sample
                    self._backpropagate(inputs_sample, output_err_signals, current_learning_rate, **opt_kwargs)
            
            avg_epoch_loss = epoch_total_loss / num_samples_processed_in_epoch if num_samples_processed_in_epoch > 0 else 0.0

            # Learning Rate Scheduling
            if lr_scheduler_name:
                current_learning_rate = self._apply_lr_schedule(
                    initial_learning_rate, current_learning_rate, epoch, epochs,
                    lr_scheduler_name, lr_decay_rate, lr_decay_steps
                )

            # Validation and Early Stopping
            val_metrics_str = ""
            if validation_data:
                # Use effective_batch_size for evaluation consistency, or a dedicated eval_batch_size
                val_results = self.evaluate(validation_data, batch_size=effective_batch_size) 
                val_loss = val_results['loss']
                val_metrics_str = f", Val Loss: {val_loss:.5f}"
                if 'accuracy' in val_results: val_metrics_str += f", Val Acc: {val_results['accuracy']:.3f}"

                # Early stopping logic (assumes lower loss is better)
                current_val_metric_for_stopping = val_loss # Could be accuracy if preferred
                if current_val_metric_for_stopping < best_val_metric - early_stopping_min_delta:
                    best_val_metric = current_val_metric_for_stopping
                    epochs_no_improve = 0
                    if save_best_model_path:
                        self.save_model(save_best_model_path)
                        if verbose: logger.info(f"  Epoch {epoch+1}: Val metric improved ({best_val_metric:.5f}). Model saved.")
                else:
                    epochs_no_improve += 1
                
                if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                    logger.info(f"Epoch {epoch + 1}: Early stopping triggered after {epochs_no_improve} epochs.")
                    break 
            
            if verbose and (epoch + 1) % print_every_n_epochs == 0:
                layer_stats_summary_parts = []
                for i_layer, layer_obj in enumerate(self.layers[:2]): # Show stats for first few layers
                    stats = layer_obj.get_layer_stats()
                    part = f"L{i_layer}[RawMean:{stats['avg_raw_activation_mean']:.2f},RawVar:{stats['avg_raw_activation_variance']:.2f}"
                    if layer_obj.use_batch_norm and 'bn_run_mean_neuron0' in stats:
                        part += f",BNM0:{stats['bn_run_mean_neuron0']:.2f}"
                    part += "]"
                    layer_stats_summary_parts.append(part)
                layer_stats_str = " ".join(layer_stats_summary_parts)
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.5f}, LR: {current_learning_rate:.6f}{val_metrics_str} | {layer_stats_str}")

        logger.info(f"--- Cyber-Security Model Training Finished (Epochs: {epoch+1}) ---")
        self._set_training_mode(False) # Set back to inference mode


    def _apply_lr_schedule(self, initial_lr, current_lr, epoch, total_epochs,
                           scheduler_name, decay_rate, decay_steps) -> float:
        if not scheduler_name or decay_rate is None: return current_lr

        if scheduler_name == 'step':
            if decay_steps and (epoch + 1) % decay_steps == 0:
                return current_lr * decay_rate
        elif scheduler_name == 'exponential':
            # decay_rate is the gamma factor here, e.g., 0.95
            # current_lr = initial_lr * (decay_rate ^ (epoch / (decay_steps or 1)))
            effective_decay_steps = decay_steps if decay_steps and decay_steps > 0 else 1.0 # Avoid division by zero, decay every epoch if steps is 0/None
            return initial_lr * (decay_rate ** (epoch / effective_decay_steps))
        elif scheduler_name == 'cosine_annealing':
            # T_max (decay_steps) is the number of epochs in one cycle.
            # decay_rate is eta_min (minimum learning rate).
            eta_min = decay_rate 
            t_max = decay_steps if decay_steps and decay_steps > 0 else total_epochs
            t_cur = epoch % t_max # Current epoch within the current cycle
            # Initial_lr for cosine annealing should be the starting LR of the cycle
            # If cycles repeat, initial_lr might need to be reset or managed. Here, simple one cycle from global initial_lr.
            return eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * t_cur / t_max))
        return current_lr

    def evaluate(self, test_data: Union[List[Tuple[List[float], List[float]]], np.ndarray],
                 batch_size: Optional[int] = 32) -> Dict[str, float]:
        """
        Evaluates the network on test data. For cybersecurity, this helps assess
        how well the model detects threats or anomalies on unseen data.
        Returns a dictionary of metrics: loss, accuracy, precision, recall, F1-score.
        """
        self._set_training_mode(False)
        total_loss = 0.0
        
        # For classification metrics
        tp, fp, tn, fn = 0, 0, 0, 0 # True/False Positives/Negatives

        if isinstance(test_data, np.ndarray) and test_data.ndim >=2:
            data_list = [(list(x_row), list(y_row) if isinstance(y_row, (list, np.ndarray)) else [y_row])
                         for x_row, y_row in zip(test_data[0], test_data[1])]
        elif isinstance(test_data, tuple) and len(test_data) == 2 and isinstance(test_data[0], np.ndarray):
            data_list = [(list(x_row), list(y_row) if isinstance(y_row, (list, np.ndarray)) else [y_row]) 
                             for x_row, y_row in zip(test_data[0], test_data[1])]
        else:
            data_list = list(test_data)
        
        num_samples = len(data_list)
        if num_samples == 0: return {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

        effective_batch_size = batch_size if batch_size and batch_size > 0 else num_samples

        for batch_idx in range(0, num_samples, effective_batch_size):
            batch_data = data_list[batch_idx : batch_idx + effective_batch_size]
            if not batch_data: continue
            
            # Batch BN considerations (similar to training, but using running stats always)
            batch_raw_activations_T_by_layer = None # Not strictly needed for eval with running stats, but passed for consistency
            # if any(layer.use_batch_norm for layer in self.layers):
            #     batch_raw_activations_T_by_layer = self._get_batch_raw_activations_for_bn(batch_data) # Uses running stats in eval

            for inputs, targets in batch_data:
                outputs = self.feed_forward_sample(inputs, batch_raw_activations_T_by_layer) # predict sets training_mode = False
                loss_val, _ = self._calculate_loss_and_output_deltas(targets, outputs) # _ uses self.loss_fn
                total_loss += loss_val

                if self.problem_type == 'binary_classification':
                    # Assuming single output neuron with sigmoid, target is [0.0] or [1.0]
                    pred_class = 1 if outputs[0] >= 0.5 else 0
                    true_class = int(round(targets[0]))
                    if pred_class == 1 and true_class == 1: tp += 1
                    elif pred_class == 1 and true_class == 0: fp += 1
                    elif pred_class == 0 and true_class == 0: tn += 1
                    elif pred_class == 0 and true_class == 1: fn += 1
                elif self.problem_type == 'multiclass_classification':
                    # Assuming targets are one-hot encoded
                    pred_class_idx = outputs.index(max(outputs))
                    true_class_idx = targets.index(1.0) if 1.0 in targets else -1 # Handle if not perfectly one-hot
                    if pred_class_idx == true_class_idx:
                        tp += 1 # Simplified: count correct multi-class as TP for accuracy
                    else:
                        fn +=1 # Simplified: count incorrect as FN for accuracy

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        metrics = {'loss': avg_loss}

        if self.problem_type == 'binary_classification':
            accuracy = (tp + tn) / num_samples if num_samples > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Also called Sensitivity or True Positive Rate
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics.update({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
                            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})
        elif self.problem_type == 'multiclass_classification':
             # For multiclass, tp here is just "correctly classified samples"
            accuracy = tp / num_samples if num_samples > 0 else 0.0
            metrics.update({'accuracy': accuracy})
            # Precision/Recall/F1 for multi-class would need micro/macro averaging, omitted for simplicity here.
        
        return metrics

    def predict(self, inputs: List[float]) -> List[float]:
        """
        Makes a prediction for a single input sample (e.g., features of a network packet).
        Returns raw output activations (e.g. probabilities if output is sigmoid/softmax).
        """
        self._set_training_mode(False)
        # For prediction, BN always uses running stats, so no need for batch_raw_activations_T_by_layer
        return self.feed_forward_sample(inputs, None)

    def predict_proba(self, inputs: List[float]) -> List[float]:
        """
        Predicts probabilities for classification tasks. For cybersecurity, this can
        represent the confidence score of a potential threat.
        Returns the output of the network, which should be probabilities if the
        output layer is configured with sigmoid (binary) or softmax (multiclass).
        """
        return self.predict(inputs)

    def predict_class(self, inputs: List[float]) -> Union[int, List[int]]:
        """
        Predicts class labels (e.g., 0 for benign, 1 for malicious).
        """
        probabilities = self.predict_proba(inputs)
        if self.problem_type == 'binary_classification':
            # Assumes single output neuron
            return 1 if probabilities[0] >= 0.5 else 0
        elif self.problem_type == 'multiclass_classification':
            return probabilities.index(max(probabilities)) # Index of max probability
        else: # Regression or unhandled
            logger.warning(f"predict_class called for non-classification problem type '{self.problem_type}'. Returning raw outputs.")
            return probabilities # Or raise error


    def get_weights_biases(self) -> List[Dict[str, Any]]:
        """Returns all weights and biases, useful for model inspection or transfer."""
        network_params = []
        for l_idx, layer in enumerate(self.layers):
            layer_params = {'layer_index': l_idx, 'neurons': []}
            if hasattr(layer, 'use_batch_norm') and layer.use_batch_norm:
                 layer_params['batch_norm_stats'] = {
                    'running_mean': list(layer.running_mean),
                    'running_variance': list(layer.running_variance),
                    'gamma': list(layer.bn_gamma), # Save learnable BN params if they were made learnable
                    'beta': list(layer.bn_beta)
                }
            for n_idx, neuron in enumerate(layer.neurons):
                neuron_params = {
                    'neuron_index': n_idx,
                    'weights': list(neuron.weights),
                    'bias': neuron.bias
                }
                # Include optimizer states if needed for resuming training precisely
                # For Adam: neuron.m_weights, neuron.v_weights, neuron.m_bias, neuron.v_bias
                # For SGD_Momentum_Adagrad: neuron.velocity_weights, neuron.velocity_bias, neuron.cache_weights, neuron.cache_bias
                layer_params['neurons'].append(neuron_params)
            network_params.append(layer_params)
        return network_params

    def set_weights_biases(self, network_params: List[Dict[str, Any]]):
        """Sets weights and biases from a saved structure. Essential for deploying trained models."""
        if len(network_params) != len(self.layers):
            raise ValueError("Mismatch in number of layers for loading weights.")
        for l_idx, layer_data in enumerate(network_params):
            layer_obj = self.layers[l_idx]
            if len(layer_data['neurons']) != len(layer_obj.neurons):
                raise ValueError(f"Mismatch in number of neurons for layer {l_idx}.")
            
            if 'batch_norm_stats' in layer_data and hasattr(layer_obj, 'use_batch_norm') and layer_obj.use_batch_norm:
                bn_stats = layer_data['batch_norm_stats']
                layer_obj.running_mean = list(bn_stats['running_mean'])
                layer_obj.running_variance = list(bn_stats['running_variance'])
                if 'gamma' in bn_stats: layer_obj.bn_gamma = list(bn_stats['gamma'])
                if 'beta' in bn_stats: layer_obj.bn_beta = list(bn_stats['beta'])

            for n_idx, neuron_data in enumerate(layer_data['neurons']):
                neuron = layer_obj.neurons[n_idx]
                if len(neuron_data['weights']) != len(neuron.weights):
                    raise ValueError(f"Mismatch in number of weights for neuron {n_idx} in layer {l_idx}.")
                neuron.weights = list(neuron_data['weights'])
                neuron.bias = float(neuron_data['bias'])
                # Restore optimizer states here if they were saved


    def save_model(self, filepath: str):
        """Saves the model architecture, weights, and relevant state (like BN stats) to a JSON file."""
        model_state = {
            'num_inputs': self.num_inputs,
            'layer_config_original': [], # Store the config used to build layers
            'loss_function_name': self.loss_function_name,
            'optimizer_name': self.optimizer_name,
            'problem_type': self.problem_type,
            'initialization_method_default': self.config.get('initialization_method_default', 'he_normal'),
            'config_used': self.config, # Save the effective configuration used
            'trained_weights_biases': self.get_weights_biases(), # Includes BN running stats per layer now
            'adam_global_timestep': self.adam_global_timestep if self.optimizer_name == 'adam' else None
        }
        for layer in self.layers:
            l_conf = { # Reconstruct essential parts of layer_config from actual layer properties
                'neurons': layer.num_neurons,
                'activation': layer.activation_name,
                'dropout': layer.dropout_rate,
                'init': layer.neurons[0].initialization_method if layer.neurons else 'unknown',
                'alpha': layer.neurons[0].activation_alpha if layer.neurons and hasattr(layer.neurons[0], 'activation_alpha') else self.config.get('default_activation_alpha', 0.01),
                'batch_norm': layer.use_batch_norm,
                'bn_momentum': layer.bn_momentum
            }
            model_state['layer_config_original'].append(l_conf)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(model_state, f, indent=4)
            logger.info(f"Cyber-security model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving cyber-security model: {e}")

    @classmethod
    def load_model(cls, filepath: str, custom_config_override: Optional[Dict] = None) -> 'NeuralNetwork':
        """Loads a model from a JSON file, ready for threat detection or further adaptation."""
        try:
            with open(filepath, 'r') as f:
                model_state = json.load(f)
        except Exception as e:
            logger.error(f"Error loading cyber-security model from {filepath}: {e}")
            raise

        # Merge loaded config with any overrides
        nn_config = model_state.get('config_used', {})
        if custom_config_override:
            nn_config.update(custom_config_override) # Simple update
        
        network = cls(
            num_inputs=model_state['num_inputs'],
            layer_config=model_state['layer_config_original'], # Use the saved original config for architecture
            loss_function_name=model_state.get('loss_function_name', 'cross_entropy'),
            optimizer_name=model_state.get('optimizer_name', 'adam'),
            initialization_method_default=model_state.get('initialization_method_default', 'he_normal'),
            problem_type=model_state.get('problem_type', 'binary_classification'),
            config=nn_config
        )
        
        if 'trained_weights_biases' in model_state:
            network.set_weights_biases(model_state['trained_weights_biases']) # Also loads BN stats
        
        if network.optimizer_name == 'adam' and 'adam_global_timestep' in model_state and model_state['adam_global_timestep'] is not None:
            network.adam_global_timestep = model_state['adam_global_timestep']
        
        logger.info(f"Cyber-security model loaded successfully from {filepath}")
        return network

# --- Example Usage: Simplified Network Intrusion Detection ---
if __name__ == "__main__":
    logger.info("--- Adaptive Neural Network for Cyber-Security Demo ---")
    config = load_config()
    layer_config = config.get('layers', [])  # Get layer architecture

    # 0. Synthetic Cyber-Security Dataset Generation (Simplified Intrusion Detection)
    def generate_security_data(num_samples: int) -> List[Tuple[List[float], List[float]]]:
        data = []
        for _ in range(num_samples):
            is_malicious = random.random() < 0.3 # 30% malicious samples
            if is_malicious:
                # Malicious patterns
                failed_logins = random.uniform(0.5, 1.0) # Higher normalized failed logins
                traffic = random.uniform(0.1, 0.8) # Can be low or high
                unusual_port = 1.0 if random.random() < 0.7 else 0.0 # Often unusual port
                duration = random.uniform(0.0, 0.3) if random.random() < 0.5 else random.uniform(0.7,1.0) # short bursts or long held
                target = [1.0]
            else:
                # Benign patterns
                failed_logins = random.uniform(0.0, 0.3)
                traffic = random.uniform(0.2, 0.6)
                unusual_port = 1.0 if random.random() < 0.1 else 0.0 # Rarely unusual port
                duration = random.uniform(0.1, 0.7)
                target = [0.0]
            
            features = [failed_logins, traffic, unusual_port, duration]
            data.append((features, target))
        return data

    num_total_samples = 500
    security_dataset = generate_security_data(num_total_samples)
    random.shuffle(security_dataset)

    # Split data: 70% train, 15% validation, 15% test
    train_end_idx = int(0.7 * num_total_samples)
    val_end_idx = int(0.85 * num_total_samples)

    train_data = security_dataset[:train_end_idx]
    val_data = security_dataset[train_end_idx:val_end_idx]
    test_data = security_dataset[val_end_idx:]
    
    logger.info(f"Dataset: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples.")

    # 1. Neural Network Configuration for Intrusion Detection
    # Input: 4 features. Output: 1 neuron (sigmoid for benign/malicious probability).
    intrusion_detection_layer_config = [
        {'neurons': 16, 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.2, 'batch_norm': True},
        {'neurons': 8, 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.1, 'batch_norm': True},
        {'neurons': 1, 'activation': 'sigmoid'} # Output layer for binary classification
    ]

    # Configuration for the network and training (can also be loaded from adaptive_config.yaml)
    nn_overall_config = {
        'initialization_method_default': 'he_normal',
        'default_dropout_rate': 0.1,
        'default_use_batch_norm': True,
        'default_bn_momentum': 0.9,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon': 1e-7, # Adam's own epsilon
        'weight_decay_lambda': 1e-5, # L2 regularization
        'gradient_clip_value': 1.0,  # Clip gradients to [-1, 1]
        'momentum_coefficient': 0.9, # For SGD_Momentum_Adagrad
        'adagrad_epsilon': 1e-7      # For SGD_Momentum_Adagrad
    }
    
    logger.info("\nInitializing Intrusion Detection Neural Network:")
    ids_nn = NeuralNetwork(
            num_inputs=4,
            layer_config=layer_config, # This 'layer_config' comes from config.get('layers', [])
            loss_function_name='cross_entropy',
            optimizer_name='adam',
            problem_type='binary_classification',
            config=config # This is the global config from load_config()
        )

    # 2. Training the IDS Model (Adaptive Learning Phase)
    logger.info("\n--- Training Intrusion Detection Model ---")
    model_save_path = "intrusion_detection_model_best.json" # Path relative to script execution

    ids_nn.train(train_data,
                 epochs=50, # Increased epochs
                 initial_learning_rate=0.001, # Adam often prefers smaller LRs
                 batch_size=32, # Mini-batch training
                 # Optimizer params (can be passed or taken from nn_overall_config if set there)
                 weight_decay_lambda=nn_overall_config['weight_decay_lambda'],
                 gradient_clip_value=nn_overall_config['gradient_clip_value'],
                 # Learning rate scheduling
                 lr_scheduler_name='cosine_annealing', 
                 lr_decay_rate=1e-6, # eta_min for cosine
                 lr_decay_steps=25,  # T_max for cosine (half of epochs for one cycle)
                 # Early stopping
                 validation_data=val_data,
                 early_stopping_patience=10,
                 early_stopping_min_delta=0.001, # Min change in val_loss to be considered improvement
                 print_every_n_epochs=5,
                 save_best_model_path=model_save_path)

    # 3. Evaluating the Trained Model on Unseen Test Data
    logger.info("\n--- Evaluating Trained IDS Model on Test Data ---")
    test_metrics = ids_nn.evaluate(test_data, batch_size=32)
    logger.info(f"Test Set Evaluation:")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric.capitalize()}: {value:.4f}")
        else:
            logger.info(f"  {metric.capitalize()}: {value}")


    # 4. Making Predictions (Simulating Real-time Threat Analysis)
    logger.info("\n--- Predicting with Trained IDS Model ---")
    sample_benign_event = [0.1, 0.2, 0.0, 0.3] # Expected benign
    sample_malicious_event = [0.8, 0.1, 1.0, 0.1] # Expected malicious

    prob_benign = ids_nn.predict_proba(sample_benign_event)
    class_benign = ids_nn.predict_class(sample_benign_event)
    logger.info(f"Benign-like event: {sample_benign_event} -> Proba: {prob_benign[0]:.4f}, Class: {'Malicious' if class_benign==1 else 'Benign'}")

    prob_malicious = ids_nn.predict_proba(sample_malicious_event)
    class_malicious = ids_nn.predict_class(sample_malicious_event)
    logger.info(f"Malicious-like event: {sample_malicious_event} -> Proba: {prob_malicious[0]:.4f}, Class: {'Malicious' if class_malicious==1 else 'Benign'}")

    # 5. Saving and Loading the Model (Persistence for Deployment)
    # Model already saved if early stopping found a best one. Explicit save:
    # ids_nn.save_model("intrusion_detection_model_final.json")
    
    logger.info(f"\n--- Loading Best Saved IDS Model from {model_save_path} (if exists) ---")
    if Path(model_save_path).exists():
        loaded_ids_nn = NeuralNetwork.load_model(model_save_path)
        
        logger.info("Evaluating Loaded IDS Model on Test Data:")
        loaded_test_metrics = loaded_ids_nn.evaluate(test_data, batch_size=32)
        for metric, value in loaded_test_metrics.items():
            if isinstance(value, float):
                 logger.info(f"  {metric.capitalize()}: {value:.4f}")
            else:
                logger.info(f"  {metric.capitalize()}: {value}")

        prob_mal_loaded = loaded_ids_nn.predict_proba(sample_malicious_event)
        logger.info(f"Malicious-like event (loaded model): {sample_malicious_event} -> Proba: {prob_mal_loaded[0]:.4f}")
    else:
        logger.warning(f"No saved model found at {model_save_path} to load and test.")

    # 6. Conceptual "Self-Improvement" / Adaptation
    logger.info("\n--- Conceptual Self-Improvement/Adaptation ---")
    logger.info("Imagine new threat data (e.g., from a honeypot or analyst feedback) becomes available.")
    num_new_threat_samples = 50
    new_threat_data = generate_security_data(num_new_threat_samples) # Simulating new data
    # We could use the loaded_ids_nn or the current ids_nn for further training
    # For fine-tuning, a smaller learning rate might be used.
    if Path(model_save_path).exists() and 'loaded_ids_nn' in locals():
        logger.info("Fine-tuning the loaded model with new threat data...")
        loaded_ids_nn.train(new_threat_data,
                            epochs=10, # Fewer epochs for fine-tuning
                            initial_learning_rate=0.0001, # Lower LR
                            batch_size=16,
                            validation_data=val_data[-len(new_threat_data):], # Use a small part of val for quick check
                            print_every_n_epochs=2,
                            verbose=True)
        
        logger.info("Re-evaluating fine-tuned model:")
        finetuned_metrics = loaded_ids_nn.evaluate(test_data, batch_size=32)
        for metric, value in finetuned_metrics.items():
            if isinstance(value, float): logger.info(f"  {metric.capitalize()}: {value:.4f}")

    logger.info("\n--- ANN in Cybersecurity: Capabilities and Caveats ---")
    logger.info("This script demonstrates a foundational Artificial Neural Network (ANN) tailored for cybersecurity tasks.")
    logger.info("Capabilities:")
    logger.info("  - Pattern Recognition: Learns complex patterns from security data to classify events or detect anomalies.")
    logger.info("  - Adaptability: Can be retrained (fine-tuned) with new data to adapt to evolving threats ('self-improvement').")
    logger.info("  - Automation: Can automate parts of threat detection and analysis, reducing analyst workload.")
    logger.info("Caveats:")
    logger.info("  - Data Dependency: Performance heavily relies on the quality and representativeness of training data.")
    logger.info("  - Adversarial Attacks: ANNs can be vulnerable to adversarial examples (inputs crafted to fool the model).")
    logger.info("  - Explainability: MLP-based models are often 'black boxes', making it hard to understand why a specific decision was made.")
    logger.info("  - False Positives/Negatives: Striking the right balance is crucial; false alarms can cause alert fatigue, while missed threats are dangerous.")
    logger.info("  - Concept Drift: The nature of threats changes over time, requiring continuous monitoring and model updates.")
    logger.info("This implementation is a starting point. Real-world cybersecurity AI often involves more sophisticated architectures (RNNs, Transformers, GNNs), robust data pipelines, MLOps practices, and human-in-the-loop systems.")
