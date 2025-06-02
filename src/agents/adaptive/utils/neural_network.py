
import json, yaml
import math
import torch
import random
import numpy as np

from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any, Union

from src.agents.base.utils.math_science import (
    sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative,
    leaky_relu, leaky_relu_derivative, elu, elu_derivative, swish, swish_derivative,
    softmax, cross_entropy, cross_entropy_derivative)
from logs.logger import get_logger

logger = get_logger("Artificial Neural Network")

CONFIG_PATH = "src/agents/adaptive/configs/adaptive_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

# --- Activation Functions and their Derivatives ---
ACTIVATION_FUNCTIONS: Dict[str, Tuple[Callable, Callable, bool]] = {
    'sigmoid': (sigmoid, sigmoid_derivative, False),
    'relu': (relu, relu_derivative, False),
    'tanh': (tanh, tanh_derivative, False),
    'leaky_relu': (leaky_relu, leaky_relu_derivative, True),
    'elu': (elu, elu_derivative, True),
    'swish': (swish, swish_derivative, False),
    'entropy': (cross_entropy, cross_entropy_derivative, False),
    'linear': (lambda x: x, lambda x: 1.0, False)
}

# --- Loss Function ---
def mean_squared_error(targets: List[float], outputs: List[float]) -> float:
    """L = 0.5 * sum((target_i - output_i)^2)"""
    if len(targets) != len(outputs):
        raise ValueError("Targets and outputs must have the same length.")
    return 0.5 * sum([(target - output) ** 2 for target, output in zip(targets, outputs)])

# --- Neuron: The Basic Processing Unit ---

class Neuron(torch.nn.Module):
    def __init__(self, num_inputs: int,
                 activation_name: str = 'sigmoid',
                 initialization_method: str = 'uniform_scaled',
                 activation_alpha: float = 0.01): # Default alpha for Leaky ReLU / ELU
        super().__init__()
        
        if activation_name not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unsupported activation: {activation_name}. Supported: {list(ACTIVATION_FUNCTIONS.keys())}")
        
        self.activation_fn_ptr, self.activation_fn_derivative_ptr, self.activation_needs_alpha = ACTIVATION_FUNCTIONS[activation_name]
        
        self.num_inputs = num_inputs
        self.initialization_method = initialization_method
        self.activation_alpha = activation_alpha # Store alpha for relevant functions

        #self.m_weights = None

        # Weights and Bias
        self.weights: List[float] = self._initialize_weights()
        self.bias: float = random.uniform(-0.1, 0.1) # Bias initialization

        # For state tracking during forward/backward pass
        self.inputs: List[float] = [0.0] * num_inputs
        self.weighted_sum: float = 0.0
        self.activation: float = 0.0
        self.delta: float = 0.0       # Error term (dE/dz)

        # For Momentum
        self.velocity_weights: List[float] = [0.0] * num_inputs
        self.velocity_bias: float = 0.0

        # For AdaGrad
        self.cache_weights: List[float] = [0.0] * num_inputs # Sum of squared gradients for weights
        self.cache_bias: float = 0.0                     # Sum of squared gradients for bias

        # Adam optimizer moment vectors (initialize by default)
        self.m_weights: List[float] = [0.0] * num_inputs
        self.v_weights: List[float] = [0.0] * num_inputs
        self.m_bias: float = 0.0
        self.v_bias: float = 0.0

    def _initialize_weights(self) -> List[float]:
        if self.initialization_method == 'uniform_scaled':
            limit = 1.0 / math.sqrt(self.num_inputs) if self.num_inputs > 0 else 0.1
            return [random.uniform(-limit, limit) for _ in range(self.num_inputs)]
        elif self.initialization_method == 'he_normal':
            stddev = math.sqrt(2.0 / self.num_inputs) if self.num_inputs > 0 else 0.01
            return [random.gauss(0, stddev) for _ in range(self.num_inputs)]
        elif self.initialization_method == 'lecun_normal':
            stddev = math.sqrt(1.0 / self.num_inputs) if self.num_inputs > 0 else 0.01
            return [random.gauss(0, stddev) for _ in range(self.num_inputs)]
        elif self.initialization_method == 'xavier_uniform':
            limit = math.sqrt(6.0 / (self.num_inputs + 1)) 
            return [random.uniform(-limit, limit) for _ in range(self.num_inputs)]
        else:
            raise ValueError(f"Unsupported weight initialization: {self.initialization_method}")

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
        self.delta = error_signal_from_downstream * self._call_activation_fn_derivative(self.weighted_sum)

    def calculate_gradients(self,
                            weight_decay_lambda: float = 0.0,
                            gradient_clip_value: Optional[float] = None
                           ) -> Tuple[List[float], float]:
        grad_weights_loss = [self.delta * inp for inp in self.inputs]
        grad_bias_loss = self.delta 
        regularized_grad_weights = [
            gw_loss + weight_decay_lambda * w
            for gw_loss, w in zip(grad_weights_loss, self.weights)
        ]
        regularized_grad_bias = grad_bias_loss
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

    def update_parameters(self,
                          grad_weights: List[float], grad_bias: float,
                          learning_rate: float,
                          momentum_coefficient: float = 0.0,
                          adagrad_epsilon: float = 1e-8):
        for i in range(len(self.weights)):
            self.cache_weights[i] += grad_weights[i] ** 2
            adjusted_lr_w = learning_rate / (math.sqrt(self.cache_weights[i]) + adagrad_epsilon)
            self.velocity_weights[i] = (momentum_coefficient * self.velocity_weights[i] +
                                        adjusted_lr_w * grad_weights[i])
            self.weights[i] += self.velocity_weights[i] 
        self.cache_bias += grad_bias ** 2
        adjusted_lr_b = learning_rate / (math.sqrt(self.cache_bias) + adagrad_epsilon)
        self.velocity_bias = (momentum_coefficient * self.velocity_bias +
                              adjusted_lr_b * grad_bias)
        self.bias += self.velocity_bias
    def __repr__(self) -> str:
        act_name = next(name for name, (fn, deriv, needs_a) in ACTIVATION_FUNCTIONS.items() if fn == self.activation_fn_ptr)
        return (f"Neuron(Act:{act_name}, Weights:{[f'{w:.3f}' for w in self.weights]}, "
                f"Bias:{self.bias:.3f}, Alpha:{self.activation_alpha if self.activation_needs_alpha else 'N/A'})")

# --- Neural Layer: A Collection of Neurons ---

class NeuralLayer(torch.nn.Module):
    def __init__(self, num_neurons: int, num_inputs_per_neuron: int,
                 activation_name: str = 'sigmoid',
                 initialization_method: str = 'uniform_scaled',
                 dropout_rate: float = 0.0,
                 activation_alpha: float = 0.01,
                 use_batch_norm: bool = False, # Flag for batch normalization
                 bn_momentum: float = 0.9,     # Momentum for running stats in BN
                 bn_epsilon: float = 1e-5):    # Epsilon for BN stability
        super().__init__()
        
        self.num_neurons = num_neurons
        self.neurons: List[Neuron] = [
            Neuron(num_inputs_per_neuron, activation_name, initialization_method, activation_alpha)
            for _ in range(num_neurons)
        ]
        self.activation_name = activation_name # Store for repr
        self.is_training: bool = False # Controlled by the NeuralNetwork

        # Dropout
        self.dropout_rate = dropout_rate
        self._dropout_mask: Optional[List[float]] = None # To store dropout mask for backprop if needed (though inverted dropout simplifies this)

        # Batch Normalization (Simplified: acts on activations post-neuron computation)
        self.use_batch_norm = use_batch_norm
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        if self.use_batch_norm:
            # These are per-activation, so size is num_neurons
            # No learnable gamma/beta for simplicity in this "native" version
            self.running_mean: List[float] = [0.0] * num_neurons
            self.running_variance: List[float] = [1.0] * num_neurons # Initialize variance to 1

        # Layer statistics (for observation)
        self.history_activation_mean: List[float] = []
        self.history_activation_variance: List[float] = []
        self.history_max_len = 100 # Keep last 100 stats

    def _update_bn_running_stats(self, current_batch_mean: List[float], current_batch_var: List[float]):
        """Updates running mean and variance using momentum."""
        for i in range(self.num_neurons):
            self.running_mean[i] = (self.bn_momentum * self.running_mean[i] +
                                    (1 - self.bn_momentum) * current_batch_mean[i])
            self.running_variance[i] = (self.bn_momentum * self.running_variance[i] +
                                        (1 - self.bn_momentum) * current_batch_var[i])

    def _apply_batch_norm(self, activations: List[float]) -> List[float]:
        """Applies batch normalization to the activations."""
        if not self.use_batch_norm:
            return activations

        normalized_activations = [0.0] * self.num_neurons

        if self.is_training:

            # During training: normalize with running_mean and running_var, then update running_mean and running_var
            # This is a slight deviation from standard BN for SGD, but more stable.
            current_sample_activations_mean = sum(activations) / self.num_neurons if self.num_neurons > 0 else 0.0
            current_sample_activations_var = sum([(a - current_sample_activations_mean)**2 for a in activations]) / self.num_neurons if self.num_neurons > 0 else 0.0
            
            # Update running stats (using the single sample's stats for the "batch" stats)
            # This is where the conceptual difficulty with SGD batch_size=1 and BN comes in.
            # A more robust way is to collect activations over a few iterations to form a "virtual batch"
            # or simply use the running stats for normalization during training as well.
            # For simplicity here, let's normalize by running stats and update running stats using current sample.
            
            # Update running stats using the current sample's activations.
            # This is not standard BN's way of calculating current batch mean/var for normalization.
            # It's more like an online update.
            self._update_bn_running_stats(activations, [(a - m)**2 for a,m in zip(activations, self.running_mean)]) # var approx

            for i in range(self.num_neurons):
                normalized_activations[i] = ((activations[i] - self.running_mean[i]) /
                                             math.sqrt(self.running_variance[i] + self.bn_epsilon))
                # Missing: * gamma + beta (learnable scale and shift, omitted for simplicity)
        else: # Inference mode
            for i in range(self.num_neurons):
                normalized_activations[i] = ((activations[i] - self.running_mean[i]) /
                                             math.sqrt(self.running_variance[i] + self.bn_epsilon))
                # Missing: * gamma + beta

        return normalized_activations

    def _apply_dropout(self, activations: List[float]) -> List[float]:
        """Applies dropout to the activations."""
        if not self.is_training or self.dropout_rate == 0.0:
            self._dropout_mask = None # No mask when not training or no dropout
            return activations

        self._dropout_mask = [0.0] * self.num_neurons # Store the mask
        scaled_activations = [0.0] * self.num_neurons
        scale_factor = 1.0 / (1.0 - self.dropout_rate)

        for i in range(self.num_neurons):
            if random.random() < self.dropout_rate:
                self._dropout_mask[i] = 0.0 # Neuron is dropped
                scaled_activations[i] = 0.0
            else:
                self._dropout_mask[i] = scale_factor # Store the scale factor
                scaled_activations[i] = activations[i] * scale_factor
        return scaled_activations

    def feed_forward(self, inputs: List[float]) -> List[float]:
        # 1. Get raw activations from neurons
        raw_activations = [neuron.activate(inputs) for neuron in self.neurons]

        # (Store stats based on raw_activations before BN/Dropout for layer observation)
        if self.is_training:
            current_mean = sum(raw_activations) / self.num_neurons if self.num_neurons > 0 else 0.0
            current_var = sum([(a - current_mean)**2 for a in raw_activations]) / self.num_neurons if self.num_neurons > 0 else 0.0
            self.history_activation_mean.append(current_mean)
            self.history_activation_variance.append(current_var)
            if len(self.history_activation_mean) > self.history_max_len:
                self.history_activation_mean.pop(0)
                self.history_activation_variance.pop(0)

        # 2. Apply Batch Normalization (if enabled)
        processed_activations = self._apply_batch_norm(raw_activations) if self.use_batch_norm else raw_activations
        
        # 3. Apply Dropout (if enabled)
        final_activations = self._apply_dropout(processed_activations)

        # 4. Update neuron.activation state to the final output of this layer
        # This is crucial for backpropagation, as it needs the value that was passed forward.
        for i, neuron in enumerate(self.neurons):
            neuron.activation = final_activations[i]
            # The neuron.weighted_sum and neuron.inputs are already set from neuron.activate()

        return final_activations

    def get_layer_stats(self) -> Dict[str, float]:
        """Returns recent average mean and variance of raw activations."""
        avg_mean = sum(self.history_activation_mean) / len(self.history_activation_mean) if self.history_activation_mean else 0.0
        avg_var = sum(self.history_activation_variance) / len(self.history_activation_variance) if self.history_activation_variance else 0.0
        bn_stats = {}
        if self.use_batch_norm:
            bn_stats = {
                f"bn_run_mean_{i}": self.running_mean[i] for i in range(self.num_neurons)
                # f"bn_run_var_{i}": self.running_variance[i] for i in range(self.num_neurons) # Can be verbose
            }
        return {
            "avg_raw_activation_mean": avg_mean,
            "avg_raw_activation_variance": avg_var,
            **bn_stats
        }

    def __repr__(self) -> str:
        bn_repr = ", BN" if self.use_batch_norm else ""
        return (f"NeuralLayer({self.num_neurons} neurons, Act: {self.activation_name}, "
                f"Dropout: {self.dropout_rate}{bn_repr})")

# --- Neural Network: The Complete Structure ---

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs: int,
                 layer_config: List[dict],
                 loss_function_name: str = 'mse', # New: 'mse' or 'cross_entropy'
                 optimizer_name: str = 'sgd_momentum_adagrad', # New: 'sgd_momentum_adagrad', 'adam'
                 initialization_method_default: str = 'he_normal',
                 problem_type: str = 'regression', # New: 'regression' or 'classification' (binary/multiclass)
                 config: Optional[Dict[str, Any]] = None): # For other general NN configs
        super().__init__()
        """
        Initializes the neural network with extensive configuration options.

        Args:
            num_inputs (int): Number of features in the input data.
            layer_config (List[dict]): Configuration for each layer. Each dict should contain:
                - 'neurons' (int): Number of neurons in the layer.
                - Optional:
                    - 'activation' (str): Name of activation function (e.g., 'relu', 'sigmoid').
                    - 'dropout' (float): Dropout rate (0 to 1).
                    - 'init' (str): Weight initialization method (e.g., 'he_normal').
                    - 'alpha' (float): Alpha parameter for activations like Leaky ReLU, ELU.
                    - 'batch_norm' (bool): Whether to use (simplified) batch normalization.
                    - 'bn_momentum' (float): Momentum for batch norm running stats.
            loss_function_name (str): Name of the loss function ('mse', 'cross_entropy').
            optimizer_name (str): Name of the optimizer ('sgd_momentum_adagrad', 'adam').
            initialization_method_default (str): Default weight init if not specified per layer.
            problem_type (str): 'regression', 'binary_classification', 'multiclass_classification'.
                                This influences output layer activation if not specified and loss choice.
            config (Optional[Dict[str, Any]]): General configuration dictionary for optimizer params, etc.
        """
        self.num_inputs = num_inputs
        self.layers: List[NeuralLayer] = []
        self.is_training: bool = False
        self.problem_type = problem_type.lower()
        self.loss_function_name = loss_function_name.lower()
        self.optimizer_name = optimizer_name.lower()
        
        self.config = config if config is not None else {}
        self._configure_loss_function()
        self._configure_optimizer_hyperparameters()


        current_num_inputs = num_inputs
        print("Initializing Neural Network Layers:")
        for i, layer_conf in enumerate(layer_config):
            num_n = layer_conf['neurons']
            
            # Determine activation: if output layer and classification, and not specified, use softmax/sigmoid
            default_act = 'sigmoid' # Default for regression or if problem_type is unknown
            if self.problem_type == 'binary_classification' and i == len(layer_config) - 1:
                default_act = 'sigmoid'
            elif self.problem_type == 'multiclass_classification' and i == len(layer_config) - 1:
                default_act = 'softmax' # Special handling for softmax in feed_forward
            elif i < len(layer_config) -1 : # Hidden layers
                default_act = 'relu' 
            
            act_name = layer_conf.get('activation', default_act)
            init_method = layer_conf.get('init', initialization_method_default)
            dropout = layer_conf.get('dropout', 0.0)
            act_alpha = layer_conf.get('alpha', 0.01)
            use_bn = layer_conf.get('batch_norm', False)
            bn_momentum = layer_conf.get('bn_momentum', 0.9) # Get from layer_conf or default

            if i == len(layer_config) - 1: # Final output layer
                dropout = 0.0 # Usually no dropout on output
                if act_name == 'softmax':
                    # Softmax is applied to the whole layer's output, not per-neuron like others.
                    # So, individual neurons in a softmax output layer might just compute linear outputs (identity activation internally),
                    # and then softmax is applied across all their outputs.
                    # For simplicity here, we'll still assign an activation (e.g. 'identity' or linear if we had one,
                    # or let them be sigmoid and then apply softmax later).
                    # Let's assume neurons compute weighted sums, and softmax is applied in feed_forward.
                    # We'll use 'identity' conceptually for neurons if layer activation is softmax.
                    # This requires NeuralLayer/Neuron to potentially handle an 'identity' activation.
                    # NEURON MODIFICATION POINT: Add 'identity' activation: (lambda x: x, lambda x: 1.0, False)
                    # For now, let's say softmax neurons output their weighted sum directly, and NN handles softmax.
                    print(f"  Layer {i} (Output): {num_n}N, Act:{act_name} (applied at NN level), Init:{init_method}, Dropout:{dropout}, BN:{use_bn}")
                    # We'll use a placeholder like 'linear' for neurons if activation is softmax,
                    # assuming NeuralLayer/Neuron can handle it or it implies no explicit neuron-level activation func.
                    # For now, they will still compute their normal activation, but feed_forward will override for softmax.
                    neuron_activation_for_softmax_layer = layer_conf.get('neuron_activation_if_softmax', 'sigmoid') # What neurons do *before* softmax
                    self.layers.append(
                        NeuralLayer(num_n, current_num_inputs, neuron_activation_for_softmax_layer, init_method, dropout, act_alpha, use_bn, bn_momentum=bn_momentum)
                    )
                    self.output_layer_activation_is_softmax = True # Flag for feed_forward
                else:
                    print(f"  Layer {i} (Output): {num_n}N, Act:{act_name}, Init:{init_method}, Dropout:{dropout}, BN:{use_bn}, Alpha:{act_alpha if act_name in ['leaky_relu', 'elu'] else 'N/A'}")
                    self.layers.append(
                        NeuralLayer(num_n, current_num_inputs, act_name, init_method, dropout, act_alpha, use_bn, bn_momentum=bn_momentum)
                    )
                    self.output_layer_activation_is_softmax = False
            else: # Hidden layers
                print(f"  Layer {i} (Hidden): {num_n}N, Act:{act_name}, Init:{init_method}, Dropout:{dropout}, BN:{use_bn}, Alpha:{act_alpha if act_name in ['leaky_relu', 'elu'] else 'N/A'}")
                self.layers.append(
                    NeuralLayer(num_n, current_num_inputs, act_name, init_method, dropout, act_alpha, use_bn, bn_momentum=bn_momentum)
                )
            current_num_inputs = num_n
        print(f"Initialized Neural Network with {len(self.layers)} layers. Problem: {self.problem_type}, Loss: {self.loss_function_name}, Opt: {self.optimizer_name}")

    def _configure_loss_function(self):
        # Assuming math_science.py has mse, cross_entropy, etc.
        # from src.agents.adaptive.utils.math_science import mse, mse_derivative, cross_entropy, cross_entropy_derivative_softmax
        # (where cross_entropy_derivative_softmax is special for CE with Softmax output)
        if self.loss_function_name == 'mse':
            self.loss_fn = mean_squared_error # Imported or defined locally
            # self.loss_fn_derivative = mse_derivative # For dE/dy_pred
        elif self.loss_function_name == 'cross_entropy':
            if self.problem_type not in ['binary_classification', 'multiclass_classification']:
                raise ValueError("Cross-entropy loss is typically for classification problems.")
            # self.loss_fn = cross_entropy_loss # Needs to be defined or imported
            # If output is softmax, dE/dz_output (pre-softmax sum) is simply (y_pred_probs - y_true)
            # This simplifies delta calculation for the output layer.
            pass # Loss calculation handled in train, derivative handled in backprop for CE+Softmax
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function_name}")

    def _configure_optimizer_hyperparameters(self):
        # Adam optimizer parameters
        self.adam_beta1 = self.config.get('adam_beta1', 0.9)
        self.adam_beta2 = self.config.get('adam_beta2', 0.999)
        self.adam_epsilon = self.config.get('adam_epsilon', 1e-8)
        
        # Initialize Adam's moment estimates for each parameter (neuron's weights and bias)
        if self.optimizer_name == 'adam':
            for layer in self.layers:
                for neuron in layer.neurons:
                    # Initialize Adam's moment vectors for each neuron
                    neuron.m_weights = [0.0] * neuron.num_inputs  # First moment for weights
                    neuron.v_weights = [0.0] * neuron.num_inputs  # Second moment for weights
                    neuron.m_bias = 0.0                           # First moment for bias
                    neuron.v_bias = 0.0                           # Second moment for bias
            self.adam_global_timestep = 0

    def _set_training_mode(self, mode: bool):
        self.is_training = mode
        for layer in self.layers:
            # NEURALLAYER MODIFICATION POINT: Ensure NeuralLayer has 'is_training' and propagates it.
            # layer.is_training = mode
            if hasattr(layer, 'is_training'):
                layer.is_training = mode


    def feed_forward(self, inputs: List[float]) -> List[float]:
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}.")
        
        current_outputs = inputs
        for i, layer in enumerate(self.layers):
            current_outputs = layer.feed_forward(current_outputs)
            
            # Special handling for softmax on the output layer
            if i == len(self.layers) - 1 and hasattr(self, 'output_layer_activation_is_softmax') and self.output_layer_activation_is_softmax:

                softmax_probs = softmax(current_outputs)

                for neuron_idx, neuron in enumerate(self.layers[-1].neurons):
                    neuron.activation = softmax_probs[neuron_idx]
                current_outputs = softmax_probs
                
        return current_outputs

    def _calculate_loss_and_output_deltas(self,
                                          target_outputs: List[float],
                                          predicted_outputs: List[float]
                                         ) -> Tuple[float, List[float]]:
        """
        Calculates loss and the initial deltas (dE/dz) for the output layer neurons.
        For CE with Softmax, delta = predicted_probs - target_one_hot.
        For MSE, delta = (predicted - target) * f'(z_output).
        """
        loss = 0.0
        output_deltas = [0.0] * len(target_outputs)
        output_layer = self.layers[-1]

        if self.loss_function_name == 'cross_entropy' and \
           hasattr(self, 'output_layer_activation_is_softmax') and self.output_layer_activation_is_softmax:

            # Simplified CE calculation for one sample:
            epsilon = 1e-12
            for i in range(len(target_outputs)):
                loss -= target_outputs[i] * math.log(predicted_outputs[i] + epsilon) # predicted_outputs are already softmax probs
            
            for i in range(len(output_layer.neurons)):
                # neuron.activation is the softmax probability p_i
                # target_outputs[i] is the true label t_i (e.g., one-hot)
                output_deltas[i] = output_layer.neurons[i].activation - target_outputs[i]
                output_layer.neurons[i].delta = output_deltas[i] # Store dE/dz directly
        
        elif self.loss_function_name == 'cross_entropy':
            # General case for CE without softmax (e.g., sigmoid per neuron)
            loss = cross_entropy(target_outputs, predicted_outputs)
            output_deltas = cross_entropy_derivative(target_outputs, predicted_outputs)
            for i in range(len(output_layer.neurons)):
                output_layer.neurons[i].delta = output_deltas[i]
        
        else:
            raise NotImplementedError(f"Loss derivative logic for {self.loss_function_name} not fully implemented here.")
            
        return loss, output_deltas


    def _backpropagate(self, inputs: List[float], output_layer_error_signals: List[float],
                       learning_rate: float,
                       # Optimizer-specific params passed from train method
                       **optimizer_kwargs):

        # --- Output Layer ---
        output_layer = self.layers[-1]
        inputs_to_output_layer = [n.activation for n in self.layers[-2].neurons] if len(self.layers) > 1 else inputs

        for neuron_idx, neuron in enumerate(output_layer.neurons):
            if not (self.loss_function_name == 'cross_entropy' and 
                    hasattr(self, 'output_layer_activation_is_softmax') and self.output_layer_activation_is_softmax):
                # If not CE+Softmax (where delta is already dE/dz), calculate delta normally
                neuron.calculate_delta(output_layer_error_signals[neuron_idx])
            # Else, neuron.delta was already set by _calculate_loss_and_output_deltas as (p_i - t_i)

            grad_weights, grad_bias = neuron.calculate_gradients(
                optimizer_kwargs.get('weight_decay_lambda', 0.0),
                optimizer_kwargs.get('gradient_clip_value')
            )
            self._apply_optimizer_step(neuron, grad_weights, grad_bias, learning_rate, **optimizer_kwargs)


        # --- Hidden Layers (iterating backwards) ---
        for layer_idx in reversed(range(len(self.layers) - 1)):
            hidden_layer = self.layers[layer_idx]
            downstream_layer = self.layers[layer_idx + 1]
            inputs_to_hidden_layer = [n.activation for n in self.layers[layer_idx-1].neurons] if layer_idx > 0 else inputs

            for neuron_idx, neuron in enumerate(hidden_layer.neurons):
                error_signal_sum_for_activation = 0.0
                for downstream_neuron in downstream_layer.neurons:
                    error_signal_sum_for_activation += downstream_neuron.delta * downstream_neuron.weights[neuron_idx]
                
                neuron.calculate_delta(error_signal_sum_for_activation)

                if hasattr(hidden_layer, '_dropout_mask') and hidden_layer._dropout_mask and \
                   hidden_layer._dropout_mask[neuron_idx] == 0.0:
                    neuron.delta = 0.0 # Correctly ensures no gradient contribution if dropped
                
                if neuron.delta != 0.0: # Only update if not dropped or delta is non-zero
                    grad_weights, grad_bias = neuron.calculate_gradients(
                        optimizer_kwargs.get('weight_decay_lambda', 0.0),
                        optimizer_kwargs.get('gradient_clip_value')
                    )
                    self._apply_optimizer_step(neuron, grad_weights, grad_bias, learning_rate, **optimizer_kwargs)
                else: # Reset velocities if delta is zero to prevent stale momentum
                    # NEURON MODIFICATION POINT: Check if neuron has velocity attributes
                    if hasattr(neuron, 'velocity_weights'):
                        neuron.velocity_weights = [0.0] * neuron.num_inputs
                        neuron.velocity_bias = 0.0


    def _apply_optimizer_step(self, neuron: Neuron, grad_weights: List[float], grad_bias: float,
                              learning_rate: float, **optimizer_kwargs):
        """Applies a single optimization step to a neuron's parameters."""
        if self.optimizer_name == 'sgd_momentum_adagrad':
            # This re-uses Neuron's existing update_parameters.
            # NEURON MODIFICATION POINT: Ensure neuron.update_parameters handles this.
            # It should take grad_weights, grad_bias, learning_rate, momentum_coefficient, adagrad_epsilon.
            neuron.update_parameters(
                grad_weights, grad_bias, learning_rate,
                optimizer_kwargs.get('momentum_coefficient', 0.9),
                optimizer_kwargs.get('adagrad_epsilon', 1e-8)
            )
        elif self.optimizer_name == 'adam':
            # NEURON MODIFICATION POINT: Neuron needs m_weights, v_weights, m_bias, v_bias attributes.
            # Also, adam_global_timestep is tracked by NeuralNetwork.
            
            beta1 = optimizer_kwargs.get('adam_beta1', self.adam_beta1)
            beta2 = optimizer_kwargs.get('adam_beta2', self.adam_beta2)
            epsilon = optimizer_kwargs.get('adam_epsilon', self.adam_epsilon)
            
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
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")


    def train(self, training_data: Union[List[Tuple[List[float], List[float]]], np.ndarray], # Can take list of tuples or numpy arrays
              epochs: int, initial_learning_rate: float,
              batch_size: Optional[int] = None, # New: for mini-batch training
              # Optimizer specific kwargs are now part of self.config or passed directly
              momentum_coefficient: float = 0.9, # Default, can be overridden by self.config
              weight_decay_lambda: float = 0.0001,
              gradient_clip_value: Optional[float] = 1.0,
              adagrad_epsilon: float = 1e-8, # For combined optimizer
              # Adam specific, also can be in self.config
              adam_beta1: Optional[float] = None, 
              adam_beta2: Optional[float] = None,
              adam_epsilon_opt: Optional[float] = None, # Renamed to avoid clash with adagrad_epsilon
              # LR Decay
              lr_scheduler_name: Optional[str] = None, # 'step', 'exponential', 'cosine_annealing'
              lr_decay_rate: Optional[float] = None, # Factor for exp/step or param for cosine
              lr_decay_steps: Optional[int] = None, # Epochs for step/exp or T_max for cosine
              # Early Stopping
              early_stopping_patience: Optional[int] = None,
              early_stopping_min_delta: float = 0.0001,
              validation_data: Optional[Union[List[Tuple[List[float], List[float]]], np.ndarray]] = None,
              # Other
              verbose: bool = True,
              print_every_n_epochs: Optional[int] = None,
              save_best_model_path: Optional[str] = None):

        self._set_training_mode(True)
        current_learning_rate = initial_learning_rate

        # Consolidate optimizer kwargs
        opt_kwargs = {
            'momentum_coefficient': self.config.get('momentum_coefficient', momentum_coefficient),
            'weight_decay_lambda': self.config.get('weight_decay_lambda', weight_decay_lambda),
            'gradient_clip_value': self.config.get('gradient_clip_value', gradient_clip_value),
            'adagrad_epsilon': self.config.get('adagrad_epsilon', adagrad_epsilon),
            'adam_beta1': self.config.get('adam_beta1', adam_beta1 if adam_beta1 is not None else self.adam_beta1),
            'adam_beta2': self.config.get('adam_beta2', adam_beta2 if adam_beta2 is not None else self.adam_beta2),
            'adam_epsilon': self.config.get('adam_epsilon_opt', adam_epsilon_opt if adam_epsilon_opt is not None else self.adam_epsilon), # Use NN's adam_epsilon
        }


        if print_every_n_epochs is None:
            print_every_n_epochs = epochs // 20 if epochs >= 20 else 1
        if print_every_n_epochs == 0: print_every_n_epochs = 1
        
        # Early stopping setup
        best_val_loss = float('inf')
        epochs_no_improve = 0
        if save_best_model_path:
            Path(save_best_model_path).parent.mkdir(parents=True, exist_ok=True)


        print(f"\n--- Training Started ({self.optimizer_name} optimizer) ---")
        # Print all effective hyperparameters
        # ... (omitted for brevity, but good to log them)

        for epoch in range(epochs):
            if self.optimizer_name == 'adam':
                self.adam_global_timestep += 1 # Increment Adam's timestep

            # Convert to list of tuples if numpy array is given
            if isinstance(training_data, np.ndarray):
                # Assuming training_data is (X, y)
                data_list = [(list(x_row), list(y_row) if isinstance(y_row, (list, np.ndarray)) else [y_row]) 
                             for x_row, y_row in zip(training_data[0], training_data[1])]
            else:
                data_list = list(training_data) # Ensure it's a list for shuffling

            random.shuffle(data_list)
            epoch_loss = 0.0

            if batch_size is None or batch_size >= len(data_list): # Full batch / SGD
                for inputs_sample, targets_sample in data_list:
                    outputs = self.feed_forward(inputs_sample)
                    loss_val, output_err_signals = self._calculate_loss_and_output_deltas(targets_sample, outputs)
                    epoch_loss += loss_val
                    self._backpropagate(inputs_sample, output_err_signals, current_learning_rate, **opt_kwargs)
                avg_epoch_loss = epoch_loss / len(data_list) if data_list else 0.0
            else: # Mini-batch training
                num_batches = math.ceil(len(data_list) / batch_size)
                total_loss_accumulator = 0.0
                for i in range(num_batches):
                    batch_data = data_list[i*batch_size : (i+1)*batch_size]
                    if not batch_data: continue

                    batch_loss_sum = 0.0
                    for inputs_sample, targets_sample in batch_data:
                        outputs = self.feed_forward(inputs_sample)
                        loss_val, output_err_signals = self._calculate_loss_and_output_deltas(targets_sample, outputs)
                        batch_loss_sum += loss_val
                        self._backpropagate(inputs_sample, output_err_signals, current_learning_rate, **opt_kwargs)
                    
                    total_loss_accumulator += batch_loss_sum
                avg_epoch_loss = total_loss_accumulator / len(data_list) if data_list else 0.0


            # Learning Rate Scheduling
            if lr_scheduler_name:
                current_learning_rate = self._apply_lr_schedule(
                    initial_learning_rate, current_learning_rate, epoch, epochs,
                    lr_scheduler_name, lr_decay_rate, lr_decay_steps
                )

            # Validation and Early Stopping
            val_loss_str = ""
            if validation_data:
                val_loss = self.evaluate(validation_data, batch_size=batch_size if batch_size else len(validation_data)) # Use eval batch_size
                val_loss_str = f", Val Loss: {val_loss:.7f}"
                if val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    if save_best_model_path:
                        self.save_model(save_best_model_path)
                        if verbose: print(f"  Epoch {epoch+1}: Validation loss improved. Model saved to {save_best_model_path}")
                else:
                    epochs_no_improve += 1
                
                if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                    if verbose: print(f"Epoch {epoch + 1}: Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                    break # Exit training loop
            
            if verbose and (epoch + 1) % print_every_n_epochs == 0:
                # ... (printing layer stats as before) ...
                layer_stats_str = "" # Rebuild for this print
                for i_layer, layer_obj in enumerate(self.layers):
                    stats = layer_obj.get_layer_stats() # Assumes NeuralLayer.get_layer_stats() exists
                    layer_stats_str += f"\n  L{i_layer} Stats: AvgRawMean={stats['avg_raw_activation_mean']:.3f}, AvgRawVar={stats['avg_raw_activation_variance']:.3f}"
                    if hasattr(layer_obj, 'use_batch_norm') and layer_obj.use_batch_norm:
                         layer_stats_str += f", BNMean0={stats.get('bn_run_mean_0', 'N/A'):.3f}"
                print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_epoch_loss:.7f}, LR: {current_learning_rate:.7f}{val_loss_str}{layer_stats_str}")

        print(f"--- Training Finished (Epochs: {epoch+1}) ---")
        self._set_training_mode(False)


    def _apply_lr_schedule(self, initial_lr, current_lr, epoch, total_epochs,
                           scheduler_name, decay_rate, decay_steps) -> float:
        if scheduler_name == 'step':
            if decay_steps and decay_rate and (epoch + 1) % decay_steps == 0:
                return current_lr * decay_rate
        elif scheduler_name == 'exponential':
            if decay_rate: # decay_rate is the exponent factor here
                 return initial_lr * (decay_rate ** (epoch / (decay_steps if decay_steps else total_epochs)))
        elif scheduler_name == 'cosine_annealing':
            # T_max (decay_steps) is the number of epochs in one cycle.
            # eta_min could be a parameter (e.g., self.config.get('lr_cosine_eta_min', 0))
            if decay_steps and decay_rate is not None: # decay_rate here is eta_min
                eta_min = decay_rate
                # We need T_cur, which is current epoch within the current cycle
                t_cur = epoch % decay_steps
                return eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * t_cur / decay_steps))
        return current_lr # No change if no schedule matched or params missing

    def evaluate(self, test_data: Union[List[Tuple[List[float], List[float]]], np.ndarray],
                 batch_size: Optional[int] = None) -> float:
        """Evaluates the network on test data and returns average loss."""
        self._set_training_mode(False)
        total_loss = 0.0
        num_samples = 0

        if isinstance(test_data, np.ndarray):
            data_list = [(list(x_row), list(y_row) if isinstance(y_row, (list, np.ndarray)) else [y_row])
                         for x_row, y_row in zip(test_data[0], test_data[1])]
        else:
            data_list = list(test_data)
        
        num_samples = len(data_list)
        if num_samples == 0: return 0.0

        if batch_size is None or batch_size >= num_samples: # Full batch / SGD style
            for inputs, targets in data_list:
                outputs = self.predict(inputs) # predict sets training_mode = False
                loss_val, _ = self._calculate_loss_and_output_deltas(targets, outputs)
                total_loss += loss_val
            return total_loss / num_samples if num_samples > 0 else 0.0
        else: # Mini-batch evaluation
            num_batches = math.ceil(num_samples / batch_size)
            for i in range(num_batches):
                batch_data = data_list[i*batch_size : (i+1)*batch_size]
                if not batch_data: continue
                for inputs, targets in batch_data:
                    outputs = self.predict(inputs)
                    loss_val, _ = self._calculate_loss_and_output_deltas(targets, outputs)
                    total_loss += loss_val
            return total_loss / num_samples if num_samples > 0 else 0.0

    def predict(self, inputs):
        self._set_training_mode(False)
        return self.feed_forward(inputs)

    def predict_proba(self, inputs: List[float]) -> List[float]:
        """Predicts probabilities (especially for classification).
           If output is not softmax/sigmoid, it returns raw activations."""
        self._set_training_mode(False)
        raw_outputs = self.feed_forward(inputs) # This applies softmax if configured
        return raw_outputs # feed_forward already handles softmax if applicable

    def predict_class(self, inputs: List[float]) -> Union[int, List[int]]:
        """Predicts class labels."""
        self._set_training_mode(False)
        probabilities = self.predict_proba(inputs)
        if self.problem_type == 'binary_classification':
            return [round(p) for p in probabilities] # Or just round(probabilities[0]) if single output
        elif self.problem_type == 'multiclass_classification':
            # Return index of max probability
            return probabilities.index(max(probabilities))
        else: # Regression or unknown
            print("Warning: predict_class called for non-classification problem type. Returning raw probabilities.")
            return probabilities


    def get_weights_biases(self) -> List[Dict[str, Any]]:
        """Returns all weights and biases of the network."""
        network_params = []
        for l_idx, layer in enumerate(self.layers):
            layer_params = {'layer_index': l_idx, 'neurons': []}
            for n_idx, neuron in enumerate(layer.neurons):
                neuron_params = {
                    'neuron_index': n_idx,
                    'weights': list(neuron.weights), # Ensure serializable
                    'bias': neuron.bias
                }
                layer_params['neurons'].append(neuron_params)
            network_params.append(layer_params)
        return network_params

    def set_weights_biases(self, network_params: List[Dict[str, Any]]):
        """Sets weights and biases from a previously saved structure."""
        if len(network_params) != len(self.layers):
            raise ValueError("Mismatch in number of layers for loading weights.")
        for l_idx, layer_data in enumerate(network_params):
            if len(layer_data['neurons']) != len(self.layers[l_idx].neurons):
                raise ValueError(f"Mismatch in number of neurons for layer {l_idx}.")
            for n_idx, neuron_data in enumerate(layer_data['neurons']):
                neuron = self.layers[l_idx].neurons[n_idx]
                if len(neuron_data['weights']) != len(neuron.weights):
                    raise ValueError(f"Mismatch in number of weights for neuron {n_idx} in layer {l_idx}.")
                neuron.weights = list(neuron_data['weights']) # Make sure it's a list of floats
                neuron.bias = float(neuron_data['bias'])


    def save_model(self, filepath: str):
        """Saves the model architecture and weights to a JSON file."""
        model_state = {
            'num_inputs': self.num_inputs,
            'layer_config': [], # We'll store the effective config used to build layers
            'loss_function_name': self.loss_function_name,
            'optimizer_name': self.optimizer_name,
            'problem_type': self.problem_type,
            'initialization_method_default': self.config.get('initialization_method_default', 'he_normal'), # Store the default used
            'trained_weights_biases': self.get_weights_biases(),
            # Store running means/vars for BN layers if used
            'batch_norm_stats': []
        }
        for layer in self.layers:
            # Reconstruct layer_config entry from actual layer properties
            # NEURALLAYER MODIFICATION POINT: Ensure NeuralLayer stores its config params like activation_name, dropout_rate, etc.
            # For now, make assumptions based on attributes existing.
            l_conf = {
                'neurons': layer.num_neurons,
                'activation': layer.activation_name if hasattr(layer, 'activation_name') else 'sigmoid', # Fallback
                'dropout': layer.dropout_rate if hasattr(layer, 'dropout_rate') else 0.0,
                'init': layer.neurons[0].initialization_method if layer.neurons else 'unknown',
                'alpha': layer.neurons[0].activation_alpha if layer.neurons and hasattr(layer.neurons[0], 'activation_alpha') else 0.01,
                'batch_norm': layer.use_batch_norm if hasattr(layer, 'use_batch_norm') else False,
                'bn_momentum': layer.bn_momentum if hasattr(layer, 'bn_momentum') else 0.9
            }
            model_state['layer_config'].append(l_conf)

            if hasattr(layer, 'use_batch_norm') and layer.use_batch_norm:
                model_state['batch_norm_stats'].append({
                    'running_mean': list(layer.running_mean),
                    'running_variance': list(layer.running_variance)
                })
            else:
                model_state['batch_norm_stats'].append(None)


        try:
            with open(filepath, 'w') as f:
                json.dump(model_state, f, indent=4)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, filepath: str, custom_config_override: Optional[Dict] = None) -> 'NeuralNetwork':
        """Loads a model architecture and weights from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                model_state = json.load(f)
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            raise

        # Use custom_config_override if provided, else use saved config or defaults
        nn_config = custom_config_override if custom_config_override is not None else {}
        
        # Create a new network instance with the loaded architecture
        # The layer_config from the file should be used directly
        network = cls(
            num_inputs=model_state['num_inputs'],
            layer_config=model_state['layer_config'], # This uses the reconstructed config
            loss_function_name=model_state.get('loss_function_name', 'mse'),
            optimizer_name=model_state.get('optimizer_name', 'sgd_momentum_adagrad'),
            initialization_method_default=model_state.get('initialization_method_default', 'he_normal'),
            problem_type=model_state.get('problem_type', 'regression'),
            config=nn_config # Pass any overrides or additional runtime configs
        )
        
        # Set weights and biases
        if 'trained_weights_biases' in model_state:
            network.set_weights_biases(model_state['trained_weights_biases'])
        
        # Load Batch Norm running statistics
        if 'batch_norm_stats' in model_state:
            for i, layer_stats in enumerate(model_state['batch_norm_stats']):
                if layer_stats and hasattr(network.layers[i], 'use_batch_norm') and network.layers[i].use_batch_norm:
                    network.layers[i].running_mean = list(layer_stats['running_mean'])
                    network.layers[i].running_variance = list(layer_stats['running_variance'])
        
        print(f"Model loaded successfully from {filepath}")
        return network

# --- Example Usage: Solving the XOR Problem ---
if __name__ == "__main__":
    print("--- Neural Network Extensive Expansion Demo ---")

    # Define a more complex configuration
    complex_config = {
        'optimizer_name': 'adam', # Try Adam
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon_opt': 1e-7, # Use _opt suffix for clarity if passing to train
        'momentum_coefficient': 0.0, # Not used by Adam directly
        'adagrad_epsilon': 1e-7, # Not used by Adam
        'weight_decay_lambda': 0.00001, # Adam can have weight decay too (often called AdamW)
        'gradient_clip_value': 1.0,
    }

    iris_layer_config = [
        {'neurons': 10, 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.1, 'batch_norm': True, 'bn_momentum': 0.95},
        {'neurons': 8, 'activation': 'relu', 'init': 'he_normal', 'dropout': 0.1, 'batch_norm': True, 'bn_momentum': 0.95},
        {'neurons': 3, 'activation': 'softmax', 'init': 'xavier_uniform'} # Output for 3 classes
    ]
    
    print("\nInitializing Iris Classification Network (Example):")
    num_iris_samples = 150
    num_features_iris = 4
    iris_X = [[random.uniform(0,1) for _ in range(num_features_iris)] for _ in range(num_iris_samples)]
    iris_y_labels = [random.randint(0,2) for _ in range(num_iris_samples)]
    # Convert y_labels to one-hot encoding for cross-entropy with softmax
    iris_y_one_hot = []
    for label in iris_y_labels:
        one_hot = [0.0, 0.0, 0.0]
        one_hot[label] = 1.0
        iris_y_one_hot.append(one_hot)

    iris_training_data = list(zip(iris_X, iris_y_one_hot))
    
    # Create validation split (e.g., 20%)
    random.shuffle(iris_training_data)
    split_idx = int(0.8 * len(iris_training_data))
    train_subset = iris_training_data[:split_idx]
    val_subset = iris_training_data[split_idx:]


    nn_iris = NeuralNetwork(num_inputs=num_features_iris,
                            layer_config=iris_layer_config,
                            loss_function_name='cross_entropy', # Suitable for multiclass
                            optimizer_name='adam',            # Using Adam
                            problem_type='multiclass_classification',
                            config=complex_config) # Pass the optimizer configs

    save_path = "src/agents/adaptive/test/iris_nn_best.json"

    nn_iris.train(train_subset, epochs=100, initial_learning_rate=0.001, # Adam often uses smaller LRs
                  batch_size=16,
                  weight_decay_lambda=complex_config['weight_decay_lambda'], # Can pass directly or ensure in config
                  gradient_clip_value=complex_config['gradient_clip_value'],
                  lr_scheduler_name='cosine_annealing', lr_decay_rate=0.00001, lr_decay_steps=50, # rate is eta_min for cosine
                  validation_data=val_subset, early_stopping_patience=10, early_stopping_min_delta=0.001,
                  print_every_n_epochs=10,
                  save_best_model_path=save_path)

    print("\n--- Testing Trained Iris Network (from training) ---")
    final_val_loss = nn_iris.evaluate(val_subset, batch_size=16)
    print(f"Final Validation Loss (from current model): {final_val_loss:.7f}")

    correct_iris = 0
    # Test with a few samples from validation set
    for inputs, target_one_hot in val_subset[:10]:
        predicted_probs = nn_iris.predict_proba(inputs)
        predicted_class = nn_iris.predict_class(inputs) # Uses predict_proba internally
        actual_class = target_one_hot.index(1.0)
        is_correct = (predicted_class == actual_class)
        if is_correct: correct_iris +=1
        print(f"Input: [...], Target: {actual_class}, PredProbs: {[f'{p:.2f}' for p in predicted_probs]}, PredClass: {predicted_class} -> {'Correct' if is_correct else 'Incorrect'}")
    print(f"Sample Iris Accuracy: {correct_iris / min(10, len(val_subset)) * 100:.2f}%")


    # --- Load the best saved model and test ---
    if Path(save_path).exists():
        print(f"\n--- Loading and Testing Best Saved Iris Model from {save_path} ---")
        loaded_nn_iris = NeuralNetwork.load_model(save_path)
        
        loaded_val_loss = loaded_nn_iris.evaluate(val_subset, batch_size=16)
        print(f"Validation Loss (from loaded best model): {loaded_val_loss:.7f}")

        correct_loaded_iris = 0
        for inputs, target_one_hot in val_subset[:10]:
            predicted_probs_loaded = loaded_nn_iris.predict_proba(inputs)
            predicted_class_loaded = loaded_nn_iris.predict_class(inputs)
            actual_class_loaded = target_one_hot.index(1.0)
            is_correct_loaded = (predicted_class_loaded == actual_class_loaded)
            if is_correct_loaded: correct_loaded_iris +=1
            print(f"Input: [...], Target: {actual_class_loaded}, PredProbs: {[f'{p:.2f}' for p in predicted_probs_loaded]}, PredClass: {predicted_class_loaded} -> {'Correct' if is_correct_loaded else 'Incorrect'}")
        print(f"Sample Iris Accuracy (Loaded Model): {correct_loaded_iris / min(10, len(val_subset)) * 100:.2f}%")


    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    
    model = NeuralNetwork.load_model("src/agents/adaptive/test/iris_nn_best.json")

    # Load and prepare test data
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)
    
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    
    _, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    test_data = list(zip(X_test.tolist(), y_test.tolist()))
    
    # Evaluate accuracy
    correct = 0
    for x, y_true in test_data:
        y_pred = model.predict_class(x)
        actual_class = y_true.index(1.0)
        if y_pred == actual_class:
            correct += 1
    
    accuracy = correct / len(test_data)
    print("Accuracy:", accuracy)

    print("\n--- A Note on \"Mimicking Human Neural Networks\" ---")
    print("This script implements a simplified mathematical model (Artificial Neural Network) ")
    print("inspired by some principles of biological neural networks, such as interconnected")
    print("processing units (neurons) and adaptive connections (weights).")
    print("Key differences from biological systems include:")
    print("  - Learning Algorithm: Backpropagation is a specific mathematical optimization,")
    print("    while biological learning (e.g., Hebbian learning, STDP) is different.")
    print("  - Neuron Model: Sigmoid neurons are a simplification of diverse biological neuron types.")
    print("  - Scale and Complexity: The human brain is vastly more complex in structure, chemistry,")
    print("    and parallel processing capabilities.")
    print("  - No True Cognition: This model performs pattern recognition based on training data;")
    print("    it does not possess consciousness, understanding, or general intelligence.")
    print("This script is a foundational exercise in understanding ANN mechanics from scratch.")
